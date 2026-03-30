use std::collections::{BTreeMap, HashMap, HashSet};

use petgraph::graph::NodeIndex;

use crate::llir::cfg::IrCfg;
use crate::llir::{Callee, Function, Inst, PReg, Rhs, Term, Ty, Value, Var};
use crate::ssa::def_use::{all_vars, def_to_blocks};
use crate::ssa::dominance_frontier::ircfg_to_dominance_frontier;
use crate::ssa::liveness::compute_live_in;

use super::{
    SsaBlock, SsaCallee, SsaFunction, SsaInst, SsaLabel, SsaPhiNode, SsaRhs, SsaTerm, SsaValue,
    SsaVar,
};

/// Minimal
fn compute_phi_sites_minimal(
    func: &Function,
    cfg: &IrCfg,
    df: &HashMap<NodeIndex, HashSet<NodeIndex>>,
) -> HashMap<Var, HashSet<NodeIndex>> {
    let all_vars = all_vars(func);
    let mut var_to_phi_sites = HashMap::new();
    for var in all_vars {
        let mut worklist: Vec<NodeIndex> = def_to_blocks(func, var)
            .into_iter()
            .map(|block| cfg.node_by_index(block).expect("block index out of bounds"))
            .collect();
        let mut phi_sites = HashSet::new();

        while let Some(node) = worklist.pop() {
            let empty = HashSet::new();
            for df_node in df.get(&node).unwrap_or(&empty) {
                if !phi_sites.contains(df_node) {
                    phi_sites.insert(*df_node);
                    worklist.push(*df_node);
                }
            }
        }
        var_to_phi_sites.insert(var, phi_sites);
    }
    var_to_phi_sites
}

/// Pruned
fn compute_phi_sites_pruned(
    func: &Function,
    cfg: &IrCfg,
    df: &HashMap<NodeIndex, HashSet<NodeIndex>>,
) -> HashMap<Var, HashSet<NodeIndex>> {
    let live_in = compute_live_in(cfg);

    let all_vars = all_vars(func);
    let mut var_to_phi_sites = HashMap::new();
    for var in all_vars {
        let mut worklist: Vec<NodeIndex> = def_to_blocks(func, var)
            .into_iter()
            .map(|block| cfg.node_by_index(block).expect("block index out of bounds"))
            .collect();
        let mut phi_sites = HashSet::new();

        while let Some(node) = worklist.pop() {
            let empty = HashSet::new();
            for df_node in df.get(&node).unwrap_or(&empty) {
                if !phi_sites.contains(df_node)
                    && live_in
                        .get(df_node)
                        .map_or(false, |live| live.contains(&var))
                {
                    phi_sites.insert(*df_node);
                    worklist.push(*df_node);
                }
            }
        }
        var_to_phi_sites.insert(var, phi_sites);
    }
    var_to_phi_sites
}

fn infer_var_ty(func: &Function, var: Var) -> Ty {
    for block in &func.blocks {
        for inst in &block.insts {
            match inst {
                Inst::Assign { dst, ty, .. } if *dst == var => return *ty,
                Inst::Call { rets, .. } => {
                    for (v, ty) in rets {
                        if *v == var {
                            return *ty;
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ty::I64
}

struct RenameContext {
    counter: HashMap<Var, u32>,
    stack: HashMap<Var, Vec<u32>>,
}

impl RenameContext {
    fn new(vars: &HashSet<Var>) -> Self {
        Self {
            counter: vars.iter().map(|var| (*var, 0u32)).collect(),
            stack: vars.iter().map(|var| (*var, vec![0u32])).collect(),
        }
    }

    /// Returns the current version of the variable.
    fn current_version(&self, var: Var) -> u32 {
        *self
            .stack
            .get(&var)
            .unwrap_or(&vec![0])
            .last()
            .unwrap_or(&0)
    }

    fn new_version(&mut self, var: Var) -> u32 {
        let new_version = *self.counter.entry(var).or_insert(0) + 1;
        self.counter.insert(var, new_version);
        self.stack.entry(var).or_insert(vec![0]).push(new_version);
        new_version
    }
}

/// φサイトを受け取り、rename → SsaFunction を構築する共通パス
fn build_ssa_from_phi_sites(
    func: &Function,
    cfg: &IrCfg,
    phi_sites: HashMap<Var, HashSet<NodeIndex>>,
) -> SsaFunction {
    // construct phi nodes for each block
    let mut node_to_phis: HashMap<NodeIndex, Vec<(Var, Ty)>> = HashMap::new();
    for (var, sites) in phi_sites {
        for site in sites {
            node_to_phis
                .entry(site)
                .or_default()
                .push((var, infer_var_ty(func, var)));
        }
    }

    // construct dominator tree
    let dom_tree = petgraph::algo::dominators::simple_fast(
        cfg.graph(),
        cfg.entry_node().expect("CFG must have an entry node"),
    );
    let mut dom_children: HashMap<NodeIndex, Vec<NodeIndex>> = HashMap::new();
    for node in cfg.graph().node_indices() {
        if Some(node) == cfg.entry_node() {
            continue;
        }
        let idom = dom_tree
            .immediate_dominator(node)
            .expect("node must have an immediate dominator");
        dom_children.entry(idom).or_default().push(node);
    }

    // initialize rename context
    let all_vars = all_vars(func);
    let mut rename_ctx = RenameContext::new(&all_vars);

    // initialize SSA blocks
    let mut ssa_blocks: HashMap<NodeIndex, SsaBlock> = HashMap::new();
    for node in cfg.graph().node_indices() {
        let block_index = cfg.block_index(node);
        let block = &func.blocks[block_index];
        let label = block.label.0.clone();
        let mut phis = Vec::new();
        // add placeholder phis for each variable that has a phi site
        for (var, ty) in node_to_phis.get(&node).unwrap_or(&vec![]).iter() {
            phis.push(SsaPhiNode {
                dst: SsaVar {
                    base: *var,
                    version: 0,
                },
                ty: *ty,
                args: BTreeMap::new(),
            });
        }

        let insts = Vec::new();
        ssa_blocks.insert(
            node,
            SsaBlock {
                label: SsaLabel(label),
                phis,                                      // placeholder phis
                insts,                                     // empty placeholder
                term: SsaTerm::Trap("placeholder".into()), // placeholder
            },
        );
    }

    // version 0 for parameters
    let params = func
        .params
        .iter()
        .map(|(var, ty)| {
            (
                SsaVar {
                    base: *var,
                    version: 0,
                },
                *ty,
            )
        })
        .collect();

    // rename blocks. replace placeholder phis, insts, and term with actual versioned ones.
    let entry_node = cfg.entry_node().expect("CFG must have an entry node");
    rename_block(
        entry_node,
        &cfg,
        &dom_children,
        &mut rename_ctx,
        &mut ssa_blocks,
    );
    let blocks = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let node = cfg.node_by_index(i).expect("block index out of bounds");
            ssa_blocks.remove(&node).expect("block must exist")
        })
        .collect();

    // construct and return SSA function
    SsaFunction {
        name: func.name.clone(),
        params,
        ret_ty: func.ret_ty.clone(),
        blocks,
    }
}

pub fn construct_minimal_ssa(func: &Function) -> SsaFunction {
    let cfg = IrCfg::build(func);
    let df = ircfg_to_dominance_frontier(&cfg);
    let phi_sites = compute_phi_sites_minimal(func, &cfg, &df);
    build_ssa_from_phi_sites(func, &cfg, phi_sites)
}

pub fn construct_pruned_ssa(func: &Function) -> SsaFunction {
    let cfg = IrCfg::build(func);
    let df = ircfg_to_dominance_frontier(&cfg);
    let phi_sites = compute_phi_sites_pruned(func, &cfg, &df);
    build_ssa_from_phi_sites(func, &cfg, phi_sites)
}

fn rename_block(
    node: NodeIndex,
    cfg: &IrCfg,
    dom_children: &HashMap<NodeIndex, Vec<NodeIndex>>,
    ctx: &mut RenameContext,
    ssa_blocks: &mut HashMap<NodeIndex, SsaBlock>,
) {
    let saved: HashMap<Var, usize> = ctx
        .stack
        .iter()
        .map(|(var, stack)| (*var, stack.len()))
        .collect();

    // rename phis
    let ssa_block = ssa_blocks.get_mut(&node).expect("block must exist");
    for phi in &mut ssa_block.phis {
        let var = phi.dst.base;
        phi.dst = SsaVar {
            base: var,
            version: ctx.new_version(var),
        }
    }

    // rename instructions
    let block = cfg.block(node);
    for inst in block.insts.iter() {
        ssa_block.insts.push(rename_inst(inst, ctx));
    }
    ssa_block.term = rename_term(&block.term, ctx);
    ssa_block.label = SsaLabel(block.label.0.clone());

    // update successors' phis
    let succs: Vec<NodeIndex> = cfg.successors(node).map(|(_, succ, _)| succ).collect();
    for succ in succs {
        let succ_block = ssa_blocks.get_mut(&succ).expect("block must exist");
        for phi in &mut succ_block.phis {
            let var = phi.dst.base;
            let version = ctx.current_version(var);
            phi.args.insert(
                SsaLabel(block.label.0.clone()),
                SsaValue::VarRef(SsaVar { base: var, version }, phi.ty),
            );
        }
    }

    // recursively rename children on dominator tree
    for &child in dom_children.get(&node).unwrap_or(&vec![]).iter() {
        rename_block(child, cfg, dom_children, ctx, ssa_blocks);
    }

    // stack rollback
    for (var, depth) in saved.iter() {
        ctx.stack
            .get_mut(var)
            .expect("variable must exist")
            .truncate(*depth);
    }
}

/// Rename an instruction.
fn rename_inst(inst: &Inst, ctx: &mut RenameContext) -> SsaInst {
    match inst {
        Inst::Assign { dst, ty, rhs } => {
            // uses should be versioned first, then def should create new version.
            let ssa_rhs = rename_rhs(rhs, ctx);
            SsaInst::Assign {
                dst: SsaVar {
                    base: *dst,
                    version: ctx.new_version(*dst),
                },
                ty: *ty,
                rhs: ssa_rhs,
            }
        }
        Inst::Store { ty, addr, val } => SsaInst::Store {
            ty: *ty,
            addr: rename_value(addr, ctx),
            val: rename_value(val, ctx),
        },
        Inst::Call {
            rets,
            callee,
            args,
            clobber,
        } => {
            let ssa_args = args.iter().map(|val| rename_value(val, ctx)).collect();
            let ssa_callee = match callee {
                Callee::Direct(name) => SsaCallee::Direct(name.clone()),
                Callee::Indirect(val) => SsaCallee::Indirect(rename_value(val, ctx)),
            };
            let ssa_rets: Vec<(SsaVar, Ty)> = rets
                .iter()
                .map(|(var, ty)| {
                    (
                        SsaVar {
                            base: *var,
                            version: ctx.new_version(*var),
                        },
                        *ty,
                    )
                })
                .collect();
            let ssa_clobber: Vec<(SsaVar, Ty)> = clobber
                .iter()
                .filter_map(|reg| {
                    if *reg == PReg::ZERO {
                        return None;
                    }
                    let var = Var::PReg(*reg);
                    Some((
                        SsaVar {
                            base: var,
                            version: ctx.new_version(var),
                        },
                        Ty::I64,
                    ))
                })
                .collect();
            SsaInst::Call {
                rets: ssa_rets,
                callee: ssa_callee,
                args: ssa_args,
                clobber: ssa_clobber,
            }
        }
        Inst::Trap(reason) => SsaInst::Trap(reason.clone()),
    }
}

fn rename_term(term: &Term, ctx: &mut RenameContext) -> SsaTerm {
    match term {
        Term::BrCond {
            cond,
            then_label,
            else_label,
        } => SsaTerm::BrCond {
            cond: rename_value(cond, ctx),
            then_label: SsaLabel(then_label.0.clone()),
            else_label: SsaLabel(else_label.0.clone()),
        },
        Term::Jmp(label) => SsaTerm::Jmp(SsaLabel(label.0.clone())),
        Term::Ret(vals) => SsaTerm::Ret(vals.iter().map(|val| rename_value(val, ctx)).collect()),
        Term::Switch {
            value,
            cases,
            default,
        } => SsaTerm::Switch {
            value: rename_value(value, ctx),
            cases: cases
                .iter()
                .map(|(n, label)| (*n, SsaLabel(label.0.clone())))
                .collect(),
            default: SsaLabel(default.0.clone()),
        },
        Term::Trap(reason) => SsaTerm::Trap(reason.clone()),
    }
}

/// Rename the right-hand side of an assignment.
fn rename_rhs(rhs: &Rhs, ctx: &mut RenameContext) -> SsaRhs {
    match rhs {
        Rhs::BinOp(op, a, b) => SsaRhs::BinOp(*op, rename_value(a, ctx), rename_value(b, ctx)),
        Rhs::Cmp(op, a, b) => SsaRhs::Cmp(*op, rename_value(a, ctx), rename_value(b, ctx)),
        Rhs::Cast(op, val, ty) => SsaRhs::Cast(*op, rename_value(val, ctx), *ty),
        Rhs::Load(ty, addr) => SsaRhs::Load(*ty, rename_value(addr, ctx)),
        Rhs::Copy(val) => SsaRhs::Copy(rename_value(val, ctx)),
    }
}

/// Rename a value.
fn rename_value(value: &Value, ctx: &mut RenameContext) -> SsaValue {
    match value {
        Value::VarRef(var, ty) => SsaValue::VarRef(
            SsaVar {
                base: *var,
                version: ctx.current_version(*var),
            },
            *ty,
        ),
        Value::Imm(n, ty) => SsaValue::Imm(*n, *ty),
        Value::Undef(ty) => SsaValue::Undef(*ty),
    }
}

#[cfg(test)]
mod tests {
    use crate::llir::cfg::IrCfg;
    use crate::llir::{
        BinOp, Block, CmpPred, Function, Inst, Label, PReg, Rhs, Temp, Term, Ty, Value, Var,
    };
    use crate::ssa::dominance_frontier::ircfg_to_dominance_frontier;

    use super::*;

    /// entry: cmp → br then, else
    /// then:  a0 = 1 → jmp join
    /// else:  a0 = 2 → jmp join
    /// join:  ret a0
    #[test]
    fn test_diamond_phi_at_join() {
        let func = Function {
            name: "diamond".into(),
            params: vec![],
            ret_ty: None,
            blocks: vec![
                Block {
                    label: Label("entry".into()),
                    insts: vec![Inst::Assign {
                        dst: Var::Temp(Temp(0)),
                        ty: Ty::I1,
                        rhs: Rhs::Cmp(
                            CmpPred::Eq,
                            Value::VarRef(Var::PReg(PReg(10)), Ty::I64),
                            Value::Imm(0, Ty::I64),
                        ),
                    }],
                    term: Term::BrCond {
                        cond: Value::VarRef(Var::Temp(Temp(0)), Ty::I1),
                        then_label: Label("then".into()),
                        else_label: Label("else".into()),
                    },
                },
                Block {
                    label: Label("then".into()),
                    insts: vec![Inst::Assign {
                        dst: Var::PReg(PReg(10)),
                        ty: Ty::I64,
                        rhs: Rhs::Copy(Value::Imm(1, Ty::I64)),
                    }],
                    term: Term::Jmp(Label("join".into())),
                },
                Block {
                    label: Label("else".into()),
                    insts: vec![Inst::Assign {
                        dst: Var::PReg(PReg(10)),
                        ty: Ty::I64,
                        rhs: Rhs::Copy(Value::Imm(2, Ty::I64)),
                    }],
                    term: Term::Jmp(Label("join".into())),
                },
                Block {
                    label: Label("join".into()),
                    insts: vec![],
                    term: Term::Ret(vec![Value::VarRef(Var::PReg(PReg(10)), Ty::I64)]),
                },
            ],
        };

        let cfg = IrCfg::build(&func);
        let df = ircfg_to_dominance_frontier(&cfg);
        let phi_sites = compute_phi_sites_minimal(&func, &cfg, &df);

        let join_node = cfg.node_by_label("join").unwrap();
        let a0 = Var::PReg(PReg(10));

        let ssa = construct_minimal_ssa(&func);
        println!("=== diamond ===\n{ssa}");

        // a0 は then/else で定義 → join に phi が必要
        assert!(phi_sites[&a0].contains(&join_node));

        // entry, then, else には a0 の phi は不要
        let entry_node = cfg.entry_node().unwrap();
        let then_node = cfg.node_by_label("then").unwrap();
        let else_node = cfg.node_by_label("else").unwrap();
        assert!(!phi_sites[&a0].contains(&entry_node));
        assert!(!phi_sites[&a0].contains(&then_node));
        assert!(!phi_sites[&a0].contains(&else_node));
    }

    /// entry: a0 = 1 → jmp exit
    /// exit:  ret a0
    #[test]
    fn test_linear_no_phi() {
        let func = Function {
            name: "linear".into(),
            params: vec![],
            ret_ty: None,
            blocks: vec![
                Block {
                    label: Label("entry".into()),
                    insts: vec![Inst::Assign {
                        dst: Var::PReg(PReg(10)),
                        ty: Ty::I64,
                        rhs: Rhs::Copy(Value::Imm(1, Ty::I64)),
                    }],
                    term: Term::Jmp(Label("exit".into())),
                },
                Block {
                    label: Label("exit".into()),
                    insts: vec![],
                    term: Term::Ret(vec![Value::VarRef(Var::PReg(PReg(10)), Ty::I64)]),
                },
            ],
        };

        let cfg = IrCfg::build(&func);
        let df = ircfg_to_dominance_frontier(&cfg);
        let phi_sites = compute_phi_sites_minimal(&func, &cfg, &df);

        let a0 = Var::PReg(PReg(10));

        let ssa = construct_minimal_ssa(&func);
        println!("=== linear ===\n{ssa}");

        // 定義が1箇所のみ → phi サイトは空
        assert!(phi_sites[&a0].is_empty());
    }

    /// ループCFG: header に a0 の phi が必要
    ///
    /// entry:  a0 = 0 → jmp header
    /// header: cmp → br body, exit
    /// body:   a0 = a0 + 1 → jmp header
    /// exit:   ret a0
    #[test]
    fn test_loop_phi_at_header() {
        let func = Function {
            name: "loop".into(),
            params: vec![],
            ret_ty: None,
            blocks: vec![
                Block {
                    label: Label("entry".into()),
                    insts: vec![Inst::Assign {
                        dst: Var::PReg(PReg(10)),
                        ty: Ty::I64,
                        rhs: Rhs::Copy(Value::Imm(0, Ty::I64)),
                    }],
                    term: Term::Jmp(Label("header".into())),
                },
                Block {
                    label: Label("header".into()),
                    insts: vec![Inst::Assign {
                        dst: Var::Temp(Temp(0)),
                        ty: Ty::I1,
                        rhs: Rhs::Cmp(
                            CmpPred::Lt,
                            Value::VarRef(Var::PReg(PReg(10)), Ty::I64),
                            Value::Imm(10, Ty::I64),
                        ),
                    }],
                    term: Term::BrCond {
                        cond: Value::VarRef(Var::Temp(Temp(0)), Ty::I1),
                        then_label: Label("body".into()),
                        else_label: Label("exit".into()),
                    },
                },
                Block {
                    label: Label("body".into()),
                    insts: vec![Inst::Assign {
                        dst: Var::PReg(PReg(10)),
                        ty: Ty::I64,
                        rhs: Rhs::BinOp(
                            BinOp::Add,
                            Value::VarRef(Var::PReg(PReg(10)), Ty::I64),
                            Value::Imm(1, Ty::I64),
                        ),
                    }],
                    term: Term::Jmp(Label("header".into())),
                },
                Block {
                    label: Label("exit".into()),
                    insts: vec![],
                    term: Term::Ret(vec![Value::VarRef(Var::PReg(PReg(10)), Ty::I64)]),
                },
            ],
        };

        let cfg = IrCfg::build(&func);
        let df = ircfg_to_dominance_frontier(&cfg);
        let phi_sites = compute_phi_sites_minimal(&func, &cfg, &df);

        let header_node = cfg.node_by_label("header").unwrap();
        let a0 = Var::PReg(PReg(10));

        let ssa = construct_minimal_ssa(&func);
        println!("=== loop ===\n{ssa}");

        // a0 は entry と body で定義 → header に phi が必要
        assert!(phi_sites[&a0].contains(&header_node));
    }

    // ── Pruned SSA tests ──────────────────────────────────────────

    /// Pruned SSA で不要なφが除去されることを確認
    ///
    /// entry: t0 = cmp(a0, 0), a1 = 0 → br then, else
    /// then:  a0 = 1, a1 = 99         → jmp join
    /// else:  a0 = 2                   → jmp join
    /// join:  ret a0
    ///
    /// a1 は entry と then で定義 → join が DF → minimal は join に a1 のφを置く
    /// しかし a1 は join で使われない (live でない) → pruned では除去される
    #[test]
    fn test_pruned_removes_dead_phi() {
        let func = Function {
            name: "pruned_test".into(),
            params: vec![],
            ret_ty: Some(Ty::I64),
            blocks: vec![
                Block {
                    label: Label("entry".into()),
                    insts: vec![
                        Inst::Assign {
                            dst: Var::Temp(Temp(0)),
                            ty: Ty::I1,
                            rhs: Rhs::Cmp(
                                CmpPred::Eq,
                                Value::VarRef(Var::PReg(PReg(10)), Ty::I64),
                                Value::Imm(0, Ty::I64),
                            ),
                        },
                        Inst::Assign {
                            dst: Var::PReg(PReg(11)),
                            ty: Ty::I64,
                            rhs: Rhs::Copy(Value::Imm(0, Ty::I64)),
                        },
                    ],
                    term: Term::BrCond {
                        cond: Value::VarRef(Var::Temp(Temp(0)), Ty::I1),
                        then_label: Label("then".into()),
                        else_label: Label("else".into()),
                    },
                },
                Block {
                    label: Label("then".into()),
                    insts: vec![
                        Inst::Assign {
                            dst: Var::PReg(PReg(10)),
                            ty: Ty::I64,
                            rhs: Rhs::Copy(Value::Imm(1, Ty::I64)),
                        },
                        Inst::Assign {
                            dst: Var::PReg(PReg(11)),
                            ty: Ty::I64,
                            rhs: Rhs::Copy(Value::Imm(99, Ty::I64)),
                        },
                    ],
                    term: Term::Jmp(Label("join".into())),
                },
                Block {
                    label: Label("else".into()),
                    insts: vec![Inst::Assign {
                        dst: Var::PReg(PReg(10)),
                        ty: Ty::I64,
                        rhs: Rhs::Copy(Value::Imm(2, Ty::I64)),
                    }],
                    term: Term::Jmp(Label("join".into())),
                },
                Block {
                    label: Label("join".into()),
                    insts: vec![],
                    term: Term::Ret(vec![Value::VarRef(Var::PReg(PReg(10)), Ty::I64)]),
                },
            ],
        };

        let minimal = construct_minimal_ssa(&func);
        let pruned = construct_pruned_ssa(&func);

        println!("=== minimal ===\n{minimal}");
        println!("=== pruned ===\n{pruned}");

        let minimal_join = minimal.blocks.iter().find(|b| b.label.0 == "join").unwrap();
        let pruned_join = pruned.blocks.iter().find(|b| b.label.0 == "join").unwrap();

        // minimal: a0 と a1 両方のφ
        assert!(
            minimal_join.phis.len() >= 2,
            "minimal should have phi for both a0 and a1, got {}",
            minimal_join.phis.len()
        );

        // pruned: a0 のφのみ (a1 は join で live でない)
        assert_eq!(
            pruned_join.phis.len(),
            1,
            "pruned should only have phi for a0, got {}",
            pruned_join.phis.len()
        );
        assert_eq!(pruned_join.phis[0].dst.base, Var::PReg(PReg(10)));
    }

    /// Pruned SSA でもループの live なφは残る
    ///
    /// entry:  a0 = 0       → jmp header
    /// header: t0 = cmp(a0) → br body, exit
    /// body:   a0 = a0 + 1  → jmp header
    /// exit:   ret a0
    ///
    /// a0 は header で live → pruned でも header のφは残る
    #[test]
    fn test_pruned_keeps_live_phi_in_loop() {
        let func = Function {
            name: "pruned_loop".into(),
            params: vec![],
            ret_ty: Some(Ty::I64),
            blocks: vec![
                Block {
                    label: Label("entry".into()),
                    insts: vec![Inst::Assign {
                        dst: Var::PReg(PReg(10)),
                        ty: Ty::I64,
                        rhs: Rhs::Copy(Value::Imm(0, Ty::I64)),
                    }],
                    term: Term::Jmp(Label("header".into())),
                },
                Block {
                    label: Label("header".into()),
                    insts: vec![Inst::Assign {
                        dst: Var::Temp(Temp(0)),
                        ty: Ty::I1,
                        rhs: Rhs::Cmp(
                            CmpPred::Lt,
                            Value::VarRef(Var::PReg(PReg(10)), Ty::I64),
                            Value::Imm(10, Ty::I64),
                        ),
                    }],
                    term: Term::BrCond {
                        cond: Value::VarRef(Var::Temp(Temp(0)), Ty::I1),
                        then_label: Label("body".into()),
                        else_label: Label("exit".into()),
                    },
                },
                Block {
                    label: Label("body".into()),
                    insts: vec![Inst::Assign {
                        dst: Var::PReg(PReg(10)),
                        ty: Ty::I64,
                        rhs: Rhs::BinOp(
                            BinOp::Add,
                            Value::VarRef(Var::PReg(PReg(10)), Ty::I64),
                            Value::Imm(1, Ty::I64),
                        ),
                    }],
                    term: Term::Jmp(Label("header".into())),
                },
                Block {
                    label: Label("exit".into()),
                    insts: vec![],
                    term: Term::Ret(vec![Value::VarRef(Var::PReg(PReg(10)), Ty::I64)]),
                },
            ],
        };

        let minimal = construct_minimal_ssa(&func);
        let pruned = construct_pruned_ssa(&func);

        println!("=== minimal loop ===\n{minimal}");
        println!("=== pruned loop ===\n{pruned}");

        let minimal_header = minimal
            .blocks
            .iter()
            .find(|b| b.label.0 == "header")
            .unwrap();
        let pruned_header = pruned
            .blocks
            .iter()
            .find(|b| b.label.0 == "header")
            .unwrap();

        // a0 は header で live → minimal も pruned も header に a0 のφを持つ
        let has_a0_phi =
            |phis: &[SsaPhiNode]| phis.iter().any(|phi| phi.dst.base == Var::PReg(PReg(10)));
        assert!(has_a0_phi(&minimal_header.phis));
        assert!(has_a0_phi(&pruned_header.phis));

        // pruned のφ数 <= minimal のφ数
        let total_phis =
            |ssa: &SsaFunction| -> usize { ssa.blocks.iter().map(|b| b.phis.len()).sum() };
        assert!(total_phis(&pruned) <= total_phis(&minimal));
    }

    /// Live なφは全て残ることの確認 (diamond で両方 live)
    ///
    /// entry: t0 = cmp(a0, 0), a1 = 0 → br then, else
    /// then:  a0 = 1, a1 = 99         → jmp join
    /// else:  a0 = 2                   → jmp join
    /// join:  ret a0 + a1
    ///
    /// a0, a1 ともに join で使用 → pruned でも両方のφが残る
    #[test]
    fn test_pruned_keeps_all_live_phis() {
        let func = Function {
            name: "both_live".into(),
            params: vec![],
            ret_ty: Some(Ty::I64),
            blocks: vec![
                Block {
                    label: Label("entry".into()),
                    insts: vec![
                        Inst::Assign {
                            dst: Var::Temp(Temp(0)),
                            ty: Ty::I1,
                            rhs: Rhs::Cmp(
                                CmpPred::Eq,
                                Value::VarRef(Var::PReg(PReg(10)), Ty::I64),
                                Value::Imm(0, Ty::I64),
                            ),
                        },
                        Inst::Assign {
                            dst: Var::PReg(PReg(11)),
                            ty: Ty::I64,
                            rhs: Rhs::Copy(Value::Imm(0, Ty::I64)),
                        },
                    ],
                    term: Term::BrCond {
                        cond: Value::VarRef(Var::Temp(Temp(0)), Ty::I1),
                        then_label: Label("then".into()),
                        else_label: Label("else".into()),
                    },
                },
                Block {
                    label: Label("then".into()),
                    insts: vec![
                        Inst::Assign {
                            dst: Var::PReg(PReg(10)),
                            ty: Ty::I64,
                            rhs: Rhs::Copy(Value::Imm(1, Ty::I64)),
                        },
                        Inst::Assign {
                            dst: Var::PReg(PReg(11)),
                            ty: Ty::I64,
                            rhs: Rhs::Copy(Value::Imm(99, Ty::I64)),
                        },
                    ],
                    term: Term::Jmp(Label("join".into())),
                },
                Block {
                    label: Label("else".into()),
                    insts: vec![Inst::Assign {
                        dst: Var::PReg(PReg(10)),
                        ty: Ty::I64,
                        rhs: Rhs::Copy(Value::Imm(2, Ty::I64)),
                    }],
                    term: Term::Jmp(Label("join".into())),
                },
                Block {
                    label: Label("join".into()),
                    insts: vec![Inst::Assign {
                        dst: Var::PReg(PReg(12)),
                        ty: Ty::I64,
                        rhs: Rhs::BinOp(
                            BinOp::Add,
                            Value::VarRef(Var::PReg(PReg(10)), Ty::I64),
                            Value::VarRef(Var::PReg(PReg(11)), Ty::I64),
                        ),
                    }],
                    term: Term::Ret(vec![Value::VarRef(Var::PReg(PReg(12)), Ty::I64)]),
                },
            ],
        };

        let minimal = construct_minimal_ssa(&func);
        let pruned = construct_pruned_ssa(&func);

        println!("=== minimal both_live ===\n{minimal}");
        println!("=== pruned both_live ===\n{pruned}");

        let pruned_join = pruned.blocks.iter().find(|b| b.label.0 == "join").unwrap();

        // a0, a1 ともに join で使用 → pruned でも両方のφが残る
        let has_phi_for = |var: Var| pruned_join.phis.iter().any(|phi| phi.dst.base == var);
        assert!(
            has_phi_for(Var::PReg(PReg(10))),
            "pruned should keep a0 phi"
        );
        assert!(
            has_phi_for(Var::PReg(PReg(11))),
            "pruned should keep a1 phi"
        );
    }
}
