use std::collections::{HashMap, HashSet};

use petgraph::{
    graph::NodeIndex,
    visit::{DfsPostOrder, Walker},
};

use crate::{
    llir::{Block, Var, cfg::IrCfg},
    ssa::def_use::{inst_to_defs, inst_to_uses, term_to_uses},
};

fn compute_def_use_in_block(block: &Block) -> (HashSet<Var>, HashSet<Var>) {
    let mut def = HashSet::new();
    let mut use_ = HashSet::new();

    for inst in &block.insts {
        for v in inst_to_uses(inst) {
            if !def.contains(&v) {
                use_.insert(v);
            }
        }

        for v in inst_to_defs(inst) {
            def.insert(v);
        }
    }

    for v in term_to_uses(&block.term) {
        if !def.contains(&v) {
            use_.insert(v);
        }
    }

    (def, use_)
}

pub fn compute_live_in(cfg: &IrCfg) -> HashMap<NodeIndex, HashSet<Var>> {
    // (NodeIndex, (Defs, Uses))
    let block_info = cfg
        .blocks()
        .map(|(node, _, block)| {
            let (def, use_) = compute_def_use_in_block(block);
            (node, (def, use_))
        })
        .collect::<HashMap<NodeIndex, (HashSet<Var>, HashSet<Var>)>>();

    let mut live_in: HashMap<NodeIndex, HashSet<Var>> = cfg
        .node_indices()
        .map(|node| (node, HashSet::new()))
        .collect();
    let mut live_out: HashMap<NodeIndex, HashSet<Var>> = cfg
        .node_indices()
        .map(|node| (node, HashSet::new()))
        .collect();

    let post_order: Vec<NodeIndex> = DfsPostOrder::new(
        cfg.graph(),
        cfg.entry_node().expect("CFG must have an entry node"),
    )
    .iter(cfg.graph())
    .collect();

    loop {
        let mut changed = false;

        for &node in &post_order {
            // live_out[B] = ∪ { live_in[S] | S ∈ successors(B) }
            let new_live_out = cfg
                .successors(node)
                .flat_map(|(_, succ, _)| live_in.get(&succ).cloned().unwrap_or_default())
                .collect::<HashSet<_>>();

            let (block_defs, block_uses) = block_info.get(&node).unwrap();

            // live_in[B] = uses[B] ∪ (live_out[B] - defs[B])
            let new_live_in = block_uses
                .union(
                    &new_live_out
                        .difference(block_defs)
                        .copied()
                        .collect::<HashSet<Var>>(),
                )
                .copied()
                .collect::<HashSet<Var>>();

            if live_in[&node] != new_live_in {
                changed = true;
            }

            live_in.insert(node, new_live_in);
            live_out.insert(node, new_live_out);
        }

        if !changed {
            break;
        }
    }

    live_in
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llir::{
        BinOp, Block, CmpPred, Function, Inst, Label, PReg, Rhs, Temp, Term, Ty, Value, Var,
    };

    /// entry: t0 = cmp(a0, 0) → br then, else
    /// then:  a0 = 1          → jmp join
    /// else:  (empty)         → jmp join
    /// join:  ret a0
    #[test]
    fn test_diamond_liveness() {
        let func = Function {
            name: "diamond".into(),
            params: vec![],
            ret_ty: Some(Ty::I64),
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
                    insts: vec![],
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
        let live_in = compute_live_in(&cfg);

        let a0 = Var::PReg(PReg(10));
        let t0 = Var::Temp(Temp(0));

        let entry = cfg.node_by_label("entry").unwrap();
        let then_node = cfg.node_by_label("then").unwrap();
        let else_node = cfg.node_by_label("else").unwrap();
        let join = cfg.node_by_label("join").unwrap();

        // entry: a0 は cmp で使用 → live_in
        assert!(live_in[&entry].contains(&a0));
        // t0 は entry 内で定義→使用 → live_in には入らない
        assert!(!live_in[&entry].contains(&t0));

        // then: a0 を再定義する → live_in に a0 は入らない
        assert!(!live_in[&then_node].contains(&a0));

        // else: a0 を定義しないが join で使う → live_in に a0
        assert!(live_in[&else_node].contains(&a0));

        // join: ret a0 → live_in に a0
        assert!(live_in[&join].contains(&a0));
    }

    /// entry: a0 = 0       → jmp header
    /// header: t0 = cmp(a0, 10) → br body, exit
    /// body:   a0 = a0 + 1 → jmp header
    /// exit:   ret a0
    #[test]
    fn test_loop_liveness() {
        let func = Function {
            name: "loop".into(),
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

        let cfg = IrCfg::build(&func);
        let live_in = compute_live_in(&cfg);

        let a0 = Var::PReg(PReg(10));
        let t0 = Var::Temp(Temp(0));

        let entry = cfg.node_by_label("entry").unwrap();
        let header = cfg.node_by_label("header").unwrap();
        let body = cfg.node_by_label("body").unwrap();
        let exit = cfg.node_by_label("exit").unwrap();

        // entry: a0 を定義するだけ → live_in に a0 は入らない
        assert!(!live_in[&entry].contains(&a0));

        // header: cmp で a0 を使用 → live_in
        assert!(live_in[&header].contains(&a0));
        // t0 は header 内で定義→使用 → live_in には入らない
        assert!(!live_in[&header].contains(&t0));

        // body: a0 + 1 で a0 を使用(定義前) → live_in
        assert!(live_in[&body].contains(&a0));

        // exit: ret a0 → live_in
        assert!(live_in[&exit].contains(&a0));
    }

    /// entry: a0 = 1 → jmp exit
    /// exit:  ret a0
    #[test]
    fn test_linear_liveness() {
        let func = Function {
            name: "linear".into(),
            params: vec![],
            ret_ty: Some(Ty::I64),
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
        let live_in = compute_live_in(&cfg);

        let a0 = Var::PReg(PReg(10));

        let entry = cfg.node_by_label("entry").unwrap();
        let exit = cfg.node_by_label("exit").unwrap();

        // entry: a0 を定義する → live_in に a0 は入らない
        assert!(!live_in[&entry].contains(&a0));

        // exit: ret a0 → live_in に a0
        assert!(live_in[&exit].contains(&a0));
    }
}
