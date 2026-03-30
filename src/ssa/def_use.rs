use std::collections::HashSet;

use crate::llir::{Callee, Function, Inst, Rhs, Term, Value, Var};

pub fn all_vars(func: &Function) -> HashSet<Var> {
    let mut vars = HashSet::new();
    for block in &func.blocks {
        for inst in &block.insts {
            vars.extend(inst_to_defs(inst));
            vars.extend(inst_to_uses(inst));
        }
        vars.extend(term_to_uses(&block.term));
    }
    vars
}

pub fn def_to_blocks(func: &Function, var: Var) -> Vec<usize> {
    let mut blocks = Vec::new();
    for (i, block) in func.blocks.iter().enumerate() {
        if block
            .insts
            .iter()
            .any(|inst| inst_to_defs(inst).contains(&var))
        {
            blocks.push(i);
        }
    }
    blocks
}

pub fn inst_to_defs(inst: &Inst) -> Vec<Var> {
    match inst {
        Inst::Assign { dst, .. } => {
            vec![*dst]
        }
        Inst::Call {
            rets,
            callee: _,
            args: _,
            clobber,
        } => {
            let ret_vars: Vec<Var> = rets.iter().map(|(var, _)| *var).collect();
            let clobber_vars: Vec<Var> = clobber.iter().map(|reg| Var::PReg(*reg)).collect();
            ret_vars.into_iter().chain(clobber_vars).collect()
        }
        Inst::Store { .. } => {
            vec![]
        }
        Inst::Trap(_) => {
            vec![]
        }
    }
}

pub fn inst_to_uses(inst: &Inst) -> Vec<Var> {
    match inst {
        Inst::Assign { dst: _, ty: _, rhs } => rhs_to_uses(rhs),
        Inst::Store { addr, val, .. } => value_to_vars(addr)
            .into_iter()
            .chain(value_to_vars(val))
            .collect(),
        Inst::Call {
            rets: _,
            callee,
            args,
            clobber: _,
        } => {
            let mut use_vars = vec![];
            if let Callee::Indirect(val) = callee {
                use_vars.extend(value_to_vars(val));
            }
            for arg in args {
                use_vars.extend(value_to_vars(arg));
            }
            use_vars
        }
        _ => {
            vec![]
        }
    }
}

pub fn term_to_uses(term: &Term) -> Vec<Var> {
    match term {
        Term::BrCond {
            cond,
            then_label: _,
            else_label: _,
        } => value_to_vars(cond),
        Term::Jmp(_) => {
            vec![]
        }
        Term::Ret(vals) => vals.iter().flat_map(value_to_vars).collect(),
        Term::Trap(_) => {
            vec![]
        }
        Term::Switch { value, .. } => value_to_vars(value),
    }
}

fn rhs_to_uses(rhs: &Rhs) -> Vec<Var> {
    match rhs {
        Rhs::BinOp(_, a, b) | Rhs::Cmp(_, a, b) => value_to_vars(a)
            .into_iter()
            .chain(value_to_vars(b))
            .collect(),
        Rhs::Cast(_, val, _) => value_to_vars(val),
        Rhs::Load(_, addr) => value_to_vars(addr),
        Rhs::Copy(val) => value_to_vars(val),
    }
}

fn value_to_vars(value: &Value) -> Vec<Var> {
    match value {
        Value::VarRef(var, _) => {
            vec![*var]
        }
        Value::Imm(_, _) => {
            vec![]
        }
        Value::Undef(_) => {
            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::llir::{BinOp, Block, CmpPred, Label, PReg, Temp, Ty};

    use super::*;

    #[test]
    fn test_inst_to_defs() {
        let inst = Inst::Assign {
            dst: Var::Temp(Temp(0)),
            ty: Ty::I1,
            rhs: Rhs::Cmp(
                CmpPred::Eq,
                Value::VarRef(Var::PReg(PReg(10)), Ty::I64),
                Value::VarRef(Var::PReg(PReg(11)), Ty::I64),
            ),
        };
        assert_eq!(inst_to_defs(&inst), vec![Var::Temp(Temp(0))]);
    }

    #[test]
    fn test_inst_to_uses() {
        let inst = Inst::Assign {
            dst: Var::Temp(Temp(0)),
            ty: Ty::I1,
            rhs: Rhs::Cmp(
                CmpPred::Eq,
                Value::VarRef(Var::PReg(PReg(10)), Ty::I64),
                Value::VarRef(Var::PReg(PReg(11)), Ty::I64),
            ),
        };
        assert_eq!(
            inst_to_uses(&inst),
            vec![Var::PReg(PReg(10)), Var::PReg(PReg(11))]
        );
    }

    #[test]
    fn test_term_to_uses() {
        let term = Term::BrCond {
            cond: Value::VarRef(Var::PReg(PReg(10)), Ty::I1),
            then_label: Label("then".into()),
            else_label: Label("else".into()),
        };
        assert_eq!(term_to_uses(&term), vec![Var::PReg(PReg(10))]);
    }

    #[test]
    fn test_all_vars() {
        let block1 = Block {
            label: Label("entry".into()),
            insts: vec![Inst::Assign {
                dst: Var::Temp(Temp(0)),
                ty: Ty::I1,
                rhs: Rhs::Cmp(
                    CmpPred::Eq,
                    Value::VarRef(Var::PReg(PReg(10)), Ty::I64),
                    Value::VarRef(Var::PReg(PReg(11)), Ty::I64),
                ),
            }],
            term: Term::BrCond {
                cond: Value::VarRef(Var::Temp(Temp(0)), Ty::I1),
                then_label: Label("then".into()),
                else_label: Label("else".into()),
            },
        };

        let block2 = Block {
            label: Label("then".into()),
            insts: vec![],
            term: Term::Jmp(Label("join".into())),
        };

        let block3 = Block {
            label: Label("join".into()),
            insts: vec![Inst::Assign {
                dst: Var::PReg(PReg(15)),
                ty: Ty::I64,
                rhs: Rhs::BinOp(
                    BinOp::Add,
                    Value::VarRef(Var::PReg(PReg(10)), Ty::I64),
                    Value::VarRef(Var::PReg(PReg(11)), Ty::I64),
                ),
            }],
            term: Term::Ret(vec![]),
        };

        let func = Function {
            name: "test".into(),
            params: vec![],
            ret_ty: None,
            blocks: vec![block1, block2, block3],
        };

        assert_eq!(
            all_vars(&func),
            vec![
                Var::Temp(Temp(0)),
                Var::PReg(PReg(10)),
                Var::PReg(PReg(11)),
                Var::PReg(PReg(15)),
            ]
            .into_iter()
            .collect()
        );
    }

    #[test]
    fn test_def_to_blocks() {
        let block = Block {
            label: Label("entry".into()),
            insts: vec![Inst::Assign {
                dst: Var::PReg(PReg(10)),
                ty: Ty::I64,
                rhs: Rhs::BinOp(
                    BinOp::Add,
                    Value::Imm(100, Ty::I64),
                    Value::Imm(200, Ty::I64),
                ),
            }],
            term: Term::Jmp(Label("then".into())),
        };

        let func = Function {
            name: "test".into(),
            params: vec![],
            ret_ty: None,
            blocks: vec![block],
        };

        assert_eq!(def_to_blocks(&func, Var::PReg(PReg(10))), vec![0]);
    }
}
