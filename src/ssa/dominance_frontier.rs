use std::collections::{HashMap, HashSet};

use petgraph::graph::NodeIndex;

use crate::llir::cfg::IrCfg;

/// Implementation of Cooper-Harvey-Kennedy algorithm for computing the dominance frontier.
pub fn ircfg_to_dominance_frontier(ircfg: &IrCfg) -> HashMap<NodeIndex, HashSet<NodeIndex>> {
    let graph = ircfg.graph();
    let dominators = petgraph::algo::dominators::simple_fast(
        graph,
        ircfg.entry_node().expect("CFG must have an entry node"),
    );

    let mut frontier = HashMap::new();
    for node in ircfg.graph().node_indices() {
        let predecessors: Vec<_> = IrCfg::predecessors(&ircfg, node).collect();
        if predecessors.len() < 2 {
            continue;
        }
        let idom = dominators
            .immediate_dominator(node)
            .expect("join node must have a idom");
        for (_, predecessor, _) in predecessors {
            let mut runner = predecessor;
            while runner != idom {
                frontier
                    .entry(runner)
                    .or_insert(HashSet::new())
                    .insert(node);
                if let Some(immediate_dominator) = dominators.immediate_dominator(runner) {
                    runner = immediate_dominator;
                } else {
                    break;
                }
            }
        }
    }
    frontier
}

#[cfg(test)]
mod tests {
    use crate::llir::{
        Block, CmpPred, Function, Inst, Label, PReg, Rhs, Temp, Term, Ty, Value, Var,
    };

    use super::*;

    #[test]
    fn test_ircfg_to_dominance_frontier() {
        let entry = Block {
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

        let then = Block {
            label: Label("then".into()),
            insts: vec![],
            term: Term::Jmp(Label("join".into())),
        };

        let else_block = Block {
            label: Label("else".into()),
            insts: vec![],
            term: Term::Jmp(Label("join".into())),
        };

        let join = Block {
            label: Label("join".into()),
            insts: vec![],
            term: Term::Ret(vec![]),
        };

        let func = Function {
            name: "test".into(),
            params: vec![],
            ret_ty: None,
            blocks: vec![entry, then, else_block, join],
        };

        let cfg = IrCfg::build(&func);
        let frontier = ircfg_to_dominance_frontier(&cfg);

        let then_node = cfg.node_by_label("then").unwrap();
        let else_node = cfg.node_by_label("else").unwrap();
        let join_node = cfg.node_by_label("join").unwrap();
        let entry_node = cfg.entry_node().unwrap();

        assert_eq!(frontier.get(&then_node), Some(&HashSet::from([join_node])));
        assert_eq!(frontier.get(&else_node), Some(&HashSet::from([join_node])));
        assert!(frontier.get(&entry_node).is_none());
        assert!(frontier.get(&join_node).is_none());
    }
}
