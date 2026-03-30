use std::collections::HashMap;
use std::fmt::Write;

use petgraph::Direction;
use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::visit::{DfsPostOrder, EdgeRef, Walker};

use super::{Block, Function, Term};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrEdgeKind {
    Unconditional,
    Then,
    Else,
    Case(i64),
    Default,
}

pub struct IrCfg<'a> {
    func: &'a Function,
    graph: StableDiGraph<usize, IrEdgeKind>,
    label_to_node: HashMap<String, NodeIndex>,
    index_to_node: Vec<NodeIndex>,
}

impl<'a> IrCfg<'a> {
    pub fn build(func: &'a Function) -> Self {
        let mut graph = StableDiGraph::new();
        let mut label_to_node = HashMap::new();
        let mut index_to_node = Vec::with_capacity(func.blocks.len());

        // Phase 1: create nodes
        for (i, block) in func.blocks.iter().enumerate() {
            let node = graph.add_node(i);
            label_to_node.insert(block.label.0.clone(), node);
            index_to_node.push(node);
        }

        // Phase 2: add edges
        for (i, block) in func.blocks.iter().enumerate() {
            let node = index_to_node[i];
            match &block.term {
                Term::Jmp(label) => {
                    if let Some(&target) = label_to_node.get(&label.0) {
                        graph.add_edge(node, target, IrEdgeKind::Unconditional);
                    }
                }
                Term::BrCond {
                    then_label,
                    else_label,
                    ..
                } => {
                    if let Some(&target) = label_to_node.get(&then_label.0) {
                        graph.add_edge(node, target, IrEdgeKind::Then);
                    }
                    if let Some(&target) = label_to_node.get(&else_label.0) {
                        graph.add_edge(node, target, IrEdgeKind::Else);
                    }
                }
                Term::Switch { cases, default, .. } => {
                    for (val, label) in cases {
                        if let Some(&target) = label_to_node.get(&label.0) {
                            graph.add_edge(node, target, IrEdgeKind::Case(*val));
                        }
                    }
                    if let Some(&target) = label_to_node.get(&default.0) {
                        graph.add_edge(node, target, IrEdgeKind::Default);
                    }
                }
                Term::Ret(_) | Term::Trap(_) => {}
            }
        }

        Self {
            func,
            graph,
            label_to_node,
            index_to_node,
        }
    }

    // ── Accessors ──────────────────────────────────────────────────

    pub fn block_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn block(&self, idx: NodeIndex) -> &'a Block {
        let block_idx = self.graph[idx];
        &self.func.blocks[block_idx]
    }

    pub fn block_index(&self, idx: NodeIndex) -> usize {
        self.graph[idx]
    }

    pub fn node_by_label(&self, label: &str) -> Option<NodeIndex> {
        self.label_to_node.get(label).copied()
    }

    pub fn entry_node(&self) -> Option<NodeIndex> {
        self.index_to_node.first().copied()
    }

    pub fn node_by_index(&self, block_index: usize) -> Option<NodeIndex> {
        self.index_to_node.get(block_index).copied()
    }

    // ── Iteration ──────────────────────────────────────────────────

    pub fn blocks(&self) -> impl Iterator<Item = (NodeIndex, usize, &'a Block)> + '_ {
        self.graph
            .node_indices()
            .map(move |n| (n, self.graph[n], &self.func.blocks[self.graph[n]]))
    }

    pub fn node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.graph.node_indices()
    }

    pub fn successors(
        &self,
        idx: NodeIndex,
    ) -> impl Iterator<Item = (IrEdgeKind, NodeIndex, &'a Block)> + '_ {
        self.graph.edges(idx).map(move |e| {
            (
                *e.weight(),
                e.target(),
                &self.func.blocks[self.graph[e.target()]],
            )
        })
    }

    pub fn predecessors(
        &self,
        idx: NodeIndex,
    ) -> impl Iterator<Item = (IrEdgeKind, NodeIndex, &'a Block)> + '_ {
        self.graph
            .edges_directed(idx, Direction::Incoming)
            .map(move |e| {
                (
                    *e.weight(),
                    e.source(),
                    &self.func.blocks[self.graph[e.source()]],
                )
            })
    }

    // ── Algorithms ─────────────────────────────────────────────────

    pub fn reverse_post_order(&self) -> Vec<NodeIndex> {
        let Some(entry) = self.entry_node() else {
            return vec![];
        };
        let mut rpo: Vec<NodeIndex> = DfsPostOrder::new(&self.graph, entry)
            .iter(&self.graph)
            .collect();
        rpo.reverse();
        rpo
    }

    // ── Visualization ──────────────────────────────────────────────

    pub fn to_dot(&self) -> String {
        fn escape(s: &str) -> String {
            let mut out = String::new();
            for ch in s.chars() {
                match ch {
                    '\"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\l"),
                    _ => out.push(ch),
                }
            }
            out
        }

        let mut out = String::new();
        writeln!(&mut out, "digraph \"{}\" {{", escape(&self.func.name)).unwrap();
        writeln!(&mut out, "    node [shape=record];").unwrap();

        for (node, _, block) in self.blocks() {
            let header = format!("{}:", block.label);
            let mut body = String::new();
            for inst in &block.insts {
                writeln!(&mut body, "  {inst}").unwrap();
            }
            writeln!(&mut body, "  {}", block.term).unwrap();
            let label = format!("{}\\l{}", escape(&header), escape(&body));
            writeln!(&mut out, "    n{} [label=\"{}\"];", node.index(), label).unwrap();
        }

        for node in self.node_indices() {
            for (kind, succ, _) in self.successors(node) {
                let label = match kind {
                    IrEdgeKind::Unconditional => "jmp".to_string(),
                    IrEdgeKind::Then => "then".to_string(),
                    IrEdgeKind::Else => "else".to_string(),
                    IrEdgeKind::Case(v) => format!("case {v}"),
                    IrEdgeKind::Default => "default".to_string(),
                };
                writeln!(
                    &mut out,
                    "    n{} -> n{} [label=\"{}\"];",
                    node.index(),
                    succ.index(),
                    label
                )
                .unwrap();
            }
        }

        writeln!(&mut out, "}}").unwrap();
        out
    }

    // ── Escape hatch ───────────────────────────────────────────────

    pub fn graph(&self) -> &StableDiGraph<usize, IrEdgeKind> {
        &self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llir::{Block, CmpPred, Function, Inst, Label, PReg, Rhs, Term, Ty, Value, Var};

    // ── Helper ─────────────────────────────────────────────────────

    fn jmp_block(label: &str, target: &str) -> Block {
        Block {
            label: Label(label.into()),
            insts: vec![],
            term: Term::Jmp(Label(target.into())),
        }
    }

    fn ret_block(label: &str) -> Block {
        Block {
            label: Label(label.into()),
            insts: vec![],
            term: Term::Ret(vec![]),
        }
    }

    fn trap_block(label: &str) -> Block {
        Block {
            label: Label(label.into()),
            insts: vec![],
            term: Term::Trap("halt".into()),
        }
    }

    fn br_block(label: &str, then_label: &str, else_label: &str) -> Block {
        Block {
            label: Label(label.into()),
            insts: vec![Inst::Assign {
                dst: Var::PReg(PReg(10)),
                ty: Ty::I1,
                rhs: Rhs::Cmp(CmpPred::Eq, Value::Imm(0, Ty::I64), Value::Imm(0, Ty::I64)),
            }],
            term: Term::BrCond {
                cond: Value::VarRef(Var::PReg(PReg(10)), Ty::I1),
                then_label: Label(then_label.into()),
                else_label: Label(else_label.into()),
            },
        }
    }

    fn simple_func(name: &str, blocks: Vec<Block>) -> Function {
        Function {
            name: name.into(),
            params: vec![],
            ret_ty: None,
            blocks,
        }
    }

    // ── Unit tests ─────────────────────────────────────────────────

    #[test]
    fn test_linear_cfg() {
        let func = simple_func(
            "f",
            vec![jmp_block("a", "b"), jmp_block("b", "c"), ret_block("c")],
        );
        let cfg = IrCfg::build(&func);
        assert_eq!(cfg.block_count(), 3);
        assert_eq!(cfg.edge_count(), 2);
        for (kind, _, _) in cfg.successors(cfg.entry_node().unwrap()) {
            assert_eq!(kind, IrEdgeKind::Unconditional);
        }
    }

    #[test]
    fn test_diamond_cfg() {
        //   entry
        //   /   \
        // then  else
        //   \   /
        //   join
        let func = simple_func(
            "f",
            vec![
                br_block("entry", "then", "else"),
                jmp_block("then", "join"),
                jmp_block("else", "join"),
                ret_block("join"),
            ],
        );
        let cfg = IrCfg::build(&func);
        assert_eq!(cfg.block_count(), 4);
        // entry->then (Then), entry->else (Else), then->join, else->join = 4 edges
        assert_eq!(cfg.edge_count(), 4);

        let entry = cfg.entry_node().unwrap();
        let succs: Vec<IrEdgeKind> = cfg.successors(entry).map(|(k, _, _)| k).collect();
        assert!(succs.contains(&IrEdgeKind::Then));
        assert!(succs.contains(&IrEdgeKind::Else));
    }

    #[test]
    fn test_ret_no_edges() {
        let func = simple_func("f", vec![ret_block("exit")]);
        let cfg = IrCfg::build(&func);
        assert_eq!(cfg.block_count(), 1);
        assert_eq!(cfg.edge_count(), 0);
        let n = cfg.entry_node().unwrap();
        assert_eq!(cfg.successors(n).count(), 0);
    }

    #[test]
    fn test_trap_no_edges() {
        let func = simple_func("f", vec![trap_block("trap")]);
        let cfg = IrCfg::build(&func);
        assert_eq!(cfg.block_count(), 1);
        assert_eq!(cfg.edge_count(), 0);
        let n = cfg.entry_node().unwrap();
        assert_eq!(cfg.successors(n).count(), 0);
    }

    #[test]
    fn test_entry_node() {
        let func = simple_func("f", vec![jmp_block("first", "second"), ret_block("second")]);
        let cfg = IrCfg::build(&func);
        let entry = cfg.entry_node().unwrap();
        assert_eq!(cfg.block(entry).label.0, "first");
        assert_eq!(cfg.block_index(entry), 0);
    }

    #[test]
    fn test_predecessors() {
        let func = simple_func(
            "f",
            vec![
                br_block("entry", "then", "else"),
                jmp_block("then", "join"),
                jmp_block("else", "join"),
                ret_block("join"),
            ],
        );
        let cfg = IrCfg::build(&func);
        let join = cfg.node_by_label("join").unwrap();
        assert_eq!(cfg.predecessors(join).count(), 2);
    }

    #[test]
    fn test_reverse_post_order() {
        let func = simple_func(
            "f",
            vec![
                br_block("entry", "then", "else"),
                jmp_block("then", "join"),
                jmp_block("else", "join"),
                ret_block("join"),
            ],
        );
        let cfg = IrCfg::build(&func);
        let rpo = cfg.reverse_post_order();
        assert_eq!(rpo.len(), 4);
        // entry must come first
        assert_eq!(cfg.block(rpo[0]).label.0, "entry");
        // join must come last (both then and else converge to it)
        assert_eq!(cfg.block(rpo[3]).label.0, "join");
    }

    #[test]
    fn test_unreachable_label_ignored() {
        let func = simple_func(
            "f",
            vec![Block {
                label: Label("entry".into()),
                insts: vec![],
                term: Term::Jmp(Label("nonexistent".into())),
            }],
        );
        let cfg = IrCfg::build(&func);
        assert_eq!(cfg.block_count(), 1);
        assert_eq!(cfg.edge_count(), 0);
    }

    // ── Integration tests (require testcase binaries) ──────────────

    #[cfg(test)]
    mod integration {
        use super::*;
        use crate::disassemble::load_elf_analysis;
        use crate::llir::lower::lower_functions;
        use std::path::PathBuf;

        fn testcase_path(name: &str) -> PathBuf {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("testcases")
                .join(name)
        }

        fn lower_main(name: &str) -> Function {
            let analysis = load_elf_analysis(testcase_path(name)).unwrap();
            let program = lower_functions(&analysis.instructions, &analysis.functions);
            program
                .functions
                .into_iter()
                .find(|f| f.name == "main")
                .expect("main not found")
        }

        #[test]
        fn test_if_out_has_branch() {
            let func = lower_main("if.out");
            let cfg = IrCfg::build(&func);
            let has_then_else = cfg.node_indices().any(|n| {
                let kinds: Vec<IrEdgeKind> = cfg.successors(n).map(|(k, _, _)| k).collect();
                kinds.contains(&IrEdgeKind::Then) && kinds.contains(&IrEdgeKind::Else)
            });
            assert!(has_then_else, "if.out CFG should have Then/Else edge pair");
        }

        #[test]
        fn test_for_out_has_back_edge() {
            let func = lower_main("for.out");
            let cfg = IrCfg::build(&func);
            // A back-edge exists if some successor has a lower block index than the source
            let has_back = cfg.node_indices().any(|n| {
                let src_idx = cfg.block_index(n);
                cfg.successors(n)
                    .any(|(_, succ, _)| cfg.block_index(succ) <= src_idx)
            });
            assert!(has_back, "for.out CFG should have a back-edge (loop)");
        }

        #[test]
        fn test_hello_out_dot() {
            let func = lower_main("hello.out");
            let cfg = IrCfg::build(&func);
            let dot = cfg.to_dot();
            assert!(dot.contains("digraph"), "DOT should start with digraph");
            assert!(dot.contains("n0"), "DOT should contain at least one node");
        }
    }
}
