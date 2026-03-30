use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};
use petgraph::visit::EdgeRef;

use super::block::{BasicBlock, BlockId, EdgeKind};
use super::instruction::{Instruction, InstructionKind};

/// Control flow graph built from a flat sequence of disassembled instructions.
///
/// Uses `StableDiGraph` so that `NodeIndex` values remain valid across
/// node/edge insertions and removals.
pub struct Cfg {
    graph: StableDiGraph<BasicBlock, EdgeKind>,
    addr_to_node: HashMap<u64, NodeIndex>,
}

#[allow(dead_code)]
impl Cfg {
    /// Build a CFG from a flat instruction stream.
    ///
    /// Three-phase algorithm:
    /// 1. Identify leader addresses (basic block start points).
    /// 2. Split the instruction stream at leaders into blocks.
    /// 3. Add blocks as graph nodes, then wire edges based on terminators.
    pub fn build(instructions: Vec<Instruction>) -> Self {
        if instructions.is_empty() {
            return Self {
                graph: StableDiGraph::new(),
                addr_to_node: HashMap::new(),
            };
        }

        let leaders = identify_leaders(&instructions);
        let block_groups = split_into_blocks(instructions, &leaders);

        let mut graph = StableDiGraph::new();
        let mut addr_to_node = HashMap::new();

        // Phase 3a: create nodes
        for insns in block_groups {
            let start_addr = insns[0].address;
            let node = graph.add_node(BasicBlock::new(BlockId(0), insns));
            // Sync BlockId with NodeIndex
            graph[node].id = BlockId(node.index() as u32);
            addr_to_node.insert(start_addr, node);
        }

        // Phase 3b: add edges
        // Collect node indices — StableGraph iteration order is insertion order.
        let nodes: Vec<NodeIndex> = graph.node_indices().collect();

        for &node in &nodes {
            let block = &graph[node];
            let term_kind = *block.terminator_kind();
            let fallthrough_addr = block.fallthrough_address();

            match term_kind {
                InstructionKind::Regular => {
                    // Block was split because the *next* instruction is a leader.
                    // Add a fallthrough edge if there is a next block.
                    if let Some(&next_node) = fallthrough_addr.and_then(|a| addr_to_node.get(&a)) {
                        graph.add_edge(node, next_node, EdgeKind::Fallthrough);
                    }
                }
                InstructionKind::ConditionalBranch { target } => {
                    if let Some(&target_node) = addr_to_node.get(&target) {
                        graph.add_edge(node, target_node, EdgeKind::ConditionalTaken);
                    }
                    if let Some(&next_node) = fallthrough_addr.and_then(|a| addr_to_node.get(&a)) {
                        graph.add_edge(node, next_node, EdgeKind::ConditionalFallthrough);
                    }
                }
                InstructionKind::Jump { target } => {
                    if let Some(&target_node) = addr_to_node.get(&target) {
                        graph.add_edge(node, target_node, EdgeKind::Unconditional);
                    }
                }
                InstructionKind::IndirectJump => {
                    // Target unknown statically — no edges.
                }
                InstructionKind::Call { .. } | InstructionKind::IndirectCall => {
                    // Intraprocedural: calls are not terminators, but a call can
                    // end up as the last instruction when the block is split
                    // because the *next* instruction is a leader for another
                    // reason (e.g. it is a branch target). Treat like Regular.
                    if let Some(&next_node) = fallthrough_addr.and_then(|a| addr_to_node.get(&a)) {
                        graph.add_edge(node, next_node, EdgeKind::Fallthrough);
                    }
                }
                InstructionKind::Return => {
                    // Function exit — no outgoing edges.
                }
                InstructionKind::Syscall => {
                    // Conservative — no outgoing edges.
                }
            }
        }

        Self {
            graph,
            addr_to_node,
        }
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// Number of basic blocks.
    pub fn block_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get a block by its `NodeIndex`.
    pub fn block(&self, idx: NodeIndex) -> Option<&BasicBlock> {
        self.graph.node_weight(idx)
    }

    /// Get a mutable reference to a block by its `NodeIndex`.
    pub fn block_mut(&mut self, idx: NodeIndex) -> Option<&mut BasicBlock> {
        self.graph.node_weight_mut(idx)
    }

    /// Look up a block by its start address.
    pub fn block_by_address(&self, addr: u64) -> Option<(NodeIndex, &BasicBlock)> {
        let &node = self.addr_to_node.get(&addr)?;
        Some((node, &self.graph[node]))
    }

    /// Look up a `NodeIndex` by block start address.
    pub fn node_index_by_address(&self, addr: u64) -> Option<NodeIndex> {
        self.addr_to_node.get(&addr).copied()
    }

    /// Iterate over all blocks.
    pub fn blocks(&self) -> impl Iterator<Item = (NodeIndex, &BasicBlock)> {
        self.graph.node_indices().map(move |n| (n, &self.graph[n]))
    }

    /// Iterate over all node indices.
    pub fn node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.graph.node_indices()
    }

    /// Outgoing edges and successor blocks.
    pub fn successors(
        &self,
        idx: NodeIndex,
    ) -> impl Iterator<Item = (EdgeKind, NodeIndex, &BasicBlock)> {
        self.graph
            .edges(idx)
            .map(move |e| (*e.weight(), e.target(), &self.graph[e.target()]))
    }

    /// Incoming edges and predecessor blocks.
    pub fn predecessors(
        &self,
        idx: NodeIndex,
    ) -> impl Iterator<Item = (EdgeKind, NodeIndex, &BasicBlock)> {
        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .map(move |e| (*e.weight(), e.source(), &self.graph[e.source()]))
    }

    /// The entry block (block with the lowest start address).
    pub fn entry_block(&self) -> Option<(NodeIndex, &BasicBlock)> {
        self.blocks().min_by_key(|(_, b)| b.start_address)
    }

    /// Render this CFG in Graphviz DOT format.
    ///
    /// Designed for debugging/visualization only; the format is stable enough
    /// for ad-hoc use but not considered a formal interface.
    pub fn to_dot(&self) -> String {
        fn escape_label(s: &str) -> String {
            let mut out = String::new();
            for ch in s.chars() {
                match ch {
                    '\"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    _ => out.push(ch),
                }
            }
            out
        }

        let mut out = String::new();
        // Header
        writeln!(&mut out, "digraph cfg {{").unwrap();

        // Nodes
        for (_, block) in self.blocks() {
            let node_name = format!("bb{}", block.id.0);
            let label = format!(
                "{} [0x{:x}..0x{:x}]",
                block.id, block.start_address, block.end_address
            );
            let label = escape_label(&label);
            writeln!(&mut out, "    {} [label=\"{}\"];", node_name, label).unwrap();
        }

        // Edges
        for node in self.node_indices() {
            let from_block = &self.graph[node];
            let from_name = format!("bb{}", from_block.id.0);
            for (kind, succ, _) in self.successors(node) {
                let to_block = &self.graph[succ];
                let to_name = format!("bb{}", to_block.id.0);
                let label = escape_label(&format!("{:?}", kind));
                writeln!(
                    &mut out,
                    "    {} -> {} [label=\"{}\"];",
                    from_name, to_name, label
                )
                .unwrap();
            }
        }

        writeln!(&mut out, "}}").unwrap();
        out
    }

    // ── Graph escape hatch ─────────────────────────────────────────────

    /// Direct access to the underlying `StableDiGraph` for petgraph algorithms
    /// (Dfs, Bfs, dominators, etc.).
    pub fn graph(&self) -> &StableDiGraph<BasicBlock, EdgeKind> {
        &self.graph
    }

    /// Mutable access to the underlying graph.
    ///
    /// **Warning:** direct mutations may desync `addr_to_node`. Prefer the
    /// dedicated mutation methods when possible.
    pub fn graph_mut(&mut self) -> &mut StableDiGraph<BasicBlock, EdgeKind> {
        &mut self.graph
    }

    // ── Mutation (keeps addr_to_node consistent) ───────────────────────

    /// Add a new block from a non-empty instruction list. Returns its `NodeIndex`.
    pub fn add_block(&mut self, instructions: Vec<Instruction>) -> NodeIndex {
        let start_addr = instructions[0].address;
        let node = self
            .graph
            .add_node(BasicBlock::new(BlockId(0), instructions));
        self.graph[node].id = BlockId(node.index() as u32);
        self.addr_to_node.insert(start_addr, node);
        node
    }

    /// Remove a block and all its edges. Returns the `BasicBlock` if it existed.
    pub fn remove_block(&mut self, idx: NodeIndex) -> Option<BasicBlock> {
        let block = self.graph.remove_node(idx)?;
        self.addr_to_node.remove(&block.start_address);
        Some(block)
    }

    /// Add an edge between two nodes.
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, kind: EdgeKind) -> EdgeIndex {
        self.graph.add_edge(from, to, kind)
    }

    /// Remove an edge by its index.
    pub fn remove_edge(&mut self, idx: EdgeIndex) -> Option<EdgeKind> {
        self.graph.remove_edge(idx)
    }
}

impl std::fmt::Display for Cfg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "CFG: {} blocks, {} edges",
            self.block_count(),
            self.edge_count()
        )?;
        for (node, block) in self.blocks() {
            write!(f, "{block}")?;
            for (kind, _, target) in self.successors(node) {
                writeln!(f, "    -> {} ({:?})", target.id, kind)?;
            }
        }
        Ok(())
    }
}

// ── Private helpers ────────────────────────────────────────────────────

/// Identify all leader addresses (basic block start points).
fn identify_leaders(instructions: &[Instruction]) -> HashSet<u64> {
    let valid_addrs: HashSet<u64> = instructions.iter().map(|i| i.address).collect();
    let mut leaders = HashSet::new();

    // First instruction is always a leader.
    leaders.insert(instructions[0].address);

    for (i, insn) in instructions.iter().enumerate() {
        if insn.kind.is_terminator() {
            // Instruction after a terminator is a leader.
            if i + 1 < instructions.len() {
                leaders.insert(instructions[i + 1].address);
            }
            // Branch/jump target is a leader (if within the instruction stream).
            if let Some(target) = insn.kind.static_target() {
                if valid_addrs.contains(&target) {
                    leaders.insert(target);
                }
            }
        }
    }

    leaders
}

/// Split the instruction stream at leader addresses into groups.
fn split_into_blocks(
    instructions: Vec<Instruction>,
    leaders: &HashSet<u64>,
) -> Vec<Vec<Instruction>> {
    let mut blocks: Vec<Vec<Instruction>> = Vec::new();
    let mut current: Vec<Instruction> = Vec::new();

    for insn in instructions {
        if leaders.contains(&insn.address) && !current.is_empty() {
            blocks.push(current);
            current = Vec::new();
        }
        current.push(insn);
    }
    if !current.is_empty() {
        blocks.push(current);
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disassemble::load_and_disassemble;
    use std::path::PathBuf;

    fn testcase_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testcases")
            .join(name)
    }

    fn load_testcase(name: &str) -> Vec<Instruction> {
        load_and_disassemble(testcase_path(name)).unwrap()
    }

    fn build_cfg(name: &str) -> Cfg {
        Cfg::build(load_testcase(name))
    }

    #[test]
    fn test_build_hello_cfg() {
        let cfg = build_cfg("hello.out");
        assert!(cfg.block_count() > 0, "CFG should have at least one block");
        for (_, block) in cfg.blocks() {
            assert!(!block.instructions.is_empty(), "blocks must be non-empty");
        }
    }

    #[test]
    fn test_if_has_conditional_edges() {
        let cfg = build_cfg("if.out");

        let has_cond_pair = cfg.node_indices().any(|n| {
            let succs: Vec<EdgeKind> = cfg.successors(n).map(|(k, _, _)| k).collect();
            succs.contains(&EdgeKind::ConditionalTaken)
                && succs.contains(&EdgeKind::ConditionalFallthrough)
        });
        assert!(
            has_cond_pair,
            "if.out CFG should have a block with both ConditionalTaken and ConditionalFallthrough edges"
        );
    }

    #[test]
    fn test_for_has_back_edge() {
        let cfg = build_cfg("for.out");

        let has_unconditional = cfg.node_indices().any(|n| {
            cfg.successors(n)
                .any(|(k, _, _)| k == EdgeKind::Unconditional)
        });
        assert!(
            has_unconditional,
            "for.out CFG should have an unconditional jump edge (loop back-edge)"
        );
    }

    #[test]
    fn test_block_id_matches_node_index() {
        let cfg = build_cfg("hello.out");

        for (node, block) in cfg.blocks() {
            assert_eq!(
                block.id,
                BlockId(node.index() as u32),
                "BlockId should match NodeIndex for block at {:#x}",
                block.start_address
            );
        }
    }

    #[test]
    fn test_addr_to_node_consistency() {
        let cfg = build_cfg("hello.out");

        for (node, block) in cfg.blocks() {
            let looked_up = cfg.node_index_by_address(block.start_address);
            assert_eq!(
                looked_up,
                Some(node),
                "addr_to_node should map {:#x} to the correct NodeIndex",
                block.start_address
            );
        }
    }

    #[test]
    fn test_remove_block() {
        let mut cfg = build_cfg("hello.out");
        let (node, block) = cfg.entry_block().unwrap();
        let addr = block.start_address;

        let removed = cfg.remove_block(node);
        assert!(removed.is_some(), "remove_block should return the block");
        assert!(cfg.block(node).is_none(), "block should be gone from graph");
        assert!(
            cfg.node_index_by_address(addr).is_none(),
            "addr_to_node should be cleared"
        );
    }

    #[test]
    fn test_entry_block_is_lowest_address() {
        let cfg = build_cfg("hello.out");
        let (_, entry) = cfg.entry_block().unwrap();

        let min_addr = cfg.blocks().map(|(_, b)| b.start_address).min().unwrap();
        assert_eq!(
            entry.start_address, min_addr,
            "entry block should have the lowest start address"
        );
    }

    #[test]
    fn test_empty_instructions() {
        let cfg = Cfg::build(vec![]);
        assert_eq!(cfg.block_count(), 0);
        assert_eq!(cfg.edge_count(), 0);
        assert!(cfg.entry_block().is_none());
    }

    #[test]
    fn test_graph_escape_hatch_dfs() {
        let cfg = build_cfg("hello.out");
        let (entry_node, _) = cfg.entry_block().unwrap();

        let mut dfs = petgraph::visit::Dfs::new(cfg.graph(), entry_node);
        let mut visited = 0;
        while dfs.next(cfg.graph()).is_some() {
            visited += 1;
        }
        assert!(visited > 0, "DFS should visit at least one node");
    }

    #[test]
    fn test_to_dot_for_out_basic() {
        let cfg = build_cfg("for.out");
        let dot = cfg.to_dot();

        assert!(
            dot.starts_with("digraph cfg"),
            "DOT output should start with graph header"
        );
        assert!(
            dot.contains("->"),
            "DOT output should contain at least one edge"
        );
        assert!(
            dot.contains("bb0"),
            "DOT output should contain at least one basic block node"
        );
    }
}
