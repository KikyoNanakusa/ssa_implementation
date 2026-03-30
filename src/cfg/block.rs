use super::instruction::{Instruction, InstructionKind};

/// Unique identifier for a basic block within a CFG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// The type of control flow edge between basic blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub enum EdgeKind {
    /// Sequential fall-through to the next block.
    Fallthrough,
    /// Conditional branch taken.
    ConditionalTaken,
    /// Conditional branch not taken (fall-through edge).
    ConditionalFallthrough,
    /// Unconditional jump.
    Unconditional,
    /// Function call edge.
    Call,
    /// Return edge.
    Return,
}

/// A basic block: a maximal sequence of instructions with a single entry
/// and a single exit (the last instruction, which is a terminator).
///
/// Designed as the node weight for `petgraph::graph::DiGraph<BasicBlock, EdgeKind>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    /// Unique identifier for this block.
    pub id: BlockId,

    /// Optional human-readable label for this block (for debugging/visualization).
    pub label: Option<String>,

    /// Address of the first instruction.
    pub start_address: u64,

    /// Address immediately after the last instruction.
    pub end_address: u64,

    /// Ordered sequence of instructions. Non-empty; only the last may be a terminator.
    pub instructions: Vec<Instruction>,
}

#[allow(dead_code)]
impl BasicBlock {
    /// Create a new basic block from a non-empty vector of instructions.
    ///
    /// # Panics
    ///
    /// Panics if `instructions` is empty.
    pub fn new(id: BlockId, instructions: Vec<Instruction>) -> Self {
        assert!(
            !instructions.is_empty(),
            "BasicBlock must have at least one instruction"
        );
        let start_address = instructions.first().unwrap().address;
        let end_address = instructions.last().unwrap().end_address();
        Self {
            id,
            label: None,
            start_address,
            end_address,
            instructions,
        }
    }

    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Returns the last instruction (the terminator).
    pub fn terminator(&self) -> &Instruction {
        self.instructions.last().unwrap()
    }

    /// Returns the control flow kind of the terminator.
    pub fn terminator_kind(&self) -> &InstructionKind {
        &self.terminator().kind
    }

    /// Number of instructions in this block.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Total byte size of all instructions.
    pub fn byte_size(&self) -> u64 {
        self.end_address - self.start_address
    }

    /// Iterator over instructions.
    pub fn iter(&self) -> std::slice::Iter<'_, Instruction> {
        self.instructions.iter()
    }

    /// Fall-through address, if the terminator supports fall-through.
    pub fn fallthrough_address(&self) -> Option<u64> {
        if self.terminator_kind().has_fallthrough() {
            Some(self.end_address)
        } else {
            None
        }
    }

    /// Statically-known successor addresses.
    pub fn successor_addresses(&self) -> Vec<u64> {
        let mut addrs = Vec::new();
        let kind = self.terminator_kind();
        if let Some(target) = kind.static_target() {
            addrs.push(target);
        }
        if kind.has_fallthrough() && !matches!(kind, InstructionKind::Regular) {
            addrs.push(self.end_address);
        }
        addrs
    }
}

impl std::fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "{} [{:#x}..{:#x}]:",
            self.id, self.start_address, self.end_address
        )?;
        for insn in &self.instructions {
            writeln!(f, "  {insn}")?;
        }
        Ok(())
    }
}
