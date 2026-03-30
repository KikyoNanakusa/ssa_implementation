use super::operand::RvOperand;

/// Classification of an instruction for control flow graph construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstructionKind {
    /// Regular instruction that falls through to the next.
    Regular,

    /// Conditional branch (beq, bne, blt, bge, bltu, bgeu, c.beqz, c.bnez).
    /// Two successors: target and fall-through.
    ConditionalBranch { target: u64 },

    /// Unconditional direct jump (jal x0, c.j, j pseudo).
    /// One successor: the target.
    Jump { target: u64 },

    /// Indirect jump (jalr x0 rs, c.jr). Target unknown statically.
    IndirectJump,

    /// Direct call (jal ra, c.jal). Saves return address.
    Call { target: u64 },

    /// Indirect call (jalr ra rs, c.jalr). Callee unknown statically.
    IndirectCall,

    /// Return (ret / jalr x0, ra, 0 / c.jr ra). No successors in function CFG.
    Return,

    /// System call or trap (ecall, ebreak). Block terminator.
    Syscall,
}

impl InstructionKind {
    /// Returns true if this instruction terminates a basic block.
    pub fn is_terminator(&self) -> bool {
        !matches!(
            self,
            InstructionKind::Regular | InstructionKind::Call { .. } | InstructionKind::IndirectCall
        )
    }

    /// Returns the statically-known branch/jump target, if any.
    pub fn static_target(&self) -> Option<u64> {
        match self {
            InstructionKind::ConditionalBranch { target } | InstructionKind::Jump { target } => {
                Some(*target)
            }
            _ => None,
        }
    }

    /// Returns true if execution can fall through to the next instruction.
    pub fn has_fallthrough(&self) -> bool {
        matches!(
            self,
            InstructionKind::Regular
                | InstructionKind::ConditionalBranch { .. }
                | InstructionKind::Call { .. }
                | InstructionKind::IndirectCall
        )
    }
}

/// A single disassembled instruction, decoupled from capstone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    /// Virtual address of this instruction in the binary.
    pub address: u64,

    /// Size in bytes (2 for compressed, 4 for standard).
    pub size: u8,

    /// Raw instruction bytes. Only the first `size` bytes are meaningful.
    pub bytes: [u8; 4],

    /// Assembly mnemonic (e.g., "addi", "beq", "c.j").
    pub mnemonic: String,

    /// Operand string (e.g., "a0, a1, 0x10").
    pub operands: String,

    /// Structured operands extracted from capstone.
    pub rv_operands: Vec<RvOperand>,

    /// Control flow classification.
    pub kind: InstructionKind,
}

#[allow(dead_code)]
impl Instruction {
    /// Address immediately after this instruction.
    pub fn end_address(&self) -> u64 {
        self.address + self.size as u64
    }

    /// Raw bytes of this instruction as a slice.
    pub fn raw_bytes(&self) -> &[u8] {
        &self.bytes[..self.size as usize]
    }

    /// Returns true if this is a compressed (16-bit) instruction.
    pub fn is_compressed(&self) -> bool {
        self.size == 2
    }
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:#x}: {} {}",
            self.address, self.mnemonic, self.operands
        )
    }
}
