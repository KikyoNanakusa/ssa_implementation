use crate::llir::PReg;

/// RISC-V physical register in the CFG layer (x0..x31).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RvReg(pub u8);

impl RvReg {
    pub const ZERO: Self = Self(0);
    pub const RA: Self = Self(1);
    pub const SP: Self = Self(2);

    /// Convert from capstone's RegId to RvReg.
    /// Capstone RISCV_REG_X0 = 1, so reg_id - 1 gives the register number.
    pub fn from_capstone(reg_id: u16) -> Option<Self> {
        if reg_id == 0 {
            return None; // RISCV_REG_INVALID
        }
        let n = reg_id.checked_sub(1)?;
        if n < 32 { Some(Self(n as u8)) } else { None }
    }

    /// Convert to LLIR PReg.
    pub fn to_preg(self) -> PReg {
        PReg(self.0)
    }
}

impl std::fmt::Display for RvReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "x{}", self.0)
    }
}

/// A structured operand extracted from capstone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RvOperand {
    Reg(RvReg),
    Imm(i64),
    Mem { base: RvReg, disp: i64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_capstone_x0() {
        let reg = RvReg::from_capstone(1).unwrap();
        assert_eq!(reg, RvReg::ZERO);
        assert_eq!(reg.0, 0);
    }

    #[test]
    fn test_from_capstone_x31() {
        let reg = RvReg::from_capstone(32).unwrap();
        assert_eq!(reg.0, 31);
    }

    #[test]
    fn test_from_capstone_invalid_zero() {
        assert!(RvReg::from_capstone(0).is_none());
    }

    #[test]
    fn test_from_capstone_out_of_range() {
        assert!(RvReg::from_capstone(33).is_none());
        assert!(RvReg::from_capstone(100).is_none());
    }

    #[test]
    fn test_to_preg() {
        assert_eq!(RvReg::ZERO.to_preg(), PReg::ZERO);
        assert_eq!(RvReg::RA.to_preg(), PReg::RA);
        assert_eq!(RvReg::SP.to_preg(), PReg::SP);
        assert_eq!(RvReg(10).to_preg(), PReg(10));
    }
}
