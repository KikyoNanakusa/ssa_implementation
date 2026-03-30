use std::collections::BTreeMap;
use std::fmt;

use crate::llir::{BinOp, CastOp, CmpPred, Ty, Var};

pub mod construct;
pub mod def_use;
pub mod dominance_frontier;
pub mod liveness;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SsaVar {
    pub base: Var,
    pub version: u32,
}

impl fmt::Display for SsaVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.base, self.version)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsaValue {
    VarRef(SsaVar, Ty),
    Imm(i64, Ty),
    Undef(Ty),
}

impl SsaValue {
    pub fn ty(&self) -> Ty {
        match self {
            SsaValue::VarRef(_, ty) | SsaValue::Imm(_, ty) | SsaValue::Undef(ty) => *ty,
        }
    }
}

impl fmt::Display for SsaValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SsaValue::VarRef(var, ty) => write!(f, "{var}:{ty}"),
            SsaValue::Imm(n, ty) => write!(f, "{n}:{ty}"),
            SsaValue::Undef(ty) => write!(f, "undef:{ty}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsaRhs {
    BinOp(BinOp, SsaValue, SsaValue),
    Cmp(CmpPred, SsaValue, SsaValue),
    Cast(CastOp, SsaValue, Ty),
    Load(Ty, SsaValue),
    Copy(SsaValue),
}

impl fmt::Display for SsaRhs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SsaRhs::BinOp(op, a, b) => write!(f, "binop({op}, {a}, {b})"),
            SsaRhs::Cmp(pred, a, b) => write!(f, "cmp({pred}, {a}, {b})"),
            SsaRhs::Cast(op, val, ty) => write!(f, "cast({op}, {val}, {ty})"),
            SsaRhs::Load(ty, addr) => write!(f, "load({ty}, {addr})"),
            SsaRhs::Copy(val) => write!(f, "copy({val})"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsaInst {
    Assign {
        dst: SsaVar,
        ty: Ty,
        rhs: SsaRhs,
    },
    Store {
        ty: Ty,
        addr: SsaValue,
        val: SsaValue,
    },
    Call {
        rets: Vec<(SsaVar, Ty)>,
        callee: SsaCallee,
        args: Vec<SsaValue>,
        clobber: Vec<(SsaVar, Ty)>,
    },
    Trap(String),
}

impl fmt::Display for SsaInst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SsaInst::Assign { dst, ty, rhs } => {
                write!(f, "{dst}:{ty} = {rhs}")
            }
            SsaInst::Store { ty, addr, val } => {
                write!(f, "store({ty}, {addr}, {val})")
            }
            SsaInst::Call {
                rets,
                callee,
                args,
                clobber,
            } => {
                if !rets.is_empty() {
                    for (i, (var, ty)) in rets.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{var}:{ty}")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "call {callee}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")?;
                if !clobber.is_empty() {
                    write!(f, " clobber{{")?;
                    for (i, (var, _ty)) in clobber.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{var}")?;
                    }
                    write!(f, "}}")?;
                }
                Ok(())
            }
            SsaInst::Trap(reason) => write!(f, "trap \"{reason}\""),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsaCallee {
    Direct(String),
    Indirect(SsaValue),
}

impl fmt::Display for SsaCallee {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SsaCallee::Direct(name) => write!(f, "{name}"),
            SsaCallee::Indirect(val) => write!(f, "indirect({val})"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsaTerm {
    BrCond {
        cond: SsaValue,
        then_label: SsaLabel,
        else_label: SsaLabel,
    },
    Jmp(SsaLabel),
    Ret(Vec<SsaValue>),
    Switch {
        value: SsaValue,
        cases: Vec<(i64, SsaLabel)>,
        default: SsaLabel,
    },
    Trap(String),
}

impl fmt::Display for SsaTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SsaTerm::BrCond {
                cond,
                then_label,
                else_label,
            } => {
                write!(f, "br {cond}, {then_label}, {else_label}")
            }
            SsaTerm::Jmp(label) => write!(f, "jmp {label}"),
            SsaTerm::Ret(vals) => {
                write!(f, "ret")?;
                for (i, val) in vals.iter().enumerate() {
                    if i == 0 {
                        write!(f, " ")?;
                    } else {
                        write!(f, ", ")?;
                    }
                    write!(f, "{val}")?;
                }
                Ok(())
            }
            SsaTerm::Switch {
                value,
                cases,
                default,
            } => {
                writeln!(f, "switch {value} {{")?;
                for (n, label) in cases {
                    writeln!(f, "  case {n}: {label}")?;
                }
                write!(f, "  default: {default}\n}}")
            }
            SsaTerm::Trap(name) => write!(f, "trap {name}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SsaLabel(pub String);

impl fmt::Display for SsaLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaPhiNode {
    pub dst: SsaVar,
    pub ty: Ty,
    pub args: BTreeMap<SsaLabel, SsaValue>,
}

impl fmt::Display for SsaPhiNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} = phi(", self.dst, self.ty)?;
        for (i, (label, val)) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{label}: {val}")?;
        }
        write!(f, ")")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaBlock {
    pub label: SsaLabel,
    pub phis: Vec<SsaPhiNode>,
    pub insts: Vec<SsaInst>,
    pub term: SsaTerm,
}

impl fmt::Display for SsaBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}:", self.label)?;
        for phi in &self.phis {
            writeln!(f, "  {phi}")?;
        }
        for inst in &self.insts {
            writeln!(f, "  {inst}")?;
        }
        write!(f, "  {}", self.term)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaFunction {
    pub name: String,
    pub params: Vec<(SsaVar, Ty)>,
    pub ret_ty: Option<Ty>,
    pub blocks: Vec<SsaBlock>,
}

impl fmt::Display for SsaFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "func {}(", self.name)?;
        for (i, (var, ty)) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{var}:{ty}")?;
        }
        write!(f, "): ")?;
        match &self.ret_ty {
            Some(ty) => writeln!(f, "{ty} {{")?,
            None => writeln!(f, "void {{")?,
        }
        for (i, block) in self.blocks.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            writeln!(f, "{block}")?;
        }
        write!(f, "}}")
    }
}
