pub mod cfg;
pub mod lower;

use std::fmt;

// ============================================================
// 1. Basic types
// ============================================================

/// LLIR type: minimal type to fix bit-width and sign-extension semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ty {
    I1,
    I8,
    I16,
    I32,
    I64,
    Ptr,
}

impl Ty {
    pub fn bit_width(&self) -> u32 {
        match self {
            Ty::I1 => 1,
            Ty::I8 => 8,
            Ty::I16 => 16,
            Ty::I32 => 32,
            Ty::I64 | Ty::Ptr => 64,
        }
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::I1 => write!(f, "i1"),
            Ty::I8 => write!(f, "i8"),
            Ty::I16 => write!(f, "i16"),
            Ty::I32 => write!(f, "i32"),
            Ty::I64 => write!(f, "i64"),
            Ty::Ptr => write!(f, "ptr"),
        }
    }
}

/// Physical register (x0--x31).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PReg(pub u8);

impl PReg {
    pub const ZERO: PReg = PReg(0);
    pub const RA: PReg = PReg(1);
    pub const SP: PReg = PReg(2);

    pub fn new(n: u8) -> Option<PReg> {
        if n < 32 { Some(PReg(n)) } else { None }
    }

    /// ABI name for this register.
    pub fn abi_name(&self) -> &'static str {
        const NAMES: [&str; 32] = [
            "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3",
            "a4", "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
            "t3", "t4", "t5", "t6",
        ];
        NAMES[self.0 as usize]
    }
}

impl fmt::Display for PReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.abi_name())
    }
}

/// Compiler-generated temporary variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Temp(pub u32);

impl fmt::Display for Temp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%t{}", self.0)
    }
}

/// Variable: either a physical register or a temporary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Var {
    PReg(PReg),
    Temp(Temp),
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Var::PReg(r) => write!(f, "{r}"),
            Var::Temp(t) => write!(f, "{t}"),
        }
    }
}

/// Value: operand for instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Value {
    VarRef(Var, Ty),
    Imm(i64, Ty),
    Undef(Ty),
}

impl Value {
    pub fn ty(&self) -> Ty {
        match self {
            Value::VarRef(_, ty) | Value::Imm(_, ty) | Value::Undef(ty) => *ty,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::VarRef(var, ty) => write!(f, "{var}:{ty}"),
            Value::Imm(n, ty) => write!(f, "{n}:{ty}"),
            Value::Undef(ty) => write!(f, "undef:{ty}"),
        }
    }
}

// ============================================================
// 2. Operators
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Sll,
    Srl,
    Sra,
    Mul,
    Div,
    DivU,
    Rem,
    RemU,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BinOp::Add => "Add",
            BinOp::Sub => "Sub",
            BinOp::And => "And",
            BinOp::Or => "Or",
            BinOp::Xor => "Xor",
            BinOp::Sll => "Sll",
            BinOp::Srl => "Srl",
            BinOp::Sra => "Sra",
            BinOp::Mul => "Mul",
            BinOp::Div => "Div",
            BinOp::DivU => "DivU",
            BinOp::Rem => "Rem",
            BinOp::RemU => "RemU",
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpPred {
    Eq,
    Ne,
    Lt,
    Ge,
    LtU,
    GeU,
}

impl fmt::Display for CmpPred {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            CmpPred::Eq => "Eq",
            CmpPred::Ne => "Ne",
            CmpPred::Lt => "Lt",
            CmpPred::Ge => "Ge",
            CmpPred::LtU => "LtU",
            CmpPred::GeU => "GeU",
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastOp {
    SExt,
    ZExt,
    Trunc,
}

impl fmt::Display for CastOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            CastOp::SExt => "SExt",
            CastOp::ZExt => "ZExt",
            CastOp::Trunc => "Trunc",
        };
        write!(f, "{s}")
    }
}

// ============================================================
// 3. Label
// ============================================================

/// Basic block label.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label(pub String);

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ============================================================
// 4. Instructions
// ============================================================

/// Right-hand side of an assignment.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Rhs {
    BinOp(BinOp, Value, Value),
    Cmp(CmpPred, Value, Value),
    Cast(CastOp, Value, Ty),
    Load(Ty, Value),
    Copy(Value),
}

impl fmt::Display for Rhs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rhs::BinOp(op, a, b) => write!(f, "binop({op}, {a}, {b})"),
            Rhs::Cmp(pred, a, b) => write!(f, "cmp({pred}, {a}, {b})"),
            Rhs::Cast(op, val, ty) => write!(f, "cast({op}, {val}, {ty})"),
            Rhs::Load(ty, addr) => write!(f, "load({ty}, {addr})"),
            Rhs::Copy(val) => write!(f, "copy({val})"),
        }
    }
}

/// Call target.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Callee {
    Direct(String),
    Indirect(Value),
}

impl fmt::Display for Callee {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Callee::Direct(name) => write!(f, "{name}"),
            Callee::Indirect(val) => write!(f, "indirect({val})"),
        }
    }
}

/// Non-terminator instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inst {
    Assign {
        dst: Var,
        ty: Ty,
        rhs: Rhs,
    },
    Store {
        ty: Ty,
        addr: Value,
        val: Value,
    },
    Call {
        rets: Vec<(Var, Ty)>,
        callee: Callee,
        args: Vec<Value>,
        clobber: Vec<PReg>,
    },
    Trap(String),
}

impl fmt::Display for Inst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Inst::Assign { dst, ty, rhs } => {
                write!(f, "{dst}:{ty} = {rhs}")
            }
            Inst::Store { ty, addr, val } => {
                write!(f, "store({ty}, {addr}, {val})")
            }
            Inst::Call {
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
                    for (i, reg) in clobber.iter().enumerate() {
                        if i > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{reg}")?;
                    }
                    write!(f, "}}")?;
                }
                Ok(())
            }
            Inst::Trap(reason) => write!(f, "trap \"{reason}\""),
        }
    }
}

/// Terminator instruction (exactly one per block, at the end).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    BrCond {
        cond: Value,
        then_label: Label,
        else_label: Label,
    },
    Jmp(Label),
    Ret(Vec<Value>),
    Switch {
        value: Value,
        cases: Vec<(i64, Label)>,
        default: Label,
    },
    Trap(String),
}

impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Term::BrCond {
                cond,
                then_label,
                else_label,
            } => {
                write!(f, "br {cond}, {then_label}, {else_label}")
            }
            Term::Jmp(label) => write!(f, "jmp {label}"),
            Term::Ret(vals) => {
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
            Term::Switch {
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
            Term::Trap(name) => write!(f, "trap {name}"),
        }
    }
}

// ============================================================
// 5. Program structure
// ============================================================

/// A basic block: a label, a sequence of non-terminator instructions,
/// and exactly one terminator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub label: Label,
    pub insts: Vec<Inst>,
    pub term: Term,
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}:", self.label)?;
        for inst in &self.insts {
            writeln!(f, "  {inst}")?;
        }
        write!(f, "  {}", self.term)
    }
}

/// A function definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub params: Vec<(Var, Ty)>,
    pub ret_ty: Option<Ty>,
    pub blocks: Vec<Block>,
}

impl fmt::Display for Function {
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

/// Top-level program: a collection of functions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub functions: Vec<Function>,
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, func) in self.functions.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{func}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preg_abi_names() {
        assert_eq!(PReg::ZERO.to_string(), "zero");
        assert_eq!(PReg::RA.to_string(), "ra");
        assert_eq!(PReg::SP.to_string(), "sp");
        assert_eq!(PReg(10).to_string(), "a0");
        assert_eq!(PReg(31).to_string(), "t6");
    }

    #[test]
    fn test_preg_new_validation() {
        assert!(PReg::new(0).is_some());
        assert!(PReg::new(31).is_some());
        assert!(PReg::new(32).is_none());
    }

    #[test]
    fn test_temp_display() {
        assert_eq!(Temp(0).to_string(), "%t0");
        assert_eq!(Temp(42).to_string(), "%t42");
    }

    #[test]
    fn test_value_display() {
        let v = Value::VarRef(Var::PReg(PReg(10)), Ty::I64);
        assert_eq!(v.to_string(), "a0:i64");

        let v = Value::Imm(10, Ty::I32);
        assert_eq!(v.to_string(), "10:i32");

        let v = Value::Undef(Ty::I64);
        assert_eq!(v.to_string(), "undef:i64");
    }

    #[test]
    fn test_ty_bit_width() {
        assert_eq!(Ty::I1.bit_width(), 1);
        assert_eq!(Ty::I32.bit_width(), 32);
        assert_eq!(Ty::Ptr.bit_width(), 64);
    }

    #[test]
    fn test_assign_display() {
        let inst = Inst::Assign {
            dst: Var::Temp(Temp(0)),
            ty: Ty::I1,
            rhs: Rhs::Cmp(
                CmpPred::Eq,
                Value::VarRef(Var::PReg(PReg(10)), Ty::I64),
                Value::VarRef(Var::PReg(PReg(11)), Ty::I64),
            ),
        };
        assert_eq!(inst.to_string(), "%t0:i1 = cmp(Eq, a0:i64, a1:i64)");
    }

    #[test]
    fn test_call_display() {
        let inst = Inst::Call {
            rets: vec![(Var::PReg(PReg(10)), Ty::I64)],
            callee: Callee::Direct("foo".into()),
            args: vec![Value::VarRef(Var::PReg(PReg(12)), Ty::I64)],
            clobber: vec![PReg(11), PReg(12)],
        };
        assert_eq!(inst.to_string(), "a0:i64 = call foo(a2:i64) clobber{a1,a2}");
    }

    #[test]
    fn test_function_display() {
        let func = Function {
            name: "f".into(),
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
                            Value::VarRef(Var::PReg(PReg(11)), Ty::I64),
                        ),
                    }],
                    term: Term::BrCond {
                        cond: Value::VarRef(Var::Temp(Temp(0)), Ty::I1),
                        then_label: Label("then".into()),
                        else_label: Label("els".into()),
                    },
                },
                Block {
                    label: Label("then".into()),
                    insts: vec![],
                    term: Term::Jmp(Label("join".into())),
                },
                Block {
                    label: Label("join".into()),
                    insts: vec![],
                    term: Term::Ret(vec![]),
                },
            ],
        };
        let output = func.to_string();
        assert!(output.starts_with("func f(): void {"));
        assert!(output.contains("%t0:i1 = cmp(Eq, a0:i64, a1:i64)"));
        assert!(output.contains("br %t0:i1, then, els"));
        assert!(output.contains("jmp join"));
        assert!(output.ends_with('}'));
    }
}
