use crate::cfg::block::BasicBlock;
use crate::cfg::graph::Cfg;
use crate::cfg::instruction::{Instruction, InstructionKind};
use crate::cfg::operand::{RvOperand, RvReg};
use crate::disassemble::{FuncSymbol, split_instructions_by_functions};

use super::{
    BinOp, Block, Callee, CastOp, CmpPred, Function, Inst, Label, PReg, Program, Rhs, Temp, Term,
    Ty, Value, Var,
};

// ============================================================
// Caller-saved registers (clobber set for calls)
// ============================================================

const CALLER_SAVED: &[PReg] = &[
    PReg(1),  // ra
    PReg(5),  // t0
    PReg(6),  // t1
    PReg(7),  // t2
    PReg(10), // a0
    PReg(11), // a1
    PReg(12), // a2
    PReg(13), // a3
    PReg(14), // a4
    PReg(15), // a5
    PReg(16), // a6
    PReg(17), // a7
    PReg(28), // t3
    PReg(29), // t4
    PReg(30), // t5
    PReg(31), // t6
];

// ============================================================
// LowerCtx
// ============================================================

pub struct LowerCtx {
    temp_counter: u32,
}

impl LowerCtx {
    pub fn new() -> Self {
        Self { temp_counter: 0 }
    }

    fn fresh_temp(&mut self) -> Temp {
        let t = Temp(self.temp_counter);
        self.temp_counter += 1;
        t
    }

    fn reg_to_value(&self, reg: RvReg) -> Value {
        if reg == RvReg::ZERO {
            Value::Imm(0, Ty::I64)
        } else {
            Value::VarRef(Var::PReg(reg.to_preg()), Ty::I64)
        }
    }

    /// Returns None if reg is x0 (writes to x0 should generate Trap).
    fn reg_to_dst(&self, reg: RvReg) -> Option<Var> {
        if reg == RvReg::ZERO {
            None
        } else {
            Some(Var::PReg(reg.to_preg()))
        }
    }
}

// ============================================================
// Operand extraction helpers
// ============================================================

fn get_reg(ops: &[RvOperand], idx: usize) -> RvReg {
    match ops.get(idx) {
        Some(RvOperand::Reg(r)) => *r,
        _ => RvReg::ZERO,
    }
}

fn get_imm(ops: &[RvOperand], idx: usize) -> i64 {
    match ops.get(idx) {
        Some(RvOperand::Imm(v)) => *v,
        _ => 0,
    }
}

fn get_mem(ops: &[RvOperand], idx: usize) -> (RvReg, i64) {
    match ops.get(idx) {
        Some(RvOperand::Mem { base, disp }) => (*base, *disp),
        _ => (RvReg::ZERO, 0),
    }
}

// ============================================================
// ALU helpers
// ============================================================

fn lower_alu_rrr(ctx: &mut LowerCtx, ops: &[RvOperand], op: BinOp) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let rs1 = get_reg(ops, 1);
    let rs2 = get_reg(ops, 2);
    match ctx.reg_to_dst(rd) {
        Some(dst) => vec![Inst::Assign {
            dst,
            ty: Ty::I64,
            rhs: Rhs::BinOp(op, ctx.reg_to_value(rs1), ctx.reg_to_value(rs2)),
        }],
        None => vec![Inst::Trap("write_x0".into())],
    }
}

fn lower_alu_rri(ctx: &mut LowerCtx, ops: &[RvOperand], op: BinOp) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let rs1 = get_reg(ops, 1);
    let imm = get_imm(ops, 2);
    match ctx.reg_to_dst(rd) {
        Some(dst) => vec![Inst::Assign {
            dst,
            ty: Ty::I64,
            rhs: Rhs::BinOp(op, ctx.reg_to_value(rs1), Value::Imm(imm, Ty::I64)),
        }],
        None => vec![Inst::Trap("write_x0".into())],
    }
}

/// Compressed ALU: rd = rd op rs2 (c.add, c.sub, c.and, c.or, c.xor)
fn lower_alu_c_rr(ctx: &mut LowerCtx, ops: &[RvOperand], op: BinOp) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let rs2 = get_reg(ops, 1);
    match ctx.reg_to_dst(rd) {
        Some(dst) => vec![Inst::Assign {
            dst,
            ty: Ty::I64,
            rhs: Rhs::BinOp(op, ctx.reg_to_value(rd), ctx.reg_to_value(rs2)),
        }],
        None => vec![Inst::Trap("write_x0".into())],
    }
}

/// Compressed ALU immediate: rd = rd op imm (c.addi, c.slli, c.srli, c.srai, c.andi)
fn lower_alu_c_ri(ctx: &mut LowerCtx, ops: &[RvOperand], op: BinOp) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let imm = get_imm(ops, 1);
    match ctx.reg_to_dst(rd) {
        Some(dst) => vec![Inst::Assign {
            dst,
            ty: Ty::I64,
            rhs: Rhs::BinOp(op, ctx.reg_to_value(rd), Value::Imm(imm, Ty::I64)),
        }],
        None => vec![Inst::Trap("write_x0".into())],
    }
}

/// W R-type: trunc both operands to i32, operate, sext to i64
fn lower_alu_w_rrr(ctx: &mut LowerCtx, ops: &[RvOperand], op: BinOp) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let rs1 = get_reg(ops, 1);
    let rs2 = get_reg(ops, 2);
    match ctx.reg_to_dst(rd) {
        Some(dst) => {
            let t1 = ctx.fresh_temp();
            let t2 = ctx.fresh_temp();
            let t3 = ctx.fresh_temp();
            vec![
                Inst::Assign {
                    dst: Var::Temp(t1),
                    ty: Ty::I32,
                    rhs: Rhs::Cast(CastOp::Trunc, ctx.reg_to_value(rs1), Ty::I32),
                },
                Inst::Assign {
                    dst: Var::Temp(t2),
                    ty: Ty::I32,
                    rhs: Rhs::Cast(CastOp::Trunc, ctx.reg_to_value(rs2), Ty::I32),
                },
                Inst::Assign {
                    dst: Var::Temp(t3),
                    ty: Ty::I32,
                    rhs: Rhs::BinOp(
                        op,
                        Value::VarRef(Var::Temp(t1), Ty::I32),
                        Value::VarRef(Var::Temp(t2), Ty::I32),
                    ),
                },
                Inst::Assign {
                    dst,
                    ty: Ty::I64,
                    rhs: Rhs::Cast(CastOp::SExt, Value::VarRef(Var::Temp(t3), Ty::I32), Ty::I64),
                },
            ]
        }
        None => vec![Inst::Trap("write_x0".into())],
    }
}

/// W I-type: trunc rs1 to i32, operate with imm, sext to i64
fn lower_alu_w_rri(ctx: &mut LowerCtx, ops: &[RvOperand], op: BinOp) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let rs1 = get_reg(ops, 1);
    let imm = get_imm(ops, 2);
    match ctx.reg_to_dst(rd) {
        Some(dst) => {
            let t1 = ctx.fresh_temp();
            let t2 = ctx.fresh_temp();
            vec![
                Inst::Assign {
                    dst: Var::Temp(t1),
                    ty: Ty::I32,
                    rhs: Rhs::Cast(CastOp::Trunc, ctx.reg_to_value(rs1), Ty::I32),
                },
                Inst::Assign {
                    dst: Var::Temp(t2),
                    ty: Ty::I32,
                    rhs: Rhs::BinOp(
                        op,
                        Value::VarRef(Var::Temp(t1), Ty::I32),
                        Value::Imm(imm, Ty::I32),
                    ),
                },
                Inst::Assign {
                    dst,
                    ty: Ty::I64,
                    rhs: Rhs::Cast(CastOp::SExt, Value::VarRef(Var::Temp(t2), Ty::I32), Ty::I64),
                },
            ]
        }
        None => vec![Inst::Trap("write_x0".into())],
    }
}

/// Compressed W R-type: rd = (sext)(trunc(rd) op trunc(rs2))
fn lower_alu_w_c_rr(ctx: &mut LowerCtx, ops: &[RvOperand], op: BinOp) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let rs2 = get_reg(ops, 1);
    match ctx.reg_to_dst(rd) {
        Some(dst) => {
            let t1 = ctx.fresh_temp();
            let t2 = ctx.fresh_temp();
            let t3 = ctx.fresh_temp();
            vec![
                Inst::Assign {
                    dst: Var::Temp(t1),
                    ty: Ty::I32,
                    rhs: Rhs::Cast(CastOp::Trunc, ctx.reg_to_value(rd), Ty::I32),
                },
                Inst::Assign {
                    dst: Var::Temp(t2),
                    ty: Ty::I32,
                    rhs: Rhs::Cast(CastOp::Trunc, ctx.reg_to_value(rs2), Ty::I32),
                },
                Inst::Assign {
                    dst: Var::Temp(t3),
                    ty: Ty::I32,
                    rhs: Rhs::BinOp(
                        op,
                        Value::VarRef(Var::Temp(t1), Ty::I32),
                        Value::VarRef(Var::Temp(t2), Ty::I32),
                    ),
                },
                Inst::Assign {
                    dst,
                    ty: Ty::I64,
                    rhs: Rhs::Cast(CastOp::SExt, Value::VarRef(Var::Temp(t3), Ty::I32), Ty::I64),
                },
            ]
        }
        None => vec![Inst::Trap("write_x0".into())],
    }
}

/// Compressed W I-type: rd = (sext)(trunc(rd) op imm)
fn lower_alu_w_c_ri(ctx: &mut LowerCtx, ops: &[RvOperand], op: BinOp) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let imm = get_imm(ops, 1);
    match ctx.reg_to_dst(rd) {
        Some(dst) => {
            let t1 = ctx.fresh_temp();
            let t2 = ctx.fresh_temp();
            vec![
                Inst::Assign {
                    dst: Var::Temp(t1),
                    ty: Ty::I32,
                    rhs: Rhs::Cast(CastOp::Trunc, ctx.reg_to_value(rd), Ty::I32),
                },
                Inst::Assign {
                    dst: Var::Temp(t2),
                    ty: Ty::I32,
                    rhs: Rhs::BinOp(
                        op,
                        Value::VarRef(Var::Temp(t1), Ty::I32),
                        Value::Imm(imm, Ty::I32),
                    ),
                },
                Inst::Assign {
                    dst,
                    ty: Ty::I64,
                    rhs: Rhs::Cast(CastOp::SExt, Value::VarRef(Var::Temp(t2), Ty::I32), Ty::I64),
                },
            ]
        }
        None => vec![Inst::Trap("write_x0".into())],
    }
}

// ============================================================
// SLT helpers
// ============================================================

fn lower_slt(ctx: &mut LowerCtx, ops: &[RvOperand], pred: CmpPred) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let rs1 = get_reg(ops, 1);
    let rs2 = get_reg(ops, 2);
    match ctx.reg_to_dst(rd) {
        Some(dst) => {
            let tc = ctx.fresh_temp();
            vec![
                Inst::Assign {
                    dst: Var::Temp(tc),
                    ty: Ty::I1,
                    rhs: Rhs::Cmp(pred, ctx.reg_to_value(rs1), ctx.reg_to_value(rs2)),
                },
                Inst::Assign {
                    dst,
                    ty: Ty::I64,
                    rhs: Rhs::Cast(CastOp::ZExt, Value::VarRef(Var::Temp(tc), Ty::I1), Ty::I64),
                },
            ]
        }
        None => vec![Inst::Trap("write_x0".into())],
    }
}

fn lower_slti(ctx: &mut LowerCtx, ops: &[RvOperand], pred: CmpPred) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let rs1 = get_reg(ops, 1);
    let imm = get_imm(ops, 2);
    match ctx.reg_to_dst(rd) {
        Some(dst) => {
            let tc = ctx.fresh_temp();
            vec![
                Inst::Assign {
                    dst: Var::Temp(tc),
                    ty: Ty::I1,
                    rhs: Rhs::Cmp(pred, ctx.reg_to_value(rs1), Value::Imm(imm, Ty::I64)),
                },
                Inst::Assign {
                    dst,
                    ty: Ty::I64,
                    rhs: Rhs::Cast(CastOp::ZExt, Value::VarRef(Var::Temp(tc), Ty::I1), Ty::I64),
                },
            ]
        }
        None => vec![Inst::Trap("write_x0".into())],
    }
}

// ============================================================
// Load/Store helpers
// ============================================================

/// Lower a load instruction: addr = base + disp; rd = load(ty, addr) [+ ext]
fn lower_load(
    ctx: &mut LowerCtx,
    ops: &[RvOperand],
    load_ty: Ty,
    ext: Option<CastOp>,
) -> Vec<Inst> {
    // operand layout: rd, mem(base, disp)
    let rd = get_reg(ops, 0);
    let (base, disp) = get_mem(ops, 1);
    match ctx.reg_to_dst(rd) {
        Some(dst) => {
            let mut insts = Vec::new();
            let addr = if disp != 0 {
                let ta = ctx.fresh_temp();
                insts.push(Inst::Assign {
                    dst: Var::Temp(ta),
                    ty: Ty::Ptr,
                    rhs: Rhs::BinOp(
                        BinOp::Add,
                        ctx.reg_to_value(base),
                        Value::Imm(disp, Ty::I64),
                    ),
                });
                Value::VarRef(Var::Temp(ta), Ty::Ptr)
            } else {
                ctx.reg_to_value(base)
            };

            match ext {
                Some(cast_op) => {
                    let tl = ctx.fresh_temp();
                    insts.push(Inst::Assign {
                        dst: Var::Temp(tl),
                        ty: load_ty,
                        rhs: Rhs::Load(load_ty, addr),
                    });
                    insts.push(Inst::Assign {
                        dst,
                        ty: Ty::I64,
                        rhs: Rhs::Cast(cast_op, Value::VarRef(Var::Temp(tl), load_ty), Ty::I64),
                    });
                }
                None => {
                    insts.push(Inst::Assign {
                        dst,
                        ty: load_ty,
                        rhs: Rhs::Load(load_ty, addr),
                    });
                }
            }
            insts
        }
        None => vec![Inst::Trap("write_x0".into())],
    }
}

/// Lower a store instruction: addr = base + disp; [trunc val]; store(ty, addr, val)
fn lower_store(ctx: &mut LowerCtx, ops: &[RvOperand], store_ty: Ty) -> Vec<Inst> {
    // operand layout: rs2, mem(base, disp)
    let rs2 = get_reg(ops, 0);
    let (base, disp) = get_mem(ops, 1);

    let mut insts = Vec::new();
    let addr = if disp != 0 {
        let ta = ctx.fresh_temp();
        insts.push(Inst::Assign {
            dst: Var::Temp(ta),
            ty: Ty::Ptr,
            rhs: Rhs::BinOp(
                BinOp::Add,
                ctx.reg_to_value(base),
                Value::Imm(disp, Ty::I64),
            ),
        });
        Value::VarRef(Var::Temp(ta), Ty::Ptr)
    } else {
        ctx.reg_to_value(base)
    };

    let val = if store_ty != Ty::I64 {
        let tt = ctx.fresh_temp();
        insts.push(Inst::Assign {
            dst: Var::Temp(tt),
            ty: store_ty,
            rhs: Rhs::Cast(CastOp::Trunc, ctx.reg_to_value(rs2), store_ty),
        });
        Value::VarRef(Var::Temp(tt), store_ty)
    } else {
        ctx.reg_to_value(rs2)
    };

    insts.push(Inst::Store {
        ty: store_ty,
        addr,
        val,
    });
    insts
}

/// Lower compressed load: c.lw, c.ld, c.lwsp, c.ldsp
/// Operand layout: rd, mem(base, disp)
fn lower_c_load(
    ctx: &mut LowerCtx,
    ops: &[RvOperand],
    load_ty: Ty,
    ext: Option<CastOp>,
) -> Vec<Inst> {
    // Same as lower_load — operand layout is identical for compressed loads
    lower_load(ctx, ops, load_ty, ext)
}

/// Lower compressed store: c.sw, c.sd, c.swsp, c.sdsp
/// Operand layout: rs2, mem(base, disp)
fn lower_c_store(ctx: &mut LowerCtx, ops: &[RvOperand], store_ty: Ty) -> Vec<Inst> {
    lower_store(ctx, ops, store_ty)
}

// ============================================================
// Upper immediate helpers
// ============================================================

fn lower_lui(ctx: &mut LowerCtx, ops: &[RvOperand]) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let imm = get_imm(ops, 1);
    let val = imm << 12;
    match ctx.reg_to_dst(rd) {
        Some(dst) => vec![Inst::Assign {
            dst,
            ty: Ty::I64,
            rhs: Rhs::Copy(Value::Imm(val, Ty::I64)),
        }],
        None => vec![Inst::Trap("write_x0".into())],
    }
}

fn lower_auipc(ctx: &mut LowerCtx, ops: &[RvOperand], insn_addr: u64) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let imm = get_imm(ops, 1);
    let val = insn_addr.wrapping_add((imm << 12) as u64);
    match ctx.reg_to_dst(rd) {
        Some(dst) => vec![Inst::Assign {
            dst,
            ty: Ty::I64,
            rhs: Rhs::Copy(Value::Imm(val as i64, Ty::I64)),
        }],
        None => vec![Inst::Trap("write_x0".into())],
    }
}

// ============================================================
// Pseudo instruction helpers
// ============================================================

fn lower_mv(ctx: &mut LowerCtx, ops: &[RvOperand]) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let rs = get_reg(ops, 1);
    match ctx.reg_to_dst(rd) {
        Some(dst) => vec![Inst::Assign {
            dst,
            ty: Ty::I64,
            rhs: Rhs::Copy(ctx.reg_to_value(rs)),
        }],
        None => vec![Inst::Trap("write_x0".into())],
    }
}

fn lower_li(ctx: &mut LowerCtx, ops: &[RvOperand]) -> Vec<Inst> {
    let rd = get_reg(ops, 0);
    let imm = get_imm(ops, 1);
    match ctx.reg_to_dst(rd) {
        Some(dst) => vec![Inst::Assign {
            dst,
            ty: Ty::I64,
            rhs: Rhs::Copy(Value::Imm(imm, Ty::I64)),
        }],
        None => vec![Inst::Trap("write_x0".into())],
    }
}

// ============================================================
// Call helper
// ============================================================

fn lower_call(ctx: &mut LowerCtx, insn: &Instruction) -> Vec<Inst> {
    let callee = match insn.kind {
        InstructionKind::Call { target } => Callee::Direct(format!("sub_{target:x}")),
        InstructionKind::IndirectCall => {
            // jalr ra, rs1, 0 => indirect call via rs1
            let rs1 = get_reg(&insn.rv_operands, 1);
            Callee::Indirect(ctx.reg_to_value(rs1))
        }
        _ => Callee::Direct("unknown".into()),
    };
    vec![Inst::Call {
        rets: vec![],
        callee,
        args: vec![],
        clobber: CALLER_SAVED.to_vec(),
    }]
}

// ============================================================
// Instruction dispatch
// ============================================================

fn lower_instruction(ctx: &mut LowerCtx, insn: &Instruction) -> Vec<Inst> {
    let ops = &insn.rv_operands;
    match insn.mnemonic.as_str() {
        // ALU R-type
        "add" => lower_alu_rrr(ctx, ops, BinOp::Add),
        "sub" => lower_alu_rrr(ctx, ops, BinOp::Sub),
        "and" => lower_alu_rrr(ctx, ops, BinOp::And),
        "or" => lower_alu_rrr(ctx, ops, BinOp::Or),
        "xor" => lower_alu_rrr(ctx, ops, BinOp::Xor),
        "sll" => lower_alu_rrr(ctx, ops, BinOp::Sll),
        "srl" => lower_alu_rrr(ctx, ops, BinOp::Srl),
        "sra" => lower_alu_rrr(ctx, ops, BinOp::Sra),
        "mul" => lower_alu_rrr(ctx, ops, BinOp::Mul),
        "div" => lower_alu_rrr(ctx, ops, BinOp::Div),
        "divu" => lower_alu_rrr(ctx, ops, BinOp::DivU),
        "rem" => lower_alu_rrr(ctx, ops, BinOp::Rem),
        "remu" => lower_alu_rrr(ctx, ops, BinOp::RemU),

        // ALU I-type
        "addi" => lower_alu_rri(ctx, ops, BinOp::Add),
        "andi" => lower_alu_rri(ctx, ops, BinOp::And),
        "ori" => lower_alu_rri(ctx, ops, BinOp::Or),
        "xori" => lower_alu_rri(ctx, ops, BinOp::Xor),
        "slli" => lower_alu_rri(ctx, ops, BinOp::Sll),
        "srli" => lower_alu_rri(ctx, ops, BinOp::Srl),
        "srai" => lower_alu_rri(ctx, ops, BinOp::Sra),

        // W R-type
        "addw" => lower_alu_w_rrr(ctx, ops, BinOp::Add),
        "subw" => lower_alu_w_rrr(ctx, ops, BinOp::Sub),
        "sllw" => lower_alu_w_rrr(ctx, ops, BinOp::Sll),
        "srlw" => lower_alu_w_rrr(ctx, ops, BinOp::Srl),
        "sraw" => lower_alu_w_rrr(ctx, ops, BinOp::Sra),
        "mulw" => lower_alu_w_rrr(ctx, ops, BinOp::Mul),
        "divw" => lower_alu_w_rrr(ctx, ops, BinOp::Div),
        "divuw" => lower_alu_w_rrr(ctx, ops, BinOp::DivU),
        "remw" => lower_alu_w_rrr(ctx, ops, BinOp::Rem),
        "remuw" => lower_alu_w_rrr(ctx, ops, BinOp::RemU),

        // W I-type
        "addiw" => lower_alu_w_rri(ctx, ops, BinOp::Add),
        "slliw" => lower_alu_w_rri(ctx, ops, BinOp::Sll),
        "srliw" => lower_alu_w_rri(ctx, ops, BinOp::Srl),
        "sraiw" => lower_alu_w_rri(ctx, ops, BinOp::Sra),

        // Compressed ALU (rd = rd op rs2)
        "c.add" => lower_alu_c_rr(ctx, ops, BinOp::Add),
        "c.sub" => lower_alu_c_rr(ctx, ops, BinOp::Sub),
        "c.and" => lower_alu_c_rr(ctx, ops, BinOp::And),
        "c.or" => lower_alu_c_rr(ctx, ops, BinOp::Or),
        "c.xor" => lower_alu_c_rr(ctx, ops, BinOp::Xor),

        // Compressed ALU immediate (rd = rd op imm)
        "c.addi" => lower_alu_c_ri(ctx, ops, BinOp::Add),
        "c.slli" => lower_alu_c_ri(ctx, ops, BinOp::Sll),
        "c.srli" => lower_alu_c_ri(ctx, ops, BinOp::Srl),
        "c.srai" => lower_alu_c_ri(ctx, ops, BinOp::Sra),
        "c.andi" => lower_alu_c_ri(ctx, ops, BinOp::And),

        // Compressed W (rd = rd op rs2, 32-bit)
        "c.addw" => lower_alu_w_c_rr(ctx, ops, BinOp::Add),
        "c.subw" => lower_alu_w_c_rr(ctx, ops, BinOp::Sub),

        // Compressed W immediate
        "c.addiw" => lower_alu_w_c_ri(ctx, ops, BinOp::Add),

        // c.addi16sp: sp = sp + imm (special form of c.addi targeting sp)
        "c.addi16sp" => lower_alu_c_ri(ctx, ops, BinOp::Add),

        // c.addi4spn: rd = sp + imm
        "c.addi4spn" => lower_alu_rri(ctx, ops, BinOp::Add),

        // Compare
        "slt" => lower_slt(ctx, ops, CmpPred::Lt),
        "sltu" => lower_slt(ctx, ops, CmpPred::LtU),
        "slti" => lower_slti(ctx, ops, CmpPred::Lt),
        "sltiu" => lower_slti(ctx, ops, CmpPred::LtU),

        // Load
        "lb" => lower_load(ctx, ops, Ty::I8, Some(CastOp::SExt)),
        "lh" => lower_load(ctx, ops, Ty::I16, Some(CastOp::SExt)),
        "lw" => lower_load(ctx, ops, Ty::I32, Some(CastOp::SExt)),
        "ld" => lower_load(ctx, ops, Ty::I64, None),
        "lbu" => lower_load(ctx, ops, Ty::I8, Some(CastOp::ZExt)),
        "lhu" => lower_load(ctx, ops, Ty::I16, Some(CastOp::ZExt)),
        "lwu" => lower_load(ctx, ops, Ty::I32, Some(CastOp::ZExt)),

        // Compressed load
        "c.lw" => lower_c_load(ctx, ops, Ty::I32, Some(CastOp::SExt)),
        "c.ld" => lower_c_load(ctx, ops, Ty::I64, None),
        "c.lwsp" => lower_c_load(ctx, ops, Ty::I32, Some(CastOp::SExt)),
        "c.ldsp" => lower_c_load(ctx, ops, Ty::I64, None),

        // Store
        "sb" => lower_store(ctx, ops, Ty::I8),
        "sh" => lower_store(ctx, ops, Ty::I16),
        "sw" => lower_store(ctx, ops, Ty::I32),
        "sd" => lower_store(ctx, ops, Ty::I64),

        // Compressed store
        "c.sw" => lower_c_store(ctx, ops, Ty::I32),
        "c.sd" => lower_c_store(ctx, ops, Ty::I64),
        "c.swsp" => lower_c_store(ctx, ops, Ty::I32),
        "c.sdsp" => lower_c_store(ctx, ops, Ty::I64),

        // Upper immediate
        "lui" => lower_lui(ctx, ops),
        "c.lui" => lower_lui(ctx, ops),
        "auipc" => lower_auipc(ctx, ops, insn.address),

        // Pseudo / nop
        "nop" | "c.nop" => vec![],

        // Pseudo: mv / li
        "c.mv" => lower_mv(ctx, ops),
        "c.li" => lower_li(ctx, ops),

        // Pseudo: ALU (synthesize operands to reuse existing helpers)
        "not" => {
            let rd = get_reg(ops, 0);
            let rs = get_reg(ops, 1);
            let synth = vec![RvOperand::Reg(rd), RvOperand::Reg(rs), RvOperand::Imm(-1)];
            lower_alu_rri(ctx, &synth, BinOp::Xor)
        }
        "neg" => {
            let rd = get_reg(ops, 0);
            let rs = get_reg(ops, 1);
            let synth = vec![
                RvOperand::Reg(rd),
                RvOperand::Reg(RvReg::ZERO),
                RvOperand::Reg(rs),
            ];
            lower_alu_rrr(ctx, &synth, BinOp::Sub)
        }
        "negw" => {
            let rd = get_reg(ops, 0);
            let rs = get_reg(ops, 1);
            let synth = vec![
                RvOperand::Reg(rd),
                RvOperand::Reg(RvReg::ZERO),
                RvOperand::Reg(rs),
            ];
            lower_alu_w_rrr(ctx, &synth, BinOp::Sub)
        }
        "sext.w" => {
            let rd = get_reg(ops, 0);
            let rs = get_reg(ops, 1);
            let synth = vec![RvOperand::Reg(rd), RvOperand::Reg(rs), RvOperand::Imm(0)];
            lower_alu_w_rri(ctx, &synth, BinOp::Add)
        }
        "zext.b" => {
            let rd = get_reg(ops, 0);
            let rs = get_reg(ops, 1);
            let synth = vec![RvOperand::Reg(rd), RvOperand::Reg(rs), RvOperand::Imm(255)];
            lower_alu_rri(ctx, &synth, BinOp::And)
        }
        "seqz" => {
            let rd = get_reg(ops, 0);
            let rs = get_reg(ops, 1);
            let synth = vec![RvOperand::Reg(rd), RvOperand::Reg(rs), RvOperand::Imm(1)];
            lower_slti(ctx, &synth, CmpPred::LtU)
        }
        "snez" => {
            let rd = get_reg(ops, 0);
            let rs = get_reg(ops, 1);
            let synth = vec![
                RvOperand::Reg(rd),
                RvOperand::Reg(RvReg::ZERO),
                RvOperand::Reg(rs),
            ];
            lower_slt(ctx, &synth, CmpPred::LtU)
        }
        "sltz" => {
            let rd = get_reg(ops, 0);
            let rs = get_reg(ops, 1);
            let synth = vec![
                RvOperand::Reg(rd),
                RvOperand::Reg(rs),
                RvOperand::Reg(RvReg::ZERO),
            ];
            lower_slt(ctx, &synth, CmpPred::Lt)
        }
        "sgtz" => {
            let rd = get_reg(ops, 0);
            let rs = get_reg(ops, 1);
            let synth = vec![
                RvOperand::Reg(rd),
                RvOperand::Reg(RvReg::ZERO),
                RvOperand::Reg(rs),
            ];
            lower_slt(ctx, &synth, CmpPred::Lt)
        }

        // ebreak
        "ebreak" | "c.ebreak" => vec![Inst::Trap("ebreak".into())],

        // Call (jal ra, jalr ra) — handled as non-terminator
        "jal" | "jalr" | "c.jal" | "c.jalr"
            if matches!(
                insn.kind,
                InstructionKind::Call { .. } | InstructionKind::IndirectCall
            ) =>
        {
            lower_call(ctx, insn)
        }

        // Unknown
        other => vec![Inst::Trap(format!("unknown_{other}"))],
    }
}

// ============================================================
// Terminator conversion
// ============================================================

fn lower_terminator(
    ctx: &mut LowerCtx,
    insn: &Instruction,
    fall_label: Option<&Label>,
) -> (Vec<Inst>, Term) {
    let ops = &insn.rv_operands;
    let default_fall = fall_label
        .cloned()
        .unwrap_or_else(|| Label("unreachable".into()));

    match &insn.kind {
        InstructionKind::ConditionalBranch { target } => {
            let target_label = Label(format!("bb_{target:x}"));
            let (pred, lhs, rhs_val) = match insn.mnemonic.as_str() {
                "beq" => (
                    CmpPred::Eq,
                    ctx.reg_to_value(get_reg(ops, 0)),
                    ctx.reg_to_value(get_reg(ops, 1)),
                ),
                "bne" => (
                    CmpPred::Ne,
                    ctx.reg_to_value(get_reg(ops, 0)),
                    ctx.reg_to_value(get_reg(ops, 1)),
                ),
                "blt" => (
                    CmpPred::Lt,
                    ctx.reg_to_value(get_reg(ops, 0)),
                    ctx.reg_to_value(get_reg(ops, 1)),
                ),
                "bge" => (
                    CmpPred::Ge,
                    ctx.reg_to_value(get_reg(ops, 0)),
                    ctx.reg_to_value(get_reg(ops, 1)),
                ),
                "bltu" => (
                    CmpPred::LtU,
                    ctx.reg_to_value(get_reg(ops, 0)),
                    ctx.reg_to_value(get_reg(ops, 1)),
                ),
                "bgeu" => (
                    CmpPred::GeU,
                    ctx.reg_to_value(get_reg(ops, 0)),
                    ctx.reg_to_value(get_reg(ops, 1)),
                ),
                "c.beqz" => (
                    CmpPred::Eq,
                    ctx.reg_to_value(get_reg(ops, 0)),
                    Value::Imm(0, Ty::I64),
                ),
                "c.bnez" => (
                    CmpPred::Ne,
                    ctx.reg_to_value(get_reg(ops, 0)),
                    Value::Imm(0, Ty::I64),
                ),
                // Pseudo: 1-operand branches (rs, offset)
                "beqz" => (
                    CmpPred::Eq,
                    ctx.reg_to_value(get_reg(ops, 0)),
                    Value::Imm(0, Ty::I64),
                ),
                "bnez" => (
                    CmpPred::Ne,
                    ctx.reg_to_value(get_reg(ops, 0)),
                    Value::Imm(0, Ty::I64),
                ),
                "bgez" => (
                    CmpPred::Ge,
                    ctx.reg_to_value(get_reg(ops, 0)),
                    Value::Imm(0, Ty::I64),
                ),
                "bltz" => (
                    CmpPred::Lt,
                    ctx.reg_to_value(get_reg(ops, 0)),
                    Value::Imm(0, Ty::I64),
                ),
                "blez" => (
                    CmpPred::Ge,
                    Value::Imm(0, Ty::I64),
                    ctx.reg_to_value(get_reg(ops, 0)),
                ),
                "bgtz" => (
                    CmpPred::Lt,
                    Value::Imm(0, Ty::I64),
                    ctx.reg_to_value(get_reg(ops, 0)),
                ),
                // Pseudo: 2-operand branches with reversed operands (rs, rt, offset)
                "bgt" => (
                    CmpPred::Lt,
                    ctx.reg_to_value(get_reg(ops, 1)),
                    ctx.reg_to_value(get_reg(ops, 0)),
                ),
                "ble" => (
                    CmpPred::Ge,
                    ctx.reg_to_value(get_reg(ops, 1)),
                    ctx.reg_to_value(get_reg(ops, 0)),
                ),
                "bgtu" => (
                    CmpPred::LtU,
                    ctx.reg_to_value(get_reg(ops, 1)),
                    ctx.reg_to_value(get_reg(ops, 0)),
                ),
                "bleu" => (
                    CmpPred::GeU,
                    ctx.reg_to_value(get_reg(ops, 1)),
                    ctx.reg_to_value(get_reg(ops, 0)),
                ),
                _ => (
                    CmpPred::Eq,
                    ctx.reg_to_value(get_reg(ops, 0)),
                    ctx.reg_to_value(get_reg(ops, 1)),
                ),
            };

            let tc = ctx.fresh_temp();
            let cmp_inst = Inst::Assign {
                dst: Var::Temp(tc),
                ty: Ty::I1,
                rhs: Rhs::Cmp(pred, lhs, rhs_val),
            };

            let term = Term::BrCond {
                cond: Value::VarRef(Var::Temp(tc), Ty::I1),
                then_label: target_label,
                else_label: default_fall,
            };

            (vec![cmp_inst], term)
        }
        InstructionKind::Jump { target } => {
            let target_label = Label(format!("bb_{target:x}"));
            (vec![], Term::Jmp(target_label))
        }
        InstructionKind::Return => (vec![], Term::Ret(vec![])),
        InstructionKind::Syscall => (vec![], Term::Trap("syscall".into())),
        InstructionKind::IndirectJump => (vec![], Term::Trap("indirect_jump".into())),
        // Regular / Call / IndirectCall at block end → fallthrough
        InstructionKind::Regular | InstructionKind::Call { .. } | InstructionKind::IndirectCall => {
            // Non-terminator at block end: emit as normal instruction + jmp
            let insts = lower_instruction(ctx, insn);
            (insts, Term::Jmp(default_fall))
        }
    }
}

// ============================================================
// Block conversion
// ============================================================

fn block_label(block: &BasicBlock) -> Label {
    match &block.label {
        Some(l) => Label(l.clone()),
        None => Label(format!("bb_{:x}", block.start_address)),
    }
}

pub fn lower_block(ctx: &mut LowerCtx, block: &BasicBlock, fall_label: Option<&Label>) -> Block {
    let label = block_label(block);
    let mut insts = Vec::new();

    let n = block.instructions.len();
    if n == 0 {
        return Block {
            label,
            insts,
            term: Term::Jmp(fall_label.cloned().unwrap_or(Label("unreachable".into()))),
        };
    }

    // Process all instructions except the last
    for insn in &block.instructions[..n - 1] {
        insts.extend(lower_instruction(ctx, insn));
    }

    // Last instruction
    let last = &block.instructions[n - 1];
    if last.kind.is_terminator() {
        let (term_insts, term) = lower_terminator(ctx, last, fall_label);
        insts.extend(term_insts);
        Block { label, insts, term }
    } else {
        // Non-terminator at end of block
        insts.extend(lower_instruction(ctx, last));
        let term = Term::Jmp(fall_label.cloned().unwrap_or(Label("unreachable".into())));
        Block { label, insts, term }
    }
}

// ============================================================
// CFG → Function
// ============================================================

pub fn lower_cfg(cfg: &Cfg, name: &str) -> Function {
    let mut ctx = LowerCtx::new();

    // Collect blocks sorted by start address
    let mut sorted_blocks: Vec<_> = cfg.blocks().map(|(_, block)| block).collect();
    sorted_blocks.sort_by_key(|b| b.start_address);

    // Pre-compute labels for each block
    let labels: Vec<Label> = sorted_blocks.iter().map(|b| block_label(b)).collect();

    let mut blocks = Vec::new();
    for (i, block) in sorted_blocks.iter().enumerate() {
        let fall_label = labels.get(i + 1);
        blocks.push(lower_block(&mut ctx, block, fall_label));
    }

    Function {
        name: name.to_string(),
        params: vec![],
        ret_ty: None,
        blocks,
    }
}

// ============================================================
// Multiple functions → Program
// ============================================================

/// Lower all functions identified by ELF symbols into a `Program`.
///
/// For each `FuncSymbol`, extracts the corresponding instructions,
/// builds a per-function CFG, and lowers it to an LLIR `Function`.
pub fn lower_functions(instructions: &[Instruction], functions: &[FuncSymbol]) -> Program {
    let split = split_instructions_by_functions(instructions, functions);
    let funcs = split
        .into_iter()
        .map(|(name, insns)| {
            let cfg = Cfg::build(insns);
            lower_cfg(&cfg, &name)
        })
        .collect();
    Program { functions: funcs }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::graph::Cfg;
    use crate::disassemble::load_and_disassemble;
    use std::path::PathBuf;

    fn testcase_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testcases")
            .join(name)
    }

    fn build_cfg(name: &str) -> Cfg {
        let insns = load_and_disassemble(testcase_path(name)).unwrap();
        Cfg::build(insns)
    }

    #[test]
    fn test_lower_add() {
        let mut ctx = LowerCtx::new();
        let insn = Instruction {
            address: 0x1000,
            size: 4,
            bytes: [0; 4],
            mnemonic: "add".into(),
            operands: "a0, a1, a2".into(),
            rv_operands: vec![
                RvOperand::Reg(RvReg(10)),
                RvOperand::Reg(RvReg(11)),
                RvOperand::Reg(RvReg(12)),
            ],
            kind: InstructionKind::Regular,
        };
        let result = lower_instruction(&mut ctx, &insn);
        assert_eq!(result.len(), 1);
        let s = result[0].to_string();
        assert!(s.contains("binop(Add,"), "got: {s}");
    }

    #[test]
    fn test_lower_addw() {
        let mut ctx = LowerCtx::new();
        let insn = Instruction {
            address: 0x1000,
            size: 4,
            bytes: [0; 4],
            mnemonic: "addw".into(),
            operands: "a0, a1, a2".into(),
            rv_operands: vec![
                RvOperand::Reg(RvReg(10)),
                RvOperand::Reg(RvReg(11)),
                RvOperand::Reg(RvReg(12)),
            ],
            kind: InstructionKind::Regular,
        };
        let result = lower_instruction(&mut ctx, &insn);
        // trunc + trunc + binop + sext = 4 instructions
        assert_eq!(result.len(), 4, "addw should produce 4 LLIR instructions");
        assert!(result[0].to_string().contains("Trunc"));
        assert!(result[1].to_string().contains("Trunc"));
        assert!(result[2].to_string().contains("binop(Add,"));
        assert!(result[3].to_string().contains("SExt"));
    }

    #[test]
    fn test_lower_beq_terminator() {
        let mut ctx = LowerCtx::new();
        let insn = Instruction {
            address: 0x1000,
            size: 4,
            bytes: [0; 4],
            mnemonic: "beq".into(),
            operands: "a0, a1, 0x20".into(),
            rv_operands: vec![
                RvOperand::Reg(RvReg(10)),
                RvOperand::Reg(RvReg(11)),
                RvOperand::Imm(0x20),
            ],
            kind: InstructionKind::ConditionalBranch { target: 0x1020 },
        };
        let fall = Label("fallthrough".into());
        let (insts, term) = lower_terminator(&mut ctx, &insn, Some(&fall));
        assert_eq!(insts.len(), 1);
        assert!(insts[0].to_string().contains("cmp(Eq,"));
        match &term {
            Term::BrCond {
                then_label,
                else_label,
                ..
            } => {
                assert_eq!(then_label.0, "bb_1020");
                assert_eq!(else_label.0, "fallthrough");
            }
            other => panic!("expected BrCond, got {other}"),
        }
    }

    #[test]
    fn test_lower_ld() {
        let mut ctx = LowerCtx::new();
        let insn = Instruction {
            address: 0x1000,
            size: 4,
            bytes: [0; 4],
            mnemonic: "ld".into(),
            operands: "a0, 8(sp)".into(),
            rv_operands: vec![
                RvOperand::Reg(RvReg(10)),
                RvOperand::Mem {
                    base: RvReg::SP,
                    disp: 8,
                },
            ],
            kind: InstructionKind::Regular,
        };
        let result = lower_instruction(&mut ctx, &insn);
        // addr calculation + load = 2 instructions
        assert_eq!(
            result.len(),
            2,
            "ld with offset should produce 2 instructions"
        );
        assert!(result[0].to_string().contains("binop(Add,"));
        assert!(result[1].to_string().contains("load(i64,"));
    }

    #[test]
    fn test_lower_sd() {
        let mut ctx = LowerCtx::new();
        let insn = Instruction {
            address: 0x1000,
            size: 4,
            bytes: [0; 4],
            mnemonic: "sd".into(),
            operands: "a0, 8(sp)".into(),
            rv_operands: vec![
                RvOperand::Reg(RvReg(10)),
                RvOperand::Mem {
                    base: RvReg::SP,
                    disp: 8,
                },
            ],
            kind: InstructionKind::Regular,
        };
        let result = lower_instruction(&mut ctx, &insn);
        // addr calculation + store = 2 instructions
        assert_eq!(
            result.len(),
            2,
            "sd with offset should produce 2 instructions"
        );
        assert!(result[0].to_string().contains("binop(Add,"));
        assert!(result[1].to_string().contains("store(i64,"));
    }

    #[test]
    fn test_lower_x0_write_trap() {
        let mut ctx = LowerCtx::new();
        let insn = Instruction {
            address: 0x1000,
            size: 4,
            bytes: [0; 4],
            mnemonic: "add".into(),
            operands: "zero, a1, a2".into(),
            rv_operands: vec![
                RvOperand::Reg(RvReg::ZERO),
                RvOperand::Reg(RvReg(11)),
                RvOperand::Reg(RvReg(12)),
            ],
            kind: InstructionKind::Regular,
        };
        let result = lower_instruction(&mut ctx, &insn);
        assert_eq!(result.len(), 1);
        assert!(
            matches!(&result[0], Inst::Trap(r) if r == "write_x0"),
            "writing to x0 should produce trap"
        );
    }

    #[test]
    fn test_lower_if_cfg() {
        let cfg = build_cfg("if.out");
        let func = lower_cfg(&cfg, "main");

        // Every block should have exactly one terminator
        for block in &func.blocks {
            // term is always present by construction
            assert!(
                matches!(
                    &block.term,
                    Term::BrCond { .. }
                        | Term::Jmp(_)
                        | Term::Ret(_)
                        | Term::Trap(_)
                        | Term::Switch { .. }
                ),
                "block {} has invalid term: {}",
                block.label,
                block.term
            );
        }

        let output = func.to_string();
        assert!(output.contains("cmp("), "should contain cmp instructions");
        assert!(output.contains("br "), "should contain br terminator");
    }

    #[test]
    fn test_lower_hello_cfg_display() {
        let cfg = build_cfg("hello.out");
        let func = lower_cfg(&cfg, "main");
        let output = func.to_string();

        // Basic structure checks
        assert!(output.contains("func main"), "should contain function name");
        assert!(output.contains("call "), "should contain call instructions");
        assert!(
            output.contains("store(") || output.contains("load("),
            "should contain load or store instructions"
        );
    }

    #[test]
    fn test_lower_functions_hello() {
        let analysis = crate::disassemble::load_elf_analysis(testcase_path("hello.out")).unwrap();
        let program = lower_functions(&analysis.instructions, &analysis.functions);
        assert!(
            program.functions.len() >= 2,
            "hello.out should produce at least 2 functions (_start and main), got {}",
            program.functions.len()
        );
        let names: Vec<&str> = program.functions.iter().map(|f| f.name.as_str()).collect();
        assert!(
            names.contains(&"main"),
            "should contain main, got: {names:?}"
        );
    }

    // ========================================
    // Pseudo-instruction tests
    // ========================================

    fn make_pseudo_2op(mnemonic: &str) -> Instruction {
        Instruction {
            address: 0x1000,
            size: 4,
            bytes: [0; 4],
            mnemonic: mnemonic.into(),
            operands: "a0, a1".into(),
            rv_operands: vec![
                RvOperand::Reg(RvReg(10)), // a0
                RvOperand::Reg(RvReg(11)), // a1
            ],
            kind: InstructionKind::Regular,
        }
    }

    #[test]
    fn test_lower_sext_w() {
        let mut ctx = LowerCtx::new();
        let insn = make_pseudo_2op("sext.w");
        let result = lower_instruction(&mut ctx, &insn);
        // trunc i32 + add 0 + sext i64 = 3 instructions
        assert_eq!(
            result.len(),
            3,
            "sext.w should produce 3 LLIR instructions, got: {result:?}"
        );
        assert!(
            result[0].to_string().contains("Trunc"),
            "first should be Trunc"
        );
        assert!(
            result[1].to_string().contains("binop(Add,"),
            "second should be Add"
        );
        assert!(
            result[2].to_string().contains("SExt"),
            "third should be SExt"
        );
    }

    #[test]
    fn test_lower_negw() {
        let mut ctx = LowerCtx::new();
        let insn = make_pseudo_2op("negw");
        let result = lower_instruction(&mut ctx, &insn);
        // trunc + trunc + sub + sext = 4 instructions
        assert_eq!(result.len(), 4, "negw should produce 4 LLIR instructions");
        assert!(result[0].to_string().contains("Trunc"));
        assert!(result[1].to_string().contains("Trunc"));
        assert!(result[2].to_string().contains("binop(Sub,"));
        assert!(result[3].to_string().contains("SExt"));
    }

    #[test]
    fn test_lower_not() {
        let mut ctx = LowerCtx::new();
        let insn = make_pseudo_2op("not");
        let result = lower_instruction(&mut ctx, &insn);
        assert_eq!(result.len(), 1, "not should produce 1 LLIR instruction");
        let s = result[0].to_string();
        assert!(s.contains("binop(Xor,") && s.contains("-1"), "got: {s}");
    }

    #[test]
    fn test_lower_neg() {
        let mut ctx = LowerCtx::new();
        let insn = make_pseudo_2op("neg");
        let result = lower_instruction(&mut ctx, &insn);
        assert_eq!(result.len(), 1, "neg should produce 1 LLIR instruction");
        let s = result[0].to_string();
        assert!(s.contains("binop(Sub,"), "got: {s}");
    }

    #[test]
    fn test_lower_seqz() {
        let mut ctx = LowerCtx::new();
        let insn = make_pseudo_2op("seqz");
        let result = lower_instruction(&mut ctx, &insn);
        // cmp LtU + zext = 2 instructions
        assert_eq!(result.len(), 2, "seqz should produce 2 LLIR instructions");
        assert!(result[0].to_string().contains("cmp(LtU,"));
        assert!(result[1].to_string().contains("ZExt"));
    }

    #[test]
    fn test_lower_snez() {
        let mut ctx = LowerCtx::new();
        let insn = make_pseudo_2op("snez");
        let result = lower_instruction(&mut ctx, &insn);
        // cmp LtU + zext = 2 instructions
        assert_eq!(result.len(), 2, "snez should produce 2 LLIR instructions");
        assert!(result[0].to_string().contains("cmp(LtU,"));
        assert!(result[1].to_string().contains("ZExt"));
    }

    #[test]
    fn test_lower_ebreak() {
        let mut ctx = LowerCtx::new();
        let insn = Instruction {
            address: 0x1000,
            size: 2,
            bytes: [0; 4],
            mnemonic: "c.ebreak".into(),
            operands: "".into(),
            rv_operands: vec![],
            kind: InstructionKind::Regular,
        };
        let result = lower_instruction(&mut ctx, &insn);
        assert_eq!(result.len(), 1);
        assert!(
            matches!(&result[0], Inst::Trap(r) if r == "ebreak"),
            "got: {:?}",
            result[0]
        );
    }

    #[test]
    fn test_lower_donut_no_unknown_traps() {
        let path = testcase_path("donut.out");
        if !path.exists() {
            eprintln!("skipping: donut.out not found");
            return;
        }
        let analysis = crate::disassemble::load_elf_analysis(&path).unwrap();
        let program = lower_functions(&analysis.instructions, &analysis.functions);
        let output = program.to_string();
        let bad_traps: Vec<&str> = output
            .lines()
            .filter(|line| {
                line.contains("trap \"unknown_sext.w\"")
                    || line.contains("trap \"unknown_negw\"")
                    || line.contains("trap \"unknown_c.ebreak\"")
            })
            .collect();
        assert!(
            bad_traps.is_empty(),
            "donut.out should not have these unknown traps:\n{}",
            bad_traps.join("\n")
        );
    }

    #[test]
    fn test_lower_functions_each_has_terminator() {
        let analysis = crate::disassemble::load_elf_analysis(testcase_path("hello.out")).unwrap();
        let program = lower_functions(&analysis.instructions, &analysis.functions);
        for func in &program.functions {
            for block in &func.blocks {
                assert!(
                    matches!(
                        &block.term,
                        Term::BrCond { .. }
                            | Term::Jmp(_)
                            | Term::Ret(_)
                            | Term::Trap(_)
                            | Term::Switch { .. }
                    ),
                    "block {} in func {} has invalid term: {}",
                    block.label,
                    func.name,
                    block.term
                );
            }
        }
    }
}
