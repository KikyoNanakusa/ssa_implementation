use std::path::Path;

use anyhow::{Result, anyhow};

use capstone::arch::ArchOperand;
use capstone::arch::riscv::{ArchExtraMode, ArchMode, RiscVOperand};
use capstone::prelude::*;
use goblin::elf::Elf;
use goblin::elf::sym::STT_FUNC;

use crate::cfg::operand::{RvOperand, RvReg};
use crate::cfg::{Instruction, InstructionKind};

/// A function symbol extracted from the ELF symbol table.
#[derive(Debug, Clone)]
pub struct FuncSymbol {
    pub name: String,
    pub addr: u64,
    pub size: u64,
}

/// Result of analyzing an ELF binary: disassembled instructions + function symbols.
pub struct ElfAnalysis {
    pub instructions: Vec<Instruction>,
    pub functions: Vec<FuncSymbol>,
}

const GRP_JUMP: u8 = 1;
const GRP_CALL: u8 = 2;
const GRP_RET: u8 = 3;
const GRP_INT: u8 = 4;

/// Read an ELF binary from `path` and disassemble its `.text` section.
pub fn load_and_disassemble(path: impl AsRef<Path>) -> Result<Vec<Instruction>> {
    let elf_bytes = std::fs::read(&path)
        .map_err(|e| anyhow!("failed to read ELF from {}: {e}", path.as_ref().display()))?;
    let (base_addr, code) = extract_text_section(&elf_bytes).ok_or_else(|| {
        anyhow!(
            "failed to find .text section in {}",
            path.as_ref().display()
        )
    })?;
    Ok(disassemble(code, base_addr))
}

/// Read an ELF binary, disassemble `.text`, and extract function symbols.
pub fn load_elf_analysis(path: impl AsRef<Path>) -> Result<ElfAnalysis> {
    let elf_bytes = std::fs::read(&path)
        .map_err(|e| anyhow!("failed to read ELF from {}: {e}", path.as_ref().display()))?;
    let (base_addr, code) = extract_text_section(&elf_bytes).ok_or_else(|| {
        anyhow!(
            "failed to find .text section in {}",
            path.as_ref().display()
        )
    })?;
    let instructions = disassemble(code, base_addr);
    let functions = extract_function_symbols(&elf_bytes).unwrap_or_default();
    Ok(ElfAnalysis {
        instructions,
        functions,
    })
}

/// Extract `STT_FUNC` symbols from the ELF symbol table.
/// Returns symbols sorted by address.
pub fn extract_function_symbols(elf_bytes: &[u8]) -> Option<Vec<FuncSymbol>> {
    let elf = Elf::parse(elf_bytes).ok()?;
    let mut syms: Vec<FuncSymbol> = elf
        .syms
        .iter()
        .filter(|sym| sym.st_type() == STT_FUNC && sym.st_size > 0)
        .filter_map(|sym| {
            let name = elf.strtab.get_at(sym.st_name)?.to_string();
            Some(FuncSymbol {
                name,
                addr: sym.st_value,
                size: sym.st_size,
            })
        })
        .collect();
    syms.sort_by_key(|s| s.addr);
    Some(syms)
}

/// Split a flat instruction stream into per-function instruction vectors.
///
/// Each function is identified by `[addr, addr+size)` from `FuncSymbol`.
/// Instructions not belonging to any function are dropped.
pub fn split_instructions_by_functions(
    instructions: &[Instruction],
    functions: &[FuncSymbol],
) -> Vec<(String, Vec<Instruction>)> {
    functions
        .iter()
        .filter_map(|func| {
            let start = func.addr;
            let end = func.addr + func.size;
            let func_insns: Vec<Instruction> = instructions
                .iter()
                .filter(|insn| insn.address >= start && insn.address < end)
                .cloned()
                .collect();
            if func_insns.is_empty() {
                None
            } else {
                Some((func.name.clone(), func_insns))
            }
        })
        .collect()
}

/// Extract the `.text` section from an ELF binary.
/// Returns `(virtual_address, section_data)`.
pub fn extract_text_section(elf_bytes: &[u8]) -> Option<(u64, &[u8])> {
    let elf = Elf::parse(elf_bytes).ok()?;
    for section in &elf.section_headers {
        let name = elf.shdr_strtab.get_at(section.sh_name)?;
        if name == ".text" {
            let offset = section.sh_offset as usize;
            let size = section.sh_size as usize;
            let data = &elf_bytes[offset..offset + size];
            return Some((section.sh_addr, data));
        }
    }
    None
}

/// Disassemble RISC-V machine code into a vector of `Instruction`.
pub fn disassemble(code: &[u8], base_addr: u64) -> Vec<Instruction> {
    let cs = Capstone::new()
        .riscv()
        .mode(ArchMode::RiscV64)
        .extra_mode([ArchExtraMode::RiscVC].iter().copied())
        .detail(true)
        .build()
        .expect("failed to create Capstone instance");

    let insns = cs
        .disasm_all(code, base_addr)
        .expect("failed to disassemble");

    insns
        .as_ref()
        .iter()
        .map(|insn| {
            let (kind, rv_operands) = classify_and_extract(&cs, insn);
            let mut bytes = [0u8; 4];
            let raw = insn.bytes();
            bytes[..raw.len()].copy_from_slice(raw);

            Instruction {
                address: insn.address(),
                size: insn.len() as u8,
                bytes,
                mnemonic: insn.mnemonic().unwrap_or("").to_string(),
                operands: insn.op_str().unwrap_or("").to_string(),
                rv_operands,
                kind,
            }
        })
        .collect()
}

fn classify_and_extract(cs: &Capstone, insn: &capstone::Insn) -> (InstructionKind, Vec<RvOperand>) {
    let detail = match cs.insn_detail(insn) {
        Ok(d) => d,
        Err(_) => return (InstructionKind::Regular, vec![]),
    };

    let rv_operands = extract_rv_operands(&detail);

    let groups: Vec<u8> = detail.groups().iter().map(|g| g.0).collect();
    let has = |id: u8| groups.contains(&id);

    if has(GRP_RET) {
        return (InstructionKind::Return, rv_operands);
    }
    if has(GRP_INT) {
        return (InstructionKind::Syscall, rv_operands);
    }

    // ebreak / c.ebreak: mnemonic-based detection (may not have GRP_INT)
    let mnemonic = insn.mnemonic().unwrap_or("");
    if mnemonic == "ebreak" || mnemonic == "c.ebreak" {
        return (InstructionKind::Syscall, rv_operands);
    }

    let is_call = has(GRP_CALL);
    let is_jump = has(GRP_JUMP);

    if !is_call && !is_jump {
        return (InstructionKind::Regular, rv_operands);
    }

    let target = extract_branch_target(&detail, insn.address());

    let op_str = insn.op_str().unwrap_or("");

    // Detect return: c.jr ra, jalr zero, ra, 0
    if is_jump && is_return_pattern(mnemonic, op_str) {
        return (InstructionKind::Return, rv_operands);
    }

    let kind = if is_call {
        match target {
            Some(t) => InstructionKind::Call { target: t },
            None => InstructionKind::IndirectCall,
        }
    } else {
        let is_conditional = matches!(
            mnemonic,
            "beq"
                | "bne"
                | "blt"
                | "bge"
                | "bltu"
                | "bgeu"
                | "c.beqz"
                | "c.bnez"
                | "beqz"
                | "bnez"
                | "blez"
                | "bgez"
                | "bltz"
                | "bgtz"
                | "bgt"
                | "ble"
                | "bgtu"
                | "bleu"
        );

        match (is_conditional, target) {
            (true, Some(t)) => InstructionKind::ConditionalBranch { target: t },
            (true, None) => InstructionKind::ConditionalBranch { target: 0 },
            (false, Some(t)) => InstructionKind::Jump { target: t },
            (false, None) => InstructionKind::IndirectJump,
        }
    };

    (kind, rv_operands)
}

fn extract_rv_operands(detail: &capstone::InsnDetail) -> Vec<RvOperand> {
    detail
        .arch_detail()
        .operands()
        .iter()
        .filter_map(|op| match op {
            ArchOperand::RiscVOperand(RiscVOperand::Reg(id)) => {
                RvReg::from_capstone(id.0 as u16).map(RvOperand::Reg)
            }
            ArchOperand::RiscVOperand(RiscVOperand::Imm(v)) => Some(RvOperand::Imm(*v)),
            ArchOperand::RiscVOperand(RiscVOperand::Mem(m)) => {
                let base = RvReg::from_capstone(m.base().0 as u16).unwrap_or(RvReg::ZERO);
                Some(RvOperand::Mem {
                    base,
                    disp: m.disp(),
                })
            }
            _ => None,
        })
        .collect()
}

/// Check if an instruction is a return (jump to ra).
/// Capstone does not tag RISC-V ret with CS_GRP_RET, so we detect it
/// by mnemonic and operand: `c.jr ra` or `jalr zero, ra, 0`.
fn is_return_pattern(mnemonic: &str, op_str: &str) -> bool {
    match mnemonic {
        "c.jr" => op_str == "ra",
        "jalr" => op_str.starts_with("zero, ra,"),
        _ => false,
    }
}

/// Extract the absolute branch/jump target from instruction operands.
///
/// For RISC-V, capstone stores the raw PC-relative offset in the immediate
/// operand. We add the instruction address to get the absolute target.
fn extract_branch_target(detail: &capstone::InsnDetail, insn_addr: u64) -> Option<u64> {
    for op in detail.arch_detail().operands() {
        if let ArchOperand::RiscVOperand(RiscVOperand::Imm(imm)) = op {
            return Some((insn_addr as i64 + imm) as u64);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn testcase_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testcases")
            .join(name)
    }

    fn load_testcase(name: &str) -> Vec<Instruction> {
        load_and_disassemble(testcase_path(name)).unwrap()
    }

    #[test]
    fn test_extract_text_section() {
        let elf_bytes = std::fs::read(testcase_path("hello.out")).unwrap();
        let result = extract_text_section(&elf_bytes);
        assert!(result.is_some());
        let (addr, data) = result.unwrap();
        assert!(addr > 0);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_disassemble_hello() {
        let insns = load_testcase("hello.out");
        assert!(!insns.is_empty());

        // Every instruction should have a valid size (2 or 4)
        for insn in &insns {
            assert!(
                insn.size == 2 || insn.size == 4,
                "unexpected size: {}",
                insn.size
            );
            assert!(!insn.mnemonic.is_empty());
        }

        // Should contain at least one Call and one Return
        assert!(
            insns
                .iter()
                .any(|i| matches!(i.kind, InstructionKind::Call { .. }))
        );
        assert!(
            insns
                .iter()
                .any(|i| matches!(i.kind, InstructionKind::Return))
        );
    }

    #[test]
    fn test_disassemble_if_has_conditional_branch() {
        let insns = load_testcase("if.out");

        let branches: Vec<_> = insns
            .iter()
            .filter(|i| matches!(i.kind, InstructionKind::ConditionalBranch { .. }))
            .collect();
        assert!(
            !branches.is_empty(),
            "if.out should contain conditional branches"
        );

        // The bge instruction in main should have a target within .text
        for br in &branches {
            if let InstructionKind::ConditionalBranch { target } = br.kind {
                assert!(target > 0, "branch target should be non-zero");
            }
        }
    }

    #[test]
    fn test_disassemble_for_has_jump_and_branch() {
        let insns = load_testcase("for.out");

        // for loop: unconditional jump (j) and conditional branch (bge)
        assert!(
            insns
                .iter()
                .any(|i| matches!(i.kind, InstructionKind::Jump { .. })),
            "for.out should contain an unconditional jump"
        );
        assert!(
            insns
                .iter()
                .any(|i| matches!(i.kind, InstructionKind::ConditionalBranch { .. })),
            "for.out should contain a conditional branch"
        );
    }

    #[test]
    fn test_disassemble_function_call() {
        let insns = load_testcase("function_call.out");

        let calls: Vec<_> = insns
            .iter()
            .filter(|i| matches!(i.kind, InstructionKind::Call { .. }))
            .collect();
        // At least: main calls print_hello, print_hello calls puts
        assert!(
            calls.len() >= 2,
            "function_call.out should have at least 2 calls, found {}",
            calls.len()
        );
    }

    #[test]
    fn test_disassemble_switch() {
        let insns = load_testcase("switch.out");
        assert!(!insns.is_empty());

        // switch/case generates branches
        let has_branch = insns.iter().any(|i| {
            matches!(
                i.kind,
                InstructionKind::ConditionalBranch { .. }
                    | InstructionKind::Jump { .. }
                    | InstructionKind::IndirectJump
            )
        });
        assert!(
            has_branch,
            "switch.out should contain branch/jump instructions"
        );
    }

    #[test]
    fn test_instruction_addresses_are_sequential() {
        let insns = load_testcase("hello.out");

        for pair in insns.windows(2) {
            assert_eq!(
                pair[0].end_address(),
                pair[1].address,
                "instructions should be sequential: {} followed by {}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn test_extract_function_symbols() {
        let elf_bytes = std::fs::read(testcase_path("hello.out")).unwrap();
        let syms = extract_function_symbols(&elf_bytes).unwrap();
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"_start"),
            "should contain _start, got: {names:?}"
        );
        assert!(
            names.contains(&"main"),
            "should contain main, got: {names:?}"
        );
    }

    #[test]
    fn test_function_call_symbols() {
        let elf_bytes = std::fs::read(testcase_path("function_call.out")).unwrap();
        let syms = extract_function_symbols(&elf_bytes).unwrap();
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"main"),
            "should contain main, got: {names:?}"
        );
        assert!(
            names.contains(&"print_hello"),
            "should contain print_hello, got: {names:?}"
        );
    }

    #[test]
    fn test_split_instructions_by_functions() {
        let elf_bytes = std::fs::read(testcase_path("function_call.out")).unwrap();
        let (base_addr, code) = extract_text_section(&elf_bytes).unwrap();
        let insns = disassemble(code, base_addr);
        let syms = extract_function_symbols(&elf_bytes).unwrap();
        let split = split_instructions_by_functions(&insns, &syms);

        for (name, func_insns) in &split {
            let sym = syms.iter().find(|s| &s.name == name).unwrap();
            for insn in func_insns {
                assert!(
                    insn.address >= sym.addr && insn.address < sym.addr + sym.size,
                    "instruction {:#x} should be within function {} [{:#x}..{:#x})",
                    insn.address,
                    name,
                    sym.addr,
                    sym.addr + sym.size,
                );
            }
        }

        let names: Vec<&str> = split.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"main"), "split should contain main");
        assert!(
            names.contains(&"print_hello"),
            "split should contain print_hello"
        );
    }

    #[test]
    fn test_return_instructions_present() {
        // All testcases have functions that return
        for name in &[
            "hello.out",
            "if.out",
            "for.out",
            "function_call.out",
            "switch.out",
        ] {
            let insns = load_testcase(name);
            assert!(
                insns
                    .iter()
                    .any(|i| matches!(i.kind, InstructionKind::Return)),
                "{name} should contain at least one return instruction"
            );
        }
    }
}
