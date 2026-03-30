use std::path::Path;
use std::time::Instant;

use anyhow::Result;

mod cfg;
mod disassemble;
mod llir;
mod ssa;

enum Command {
    MinimalSSA, 
    PrunedSSA, 
    LLIR, 
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<String>>();
    let command = args.get(1).unwrap_or_else(|| {
        eprintln!("Usage: {} <command>", args[0]);
        eprintln!("Commands:");
        eprintln!("  minimal-ssa");
        eprintln!("  pruned-ssa");
        eprintln!("  llir");
        std::process::exit(1);
    });
    let binary_path = args.get(2).unwrap_or_else(|| {
        eprintln!("Usage: {} <command> <binary path>", args[0]);
        std::process::exit(1);
    });
    let command = match command.as_str() {
        "minimal-ssa" => Command::MinimalSSA,
        "pruned-ssa" => Command::PrunedSSA,
        "llir" => Command::LLIR,
        _ => {
            eprintln!("Invalid command: {}", command);
            std::process::exit(1);
        }
    };

    match command {
        Command::MinimalSSA => {
            let analysis = disassemble::load_elf_analysis(Path::new(binary_path))?;
            let program = llir::lower::lower_functions(&analysis.instructions, &analysis.functions);
            for func in program.functions.iter() {
                let ssa = ssa::construct::construct_minimal_ssa(func);
                println!("=== {name} ===\n{ssa}", name = func.name);
            }
        },
        Command::PrunedSSA => {
            let analysis = disassemble::load_elf_analysis(Path::new(binary_path))?;
            let program = llir::lower::lower_functions(&analysis.instructions, &analysis.functions);
            for func in program.functions.iter() {
                let ssa = ssa::construct::construct_pruned_ssa(func);
                println!("{ssa}");
            }
        },
        Command::LLIR => {
            let analysis = disassemble::load_elf_analysis(Path::new(binary_path))?;
            let program = llir::lower::lower_functions(&analysis.instructions, &analysis.functions);
            println!("{program}");
        }
    }
    Ok(())
}
