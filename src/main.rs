use std::path::Path;
use std::time::Instant;

use anyhow::Result;

mod cfg;
mod disassemble;
mod llir;
mod ssa;

fn main() -> Result<()> {
    let analysis = disassemble::load_elf_analysis(Path::new("testcases/donut.out"))?;
    let program = llir::lower::lower_functions(&analysis.instructions, &analysis.functions);
    // println!("{program}");
    for func in program.functions.iter() {
        let start = Instant::now();
        let ssa = ssa::construct::construct_minimal_ssa(func);
        // let ssa = ssa::construct::construct_pruned_ssa(func);
        let elapsed = start.elapsed();
        println!(
            "[timing] construct_pruned_ssa({name}): {elapsed_ms:.3} ms\n\n",
            name = func.name,
            elapsed_ms = elapsed.as_secs_f64() * 1000.0
        );
        println!("=== {name} ===\n{ssa}", name = func.name);
    }
    Ok(())
}
