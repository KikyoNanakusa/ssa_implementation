# SSA Implementation

ELF形式かつRISC-V64アーキテクチャのバイナリをSSA形式の中間表現, 及びより低レベルな中間表現に変更することができます. 

## Usage

```bash 
# LLIR(Low Level IR)を生成
cargo run llir <Binary Path>

# Minimal SSAを生成
cargo run minimal-ssa <Binary Path>

# Pruned SSAを生成
cargo run pruned-ssa <Binary Path>
```

`testcases`ディレクトリにC言語で書かれた適当なテストケースとMakefileを用意しています. 
