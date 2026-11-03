# spider-core

Rust implementation of **Spider-Web Looped MoE** for AERIS_64.

```text
Prelude (2) → Spider-MoE × r → Coda (2) → LM head
```

## Build

```bash
cd crates/spider-core
cargo test
cargo build --release
```

## Status

Scaffold: control flow, modules, stability hooks, CSR graph stubs.  
Weights / real attn-FFN / grouped GEMM kernels still TODO.

Design: `docs/SPIDER_WEB_MOE.md`, `docs/ARCHITECTURE.md`.

## Layout

- `model/` — prelude, recurrent, spider_moe, coda, early_exit
- `graph/` — edge generator, CSR adjacency
- `moe/` — router, experts, load balance
- `stability/` — variance / norm monitors
