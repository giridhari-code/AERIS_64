# LEGACY — Python SENTINEL

**Status: retired as the primary model architecture (2026-11).**

The active architecture is **Spider-Web Looped MoE** in Rust:

- `crates/spider-core`
- `docs/ARCHITECTURE.md`
- `docs/SPIDER_WEB_MOE.md`

This package (`packages/sentinel`) retains historical two-speed memory / neural-field / serve code for reference and gradual API migration. **Do not add new model features here.**

New training and inference work targets `spider-core`.
