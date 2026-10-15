# Legacy static UI (optional)

**Canonical frontend:** `apps/web/public/` (Bun on port **3000**).

These HTML files are only served if `SERVE_UI=1` / `NEUROFIELD_SERVE_UI=1`.
Default backend is API-only so model and UI stay separate.

Do not develop new UI here — edit `apps/web/public/` instead.
See `docs/STRUCTURE.md`.
