# Backend has **no** UI assets here

Duplicate `index.html` / `product.html` / `logo.svg` were **removed** so frontend and backend do not mix.

| Role | Location |
|------|----------|
| **Frontend (only)** | `apps/web/public/` — Bun `:3000` |
| **Backend** | `src/sentinel/serving/` — JSON API `:8000` |

If `SERVE_UI=1`, Python serves files from **`apps/web/public`** (same folder), not from here.
Prefer always using Bun for UI. See `docs/STRUCTURE.md`.
