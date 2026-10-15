# Frontend — product UI (frontier-style)

Same role as ChatGPT / Claude / Grok **web client**: browser UI only.
Talks to the **backend API** on `:8000`.

| Port | Role |
|------|------|
| **3000** | This app (Bun + TypeScript) |
| **8000** | `src/sentinel` inference API |

```bash
cd frontend && bun install && bun run dev
```

```text
frontend/
├── public/     # index.html · product.html · assets
│   └── static/vendor/   # marked · DOMPurify · highlight.js (local, offline)
└── src/        # server · proxy · buddy · git
```

Model weights and training code do **not** live here — see `src/sentinel/` and `backend/README.md`.

## Vendored browser libs

`public/static/vendor/` — marked (markdown) · DOMPurify (sanitize) · highlight.js (code colors),
used for assistant chat bubbles. No CDN — the UI works fully offline.
Refresh copies: `bun add -d marked dompurify @highlightjs/cdn-assets && scripts/vendor.sh`.
