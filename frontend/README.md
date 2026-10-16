# Frontend — product UI (frontier-style, React)

Same role as ChatGPT / Claude / Grok **web client**: browser UI only.
Talks to the **backend API** on `:8000`.

| Port | Role |
|------|------|
| **3000** | This app (Bun + React 19) |
| **8000** | `sentinel` inference API (`packages/sentinel`) |

```bash
cd frontend && bun install && bun run dev
```

```text
frontend/
├── public/              # index.html (thin React shell) · static/app.css · assets
│   └── static/vendor/   # marked · DOMPurify · highlight.js (local, offline)
└── src/
    ├── app/             # React app (App, Topbar, Welcome, Message, Composer, BuddyFloat)
    │   └── lib/         # markdown renderer · buddy float controller
    ├── app-build.ts     # in-memory Bun.build bundle → /static/app.js (rebuild-on-request)
    ├── server.ts        # Hono: API proxy + static files + app bundle route
    └── buddy.ts · proxy.ts · git.ts
```

The React bundle is built in memory and served at `/static/app.js`; it rebuilds
automatically when files under `src/app/` change (mtime check). Legacy DOM
ids/classes (`#thread`, `#prompt`, `#btnNew`, `#buddy-float`, …) are kept, so
the CSS and the jsdom end-to-end tests keep working unchanged.

Model weights and training code do **not** live here — see `packages/sentinel/` and `backend/README.md`.

## Vendored browser libs

`public/static/vendor/` — marked (markdown) · DOMPurify (sanitize) · highlight.js (code colors),
used for assistant chat bubbles. No CDN — the UI works fully offline.
Refresh copies: `bun add -d marked dompurify @highlightjs/cdn-assets && scripts/vendor.sh`.
