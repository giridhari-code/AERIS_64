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
    ├── app/             # React app (App, Topbar, Sidebar, Welcome, Message, Composer, BuddyFloat)
    │   └── lib/         # markdown renderer · buddy float controller · stream transports · chat history
    ├── app-build.ts     # in-memory Bun.build bundle → /static/app.js (rebuild-on-request)
    ├── server.ts        # Hono: API proxy + static files + app bundle route
    └── buddy.ts · proxy.ts · git.ts
```

The React bundle is built in memory and served at `/static/app.js`; it rebuilds
automatically when files under `src/app/` change (mtime check). Legacy DOM
ids/classes (`#thread`, `#prompt`, `#btnNew`, `#buddy-float`, …) are kept, so
the CSS and the jsdom end-to-end tests keep working unchanged.

## Chat client

- **Streaming** — replies render token-by-token. Transport fallback chain:
  `/v1/ag-ui` (AG-UI events) → `/v1/chat/stream` (SSE tokens) → `/v1/chat`
  (buffered JSON). A transport is skipped when the server 404s/501s or replies
  non-SSE. `src/app/lib/stream.ts`.
- **Stop** — the composer turns into a Stop button while streaming; it aborts
  the fetch (the backend releases the model lock promptly, see
  `_TokenPump` in `packages/sentinel/serving/server.py`) and keeps the partial
  reply with a `stopped` marker.
- **Regenerate / Edit** — per-message actions (copy · regenerate · edit).
  Both reset the server-side session and replay the turn from the edited
  prompt (there is no server-side truncate API).
- **History sidebar** — chats persist in `localStorage` under
  `sentinel_chats_v1` (legacy `sentinel_session` key stays in sync). New
  chat / switch / rename-by-first-message / delete; `#btnMenu` toggles the
  sidebar (overlay on mobile).

Both streaming endpoints are proxied unbuffered by `proxyStream()` in
`src/proxy.ts`; the Bun server sets `idleTimeout: 255` so slow multi-second
token gaps are not reset mid-stream.

Model weights and training code do **not** live here — see `packages/sentinel/` and `backend/README.md`.

## Vendored browser libs

`public/static/vendor/` — marked (markdown) · DOMPurify (sanitize) · highlight.js (code colors),
used for assistant chat bubbles. No CDN — the UI works fully offline.
Refresh copies: `bun add -d marked dompurify @highlightjs/cdn-assets && scripts/vendor.sh`.
