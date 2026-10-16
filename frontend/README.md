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

## Accounts (login / signup)

- The app probes `/v1/auth/status` on load. When the backend has accounts
  enabled it shows a full-screen **log in / sign up** card
  (`src/app/components/AuthGate.tsx`); "Continue as guest" skips it in one
  click (hidden when the server sets `required: true`).
- If the auth API is missing (older server, offline dev, stubbed tests) the
  app falls back to guest mode silently — chat is never blocked by auth.
- Sessions: localStorage `sentinel_auth_v1` (`{kind:"user", token, user}` or
  `{kind:"guest"}`). Stored tokens are re-validated through `/v1/auth/me`;
  a 401 clears the session and returns to the gate.
- Chats are per identity: `sentinel_chats_v1` entries carry an `owner`
  (`"guest"` or the account id) and the sidebar only shows the current
  user's chats. The topbar chip (`#btnUser`) shows the account and offers
  log out / log in.
- `Authorization: Bearer <token>` rides every model call
  (`src/app/lib/auth.ts`, `src/app/lib/stream.ts`) and the Bun edge forwards
  it (`bearerOf` in `src/server.ts`).
- Server side: `packages/sentinel/serving/accounts.py`, routes `/v1/auth/*`.
  Env: `NEUROFIELD_AUTH_DB` (default `data/auth/accounts.db`),
  `NEUROFIELD_AUTH=0` disables accounts, `NEUROFIELD_AUTH_REQUIRED=1` makes
  the chat endpoints require a login (or X-API-Key),
  `NEUROFIELD_AUTH_TTL_DAYS` (default 30).

## Buddy companion (egg hatching)

- Pokémon-style reveal gate: while the egg is unhatched the float shows the
  egg instead of the buddy. Every user chat message is one incubation step
  (`POST /v1/buddy/step`, fired by `App.send`); passive time also counts
  (default one step per 5 min). The meta line reads
  `Egg · 3/8 steps · next ~4m` and the wobble speeds up near hatching.
- The float auto-opens once on the first visit (the egg is the discovery
  moment); an explicit `/buddy off` is respected. On hatching it plays the
  crack sequence, reveals the buddy and fires a one-time branded celebration
  overlay **outside the chat** (click/Enter/Esc dismiss; seen-flag kept in
  `sentinel_buddy_hatch_seen:<id>`). Chat never shows hatch art — `/buddy`
  replies point at the float.
- Identity follows the account: signed-in users incubate `u-<account id>`
  (same egg on any device), guests keep the per-browser device id. First
  login adopts the guest egg into the account (`POST /v1/buddy/adopt`,
  pinned companion key) so the same buddy carries over.
- New chat commands surface through `meta_rag`: `mimic`/`joker` (echo
  spoof), `mood` (emotion sensing), `stats`, `who`, `about`,
  `dance|wave|happy|sleep|excited`. The app routes `meta_rag.buddy_anim` to
  `window.sentinelBuddyAnim`; hooks `sentinelBuddyAnim/Refresh/Adopt`.
- Server side: `packages/sentinel/buddy/eggs.py` (stdlib JSON store
  `NEUROFIELD_BUDDY_DB`, default `data/buddy/eggs.json`; passive period
  `NEUROFIELD_BUDDY_EGG_PERIOD`, default 300 s).

Model weights and training code do **not** live here — see `packages/sentinel/` and `backend/README.md`.

## Vendored browser libs

`public/static/vendor/` — marked (markdown) · DOMPurify (sanitize) · highlight.js (code colors),
used for assistant chat bubbles. No CDN — the UI works fully offline.
Refresh copies: `bun add -d marked dompurify @highlightjs/cdn-assets && scripts/vendor.sh`.
