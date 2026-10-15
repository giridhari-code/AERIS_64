# Buddy — ASCII companion

Decorative only. Does **not** write or debug code.

## Commands (chat)

- `/buddy` — hatch/show your buddy (deterministic from user id)
- `/buddy on` — show the floating companion
- `/buddy off` — hide it
- `/buddy pet` — pet animation with hearts
- `/buddy help` — short help

## API

- `GET /v1/buddy` — JSON + ascii for API key owner
- `POST /v1/buddy` `{"command":"/buddy pet"}`

## Design

- 18 species with fixed sprites
- Rarity: common 60%, uncommon 25%, rare 10%, epic 4%, legendary 1%
- Shiny 1% independent of rarity
- 6 eye styles, 8 hats (rarer hats gated)
- Name + personality from user hash (no external LLM required)

## Web UI (floating)

- Navbar me koi buddy icon **nahi** — chat me **`/buddy on`** (ya `--buddy on`) se show, **`/buddy off`** (ya `--buddy off`) se hide.
- On hone par buddy **navbar ke thoda neeche, top-right floating** dikhta hai — content ke **upar overlay** (display push nahi hota), thread ke **andar nahi**.
- Soft **bob animation** + idle blink; **Pet** se short pet animation.
- Preference `localStorage.sentinel_buddy_on` me save.

## Animation types

ASCII frame sets + CSS motion on the float panel:

| Type | Effect |
|------|--------|
| idle | Soft presence |
| blink | Eyes close |
| float | Hover bob |
| talk | Talking frames |
| think | Thought dots |
| wave | Wave |
| happy | Happy eyes |
| sleep | Zzz |
| excited | Bounce/shake |
| look_left / look_right | Glance |
| pet | Hearts |

Koi manual mood buttons nahi — float shows **name + sprite only**; mood chat se auto (neeche dekho). Burst ke ~4.5s baad idle pe wapas.

## Auto mood (chat-linked)

Float shows **only name + sprite** (no manual mood buttons).
Animation follows AERIS chat:

- User question → think / blink
- Greeting → wave
- Model reply → talk
- Funny / excited text → happy / excited
- Double-click float to hide; chat `/buddy on` / `/buddy off` toggles on/off


## One HTML file (single source)

| File | Used when |
|------|-----------|
| `frontend/public/index.html` | **Bun UI** on `:3000` (normal) |

Python backend (API-only) bhi yahi file serve karta hai jab `SERVE_UI=1` (ya `NEUROFIELD_SERVE_UI=1`) ho — koi duplicate copy nahi. Prefer Bun `:3000` for chat.
