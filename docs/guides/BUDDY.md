# Buddy — ASCII companion

Decorative only. Does **not** write or debug code.

## Commands (chat)

Sab commands `/buddy …`, `--buddy …` aur bare `buddy …` — teeno spellings chalte hain.

- `/buddy` / `/buddy status` — short status (incubation **float pe**, chat me nahi)
- `/buddy on` / `/buddy off` — float show/hide
- `/buddy egg` — float ka egg refresh
- `/buddy pet` — pet animation with hearts
- `/buddy mimic [text]` — joker nakal: last message (ya diya gaya text) ki funny copy
- `/buddy joker [text]` — full joker-mode mimicry
- `/buddy mood` — aapki last message ki emotion padho (happy/sad/angry/tired/…)
- `/buddy stats` — curiosity / calm / loyalty bars + personality
- `/buddy who` (alias `link`) — user + AERIS-64 link info
- `/buddy about` — AERIS-64 companion info
- `/buddy dance|party|wave|happy|sleep|excited` — float animation
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
- **Hatching sirf float pe** hoti hai: chat kabhi egg art nahi dikhata — `/buddy` "see Buddy float" bolta hai. Progress bar + `Egg · 3/8 steps · next ~4m` meta float pe.
- Hatch hote hi **one-time celebration overlay** (poori screen, chat ke bahar): crack → confetti → reveal; click / Enter / Esc se dismiss. Per user sirf ek baar (`sentinel_buddy_hatch_seen:<id>`).
- Float header: **AERIS-64 · Buddy** — buddy account + AERIS-64 dono se linked (`/buddy who`, `/buddy about`).

## Account link (one user, one hatch)

- Pehli baar login karne par aapka **guest (device) egg + buddy account me adopt**
  ho jata hai (`POST /v1/buddy/adopt`) — companion key pin hoti hai, isliye
  **same buddy har device pe** milta hai.
- Account ka *started* egg (jisme messages ho chuke) kabhi overwrite nahi hota;
  sirf untouched (khaali) egg adopt hone deta hai — dobara login par adopt
  safely no-op hota hai.
- **One user, one time**: hatch hone ke baad progress unlock ho jata hai,
  dobara hatch nahi hota; steps buddy ko wipe nahi karte.

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
