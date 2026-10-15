# Buddy — ASCII companion

Decorative only. Does **not** write or debug code.

## Commands (chat)

- `/buddy` — hatch/show your buddy (deterministic from user id)
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

- Navbar me **Buddy** toggle (◉).
- On hone par buddy **navbar ke thoda neeche, right side floating** dikhta hai — chat/thread/chart ke **andar nahi**.
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

UI buttons under the float switch mood; idle auto-cycles expressions about every 12s.

## Auto mood (chat-linked)

Float shows **only name + sprite** (no manual mood buttons).
Animation follows AERIS chat:

- User question → think / blink
- Greeting → wave
- Model reply → talk
- Funny / excited text → happy / excited
- Double-click float to hide; topbar ◉ toggles on/off
