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
