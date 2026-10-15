# packages/ — model & library code

| Package | Role |
|---------|------|
| **sentinel/** | Frontier model core: architecture, train, tools, Megha, API impl |

This is the **model layer** (like an internal Python monorepo package).
Frontend never imports training code; backend only serves via `sentinel.api` / `sentinel.serving`.
