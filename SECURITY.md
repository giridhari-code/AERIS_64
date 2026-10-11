# Security policy

## Reporting a vulnerability
Please **do not open a public issue**. Email **giriisdev@gmail.com** with steps to reproduce,
the affected version/commit and the impact. You will get a reply within a few days.

## Supported versions
Only the latest commit on `main` receives fixes.

## Deployment guidance
- The model API is open when `NEUROFIELD_API_KEYS` is empty. Always set keys on anything reachable from a network.
- `/v1/tools/shell` and `/v1/tools/python` **always require an API key** (503 otherwise; `NEUROFIELD_TOOLS_ALLOW_OPEN=1`
  is for local development only). Set `NEUROFIELD_SANDBOX=0` to disable them entirely.
- The sandbox is **defence in depth, not a security boundary** (`python3` is allowed). Run the server in a container
  or VM with no secrets, no credentials in the environment and no outbound network.
- Bind to `127.0.0.1` unless a reverse proxy with TLS and authentication sits in front.
- Checkpoints: only `.safetensors` are loaded by default. `NEUROFIELD_ALLOW_PICKLE=1` (legacy `.pt`) executes code from the
  file; use it only for files you created yourself.
- Never paste access tokens into chats, issues or commits. Revoke any token that has been exposed.
