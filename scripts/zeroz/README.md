# Open with: `0zcode`

# 0z code — Claude Code–similar agent (zero coding)

Not a clone. **Similar workflow**: project context, plan, tools, todos, slash CLI.

```bash
export PYTHONPATH=packages
# or: pip install -e .

python -m sentinel.zeroz.cli -i --no-model
```

## Interactive (similar to Claude Code)

```
0z> /help
0z> /context
0z> /tools
0z> /plan Weather in Delhi and convert 10 USD to INR
0z> /todo add Research SME AI pricing
0z> /todos
0z> Weather in Mumbai?
```

## Policy

- **No** source code write / edit / execute
- **Yes** research, weather, wiki, files read, git status, planning

## HTTP

```bash
python -m sentinel.zeroz.server
# POST /v1/chat
```


## Install command name `0zcode`

```bash
bash scripts/zeroz/install_0zcode.sh
source ~/.bashrc   # or ~/.zshrc
0zcode             # opens interactive agent
0zcode "Weather in Delhi?"
```

Or from repo without install:

```bash
./bin/0zcode
```
