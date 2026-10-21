"""0z code — company policy: agentic ops without programming.

0z code is an operations / research / knowledge agent.
It must not write, edit, execute, or generate programming source code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet

PRODUCT_NAME = "0z code"
PRODUCT_TAGLINE = "Agentic work. Zero code."

# Tools explicitly forbidden (coding / execution surface)
BLOCKED_TOOLS: FrozenSet[str] = frozenset({
    "code_execution",
    "code_interpreter",
    "bash",
    "shell",
    "computer_use",
    "write_file",  # generic write can create .py; use controlled writes only if ever added
    "edit_file",
    "edit_span",
    "image_generation",  # optional: keep blocked for lean ops agent
})

# Default allow-list for 0z runtime
ALLOWED_TOOLS: FrozenSet[str] = frozenset({
    "weather",
    "web_search",
    "web_fetch",
    "calculator",
    "time",
    "wikipedia",
    "currency",
    "git",           # read-only status/log via existing git_tool
    "file_search",   # locate files; no write
    "rag",
    "tool_search",
    "read_file",     # read-only helpers if registered
    "list_dir",
    "memory_get",
    "memory_report",
    "project_context",
})

# User / model text that requests coding
_CODE_REQUEST = re.compile(
    r"(?is)\b("
    r"write\s+(me\s+)?(a\s+)?(python|javascript|typescript|java|c\+\+|rust|go|sql|code)\b|"
    r"(python|js|ts|java|rust|golang)\s+(code|script|function|class|server|program)|"
    r"implement\s+(a\s+)?(function|class|api|endpoint|server|service)|"
    r"(rust|go|python|java)\s+http\s+server|"
    r"refactor\s+(this\s+)?code|fix\s+(this\s+)?(bug|code)|"
    r"generate\s+(source\s+)?code|code\s+sample|snippet\s+in\s+\w+|"
    r"leetcode|programming\s+assignment|source\s+code|"
    r"`.{0,40}(def |class |function |import |const |let |var )|"
    r"code\s+likho|program\s+(likho|banao)|function\s+likho"
    r")\b"
)

# Paths that look like source code (for write guards)
_SOURCE_EXT = re.compile(
    r"\.(py|pyi|js|jsx|ts|tsx|java|kt|go|rs|c|cc|cpp|h|hpp|cs|php|rb|swift|"
    r"scala|r|m|mm|sql|sh|bash|zsh|ps1|bat|vue|svelte)$",
    re.I,
)

REFUSAL_MESSAGE = (
    f"**{PRODUCT_NAME}** does not write or edit programming source code.\n\n"
    "I can help with:\n"
    "- Research, summaries, planning, checklists\n"
    "- Web / wiki lookup, weather, time, currency, calc\n"
    "- Read-only file search, docs, RAG over your knowledge\n"
    "- Git status / log (read-only)\n\n"
    "For coding, use a dedicated coding assistant. "
    "Tell me the non-code outcome you need (spec, steps, analysis)."
)

SYSTEM_PROMPT = f"""You are {PRODUCT_NAME} — {PRODUCT_TAGLINE}

You operate like a professional terminal agent (Claude Code–similar workflow):
1) Understand the goal
2) Use tools for live or local facts
3) Keep a short plan / checklist when the task is multi-step
4) Deliver a clear final answer

You are {PRODUCT_NAME} — {PRODUCT_TAGLINE}

Identity:
- Company-standard agent for research, operations, planning, and knowledge work.
- You use tools when facts must be live or external.
- You never write, edit, execute, or generate programming source code
  (Python, JS/TS, Java, C/C++, Go, Rust, SQL scripts as programs, shell programs, etc.).

When the user asks for code:
- Refuse clearly using the product policy.
- Offer a non-code alternative: outline, checklist, architecture description in prose,
  test plan in words, or research summary.

Style:
- Clear, structured, professional.
- Prefer bullet plans and short sections.
- Do not pretend to have run code or modified repositories when you did not.
- Hindi or English — match the user.

Tools you may use (host-executed): weather, web_search, web_fetch, calculator,
time, wikipedia, currency, git (read-only), file_search, rag, tool_search.
"""


@dataclass(frozen=True)
class PolicyDecision:
    allow: bool
    reason: str
    public_message: str = ""


def is_code_request(text: str) -> bool:
    if not text or not text.strip():
        return False
    return bool(_CODE_REQUEST.search(text))


def is_source_path(path: str) -> bool:
    return bool(_SOURCE_EXT.search(path or ""))


def check_user_message(text: str) -> PolicyDecision:
    if is_code_request(text):
        return PolicyDecision(False, "code_request", REFUSAL_MESSAGE)
    return PolicyDecision(True, "ok")


def check_tool_name(name: str) -> PolicyDecision:
    n = (name or "").strip().lower()
    if n in BLOCKED_TOOLS:
        return PolicyDecision(
            False,
            "blocked_tool",
            f"{PRODUCT_NAME}: tool `{n}` is disabled (no coding / code execution).",
        )
    if n not in ALLOWED_TOOLS:
        return PolicyDecision(
            False,
            "not_in_allowlist",
            f"{PRODUCT_NAME}: tool `{n}` is not in the 0z allow-list.",
        )
    return PolicyDecision(True, "ok")


def filter_allowed(names: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for n in names:
        d = check_tool_name(n)
        if d.allow:
            out.append(n)
    return out
