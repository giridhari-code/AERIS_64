"""Multi-format corpus loader → unified plain text for SENTINEL training.

Supported file types
--------------------
  .txt .md .markdown .text     plain / markdown (headers kept as text; # comments only if whole-line #)
  .jsonl                       one JSON object per line
  .json                        list | {"data":[...]} | single record | ShareGPT
  .csv .tsv                    tabular SFT / text columns
  .parquet                     columnar (optional: pyarrow/pandas)
  .html .htm                   tags stripped to text
  .csv.gz .jsonl.gz .txt.gz    gzip-compressed of the above

Recognised record schemas
-------------------------
  messages / conversations     ChatML, OpenAI, ShareGPT (role/from + content/value)
  user+assistant               also prompt/response, instruction/output, question/answer, ...
  text / content / body        plain document field
  instruction + optional input + output   Alpaca

HuggingFace
-----------
  Pass ``hf:dataset_name`` or ``hf:dataset_name:config:split`` as a --data path
  (requires ``datasets`` package).

No demo hardcoding — schema normalization only.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

logger = logging.getLogger(__name__)

_USER_KEYS = (
    "user", "prompt", "instruction", "question", "input", "query", "human", "human_input",
)
_ASSIST_KEYS = (
    "assistant", "response", "output", "answer", "completion", "target", "gpt", "bot",
)
_TEXT_KEYS = ("text", "content", "body", "document", "raw", "article", "passage", "story")
_USER_ROLES = frozenset({"user", "human", "prompt", "instruction", "prompter", "customer"})
_ASSIST_ROLES = frozenset({"assistant", "gpt", "bot", "model", "response", "chatgpt", "ai", "system_assistant"})
# ShareGPT sometimes uses system; we keep as comment-style so train can skip via #
_SYSTEM_ROLES = frozenset({"system", "context"})

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")


# --------------------------------------------------------------------------- helpers
def _pick(obj: dict[str, Any], keys: Iterable[str]) -> str | None:
    for k in keys:
        if k in obj and obj[k] is not None:
            v = obj[k]
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, (int, float, bool)):
                return str(v)
            if isinstance(v, (list, dict)):
                # nested content blocks
                t = record_to_text(v)
                if t:
                    return t
    return None


def _content_to_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or p.get("value") or ""))
            else:
                parts.append(str(p))
        return " ".join(x for x in parts if x).strip()
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or content.get("value") or "").strip()
    return str(content).strip()


def _messages_to_text(messages: list) -> str:
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(
            m.get("role") or m.get("from") or m.get("speaker") or m.get("author") or ""
        ).lower().strip()
        content = _content_to_str(
            m.get("content") if m.get("content") is not None else m.get("value") if m.get("value") is not None else m.get("text")
        )
        if not content:
            continue
        if role in _USER_ROLES:
            parts.append(f"User: {content}")
        elif role in _ASSIST_ROLES:
            parts.append(f"Assistant: {content}")
        elif role in _SYSTEM_ROLES:
            parts.append(f"# system: {content}")
        else:
            # unknown role → plain
            parts.append(content)
    return "\n".join(parts)


def record_to_text(obj: Any) -> str:
    """Normalize one record (dict / list / str) to training text."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and (
            "role" in obj[0] or "from" in obj[0] or "content" in obj[0] or "value" in obj[0]
        ):
            return _messages_to_text(obj)
        return "\n\n".join(t for t in (record_to_text(x) for x in obj) if t)
    if not isinstance(obj, dict):
        return str(obj).strip()

    # Chat lists
    for key in ("messages", "conversations", "conversation", "chat", "turns", "dialogue"):
        if key in obj and isinstance(obj[key], list):
            return _messages_to_text(obj[key])

    # Alpaca: instruction (+ optional input) + output
    instr = obj.get("instruction")
    if isinstance(instr, str) and instr.strip():
        user_parts = [instr.strip()]
        inp = obj.get("input")
        if isinstance(inp, str) and inp.strip():
            user_parts.append(inp.strip())
        out = _pick(obj, _ASSIST_KEYS) or (
            str(obj["output"]).strip() if isinstance(obj.get("output"), str) else None
        )
        user = "\n".join(user_parts)
        if out:
            return f"User: {user}\nAssistant: {out}"
        return f"User: {user}"

    user = _pick(obj, _USER_KEYS)
    assistant = _pick(obj, _ASSIST_KEYS)
    if user and assistant:
        return f"User: {user}\nAssistant: {assistant}"
    if user and not assistant:
        return f"User: {user}"
    if assistant and not user:
        return f"Assistant: {assistant}"

    text = _pick(obj, _TEXT_KEYS)
    if text:
        return text
    return ""


# --------------------------------------------------------------------------- file readers
def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz" or path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")  # type: ignore
    return path.open(encoding="utf-8", errors="ignore")


def _strip_gz_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    # return effective suffix e.g. .jsonl
    for ext in (".jsonl", ".json", ".csv", ".tsv", ".txt", ".md", ".markdown", ".html", ".htm", ".parquet"):
        if name.endswith(ext):
            return ext
    return Path(name).suffix.lower()


def load_txt(path: Path) -> str:
    with _open_text(path) as f:
        raw = f.read()
    lines = []
    for ln in raw.splitlines():
        s = ln.strip()
        # keep markdown headers; only skip pure comment lines that are "# comment"
        # and not markdown AT1 that might be intentional — skip if starts with "# "
        # and looks like a file comment (optional). Keep all non-empty for md.
        if s.startswith("#") and not s.startswith("# "):
            # "#foo" rare; treat "# " as possible md title — still include
            pass
        if s.startswith("# ") and path.suffix.lower() in {".txt", ".text"}:
            # txt comments
            continue
        lines.append(ln)
    return "\n".join(lines).strip()


def load_markdown(path: Path) -> str:
    with _open_text(path) as f:
        raw = f.read()
    # light cleanup: images → alt text, links → label
    raw = _MD_IMAGE_RE.sub(r"\1", raw)
    raw = _MD_LINK_RE.sub(r"\1", raw)
    return raw.strip()


def load_html(path: Path) -> str:
    with _open_text(path) as f:
        raw = f.read()
    # remove script/style
    raw = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.I | re.S)
    text = _HTML_TAG_RE.sub(" ", raw)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_jsonl(path: Path) -> str:
    blocks: list[str] = []
    with _open_text(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("%s:%d skip invalid JSON: %s", path.name, i, e)
                continue
            t = record_to_text(obj)
            if t:
                blocks.append(t)
    return "\n\n".join(blocks)


def load_json(path: Path) -> str:
    with _open_text(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "examples", "items", "conversations", "messages", "dataset", "rows"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            return record_to_text(data)
    if not isinstance(data, list):
        return record_to_text(data)
    blocks = [record_to_text(x) for x in data]
    return "\n\n".join(b for b in blocks if b)


def _csv_row_to_record(row: dict[str, str]) -> dict[str, Any]:
    # normalize keys lower-strip
    norm = {str(k).strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k is not None}
    return norm


def load_csv(path: Path, delimiter: str | None = None) -> str:
    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" or path.name.lower().endswith(".tsv.gz") else ","
    blocks: list[str] = []
    with _open_text(path) as f:
        # sniff if possible
        sample = f.read(4096)
        f.seek(0)
        if delimiter == "," and sample:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
                reader = csv.DictReader(f, dialect=dialect)
            except csv.Error:
                f.seek(0)
                reader = csv.DictReader(f, delimiter=delimiter)
        else:
            reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            rec = _csv_row_to_record(row)
            t = record_to_text(rec)
            if not t:
                # fallback: join all cell values
                vals = [str(v) for v in rec.values() if v]
                t = " ".join(vals)
            if t:
                blocks.append(t)
    return "\n\n".join(blocks)


def load_parquet(path: Path) -> str:
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Reading .parquet requires pandas (+ pyarrow). pip install pandas pyarrow"
        ) from e
    df = pd.read_parquet(path)
    blocks: list[str] = []
    for rec in df.to_dict(orient="records"):
        # stringify nan
        clean = {k: ("" if v is None or (isinstance(v, float) and v != v) else v) for k, v in rec.items()}
        t = record_to_text(clean)
        if t:
            blocks.append(t)
    return "\n\n".join(blocks)


def load_hf(spec: str) -> str:
    """spec: hf:name | hf:name:config | hf:name:config:split | hf:name:::split"""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "HuggingFace paths require `datasets`. pip install datasets"
        ) from e
    raw = spec[3:] if spec.startswith("hf:") else spec
    parts = raw.split(":")
    name = parts[0]
    config = parts[1] if len(parts) > 1 and parts[1] else None
    split = parts[2] if len(parts) > 2 and parts[2] else "train"
    logger.info("loading HF dataset %s config=%s split=%s", name, config, split)
    if config:
        ds = load_dataset(name, config, split=split)
    else:
        ds = load_dataset(name, split=split)
    blocks: list[str] = []
    for row in ds:
        t = record_to_text(dict(row))
        if t:
            blocks.append(t)
    return "\n\n".join(blocks)


def load_corpus_file(path: str | Path) -> str:
    """Load one local file (any supported type) into plain training text."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ext = _strip_gz_suffix(path)
    if ext in {".txt", ".text"}:
        return load_txt(path)
    if ext in {".md", ".markdown"}:
        return load_markdown(path)
    if ext in {".html", ".htm"}:
        return load_html(path)
    if ext == ".jsonl":
        return load_jsonl(path)
    if ext == ".json":
        return load_json(path)
    if ext == ".csv":
        return load_csv(path, delimiter=",")
    if ext == ".tsv":
        return load_csv(path, delimiter="\t")
    if ext == ".parquet":
        return load_parquet(path)
    # fallback
    logger.warning("unknown suffix %s for %s — trying text", ext, path)
    return load_txt(path)


def load_corpus(paths: list[str | Path]) -> tuple[str, list[dict[str, Any]]]:
    """Load many paths (files or hf: specs). Returns (merged_text, manifest)."""
    parts: list[str] = []
    manifest: list[dict[str, Any]] = []
    for p in paths:
        s = str(p)
        if s.startswith("hf:"):
            text = load_hf(s)
            n = len(text)
            parts.append(text)
            manifest.append({"path": s, "bytes": n, "chars": n, "format": "hf"})
            logger.info("loaded %s (%d chars, format=hf)", s, n)
            continue
        path = Path(p)
        if not path.is_file():
            logger.warning("skip missing data file: %s", path)
            continue
        text = load_corpus_file(path)
        n = len(text)
        parts.append(text)
        manifest.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "chars": n,
            "format": _strip_gz_suffix(path) or path.suffix.lower() or ".txt",
        })
        logger.info("loaded %s (%s, %d chars)", path, manifest[-1]["format"], n)
    merged = "\n\n".join(t for t in parts if t).strip() + "\n"
    return merged, manifest


SUPPORTED_FORMATS = (
    ".txt", ".text", ".md", ".markdown",
    ".jsonl", ".json",
    ".csv", ".tsv",
    ".parquet",
    ".html", ".htm",
    ".gz",  # with inner type
    "hf:...",
)
