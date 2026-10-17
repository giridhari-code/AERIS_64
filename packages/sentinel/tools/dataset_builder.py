"""Dataset Builder — scalable collect / clean / quality / balance / export.

Optimizations vs v1:
  - concurrent URL + file batch (ThreadPoolExecutor)
  - directory walk + glob for bulk local ingest
  - chunked large-file read; near-dupe paragraph fingerprints
  - rewritten compact state (docs/ + index.json) for scale
  - imbalance + target-complete notifications
  - training-ready export (jsonl records with text + meta)
  - env limits: workers, max docs, max file bytes, fetch timeout

Env:
  NEUROFIELD_DATASET_BUILDER=1
  NEUROFIELD_DATASET_WORK_DIR=path
  NEUROFIELD_DATASET_WORKERS=8
  NEUROFIELD_DATASET_MAX_DOCS=50000
  NEUROFIELD_DATASET_MAX_FILE_BYTES=8000000
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Defaults / limits
# ---------------------------------------------------------------------------

DEFAULT_CATEGORY_TARGETS: dict[str, int] = {
    "general_language": 300_000,
    "science_factual": 180_000,
    "mathematics": 180_000,
    "reasoning_logic": 200_000,
    "hinglish": 140_000,
}

CATEGORY_ALIASES = {
    "general": "general_language",
    "general_language_high_quality_text": "general_language",
    "science": "science_factual",
    "science_factual_knowledge": "science_factual",
    "math": "mathematics",
    "mathematics": "mathematics",
    "reasoning": "reasoning_logic",
    "reasoning_logic": "reasoning_logic",
    "logic": "reasoning_logic",
    "hinglish": "hinglish",
}

_TEXT_SUFFIXES = {".txt", ".md", ".html", ".htm", ".jsonl", ".json", ".csv", ".pdf"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def normalize_category(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return CATEGORY_ALIASES.get(key, key or "general_language")


def workers_default() -> int:
    return max(1, min(32, _env_int("NEUROFIELD_DATASET_WORKERS", 8)))


def max_docs_default() -> int:
    return max(100, _env_int("NEUROFIELD_DATASET_MAX_DOCS", 50_000))


def max_file_bytes() -> int:
    return max(50_000, _env_int("NEUROFIELD_DATASET_MAX_FILE_BYTES", 8_000_000))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class DocumentRecord:
    doc_id: str
    text: str
    source: str = ""
    collected_at: float = 0.0
    license_note: str = "unknown"
    category: str = "general_language"
    quality: str = "medium"
    quality_score: float = 0.5
    quality_flags: list[str] = field(default_factory=list)
    needs_review: bool = False
    char_count: int = 0
    word_count: int = 0
    token_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_training_row(self) -> dict[str, Any]:
        """Training-ready row (text + metadata; no demo fields)."""
        return {
            "id": self.doc_id,
            "text": self.text,
            "category": self.category,
            "source": self.source,
            "license": self.license_note,
            "quality": self.quality,
            "quality_score": self.quality_score,
            "tokens": self.token_count,
            "words": self.word_count,
            "chars": self.char_count,
            "collected_at": self.collected_at,
            "needs_review": self.needs_review,
        }


@dataclass
class BuilderReport:
    action: str
    ok: bool
    message: str
    documents: int = 0
    tokens: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        lines = [
            f"DatasetBuilder [{self.action}]: {'ok' if self.ok else 'error'}",
            self.message,
        ]
        if self.documents:
            lines.append(f"documents={self.documents}")
        if self.tokens:
            lines.append(f"tokens≈{self.tokens}")
        if self.details:
            lines.append(json.dumps(self.details, ensure_ascii=False, indent=2)[:6000])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML / clean / tokens / quality
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "nav", "footer", "header", "aside", "noscript", "form"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "header", "aside", "noscript", "form"}:
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data: str) -> None:
        if self._skip == 0 and data and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        return "\n".join(self._chunks)


def html_to_text(raw: str) -> str:
    p = _HTMLTextExtractor()
    try:
        p.feed(raw or "")
        p.close()
    except Exception:
        return re.sub(r"<[^>]+>", " ", raw or "")
    return p.text()


def approximate_tokens(text: str) -> int:
    t = text or ""
    if not t:
        return 0
    dev = len(re.findall(r"[\u0900-\u097F]", t))
    if dev > max(20, len(t) // 10):
        return max(1, int(len(t) / 3.2))
    return max(1, int(max(len(t.split()) * 1.3, len(t) / 4)))


def count_tokens(text: str, tokenizer: Any = None) -> int:
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "encode"):
                ids = tokenizer.encode(text)
                return len(ids) if ids is not None else approximate_tokens(text)
        except Exception:
            pass
    try:
        from sentinel.tokenizer.char import CharTokenizer

        tok = CharTokenizer()
        if hasattr(tok, "encode"):
            return len(tok.encode(text) or [])
    except Exception:
        pass
    return approximate_tokens(text)


_BOILERPLATE = re.compile(
    r"(?i)\b(subscribe|cookie policy|all rights reserved|click here|sign up|"
    r"newsletter|advertisement|related articles|share this|follow us|"
    r"accept (all )?cookies|privacy policy)\b"
)
_MULTI_NL = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    t = text or ""
    if "<html" in t[:500].lower() or "<body" in t[:800].lower():
        t = html_to_text(t)
    lines: list[str] = []
    for line in t.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if _BOILERPLATE.search(s) and len(s) < 140:
            continue
        lines.append(s)
    t = _MULTI_NL.sub("\n\n", "\n".join(lines))
    t = _MULTI_SPACE.sub(" ", t)
    return t.strip()


def content_fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()[:32]


def paragraph_fingerprints(text: str) -> set[str]:
    """Near-duplicate detection at paragraph grain."""
    fps: set[str] = set()
    for para in re.split(r"\n\s*\n", text or ""):
        p = re.sub(r"\s+", " ", para.strip().lower())
        if len(p) < 80:
            continue
        fps.add(hashlib.sha256(p.encode()).hexdigest()[:20])
    return fps


def score_quality(text: str, *, source: str = "") -> tuple[str, float, list[str], bool]:
    flags: list[str] = []
    t = (text or "").strip()
    words = t.split()
    if len(t) < 80:
        flags.append("too_short")
    if len(t) > 500_000:
        flags.append("very_long")
    if len(words) >= 20:
        ratio = len(set(w.lower() for w in words)) / max(1, len(words))
        if ratio < 0.35:
            flags.append("repetitive")
    if t.count("\ufffd") > 3 or "\x00" in t:
        flags.append("encoding_noise")
    letters = len(re.findall(r"[A-Za-z\u0900-\u097F]", t))
    if t and letters / max(1, len(t)) < 0.25:
        flags.append("low_alpha_ratio")
    # weak source signal (not final truth)
    if source and not source.startswith(("http://", "https://", "/")):
        flags.append("source_unclear")

    score = 0.75
    for f, d in (
        ("too_short", 0.35),
        ("repetitive", 0.25),
        ("encoding_noise", 0.3),
        ("low_alpha_ratio", 0.2),
        ("source_unclear", 0.05),
    ):
        if f in flags:
            score -= d
    if len(words) > 120 and "repetitive" not in flags:
        score += 0.1
    score = max(0.0, min(1.0, score))
    label = "high" if score >= 0.7 and not flags else ("medium" if score >= 0.4 else "low")
    needs_review = label == "low" or score < 0.55 or "encoding_noise" in flags
    return label, score, flags, needs_review


_RISKY_LICENSE = re.compile(
    r"(?i)\b(all rights reserved|copyright|proprietary|no[t]? for training|"
    r"personal use only|creative commons(?!.*zero)|cc by-nc)\b"
)


def license_review_flag(license_note: str, source: str) -> bool:
    if not (license_note or "").strip() or (license_note or "").lower() in {"unknown", "unclear"}:
        return True
    return bool(_RISKY_LICENSE.search(f"{license_note} {source}"))


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _read_local_file(path: Path) -> str:
    limit = max_file_bytes()
    size = path.stat().st_size
    suf = path.suffix.lower()
    if size > limit and suf not in {".jsonl"}:
        # stream first chunk only for huge files
        raw = path.read_bytes()[:limit]
    else:
        raw = path.read_bytes()
        if len(raw) > limit:
            raw = raw[:limit]

    if suf in {".html", ".htm"}:
        return html_to_text(raw.decode("utf-8", errors="replace"))
    if suf == ".pdf":
        try:
            import pypdf  # type: ignore

            reader = pypdf.PdfReader(path.open("rb"))
            parts = [(page.extract_text() or "") for page in reader.pages[:80]]
            return "\n".join(parts)
        except Exception as e:
            return f"[pdf_extract_failed:{type(e).__name__}]"
    if suf == ".jsonl":
        lines: list[str] = []
        n = 0
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            n += 1
            if n > 5000:
                break
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    lines.append(
                        str(obj.get("text") or obj.get("content") or obj.get("body") or json.dumps(obj))
                    )
                else:
                    lines.append(str(obj))
            except json.JSONDecodeError:
                lines.append(line)
        return "\n\n".join(lines)
    if suf == ".json":
        try:
            obj = json.loads(raw.decode("utf-8", errors="replace"))
            if isinstance(obj, dict) and "text" in obj:
                return str(obj["text"])
            if isinstance(obj, list):
                bits = []
                for item in obj[:2000]:
                    if isinstance(item, dict):
                        bits.append(str(item.get("text") or item.get("content") or item))
                    else:
                        bits.append(str(item))
                return "\n\n".join(bits)
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return raw.decode("utf-8", errors="replace")
    if suf == ".csv":
        # best-effort: join non-empty cells
        text = raw.decode("utf-8", errors="replace")
        return re.sub(r",+", " ", text)
    return raw.decode("utf-8", errors="replace")


def _fetch_url(url: str, *, timeout: float = 20.0, max_bytes: int | None = None) -> str:
    cap = max_bytes or max_file_bytes()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AERIS64-DatasetBuilder/2.0 (research; +local)"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        data = resp.read(cap + 1)
        if len(data) > cap:
            data = data[:cap]
        charset = resp.headers.get_content_charset() or "utf-8"
        text = data.decode(charset, errors="replace")
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "html" in ctype or text.lstrip()[:20].lower().startswith(("<!doctype", "<html")):
            return html_to_text(text)
        return text


def _doc_id(source: str, text: str) -> str:
    h = hashlib.sha256(f"{source}\n{content_fingerprint(text)}".encode()).hexdigest()[:16]
    return f"doc_{h}"


def expand_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files + directories into a flat file list (scaled local collect)."""
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            out.append(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in _TEXT_SUFFIXES:
                    out.append(child)
                    if len(out) >= 20_000:
                        return out
        else:
            # glob pattern
            for child in sorted(Path().glob(str(p))):
                if child.is_file():
                    out.append(child)
    return out


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    def __init__(
        self,
        work_dir: Optional[str | Path] = None,
        category_targets: Optional[dict[str, int]] = None,
        tokenizer: Any = None,
        max_docs: Optional[int] = None,
        workers: Optional[int] = None,
    ) -> None:
        env_dir = os.environ.get("NEUROFIELD_DATASET_WORK_DIR") or os.environ.get(
            "AERIS_DATASET_WORK_DIR"
        )
        self.work_dir = Path(work_dir or env_dir or "artifacts/dataset_builder")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "docs").mkdir(exist_ok=True)
        self.category_targets = dict(category_targets or DEFAULT_CATEGORY_TARGETS)
        self.tokenizer = tokenizer
        self.max_docs = max_docs or max_docs_default()
        self.workers = workers or workers_default()
        self.documents: dict[str, DocumentRecord] = {}
        self._fingerprints: set[str] = set()
        self._para_fps: set[str] = set()
        self._load_state()

    def _index_path(self) -> Path:
        return self.work_dir / "index.json"

    def _doc_path(self, doc_id: str) -> Path:
        return self.work_dir / "docs" / f"{doc_id}.json"

    def _load_state(self) -> None:
        idx = self._index_path()
        if idx.is_file():
            try:
                data = json.loads(idx.read_text(encoding="utf-8"))
                for doc_id in data.get("ids") or []:
                    dp = self._doc_path(doc_id)
                    if not dp.is_file():
                        continue
                    obj = json.loads(dp.read_text(encoding="utf-8"))
                    rec = DocumentRecord(
                        **{k: obj[k] for k in DocumentRecord.__dataclass_fields__ if k in obj}
                    )
                    self.documents[rec.doc_id] = rec
                    self._fingerprints.add(content_fingerprint(rec.text))
                    self._para_fps |= paragraph_fingerprints(rec.text)
                return
            except Exception:
                pass
        # legacy state.jsonl
        legacy = self.work_dir / "state.jsonl"
        if legacy.is_file():
            try:
                for line in legacy.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    rec = DocumentRecord(
                        **{k: obj[k] for k in DocumentRecord.__dataclass_fields__ if k in obj}
                    )
                    self.documents[rec.doc_id] = rec
                    self._fingerprints.add(content_fingerprint(rec.text))
                    self._para_fps |= paragraph_fingerprints(rec.text)
            except Exception:
                pass

    def _persist_index(self) -> None:
        try:
            self._index_path().write_text(
                json.dumps({"ids": list(self.documents.keys()), "updated": time.time()}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _persist_doc(self, rec: DocumentRecord) -> None:
        try:
            self._doc_path(rec.doc_id).write_text(
                json.dumps(rec.to_dict(), ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def add_document(
        self,
        text: str,
        *,
        source: str = "",
        category: str = "general_language",
        license_note: str = "unknown",
        clean: bool = True,
    ) -> Optional[DocumentRecord]:
        if len(self.documents) >= self.max_docs:
            return None
        body = clean_text(text) if clean else (text or "")
        if len(body.strip()) < 40:
            return None
        fp = content_fingerprint(body)
        if fp in self._fingerprints:
            return None
        # paragraph near-dupe: drop if >60% paragraphs already seen
        paras = paragraph_fingerprints(body)
        if paras:
            overlap = len(paras & self._para_fps) / max(1, len(paras))
            if overlap >= 0.6:
                return None

        q_label, q_score, flags, review = score_quality(body, source=source)
        lic_review = license_review_flag(license_note, source)
        rec = DocumentRecord(
            doc_id=_doc_id(source, body),
            text=body,
            source=source,
            collected_at=time.time(),
            license_note=license_note or "unknown",
            category=normalize_category(category),
            quality=q_label,
            quality_score=q_score,
            quality_flags=flags,
            needs_review=review or lic_review,
            char_count=len(body),
            word_count=len(body.split()),
            token_count=count_tokens(body, self.tokenizer),
            meta={"license_flagged": lic_review},
        )
        self.documents[rec.doc_id] = rec
        self._fingerprints.add(fp)
        self._para_fps |= paras
        self._persist_doc(rec)
        self._persist_index()
        return rec

    def collect_paths(
        self,
        paths: Iterable[str | Path],
        *,
        category: str = "general_language",
        license_note: str = "unknown",
    ) -> BuilderReport:
        files = expand_paths(paths)
        added = skipped = 0
        errors: list[str] = []

        def _one(path: Path) -> tuple[str, Optional[DocumentRecord], str]:
            try:
                text = _read_local_file(path)
                rec = self.add_document(
                    text,
                    source=str(path.resolve()),
                    category=category,
                    license_note=license_note,
                )
                return ("ok", rec, "")
            except Exception as e:
                return ("err", None, f"{path}:{type(e).__name__}")

        # sequential add is safer for shared state; parallelize IO reads
        texts: list[tuple[Path, str]] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futs = {pool.submit(_read_local_file, f): f for f in files}
            for fut in as_completed(futs):
                f = futs[fut]
                try:
                    texts.append((f, fut.result()))
                except Exception as e:
                    errors.append(f"{f}:{type(e).__name__}")
                    skipped += 1

        for path, text in texts:
            rec = self.add_document(
                text,
                source=str(path.resolve()),
                category=category,
                license_note=license_note,
            )
            if rec:
                added += 1
            else:
                skipped += 1

        return BuilderReport(
            action="collect_paths",
            ok=True,
            message=f"files={len(files)} added={added} skipped={skipped}",
            documents=added,
            tokens=self.total_tokens(),
            details={"errors": errors[:30], "error_count": len(errors)},
        )

    def collect_urls(
        self,
        urls: Iterable[str],
        *,
        category: str = "general_language",
        license_note: str = "public_web_review",
        timeout: float = 20.0,
    ) -> BuilderReport:
        url_list = [u.strip() for u in urls if (u or "").strip()]
        added = skipped = 0
        errors: list[str] = []

        def _fetch_one(u: str) -> tuple[str, str, str]:
            if not u.startswith(("http://", "https://")):
                return (u, "", "bad_url")
            try:
                return (u, _fetch_url(u, timeout=timeout), "")
            except Exception as e:
                return (u, "", type(e).__name__)

        results: list[tuple[str, str, str]] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futs = [pool.submit(_fetch_one, u) for u in url_list]
            for fut in as_completed(futs):
                results.append(fut.result())

        for u, text, err in results:
            if err or not text:
                errors.append(f"{u[:80]}:{err or 'empty'}")
                skipped += 1
                continue
            rec = self.add_document(
                text, source=u, category=category, license_note=license_note
            )
            if rec:
                added += 1
            else:
                skipped += 1

        return BuilderReport(
            action="collect_urls",
            ok=True,
            message=f"urls={len(url_list)} added={added} skipped={skipped}",
            documents=added,
            tokens=self.total_tokens(),
            details={"errors": errors[:30], "error_count": len(errors)},
        )

    def total_tokens(self, *, category: Optional[str] = None) -> int:
        cat = normalize_category(category) if category else None
        return sum(
            d.token_count
            for d in self.documents.values()
            if not cat or d.category == cat
        )

    def total_words(self) -> int:
        return sum(d.word_count for d in self.documents.values())

    def total_chars(self) -> int:
        return sum(d.char_count for d in self.documents.values())

    def category_status(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, target in self.category_targets.items():
            docs = [d for d in self.documents.values() if d.category == name]
            toks = sum(d.token_count for d in docs)
            words = sum(d.word_count for d in docs)
            chars = sum(d.char_count for d in docs)
            pct = min(100.0, 100.0 * toks / target) if target else 100.0
            out.append(
                {
                    "name": name,
                    "target_tokens": target,
                    "current_tokens": toks,
                    "words": words,
                    "chars": chars,
                    "doc_count": len(docs),
                    "remaining": max(0, target - toks),
                    "progress_pct": round(pct, 2),
                    "complete": toks >= target,
                }
            )
        return out

    def imbalance_report(self) -> dict[str, Any]:
        rows = self.category_status()
        filled = [r for r in rows if r["target_tokens"] > 0]
        ratios = [
            r["current_tokens"] / r["target_tokens"] for r in filled if r["target_tokens"]
        ]
        return {
            "categories": rows,
            "over_target": [r["name"] for r in rows if r["current_tokens"] > r["target_tokens"]],
            "under_target": [r["name"] for r in rows if r["current_tokens"] < r["target_tokens"]],
            "complete": [r["name"] for r in rows if r.get("complete")],
            "max_fill_ratio": max(ratios) if ratios else 0.0,
            "min_fill_ratio": min(ratios) if ratios else 0.0,
        }

    def progress_report(self) -> BuilderReport:
        rows = self.category_status()
        total = self.total_tokens()
        target_sum = sum(self.category_targets.values())
        complete = [r["name"] for r in rows if r.get("complete")]
        msg = f"total_tokens≈{total} / target={target_sum}"
        if complete:
            msg += f" | complete: {', '.join(complete)}"
        return BuilderReport(
            action="progress",
            ok=True,
            message=msg,
            documents=len(self.documents),
            tokens=total,
            details={
                "categories": rows,
                "words": self.total_words(),
                "chars": self.total_chars(),
                "needs_review": sum(1 for d in self.documents.values() if d.needs_review),
                "quality_counts": {
                    "high": sum(1 for d in self.documents.values() if d.quality == "high"),
                    "medium": sum(1 for d in self.documents.values() if d.quality == "medium"),
                    "low": sum(1 for d in self.documents.values() if d.quality == "low"),
                },
                "imbalance": self.imbalance_report(),
                "max_docs": self.max_docs,
                "workers": self.workers,
            },
        )

    def balance(self, *, drop_low_quality_first: bool = True) -> BuilderReport:
        removed = 0
        for name, target in self.category_targets.items():
            docs = [d for d in self.documents.values() if d.category == name]
            current = sum(d.token_count for d in docs)
            if current <= target:
                continue
            docs_sorted = sorted(
                docs,
                key=lambda d: (
                    0 if (drop_low_quality_first and d.quality == "low") else 1,
                    0 if d.needs_review else 1,
                    d.quality_score,
                    -d.token_count,
                ),
            )
            for d in docs_sorted:
                if current <= target:
                    break
                current -= d.token_count
                self.documents.pop(d.doc_id, None)
                try:
                    self._doc_path(d.doc_id).unlink(missing_ok=True)
                except Exception:
                    pass
                removed += 1
        self._fingerprints = {content_fingerprint(d.text) for d in self.documents.values()}
        self._para_fps = set()
        for d in self.documents.values():
            self._para_fps |= paragraph_fingerprints(d.text)
        self._persist_index()
        return BuilderReport(
            action="balance",
            ok=True,
            message=f"removed={removed} documents to match targets",
            documents=len(self.documents),
            tokens=self.total_tokens(),
            details={"categories": self.category_status()},
        )

    def export(
        self,
        fmt: str = "jsonl",
        *,
        out_name: Optional[str] = None,
        min_quality: Optional[str] = None,
        training_ready: bool = False,
    ) -> BuilderReport:
        fmt = (fmt or "jsonl").lower().strip()
        q_order = {"low": 0, "medium": 1, "high": 2}
        min_q = q_order.get((min_quality or "low").lower(), 0)
        docs = [d for d in self.documents.values() if q_order.get(d.quality, 0) >= min_q]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if fmt in {"train", "training", "training_jsonl"}:
            training_ready = True
            fmt = "jsonl"
        if fmt == "txt":
            path = self.work_dir / (out_name or f"export_{stamp}.txt")
            path.write_text("\n\n---\n\n".join(d.text for d in docs), encoding="utf-8")
        elif fmt == "csv":
            path = self.work_dir / (out_name or f"export_{stamp}.csv")
            with path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "doc_id",
                        "category",
                        "quality",
                        "quality_score",
                        "token_count",
                        "word_count",
                        "char_count",
                        "source",
                        "license_note",
                        "needs_review",
                    ],
                )
                w.writeheader()
                for d in docs:
                    w.writerow(
                        {
                            "doc_id": d.doc_id,
                            "category": d.category,
                            "quality": d.quality,
                            "quality_score": d.quality_score,
                            "token_count": d.token_count,
                            "word_count": d.word_count,
                            "char_count": d.char_count,
                            "source": d.source,
                            "license_note": d.license_note,
                            "needs_review": d.needs_review,
                        }
                    )
        else:
            path = self.work_dir / (out_name or f"export_{stamp}.jsonl")
            with path.open("w", encoding="utf-8") as f:
                for d in docs:
                    row = d.to_training_row() if training_ready else d.to_dict()
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return BuilderReport(
            action="export",
            ok=True,
            message=f"wrote {path}",
            documents=len(docs),
            tokens=sum(d.token_count for d in docs),
            details={
                "path": str(path),
                "format": fmt,
                "training_ready": training_ready,
            },
        )


# ---------------------------------------------------------------------------
# Tool entry
# ---------------------------------------------------------------------------

_SESSION: Optional[DatasetBuilder] = None


def get_builder(work_dir: Optional[str] = None) -> DatasetBuilder:
    global _SESSION
    if _SESSION is None or (work_dir and str(_SESSION.work_dir) != str(work_dir)):
        _SESSION = DatasetBuilder(work_dir=work_dir)
    return _SESSION


def dataset_builder_report(
    action: str = "progress",
    *,
    paths: Optional[list[str]] = None,
    urls: Optional[list[str]] = None,
    category: str = "general_language",
    license_note: str = "unknown",
    export_format: str = "jsonl",
    work_dir: Optional[str] = None,
    text: Optional[str] = None,
    training_ready: bool = False,
) -> str:
    if os.environ.get("NEUROFIELD_DATASET_BUILDER", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return "DatasetBuilder disabled."
    b = get_builder(work_dir)
    act = (action or "progress").strip().lower()
    if act in {"progress", "status", "categories"}:
        return b.progress_report().to_text()
    if act in {"imbalance", "balance_report"}:
        return BuilderReport(
            "imbalance", True, "category fill ratios", len(b.documents), b.total_tokens(), b.imbalance_report()
        ).to_text()
    if act in {"collect", "collect_paths", "add_files"} and paths:
        return b.collect_paths(paths, category=category, license_note=license_note).to_text()
    if act in {"collect_urls", "urls", "fetch"} and urls:
        return b.collect_urls(urls, category=category, license_note=license_note).to_text()
    if act in {"add_text", "ingest"} and text:
        rec = b.add_document(text, source="inline", category=category, license_note=license_note)
        if not rec:
            return "DatasetBuilder: text skipped (short/duplicate/limit)."
        return BuilderReport(
            "add_text", True, f"added {rec.doc_id} quality={rec.quality}", 1, rec.token_count,
            {"category": rec.category, "needs_review": rec.needs_review},
        ).to_text()
    if act in {"balance", "trim"}:
        return b.balance().to_text()
    if act in {"export", "write"}:
        return b.export(export_format, training_ready=training_ready).to_text()
    if act in {"export_train", "training"}:
        return b.export("jsonl", training_ready=True).to_text()
    if act in {"help", "?"}:
        return (
            "DatasetBuilder v2 actions: progress | imbalance | collect_paths | collect_urls | "
            "add_text | balance | export | export_train\n"
            "Scale: workers=%d max_docs=%d\n"
            "Categories: %s"
            % (workers_default(), max_docs_default(), ", ".join(DEFAULT_CATEGORY_TARGETS))
        )
    return f"Unknown or incomplete action '{act}'."


def dataset_builder_json(action: str = "progress", **kwargs: Any) -> dict[str, Any]:
    text = dataset_builder_report(action, **kwargs)
    b = get_builder(kwargs.get("work_dir"))
    return {
        "ok": True,
        "action": action,
        "report": text,
        "documents": len(b.documents),
        "tokens": b.total_tokens(),
        "words": b.total_words(),
        "chars": b.total_chars(),
        "categories": b.category_status(),
        "needs_review": sum(1 for d in b.documents.values() if d.needs_review),
        "imbalance": b.imbalance_report(),
    }
