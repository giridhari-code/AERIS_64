"""Dataset Builder — collect, clean, quality, categories, balance, export."""
from __future__ import annotations
import csv, hashlib, json, os, re, time, urllib.request
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional

DEFAULT_CATEGORY_TARGETS = {
    "general_language": 300_000, "science_factual": 180_000, "mathematics": 180_000,
    "reasoning_logic": 200_000, "hinglish": 140_000,
}
CATEGORY_ALIASES = {"general": "general_language", "science": "science_factual", "math": "mathematics", "reasoning": "reasoning_logic", "hinglish": "hinglish"}

def normalize_category(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return CATEGORY_ALIASES.get(key, key or "general_language")

@dataclass
class DocumentRecord:
    doc_id: str; text: str; source: str = ""; collected_at: float = 0.0
    license_note: str = "unknown"; category: str = "general_language"
    quality: str = "medium"; quality_score: float = 0.5
    quality_flags: list = field(default_factory=list); needs_review: bool = False
    char_count: int = 0; word_count: int = 0; token_count: int = 0
    meta: dict = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass
class BuilderReport:
    action: str; ok: bool; message: str; documents: int = 0; tokens: int = 0
    details: dict = field(default_factory=dict)
    def to_text(self) -> str:
        lines = [f"DatasetBuilder [{self.action}]: {'ok' if self.ok else 'error'}", self.message]
        if self.documents: lines.append(f"documents={self.documents}")
        if self.tokens: lines.append(f"tokens≈{self.tokens}")
        if self.details: lines.append(json.dumps(self.details, ensure_ascii=False, indent=2)[:4000])
        return "\n".join(lines)

class _H(HTMLParser):
    def __init__(self):
        super().__init__(); self.c=[]; self.s=0
    def handle_starttag(self, t, a):
        if t in {"script","style","nav","footer","header","aside","noscript"}: self.s+=1
    def handle_endtag(self, t):
        if t in {"script","style","nav","footer","header","aside","noscript"}: self.s=max(0,self.s-1)
    def handle_data(self, d):
        if self.s==0 and d and d.strip(): self.c.append(d)
    def text(self): return "\n".join(self.c)

def html_to_text(raw: str) -> str:
    p=_H()
    try: p.feed(raw or ""); p.close()
    except Exception: return re.sub(r"<[^>]+>"," ", raw or "")
    return p.text()

def approximate_tokens(text: str) -> int:
    t=text or ""
    if not t: return 0
    if len(re.findall(r"[\u0900-\u097F]", t)) > max(20, len(t)//10): return max(1, int(len(t)/3.2))
    return max(1, int(max(len(t.split())*1.3, len(t)/4)))

def count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        try:
            if hasattr(tokenizer,"encode"):
                ids=tokenizer.encode(text); return len(ids) if ids is not None else approximate_tokens(text)
        except Exception: pass
    try:
        from sentinel.tokenizer.char import CharTokenizer
        tok=CharTokenizer()
        if hasattr(tok,"encode"): return len(tok.encode(text) or [])
    except Exception: pass
    return approximate_tokens(text)

_BOILER=re.compile(r"(?i)\b(subscribe|cookie policy|all rights reserved|click here|sign up|newsletter|advertisement)\b")

def clean_text(text: str) -> str:
    t=text or ""
    if "<html" in t.lower() or "<body" in t.lower(): t=html_to_text(t)
    lines=[]
    for line in t.splitlines():
        s=line.strip()
        if not s: lines.append(""); continue
        if _BOILER.search(s) and len(s)<120: continue
        lines.append(s)
    t=re.sub(r"\n{3,}","\n\n","\n".join(lines)); return re.sub(r"[ \t]{2,}"," ",t).strip()

def content_fingerprint(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+"," ",(text or "").lower()).strip().encode()).hexdigest()[:32]

def score_quality(text: str):
    flags=[]; t=(text or "").strip(); words=t.split()
    if len(t)<80: flags.append("too_short")
    if len(words)>=20 and len(set(w.lower() for w in words))/max(1,len(words))<0.35: flags.append("repetitive")
    if t.count("\ufffd")>3: flags.append("encoding_noise")
    letters=len(re.findall(r"[A-Za-z\u0900-\u097F]",t))
    if t and letters/max(1,len(t))<0.25: flags.append("low_alpha_ratio")
    score=0.75
    for f,d in [("too_short",0.35),("repetitive",0.25),("encoding_noise",0.3),("low_alpha_ratio",0.2)]:
        if f in flags: score-=d
    if len(words)>120 and "repetitive" not in flags: score+=0.1
    score=max(0.0,min(1.0,score))
    label="high" if score>=0.7 and not flags else ("medium" if score>=0.4 else "low")
    return label, score, flags, label=="low" or score<0.55 or "encoding_noise" in flags

def license_review_flag(license_note: str, source: str) -> bool:
    if not (license_note or "").strip() or (license_note or "").lower() in {"unknown","unclear"}: return True
    return bool(re.search(r"(?i)\b(all rights reserved|copyright|proprietary|personal use only|cc by-nc)\b", f"{license_note} {source}"))

def _read_local(path: Path) -> str:
    raw=path.read_bytes(); suf=path.suffix.lower()
    if suf in {".html",".htm"}: return html_to_text(raw.decode("utf-8","replace"))
    if suf==".jsonl":
        out=[]
        for line in raw.decode("utf-8","replace").splitlines():
            if not line.strip(): continue
            try:
                o=json.loads(line); out.append(str(o.get("text") or o.get("content") or o) if isinstance(o,dict) else str(o))
            except Exception: out.append(line)
        return "\n\n".join(out)
    return raw.decode("utf-8","replace")

def _fetch(url: str, timeout=20.0, max_bytes=2_000_000) -> str:
    req=urllib.request.Request(url, headers={"User-Agent":"AERIS64-DatasetBuilder/1.0"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        data=resp.read(max_bytes+1)[:max_bytes]
        text=data.decode(resp.headers.get_content_charset() or "utf-8", "replace")
        if "html" in (resp.headers.get("Content-Type") or "").lower() or "<html" in text[:200].lower(): return html_to_text(text)
        return text

def _doc_id(source, text): return "doc_"+hashlib.sha256(f"{source}\n{content_fingerprint(text)}".encode()).hexdigest()[:16]

class DatasetBuilder:
    def __init__(self, work_dir=None, category_targets=None, tokenizer=None):
        env=os.environ.get("NEUROFIELD_DATASET_WORK_DIR") or os.environ.get("AERIS_DATASET_WORK_DIR")
        self.work_dir=Path(work_dir or env or "artifacts/dataset_builder"); self.work_dir.mkdir(parents=True, exist_ok=True)
        self.category_targets=dict(category_targets or DEFAULT_CATEGORY_TARGETS)
        self.tokenizer=tokenizer; self.documents={}; self._fps=set(); self._load()
    def _sp(self): return self.work_dir/"state.jsonl"
    def _load(self):
        if not self._sp().is_file(): return
        try:
            for line in self._sp().read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                o=json.loads(line)
                rec=DocumentRecord(**{k:o[k] for k in DocumentRecord.__dataclass_fields__ if k in o})
                self.documents[rec.doc_id]=rec; self._fps.add(content_fingerprint(rec.text))
        except Exception: pass
    def _save(self, rec):
        try:
            with self._sp().open("a",encoding="utf-8") as f: f.write(json.dumps(rec.to_dict(), ensure_ascii=False)+"\n")
        except Exception: pass
    def add_document(self, text, *, source="", category="general_language", license_note="unknown", clean=True):
        body=clean_text(text) if clean else (text or "")
        if len(body.strip())<40: return None
        fp=content_fingerprint(body)
        if fp in self._fps: return None
        ql,qs,fl,rev=score_quality(body); lic=license_review_flag(license_note, source)
        rec=DocumentRecord(_doc_id(source,body), body, source, time.time(), license_note or "unknown",
            normalize_category(category), ql, qs, fl, rev or lic, len(body), len(body.split()), count_tokens(body,self.tokenizer), {"license_flagged": lic})
        self.documents[rec.doc_id]=rec; self._fps.add(fp); self._save(rec); return rec
    def collect_paths(self, paths, *, category="general_language", license_note="unknown"):
        a=s=0; err=[]
        for p in paths:
            path=Path(p)
            try:
                if not path.is_file(): err.append(f"missing:{path}"); continue
                rec=self.add_document(_read_local(path), source=str(path.resolve()), category=category, license_note=license_note)
                a+=1 if rec else 0; s+=0 if rec else 1
            except Exception as e: err.append(f"{path}:{type(e).__name__}"); s+=1
        return BuilderReport("collect_paths", True, f"added={a} skipped={s}", a, self.total_tokens(), {"errors":err[:20]})
    def collect_urls(self, urls, *, category="general_language", license_note="public_web_review", timeout=20.0):
        a=s=0; err=[]
        for url in urls:
            u=(url or "").strip()
            if not u.startswith(("http://","https://")): err.append(f"bad_url:{u[:80]}"); s+=1; continue
            try:
                rec=self.add_document(_fetch(u,timeout=timeout), source=u, category=category, license_note=license_note)
                a+=1 if rec else 0; s+=0 if rec else 1
            except Exception as e: err.append(f"{u[:60]}:{type(e).__name__}"); s+=1
        return BuilderReport("collect_urls", True, f"added={a} skipped={s}", a, self.total_tokens(), {"errors":err[:20]})
    def total_tokens(self, *, category=None):
        cat=normalize_category(category) if category else None
        return sum(d.token_count for d in self.documents.values() if not cat or d.category==cat)
    def category_status(self):
        out=[]
        for name,target in self.category_targets.items():
            docs=[d for d in self.documents.values() if d.category==name]
            toks=sum(d.token_count for d in docs)
            out.append({"name":name,"target_tokens":target,"current_tokens":toks,"doc_count":len(docs),
                        "remaining":max(0,target-toks),"progress_pct":min(100.0,100.0*toks/target) if target else 100.0})
        return out
    def progress_report(self):
        total=self.total_tokens(); ts=sum(self.category_targets.values())
        return BuilderReport("progress", True, f"total_tokens≈{total} / target={ts}", len(self.documents), total,
            {"categories": self.category_status(), "needs_review": sum(1 for d in self.documents.values() if d.needs_review)})
    def balance(self, *, drop_low_quality_first=True):
        removed=0
        for name,target in self.category_targets.items():
            docs=[d for d in self.documents.values() if d.category==name]
            cur=sum(d.token_count for d in docs)
            if cur<=target: continue
            for d in sorted(docs, key=lambda d:(0 if (drop_low_quality_first and d.quality=="low") else 1, 0 if d.needs_review else 1, -d.token_count)):
                if cur<=target: break
                cur-=d.token_count; self.documents.pop(d.doc_id,None); removed+=1
        self._fps={content_fingerprint(d.text) for d in self.documents.values()}
        return BuilderReport("balance", True, f"removed={removed}", len(self.documents), self.total_tokens(), {"categories": self.category_status()})
    def export(self, fmt="jsonl", *, out_name=None, min_quality=None):
        fmt=(fmt or "jsonl").lower().strip(); qo={"low":0,"medium":1,"high":2}; mq=qo.get((min_quality or "low").lower(),0)
        docs=[d for d in self.documents.values() if qo.get(d.quality,0)>=mq]; stamp=time.strftime("%Y%m%d_%H%M%S")
        if fmt=="txt":
            path=self.work_dir/(out_name or f"export_{stamp}.txt"); path.write_text("\n\n---\n\n".join(d.text for d in docs), encoding="utf-8")
        elif fmt=="csv":
            path=self.work_dir/(out_name or f"export_{stamp}.csv")
            with path.open("w",encoding="utf-8",newline="") as f:
                w=csv.DictWriter(f, fieldnames=["doc_id","category","quality","token_count","source","license_note","needs_review"]); w.writeheader()
                for d in docs: w.writerow({"doc_id":d.doc_id,"category":d.category,"quality":d.quality,"token_count":d.token_count,"source":d.source,"license_note":d.license_note,"needs_review":d.needs_review})
        else:
            path=self.work_dir/(out_name or f"export_{stamp}.jsonl")
            with path.open("w",encoding="utf-8") as f:
                for d in docs: f.write(json.dumps(d.to_dict(), ensure_ascii=False)+"\n")
        return BuilderReport("export", True, f"wrote {path}", len(docs), sum(d.token_count for d in docs), {"path":str(path),"format":fmt})

_SESSION=None
def get_builder(work_dir=None):
    global _SESSION
    if _SESSION is None or (work_dir and str(_SESSION.work_dir)!=str(work_dir)): _SESSION=DatasetBuilder(work_dir=work_dir)
    return _SESSION

def dataset_builder_report(action="progress", *, paths=None, urls=None, category="general_language", license_note="unknown", export_format="jsonl", work_dir=None, text=None):
    if os.environ.get("NEUROFIELD_DATASET_BUILDER","1").strip().lower() in {"0","false","no","off"}: return "DatasetBuilder disabled."
    b=get_builder(work_dir); act=(action or "progress").strip().lower()
    if act in {"progress","status","categories"}: return b.progress_report().to_text()
    if act in {"collect","collect_paths","add_files"} and paths: return b.collect_paths(paths, category=category, license_note=license_note).to_text()
    if act in {"collect_urls","urls","fetch"} and urls: return b.collect_urls(urls, category=category, license_note=license_note).to_text()
    if act in {"add_text","ingest"} and text:
        rec=b.add_document(text, source="inline", category=category, license_note=license_note)
        return "skipped" if not rec else BuilderReport("add_text", True, f"added {rec.doc_id}", 1, rec.token_count, {"quality":rec.quality}).to_text()
    if act in {"balance","trim"}: return b.balance().to_text()
    if act in {"export","write"}: return b.export(export_format).to_text()
    if act in {"help","?"}: return "Actions: progress|collect_paths|collect_urls|add_text|balance|export"
    return f"Unknown action '{act}'."

def dataset_builder_json(action="progress", **kwargs) -> dict:
    text=dataset_builder_report(action, **kwargs); b=get_builder(kwargs.get("work_dir"))
    return {"ok": True, "action": action, "report": text, "documents": len(b.documents), "tokens": b.total_tokens(),
            "categories": b.category_status(), "needs_review": sum(1 for d in b.documents.values() if d.needs_review)}
