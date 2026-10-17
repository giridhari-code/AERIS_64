#!/usr/bin/env python3
"""Build ~1M-token pack: Wikipedia + Gutenberg + HF literature (no code)."""
from __future__ import annotations
import json, re, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sentinel.tools.dataset_builder import DatasetBuilder, DEFAULT_CATEGORY_TARGETS

UA = "AERIS64-DatasetBuilder/2.0 (research; educational)"
WORK = Path("artifacts/dataset_1m")
WIKI = {
    "general_language": ["English_language","Literature","Poetry","Essay","Narrative","Rhetoric","Grammar","Vocabulary","Proverb","Fable","Novel","Short_story","Oral_tradition","Folklore","Mythology","Epic_poetry","Journalism","Biography","Translation","Idiom","Metaphor","Allegory","Dialogue","Rhetoric_of_Aristotle","Letter_(message)","Diary","Autobiography"],
    "science_factual": ["Physics","Chemistry","Biology","Astronomy","Geology","Meteorology","Atom","Molecule","Photosynthesis","Cell_(biology)","DNA","Evolution","Solar_System","Planet","Climate","Water","Oxygen","Carbon","Electricity","Magnetism","Gravity","Thermodynamics","Optics","Periodic_table","Scientific_method","Ecology","Genetics","Neuroscience","Plate_tectonics","Oceanography","Botany","Zoology"],
    "mathematics": ["Mathematics","Arithmetic","Algebra","Geometry","Trigonometry","Calculus","Number","Prime_number","Fraction_(mathematics)","Percentage","Equation","Function_(mathematics)","Probability","Statistics","Set_theory","Linear_algebra","Pythagorean_theorem","Pi","Euclidean_geometry","Number_theory","Combinatorics","Graph_theory","Logic","Proof_(mathematics)","Derivative","Integral","Matrix_(mathematics)","Vector_(mathematics)","Polynomial","Ratio"],
    "reasoning_logic": ["Logic","Critical_thinking","Reasoning","Syllogism","Deductive_reasoning","Inductive_reasoning","Abductive_reasoning","Argument","Fallacy","Problem_solving","Decision-making","Causal_reasoning","Analogy","Hypothesis","Scientific_method","Thought_experiment","Paradox","Dialectic","Inference","Propositional_logic","Validity_(logic)","Soundness","Cognitive_bias","Bayesian_probability","Game_theory","Puzzle","Riddle","Strategy","Planning"],
    "hinglish": ["Hindi","Hindustani_language","India","Indian_literature","Bollywood","Hindi_literature","Urdu","Devanagari","Sanskrit","Bhakti_movement","Mahabharata","Ramayana","Kabir","Tulsidas","Premchand","Indian_philosophy","Yoga","Ayurveda","Indian_cuisine","Diwali","Holi","Ganges","Himalayas","Mumbai","Delhi","Tagore","Gandhi","Indian_English"],
}
GUTENBERG = {
    "general_language": [11, 1342, 84, 98, 76, 2701, 1661, 174, 345, 5200, 2591, 74],
    "science_factual": [2009, 1228, 2240],
    "mathematics": [17384],
    "reasoning_logic": [1232, 3207, 55],
    "hinglish": [],
}

def _get(url, timeout=25.0):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(6_000_000)

def wiki_extract(title, lang="en"):
    q = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&exlimit=1&redirects=1&format=json&titles={q}"
    raw = json.loads(_get(url).decode("utf-8", "replace"))
    pages = (raw.get("query") or {}).get("pages") or {}
    for page in pages.values():
        if int(page.get("id") or 0) < 0: continue
        text = (page.get("extract") or "").strip()
        if len(text) > 200: return text
    return ""

def gutenberg_text(book_id):
    for tpl in (
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-8.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
    ):
        try:
            data = _get(tpl, timeout=40).decode("utf-8", "replace")
            if "Project Gutenberg" in data[:2000] or len(data) > 5000:
                m = re.search(r"\*\*\*\s*START[^\n]*\n", data, re.I)
                if m: data = data[m.end():]
                m2 = re.search(r"\*\*\*\s*END", data, re.I)
                if m2: data = data[:m2.start()]
                return data.strip()
        except Exception:
            continue
    return ""

def hf_shakespeare():
    try:
        return _get("https://huggingface.co/datasets/karpathy/tiny_shakespeare/resolve/main/input.txt", timeout=30).decode("utf-8","replace")
    except Exception:
        return ""

def fill_category(b, cat, target):
    titles = WIKI.get(cat, [])
    langs = ["hi", "en"] if cat == "hinglish" else ["en"]
    def fetch_wiki(title):
        for lang in langs:
            try:
                t = wiki_extract(title, lang=lang)
                if t: return title, t
            except Exception:
                continue
        return title, ""
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(fetch_wiki, t) for t in titles]
        for fut in as_completed(futs):
            if b.total_tokens(category=cat) >= target: break
            title, text = fut.result()
            if not text: continue
            b.add_document(text, source=f"https://wikipedia.org/wiki/{urllib.parse.quote(title)}", category=cat, license_note="CC-BY-SA Wikipedia")
    for bid in GUTENBERG.get(cat, []):
        if b.total_tokens(category=cat) >= target: break
        try: book = gutenberg_text(bid)
        except Exception: continue
        if len(book) < 1000: continue
        chunk = 8000
        for i in range(0, min(len(book), 500_000), chunk):
            if b.total_tokens(category=cat) >= target: break
            b.add_document(book[i:i+chunk], source=f"https://www.gutenberg.org/ebooks/{bid}#c{i}", category=cat, license_note="public_domain_gutenberg")

def main():
    WORK.mkdir(parents=True, exist_ok=True)
    b = DatasetBuilder(work_dir=WORK, workers=8, max_docs=100_000)
    sh = hf_shakespeare()
    if sh:
        for i in range(0, len(sh), 6000):
            b.add_document(sh[i:i+6000], source="huggingface:karpathy/tiny_shakespeare", category="general_language", license_note="public_domain_literature")
    for cat, target in DEFAULT_CATEGORY_TARGETS.items():
        print(f"=== {cat} → {target}", flush=True)
        fill_category(b, cat, target)
        st = [s for s in b.category_status() if s["name"]==cat][0]
        print(f"  tokens={st['current_tokens']} docs={st['doc_count']} pct={st['progress_pct']}", flush=True)
    print(b.progress_report().to_text())
    exp = b.export("jsonl", training_ready=True, out_name="aeris64_1m_train.jsonl")
    b.export("csv", out_name="aeris64_1m_meta.csv")
    summary = {"targets": DEFAULT_CATEGORY_TARGETS, "categories": b.category_status(), "total_tokens": b.total_tokens(), "documents": len(b.documents), "export": exp.details}
    (WORK / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", WORK, "tokens", b.total_tokens())

if __name__ == "__main__":
    main()
