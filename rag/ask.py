#!/usr/bin/env python3
"""RAG maintenance copilot: TF-IDF retrieval over the sample maintenance
guides in rag/docs, returning the best-matching passages plus the
recommended actions from the top guide. Retrieval-only by design: no LLM
call is made here, so answers are always grounded in the retrieved text.
Swap in an LLM generation step behind the same ask() interface later.

Run from the project root:
  python3 rag/ask.py "engine shows rising EGT and falling compressor efficiency"
"""
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "rag" / "docs"


def load_docs():
    docs = []
    for path in sorted(DOCS.glob("*.md")):
        text = path.read_text()
        title = text.splitlines()[0].lstrip("# ").strip()
        actions = ""
        m = re.search(r"## Recommended actions\n(.*?)(?=\n## |\Z)", text, re.S)
        if m:
            actions = m.group(1).strip()
        docs.append({"file": path.name, "title": title, "text": text,
                     "actions": actions})
    return docs


def ask(query: str, top_k: int = 2):
    docs = load_docs()
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vec.fit_transform([d["text"] for d in docs])
    sims = cosine_similarity(vec.transform([query]), matrix)[0]
    order = sims.argsort()[::-1][:top_k]
    matches = []
    for i in order:
        d = docs[int(i)]
        passage = d["text"][:600].replace("\n", " ")
        matches.append({"title": d["title"], "score": round(float(sims[i]), 3),
                        "passage": passage, "actions": d["actions"]})
    return {"query": query, "matches": matches}


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python3 rag/ask.py \"<question>\"")
        raise SystemExit(1)
    result = ask(" ".join(sys.argv[1:]))
    print(f"Q: {result['query']}\n")
    for m in result["matches"]:
        print(f"[{m['score']}] {m['title']}")
        print(f"  {m['passage'][:400]}")
        print(f"  Recommended actions:\n  {m['actions']}\n")


if __name__ == "__main__":
    main()
