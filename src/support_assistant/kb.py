"""Knowledge base: Markdown articles split into sections and searched with BM25.

BM25 needs no embeddings API key, which keeps the demo runnable with a single
model key. In production this is a vector store with embeddings and reranking.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

WORD = re.compile(r"\w+", re.UNICODE)

# Crude stemming for Ukrainian inflection: "пароль", "пароля", "паролем" share
# a prefix. Good enough for a handful of articles, not for a real search engine.
STEM_LENGTH = 5


@dataclass(frozen=True)
class Chunk:
    source: str  # "file.md#Section title"
    text: str


def tokenize(text: str) -> list[str]:
    return [w[:STEM_LENGTH] for w in WORD.findall(text.lower())]


def split_article(path: Path) -> list[Chunk]:
    """One chunk per "## " section; the article title is prepended for context."""
    lines = path.read_text(encoding="utf-8").splitlines()
    title = next((line[2:].strip() for line in lines if line.startswith("# ")), path.stem)
    chunks, heading, body = [], None, []

    def flush():
        if heading and any(b.strip() for b in body):
            text = f"{title}. {heading}.\n" + "\n".join(body).strip()
            chunks.append(Chunk(f"{path.name}#{heading}", text))

    for line in lines:
        if line.startswith("## "):
            flush()
            heading, body = line[3:].strip(), []
        elif heading:
            body.append(line)
    flush()
    return chunks


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    @classmethod
    def from_dir(cls, directory: Path) -> "KnowledgeBase":
        chunks = [c for p in sorted(directory.glob("*.md")) for c in split_article(p)]
        if not chunks:
            raise FileNotFoundError(f"No articles found in {directory}")
        return cls(chunks)

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(scores, self.chunks, strict=True), key=lambda p: p[0], reverse=True)
        return [chunk for score, chunk in ranked[:k] if score > 0]
