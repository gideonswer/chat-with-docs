"""Retrieve -> prompt -> answer with [n] citations."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from app.embeddings import Embedder
from app.llm import LLM, call_llm
from app.vectorstore import ChromaStore, Hit

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.
Rules:
- Cite every claim with the source number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "I couldn't find that in the uploaded documents."
- Do not use outside knowledge. Be concise."""

NOT_FOUND = "I couldn't find that in the uploaded documents."


@dataclass
class RagResult:
    answer: str
    hits: list[Hit]
    cited_refs: list[int]
    model: str
    latency_ms: int


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [
        f"[{i}] (file: {h.chunk.filename}, page {h.chunk.page})\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
    ]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def parse_citations(answer: str, n_sources: int) -> list[int]:
    refs = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    return sorted(r for r in refs if 1 <= r <= n_sources)


def answer_question(
    question: str,
    workspace_id: int,
    *,
    llm: LLM,
    embedder: Embedder,
    store: ChromaStore,
    top_k: int,
) -> RagResult:
    start = time.perf_counter()
    query_vec = embedder.embed([question])[0]
    hits = store.search(workspace_id, query_vec, top_k)
    if not hits:
        return RagResult(NOT_FOUND, [], [], "none", int((time.perf_counter() - start) * 1000))

    answer, model = call_llm(llm, SYSTEM_PROMPT, build_prompt(question, hits))
    return RagResult(
        answer=answer,
        hits=hits,
        cited_refs=parse_citations(answer, len(hits)),
        model=model,
        latency_ms=int((time.perf_counter() - start) * 1000),
    )
