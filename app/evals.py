"""Evaluation: how good are the answers, really?

For every test question we measure:
  retrieval_hit  - did the search find the expected page at all? (and at what rank -> MRR)
  citation_hit   - did the answer actually cite the expected page?
  cites_sources  - does the answer cite anything? (catches answers made up without sources)
  token_f1       - word-overlap with the expected answer (free, deterministic)
  judge_score    - an LLM grades correctness 1-5 against the expected answer
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter

from app.embeddings import Embedder
from app.llm import LLM, LLMError, call_llm
from app.rag import NOT_FOUND, answer_question
from app.schemas import EvalCaseIn
from app.vectorstore import ChromaStore

JUDGE_SYSTEM = """You grade answers from a document Q&A system.
Compare the CANDIDATE answer to the REFERENCE answer for the QUESTION.
Score 1-5: 5 = fully correct and complete, 4 = correct with minor omissions,
3 = partially correct, 2 = mostly wrong, 1 = wrong or no answer.
Reply with JSON only: {"score": <1-5>, "reason": "<one sentence>"}"""


def normalize(text: str) -> list[str]:
    text = re.sub(r"\[\d+\]", " ", text.lower())
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize(prediction), normalize(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return round(2 * precision * recall / (precision + recall), 4)


def parse_judge(raw: str) -> tuple[int | None, str]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            score = int(data.get("score"))
            if 1 <= score <= 5:
                return score, str(data.get("reason", ""))
        except (ValueError, TypeError):
            pass
    m = re.search(r"\b([1-5])\b", raw)
    return (int(m.group(1)), raw.strip()[:200]) if m else (None, "Could not parse judge output")


def judge(llm: LLM, question: str, reference: str, candidate: str) -> tuple[int | None, str]:
    prompt = f"QUESTION: {question}\nREFERENCE: {reference}\nCANDIDATE: {candidate}"
    raw, _ = call_llm(llm, JUDGE_SYSTEM, prompt)
    return parse_judge(raw)


def _matches(hit_filename: str, hit_page: int, case: EvalCaseIn) -> bool:
    if case.expected_page is None or hit_page != case.expected_page:
        return False
    return case.expected_filename is None or hit_filename.lower() == case.expected_filename.lower()


def run_case(case: EvalCaseIn, workspace_id: int, *, llm, embedder, store, top_k: int, use_judge: bool) -> dict:
    row: dict = {"question": case.question, "expected_answer": case.expected_answer}
    try:
        res = answer_question(case.question, workspace_id, llm=llm, embedder=embedder, store=store, top_k=top_k)
    except LLMError as e:
        return {**row, "error": str(e)}

    row.update(answer=res.answer, model=res.model, latency_ms=res.latency_ms)
    row["refused"] = res.answer.strip() == NOT_FOUND
    row["cites_sources"] = bool(res.cited_refs)
    row["token_f1"] = token_f1(res.answer, case.expected_answer)
    row["retrieved"] = [f"{h.chunk.filename} p.{h.chunk.page}" for h in res.hits]

    if case.expected_page is not None:
        rank = next((i for i, h in enumerate(res.hits, 1) if _matches(h.chunk.filename, h.chunk.page, case)), None)
        row["retrieval_hit"] = rank is not None
        row["reciprocal_rank"] = round(1 / rank, 4) if rank else 0.0
        row["citation_hit"] = any(_matches(res.hits[r - 1].chunk.filename, res.hits[r - 1].chunk.page, case)
                                  for r in res.cited_refs)

    if use_judge:
        try:
            row["judge_score"], row["judge_reason"] = judge(llm, case.question, case.expected_answer, res.answer)
        except LLMError as e:
            row["judge_score"], row["judge_reason"] = None, f"Judge failed: {e}"
    return row


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    with_page = [r for r in ok if "retrieval_hit" in r]
    scores = [r.get("judge_score") for r in ok]
    return {
        "cases": len(rows),
        "errors": len(rows) - len(ok),
        "retrieval_hit_rate": _mean([float(r["retrieval_hit"]) for r in with_page]),
        "mrr": _mean([r["reciprocal_rank"] for r in with_page]),
        "citation_hit_rate": _mean([float(r["citation_hit"]) for r in with_page]),
        "cites_sources_rate": _mean([float(r["cites_sources"]) for r in ok]),
        "refusal_rate": _mean([float(r["refused"]) for r in ok]),
        "avg_token_f1": _mean([r["token_f1"] for r in ok]),
        "avg_judge_score": _mean(scores),
        "judge_pass_rate": _mean([float(s >= 4) for s in scores if s is not None]),
        "avg_latency_ms": _mean([r["latency_ms"] for r in ok]),
    }


def run_eval(cases: list[EvalCaseIn], workspace_id: int, *, llm: LLM, embedder: Embedder,
             store: ChromaStore, top_k: int, use_judge: bool) -> tuple[dict, list[dict]]:
    rows = [
        run_case(c, workspace_id, llm=llm, embedder=embedder, store=store, top_k=top_k, use_judge=use_judge)
        for c in cases
    ]
    return summarize(rows), rows
