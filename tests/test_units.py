import pytest

from app.evals import parse_judge, summarize, token_f1
from app.ingest import chunk_pages, split_text
from app.llm import FallbackLLM, LLMError
from app.rag import parse_citations


def test_split_text_respects_size_and_overlaps():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = split_text(text, size=200, overlap=50)
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]  # overlap carried forward


def test_chunks_keep_page_numbers():
    chunks = chunk_pages(["page one text", "", "page three text"], document_id=1, filename="a.pdf", size=500, overlap=50)
    assert [c.page for c in chunks] == [1, 3]


def test_parse_citations_ignores_out_of_range():
    assert parse_citations("A [1], B [3][2], fake [9]", n_sources=3) == [1, 2, 3]


def test_token_f1():
    assert token_f1("The refund period is 30 days [1]", "30 days") == pytest.approx(0.5714, abs=1e-3)
    assert token_f1("nothing related", "30 days") == 0.0


@pytest.mark.parametrize("raw,score", [('{"score": 4, "reason": "ok"}', 4), ('Sure! {"score":2}', 2), ("Score: 3", 3), ("??", None)])
def test_parse_judge(raw, score):
    assert parse_judge(raw)[0] == score


def test_summarize_handles_errors():
    s = summarize([{"question": "q", "error": "boom"}])
    assert s["errors"] == 1 and s["retrieval_hit_rate"] is None


class Broken:
    name = "broken:model"

    def complete(self, *a, **k):
        raise LLMError("429 rate limit")


class Works:
    name = "backup:model"

    def complete(self, *a, **k):
        return "ok"


def test_fallback_used_when_primary_fails():
    assert FallbackLLM(Broken(), Works()).complete_traced("s", "u") == ("ok", "backup:model")
    with pytest.raises(LLMError):
        FallbackLLM(Broken(), None).complete("s", "u")
