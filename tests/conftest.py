"""Tests run fully offline: a fake embedder and a fake LLM replace the real ones,
so no API keys, no model downloads and no cost."""
import hashlib
import io
import math
import os
import re
import tempfile

import pytest

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["CHROMA_DIR"] = f"{_tmp}/chroma"
os.environ["UPLOAD_DIR"] = f"{_tmp}/uploads"
os.environ["JWT_SECRET"] = "test-secret-that-is-at-least-32-bytes-long"

from fastapi.testclient import TestClient  # noqa: E402

from app.embeddings import get_embedder  # noqa: E402
from app.llm import get_llm  # noqa: E402
from app.main import app  # noqa: E402
from app.vectorstore import ChromaStore, get_store  # noqa: E402


class FakeEmbedder:
    """Hashed bag-of-words vectors: crude, but similar texts really do score higher."""
    dim = 512

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class FakeLLM:
    name = "fake:test-model"

    def __init__(self):
        self.calls = []

    def complete(self, system, user, temperature=0.0):
        self.calls.append(user)
        if "REFERENCE:" in user:  # judge call
            return '{"score": 5, "reason": "Matches the reference."}'
        first_source = re.search(r"\[1\] \(file: .*?\)\n(.*?)(?:\n\n\[2\]|\n\nQuestion:)", user, re.S)
        return f"{first_source.group(1)[:120]} [1]" if first_source else "I couldn't find that in the uploaded documents."


def make_pdf(pages: list[str]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture(scope="session")
def fake_llm():
    return FakeLLM()


@pytest.fixture(scope="session")
def client(fake_llm):
    store = ChromaStore(path=f"{_tmp}/chroma")
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as c:
        yield c


_counter = iter(range(10_000))


@pytest.fixture
def auth(client):
    """Registers a brand-new user and returns their auth headers."""
    r = client.post("/api/auth/register", json={"email": f"user{next(_counter)}@example.com", "password": "password123"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def sample_pdf():
    return make_pdf([
        "Acme Store Policy Handbook\nWelcome to Acme. This handbook explains our store policies.",
        "Refund policy\nCustomers can request a full refund within 30 days of purchase.\nRefunds go back to the original payment method.",
        "Shipping policy\nStandard shipping takes 5 to 7 business days.\nExpress shipping arrives in 2 days.",
    ])
