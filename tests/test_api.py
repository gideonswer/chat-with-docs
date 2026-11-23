def make_workspace(client, auth, name="Test"):
    r = client.post("/api/workspaces", json={"name": name}, headers=auth)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def upload(client, auth, ws, pdf, name="policy.pdf"):
    return client.post(f"/api/workspaces/{ws}/documents", files={"file": (name, pdf, "application/pdf")}, headers=auth)


# ---------- auth ----------
def test_register_login_me(client):
    r = client.post("/api/auth/register", json={"email": "Gideon@Example.com", "password": "password123"})
    assert r.status_code == 201
    r = client.post("/api/auth/login", data={"username": "gideon@example.com", "password": "password123"})
    assert r.status_code == 200
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"})
    assert me.json()["email"] == "gideon@example.com"


def test_duplicate_email_and_bad_password(client):
    client.post("/api/auth/register", json={"email": "dup@example.com", "password": "password123"})
    assert client.post("/api/auth/register", json={"email": "dup@example.com", "password": "password123"}).status_code == 409
    assert client.post("/api/auth/login", data={"username": "dup@example.com", "password": "wrongpass1"}).status_code == 401


def test_requires_token(client):
    assert client.get("/api/workspaces").status_code == 401
    assert client.get("/api/workspaces", headers={"Authorization": "Bearer garbage"}).status_code == 401


# ---------- workspaces are private ----------
def test_users_cannot_see_each_others_workspaces(client, auth, sample_pdf):
    ws = make_workspace(client, auth)
    upload(client, auth, ws, sample_pdf)
    other = client.post("/api/auth/register", json={"email": "intruder@example.com", "password": "password123"})
    other_auth = {"Authorization": f"Bearer {other.json()['access_token']}"}
    assert client.get(f"/api/workspaces/{ws}/documents", headers=other_auth).status_code == 404
    assert client.post(f"/api/workspaces/{ws}/chat", json={"question": "refund?"}, headers=other_auth).status_code == 404


def test_search_is_scoped_to_workspace(client, auth, sample_pdf):
    ws_a, ws_b = make_workspace(client, auth, "A"), make_workspace(client, auth, "B")
    upload(client, auth, ws_a, sample_pdf)
    r = client.post(f"/api/workspaces/{ws_b}/chat", json={"question": "What is the refund policy?"}, headers=auth)
    assert r.status_code == 200
    assert r.json()["sources"] == []


# ---------- documents ----------
def test_upload_and_list(client, auth, sample_pdf):
    ws = make_workspace(client, auth)
    r = upload(client, auth, ws, sample_pdf)
    assert r.status_code == 201, r.text
    assert r.json()["num_pages"] == 3 and r.json()["num_chunks"] >= 3
    docs = client.get(f"/api/workspaces/{ws}/documents", headers=auth).json()
    assert [d["filename"] for d in docs] == ["policy.pdf"]


def test_rejects_non_pdf_and_broken_pdf(client, auth):
    ws = make_workspace(client, auth)
    assert upload(client, auth, ws, b"hello", name="notes.txt").status_code == 400
    assert upload(client, auth, ws, b"not really a pdf", name="broken.pdf").status_code == 422


def test_delete_document_removes_its_chunks(client, auth, sample_pdf):
    ws = make_workspace(client, auth)
    doc_id = upload(client, auth, ws, sample_pdf).json()["id"]
    assert client.delete(f"/api/workspaces/{ws}/documents/{doc_id}", headers=auth).status_code == 204
    r = client.post(f"/api/workspaces/{ws}/chat", json={"question": "refund"}, headers=auth)
    assert r.json()["sources"] == []


# ---------- chat ----------
def test_chat_returns_page_citations(client, auth, sample_pdf):
    ws = make_workspace(client, auth)
    upload(client, auth, ws, sample_pdf)
    r = client.post(f"/api/workspaces/{ws}/chat", json={"question": "How many days do I have to get a refund?"}, headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sources"][0]["page"] == 2          # best match is the refund page
    assert body["cited_refs"] == [1]
    assert "30 days" in body["answer"]
    assert body["model"] == "fake:test-model"


# ---------- evals ----------
def test_eval_run_scores_and_is_saved(client, auth, sample_pdf):
    ws = make_workspace(client, auth)
    upload(client, auth, ws, sample_pdf)
    cases = [
        {"question": "How many days do I have to request a refund?", "expected_answer": "within 30 days of purchase",
         "expected_filename": "policy.pdf", "expected_page": 2},
        {"question": "How long does express shipping take?", "expected_answer": "Express shipping arrives in 2 days",
         "expected_page": 3},
    ]
    r = client.post(f"/api/workspaces/{ws}/evals", json={"cases": cases, "notes": "baseline"}, headers=auth)
    assert r.status_code == 201, r.text
    s = r.json()["summary"]
    assert s["cases"] == 2 and s["errors"] == 0
    assert s["retrieval_hit_rate"] == 1.0
    assert s["citation_hit_rate"] == 1.0
    assert s["avg_judge_score"] == 5.0
    runs = client.get(f"/api/workspaces/{ws}/evals", headers=auth).json()
    assert runs[0]["notes"] == "baseline"


def test_eval_needs_documents(client, auth):
    ws = make_workspace(client, auth)
    r = client.post(f"/api/workspaces/{ws}/evals",
                    json={"cases": [{"question": "q", "expected_answer": "a"}]}, headers=auth)
    assert r.status_code == 400


def test_ui_and_health(client):
    assert "Chat with Docs" in client.get("/").text
    assert client.get("/api/health").json()["status"] == "ok"
