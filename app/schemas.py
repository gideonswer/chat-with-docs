from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


# ---------- Auth ----------
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: str


# ---------- Workspaces & documents ----------
class WorkspaceIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class WorkspaceOut(BaseModel):
    id: int
    name: str
    created_at: datetime
    document_count: int = 0


class DocumentOut(BaseModel):
    id: int
    filename: str
    num_pages: int
    num_chunks: int
    created_at: datetime


# ---------- Chat ----------
class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    ref: int  # the [n] number used in the answer
    document_id: int
    filename: str
    page: int
    text: str
    score: float


class ChatOut(BaseModel):
    answer: str
    sources: list[Source]          # everything retrieved
    cited_refs: list[int]          # which [n] the model actually cited
    model: str
    latency_ms: int


# ---------- Evals ----------
class EvalCaseIn(BaseModel):
    question: str
    expected_answer: str
    expected_filename: str | None = None
    expected_page: int | None = None


class EvalRunIn(BaseModel):
    cases: list[EvalCaseIn] = Field(min_length=1, max_length=50)
    use_llm_judge: bool = True
    notes: str = ""


class EvalRunOut(BaseModel):
    id: int
    model: str
    summary: dict
    results: list
    notes: str
    created_at: datetime
