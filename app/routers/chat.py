from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_owned_workspace
from app.config import get_settings
from app.embeddings import Embedder, get_embedder
from app.llm import LLM, LLMError, get_llm
from app.models import Workspace
from app.rag import answer_question
from app.schemas import ChatIn, ChatOut, Source
from app.vectorstore import ChromaStore, get_store

router = APIRouter(prefix="/api/workspaces/{workspace_id}", tags=["chat"])


@router.post("/chat", response_model=ChatOut)
def chat(
    body: ChatIn,
    ws: Workspace = Depends(get_owned_workspace),
    llm: LLM = Depends(get_llm),
    embedder: Embedder = Depends(get_embedder),
    store: ChromaStore = Depends(get_store),
):
    try:
        res = answer_question(
            body.question, ws.id, llm=llm, embedder=embedder, store=store,
            top_k=body.top_k or get_settings().top_k,
        )
    except LLMError as e:
        raise HTTPException(status_code=503, detail=f"AI model unavailable: {e}")

    return ChatOut(
        answer=res.answer,
        sources=[
            Source(ref=i, document_id=h.chunk.document_id, filename=h.chunk.filename, page=h.chunk.page,
                   text=h.chunk.text, score=h.score)
            for i, h in enumerate(res.hits, start=1)
        ],
        cited_refs=res.cited_refs,
        model=res.model,
        latency_ms=res.latency_ms,
    )
