from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import get_owned_workspace
from app.config import get_settings
from app.db import get_db
from app.embeddings import Embedder, get_embedder
from app.evals import run_eval
from app.llm import LLM, get_llm
from app.models import EvalRun, Workspace
from app.schemas import EvalRunIn, EvalRunOut
from app.vectorstore import ChromaStore, get_store

router = APIRouter(prefix="/api/workspaces/{workspace_id}/evals", tags=["evals"])


def to_out(r: EvalRun) -> EvalRunOut:
    return EvalRunOut(id=r.id, model=r.model, summary=r.summary, results=r.results, notes=r.notes,
                      created_at=r.created_at)


@router.post("", response_model=EvalRunOut, status_code=201)
def create_eval_run(
    body: EvalRunIn,
    ws: Workspace = Depends(get_owned_workspace),
    db: Session = Depends(get_db),
    llm: LLM = Depends(get_llm),
    embedder: Embedder = Depends(get_embedder),
    store: ChromaStore = Depends(get_store),
):
    if not ws.documents:
        raise HTTPException(status_code=400, detail="Upload at least one document before running an eval")
    summary, rows = run_eval(body.cases, ws.id, llm=llm, embedder=embedder, store=store,
                             top_k=get_settings().top_k, use_judge=body.use_llm_judge)
    models = sorted({r["model"] for r in rows if r.get("model")})
    run = EvalRun(workspace_id=ws.id, model=", ".join(models) or "n/a", summary=summary, results=rows,
                  notes=body.notes)
    db.add(run)
    db.commit()
    db.refresh(run)
    return to_out(run)


@router.get("", response_model=list[EvalRunOut])
def list_eval_runs(ws: Workspace = Depends(get_owned_workspace), db: Session = Depends(get_db)):
    runs = db.scalars(select(EvalRun).where(EvalRun.workspace_id == ws.id).order_by(EvalRun.id.desc()))
    return [to_out(r) for r in runs]


@router.get("/{run_id}", response_model=EvalRunOut)
def get_eval_run(run_id: int, ws: Workspace = Depends(get_owned_workspace), db: Session = Depends(get_db)):
    run = db.get(EvalRun, run_id)
    if run is None or run.workspace_id != ws.id:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return to_out(run)
