import shutil
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth import get_current_user, get_owned_workspace
from app.config import get_settings
from app.db import get_db
from app.models import User, Workspace
from app.schemas import WorkspaceIn, WorkspaceOut
from app.vectorstore import ChromaStore, get_store

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])


def to_out(ws: Workspace) -> WorkspaceOut:
    return WorkspaceOut(id=ws.id, name=ws.name, created_at=ws.created_at, document_count=len(ws.documents))


@router.get("", response_model=list[WorkspaceOut])
def list_workspaces(user: User = Depends(get_current_user)):
    return [to_out(ws) for ws in user.workspaces]


@router.post("", response_model=WorkspaceOut, status_code=201)
def create_workspace(body: WorkspaceIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ws = Workspace(name=body.name.strip(), owner_id=user.id)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return to_out(ws)


@router.delete("/{workspace_id}", status_code=204)
def delete_workspace(
    ws: Workspace = Depends(get_owned_workspace),
    db: Session = Depends(get_db),
    store: ChromaStore = Depends(get_store),
):
    store.delete_workspace(ws.id)
    shutil.rmtree(Path(get_settings().upload_dir) / str(ws.id), ignore_errors=True)
    db.delete(ws)
    db.commit()
