from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.auth import get_owned_workspace
from app.config import get_settings
from app.db import get_db
from app.embeddings import Embedder, get_embedder
from app.ingest import IngestError, chunk_pages, extract_pages
from app.models import Document, Workspace
from app.schemas import DocumentOut
from app.vectorstore import ChromaStore, get_store

router = APIRouter(prefix="/api/workspaces/{workspace_id}/documents", tags=["documents"])

EMBED_BATCH = 64


def to_out(d: Document) -> DocumentOut:
    return DocumentOut(id=d.id, filename=d.filename, num_pages=d.num_pages, num_chunks=d.num_chunks,
                       created_at=d.created_at)


@router.get("", response_model=list[DocumentOut])
def list_documents(ws: Workspace = Depends(get_owned_workspace)):
    return [to_out(d) for d in ws.documents]


@router.post("", response_model=DocumentOut, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    ws: Workspace = Depends(get_owned_workspace),
    db: Session = Depends(get_db),
    embedder: Embedder = Depends(get_embedder),
    store: ChromaStore = Depends(get_store),
):
    s = get_settings()
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    data = file.file.read(s.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File is larger than {s.max_upload_mb} MB")

    try:
        pages = extract_pages(data)
    except IngestError as e:
        raise HTTPException(status_code=422, detail=str(e))

    doc = Document(workspace_id=ws.id, filename=filename, num_pages=len(pages))
    db.add(doc)
    db.flush()  # get doc.id without committing yet

    chunks = chunk_pages(pages, doc.id, filename, s.chunk_size, s.chunk_overlap)
    try:
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            store.add(ws.id, batch, embedder.embed([c.text for c in batch]))
    except Exception:
        db.rollback()
        store.delete_document(doc.id)
        raise

    folder = Path(s.upload_dir) / str(ws.id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{doc.id}_{filename}").write_bytes(data)

    doc.num_chunks = len(chunks)
    db.commit()
    return to_out(doc)


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: int,
    ws: Workspace = Depends(get_owned_workspace),
    db: Session = Depends(get_db),
    store: ChromaStore = Depends(get_store),
):
    doc = db.get(Document, document_id)
    if doc is None or doc.workspace_id != ws.id:
        raise HTTPException(status_code=404, detail="Document not found")
    store.delete_document(doc.id)
    (Path(get_settings().upload_dir) / str(ws.id) / f"{doc.id}_{doc.filename}").unlink(missing_ok=True)
    db.delete(doc)
    db.commit()
