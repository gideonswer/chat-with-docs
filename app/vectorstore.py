"""ChromaDB vector store. One collection; every chunk is tagged with its workspace_id,
and every query is filtered by it, so workspaces stay isolated."""
from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings


@dataclass
class Chunk:
    id: str
    text: str
    document_id: int
    filename: str
    page: int


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, higher = more relevant


class ChromaStore:
    def __init__(self, path: str | None = None, client=None):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        if client is None:
            client = chromadb.PersistentClient(
                path=path or get_settings().chroma_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        self.col = client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})

    def add(self, workspace_id: int, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.col.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"workspace_id": workspace_id, "document_id": c.document_id, "filename": c.filename, "page": c.page}
                for c in chunks
            ],
        )

    def search(self, workspace_id: int, query_embedding: list[float], k: int) -> list[Hit]:
        res = self.col.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"workspace_id": workspace_id},
        )
        hits = []
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            chunk = Chunk(
                id=cid,
                text=text,
                document_id=int(meta["document_id"]),
                filename=str(meta["filename"]),
                page=int(meta["page"]),
            )
            hits.append(Hit(chunk=chunk, score=round(1 - float(dist), 4)))
        return hits

    def delete_document(self, document_id: int) -> None:
        self.col.delete(where={"document_id": document_id})

    def delete_workspace(self, workspace_id: int) -> None:
        self.col.delete(where={"workspace_id": workspace_id})


_store: ChromaStore | None = None


def get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store
