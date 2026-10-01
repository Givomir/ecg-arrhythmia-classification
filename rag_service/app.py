"""
RAG microservice - the guideline index behind a small REST API
================================================================
Optional: the desktop app has the same RAG engine built in (ONNX). The
service is for sharing one index between several clients (web app,
rag_recommend.py with RAG_SERVICE_URL, other machines). It embeds documents
and queries live with sentence-transformers (PyTorch).

Endpoints:
    GET    /health               status, model, number of documents/chunks
    GET    /documents            indexed documents with their chunk counts
    POST   /documents            upload a .pdf/.txt -> extract, chunk, embed, index
                                 (409 if a document with that name exists,
                                 unless ?replace=true)
    DELETE /documents/{source}   remove a document from the index
    POST   /search               {"query": ..., "top_k": 4} -> best passages
    POST   /embed                {"texts": [...]} -> normalized embeddings

The index logic (atomic saves, reloading after external rewrites, document
names) is in rag_store.py, shared with the desktop app.

Run (from the project root, needs rag_service/requirements.txt):
    uvicorn rag_service.app:app --port 8001
"""
import os
import sys
import logging
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# Shared code lives one folder up (the project root)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from build_rag_index import source_name  # noqa: E402
from rag_store import IndexStore, DocumentExists, ALLOWED  # noqa: E402

ROOT = os.path.dirname(HERE)
INDEX_DIR = os.environ.get('RAG_INDEX_DIR', os.path.join(ROOT, 'rag_index'))
DOCS_DIR = os.environ.get('RAG_DOCS_DIR', os.path.join(ROOT, 'guidelines'))
DEFAULT_MODEL = os.environ.get('RAG_MODEL', 'all-MiniLM-L6-v2')


def sentence_transformer(model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    class Encoder:
        def encode(self, texts):
            return model.encode(texts, batch_size=64, convert_to_numpy=True,
                                normalize_embeddings=True)
    return Encoder()


app = FastAPI(title="ECG RAG service",
              description="Guideline index for the ECG analyzer. Educational prototype.")
store = None


@app.on_event("startup")
def _load():
    global store
    store = IndexStore(INDEX_DIR, DOCS_DIR, sentence_transformer, DEFAULT_MODEL,
                       chunk_size=int(os.environ.get('RAG_CHUNK_SIZE', 250)),
                       overlap=int(os.environ.get('RAG_OVERLAP', 50)),
                       logger=logging.getLogger('uvicorn.error'))


class SearchRequest(BaseModel):
    query: str
    top_k: int = 4


class EmbedRequest(BaseModel):
    texts: List[str]


@app.get("/health")
def health():
    return {'status': 'ok', **store.stats()}


@app.get("/documents")
def documents():
    return [{'source': s, 'n_chunks': n} for s, n in sorted(store.documents().items())]


@app.post("/documents")
async def upload(file: UploadFile = File(...), replace: bool = False):
    name = source_name(file.filename or '')
    if not name.lower().endswith(ALLOWED):
        raise HTTPException(400, f"Only {', '.join(ALLOWED)} files are supported")
    # write to a temp file first: an existing original stays intact if
    # indexing fails or the name is already taken
    tmp = os.path.join(DOCS_DIR, '.upload-' + name)
    with open(tmp, 'wb') as f:
        f.write(await file.read())
    try:
        return store.add_file(tmp, name=name, replace=replace, move=True)
    except DocumentExists:
        raise HTTPException(409, f"A document named {name!r} is already in the index. "
                                 "Upload it with replace=true to replace it.")
    except Exception as e:
        raise HTTPException(422, f"Could not index {name}: {e}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.delete("/documents/{source:path}")
def delete(source: str):
    removed = store.remove(source)
    if not removed:
        raise HTTPException(404, f"No document named {source!r} in the index")
    return {'source': source, 'removed_chunks': removed}


@app.post("/search")
def search(req: SearchRequest):
    return store.search(req.query, max(1, min(req.top_k, 20)))


@app.post("/embed")
def embed(req: EmbedRequest):
    return {'model': store.model_name, 'embeddings': store.encode(req.texts).tolist()}
