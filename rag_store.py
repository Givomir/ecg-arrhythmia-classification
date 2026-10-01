"""
Guideline index store - shared by the desktop app and the RAG microservice
============================================================================
Holds the chunks + embeddings of the guideline documents and keeps them in
sync with the index files on disk (the same files build_rag_index.py writes).

  * documents are identified by their full file name; adding a document that
    is already indexed raises DocumentExists unless replace=True
  * every save is atomic; before every operation the store checks whether
    another process rewrote the files and reloads them, so it never serves a
    stale index and never overwrites a rebuilt one
  * the embedding model is pluggable: the desktop app uses ONNX
    (rag_embedder.OnnxEmbedder, no PyTorch), the microservice uses
    sentence-transformers - both give the same vectors
"""
import os
import shutil
import logging
import threading
from collections import Counter

import numpy as np

from build_rag_index import (read_text_file, clean_text, chunk_text, source_name,
                             index_signature, load_index_files, save_index)

LEGACY_NAME_LEN = 60   # older indexes stored only the first 60 characters
ALLOWED = ('.pdf', '.txt')


class DocumentExists(Exception):
    """A document with that name is already in the index."""


class IndexStore:
    """
    index_dir:        folder with chunks.json / embeddings.npy / config.json
    docs_dir:         where the original files of added documents are kept
    encoder_factory:  model_name -> object with encode(texts) returning
                      L2-normalized float32 vectors
    """

    def __init__(self, index_dir, docs_dir, encoder_factory, default_model='all-MiniLM-L6-v2',
                 chunk_size=250, overlap=50, logger=None):
        self.dir, self.docs_dir = index_dir, docs_dir
        self.encoder_factory = encoder_factory
        self.default_model = default_model
        self.chunk_size, self.overlap = chunk_size, overlap
        self.log = logger or logging.getLogger(__name__)
        self.lock = threading.RLock()
        self.encoder, self.model_name = None, None
        self.chunks, self.embeddings, self.sig = [], np.zeros((0, 0), np.float32), None
        os.makedirs(index_dir, exist_ok=True)
        os.makedirs(docs_dir, exist_ok=True)
        self._refresh()

    # -- keeping the in-memory index in sync with the files ------------------
    def _refresh(self):
        """Reloads the index if another process rewrote its files."""
        sig = index_signature(self.dir)
        if sig == self.sig:
            return
        if sig[-1] is None:          # no config.json yet -> empty index
            model_name, chunks, emb = self.default_model, [], np.zeros((0, 0), np.float32)
        else:
            try:
                model_name, chunks, emb = load_index_files(self.dir)
            except (ValueError, OSError) as e:
                # caught in the middle of an external rewrite - keep the old
                # index for now, the next call will try again
                self.log.warning("index not reloaded yet: %s", e)
                return
        if self.sig is not None:
            self.log.info("index changed on disk - reloaded (%d chunks)", len(chunks))
        self.chunks, self.embeddings, self.sig = chunks, emb, sig
        if model_name != self.model_name:
            # the index decides the model: query and document vectors must match
            self.encoder = self.encoder_factory(model_name)
            self.model_name = model_name

    def _save(self):
        save_index(self.dir, self.chunks, self.embeddings, self.model_name)
        self.sig = index_signature(self.dir)

    # -- queries ---------------------------------------------------------------
    def encode(self, texts):
        with self.lock:
            self._refresh()
            encoder = self.encoder
        return np.asarray(encoder.encode(list(texts)), dtype=np.float32)

    def stats(self):
        with self.lock:
            self._refresh()
            return {'model': self.model_name, 'n_chunks': len(self.chunks),
                    'n_documents': len({c['source'] for c in self.chunks})}

    def documents(self):
        with self.lock:
            self._refresh()
            return Counter(c['source'] for c in self.chunks)

    def search(self, query, top_k=4):
        q = self.encode([query])[0]
        with self.lock:
            self._refresh()
            if not self.chunks:
                return []
            scores = self.embeddings @ q
            top = np.argsort(-scores)[:top_k]
            return [{'source': self.chunks[i]['source'], 'text': self.chunks[i]['text'],
                     'score': float(scores[i])} for i in top]

    # -- documents -------------------------------------------------------------
    def _same_document(self, name):
        """Index sources that are this document: the same full name, or the
        60-character name an older index used for exactly this file name."""
        legacy = name[:LEGACY_NAME_LEN]
        names = {c['source'] for c in self.chunks}
        return {s for s in names
                if s == name or (s == legacy and len(name) > LEGACY_NAME_LEN)}

    def _remove_sources(self, sources):
        keep = [i for i, c in enumerate(self.chunks) if c['source'] not in sources]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.embeddings = self.embeddings[keep]
        return removed

    def exists(self, name):
        with self.lock:
            self._refresh()
            return bool(self._same_document(name))

    def add_text(self, name, text, replace=False):
        """Chunks, embeds and indexes a document. Raises DocumentExists if a
        document with this name is indexed and replace is False."""
        chunks = chunk_text(clean_text(text), name, self.chunk_size, self.overlap)
        if not chunks:
            raise ValueError("no usable text found in the document")
        if self.exists(name) and not replace:          # fail fast, before embedding
            raise DocumentExists(name)
        # embedding a large PDF takes a while - searches keep working meanwhile
        vectors = self.encode([c['text'] for c in chunks])
        with self.lock:
            self._refresh()          # the index may have changed in the meantime
            existing = self._same_document(name)
            if existing and not replace:
                raise DocumentExists(name)
            replaced = self._remove_sources(existing)
            self.chunks.extend(chunks)
            self.embeddings = (vectors if self.embeddings.size == 0
                               else np.vstack([self.embeddings, vectors]))
            self._save()
        return len(chunks), replaced

    def add_file(self, path, name=None, replace=False, move=False):
        """Indexes a .pdf/.txt file and keeps its original in docs_dir
        (moved or copied there only after indexing succeeded)."""
        name = source_name(name or path)
        if not name.lower().endswith(ALLOWED):
            raise ValueError(f"only {', '.join(ALLOWED)} files are supported")
        n_chunks, replaced = self.add_text(name, read_text_file(path), replace=replace)
        dest = os.path.join(self.docs_dir, name)
        if os.path.abspath(path) != os.path.abspath(dest):
            (shutil.move if move else shutil.copyfile)(path, dest)
        return {'source': name, 'n_chunks': n_chunks, 'replaced_chunks': replaced}

    def remove(self, name):
        with self.lock:
            self._refresh()
            removed = self._remove_sources(self._same_document(name))
            if removed:
                self._save()
        return removed
