"""The guideline index store shared by the desktop app and the service (rag_store.py)."""
import json
import os
import time

import numpy as np
import pytest

from build_rag_index import load_index_files, save_index
from rag_store import DocumentExists, IndexStore
from tests.conftest import AF_TEXT, MODEL_NAME, VA_TEXT, FakeEncoder


@pytest.fixture
def store(tmp_path, fake_encoder_factory):
    return IndexStore(str(tmp_path / 'index'), str(tmp_path / 'docs'), fake_encoder_factory,
                      default_model=MODEL_NAME, chunk_size=60, overlap=10)


def test_empty_store(store):
    assert store.stats() == {'model': MODEL_NAME, 'n_chunks': 0, 'n_documents': 0}
    assert store.search('anything') == []


def test_add_and_search(store):
    store.add_text('af.txt', AF_TEXT)
    store.add_text('va.txt', VA_TEXT)
    assert set(store.documents()) == {'af.txt', 'va.txt'}
    assert store.search('implantable cardioverter defibrillator ICD', 1)[0]['source'] == 'va.txt'
    assert store.search('anticoagulation CHA2DS2-VASc stroke', 1)[0]['source'] == 'af.txt'


def test_same_name_is_refused_unless_replace(store):
    n, _ = store.add_text('doc.txt', AF_TEXT)
    with pytest.raises(DocumentExists):
        store.add_text('doc.txt', VA_TEXT)
    assert store.search('ICD defibrillator', 1)[0]['text'] in AF_TEXT     # unchanged
    n2, replaced = store.add_text('doc.txt', VA_TEXT, replace=True)
    assert replaced == n
    assert store.documents()['doc.txt'] == n2


def test_names_with_the_same_60_character_prefix_are_different_documents(store):
    prefix = 'x' * 60
    store.add_text(prefix + '-part-A.txt', AF_TEXT)
    store.add_text(prefix + '-part-B.txt', VA_TEXT)
    assert len(store.documents()) == 2


def test_legacy_60_character_source_counts_as_the_same_document(store, tmp_path):
    full = 'joglar-et-al-2023-2023-acc-aha-accp-hrs-guideline-for-the-diagnosis.pdf'
    legacy = full[:60]
    store.add_text(legacy, AF_TEXT)                # as an old index stored it
    with pytest.raises(DocumentExists):
        store.add_text(full, VA_TEXT)
    _, replaced = store.add_text(full, VA_TEXT, replace=True)
    assert replaced > 0 and set(store.documents()) == {full}


def test_remove(store):
    store.add_text('af.txt', AF_TEXT)
    assert store.remove('af.txt') > 0
    assert store.remove('af.txt') == 0
    assert store.stats()['n_chunks'] == 0


def test_add_file_keeps_a_copy_of_the_original(store, guideline_files):
    af, _ = guideline_files
    result = store.add_file(af)
    assert result['source'] == 'af-guideline.txt' and result['n_chunks'] > 0
    assert os.path.exists(os.path.join(store.docs_dir, 'af-guideline.txt'))
    assert os.path.exists(af)                     # copied, not moved


def test_add_file_rejects_unsupported_types(store, tmp_path):
    p = tmp_path / 'notes.docx'
    p.write_text('x', encoding='utf-8')
    with pytest.raises(ValueError):
        store.add_file(str(p))


def test_empty_document_is_rejected(store):
    with pytest.raises(ValueError):
        store.add_text('empty.txt', '   ')


def test_external_rewrite_is_reloaded_and_not_overwritten(store):
    store.add_text('af.txt', AF_TEXT)
    store.add_text('va.txt', VA_TEXT)
    model, chunks, emb = load_index_files(store.dir)
    keep = [i for i, c in enumerate(chunks) if c['source'] != 'va.txt']
    time.sleep(0.02)
    save_index(store.dir, [chunks[i] for i in keep], emb[keep], model)   # e.g. a rebuild
    assert set(store.documents()) == {'af.txt'}                          # reloaded
    store.add_text('new.txt', VA_TEXT)
    _, after, _ = load_index_files(store.dir)
    assert {c['source'] for c in after} == {'af.txt', 'new.txt'}         # rebuild kept


def test_half_written_index_keeps_the_last_good_version(store):
    store.add_text('af.txt', AF_TEXT)
    n = store.stats()['n_chunks']
    with open(os.path.join(store.dir, 'chunks.json'), 'w', encoding='utf-8') as f:
        json.dump([], f)
    assert store.stats()['n_chunks'] == n
    assert store.search('atrial fibrillation', 1)


def test_index_decides_the_embedding_model(tmp_path):
    used = []

    def factory(name):
        used.append(name)
        return FakeEncoder(name)

    save_index(str(tmp_path / 'i'), [{'text': 'a b c', 'source': 's'}],
               FakeEncoder().encode(['a b c']), 'model-from-index')
    IndexStore(str(tmp_path / 'i'), str(tmp_path / 'd'), factory, default_model='other')
    assert used == ['model-from-index']
