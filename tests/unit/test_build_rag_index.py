"""Text cleaning, chunking and index files (build_rag_index.py)."""
import json
import os
import time

import numpy as np
import pytest

from build_rag_index import (chunk_quality, chunk_text, clean_text, index_signature,
                             load_index_files, read_text_file, save_index, source_name)


def test_clean_text_collapses_whitespace_and_joins_hyphenation():
    assert clean_text("atrial   fibril-\n lation\x00 is\tcommon") == "atrial fibrillation is common"


def test_chunk_quality():
    assert chunk_quality("Plain English text, with numbers 123.") == 1.0
    assert chunk_quality("\x01\x02\x03\x04ab") < 0.85


def test_chunks_overlap_and_cover_the_text():
    words = [f"word{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), 'doc.txt', chunk_size=100, overlap=20)
    assert all(c['source'] == 'doc.txt' for c in chunks)
    first, second = chunks[0]['text'].split(), chunks[1]['text'].split()
    assert len(first) == 100
    assert first[-20:] == second[:20]                 # 20-word overlap
    assert chunks[-1]['text'].split()[-1] == 'word499'


def test_short_and_garbage_chunks_are_dropped():
    assert chunk_text("too short", 'a') == []
    garbage = " ".join(["\x01\x02\x03\x04\x05\x06\x07"] * 100)
    assert chunk_text(garbage, 'b', chunk_size=50, overlap=0) == []


def test_source_name_is_the_full_file_name():
    long = 'x' * 80 + '.pdf'
    assert source_name(os.path.join('some', 'dir', long)) == long


def test_read_text_file_plain_text_with_pdf_extension(tmp_path):
    p = tmp_path / 'not-really.pdf'
    p.write_text('plain text guideline', encoding='utf-8')
    assert read_text_file(str(p)) == 'plain text guideline'


def test_save_and_load_roundtrip(tmp_path):
    chunks = [{'text': 'a', 'source': 's'}, {'text': 'b', 'source': 's'}]
    emb = np.eye(2, dtype=np.float32)
    save_index(str(tmp_path), chunks, emb, 'model-x')
    model, c2, e2 = load_index_files(str(tmp_path))
    assert model == 'model-x' and c2 == chunks
    np.testing.assert_array_equal(e2, emb)
    assert not [f for f in os.listdir(tmp_path) if f.endswith('.tmp')]   # no leftovers


def test_inconsistent_index_is_rejected(tmp_path):
    save_index(str(tmp_path), [{'text': 'a', 'source': 's'}], np.ones((1, 2)), 'm')
    with open(tmp_path / 'chunks.json', 'w', encoding='utf-8') as f:
        json.dump([], f)                                   # half-written rewrite
    with pytest.raises(ValueError):
        load_index_files(str(tmp_path))


def test_signature_changes_when_the_index_is_rewritten(tmp_path):
    assert index_signature(str(tmp_path)) == (None, None, None)
    save_index(str(tmp_path), [{'text': 'a', 'source': 's'}], np.ones((1, 2)), 'm')
    sig1 = index_signature(str(tmp_path))
    time.sleep(0.02)
    save_index(str(tmp_path), [{'text': 'a', 'source': 's'}, {'text': 'b', 'source': 's'}],
               np.ones((2, 2)), 'm')
    assert index_signature(str(tmp_path)) != sig1
