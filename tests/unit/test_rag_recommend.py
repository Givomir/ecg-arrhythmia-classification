"""Dominant class, prompt building and retrieval (rag_recommend.py)."""
import json
import os
import time
from collections import Counter

import numpy as np
import pytest

import rag_recommend as R
from build_rag_index import load_index_files, save_index
from tests.conftest import FakeEncoder


# -- dominant class -----------------------------------------------------------
@pytest.mark.parametrize('counts, expected', [
    ({'N': 900, 'V': 100}, 'V'),
    ({'N': 960, 'V': 40}, None),                 # 4% < 5% burden
    ({'N': 950, 'V': 50}, 'V'),                  # exactly 5%
    ({'N': 800, 'S': 120, 'V': 80}, 'S'),
    ({'N': 1000}, None),
    ({}, None),
])
def test_find_dominant(counts, expected):
    assert R.find_dominant(Counter(counts)) == expected


def test_find_dominant_custom_threshold():
    assert R.find_dominant({'N': 98, 'S': 2}, min_burden=0.01) == 'S'


def test_summarize_counts_lists_classes_by_frequency():
    text = R.summarize_counts('232', Counter({'N': 30, 'S': 70}), fs=360)
    lines = text.splitlines()
    assert lines[0] == 'ECG record 232: 100 beats analyzed, sampling 360 Hz.'
    assert 'supraventricular ectopic (S): 70 beats (70.0%)' in lines[1]
    assert 'normal (N): 30 beats (30.0%)' in lines[2]


# -- queries and prompt ---------------------------------------------------------
def test_s_query_avoids_the_word_ventricular():
    q = R.build_query_for_class('S')
    assert 'atrial fibrillation' in q
    assert 'ventricular' not in q.replace('supraventricular', '')


@pytest.mark.parametrize('dominant', ['V', 'F', 'Q'])
def test_ventricular_family_queries(dominant):
    assert 'ventricular' in R.build_query_for_class(dominant)


def test_normal_query():
    assert 'normal' in R.build_query_for_class(None)


def test_prompt_contains_findings_and_passages():
    summary = R.summarize_counts('100', Counter({'N': 90, 'V': 10}))
    passages = [({'source': 'guide-a.pdf', 'text': 'A' * 1000}, 0.9),
                ({'source': 'guide-b.pdf', 'text': 'beta text'}, 0.8)]
    prompt = R.build_prompt(summary, 'ventricular ectopic', passages, max_chars=50)
    assert '=== ECG FINDINGS ===' in prompt and summary in prompt
    assert 'Predominant abnormality: ventricular ectopic' in prompt
    assert '[Source 1: guide-a.pdf]' in prompt and '[Source 2: guide-b.pdf]' in prompt
    assert 'A' * 51 not in prompt                # passages are truncated
    assert 'Do NOT invent' in prompt


def test_prompt_without_dominant_class():
    prompt = R.build_prompt('ECG record 1: 10 beats analyzed.', None, [])
    assert 'none (mostly normal)' in prompt


# -- retrieval -------------------------------------------------------------------
def test_local_retrieval_uses_the_query_cache(small_index):
    query = 'anticoagulation stroke prevention atrial fibrillation'
    with open(os.path.join(small_index, R.QUERY_CACHE_FILE), 'w', encoding='utf-8') as f:
        json.dump({query: FakeEncoder().encode([query])[0].tolist()}, f)
    hits = R.retrieve(query, small_index, top_k=2, service_url='')
    assert len(hits) == 2
    assert all(chunk['source'] == 'af-guideline.txt' for chunk, _ in hits)
    assert hits[0][1] >= hits[1][1]


def test_local_index_reloads_after_the_files_change(small_index):
    n1 = len(R._load_index(small_index)[1])
    model, chunks, emb = load_index_files(small_index)
    time.sleep(0.02)
    save_index(small_index, chunks[:-1], emb[:-1], model)
    assert len(R._load_index(small_index)[1]) == n1 - 1


def test_retrieval_through_the_service(monkeypatch):
    calls = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return [{'source': 'x.pdf', 'text': 'passage', 'score': 0.7}]

    def fake_post(url, json=None, timeout=None):
        calls['url'], calls['json'] = url, json
        return Resp()

    monkeypatch.setattr(R.requests, 'post', fake_post)
    hits = R.retrieve('q', 'unused', top_k=3, service_url='http://svc:8001/')
    assert calls == {'url': 'http://svc:8001/search', 'json': {'query': 'q', 'top_k': 3}}
    assert hits == [({'text': 'passage', 'source': 'x.pdf'}, 0.7)]


def test_ask_ollama_sends_the_prompt(monkeypatch):
    sent = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {'response': 'answer'}

    def fake_post(url, json=None, timeout=None):
        sent.update(json)
        return Resp()

    monkeypatch.setattr(R.requests, 'post', fake_post)
    assert R.ask_ollama('PROMPT', model='m', seed=3) == 'answer'
    assert sent['prompt'] == 'PROMPT' and sent['model'] == 'm'
    assert sent['options'] == {'temperature': 0.2, 'seed': 3} and sent['stream'] is False
