"""
RAG Evaluation Framework
=========================
Measures the quality of the RAG system against gold_set.json - turns
"generates recommendations" into measurable metrics. Uses the SAME functions as
the real system (find_dominant, build_query_for_class, retrieve,
build_prompt) - the real path is evaluated, not a copy of it.

Metrics:
  0. ROUTING (no LLM) - is the dominant class correct?
     - synthetic_cases: find_dominant(beat_counts) vs. expected_dominant
       (tests the MIN_BURDEN threshold)
     - record_cases (--data_dir): the whole pipeline (model + threshold) vs. the class
       computed from the physician annotations of the record (DS2 - unseen in training)

  1. RETRIEVAL (objective, no LLM)
     - source_precision: share of the top-k passages from the correct guideline
     - source_hit: is the correct guideline the majority in the top-k
     - term_coverage: share of the expected clinical terms in the passages
     - comparison with a BASELINE query (the generic "management and evaluation of X
       arrhythmia"), to show the real effect of the targeted queries.
       Caution: the targeted queries and expected_terms are written with the terms
       of the guidelines -> the absolute value is optimistic; the difference
       from the baseline is more informative.

  2. FAITHFULNESS (needs Ollama; --mode full), mean ± std over --runs runs
     - forbidden: forbidden terms in the answer that are NOT in the passages
     - unsupported_doses: doses/numbers with units that are not in the
       context (the most common dangerous hallucination)
     - citations: does it cite [Source N] and are the numbers valid (1..k)
     - support: share of sentences whose content words are at least 50%
       present in the context (passages + ECG findings) - a rough lexical measure

  3. RELEVANCE - requires clinician review (relevance_notes in the gold set).

Usage:
    # routing (synthetic) + retrieval - fast, no LLM:
    python evaluate_rag.py

    # + real DS2 records through the whole pipeline:
    python evaluate_rag.py --data_dir "C:\\...\\mit-bih" --artifacts artifacts

    # + faithfulness (needs ollama serve):
    python evaluate_rag.py --data_dir "..." --mode full --runs 3
"""
import os
import re
import json
import argparse
from collections import Counter

import numpy as np

from rag_recommend import (CLASS_NAMES_EN, MIN_BURDEN, find_dominant,
                           summarize_counts, build_query_for_class, retrieve,
                           build_prompt, ask_ollama)

HERE = os.path.dirname(os.path.abspath(__file__))

# Words without content - not counted in the lexical support check
STOPWORDS = set("""a an the and or of to in on for with by as at from is are was were be been
being this that these those it its which who whom what when where why how not no nor but if
than then so such can could may might must should would will shall do does did done has have
had having also any all each other some more most very into over under about between based
per there their they them we our you your he she his her i me my generally typically including
provided excerpts excerpt guideline guidelines source sources note findings suggest""".split())

DOSE_RE = re.compile(r'\b\d+(?:[.,]\d+)?\s?(?:mg|mcg|µg|ug|g|ml|mmol|meq|units?|iu|j|joules?)\b',
                     re.IGNORECASE)
# [Source 1], [Source 1: al-khatib...], [Source 1, 2] - the model uses all three forms
CITE_RE = re.compile(r'\[source\s*(\d+(?:\s*,\s*\d+)*)[^\]]*\]', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_gold(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def expectations(gold, dominant):
    return gold['class_expectations'][dominant or 'none']


def classify_source(source_name):
    """Determines which guideline a passage comes from, by file name."""
    s = source_name.lower()
    if 'khatib' in s or 'ventricular-arrhythmias' in s:
        return 'ventricular'
    if 'joglar' in s or 'atrial-fibrillation' in s:
        return 'atrial'
    return 'unknown'


def baseline_query(dominant):
    """The generic query from the first version of the system - for comparison."""
    if not dominant:
        return "normal sinus rhythm evaluation"
    return (f"management and evaluation of {CLASS_NAMES_EN[dominant]} arrhythmia "
            f"clinical recommendations")


def term_regex(term):
    """Match from the start of a word, case-insensitive."""
    return re.compile(r'\b' + re.escape(term.lower()))


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    return f"{np.mean(values):.0%} ± {np.std(values):.0%}" if len(values) > 1 \
        else f"{values[0]:.0%}"


# ---------------------------------------------------------------------------
# 0. Routing - prepares the cases (dominant class + ECG summary)
# ---------------------------------------------------------------------------
def prepare_cases(gold, data_dir=None, artifacts=None, min_burden=MIN_BURDEN):
    """
    Returns a list of cases with: predicted dominant (what the system will
    actually use), expected dominant and the ECG summary for the prompt.
    """
    cases = []
    for c in gold['synthetic_cases']:
        counts = Counter(c['beat_counts'])
        cases.append({
            'id': c['id'], 'kind': 'synthetic',
            'dominant': find_dominant(counts, min_burden),
            'expected_dominant': c['expected_dominant'],
            'summary': summarize_counts(c['id'], counts),
        })

    if data_dir:
        from rag_recommend import analyze_record
        from train_arrhythmia import DS2
        for c in gold['record_cases']:
            if c['record'] not in DS2:
                # DS1 records were seen in training -> inflated result
                print(f"  [!] Record {c['record']} is not from DS2 (the test set) - skipped")
                continue
            if not os.path.exists(os.path.join(data_dir, c['record'] + '.dat')):
                print(f"  [!] Record {c['record']} is missing - skipped")
                continue
            res = analyze_record(data_dir, c['record'], artifacts, min_burden=min_burden)
            cases.append({
                'id': c['id'], 'kind': 'record',
                'dominant': res['dominant'],
                # the "truth" comes from the physician annotations, not from the model
                'expected_dominant': find_dominant(res['true_counts'], min_burden),
                'summary': res['summary'],
                'pred_counts': dict(res['counts']),
                'true_counts': dict(res['true_counts']),
            })
    for c in cases:
        c['routing_ok'] = (c['dominant'] == c['expected_dominant'])
        # Weaker, but clinically more important: does it lead to the CORRECT guideline?
        # (e.g. F and V use the same ventricular guideline)
        got = expectations(gold, c['dominant'])['expected_source']
        want = expectations(gold, c['expected_dominant'])['expected_source']
        c['guideline_ok'] = (got == want)
    return cases


# ---------------------------------------------------------------------------
# 1. Retrieval
# ---------------------------------------------------------------------------
def retrieval_metrics(query, exp, index_dir, top_k):
    passages = retrieve(query, index_dir, top_k=top_k)
    sources = [classify_source(p['source']) for p, _ in passages]
    expected = exp['expected_source']
    if expected == 'either':
        precision, hit = 1.0, True
    else:
        precision = sources.count(expected) / len(sources)
        hit = Counter(sources).most_common(1)[0][0] == expected

    text = ' '.join(p['text'].lower() for p, _ in passages)
    terms = exp['expected_terms']
    found = [t for t in terms if term_regex(t).search(text)]
    return {
        'passages': passages,
        'source_precision': precision,
        'source_hit': hit,
        'term_coverage': len(found) / len(terms) if terms else 1.0,
        'terms_missing': [t for t in terms if t not in found],
        'avg_score': float(np.mean([s for _, s in passages])),
    }


def evaluate_retrieval(cases, gold, index_dir, top_k):
    """
    Retrieval is evaluated against the EXPECTED class (with correct routing). This
    way errors in routing (metric 0) and in retrieval are not mixed.
    """
    for c in cases:
        exp = expectations(gold, c['expected_dominant'])
        c['query'] = build_query_for_class(c['expected_dominant'])
        c['retrieval'] = retrieval_metrics(c['query'], exp, index_dir, top_k)
        c['baseline'] = retrieval_metrics(baseline_query(c['expected_dominant']),
                                          exp, index_dir, top_k)
    return cases


# ---------------------------------------------------------------------------
# 2. Faithfulness
# ---------------------------------------------------------------------------
def content_words(text):
    return [w for w in re.findall(r"[a-z][a-z0-9\-]+", text.lower())
            if len(w) > 3 and w not in STOPWORDS]


def split_sentences(answer):
    # Remove markdown markers and numbering, then split into sentences
    clean = re.sub(r'[*#>`]|^\s*\d+[.)]\s*', ' ', answer, flags=re.MULTILINE)
    parts = re.split(r'(?<=[.!?])\s+|\n+', clean)
    return [p.strip() for p in parts if len(content_words(p)) >= 4]


def faithfulness_metrics(answer, context, ecg_summary, exp, top_k):
    answer_low = answer.lower()
    context_low = context.lower()

    # Forbidden terms: in the answer but NOT in the passages
    forbidden = [t for t in exp['forbidden_terms']
                 if term_regex(t).search(answer_low) and not term_regex(t).search(context_low)]

    # Doses/numbers with units that are not in the context
    norm = lambda s: re.sub(r'\s+', '', s.lower()).replace(',', '.')
    context_norm = norm(context)
    doses = [m.group(0) for m in DOSE_RE.finditer(answer)
             if norm(m.group(0)) not in context_norm]

    # Citations
    cited = [int(n) for group in CITE_RE.findall(answer) for n in re.findall(r'\d+', group)]
    invalid_cites = sorted({n for n in cited if not 1 <= n <= top_k})

    # Lexical support of the sentences
    support_vocab = set(content_words(context + ' ' + ecg_summary))
    sentences = split_sentences(answer)
    unsupported = []
    for s in sentences:
        words = content_words(s)
        if sum(w in support_vocab for w in words) / len(words) < 0.5:
            unsupported.append(s)
    support = 1 - len(unsupported) / len(sentences) if sentences else None

    return {
        'forbidden': forbidden,
        'unsupported_doses': doses,
        'cites': bool(cited),
        'invalid_cites': invalid_cites,
        'support': support,
        'unsupported_sentences': unsupported,
        'clean': not forbidden and not doses and not invalid_cites,
        'answer_words': len(answer.split()),
    }


def evaluate_faithfulness(cases, gold, index_dir, model, top_k, runs, temperature):
    """
    Generates an answer along the REAL path of the system (predicted dominant ->
    query -> passages -> prompt) - as the user would see it.
    """
    for c in cases:
        dominant = c['dominant']
        exp = expectations(gold, dominant)
        passages = retrieve(build_query_for_class(dominant), index_dir, top_k=top_k)
        # The context is exactly what the model sees (the truncated passages)
        prompt = build_prompt(c['summary'], CLASS_NAMES_EN.get(dominant), passages)
        context = prompt.split('=== GUIDELINE EXCERPTS ===')[1].split('=== TASK ===')[0]

        c['faithfulness'] = []
        for run in range(runs):
            try:
                answer = ask_ollama(prompt, model=model, temperature=temperature, seed=run)
            except Exception as e:
                c['faithfulness'].append({'error': str(e)})
                continue
            m = faithfulness_metrics(answer, context, c['summary'], exp, top_k)
            m['answer'] = answer
            c['faithfulness'].append(m)
            print(f"  {c['id']} run {run + 1}/{runs}: "
                  f"{'CLEAN' if m['clean'] else 'FLAG'} | support "
                  f"{m['support'] if m['support'] is None else round(m['support'], 2)}")
    return cases


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(cases, mode):
    print("\n--- 0. ROUTING (dominant class) ---")
    for kind in ('synthetic', 'record'):
        sub = [c for c in cases if c['kind'] == kind]
        if not sub:
            continue
        for c in sub:
            mark = 'OK ' if c['routing_ok'] else 'MISS'
            extra = f" | pred {c['pred_counts']} | true {c['true_counts']}" \
                if kind == 'record' else ''
            print(f"  [{mark}] {c['id']}: system={c['dominant']} "
                  f"expected={c['expected_dominant']}{extra}")
        ok = sum(c['routing_ok'] for c in sub)
        g_ok = sum(c['guideline_ok'] for c in sub)
        print(f"  Routing accuracy ({kind}): {ok}/{len(sub)} = {ok / len(sub):.0%} "
              f"| correct guideline: {g_ok}/{len(sub)} = {g_ok / len(sub):.0%}")

    print("\n--- 1. RETRIEVAL (against the expected class) ---")
    uniq = len({c['query'] for c in cases})
    print(f"  Unique queries: {uniq} (retrieval depends only on the class; "
          f"{len(cases)} cases -> {uniq} distinct searches)")
    for c in cases:
        r = c['retrieval']
        print(f"  {c['id']}: precision {r['source_precision']:.0%} | "
              f"hit {'OK ' if r['source_hit'] else 'MISS'} | "
              f"terms {r['term_coverage']:.0%} | score {r['avg_score']:.3f}"
              + (f" | missing {r['terms_missing']}" if r['terms_missing'] else ''))

    def avg(key, which):
        return np.mean([float(c[which][key]) for c in cases])
    print(f"\n  {'':22s}{'targeted':>13s}{'baseline':>11s}")
    for key, label in [('source_precision', 'Source precision'),
                       ('source_hit', 'Source hit'),
                       ('term_coverage', 'Term coverage')]:
        print(f"  {label:22s}{avg(key, 'retrieval'):>13.0%}{avg(key, 'baseline'):>11.0%}")

    if mode != 'full':
        return
    print("\n--- 2. FAITHFULNESS (real path, mean ± std over runs) ---")
    per_run = {}
    for c in cases:
        for i, m in enumerate(c['faithfulness']):
            if 'error' in m:
                print(f"  [ERR] {c['id']}: {m['error']}")
                continue
            per_run.setdefault(i, []).append(m)
            if m['forbidden'] or m['unsupported_doses'] or m['invalid_cites']:
                print(f"  [FLAG] {c['id']} run {i + 1}: forbidden={m['forbidden']} "
                      f"doses={m['unsupported_doses']} invalid citations={m['invalid_cites']}")
    if not per_run:
        return

    def rate(fn):
        return [np.mean([fn(m) for m in ms]) for ms in per_run.values()]
    print(f"  Clean (no flags):        {mean_std(rate(lambda m: m['clean']))}")
    print(f"  Cites sources:           {mean_std(rate(lambda m: m['cites']))}")
    print(f"  No forbidden terms:      {mean_std(rate(lambda m: not m['forbidden']))}")
    print(f"  No unsupported doses:    {mean_std(rate(lambda m: not m['unsupported_doses']))}")
    support = [np.mean([m['support'] for m in ms if m['support'] is not None])
               for ms in per_run.values()]
    print(f"  Supported sentences:     {mean_std(support)}")


def to_jsonable(cases):
    out = []
    for c in cases:
        c = dict(c)
        for key in ('retrieval', 'baseline'):
            if key in c:
                r = dict(c[key])
                r['passages'] = [{'source': p['source'][:60], 'score': round(s, 4),
                                  'text': p['text'][:300]} for p, s in r.pop('passages')]
                c[key] = r
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default=os.path.join(HERE, 'rag_index'))
    ap.add_argument('--gold', default=os.path.join(HERE, 'gold_set.json'))
    ap.add_argument('--mode', choices=['retrieval', 'full'], default='retrieval')
    ap.add_argument('--data_dir', default=None,
                    help='MIT-BIH folder - enables record_cases (the whole pipeline)')
    ap.add_argument('--artifacts', default=os.path.join(HERE, 'artifacts_cascade'))
    ap.add_argument('--model', default='llama3.2')
    ap.add_argument('--top_k', type=int, default=4)
    ap.add_argument('--runs', type=int, default=3, help='LLM runs per case')
    ap.add_argument('--temperature', type=float, default=0.2,
                    help='the same as in the app; the seed is the run number')
    ap.add_argument('--min_burden', type=float, default=MIN_BURDEN)
    ap.add_argument('--out', default=os.path.join(HERE, 'rag_eval_results.json'))
    args = ap.parse_args()

    gold = load_gold(args.gold)
    print("=" * 68)
    print("RAG EVALUATION")
    print("=" * 68)
    print(f"Gold set: {len(gold['synthetic_cases'])} synthetic + "
          f"{len(gold['record_cases']) if args.data_dir else 0} real records")
    print(f"Verification: {gold.get('_verification_status', 'unknown')}")
    print(f"Dominant class threshold: {args.min_burden:.0%}")

    cases = prepare_cases(gold, args.data_dir, args.artifacts, args.min_burden)
    evaluate_retrieval(cases, gold, args.index, args.top_k)
    if args.mode == 'full':
        print(f"\nGenerating with {args.model} ({args.runs} runs x {len(cases)} cases)...")
        evaluate_faithfulness(cases, gold, args.index, args.model, args.top_k,
                              args.runs, args.temperature)
    report(cases, args.mode)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'cases': to_jsonable(cases)}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nDetailed results: {args.out}")
    print("=" * 68)
    print("NOTE: Metric 3 (Relevance) requires clinician review.")
    print("See relevance_notes in gold_set.json.")
    print("=" * 68)


if __name__ == '__main__':
    main()
