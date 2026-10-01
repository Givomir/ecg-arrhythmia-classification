# ECG Arrhythmia Classification with Clinical Recommendations

[![CI](https://github.com/Givomir/ecg-arrhythmia-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/Givomir/ecg-arrhythmia-classification/actions/workflows/ci.yml)

Multi-class ECG heartbeat classification on the **MIT-BIH Arrhythmia Database**,
with model comparison, an explainable visualizer, a REST API, and a
**RAG system** that connects classifier output to ACC/AHA clinical guidelines
via a local LLM (Llama 3.2 through Ollama).

> ⚠ **Educational prototype — not a medical device.** Does not replace clinical
> judgment. Not for real diagnosis or treatment.

## Features

- **Multi-class classification** by AAMI EC57 groups (N, S, V, F, Q) — not binary
- **Three models compared**: Random Forest, MLP, 1D CNN, on the standard
  inter-patient **DS1/DS2 split** (de Chazal et al., 2004) — no patient appears
  in both train and test, and results are comparable with the literature
- **Patient-wise validation**: early stopping for MLP/CNN monitors 4 held-out
  DS1 records, separated *before* oversampling (no duplicated beats leak into
  validation)
- **Feature engineering**: beat morphology + RR intervals + rhythm-normalized RR
  + P-wave features (239 features per beat), plus optional long-window RR and
  rhythm-irregularity features (247 per beat) for the cascade model
- **Cascade model for S beats**: a multi-class Random Forest followed by a
  dedicated S-vs-N detector, chosen by an ablation study (see below)
- **Explainable visualization**: full ECG plot with beats colored by class,
  abnormalities starred, plus which feature groups drive the decision
- **REST API** (FastAPI) serving any of the models
- **RAG evaluation** against a gold set: routing, retrieval (vs. a baseline
  query) and faithfulness checks, including real DS2 records end-to-end
- **RAG clinical recommendations**: retrieves relevant guideline passages and
  generates an educational assessment with a local Llama model
- **Web app** (Streamlit) tying everything together
- **Desktop app**: a single `ECGAnalyzer.exe` that runs without installation,
  with the full RAG engine built in: guideline documents can be added or
  removed from the app at any time
- **RAG microservice** (optional, PyTorch): the same guideline index as a
  REST service, for sharing one index between several clients

## Results (DS2 test set, inter-patient split)

Trained on DS1 (22 records; MLP/CNN use 18 for training + 4 for validation), tested on DS2 (22 unseen patients, 49,696 beats).

| Model | Accuracy | Macro F1 | S recall | S precision | V recall | RAG routing* |
|-------|:--------:|:--------:|:--------:|:-----------:|:--------:|:------------:|
| **Cascade RF (default)** | 0.92 | **0.44** | 0.25 | **0.41** | **0.96** | **15/15** |
| Random Forest | **0.93** | 0.41 | 0.12 | 0.35 | 0.95 | 14/15 |
| MLP | 0.79 | 0.37 | 0.30 | 0.17 | 0.88 | 12/15 |
| CNN | 0.84 | 0.39 | **0.40** | – | 0.77 | 13/15 |

\* Real DS2 records run end-to-end through `evaluate_rag.py`: does the
predicted dominant class lead to the same guideline as the cardiologist
annotations? S precision for the CNN was not re-measured.

**The Cascade RF is the default model** for the API, the visualizer, the web
app and the RAG pipeline. It is the plain Random Forest's pipeline plus a
dedicated S-vs-N stage (`cascade.py`, `train_cascade.py`). How the model was
found, and why it was chosen, is described below.

On the standard DS2 split no model handles S or F well. These are the known
hard classes in inter-patient evaluation. (Earlier numbers from a sequential
80/20 record split were optimistic and are not comparable with the literature.)

## Improving S detection: tests and model choice

### Why these tests were done

The first three models all missed most supraventricular (S) beats: the Random
Forest found only 12% of them. There were two suspected causes:

1. **Class imbalance.** S is only 1.9% of the training beats (944 of 51,005).
2. **S looks like N.** An S beat starts above the ventricles (atria or AV node).
   It conducts normally, so its QRS is narrow and almost the same shape as a
   normal beat. What sets S apart is **timing**: it comes early. It also often
   has an abnormal or missing P wave. V beats, in contrast, have a wide,
   deformed QRS.

To find out which part of the pipeline limits S recall, `ablation_s.py` runs a
series of experiments. Each experiment changes one thing: the feature groups,
the class balancing, the decision threshold, or the model structure. The
experiments were done in three rounds. Each round fixed a weakness that the
previous round exposed. All tables are in `ablation_results/`.

### Round 1: features, balancing and threshold (single validation split)

*Setup:* the model was trained on 18 DS1 records. The S threshold was tuned on
the 4 validation records, and the model was tested on DS2.

- **The problem is S → N, not S → V.** Of the 1,837 DS2 S beats, 1,483 were
  called N and only 26 were called V.
- **Rhythm features carry the S signal.** S recall was 0.04 with morphology
  only, 0.09 with RR intervals added, and 0.23 with normalized RR added.
- **Adding the P-wave features lowered S recall** (0.23 → 0.18). They are
  probably noisy, because the signal is not corrected for baseline wander.
- **A tuned threshold helps more than any balancing method.** SMOTE was about
  30 times slower than the other options and brought no gain.
- *Weakness:* the tuned threshold was unstable. The same threshold of 0.28 gave
  an S recall of 0.72 in one setup and 0.37 in another. Four validation records
  (279 S beats) are too few to tune it on.

### Round 2: a robust protocol

*Setup:* the S threshold is now chosen on **out-of-fold scores from a 5-fold
GroupKFold by record over all 22 DS1 records**. Every scored beat comes from a
model that never saw that patient. The final model is trained on all of DS1 and
tested on DS2. Each experiment runs with 3 seeds and is reported as mean ± std.
Two new RR features were added:

- RR relative to the mean RR of the whole record (as in de Chazal 2004). This
  works offline only.
- RR relative to the mean of the preceding 300 beats. This also works on a
  live signal.

Results:

- The threshold choice became stable (±0.01–0.02).
- **Removing the P-wave features helped** (S F1 0.54 → 0.61).
- The long-window RR feature (real-time) was as good as the whole-record one.
  With class weights it reached an S recall of 0.82 and an S F1 of 0.58.
- Undersampling N gave the lowest precision and the highest variance. Training
  without class weights was 4 times slower and lowered V recall to about 0.90.
- *Weakness:* a per-record breakdown showed that **record 232 holds 1,382 of
  the 1,837 DS2 S beats (75%)**. Without it, the "best" model had an S precision
  of only 0.16: about 6 of every 7 S alarms were false. Most false alarms came
  from atrial fibrillation or flutter records (219, 221, 222). There the rhythm
  is irregular by nature, so normal beats look "early".

### Round 3: rhythm irregularity and a dedicated S-vs-N stage

*Setup:* the same protocol as round 2. Every DS2 metric is now also reported
**without record 232**, so that one patient cannot dominate the result.

- **Rhythm-irregularity features** were added: the variation, RMSSD and pNN50
  of the preceding RR intervals, and how unusual the current RR is relative to
  that variation. The idea is that an early beat inside an already irregular
  rhythm is not evidence of S. On the 21 patients other than 232 this raised S
  precision (0.16 → 0.18–0.23). But it also cut the overall DS2 S F1
  (0.58 → 0.45), because recall on record 232 dropped.
- **A cascade (two-stage model).** Stage 1 is the multi-class Random Forest.
  Stage 2 is a separate S-vs-N classifier, applied only to the beats that
  stage 1 calls N or S. It uses compressed morphology (PCA) plus the rhythm
  features. Three variants of stage 2 were compared:
  - *Gradient boosting:* it overfit the few S patients. It ranked S beats worse
    than stage 1 alone (DS2 average precision 0.27 vs. 0.40) and was unstable
    across seeds (threshold 0.34 ± 0.20).
  - *Random Forest* with larger leaves: this was the best stage 2.
  - *Rhythm features only, without morphology:* clearly worse (S precision
    0.09). The shape of the beat is needed even for S vs. N.

### Why record 232 behaves differently

In record 232 the S beats are **the dominant rhythm, not isolated premature
beats**:

- The record has 1,382 S beats and only 398 N beats.
- 80% of the S beats directly follow another S beat, all at an RR of about
  0.73 s. They are interrupted by pauses of 1.5–7 s, and the N beats are the
  beats that come after those pauses.
- Relative to the local *median* RR, the S beats are not early at all
  (ratio ≈ 1.0).
- Features based on the *mean* RR flag them only because the pauses inflate the
  mean. So the high S recall that the simpler models get on 232 is partly
  accidental.
- The irregularity features read this rhythm correctly as highly irregular:
  a coefficient of variation of 0.50, compared with about 0.2 in atrial
  fibrillation. That is why they suppress S there.

The overall DS2 S F1 therefore mostly measures one unusual patient. The metric
without record 232 is a better guide to how the model behaves on a typical
patient.

### Why the cascade was chosen

| DS2, tuned threshold (mean of 3 seeds) | S F1 | S precision | S F1 w/o 232 | S precision w/o 232 | false S (N→S) |
|---|:--:|:--:|:--:|:--:|:--:|
| RF, original 239 features | 0.29 | 0.31 | 0.32 | 0.21 | 1,117 |
| RF + long-window RR, no P wave | **0.58** | **0.45** | 0.27 | 0.16 | 1,779 |
| RF + long-window RR + irregularity | 0.45 | 0.40 | 0.29 | 0.18 | 1,418 |
| **Cascade RF (chosen)** | 0.29 | 0.39 | **0.41** | **0.31** | **606** |

The cascade was chosen for these reasons:

1. **It is the best on typical patients.** Without record 232 it has the highest
   S F1 (0.41) and S precision (0.31). The model with the best overall DS2 score
   owes that score to record 232 (see above). On the other 21 patients it is
   *worse* than the original features.
2. **It raises the fewest false alarms.** False S beats drop from about 1,800 to
   about 600. This matters most in atrial fibrillation, where false S alarms
   would send the RAG pipeline to the wrong guideline.
3. **It is the best end to end.** It has the highest macro F1 (0.44) and V
   recall (0.96), and it routes all 15 real DS2 records to the correct guideline
   (15/15). It is the only model that recognizes S as the dominant class in
   record 232.
4. **It works on a live signal.** All of its rhythm features use only preceding
   beats (plus the next RR, like the base features). It needs neither the whole
   record nor P-wave features.
5. **It is stable.** Across seeds, the S metrics without 232 vary by at most
   about ±0.02.

The deployed model (`train_cascade.py`, seed 42) gives on DS2: S recall 0.25,
S precision 0.41, and S F1 0.31. Without record 232 it gives S recall 0.64,
S precision 0.31, and S F1 0.415. Its stage-2 threshold, 0.38, comes from the
DS1 out-of-fold scores.

**Accepted trade-offs and caveats:**

- **More false F alarms.** 1,324 N beats are called F, mostly in record 233;
  the plain RF makes 663 such errors. F is almost never detected in
  inter-patient evaluation by any of the Random Forest variants tested.
  Because F and V share the ventricular guideline, this does not change the
  RAG routing.
- **Slightly lower accuracy:** 0.92 vs. 0.93 for the plain RF.
- **The candidates were compared on DS2.** The chosen model's DS2 numbers are
  therefore somewhat optimistic. On the DS1 out-of-fold scores, which were not
  used for this comparison, every candidate reaches an S F1 of about 0.33–0.35.
  Inter-patient S detection stays hard with only about eight S-rich training
  records.

## Project structure

```
train_arrhythmia.py   # Random Forest training
train_mlp.py          # MLP training
train_cnn.py          # 1D CNN training
train_cascade.py      # cascade (RF + S-vs-N detector) training
cascade.py            # the cascade classifier (needed to load its model.pkl)
ablation_s.py         # S-beat ablation study
utility.py            # feature extraction (single source for training + deployment)
predict.py            # unified prediction (auto-detects model type)
app.py                # FastAPI server
test_api.py           # API test client
visualize_ecg.py      # ECG visualization with explainability
build_rag_index.py    # builds the guideline vector index
rag_recommend.py      # RAG recommendation engine
evaluate_rag.py       # RAG evaluation against gold_set.json
gold_set.json         # gold cases: synthetic beat counts + real DS2 records
app_web.py            # Streamlit web app
desktop_app.py        # desktop app (pywebview window + the same Python backend)
desktop/              # desktop UI (HTML/JS), icon and the .exe build script
rag_store.py          # guideline index store (shared by the desktop app and the service)
rag_embedder.py       # ONNX sentence embeddings (no PyTorch) + export script
rag_service/          # optional RAG microservice (FastAPI + PyTorch)
```

## Setup

```bash
# Create an isolated environment (recommended)
conda create -n ecg python=3.11 -y
conda activate ecg
pip install -r requirements.txt
```

Download the MIT-BIH database (e.g. via `wfdb.dl_database('mitdb', ...)`) or
from PhysioNet, and note the folder path.

## Usage

**1. Train a model** (the cascade is the default model used by the apps)
```bash
python train_mlp.py --data_dir /path/to/mit-bih --out_dir artifacts_mlp
python train_cascade.py --data_dir /path/to/mit-bih --out_dir artifacts_cascade
```
The apps read `meta.pkl` and extract the rhythm features automatically for the
cascade model.

**1b. Reproduce the S ablation study**
```bash
python ablation_s.py --data_dir /path/to/mit-bih            # all experiments, ~30-40 min
python ablation_s.py --data_dir ... --only cascade_rf --seeds 0
```

**2. Visualize with explainability**
```bash
python visualize_ecg.py --data_dir /path/to/mit-bih --record 208   # default: artifacts_cascade
```

**3. Serve via API**
```bash
uvicorn app:app --reload                             # default: artifacts_cascade
ARTIFACT_DIR=artifacts_mlp uvicorn app:app --reload   # any other model
python test_api.py --data_dir /path/to/mit-bih --record 200 --beat 10
```

**4. Build the RAG index** (needs ACC/AHA guideline PDFs in `guidelines/`)
```bash
python build_rag_index.py --guidelines_dir guidelines --out rag_index
```

**4b. RAG microservice** (optional)

The desktop app does not need it: it has the same RAG engine built in (step
6). The service is useful when several clients should share one guideline
index, for example the web app, `rag_recommend.py` or other computers. It
holds the guideline index and embeds documents and queries live with
sentence-transformers on PyTorch (CPU), so new guideline documents can be
added at any time without rebuilding anything.

```bash
pip install -r rag_service/requirements.txt
uvicorn rag_service.app:app --port 8001     # http://localhost:8001 (API docs: /docs)
```
`RAG_INDEX_DIR` and `RAG_DOCS_DIR` choose the index and documents folders
(default: `rag_index/` and `guidelines/` in the project).

| Endpoint | What it does |
|---|---|
| `GET /health` | status, embedding model, number of documents and chunks |
| `GET /documents` | indexed documents with their chunk counts |
| `POST /documents` | upload a .pdf/.txt: extract, chunk, embed and add it to the index; returns 409 if a document with that name is already indexed, unless `?replace=true` (the desktop app asks before replacing) |
| `DELETE /documents/{source}` | remove a document from the index |
| `POST /search` | `{"query": "...", "top_k": 4}` returns the best passages with scores |
| `POST /embed` | `{"texts": [...]}` returns normalized embeddings |

- **Shared index.** By default the service uses `rag_index/` and `guidelines/`,
  so the index is shared with `build_rag_index.py`.
- **Safe with other writers.** Every save is atomic (temp file, then rename),
  both in the service and in `build_rag_index.py`. Before each request the
  service checks whether the index files were rewritten by another process
  and reloads them. A rebuild done while the service runs is therefore picked
  up rather than overwritten, and a half-written index is never served. Local
  readers (`rag_recommend.py`, the web app) also reload a changed index
  without a restart. Two processes *writing at the same moment* are not
  coordinated, so do not rebuild the index while uploading a document.
- **Document names.** A document is identified by its full file name.
  Indexes built before this change stored only the first 60 characters; such
  an entry counts as the same document as a file whose name starts with it.
  Rebuilding the index with `build_rag_index.py` switches it to full names.
- **Who uses it.** The web app and `rag_recommend.py` use it when
  `RAG_SERVICE_URL` is set (for example `RAG_SERVICE_URL=http://localhost:8001`);
  otherwise they use the local index. The desktop app uses it only when you
  choose **RAG engine: RAG service** in its Guideline library.
- **Shared code.** The index logic is in `rag_store.py`, which the desktop
  app uses as well. Only the embedding back end differs: PyTorch in the
  service, ONNX in the desktop app.
- **Same results.** Searching through the service gives exactly the same
  passages and scores as the local index.
- **English only.** The embedding model (all-MiniLM-L6-v2) is English-only,
  and chunks with less than 85% ASCII text are dropped as PDF garbage. So
  documents in other languages, such as Bulgarian, are not indexed usefully.

In every recommendation the LLM receives **both** the beat classification
summary (beat counts per class and the dominant abnormality) **and** the
guideline passages retrieved for that abnormality (`build_prompt` in
`rag_recommend.py`). It is told to answer only from those passages and to
cite them as [Source N].

**5. Run the web app** (needs Ollama running with llama3.2)
```bash
ollama serve          # in another terminal
streamlit run app_web.py
```

**6. Desktop app (single .exe, no installation)**

`ECGAnalyzer.exe` is one file: double-click it to start. It contains the
cascade, Random Forest and MLP models and a copy of the guideline index. The
MIT-BIH records stay in a separate folder, chosen with **Change…**. The choice
is remembered in `%APPDATA%\ECGAnalyzer\settings.json`, and a folder with
records next to the .exe is found automatically. The app has the same features
as the web app: the whole-record plot with a scroll bar, the beat
distribution, the guideline passages, and the Llama recommendation.

**The RAG engine is built in.** In the **Guideline library** section you can
add guideline documents (**Add document…**, .pdf/.txt) or remove them. The app
extracts the text, splits it into passages, embeds them, and uses them in
every later recommendation; a large guideline PDF takes about 40 s. Nothing
else needs to be installed or started.

- **Where the index lives.** The app's own index is in
  `%APPDATA%\ECGAnalyzer\rag_index`. On the first run it is created from the
  bundled guidelines, and copies of added documents are kept in
  `%APPDATA%\ECGAnalyzer\guidelines`.
- **Embeddings without PyTorch.** The embedding model (all-MiniLM-L6-v2) is
  bundled as ONNX (`rag_embedder.py`) and runs with onnxruntime. Its vectors
  are identical to sentence-transformers (cosine similarity 1.000000 on 300
  index chunks; the same top-10 search results).
- **Same index logic as the service.** It uses `rag_store.py`: documents are
  identified by their full file name, the app asks before replacing a
  document with the same name, and saves are atomic.
- **Optional microservice.** With **RAG engine: RAG service** the
  app uses the microservice from step 4b instead. If the service is offline,
  the app searches with the built-in engine meanwhile.

The section **What is sent to the LLM** shows the exact prompt: the beat
classification summary and the retrieved guideline passages.

- **Size and start-up:** about 208 MB, about half of it the ONNX embedding
  model. The window opens in roughly 10-15 s, because a single-file .exe
  unpacks itself to a temporary folder at every start.
- **Ollama is optional and not bundled.** Without it everything works except
  the generated recommendation text; the retrieved guideline passages are
  still shown. When Ollama is running, its installed models are listed in
  the app.
- **The CNN is not included,** because it would need TensorFlow.
- **The .exe is not code-signed.** Windows SmartScreen may warn on the first
  start: choose *More info → Run anyway*.
- **Requires WebView2,** which is built into Windows 11 and up-to-date
  Windows 10.

Build it in a clean environment. The numpy and scikit-learn versions must
match the ones the models were saved with. The ONNX model is exported once,
in an environment that has PyTorch and sentence-transformers:
```bash
python rag_embedder.py --export desktop/onnx_model   # once, needs PyTorch
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r desktop\requirements-desktop.txt
.venv\Scripts\python desktop\build_exe.py          # -> dist\ECGAnalyzer.exe
```
To check a built .exe without opening the window (it writes a JSON report):
`ECGAnalyzer.exe --selftest <data_dir> <record> <report.json>`. It classifies
the record with every model, retrieves passages, and adds, finds and removes
a test document. Set `ECG_ANALYZER_HOME` to a test folder so that the
self-test does not touch your own index.
You can also run the desktop app from source with `python desktop_app.py`.

**7. Evaluate the RAG system**
```bash
# routing + retrieval on synthetic cases (fast, no LLM)
python evaluate_rag.py
# + real DS2 records through the whole pipeline
python evaluate_rag.py --data_dir /path/to/mit-bih --artifacts artifacts_cascade
# + faithfulness (needs ollama serve), 3 runs per case
python evaluate_rag.py --data_dir /path/to/mit-bih --mode full --runs 3
```
Detailed per-case output is written to `rag_eval_results.json`.

## Tests and CI

The data, the trained models, the guideline PDFs and the ONNX model are not in
git. So the tests build their own inputs (`tests/conftest.py`):

- a synthetic ECG record in the real MIT-BIH format, written with wfdb, with
  normal, premature supraventricular and wide ventricular beats;
- small models trained on that record, using the same code paths as the
  training scripts;
- a deterministic fake embedding model in place of sentence-transformers or
  ONNX.

| Suite | What it covers |
|---|---|
| `tests/unit/` | feature extraction and rhythm features, the cascade classifier, dominant class / queries / prompt / retrieval, chunking and atomic index files, the guideline index store (name conflicts, legacy names, external rewrites, half-written files), the data split, desktop helpers |
| `tests/integration/` (marker `integration`) | record analysis end to end, the classification REST API, the RAG service REST API, the desktop backend (analysis, built-in RAG library, the LLM prompt with classification + passages, service fallback) |
| `tests/integration/test_mitbih_data.py` (marker `data`) | real MIT-BIH records 100 and 232, downloaded from PhysioNet (about 6 MB), or taken from `MITBIH_DIR` |

```bash
py -3.11 -m venv .venv-test
.venv-test\Scripts\python -m pip install -r requirements-test.txt
.venv-test\Scripts\python -m pytest -m "not data"     # ~15 s, no network
.venv-test\Scripts\python -m pytest -m data           # real records
.venv-test\Scripts\python -m pytest --cov             # with coverage
```

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push to
`main` and on pull requests. It has two jobs:

1. **Unit + integration tests.** Checks that every script compiles, runs the
   tests with coverage, and uploads the JUnit and coverage reports.
2. **Real MIT-BIH records.** Runs the `data` tests. The downloaded records
   are cached between runs.

The Windows .exe is not built in CI, because it needs the trained models,
which are not in git.

## Notes

- Data, trained models, and guideline PDFs are **not** included in the repo
  (see `.gitignore`) — they are generated or downloaded locally.
- The MIT-BIH database and ACC/AHA guidelines are subject to their own licenses.
- A non-N class is treated as the *dominant abnormality* only if it makes up
  at least 5% of beats (`MIN_BURDEN` in `rag_recommend.py`); otherwise the
  record is handled as predominantly normal.
- Class S (supraventricular) is the hardest — a known challenge in this dataset
  due to few examples and near-normal morphology. See *Improving S detection: tests and model choice*.

## License

Educational project. The included code is provided as-is for learning purposes.
