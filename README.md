# ECG Arrhythmia Classification with Clinical Recommendations

Multi-class ECG heartbeat classification on the **MIT-BIH Arrhythmia Database**,
with model comparison, an explainable visualizer, a REST API, and a
**RAG system** that connects classifier output to ACC/AHA clinical guidelines
via a local LLM (Llama 3.2 through Ollama).

> ⚠ **Educational prototype — not a medical device.** Does not replace clinical
> judgment. Not for real diagnosis or treatment.

## Features

- **Multi-class classification** by AAMI EC57 groups (N, S, V, F, Q) — not binary
- **Three models compared**: Random Forest, MLP, 1D CNN, with patient-wise
  train/test split (no data leakage between patients)
- **Feature engineering**: beat morphology + RR intervals + rhythm-normalized RR
  + P-wave features (239 features per beat)
- **Explainable visualization**: full ECG plot with beats colored by class,
  abnormalities starred, plus which feature groups drive the decision
- **REST API** (FastAPI) serving any of the three models
- **RAG clinical recommendations**: retrieves relevant guideline passages and
  generates an educational assessment with a local Llama model
- **Web app** (Streamlit) tying everything together

## Results (test set, patient-wise split)

| Model | Accuracy | Macro F1 | S recall | V recall |
|-------|:--------:|:--------:|:--------:|:--------:|
| Random Forest | 0.90 | 0.42 | 0.13 | 0.87 |
| MLP | 0.89 | 0.45 | 0.32 | 0.96 |
| CNN | 0.83 | 0.39 | 0.26 | 0.80 |

MLP offers the best balance. See `model_comparison.md` for full analysis.

## Project structure

```
train_arrhythmia.py   # Random Forest training
train_mlp.py          # MLP training
train_cnn.py          # 1D CNN training
utility.py            # feature extraction for deployment
predict.py            # unified prediction (auto-detects model type)
app.py                # FastAPI server
test_api.py           # API test client
visualize_ecg.py      # ECG visualization with explainability
build_rag_index.py    # builds the guideline vector index
rag_recommend.py      # RAG recommendation engine
app_web.py            # Streamlit web app
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

**1. Train a model**
```bash
python train_mlp.py --data_dir /path/to/mit-bih --out_dir artifacts_mlp
```

**2. Visualize with explainability**
```bash
python visualize_ecg.py --data_dir /path/to/mit-bih --record 208 --artifacts artifacts_mlp
```

**3. Serve via API**
```bash
ARTIFACT_DIR=artifacts_mlp uvicorn app:app --reload
python test_api.py --data_dir /path/to/mit-bih --record 200 --beat 10
```

**4. Build the RAG index** (needs ACC/AHA guideline PDFs in `guidelines/`)
```bash
python build_rag_index.py --guidelines_dir guidelines --out rag_index
```

**5. Run the web app** (needs Ollama running with llama3.2)
```bash
ollama serve          # in another terminal
streamlit run app_web.py
```

## Notes

- Data, trained models, and guideline PDFs are **not** included in the repo
  (see `.gitignore`) — they are generated or downloaded locally.
- The MIT-BIH database and ACC/AHA guidelines are subject to their own licenses.
- Class S (supraventricular) is the hardest — a known challenge in this dataset
  due to few examples and near-normal morphology.

## License

Educational project. The included code is provided as-is for learning purposes.
