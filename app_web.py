"""
ECG Analyzer — Web Application
================================
Combines the full system into one interactive app:
  * Record and model selection
  * ECG plot with beats colored by class
  * Class distribution
  * RAG clinical recommendation (guidelines + Llama)

EDUCATIONAL PROTOTYPE — not a medical device.

Run:
    streamlit run app_web.py

It opens automatically in the browser (http://localhost:8501).
The data path is set in the sidebar.
"""
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import Counter

# Reuse the existing logic
from train_arrhythmia import load_beats
from utility import record_features, uses_rhythm_features
from rag_recommend import CLASS_NAMES_EN, find_dominant, summarize_counts

CLASS_COLORS = {'N': '#2ca02c', 'S': '#ff7f0e', 'V': '#d62728',
                'F': '#9467bd', 'Q': '#7f7f7f'}
CLASS_NAMES = {'N': 'Normal', 'S': 'Supraventricular', 'V': 'Ventricular',
               'F': 'Fusion', 'Q': 'Unknown'}

st.set_page_config(page_title="ECG Analyzer", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders (avoid reloading on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(artifacts):
    import joblib
    scaler = joblib.load(os.path.join(artifacts, 'scaler.pkl'))
    le = joblib.load(os.path.join(artifacts, 'label_encoder.pkl'))
    keras_path = os.path.join(artifacts, 'model.keras')
    if os.path.exists(keras_path):
        import tensorflow as tf
        model = tf.keras.models.load_model(keras_path)
        meta = joblib.load(os.path.join(artifacts, 'meta.pkl'))
        return ('cnn', model, scaler, le, meta['n_context'])
    else:
        model = joblib.load(os.path.join(artifacts, 'model.pkl'))
        return ('sklearn', model, scaler, le, 9)


@st.cache_data
def analyze_record(data_dir, record, artifacts, lead=0):
    """Classify all beats. Returns signal, beats, predictions."""
    signals, fs, beats = load_beats(os.path.join(data_dir, record))
    signal = signals[:, lead]
    X, kept_idx = record_features(signal, [b[0] for b in beats], fs,
                                  rhythm=uses_rhythm_features(artifacts))
    X = np.nan_to_num(X)
    kept = [beats[i] for i in kept_idx]   # (r_peak, true AAMI class)

    model_type, model, scaler, le, nc = load_model(artifacts)
    Xs = scaler.transform(X)
    if model_type == 'cnn':
        p = model.predict({'morphology': Xs[:, :-nc][..., np.newaxis],
                           'context': Xs[:, -nc:]}, verbose=0)
        pred = le.inverse_transform(np.argmax(p, axis=1))
    else:
        pred = le.inverse_transform(model.predict(Xs))

    return signal, fs, kept, pred, model_type


def plot_ecg(signal, fs, kept, pred, start_sec, seconds):
    """Draw ECG with colored beats."""
    s0 = int(start_sec * fs)
    s1 = min(int((start_sec + seconds) * fs), len(signal))
    t = np.arange(s0, s1) / fs

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, signal[s0:s1], color='#333', linewidth=0.8, zorder=1)

    shown = set()
    for k, (r_peak, true) in enumerate(kept):
        if not (s0 <= r_peak < s1):
            continue
        p = pred[k]
        color = CLASS_COLORS.get(p, '#000')
        tsec = r_peak / fs
        ax.scatter([tsec], [signal[r_peak]], color=color, s=70, zorder=3,
                   edgecolors='white', linewidths=0.7,
                   label=f"{p} ({CLASS_NAMES.get(p)})" if p not in shown else None)
        shown.add(p)
        ax.annotate(p, (tsec, signal[r_peak]), textcoords="offset points",
                    xytext=(0, 11), ha='center', fontsize=8, fontweight='bold',
                    color=color)
        if p != 'N':
            ax.annotate('*', (tsec, signal[r_peak]), textcoords="offset points",
                        xytext=(7, 3), ha='center', fontsize=14, color=color)
        if p != true:
            rect = Rectangle((tsec - 0.12, signal[r_peak] - 0.4), 0.24, 0.8,
                             fill=False, edgecolor='red', linestyle='--',
                             linewidth=1.1, zorder=2)
            ax.add_patch(rect)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (mV)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.title("🫀 ECG Analyzer with Clinical Recommendations")
st.caption("⚠ Educational prototype — not a medical device. "
           "Does not replace clinical judgment.")

# --- Sidebar: settings ---
with st.sidebar:
    st.header("Settings")
    data_dir = st.text_input(
        "MIT-BIH data folder",
        value=r"C:\stasi\SoftUni_Machine_learning\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0")
    record = st.text_input("Record number", value="200")
    artifacts = st.selectbox("Model",
                             ["artifacts_cascade", "artifacts", "artifacts_mlp", "artifacts_cnn"],
                             format_func=lambda x: {
                                 "artifacts_cascade": "Cascade RF (recommended)",
                                 "artifacts": "Random Forest",
                                 "artifacts_mlp": "MLP",
                                 "artifacts_cnn": "CNN"}.get(x, x))
    st.divider()
    start_sec = st.slider("Start (second)", 0, 1800, 0, step=5)
    seconds = st.slider("Duration (sec)", 5, 30, 10)

# --- Main view ---
if st.button("Analyze record", type="primary"):
    if not os.path.exists(os.path.join(data_dir, record + '.dat')):
        st.error(f"Record {record} not found in {data_dir}")
    else:
        with st.spinner("Classifying beats..."):
            signal, fs, kept, pred, model_type = analyze_record(
                data_dir, record, artifacts)
        st.session_state['analysis'] = (record, artifacts, kept, pred, model_type,
                                        len(signal), fs)

        # Plot
        st.subheader(f"ECG record {record} ({model_type})")
        fig = plot_ecg(signal, fs, kept, pred, start_sec, seconds)
        st.pyplot(fig)
        st.caption("Color = predicted class | * = abnormality | "
                   "red dashed = differs from annotation")

        # Distribution
        counts = Counter(pred)
        total = len(pred)
        st.subheader("Beat distribution")
        cols = st.columns(len(counts))
        for col, (cls, cnt) in zip(cols, counts.most_common()):
            col.metric(f"{cls} — {CLASS_NAMES.get(cls)}",
                       f"{cnt}", f"{100*cnt/total:.1f}%")

# --- RAG recommendation ---
if 'analysis' in st.session_state:
    st.divider()
    st.subheader("Clinical Recommendation (RAG + Llama)")
    st.caption("Retrieves relevant ACC/AHA guideline passages and generates "
               "an educational assessment with a local Llama model.")

    llama_model = st.text_input("Ollama model", value="llama3.2")

    if st.button("Generate recommendation"):
        record, artifacts, kept, pred, model_type, _, fs = st.session_state['analysis']
        counts = Counter(pred)

        # Same logic as rag_recommend.py (dominant-class threshold, targeted
        # query), so the app matches what evaluate_rag.py measures
        summary = summarize_counts(record, counts, fs)
        dominant = find_dominant(counts)
        dominant_name = CLASS_NAMES_EN.get(dominant) if dominant else None

        try:
            from rag_recommend import (retrieve, build_prompt, ask_ollama,
                                       build_query_for_class)
            query = build_query_for_class(dominant)

            with st.spinner("Searching guidelines..."):
                passages = retrieve(query, 'rag_index', top_k=4)

            with st.expander("Retrieved guideline passages"):
                for i, (p, score) in enumerate(passages):
                    st.markdown(f"**[Source {i+1}]** ({score:.3f}) "
                               f"_{p['source'][:45]}_")
                    st.text(p['text'][:400] + "...")

            with st.spinner(f"Generating with {llama_model} (may take a minute)..."):
                prompt = build_prompt(summary, dominant_name, passages)
                answer = ask_ollama(prompt, model=llama_model)

            st.markdown(answer)
            st.caption("⚠ Educational prototype — not a diagnosis.")

        except FileNotFoundError:
            st.error("RAG index missing. Run: python build_rag_index.py "
                     "--guidelines_dir guidelines --out rag_index")
        except Exception as e:
            if 'ConnectionError' in type(e).__name__ or 'timed out' in str(e).lower():
                st.error("Cannot connect to Ollama. Make sure it is running "
                         "(ollama serve) and the model is pulled.")
            else:
                st.error(f"Error: {e}")
                