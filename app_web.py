"""
ECG Analyzer — Web Application
================================
Combines the full system into one interactive app:
  * Record and model selection
  * Interactive plot of the whole ECG (scroll bar), beats colored by class
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
import plotly.graph_objects as go
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


def overview_envelope(signal, fs, bucket_sec=0.25):
    """
    Min/max envelope of the whole signal - a light trace (a few thousand
    points instead of ~650k) drawn in the scroll bar under the plot.
    """
    b = max(1, int(bucket_sec * fs))
    n = len(signal) // b * b
    blocks = signal[:n].reshape(-1, b)
    t = (np.arange(len(blocks)) * b + b / 2) / fs
    # alternate min and max, so the line traces the envelope
    return np.repeat(t, 2), np.column_stack([blocks.min(1), blocks.max(1)]).ravel()


def plot_ecg(signal, fs, kept, pred, window_sec):
    """
    Interactive plot of the WHOLE record with a scroll bar (range slider).
    The main view shows `window_sec` seconds; drag the window in the scroll
    bar (or pan the plot) to move through the record, drag its edges to zoom.

    Axes trick: the full-resolution ECG and the beat markers are WebGL traces
    on y2 (fast, but not drawn in the range slider). The light overview
    envelope is on the primary y axis, which is hidden far outside the main
    view, so the envelope is visible only in the scroll bar.
    """
    t = np.arange(len(signal)) / fs
    peaks = np.array([r for r, _ in kept])
    true = np.array([c for _, c in kept])
    pred = np.asarray(pred)

    fig = go.Figure()
    env_t, env_y = overview_envelope(signal, fs)
    fig.add_trace(go.Scatter(x=env_t, y=env_y, mode='lines', yaxis='y',
                             line=dict(color='#555', width=0.6),
                             hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scattergl(x=t, y=signal, mode='lines', yaxis='y2',
                               line=dict(color='#333', width=1),
                               hoverinfo='skip', showlegend=False))

    for cls, color in CLASS_COLORS.items():
        m = pred == cls
        if not m.any():
            continue
        label = cls + ('*' if cls != 'N' else '')   # * = abnormality
        fig.add_trace(go.Scattergl(
            x=peaks[m] / fs, y=signal[peaks[m]], mode='markers+text', yaxis='y2',
            text=[label] * int(m.sum()), textposition='top center',
            textfont=dict(color=color, size=11),
            marker=dict(color=color, size=9, line=dict(color='white', width=1)),
            name=f"{cls} ({CLASS_NAMES.get(cls)})", customdata=true[m],
            hovertemplate=(f"%{{x:.2f}} s<br>predicted: {cls}"
                           "<br>annotated: %{customdata}<extra></extra>")))

    wrong = pred != true
    if wrong.any():
        fig.add_trace(go.Scattergl(
            x=peaks[wrong] / fs, y=signal[peaks[wrong]], mode='markers', yaxis='y2',
            marker=dict(symbol='square-open', size=22, color='red', line=dict(width=1.5)),
            name='Differs from annotation', hoverinfo='skip'))

    lo, hi = np.percentile(signal, [0.05, 99.95])
    pad = 0.1 * (hi - lo)
    fig.update_layout(
        height=480, margin=dict(l=60, r=20, t=40, b=20),
        dragmode='pan', hovermode='closest',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        xaxis=dict(title='Time (seconds)', range=[0, min(window_sec, t[-1])],
                   rangeslider=dict(visible=True, thickness=0.14,
                                    range=[0, t[-1]], autorange=False,
                                    yaxis=dict(rangemode='auto'))),
        # primary y: only carries the overview envelope -> far outside the view
        yaxis=dict(range=[1e6, 1e6 + 1], visible=False, fixedrange=True),
        yaxis2=dict(title='Amplitude (mV)', overlaying='y', side='left',
                    range=[lo - pad, hi + 2 * pad], fixedrange=True,
                    zeroline=False),
    )
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
    window_sec = st.slider("Visible window (sec)", 5, 60, 10,
                           help="Initial width of the view; drag the edges of "
                                "the scroll bar window to zoom in or out.")

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
        st.session_state['view'] = (data_dir, record, artifacts)

# The results stay visible across reruns (e.g. while generating the RAG
# recommendation); analyze_record is cached, so this is cheap
if 'view' in st.session_state:
    v_dir, v_record, v_artifacts = st.session_state['view']
    signal, fs, kept, pred, model_type = analyze_record(v_dir, v_record, v_artifacts)

    # Plot
    st.subheader(f"ECG record {v_record} ({model_type}, "
                 f"{len(signal) / fs / 60:.1f} min)")
    fig = plot_ecg(signal, fs, kept, pred, window_sec)
    st.plotly_chart(fig, use_container_width=True,
                    config={'scrollZoom': False, 'displaylogo': False})
    st.caption("Scroll bar: drag the window to move through the whole record, "
               "drag its edges to zoom | Color = predicted class | "
               "* = abnormality | red square = differs from annotation | "
               "hover a beat for details")

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
                