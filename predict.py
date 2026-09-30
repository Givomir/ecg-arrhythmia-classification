"""
Unified prediction module - works with ALL models (cascade, RF, MLP, CNN).
Detects the model type automatically:
  * model.pkl   -> scikit-learn compatible model (cascade / Random Forest / MLP)
  * model.keras -> Keras CNN (hybrid input: morphology + context)

Fixed compared to the original:
  * the labels come from the saved LabelEncoder (not a hard-coded reversed dict)
  * the scaler is the one from training (transform only, no re-fit)
  * one and the same make_prediction() interface for all three models
"""
import os
import joblib
import numpy as np

# Default: the cascade model (best S detection and RAG routing, see README)
ARTIFACT_DIR = os.environ.get('ARTIFACT_DIR', 'artifacts_cascade')

# Load the shared artifacts (present for every model)
_scaler = joblib.load(os.path.join(ARTIFACT_DIR, 'scaler.pkl'))
_le = joblib.load(os.path.join(ARTIFACT_DIR, 'label_encoder.pkl'))

# Detect the model type from the file that is present
_keras_path = os.path.join(ARTIFACT_DIR, 'model.keras')
_pkl_path = os.path.join(ARTIFACT_DIR, 'model.pkl')

if os.path.exists(_keras_path):
    MODEL_TYPE = 'cnn'
    _meta = joblib.load(os.path.join(ARTIFACT_DIR, 'meta.pkl'))
    N_CONTEXT = _meta['n_context']
    _model = None  # lazy loading (TF is heavy)
elif os.path.exists(_pkl_path):
    MODEL_TYPE = 'sklearn'
    _model = joblib.load(_pkl_path)
else:
    raise FileNotFoundError(
        f"No model found in {ARTIFACT_DIR} (looked for model.keras or model.pkl)")

# Clinical descriptions of the AAMI groups
AAMI_DESC = {
    'N': 'Normal / bundle branch (normal beat)',
    'S': 'Supraventricular ectopic (supraventricular premature beat)',
    'V': 'Ventricular ectopic (ventricular premature beat)',
    'F': 'Fusion beat (fusion of normal and ventricular beat)',
    'Q': 'Unknown / paced (unclassifiable / paced beat)',
}


def _get_keras_model():
    """Lazy loading of the Keras model (so importing does not pull in TF)."""
    global _model
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model(_keras_path)
    return _model


def _predict_sklearn(X):
    pred_idx = int(_model.predict(X)[0])
    proba = {}
    if hasattr(_model, 'predict_proba'):
        probs = _model.predict_proba(X)[0]
        proba = {_le.inverse_transform([i])[0]: float(round(p, 4))
                 for i, p in enumerate(probs)}
    return pred_idx, proba


def _predict_cnn(X):
    # The CNN expects a split input: morphology (Conv branch) + context (dense branch)
    morph = X[:, :-N_CONTEXT][..., np.newaxis]
    ctx = X[:, -N_CONTEXT:]
    model = _get_keras_model()
    probs = model.predict({'morphology': morph, 'context': ctx}, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    proba = {_le.inverse_transform([i])[0]: float(round(p, 4))
             for i, p in enumerate(probs)}
    return pred_idx, proba


def make_prediction(beat_features):
    """
    beat_features: the full feature vector for ONE beat (morphology + context),
                   with the same length as in training.
    Works identically for RF, MLP and CNN - the calling code does not change.
    """
    try:
        # Scaling with the SAVED scaler (transform only)
        X = np.nan_to_num(np.asarray(beat_features, dtype=float)).reshape(1, -1)
        X = _scaler.transform(X)

        if MODEL_TYPE == 'cnn':
            pred_idx, proba = _predict_cnn(X)
        else:
            pred_idx, proba = _predict_sklearn(X)

        label = _le.inverse_transform([pred_idx])[0]
        return {
            'model_type': MODEL_TYPE,
            'prediction': str(label),
            'diagnosis': AAMI_DESC.get(label, 'Unknown'),
            'probabilities': proba,
        }
    except Exception as e:
        return {'error': str(e)}