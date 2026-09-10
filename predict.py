"""
Единен модул за предсказание - работи с ВСИЧКИ модели (RF, MLP, CNN).
Автоматично разпознава типа:
  * model.pkl   -> scikit-learn модел (Random Forest / MLP)
  * model.keras -> Keras CNN (хибриден вход: морфология + контекст)

Поправено спрямо оригинала:
  * етикетите идват от запазения LabelEncoder (не хардкоднат обърнат речник)
  * scaler-ът е този от трениране (само transform, без повторен fit)
  * един и същ интерфейс make_prediction() за трите модела
"""
import os
import joblib
import numpy as np

ARTIFACT_DIR = os.environ.get('ARTIFACT_DIR', 'artifacts')

# Зареждаме общите артефакти (има ги при всички модели)
_scaler = joblib.load(os.path.join(ARTIFACT_DIR, 'scaler.pkl'))
_le = joblib.load(os.path.join(ARTIFACT_DIR, 'label_encoder.pkl'))

# Разпознаваме типа модел по наличния файл
_keras_path = os.path.join(ARTIFACT_DIR, 'model.keras')
_pkl_path = os.path.join(ARTIFACT_DIR, 'model.pkl')

if os.path.exists(_keras_path):
    MODEL_TYPE = 'cnn'
    _meta = joblib.load(os.path.join(ARTIFACT_DIR, 'meta.pkl'))
    N_CONTEXT = _meta['n_context']
    _model = None  # отложено зареждане (TF е тежък)
elif os.path.exists(_pkl_path):
    MODEL_TYPE = 'sklearn'
    _model = joblib.load(_pkl_path)
else:
    raise FileNotFoundError(
        f"Няма намерен модел в {ARTIFACT_DIR} (търсих model.keras или model.pkl)")

# Клинични описания на AAMI групите
AAMI_DESC = {
    'N': 'Normal / bundle branch (нормален удар)',
    'S': 'Supraventricular ectopic (надкамерна екстрасистола)',
    'V': 'Ventricular ectopic (камерна екстрасистола)',
    'F': 'Fusion beat (сливен удар)',
    'Q': 'Unknown / paced (неопределим / пейсиран)',
}


def _get_keras_model():
    """Отложено зареждане на Keras модела (за да не тегли TF при импорт)."""
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
    # CNN очаква разделен вход: морфология (Conv клон) + контекст (dense клон)
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
    beat_features: пълният вектор признаци за ЕДИН удар (морфология + контекст),
                   със същата дължина като при трениране.
    Работи идентично за RF, MLP и CNN - извикващият код не се променя.
    """
    try:
        # Скалиране със ЗАПАЗЕНИЯ scaler (само transform)
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