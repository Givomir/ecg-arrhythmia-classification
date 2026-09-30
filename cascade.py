"""
Two-stage (cascade) classifier with a dedicated S-vs-N detector
================================================================
Stage 1: multi-class Random Forest (N/S/V/F/Q) on morphology + rhythm.
Stage 2: a Random Forest trained ONLY on N and S beats, applied to the beats
         that stage 1 calls N or S. It sees compressed morphology (PCA) plus
         the rhythm features and decides S vs N with a threshold that is tuned
         by cross-validation by patient (see train_cascade.py).

Why: S is confused with N (narrow QRS, nearly the same shape), not with V.
A dedicated S/N stage with rhythm-irregularity features cuts the false S
alarms in atrial fibrillation records (see ablation_s.py / README).

Input: the SCALED feature vector from utility.record_features(rhythm=True)
(239 base + 8 rhythm features). The column layout is defined here, so any
code that uses scaler.pkl + model.pkl works with this model unchanged.

This module must be importable when model.pkl is unpickled.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utility import N_MORPH, N_CONTEXT, N_RHYTHM

_BASE = N_MORPH + N_CONTEXT
MORPH_COLS = np.arange(0, N_MORPH)
# rr (3) + normalized rr (2) + long-window rr (2) + irregularity (6); no P wave
RHYTHM_COLS = np.concatenate([np.arange(N_MORPH, N_MORPH + 5),
                              np.arange(_BASE, _BASE + N_RHYTHM)])
STAGE1_COLS = np.concatenate([MORPH_COLS, RHYTHM_COLS])
N_FEATURES = _BASE + N_RHYTHM


class CascadeClassifier:
    """
    s_class / n_class: encoded label indices of S and N (from the LabelEncoder)
    n_classes:         total number of classes
    threshold:         P(S) threshold of stage 2 (set after tuning)
    """

    def __init__(self, s_class, n_class, n_classes, threshold=0.5,
                 n_estimators=150, n_estimators_stage2=300, n_pca=10,
                 random_state=42):
        self.s_class = s_class
        self.n_class = n_class
        self.n_classes = n_classes
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.n_estimators_stage2 = n_estimators_stage2
        self.n_pca = n_pca
        self.random_state = random_state
        self.classes_ = np.arange(n_classes)

    # -- training --------------------------------------------------------
    def fit(self, X, y):
        self.stage1_ = RandomForestClassifier(
            n_estimators=self.n_estimators, class_weight='balanced',
            min_samples_leaf=2, n_jobs=-1, random_state=self.random_state)
        self.stage1_.fit(X[:, STAGE1_COLS], y)

        m = (y == self.n_class) | (y == self.s_class)
        self.pca_ = make_pipeline(
            StandardScaler(), PCA(n_components=self.n_pca, random_state=self.random_state))
        self.pca_.fit(X[m][:, MORPH_COLS])
        # larger leaves + balanced_subsample: more robust across patients
        # than gradient boosting, which overfits the few S patients
        self.stage2_ = RandomForestClassifier(
            n_estimators=self.n_estimators_stage2, min_samples_leaf=5,
            class_weight='balanced_subsample', n_jobs=-1,
            random_state=self.random_state)
        self.stage2_.fit(self._stage2_input(X[m]), (y[m] == self.s_class).astype(int))
        return self

    def _stage2_input(self, X):
        return np.hstack([self.pca_.transform(X[:, MORPH_COLS]), X[:, RHYTHM_COLS]])

    # -- inference -------------------------------------------------------
    def stage_scores(self, X):
        """Returns (P1, gate, p2): stage-1 probabilities, the beats that go to
        stage 2 (stage 1 says N or S), and stage-2 P(S) (0 outside the gate)."""
        P1 = np.zeros((len(X), self.n_classes))
        P1[:, self.stage1_.classes_] = self.stage1_.predict_proba(X[:, STAGE1_COLS])
        gate = np.isin(P1.argmax(1), [self.n_class, self.s_class])
        p2 = np.zeros(len(X))
        if gate.any():
            p2[gate] = self.stage2_.predict_proba(self._stage2_input(X[gate]))[:, 1]
        return P1, gate, p2

    def predict_from_scores(self, P1, gate, p2, threshold=None):
        t = self.threshold if threshold is None else threshold
        rest = P1.copy()
        rest[:, self.s_class] = -1
        pred = rest.argmax(1)
        pred[gate] = self.n_class
        pred[gate & (p2 >= t)] = self.s_class
        return pred

    def predict(self, X):
        return self.predict_from_scores(*self.stage_scores(X))

    def predict_proba(self, X):
        """
        Stage-1 probabilities, with the N+S mass of the gated beats split by
        stage 2. Informative only: predict() uses the tuned threshold, not
        the argmax of these probabilities.
        """
        P1, gate, p2 = self.stage_scores(X)
        P = P1.copy()
        ns = P1[gate, self.n_class] + P1[gate, self.s_class]
        P[gate, self.s_class] = ns * p2[gate]
        P[gate, self.n_class] = ns * (1 - p2[gate])
        return P
