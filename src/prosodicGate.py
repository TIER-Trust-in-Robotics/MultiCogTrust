"""
ProsodicGate
============
Uses ProsodicFeatureExtractor (speechProsodic.py) to extract features from a
speech segment and a trained classifier to decide whether to call the NLP model.

Datasets are used to train the classifier via fit(). Once trained, 
should_analyze() is the only method called at runtime."""

import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from speechProsodic import ProsodicFeatureExtractor

class ProsodicGate:

    def __init__(self, threshold:float = 0.5, estimator=None):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        
        self.threshold = threshold
        self._estimator = estimator
        self._extractor = ProsodicFeatureExtractor()
        self._pipeline = None


    # Feature extraction
    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Audio must be a numpy array, got {type(audio)}")
        if audio.ndim != 1:
            raise ValueError(f"Audio must be a 1D array, got shape {audio.shape}")
        if sr <= 0:
            raise ValueError(f"Sample rate must be positive, got {sr}")
        
        features = self._extractor.extract(audio, sr)
        return features["opensmile_features"].astype(np.float32)
    
    # Training : trains the gate on labelled data
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProsodicGate":
        if X.ndim != 2:
            raise ValueError(f"Feature matrix X must be 2D, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if len(np.unique(y)) < 2:
            raise ValueError("y must contain at least both classes (0 and 1)")
        
        clf = self._estimator or LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced", # accounts for more neutral than non-neutral speech
            solver="lbfgs"
        )
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        self._pipeline.fit(X, y)
        return self
    
    #Runtime inference

    def score(self, audio: np.ndarray, sr: int = 16000) -> float:
        if self._pipeline is None:
            raise RuntimeError("Gate has not been trained. Call fit() or load() first.")
        
        feats = self.extract_features(audio, sr)
        proba = self._pipeline.predict_proba(feats.reshape(1, -1))[0]  
        return float(proba[1])  # Probability of class 1 (non-neutral)
    
    def should_analyze(self, audio: np.ndarray, sr: int = 16000) -> tuple[bool, float]:
        """
        Returns (pass_gate: bool, score: float).

            pass_gate — True means call the NLP model
            score     — raw probability for logging/debugging
        """
        s = self.score(audio, sr)
        return s >= self.threshold, s
    
    # Persistence
    def save(self, path: str) -> None:
        if self._pipeline is None:
            raise RuntimeError("Gate has not been trained. Nothing to save. Call fit() first.")
        joblib.dump({
            "threshold": self.threshold,
            "pipeline": self._pipeline
        }, path)
        
    @classmethod
    def load(cls, path: str) -> "ProsodicGate":
        if not os.path.exists(path):
            raise FileNotFoundError(f"No gate model found at {path!r}")
        payload = joblib.load(path)
        gate = cls(threshold=payload["threshold"])
        gate._pipeline = payload["pipeline"]
        return gate