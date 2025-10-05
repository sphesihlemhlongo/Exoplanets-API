import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os


class ExoplanetMLService:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
            'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol',
            'koi_steff', 'koi_slogg', 'koi_srad'
        ]
        self.is_trained = False

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._initialize_placeholder_model()

    def _initialize_placeholder_model(self):
        """Initialize a placeholder stacked ensemble model"""
        base_models = [
            RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
        ]
        meta_model = LogisticRegression(random_state=42)

        self.model = {
            'base_models': base_models,
            'meta_model': meta_model,
            'scaler': self.scaler
        }

        X_dummy = np.random.randn(100, len(self.feature_names))
        y_dummy = np.random.randint(0, 2, 100)

        self.scaler.fit(X_dummy)
        X_scaled = self.scaler.transform(X_dummy)

        for model in base_models:
            model.fit(X_scaled, y_dummy)

        base_predictions = np.column_stack([
            model.predict_proba(X_scaled) for model in base_models
        ])
        meta_model.fit(base_predictions, y_dummy)

        self.is_trained = True

    def preprocess_features(self, features: Dict[str, float]) -> np.ndarray:
        """Preprocess and validate input features"""
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name not in features:
                raise ValueError(f"Missing required feature: {feature_name}")
            feature_vector.append(features[feature_name])

        return np.array(feature_vector).reshape(1, -1)

    def predict(self, features: Dict[str, float]) -> Tuple[int, float, Dict[str, float]]:
        """Make a prediction for a single exoplanet candidate"""
        if not self.is_trained:
            raise RuntimeError("Model is not trained")

        X = self.preprocess_features(features)
        X_scaled = self.model['scaler'].transform(X)

        base_predictions = np.column_stack([
            model.predict_proba(X_scaled) for model in self.model['base_models']
        ])

        probabilities = self.model['meta_model'].predict_proba(base_predictions)[0]
        prediction = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        return prediction, confidence, {
            'false_positive': float(probabilities[0]),
            'exoplanet': float(probabilities[1])
        }

    def batch_predict(self, batch_features: List[Dict[str, float]]) -> List[Tuple[int, float, Dict[str, float]]]:
        """Make predictions for multiple exoplanet candidates"""
        results = []
        for features in batch_features:
            try:
                result = self.predict(features)
                results.append(result)
            except Exception as e:
                results.append((0, 0.0, {'error': str(e)}))
        return results

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the ensemble"""
        if not self.is_trained:
            return {}

        importances = []
        for model in self.model['base_models']:
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)

        if importances:
            avg_importance = np.mean(importances, axis=0)
            return dict(zip(self.feature_names, avg_importance.tolist()))
        return {}

    def save_model(self, path: str):
        """Save the trained model"""
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """Load a trained model"""
        self.model = joblib.load(path)
        self.is_trained = True


ml_service = ExoplanetMLService()
