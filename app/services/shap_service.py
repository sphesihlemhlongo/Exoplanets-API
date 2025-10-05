import numpy as np
from typing import Dict, List
import shap


class SHAPExplainerService:
    def __init__(self, ml_service):
        self.ml_service = ml_service
        self.explainer = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize SHAP explainer with background data"""
        try:
            X_background = np.random.randn(100, len(self.ml_service.feature_names))
            X_background = self.ml_service.scaler.transform(X_background)

            def model_predict(X):
                base_predictions = np.column_stack([
                    model.predict_proba(X) for model in self.ml_service.model['base_models']
                ])
                return self.ml_service.model['meta_model'].predict_proba(base_predictions)[:, 1]

            self.explainer = shap.KernelExplainer(model_predict, X_background[:50])
        except Exception as e:
            print(f"SHAP explainer initialization warning: {e}")
            self.explainer = None

    def get_explanation(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get SHAP values for feature contributions"""
        if not self.explainer:
            return self._fallback_explanation(features)

        try:
            X = self.ml_service.preprocess_features(features)
            X_scaled = self.ml_service.scaler.transform(X)

            shap_values = self.explainer.shap_values(X_scaled, nsamples=100)

            explanations = {}
            for i, feature_name in enumerate(self.ml_service.feature_names):
                explanations[feature_name] = float(shap_values[0][i])

            return explanations
        except Exception as e:
            print(f"SHAP explanation error: {e}")
            return self._fallback_explanation(features)

    def _fallback_explanation(self, features: Dict[str, float]) -> Dict[str, float]:
        """Fallback to feature importance when SHAP fails"""
        importance = self.ml_service.get_feature_importance()

        explanations = {}
        for feature_name in self.ml_service.feature_names:
            feature_value = features.get(feature_name, 0.0)
            base_importance = importance.get(feature_name, 0.0)
            explanations[feature_name] = float(feature_value * base_importance * 0.01)

        return explanations


def create_shap_service(ml_service):
    return SHAPExplainerService(ml_service)
