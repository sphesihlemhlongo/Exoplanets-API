import optuna
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class OptunaOptimizationService:
    def __init__(self):
        self.studies: Dict[str, optuna.Study] = {}
        self.study_status: Dict[str, Dict[str, Any]] = {}

    def create_objective(self, X_train, y_train, metric='accuracy'):
        """Create an objective function for Optuna"""
        def objective(trial):
            rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
            rf_max_depth = trial.suggest_int('rf_max_depth', 3, 20)
            rf_min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 20)

            gb_n_estimators = trial.suggest_int('gb_n_estimators', 50, 300)
            gb_max_depth = trial.suggest_int('gb_max_depth', 3, 15)
            gb_learning_rate = trial.suggest_float('gb_learning_rate', 0.01, 0.3)

            rf_model = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_split=rf_min_samples_split,
                random_state=42,
                n_jobs=-1
            )

            gb_model = GradientBoostingClassifier(
                n_estimators=gb_n_estimators,
                max_depth=gb_max_depth,
                learning_rate=gb_learning_rate,
                random_state=42
            )

            rf_score = cross_val_score(rf_model, X_train, y_train, cv=3, scoring=metric, n_jobs=-1).mean()
            gb_score = cross_val_score(gb_model, X_train, y_train, cv=3, scoring=metric, n_jobs=-1).mean()

            ensemble_score = (rf_score + gb_score) / 2

            return ensemble_score

        return objective

    def start_optimization(
        self,
        study_id: str,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        metric: str = 'accuracy'
    ) -> str:
        """Start a new optimization study"""
        X_dummy = np.random.randn(500, 11)
        y_dummy = np.random.randint(0, 2, 500)

        study = optuna.create_study(
            study_name=study_id,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        self.studies[study_id] = study
        self.study_status[study_id] = {
            'status': 'running',
            'started_at': datetime.utcnow(),
            'current_trial': 0,
            'total_trials': n_trials
        }

        objective = self.create_objective(X_dummy, y_dummy, metric)

        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        self.study_status[study_id]['status'] = 'completed'
        self.study_status[study_id]['completed_at'] = datetime.utcnow()

        return study_id

    def get_study_status(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an optimization study"""
        if study_id not in self.studies:
            return None

        study = self.studies[study_id]
        status_info = self.study_status.get(study_id, {})

        best_trial = study.best_trial if study.trials else None

        return {
            'study_id': study_id,
            'status': status_info.get('status', 'unknown'),
            'current_trial': len(study.trials),
            'total_trials': status_info.get('total_trials', 0),
            'best_score': study.best_value if best_trial else None,
            'best_params': study.best_params if best_trial else None,
            'elapsed_time': (datetime.utcnow() - status_info.get('started_at', datetime.utcnow())).total_seconds()
        }

    def get_study_results(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Get the results of a completed optimization study"""
        if study_id not in self.studies:
            return None

        study = self.studies[study_id]

        optimization_history = []
        for trial in study.trials:
            optimization_history.append({
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })

        return {
            'study_id': study_id,
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'optimization_history': optimization_history,
            'timestamp': datetime.utcnow()
        }

    def get_parameter_importance(self, study_id: str) -> Optional[Dict[str, float]]:
        """Get parameter importance for a study"""
        if study_id not in self.studies:
            return None

        study = self.studies[study_id]

        try:
            importance = optuna.importance.get_param_importances(study)
            return importance
        except Exception as e:
            print(f"Parameter importance calculation error: {e}")
            return None


optimization_service = OptunaOptimizationService()
