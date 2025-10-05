import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from io import StringIO


class DataProcessingService:
    def __init__(self):
        self.required_features = [
            'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
            'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol',
            'koi_steff', 'koi_slogg', 'koi_srad'
        ]

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate uploaded dataframe"""
        errors = []

        missing_cols = set(self.required_features) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")

        if df.empty:
            errors.append("DataFrame is empty")

        for col in self.required_features:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column {col} must be numeric")

        return len(errors) == 0, errors

    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality report"""
        total_rows = len(df)
        total_features = len(df.columns)

        missing_values = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_values[col] = int(missing_count)

        feature_stats = {}
        for col in self.required_features:
            if col in df.columns:
                feature_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                }

        anomalies_detected = 0
        for col in self.required_features:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                anomalies = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()
                anomalies_detected += int(anomalies)

        missing_ratio = sum(missing_values.values()) / (total_rows * len(self.required_features)) if total_rows > 0 else 0
        quality_score = max(0, 100 - (missing_ratio * 100) - (anomalies_detected / total_rows * 10 if total_rows > 0 else 0))

        recommendations = []
        if missing_ratio > 0.1:
            recommendations.append("High percentage of missing values detected. Consider data imputation.")
        if anomalies_detected > total_rows * 0.05:
            recommendations.append("Significant outliers detected. Review data for quality issues.")
        if total_rows < 10:
            recommendations.append("Small dataset size may affect prediction reliability.")

        return {
            'total_rows': total_rows,
            'total_features': total_features,
            'missing_values': missing_values,
            'feature_stats': feature_stats,
            'anomalies_detected': anomalies_detected,
            'quality_score': float(quality_score),
            'recommendations': recommendations
        }

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess uploaded dataframe"""
        df_processed = df.copy()

        for col in self.required_features:
            if col in df_processed.columns:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)

        return df_processed[self.required_features]

    def parse_csv_file(self, file_content: str) -> pd.DataFrame:
        """Parse CSV file content"""
        return pd.read_csv(StringIO(file_content))

    def get_sample_data(self) -> List[Dict[str, float]]:
        """Generate sample exoplanet data"""
        samples = [
            {
                'koi_period': 3.52474859,
                'koi_time0bk': 170.53875,
                'koi_impact': 0.146,
                'koi_duration': 2.95750,
                'koi_depth': 615.8,
                'koi_prad': 2.26,
                'koi_teq': 793,
                'koi_insol': 93.59,
                'koi_steff': 5455,
                'koi_slogg': 4.467,
                'koi_srad': 0.927
            },
            {
                'koi_period': 9.4877,
                'koi_time0bk': 133.5,
                'koi_impact': 0.586,
                'koi_duration': 4.95,
                'koi_depth': 874.2,
                'koi_prad': 3.12,
                'koi_teq': 1285,
                'koi_insol': 250.3,
                'koi_steff': 6117,
                'koi_slogg': 4.234,
                'koi_srad': 1.456
            },
            {
                'koi_period': 122.387,
                'koi_time0bk': 162.51,
                'koi_impact': 0.923,
                'koi_duration': 8.124,
                'koi_depth': 127.5,
                'koi_prad': 0.89,
                'koi_teq': 412,
                'koi_insol': 8.45,
                'koi_steff': 5234,
                'koi_slogg': 4.512,
                'koi_srad': 0.812
            }
        ]
        return samples


data_service = DataProcessingService()
