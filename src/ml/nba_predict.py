"""
script to predict the TARGET_5Yrs column from a CSV file. This work on a
new dataset, only if it has the same features than the subject file (nba_logreg.csv)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class NBAPredictor:
    """
    Class to handle preprocessing, training and prediction for TARGET_5Yrs.
    """

    def __init__(self) -> None:
        """
        Initialize the predictor.
        """

        self.model_: Optional[XGBClassifier] = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_columns_: Optional[list[str]] = None

    def data_load(self, csv_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        """

        return pd.read_csv(csv_path)

    def data_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        data cleaning (percentages) and features engineering
        """
        res = df.copy()

        res["3P%"] = res.apply(
            lambda row: 0.0 if row["3PA"] == 0 else row["3P Made"] / row["3PA"], axis=1
        )
        res["FT%"] = res.apply(
            lambda row: 0.0 if row["FTA"] == 0 else row["FTM"] / row["FTA"], axis=1
        )
        res["FG%"] = res.apply(
            lambda row: 0.0 if row["FGA"] == 0 else row["FGM"] / row["FGA"], axis=1
        )

        res["ASTperTOV"] = res["AST"] / res["TOV"]
        res["ASTPTSperTOV"] = (res["AST"] + res["PTS"]) / res["TOV"]

        cols = [
            "GP",
            "PTS",
            "FGM",
            "FGA",
            "FG%",
            "3P Made",
            "3PA",
            "3P%",
            "FTM",
            "FTA",
            "FT%",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "ASTperTOV",
            "ASTPTSperTOV",
        ]

        for col in cols:
            res[f"{col}perMIN"] = res[col] / res["MIN"]

        return res

    def data_split(
        self, data: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        """

        x_data = data.drop(columns=["TARGET_5Yrs"])
        y_data = data["TARGET_5Yrs"]

        x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_data,
            test_size=test_size,
            random_state=42,
            stratify=y_data,
        )
        return x_train, x_test, y_train, y_test

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        save_dir: Optional[str] = True,
    ) -> None:
        """
        Train an XGBoost classifier with 5-fold cross validation, optimizing
        hyperparameters for positive-class recall.
        """
        # Keep only numeric features for the model
        x_train_numeric = x_train.select_dtypes(include=[np.number])
        self.feature_columns_ = x_train_numeric.columns

        if not list(self.feature_columns_):
            raise ValueError("No numeric feature columns found for training.")

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        # Booster parameters
                        eta=0.01,
                        gamma=0,
                        max_depth=3,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=1,
                        lambda_=1,
                        alpha=0,
                        n_estimators=500,
                        tree_method="auto",
                        importance_type="gain",
                        # Learning task parameters
                        objective="binary:logistic",
                        eval_metric="auc",
                        random_state=42,
                    ),
                ),
            ]
        )

        pipeline.fit(x_train_numeric, y_train)

        self.scaler_ = pipeline.named_steps["scaler"]
        self.model_ = pipeline.named_steps["clf"]

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            scaler_path = os.path.join(save_dir, "scaler.joblib")
            model_path = os.path.join(save_dir, "model.joblib")
            features_path = os.path.join(save_dir, "feature_columns.joblib")

            joblib.dump(self.scaler_, scaler_path)
            joblib.dump(self.model_, model_path)
            joblib.dump(list(self.feature_columns_), features_path)

    def model_load(self, model_dir: str) -> None:
        """
        Load the StandardScaler, model weights and feature list from disk.
        """
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        model_path = os.path.join(model_dir, "model.joblib")
        features_path = os.path.join(model_dir, "feature_columns.joblib")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at '{scaler_path}'.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'.")
        if not os.path.exists(features_path):
            raise FileNotFoundError(
                f"Feature list file not found at '{features_path}'."
            )

        self.scaler_ = joblib.load(scaler_path)
        self.model_ = joblib.load(model_path)
        self.feature_columns_ = joblib.load(features_path)

    def predict(self, x_test, y_test) -> pd.DataFrame:
        """
        Predict TARGET_5Yrs using the trained model.
        """
        if self.model_ is None or self.scaler_ is None or self.feature_columns_ is None:
            raise ValueError("Model, scaler or feature columns are not loaded/trained.")

        missing_features = [
            col for col in self.feature_columns_ if col not in (x_test.columns.tolist())
        ]
        if missing_features:
            missing_str = ", ".join(missing_features)
            raise ValueError(
                f"The following required feature columns are missing in "
                f"test data: {missing_str}"
            )

        x_test_numeric = x_test[list(self.feature_columns_)]
        x_test_scaled = self.scaler_.transform(x_test_numeric)

        y_pred = self.model_.predict(x_test_scaled)
        y_proba = self.model_.predict_proba(x_test_scaled)[:, 1]

        result_df = pd.concat([x_test, y_test], axis=1)
        result_df["TARGET_5Yrs_pred"] = y_pred
        result_df["TARGET_5Yrs_pred_proba"] = y_proba

        return result_df

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        results_dir: Optional[str] = None,
    ) -> None:
        """
        Print classification report and optionally save confusion matrix, feature importances and
        classification report into a single PDF file.
        """
        print("Classification report:")
        print(classification_report(y_true, y_pred))

        if results_dir is None:
            return

        os.makedirs(results_dir, exist_ok=True)
        pdf_path = os.path.join(results_dir, "evaluation_report.pdf")

        with PdfPages(pdf_path) as pdf:
            # Confusion matrix page
            fig_cm, ax_cm = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax_cm)
            ax_cm.set_title("Confusion Matrix")
            fig_cm.tight_layout()
            pdf.savefig(fig_cm)
            plt.close(fig_cm)

            # Classification report page
            report_text = classification_report(y_true, y_pred)

            fig_text, ax_text = plt.subplots()
            ax_text.axis("off")
            ax_text.text(
                0.01,
                0.99,
                "Classification Report\n\n" + report_text,
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
            )
            fig_text.tight_layout()
            pdf.savefig(fig_text)
            plt.close(fig_text)

            # Feature importances
            importances = self.model_.feature_importances_
            indices = np.argsort(importances)[::-1]
            sorted_features = self.feature_columns_[indices]
            sorted_importances = importances[indices]
            # Plot
            fig_fi, ax_fi = plt.subplots(figsize=(6, 10))
            ax_fi.barh(sorted_features, sorted_importances)
            ax_fi.set_title("XGBoost Feature Importance (Gain)")
            fig_fi.tight_layout()
            pdf.savefig(fig_fi)
            plt.close(fig_text)


if __name__ == "__main__":
    # Example usage (can be removed or adapted as needed).
    #
    predictor = NBAPredictor()
    data = predictor.data_load("./data/nba_logreg.csv")
    data = predictor.data_preprocess(data)
    x_train_df, x_test_df, y_train_series, y_test_series = predictor.data_split(data)

    predictor.train(x_train_df, y_train_series, save_dir="models")
    y_test_pred = predictor.model_.predict(
        predictor.scaler_.transform(
            x_test_df[predictor.feature_columns_].select_dtypes(include=[np.number])
        )
    )
    predictor.evaluate_model(y_test_series.values, y_test_pred, results_dir="results/")
