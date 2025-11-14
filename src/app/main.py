"""
FastAPI application exposing a simple prediction endpoint.
"""

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.ml.nba_predict import NBAPredictor

APP = FastAPI(
    title="NBA Player Longevity API",
    version="0.2.0",
)

PREDICTOR = NBAPredictor()


class PlayerFeatures(BaseModel):
    """Input features for a single player prediction."""

    Name: str
    GP: float
    MIN: float = Field(1.0)
    PTS: float
    FGM: float
    FGA: float
    FGperc: float = Field(alias="FG%")
    TP_made: float = Field(alias="3P Made")
    TPA: float = Field(alias="3PA")
    TPperc: float = Field(alias="3P%")
    FTM: float
    FTA: float
    FTperc: float = Field(alias="FT%")
    OREB: float
    DREB: float
    REB: float
    AST: float
    STL: float
    BLK: float
    TOV: float


@APP.on_event("startup")
def load_artifacts() -> None:
    """Load model, scaler and feature list into memory at startup."""
    global PREDICTOR

    PREDICTOR.model_load("models/")


def player_to_dataframe(player: PlayerFeatures) -> pd.DataFrame:
    """
    Convert PlayerFeatures into a dataframe aligned with training features.
    """
    # Use aliases so column names match the original dataset
    data_dict = player.dict(by_alias=True)

    df = pd.DataFrame([data_dict])
    print(df.columns)

    # Reorder and select columns to match training
    return df


@APP.post("/predict")
def predict(player: PlayerFeatures) -> dict:
    """
    Return a prediction for a single player using the trained model.
    """
    if PREDICTOR.model_ is None or PREDICTOR.scaler_ is None:
        return {"error": "Model not loaded"}

    player_df = player_to_dataframe(player)

    player_df = PREDICTOR.data_preprocess(player_df)
    print(player_df.head())
    predictions = PREDICTOR.predict(player_df.select_dtypes(include=[np.number]))
    print(predictions)
    pred = float(predictions[0][0])
    proba = float(predictions[1][0])
    return {
        "prediction": pred,
        "probability": proba,
    }
