"""Load the trained expected-power models for live prediction and SHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBRegressor

from app.data import MODEL_FEATURES, PROJECT_ROOT

PLANT_MODEL_PATHS = {
    "Plant 1": PROJECT_ROOT / "models" / "expected_power_model_4135001.json",
    "Plant 2": PROJECT_ROOT / "models" / "expected_power_model_4136001.json",
}


@st.cache_resource(show_spinner=False)
def load_model(plant_name: str) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(str(PLANT_MODEL_PATHS[plant_name]))
    return model


def predict_ac_power(plant_name: str, features: pd.DataFrame) -> np.ndarray:
    model = load_model(plant_name)
    return model.predict(features[list(MODEL_FEATURES)])


def predict_single(
    plant_name: str,
    irradiation: float,
    ambient_temperature: float,
    module_temperature: float,
    hour: float,
) -> float:
    """Score one what-if scenario using the same feature encoding as training."""

    hour_radians = 2 * np.pi * hour / 24
    row = pd.DataFrame(
        [
            {
                "IRRADIATION": irradiation,
                "AMBIENT_TEMPERATURE": ambient_temperature,
                "MODULE_TEMPERATURE": module_temperature,
                "HOUR_SIN": np.sin(hour_radians),
                "HOUR_COS": np.cos(hour_radians),
            }
        ]
    )
    return float(predict_ac_power(plant_name, row)[0])


@st.cache_data(show_spinner=False)
def shap_importance(plant_name: str, sample: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, computed on a sampled subset."""

    model = load_model(plant_name)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[list(MODEL_FEATURES)])
    return (
        pd.DataFrame(
            {
                "feature": MODEL_FEATURES,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
