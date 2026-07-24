"""Data loading, validation, and reusable dashboard aggregations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paper"
    / "paper_dataset_with_masked_loss.csv"
)
PAPER_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "paper"

EMISSION_FACTOR_KG_PER_KWH = 0.703
FAULT_CLASSES = ("TOTAL_LOSS", "PARTIAL_LOSS")
INTERVAL_MINUTES = 15
RECOMMENDED_ACTIONS = {
    "TOTAL_LOSS": "Dispatch crew now",
    "PARTIAL_LOSS": "Inspect within 24 h",
}

REQUIRED_COLUMNS = {
    "DATE_TIME",
    "PLANT_NAME",
    "SOURCE_KEY",
    "ANOMALY_CLASS",
    "TRUE_ENERGY_LOSS_KWH",
    "TRUE_CO2_LOSS_KG",
    "ENERGY_KWH_INTERVAL",
    "IS_DAY",
    "AC_POWER",
    "EXPECTED_AC_POWER",
    "POWER_LOSS_KW",
    "DC_POWER_CORRECTED",
}

RAW_LOSS_KWH = 847_213.65
TRAIN_TEST_CUTOFF = pd.Timestamp("2020-06-11")
MODEL_FEATURES = (
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "HOUR_SIN",
    "HOUR_COS",
)

MODEL_PERFORMANCE = pd.DataFrame(
    [
        {"plant": "Plant 1", "split": "Train", "r2": 0.9775, "mae_kw": 29.10, "rmse_kw": 58.00, "mape_pct": 6.03},
        {"plant": "Plant 1", "split": "Test", "r2": 0.9619, "mae_kw": 31.08, "rmse_kw": 68.59, "mape_pct": 9.24},
        {"plant": "Plant 2", "split": "Train", "r2": 0.5403, "mae_kw": 156.72, "rmse_kw": 279.36, "mape_pct": 22.18},
        {"plant": "Plant 2", "split": "Test", "r2": 0.5386, "mae_kw": 101.45, "rmse_kw": 199.23, "mape_pct": 33.38},
    ]
)

BASELINE_COMPARISON = pd.DataFrame(
    [
        {"method": "Rule-Based Taxonomy", "precision": 1.00, "recall": 0.58, "f1": 0.74, "kappa": 0.74},
        {"method": "Isolation Forest", "precision": 0.32, "recall": 0.29, "f1": 0.30, "kappa": 0.30},
        {"method": "One-Class SVM", "precision": 0.17, "recall": 0.52, "f1": 0.25, "kappa": 0.24},
        {"method": "LOF", "precision": 0.15, "recall": 0.39, "f1": 0.21, "kappa": 0.21},
    ]
)


@dataclass(frozen=True)
class DatasetMeta:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    row_count: int
    plant_count: int
    inverter_count: int


def _file_fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner=False)
def _read_master_data(
    path_text: str, fingerprint: tuple[int, int]
) -> pd.DataFrame:
    """Read and normalize the master dataset.

    ``fingerprint`` deliberately participates in Streamlit's cache key so that
    regenerating the pipeline output invalidates the cached dataframe.
    """

    del fingerprint
    frame = pd.read_csv(path_text, parse_dates=["DATE_TIME"])

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Master dataset is missing columns: {missing_text}")

    numeric_columns = (
        "TRUE_ENERGY_LOSS_KWH",
        "TRUE_CO2_LOSS_KG",
        "ENERGY_KWH_INTERVAL",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    if frame["IS_DAY"].dtype != bool:
        frame["IS_DAY"] = (
            frame["IS_DAY"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
            .astype(bool)
        )

    frame["DATE"] = frame["DATE_TIME"].dt.normalize()
    frame["ANOMALY_CLASS"] = frame["ANOMALY_CLASS"].astype("string")
    frame["PLANT_NAME"] = frame["PLANT_NAME"].astype("string")
    frame["SOURCE_KEY"] = frame["SOURCE_KEY"].astype("string")
    return frame


def load_master_data(path: Path = MASTER_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "Master dashboard dataset was not found. Run `python reproduce_all.py` "
            f"to generate: {path}"
        )
    return _read_master_data(str(path), _file_fingerprint(path))


def dataset_meta(frame: pd.DataFrame) -> DatasetMeta:
    return DatasetMeta(
        start_date=frame["DATE_TIME"].min(),
        end_date=frame["DATE_TIME"].max(),
        row_count=len(frame),
        plant_count=frame["PLANT_NAME"].nunique(),
        inverter_count=frame["SOURCE_KEY"].nunique(),
    )


@st.cache_data(show_spinner=False)
def overview_aggregates(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return plant comparison and daily fault activity tables."""

    plant_totals = (
        frame.groupby("PLANT_NAME", observed=True)
        .agg(
            masked_loss_kwh=("TRUE_ENERGY_LOSS_KWH", "sum"),
            co2_loss_kg=("TRUE_CO2_LOSS_KG", "sum"),
            actual_energy_kwh=("ENERGY_KWH_INTERVAL", "sum"),
        )
        .reset_index()
    )

    affected = (
        frame.loc[frame["ANOMALY_CLASS"].isin(FAULT_CLASSES)]
        .groupby("PLANT_NAME", observed=True)["SOURCE_KEY"]
        .nunique()
        .rename("affected_inverters")
        .reset_index()
    )
    plant_totals = plant_totals.merge(
        affected, on="PLANT_NAME", how="left"
    ).fillna({"affected_inverters": 0})
    plant_totals["affected_inverters"] = plant_totals[
        "affected_inverters"
    ].astype(int)
    plant_totals["loss_rate_pct"] = (
        plant_totals["masked_loss_kwh"]
        .div(plant_totals["actual_energy_kwh"].replace(0, pd.NA))
        .mul(100)
        .fillna(0.0)
    )
    plant_totals["co2_loss_tonnes"] = plant_totals["co2_loss_kg"] / 1000

    daily_faults = (
        frame.loc[frame["ANOMALY_CLASS"].isin(FAULT_CLASSES)]
        .groupby(["DATE", "PLANT_NAME", "ANOMALY_CLASS"], observed=True)
        .agg(
            affected_inverters=("SOURCE_KEY", "nunique"),
            flagged_intervals=("SOURCE_KEY", "size"),
            masked_loss_kwh=("TRUE_ENERGY_LOSS_KWH", "sum"),
        )
        .reset_index()
    )
    return plant_totals, daily_faults


def filter_plants(frame: pd.DataFrame, plants: list[str]) -> pd.DataFrame:
    if not plants:
        return frame.iloc[0:0]
    return frame.loc[frame["PLANT_NAME"].isin(plants)]


@st.cache_data(show_spinner=False)
def build_fault_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive same-class fault intervals per inverter into events.

    A run ends whenever ``ANOMALY_CLASS`` changes for that inverter (night hours
    fall back to NORMAL, so multi-day faults naturally split into daily events,
    matching how an operator would encounter them).
    """

    ordered = frame.sort_values(["SOURCE_KEY", "DATE_TIME"])
    run_id = ordered.groupby("SOURCE_KEY", observed=True)["ANOMALY_CLASS"].transform(
        lambda classes: (classes != classes.shift()).cumsum()
    )
    faulty = ordered.loc[ordered["ANOMALY_CLASS"].isin(FAULT_CLASSES)].copy()
    faulty["EVENT_RUN"] = run_id.loc[faulty.index]

    events = (
        faulty.groupby(
            ["SOURCE_KEY", "PLANT_NAME", "ANOMALY_CLASS", "EVENT_RUN"],
            observed=True,
        )
        .agg(
            start_time=("DATE_TIME", "min"),
            end_time=("DATE_TIME", "max"),
            interval_count=("DATE_TIME", "size"),
            masked_loss_kwh=("TRUE_ENERGY_LOSS_KWH", "sum"),
            co2_loss_kg=("TRUE_CO2_LOSS_KG", "sum"),
        )
        .reset_index()
        .drop(columns="EVENT_RUN")
    )
    events["duration_minutes"] = events["interval_count"] * INTERVAL_MINUTES
    events["recommended_action"] = events["ANOMALY_CLASS"].map(RECOMMENDED_ACTIONS)
    return events.sort_values("masked_loss_kwh", ascending=False).reset_index(
        drop=True
    )


def inverter_window(
    frame: pd.DataFrame,
    source_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Return one inverter's raw record between two timestamps, inclusive."""

    mask = (
        frame["SOURCE_KEY"].eq(source_key)
        & frame["DATE_TIME"].ge(start)
        & frame["DATE_TIME"].le(end)
    )
    return frame.loc[
        mask,
        [
            "DATE_TIME",
            "AC_POWER",
            "DC_POWER_CORRECTED",
            "EXPECTED_AC_POWER",
            "ANOMALY_CLASS",
        ],
    ].sort_values("DATE_TIME")


def inverter_timeseries(frame: pd.DataFrame, source_key: str) -> pd.DataFrame:
    """Return one inverter's chronological actual/expected power record."""

    return (
        frame.loc[frame["SOURCE_KEY"].eq(source_key)]
        .sort_values("DATE_TIME")[
            [
                "DATE_TIME",
                "AC_POWER",
                "EXPECTED_AC_POWER",
                "POWER_LOSS_KW",
                "ANOMALY_CLASS",
                "TRUE_ENERGY_LOSS_KWH",
            ]
        ]
        .reset_index(drop=True)
    )


def inverter_summary(events: pd.DataFrame, source_key: str) -> dict[str, float | int]:
    """Aggregate an inverter's fault events into headline drill-down stats."""

    inverter_events = events.loc[events["SOURCE_KEY"].eq(source_key)]
    return {
        "event_count": int(len(inverter_events)),
        "total_loss_events": int(
            (inverter_events["ANOMALY_CLASS"] == "TOTAL_LOSS").sum()
        ),
        "partial_loss_events": int(
            (inverter_events["ANOMALY_CLASS"] == "PARTIAL_LOSS").sum()
        ),
        "masked_loss_kwh": float(inverter_events["masked_loss_kwh"].sum()),
        "downtime_hours": float(inverter_events["duration_minutes"].sum() / 60),
    }


@st.cache_data(show_spinner=False)
def loss_by_plant_class(frame: pd.DataFrame) -> pd.DataFrame:
    """Masked loss and carbon impact split by plant and fault class."""

    breakdown = (
        frame.loc[frame["ANOMALY_CLASS"].isin(FAULT_CLASSES)]
        .groupby(["PLANT_NAME", "ANOMALY_CLASS"], observed=True)
        .agg(
            masked_loss_kwh=("TRUE_ENERGY_LOSS_KWH", "sum"),
            co2_loss_kg=("TRUE_CO2_LOSS_KG", "sum"),
        )
        .reset_index()
    )
    breakdown["co2_loss_tonnes"] = breakdown["co2_loss_kg"] / 1000
    return breakdown


@st.cache_data(show_spinner=False)
def daily_loss_trend(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily masked energy loss and carbon impact, by plant."""

    trend = (
        frame.loc[frame["ANOMALY_CLASS"].isin(FAULT_CLASSES)]
        .groupby(["DATE", "PLANT_NAME"], observed=True)
        .agg(
            masked_loss_kwh=("TRUE_ENERGY_LOSS_KWH", "sum"),
            co2_loss_kg=("TRUE_CO2_LOSS_KG", "sum"),
        )
        .reset_index()
    )
    trend["co2_loss_tonnes"] = trend["co2_loss_kg"] / 1000
    return trend


@st.cache_data(show_spinner=False)
def inverter_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-inverter masked loss ranking, worst first."""

    ranking = (
        frame.groupby(["SOURCE_KEY", "PLANT_NAME"], observed=True)
        .agg(
            masked_loss_kwh=("TRUE_ENERGY_LOSS_KWH", "sum"),
            co2_loss_kg=("TRUE_CO2_LOSS_KG", "sum"),
        )
        .reset_index()
    )
    ranking["co2_loss_tonnes"] = ranking["co2_loss_kg"] / 1000
    return ranking.loc[ranking["masked_loss_kwh"] > 0].sort_values(
        "masked_loss_kwh", ascending=False
    ).reset_index(drop=True)


def snapshot_summary(
    frame: pd.DataFrame, selected_date: pd.Timestamp
) -> dict[str, int]:
    day_frame = frame.loc[frame["DATE"].eq(selected_date)]
    total_loss = day_frame.loc[day_frame["ANOMALY_CLASS"].eq("TOTAL_LOSS")]
    partial_loss = day_frame.loc[
        day_frame["ANOMALY_CLASS"].eq("PARTIAL_LOSS")
    ]
    faulty = day_frame.loc[day_frame["ANOMALY_CLASS"].isin(FAULT_CLASSES)]
    return {
        "affected_inverters": int(faulty["SOURCE_KEY"].nunique()),
        "total_loss_inverters": int(total_loss["SOURCE_KEY"].nunique()),
        "partial_loss_inverters": int(partial_loss["SOURCE_KEY"].nunique()),
        "fault_intervals": int(len(faulty)),
    }
