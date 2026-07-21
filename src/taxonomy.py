"""
anomaly_taxonomy.py — Multi-class rule-based anomaly classification
for solar PV inverter fault diagnosis.

Each class maps to a specific O&M action, making the taxonomy *actionable*
rather than simply binary (normal / anomaly).

Usage:
    from src.taxonomy import apply_taxonomy
    df = apply_taxonomy(df)   # adds ANOMALY_CLASS column
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Anomaly class definitions
# ---------------------------------------------------------------------------
# Each class is defined by a set of rules applied to daytime records.
# Rules are evaluated in priority order (first match wins).

CLASSES = [
    "TOTAL_LOSS",
    "SENSOR_DRIFT",
    "THERMAL_DEGRADE",
    "PARTIAL_LOSS",
    "NORMAL",
]

CLASS_DESCRIPTIONS = {
    "TOTAL_LOSS":        "AC output is zero despite sufficient irradiation → dispatch crew immediately",
    "PARTIAL_LOSS":      "AC output < 50% of peer average under adequate sunlight → inspect within 24h",
    "THERMAL_DEGRADE":   "Efficiency drops significantly at high module temperature → schedule preventive maintenance",
    "SENSOR_DRIFT":      "Weather sensor readings diverge from expected patterns → calibrate sensors",
    "NORMAL":            "Performance within expected range",
}


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------
def _classify_row(
    ac_power: float,
    dc_power_corrected: float,
    irradiation: float,
    module_temperature: float,
    efficiency_corrected: float,
    peer_avg_ac: float,
    power_gap_pct: float,
    is_day: bool,
    # Thresholds (tunable)
    irr_threshold: float = 0.2,
    partial_loss_ratio: float = 0.5,
    thermal_temp_threshold: float = 50.0,
    thermal_eff_threshold: float = 0.85,
) -> str:
    """
    Classify a single record into an anomaly class.
    Only meaningful for daytime records (IS_DAY = True).
    Night-time records are always NORMAL.
    """
    # Night → always normal
    if not is_day or irradiation <= 0:
        return "NORMAL"

    # --- Priority 1: TOTAL_LOSS ---
    # AC = 0 while there is sufficient sunlight
    if ac_power == 0 and irradiation > irr_threshold:
        return "TOTAL_LOSS"

    # --- Priority 2: SENSOR_DRIFT ---
    # Irradiation is abnormally negative or temperature readings are extreme outliers
    # (simplified heuristic: irradiation < 0 or module_temp < -10 or > 80)
    if irradiation < 0 or module_temperature < -10 or module_temperature > 80:
        return "SENSOR_DRIFT"

    # --- Priority 4: THERMAL_DEGRADE ---
    # High module temperature AND low efficiency
    if (module_temperature > thermal_temp_threshold
            and efficiency_corrected is not None
            and not np.isnan(efficiency_corrected)
            and efficiency_corrected < thermal_eff_threshold
            and efficiency_corrected > 0):
        return "THERMAL_DEGRADE"

    # --- Priority 5: PARTIAL_LOSS ---
    # AC power is less than 50% of peer average under adequate irradiation
    if (peer_avg_ac > 0
            and irradiation > irr_threshold
            and ac_power < partial_loss_ratio * peer_avg_ac
            and ac_power > 0):
        return "PARTIAL_LOSS"

    # --- Default: NORMAL ---
    return "NORMAL"


def apply_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply multi-class anomaly taxonomy to the entire DataFrame.
    Adds column `ANOMALY_CLASS` with one of the defined class labels.
    Also adds `ANOMALY_BINARY` (0/1) for ML evaluation convenience.
    
    Rationale (Actionable Taxonomy):
    Rather than a simple binary (Normal/Anomaly) classification, this taxonomy defines faults based 
    on the required Operations and Maintenance (O&M) actions. For example, TOTAL_LOSS triggers immediate 
    dispatch (AC=0 under sufficient sunlight), while PARTIAL_LOSS triggers inspection within 24h, and 
    THERMAL_DEGRADE triggers preventive maintenance. This bridges the gap between raw ML predictions 
    and practical plant operations.
    """
    df = df.copy()

    # Vectorized classification using apply (safe for mixed types)
    df["ANOMALY_CLASS"] = df.apply(
        lambda row: _classify_row(
            ac_power=row.get("AC_POWER", 0),
            dc_power_corrected=row.get("DC_POWER_CORRECTED", 0),
            irradiation=row.get("IRRADIATION", 0),
            module_temperature=row.get("MODULE_TEMPERATURE", 25),
            efficiency_corrected=row.get("EFFICIENCY_CORRECTED", np.nan),
            peer_avg_ac=row.get("PEER_AVG_AC_POWER", 0),
            power_gap_pct=row.get("POWER_GAP_PERCENT", 0),
            is_day=row.get("IS_DAY", False),
        ),
        axis=1,
    )

    # Binary flag: anything not NORMAL is an actionable anomaly
    df["ANOMALY_BINARY"] = (df["ANOMALY_CLASS"] != "NORMAL").astype(int)
    
    return df

def run_taxonomy(input_csv="data/processed/paper/paper_dataset_with_expected.csv", 
                 output_csv="data/processed/paper/paper_dataset_with_taxonomy.csv"):
    """
    Load dataset, apply the actionable taxonomy, and save to a new CSV.
    """
    import os
    df = pd.read_csv(input_csv)
    df = apply_taxonomy(df)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Taxonomy applied. Output saved to {output_csv}")

if __name__ == "__main__":
    run_taxonomy()


def get_taxonomy_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table of anomaly class distribution.
    Useful for paper Table II.
    """
    if "ANOMALY_CLASS" not in df.columns:
        df = apply_taxonomy(df)

    summary = (
        df["ANOMALY_CLASS"]
        .value_counts()
        .reset_index()
        .rename(columns={"ANOMALY_CLASS": "Class", "count": "Count"})
    )
    summary["Percentage"] = (summary["Count"] / summary["Count"].sum() * 100).round(2)
    summary["O&M Action"] = summary["Class"].map(CLASS_DESCRIPTIONS)

    return summary
