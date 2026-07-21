"""
fault_injection.py — Synthetic fault injection for controlled evaluation.

Injects known faults into clean data so that true Precision/Recall/F1
can be computed against verified ground-truth labels.

Usage:
    from src.fault_injection import create_injected_dataset
    df_injected, gt_labels = create_injected_dataset(df_clean, seed=42)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual fault injectors
# ---------------------------------------------------------------------------
def inject_total_loss(
    df: pd.DataFrame,
    inverter: str,
    n_intervals: int = 15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Simulate TOTAL_LOSS: set AC_POWER = 0 for random daytime intervals
    of a specific inverter.

    Returns (modified_df, boolean_mask_of_injected_rows).
    """
    rng = np.random.RandomState(seed)
    df = df.copy()

    # Select daytime rows for this inverter
    mask_inv = (df["SOURCE_KEY"] == inverter) & (df["IS_DAY"] == True) & (df["AC_POWER"] > 0)
    candidates = df.loc[mask_inv].index.tolist()

    if len(candidates) < n_intervals:
        n_intervals = len(candidates)

    # Pick random contiguous blocks (simulate real outages)
    injected_indices = []
    if len(candidates) > 0:
        # Pick random start points and inject blocks of 3-8 consecutive timestamps
        block_starts = rng.choice(
            range(len(candidates) - 8),
            size=min(n_intervals // 5 + 1, len(candidates) // 8),
            replace=False,
        )
        for start in block_starts:
            block_size = rng.randint(3, min(9, len(candidates) - start))
            block = candidates[start : start + block_size]
            injected_indices.extend(block)

    injected_indices = list(set(injected_indices))[:n_intervals]

    # Apply fault
    df.loc[injected_indices, "AC_POWER"] = 0.0
    df.loc[injected_indices, "DC_POWER_RAW"] = 0.0
    df.loc[injected_indices, "DC_POWER_CORRECTED"] = 0.0
    df.loc[injected_indices, "ENERGY_KWH_INTERVAL"] = 0.0

    # Ground truth mask
    gt = pd.Series(False, index=df.index)
    gt.loc[injected_indices] = True

    return df, gt


def inject_partial_loss(
    df: pd.DataFrame,
    inverter: str,
    factor: float = 0.3,
    n_intervals: int = 20,
    seed: int = 43,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Simulate PARTIAL_LOSS: multiply AC_POWER by a degradation factor
    for random daytime intervals.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()

    mask_inv = (df["SOURCE_KEY"] == inverter) & (df["IS_DAY"] == True) & (df["AC_POWER"] > 0)
    candidates = df.loc[mask_inv].index.tolist()

    if len(candidates) < n_intervals:
        n_intervals = len(candidates)

    selected = rng.choice(candidates, size=n_intervals, replace=False).tolist()

    # Apply fault
    df.loc[selected, "AC_POWER"] = df.loc[selected, "AC_POWER"] * factor
    df.loc[selected, "DC_POWER_RAW"] = df.loc[selected, "DC_POWER_RAW"] * factor
    df.loc[selected, "DC_POWER_CORRECTED"] = df.loc[selected, "DC_POWER_CORRECTED"] * factor
    df.loc[selected, "ENERGY_KWH_INTERVAL"] = df.loc[selected, "AC_POWER"] * 0.25

    gt = pd.Series(False, index=df.index)
    gt.loc[selected] = True

    return df, gt


def inject_thermal_degrade(
    df: pd.DataFrame,
    inverter: str,
    degradation_pct: float = 0.20,
    n_intervals: int = 15,
    seed: int = 44,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Simulate THERMAL_DEGRADE: gradually reduce efficiency during high-temperature
    periods for a specific inverter.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()

    # Select high-temperature daytime rows
    mask_inv = (
        (df["SOURCE_KEY"] == inverter)
        & (df["IS_DAY"] == True)
        & (df["AC_POWER"] > 0)
        & (df["MODULE_TEMPERATURE"] > 45)
    )
    candidates = df.loc[mask_inv].index.tolist()

    if len(candidates) < n_intervals:
        n_intervals = len(candidates)

    selected = rng.choice(candidates, size=n_intervals, replace=False).tolist()

    # Apply gradual degradation (each record gets a slightly different factor)
    for i, idx in enumerate(selected):
        deg_factor = 1.0 - degradation_pct * (0.5 + 0.5 * rng.random())
        df.loc[idx, "AC_POWER"] = df.loc[idx, "AC_POWER"] * deg_factor
        df.loc[idx, "ENERGY_KWH_INTERVAL"] = df.loc[idx, "AC_POWER"] * 0.25

    gt = pd.Series(False, index=df.index)
    gt.loc[selected] = True

    return df, gt


# ---------------------------------------------------------------------------
# Combined injection
# ---------------------------------------------------------------------------
def create_injected_dataset(
    df: pd.DataFrame,
    seed: int = 42,
    target_inverters: list[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply a mixture of fault types to create a dataset with known ground truth.

    Parameters
    ----------
    df : clean DataFrame (should exclude already-anomalous records if possible)
    seed : random seed for reproducibility
    target_inverters : list of inverter SOURCE_KEYs to inject faults into.
                       If None, auto-selects 3 inverters per plant.

    Returns
    -------
    df_injected : DataFrame with faults injected
    gt_labels : Series of bool (True = injected fault)
    """
    rng = np.random.RandomState(seed)
    df_out = df.copy()
    gt_combined = pd.Series(False, index=df_out.index)

    if target_inverters is None:
        # Auto-select: 3 inverters per plant
        inv_per_plant = df.groupby("PLANT_ID")["SOURCE_KEY"].unique()
        target_inverters = []
        for plant_id, invs in inv_per_plant.items():
            chosen = rng.choice(invs, size=min(3, len(invs)), replace=False)
            target_inverters.extend(chosen)

    # Distribute fault types across selected inverters
    fault_types = [inject_total_loss, inject_partial_loss, inject_thermal_degrade]

    for i, inv in enumerate(target_inverters):
        fault_fn = fault_types[i % len(fault_types)]
        df_out, gt = fault_fn(df_out, inverter=inv, seed=seed + i)
        gt_combined = gt_combined | gt

    # Fix: Recalculate derived features after AC_POWER modification
    df_out["EFFICIENCY_CORRECTED"] = np.where(
        df_out["DC_POWER_CORRECTED"] > 0,
        df_out["AC_POWER"] / df_out["DC_POWER_CORRECTED"],
        0.0
    )
    df_out["POWER_GAP_FROM_PEER"] = df_out["PEER_AVG_AC_POWER"] - df_out["AC_POWER"]
    df_out["POWER_GAP_PERCENT"] = np.where(
        df_out["PEER_AVG_AC_POWER"] > 0,
        df_out["POWER_GAP_FROM_PEER"] / df_out["PEER_AVG_AC_POWER"],
        0.0
    )

    return df_out, gt_combined
