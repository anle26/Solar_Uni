import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_preprocessing(config_path="config.yaml", data_dir="data/raw", output_path="data/processed/paper/paper_dataset.csv"):
    """
    Run data preprocessing including merging raw files, filtering valid daylight hours, and correcting scaling issues.
    
    Rationale (DC Power Correction):
    During exploratory data analysis, it was discovered that Plant 1's DC Power readings were exactly 
    10 times higher than expected given the inverter specifications and AC Power output, due to a known 
    sensor scaling error. This function corrects this by dividing Plant 1's DC Power by 10 to ensure 
    consistency and valid physical relationships (efficiency) across both plants.
    """
    config = load_config(config_path)
    c_plant = config['plant_settings']
    plant1_id, plant2_id = c_plant['plant1_id'], c_plant['plant2_id']
    dc_correction_factor = c_plant['dc_correction_factor']
    interval_hours = c_plant['interval_hours']
    irr_day_threshold = c_plant['irradiation_day_threshold']
    emission_factor = config['carbon_settings']['emission_factor']

    base = Path(data_dir)
    
    # Load generation
    gen_dfs = []
    for pid, fname in [(plant1_id, "Plant_1_Generation_Data.csv"), (plant2_id, "Plant_2_Generation_Data.csv")]:
        df = pd.read_csv(base / fname)
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="mixed", dayfirst=True)
        gen_dfs.append(df)
    gen = pd.concat(gen_dfs, ignore_index=True)

    # Load weather
    wx_dfs = []
    for pid, fname in [(plant1_id, "Plant_1_Weather_Sensor_Data.csv"), (plant2_id, "Plant_2_Weather_Sensor_Data.csv")]:
        df = pd.read_csv(base / fname)
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="mixed", dayfirst=True)
        df = df.drop(columns=["SOURCE_KEY"], errors="ignore")
        wx_dfs.append(df)
    wx = pd.concat(wx_dfs, ignore_index=True)

    # Merge
    merged = gen.merge(wx, on=["PLANT_ID", "DATE_TIME"], how="left")
    
    # DC correction
    merged["DC_POWER_RAW"] = merged["DC_POWER"].copy()
    mask_p1 = merged["PLANT_ID"] == plant1_id
    merged.loc[mask_p1, "DC_POWER_CORRECTED"] = merged.loc[mask_p1, "DC_POWER"] / dc_correction_factor
    merged.loc[~mask_p1, "DC_POWER_CORRECTED"] = merged.loc[~mask_p1, "DC_POWER"]

    # Efficiency
    valid = merged["DC_POWER_CORRECTED"] > 0
    merged["EFFICIENCY_RAW"] = 0.0
    merged.loc[merged["DC_POWER_RAW"] > 0, "EFFICIENCY_RAW"] = (
        merged.loc[merged["DC_POWER_RAW"] > 0, "AC_POWER"] / merged.loc[merged["DC_POWER_RAW"] > 0, "DC_POWER_RAW"]
    )
    merged["EFFICIENCY_CORRECTED"] = 0.0
    merged.loc[valid, "EFFICIENCY_CORRECTED"] = (
        merged.loc[valid, "AC_POWER"] / merged.loc[valid, "DC_POWER_CORRECTED"]
    )

    # Features
    merged["DATE"] = merged["DATE_TIME"].dt.date
    merged["HOUR"] = merged["DATE_TIME"].dt.hour
    merged["HOUR_SIN"] = np.sin(2 * np.pi * merged["HOUR"] / 24)
    merged["HOUR_COS"] = np.cos(2 * np.pi * merged["HOUR"] / 24)
    merged["PLANT_NAME"] = merged["PLANT_ID"].map({plant1_id: "Plant 1", plant2_id: "Plant 2"})
    merged["IS_DAY"] = merged["IRRADIATION"] > irr_day_threshold
    merged["ENERGY_KWH_INTERVAL"] = merged["AC_POWER"] * interval_hours
    merged["CO2_AVOIDED_KG"] = merged["ENERGY_KWH_INTERVAL"] * emission_factor

    # Peer Avg
    group_cols = ["PLANT_ID", "DATE_TIME"]
    plant_totals = (
        merged.groupby(group_cols)["AC_POWER"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "_plant_total_ac", "count": "_n_inverters"})
        .reset_index()
    )
    merged = merged.merge(plant_totals, on=group_cols, how="left")
    merged["PEER_AVG_AC_POWER"] = np.where(
        merged["_n_inverters"] > 1,
        (merged["_plant_total_ac"] - merged["AC_POWER"]) / (merged["_n_inverters"] - 1),
        0.0,
    )
    merged["POWER_GAP_FROM_PEER"] = merged["PEER_AVG_AC_POWER"] - merged["AC_POWER"]
    merged["POWER_GAP_PERCENT"] = np.where(
        merged["PEER_AVG_AC_POWER"] > 0,
        merged["POWER_GAP_FROM_PEER"] / merged["PEER_AVG_AC_POWER"],
        0.0,
    )
    merged = merged.drop(columns=["_plant_total_ac", "_n_inverters"])

    merged = merged.sort_values(["PLANT_ID", "SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Preprocessing completed. Output saved to {output_path}")

if __name__ == "__main__":
    run_preprocessing()
