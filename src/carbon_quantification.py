import pandas as pd
import numpy as np
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_carbon_quantification(input_csv="data/processed/paper/paper_dataset_with_masked_loss.csv", config_path="config.yaml"):
    """
    Calculate final CO2 emissions loss and generate Table 4 summaries.
    Critically, the loss rate MUST be calculated as Masked Loss / Total ACTUAL Energy,
    not Expected Energy, to properly reflect the percentage of generated energy lost.
    """
    config = load_config(config_path)
    emission_factor = config['carbon_settings']['emission_factor']
    
    df = pd.read_csv(input_csv)
    
    print("\n=== Carbon Quantification & Summary (Table 4) ===")
    
    # Check if necessary columns exist
    for col in ['PLANT_NAME', 'TRUE_ENERGY_LOSS_KWH', 'ENERGY_KWH_INTERVAL']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    # Calculate CO2 Loss
    df['TRUE_CO2_LOSS_KG'] = df['TRUE_ENERGY_LOSS_KWH'] * emission_factor
    
    # Calculate totals
    total_energy_loss = df['TRUE_ENERGY_LOSS_KWH'].sum()
    total_co2_loss = df['TRUE_CO2_LOSS_KG'].sum()
    # --- Original Filter (IS_DAY == True & dropna AD_FEATURES) ---
    AD_FEATURES = [
        'AC_POWER', 'DC_POWER_CORRECTED', 'EFFICIENCY_CORRECTED',
        'IRRADIATION', 'MODULE_TEMPERATURE', 'HOUR_SIN', 'HOUR_COS',
        'POWER_GAP_PERCENT'
    ]
    day_valid = df[df['IS_DAY'] == True].dropna(subset=AD_FEATURES).copy()
    
    # Sum Masked Loss (energy loss where ANOMALY_CLASS is not NORMAL)
    plant_groups = day_valid.groupby('PLANT_NAME')
    
    summary = plant_groups.agg(
        Total_Actual_Energy_kWh=('ENERGY_KWH_INTERVAL', 'sum'),
        Masked_Energy_Loss_kWh=('TRUE_ENERGY_LOSS_KWH', 'sum'),
        Masked_CO2_Loss_kg=('TRUE_CO2_LOSS_KG', 'sum')
    ).reset_index()
    
    summary['Loss_Rate_Pct'] = (summary['Masked_Energy_Loss_kWh'] / summary['Total_Actual_Energy_kWh']) * 100
    
    print("\n=== Carbon Quantification & Summary (Table 4) ===")
    print("\nBreakdown by Plant (Matches original Table 4):")
    for _, row in summary.iterrows():
        print(f"  {row['PLANT_NAME']}:")
        print(f"    - Total Actual Energy : {row['Total_Actual_Energy_kWh']:,.2f} kWh")
        print(f"    - Masked Energy Loss  : {row['Masked_Energy_Loss_kWh']:,.2f} kWh")
        print(f"    - Masked CO2 Loss     : {row['Masked_CO2_Loss_kg']:,.2f} kg")
        print(f"    - Loss Rate (%)       : {row['Loss_Rate_Pct']:.2f}%")
        
    print("=================================================\n")

if __name__ == "__main__":
    run_carbon_quantification()
