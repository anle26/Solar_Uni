import pandas as pd
import numpy as np

def calculate_masked_loss(y: pd.Series, y_hat: pd.Series, is_anomaly: pd.Series, interval_hours: float = 0.25) -> pd.Series:
    """
    Calculate the masked energy loss to prevent positive drift bias.
    
    Rationale:
    Standard ML models have natural prediction variance (noise/positive drift) even during NORMAL periods.
    If we aggregate this naive raw residual (y_hat - y) over time, it artificially inflates the reported 
    energy loss, resulting in overestimation. By masking the loss calculation and strictly accumulating it 
    only when an actionable fault occurs (is_anomaly == 1), we remove normal prediction noise and isolate 
    the true systemic loss.
    
    Parameters:
    - y: Actual power output
    - y_hat: Expected power output predicted by the ML model
    - is_anomaly: Binary flag (1 if an anomaly is detected, 0 if NORMAL)
    - interval_hours: Duration between samples in hours (e.g., 0.25 for 15-minute intervals)
    
    Returns:
    - pd.Series representing the true energy loss in kWh.
    """
    # Calculate power loss (expected - actual), clipping negative values (we don't count over-performance as loss)
    power_loss = np.clip(y_hat - y, 0, None)
    
    # Convert power loss to energy loss for the given interval
    energy_loss = power_loss * interval_hours
    
    # Masking: only accumulate loss if an anomaly is actually detected
    masked_energy_loss = np.where(is_anomaly == 1, energy_loss, 0.0)
    
    return pd.Series(masked_energy_loss, index=y.index)

def run_masked_loss(input_csv="data/processed/paper/paper_dataset_with_taxonomy.csv", 
                    output_csv="data/processed/paper/paper_dataset_with_masked_loss.csv",
                    interval_hours: float = 0.25):
    """
    Run the masked loss aggregation on the full dataset, print summaries, and save.
    """
    import os
    df = pd.read_csv(input_csv)
    
    # Check if necessary columns exist
    if not all(c in df.columns for c in ['AC_POWER', 'EXPECTED_AC_POWER', 'ANOMALY_BINARY', 'ENERGY_LOSS_KWH']):
        raise ValueError("Missing required columns in dataset.")
        
    # Calculate masked loss
    df['TRUE_ENERGY_LOSS_KWH'] = calculate_masked_loss(
        y=df['AC_POWER'],
        y_hat=df['EXPECTED_AC_POWER'],
        is_anomaly=df['ANOMALY_BINARY'],
        interval_hours=interval_hours
    )
    
    # Calculate masked CO2 loss based on ratio (if needed) or recalculate
    # For now, just focus on ENERGY
    # We can also mask CO2 loss directly
    df['TRUE_CO2_LOSS_KG'] = np.where(df['ANOMALY_BINARY'] == 1, df['CO2_LOSS_KG'], 0.0)
    
    raw_loss = df['ENERGY_LOSS_KWH'].sum()
    masked_loss = df['TRUE_ENERGY_LOSS_KWH'].sum()
    
    print(f"\n--- Masked Loss Aggregation ---")
    print(f"Total Raw Loss (Unmasked): {raw_loss:,.2f} kWh")
    print(f"Total Masked Loss (Systemic): {masked_loss:,.2f} kWh")
    print("\nBreakdown of Masked Loss by Plant:")
    
    breakdown = df.groupby('PLANT_NAME')['TRUE_ENERGY_LOSS_KWH'].sum()
    for plant, loss in breakdown.items():
        print(f"  - {plant}: {loss:,.2f} kWh")
    print(f"-------------------------------\n")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Masked loss applied. Output saved to {output_csv}")

if __name__ == "__main__":
    run_masked_loss()
