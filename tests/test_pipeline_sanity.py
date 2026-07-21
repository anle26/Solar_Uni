import pandas as pd
import numpy as np

# Import functions from actual source
from src.taxonomy import apply_taxonomy
from src.expected_power_model import load_config

def test_masked_loss_zero_when_normal():
    # Create a mock dataframe with NORMAL instances
    # Specifically, we want ANOMALY_BINARY = 0
    data = {
        'AC_POWER': [500.0, 600.0, 0.0],
        'EXPECTED_AC_POWER': [510.0, 610.0, 0.0],
        'ANOMALY_BINARY': [0, 0, 0] # 0 = NORMAL
    }
    df = pd.DataFrame(data)
    INTERVAL_HOURS = 0.25
    
    # 1. Manual calculation (the old way)
    power_loss = df['EXPECTED_AC_POWER'] - df['AC_POWER']
    manual_loss = (power_loss * INTERVAL_HOURS).where(df['ANOMALY_BINARY'] == 1, 0.0)
    manual_loss_sum = manual_loss.sum()
    
    # 2. Module calculation (the new way)
    from src.masked_loss import calculate_masked_loss
    module_loss = calculate_masked_loss(
        y=df['AC_POWER'], 
        y_hat=df['EXPECTED_AC_POWER'], 
        is_anomaly=df['ANOMALY_BINARY'], 
        interval_hours=INTERVAL_HOURS
    )
    module_loss_sum = module_loss.sum()
    
    print(f"    -> Manual Masked Loss for NORMAL records: {manual_loss_sum}")
    print(f"    -> Module Masked Loss for NORMAL records: {module_loss_sum}")
    
    # Assert total loss is exactly 0 and they match
    assert manual_loss_sum == 0.0, "Manual masked loss should be exactly 0 for NORMAL records"
    assert module_loss_sum == 0.0, "Module masked loss should be exactly 0 for NORMAL records"
    pd.testing.assert_series_equal(manual_loss, module_loss, check_names=False)

def test_no_dc_power_leakage():
    # Load feature list actually passed to the model
    config = load_config("config.yaml")
    features = config['features']['regression']
    print(f"    -> Full regression feature list from config.yaml: {features}")
    
    # Check that this matches what is expected for the final model
    expected = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'HOUR_SIN', 'HOUR_COS']
    print(f"    -> Match with expected regression FEATURES: {features == expected}")
    
    # Assert DC_POWER is not in the feature list
    assert 'DC_POWER' not in features, "Target leakage detected: DC_POWER is in the regression features!"
    assert 'DC_POWER_RAW' not in features, "Target leakage detected: DC_POWER_RAW is in the regression features!"
    assert 'DC_POWER_CORRECTED' not in features, "Target leakage detected: DC_POWER_CORRECTED is in the regression features!"

def test_taxonomy_total_count():
    # Load the actual paper dataset
    df = pd.read_csv("data/processed/paper/paper_dataset.csv")
    
    # Apply taxonomy classification
    df_tax = apply_taxonomy(df)
    
    # Count rows for NORMAL, TOTAL_LOSS, PARTIAL_LOSS
    counts = df_tax['ANOMALY_CLASS'].value_counts()
    
    total_normal = counts.get('NORMAL', 0)
    total_loss = counts.get('TOTAL_LOSS', 0)
    partial_loss = counts.get('PARTIAL_LOSS', 0)
    
    # The user asked to assert the sum of NORMAL + TOTAL_LOSS + PARTIAL_LOSS = 136,476
    # Wait, the dataset size is 136,476 total. Are there SENSOR_DRIFT or THERMAL_DEGRADE?
    # In the original pipeline, the total records were 136,476.
    # Let's assert the sum of these three classes exactly matches the user's provided number.
    total_3_classes = total_normal + total_loss + partial_loss
    print(f"    -> Total rows for (NORMAL + TOTAL_LOSS + PARTIAL_LOSS): {total_3_classes}")
    
    assert total_3_classes == 136476, f"Expected 136,476 records across NORMAL, TOTAL_LOSS, PARTIAL_LOSS, but got {total_3_classes}"

def test_figure_axes():
    # Verify the residual plot Y-axis bounds are within reasonable range
    # to prevent the 4 orders of magnitude scaling bug.
    df = pd.read_csv("data/processed/paper/paper_dataset_with_masked_loss.csv")
    plant2 = df[df['PLANT_ID'] == 4136001].copy()
    if 'EXPECTED_AC_POWER' in df.columns:
        plant2['RESIDUAL'] = plant2['AC_POWER'] - plant2['EXPECTED_AC_POWER']
        
        y_min = plant2['RESIDUAL'].min()
        y_max = plant2['RESIDUAL'].max()
        print(f"    -> Plant 2 Residual range: [{y_min:.2f}, {y_max:.2f}] kW")
        
        assert y_min >= -1500, f"Y-axis min {y_min} is too small, likely wrong scale (expected >= -1500)"
    assert y_max <= 1500, f"Y-axis max {y_max} is too large, likely wrong scale (expected <= 1500)"

if __name__ == "__main__":
    print("Running test_masked_loss_zero_when_normal...")
    test_masked_loss_zero_when_normal()
    print("Running test_no_dc_power_leakage...")
    test_no_dc_power_leakage()
    print("Running test_taxonomy_total_count...")
    test_taxonomy_total_count()
    print("Running test_figure_axes...")
    test_figure_axes()
    print("\nAll sanity tests PASSED successfully!")
