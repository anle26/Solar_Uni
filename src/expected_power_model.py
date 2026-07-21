import os
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_expected_power_model(config_path="config.yaml", 
                               input_csv="data/processed/paper/paper_dataset.csv", 
                               output_csv="data/processed/paper/paper_dataset_with_expected.csv"):
    """
    Train an XGBoost model to predict the expected AC Power based on weather conditions.
    
    Rationale (Target Leakage):
    The model strictly uses exogenous weather variables (IRRADIATION, AMBIENT_TEMPERATURE, 
    MODULE_TEMPERATURE) and temporal features (HOUR_SIN, HOUR_COS). It EXCLUDES DC_POWER.
    Including DC_POWER would cause target leakage because DC and AC power are deterministically 
    linked by the inverter efficiency. If an inverter fault occurs, both DC and AC power drop,
    so predicting AC from DC would completely mask the fault (the expected AC power would just 
    follow the faulty DC power).
    """
    config = load_config(config_path)
    features = config['features']['regression']
    target = 'AC_POWER'
    params = config['model_hyperparameters']['xgboost']
    plant1_id = config['plant_settings']['plant1_id']
    plant2_id = config['plant_settings']['plant2_id']

    df = pd.read_csv(input_csv)
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Chronological split
    day_valid = df[
        (df['IRRADIATION'] > 0) &
        (df['AC_POWER'].notna()) &
        (df['DC_POWER_CORRECTED'].notna()) &
        (df['MODULE_TEMPERATURE'].notna())
    ].copy()

    all_dates = sorted(day_valid['DATE'].unique())
    split_idx = int(len(all_dates) * 0.8)
    train_dates = set(all_dates[:split_idx])
    test_dates = set(all_dates[split_idx:])

    exclude_from_train = ['bvBOhCH3iADSZry']

    models = {}
    results_list = []

    for plant_id, plant_name in [(plant1_id, 'Plant 1'), (plant2_id, 'Plant 2')]:
        plant_data = day_valid[day_valid['PLANT_ID'] == plant_id].copy()

        train_mask = (plant_data['DATE'].isin(train_dates)) & (~plant_data['SOURCE_KEY'].isin(exclude_from_train))
        test_mask = plant_data['DATE'].isin(test_dates)

        X_train = plant_data.loc[train_mask, features]
        y_train = plant_data.loc[train_mask, target]
        X_test = plant_data.loc[test_mask, features]
        y_test = plant_data.loc[test_mask, target]

        model = xgb.XGBRegressor(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)

        for split_name, y, y_hat in [('Train', y_train, y_hat_train), ('Test', y_test, y_hat_test)]:
            r2 = r2_score(y, y_hat)
            mae = mean_absolute_error(y, y_hat)
            rmse = np.sqrt(mean_squared_error(y, y_hat))
            mask_nonzero = y > 0
            mape = np.mean(np.abs((y[mask_nonzero] - y_hat[mask_nonzero]) / y[mask_nonzero])) * 100

            results_list.append({
                'Plant': plant_name,
                'Split': split_name,
                'R2': round(r2, 4),
                'MAE_kW': round(mae, 2),
                'RMSE_kW': round(rmse, 2),
                'MAPE_pct': round(mape, 2),
            })
        models[plant_id] = model
        
        # Save model for SHAP/visualization
        os.makedirs('models', exist_ok=True)
        model.save_model(f'models/expected_power_model_{plant_id}.json')

    results_df = pd.DataFrame(results_list)
    print("\n--- XGBoost Regression Results ---")
    print(results_df.to_string(index=False))
    print("----------------------------------\n")
    os.makedirs('data/processed/paper', exist_ok=True)
    results_df.to_csv('data/processed/paper/regression_results.csv', index=False)

    df['EXPECTED_AC_POWER'] = 0.0
    for plant_id in [plant1_id, plant2_id]:
        mask = (df['PLANT_ID'] == plant_id) & (df['IS_DAY'] == True) & (df['IRRADIATION'] > 0)
        X = df.loc[mask, features].fillna(0)
        preds = models[plant_id].predict(X)
        df.loc[mask, 'EXPECTED_AC_POWER'] = np.clip(preds, 0, None)

    # Power loss
    interval_hours = config['plant_settings']['interval_hours']
    emission_factor = config['carbon_settings']['emission_factor']
    
    df['POWER_LOSS_KW'] = np.clip(df['EXPECTED_AC_POWER'] - df['AC_POWER'], 0, None)
    df['ENERGY_LOSS_KWH'] = df['POWER_LOSS_KW'] * interval_hours
    df['CO2_LOSS_KG'] = df['ENERGY_LOSS_KWH'] * emission_factor

    df.to_csv(output_csv, index=False)
    print(f"Regression completed. Output saved to {output_csv}")

if __name__ == "__main__":
    run_expected_power_model()
