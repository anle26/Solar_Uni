"""
generate_all_figures.py
=======================
Single source of truth for generating Figures 6, 7, 8 in the paper.
All figures are derived from:
  - data/processed/paper/pr_curve_data.pkl         (PR curve scores, Seed 11)
  - data/processed/paper/synthetic_fault_evaluation_tuned.csv  (Table 3 summary)
  - data/processed/paper/best_params.json           (optimal hyperparameters)

Run:
    python figures/generate_all_figures.py

Outputs to: reports/figures/paper/
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import precision_recall_curve

# ─── Paths ────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join('reports', 'figures', 'paper')
os.makedirs(OUT_DIR, exist_ok=True)

PR_DATA_PATH  = 'data/processed/paper/pr_curve_data.pkl'
SUMMARY_PATH  = 'data/processed/paper/synthetic_fault_evaluation_tuned.csv'
PARAMS_PATH   = 'data/processed/paper/best_params.json'

# ─── Style helpers ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.5,
})

COLORS = {
    'Rule':  '#2ca02c',
    'IF':    '#1f77b4',
    'OCSVM': '#ff7f0e',
    'LOF':   '#9467bd',
}
LABELS = {
    'Rule':  'Rule-Based Taxonomy',
    'IF':    'Isolation Forest',
    'OCSVM': 'One-Class SVM',
    'LOF':   'Local Outlier Factor',
}

def single_col(height_in=3.0):
    return (3.5, height_in)

def dual_col(height_in=3.0):
    return (7.16, height_in)



# ─── Figure 1: DC Correction ──────────────────────────────────────────────────
def plot_dc_correction():
    df = pd.read_csv('data/processed/paper/paper_dataset_with_masked_loss.csv')
    day = df[df['IS_DAY'] == True].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=dual_col(3.5))
    
    # --- Before correction ---
    for pid, color, label in [(4135001, '#1f77b4', 'Plant 1'), (4136001, '#ff7f0e', 'Plant 2')]:
        subset = day[day['PLANT_ID'] == pid].sample(min(2000, len(day[day['PLANT_ID'] == pid])), random_state=42)
        axes[0].scatter(subset['DC_POWER_RAW'], subset['AC_POWER'], s=2, alpha=0.5, c=color, label=label)
    
    axes[0].set_title('Raw DC Power vs AC Power\n(Before Correction)', fontsize=10, fontweight='bold')
    axes[0].set_xlabel('DC Power (kW)')
    axes[0].set_ylabel('AC Power (kW)')
    axes[0].legend()
    
    # --- After correction ---
    for pid, color, label in [(4135001, '#1f77b4', 'Plant 1'), (4136001, '#ff7f0e', 'Plant 2')]:
        subset = day[day['PLANT_ID'] == pid].sample(min(2000, len(day[day['PLANT_ID'] == pid])), random_state=42)
        axes[1].scatter(subset['DC_POWER_CORRECTED'], subset['AC_POWER'], s=2, alpha=0.5, c=color, label=label)
        
    axes[1].set_title('Corrected DC Power vs AC Power\n(Scaling Issue Fixed)', fontsize=10, fontweight='bold')
    axes[1].set_xlabel('DC Power (kW)')
    axes[1].set_ylabel('AC Power (kW)')
    axes[1].legend()
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'fig_01_data_quality_before_after.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ─── Figure 3: Actual vs Predicted ───────────────────────────────────────────
def plot_actual_vs_predicted():
    import xgboost as xgb
    from sklearn.metrics import r2_score
    import yaml
    
    df = pd.read_csv('data/processed/paper/paper_dataset_with_masked_loss.csv')
    
    with open('config.yaml', 'r', encoding='utf-8') as cf:
        config = yaml.safe_load(cf)
    FEATURES = config['features']['regression']
    TARGET = 'AC_POWER'
    
    models = {}
    for pid in [4135001, 4136001]:
        m = xgb.XGBRegressor()
        m.load_model(f'models/expected_power_model_{pid}.json')
        models[pid] = m

    fig, axes = plt.subplots(1, 2, figsize=dual_col(3.5))
    
    day_valid = df[(df['IS_DAY'] == True)].dropna(subset=FEATURES + [TARGET]).copy()
    
    all_dates = sorted(day_valid['DATE'].unique())
    split_idx = int(len(all_dates) * 0.8)
    test_dates = set(all_dates[split_idx:])
    
    for i, (plant_id, plant_name) in enumerate([(4135001, 'Plant 1'), (4136001, 'Plant 2')]):
        plant_data = day_valid[(day_valid['PLANT_ID'] == plant_id) & (day_valid['DATE'].isin(test_dates))]
        X = plant_data[FEATURES]
        y_true = plant_data[TARGET]
        y_pred = models[plant_id].predict(X)
        
        r2 = r2_score(y_true, y_pred)
        
        axes[i].scatter(y_true, y_pred, alpha=0.3, s=2, c='#1f77b4' if i==0 else '#ff7f0e')
        axes[i].plot([0, 1500], [0, 1500], 'k--', lw=1)
        axes[i].set_title(f'{plant_name} (Test $R^2$ = {r2:.4f})', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Actual AC Power (kW)')
        axes[i].set_ylabel('Expected AC Power (kW)')
        axes[i].set_xlim([0, max(y_true.max(), y_pred.max()) * 1.05])
        axes[i].set_ylim([0, max(y_true.max(), y_pred.max()) * 1.05])
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'fig_03_regression_actual_vs_predicted.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ─── Figure 4: SHAP Feature Importance ───────────────────────────────────────
def plot_shap_feature_importance():
    import xgboost as xgb
    import shap
    import yaml
    
    df = pd.read_csv('data/processed/paper/paper_dataset_with_masked_loss.csv')
    
    with open('config.yaml', 'r', encoding='utf-8') as cf:
        config = yaml.safe_load(cf)
    FEATURES = config['features']['regression']
    
    # Load Plant 1 model
    model = xgb.XGBRegressor()
    model.load_model('models/expected_power_model_4135001.json')
    
    day_valid = df[(df['IS_DAY'] == True)].dropna(subset=FEATURES).copy()
    
    all_dates = sorted(day_valid['DATE'].unique())
    split_idx = int(len(all_dates) * 0.8)
    test_dates = set(all_dates[split_idx:])
    
    plant1_test = day_valid[
        (day_valid['PLANT_ID'] == 4135001) &
        (day_valid['DATE'].isin(test_dates))
    ][FEATURES]
    
    shap_sample = plant1_test.sample(min(1000, len(plant1_test)), random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample)
    
    # Print the ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking_idx = np.argsort(mean_abs_shap)[::-1]
    ranked_features = [FEATURES[i] for i in ranking_idx]
    
    print('\n--- SHAP Feature Ranking (Figure 4) ---')
    print(' > '.join(ranked_features))
    print('---------------------------------------')
    
    fig, ax = plt.subplots(figsize=single_col(3.0))
    shap.summary_plot(shap_values, shap_sample, plot_type='bar', show=False,
                      feature_names=FEATURES)
    ax.set_xlabel('mean(|SHAP value|)', fontsize=9)
    plt.title('Feature Importance (Mean |SHAP|)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'fig_04_shap_feature_importance.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
# ─── Figure 6: Precision–Recall Curves (Illustrative Seed 11) ─────────────────
def plot_pr_curves():
    with open(PR_DATA_PATH, 'rb') as f:
        pr_data = pickle.load(f)

    fig, ax = plt.subplots(figsize=dual_col(3.0))

    for model_name in ['IF', 'OCSVM', 'LOF']:
        gt, scores = pr_data[model_name]
        prec, rec, _ = precision_recall_curve(gt, scores)
        ax.plot(rec, prec,
                color=COLORS[model_name],
                label=LABELS[model_name],
                linewidth=1.5)

    # Rule-based: single point — note aggregate mean Recall, not seed-11 specific
    # Rule-based produces hard binary labels (no continuous score/threshold),
    # so only the aggregate mean Recall across all 20 seeds is meaningful here.
    summary = pd.read_csv(SUMMARY_PATH)
    rule_row = summary[summary['Method'] == 'Rule'].iloc[0]
    rule_r = float(rule_row['Recall'].split()[0])
    ax.scatter([rule_r], [1.0],
               color=COLORS['Rule'], zorder=5, s=60,
               label=LABELS['Rule'] + ' (aggregate mean, N=20 seeds)')

    # Baseline (random)
    ax.axhline(y=0.05, color='grey', linestyle='--', linewidth=1.0, label='No-skill baseline')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(OUT_DIR, 'fig_06_pr_curves.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ─── Figure 8: Baseline Comparison Bar Chart (Table 3 — Test Set Mean) ────────
def plot_baseline_comparison():
    summary = pd.read_csv(SUMMARY_PATH)
    params  = json.load(open(PARAMS_PATH))

    model_order = ['Rule', 'IF', 'OCSVM', 'LOF']
    display_labels = [LABELS[m] for m in model_order]

    metrics = {}
    stds    = {}
    for m in model_order:
        row = summary[summary['Method'] == m].iloc[0]
        metrics[m] = {
            'Precision': float(row['Precision'].split()[0]),
            'Recall':    float(row['Recall'].split()[0]),
            'F1':        float(row['F1'].split()[0]),
        }
        stds[m] = {
            'Precision': float(row['Precision'].split()[-1]),
            'Recall':    float(row['Recall'].split()[-1]),
            'F1':        float(row['F1'].split()[-1]),
        }

    metric_names = ['Precision', 'Recall', 'F1']
    x = np.arange(len(model_order))
    width = 0.25

    fig, ax = plt.subplots(figsize=dual_col(3.5))

    for i, metric in enumerate(metric_names):
        vals  = [metrics[m][metric] for m in model_order]
        errs  = [stds[m][metric]   for m in model_order]
        bars  = ax.bar(x + i * width, vals, width,
                       yerr=errs, capsize=3,
                       label=metric,
                       error_kw={'linewidth': 1.0})
        # Annotate values above the error bar cap, not the bar top
        for bar, v, e in zip(bars, vals, errs):
            y_pos = v + e + 0.025  # clear the error bar cap
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(display_labels, rotation=12, ha='right', fontsize=8)
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1.15])

    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    out_path = os.path.join(OUT_DIR, 'fig_08_baseline_comparison.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ─── Figure 2: Residual Plot for Plant 2 ─────────────────────────────────────
def plot_residual_plant2():
    try:
        import pandas as pd
        df = pd.read_csv("data/processed/paper/paper_dataset_with_masked_loss.csv")
        if 'EXPECTED_AC_POWER' in df.columns and 'ANOMALY_CLASS' in df.columns:
            df['ANOMALY_BINARY'] = (df['ANOMALY_CLASS'] != 'NORMAL').astype(int)
            plant2 = df[df['PLANT_ID'] == 4136001].copy()
            plant2['DATE_TIME'] = pd.to_datetime(plant2['DATE_TIME'])
            plant2 = plant2.sort_values('DATE_TIME')
            plant2['RESIDUAL'] = plant2['AC_POWER'] - plant2['EXPECTED_AC_POWER']

            fig, ax = plt.subplots(figsize=dual_col(4.0))
            ax.scatter(plant2['DATE_TIME'], plant2['RESIDUAL'], s=2, alpha=0.5, label='Residual (Actual - Expected)', color='gray')
            # Highlight anomalies
            anomalies = plant2[plant2['ANOMALY_BINARY'] == 1]
            ax.scatter(anomalies['DATE_TIME'], anomalies['RESIDUAL'], s=10, alpha=0.8, color='red', label='Detected Anomalies')
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.set_ylabel('Residual AC Power (kW)')
            ax.set_xlabel('Date')
            ax.set_title('Plant 2: Systematic Degradation Residuals over Time', fontweight='bold')
            ax.legend(loc='lower right')
            plt.tight_layout()
            out_path = os.path.join(OUT_DIR, 'fig_residual_plant2.png')
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved: {out_path}')
    except Exception as e:
        print(f"Error plotting residual plant 2: {e}")



# ─── Figure 5: Anomaly Class Distribution ────────────────────────────────────
def plot_anomaly_class_distribution():
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    df = pd.read_csv('data/processed/paper/paper_dataset_with_masked_loss.csv')
    
    # Only daytime
    day = df[df['IS_DAY'] == True].copy()
    day_classes = day['ANOMALY_CLASS'].value_counts()
    
    print('\n--- Figure 5 Check: Anomaly Class Counts (Daytime) ---')
    print(day_classes)
    print('------------------------------------------------------')
    
    ANOMALY_COLORS = {
        'NORMAL': '#999999',
        'TOTAL_LOSS': '#d62728',
        'PARTIAL_LOSS': '#ff7f0e'
    }
    
    fig, ax = plt.subplots(figsize=single_col(3.5))
    colors = [ANOMALY_COLORS.get(cls, '#999999') for cls in day_classes.index]
    bars = ax.barh(day_classes.index, day_classes.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Number of Records')
    ax.set_title('Anomaly Class Distribution (Daytime)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()  # Largest on top
    
    for bar in bars:
        ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2,
                f'{int(bar.get_width()):,}', 
                va='center', fontsize=9)
        
    ax.set_xlim(0, max(day_classes.values) * 1.35)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'fig_06_anomaly_class_distribution.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')

# ─── Figure 8: Energy Loss by Class ──────────────────────────────────────────
def plot_energy_loss_by_class():
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    df = pd.read_csv('data/processed/paper/paper_dataset_with_masked_loss.csv')
    
    AD_FEATURES = [
        'AC_POWER', 'DC_POWER_CORRECTED', 'EFFICIENCY_CORRECTED',
        'IRRADIATION', 'MODULE_TEMPERATURE', 'HOUR_SIN', 'HOUR_COS',
        'POWER_GAP_PERCENT'
    ]
    day = df[df['IS_DAY'] == True].dropna(subset=AD_FEATURES).copy()
    
    class_loss = day[day['ANOMALY_BINARY'] == 1].groupby(['PLANT_NAME', 'ANOMALY_CLASS'])['TRUE_ENERGY_LOSS_KWH'].sum().reset_index()
    
    print('\n--- Figure 8 Check: Energy Loss by Plant & Class ---')
    plant_totals = class_loss.groupby('PLANT_NAME')['TRUE_ENERGY_LOSS_KWH'].sum()
    for plant, val in plant_totals.items():
        print(f'  {plant} Total: {val:,.2f} kWh')
    print(class_loss)
    print('----------------------------------------------------')
    
    fig, ax = plt.subplots(figsize=dual_col(3.5))
    
    pivot = class_loss.pivot_table(index='PLANT_NAME', columns='ANOMALY_CLASS', values='TRUE_ENERGY_LOSS_KWH', fill_value=0)
    
    ANOMALY_COLORS = {
        'TOTAL_LOSS': '#d62728',
        'PARTIAL_LOSS': '#ff7f0e'
    }
    
    if not pivot.empty:
        col_order = pivot.sum().sort_values(ascending=False).index
        pivot = pivot[col_order]
        
        bottom = np.zeros(len(pivot))
        for col in pivot.columns:
            ax.bar(pivot.index, pivot[col], label=col, color=ANOMALY_COLORS.get(col, '#999999'), bottom=bottom, width=0.4)
            bottom += pivot[col]
            
        for i, plant in enumerate(pivot.index):
            ax.text(i, bottom[i] + (bottom.max() * 0.02), f'{bottom[i]:,.0f} kWh', ha='center', fontweight='bold')
            
        ax.set_ylabel('Energy Loss (kWh)')
        ax.set_title('Energy Loss by Anomaly Class and Plant', fontsize=12, fontweight='bold')
        ax.legend(title='Anomaly Class')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, bottom.max() * 1.15])
        
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'fig_14_energy_loss_by_class.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')



# ─── Figure 9: Case Study Timeline ───────────────────────────────────────────
def plot_case_study_timeline():
    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    df = pd.read_csv('data/processed/paper/paper_dataset_with_masked_loss.csv')
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    
    AD_FEATURES = [
        'AC_POWER', 'DC_POWER_CORRECTED', 'EFFICIENCY_CORRECTED',
        'IRRADIATION', 'MODULE_TEMPERATURE', 'HOUR_SIN', 'HOUR_COS',
        'POWER_GAP_PERCENT'
    ]
    
    day = df[df['IS_DAY'] == True].dropna(subset=AD_FEATURES).copy()
    
    TARGET_INVERTER = 'bvBOhCH3iADSZry'
    inv1 = day[day['SOURCE_KEY'] == TARGET_INVERTER].sort_values('DATE_TIME')
    
    # Verification Info
    anomalous = inv1[inv1['ANOMALY_BINARY'] == 1]
    worst_idx = anomalous['POWER_LOSS_KW'].idxmax()
    worst_ts = inv1.loc[worst_idx, 'DATE_TIME']
    
    start_dt = inv1['DATE_TIME'].min().strftime('%m/%d/%Y')
    end_dt = inv1['DATE_TIME'].max().strftime('%m/%d/%Y')
    
    print('\n--- Figure 9 Check: Case Study Timeline ---')
    print(f'Target Inverter: {TARGET_INVERTER}')
    print(f'Worst Anomaly Timestamp: {worst_ts}')
    print(f'Timeline Range: {start_dt} - {end_dt}')
    print('-------------------------------------------')
    
    # --- Timeline Plot ---
    fig, axes = plt.subplots(2, 1, figsize=dual_col(5.0), gridspec_kw={'height_ratios': [3, 1]})
    
    ax = axes[0]
    ax.plot(inv1['DATE_TIME'], inv1['AC_POWER'], color='#1f77b4',
            linewidth=0.8, alpha=0.8, label='Actual AC Power')
    if 'EXPECTED_AC_POWER' in inv1.columns:
        ax.plot(inv1['DATE_TIME'], inv1['EXPECTED_AC_POWER'], color='#2ca02c',
                linewidth=0.8, alpha=0.6, linestyle='--', label='Expected AC Power')
                
    ANOMALY_COLORS = {
        'TOTAL_LOSS': '#d62728',
        'PARTIAL_LOSS': '#ff7f0e',
        'THERMAL_DEGRADE': '#8c564b'
    }
    
    for cls in ['TOTAL_LOSS', 'PARTIAL_LOSS', 'THERMAL_DEGRADE']:
        mask = inv1['ANOMALY_CLASS'] == cls
        if mask.any():
            ax.scatter(inv1.loc[mask, 'DATE_TIME'], inv1.loc[mask, 'AC_POWER'],
                      s=8, color=ANOMALY_COLORS.get(cls, 'red'), alpha=0.7, label=cls, zorder=5)
                      
    ax.set_ylabel('AC Power (kW)')
    ax.set_title(f'Plant 1 - Inverter {TARGET_INVERTER}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    ax2 = axes[1]
    if 'POWER_LOSS_KW' in inv1.columns:
        ax2.fill_between(inv1['DATE_TIME'], inv1['POWER_LOSS_KW'],
                         color='#ff7f0e', alpha=0.4)
        ax2.set_ylabel('Power Loss (kW)')
    ax2.set_xlabel('Date')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    plt.tight_layout()
    out_path_tl = os.path.join(OUT_DIR, 'fig_11_case1_timeline.png')
    fig.savefig(out_path_tl, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path_tl}')


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Generating Paper Figures ===\n')
    plot_dc_correction()
    plot_actual_vs_predicted()
    plot_shap_feature_importance()
    plot_anomaly_class_distribution()
    plot_pr_curves()
    plot_baseline_comparison()
    plot_energy_loss_by_class()
    plot_case_study_timeline()
    plot_residual_plant2()
    print('\nAll figures generated successfully.')
    print(f'Output directory: {os.path.abspath(OUT_DIR)}')
