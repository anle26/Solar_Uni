import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_curve, cohen_kappa_score

from src.fault_injection import create_injected_dataset
from src.taxonomy import apply_taxonomy

VERBOSE = False

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_grid_search(df_clean, ad_features, config):
    print("=== GRID SEARCH (Validation Set: Seeds 1-10) ===")

    val_seeds = range(1, 11)

    # Define search space
    # NOTE: IsolationForest.contamination does NOT affect score_samples() output —
    # it only shifts the decision_function threshold for predict(). Since we use
    # score_samples() + PR curve to find the optimal threshold, the key tunable
    # parameter is n_estimators (number of trees), which governs ensemble diversity.
    # max_samples is fixed at 'auto' (sklearn default = min(256, n_samples));
    # Liu et al. (2008) showed sub-sample size beyond 256 offers negligible gain
    # and can introduce swamping/masking artefacts.
    search_space = {
        'iforest': {
            'n_estimators': [50, 100, 200, 300, 500]
        },
        'ocsvm': {'nu': [0.01, 0.02, 0.03, 0.05, 0.1]},
        'lof': {'n_neighbors': [10, 15, 20, 30, 50]}
    }

    best_params = {}

    # 1. Isolation Forest — tune n_estimators only; max_samples fixed at 'auto'
    print("\n--- Tuning Isolation Forest (max_samples='auto' fixed) ---")
    best_f1 = -1
    best_ne = None
    for ne in search_space['iforest']['n_estimators']:
        f1_list = []
        for seed in val_seeds:
            df_inj, gt = create_injected_dataset(df_clean, seed=seed)
            X = df_inj[ad_features].fillna(0).values
            scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)

            # DEBUG: confirm exact params being passed to constructor
            if VERBOSE: print(f"      [IF] Fitting n_estimators={ne}, max_samples='auto', seed={seed}")
            model = IsolationForest(
                n_estimators=ne, max_samples='auto',
                contamination='auto', random_state=42, n_jobs=1
            )
            model.fit(X_scaled)
            scores = -model.score_samples(X_scaled)

            prec, rec, _ = precision_recall_curve(gt, scores)
            f1_curve = 2 * (prec * rec) / (prec + rec + 1e-9)
            f1 = np.max(f1_curve)
            f1_list.append(f1)
            if VERBOSE: print(f"        -> Seed {seed}: F1={f1:.4f}")
        avg_f1 = np.mean(f1_list)
        if VERBOSE: print(f"  n_estimators={ne}: Avg F1 = {avg_f1:.4f}")
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_ne = ne
    best_params['iforest_n_estimators'] = best_ne
    best_params['iforest_max_samples'] = 'auto'
    print(f"-> Best IF: n_estimators={best_ne}, max_samples='auto' (F1: {best_f1:.4f})")

    # 2. One-Class SVM
    print("\n--- Tuning One-Class SVM ---")
    best_f1 = -1
    best_nu = None
    for nu in search_space['ocsvm']['nu']:
        f1_list = []
        for seed in val_seeds:
            df_inj, gt = create_injected_dataset(df_clean, seed=seed)
            X = df_inj[ad_features].fillna(0).values
            scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)

            rng = np.random.RandomState(seed)
            sub_idx = rng.choice(len(X_scaled), min(5000, len(X_scaled)), replace=False)

            # DEBUG: confirm exact params being passed to constructor
            if VERBOSE: print(f"      [OCSVM] Fitting nu={nu}, seed={seed}")
            model = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
            model.fit(X_scaled[sub_idx])
            scores = -model.score_samples(X_scaled)

            prec, rec, _ = precision_recall_curve(gt, scores)
            f1_curve = 2 * (prec * rec) / (prec + rec + 1e-9)
            f1 = np.max(f1_curve)
            f1_list.append(f1)
            if VERBOSE: print(f"        -> Seed {seed}: F1={f1:.4f}")
        avg_f1 = np.mean(f1_list)
        if VERBOSE: print(f"  nu={nu}: Avg F1 = {avg_f1:.4f}")
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_nu = nu
    best_params['ocsvm_nu'] = best_nu
    print(f"-> Best OCSVM nu: {best_nu} (F1: {best_f1:.4f})")

    # 3. LOF
    print("\n--- Tuning LOF ---")
    best_f1 = -1
    best_n = None
    for n in search_space['lof']['n_neighbors']:
        f1_list = []
        for seed in val_seeds:
            df_inj, gt = create_injected_dataset(df_clean, seed=seed)
            X = df_inj[ad_features].fillna(0).values
            scaler = StandardScaler().fit(X)
            X_scaled = scaler.transform(X)

            rng = np.random.RandomState(seed)
            sub_idx = rng.choice(len(X_scaled), min(5000, len(X_scaled)), replace=False)

            # DEBUG: confirm exact params being passed to constructor
            if VERBOSE: print(f"      [LOF] Fitting n_neighbors={n}, seed={seed}")
            model = LocalOutlierFactor(n_neighbors=n, contamination=0.05, novelty=True)
            model.fit(X_scaled[sub_idx])
            scores = -model.score_samples(X_scaled)

            prec, rec, _ = precision_recall_curve(gt, scores)
            f1_curve = 2 * (prec * rec) / (prec + rec + 1e-9)
            f1 = np.max(f1_curve)
            f1_list.append(f1)
            if VERBOSE: print(f"        -> Seed {seed}: F1={f1:.4f}")
        avg_f1 = np.mean(f1_list)
        if VERBOSE: print(f"  n_neighbors={n}: Avg F1 = {avg_f1:.4f}")
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_n = n
    best_params['lof_n_neighbors'] = best_n
    print(f"-> Best LOF n_neighbors: {best_n} (F1: {best_f1:.4f})")

    return best_params

def evaluate_baselines(df_clean, ad_features, best_params):
    print("\n=== EVALUATION (Test Set: Seeds 11-30) ===")
    test_seeds = range(11, 31)

    results = {'IF': [], 'OCSVM': [], 'LOF': [], 'Rule': []}
    # Raw per-seed records for A9 cross-check
    raw_records = []

    # Store scores/labels for PR curve plots (seed 11)
    pr_data = {}

    ne = best_params['iforest_n_estimators']
    ms = best_params['iforest_max_samples']
    nu = best_params['ocsvm_nu']
    nn = best_params['lof_n_neighbors']
    print(f"\n=== EVALUATION (Test Set: Seeds 11-30) ===")
    print(f"  Using: IF(n_estimators={ne}, max_samples='{ms}'), OCSVM(nu={nu}), LOF(n_neighbors={nn})")

    for i, seed in enumerate(test_seeds):
        if VERBOSE: print(f"\n  --- Seed {seed} ---")
        df_inj, gt = create_injected_dataset(df_clean, seed=seed)
        gt = gt.astype(int).values

        X = df_inj[ad_features].fillna(0).values
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        rng = np.random.RandomState(seed)
        sub_idx = rng.choice(len(X_scaled), min(5000, len(X_scaled)), replace=False)

        # Rule-Based — hard binary labels, Kappa computed directly
        df_tax = apply_taxonomy(df_inj)
        rule_preds = df_tax['ANOMALY_BINARY'].values
        rule_tp = np.sum((rule_preds == 1) & (gt == 1))
        rule_fp = np.sum((rule_preds == 1) & (gt == 0))
        rule_fn = np.sum((rule_preds == 0) & (gt == 1))
        rule_p = rule_tp / (rule_tp + rule_fp + 1e-9)
        rule_r = rule_tp / (rule_tp + rule_fn + 1e-9)
        rule_f1 = 2 * rule_p * rule_r / (rule_p + rule_r + 1e-9)
        rule_kappa = cohen_kappa_score(gt, rule_preds)
        results['Rule'].append((rule_p, rule_r, rule_f1, rule_kappa))
        raw_records.append({'Seed': seed, 'Method': 'Rule', 'Precision': rule_p, 'Recall': rule_r, 'F1': rule_f1, 'Kappa': rule_kappa})
        if VERBOSE: print(f"    [Rule]  P={rule_p:.4f}, R={rule_r:.4f}, F1={rule_f1:.4f}, Kappa={rule_kappa:.4f}")

        # Isolation Forest
        if VERBOSE: print(f"    [IF] Fitting n_estimators={ne}, max_samples='{ms}', seed={seed}")
        model_if = IsolationForest(
            n_estimators=ne, max_samples=ms,
            contamination='auto', random_state=42, n_jobs=1
        )
        model_if.fit(X_scaled)
        scores_if = -model_if.score_samples(X_scaled)
        p, r, t = precision_recall_curve(gt, scores_if)
        f1_curve = 2 * p * r / (p + r + 1e-9)
        best_idx = np.argmax(f1_curve)
        # Kappa at the optimal threshold (t is 1 shorter than p/r — sklearn convention)
        thresh_if = t[min(best_idx, len(t)-1)]
        preds_if = (scores_if >= thresh_if).astype(int)
        kappa_if = cohen_kappa_score(gt, preds_if)
        results['IF'].append((p[best_idx], r[best_idx], f1_curve[best_idx], kappa_if))
        raw_records.append({'Seed': seed, 'Method': 'IF', 'Precision': p[best_idx], 'Recall': r[best_idx], 'F1': f1_curve[best_idx], 'Kappa': kappa_if})
        if VERBOSE: print(f"      -> P={p[best_idx]:.4f}, R={r[best_idx]:.4f}, F1={f1_curve[best_idx]:.4f}, Kappa={kappa_if:.4f}")
        if i == 0: pr_data['IF'] = (gt.copy(), scores_if.copy())

        # One-Class SVM
        if VERBOSE: print(f"    [OCSVM] Fitting nu={nu}, seed={seed}")
        model_oc = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
        model_oc.fit(X_scaled[sub_idx])
        scores_oc = -model_oc.score_samples(X_scaled)
        p, r, t = precision_recall_curve(gt, scores_oc)
        f1_curve = 2 * p * r / (p + r + 1e-9)
        best_idx = np.argmax(f1_curve)
        thresh_oc = t[min(best_idx, len(t)-1)]
        preds_oc = (scores_oc >= thresh_oc).astype(int)
        kappa_oc = cohen_kappa_score(gt, preds_oc)
        results['OCSVM'].append((p[best_idx], r[best_idx], f1_curve[best_idx], kappa_oc))
        raw_records.append({'Seed': seed, 'Method': 'OCSVM', 'Precision': p[best_idx], 'Recall': r[best_idx], 'F1': f1_curve[best_idx], 'Kappa': kappa_oc})
        if VERBOSE: print(f"      -> P={p[best_idx]:.4f}, R={r[best_idx]:.4f}, F1={f1_curve[best_idx]:.4f}, Kappa={kappa_oc:.4f}")
        if i == 0: pr_data['OCSVM'] = (gt.copy(), scores_oc.copy())

        # LOF
        if VERBOSE: print(f"    [LOF] Fitting n_neighbors={nn}, seed={seed}")
        model_lof = LocalOutlierFactor(n_neighbors=nn, contamination=0.05, novelty=True)
        model_lof.fit(X_scaled[sub_idx])
        scores_lof = -model_lof.score_samples(X_scaled)
        p, r, t = precision_recall_curve(gt, scores_lof)
        f1_curve = 2 * p * r / (p + r + 1e-9)
        best_idx = np.argmax(f1_curve)
        thresh_lof = t[min(best_idx, len(t)-1)]
        preds_lof = (scores_lof >= thresh_lof).astype(int)
        kappa_lof = cohen_kappa_score(gt, preds_lof)
        results['LOF'].append((p[best_idx], r[best_idx], f1_curve[best_idx], kappa_lof))
        raw_records.append({'Seed': seed, 'Method': 'LOF', 'Precision': p[best_idx], 'Recall': r[best_idx], 'F1': f1_curve[best_idx], 'Kappa': kappa_lof})
        if VERBOSE: print(f"      -> P={p[best_idx]:.4f}, R={r[best_idx]:.4f}, F1={f1_curve[best_idx]:.4f}, Kappa={kappa_lof:.4f}")
        if i == 0: pr_data['LOF'] = (gt.copy(), scores_lof.copy())

    # Summarize (mean ± std over 20 test seeds)
    summary = []
    for model_name in ['Rule', 'IF', 'OCSVM', 'LOF']:
        metrics = np.array(results[model_name])
        summary.append({
            'Method': model_name,
            'Precision': f"{metrics[:, 0].mean():.2f} +/- {metrics[:, 0].std():.2f}",
            'Recall':    f"{metrics[:, 1].mean():.2f} +/- {metrics[:, 1].std():.2f}",
            'F1':        f"{metrics[:, 2].mean():.2f} +/- {metrics[:, 2].std():.2f}",
            'Kappa':     f"{metrics[:, 3].mean():.2f} +/- {metrics[:, 3].std():.2f}",
        })
    df_summary = pd.DataFrame(summary)
    print("\n=== Final Table 3 Summary (Mean +/- Std, N=20 Test Seeds) ===")
    print(df_summary.to_string(index=False))

    # Save outputs
    import pickle
    os.makedirs('data/processed/paper', exist_ok=True)
    with open('data/processed/paper/pr_curve_data.pkl', 'wb') as f:
        pickle.dump(pr_data, f)
    df_summary.to_csv('data/processed/paper/synthetic_fault_evaluation_tuned.csv', index=False)
    # Raw per-seed for A9 cross-check
    pd.DataFrame(raw_records).to_csv('data/processed/paper/synthetic_fault_evaluation_raw.csv', index=False)
    # Save best_params for figures script
    import json
    with open('data/processed/paper/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print("\nSaved: pr_curve_data.pkl, synthetic_fault_evaluation_tuned.csv, synthetic_fault_evaluation_raw.csv, best_params.json")

    return df_summary, best_params

if __name__ == "__main__":
    config = load_config()
    ad_features = config['features']['anomaly_detection']
    df = pd.read_csv("data/processed/paper/paper_dataset_with_expected.csv")
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    # Only daytime for anomaly detection
    df_day = df[df['IS_DAY'] == True].copy()
    
    # Ensure no pre-existing severe anomalies are used as the "clean" base
    # (Simplified: just drop the single known broken inverter from being the base)
    df_clean_raw = apply_taxonomy(df_day)
    df_clean = df_clean_raw[df_clean_raw['ANOMALY_CLASS'] == 'NORMAL'].copy()
    
    best_params = run_grid_search(df_clean, ad_features, config)
    evaluate_baselines(df_clean, ad_features, best_params)
