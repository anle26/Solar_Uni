# Reproduction Pipeline

To reproduce all datasets, models, metrics, and figures from scratch, there is now **a single unified entrypoint**:

```bash
python reproduce_all.py
```

This script will sequentially execute the entire pipeline in the correct order:
1. `src/preprocessing.py`
2. `src/expected_power_model.py`
3. `src/taxonomy.py`
4. `src/masked_loss.py`
5. `src/carbon_quantification.py`
6. `src/baselines.py` (which internally triggers `fault_injection.py`)
7. `figures/generate_all_figures.py`

**Note:** You no longer need to manually run Jupyter notebooks. All logic has been migrated and refactored into the `src/` and `figures/` modules.
