# Renewable Energy Production & Carbon Footprint Optimization

## Project Overview
This project focuses on optimizing renewable energy production and monitoring the carbon footprint for solar power plants. It encompasses an end-to-end data pipeline from raw data ingestion, exploratory data analysis (EDA), anomaly detection (rule-based taxonomy and ML baselines), carbon quantification, all the way to generating reproducible figures for academic reporting.

## Data Availability

The raw dataset is not included in this repository to keep it lightweight. 
The data used is the **"Solar Power Generation Data"** by Ani Kannal, which is publicly available on Kaggle.

### Reproduction Steps
1. Search for the **"Solar Power Generation Data"** dataset on Kaggle (or use a direct search engine query) and download the archive.
2. Extract and place the raw CSV files into the `data/raw/` directory. Ensure the filenames match these exactly:
   - `Plant_1_Generation_Data.csv`
   - `Plant_2_Generation_Data.csv`
   - `Plant_1_Weather_Sensor_Data.csv`
   - `Plant_2_Weather_Sensor_Data.csv`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the full pipeline: `python reproduce_all.py`

This regenerates all datasets, models, metrics, and the 9 paper figures from scratch. The pipeline has been verified for deterministic reproducibility (identical results across independent runs).

## Folder Structure
The repository is organized following industry-standard Data Engineering and Analytics practices, recently refactored to a modular architecture:

```text
.
├── README.md               # This file
├── Data_Dictionary.md      # Detailed description of all datasets
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore file
│
├── docs/                   # Documentation, LaTeX paper, and reference materials
├── src/                    # Core Python modules (preprocessing, taxonomy, baselines, etc.)
├── figures/                # Scripts to generate paper figures
├── tests/                  # Pipeline sanity checks
├── notebooks/              # Jupyter notebooks for EDA (01, 03) and archived scripts
├── power_bi/               # Power BI dashboard templates
```
*(Note: `data/`, `models/`, and `reports/figures/` directories are dynamically generated when running the pipeline).*

## Data & Machine Learning Pipeline
The pipeline is fully automated via `reproduce_all.py`, executing the following modules sequentially from `src/`:
1. **Preprocessing**: Merging and cleaning raw data, computing derived features, and applying DC corrections (`src/preprocessing.py`).
2. **Expected Power Model**: Training an XGBoost regression model to establish an ideal baseline for power generation (`src/expected_power_model.py`).
3. **Taxonomy**: Applying a multi-class rule-based taxonomy for actionable anomaly detection (`src/taxonomy.py`).
4. **Masked Loss**: Calculating carbon and energy loss precisely by masking out acceptable ML variance (`src/masked_loss.py`).
5. **Carbon Quantification**: Generating plant-level summaries of masked energy/carbon loss (`src/carbon_quantification.py`).
6. **Baselines & Evaluation**: Injecting synthetic faults and evaluating unsupervised ML baselines (IF, OCSVM, LOF) against the rule-based taxonomy (`src/baselines.py`, `src/fault_injection.py`).
7. **Figure Generation**: Producing all deterministic figures and visualizations for the final report/paper (`figures/generate_all_figures.py`).

