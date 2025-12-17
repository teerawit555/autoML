# Waveform ML: End-to-End Prediction Pipeline

A comprehensive Python pipeline for synthetic waveform generation, feature engineering, and automated machine learning (AutoML) to predict target parameters such as wait times.

## 1. Environment Setup

This project requires **Python 3.11** on Windows. Follow these steps to initialize your local environment using PowerShell:

```powershell
# Verify available Python versions
py -0

# Create a virtual environment using Python 3.11
py -V:3.11 -m venv venv311

# Enable script execution (Required for PowerShell activation)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# Activate the virtual environment
.\venv311\Scripts\Activate.ps1

# Verify the current Python version
python --version

# Install dependencies
pip install -r requirements.txt

## 2. CLI part

# 1. Generate synthetic data (1000 samples)
python scripts\generate_raw_long.py --out data\raw\data1000samples_test.csv --dt_ms 0.01 --t_end_ms 9.99

# 2. Transform Long data to Wide format
    #------ 2.1 Data for training (with labels) ------#
    python scripts/make_wide_csv.py --mode train --in data/raw/data_for_train.csv --out data/processed/train/train_features.csv
    #------ 2.2 Data for inference (without labels) ------#
    python scripts/make_wide_csv.py --mode inference --in data/raw/data_1000_samples_to_pred.csv --out data\processed\inference\wide_1000_samples_to_pred.csv

# 3. Extract features (Peak, Slope, etc.)
python scripts\extract_features.py --in data\processed\inference\wide_1000_demo_x.csv --out data\processed\inference\train_features_1000_x.csv

# 4. Train the AutoML model
python scripts\autoML.py --mode train --data data\processed\train\train_features.csv --label wait_time_ms --time-limit 120

# 5. Run Prediction (Inference)
python scripts\autoML.py --mode predict --model-path AutogluonModels --inference-csv data\processed\inference\train_features_1000_x.csv --out data\processed\prediction\predicted_wait_time_1000_x.csv

# 6. Generate Visualization plots
python scripts\plot_all_waves.py --raw data\raw\data1000samples_test.csv --pred data\processed\prediction\predicted_wait_time_1000_x.csv --out plots\waves\waves_with_pred_wait_x.png