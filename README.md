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
    #------ 1.1 Data for train ------#
    python scripts\generate_train_sample.py --out data\raw\data_for_train.csv --dt_ms 0.01 --t_end_ms 9.99 --n_waves 100
    #------ 1.2 Data for ,pred ------#
    python scripts\generate_predict_sample.py --out data\raw\data1000samples_test.csv --dt_ms 0.01 --t_end_ms 9.99 --n_waves 500

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
python scripts\autoML.py --mode predict --model-path AutogluonModels/ag-20260105_141110 --inference-csv data\processed\inference\wide_v13.csv --out data\processed\prediction\predicted_wait_time.csv

# 6. Generate Visualization plots
python scripts\plot_all_waves.py --raw data\raw\data1000samples_test.csv --pred data\processed\prediction\predicted_wait_time.csv --out plots\waves\pred_no_x_waves_with_pred_wait_x.png

python scripts/plot_all_waves.py --raw data/raw/data1000samples_test_500.csv --pred data/processed/prediction/predicted_wait_time_10000_500wave.csv --actual data/processed/train/train_with_predictions_20251218_143759.csv --out plots/waves/test_no4_fix/final_comparison.png

python scripts/plot_raw_compare.py --raw_train data/raw/data_for_train.csv --raw_test data/raw/data1000samples_test.csv --num_samples 5 --out plots/check_raw_signals.png

python scripts/plot_raw_compare.py --raw_train data/raw/data_for_train.csv --raw_test data/raw/data1000samples_test.csv --limit 100 --out_prefix plots/my_test_results/compare_signal


# Debugging plot wave
python scripts/plot_all_waves.py --mode check_train --raw data/raw/data_for_train.csv --result data/processed/train/train_with_predictions_20251226_143500.csv --out plots/check_train/check.png

python scripts/plot_all_waves.py --mode check_pred --raw data\raw\data_for_train_new_ver.csv --result data/processed/prediction/predicted_wait_time.csv --out plots/check_pred/check.png


PS C:\Users\TPongkun\Documents\autoML> python scripts/make_wide_csv.py --mode train --in data/raw/data_for_train_new_ver.csv --out data/processed/train/train_features_v13.csv                    
Processing Train Data: data/raw/data_for_train_new_ver.csv
C:\Users\TPongkun\Documents\autoML\scripts\make_wide_csv.py:399: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  train_features = df_raw.groupby(args.id_col, group_keys=False).apply(extract_features_and_label).reset_index(drop=True)
 Training features with labels saved: data/processed/train/train_features_v13.csv
PS C:\Users\TPongkun\Documents\autoML> python scripts\autoML.py --mode train --data data\processed\train\train_features_v13.csv --label wait_time_ms --time-limit 120                             
Dropping meta columns: ['wave_id']          
🚀 Training device: CPU
Verbosity: 2 (Standard Logging)
=================== System Info ===================
AutoGluon Version:  1.4.0
Python Version:     3.11.9
Operating System:   Windows
Platform Machine:   AMD64
Platform Version:   10.0.22631
CPU Count:          8
Memory Avail:       2.89 GB / 15.72 GB (18.4%)
Disk Space Avail:   141.64 GB / 475.02 GB (29.8%)
===================================================
Presets specified: ['medium_quality']
Using hyperparameters preset: hyperparameters='default'
Beginning AutoGluon training ... Time limit = 120s
AutoGluon will save models to "C:\Users\TPongkun\Documents\autoML\AutogluonModels\ag-20260105_154843"
Train Data Rows:    100
Train Data Columns: 10
Label Column:       wait_time_ms
Problem Type:       regression
Preprocessing data ...
Using Feature Generators to preprocess the data ...
Fitting AutoMLPipelineFeatureGenerator...
        Available Memory:                    2972.31 MB
        Train Data (Original)  Memory Usage: 0.01 MB (0.0% of available memory)
        Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
        Stage 1 Generators:
                Fitting AsTypeFeatureGenerator...
        Stage 2 Generators:
                Fitting FillNaFeatureGenerator...
        Stage 3 Generators:
                Fitting IdentityFeatureGenerator...
        Stage 4 Generators:
                Fitting DropUniqueFeatureGenerator...
        Stage 5 Generators:
                Fitting DropDuplicatesFeatureGenerator...
        Types of features in original data (raw dtype, special dtypes):
                ('float', []) : 10 | ['x_end', 'std_all', 'ringing_energy', 'max_rolling_std_half', 'std_tail_50', ...]
        Types of features in processed data (raw dtype, special dtypes):
                ('float', []) : 10 | ['x_end', 'std_all', 'ringing_energy', 'max_rolling_std_half', 'std_tail_50', ...]
        0.1s = Fit runtime
        10 features in original data used to generate 10 features in processed data.
        Train Data (Processed) Memory Usage: 0.01 MB (0.0% of available memory)
Data preprocessing and feature engineering runtime = 0.1s ...
AutoGluon will gauge predictive performance using evaluation metric: 'mean_absolute_error'
        This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
        To change this, specify the eval_metric parameter of Predictor()
Automatically generating train/validation split with holdout_frac=0.2, Train Rows: 80, Val Rows: 20
User-specified model hyperparameters to be fit:
{
        'NN_TORCH': [{}],
        'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, {}, {'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3, 'ag_args': {'name_suffix': 'Large', 'priority': 0, 'hyperparameter_tune_kwargs': None}}],
        'CAT': [{}],
        'XGB': [{}],
        'FASTAI': [{}],
        'RF': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
        'XT': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
}
Fitting 9 L1 models, fit_strategy="sequential" ...
Fitting model: LightGBMXT ... Training model for up to 119.90s of the 119.90s of remaining time.
        Fitting with cpus=4, gpus=0, mem=0.0/2.9 GB
        -1.2084  = Validation score   (-mean_absolute_error)
        2.57s    = Training   runtime
        0.0s     = Validation runtime
Fitting model: LightGBM ... Training model for up to 117.31s of the 117.31s of remaining time.
        Fitting with cpus=4, gpus=0, mem=0.0/2.9 GB
        -1.152   = Validation score   (-mean_absolute_error)
        0.52s    = Training   runtime
        0.0s     = Validation runtime
Fitting model: RandomForestMSE ... Training model for up to 116.77s of the 116.77s of remaining time.
        Fitting with cpus=8, gpus=0
        -1.0188  = Validation score   (-mean_absolute_error)
        0.66s    = Training   runtime
        0.05s    = Validation runtime
Fitting model: CatBoost ... Training model for up to 116.04s of the 116.03s of remaining time.
        Fitting with cpus=4, gpus=0
        -1.0455  = Validation score   (-mean_absolute_error)
        0.71s    = Training   runtime
        0.0s     = Validation runtime
Fitting model: ExtraTreesMSE ... Training model for up to 115.32s of the 115.31s of remaining time.
        Fitting with cpus=8, gpus=0
        -1.1755  = Validation score   (-mean_absolute_error)
        0.61s    = Training   runtime
        0.04s    = Validation runtime
Fitting model: NeuralNetFastAI ... Training model for up to 114.62s of the 114.61s of remaining time.
        Fitting with cpus=4, gpus=0, mem=0.0/2.9 GB
        -1.0943  = Validation score   (-mean_absolute_error)
        2.12s    = Training   runtime
        0.01s    = Validation runtime
Fitting model: XGBoost ... Training model for up to 112.47s of the 112.47s of remaining time.
        Fitting with cpus=4, gpus=0
        -1.0464  = Validation score   (-mean_absolute_error)
        0.49s    = Training   runtime
        0.0s     = Validation runtime
Fitting model: NeuralNetTorch ... Training model for up to 111.97s of the 111.96s of remaining time.
        Fitting with cpus=4, gpus=0, mem=0.0/2.8 GB
C:\Users\TPongkun\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\compose\_column_transformer.py:975: FutureWarning: The parameter `force_int_remainder_cols` is deprecated and will be removed in 1.9. It has no effect. Leave it to its default value to avoid this warning.
  warnings.warn(
        -0.8882  = Validation score   (-mean_absolute_error)
        5.0s     = Training   runtime
        0.02s    = Validation runtime
Fitting model: LightGBMLarge ... Training model for up to 106.94s of the 106.94s of remaining time.
        Fitting with cpus=4, gpus=0, mem=0.0/2.7 GB
        -1.0586  = Validation score   (-mean_absolute_error)
        0.56s    = Training   runtime
        0.0s     = Validation runtime
Fitting model: WeightedEnsemble_L2 ... Training model for up to 119.90s of the 106.19s of remaining time.
        Ensemble Weights: {'NeuralNetTorch': 0.75, 'CatBoost': 0.25}
        -0.8693  = Validation score   (-mean_absolute_error)
        0.08s    = Training   runtime
        0.0s     = Validation runtime
AutoGluon training complete, total runtime = 13.95s ... Best model: WeightedEnsemble_L2 | Estimated inference throughput: 1219.0 rows/s (20 batch size)
TabularPredictor saved. To load, use: predictor = TabularPredictor.load("C:\Users\TPongkun\Documents\autoML\AutogluonModels\ag-20260105_154843")

============================================================
🔍 DEEP MODEL ANALYSIS & DIAGNOSIS
============================================================

[1] Calculating Feature Importance...
Computing feature importance via permutation shuffling for 10 features using 100 rows with 5 shuffle sets...
        3.01s   = Expected runtime (0.6s per shuffle set)
        0.22s   = Actual runtime (Completed 5 of 5 shuffle sets)
                      importance    stddev   p_value  n  p99_high   p99_low
drift_score             0.434798  0.064615  0.000057  5  0.567842  0.301754
crossing_rate           0.393560  0.091193  0.000322  5  0.581328  0.205792
max_rolling_std_half    0.302568  0.096976  0.001110  5  0.502242  0.102893
max_dev_tail_50         0.260488  0.061418  0.000345  5  0.386948  0.134027
std_tail_50             0.247323  0.084867  0.001431  5  0.422065  0.072582
std_all                 0.238858  0.038263  0.000076  5  0.317642  0.160075
mid_to_tail_ratio       0.191559  0.039813  0.000212  5  0.273534  0.109583
x_end                   0.184867  0.048158  0.000506  5  0.284026  0.085709
ringing_energy          0.174185  0.024133  0.000043  5  0.223875  0.124495
max_slope               0.159856  0.031044  0.000162  5  0.223775  0.095937

[2] Model Leaderboard:
                 model  score_val  pred_time_val  fit_time
0              XGBoost  -1.046437       0.004002  0.486560
1        LightGBMLarge  -1.058593       0.003003  0.564224
2             CatBoost  -1.045490       0.001086  0.708230
3      RandomForestMSE  -1.018830       0.053039  0.658756
4  WeightedEnsemble_L2  -0.869309       0.016406  5.785375

[3] TOP 10 WORST PREDICTIONS (Review these Wave IDs!):
    wave_id  wait_time_ms  pred_wait_time_ms  error_ms
26     27.0          8.16           0.112322 -8.047678
60     61.0          8.10           0.698896 -7.401104
74     75.0          8.34           1.177199 -7.162801
81     82.0          7.71           1.104595 -6.605405
39     40.0          5.72           0.327970 -5.392030
93     94.0          5.66           3.107691 -2.552309
98     99.0          8.44           6.828873 -1.611127
22     23.0          4.25           2.752804 -1.497196
73     74.0          6.36           4.937285 -1.422715
46     47.0          4.77           3.410289 -1.359711

✅ Analysis Report Saved: data/processed/analysis/diagnosis_report_20260105_154843.csv
✅ Feature Importance Saved: data/processed/analysis/feature_importance_20260105_154843.csv
✅ Model saved at: AutogluonModels/ag-20260105_154843
============================================================
PS C:\Users\TPongkun\Documents\autoML> python scripts\autoML.py --mode predict --model-path AutogluonModels/ag-20260105_154843 --inference-csv data\processed\inference\wide_v13.csv --out data\processed\prediction\predicted_wait_time.csv  
🔮 Loading model and predicting: data\processed\inference\wide_v13.csv
✅ Prediction Results saved: data\processed\prediction\predicted_wait_time.csv
PS C:\Users\TPongkun\Documents\autoML> python scripts/plot_all_waves.py --mode check_pred --raw data\raw\data_for_train_new_ver.csv --result data/processed/prediction/predicted_wait_time.csv --out plots/check_pred_51_20per_fix_2/check.png