# Waveform ML: End-to-End Prediction Pipeline

An intelligent waveform analysis framework designed to predict signal settling times (`wait_time_ms`) using a hybrid approach of Signal Processing and Automated Machine Learning (AutoGluon).

This comprehensive pipeline handles the entire lifecycle:
1.  **Synthetic Data Generation**: Creating diverse waveform patterns (Sine, Pulse, Step, Glitch).
2.  **Advanced Feature Engineering**: extracting robust metrics using digital filters and rule-based logic (V19).
3.  **AutoML Training & Inference**: Using AutoGluon to train high-accuracy regressors and classifiers.

## 1. Environment Setup

This project requires **Python 3.11** on Windows. Follow these steps to initialize your local environment using PowerShell:

### Verify available Python versions
```powershell
py -0
```
### Create a virtual environment using Python 3.11
```powershell
py -V:3.11 -m venv venv311
```
### Enable script execution (Required for PowerShell activation)
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
### Activate the virtual environment
```powershell
.\venv311\Scripts\Activate.ps1
```
### Verify the current Python version
```powershell
python --version
```
### Install dependencies
```powershell
pip install -r requirements.txt
```
## 2. CLI part

### 1. Generate data (1000 samples)
#### 1.1 Data for train
    
```powershell
python scripts\generate_train_sample.py --out data\raw\data_for_train.csv --dt_ms 0.01 --t_end_ms 9.99 --n_waves 10000
```
#### 1.2 Data for pred
```powershell
python scripts\generate_predict_sample.py --out data\raw\data_for_pred.csv --dt_ms 0.01 --t_end_ms 9.99 --n_waves 2000
```
### 2. Transform Long data to Wide format
#### 2.1 Data for training (with labels) 
```powershell
python scripts/extract_features.py --mode train --in data/raw/data_for_train.csv --out data/processed/train/train_features.csv
```
#### 2.2 Data for inference (without labels) 
```powershell
python scripts/extract_features.py --mode inference --in data/raw/data_for_pred.csv --out data\processed\inference\pred_features.csv
```
### 3. Train the AutoML model
```powershell
python scripts\train_ag_TT.py --mode train --data data\processed\train\train_features.csv --label wait_time_ms --time-limit 120
```

### 4. Run Prediction (Inference)
```powershell
python scripts\predict_ag_TT.py --mode predict --model-path AutogluonModels/ag-20260129_142046 --inference-csv data\processed\inference\pred_features.csv --out data\processed\prediction\predicted_wait_time.csv
```

### 5. Generate Visualization plots

#### Debugging plot waveform
```powershell
# Check waveform data for train
python scripts/check_plot_v22.py --mode check_train --raw data\raw\data_for_train.csv --result data\processed\train\train_features_V_TT.csv --out plots/V_xx/train_scale.png     
# Check waveform data for pred
python scripts/check_plot_v22.py --mode check_pred --raw data\raw\data_for_pred.csv --result data/processed/prediction/pred_wait_time.csv --out plots/V_xx/check.png       
```
### 6. audit features
```powershell
python scripts/audit_features.py --train data/processed/train/train_features_V_TT.csv --pred data/processed/inference/wide_v_TT.csv --outdir AutogluonModels/ag-v27-20260129_142046/feature_audit --min-unique 10 --corr-thr 0.95
```
## AutoML Architecture (autoML.py)
The machine learning pipeline implements a Two-Stage Hybrid Model designed to handle the specific nature of waveform settling times (sparse non-zero values and specific signal patterns).

### The Two-Stage Approach
Instead of a single regression model, we split the problem into two tasks:

#### 1. Stage A (Zero Classifier): 
A Binary Classifier (is_zero) that determines if the signal settles immediately (0ms) or needs time to settle.

#### 2. Stage B (Wait Regressor): 
A Regressor trained only on non-zero samples (using log1p transformation) to predict the exact wait time for active signals.

### Hybrid Logic Override
To ensure 100% accuracy on known patterns, the pipeline implements a Hybrid Logic mechanism during inference:

#### AI Prediction: 
The default path using the Two-Stage model.

#### Logic Override: 
If the feature engineering step detects specific patterns (e.g., Continuous Sine Wave or Glitch), the AI prediction is overridden, and the wait time is forced to 0.0ms.

graph TD
    Start[Input Data] --> Step1{1. Rule Check}
    
    Step1 -- "Yes (Sine / Glitch)" --> ResultZero[Force 0.0ms]
    Step1 -- "No" --> Step2{2. AI: Is it Zero?}
    
    Step2 -- "Yes (Prob > Threshold)" --> ResultZero
    Step2 -- "No (It has settling time)" --> Step3[3. AI: Calculate Time]
    
    Step3 --> Final[Final Prediction]
    ResultZero --> Final

## Members QOET AI TEAM

1.  **P'Note**
2.  **P'Folk**
3.  **TT** 
4.  **MOS** 

## 3. Example Result 
#### ผลอยู่ที่ plots\V19_label_202316
```powershell
(venv311) PS C:\Users\Zerefany11\Documents\autoML> python scripts\autoML.py --mode train --data data\processed\train\train_features_V19.csv --label wait_time_ms --time-limit 400 --presets medium_quality
🚀 Training device: CPU
presets=medium_quality time_limit=400s (zero=100s reg=300s)
Verbosity: 2 (Standard Logging)
=================== System Info ===================
AutoGluon Version:  1.5.0
Python Version:     3.11.9
Operating System:   Windows
Platform Machine:   AMD64
Platform Version:   10.0.26100
CPU Count:          16
Pytorch Version:    2.5.1+cu121
CUDA Version:       CUDA is not available
Memory Avail:       5.77 GB / 15.69 GB (36.8%)
Disk Space Avail:   125.51 GB / 475.34 GB (26.4%)
===================================================
Presets specified: ['medium_quality']
Using hyperparameters preset: hyperparameters='default'
Beginning AutoGluon training ... Time limit = 100s
AutoGluon will save models to "C:\Users\Zerefany11\Documents\autoML\AutogluonModels\ag-20260106_202316\zero_clf"
Train Data Rows:    10000
Train Data Columns: 19
Label Column:       is_zero
Problem Type:       binary
Preprocessing data ...
Selected class <--> label mapping:  class 1 = 1, class 0 = 0
Using Feature Generators to preprocess the data ...
Fitting AutoMLPipelineFeatureGenerator...
        Available Memory:                    5920.69 MB
        Train Data (Original)  Memory Usage: 1.45 MB (0.0% of available memory)
        Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
        Stage 1 Generators:
                Fitting AsTypeFeatureGenerator...
                        Note: Converting 1 features to boolean dtype as they only contain 2 unique values.
        Stage 2 Generators:
                Fitting FillNaFeatureGenerator...
        Stage 3 Generators:
                Fitting IdentityFeatureGenerator...
        Stage 4 Generators:
                Fitting DropUniqueFeatureGenerator...
        Stage 5 Generators:
                Fitting DropDuplicatesFeatureGenerator...
        Unused Original Features (Count: 1): ['logic_tail_range']
                These features were not used to generate any of the output features. Add a feature generator compatible with these features to utilize them.
                Features can also be unused if they carry very little information, such as being categorical but having almost entirely unique values or being duplicates of other features.
                These features do not need to be present at inference time.
                ('float', []) : 1 | ['logic_tail_range']
        Types of features in original data (raw dtype, special dtypes):
                ('float', []) : 18 | ['x_end', 'std_all', 'ringing_energy', 'max_rolling_std_half', 'std_tail_50', ...]
        Types of features in processed data (raw dtype, special dtypes):
                ('float', [])     : 17 | ['x_end', 'std_all', 'ringing_energy', 'max_rolling_std_half', 'std_tail_50', ...]
                ('int', ['bool']) :  1 | ['logic_flag_continuous']
        0.1s = Fit runtime
        18 features in original data used to generate 18 features in processed data.
        Train Data (Processed) Memory Usage: 1.31 MB (0.0% of available memory)
Data preprocessing and feature engineering runtime = 0.15s ...
AutoGluon will gauge predictive performance using evaluation metric: 'f1'
        To change this, specify the eval_metric parameter of Predictor()
Automatically generating train/validation split with holdout_frac=0.1, Train Rows: 9000, Val Rows: 1000
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
Fitting 11 L1 models, fit_strategy="sequential" ...
Fitting model: LightGBMXT ... Training model for up to 99.85s of the 99.85s of remaining time.
        Fitting with cpus=12, gpus=0, mem=0.0/5.8 GB
        1.0      = Validation score   (f1)
        3.66s    = Training   runtime
        0.01s    = Validation runtime
Fitting model: LightGBM ... Training model for up to 96.17s of the 96.16s of remaining time.
        Fitting with cpus=12, gpus=0, mem=0.0/5.7 GB
        1.0      = Validation score   (f1)
        1.41s    = Training   runtime
        0.01s    = Validation runtime
Fitting model: RandomForestGini ... Training model for up to 94.73s of the 94.73s of remaining time.
        Fitting with cpus=16, gpus=0, mem=0.0/5.7 GB
        1.0      = Validation score   (f1)
        1.52s    = Training   runtime
        0.07s    = Validation runtime
Fitting model: RandomForestEntr ... Training model for up to 93.11s of the 93.10s of remaining time.
        Fitting with cpus=16, gpus=0, mem=0.0/5.7 GB
        1.0      = Validation score   (f1)
        1.46s    = Training   runtime
        0.05s    = Validation runtime
Fitting model: CatBoost ... Training model for up to 91.58s of the 91.57s of remaining time.
        Fitting with cpus=12, gpus=0
        1.0      = Validation score   (f1)
        46.18s   = Training   runtime
        0.0s     = Validation runtime
Fitting model: ExtraTreesGini ... Training model for up to 45.37s of the 45.36s of remaining time.
        Fitting with cpus=16, gpus=0, mem=0.0/5.8 GB
        1.0      = Validation score   (f1)
        1.18s    = Training   runtime
        0.1s     = Validation runtime
Fitting model: ExtraTreesEntr ... Training model for up to 44.05s of the 44.05s of remaining time.
        Fitting with cpus=16, gpus=0, mem=0.0/5.8 GB
        1.0      = Validation score   (f1)
        1.08s    = Training   runtime
        0.06s    = Validation runtime
Fitting model: NeuralNetFastAI ... Training model for up to 42.90s of the 42.90s of remaining time.
        Fitting with cpus=12, gpus=0, mem=0.0/5.7 GB
        Warning: Exception caused NeuralNetFastAI to fail during training (ImportError)... Skipping this model.
                Import fastai failed. A quick tip is to install via `pip install autogluon.tabular[fastai]==1.5.0`. 
Fitting model: XGBoost ... Training model for up to 41.75s of the 41.74s of remaining time.
        Fitting with cpus=12, gpus=0
        1.0      = Validation score   (f1)
        1.56s    = Training   runtime
        0.02s    = Validation runtime
Fitting model: NeuralNetTorch ... Training model for up to 40.15s of the 40.14s of remaining time.
        Fitting with cpus=12, gpus=0, mem=0.0/5.7 GB
C:\Users\Zerefany11\Documents\autoML\venv311\Lib\site-packages\sklearn\compose\_column_transformer.py:975: FutureWarning: The parameter `force_int_remainder_cols` is deprecated and will be removed in 1.9. It has no effect. Leave it to its default value to avoid this warning.
  warnings.warn(
        1.0      = Validation score   (f1)
        19.09s   = Training   runtime
        0.02s    = Validation runtime
Fitting model: LightGBMLarge ... Training model for up to 21.03s of the 21.02s of remaining time.
        Fitting with cpus=12, gpus=0, mem=0.1/5.7 GB
        1.0      = Validation score   (f1)
        3.02s    = Training   runtime
        0.01s    = Validation runtime
Fitting model: WeightedEnsemble_L2 ... Training model for up to 99.85s of the 17.96s of remaining time.
        Fitting 1 model on all data | Fitting with cpus=16, gpus=0, mem=0.0/5.6 GB
        Ensemble Weights: {'ExtraTreesGini': 1.0}
        1.0      = Validation score   (f1)
        0.36s    = Training   runtime
        0.0s     = Validation runtime
AutoGluon training complete, total runtime = 82.46s ... Best model: WeightedEnsemble_L2 | Estimated inference throughput: 9549.2 rows/s (1000 batch size)
Enabling decision threshold calibration (calibrate_decision_threshold='auto', metric is valid, problem_type is 'binary')
Calibrating decision threshold to optimize metric f1 | Checking 51 thresholds...
Calibrating decision threshold via fine-grained search | Checking 38 thresholds...
        Base Threshold: 0.500   | val: 1.0000
        Best Threshold: 0.500   | val: 1.0000
TabularPredictor saved. To load, use: predictor = TabularPredictor.load("C:\Users\Zerefany11\Documents\autoML\AutogluonModels\ag-20260106_202316\zero_clf")
Verbosity: 2 (Standard Logging)
=================== System Info ===================
AutoGluon Version:  1.5.0
Python Version:     3.11.9
Operating System:   Windows
Platform Machine:   AMD64
Platform Version:   10.0.26100
CPU Count:          16
Pytorch Version:    2.5.1+cu121
CUDA Version:       CUDA is not available
Memory Avail:       5.63 GB / 15.69 GB (35.9%)
Disk Space Avail:   125.50 GB / 475.34 GB (26.4%)
===================================================
Presets specified: ['medium_quality']
Using hyperparameters preset: hyperparameters='default'
Beginning AutoGluon training ... Time limit = 300s
AutoGluon will save models to "C:\Users\Zerefany11\Documents\autoML\AutogluonModels\ag-20260106_202316\wait_reg"
Train Data Rows:    6061
Train Data Columns: 19
Label Column:       wait_time_log
Problem Type:       regression
Preprocessing data ...
Using Feature Generators to preprocess the data ...
Fitting AutoMLPipelineFeatureGenerator...
        Available Memory:                    5770.21 MB
        Train Data (Original)  Memory Usage: 0.88 MB (0.0% of available memory)
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
        Useless Original Features (Count: 2): ['logic_flag_continuous', 'logic_flag_glitch']
                These features carry no predictive signal and should be manually investigated.
                This is typically a feature which has the same value for all rows.
                These features do not need to be present at inference time.
        Unused Original Features (Count: 1): ['logic_tail_range']
                These features were not used to generate any of the output features. Add a feature generator compatible with these features to utilize them.
                Features can also be unused if they carry very little information, such as being categorical but having almost entirely unique values or being duplicates of other features.
                These features do not need to be present at inference time.
                ('float', []) : 1 | ['logic_tail_range']
        Types of features in original data (raw dtype, special dtypes):
                ('float', []) : 16 | ['x_end', 'std_all', 'ringing_energy', 'max_rolling_std_half', 'std_tail_50', ...]
        Types of features in processed data (raw dtype, special dtypes):
                ('float', []) : 16 | ['x_end', 'std_all', 'ringing_energy', 'max_rolling_std_half', 'std_tail_50', ...]
        0.1s = Fit runtime
        16 features in original data used to generate 16 features in processed data.
        Train Data (Processed) Memory Usage: 0.74 MB (0.0% of available memory)
Data preprocessing and feature engineering runtime = 0.09s ...
AutoGluon will gauge predictive performance using evaluation metric: 'mean_absolute_error'
        This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
        To change this, specify the eval_metric parameter of Predictor()
Automatically generating train/validation split with holdout_frac=0.1, Train Rows: 5454, Val Rows: 607
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
Fitting model: LightGBMXT ... Training model for up to 299.91s of the 299.91s of remaining time.
        Fitting with cpus=12, gpus=0, mem=0.0/5.6 GB
[1000]  valid_set's l1: 0.0492574
[2000]  valid_set's l1: 0.04863
[3000]  valid_set's l1: 0.0485865
        -0.0485  = Validation score   (-mean_absolute_error)
        5.12s    = Training   runtime
        0.05s    = Validation runtime
Fitting model: LightGBM ... Training model for up to 294.50s of the 294.50s of remaining time.
        Fitting with cpus=12, gpus=0, mem=0.0/5.6 GB
[1000]  valid_set's l1: 0.0524672
        -0.0521  = Validation score   (-mean_absolute_error)
        1.61s    = Training   runtime
        0.03s    = Validation runtime
Fitting model: RandomForestMSE ... Training model for up to 292.77s of the 292.77s of remaining time.
        Fitting with cpus=16, gpus=0, mem=0.0/5.6 GB
        -0.0529  = Validation score   (-mean_absolute_error)
        3.68s    = Training   runtime
        0.06s    = Validation runtime
Fitting model: CatBoost ... Training model for up to 288.86s of the 288.86s of remaining time.
        Fitting with cpus=12, gpus=0
        -0.0488  = Validation score   (-mean_absolute_error)
        132.37s  = Training   runtime
        0.01s    = Validation runtime
Fitting model: ExtraTreesMSE ... Training model for up to 156.42s of the 156.42s of remaining time.
        Fitting with cpus=16, gpus=0, mem=0.0/5.6 GB
        -0.0537  = Validation score   (-mean_absolute_error)
        1.34s    = Training   runtime
        0.11s    = Validation runtime
Fitting model: NeuralNetFastAI ... Training model for up to 154.75s of the 154.75s of remaining time.
        Fitting with cpus=12, gpus=0, mem=0.0/5.5 GB
        Warning: Exception caused NeuralNetFastAI to fail during training (ImportError)... Skipping this model.
                Import fastai failed. A quick tip is to install via `pip install autogluon.tabular[fastai]==1.5.0`.
Fitting model: XGBoost ... Training model for up to 154.56s of the 154.56s of remaining time.
        Fitting with cpus=12, gpus=0
        -0.0531  = Validation score   (-mean_absolute_error)
        3.71s    = Training   runtime
        0.02s    = Validation runtime
Fitting model: NeuralNetTorch ... Training model for up to 150.74s of the 150.74s of remaining time.
        Fitting with cpus=12, gpus=0, mem=0.0/5.5 GB
C:\Users\Zerefany11\Documents\autoML\venv311\Lib\site-packages\sklearn\compose\_column_transformer.py:975: FutureWarning: The parameter `force_int_remainder_cols` is deprecated and will be removed in 1.9. It has no effect. Leave it to its default value to avoid this warning.
  warnings.warn(
        -0.0404  = Validation score   (-mean_absolute_error)
        146.96s  = Training   runtime
        0.03s    = Validation runtime
Fitting model: LightGBMLarge ... Training model for up to 3.74s of the 3.74s of remaining time.
        Fitting with cpus=12, gpus=0, mem=0.1/5.9 GB
        Ran out of time, early stopping on iteration 535. Best iteration is:
        [465]   valid_set's l1: 0.0503766
        -0.0504  = Validation score   (-mean_absolute_error)
        3.89s    = Training   runtime
        0.02s    = Validation runtime
Fitting model: WeightedEnsemble_L2 ... Training model for up to 299.91s of the -0.33s of remaining time.
        Fitting 1 model on all data | Fitting with cpus=16, gpus=0, mem=0.0/5.9 GB
        Ensemble Weights: {'NeuralNetTorch': 0.75, 'LightGBMXT': 0.125, 'CatBoost': 0.125}
        -0.0397  = Validation score   (-mean_absolute_error)
        0.08s    = Training   runtime
        0.0s     = Validation runtime
AutoGluon training complete, total runtime = 300.48s ... Best model: WeightedEnsemble_L2 | Estimated inference throughput: 6916.7 rows/s (607 batch size)
TabularPredictor saved. To load, use: predictor = TabularPredictor.load("C:\Users\Zerefany11\Documents\autoML\AutogluonModels\ag-20260106_202316\wait_reg")

============================================================
 DEEP MODEL ANALYSIS & DIAGNOSIS
============================================================

[1] Calculating Feature Importance (Regressor)...
These features in provided data are not utilized by the predictor and will be ignored: ['logic_tail_range', 'logic_flag_continuous', 'logic_flag_glitch']
Computing feature importance via permutation shuffling for 16 features using 5000 rows with 5 shuffle sets...
        47.03s  = Expected runtime (9.41s per shuffle set)
        26.47s  = Actual runtime (Completed 5 of 5 shuffle sets)
                      importance    stddev       p_value  n  p99_high   p99_low
logic_baseline_diff     0.165068  0.002181  3.655749e-09  5  0.169558  0.160577
ringing_energy          0.145655  0.004139  7.814217e-08  5  0.154177  0.137133
max_slope               0.097046  0.001640  9.783379e-09  5  0.100423  0.093669
max_rolling_std_half    0.091530  0.001177  3.279194e-09  5  0.093953  0.089107
x_end                   0.062879  0.001061  9.730171e-09  5  0.065064  0.060694
logic_noise_energy      0.052446  0.001306  4.608870e-08  5  0.055135  0.049757
logic_global_range      0.046196  0.000952  2.162629e-08  5  0.048156  0.044236
std_all                 0.043169  0.000830  1.638444e-08  5  0.044877  0.041460
mid_to_tail_ratio       0.035519  0.000704  1.855495e-08  5  0.036969  0.034068
logic_heavy_amp         0.023669  0.000721  1.029646e-07  5  0.025153  0.022186
std_tail_50             0.021363  0.000582  6.615755e-08  5  0.022562  0.020164
max_dev_tail_50         0.012068  0.000656  1.042709e-06  5  0.013418  0.010718
drift_score             0.011727  0.000335  7.939971e-08  5  0.012415  0.011038
crossing_rate           0.007768  0.000381  6.920671e-07  5  0.008552  0.006983
logic_per_score         0.007144  0.000265  2.261535e-07  5  0.007690  0.006599

[2] Model Leaderboard (Regressor):
           model  score_val  pred_time_val    fit_time
0        XGBoost  -0.053068       0.016988    3.707388
1  LightGBMLarge  -0.050377       0.019362    3.890773
2       LightGBM  -0.052130       0.027975    1.614197
3       CatBoost  -0.048751       0.009622  132.369889
4     LightGBMXT  -0.048486       0.046067    5.123585

[2.5] Auto-picked zero threshold = 0.215 (train MAE=0.085)
proba_is_zero min/mean/max = 0.000000 0.393868 1.000000
predicted zeros = 3939 / 10000

[3] TOP 10 WORST PREDICTIONS (Review these Wave IDs!):
      wave_id  wait_time_ms  pred_wait_time_ms  error_ms
9104   9105.0          5.51           2.774882 -2.735118
7840   7841.0          5.48           2.754149 -2.725851
7853   7854.0          5.30           3.062973 -2.237027
6815   6816.0          4.80           2.890680 -1.909320
1783   1784.0          4.90           3.002419 -1.897581
1713   1714.0          5.68           3.883425 -1.796575
9475   9476.0          5.22           3.490641 -1.729359
4666   4667.0          4.56           2.966526 -1.593474
6784   6785.0          5.21           3.638760 -1.571240
7015   7016.0          3.44           4.830713  1.390713

✅ Analysis Report Saved: data/processed/analysis/diagnosis_report_20260106_202316.csv
✅ Feature Importance Saved: data/processed/analysis/feature_importance_20260106_202316.csv
✅ Models saved at: AutogluonModels/ag-20260106_202316
   - Zero classifier: AutogluonModels/ag-20260106_202316\zero_clf
   - Wait regressor:  AutogluonModels/ag-20260106_202316\wait_reg
   - Zero threshold:  0.215  (saved: AutogluonModels/ag-20260106_202316\zero_threshold.txt)
============================================================
(venv311) PS C:\Users\Zerefany11\Documents\autoML> python scripts\autoML.py --mode predict --model-path AutogluonModels/ag-20260106_202316 --inference-csv data\processed\inference\wide_v19.csv --out data\processed\prediction\predicted_wait_time.csv    
🔮 Loading models and predicting: data\processed\inference\wide_v19.csv
using zero threshold = 0.215
proba_is_zero min/mean/max = 0.000000 0.389447 1.000000
predicted zeros = 783 / 2000
 Applying Logic Override for Continuous Waves: 737 items forced to 0.0ms
 Applying Logic Override for Glitch: 38 items forced to 0.0ms
✅ Prediction Results saved: data\processed\prediction\predicted_wait_time.csv
```
