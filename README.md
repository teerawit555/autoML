## Setup Python Environment (Windows)

```powershell
# check python versions
py -0

# create venv (Python 3.11)
py -V:3.11 -m venv venv311

# allow script (run once)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# activate venv
.\venv311\Scripts\Activate.ps1

# python version
python --version

pip install -r requirements.txt

# 1 to generate fake data 1000 samples
python scripts\generate_raw_long.py --out data\raw\data1000samples_test.csv --dt_ms 0.01 --t_end_ms 9.99

# 2 เปลี่ยนข้อมูล Waveform ที่เรียงกันเป็นลำดับเวลาให้กลายเป็นคอลัมน์ฟีเจอร์ (i_0, i_1, i_2, ...) เพื่อให้ขั้นตอน extract_features.py สามารถสกัดคุณลักษณะต่าง ๆ (เช่น peak_rel, max_slope) ได้ง่ายขึ้น
python scripts\make_wide_csv.py --in data\raw\data1000samples_test.csv --out data\processed\inference\wide_1000_demo_x.csv

# 3 (Feature Engineering) จากข้อมูล Waveform ที่อยู่ในรูปแบบกว้าง (Wide Format) ที่สร้างโดย make_wide_csv.py
python scripts\extract_features.py --in data\processed\inference\wide_1000_demo_x.csv --out data\processed\inference\train_features_1000_x.csv

# 4 train models
python scripts\autoML.py --mode train --data data\processed\train\train_features.csv --label wait_time_ms --time-limit 120

# 5 predict 
python scripts\autoML.py --mode predict --model-path AutogluonModels --inference-csv data\processed\inference\train_features_1000_x.csv --out data\processed\prediction\predicted_wait_time_1000_x.csv

# 6 plot
python scripts\plot_all_waves.py --raw data\raw\data1000samples_test.csv --pred data\processed\prediction\predicted_wait_time_1000_x.csv --out plots\waves\waves_with_pred_wait_x.png

# deactivate
deactivate

# =================================================================
# ML WORKFLOW: TRAIN (20 SAMPLES) & PREDICT (1000 SAMPLES)
# =================================================================

# 1. SETUP: Activate Virtual Environment (ต้องทำก่อนเริ่มงาน)
# หากคุณใช้ชื่อ venv311
.\venv311\Scripts\Activate.ps1

# --- PART 1: PREPARE INFERENCE DATA (1000 SAMPLES) ---

# 2. GENERATE RAW INFERENCE DATA (สร้างข้อมูล Waveform ดิบ 1000 Samples ใหม่)
python scripts\generate_raw_long.py --out data\raw\data1000samples_test.csv --dt_ms 0.01 --t_end_ms 9.99

# 3. MAKE WIDE CSV (แปลง Raw Long Format -> Wide Format)
python scripts\make_wide_csv.py --in data\raw\data1000samples_test.csv --out data\processed\inference\wide_1000_demo_x.csv

# 4. FEATURE ENGINEERING (สกัดคุณลักษณะ ML จาก Wide Format)
# Output ไฟล์นี้ (train_features_1000_x.csv) ไม่มี Label (wait_time_ms)
python scripts\extract_features.py --in data\processed\inference\wide_1000_demo_x.csv --out data\processed\inference\train_features_1000_x.csv

# --- PART 2: TRAIN MODEL (ใช้ไฟล์ชุดเก่า 20 SAMPLES ที่มี Label) ---

# 5. TRAIN MODEL (AutoML/AutoGluon)
# **สำคัญ:** ใช้ไฟล์ train_features.csv (จากโฟลเดอร์ train ซึ่งมี Label)
# หากโมเดลเดิมไม่ถูกลบ จะทำการฝึกทับใน AutogluonModels
python scripts\autoML.py --mode train --data data\processed\train\train_features.csv --label wait_time_ms --time-limit 120

# --- PART 3: PREDICT & PLOT RESULTS ---

# 6. PREDICT NEW DATA (ทำนายข้อมูล 1000 Samples ที่เพิ่งสกัดฟีเจอร์มา)
# Input คือไฟล์ที่สร้างในขั้นตอนที่ 4
python scripts\autoML.py --mode predict --model-path AutogluonModels --inference-csv data\processed\inference\train_features_1000_x.csv --out data\processed\prediction\predicted_wait_time_1000_x.csv

# 7. PLOT RESULTS (พล็อตกราฟ Waveform พร้อมเส้นทำนาย)
python scripts\plot_waves.py --raw data\raw\data1000samples_test.csv --pred data\processed\prediction\predicted_wait_time_1000_x.csv --out plots\waves\waves_with_pred_wait_x.png
