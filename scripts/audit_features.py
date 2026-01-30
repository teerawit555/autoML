# scripts/audit_features.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------
def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    return pd.read_csv(p)

def numeric_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def safe_quantile(s: pd.Series, q: float) -> float:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))

def summarize_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        x = s.to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        nun = int(pd.Series(x).nunique()) if x.size else 0
        std = float(np.std(x)) if x.size else 0.0
        rows.append({
            "feature": c,
            "n": int(x.size),
            "nunique": nun,
            "std": std,
            "min": float(np.min(x)) if x.size else np.nan,
            "p05": float(np.quantile(x, 0.05)) if x.size else np.nan,
            "p50": float(np.quantile(x, 0.50)) if x.size else np.nan,
            "p95": float(np.quantile(x, 0.95)) if x.size else np.nan,
            "max": float(np.max(x)) if x.size else np.nan,
            "nan_ratio": float(1.0 - (x.size / max(len(s), 1))),
        })
    out = pd.DataFrame(rows).sort_values(["nunique", "std"], ascending=[True, True]).reset_index(drop=True)
    return out

def compare_train_pred(
    train: pd.DataFrame,
    pred: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    rows = []
    for c in cols:
        tr = pd.to_numeric(train[c], errors="coerce")
        pr = pd.to_numeric(pred[c], errors="coerce")

        tr_mu = float(np.nanmean(tr))
        tr_sd = float(np.nanstd(tr))
        pr_mu = float(np.nanmean(pr))
        pr_sd = float(np.nanstd(pr))

        # normalized shift: |mu_pred - mu_train| / (sd_train + eps)
        eps = 1e-12
        z_shift = abs(pr_mu - tr_mu) / (tr_sd + eps)

        # range overlap check (p05/p95)
        tr_p05 = safe_quantile(tr, 0.05)
        tr_p95 = safe_quantile(tr, 0.95)
        pr_p05 = safe_quantile(pr, 0.05)
        pr_p95 = safe_quantile(pr, 0.95)

        overlap = not (pr_p95 < tr_p05 or pr_p05 > tr_p95)

        rows.append({
            "feature": c,
            "train_mean": tr_mu, "train_std": tr_sd, "train_p05": tr_p05, "train_p95": tr_p95,
            "pred_mean": pr_mu,  "pred_std": pr_sd,  "pred_p05": pr_p05,  "pred_p95": pr_p95,
            "z_shift": float(z_shift),
            "p05_p95_overlap": bool(overlap),
        })

    out = pd.DataFrame(rows)
    # sort by biggest distribution shift first
    out = out.sort_values(["z_shift", "p05_p95_overlap"], ascending=[False, True]).reset_index(drop=True)
    return out

def high_corr_pairs(df: pd.DataFrame, cols: list[str], corr_thr: float = 0.98, max_pairs: int = 2000) -> pd.DataFrame:
    X = df[cols].copy()
    # coerce numeric
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # drop columns that are all nan or constant
    nun = X.nunique(dropna=True)
    keep = nun[nun > 1].index.tolist()
    X = X[keep]
    cols2 = keep

    # compute correlation
    corr = X.corr(numeric_only=True)
    pairs = []
    n = len(cols2)
    for i in range(n):
        for j in range(i + 1, n):
            a = cols2[i]
            b = cols2[j]
            v = corr.iat[i, j]
            if np.isfinite(v) and abs(v) >= corr_thr:
                pairs.append({"a": a, "b": b, "corr": float(v)})
                if len(pairs) >= max_pairs:
                    break
        if len(pairs) >= max_pairs:
            break
    # ✅ FIX: handle no pairs found
    if len(pairs) == 0:
        return pd.DataFrame(columns=["feat_a", "feat_b", "corr"])

    out = pd.DataFrame(pairs)

    # กันกรณี pairs มีแต่ dict คนละคีย์ (เผื่ออนาคต)
    if "corr" not in out.columns:
        return pd.DataFrame(columns=["feat_a", "feat_b", "corr"])

    out = out.sort_values("corr", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    out = pd.DataFrame(pairs).sort_values("corr", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    
    return out

def sanity_gate_suggest(
    calib: pd.DataFrame,
    y_col: str,
    proba_col: str,
    thr_high: float,
    thr_low: float,
    fast_ms: float,
    *,
    feat_cols: list[str],
    q_feat: float = 0.98,
    q_reg_fp: float = 0.95,
    reg_pred_col: str = "reg_wait_pred_ms",
) -> dict:
    """
    Derive sanity thresholds ONLY from CALIB true-fast and near the soft band.
    Uses quantiles so it’s data-driven.
    """
    y = pd.to_numeric(calib[y_col], errors="coerce").to_numpy(float)
    p = pd.to_numeric(calib[proba_col], errors="coerce").to_numpy(float)

    is_true_fast = np.isfinite(y) & (y <= fast_ms + 1e-12)
    near_soft = np.isfinite(p) & (p >= thr_low) & (p < thr_high)

    mask = is_true_fast & near_soft
    if not np.any(mask):
        mask = is_true_fast

    out = {
        "enabled": True,
        "policy": "soft_zone_fp_firewall",
        "thr_high": float(thr_high),
        "thr_low": float(thr_low),
        "q_feat": float(q_feat),
        "q_reg_fp": float(q_reg_fp),
        "features_present": {},
        "thresholds": {},
    }

    for f in feat_cols:
        if f in calib.columns:
            out["features_present"][f] = True
            out["thresholds"][f] = float(safe_quantile(calib.loc[mask, f], q_feat))
        else:
            out["features_present"][f] = False

    if reg_pred_col in calib.columns:
        out["features_present"][reg_pred_col] = True
        out["thresholds"][reg_pred_col] = float(safe_quantile(calib.loc[mask, reg_pred_col], q_reg_fp))
    else:
        out["features_present"][reg_pred_col] = False

    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train features csv (wide, includes wait_time_ms)")
    ap.add_argument("--pred", required=True, help="pred features csv (wide, no label)")
    ap.add_argument("--outdir", required=True, help="output folder for reports")

    ap.add_argument("--label", default="wait_time_ms")
    ap.add_argument("--exclude", default="wave_id,type,type_debug,dbg_label_reason,is_fast,wait_time_log",
                    help="comma-separated columns to exclude from feature audits")

    ap.add_argument("--corr-thr", type=float, default=0.98)
    ap.add_argument("--min-unique", type=int, default=10, help="flag as 'flat' if nunique < this")
    ap.add_argument("--min-std", type=float, default=1e-6, help="flag as 'flat' if std < this")

    # optional sanity suggest inputs (if you have calib_diagnostics csv)
    ap.add_argument("--calib-diag", default=None, help="calib_diagnostics_*.csv from training (optional)")
    ap.add_argument("--thr-high", type=float, default=None)
    ap.add_argument("--thr-low", type=float, default=None)
    ap.add_argument("--fast-ms", type=float, default=0.1)
    ap.add_argument("--proba-col", default="proba_is_fast")
    ap.add_argument("--reg-col", default="reg_wait_pred_ms")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train = read_csv(args.train)
    pred = read_csv(args.pred)

    exclude = {c.strip() for c in args.exclude.split(",") if c.strip()}

    # columns to audit: intersection numeric + common
    tr_cols = set(train.columns)
    pr_cols = set(pred.columns)
    common = sorted(list((tr_cols & pr_cols) - exclude))

    cols = [c for c in common if pd.api.types.is_numeric_dtype(train[c]) or pd.api.types.is_numeric_dtype(pred[c])]
    # ensure numeric coercible
    cols = [c for c in cols if c not in exclude]

    # 1) summarize train/pred
    sum_train = summarize_df(train, cols)
    sum_pred = summarize_df(pred, cols)

    sum_train.to_csv(outdir / "summary_train.csv", index=False)
    sum_pred.to_csv(outdir / "summary_pred.csv", index=False)

    # 2) flag flat features in pred
    flat_pred = sum_pred[(sum_pred["nunique"] < args.min_unique) | (sum_pred["std"] < args.min_std)].copy()
    flat_pred.to_csv(outdir / "flat_features_pred.csv", index=False)

    # 3) compare distribution shift
    comp = compare_train_pred(train, pred, cols)
    comp.to_csv(outdir / "compare_train_vs_pred.csv", index=False)

    # 4) high correlation pairs (use TRAIN to decide redundancy)
    corr_pairs = high_corr_pairs(train, cols, corr_thr=args.corr_thr)
    corr_pairs.to_csv(outdir / "high_corr_pairs_train.csv", index=False)

    print(f"✅ Saved reports to: {outdir}")
    print(f"- summary_train.csv / summary_pred.csv")
    print(f"- flat_features_pred.csv (nunique<{args.min_unique} or std<{args.min_std})")
    print(f"- compare_train_vs_pred.csv (sorted by z_shift)")
    print(f"- high_corr_pairs_train.csv (|corr|>={args.corr_thr})")

    # 5) sanity suggestion (optional)
    if args.calib_diag and (args.thr_high is not None) and (args.thr_low is not None):
        calib = read_csv(args.calib_diag)

        # features that are meaningful for FP firewall (recommend list)
        # you can add/remove here
        sanity_feats = [
            "last_edge_pos_ratio",
            "edge_span_ratio",
            "tail_std",
            "post_head_std",
            "meta_step_to_span",
        ]
        # keep only those existing in calib
        sanity_feats = [f for f in sanity_feats if f in calib.columns]

        sanity = sanity_gate_suggest(
            calib=calib,
            y_col=args.label,
            proba_col=args.proba_col,
            thr_high=float(args.thr_high),
            thr_low=float(args.thr_low),
            fast_ms=float(args.fast_ms),
            feat_cols=sanity_feats,
            q_feat=0.98,
            q_reg_fp=0.95,
            reg_pred_col=args.reg_col,
        )
        # save json
        import json
        (outdir / "sanity_suggest.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
        print(f"- sanity_suggest.json (derived from CALIB true-fast near soft band)")
    else:
        print("ℹ️ sanity_suggest skipped (provide --calib-diag --thr-high --thr-low to enable)")

if __name__ == "__main__":
    main()

'''
=============================================================================
audit_features.py — รายงานแต่ละไฟล์เอาไว้ดูอะไร / ใช้วิเคราะห์อะไรได้บ้าง

Script นี้ทำ Feature Audit เพื่อเช็คว่า
1) Feature ใน train กับ pred มีหน้าตาเหมือนกันไหม (distribution shift)
2) Feature ใน pred มีพัง/แบน/คงที่/กลายเป็น 0 เยอะไหม (flat / missing-signal)
3) มี feature ซ้ำกันมาก (correlation สูง) จนควร drop เพื่อลด redundancy ไหม
4) (optional) ถ้ามี calib_diagnostics + thresholds -> แนะนำ sanity gate thresholds ได้

---------------------------------------------------------------------------
Outputs (ไฟล์ที่ได้ใน outdir)
---------------------------------------------------------------------------

1) summary_train.csv
   - สรุปสถิติ “ต่อ feature” บน TRAIN (เฉพาะ numeric และ common columns)
   - ใช้เพื่อ:
     • หา feature ที่แทบไม่ขยับ (nunique ต่ำ, std ต่ำ) -> ไม่ช่วยโมเดล / เสี่ยงให้โมเดลงง
     • ดู scale ของ feature ว่าหลุดโลกไหม (max ใหญ่มาก, std ใหญ่มาก)
     • ดู quantile (p05/p50/p95) เพื่อเห็นรูปทรงคร่าว ๆ

   คอลัมน์สำคัญ:
     - nunique: จำนวนค่าที่ไม่ซ้ำ (ค่าต่ำมาก = flat/constant)
     - std: ส่วนเบี่ยงเบนมาตรฐาน (ต่ำมาก = flat) / สูงมาก = scale อันตราย/มี outlier
     - min/max: ดูว่าค่ากระโดดเกินเหตุไหม
     - p05/p50/p95: ดูช่วงค่าหลัก ๆ (กันโดน max ที่เป็น outlier หลอก)
     - nan_ratio: สัดส่วนที่เป็น NaN (สูง = feature extractor มีปัญหา/ key หาย)

   วิธีใช้จริง:
     • ถ้า feature nunique=1 หรือ std≈0 -> drop ทิ้งได้เลย (ยิ่งใน pred ถ้าเป็นแบบนี้ยิ่งอันตราย)
     • ถ้า std ใหญ่มาก + max ใหญ่มาก (เช่น 1e6) -> เสี่ยงทำให้ tree split พัง/โมเดลมั่นใจผิด
     • ถ้า nan_ratio > 0 ใน pred -> ตรวจ feature extractor / input pipeline


2) summary_pred.csv
   - เหมือน summary_train แต่คำนวณจาก PRED / INFERENCE DATA
   - ใช้เพื่อ:
     • ตรวจว่า feature extractor ที่ใช้กับ pred “ผลิต feature ได้ปกติไหม”
     • หา feature ที่กลายเป็น constant ใน pred ทั้งที่ train ไม่ constant (สัญญาณหาย)
     • หา feature ที่กลายเป็น 0 เยอะผิดปกติ (เช่น edge_* กลายเป็น 0 เพราะ detect edge fail)

   วิธีใช้จริง:
     • ถ้า pred nunique ต่ำผิดปกติ -> โมเดลจะ “มองไม่เห็นสัญญาณ” => ทำนายเพี้ยน/ตก type บางกลุ่ม
     • ถ้า pred p50/p95 shift มากจาก train -> โมเดลเจอ distribution ใหม่ => proba แปลก/ทำนายหลุด


3) flat_features_pred.csv
   - ดึงจาก summary_pred โดย flag feature ที่:
       nunique < --min-unique  (default=10)
       OR std < --min-std      (default=1e-6)
   - ใช้เพื่อ:
     • ลิสต์ “feature ที่ pred แบน/นิ่ง” ซึ่งมักเป็นสาเหตุที่โมเดลทาย type บางอันหาย
     • แยกสาเหตุ “โมเดลแย่” vs “feature extractor พัง” ได้เร็วมาก

   วิธีใช้จริง:
     • ถ้า feature สำคัญ (เช่น edge_rate, last_edge_pos_ratio, tail_std ฯลฯ) โผล่ในนี้
       -> ปัญหาอยู่ที่ extractor / preprocessing / input waveform (ไม่ใช่ที่ threshold)


4) compare_train_vs_pred.csv
   - เปรียบเทียบ distribution shift ระหว่าง TRAIN และ PRED ต่อ feature
   - มีตัวชี้วัดหลัก:
       z_shift = |mean_pred - mean_train| / (std_train + eps)
     และเช็คช่วง:
       p05_p95_overlap = True/False

   ใช้เพื่อ:
     • หาว่า pred data “หลุด distribution” จาก train ตรงไหน (big z_shift)
     • เป็นตัวชี้ว่าควร retrain ด้วย data เพิ่ม / normalize / clip / แก้ extractor

   วิธีอ่านคอลัมน์:
     - train_mean/train_std/train_p05/train_p95
     - pred_mean/pred_std/pred_p05/pred_p95
     - z_shift:
         ~0-1  : ใกล้เคียง train (โอเค)
         2-3+  : เริ่มน่าห่วง (shift ชัด)
         5-10+ : หลุดโลก (มักทำให้โมเดลทายเพี้ยนหนัก)
     - p05_p95_overlap:
         False = ช่วงค่าหลัก ๆ ของ pred ไม่ทับกับ train เลย => distribution drift แรงมาก

   วิธีใช้จริง:
     • sort ไฟล์นี้ตาม z_shift (สคริปต์ทำให้แล้ว) แล้วดู top-20
     • ถ้า feature ที่เกี่ยวกับ edge/ring/tail shift แรง -> อาจเป็น sampling rate/scale/normalization เปลี่ยน
     • ถ้า feature meta_* shift แรง -> input setting เปลี่ยน หรือคำนวณ reference ผิด


5) high_corr_pairs_train.csv
   - หา pair ของ feature ที่มี |corr| >= --corr-thr (default=0.98) จาก TRAIN
   - ใช้เพื่อ:
     • ลด redundancy: feature ซ้ำกันมาก -> ทำให้โมเดล “สับสน/overfit” หรือ proba แข็งเป็นก้อน
     • ช่วยตัด feature ชุดที่ซ้ำก่อน train ใหม่ (cleaner feature set)

   คอลัมน์:
     - a, b: feature คู่ที่ correlation สูง
     - corr: ค่าสหสัมพันธ์ (ใกล้ 1 หรือ -1 = ซ้ำมาก)

   วิธีใช้จริง:
     • ถ้า pair ซ้ำมาก ให้เลือก drop ตัวหนึ่ง โดยใช้หลัก:
       - ตัวที่ stable กว่า / normalize แล้ว -> เก็บ
       - ตัวที่เป็น p2p/outlier sensitive -> มัก drop
       - ตัวที่ใน pred กลายเป็น flat -> drop หรือแก้ extractor

   หมายเหตุสำคัญ:
     • ถ้า “ไม่มีคู่ที่ถึง threshold” -> ไฟล์อาจว่าง (script handle ด้วย empty dataframe)
     • ถ้าคุณตั้ง corr_thr ต่ำมาก -> pairs จะเยอะ และอาจไม่ useful


6) sanity_suggest.json  (OPTIONAL)
   - จะถูกสร้างเมื่อส่ง:
       --calib-diag  (ไฟล์ calib_diagnostics_*.csv)
       --thr-high, --thr-low
   - ใช้เพื่อ:
     • แนะนำ threshold ของ sanity gate แบบ data-driven
     • อิง “CALIB true-fast” และ “near soft band” (p อยู่ระหว่าง thr_low ถึง thr_high)
     • เหมาะมากสำหรับกัน FP-fast ใน soft zone

   โครงสร้างไฟล์:
     {
       "policy": "soft_zone_fp_firewall",
       "thr_high": ...,
       "thr_low": ...,
       "q_feat": 0.98,
       "q_reg_fp": 0.95,
       "features_present": { ... },
       "thresholds": { feature: value, ... }
     }

   วิธีใช้จริง:
     • เอาค่า thresholds ไปตั้ง sanity gate ใน predict script
     • ถ้า feature ใด features_present=False -> หมายถึง calib_diag ไม่มีคอลัมน์นั้น
     • threshold ของ reg_pred_col (เช่น reg_wait_pred_ms) คือ “firewall” กัน fast แต่ reg พุ่งสูง

---------------------------------------------------------------------------
แนวทางการไล่แก้ปัญหาด้วยรายงานชุดนี้
---------------------------------------------------------------------------
- case : pred ขาด type บางอัน / fast หาย:
    1) เปิด flat_features_pred.csv ดูว่า edge/tail features แบนไหม
    2) เปิด compare_train_vs_pred.csv ดูว่า z_shift ของ feature สำคัญสูงผิดปกติไหม
    3) ถ้าใช่ -> แก้ feature extractor (window, normalization, edge detect)

- case : proba เป็นก้อน 0/1 หรือ unique count น้อย:
    1) เปิด high_corr_pairs_train.csv ตัด redundancy
    2) เปิด summary_train.csv ดู feature scale หลุดโลก (std/max ใหญ่มาก)
    3) drop/transform เฉพาะใน zero_clf (กันโมเดลมั่นใจผิด)

- case : train ดี แต่ pred เพี้ยน:
    1) compare_train_vs_pred.csv (z_shift) คือคำตอบอันดับ 1
    2) ถ้า drift แรง -> ต้อง retrain ด้วย data ใหม่ หรือ normalize ให้เหมือน train
=============================================================================
'''