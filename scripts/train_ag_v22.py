from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor

# ปรับได้: meta ที่ไม่อยากให้โมเดลใช้ (ถ้าไม่มีในไฟล์ก็ไม่เป็นไร)
COLS_TO_DROP = ["force_mA", "range_V", "temp_C"]
DROP_ALWAYS = ["type"]   # จะเพิ่มอย่างอื่นก็ได้ถ้าไม่ stable


ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # ✅ FIX
DEFAULT_SAVE_PATH = f"AutogluonModels/ag-v22-{ts}"
FAST_MS = 0.1

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text))


def append_log(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def log_print(log_path: str, msg: str) -> None:
    print(msg)
    append_log(log_path, msg)


def align_columns(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in required_cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[required_cols]


def proba_class1(p) -> np.ndarray:
    if hasattr(p, "columns") and (1 in list(p.columns)):
        return p[1].to_numpy()
    arr = np.asarray(p)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return arr.astype(float)


def threshold_scan_report(
    proba_is_fast: np.ndarray,
    wait_pred: np.ndarray,
    y_true: np.ndarray,
    thr_min: float = 0.10,
    thr_max: float = 0.90,
    steps: int = 161,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    is_fast_true = (y_true <= FAST_MS + 1e-12)


    rows = []
    for thr in np.linspace(thr_min, thr_max, steps):
        is_fast_pred = proba_is_fast >= thr
        y_hat = np.where(is_fast_pred, FAST_MS, wait_pred)

        mae_all = float(np.mean(np.abs(y_hat - y_true)))

        # zero confusion
        zp = is_fast_pred
        zt = is_fast_true
        tp = int(np.sum(zp & zt))          # true fast predicted fast
        fn = int(np.sum((~zp) & zt))       # true fast predicted slow (BAD)
        fp = int(np.sum(zp & (~zt)))       # true slow predicted fast
        tn = int(np.sum((~zp) & (~zt)))

        fast_fn_rate = fn / max(int(np.sum(zt)), 1)
        fast_recall = tp / max(int(np.sum(zt)), 1)

        rows.append(
            {
                "thr": float(thr),
                "mae_all": mae_all,
                "fast_recall": float(fast_recall),
                "fast_fn_rate": float(fast_fn_rate),
                "TP_zero": tp,
                "FN_zero": fn,
                "FP_zero": fp,
                "TN_nonzero": tn,
                "pred_fast": int(np.sum(zp)),
            }
        )
    return pd.DataFrame(rows)

def pick_threshold_prefer_zero(scan: pd.DataFrame) -> tuple[float, float, str]:
    """
    เลือก threshold โดย "ห้ามพลาด zero" ก่อน
    1) หาแถวที่ fast_fn_rate == 0 แล้วเลือก mae_all ต่ำสุด
    2) ถ้าไม่มีเลย (rare) ค่อย fallback เป็น mae_all ต่ำสุดทั้งตาราง
    คืนค่า: (thr, mae, reason)
    """
    s = scan.copy()

    # กัน floating error
    zero_safe = s[s["fast_fn_rate"] <= 1e-12].copy()
    if len(zero_safe) > 0:
        row = zero_safe.sort_values("mae_all", ascending=True).iloc[0]
        return float(row["thr"]), float(row["mae_all"]), "zero_safe(best_mae)"

    row = s.sort_values("mae_all", ascending=True).iloc[0]
    return float(row["thr"]), float(row["mae_all"]), "mae_best(fallback)"


def pick_best_zero_threshold(
    proba_is_fast: np.ndarray,
    wait_pred: np.ndarray,
    y_true: np.ndarray,
    thr_min: float = 0.20,
    thr_max: float = 0.80,
    steps: int = 121,
) -> tuple[float, float]:
    best_thr = 0.5
    best_mae = float("inf")

    thr_grid = np.linspace(thr_min, thr_max, steps)
    for thr in thr_grid:
        is_fast = proba_is_fast >= thr
        y_hat = np.where(is_fast, FAST_MS, wait_pred)
        mae = float(np.mean(np.abs(y_hat - y_true)))
        if mae < best_mae:
            best_mae = mae
            best_thr = float(thr)

    return best_thr, best_mae


def train(
    data_path: str,
    label: str,
    model_dir: str | None,
    presets: str,
    time_limit: int,
) -> None:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Input file not found: "{data_path}"')

    df = pd.read_csv(data_path)

    if "wave_id" not in df.columns:
        df["wave_id"] = np.arange(len(df), dtype=int)

    # train ต้องมี label
    df_train = df.dropna(subset=[label]).reset_index(drop=True)
    if len(df_train) == 0:
        raise ValueError(f"No rows with label '{label}' found in training file.")

    # binary label สำหรับ stage A
    #df_train["is_fast"] = (df_train[label].astype(float) == 0.0).astype(int)
    df_train["is_fast"] = (df_train[label].astype(float) <= FAST_MS + 1e-12).astype(int)

    df_train = df_train.drop(columns=DROP_ALWAYS, errors="ignore")

    u = df_train["is_fast"].unique()
    if len(u) < 2:
        raise ValueError(
            f"is_fast has only 1 class in training data: {u}. "
            f"ต้องมีทั้งเคส wait=0 และ wait>0 ในชุดเทรน"
        )

    save_path = model_dir or DEFAULT_SAVE_PATH
    zero_model_path = os.path.join(save_path, "zero_clf")
    reg_model_path = os.path.join(save_path, "wait_reg")
    _ensure_dir(save_path)

    # log file path
    log_path = os.path.join(save_path, f"train_log_{ts}.txt")
    append_log(log_path, f"=== TRAIN LOG START {ts} ===")

    gpu_count = 1 if torch.cuda.is_available() else 0
    log_print(log_path, f"🚀 Training device: {'GPU (CUDA)' if gpu_count > 0 else 'CPU'}")
    log_print(log_path, f"data_path={data_path}")
    log_print(log_path, f"rows={len(df_train)} cols={len(df_train.columns)}")
    log_print(log_path, f"presets={presets} time_limit={time_limit}s")

    # distribution
    z_cnt = int((df_train["is_fast"] == 1).sum())
    nz_cnt = int((df_train["is_fast"] == 0).sum())
    log_print(log_path, f"label distribution: zero={z_cnt} nonzero={nz_cnt} zero_ratio={z_cnt/max(len(df_train),1):.4f}")

    # แบ่งเวลาเทรน 2-stage
    t_zero = max(30, int(time_limit * 0.25))
    t_reg = max(60, int(time_limit * 0.75))
    log_print(log_path, f"stage time: zero={t_zero}s reg={t_reg}s")

    # drop meta cols
    cols_to_drop_found = [c for c in COLS_TO_DROP if c in df_train.columns]
    if cols_to_drop_found:
        log_print(log_path, f"Dropping meta columns: {cols_to_drop_found}")

    # NOTE: log if 'type' exists (มันมักทำให้โมเดลพึ่ง metadata)
    if "type" in df_train.columns:
        nunq = int(df_train["type"].nunique(dropna=False))
        log_print(log_path, f"⚠ column 'type' detected (n_unique incl NaN = {nunq}). If pred has no type, consider dropping it.")

    # =========================
    # Stage A: Zero Classifier
    # =========================
    log_print(log_path, "\n=== Stage A: zero_clf (binary) ===")

    #df_train = df_train.drop(columns=DROP_ALWAYS, errors="ignore")

    

    # drop ring features for classifier only
    # ring_cols = [c for c in df_zero.columns if c.startswith("ring_")]
    # df_zero = df_zero.drop(columns=ring_cols, errors="ignore")

    # กัน leakage: ห้ามให้ classifier เห็น label จริง + wave_id
    df_zero = df_train.drop(columns=cols_to_drop_found, errors="ignore").copy()
    df_zero = df_zero.drop(columns=[label, "wave_id"], errors="ignore")

    ZERO_DROP_EXTRA = [
        "min_pos_ratio","min_time","min_to_end_norm","rebound_norm",
        "post_min_slope_norm","tail_creep_norm",
    ]
    df_zero = df_zero.drop(columns=ZERO_DROP_EXTRA, errors="ignore")

    # ✅ save หลัง drop
    zero_feature_cols = [c for c in df_zero.columns if c != "is_fast"]
    save_json(os.path.join(save_path, "zero_feature_cols.json"), zero_feature_cols)



    zero_clf = TabularPredictor(
        label="is_fast",
        path=zero_model_path,
        problem_type="binary",
        eval_metric="f1",
        verbosity=2,
    ).fit(
        train_data=df_zero,
        presets=presets,
        time_limit=t_zero,
        num_gpus=gpu_count,
        dynamic_stacking=False,
    )

    # leaderboard stage A (บน train data เดิม)
        # leaderboard stage A
    try:
        lb0 = zero_clf.leaderboard(df_zero, silent=True)
        lb0_path = os.path.join(save_path, f"leaderboard_zero_{ts}.csv")
        lb0.to_csv(lb0_path, index=False)
        log_print(log_path, f"saved: {lb0_path}")
        log_print(log_path, "zero_clf top models:\n" + lb0[["model", "score_val"]].head(5).to_string(index=False))
    except Exception as e:
        log_print(log_path, f"leaderboard zero_clf failed: {e}")

    # fit_summary stage A  ✅ ต้องอยู่นอก except
    try:
        fs0 = zero_clf.fit_summary(verbosity=0)
        fs0_path = os.path.join(save_path, f"fit_summary_zero_{ts}.json")
        save_json(fs0_path, fs0)
        log_print(log_path, f"saved: {fs0_path}")
    except Exception as e:
        log_print(log_path, f"fit_summary zero_clf failed: {e}")

    # =========================
    # Stage B: Wait Regressor (non-zero only)
    # =========================
    log_print(log_path, "\n=== Stage B: wait_reg (regression on non-zero) ===")

    #df_slow = df_train[df_train[label].astype(float) > 0].copy()
    df_slow = df_train[df_train[label].astype(float) > FAST_MS].copy()

    log_print(log_path, f"non-zero rows for regressor = {len(df_slow)}")
    if len(df_slow) < 5:
        raise ValueError(
            f"non-zero samples too few ({len(df_slow)}). "
            f"ต้องมีเคส wait>0 มากกว่านี้เพื่อเทรน regressor"
        )

    df_slow["wait_time_log"] = np.log1p(df_slow[label].astype(float))

    df_reg = df_slow.drop(columns=cols_to_drop_found, errors="ignore").copy()
    df_reg_for_fit = df_reg.drop(columns=["wave_id", label, "is_fast"], errors="ignore")

    reg_feature_cols = [c for c in df_reg_for_fit.columns if c != "wait_time_log"]
    save_json(os.path.join(save_path, "reg_feature_cols.json"), reg_feature_cols)
    log_print(log_path, f"wait_reg feature cols = {len(reg_feature_cols)} (saved reg_feature_cols.json)")

    wait_reg = TabularPredictor(
        label="wait_time_log",
        path=reg_model_path,
        problem_type="regression",
        eval_metric="mean_absolute_error",
        verbosity=2,
    ).fit(
        train_data=df_reg_for_fit,
        presets=presets,
        time_limit=t_reg,
        num_gpus=gpu_count,
        dynamic_stacking=False,
    )

    try:
        lbr = wait_reg.leaderboard(df_reg_for_fit, silent=True)
        lbr_path = os.path.join(save_path, f"leaderboard_reg_{ts}.csv")
        lbr.to_csv(lbr_path, index=False)
        log_print(log_path, f"saved: {lbr_path}")
        keep = [c for c in ["model", "score_val", "pred_time_val", "fit_time"] if c in lbr.columns]
        log_print(log_path, "wait_reg top models:\n" + lbr[keep].head(5).to_string(index=False))
    except Exception as e:
        log_print(log_path, f"leaderboard wait_reg failed: {e}")

    # fit_summary stage B
    try:
        fsr = wait_reg.fit_summary(verbosity=0)
        fsr_path = os.path.join(save_path, f"fit_summary_reg_{ts}.json")
        save_json(fsr_path, fsr)
        log_print(log_path, f"saved: {fsr_path}")
    except Exception as e:
        log_print(log_path, f"fit_summary wait_reg failed: {e}")


    # =========================
    # Threshold selection + diagnostic
    # =========================
    log_print(log_path, "\n=== Threshold selection & diagnostics ===")

    X_all = df_train.drop(columns=[label], errors="ignore").copy()
    y_true = df_train[label].to_numpy(dtype=float)

    X_feat = X_all.drop(columns=cols_to_drop_found, errors="ignore").copy()

    # zero proba
    X_zero_in = X_feat.drop(columns=["wave_id"], errors="ignore").copy()
    X_zero_in = align_columns(X_zero_in, zero_feature_cols)
    p = zero_clf.predict_proba(X_zero_in)
    proba_is_fast = proba_class1(p)

    # reg pred (predict ทุกแถว เพื่อ optimize threshold)
    X_reg_in = X_feat.drop(columns=["wave_id", "is_fast", "wait_time_ms", "wait_time_log"], errors="ignore").copy()
    X_reg_in = align_columns(X_reg_in, reg_feature_cols)
    wait_log_pred = wait_reg.predict(X_reg_in)
    wait_pred = np.expm1(np.asarray(wait_log_pred, dtype=float))
    wait_pred = np.clip(wait_pred, 0, None)

    # scan report
    scan = threshold_scan_report(proba_is_fast, wait_pred, y_true, thr_min=0.10, thr_max=0.90, steps=161)
    scan_path = os.path.join(save_path, f"threshold_scan_{ts}.csv")
    scan.to_csv(scan_path, index=False)
    log_print(log_path, f"saved: {scan_path}")

    # pick best by MAE (เดิม)
    # best_thr, best_mae = pick_best_zero_threshold(proba_is_fast, wait_pred, y_true)
    # save_text(os.path.join(save_path, "zero_threshold.txt"), f"{best_thr:.6f}")

    best_thr, best_mae, thr_reason = pick_threshold_prefer_zero(scan)
    save_text(os.path.join(save_path, "zero_threshold.txt"), f"{best_thr:.6f}")
    log_print(log_path, f"threshold_pick_reason = {thr_reason}")


    # show a few candidate thresholds
    top_mae = scan.sort_values("mae_all", ascending=True).head(10)
    log_print(log_path, "\nTop 10 thresholds by MAE:\n" + top_mae[["thr", "mae_all", "fast_recall", "fast_fn_rate", "pred_fast"]].to_string(index=False))

    # also show thresholds with high zero recall (useful for sine/glitch)
    top_recall = scan.sort_values(["fast_recall", "mae_all"], ascending=[False, True]).head(10)
    log_print(log_path, "\nTop 10 thresholds by ZERO RECALL (then MAE):\n" + top_recall[["thr", "fast_recall", "fast_fn_rate", "mae_all", "pred_fast"]].to_string(index=False))

    # compute final train metrics with selected threshold
    is_fast_pred = proba_is_fast >= best_thr
    y_hat = np.where(is_fast_pred, FAST_MS, wait_pred)

    mae_all = float(np.mean(np.abs(y_hat - y_true)))
    nonzero_mask = y_true > FAST_MS + 1e-12
    mae_nz = float(np.mean(np.abs(y_hat[nonzero_mask] - y_true[nonzero_mask]))) if np.any(nonzero_mask) else float("nan")

    zt = (y_true <= FAST_MS + 1e-12)

    fn_zero = int(np.sum((~is_fast_pred) & zt))
    zero_total = int(np.sum(zt))
    fast_fn_rate = fn_zero / max(zero_total, 1)

    log_print(log_path, f"\nSelected threshold (MAE-best) = {best_thr:.3f}")
    log_print(log_path, f"Train MAE(all)={mae_all:.4f} | MAE(nonzero)={mae_nz:.4f}")
    log_print(log_path, f"Zero FN rate={fast_fn_rate:.4f} (FN={fn_zero} / total_zero={zero_total})")
    log_print(log_path, f"proba_is_fast min/mean/max = {proba_is_fast.min():.6f} {proba_is_fast.mean():.6f} {proba_is_fast.max():.6f}")
    log_print(log_path, f"predicted zeros = {int(is_fast_pred.sum())} / {len(is_fast_pred)}")

    # save detailed diagnostics per row
    diag = df_train[["wave_id", label, "is_fast"]].copy()
    diag["proba_is_fast"] = proba_is_fast
    diag["reg_wait_pred_ms"] = wait_pred
    diag["pred_is_fast"] = is_fast_pred.astype(int)
    diag["pred_wait_ms"] = y_hat
    diag["abs_error"] = np.abs(diag["pred_wait_ms"].to_numpy(dtype=float) - diag[label].to_numpy(dtype=float))

    diag_path = os.path.join(save_path, f"train_diagnostics_{ts}.csv")
    diag.sort_values("abs_error", ascending=False).to_csv(diag_path, index=False)
    log_print(log_path, f"saved: {diag_path}")
    log_print(log_path, "\nWorst 15 rows (by abs_error):\n" + diag.sort_values("abs_error", ascending=False).head(15).to_string(index=False))

    log_print(log_path, "\n✅ Saved models at: " + save_path)
    log_print(log_path, f" - zero_clf: {zero_model_path}")
    log_print(log_path, f" - wait_reg: {reg_model_path}")
    log_print(log_path, f" - zero_threshold: {best_thr:.3f} (train MAE={best_mae:.4f})")
    log_print(log_path, f" - log file: {log_path}")
    log_print(log_path, "=== TRAIN LOG END ===")


def main():
    ap = argparse.ArgumentParser("Train AutoGluon v22 (2-stage: zero clf + wait reg) + logging")
    ap.add_argument("--data", required=True, help="train features csv (must include wait_time_ms)")
    ap.add_argument("--label", default="wait_time_ms")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--presets", default="medium_quality")  # medium_quality / high_quality / best_quality

    args = ap.parse_args()
    train(
        data_path=args.data,
        label=args.label,
        model_dir=args.model_dir,
        presets=args.presets,
        time_limit=args.time_limit,
    )


if __name__ == "__main__":
    main()
