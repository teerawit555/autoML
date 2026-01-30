import pandas as pd

pred = pd.read_csv("data/processed/prediction/pred_v_TT.csv")
miss = pred[(pred.type_debug=="type2_HARD") & (pred.pred_is_fast==0)]
print("miss count =", len(miss))
print(miss[["wave_id","proba_is_fast","fast_zone","reg_wait_pred_ms"]])


X = pd.read_csv("data/processed/inference/wide_v_TT.csv")
pred = pd.read_csv("data/processed/prediction/pred_v_TT.csv")

df = X.merge(pred[["wave_id","type_debug","pred_is_fast","proba_is_fast"]], on="wave_id")

cols = [
  "meta_step_to_span","tail_std",
  #"late_activity",
  "last_edge_pos_ratio","base_max_slope",
  "edge_rate","edge_max_ratio","tail_p2p","tail_mean_abs_slope","ring_peak_count"
]

good = df[(df.type_debug=="type2_HARD") & (df.pred_is_fast==1)][cols].describe()
bad  = df[(df.type_debug=="type2_HARD") & (df.pred_is_fast==0)][cols].describe()

print("=== FAST type2_HARD (93) ===\n", good)
print("=== MISSED type2_HARD (7) ===\n", bad)


wide = pd.read_csv("data/processed/inference/wide_v_TT.csv")
pred = pd.read_csv("data/processed/prediction/pred_v_TT.csv")

miss_ids = [317,345,430,441,1202,1204,1885]
m = wide[wide["wave_id"].isin(miss_ids)].copy()

cols = [
    "wave_id",
    "edge_count","edge_thr","edge_rate",
    "first_edge_pos_ratio","last_edge_pos_ratio",
    "edge_span_ratio","edge_density",
    "tail_edge_count","tail_edge_rate",
    "edge_max_ratio",
    "meta_step_to_span","tail_std","base_max_slope",
    "ring_peak_count",
    "glitch_phase",
]
print(m[cols].to_string(index=False))