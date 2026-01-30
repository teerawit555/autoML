# import argparse
# import math
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# def find_wide_value_cols(df: pd.DataFrame, prefix: str = "v") -> list[str]:
#     # เลือกคอลัมน์ v0000..v0999 แบบปลอดภัย
#     cols = [c for c in df.columns if c.startswith(prefix)]
#     # sort ตามเลขท้าย ถ้าเป็น v0001 จะเรียงถูก
#     def key_fn(c):
#         tail = c[len(prefix):]
#         return int(tail) if tail.isdigit() else 10**9
#     cols = sorted(cols, key=key_fn)
#     return cols


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument(
#         "--mode",
#         required=True,
#         choices=["check_train", "check_pred", "check_pred_wide"],
#         help=(
#             "check_train: raw long vs wait_time_ms\n"
#             "check_pred:  raw long vs pred_wait_time_ms\n"
#             "check_pred_wide: wide_v22.csv vs pred_wait_time_ms (ใช้ dt_ms สร้างแกนเวลา)"
#         ),
#     )
#     ap.add_argument("--raw", help="Path to raw LONG CSV (มี wave_id,time/value) สำหรับ check_train/check_pred")
#     ap.add_argument("--wide", help="Path to WIDE CSV (มี wave_id,v0000..) สำหรับ check_pred_wide")
#     ap.add_argument("--result", required=True, help="Path to result CSV (มี wave_id + wait_time_ms/pred_wait_time_ms)")
#     ap.add_argument("--out", default="plots/check_results/result.png")
#     ap.add_argument("--ncols", type=int, default=5)
#     ap.add_argument("--per_page", type=int, default=20)

#     # สำหรับ wide plotting
#     ap.add_argument("--dt-ms", type=float, default=None, help="dt ของ sample (ms). ถ้าไม่ใส่ จะ plot เป็น sample_idx แทน")
#     ap.add_argument("--value-prefix", default="v", help="prefix ของคอลัมน์ wide เช่น v0000.. (default=v)")
#     args = ap.parse_args()

#     # =========================
#     # Load result (pred/label)
#     # =========================
#     print(f"📂 Loading Result Data: {args.result}")
#     res_df = pd.read_csv(args.result)

#     if args.mode == "check_train":
#         target_col = "wait_time_ms"
#         line_color = "royalblue"
#         line_style = "-"
#         title_prefix = "CHECK TRAIN LABEL"
#     else:
#         target_col = "pred_wait_time_ms"
#         line_color = "darkorange"
#         line_style = "--"
#         title_prefix = "CHECK PREDICTION"

#     if target_col not in res_df.columns:
#         raise KeyError(f"Result file missing column: {target_col}")

#     if "wave_id" not in res_df.columns:
#         raise KeyError("Result file missing column: wave_id")

#     # Map target value
#     val_map = dict(zip(res_df["wave_id"], res_df[target_col]))

#     # [NEW] Map flags (Need More Sample)
#     flag_map = {}
#     reason_map = {}
#     if "need_more_sample" in res_df.columns:
#         flag_map = dict(zip(res_df["wave_id"], res_df["need_more_sample"]))
#     if "need_more_reason" in res_df.columns:
#         reason_map = dict(zip(res_df["wave_id"], res_df["need_more_reason"].fillna("")))

#     # =========================
#     # Load raw/wide depending on mode
#     # =========================
#     if args.mode in ["check_train", "check_pred"]:
#         if not args.raw:
#             raise ValueError("--raw is required for check_train/check_pred")
#         print(f"📂 Loading Raw LONG Data: {args.raw}")
#         raw = pd.read_csv(args.raw)

#         if "wave_id" not in raw.columns:
#             raise KeyError("Raw file missing column: wave_id")

#         val_col = "value" if "value" in raw.columns else ("current" if "current" in raw.columns else None)
#         time_col = "time_ms" if "time_ms" in raw.columns else ("time" if "time" in raw.columns else None)
#         if val_col is None or time_col is None:
#             raise KeyError("Raw LONG must contain value/current and time_ms/time")

#         all_wave_ids = sorted(raw["wave_id"].unique())
#         num_pages = math.ceil(len(all_wave_ids) / args.per_page)

#     else:
#         if not args.wide:
#             raise ValueError("--wide is required for check_pred_wide")
#         print(f"📂 Loading WIDE Data: {args.wide}")
#         wide = pd.read_csv(args.wide)

#         if "wave_id" not in wide.columns:
#             raise KeyError("Wide file missing column: wave_id")

#         vcols = find_wide_value_cols(wide, prefix=args.value_prefix)
#         if len(vcols) == 0:
#             raise KeyError(f"No wide value columns found with prefix='{args.value_prefix}' (e.g. v0000..)")

#         all_wave_ids = sorted(wide["wave_id"].unique())
#         num_pages = math.ceil(len(all_wave_ids) / args.per_page)

#         # สร้างแกน X
#         n = len(vcols)
#         if args.dt_ms is None:
#             x_axis = np.arange(n, dtype=float)  # sample index
#             x_label = "sample_idx"
#         else:
#             x_axis = np.arange(n, dtype=float) * float(args.dt_ms)
#             x_label = "time_ms"

#     # =========================
#     # Plot loop
#     # =========================
#     base_out = Path(args.out)
#     base_out.parent.mkdir(parents=True, exist_ok=True)

#     for page in range(num_pages):
#         start_idx = page * args.per_page
#         end_idx = min(start_idx + args.per_page, len(all_wave_ids))
#         batch_ids = all_wave_ids[start_idx:end_idx]

#         nrows = math.ceil(len(batch_ids) / args.ncols)
#         fig, axes = plt.subplots(nrows, args.ncols, figsize=(4 * args.ncols, 3 * nrows))
#         axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

#         print(f"🎨 Drawing {args.mode} - Page {page+1}/{num_pages}...")

#         for i, wid in enumerate(batch_ids):
#             ax = axes[i]

#             # -------- plot waveform --------
#             if args.mode in ["check_train", "check_pred"]:
#                 sub = raw[raw["wave_id"] == wid].sort_values(time_col)
#                 ax.plot(sub[time_col], sub[val_col], color="gray", alpha=0.6, linewidth=0.9)
#                 x_for_vline = time_col
#                 x_vals = sub[time_col].to_numpy(dtype=float)
#                 y_vals = sub[val_col].to_numpy(dtype=float)
#                 ax.set_xlabel("time_ms" if time_col == "time_ms" else time_col, fontsize=8)
#             else:
#                 row = wide[wide["wave_id"] == wid]
#                 if row.empty:
#                     ax.set_title(f"ID:{wid} | missing in wide", fontsize=8)
#                     ax.axis("off")
#                     continue
#                 y = row[vcols].iloc[0].to_numpy(dtype=float)
#                 ax.plot(x_axis, y, color="gray", alpha=0.7, linewidth=0.9)
#                 x_for_vline = x_label
#                 x_vals = x_axis
#                 y_vals = y
#                 ax.set_xlabel(x_label, fontsize=8)

#             # -------- vertical line (label/pred) --------
#             if wid in val_map:
#                 t_val = float(val_map[wid])

#                 if args.mode == "check_pred_wide" and args.dt_ms is None:
#                     ax.text(
#                         0.02, 0.95,
#                         f"⚠ pred={t_val:.3f}ms\n(no dt_ms)",
#                         transform=ax.transAxes,
#                         fontsize=7, va="top",
#                         color=line_color, fontweight="bold",
#                     )
#                 else:
#                     ax.axvline(t_val, color=line_color, linestyle=line_style, linewidth=2)

#                     # สถิติหลังเส้น
#                     post_mask = x_vals >= t_val
#                     if np.any(post_mask):
#                         post_y = y_vals[post_mask]
#                         y_max, y_min = float(np.max(post_y)), float(np.min(post_y))
#                         ax.set_title(f"{title_prefix}\nID:{wid} | Δpost:{(y_max-y_min):.4f}", fontsize=8)
#                     else:
#                         ax.set_title(f"{title_prefix}\nID:{wid}", fontsize=8)

#                     ax.text(
#                         t_val, ax.get_ylim()[1],
#                         f"{t_val:.2f}",
#                         color=line_color, fontsize=7,
#                         ha="center", va="bottom", fontweight="bold",
#                     )
#             else:
#                 ax.set_title(f"{title_prefix}\nID:{wid} | no target", fontsize=8)

#             # -------- [NEW] Show Need More Sample Flag --------
#             if wid in flag_map and flag_map[wid] == 1:
#                 reason = reason_map.get(wid, "unknown")
#                 # ตัดคำ reason ถ้ายาวเกินไป
#                 if len(reason) > 20:
#                     reason = reason[:20] + "..."
                
#                 ax.text(
#                     0.03, 0.85, 
#                     f"⚠ NEED MORE SAMPLE\n({reason})", 
#                     transform=ax.transAxes, 
#                     color="red", 
#                     fontsize=7, 
#                     fontweight="bold",
#                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.2')
#                 )

#             ax.grid(True, alpha=0.2)
#             ax.tick_params(labelsize=7)

#         # ลบแกนว่าง
#         for j in range(i + 1, len(axes)):
#             fig.delaxes(axes[j])

#         plt.tight_layout()
#         save_name = f"{base_out.stem}_{args.mode}_pg{page+1}.png"
#         fig.savefig(base_out.parent / save_name, dpi=130)
#         plt.close(fig)

#     print(f"✨ Done! Graphs saved in {base_out.parent}")


# if __name__ == "__main__":
#     main()

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_wide_value_cols(df: pd.DataFrame, prefix: str = "v") -> list[str]:
    # เลือกคอลัมน์ v0000..v0999 แบบปลอดภัย
    cols = [c for c in df.columns if c.startswith(prefix)]
    # sort ตามเลขท้าย ถ้าเป็น v0001 จะเรียงถูก
    def key_fn(c):
        tail = c[len(prefix):]
        return int(tail) if tail.isdigit() else 10**9
    cols = sorted(cols, key=key_fn)
    return cols


def apply_focus_xlim(ax, x_vals: np.ndarray, t_settle_ms: float | None, focus_ms: float, post_ms: float):
    """
    ถ้า settle <= focus_ms -> show 0..focus_ms
    ถ้า settle >  focus_ms -> show 0..(settle + post_ms)
    clamp ไม่ให้เกินช่วงแกนจริง
    """
    if x_vals is None or len(x_vals) == 0:
        return

    x_min = float(np.min(x_vals))
    x_max = float(np.max(x_vals))

    left = max(0.0, x_min)

    if t_settle_ms is None or (not np.isfinite(t_settle_ms)):
        ax.set_xlim(left, x_max)
        return

    if t_settle_ms <= focus_ms:
        right = focus_ms
    else:
        right = t_settle_ms + post_ms

    right = min(right, x_max)
    if right <= left:
        right = x_max

    ax.set_xlim(left, right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        required=True,
        choices=["check_train", "check_pred", "check_pred_wide"],
        help=(
            "check_train: raw long vs wait_time_ms\n"
            "check_pred:  raw long vs pred_wait_time_ms\n"
            "check_pred_wide: wide_v22.csv vs pred_wait_time_ms (ใช้ dt_ms สร้างแกนเวลา)"
        ),
    )
    ap.add_argument("--raw", help="Path to raw LONG CSV (มี wave_id,time/value) สำหรับ check_train/check_pred")
    ap.add_argument("--wide", help="Path to WIDE CSV (มี wave_id,v0000..) สำหรับ check_pred_wide")
    ap.add_argument("--result", required=True, help="Path to result CSV (มี wave_id + wait_time_ms/pred_wait_time_ms)")
    ap.add_argument("--out", default="plots/check_results/result.png")
    ap.add_argument("--ncols", type=int, default=5)
    ap.add_argument("--per_page", type=int, default=20)

    # สำหรับ wide plotting
    ap.add_argument("--dt-ms", type=float, default=None, help="dt ของ sample (ms). ถ้าไม่ใส่ จะ plot เป็น sample_idx แทน")
    ap.add_argument("--value-prefix", default="v", help="prefix ของคอลัมน์ wide เช่น v0000.. (default=v)")

    # ✅ NEW: focus window controls
    ap.add_argument("--focus-ms", type=float, default=10.0,
                    help="If settle <= focus-ms, show only 0..focus-ms (default 10ms)")
    ap.add_argument("--post-ms", type=float, default=10.0,
                    help="If settle > focus-ms, show 0..(settle+post-ms) (default +10ms)")
    ap.add_argument("--focus-even-without-target", action="store_true",
                    help="If wave has no target value, still focus to 0..focus-ms")

    args = ap.parse_args()

    # =========================
    # Load result (pred/label)
    # =========================
    print(f"📂 Loading Result Data: {args.result}")
    res_df = pd.read_csv(args.result)

    if args.mode == "check_train":
        target_col = "wait_time_ms"
        line_color = "royalblue"
        line_style = "-"
        title_prefix = "CHECK TRAIN LABEL"
    else:
        target_col = "pred_wait_time_ms"
        line_color = "darkorange"
        line_style = "--"
        title_prefix = "CHECK PREDICTION"

    if target_col not in res_df.columns:
        raise KeyError(f"Result file missing column: {target_col}")

    if "wave_id" not in res_df.columns:
        raise KeyError("Result file missing column: wave_id")

    # Map target value
    val_map = dict(zip(res_df["wave_id"], res_df[target_col]))

    # [NEW] Map flags (Need More Sample)
    flag_map = {}
    reason_map = {}
    if "need_more_sample" in res_df.columns:
        flag_map = dict(zip(res_df["wave_id"], res_df["need_more_sample"]))
    if "need_more_reason" in res_df.columns:
        reason_map = dict(zip(res_df["wave_id"], res_df["need_more_reason"].fillna("")))

    # =========================
    # Load raw/wide depending on mode
    # =========================
    if args.mode in ["check_train", "check_pred"]:
        if not args.raw:
            raise ValueError("--raw is required for check_train/check_pred")
        print(f"📂 Loading Raw LONG Data: {args.raw}")
        raw = pd.read_csv(args.raw)

        if "wave_id" not in raw.columns:
            raise KeyError("Raw file missing column: wave_id")

        val_col = "value" if "value" in raw.columns else ("current" if "current" in raw.columns else None)
        time_col = "time_ms" if "time_ms" in raw.columns else ("time" if "time" in raw.columns else None)
        if val_col is None or time_col is None:
            raise KeyError("Raw LONG must contain value/current and time_ms/time")

        all_wave_ids = sorted(raw["wave_id"].unique())
        num_pages = math.ceil(len(all_wave_ids) / args.per_page)

        # sort wave_id by type then wave_id
        # if "type" in raw.columns:
        #     meta = raw[["wave_id", "type"]].drop_duplicates()
        #     meta = meta.sort_values(["type", "wave_id"], kind="mergesort")  # stable sort
        #     all_wave_ids = meta["wave_id"].tolist()
        #     type_map = dict(zip(meta["wave_id"], meta["type"]))
        # else:
        #     all_wave_ids = sorted(raw["wave_id"].unique())
        #     type_map = {}
        # num_pages = math.ceil(len(all_wave_ids) / args.per_page)

    else:
        if not args.wide:
            raise ValueError("--wide is required for check_pred_wide")
        print(f"📂 Loading WIDE Data: {args.wide}")
        wide = pd.read_csv(args.wide)

        if "wave_id" not in wide.columns:
            raise KeyError("Wide file missing column: wave_id")

        vcols = find_wide_value_cols(wide, prefix=args.value_prefix)
        if len(vcols) == 0:
            raise KeyError(f"No wide value columns found with prefix='{args.value_prefix}' (e.g. v0000..)")

        all_wave_ids = sorted(wide["wave_id"].unique())
        num_pages = math.ceil(len(all_wave_ids) / args.per_page)

        # สร้างแกน X
        n = len(vcols)
        if args.dt_ms is None:
            # ⚠ focus แบบ ms ทำไม่ได้ถ้าไม่มี dt_ms (เพราะแกนเป็น sample_idx)
            x_axis = np.arange(n, dtype=float)  # sample index
            x_label = "sample_idx"
        else:
            x_axis = np.arange(n, dtype=float) * float(args.dt_ms)
            x_label = "time_ms"

    # =========================
    # Plot loop
    # =========================
    base_out = Path(args.out)
    base_out.parent.mkdir(parents=True, exist_ok=True)

    for page in range(num_pages):
        start_idx = page * args.per_page
        end_idx = min(start_idx + args.per_page, len(all_wave_ids))
        batch_ids = all_wave_ids[start_idx:end_idx]

        nrows = math.ceil(len(batch_ids) / args.ncols)
        fig, axes = plt.subplots(nrows, args.ncols, figsize=(4 * args.ncols, 3 * nrows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        print(f"🎨 Drawing {args.mode} - Page {page+1}/{num_pages}...")

        for i, wid in enumerate(batch_ids):
            ax = axes[i]

            # -------- plot waveform --------
            if args.mode in ["check_train", "check_pred"]:
                sub = raw[raw["wave_id"] == wid].sort_values(time_col)
                ax.plot(sub[time_col], sub[val_col], color="gray", alpha=0.6, linewidth=0.9)
                x_vals = sub[time_col].to_numpy(dtype=float)
                y_vals = sub[val_col].to_numpy(dtype=float)
                ax.set_xlabel("time_ms" if time_col == "time_ms" else time_col, fontsize=8)
            else:
                row = wide[wide["wave_id"] == wid]
                if row.empty:
                    ax.set_title(f"ID:{wid} | missing in wide", fontsize=8)
                    ax.axis("off")
                    continue
                y = row[vcols].iloc[0].to_numpy(dtype=float)
                ax.plot(x_axis, y, color="gray", alpha=0.7, linewidth=0.9)
                x_vals = np.asarray(x_axis, dtype=float)
                y_vals = y
                ax.set_xlabel(x_label, fontsize=8)

            # -------- vertical line (label/pred) + focus window --------
            if wid in val_map:
                t_val = float(val_map[wid])

                # wide แต่ไม่มี dt_ms -> วาดเส้นเป็น ms ไม่ตรงแกน sample_idx
                if args.mode == "check_pred_wide" and args.dt_ms is None:
                    ax.text(
                        0.02, 0.95,
                        f"⚠ {target_col}={t_val:.3f}ms\n(no --dt-ms, x=sample_idx)",
                        transform=ax.transAxes,
                        fontsize=7, va="top",
                        color=line_color, fontweight="bold",
                    )
                    # focus แบบ ms ทำไม่ได้ -> ปล่อย full range
                else:
                    ax.axvline(t_val, color=line_color, linestyle=line_style, linewidth=2)

                    # ✅ NEW: focus xlim ตาม settle
                    apply_focus_xlim(ax, x_vals, t_val, args.focus_ms, args.post_ms)

                    # สถิติหลังเส้น
                    post_mask = x_vals >= t_val
                    if np.any(post_mask):
                        post_y = y_vals[post_mask]
                        y_max, y_min = float(np.max(post_y)), float(np.min(post_y))
                        ax.set_title(f"{title_prefix}\nID:{wid} | Δpost:{(y_max-y_min):.4f}", fontsize=8)
                    else:
                        ax.set_title(f"{title_prefix}\nID:{wid}", fontsize=8)

                    ax.text(
                        t_val, ax.get_ylim()[1],
                        f"{t_val:.2f}",
                        color=line_color, fontsize=7,
                        ha="center", va="bottom", fontweight="bold",
                    )
            else:
                ax.set_title(f"{title_prefix}\nID:{wid} | no target", fontsize=8)

                # ✅ optional: focus even when no target
                if args.focus_even_without_target:
                    apply_focus_xlim(ax, x_vals, 0.0, args.focus_ms, args.post_ms)

            # -------- Show Need More Sample Flag --------
            if wid in flag_map and flag_map[wid] == 1:
                reason = reason_map.get(wid, "unknown")
                if len(reason) > 20:
                    reason = reason[:20] + "..."
                ax.text(
                    0.03, 0.85,
                    f"⚠ NEED MORE SAMPLE\n({reason})",
                    transform=ax.transAxes,
                    color="red",
                    fontsize=7,
                    fontweight="bold",
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.2')
                )

            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=7)

        # ลบแกนว่าง
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        save_name = f"{base_out.stem}_{args.mode}_pg{page+1}.png"
        fig.savefig(base_out.parent / save_name, dpi=130)
        plt.close(fig)

    print(f"✨ Done! Graphs saved in {base_out.parent}")


if __name__ == "__main__":
    main()
