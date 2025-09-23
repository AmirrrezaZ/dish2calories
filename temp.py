
from __future__ import annotations
import re, ast
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config ----------------
config = {
    "log_path": "logs/logs.txt",        # path to training log
    "csv_out": "logs/losses.csv",     # "" to disable CSV export
    "smooth": 1,                        # moving average window (epochs); 1 = raw
    "out_png": "logs/train_overview.png",  # PNG path for the single figure
}

# --- Regex to capture epoch lines (robust to whitespace / sci notation) ---
FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
pat = re.compile(
    rf"Epoch\s+(\d+)\s*\|\s*"
    rf"Train\s*Loss:\s*({FLOAT})\s*,\s*"
    rf"Val\s*Loss:\s*({FLOAT})\s*,\s*"
    rf"Train\s*Loss\s*Dict:\s*({{.*?}})\s*,\s*"
    rf"Val\s*Loss\s*Dict:\s*({{.*?}})",
    flags=re.DOTALL
)

def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _normalize_dict(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str)):
            out[k] = _to_float(v) if isinstance(v, str) else float(v)
    return out

def parse_log_text(text: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for m in pat.finditer(text):
        epoch = int(m.group(1))
        try:
            td = _normalize_dict(ast.literal_eval(m.group(4)))
            vd = _normalize_dict(ast.literal_eval(m.group(5)))
        except Exception:
            continue
        rows.append({
            "epoch": epoch,
            "train_total": td.get("total", float("nan")),
            "val_total":   vd.get("total", float("nan")),
            "train_cls":   td.get("cls",   float("nan")),
            "val_cls":     vd.get("cls",   float("nan")),
            "train_mass":  td.get("mass",  float("nan")),
            "val_mass":    vd.get("mass",  float("nan")),
        })
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)

def moving_average(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=max(1, window // 2)).mean()

def _plot_pair(ax, x, y_train, y_val, title: str):
    """markers on, grid on."""
    ax.plot(x, y_train, label="train", linestyle="-", marker="o", markersize=4)
    ax.plot(x, y_val,   label="val",   linestyle="-", marker="o", markersize=4)
    ax.set_title(title)
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

def _annotate_best_on_totals(ax, df: pd.DataFrame):
    """ annotate best epoch by lowest val_total."""
    if "val_total" not in df or df["val_total"].isna().all():
        return
    idx = int(df["val_total"].idxmin())
    best_epoch = int(df.loc[idx, "epoch"])
    best_val   = float(df.loc[idx, "val_total"])
    ax.axvline(best_epoch, linestyle="--", alpha=0.6)
    ax.annotate(
        f"best val_total @ {best_epoch}\n{best_val:.4g}",
        xy=(best_epoch, best_val),
        xytext=(5, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.3),
        arrowprops=dict(arrowstyle="->", alpha=0.6),
    )

def plot_all_in_one(df: pd.DataFrame, smooth: int, out_png: str):
    if df.empty:
        print("No epochs parsed. Nothing to plot.")
        return

    # Smooth series
    x = df["epoch"]
    train_total_s = moving_average(df["train_total"], smooth)
    val_total_s   = moving_average(df["val_total"],   smooth)
    train_cls_s   = moving_average(df["train_cls"],   smooth)
    val_cls_s     = moving_average(df["val_cls"],     smooth)
    train_mass_s  = moving_average(df["train_mass"],  smooth)
    val_mass_s    = moving_average(df["val_mass"],    smooth)

    # Layout: one figure, 3 stacked subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)
    _plot_pair(axes[0], x, train_total_s, val_total_s, "Total Loss (train vs val)")
    _annotate_best_on_totals(axes[0], df)
    _plot_pair(axes[1], x, train_cls_s,   val_cls_s,   "Classification Loss (train vs val)")
    _plot_pair(axes[2], x, train_mass_s,  val_mass_s,  "Mass Loss (train vs val)")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()

    # Always save PNG
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved overview figure → {out_path}")

    plt.show()

def summarize(df: pd.DataFrame) -> None:
    if df.empty:
        print("No data parsed.")
        return
    idx = int(df["val_total"].idxmin())
    best_epoch = int(df.loc[idx, "epoch"])
    best_val   = float(df.loc[idx, "val_total"])
    print(f"Best Val Total: {best_val:.6g} at epoch {best_epoch}")
    for col in ("val_cls", "val_mass"):
        if col in df and not df[col].isna().all():
            print(f"  {col} at best epoch: {df.loc[idx, col]:.6g}")

# ---------------- Main ----------------
if __name__ == "__main__":
    text = Path(config["log_path"]).read_text(encoding="utf-8", errors="ignore")
    df = parse_log_text(text)

    if config["csv_out"]:
        Path(config["csv_out"]).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config["csv_out"], index=False)
        print(f"Wrote parsed data → {config['csv_out']}")

    plot_all_in_one(df, smooth=max(1, config["smooth"]), out_png=config["out_png"])
    summarize(df)
