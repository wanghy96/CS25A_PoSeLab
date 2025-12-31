# plotting.py
from __future__ import annotations
import csv
from typing import Dict, List

def save_metrics_csv(records: List[Dict], csv_path: str) -> None:
    if not records:
        return
    keys = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(records)

def plot_metrics(records: List[Dict], png_path: str) -> None:
    """
    Save a PNG plot of metrics vs time.
    Uses matplotlib (stable for packaging).
    """
    if not records:
        return

    import matplotlib
    matplotlib.use("Agg")  # headless safe for packaging
    import matplotlib.pyplot as plt

    t = [r["time_s"] for r in records]

    # 你关心哪些指标就画哪些（可自行增删）
    keys_to_plot = ["FHA", "FHP", "SBA", "PPA", "PPT", "TKA", "total_score"]

    plt.figure(figsize=(12, 6))
    for k in keys_to_plot:
        if k not in records[0]:
            continue
        plt.plot(t, [r.get(k, 0.0) for r in records], label=k)

    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()