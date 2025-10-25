
# E6_heatmap.py — Atom-Free Pass/Alert visualization
import json, os, math, numpy as np, matplotlib.pyplot as plt

def load_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def group_by_kernel(data):
    from collections import defaultdict
    by_kernel = defaultdict(list)
    for r in data:
        by_kernel[r["kernel"]].append(r)
    return by_kernel

def plot_heatmaps(by_kernel, out_dir="figs"):
    os.makedirs(out_dir, exist_ok=True)
    for kernel, recs in by_kernel.items():
        recs_sorted = sorted(recs, key=lambda r: float(r.get("tau", 0.0)))
        taus = np.array([float(r.get("tau", 0.0)) for r in recs_sorted])
        lam = np.array([math.log10(max(r.get("lambda_min", 0.0), 1e-300)) for r in recs_sorted])
        ent = np.array([r.get("entropy", 0.0) for r in recs_sorted])
        cond = np.array([math.log10(max(r.get("cond_est", 1.0), 1.0)) for r in recs_sorted])
        metrics = [("lambda_min_log10", lam), ("entropy", ent), ("cond_est_log10", cond)]
        for name, arr in metrics:
            fig, ax = plt.subplots(figsize=(8, 1.8))
            im = ax.imshow(arr.reshape(1, -1), aspect="auto")
            ax.set_title(f"{kernel} — {name}")
            ax.set_xticks(range(len(taus)))
            ax.set_xticklabels([f"{t:.2f}" for t in taus], rotation=45, ha="right")
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            out = os.path.join(out_dir, f"{kernel}_{name}.png")
            plt.savefig(out, dpi=160)
            plt.close(fig)
            print(f"[E6] Saved {out}")

if __name__ == "__main__":
    json_path = "results/results.json"
    if not os.path.exists(json_path):
        print("[E6] results.json not found, please run v20_runner.py first.")
    else:
        data = load_results(json_path)
        grouped = group_by_kernel(data)
        plot_heatmaps(grouped, out_dir="figs")
        print("[E6] Heatmaps complete.")
