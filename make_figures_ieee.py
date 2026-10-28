"""Single-column (IEEE) versions of the ablation figures: the two dataset panels are
stacked vertically so each figure fits one 3.5-inch column with readable text.

    python make_figures_ieee.py   -> figures/ieee_fig_{hash,alpha,depth}.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

R = Path("results_final")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DS = {"scifact": "SciFact", "nfcorpus": "NFCorpus"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 7.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 4,
})
main = pd.read_csv(R / "main_table.csv").set_index(["dataset", "key"])["ndcg@10"]


def two_rows():
    return plt.subplots(2, 1, figsize=(3.45, 4.3))


# hash size
abl = pd.read_csv("results_ablation/hash_ablation.csv")
abl = abl[abl.signed]
fig, axes = two_rows()
for ax, (ds, title) in zip(axes, DS.items()):
    for (label, c), mk in zip([("word 1", BLUE), ("word 1-2", ORANGE), ("word 1-3", AQUA),
                               ("word 1 + char 3-5", YELLOW)], ["o", "s", "^", "D"]):
        s = abl[(abl.dataset == ds) & (abl.features == label)].sort_values("bits")
        ax.plot(s.bits, s["ndcg@10"], color=c, marker=mk, label=label)
    ref = main[(ds, "bm25-stem")]
    ax.axhline(ref, color=INK2, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.annotate(f"BM25-stem {ref:.3f}", (10, ref), xytext=(0, 2), textcoords="offset points",
                fontsize=6.5, color=INK2)
    ax.set_xticks([10, 12, 14, 16, 18], ["$2^{10}$", "$2^{12}$", "$2^{14}$", "$2^{16}$", "$2^{18}$"])
    ax.set_title(title, loc="left")
    ax.set_ylabel("nDCG@10")
axes[1].set_xlabel("Number of hash buckets")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.07), fontsize=6.8)
fig.tight_layout()
fig.savefig(OUT / "ieee_fig_hash.pdf", bbox_inches="tight")
plt.close(fig)

# fusion weight
sw = pd.read_csv(R / "fusion_alpha_sweep.csv")
fig, axes = two_rows()
for ax, (ds, title) in zip(axes, DS.items()):
    s = sw[(sw.dataset == ds) & (sw.dense == "bge-small")]
    ax.plot(s.alpha, s["ndcg@10_tune"], color=BLUE, label="Tuning queries")
    ax.plot(s.alpha, s["ndcg@10_test"], color=ORANGE, label="Test queries")
    ch = s[s.chosen].iloc[0]
    ax.axvline(ch.alpha, color=INK2, linestyle=(0, (4, 3)), linewidth=0.9)
    lo = min(s["ndcg@10_tune"].min(), s["ndcg@10_test"].min())
    ax.annotate(f"chosen $\\alpha$ = {ch.alpha:.2f}\ntest {ch['ndcg@10_test']:.3f}", (ch.alpha, lo),
                xytext=(-4, 2), textcoords="offset points", ha="right", va="bottom",
                fontsize=6.5, color=INK2)
    ax.set_title(title, loc="left")
    ax.set_ylabel("nDCG@10")
axes[1].set_xlabel("$\\alpha$  (0 = BM25-stem only, 1 = bge-small only)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.05), fontsize=6.8)
fig.tight_layout()
fig.savefig(OUT / "ieee_fig_alpha.pdf", bbox_inches="tight")
plt.close(fig)

# re-ranking depth
dep = pd.read_csv(R / "rerank_depth.csv")
base = {"BM25 + stemming": ("bm25-stem", "BM25-stem"),
        "RRF, BM25-stem + bge-small": ("hybrid-rrf-bge-small", "RRF (BM25-stem + bge)")}
fig, axes = two_rows()
for ax, (ds, title) in zip(axes, DS.items()):
    for (fs, (key, lab)), c in zip(base.items(), [BLUE, ORANGE]):
        s = dep[(dep.dataset == ds) & (dep.first_stage == fs)].sort_values("depth")
        ax.plot(s.depth, s["ndcg@10"], color=c, marker="o", label=f"{lab} + cross-encoder")
        b = main[(ds, key)]
        ax.axhline(b, color=c, linestyle=(0, (4, 3)), linewidth=0.9)
        ax.annotate(f"{lab.split(' (')[0]} alone", (9, b), xytext=(0, 2), textcoords="offset points",
                    va="bottom", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 30, 50, 100], ["10", "20", "30", "50", "100"])
    ax.minorticks_off()
    ax.set_xlim(8.5, 115)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.1)
    ax.set_title(title, loc="left")
    ax.set_ylabel("nDCG@10")
axes[1].set_xlabel("Re-ranking depth $k$ (log scale)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=1,
           bbox_to_anchor=(0.5, -0.08), fontsize=6.8)
fig.tight_layout()
fig.savefig(OUT / "ieee_fig_depth.pdf", bbox_inches="tight")
plt.close(fig)
print("written: ieee_fig_hash.pdf, ieee_fig_alpha.pdf, ieee_fig_depth.pdf")
