"""Report figures (PNG, 300 dpi) from results_final/ and results_ablation/.

    python make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

R = Path("results_final")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# validated categorical slots (light mode): blue, orange, aqua, yellow
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DS = {"scifact": "SciFact", "nfcorpus": "NFCorpus"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "axes.labelsize": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    fig.savefig(OUT / name.replace(".png", ".pdf"), bbox_inches="tight")  # vector copy for LaTeX
    plt.close(fig)


# 1. accuracy vs latency ------------------------------------------------------
SHORT = {
    "tfidf": "TF-IDF", "bm25": "BM25", "bm25-stem": "BM25-stem",
    "hashed-w1-2-b16": "Hashed w1-2", "hashed-w1-1+c3-5-b16": "Hashed w1+c3-5",
    "dense-minilm": "MiniLM", "dense-bge-small": "bge-small",
    "hybrid-wsum-minilm": "WSum MiniLM", "hybrid-rrf-minilm": "RRF MiniLM",
    "hybrid-wsum-bge-small": "WSum bge", "hybrid-rrf-bge-small": "RRF bge",
    "ce100-bm25-stem": "BM25+CE", "ce100-hybrid-rrf-bge-small": "RRF bge+CE",
}
GROUP = {"lexical": (BLUE, "o", "Lexical / hashed"), "hashed": (BLUE, "s", "Lexical / hashed"),
         "dense": (ORANGE, "D", "Dense"), "hybrid": (AQUA, "^", "Hybrid / re-ranked"),
         "re-ranked": (AQUA, "v", "Hybrid / re-ranked")}
main = pd.read_csv(R / "main_table.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
for ax, (ds, title) in zip(axes, DS.items()):
    d = main[main.dataset == ds]
    for _, r in d.iterrows():
        c, mk, _ = GROUP[r.family]
        ax.scatter(r.latency_ms, r["ndcg@10"], color=c, marker=mk, s=34, zorder=3,
                   edgecolor=SURFACE, linewidth=1.2)
    # label only the points that carry the story, to avoid collisions
    for key in ["bm25", "bm25-stem", "hashed-w1-1+c3-5-b16", "dense-minilm", "dense-bge-small",
                "hybrid-wsum-bge-small", "hybrid-rrf-bge-small", "ce100-bm25-stem",
                "ce100-hybrid-rrf-bge-small"]:
        r = d[d.key == key].iloc[0]
        other = "ce100-hybrid-rrf-bge-small" if key == "ce100-bm25-stem" else "ce100-bm25-stem"
        dy = 3
        if key.startswith("ce100"):  # put the higher of the two CE points above, the other below
            dy = 4 if r["ndcg@10"] >= d[d.key == other]["ndcg@10"].iloc[0] else -10
        ax.annotate(SHORT[key], (r.latency_ms, r["ndcg@10"]), xytext=(-4 if key.startswith("ce100") else 4, dy),
                    textcoords="offset points", fontsize=6.8, color=INK,
                    ha="right" if key.startswith("ce100") else "left")
    ax.set_xscale("log")
    ax.set_title(title, loc="left")
    ax.set_xlabel("Latency per query, ms (log scale, laptop CPU)")
    ax.set_xlim(0.3, 1e5)
axes[0].set_ylabel("nDCG@10")
handles = [plt.Line2D([], [], color=c, marker=m, linestyle="", label=l)
           for c, m, l in [(BLUE, "o", "Lexical"), (BLUE, "s", "Hashed"), (ORANGE, "D", "Dense"),
                           (AQUA, "^", "Hybrid"), (AQUA, "v", "Re-ranked (CE@100)")]]
fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=7.5)
fig.tight_layout()
save(fig, "fig1_accuracy_vs_latency.png")

# 2. hash size ablation ----------------------------------------------------------
abl = pd.read_csv("results_ablation/hash_ablation.csv")
abl = abl[abl.signed]
if len(abl) < 60:  # 6 feature sets x 5 sizes x 2 datasets
    raise SystemExit(f"hash ablation incomplete ({len(abl)} of 60 signed runs); re-run when it finishes")
series = [("word 1", BLUE), ("word 1-2", ORANGE), ("word 1-3", AQUA), ("word 1 + char 3-5", YELLOW)]
bm = main.set_index(["dataset", "key"])["ndcg@10"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, (ds, title) in zip(axes, DS.items()):
    for (label, c), mk in zip(series, ["o", "s", "^", "D"]):
        s = abl[(abl.dataset == ds) & (abl.features == label)].sort_values("bits")
        ax.plot(s.bits, s["ndcg@10"], color=c, marker=mk, markersize=5.5, label=label)
    ref = bm[(ds, "bm25-stem")]
    ax.axhline(ref, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    ax.annotate(f"BM25-stem {ref:.3f}", (10, ref), xytext=(0, 3), textcoords="offset points",
                fontsize=6.8, color=INK2)
    ax.set_xticks([10, 12, 14, 16, 18], ["2^10", "2^12", "2^14", "2^16", "2^18"])
    ax.set_xlim(9.6, 18.4)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Number of hash buckets")
axes[0].set_ylabel("nDCG@10")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.06), fontsize=7.5)
fig.tight_layout()
save(fig, "fig2_hash_size.png")

# 3. fusion weight --------------------------------------------------------------
sw = pd.read_csv(R / "fusion_alpha_sweep.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, (ds, title) in zip(axes, DS.items()):
    s = sw[(sw.dataset == ds) & (sw.dense == "bge-small")]
    ax.plot(s.alpha, s["ndcg@10_tune"], color=BLUE, label="Tuning queries")
    ax.plot(s.alpha, s["ndcg@10_test"], color=ORANGE, label="Test queries")
    ch = s[s.chosen].iloc[0]
    ax.axvline(ch.alpha, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    lo = min(s["ndcg@10_tune"].min(), s["ndcg@10_test"].min())
    ax.annotate(f"chosen on tuning set\nalpha = {ch.alpha:.2f}, test {ch['ndcg@10_test']:.3f}",
                (ch.alpha, lo), xytext=(5, 2), textcoords="offset points",
                ha="left", va="bottom", fontsize=6.8, color=INK2)
    for col, lab in [("ndcg@10_tune", "Tuning"), ("ndcg@10_test", "Test")]:
        ax.annotate(lab, (1.0, s[col].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=6.8, color=INK)
    ax.set_xlim(0, 1.12)
    ax.set_title(title, loc="left")
    ax.set_xlabel("alpha  (0 = BM25-stem only, 1 = bge-small only)")
axes[0].set_ylabel("nDCG@10")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.06), fontsize=7.5)
fig.tight_layout()
save(fig, "fig3_fusion_weight.png")

# 4. re-ranking depth -----------------------------------------------------------
dep = pd.read_csv(R / "rerank_depth.csv")
base = {"BM25 + stemming": "bm25-stem", "RRF, BM25-stem + bge-small": "hybrid-rrf-bge-small"}
lab = {"BM25 + stemming": "BM25-stem", "RRF, BM25-stem + bge-small": "RRF (BM25-stem + bge)"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, (ds, title) in zip(axes, DS.items()):
    for (fs, key), c in zip(base.items(), [BLUE, ORANGE]):
        s = dep[(dep.dataset == ds) & (dep.first_stage == fs)].sort_values("depth")
        ax.plot(s.depth, s["ndcg@10"], color=c, marker="o", label=f"{lab[fs]} + cross-encoder")
        b = bm[(ds, key)]
        ax.axhline(b, color=c, linestyle=(0, (4, 3)), linewidth=1)
        ax.annotate(f"{lab[fs].split(' (')[0]} alone", (9, b), xytext=(0, 3), textcoords="offset points",
                    va="bottom", fontsize=6.8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 30, 50, 100], ["10", "20", "30", "50", "100"])
    ax.minorticks_off()
    ax.set_xlim(8.5, 115)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.08)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Re-ranking depth k (log scale)")
axes[0].set_ylabel("nDCG@10")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.06), fontsize=7.5)
fig.tight_layout()
save(fig, "fig4_rerank_depth.png")

# 5. error categories -------------------------------------------------------------
err = pd.read_csv(R / "error_categories.csv")
names = {"A": "Near miss (rank 11-25)", "B": "Indirect evidence", "C": "Topical decoy",
         "D": "Short / ambiguous query", "E": "Loosely linked label"}
ct = pd.crosstab(err.category, err.dataset).reindex(list(names), fill_value=0)
fig, ax = plt.subplots(figsize=(5.2, 2.6))
y = range(len(ct))
h = 0.36
for i, (ds, c) in enumerate([("scifact", BLUE), ("nfcorpus", ORANGE)]):
    pos = [v + (i - 0.5) * (h + 0.04) for v in y]
    ax.barh(pos, ct[ds], height=h, color=c, label=DS[ds])
    for p, v in zip(pos, ct[ds]):
        if v:
            ax.annotate(str(v), (v, p), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK)
ax.set_yticks(list(y), [names[k] for k in ct.index])
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("Failed queries in the sample (15 per dataset)")
ax.set_xlim(0, 9.5)
ax.legend(loc="center right", fontsize=7.5)
fig.tight_layout()
save(fig, "fig5_error_categories.png")
print("figures written to", OUT)
