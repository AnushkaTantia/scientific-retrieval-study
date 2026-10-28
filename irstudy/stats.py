"""Paired significance tests over per-query scores."""
import numpy as np
from scipy import stats


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Compare two systems on the same queries.

    Returns the mean difference (a - b), a 95% bootstrap confidence interval and a
    two-sided p-value for the null hypothesis that the mean difference is zero.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"mean_diff": float(diff.mean()), "ci95": (float(lo), float(hi)),
            "p_bootstrap": float(min(p, 1.0)),
            "p_ttest": float(stats.ttest_rel(a, b).pvalue) if diff.any() else 1.0}
