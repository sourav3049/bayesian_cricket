# src/bayes.py
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta as beta_dist

@dataclass
class BetaBinomialPosterior:
    alpha_prior: float  # prior "successes" + 1 (Beta alpha)
    beta_prior: float   # prior "failures" + 1 (Beta beta)

    def update(self, successes: int, trials: int):
        failures = max(trials - successes, 0)
        return (
            self.alpha_prior + successes,
            self.beta_prior + failures
        )

    @staticmethod
    def mean(alpha_post: float, beta_post: float) -> float:
        return alpha_post / (alpha_post + beta_post)

    @staticmethod
    def ci(alpha_post: float, beta_post: float, level: float = 0.95):
        lo = (1 - level) / 2.0
        hi = 1 - lo
        return float(beta_dist.ppf(lo, alpha_post, beta_post)), float(beta_dist.ppf(hi, alpha_post, beta_post))

def adjust_categorical_with_boundary(base_probs, p_boundary_post, boundary_idx):
    """
    Replaces total boundary mass with posterior mean p_boundary_post,
    allocating between boundary categories proportionally to their base ratio.
    Non-boundary categories are scaled to keep sum=1.
    """
    p = np.array(base_probs, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    if p.size == 0:
        return p

    # total boundary base mass
    bmask = np.zeros_like(p, dtype=bool)
    bmask[list(boundary_idx)] = True
    pB_base = p[bmask].sum()
    pNB_base = 1.0 - pB_base
    if pB_base <= 0:
        # nothing to reallocate proportionally; just inject all mass into first boundary index
        p_adj = p.copy()
        p_adj[~bmask] *= (1 - p_boundary_post) / max(pNB_base, 1e-12)
        # split to boundary indices equally
        each = p_boundary_post / len(boundary_idx)
        for bi in boundary_idx:
            p_adj[bi] = each
        return p_adj / p_adj.sum()

    # scale non-boundary part to 1 - pB_post while preserving ratios
    p_adj = p.copy()
    p_adj[~bmask] *= (1 - p_boundary_post) / max(pNB_base, 1e-12)

    # allocate boundary mass proportionally to base 4 vs 6 split
    ratios = p[bmask] / pB_base
    p_adj[bmask] = ratios * p_boundary_post

    # numerical safety
    s = p_adj.sum()
    if s <= 0:
        return p
    return p_adj / s
