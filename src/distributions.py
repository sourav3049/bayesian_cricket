# src/distributions.py
from dataclasses import dataclass
import numpy as np
from scipy.stats import bernoulli, binom, multinomial, poisson, nbinom

# ---------- Bernoulli ----------
@dataclass
class BernoulliDist:
    p: float  # success probability (e.g., "boundary on this ball")

    def pmf(self, k):
        return bernoulli.pmf(k, self.p)

    @property
    def mean(self):
        return self.p

    @property
    def var(self):
        return self.p * (1 - self.p)

    def sample(self, size=1, rng=None):
        rng = rng or np.random.default_rng()
        return rng.binomial(1, self.p, size=size)

# ---------- Binomial ----------
@dataclass
class BinomialDist:
    n: int
    p: float  # per-trial success probability

    def pmf(self, k):
        return binom.pmf(k, self.n, self.p)

    @property
    def mean(self):
        return self.n * self.p

    @property
    def var(self):
        return self.n * self.p * (1 - self.p)

    def sample(self, size=1, rng=None):
        rng = rng or np.random.default_rng()
        return rng.binomial(self.n, self.p, size=size)

# ---------- Categorical (1 trial over K categories) ----------
@dataclass
class CategoricalDist:
    probs: np.ndarray  # shape (K,)

    def pmf(self, k):
        # k is the category index
        probs = np.asarray(self.probs)
        if k < 0 or k >= probs.size:
            return 0.0
        return probs[k]

    @property
    def mean(self):
        # expected category index; often less meaningful—still provided
        probs = np.asarray(self.probs)
        return np.sum(np.arange(probs.size) * probs)

    @property
    def var(self):
        probs = np.asarray(self.probs)
        mu = self.mean
        return np.sum(((np.arange(probs.size) - mu) ** 2) * probs)

    def sample(self, size=1, rng=None):
        rng = rng or np.random.default_rng()
        probs = np.asarray(self.probs)
        return rng.choice(len(probs), size=size, p=probs)

# ---------- Multinomial (n trials over K categories) ----------
@dataclass
class MultinomialDist:
    n: int
    probs: np.ndarray  # shape (K,)

    def pmf(self, counts):
        counts = np.asarray(counts)
        return multinomial.pmf(counts, self.n, self.probs)

    @property
    def mean(self):
        return self.n * np.asarray(self.probs)

    @property
    def cov(self):
        p = np.asarray(self.probs)
        diag = np.diag(self.n * p * (1 - p))
        off = -self.n * np.outer(p, p)
        np.fill_diagonal(off, 0)
        return diag + off

    def sample(self, size=1, rng=None):
        rng = rng or np.random.default_rng()
        return rng.multinomial(self.n, self.probs, size=size)

# ---------- Poisson ----------
@dataclass
class PoissonDist:
    lam: float  # expected count per interval (e.g., runs per over events)

    def pmf(self, k):
        return poisson.pmf(k, self.lam)

    @property
    def mean(self):
        return self.lam

    @property
    def var(self):
        return self.lam

    def sample(self, size=1, rng=None):
        rng = rng or np.random.default_rng()
        return rng.poisson(self.lam, size=size)

# ---------- Negative Binomial ----------
@dataclass
class NegativeBinomialDist:
    r: int      # number of successes to stop (or 'failures' depending on paramization)
    p: float    # per-trial success prob (SciPy nbinom uses number of failures before r successes with prob p)
    # SciPy parameterization: nbinom(r, p) gives number of failures before r successes.

    def pmf(self, k):
        return nbinom.pmf(k, self.r, self.p)

    @property
    def mean(self):
        return self.r * (1 - self.p) / self.p

    @property
    def var(self):
        return self.r * (1 - self.p) / (self.p ** 2)

    def sample(self, size=1, rng=None):
        rng = rng or np.random.default_rng()
        return rng.negative_binomial(self.r, self.p, size=size)
