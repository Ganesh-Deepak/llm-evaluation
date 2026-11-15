"""
Chapters 1-2: how many outputs to read, how sure a rate is, how big a golden set must be.

Notation used throughout the course:
    n   number of items                p   a true rate (pass rate, failure rate)
    k   number of successes            m   a margin (half-width of an interval)
    z   a standard-normal quantile     alpha = 1 - confidence
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

Z = NormalDist().inv_cdf


def z_for(conf: float = 0.95) -> float:
    """Two-sided normal quantile: 1.95996... for 95%."""
    return Z(1 - (1 - conf) / 2)


# ------------------------------------------------------------ Ch 1: error analysis

def reads_to_see(p: float, conf: float = 0.95) -> int:
    """
    How many independent outputs must you read to see a failure mode of frequency p
    at least once with probability conf?   P(miss) = (1-p)^n <= 1-conf
      =>  n >= ln(1-conf) / ln(1-p)
    """
    return math.ceil(math.log(1 - conf) / math.log(1 - p))


def prob_seen(p: float, n: int) -> float:
    """P(at least one occurrence in n reads) = 1 - (1-p)^n."""
    return 1 - (1 - p) ** n


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """
    Wilson score interval for a proportion. Unlike p_hat +- z*sqrt(p_hat(1-p_hat)/n)
    it never leaves [0, 1] and behaves at k = 0 or k = n, which is exactly where
    error analysis lives (rare failure modes).
    """
    z = z_for(conf)
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def wald(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """The textbook interval, kept to show where it fails."""
    z = z_for(conf)
    ph = k / n
    h = z * math.sqrt(ph * (1 - ph) / n)
    return ph - h, ph + h


# ------------------------------------------------------------ Ch 2: golden-set size

def half_width(p: float, n: int, conf: float = 0.95) -> float:
    """Normal-approximation half-width of a pass-rate interval: z*sqrt(p(1-p)/n)."""
    return z_for(conf) * math.sqrt(p * (1 - p) / n)


def n_for_margin(m: float, p: float = 0.5, conf: float = 0.95) -> int:
    """Items needed so the pass-rate interval is +-m:  n = z^2 p(1-p) / m^2."""
    z = z_for(conf)
    return math.ceil(z * z * p * (1 - p) / (m * m))


def stratified_allocation(sizes: dict, sds: dict, n: int, rule: str = "proportional") -> dict:
    """
    How to split n golden-set items across strata (intents, lengths, difficulties).
      proportional: n_h proportional to N_h            (mirrors traffic)
      neyman:       n_h proportional to N_h * sigma_h   (minimises the variance of the overall mean)
      equal:        n / H each                          (best for per-stratum reporting)
    Returns integer allocations that sum to n (largest-remainder rounding).
    """
    keys = list(sizes)
    if rule == "proportional":
        w = {h: sizes[h] for h in keys}
    elif rule == "neyman":
        w = {h: sizes[h] * sds[h] for h in keys}
    elif rule == "equal":
        w = {h: 1.0 for h in keys}
    else:
        raise ValueError(rule)
    tot = sum(w.values())
    raw = {h: n * w[h] / tot for h in keys}
    out = {h: int(math.floor(raw[h])) for h in keys}
    rest = n - sum(out.values())
    for h in sorted(keys, key=lambda h: raw[h] - out[h], reverse=True)[:rest]:
        out[h] += 1
    return out


def stratified_se(sizes: dict, sds: dict, alloc: dict) -> float:
    """Standard error of the traffic-weighted mean under a given allocation:
       SE^2 = sum_h W_h^2 * sigma_h^2 / n_h, with W_h = N_h / N."""
    N = sum(sizes.values())
    return math.sqrt(sum((sizes[h] / N) ** 2 * sds[h] ** 2 / alloc[h] for h in sizes))


def simulate_failure_reading(rates: dict, n: int, seed: int = 0) -> dict:
    """Read n outputs from a system whose failure modes occur at `rates`; count what you see."""
    rng = random.Random(seed)
    counts = {k: 0 for k in rates}
    for _ in range(n):
        for k, p in rates.items():
            if rng.random() < p:
                counts[k] += 1
    return counts


def interval_coverage(p: float, n: int, method: str = "wilson", trials: int = 10_000, seed: int = 0) -> float:
    """Fraction of simulated samples whose 95% interval contains the true rate p."""
    rng = random.Random(seed)
    f = wilson if method == "wilson" else wald
    hit = 0
    for _ in range(trials):
        k = sum(rng.random() < p for _ in range(n))
        lo, hi = f(k, n)
        hit += lo <= p <= hi
    return hit / trials


def reading_runs(rates: dict, n: int, runs: int = 1000, seed: int = 0) -> list[dict]:
    """Repeat simulate_failure_reading `runs` times with different seeds."""
    return [simulate_failure_reading(rates, n, seed=seed + r) for r in range(runs)]


# ------------------------------------------------------------ Ch 2: weighting and reuse

def weighted_rate(strata: dict) -> tuple[float, float]:
    """
    Traffic-weighted pass rate from a golden set that over-samples some slices.
    strata: {name: (traffic_share W_h, items n_h, passes k_h)}
    Returns (estimate, standard error):  p = sum W_h p_h,  SE^2 = sum W_h^2 p_h(1-p_h)/n_h
    """
    p = sum(W * k / n for W, n, k in strata.values())
    se = math.sqrt(sum(W * W * (k / n) * (1 - k / n) / n for W, n, k in strata.values()))
    return p, se


def expected_max_normal(K: int, steps: int = 20_000) -> float:
    """
    E[max of K independent standard normals], by numerical integration of
    E[M] = integral of x * K * phi(x) * Phi(x)^(K-1) dx over [-8, 8].
    """
    nd = NormalDist()
    lo, hi = -8.0, 8.0
    h = (hi - lo) / steps
    tot = 0.0
    for i in range(steps + 1):
        x = lo + i * h
        w = 0.5 if i in (0, steps) else 1.0
        tot += w * x * K * nd.pdf(x) * nd.cdf(x) ** (K - 1)
    return tot * h


def selection_bias(p: float, n: int, K: int) -> float:
    """Expected inflation of the best score when K equally good variants are compared on the same n items."""
    return expected_max_normal(K) * math.sqrt(p * (1 - p) / n)


def simulate_selection(p: float, n: int, K: int, trials: int = 2000, seed: int = 0) -> float:
    """Measured: mean of (best observed pass rate - true p) over trials."""
    rng = random.Random(seed)
    tot = 0.0
    for _ in range(trials):
        best = max(sum(rng.random() < p for _ in range(n)) / n for _ in range(K))
        tot += best - p
    return tot / trials
