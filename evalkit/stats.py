"""
Chapters 5-6: honest error bars and honest comparisons.

    bootstrap_ci        percentile and BCa intervals for a mean (resample ITEMS)
    clustered_se        standard error of a mean when items come in correlated clusters
    design_effect       1 + (m - 1) * rho: how much clustering inflates the variance
    mcnemar             exact and chi-square tests on paired pass/fail outcomes
    paired_bootstrap    interval on a difference, resampling item ids
    permutation_test    sign-flip test on paired differences
    n_paired            items needed to detect a gain, from the discordant rate
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

ND = NormalDist()


def mean(xs):
    return sum(xs) / len(xs)


def se_mean(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


# ------------------------------------------------------------ Ch 5: the bootstrap

def bootstrap_means(xs, n_boot=10_000, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    return sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))


def bootstrap_ci(xs, n_boot=10_000, alpha=0.05, seed=0, method="percentile"):
    """
    Percentile: the alpha/2 and 1-alpha/2 quantiles of the bootstrap means.
    BCa (Efron 1987): shift the quantiles for bias (z0) and skew (acceleration a, from the jackknife).
    """
    bm = bootstrap_means(xs, n_boot, seed)
    if method == "percentile":
        return bm[int(alpha / 2 * n_boot)], bm[int((1 - alpha / 2) * n_boot) - 1]
    th = mean(xs)
    prop = sum(b < th for b in bm) / n_boot
    prop = min(max(prop, 1 / n_boot), 1 - 1 / n_boot)
    z0 = ND.inv_cdf(prop)
    n = len(xs)
    tot = sum(xs)
    jack = [(tot - x) / (n - 1) for x in xs]
    jm = mean(jack)
    num = sum((jm - j) ** 3 for j in jack)
    den = 6 * (sum((jm - j) ** 2 for j in jack) ** 1.5)
    a = num / den if den else 0.0

    def q(p):
        z = ND.inv_cdf(p)
        adj = ND.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        return bm[min(n_boot - 1, max(0, int(adj * n_boot)))]
    return q(alpha / 2), q(1 - alpha / 2)


# ------------------------------------------------------------ Ch 5: clustering

def design_effect(m: float, rho: float) -> float:
    """Variance inflation for clusters of size m with intra-cluster correlation rho (Kish 1965)."""
    return 1 + (m - 1) * rho


def effective_n(n_rows: int, m: float, rho: float) -> float:
    return n_rows / design_effect(m, rho)


def clustered_se(values: list[float], clusters: list) -> float:
    """
    Cluster-robust standard error of the overall mean (CR1-style small-sample factor G/(G-1)):
        SE^2 = G/(G-1) * sum_g (sum_{i in g} (x_i - xbar))^2 / N^2
    """
    N = len(values)
    xbar = mean(values)
    sums = {}
    for x, g in zip(values, clusters):
        sums[g] = sums.get(g, 0.0) + (x - xbar)
    G = len(sums)
    return math.sqrt(G / (G - 1) * sum(s * s for s in sums.values()) / (N * N))


def cluster_bootstrap_ci(values, clusters, n_boot=4000, alpha=0.05, seed=0):
    """Resample whole clusters with replacement, recompute the mean over all their rows."""
    rng = random.Random(seed)
    groups = {}
    for x, g in zip(values, clusters):
        groups.setdefault(g, []).append(x)
    keys = list(groups)
    G = len(keys)
    ms = []
    for _ in range(n_boot):
        pick = [groups[keys[rng.randrange(G)]] for _ in range(G)]
        rows = [x for grp in pick for x in grp]
        ms.append(mean(rows))
    ms.sort()
    return ms[int(alpha / 2 * n_boot)], ms[int((1 - alpha / 2) * n_boot) - 1]


def simulate_clustered(G: int, m: int, p: float, rho: float, seed: int = 0):
    """
    G clusters (prompts) of m binary rows each (samples), intra-cluster correlation rho.
    Each prompt has its own pass probability q_g ~ Beta with mean p and the right rho
    (beta-binomial: rho = 1 / (a + b + 1)).
    """
    rng = random.Random(seed)
    s = 1 / rho - 1                      # a + b
    a, b = p * s, (1 - p) * s
    vals, cl = [], []
    for g in range(G):
        q = rng.betavariate(a, b)
        for _ in range(m):
            vals.append(1.0 if rng.random() < q else 0.0)
            cl.append(g)
    return vals, cl


def coverage_naive_vs_clustered(G, m, p, rho, trials=1000, seed=0):
    """Fraction of trials whose 95% interval covers p: naive (rows as independent) vs clustered SE."""
    z = 1.959963984540054
    hit_n = hit_c = 0
    for t in range(trials):
        v, c = simulate_clustered(G, m, p, rho, seed=seed + t)
        mu = mean(v)
        se_n = math.sqrt(mu * (1 - mu) / len(v))
        se_c = clustered_se(v, c)
        hit_n += abs(mu - p) <= z * se_n
        hit_c += abs(mu - p) <= z * se_c
    return hit_n / trials, hit_c / trials


# ------------------------------------------------------------ Ch 6: paired comparisons

def discordant(a: list[int], b: list[int]) -> tuple[int, int]:
    """(b01, b10): items where A failed and B passed, and where A passed and B failed."""
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    return b01, b10


def mcnemar_exact(b01: int, b10: int) -> float:
    """Two-sided exact McNemar: binomial test of b01 vs b10 with p = 1/2 on the discordant items."""
    n = b01 + b10
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar_chi2(b01: int, b10: int, correction: bool = True) -> float:
    """Chi-square McNemar (1 df) p-value, with Edwards' continuity correction by default."""
    d = abs(b01 - b10) - (1 if correction else 0)
    stat = max(d, 0) ** 2 / (b01 + b10)
    return 2 * (1 - ND.cdf(math.sqrt(stat)))


def paired_bootstrap(a, b, n_boot=10_000, alpha=0.05, seed=0):
    """Interval for mean(b) - mean(a) on the same items: resample item ids."""
    return bootstrap_ci([y - x for x, y in zip(a, b)], n_boot, alpha, seed)


def unpaired_bootstrap(a, b, n_boot=10_000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n, m = len(a), len(b)
    d = sorted(sum(b[rng.randrange(m)] for _ in range(m)) / m - sum(a[rng.randrange(n)] for _ in range(n)) / n
               for _ in range(n_boot))
    return d[int(alpha / 2 * n_boot)], d[int((1 - alpha / 2) * n_boot) - 1]


def permutation_test(a, b, n_perm=10_000, seed=0) -> float:
    """Sign-flip permutation test on paired differences: two-sided p-value for mean difference = 0."""
    rng = random.Random(seed)
    d = [y - x for x, y in zip(a, b)]
    obs = abs(sum(d))
    hits = 0
    for _ in range(n_perm):
        s = sum(x if rng.random() < 0.5 else -x for x in d)
        hits += abs(s) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def n_paired(discord: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """
    Items to detect a gain delta when both systems answer the same items.
    Per item d in {-1, 0, +1}; Var(d) = discord - delta^2 (discord = P(systems disagree)).
    """
    z = ND.inv_cdf(1 - alpha / 2) + ND.inv_cdf(power)
    return math.ceil(z * z * (discord - delta * delta) / (delta * delta))


def n_unpaired(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Items PER SYSTEM for independent samples (two-proportion z-test)."""
    z = ND.inv_cdf(1 - alpha / 2) + ND.inv_cdf(power)
    return math.ceil(z * z * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2)


def mde_paired(n: int, discord: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest gain detectable with n paired items (solve n_paired for delta; small-delta approximation)."""
    z = ND.inv_cdf(1 - alpha / 2) + ND.inv_cdf(power)
    return z * math.sqrt(discord / n)


def simulate_pair(n: int, pa: float, pb: float, corr_items: float, seed: int = 0):
    """
    Two systems on the same n items. Each item has a difficulty; with weight corr_items both
    systems share it (correlated outcomes), else each draws independently.
    Returns two 0/1 lists.
    """
    rng = random.Random(seed)
    A, Bl = [], []
    for _ in range(n):
        u = rng.random()
        ua = u if rng.random() < corr_items else rng.random()
        ub = u if rng.random() < corr_items else rng.random()
        A.append(int(ua < pa))
        Bl.append(int(ub < pb))
    return A, Bl
