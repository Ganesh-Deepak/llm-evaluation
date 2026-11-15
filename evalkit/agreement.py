"""
Chapter 3: agreement between labellers, corrected for chance.

    observed agreement   p_o = fraction of items where raters agree
    chance agreement     p_e = what two raters with these label frequencies would agree on by luck
    kappa                (p_o - p_e) / (1 - p_e)

All three statistics below are kappa-shaped: 1 = perfect, 0 = chance level, < 0 = worse than chance.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations


def observed_agreement(a: list, b: list) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: list, b: list) -> float:
    """Two raters, any number of categories. p_e uses each rater's OWN marginal frequencies."""
    n = len(a)
    po = observed_agreement(a, b)
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe)


def kappa_from_table(both_pass: int, a_only: int, b_only: int, both_fail: int) -> dict:
    """2x2 table (rater A pass / B pass) -> p_o, p_e, kappa. a_only: A pass, B fail."""
    n = both_pass + a_only + b_only + both_fail
    po = (both_pass + both_fail) / n
    pa, pb = (both_pass + a_only) / n, (both_pass + b_only) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {"po": po, "pe": pe, "kappa": (po - pe) / (1 - pe)}


def table_to_labels(both_pass: int, a_only: int, b_only: int, both_fail: int) -> tuple[list, list]:
    a = [1] * both_pass + [1] * a_only + [0] * b_only + [0] * both_fail
    b = [1] * both_pass + [0] * a_only + [1] * b_only + [0] * both_fail
    return a, b


def fleiss_kappa(counts: list[list[int]]) -> float:
    """
    Many raters, the same number per item. counts[i][j] = raters who put item i in category j.
    P_i = (sum_j n_ij^2 - r) / (r (r - 1));  P_bar = mean P_i;  p_j = category shares;
    P_e = sum_j p_j^2;  kappa = (P_bar - P_e) / (1 - P_e)
    """
    N = len(counts)
    r = sum(counts[0])
    assert all(sum(row) == r for row in counts), "Fleiss' kappa needs the same number of raters per item"
    Pi = [(sum(x * x for x in row) - r) / (r * (r - 1)) for row in counts]
    Pbar = sum(Pi) / N
    k = len(counts[0])
    pj = [sum(row[j] for row in counts) / (N * r) for j in range(k)]
    Pe = sum(p * p for p in pj)
    return (Pbar - Pe) / (1 - Pe)


def krippendorff_alpha_nominal(units: list[list]) -> float:
    """
    Krippendorff's alpha for nominal labels, with missing values allowed.
    units: one list per item of the labels it received (None = not labelled; items with < 2 labels are dropped).
    alpha = 1 - D_o / D_e, built from the coincidence matrix: each item with m labels contributes
    every ordered pair of its labels with weight 1/(m-1).
    """
    coinc: Counter = Counter()
    for u in units:
        vals = [v for v in u if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    coinc[(vals[i], vals[j])] += 1 / (m - 1)
    cats = sorted({c for pair in coinc for c in pair}, key=str)
    nc = {c: sum(coinc[(c, d)] for d in cats) for c in cats}
    n = sum(nc.values())
    Do = sum(coinc[(c, d)] for c in cats for d in cats if c != d) / n
    De = sum(nc[c] * nc[d] for c in cats for d in cats if c != d) / (n * (n - 1))
    return 1 - Do / De


def pairwise_cohen(labels: dict) -> dict:
    """Cohen's kappa for every pair of raters. labels: {rater: [label per item]}."""
    return {(a, b): cohen_kappa(labels[a], labels[b]) for a, b in combinations(labels, 2)}


def majority(votes: list) -> object:
    """Majority label; ties return None (send to adjudication)."""
    c = Counter(votes).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None
    return c[0][0]


def simulate_kappa(kappa: float, p_pass: float, n: int, trials: int = 2000, seed: int = 0) -> list[float]:
    """
    Two raters with the same pass rate p and true kappa: the joint table is
      both pass = p - d/2, both fail = (1-p) - d/2, each disagreement cell = d/2,
      where d = (1 - kappa) * (1 - p_e) and p_e = p^2 + (1-p)^2.
    Draw n items per trial and return the measured kappas.
    """
    import random
    rng = random.Random(seed)
    pe = p_pass ** 2 + (1 - p_pass) ** 2
    d = (1 - kappa) * (1 - pe)
    cells = [p_pass - d / 2, d / 2, d / 2, 1 - p_pass - d / 2]
    out = []
    for _ in range(trials):
        cnt = [0, 0, 0, 0]
        for _ in range(n):
            u, acc = rng.random(), 0.0
            for i, c in enumerate(cells):
                acc += c
                if u < acc:
                    cnt[i] += 1
                    break
            else:
                cnt[3] += 1
        r = kappa_from_table(*cnt)
        if r["pe"] < 1:
            out.append(r["kappa"])
    return out


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(q * len(s))))]
