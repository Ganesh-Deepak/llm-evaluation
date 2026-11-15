"""
Chapter 4: metrics for a single output, calibration, and cost as an outcome.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

# ------------------------------------------------------------ reference-based

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s: str) -> str:
    """SQuAD normalisation: lower-case, drop punctuation and articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str, norm: bool = True) -> int:
    return int((normalize(pred) if norm else pred) == (normalize(gold) if norm else gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def sentence_bleu(pred: str, refs: list[str], max_n: int = 4) -> float:
    """
    BLEU (Papineni et al. 2002) for one sentence: geometric mean of clipped n-gram precisions
    (n = 1..4) times a brevity penalty, with add-one smoothing on n >= 2 (Lin & Och 2004) so a
    single missing 4-gram does not zero the score.
    """
    p = pred.lower().split()
    R = [r.lower().split() for r in refs]
    logs = []
    for n in range(1, max_n + 1):
        cand = _ngrams(p, n)
        maxref = Counter()
        for r in R:
            for g, c in _ngrams(r, n).items():
                maxref[g] = max(maxref[g], c)
        clipped = sum(min(c, maxref[g]) for g, c in cand.items())
        total = max(sum(cand.values()), 0)
        if n == 1:
            if clipped == 0:
                return 0.0
            logs.append(math.log(clipped / total))
        else:
            logs.append(math.log((clipped + 1) / (total + 1)))
    c = len(p)
    r = min((abs(len(x) - c), len(x)) for x in R)[1]
    bp = 1.0 if c > r else math.exp(1 - r / max(c, 1))
    return bp * math.exp(sum(logs) / max_n)


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1 from the longest common subsequence of tokens."""
    a, b = pred.lower().split(), ref.lower().split()
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


# ------------------------------------------------------------ calibration

def brier(conf: list[float], correct: list[int]) -> float:
    """Mean squared error between stated confidence and the 0/1 outcome."""
    return sum((c - y) ** 2 for c, y in zip(conf, correct)) / len(conf)


def ece(conf: list[float], correct: list[int], bins: int = 10) -> float:
    """
    Expected calibration error: bucket by confidence, then the item-weighted mean of
    |accuracy in bucket - mean confidence in bucket|.
    """
    n = len(conf)
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        mc = sum(conf[i] for i in idx) / len(idx)
        tot += len(idx) / n * abs(acc - mc)
    return tot


def simulate_confidences(n: int, overconfidence: float, seed: int = 0) -> tuple[list[float], list[int]]:
    """
    A system whose true probability of being right is q ~ Uniform(0.5, 1), and which REPORTS
    min(1, q + overconfidence). Returns (stated confidences, 0/1 outcomes).
    """
    import random
    rng = random.Random(seed)
    conf, ok = [], []
    for _ in range(n):
        q = 0.5 + 0.5 * rng.random()
        conf.append(min(1.0, q + overconfidence))
        ok.append(int(rng.random() < q))
    return conf, ok


# ------------------------------------------------------------ cost as an outcome

def cost_per_call(tokens_in: int, tokens_out: int, price_in: float, price_out: float) -> float:
    """Dollars for one call; prices are per million tokens."""
    return tokens_in * price_in / 1e6 + tokens_out * price_out / 1e6


def pareto_front(points: dict) -> list[str]:
    """
    points: {name: (quality, cost)}. A point is on the front if no other point has
    quality >= and cost <= with at least one strictly better.
    """
    front = []
    for a, (qa, ca) in points.items():
        dominated = any((qb >= qa and cb <= ca) and (qb > qa or cb < ca) for b, (qb, cb) in points.items() if b != a)
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda k: points[k][1])
