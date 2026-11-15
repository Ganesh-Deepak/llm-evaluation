"""
Every number the Evaluation course states, asserted.

    python tests/test_evalkit.py            # no pytest needed
    python -m pytest tests/test_evalkit.py -q
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from evalkit import agreement as A  # noqa: E402
from evalkit import basics as B  # noqa: E402
from evalkit import metrics as M  # noqa: E402

R1 = lambda x: round(x * 100, 1)  # noqa: E731   proportion -> percent, one decimal

# ============================================================ Ch 1 — defining good

RATES = {"wrong policy": 0.06, "invented order status": 0.03, "missed escalation": 0.02, "tone": 0.08}


def test_ch01_reads_table():
    assert [B.reads_to_see(p) for p in (.01, .02, .03, .05, .10, .20)] == [299, 149, 99, 59, 29, 14]
    assert [R1(B.prob_seen(p, 100)) for p in (.01, .02, .03, .05)] == [63.4, 86.7, 95.2, 99.4]
    assert [R1(B.prob_seen(p, 30)) for p in (.01, .02, .03, .05, .10)] == [26.0, 45.5, 59.9, 78.5, 95.8]
    assert round(0.97 ** 30, 3) == 0.401                       # quiz: 30 reads, 3% mode, none seen
    assert round(math.log(0.05) / math.log(0.95), 1) == 58.4   # quiz: 59 reads for a 5% mode
    assert round(1 - 0.95 ** 20, 2) == 0.64


def test_ch01_wald_vs_wilson():
    wil = {k: tuple(R1(x) for x in B.wilson(k, 100)) for k in (0, 1, 3, 5, 12)}
    assert wil == {0: (0.0, 3.7), 1: (0.2, 5.4), 3: (1.0, 8.5), 5: (2.2, 11.2), 12: (7.0, 19.8)}
    wal = {k: tuple(R1(x) for x in B.wald(k, 100)) for k in (0, 1, 3, 5, 12)}
    assert wal == {0: (0.0, 0.0), 1: (-1.0, 3.0), 3: (-0.3, 6.3), 5: (0.7, 9.3), 12: (5.6, 18.4)}
    z = B.z_for()
    assert round(z * z / (100 + z * z) * 100, 1) == 3.7            # the 0-in-100 upper bound


def test_ch01_worked_example():
    seen = B.simulate_failure_reading(RATES, 100, seed=1)
    assert seen == {"wrong policy": 9, "invented order status": 7, "missed escalation": 0, "tone": 13}
    iv = {k: tuple(R1(x) for x in B.wilson(v, 100)) for k, v in seen.items()}
    assert iv == {"wrong policy": (4.8, 16.2), "invented order status": (3.4, 13.7),
                  "missed escalation": (0.0, 3.7), "tone": (7.8, 21.0)}
    inside = {k: iv[k][0] <= p * 100 <= iv[k][1] for k, p in RATES.items()}
    assert inside == {"wrong policy": True, "invented order status": False,   # 3.0 < 3.4: the one miss
                      "missed escalation": True, "tone": True}
    assert round(0.98 ** 100, 3) == 0.133


def test_ch01_exercises():
    assert B.reads_to_see(0.005) == 598 and B.reads_to_see(0.005, 0.80) == 322
    assert round((1 - 0.05 ** (1 / 200)) * 100, 2) == 1.49
    assert B.interval_coverage(0.02, 100, "wald") == 0.8645
    assert B.interval_coverage(0.02, 100, "wilson") == 0.9447
    runs = B.reading_runs(RATES, 100)
    assert sum(r["missed escalation"] == 0 for r in runs) / 1000 == 0.135
    assert sum(r["invented order status"] > r["wrong policy"] for r in runs) / 1000 == 0.120
    assert sum(r["invented order status"] == r["wrong policy"] for r in runs) / 1000 == 0.083
    item = 1200 * 3e-6 + 300 * 15e-6 + 2000 * 3e-6 + 50 * 15e-6
    assert round(item, 3) == 0.015 and round(item * 1500) == 22
    assert 3 / 1500 == 0.002


# ============================================================ Ch 2 — golden sets

SIZES = {"billing": 6000, "shipping": 3000, "returns": 800, "account": 200}
SDS = {"billing": 0.3, "shipping": 0.3, "returns": 0.45, "account": 0.5}


def test_ch02_allocation():
    al = {r: B.stratified_allocation(SIZES, SDS, 400, r) for r in ("proportional", "neyman", "equal")}
    assert list(al["proportional"].values()) == [240, 120, 32, 8]
    assert list(al["neyman"].values()) == [228, 114, 45, 13]
    se = {r: round(B.stratified_se(SIZES, SDS, a) * 100, 2) for r, a in al.items()}
    assert se == {"proportional": 1.6, "neyman": 1.58, "equal": 2.05}
    assert round(1.96 * 0.3 / 10 * 100, 1) == 5.9 and round(1.96 * 0.5 / 10 * 100, 1) == 9.8   # equal: +-6..10
    assert round(1.96 * math.sqrt(0.25 / 8), 2) == 0.35


def test_ch02_weighting():
    p, se = B.weighted_rate({"normal": (0.95, 300, 276), "risky": (0.05, 300, 240)})
    assert round(p, 3) == 0.914 and round(se * 100, 2) == 1.49 and round(1.96 * se * 100, 1) == 2.9
    assert (276 + 240) / 600 == 0.86
    assert round(math.sqrt(0.92 * 0.08 / 300) * 100, 2) == 1.57


def test_ch02_size_table():
    tab = {m: [B.n_for_margin(m, p) for p in (.5, .8, .9)] for m in (.01, .02, .03, .05, .07)}
    assert tab == {.01: [9604, 6147, 3458], .02: [2401, 1537, 865], .03: [1068, 683, 385],
                   .05: [385, 246, 139], .07: [196, 126, 71]}
    assert round(B.half_width(0.5, 200) * 100, 1) == 6.9 and round(B.half_width(0.8, 300) * 100, 1) == 4.5


def test_ch02_selection():
    em = [round(B.expected_max_normal(K), 3) for K in (2, 5, 10, 20, 50)]
    assert em == [0.564, 1.163, 1.539, 1.867, 2.249]
    infl = [round(B.selection_bias(0.7, 200, K) * 100, 1) for K in (2, 5, 10, 20, 50)]
    assert infl == [1.8, 3.8, 5.0, 6.1, 7.3]
    assert round(math.sqrt(0.21 / 200) * 100, 2) == 3.24
    assert round(B.simulate_selection(0.7, 200, 20) * 100, 1) == 5.9


def test_ch02_exercises():
    sizes, sds = {"en": 85, "es": 10, "de": 5}, {k: math.sqrt(p * (1 - p)) for k, p in {"en": .92, "es": .85, "de": .75}.items()}
    ney = B.stratified_allocation(sizes, sds, 600, "neyman")
    assert list(ney.values()) == [481, 74, 45]
    assert [round(B.stratified_se(sizes, sds, B.stratified_allocation(sizes, sds, 600, r)) * 100, 2)
            for r in ("proportional", "neyman", "equal")] == [1.19, 1.18, 1.66]
    fl = {"en": 400, "es": 100, "de": 100}
    assert round(B.stratified_se(sizes, sds, fl) * 100, 2) == 1.23
    assert [round(1.96 * sds[k] / math.sqrt(fl[k]) * 100, 1) for k in ("es", "de")] == [7.0, 8.5]
    p, se = B.weighted_rate({"g": (0.915, 500, 440), "l": (0.08, 200, 140), "i": (0.005, 100, 60)})
    assert round(p * 100, 1) == 86.4 and round(se * 100, 2) == 1.35 and (440 + 140 + 60) / 800 == 0.8
    assert round(B.selection_bias(0.8, 400, 10) * 100, 1) == 3.1
    assert round(B.simulate_selection(0.8, 400, 10) * 100, 1) == 3.0
    assert B.n_for_margin(0.02, 0.85) == 1225 and B.n_for_margin(0.05, 0.85) == 196
    assert 1225 * 1.5 + 980 * 6 == 7717.5


# ============================================================ Ch 3 — labels and agreement

THREE = [[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]]


def test_ch03_kappa_tables():
    rows = {t: A.kappa_from_table(*t) for t in [(70, 10, 5, 15), (45, 5, 5, 45), (90, 4, 4, 2), (92, 3, 3, 2)]}
    assert [(round(r["po"], 2), round(r["pe"], 3), round(r["kappa"], 3)) for r in rows.values()] == [
        (0.85, 0.65, 0.571), (0.9, 0.5, 0.8), (0.92, 0.887, 0.291), (0.94, 0.905, 0.368)]
    a, b = A.table_to_labels(70, 10, 5, 15)
    assert round(A.cohen_kappa(a, b), 3) == 0.571                      # two routes, one number
    assert round((0.75 - 0.5) / 0.5, 2) == 0.5                          # quiz
    assert round((0.85 - 0.5) / 0.5, 2) == 0.7                          # 50/50 baseline would claim 0.70


def test_ch03_published_examples():
    N = None
    units = [list(t) for t in zip([1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N], [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
                                  [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N], [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N])]
    assert round(A.krippendorff_alpha_nominal(units), 3) == 0.743       # Krippendorff (2011)
    fl = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
          [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert round(A.fleiss_kappa(fl), 3) == 0.210                        # the standard Fleiss example


def test_ch03_three_raters():
    cnt = [[sum(x == 1 for x in u), sum(x == 0 for x in u)] for u in THREE]
    assert round(A.fleiss_kappa(cnt), 3) == 0.550 and round(A.krippendorff_alpha_nominal(THREE), 3) == 0.565
    lab = {r: [u[i] for u in THREE] for i, r in enumerate("ABC")}
    assert ["".join(map(str, lab[r])) for r in "ABC"] == ["1111011101", "1111001101", "1101001001"]
    pw = {k: round(v, 3) for k, v in A.pairwise_cohen(lab).items()}
    assert pw == {("A", "B"): 0.737, ("A", "C"): 0.4, ("B", "C"): 0.6}
    gaps = [list(u) for u in THREE]
    gaps[1][2] = gaps[6][1] = gaps[8][2] = None
    assert round(A.krippendorff_alpha_nominal(gaps), 3) == 0.519
    assert A.majority([1, 0]) is None and A.majority([1, 1, 0]) == 1


def test_ch03_exercises():
    r1, r2 = A.kappa_from_table(120, 20, 10, 50), A.kappa_from_table(165, 20, 10, 5)
    assert (round(r1["pe"], 2), round(r1["kappa"], 3)) == (0.56, 0.659)
    assert (round(r2["pe"], 3), round(r2["kappa"], 2)) == (0.819, 0.17) and r1["po"] == r2["po"] == 0.85
    spread = {n: (round(A.percentile(k, .025), 2), round(A.percentile(k, .975), 2))
              for n in (50, 100, 200, 400) for k in [A.simulate_kappa(0.6, 0.7, n)]}
    assert spread == {50: (0.33, 0.82), 100: (0.42, 0.76), 200: (0.48, 0.72), 400: (0.51, 0.68)}
    gen = (200 * 2 + 1300) * 40 / 3600 * 25
    exp = (0.15 * 200 + 0.06 * 1500) * 3 / 60 * 120
    assert round(gen) == 472 and round(exp) == 720 and round(exp / (gen + exp), 2) == 0.60


# ============================================================ Ch 4 — metrics

GOLD = "The order ships on March 3."
ROWS = ["the order ships on March 3", "It ships March 3rd.", "Your order will ship on 3 March", "The order ships on March 13."]
REF = "You can return the item within 30 days for a full refund."
PARA = "Returns are accepted for a full refund if you send it back within a month."
NEG = "You cannot return the item within 30 days for a full refund."
CONFIGS = {"large": (0.91, 0.0140), "medium": (0.88, 0.0042), "small": (0.79, 0.0009),
           "medium+retry": (0.90, 0.0061), "large-old": (0.87, 0.0150)}


def test_ch04_em_f1():
    assert [M.exact_match(r, GOLD, norm=False) for r in ROWS] == [0, 0, 0, 0]
    assert [M.exact_match(r, GOLD) for r in ROWS] == [1, 0, 0, 0]
    assert [round(M.token_f1(r, GOLD), 2) for r in ROWS] == [1.0, 0.44, 0.67, 0.8]


def test_ch04_overlap_fails():
    assert (round(M.sentence_bleu(PARA, [REF]), 3), round(M.rouge_l(PARA, REF), 3), round(M.token_f1(PARA, REF), 3)) == (0.165, 0.222, 0.435)
    assert (round(M.sentence_bleu(NEG, [REF]), 3), round(M.rouge_l(NEG, REF), 3), round(M.token_f1(NEG, REF), 3)) == (0.841, 0.917, 0.9)
    assert M.sentence_bleu(REF, [REF]) == 1.0 and M.rouge_l(REF, REF) == 1.0


def test_ch04_calibration():
    out = []
    for oc in (0.0, 0.1, 0.2):
        c, o = M.simulate_confidences(5000, oc)
        out.append((round(sum(o) / 5000, 3), round(sum(c) / 5000, 3), round(M.ece(c, o) * 100, 1), round(M.brier(c, o), 3)))
    assert out == [(0.745, 0.749, 1.7, 0.169), (0.745, 0.84, 9.5, 0.178), (0.745, 0.91, 16.5, 0.199)]
    assert M.ece([1.0, 0.0], [1, 0]) == 0.0 and M.brier([1.0, 0.0], [1, 0]) == 0.0


def test_ch04_cost():
    assert M.pareto_front(CONFIGS) == ["small", "medium", "medium+retry", "large"]
    assert round(0.0140 / 0.0061, 1) == 2.3 and round((0.0140 - 0.0061) * 1e6) == 7900
    big, small = M.cost_per_call(1200, 300, 3, 15), M.cost_per_call(1200, 300, 0.25, 1.25)
    assert round(big, 4) == 0.0081 and round(small, 6) == 0.000675
    casc = small + 0.22 * big
    assert round(casc, 5) == 0.00246 and round((big - casc) * 1e6) == 5643 and round(1 - casc / big, 2) == 0.70
    assert round(1.96 * math.sqrt(0.9 * 0.1 / 400) * 100, 1) == 2.9


# ============================================================ runner

def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}\n          {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
