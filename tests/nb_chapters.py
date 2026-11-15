"""The companion notebooks, one function per chapter. Every code cell asserts what it recomputes."""

from nb_lib import SETUP, code, md, notebook


@notebook("01_define")
def _nb01():
    return [
        md("""
        # Chapter 1 — Defining "good" before writing code

        Every number in Chapter 1, recomputed from `evalkit/basics.py`: how many outputs you must read to see a
        failure mode, the Wald and Wilson intervals for small counts, the worked example's simulated reading of 100
        support conversations, and the exercises' simulations. Every cell **asserts** what it checks.
        """),
        code(SETUP, "from evalkit import basics as B\nR1 = lambda x: round(x * 100, 1)"),
        md("## 1.3 — reads needed to see a failure mode: n ≥ ln(1 − c) / ln(1 − p)"),
        code("""
        for p in (.01, .02, .03, .05, .10, .20):
            print(f"p={p:5.0%}  reads for 95%: {B.reads_to_see(p):4d}   P(seen) in 100: {B.prob_seen(p, 100):6.1%}   in 30: {B.prob_seen(p, 30):6.1%}")
        assert [B.reads_to_see(p) for p in (.01, .02, .03, .05, .10, .20)] == [299, 149, 99, 59, 29, 14]
        assert [R1(B.prob_seen(p, 30)) for p in (.01, .02, .03, .05, .10)] == [26.0, 45.5, 59.9, 78.5, 95.8]
        # the same number by brute force: simulate 20,000 readings of 100 outputs at p = 3%
        rng = np.random.default_rng(0)
        sim = (rng.random((20_000, 100)) < 0.03).any(axis=1).mean()
        print("simulated P(seen) for p=3%, n=100:", sim)
        assert abs(sim - B.prob_seen(0.03, 100)) < 0.005
        """),
        md("## 1.3 — Wald against Wilson at small counts"),
        code("""
        for k in (0, 1, 3, 5, 12):
            print(k, "Wald", tuple(R1(x) for x in B.wald(k, 100)), " Wilson", tuple(R1(x) for x in B.wilson(k, 100)))
        assert tuple(R1(x) for x in B.wald(1, 100)) == (-1.0, 3.0)       # a negative rate
        assert tuple(R1(x) for x in B.wald(0, 100)) == (0.0, 0.0)        # certainty from nothing
        assert tuple(R1(x) for x in B.wilson(5, 100)) == (2.2, 11.2)
        z = B.z_for(); assert round(z * z / (100 + z * z) * 100, 1) == 3.7
        """),
        md("## 1.5 — the worked example: 100 support conversations, four failure modes"),
        code("""
        RATES = {"wrong policy": 0.06, "invented order status": 0.03, "missed escalation": 0.02, "tone": 0.08}
        seen = B.simulate_failure_reading(RATES, 100, seed=1)
        for k, v in seen.items():
            lo, hi = B.wilson(v, 100)
            print(f"{k:22s} true {RATES[k]:4.0%}   seen {v:2d}/100   Wilson [{lo:.1%}, {hi:.1%}]")
        assert seen == {"wrong policy": 9, "invented order status": 7, "missed escalation": 0, "tone": 13}
        assert B.wilson(7, 100)[0] > 0.03          # the one interval that misses its true rate
        assert round(0.98 ** 100, 3) == 0.133      # chance the 2% escalation failure is invisible
        """),
        md("## Exercises 1.2–1.5"),
        code("""
        assert B.reads_to_see(0.005) == 598 and B.reads_to_see(0.005, 0.80) == 322
        assert round(1 - 0.05 ** (1 / 200), 4) == 0.0149
        cov_wald, cov_wilson = B.interval_coverage(0.02, 100, "wald"), B.interval_coverage(0.02, 100, "wilson")
        print("coverage at p=2%, n=100:  Wald", cov_wald, "  Wilson", cov_wilson)
        assert (cov_wald, cov_wilson) == (0.8645, 0.9447)
        runs = B.reading_runs(RATES, 100)
        miss = sum(r["missed escalation"] == 0 for r in runs) / 1000
        inv = sum(r["invented order status"] > r["wrong policy"] for r in runs) / 1000
        tie = sum(r["invented order status"] == r["wrong policy"] for r in runs) / 1000
        print("escalation invisible:", miss, "  3% mode ranked above 6% mode:", inv, "  tied:", tie)
        assert (miss, inv, tie) == (0.135, 0.120, 0.083)
        item = 1200 * 3e-6 + 300 * 15e-6 + 2000 * 3e-6 + 50 * 15e-6
        print(f"cost per item ${item:.4f}, per 1,500-item run ${item * 1500:.2f}")
        assert round(item * 1500) == 22
        """),
    ]


@notebook("02_golden")
def _nb02():
    return [
        md("""
        # Chapter 2 — Golden sets: building, sampling, maintaining

        Stratified allocation and its standard errors, weighting an over-sampled risk slice back to traffic, the
        size table n = z²p(1−p)/m², and the selection inflation from picking the best of K equally good variants,
        predicted by extreme-value theory and measured by simulation. Every cell **asserts** what it checks.
        """),
        code(SETUP, "from evalkit import basics as B"),
        md("## 2.1 — three allocations of 400 items across four intents"),
        code("""
        SIZES = {"billing": 6000, "shipping": 3000, "returns": 800, "account": 200}
        SDS = {"billing": 0.3, "shipping": 0.3, "returns": 0.45, "account": 0.5}
        for rule in ("proportional", "neyman", "equal"):
            a = B.stratified_allocation(SIZES, SDS, 400, rule)
            print(f"{rule:12s} {a}  overall SE {B.stratified_se(SIZES, SDS, a) * 100:.2f} points")
        assert list(B.stratified_allocation(SIZES, SDS, 400, "neyman").values()) == [228, 114, 45, 13]
        # Monte Carlo check of the stratified SE formula under proportional allocation
        rng = np.random.default_rng(0); N = sum(SIZES.values())
        p_true = {"billing": 0.9, "shipping": 0.9, "returns": 0.7, "account": 0.5}
        a = B.stratified_allocation(SIZES, SDS, 400, "proportional")
        est = [sum(SIZES[h] / N * rng.binomial(a[h], p_true[h]) / a[h] for h in SIZES) for _ in range(20_000)]
        print("simulated SE:", round(np.std(est) * 100, 2), " formula:", round(B.stratified_se(SIZES, SDS, a) * 100, 2))
        assert abs(np.std(est) * 100 - B.stratified_se(SIZES, SDS, a) * 100) < 0.05
        """),
        md("## 2.2 — weighting an over-sampled risk slice back to traffic"),
        code("""
        p, se = B.weighted_rate({"normal": (0.95, 300, 276), "risky": (0.05, 300, 240)})
        print(f"unweighted {(276 + 240) / 600:.1%}   weighted {p:.1%} ± {1.96 * se * 100:.1f}")
        assert round(p, 3) == 0.914 and round(se * 100, 2) == 1.49
        """),
        md("## 2.3 — the size table"),
        code("""
        for m in (.01, .02, .03, .05, .07):
            print(f"±{m * 100:.0f} points:", [B.n_for_margin(m, p) for p in (.5, .8, .9)])
        assert [B.n_for_margin(m) for m in (.01, .02, .03, .05, .07)] == [9604, 2401, 1068, 385, 196]
        """),
        md("## 2.5 — selection inflation: predicted by E[max of K normals], then measured"),
        code("""
        for K in (2, 5, 10, 20, 50):
            print(K, round(B.expected_max_normal(K), 3), f"+{B.selection_bias(0.7, 200, K) * 100:.1f} points")
        # the numerical integral against a direct simulation of the maximum of K normals
        sim = np.random.default_rng(1).standard_normal((200_000, 20)).max(axis=1).mean()
        assert abs(sim - B.expected_max_normal(20)) < 0.01
        meas = B.simulate_selection(0.7, 200, 20)
        print("measured inflation, K=20:", round(meas * 100, 2), "points")
        assert round(meas * 100, 1) == 5.9 and round(B.selection_bias(0.7, 200, 20) * 100, 1) == 6.1
        """),
        md("## Exercises"),
        code("""
        sizes = {"en": 85, "es": 10, "de": 5}
        sds = {k: math.sqrt(p * (1 - p)) for k, p in {"en": .92, "es": .85, "de": .75}.items()}
        for rule in ("proportional", "neyman", "equal"):
            a = B.stratified_allocation(sizes, sds, 600, rule)
            print(rule, a, round(B.stratified_se(sizes, sds, a) * 100, 2))
        assert list(B.stratified_allocation(sizes, sds, 600, "neyman").values()) == [481, 74, 45]
        p, se = B.weighted_rate({"g": (0.915, 500, 440), "l": (0.08, 200, 140), "i": (0.005, 100, 60)})
        assert round(p * 100, 1) == 86.4 and round(se * 100, 2) == 1.35
        assert round(B.simulate_selection(0.8, 400, 10) * 100, 1) == 3.0
        """),
    ]


@notebook("03_labels")
def _nb03():
    return [
        md("""
        # Chapter 3 — Labels and agreement

        Cohen's kappa from the 2×2 tables, the kappa paradox, Fleiss' kappa and Krippendorff's alpha on the
        published examples, the three-rater worked example, and the sampling spread of kappa. Every cell
        **asserts** what it checks.
        """),
        code(SETUP, "from evalkit import agreement as A"),
        md("## 3.2 — kappa, and the paradox"),
        code("""
        for t in [(70, 10, 5, 15), (45, 5, 5, 45), (90, 4, 4, 2), (92, 3, 3, 2)]:
            r = A.kappa_from_table(*t)
            print(t, f"p_o={r['po']:.2f}  p_e={r['pe']:.3f}  kappa={r['kappa']:.3f}")
        assert round(A.kappa_from_table(70, 10, 5, 15)["kappa"], 3) == 0.571
        assert round(A.kappa_from_table(90, 4, 4, 2)["kappa"], 3) == 0.291     # 92% agreement, kappa 0.29
        a, b = A.table_to_labels(70, 10, 5, 15)
        assert round(A.cohen_kappa(a, b), 3) == 0.571
        """),
        md("## 3.3 — the published examples (Krippendorff 2011; the standard Fleiss example)"),
        code("""
        N = None
        units = [list(t) for t in zip([1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N], [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
                                      [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N], [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N])]
        print("alpha:", round(A.krippendorff_alpha_nominal(units), 3))
        assert round(A.krippendorff_alpha_nominal(units), 3) == 0.743
        fl = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
              [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
        print("Fleiss:", round(A.fleiss_kappa(fl), 3))
        assert round(A.fleiss_kappa(fl), 3) == 0.210
        """),
        md("## 3.3 — three raters, ten items: find the rater who drifted"),
        code("""
        THREE = [[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]]
        cnt = [[sum(x == 1 for x in u), sum(x == 0 for x in u)] for u in THREE]
        lab = {r: [u[i] for u in THREE] for i, r in enumerate("ABC")}
        print("Fleiss", round(A.fleiss_kappa(cnt), 3), " alpha", round(A.krippendorff_alpha_nominal(THREE), 3))
        print({k: round(v, 3) for k, v in A.pairwise_cohen(lab).items()})
        assert round(A.pairwise_cohen(lab)[("A", "C")], 3) == 0.4
        gaps = [list(u) for u in THREE]; gaps[1][2] = gaps[6][1] = gaps[8][2] = None
        assert round(A.krippendorff_alpha_nominal(gaps), 3) == 0.519
        """),
        md("## Exercises — kappa by hand, and kappa's sampling spread"),
        code("""
        assert round(A.kappa_from_table(120, 20, 10, 50)["kappa"], 3) == 0.659
        assert round(A.kappa_from_table(165, 20, 10, 5)["kappa"], 2) == 0.17
        for n in (50, 100, 200, 400):
            ks = A.simulate_kappa(0.6, 0.7, n)
            print(n, "items: 95% of measured kappas in", (round(A.percentile(ks, .025), 2), round(A.percentile(ks, .975), 2)))
        ks = A.simulate_kappa(0.6, 0.7, 50)
        assert (round(A.percentile(ks, .025), 2), round(A.percentile(ks, .975), 2)) == (0.33, 0.82)
        """),
    ]


@notebook("04_metrics")
def _nb04():
    return [
        md("""
        # Chapter 4 — Task metrics, proxy metrics, and cost

        Exact match and F1 on the date example, BLEU / ROUGE-L / F1 ranking a negation above a paraphrase,
        calibration (ECE, Brier) under three amounts of overconfidence, and the Pareto front of five
        configurations. Every cell **asserts** what it checks.
        """),
        code(SETUP, "from evalkit import metrics as M"),
        md("## 4.1 — exact match and F1"),
        code("""
        GOLD = "The order ships on March 3."
        for r in ["the order ships on March 3", "It ships March 3rd.", "Your order will ship on 3 March", "The order ships on March 13."]:
            print(f"{r:36s} EM {M.exact_match(r, GOLD, norm=False)}  normEM {M.exact_match(r, GOLD)}  F1 {M.token_f1(r, GOLD):.2f}")
        assert round(M.token_f1("The order ships on March 13.", GOLD), 2) == 0.80     # the wrong date, high partial credit
        """),
        md("## 4.2 — overlap prefers the negation"),
        code("""
        REF = "You can return the item within 30 days for a full refund."
        PARA = "Returns are accepted for a full refund if you send it back within a month."
        NEG = "You cannot return the item within 30 days for a full refund."
        for name, s in (("paraphrase (right)", PARA), ("negation (wrong)", NEG)):
            print(f"{name:20s} BLEU {M.sentence_bleu(s, [REF]):.3f}  ROUGE-L {M.rouge_l(s, REF):.3f}  F1 {M.token_f1(s, REF):.3f}")
        assert M.sentence_bleu(NEG, [REF]) > 5 * M.sentence_bleu(PARA, [REF])
        assert M.rouge_l(NEG, REF) > M.rouge_l(PARA, REF) and M.token_f1(NEG, REF) > M.token_f1(PARA, REF)
        """),
        md("## 4.4 — calibration, and Exercise 4.3's threshold"),
        code("""
        for oc in (0.0, 0.1, 0.2):
            c, o = M.simulate_confidences(5000, oc)
            print(f"overconfidence {oc:.1f}: accuracy {np.mean(o):.3f}  mean conf {np.mean(c):.3f}  ECE {M.ece(c, o) * 100:.1f}  Brier {M.brier(c, o):.3f}")
        c, o = M.simulate_confidences(5000, 0.2)
        shown = [y for x, y in zip(c, o) if x >= 0.9]
        err = 1 - np.mean(shown)
        print(f"policy 'show if >= 0.9' at +20 overconfidence: {len(shown)} shown, {err:.1%} of them wrong")
        promised = 1 - np.mean([x for x in c if x >= 0.9])
        print(f"stated confidence of shown answers averages {1 - promised:.3f}: a promised error rate of {promised:.1%}")
        assert len(shown) == 3015 and round(err, 3) == 0.148 and err > 8 * promised
        # recalibrate by bins on the first half, apply to the second half
        half = 2500
        bins = {}
        for x, y in zip(c[:half], o[:half]):
            bins.setdefault(min(9, int(x * 10)), []).append(y)
        rec = {b: np.mean(v) for b, v in bins.items()}
        print("recalibrated accuracy per bin:", {b: round(v, 3) for b, v in sorted(rec.items())})
        c2 = [rec.get(min(9, int(x * 10)), x) for x in c[half:]]
        shown2 = [y for x, y in zip(c2, o[half:]) if x >= 0.9]
        print(f"after recalibration: {len(shown2)} of {half} clear 0.9")
        assert len(shown2) == 0 and round(max(rec.values()), 2) == 0.86     # nothing is 90% reliable
        """),
        md("## 4.5 — the Pareto front and the cascade"),
        code("""
        CONFIGS = {"large": (0.91, 0.0140), "medium": (0.88, 0.0042), "small": (0.79, 0.0009),
                   "medium+retry": (0.90, 0.0061), "large-old": (0.87, 0.0150)}
        print("front:", M.pareto_front(CONFIGS))
        assert M.pareto_front(CONFIGS) == ["small", "medium", "medium+retry", "large"]
        big, small = M.cost_per_call(1200, 300, 3, 15), M.cost_per_call(1200, 300, 0.25, 1.25)
        casc = small + 0.22 * big
        print(f"large ${big:.4f}  small ${small:.6f}  cascade ${casc:.5f}  monthly saving ${(big - casc) * 1e6:,.0f}")
        assert round((big - casc) * 1e6) == 5643
        """),
    ]
