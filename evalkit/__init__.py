"""
evalkit: the reference code for Evaluation & Experimentation for LLM Systems.

Every number the course states is computed by a function in this package and
asserted by tests/test_evalkit.py. Pure Python plus numpy; no scipy, so every
statistic is derived here rather than imported.

    basics      reading budgets, proportions and their intervals, sample sizes (Ch 1-2)
    agreement   Cohen's and Fleiss' kappa, Krippendorff's alpha (Ch 3)
    metrics     exact match, F1, BLEU, ECE, Brier, Pareto fronts (Ch 4)
    stats       bootstrap, clustered errors, paired tests, power (Ch 5-6)
    rank        multiple testing, Bradley-Terry, Elo, item response theory (Ch 7)
    variance    sampling, pass@k / pass^k, variance decomposition (Ch 8)
    judges      judge biases, Rogan-Gladen, prediction-powered inference (Ch 9-11)
    systems     retrieval metrics, agents, rare events, contamination (Ch 12-15)
    online      CI gates, CUPED, sequential tests (Ch 16-17)
"""
