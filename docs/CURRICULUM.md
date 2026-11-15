# Evaluation & Experimentation for LLM Systems: curriculum

*Course 1 of the library · Everyone track · 18 chapters · about 23 hours · status: preview. Generated from `curricula/evals.txt` in the hub repository; edit it there.*

How to know whether an AI system actually works. You will define "good" before writing code, build a golden set and a judge you have validated against people, put an honest error bar on every number, and turn all of it into a regression suite that gates every change. Transformers Ch 26 taught how to evaluate a model; this course evaluates whole systems, in development and in production.

**For:** Engineers who can build LLM features but cannot yet prove one version beats another.

**By the end you can:**

- Design a golden set and say how big it must be
- Build an LLM judge and correct its scores for its own measured error
- Decide whether a 3% win is real, with the right paired test
- Evaluate retrieval, agents and safety on their own terms
- Run evals as CI gates and as online experiments

**Before you start:** Transformers From Scratch Ch 26 (Evaluating LLMs); Transformers From Scratch Ch 11 (Training a Transformer); Transformers From Scratch Ch 13 (Decoding and Sampling)

**Current to November 2026:** Judge-error correction with prediction-powered inference (PPI++) and Rogan–Gladen; rubric-based judging and rubrics as RL rewards; agent evaluation by end state and pass^k; harness sensitivity (the same model ranks differently in another harness); item response theory for small, adaptive benchmarks; batch-invariant inference as a variance source; always-valid sequential tests for online experiments.

**Not covered:** Model training (see Post-Training), and any public benchmark you cannot reproduce yourself.

**Reference package:** `evalkit/` · **Semantic hues:** dim-1 = the system under test; dim-2 = the reference or baseline; dim-3 = human labels; dim-4 = the judge; dim-5 = the uncertainty


## Part I · What "works" means

*Define the target before measuring anything.*

### 01 · Defining "Good" Before Writing Code (55 min)

*Operational definitions, metric trees, and the eval-driven loop.*

Builds on: Transformers From Scratch Ch 26 (Evaluating LLMs)

1. **From a product goal to a measurable property** — the user, the task, and what a failure costs
2. **Metric trees** — one north-star outcome, the drivers under it, and the guardrails beside it
3. **Error analysis first** — read 100 outputs, cluster the failures, then decide what to measure
4. **The eval-driven development loop** — hypothesis, change, measure, keep or revert
5. **Worked example** — defining "good" for a support assistant, in four failure modes

**Build:** an error-analysis notebook that turns 100 raw outputs into a failure taxonomy with counts.

### 02 · Golden Sets: Building, Sampling, Maintaining (65 min)

*Where the items come from, and how many you need.*

Builds on: this course Ch 01 (Defining "Good" Before Writing Code)

1. **Sampling from real traffic** — stratify by intent, length and difficulty, not by convenience
2. **Hard cases on purpose** — adversarial, boundary and long-tail items, labelled as such
3. **Size from the error bar you can tolerate** — the n you need for a ±2-point interval
4. **Versioning and refresh** — golden sets rot as the product and the traffic change
5. **Held-out discipline** — the set you tune on is not the set you report

**Build:** a stratified sampler and a size calculator; a versioned golden-set format with provenance.

### 03 · Labels and Agreement (60 min)

*Guidelines, kappa and alpha, and disagreement as signal.*

Builds on: this course Ch 02 (Golden Sets: Building, Sampling, Maintaining); Transformers From Scratch Ch 26 (Evaluating LLMs)

1. **Annotation guidelines that work** — definitions, examples, counterexamples, and an escalation rule
2. **Agreement, corrected for chance** — Cohen's kappa, Fleiss' kappa, Krippendorff's alpha, derived
3. **When agreement is low** — an ambiguous task, a bad guideline, or a genuinely subjective property
4. **Adjudication and gold labels** — majority, expert override, or keep the distribution
5. **What labels cost** — time per item, the budget, and where expert labels matter most

**Build:** kappa and alpha from scratch, checked against reference implementations; a label-audit report.

### 04 · Task Metrics, Proxy Metrics, and Cost (60 min)

*Exact match to execution, and why BLEU misleads.*

Builds on: this course Ch 01 (Defining "Good" Before Writing Code)

1. **Reference-based metrics** — exact match, token F1, normalisation, and what they reward
2. **Why n-gram overlap fails for generation** — BLEU and ROUGE against human judgement, measured
3. **Execution-based metrics** — unit tests, schema validation, programmatic checkers
4. **Calibration as a metric** — ECE and the Brier score for systems that report confidence
5. **Cost and latency are outcomes too** — tokens, dollars and p95 latency in the same table as quality

**Build:** a metric library with a test for every metric; a Pareto plot of quality against cost.


## Part II · Statistics you cannot skip

*Every number gets an error bar.*

### 05 · Uncertainty: Error Bars That Are Honest (70 min)

*Standard errors, bootstrap, and clustered data.*

Builds on: Transformers From Scratch Ch 26 (Evaluating LLMs)

Related: Mathematics for Machine Learning, from First Principles Ch 10 (Concentration and Limits)

1. **The standard error of a mean score** — and why 200 items gives ±7 points at 50%
2. **The bootstrap** — resample items, never tokens; percentile and BCa intervals
3. **Clustered data** — several samples per prompt, several turns per conversation, several chunks per document
4. **Clustered standard errors** — the correction most leaderboards forget, derived and measured
5. **Worked example** — the same eval reported naively and correctly, side by side

**Build:** bootstrap and clustered-SE routines; a simulation showing naive intervals undercover.

### 06 · Comparing Two Systems (75 min)

*Paired tests, power, and the size of a real win.*

Builds on: this course Ch 05 (Uncertainty: Error Bars That Are Honest)

1. **Paired designs** — both systems on the same items, and why that shrinks the interval
2. **McNemar's test, the paired bootstrap and the permutation test, compared on one dataset**
3. **Power and minimum detectable effect** — how many items to detect a 2-point change
4. **The disagreement rate decides everything** — n from the discordant fraction, derived
5. **Reporting a comparison** — effect size, interval, n, and what it does not show

**Build:** a compare() that picks the paired test, reports the interval, and refuses underpowered claims.

### 07 · Many Comparisons, Rankings, and Small Benchmarks (70 min)

*Multiple testing, Bradley–Terry, and item response theory.*

Builds on: this course Ch 06 (Comparing Two Systems); Transformers From Scratch Ch 23 (SFT, LoRA, RLHF, and PPO)

1. **Multiple comparisons** — why testing 20 prompts finds a "winner" by chance; Holm and Benjamini–Hochberg
2. **Rankings from pairwise votes** — Bradley–Terry and Elo with confidence intervals
3. **Item response theory** — item difficulty and discrimination, and what each item tells you
4. **Small, adaptive benchmarks** — choosing the 100 items that rank models like the full 10,000
5. **Leaderboard hygiene** — ties, overlapping intervals, and the ranks you are allowed to report

**Build:** Bradley–Terry with bootstrap intervals; a 2-parameter IRT fit; an adaptive subset selector.

### 08 · Where the Variance Comes From (60 min)

*Sampling, seeds, prompts, and nondeterministic inference.*

Builds on: Transformers From Scratch Ch 13 (Decoding and Sampling); this course Ch 05 (Uncertainty: Error Bars That Are Honest)

1. **Sampling variance** — temperature, n samples per item, and pass@k against pass^k
2. **Prompt sensitivity** — formatting and wording changes that move scores more than models do
3. **Nondeterminism in inference** — batch size and kernel choice change outputs; batch-invariant inference
4. **Decomposing variance** — items, samples and seeds, and where to spend the budget
5. **Worked example** — a "2-point regression" that is 80% prompt formatting

**Build:** a variance-decomposition harness; a demonstration of batch-dependent outputs on one model.


## Part III · Judges

*Models grading models, and how far to trust them.*

### 09 · LLM-as-Judge (70 min)

*Pointwise, pairwise, rubric — and the biases you can measure.*

Builds on: Transformers From Scratch Ch 26 (Evaluating LLMs); this course Ch 03 (Labels and Agreement)

1. **Three formats** — pointwise scores, pairwise preference, rubric checklists; what each can detect
2. **Measured biases** — position, verbosity, self-preference, and style over substance
3. **Debiasing** — order swaps, length control, reference answers, and a judge from another model family
4. **Thinking judges** — reasoning before the verdict, its cost, and calibration of the vote
5. **Prompting a judge** — criteria, scales, examples, and forcing a decision

**Build:** a judge harness that runs both orders, controls for length, and logs every rationale.

### 10 · Validating and Correcting a Judge (75 min)

*Agreement with people, and fixing the estimate.*

Builds on: this course Ch 09 (LLM-as-Judge); this course Ch 05 (Uncertainty: Error Bars That Are Honest)

1. **The judge is a measuring instrument** — agreement with human labels, by slice
2. **Its error rates, measured** — false-positive and false-negative rates against a gold sample
3. **Correcting the estimate** — Rogan–Gladen for a judged pass rate, derived
4. **Prediction-powered inference** — PPI and PPI++: a few human labels plus many judged ones, with valid intervals
5. **Where to spend human labels** — budget allocation and active selection
6. **Drift** — re-validate on a schedule, because the judge and the traffic both move

**Build:** Rogan–Gladen and PPI++ estimators, with a simulation proving interval coverage.

### 11 · Rubrics, Reward Models, and Goodhart (60 min)

*When the evaluator becomes the target.*

Builds on: this course Ch 10 (Validating and Correcting a Judge)

Related: Post-Training in Practice Ch 17 (Rewards Without Verifiers)

1. **Writing rubrics** — atomic, checkable criteria written by domain experts
2. **Rubrics as rewards** — the same rubric scoring RL rollouts, and what changes when it is optimised against
3. **Reward models as evaluators** — and why their scores drift off-distribution
4. **Goodhart in practice** — evaluator hacking, measured on a small example
5. **Keeping evaluators honest** — held-out judges, human spot checks, and adversarial audits

**Build:** a rubric grader; a toy optimisation loop that exploits a weak rubric, then the fix.


## Part IV · Evaluating systems

*Retrieval, agents, safety, and contamination.*

### 12 · Evaluating Retrieval-Augmented Systems (65 min)

*Retrieval first, then grounding, then the answer.*

Builds on: this course Ch 04 (Task Metrics, Proxy Metrics, and Cost)

Related: Retrieval & Context Engineering Ch 17 (Evaluating Retrieval End to End)

1. **Evaluate the parts separately** — retrieval, grounding, and answer quality fail independently
2. **Retrieval metrics** — recall@k, MRR, nDCG, derived and computed on one example
3. **Groundedness and faithfulness** — is every claim supported by the retrieved context?
4. **Citation correctness** — the cited passage must support the sentence that cites it
5. **Building retrieval labels cheaply** — pooled judgments and synthetic queries, with their biases

**Build:** a RAG eval that reports retrieval, grounding and answer scores per query.

### 13 · Evaluating Agents (75 min)

*Trajectory and outcome, end state, and pass^k.*

Builds on: this course Ch 08 (Where the Variance Comes From)

Related: Agents & Tool Use Ch 15 (Evaluating Agents)

1. **Outcome against trajectory** — what the end state proves and what the path reveals
2. **State-checked environments** — compare the database after the run, not the transcript
3. **Reliability** — pass^k, and why 90% per attempt is 35% across ten attempts
4. **Cost, steps and time as part of the score**
5. **Harness sensitivity** — the same model, different tools and prompts, a different rank
6. **Reproducible environments** — seeds, snapshots, and simulated users

**Build:** a small state-checked environment with a simulated user; pass@k and pass^k reports.

### 14 · Safety, Refusals, and Rare Events (55 min)

*Over-refusal, under-refusal, and the rule of three.*

Builds on: this course Ch 05 (Uncertainty: Error Bars That Are Honest)

Related: AI Security & Red Teaming Ch 12 (Red-Team Methodology)

1. **Two errors** — harmful compliance and needless refusal, measured on separate sets
2. **Rare events** — zero failures in 300 tries means up to 1%: the rule of three
3. **Guardrail metrics** — precision and recall of a filter, and their cost to helpfulness
4. **Red-team success rates with intervals** — attack success is a proportion too
5. **Severity weighting** — not all failures are equal; reporting by severity

**Build:** an over-refusal and under-refusal suite; exact binomial bounds for rare failures.

### 15 · Contamination and Leakage (55 min)

*Your benchmark is in someone's training data.*

Builds on: Transformers From Scratch Ch 26 (Evaluating LLMs)

1. **How contamination happens** — web scrapes, synthetic data, and fine-tuning on your own eval
2. **Detection** — n-gram overlap, Min-K% probability, canaries, and memorisation probes
3. **Temporal splits** — items written after the training cutoff
4. **Rephrased and perturbed benchmarks** — the score that survives a rewrite is the real one
5. **Private holdouts** — keeping an evaluation set that never leaves your control

**Build:** an n-gram decontamination tool and a rephrase-robustness test.


## Part V · Evaluation in production

*From a notebook to a gate to an experiment.*

### 16 · Regression Suites and CI Gates (60 min)

*Evals that run on every change.*

Builds on: this course Ch 06 (Comparing Two Systems)

Related: Production AI Systems Ch 09 (Prompts and Configs as Code)

1. **Three tiers** — fast deterministic checks, a sampled model-graded suite, a full nightly run
2. **Thresholds with error bars** — fail on a significant regression, not on noise
3. **Flaky evals** — find them, quarantine them, and keep the gate trusted
4. **Cost of an eval run** — caching responses, sampling the suite, and budgeting judge calls
5. **Snapshot testing for prompts** — diffs a reviewer can read

**Build:** a CI gate (GitHub Actions) that runs a tiered suite and blocks significant regressions.

### 17 · Online Evaluation and A/B Tests (75 min)

*Real users, guardrail metrics, and sequential tests.*

Builds on: this course Ch 06 (Comparing Two Systems)

Related: Production AI Systems Ch 10 (Releasing Changes)

1. **What to measure live** — task success proxies, implicit feedback, and guardrail metrics
2. **Randomisation units** — users, sessions or requests, and the interference between them
3. **Variance reduction** — CUPED with pre-experiment data, derived
4. **Peeking** — why stopping when p < 0.05 is wrong, and always-valid sequential tests
5. **Interleaving for rankers, and novelty effects for everything else**

**Build:** an A/B analysis with CUPED and an always-valid sequential test, on simulated traffic.

### 18 · Capstone: An Evaluation Suite for a Real Feature (300 min)

*Everything above, on one system.*

Builds on: this course Ch 16 (Regression Suites and CI Gates); this course Ch 17 (Online Evaluation and A/B Tests)

1. **The specification** — a feature, its failure modes, and the decisions the suite must support
2. **Golden set with measured agreement** — and its size justified by the interval you need
3. **A validated judge, corrected with PPI** — and its error rates reported
4. **Paired comparison of two versions** — with power, intervals, and a verdict
5. **The CI gate and the online experiment plan** — with the power calculation for the experiment

**Build:** the complete suite, its report, and the tests that keep it honest.

## Capstone

An evaluation suite for a real LLM feature: golden set with measured agreement, a validated and corrected judge, paired significance, a CI gate, and an online experiment plan with its power calculation.
