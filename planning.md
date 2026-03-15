# Research Plan: Lying Style

## Motivation & Novelty Assessment

### Why This Research Matters
Instruction-conditioned deception is a practical safety concern: if lie-inducing prompts produce distinct writing distributions, systems can monitor outputs for deception risk without model internals. This benefits model providers, red-teamers, and downstream users who need lightweight runtime detection and auditing tools.

### Gap in Existing Work
From `literature_review.md`, most strong results detect deception via internal activations/probes, while fewer studies test whether *output text alone* reliably separates truthful vs lie-conditioned generation under paired prompts and modern API models. Existing work also underemphasizes distribution tests plus practical black-box detectors in one unified protocol.

### Our Novel Contribution
We test a black-box, output-only hypothesis on a current real API model (OpenAI GPT-4.1 family), using paired truthful vs lie-conditioned responses to identical prompts. We combine stylometric distribution divergence (JS, Wasserstein, MMD) with downstream detectability (cross-validated classifier AUROC/F1) and robustness checks across seeds and instruction variants.

### Experiment Justification
- Experiment 1: Paired truthful vs lie generation on factual prompts. Needed to isolate instruction effect while controlling topic.
- Experiment 2: Distribution divergence analysis (length/lexical/syntactic proxies + embedding geometry). Needed to quantify “noticeably different” beyond classifier accuracy.
- Experiment 3: Output-only deception classification with strict train/test split and permutation significance. Needed to assess practical detectability.
- Experiment 4: Robustness across multiple lie instructions and decoding seeds. Needed to rule out brittle prompt artifacts.

## Research Question
Do outputs from a language model under lie-inducing instructions (roleplay/jailbreak framing) come from a measurably different text distribution than outputs under truthful instructions?

## Background and Motivation
Prior studies show deception signals in internal representations and behavior under deceptive prompts. Operationally, many deployments only have access to text outputs, so establishing output-level separability is necessary for practical monitoring and model governance.

## Hypothesis Decomposition
- H1 (distribution shift): Lie-conditioned outputs differ in stylometric/semantic feature distribution from truthful outputs.
- H2 (detectability): A text-only classifier can distinguish conditions significantly above chance.
- H3 (robustness): The shift persists across multiple deceptive prompt templates and random seeds.

Independent variables:
- Instruction condition: truthful vs lie-roleplay vs jailbreak-style lie.
- Decoding seed.

Dependent variables:
- Divergence metrics: Jensen-Shannon, Wasserstein, MMD.
- Classifier metrics: AUROC, F1, accuracy.
- Auxiliary style metrics: length, hedging rate, certainty markers, punctuation entropy, type-token ratio.

Success criteria:
- AUROC >= 0.75 on held-out split for truthful vs lie (text-only).
- At least two divergence metrics significantly > permutation null (p < 0.05, FDR corrected).
- Effect size at least medium (Cohen's d >= 0.5) on >=2 core features.

Alternative explanations:
- Prompt-template artifacts drive separation rather than deception itself.
- Length effects dominate stylometric metrics.
- Topic imbalance across prompt subsets.
Mitigations:
- Paired prompting, length-normalized features, topic-stratified splits, robustness checks.

## Proposed Methodology

### Approach
Use `truthful_qa` questions as shared prompt backbone. For each question, request short answers under different system instructions (truthful baseline vs lie-roleplay vs jailbreak-like deceptive framing), keeping temperature/max token settings fixed except in seed robustness runs. Build a feature matrix from outputs; evaluate divergence and classification with statistical testing.

### Experimental Steps
1. Validate environment, seeds, and API access; log versions and hardware.
2. Load and quality-check datasets (`truthful_qa_generation`, jailbreak prompt banks).
3. Construct evaluation prompt set (target n=200 questions; balanced categories where available).
4. Generate outputs with real API calls for each condition; cache raw JSONL with metadata.
5. Extract textual features and embeddings; save tidy analysis tables.
6. Compute descriptive statistics and divergences with permutation tests.
7. Train/evaluate baselines and main classifier using stratified CV + held-out test.
8. Run robustness ablations (additional lie template, extra seed).
9. Perform error analysis on false positives/false negatives.

### Baselines
- Random/majority classifier baseline.
- Length-only logistic regression baseline.
- TF-IDF + logistic regression (main practical baseline).
- Optional embedding + linear classifier as stronger comparator.

### Evaluation Metrics
- Classification: AUROC, AUPRC, F1, accuracy, balanced accuracy.
- Distribution: Jensen-Shannon divergence, Wasserstein distance, MMD (RBF kernel).
- Effect size/statistics: Cohen's d, 95% bootstrap CI, permutation p-values.

### Statistical Analysis Plan
- Null hypotheses:
  - H0-1: No difference in feature distributions between truthful and lie conditions.
  - H0-2: Classifier AUROC equals chance (0.5).
- Tests:
  - Mann-Whitney U or Welch t-test (assumption-dependent) for feature differences.
  - Permutation test for AUROC and MMD significance.
  - Benjamini-Hochberg FDR for multiple comparisons.
- Alpha: 0.05.
- Report effect sizes and confidence intervals alongside p-values.

## Expected Outcomes
Support for hypothesis if lie-conditioned outputs show consistent divergence and are classifiable above chance across templates/seeds. Refutation if distributions strongly overlap and classifiers fail to generalize.

## Timeline and Milestones
- Phase 0-1 planning: complete now.
- Phase 2 setup + data checks: ~20 min.
- Phase 3 implementation: ~60 min.
- Phase 4 experiments: ~60 min (API latency dependent).
- Phase 5 analysis: ~40 min.
- Phase 6 documentation + validation: ~30 min.
Buffer (25%) included for retries/debugging.

## Potential Challenges
- API rate limits/cost variability: use batching, retries, and caching.
- Safety refusals in lie condition: keep prompts non-harmful and focused on factual lying.
- Small-sample instability: run bootstrap CIs and multiple seeds.
- Confounding by verbosity: include normalized features and length-matched checks.

## Success Criteria
- End-to-end reproducible pipeline with saved raw outputs and analysis artifacts.
- Statistically supported conclusion regarding distributional difference.
- REPORT.md with actual tables/figures and transparent limitations.
