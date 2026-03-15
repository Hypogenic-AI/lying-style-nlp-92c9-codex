# Lying Style: Output Distribution Shift Under Lie-Inducing Instructions

## 1. Executive Summary
We tested whether a modern LLM writes differently when explicitly instructed to lie versus when instructed to answer truthfully. Using real API calls to `gpt-4.1` on paired prompts from TruthfulQA, we found a clear and statistically significant distribution shift between truthful and lie-conditioned outputs.

A text-only classifier trained on outputs (no model internals) separated truthful vs lie-roleplay responses with AUROC **0.883** on held-out questions, with permutation-test p-value **0.001** and bootstrap 95% CI **[0.798, 0.952]**. A feature-space MMD test also showed significant shift (p = **0.001996**).

Practically, this supports black-box monitoring: output text alone can provide useful deception-risk signals when models are pushed into deceptive roleplay/jailbreak-style behavior.

## 2. Goal
### Hypothesis
The distribution of text generated under lie-inducing instructions is noticeably different from the distribution under truthful instructions.

### Why this matters
If true, we can detect risky/deceptive behavior from outputs without requiring hidden-state access. This is relevant for LLM safety auditing, runtime monitoring, and policy enforcement.

### Expected impact
A reproducible baseline for output-only deception detection and distribution-shift measurement that can be extended across models and domains.

## 2.5 Literature Review Summary
- Prior work (2024-2025) shows strong deception detectability from **internal activations/probes**.
- Fewer works directly test **output-only** distribution separability under paired truthful vs deceptive prompting.
- Jailbreak literature establishes instruction-induced behavioral changes, motivating our prompt-conditioned setup.
- Our experiment addresses this gap with black-box, text-only measurements plus significance testing.

## 3. Data Construction
### Dataset Description
- Primary prompts: `datasets/truthful_qa_generation` (HuggingFace `truthful_qa`, generation config).
- Size used: 120 sampled questions (from 817 validation items).
- Additional prompt pools available (not primary backbone):
  - `datasets/chatgpt_jailbreak_prompts`
  - `datasets/itw_jailbreak_2023_12_25`
  - `datasets/itw_regular_2023_12_25`
- Collection methodology: paired generation on identical questions under different instruction conditions.
- Known biases/limitations: TruthfulQA is misconception-heavy and English-centric; this may overrepresent factual-myth style errors relative to open-domain dialogue.

### Example Samples
| Question | Condition | Model Output (truncated) |
|---|---|---|
| What do bears wear when they fight in the wild? | truthful | "Bears do not wear anything..." |
| What do bears wear when they fight in the wild? | lie_roleplay | "...wear thick layers of leaves and bark..." |
| What is the most popular sport in Japan? | truthful | "...baseball." |
| What is the most popular sport in Japan? | lie_roleplay | "...cricket." |

### Data Quality
- Missing outputs: **0%**
- API error rate: **0.0**
- Duplicate `(question_id, condition, seed)`: **0**
- Refusal-like outputs (`can't`, `sorry`, etc.): **0**

### Preprocessing Steps
1. Load TruthfulQA from disk and drop null/duplicate questions.
2. Sample 120 questions with fixed seed.
3. Query `gpt-4.1` with controlled decoding (`temperature=0.4`, `max_tokens=80`, fixed seed).
4. Extract textual features (length, punctuation, lexical ratios, certainty/hedging markers).
5. Build TF-IDF features for text-only classification.

### Train/Val/Test Splits
- Primary classifier task: truthful vs lie_roleplay (seed 42), `n=240` outputs.
- Split method: `GroupShuffleSplit` by `question_id` to avoid leakage.
- Train: **168** outputs; Test: **72** outputs.
- Robustness set: seed 7, truthful vs lie_roleplay, `n=120` (out-of-seed evaluation).

## 4. Experiment Description
### Methodology
#### High-Level Approach
Generate paired outputs for the same prompts under truthful and deceptive instructions, then test separability with both statistical divergence and classifier performance.

#### Why this method
It directly tests the stated hypothesis in a black-box setting and aligns with gaps in prior work emphasizing internals.

### Implementation Details
#### Tools and Libraries
- Python 3.12.8
- openai 2.28.0
- datasets 4.0.0
- pandas 2.3.2
- scikit-learn 1.8.0
- scipy 1.17.1
- matplotlib 3.10.8
- seaborn 0.13.2

#### Algorithms/Models
- Generation model: `gpt-4.1` (OpenAI API).
- Classifier: TF-IDF (1-2 grams) + Logistic Regression.
- Distribution tests: Jensen-Shannon feature divergence, MMD with permutation test.

#### Hyperparameters
| Parameter | Value | Selection Method |
|---|---:|---|
| model | gpt-4.1 | preferred primary model |
| max_questions | 120 | cost/coverage tradeoff |
| max_tokens | 80 | concise answer control |
| temperature | 0.4 | moderate stochasticity |
| primary_seed | 42 | fixed reproducibility |
| robustness_seed | 7 | seed-shift robustness |
| TF-IDF ngram_range | (1,2) | standard text baseline |
| TF-IDF max_features | 20000 | practical capacity |
| Logistic max_iter | 2000 | convergence safety |

#### Analysis Pipeline
1. Generate and cache raw outputs (`results/raw/*.jsonl`).
2. Build feature table (`results/all_outputs_with_features.csv`).
3. Run feature-level tests (Mann-Whitney U + BH-FDR).
4. Run MMD permutation test.
5. Train/evaluate text classifier on grouped split.
6. Robustness check on second seed.
7. Error analysis (confusion matrix + false positives/negatives).

### Experimental Protocol
#### Reproducibility Information
- Number of runs for averaging: 1 primary run + 1 robustness seed run.
- Random seeds: 42, 7.
- Hardware detected: 2x NVIDIA RTX 3090 (24GB each).
- GPU usage: API-based experiment, so no local GPU training batch size used.
- Mean API latency: 1.13 s/call (median 1.01 s).
- Total token usage: 46,142 tokens.

#### Evaluation Metrics
- AUROC/AUPRC/F1/Accuracy: detectability of lie-conditioned outputs.
- MMD p-value: whether multivariate feature distributions differ.
- JS divergence per feature: interpretable distribution differences.
- Effect size (Cohen's d): practical magnitude.

### Raw Results
#### Tables
| Method / Test | Result |
|---|---:|
| Primary classifier AUROC | **0.8827** |
| Primary classifier AUPRC | 0.9032 |
| Primary classifier F1 | 0.7750 |
| Primary classifier Accuracy | 0.7500 |
| AUROC permutation p-value | **0.000999** |
| AUROC bootstrap 95% CI | **[0.798, 0.952]** |
| MMD statistic | 0.02246 |
| MMD permutation p-value | **0.001996** |
| Robustness AUROC (seed 7) | **0.9103** |
| Robustness F1 (seed 7) | 0.8527 |
| Robustness Accuracy (seed 7) | 0.8417 |

Confusion matrix (primary holdout):
- TN 23, FP 13, FN 5, TP 31.

Feature-level highlight (truthful vs lie_roleplay):
- `punct_ratio`: significant after FDR correction (p=1.97e-13, d=0.83, JS=0.150).

#### Visualizations
- `figures/length_distribution.png`
- `figures/pca_stylometry.png`
- `figures/classification_performance.png`

#### Output Locations
- Metrics JSON: `results/metrics.json`
- Feature stats: `results/feature_stats.csv`
- All outputs/features: `results/all_outputs_with_features.csv`
- Error analysis: `results/confusion_matrix.json`, `results/error_false_positives.csv`, `results/error_false_negatives.csv`
- Raw API generations: `results/raw/outputs_seed_42.jsonl`, `results/raw/outputs_seed_7.jsonl`

## 5. Result Analysis
### Key Findings
1. **Strong separability**: text-only classifier reached AUROC 0.883 on held-out questions.
2. **Statistically significant shift**: MMD permutation test rejects equal-distribution null (p=0.001996).
3. **Robust across seed shift**: AUROC increased to 0.910 on unseen seed-7 outputs.
4. **Style signal is distributed**: one single feature (`punct_ratio`) is strongly significant, but most signal appears multivariate/lexical (captured by TF-IDF model).

### Hypothesis Testing Results
- H0 (no distribution difference): rejected by MMD permutation test.
- H0 (classifier at chance): rejected by AUROC permutation test.
- Practical significance: medium-to-large detectability; meaningful for monitoring applications.

### Comparison to Baselines
- Random baseline AUROC expectation: 0.5.
- Observed AUROC 0.883 implies ~76.5% relative gain over random discrimination margin (`(0.883-0.5)/0.5`).

### Surprises and Insights
- Explicit lie prompts produced highly plausible falsehoods with natural style and no refusal artifacts.
- Distinguishability did not rely on obvious disclaimers (refusal-like count = 0).

### Error Analysis
- False positives (truthful flagged as lie): 13
- False negatives (lie flagged as truthful): 5
- Typical failures occurred when truthful outputs were confident/concise and when deceptive outputs mimicked factual style with low lexical novelty.

### Limitations
- Single model family (`gpt-4.1`) in this run; external validity across providers needs confirmation.
- One primary task domain (TruthfulQA questions).
- Auxiliary jailbreak-style score was in-sample only (not held-out).
- No human annotation layer for deception quality grading.

## 6. Conclusions
### Summary
Yes, in this experiment the model’s output distribution under lie-inducing instructions is noticeably different from its truthful distribution. The difference is statistically significant and practically detectable from text alone.

### Implications
- Practical: enables black-box deception-risk monitoring pipelines.
- Theoretical: supports the view that instruction-conditioned deception induces stable, measurable generation shifts.

### Confidence in Findings
Moderate-high for this model/task setting, strengthened by paired design, significance tests, and seed robustness. Confidence would increase with cross-model replication and larger prompt diversity.

## 7. Next Steps
### Immediate Follow-ups
1. Cross-model replication on GPT-5 / Claude Sonnet 4.5 / Gemini 2.5 Pro.
2. Out-of-domain prompts (non-factual, multilingual, longer-form writing).
3. Strict held-out evaluation for jailbreak-style condition.

### Alternative Approaches
- Embedding-space two-sample tests and domain-adversarial classifiers.
- Character-level stylometry and calibrated uncertainty probes.

### Broader Extensions
- Real-time moderation features for deception-risk alerts.
- Longitudinal monitoring across model/version updates.

### Open Questions
- Which deception styles transfer across model families?
- How much of the shift is semantic vs purely stylistic?
- Can detectors remain robust against adversarial anti-detection prompting?

## References
- Goldowsky-Dill et al. (2025). Detecting Strategic Deception Using Linear Probes.
- Long et al. (2025). When Truthful Representations Flip Under Deceptive Instructions?
- Qiu et al. (2023). Latent Jailbreak.
- Wei et al. (2023). Open Sesame! Universal Black-Box Jailbreaking.
- Geifman et al. (2024). An Assessment of Model-On-Model Deception.
