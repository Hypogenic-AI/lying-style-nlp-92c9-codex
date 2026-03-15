# Literature Review: Lying Style in Language Models

## Review Scope

### Research Question
Do language models produce measurably different text distributions when explicitly instructed to lie (e.g., jailbreak/roleplay/deceptive prompts) compared with truthful/normal instructions?

### Inclusion Criteria
- Studies on LLM deception, lying, strategic misrepresentation, or jailbreak behavior.
- Papers with concrete methodology and datasets/benchmarks.
- Papers with direct relevance to text-output or representation-level distribution shifts.
- Priority window: 2023-2025, with foundational references where needed.

### Exclusion Criteria
- Non-LLM deception work without actionable methods for this hypothesis.
- Papers without usable datasets/code or with minimal methodological detail.

### Time Frame
- Primary: 2023-2025.

### Sources
- arXiv API/manual search.
- Local `paper-finder` script attempted first; service unavailable on localhost.

## Search Log

| Date | Query | Source | Results | Notes |
|------|-------|--------|---------|------|
| 2026-03-15 | "llm deception truthful deceptive text" | arXiv API | 8 | Strongly relevant deception hits |
| 2026-03-15 | "language model hallucination truthfulness generation" | arXiv API | 8 | Some relevance, mixed domain |
| 2026-03-15 | "jailbreak roleplay large language models safety" | arXiv API | 7 | Useful jailbreak baseline papers |

## Screening Results

| Paper | Title Screen | Abstract Screen | Full-Text | Notes |
|------|--------------|-----------------|----------|------|
| 2507.22149 | Include | Include | Include (deep) | Closest match to hypothesis |
| 2502.03407 | Include | Include | Include (deep) | Strong detection baseline |
| 2403.09676 | Include | Include | Include (deep) | Deception taxonomy and risk framing |
| 2405.12999 | Include | Include | Abstract+methods skim | Interactive model deception |
| 2307.08487 | Include | Include | Abstract+intro skim | Benchmark for jailbreak conditions |
| 2309.01446 | Include | Include | Abstract+intro skim | Universal black-box jailbreaks |
| 2403.00867 | Include | Include | Abstract+methods skim | Jailbreak detection baseline |
| 2310.00905 | Include | Include | Abstract skim | Condition shifts across languages |

## Key Papers

### Paper 1: Detecting Strategic Deception Using Linear Probes
- **Authors**: Goldowsky-Dill et al.
- **Year**: 2025
- **Source**: arXiv (2502.03407)
- **Key Contribution**: Demonstrates high-AUROC detection of deceptive behavior from internal activations.
- **Methodology**: Linear probes over hidden activations (logistic regression/LDA variants), multi-setting evaluation.
- **Datasets Used**: Roleplaying, Insider Trading Report/Confirmation, Sandbagging, control chat data.
- **Baselines**: Black-box GPT-4o classifier and probe variants.
- **Results**: AUROC roughly 0.96-0.999 in key settings; high recall at 1% FPR in evaluated scenarios.
- **Code Available**: Yes (ApolloResearch/deception-detection).
- **Relevance to Our Research**: Strong baseline for deception detection; suggests distribution shifts may also be detectable from outputs.

### Paper 2: When Truthful Representations Flip Under Deceptive Instructions?
- **Authors**: Long et al.
- **Year**: 2025
- **Source**: arXiv (2507.22149)
- **Key Contribution**: Directly studies representational "flip" under deceptive instructions vs truthful/neutral prompts.
- **Methodology**: Layerwise probing + Sparse Autoencoder (SAE) feature-shift analysis.
- **Datasets Used**: Curated factual datasets and more complex claim sets (e.g., CommonClaim/CounterFact-style sets).
- **Baselines**: Truthful vs neutral vs deceptive prompt conditions; LR and TTPD probes.
- **Results**: Deceptive instructions induce stronger feature-space shifts; truthful and neutral states are closer.
- **Code Available**: Yes (ivyllll/truthful-representation-flip).
- **Relevance to Our Research**: Closest conceptual and methodological precedent for testing distributional differences under lie instructions.

### Paper 3: Latent Jailbreak
- **Authors**: Qiu et al.
- **Year**: 2023
- **Source**: arXiv (2307.08487)
- **Key Contribution**: Benchmark framing for latent jailbreak prompts and robustness evaluation.
- **Methodology**: Prompt-template-based jailbreak evaluation across models.
- **Datasets Used**: Latent-jailbreak prompt/data assets.
- **Baselines**: Safety robustness under jailbreak prompts.
- **Results**: Demonstrates substantial prompt-induced safety behavior changes.
- **Code Available**: Yes (latent-jailbreak repo).
- **Relevance to Our Research**: Supplies practical lie/jailbreak prompt conditions to generate deceptive outputs.

### Paper 4: Open Sesame! Universal Black Box Jailbreaking of LLMs
- **Authors**: Wei et al.
- **Year**: 2023
- **Source**: arXiv (2309.01446)
- **Key Contribution**: Universal transfer jailbreak prompts in black-box settings.
- **Methodology**: Automated adversarial prompt search/transfer.
- **Datasets Used**: Safety prompt suites.
- **Results**: High transferability of jailbreak prompts across target models.
- **Code Available**: Not confirmed during this run.
- **Relevance to Our Research**: Supports use of jailbreak prompts as intervention for inducing deceptive style shifts.

### Paper 5: Gradient Cuff
- **Authors**: Ye et al.
- **Year**: 2024
- **Source**: arXiv (2403.00867)
- **Key Contribution**: Refusal-loss-landscape-based jailbreak detection.
- **Methodology**: Gradient/refusal behavior analysis.
- **Datasets Used**: Jailbreak attack corpora.
- **Results**: Competitive detection of attack/jailbreak conditions.
- **Code Available**: Not confirmed during this run.
- **Relevance to Our Research**: Useful detector baseline to compare against output-distribution classifiers.

### Paper 6: An Assessment of Model-On-Model Deception
- **Authors**: Geifman et al.
- **Year**: 2024
- **Source**: arXiv (2405.12999)
- **Key Contribution**: Studies deceptive behavior in model-model interaction settings.
- **Methodology**: Structured interactive tasks measuring deceptive actions.
- **Datasets Used**: Task-specific interaction scenarios.
- **Results**: Evidence that deception emerges under strategic settings.
- **Code Available**: Not confirmed during this run.
- **Relevance to Our Research**: Broadens experimental settings beyond static QA.

### Paper 7: All Languages Matter
- **Authors**: Casadei et al.
- **Year**: 2023
- **Source**: arXiv (2310.00905)
- **Key Contribution**: Safety robustness varies across languages.
- **Methodology**: Multilingual prompt evaluation.
- **Datasets Used**: Multilingual safety/jailbreak prompt sets.
- **Results**: Significant cross-lingual variation in unsafe responses.
- **Code Available**: Not confirmed during this run.
- **Relevance to Our Research**: Suggests language condition should be controlled in distribution-shift studies.

### Paper 8: Unmasking the Shadows of AI
- **Authors**: Dey
- **Year**: 2024
- **Source**: arXiv (2403.09676)
- **Key Contribution**: Deception taxonomy and governance framing.
- **Methodology**: Survey/discussion paper.
- **Datasets Used**: N/A (conceptual).
- **Results**: Consolidates deception categories and risks.
- **Code Available**: No.
- **Relevance to Our Research**: Useful conceptual framework for labeling lie/deception output types.

## Common Methodologies
- Prompt-conditioned behavior testing: truthful vs deceptive/jailbreak instruction contrasts.
- Probe-based detection: linear probing of hidden states (strongest in 2502.03407, 2507.22149).
- Robustness benchmarking: jailbreak prompts and refusal/safety metrics.

## Standard Baselines
- Black-box classifier over outputs (including strong LLM judge baselines).
- Linear probe over final/mid-layer activations.
- Refusal-oriented jailbreak detectors (e.g., loss-landscape or refusal-pattern methods).

## Evaluation Metrics
- AUROC, AUPRC, recall at low FPR for deception detection.
- Attack success rate / jailbreak success rate.
- Accuracy/F1 when framing deception detection as classification.
- Distribution divergence metrics for text: Jensen-Shannon, Wasserstein, MMD (recommended for this project).

## Datasets in the Literature
- Roleplaying deception scenarios.
- Insider trading concealment scenarios.
- Sandbagging datasets.
- Jailbreak prompt corpora.
- Truthfulness factual QA benchmarks.

## Gaps and Opportunities
- Most work emphasizes internal activations; fewer studies test pure text-distribution separability under controlled lying instructions.
- Cross-model generalization of deception-style detectors is underexplored.
- Limited standardized datasets explicitly pairing truthful and deceptive responses for the same prompts.

## Recommendations for Our Experiment
- **Recommended datasets**:
- `truthful_qa` for factual truth-conditioned prompting.
- `TrustAIRLab/in-the-wild-jailbreak-prompts` + `ChatGPT-Jailbreak-Prompts` for deception-inducing instruction pools.
- **Recommended baselines**:
- Bag-of-ngrams/logistic regression on outputs.
- Transformer-based text classifier (frozen encoder + linear head).
- Optional hidden-state probe baseline if model internals are available.
- **Recommended metrics**:
- AUROC/F1 for condition classification.
- Jensen-Shannon divergence and MMD between truthful vs lie-conditioned output distributions.
- **Methodological considerations**:
- Pair prompts across truthful and deceptive instructions to isolate instruction effect.
- Control for topic, length, temperature, and language.
- Measure both in-domain and out-of-domain generalization.
