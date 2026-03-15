# Resources Catalog

## Summary
This document catalogs all resources gathered for the Lying Style project, including papers, datasets, and code repositories for downstream automated experimentation.

### Search Strategy
- Attempted local paper-finder tool first (`.claude/skills/paper-finder/scripts/find_papers.py`), but localhost service was unavailable/hanging.
- Performed manual arXiv search with multiple targeted queries.
- Prioritized papers directly on deception under instruction, jailbreak behavior, and detection methods.
- Pulled datasets from Hugging Face focused on truthfulness and jailbreak prompt pools.
- Cloned code repositories tied to the strongest matched papers.

### Selection Criteria
- Direct relevance to hypothesis.
- Actionable methodology and benchmark setup.
- Availability of code/data.
- Recent work (2023-2025), with practical baseline value.

## Papers
Total papers downloaded: 8

| Title | Authors | Year | File | Key Info |
|------|---------|------|------|---------|
| Latent Jailbreak | Qiu et al. | 2023 | papers/2307.08487_latent_jailbreak_a_benchmark_for_evaluating_text_safety.pdf | Jailbreak benchmark |
| Open Sesame | Wei et al. | 2023 | papers/2309.01446_open_sesame_universal_black_box_jailbreaking_of_large_l.pdf | Universal black-box jailbreak |
| All Languages Matter | Casadei et al. | 2023 | papers/2310.00905_all_languages_matter_on_the_multilingual_safety_of_larg.pdf | Multilingual safety variance |
| Gradient Cuff | Ye et al. | 2024 | papers/2403.00867_gradient_cuff_detecting_jailbreak_attacks_on_large_lang.pdf | Detection baseline |
| Unmasking the Shadows of AI | Dey | 2024 | papers/2403.09676_unmasking_the_shadows_of_ai_investigating_deceptive_cap.pdf | Deception taxonomy |
| Model-On-Model Deception | Geifman et al. | 2024 | papers/2405.12999_an_assessment_of_model_on_model_deception.pdf | Interactive deception |
| Detecting Strategic Deception Using Linear Probes | Goldowsky-Dill et al. | 2025 | papers/2502.03407_detecting_strategic_deception_using_linear_probes.pdf | Internal probe baseline |
| Truthful Representations Flip | Long et al. | 2025 | papers/2507.22149_when_truthful_representations_flip_under_deceptive_inst.pdf | Closest hypothesis match |

See `papers/README.md` for detailed descriptions.

## Datasets
Total datasets downloaded: 4 saved artifacts (3 logical sources)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|------|
| truthful_qa (generation) | HuggingFace | 817 | Truthfulness QA | datasets/truthful_qa_generation | Truth/false candidate answers |
| ChatGPT-Jailbreak-Prompts | HuggingFace | 79 | Prompt attack set | datasets/chatgpt_jailbreak_prompts | Lie/jailbreak instruction bank |
| ITW jailbreak prompts (2023-12-25) | HuggingFace | 1,405 | Jailbreak prompt corpus | datasets/itw_jailbreak_2023_12_25 | Real-world jailbreak prompts |
| ITW regular prompts (2023-12-25) | HuggingFace | 13,735 | Non-jailbreak contrast set | datasets/itw_regular_2023_12_25 | Useful control prompts |

See `datasets/README.md` for download instructions and sample files.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|------|
| deception-detection | https://github.com/ApolloResearch/deception-detection | Probe-based deception detection | code/deception-detection | Strong baseline for deception monitoring |
| truthful-representation-flip | https://github.com/ivyllll/truthful-representation-flip | Truthful vs deceptive representation analysis | code/truthful-representation-flip | Closest to target hypothesis |
| latent-jailbreak | https://github.com/qiuhuachuan/latent-jailbreak | Jailbreak benchmark prompts/data | code/latent-jailbreak | Large benchmark repository |

See `code/README.md` for key files and usage notes.

## Resource Gathering Notes

### Challenges Encountered
- Local paper-finder API endpoint (`localhost:8000`) was unavailable during this run.
- Some HF datasets were inaccessible (gated) or deprecated (`liar` script-style dataset).
- Resolved by switching to accessible alternatives with strong task alignment.

### Gaps and Workarounds
- No single canonical paired truthful/deceptive response dataset was found ready-made for this exact hypothesis.
- Workaround: combine truthfulness QA and jailbreak prompt pools to generate paired outputs under controlled instructions.

## Recommendations for Experiment Design
1. **Primary dataset(s)**: `truthful_qa` + ITW jailbreak/regular prompts, with prompt pairing protocol.
2. **Baseline methods**: n-gram logistic regression, transformer text classifier, optional hidden-state linear probe.
3. **Evaluation metrics**: AUROC/F1 + Jensen-Shannon divergence + MMD between condition distributions.
4. **Code to adapt/reuse**: start with `code/truthful-representation-flip` and `code/deception-detection`; use `code/latent-jailbreak` for prompt templates.

## Research Execution Update (2026-03-15)

This resource set was used for a full automated experiment run.

- Planning document created: `planning.md`
- Main pipeline implemented and executed: `src/run_experiments.py`
- Post-hoc error analysis implemented and executed: `src/post_analysis.py`
- Primary outputs:
  - `results/metrics.json`
  - `results/feature_stats.csv`
  - `results/all_outputs_with_features.csv`
  - `results/confusion_matrix.json`
  - `figures/*.png`
- Final documentation:
  - `REPORT.md`
  - `README.md`

The run used real OpenAI API calls (`gpt-4.1`) with paired truthful vs deceptive instructions and produced statistically significant distribution-shift findings.
