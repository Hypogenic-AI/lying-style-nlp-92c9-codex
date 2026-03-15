# Cloned Repositories

## Repo 1: deception-detection
- URL: https://github.com/ApolloResearch/deception-detection
- Purpose: Linear-probe deception detection on internal activations
- Location: `code/deception-detection/`
- Key files:
- `deception_detection/experiment.py`
- `deception_detection/scripts/experiment.py`
- `deception_detection/scripts/roleplaying_data_generation/generate.py`
- `data/`
- Notes:
- Includes paper-aligned datasets (roleplaying, insider trading, sandbagging).
- Requires API keys for some data generation/evals.

## Repo 2: truthful-representation-flip
- URL: https://github.com/ivyllll/truthful-representation-flip
- Purpose: Analyze truthful vs deceptive representational shifts (probing + SAE)
- Location: `code/truthful-representation-flip/`
- Key files:
- `scripts/extract_activations.py`
- `scripts/run_probing_pipeline.py`
- `scripts/analyze_feature_shift_sae.py`
- `requirements.txt`
- Notes:
- Closely aligned to current hypothesis.
- Supports truthful/deceptive/neutral prompt conditions and layer-wise probing.

## Repo 3: latent-jailbreak
- URL: https://github.com/qiuhuachuan/latent-jailbreak
- Purpose: Latent jailbreak benchmark prompts/data
- Location: `code/latent-jailbreak/`
- Key files:
- `data/`
- `src/`
- Notes:
- Provides jailbreak templates and baseline generation scripts.
- Repo is large and includes substantial benchmark assets.
