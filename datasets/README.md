# Downloaded Datasets

This directory contains datasets for studying distributional differences between truthful and lie-inducing prompt conditions. Data files are stored locally and excluded from git via `.gitignore`.

## Dataset 1: truthful_qa (generation)

### Overview
- Source: HuggingFace `truthful_qa` (config: `generation`)
- Size: 817 examples
- Format: HuggingFace DatasetDict
- Task: Truthfulness-focused QA generation/evaluation
- Splits: validation (817)
- License: See dataset card on HuggingFace

### Download Instructions
Using HuggingFace (recommended):
```python
from datasets import load_dataset

dataset = load_dataset("truthful_qa", "generation")
dataset.save_to_disk("datasets/truthful_qa_generation")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/truthful_qa_generation")
```

### Sample Data
- Saved at: `datasets/truthful_qa_generation/samples/samples.json`

### Notes
- Contains true and false answer candidates; useful for truthful-vs-false output tendency analysis.

## Dataset 2: rubend18/ChatGPT-Jailbreak-Prompts

### Overview
- Source: HuggingFace `rubend18/ChatGPT-Jailbreak-Prompts`
- Size: 79 prompts
- Format: HuggingFace DatasetDict
- Task: Jailbreak prompt bank
- Splits: train (79)
- License: See dataset card on HuggingFace

### Download Instructions
```python
from datasets import load_dataset

dataset = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts")
dataset.save_to_disk("datasets/chatgpt_jailbreak_prompts")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/chatgpt_jailbreak_prompts")
```

### Sample Data
- Saved at: `datasets/chatgpt_jailbreak_prompts/samples/samples.json`

### Notes
- Useful for constructing lie/deception-inducing prompt conditions.

## Dataset 3: TrustAIRLab/in-the-wild-jailbreak-prompts (jailbreak + regular)

### Overview
- Source: HuggingFace `TrustAIRLab/in-the-wild-jailbreak-prompts`
- Configs used:
- `jailbreak_2023_12_25` (1,405 prompts)
- `regular_2023_12_25` (13,735 prompts)
- Format: HuggingFace DatasetDict
- Task: In-the-wild jailbreak vs regular prompt distribution
- Splits: train
- License: See dataset card on HuggingFace

### Download Instructions
```python
from datasets import load_dataset

jb = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25")
rg = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "regular_2023_12_25")

jb.save_to_disk("datasets/itw_jailbreak_2023_12_25")
rg.save_to_disk("datasets/itw_regular_2023_12_25")
```

### Loading the Dataset
```python
from datasets import load_from_disk

jb = load_from_disk("datasets/itw_jailbreak_2023_12_25")
rg = load_from_disk("datasets/itw_regular_2023_12_25")
```

### Sample Data
- Saved at: `datasets/itw_jailbreak_2023_12_25/samples/samples.json`
- Saved at: `datasets/itw_regular_2023_12_25/samples/samples.json`

### Notes
- Directly supports creating contrasted prompt sets (jailbreak-like vs regular) for conditional generation experiments.

## Validation Summary
- Dataset inventory: `datasets/dataset_inventory.json`
- Local size check:
- truthful_qa_generation: ~276 KB
- chatgpt_jailbreak_prompts: ~124 KB
- itw_jailbreak_2023_12_25: ~2.0 MB
- itw_regular_2023_12_25: ~1.7 MB
