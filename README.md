# Lying Style (NLP)

This project tests whether an LLM writes differently when instructed to lie versus answer truthfully. We ran real OpenAI API experiments (`gpt-4.1`) with paired prompts from TruthfulQA and analyzed output distribution shift using both statistics and text-only classifiers.

## Key Findings
- Truthful vs lie-roleplay outputs are clearly separable (held-out AUROC: **0.883**, permutation p=**0.001**).
- Multivariate feature distributions differ significantly (MMD p=**0.001996**).
- Results remain strong on a second decoding seed (AUROC: **0.910**).
- Deceptive outputs were generated without refusal/disclaimer artifacts (0 refusal-like outputs).

## Reproduce
```bash
# 1) Activate workspace env
source .venv/bin/activate

# 2) Run main pipeline (uses cache if raw outputs already exist)
python src/run_experiments.py

# 3) Run post-hoc error analysis
python src/post_analysis.py
```

## File Structure
- `planning.md`: phase 0/1 research plan and motivation/novelty.
- `src/run_experiments.py`: full generation + analysis pipeline.
- `src/post_analysis.py`: confusion matrix and failure-case export.
- `results/`: metrics, tables, raw outputs, diagnostics.
- `figures/`: generated plots.
- `REPORT.md`: full research report with interpretation and limitations.

See `REPORT.md` for full methodology and results.
