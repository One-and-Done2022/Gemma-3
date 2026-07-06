# LLM Math Part B SVG Logo LoRA

This repository contains the code and report artifacts for the Part B task:
design a programmatic SVG logo reward, fine-tune Gemma 3 270M with LoRA, and
compare the adapter against the base model on the validation split.

## Data and Model

Data is cloned locally, but not committed:

```bash
git clone --depth 1 https://github.com/roboticcam/logo-detailed-prompt data/logo-detailed-prompt
```

The base model is downloaded from ModelScope, but not committed:

```bash
modelscope download --repo-type model google/gemma-3-270m --local-dir models/gemma-3-270m
```

## Commands

```bash
python -m unittest tests/test_reward.py
python -m student_kit.train_peft --config train_config.yaml
python -m student_kit.eval_self --base-model models/gemma-3-270m --adapter adapter --data data/logo-detailed-prompt/valid.jsonl --output results.json
```

The main deliverables are `adapter/`, `reward.py`, `train_config.yaml`,
`results.json`, and the UESTC-format report under `UESTC-Report/`.

