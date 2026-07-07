---
base_model: models/gemma-3-270m
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- svg
- reward-modeling
---

# Gemma 3 270M SVG Logo LoRA Adapter

This adapter was trained for the LLM Math Part B SVG logo generation task.
It fine-tunes `models/gemma-3-270m` with LoRA on the
`roboticcam/logo-detailed-prompt` training split.

Evaluation on the 17-sample validation split is recorded in `results.json`:

- Base model mean reward: `0.000000`
- LoRA mean reward: `0.358181`
- Base valid XML/SVG rate: `0.000000`
- LoRA valid XML/SVG rate: `0.411765`

Load it with `PeftModel.from_pretrained(base_model, "adapter")`.
