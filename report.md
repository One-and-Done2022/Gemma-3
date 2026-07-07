# Gemma 3 270M SVG 徽标 LoRA 微调报告

## 结论

本次实验在 `valid.jsonl` 的 17 条验证样本上，LoRA 相比 Gemma 3 270M 基座模型有明确提升：基座模型在自定义 reward 下均值为 `0.000000`，LoRA 均值为 `0.358181`，有效 XML/SVG 率从 `0/17` 提升到 `7/17`。提升主要来自模型学会了输出完整 SVG 结构，而不是语义保真度的显著提升。

## Reward 设计

`student_kit/reward.py` 将总分归一到 `[0, 1]`，由以下分项加权组成：

| 分项 | 权重 | 目的 |
| --- | ---: | --- |
| validity | 0.25 | XML 可解析、根节点为 `<svg>`、`xmlns` 和 `viewBox="0 0 256 256"` 正确、无额外文本 |
| safety | 0.15 | 禁止 `script/image/foreignObject`、事件属性、外链引用等不安全内容 |
| geometry | 0.15 | 坐标大多落在 `-32..288` 的软范围内，避免极端越界和坍缩 |
| structure | 0.15 | SVG primitive 数量和类型多样性合理，避免空壳或无限重复 |
| colors | 0.10 | 颜色属性数量、十六进制调色板规模和颜色合法性合理 |
| grounding | 0.10 | 检查提示词中的形状词、颜色词和显式 hex 色值是否在输出中被覆盖 |
| degeneracy | 0.10 | 惩罚过短/过长、重复 path、markdown fence 等退化输出 |

这个 reward 有意把“有效 SVG”和“安全、可渲染”放在较高权重，因为 Gemma 3 270M 的主要失败模式不是美观度，而是无法稳定输出闭合 SVG。`grounding` 权重较低，是因为简单关键词覆盖无法可靠衡量视觉语义，只作为弱信号使用。

## 环境与数据

- Conda 环境：`llm-math`
- Python：`3.11.15`
- GPU：`NVIDIA GeForce RTX 4090`
- PyTorch：`2.11.0+cu128`
- Transformers：`5.13.0`
- PEFT：`0.19.1`
- 基座模型：ModelScope `google/gemma-3-270m`
- 数据集：`roboticcam/logo-detailed-prompt`，commit `8468c4b`

## 训练配置

使用 `student_kit/train_peft.py` 和 `train_config.yaml` 训练 LoRA：

| 参数 | 值 |
| --- | --- |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| epochs | 8 |
| batch size | 1 |
| gradient accumulation | 8 |
| learning rate | `2e-4` |
| max sequence length | 4096 |
| dtype | bf16 |

训练时只对 assistant 的 SVG 部分计算 loss，prompt token 使用 `-100` mask。

## 训练过程

| Epoch | Train loss | Valid loss |
| ---: | ---: | ---: |
| 1 | 0.946210 | 0.809480 |
| 2 | 0.745076 | 0.735340 |
| 3 | 0.687376 | 0.706011 |
| 4 | 0.652134 | 0.693109 |
| 5 | 0.624803 | 0.679605 |
| 6 | 0.603479 | 0.673426 |
| 7 | 0.585965 | 0.672937 |
| 8 | 0.573578 | 0.673474 |

最佳验证 loss 为 `0.6729366008`，出现在第 7 个 epoch。最终 `adapter/` 保存的是验证 loss 最低时的 LoRA 权重。

## 自评结果

评测命令：

```bash
python -m student_kit.eval_self \
  --base-model models/gemma-3-270m \
  --adapter adapter \
  --data data/logo-detailed-prompt/valid.jsonl \
  --output results.json \
  --max-new-tokens 1536 \
  --seed 42
```

| 模型 | Mean reward | Valid XML rate | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Gemma 3 270M base | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| LoRA | 0.358181 | 0.411765 | 0.000000 | 0.884057 |
| Delta | +0.358181 | +0.411765 | - | - |

LoRA 的分项均值：

| 分项 | 均值 |
| --- | ---: |
| validity | 0.411765 |
| safety | 0.411765 |
| geometry | 0.264562 |
| structure | 0.411765 |
| colors | 0.382353 |
| grounding | 0.237582 |
| degeneracy | 0.300326 |

## 样例观察

基座模型输出经常是占位符式文本，不能被 XML 解析：

```xml
<svg ...>...</svg> <path ...>...</path> <circle ...>...</circle>
```

LoRA 的有效样本能生成完整闭合的 SVG，例如验证集第 8 条得分 `0.884057`：

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <rect x="-9999" y="-9999" width="19998" height="19998" fill="#FBF7E3"/>
  <defs><linearGradient id="ring" ...></linearGradient></defs>
  <circle cx="128" cy="128" r="104" fill="none" .../>
  ...
</svg>
```

但 LoRA 仍有 10 条样本得分为 0。失败样本通常不是完全不会写 SVG，而是在生成后半段进入重复模式，未能在 `max_new_tokens=1536` 内闭合，例如：

```xml
<line x1="128" y1="128" x2="128" y2="128"/>
<line x1="128" y1="128" x2="128" y2="128"/>
...
```

## 分析

本次提升主要来自监督微调让模型学到 SVG 外壳、常见 primitive、颜色属性和部分组合模式。基座模型虽然知道 SVG 相关词汇，但输出多为占位符或重复标签，不能形成可解析文档，因此 reward 为 0。

LoRA 的弱点也很明显：`grounding` 均值只有 `0.237582`，说明提示词语义覆盖仍差；无效样本出现明显重复，说明模型学到局部 SVG 模板后缺少稳定的结束控制。这也是 Goodhart 风险：程序化 reward 上升主要来自“闭合、合法、配色合理”，并不保证视觉上符合提示词，更不保证能通过私有 Sonnet 视觉评审。

如果继续改进，优先方向是缩短和清洗训练目标、加入重复惩罚或停止序列调参、扩大有效 SVG 结构样本，并单独分析 prompt grounding，而不是只追求 validation loss。

## 复现

```bash
python -m unittest tests/test_reward.py
python -m student_kit.train_peft --config train_config.yaml
python -m student_kit.eval_self --base-model models/gemma-3-270m --adapter adapter --data data/logo-detailed-prompt/valid.jsonl --output results.json
```

主要提交物：

- `adapter/adapter_config.json`
- `adapter/adapter_model.safetensors`
- `reward.py`
- `student_kit/reward.py`
- `train_config.yaml`
- `results.json`
- `report.md`
- `UESTC-Report/main.tex`

