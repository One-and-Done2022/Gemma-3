from __future__ import annotations

import argparse
import gc
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from student_kit.data import build_prompt, get_role_content, load_jsonl
from student_kit.reward import extract_svg, score_svg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_path: str, adapter_path: str | None, device: torch.device, use_bf16: bool):
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_svg(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, str]:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output[0, inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return raw, extract_svg(raw)


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"count": 0}
    scores = [item["score"]["score"] for item in items]
    components: dict[str, float] = {}
    for key in items[0]["score"]["components"]:
        values = [item["score"]["components"].get(key, 0.0) for item in items]
        components[key] = round(sum(values) / len(values), 6)
    valid_xml = [
        bool(item["score"]["details"].get("validity", {}).get("xml_valid", False))
        for item in items
    ]
    return {
        "count": len(items),
        "mean_reward": round(sum(scores) / len(scores), 6),
        "min_reward": round(min(scores), 6),
        "max_reward": round(max(scores), 6),
        "valid_xml_rate": round(sum(valid_xml) / len(valid_xml), 6),
        "mean_components": components,
    }


def evaluate_run(
    name: str,
    model_path: str,
    adapter_path: str | None,
    data: list[dict[str, Any]],
    device: torch.device,
    use_bf16: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    tokenizer, model = load_model(model_path, adapter_path, device, use_bf16)
    items: list[dict[str, Any]] = []
    for index, record in enumerate(tqdm(data, desc=f"eval {name}")):
        messages = record["messages"]
        prompt = build_prompt(messages)
        user_prompt = get_role_content(messages, "user")
        raw, svg = generate_svg(model, tokenizer, prompt, device, max_new_tokens)
        items.append(
            {
                "index": index,
                "prompt": user_prompt,
                "raw_output": raw,
                "svg": svg,
                "score": score_svg(svg, prompt=user_prompt),
            }
        )
    result = {"name": name, "adapter": adapter_path, "summary": summarize(items), "items": items}
    del model
    del tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="models/gemma-3-270m")
    parser.add_argument("--adapter", default="adapter")
    parser.add_argument("--data", default="data/logo-detailed-prompt/valid.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-base", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    data = load_jsonl(args.data, args.limit)

    runs: dict[str, Any] = {}
    if not args.skip_base:
        runs["base"] = evaluate_run("base", args.base_model, None, data, device, use_bf16, args.max_new_tokens)
    runs["lora"] = evaluate_run("lora", args.base_model, args.adapter, data, device, use_bf16, args.max_new_tokens)

    delta = {}
    if "base" in runs:
        delta["mean_reward"] = round(
            runs["lora"]["summary"]["mean_reward"] - runs["base"]["summary"]["mean_reward"], 6
        )
        delta["valid_xml_rate"] = round(
            runs["lora"]["summary"]["valid_xml_rate"] - runs["base"]["summary"]["valid_xml_rate"], 6
        )

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_model": args.base_model,
        "adapter": args.adapter,
        "data": args.data,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "runs": runs,
        "delta": delta,
    }
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps({"output": args.output, "delta": delta, "lora": runs["lora"]["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
