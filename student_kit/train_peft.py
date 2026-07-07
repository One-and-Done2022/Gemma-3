from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from student_kit.data import build_prompt, get_target_svg, load_jsonl


class SvgDataset(Dataset):
    def __init__(self, items: list[dict[str, Any]], tokenizer: Any, max_seq_length: int):
        self.items = items
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        messages = self.items[index]["messages"]
        prompt = build_prompt(messages)
        target = get_target_svg(messages) + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + target_ids)[: self.max_seq_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_seq_length]
        if all(label == -100 for label in labels):
            labels[-1] = input_ids[-1]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


class DataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_loss(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, use_bf16: bool) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
                output = model(**batch)
            losses.append(float(output.loss.detach().cpu()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(cfg.get("bf16", True)) and device.type == "cuda" and torch.cuda.is_bf16_supported()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_items = load_jsonl(cfg["train_file"], cfg.get("train_limit"))
    valid_items = load_jsonl(cfg["valid_file"], cfg.get("valid_limit"))
    train_dataset = SvgDataset(train_items, tokenizer, int(cfg["max_seq_length"]))
    valid_dataset = SvgDataset(valid_items, tokenizer, int(cfg["max_seq_length"]))
    collator = DataCollator(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["per_device_train_batch_size"]),
        shuffle=True,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        shuffle=False,
        collate_fn=collator,
    )

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], torch_dtype=dtype)
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    model.to(device)

    lora_cfg = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg["target_modules"]),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    grad_accum = int(cfg["gradient_accumulation_steps"])
    epochs = int(cfg["epochs"])
    total_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    warmup_steps = int(total_steps * float(cfg.get("warmup_ratio", 0.0)))
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)

    output_dir = Path(cfg["output_dir"])
    run_dir = Path(cfg.get("run_dir", "runs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, epochs + 1):
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        running_loss = 0.0
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
                output = model(**batch)
                loss = output.loss / grad_accum
            loss.backward()
            running_loss += float(output.loss.detach().cpu())

            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.set_postfix(loss=running_loss / step, lr=scheduler.get_last_lr()[0])

        train_loss = running_loss / max(1, len(train_loader))
        val_loss = evaluate_loss(model, valid_loader, device, use_bf16)
        row = {"epoch": epoch, "global_step": global_step, "train_loss": train_loss, "valid_loss": val_loss}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(output_dir)

    metadata = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "config": cfg,
        "best_valid_loss": best_val,
        "history": history,
        "torch": torch.__version__,
    }
    with (run_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved best adapter to {output_dir}")


if __name__ == "__main__":
    main()
