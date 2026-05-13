from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


def _messages_to_text(obj: dict[str, Any], tokenizer: PreTrainedTokenizer) -> str | None:
    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            return None
    return None


def _load_texts(jsonl_path: str, tokenizer: PreTrainedTokenizer) -> list[str]:
    texts: list[str] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = _messages_to_text(obj, tokenizer)
            if not t and obj.get("text"):
                t = str(obj["text"]).strip()
            if not t and "instruction" in obj:
                t = (
                    "### Instruction\n"
                    f"{obj.get('instruction', '')}\n### Input\n"
                    f"{obj.get('input', '')}\n### Output\n"
                    f"{obj.get('output', '')}"
                )
            if isinstance(t, str) and len(t.strip()) > 40:
                texts.append(t.strip()[:12000])
    return texts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="JSONL: messages (preferred) или text / instruction.")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--model_name",
        default=os.environ.get("SFT_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
    )
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=1)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    texts = _load_texts(args.jsonl, tokenizer)
    if not texts:
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        use_bf16, use_fp16 = True, False
    elif use_cuda:
        dtype = torch.float16
        use_bf16, use_fp16 = False, True
    else:
        dtype = torch.float32
        use_bf16, use_fp16 = False, False

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            trust_remote_code=False,
            attn_implementation="sdpa",
        )
    except (TypeError, ValueError, RuntimeError):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            trust_remote_code=False,
        )

    model.to(device)

    ds = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tok_ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    t0 = time.perf_counter()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_steps=max(5, args.max_steps),
        learning_rate=args.learning_rate,
        warmup_steps=max(1, args.max_steps // 20),
        logging_steps=max(1, args.max_steps // 10),
        save_steps=args.max_steps + 1,
        save_total_limit=1,
        prediction_loss_only=True,
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=use_cuda,
        dataloader_pin_memory=use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds,
        data_collator=collator,
    )
    train_out = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    elapsed = time.perf_counter() - t0
    loss = getattr(train_out, "training_loss", None)
    metrics_dict = getattr(train_out, "metrics", None) or {}
    if loss is None:
        loss = metrics_dict.get("train_loss")
    if loss is None:
        for log in reversed(getattr(trainer.state, "log_history", []) or []):
            if isinstance(log, dict) and "loss" in log:
                loss = log["loss"]
                break

    metrics = {
        "train_loss": float(loss) if loss is not None else None,
        "max_steps": training_args.max_steps,
        "n_samples": len(texts),
        "model_name": args.model_name,
        "device": str(device),
        "dtype": str(dtype),
        "runtime_seconds": round(elapsed, 2),
        "jsonl": os.path.abspath(args.jsonl),
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    manifest = {
        "status": "finished_micro_sft",
        "mock": False,
        "checkpoint_dir": os.path.abspath(args.output_dir),
        "metrics_path": os.path.abspath(metrics_path),
        "base_model": args.model_name,
        "train_samples": len(texts),
        "device": str(device),
    }
    manifest_path = os.path.join(args.output_dir, "train_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
