# finetune_t5.py (legacy-transformers compatible)
import argparse
import json
import os
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

import sacrebleu


def load_jsonl(path: str) -> Dataset:
    zh, en = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            z = str(obj.get("zh", "")).strip()
            e = str(obj.get("en", "")).strip()
            if not z or not e:
                continue
            zh.append(z)
            en.append(e)
    return Dataset.from_dict({"zh": zh, "en": en})


def save_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s.strip() + "\n")


@torch.no_grad()
def generate_texts(
    model,
    tokenizer,
    src_texts: List[str],
    device: torch.device,
    max_src_len: int,
    max_tgt_len: int,
    num_beams: int,
    length_penalty: float,
    batch_size: int,
) -> List[str]:
    """Manual generation for maximum compatibility across transformers versions."""
    model.eval()
    outs: List[str] = []

    for i in range(0, len(src_texts), batch_size):
        batch = src_texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_src_len,
        ).to(device)

        gen_ids = model.generate(
            **enc,
            max_length=max_tgt_len,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True,
        )
        txt = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        outs.extend(txt)

    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--model_name", default="/afs/250010026/nlp/t5-base")
    ap.add_argument("--max_src_len", type=int, default=256)
    ap.add_argument("--max_tgt_len", type=int, default=256)

    ap.add_argument("--per_device_bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)

    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--len_penalty", type=float, default=0.6)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print_samples", type=int, default=3)
    ap.add_argument("--tokenize", default="13a")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    train_ds = load_jsonl(args.train)
    valid_ds = load_jsonl(args.valid)
    test_ds = load_jsonl(args.test)

    prefix = "translate Chinese to English: "

    def preprocess(examples: Dict[str, List[str]]) -> Dict[str, Any]:
        inputs = [prefix + x for x in examples["zh"]]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_src_len,
            truncation=True,
        )
        # labels
        try:
            labels = tokenizer(
                text_target=examples["en"],
                max_length=args.max_tgt_len,
                truncation=True,
            )
        except TypeError:
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["en"],
                    max_length=args.max_tgt_len,
                    truncation=True,
                )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    valid_tok = valid_ds.map(preprocess, batched=True, remove_columns=valid_ds.column_names)
    test_tok = test_ds.map(preprocess, batched=True, remove_columns=test_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training args: NO generation_* here for old versions
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        fp16=False,
        logging_steps=100,
        seed=args.seed,
        save_total_limit=2,
    )

    # We will compute BLEU manually after training, so trainer doesn't need compute_metrics
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save last
    last_dir = os.path.join(args.out_dir, "last")
    trainer.save_model(last_dir)
    tokenizer.save_pretrained(last_dir)

    # ---------- Manual eval & dump ----------
    eval_dir = os.path.join(args.out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    def eval_split(name: str, raw_ds: Dataset) -> Tuple[float, List[str]]:
        src = [prefix + s for s in raw_ds["zh"]]
        ref = list(raw_ds["en"])

        hyp = generate_texts(
            model=model,
            tokenizer=tokenizer,
            src_texts=src,
            device=device,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len,
            num_beams=args.beam,
            length_penalty=args.len_penalty,
            batch_size=args.per_device_bs,
        )

        bleu = sacrebleu.corpus_bleu(hyp, [ref], tokenize=args.tokenize).score

        save_lines(os.path.join(eval_dir, f"{name}.src.txt"), list(raw_ds["zh"]))
        save_lines(os.path.join(eval_dir, f"{name}.ref.txt"), ref)
        save_lines(os.path.join(eval_dir, f"{name}.hyp.beam{args.beam}.txt"), hyp)

        with open(os.path.join(eval_dir, f"{name}.bleu.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "bleu": float(bleu),
                    "tokenize": args.tokenize,
                    "beam": args.beam,
                    "len_penalty": args.len_penalty,
                    "max_src_len": args.max_src_len,
                    "max_tgt_len": args.max_tgt_len,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        n = min(args.print_samples, len(hyp))
        if n > 0:
            print(f"\n[{name} SANITY CHECK]")
            for i in range(n):
                print(f"\n[{name} #{i}]")
                print("ZH:", raw_ds["zh"][i])
                print("REF:", ref[i])
                print("HYP:", hyp[i])

        print(f"\n{name} BLEU = {bleu:.2f}")
        return float(bleu), hyp

    valid_bleu, _ = eval_split("valid", valid_ds)
    test_bleu, _ = eval_split("test", test_ds)

    print("\n[SUMMARY]")
    print("last_model:", last_dir)
    print("eval_dir:", eval_dir)
    print("valid_bleu:", valid_bleu)
    print("test_bleu:", test_bleu)


if __name__ == "__main__":
    main()
