# train_t5_scratch.py
import argparse
import json
import os
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,   # 改这里
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

import sacrebleu
import sentencepiece as spm


def load_jsonl_pairs(path: str) -> Tuple[List[str], List[str]]:
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
    return zh, en


def save_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s.strip() + "\n")


def train_joint_sentencepiece(
    train_paths: List[str],
    model_prefix: str,
    vocab_size: int,
    model_type: str = "unigram",
    character_coverage: float = 1.0,
) -> str:
    """
    Train a joint SentencePiece model on concatenated zh+en lines.
    Returns model_file path.
    """
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    tmp_corpus = model_prefix + ".corpus.txt"
    with open(tmp_corpus, "w", encoding="utf-8") as w:
        for p in train_paths:
            zh, en = load_jsonl_pairs(p)
            for s in zh:
                w.write(s.replace("\n", " ").strip() + "\n")
            for s in en:
                w.write(s.replace("\n", " ").strip() + "\n")

    spm.SentencePieceTrainer.train(
        input=tmp_corpus,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        # IDs: keep a stable scheme
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,  # T5 typically doesn't use BOS
    )

    model_file = model_prefix + ".model"
    return model_file


@torch.no_grad()
def generate_texts(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    src_texts: List[str],
    device: torch.device,
    max_src_len: int,
    max_tgt_len: int,
    num_beams: int,
    length_penalty: float,
    batch_size: int,
) -> List[str]:
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

    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--spm_type", choices=["unigram", "bpe"], default="unigram")

    ap.add_argument("--max_src_len", type=int, default=256)
    ap.add_argument("--max_tgt_len", type=int, default=256)

    ap.add_argument("--per_device_bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)

    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--len_penalty", type=float, default=0.6)

    # T5 scratch model size (adjust if resources limited)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tokenize", default="13a")
    ap.add_argument("--print_samples", type=int, default=3)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Train joint SentencePiece ----
    spm_dir = os.path.join(args.out_dir, "spm")
    spm_prefix = os.path.join(spm_dir, f"spm_joint_v{args.vocab_size}_{args.spm_type}")
    spm_model_file = spm_prefix + ".model"
    if not os.path.exists(spm_model_file):
        print(f"[SPM] training joint sentencepiece vocab_size={args.vocab_size} type={args.spm_type}")
        spm_model_file = train_joint_sentencepiece(
            train_paths=[args.train],
            model_prefix=spm_prefix,
            vocab_size=args.vocab_size,
            model_type=args.spm_type,
            character_coverage=1.0,
        )
    else:
        print(f"[SPM] reuse existing: {spm_model_file}")

    # ---- Build tokenizer from joint SPM ----
    tokenizer = T5Tokenizer(vocab_file=spm_model_file)

    # Set special tokens explicitly to match SPM ids used above
    # pad=0, eos=1, unk=2
    # tokenizer.pad_token = "<pad>"
    # tokenizer.eos_token = "</s>"
    # tokenizer.unk_token = "<unk>"

    # ---- Build scratch T5 model ----
    config = T5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        d_kv=args.d_model // args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
        layer_norm_epsilon=1e-6,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,  # T5 convention
        # relative attention bias is part of T5; keep default True
    )
    model = T5ForConditionalGeneration(config).to(device)

    # ---- Load datasets ----
    train_zh, train_en = load_jsonl_pairs(args.train)
    valid_zh, valid_en = load_jsonl_pairs(args.valid)
    test_zh, test_en = load_jsonl_pairs(args.test)

    train_ds = Dataset.from_dict({"zh": train_zh, "en": train_en})
    valid_ds = Dataset.from_dict({"zh": valid_zh, "en": valid_en})
    test_ds = Dataset.from_dict({"zh": test_zh, "en": test_en})

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

    # ---- Training args (legacy compatible; no generation_* args) ----
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

    # ---- Manual eval with generate ----
    eval_dir = os.path.join(args.out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    def eval_split(name: str, raw_ds: Dataset) -> float:
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
                    "vocab_size": args.vocab_size,
                    "spm_model": spm_model_file,
                    "t5_config": {
                        "d_model": args.d_model,
                        "num_layers": args.num_layers,
                        "num_heads": args.num_heads,
                        "d_ff": args.d_ff,
                        "dropout": args.dropout,
                    },
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
        return float(bleu)

    valid_bleu = eval_split("valid", valid_ds)
    test_bleu = eval_split("test", test_ds)

    print("\n[SUMMARY]")
    print("spm_model:", spm_model_file)
    print("last_model:", last_dir)
    print("eval_dir:", eval_dir)
    print("valid_bleu:", valid_bleu)
    print("test_bleu:", test_bleu)


if __name__ == "__main__":
    main()
