# tokenize_and_build.py
import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Any, Optional, Tuple, Iterable

import sentencepiece as spm

def iter_jsonl(path: str) -> Iterable[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                yield line_no, None, "empty_line"
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield line_no, None, "bad_json"
                continue
            yield line_no, obj, None

def dump_corpus(clean_path: str, field: str, out_txt: str) -> None:
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as fout:
        for _, obj, err in iter_jsonl(clean_path):
            if err or obj is None:
                continue
            if field not in obj or not isinstance(obj[field], str):
                continue
            fout.write(obj[field].strip() + "\n")

def train_sentencepiece(
    input_txt: str,
    model_prefix: str,
    vocab_size: int,
    model_type: str,
) -> None:
    spm.SentencePieceTrainer.train(
        input=input_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,          # unigram 或 bpe
        character_coverage=1.0,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )

def encode_split(
    clean_path: str,
    out_ids_path: str,
    sp_src: spm.SentencePieceProcessor,
    sp_tgt: spm.SentencePieceProcessor,
    max_tokens: int,
    len_mode: str,   # filter 或 truncate
    ratio_min: float,
    ratio_max: float,
) -> Dict[str, Any]:
    stats = Counter()
    len_bins_src = Counter()
    len_bins_tgt = Counter()

    def bin_len(x: int) -> str:
        if x <= 32: return "0-32"
        if x <= 64: return "33-64"
        if x <= 128: return "65-128"
        if x <= 256: return "129-256"
        if x <= 512: return "257-512"
        return "513+"

    os.makedirs(os.path.dirname(out_ids_path), exist_ok=True)
    with open(out_ids_path, "w", encoding="utf-8") as fout:
        for _, obj, err in iter_jsonl(clean_path):
            if err or obj is None:
                continue
            if "zh" not in obj or "en" not in obj:
                stats["skip_missing_fields"] += 1
                continue

            zh = obj["zh"]
            en = obj["en"]
            if not isinstance(zh, str) or not isinstance(en, str):
                stats["skip_non_str"] += 1
                continue

            src_ids = [sp_src.bos_id()] + sp_src.encode(zh, out_type=int) + [sp_src.eos_id()]
            tgt_ids = [sp_tgt.bos_id()] + sp_tgt.encode(en, out_type=int) + [sp_tgt.eos_id()]

            if len(tgt_ids) == 0:
                stats["drop_empty_tgt"] += 1
                continue

            ratio = len(src_ids) / max(1, len(tgt_ids))
            if ratio < ratio_min or ratio > ratio_max:
                stats["drop_len_ratio"] += 1
                continue

            if len(src_ids) > max_tokens or len(tgt_ids) > max_tokens:
                if len_mode == "filter":
                    stats["drop_too_long_tok"] += 1
                    continue
                if len_mode == "truncate":
                    if len(src_ids) > max_tokens:
                        src_ids = src_ids[:max_tokens]
                    if len(tgt_ids) > max_tokens:
                        tgt_ids = tgt_ids[:max_tokens]
                    stats["truncate_too_long_tok"] += 1

            out = {"src_ids": src_ids, "tgt_ids": tgt_ids}
            if "index" in obj:
                out["index"] = obj["index"]

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["keep"] += 1
            stats["src_len_sum"] += len(src_ids)
            stats["tgt_len_sum"] += len(tgt_ids)
            stats["src_len_max"] = max(stats["src_len_max"], len(src_ids))
            stats["tgt_len_max"] = max(stats["tgt_len_max"], len(tgt_ids))

            len_bins_src[bin_len(len(src_ids))] += 1
            len_bins_tgt[bin_len(len(tgt_ids))] += 1

    if stats["keep"] > 0:
        stats["src_len_avg"] = stats["src_len_sum"] / stats["keep"]
        stats["tgt_len_avg"] = stats["tgt_len_sum"] / stats["keep"]

    return {
        "counts": dict(stats),
        "len_bins_src": dict(len_bins_src),
        "len_bins_tgt": dict(len_bins_tgt),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)  # 包含 train.jsonl, valid.jsonl, test.jsonl
    ap.add_argument("--out_dir", required=True)    # 输出 tok 目录

    ap.add_argument("--vocab_size_src", type=int, default=16000)
    ap.add_argument("--vocab_size_tgt", type=int, default=16000)
    ap.add_argument("--spm_type", type=str, default="unigram", choices=["unigram", "bpe"])

    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--len_mode", type=str, default="filter", choices=["filter", "truncate"])

    ap.add_argument("--ratio_min", type=float, default=0.2)
    ap.add_argument("--ratio_max", type=float, default=5.0)

    args = ap.parse_args()

    train_clean = os.path.join(args.clean_dir, "train.jsonl")
    valid_clean = os.path.join(args.clean_dir, "valid.jsonl")
    test_clean  = os.path.join(args.clean_dir, "test.jsonl")

    tok_dir = os.path.join(args.out_dir, "tok")
    os.makedirs(tok_dir, exist_ok=True)

    train_zh_txt = os.path.join(tok_dir, "train.zh.txt")
    train_en_txt = os.path.join(tok_dir, "train.en.txt")
    dump_corpus(train_clean, "zh", train_zh_txt)
    dump_corpus(train_clean, "en", train_en_txt)

    spm_src_prefix = os.path.join(tok_dir, "spm_src")
    spm_tgt_prefix = os.path.join(tok_dir, "spm_tgt")

    train_sentencepiece(train_zh_txt, spm_src_prefix, args.vocab_size_src, args.spm_type)
    train_sentencepiece(train_en_txt, spm_tgt_prefix, args.vocab_size_tgt, args.spm_type)

    sp_src = spm.SentencePieceProcessor(model_file=spm_src_prefix + ".model")
    sp_tgt = spm.SentencePieceProcessor(model_file=spm_tgt_prefix + ".model")

    report = defaultdict(dict)
    report["encode"]["train"] = encode_split(
        train_clean,
        os.path.join(tok_dir, "train.ids.jsonl"),
        sp_src, sp_tgt,
        max_tokens=args.max_tokens,
        len_mode=args.len_mode,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
    )
    report["encode"]["valid"] = encode_split(
        valid_clean,
        os.path.join(tok_dir, "valid.ids.jsonl"),
        sp_src, sp_tgt,
        max_tokens=args.max_tokens,
        len_mode=args.len_mode,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
    )
    report["encode"]["test"] = encode_split(
        test_clean,
        os.path.join(tok_dir, "test.ids.jsonl"),
        sp_src, sp_tgt,
        max_tokens=args.max_tokens,
        len_mode=args.len_mode,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
    )

    report["spm"] = {
        "src_model": spm_src_prefix + ".model",
        "tgt_model": spm_tgt_prefix + ".model",
        "src_vocab": spm_src_prefix + ".vocab",
        "tgt_vocab": spm_tgt_prefix + ".vocab",
        "vocab_size_src": args.vocab_size_src,
        "vocab_size_tgt": args.vocab_size_tgt,
        "spm_type": args.spm_type,
        "max_tokens": args.max_tokens,
        "len_mode": args.len_mode,
        "ratio_min": args.ratio_min,
        "ratio_max": args.ratio_max,
    }

    stats_path = os.path.join(tok_dir, "token_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Token outputs:", tok_dir)
    print("Stats:", stats_path)

if __name__ == "__main__":
    main()
