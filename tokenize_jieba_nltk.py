# tokenize_jieba_nltk.py
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Any, Iterable, List, Optional, Tuple

import jieba
from nltk.tokenize import WordPunctTokenizer

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = 0, 1, 2, 3

# 英文分词器，不依赖 punkt
EN_TOKENIZER = WordPunctTokenizer()

# 轻量归一化：把连续空白折叠成单空格
_space_re = re.compile(r"\s+")

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

def normalize_ws(s: str) -> str:
    return _space_re.sub(" ", s).strip()

def tok_zh_jieba(s: str, hmm: bool = False) -> List[str]:
    s = normalize_ws(s)
    # 精确模式，默认关闭 HMM 更稳定可复现
    return [t for t in jieba.lcut(s, cut_all=False, HMM=hmm) if t.strip()]

def tok_en_nltk(s: str) -> List[str]:
    s = normalize_ws(s)
    # WordPunct: 按字母数字串与标点分割，比较接近经典 tokenization baseline
    return [t for t in EN_TOKENIZER.tokenize(s) if t.strip()]

def build_vocab(
    token_counter: Counter,
    min_freq: int,
    max_vocab_size: int,
) -> Dict[str, int]:
    """
    max_vocab_size: 包含 special tokens 的总词表大小上限
    """
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    remain = max(0, max_vocab_size - len(SPECIAL_TOKENS))

    # 先按频次降序，再按字典序作为稳定 tie-break
    candidates = [(tok, c) for tok, c in token_counter.items() if c >= min_freq and tok not in vocab]
    candidates.sort(key=lambda x: (-x[1], x[0]))

    for tok, _ in candidates[:remain]:
        vocab[tok] = len(vocab)

    return vocab

def encode_tokens(tokens: List[str], vocab: Dict[str, int]) -> Tuple[List[int], int]:
    """
    返回 ids 与 unk_count
    """
    unk = vocab.get("<unk>", UNK)
    ids = []
    unk_count = 0
    for t in tokens:
        i = vocab.get(t, unk)
        if i == unk:
            unk_count += 1
        ids.append(i)
    return ids, unk_count

def bin_len(x: int) -> str:
    if x <= 32: return "0-32"
    if x <= 64: return "33-64"
    if x <= 128: return "65-128"
    if x <= 256: return "129-256"
    if x <= 512: return "257-512"
    return "513+"

def process_split(
    clean_path: str,
    out_ids_path: str,
    vocab_zh: Dict[str, int],
    vocab_en: Dict[str, int],
    max_tokens: int,
    len_mode: str,     # filter or truncate
    ratio_min: float,
    ratio_max: float,
    hmm: bool,
) -> Dict[str, Any]:
    stats = Counter()
    len_bins_src = Counter()
    len_bins_tgt = Counter()

    src_unk = 0
    tgt_unk = 0
    src_tok_total = 0
    tgt_tok_total = 0

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

            zh_tok = tok_zh_jieba(zh, hmm=hmm)
            en_tok = tok_en_nltk(en)

            if len(zh_tok) == 0 or len(en_tok) == 0:
                stats["drop_empty_after_tokenize"] += 1
                continue

            src_ids_core, src_unk_cnt = encode_tokens(zh_tok, vocab_zh)
            tgt_ids_core, tgt_unk_cnt = encode_tokens(en_tok, vocab_en)

            src_ids = [BOS] + src_ids_core + [EOS]
            tgt_ids = [BOS] + tgt_ids_core + [EOS]

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

            src_unk += src_unk_cnt
            tgt_unk += tgt_unk_cnt
            src_tok_total += len(src_ids_core)
            tgt_tok_total += len(tgt_ids_core)

    if stats["keep"] > 0:
        stats["src_len_avg"] = stats["src_len_sum"] / stats["keep"]
        stats["tgt_len_avg"] = stats["tgt_len_sum"] / stats["keep"]

    oov = {
        "src_unk_tokens": src_unk,
        "tgt_unk_tokens": tgt_unk,
        "src_total_tokens": src_tok_total,
        "tgt_total_tokens": tgt_tok_total,
        "src_unk_rate": (src_unk / src_tok_total) if src_tok_total > 0 else 0.0,
        "tgt_unk_rate": (tgt_unk / tgt_tok_total) if tgt_tok_total > 0 else 0.0,
    }

    return {
        "counts": dict(stats),
        "len_bins_src": dict(len_bins_src),
        "len_bins_tgt": dict(len_bins_tgt),
        "oov": oov,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)   # 包含 train.jsonl valid.jsonl test.jsonl
    ap.add_argument("--out_dir", required=True)     # 输出目录，会创建 word_tok 子目录

    ap.add_argument("--min_freq_zh", type=int, default=2)
    ap.add_argument("--min_freq_en", type=int, default=2)
    ap.add_argument("--max_vocab_zh", type=int, default=30000)  # 含 special tokens
    ap.add_argument("--max_vocab_en", type=int, default=30000)

    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--len_mode", type=str, default="filter", choices=["filter", "truncate"])
    ap.add_argument("--ratio_min", type=float, default=0.2)
    ap.add_argument("--ratio_max", type=float, default=5.0)

    ap.add_argument("--jieba_hmm", action="store_true")  # 若指定则开启 HMM

    args = ap.parse_args()

    train_path = os.path.join(args.clean_dir, "train.jsonl")
    valid_path = os.path.join(args.clean_dir, "valid.jsonl")
    test_path  = os.path.join(args.clean_dir, "test.jsonl")

    out_base = os.path.join(args.out_dir, "word_tok")
    os.makedirs(out_base, exist_ok=True)

    # 1) 统计词频只用训练集
    zh_counter = Counter()
    en_counter = Counter()
    for _, obj, err in iter_jsonl(train_path):
        if err or obj is None:
            continue
        if "zh" not in obj or "en" not in obj:
            continue
        zh, en = obj["zh"], obj["en"]
        if not isinstance(zh, str) or not isinstance(en, str):
            continue
        zh_counter.update(tok_zh_jieba(zh, hmm=args.jieba_hmm))
        en_counter.update(tok_en_nltk(en))

    # 2) 构建词表
    vocab_zh = build_vocab(zh_counter, args.min_freq_zh, args.max_vocab_zh)
    vocab_en = build_vocab(en_counter, args.min_freq_en, args.max_vocab_en)

    vocab_zh_path = os.path.join(out_base, "vocab_zh.json")
    vocab_en_path = os.path.join(out_base, "vocab_en.json")
    with open(vocab_zh_path, "w", encoding="utf-8") as f:
        json.dump(vocab_zh, f, ensure_ascii=False, indent=2)
    with open(vocab_en_path, "w", encoding="utf-8") as f:
        json.dump(vocab_en, f, ensure_ascii=False, indent=2)

    # 3) 编码并导出 ids
    report = defaultdict(dict)
    report["vocab"] = {
        "min_freq_zh": args.min_freq_zh,
        "min_freq_en": args.min_freq_en,
        "max_vocab_zh": args.max_vocab_zh,
        "max_vocab_en": args.max_vocab_en,
        "vocab_size_zh": len(vocab_zh),
        "vocab_size_en": len(vocab_en),
        "vocab_zh_path": vocab_zh_path,
        "vocab_en_path": vocab_en_path,
        "jieba_hmm": bool(args.jieba_hmm),
    }
    report["encode"]["train"] = process_split(
        train_path,
        os.path.join(out_base, "train.ids.jsonl"),
        vocab_zh, vocab_en,
        max_tokens=args.max_tokens,
        len_mode=args.len_mode,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        hmm=args.jieba_hmm,
    )
    report["encode"]["valid"] = process_split(
        valid_path,
        os.path.join(out_base, "valid.ids.jsonl"),
        vocab_zh, vocab_en,
        max_tokens=args.max_tokens,
        len_mode=args.len_mode,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        hmm=args.jieba_hmm,
    )
    report["encode"]["test"] = process_split(
        test_path,
        os.path.join(out_base, "test.ids.jsonl"),
        vocab_zh, vocab_en,
        max_tokens=args.max_tokens,
        len_mode=args.len_mode,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        hmm=args.jieba_hmm,
    )

    report["config"] = {
        "max_tokens": args.max_tokens,
        "len_mode": args.len_mode,
        "ratio_min": args.ratio_min,
        "ratio_max": args.ratio_max,
        "special_tokens": SPECIAL_TOKENS,
    }

    stats_path = os.path.join(out_base, "word_token_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Outputs:", out_base)
    print("Vocab zh:", vocab_zh_path)
    print("Vocab en:", vocab_en_path)
    print("Stats:", stats_path)

if __name__ == "__main__":
    main()
