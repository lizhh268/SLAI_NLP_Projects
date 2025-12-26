# clean_data.py
import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Any, Tuple, Optional, Iterable

CONTROL_CATEGORIES = {"Cc"}  # Unicode Control

_space_re = re.compile(r"\s+")
_paren_space_re = re.compile(r"([\(\（])\s+|\s+([\)\）])")
_punct_space_re = re.compile(r"\s+([,.;:!?，。；：！？\)\）])")

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

def remove_control_chars(s: str) -> str:
    return "".join(ch for ch in s if unicodedata.category(ch) not in CONTROL_CATEGORIES)

def normalize_nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def fix_unpaired_quotes(s: str) -> str:
    # 若弯引号不配对，则删除该类引号，避免引入异常 token
    for ql, qr in [("“", "”"), ("‘", "’")]:
        if s.count(ql) != s.count(qr):
            s = s.replace(ql, "").replace(qr, "")
    return s

def fix_spacing(s: str) -> str:
    # 统一空白符，清理括号内侧空格，清理标点前空格
    s = _space_re.sub(" ", s).strip()
    s = _paren_space_re.sub(lambda m: (m.group(1) or "") + (m.group(2) or ""), s)
    s = _punct_space_re.sub(r"\1", s)
    return s

def clean_text(s: str) -> str:
    s = remove_control_chars(s)
    s = normalize_nfkc(s)
    s = fix_unpaired_quotes(s)
    s = fix_spacing(s)
    return s

def clean_jsonl(
    in_path: str,
    out_path: str,
    max_char_zh: int,
    max_char_en: int,
) -> Dict[str, Any]:
    stats = Counter()
    lens = {"zh_sum": 0, "en_sum": 0, "zh_max": 0, "en_max": 0}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for _, obj, err in iter_jsonl(in_path):
            if err:
                stats[f"skip_{err}"] += 1
                continue

            if not isinstance(obj, dict) or "zh" not in obj or "en" not in obj:
                stats["skip_missing_fields"] += 1
                continue

            zh0, en0 = obj["zh"], obj["en"]
            if not isinstance(zh0, str) or not isinstance(en0, str):
                stats["skip_non_str"] += 1
                continue

            zh = clean_text(zh0)
            en = clean_text(en0)

            if len(zh) == 0 or len(en) == 0:
                stats["drop_empty_after_clean"] += 1
                continue

            if len(zh) > max_char_zh or len(en) > max_char_en:
                stats["drop_too_long_char"] += 1
                continue

            obj["zh"] = zh
            obj["en"] = en

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats["keep"] += 1

            lens["zh_sum"] += len(zh)
            lens["en_sum"] += len(en)
            lens["zh_max"] = max(lens["zh_max"], len(zh))
            lens["en_max"] = max(lens["en_max"], len(en))

    out = {"counts": dict(stats), "char_len": lens}
    if stats["keep"] > 0:
        out["char_len"]["zh_avg"] = lens["zh_sum"] / stats["keep"]
        out["char_len"]["en_avg"] = lens["en_sum"] / stats["keep"]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", required=True)
    ap.add_argument("--valid_in", required=True)
    ap.add_argument("--test_in", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--max_char_zh", type=int, default=2000)
    ap.add_argument("--max_char_en", type=int, default=2000)

    args = ap.parse_args()

    out_clean_dir = os.path.join(args.out_dir, "clean")
    os.makedirs(out_clean_dir, exist_ok=True)

    paths = {
        "train": (args.train_in, os.path.join(out_clean_dir, "train.jsonl")),
        "valid": (args.valid_in, os.path.join(out_clean_dir, "valid.jsonl")),
        "test":  (args.test_in,  os.path.join(out_clean_dir, "test.jsonl")),
    }

    report = defaultdict(dict)
    for split, (inp, outp) in paths.items():
        report[split] = clean_jsonl(
            inp, outp,
            max_char_zh=args.max_char_zh,
            max_char_en=args.max_char_en,
        )

    stats_path = os.path.join(out_clean_dir, "clean_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Clean files:", out_clean_dir)
    print("Stats:", stats_path)

if __name__ == "__main__":
    main()
