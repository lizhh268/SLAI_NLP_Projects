# evaluate_bleu.py
import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

def _read_text_lines(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if s is None:
                continue
            lines.append(s)
    return lines

def _read_jsonl_field(path: str, field: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at line {i} in {path}: {e}") from e
            if field not in obj:
                raise ValueError(f"Missing field '{field}' at line {i} in {path}")
            if not isinstance(obj[field], str):
                raise ValueError(f"Field '{field}' is not a string at line {i} in {path}")
            lines.append(obj[field])
    return lines

def _load_sentences(path: str, fmt: str, field: Optional[str]) -> List[str]:
    if fmt == "txt":
        return _read_text_lines(path)
    if fmt == "jsonl":
        if not field:
            raise ValueError("When fmt=jsonl, you must provide --field.")
        return _read_jsonl_field(path, field)
    raise ValueError(f"Unknown format: {fmt}")

def _validate_lengths(hyp: List[str], ref: List[str]) -> None:
    if len(hyp) != len(ref):
        raise ValueError(
            f"Length mismatch: hyp has {len(hyp)} lines but ref has {len(ref)} lines. "
            "They must be aligned 1-to-1."
        )

def compute_bleu(
    hyp: List[str],
    ref: List[str],
    tokenize: str,
    lowercase: bool,
    force: bool,
) -> Tuple[float, str, str]:
    try:
        import sacrebleu
    except ImportError as e:
        raise ImportError(
            "sacrebleu is not installed. Please run: pip install sacrebleu"
        ) from e

    bleu = sacrebleu.corpus_bleu(
        hyp,
        [ref],
        tokenize=tokenize,
        lowercase=lowercase,
        force=force,
    )

    score = float(bleu.score)
    signature = bleu.signature
    full_str = str(bleu)
    return score, signature, full_str

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, help="Hypothesis file path.")
    ap.add_argument("--ref", required=True, help="Reference file path.")

    ap.add_argument("--hyp_fmt", default="txt", choices=["txt", "jsonl"])
    ap.add_argument("--ref_fmt", default="txt", choices=["txt", "jsonl"])
    ap.add_argument("--hyp_field", default=None, help="Field name when hyp_fmt=jsonl.")
    ap.add_argument("--ref_field", default=None, help="Field name when ref_fmt=jsonl.")

    ap.add_argument("--tokenize", default="13a", help="SacreBLEU tokenizer, default=13a.")
    ap.add_argument("--lowercase", action="store_true", help="Apply lowercase before scoring.")
    ap.add_argument("--force", action="store_true", help="Allow empty lines.")
    ap.add_argument("--out_json", default=None, help="Optional output JSON path for results.")
    ap.add_argument("--quiet", action="store_true", help="Only print JSON if out_json is not set.")

    args = ap.parse_args()

    hyp = _load_sentences(args.hyp, args.hyp_fmt, args.hyp_field)
    ref = _load_sentences(args.ref, args.ref_fmt, args.ref_field)
    _validate_lengths(hyp, ref)

    score, signature, full_str = compute_bleu(
        hyp=hyp,
        ref=ref,
        tokenize=args.tokenize,
        lowercase=args.lowercase,
        force=args.force,
    )

    result = {
        "bleu": score,
        "signature": signature,
        "n_sentences": len(hyp),
        "hyp_path": os.path.abspath(args.hyp),
        "ref_path": os.path.abspath(args.ref),
        "hyp_fmt": args.hyp_fmt,
        "ref_fmt": args.ref_fmt,
        "hyp_field": args.hyp_field,
        "ref_field": args.ref_field,
        "tokenize": args.tokenize,
        "lowercase": bool(args.lowercase),
        "force": bool(args.force),
        "detail": full_str,
    }

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
