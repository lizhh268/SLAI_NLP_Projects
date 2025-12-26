# decode_rnn.py
import argparse
import json
import os
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader

# 依赖你已有的训练脚本里的模型与数据定义
# 确保 decode_rnn.py 与 train_rnn_attn.py 在同一目录
from train_rnn_attn import Seq2Seq, IdsJsonlDataset, collate_fn, load_refs_from_clean


def clean_ids_for_spm(ids: List[int], pad_id: int = 0, bos_id: int = 1, eos_id: int = 2) -> List[int]:
    out = []
    for t in ids:
        if t == eos_id:
            break
        if t == pad_id or t == bos_id:
            continue
        out.append(int(t))
    return out


def compute_sacrebleu(hyp: List[str], ref: List[str], tokenize: str = "13a") -> Dict[str, Any]:
    import sacrebleu
    bleu_obj = sacrebleu.corpus_bleu(hyp, [ref], tokenize=tokenize)

    sig = None
    if hasattr(bleu_obj, "signature"):
        sig = bleu_obj.signature
    else:
        try:
            bleu_metric = sacrebleu.metrics.BLEU(tokenize=tokenize)
            sig = bleu_metric.get_signature()
        except Exception:
            sig = None

    return {"bleu": float(bleu_obj.score), "signature": sig, "detail": str(bleu_obj)}


@torch.no_grad()
def beam_search_one(
    model: Seq2Seq,
    src_1: torch.Tensor,      # [1, S]
    src_len_1: torch.Tensor,  # [1]
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    beam_size: int,
    len_penalty: float,
) -> List[int]:
    """
    返回不含 BOS/EOS 的 token ids
    """
    model.eval()
    device = src_1.device

    enc_out, enc_state = model.encoder(src_1, src_len_1)  # enc_out: [1,S,H]
    B, S, H = enc_out.shape
    enc_mask = (src_1 != pad_id)[:, :S]  # [1,S]

    def clone_state(state):
        if isinstance(state, tuple):
            return (state[0].clone(), state[1].clone())
        return state.clone()

    def length_penalized(score: float, length: int, alpha: float) -> float:
        if alpha <= 0.0:
            return score
        lp = ((5.0 + length) / 6.0) ** alpha
        return score / lp

    init_ctx = torch.zeros(1, H, device=device)
    init_state = enc_state

    # beam 元素: (tokens, logp, state, ctx, ended)
    beams = [([], 0.0, init_state, init_ctx, False)]

    for _ in range(max_len):
        all_cands = []
        for tokens, logp, state, ctx, ended in beams:
            if ended:
                all_cands.append((tokens, logp, state, ctx, True))
                continue

            y_prev = bos_id if len(tokens) == 0 else tokens[-1]
            y_prev_t = torch.tensor([y_prev], dtype=torch.long, device=device)

            logits, new_state, new_ctx, _ = model.decoder.forward_step(
                y_prev_t, state, enc_out, enc_mask, ctx
            )
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # [V]
            topk_logp, topk_ids = torch.topk(log_probs, k=beam_size)

            for k in range(beam_size):
                tok = int(topk_ids[k].item())
                cand_logp = logp + float(topk_logp[k].item())
                cand_tokens = tokens + [tok]
                cand_ended = (tok == eos_id)
                all_cands.append((cand_tokens, cand_logp, clone_state(new_state), new_ctx.clone(), cand_ended))

        # 先按未加长度惩罚的 logp 排序截断，减少开销
        all_cands.sort(key=lambda x: x[1], reverse=True)
        beams = all_cands[:beam_size]

        if all(b[4] for b in beams):
            break

    # 选择最终最优，按长度惩罚后的分数
    best = None
    best_score = -1e18
    for tokens, logp, _, _, _ in beams:
        # tokens 里可能含 eos
        trimmed = []
        for t in tokens:
            if t == eos_id:
                break
            trimmed.append(t)
        score = length_penalized(logp, max(1, len(trimmed)), len_penalty)
        if score > best_score:
            best_score = score
            best = trimmed

    return best if best is not None else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pt or last.pt")
    ap.add_argument("--ids", required=True, help="valid.ids.jsonl or test.ids.jsonl")
    ap.add_argument("--clean", required=True, help="valid.jsonl or test.jsonl with 'en' field")
    ap.add_argument("--spm_tgt_model", required=True)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_name", default="valid", help="used in output filenames")

    ap.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    ap.add_argument("--beam_size", type=int, default=5)
    ap.add_argument("--len_penalty", type=float, default=0.6)
    ap.add_argument("--max_len", type=int, default=256)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--tokenize", default="13a")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt.get("args", {})

    pad_id = int(train_args.get("pad_id", 0))
    bos_id = int(train_args.get("bos_id", 1))
    eos_id = int(train_args.get("eos_id", 2))

    model = Seq2Seq(
        src_vocab=int(train_args["src_vocab"]),
        tgt_vocab=int(train_args["tgt_vocab"]),
        emb_size=int(train_args["emb_size"]),
        hidden_size=int(train_args["hidden_size"]),
        num_layers=int(train_args["num_layers"]),
        rnn_type=train_args["rnn_type"],
        pad_id=pad_id,
        dropout=float(train_args["dropout"]),
        alignment=train_args["alignment"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    ds = IdsJsonlDataset(args.ids)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_id, bos_id, eos_id),
    )

    refs_in_order, refs_by_index = load_refs_from_clean(args.clean)

    import sentencepiece as spm
    sp_tgt = spm.SentencePieceProcessor(model_file=args.spm_tgt_model)

    hyps: List[str] = []
    refs: List[str] = []
    line_ptr = 0

    for batch in loader:
        src = batch.src.to(device)
        src_len = batch.src_len.to(device)

        if args.decode == "greedy":
            outs = model.greedy_decode(
                src, src_len,
                max_len=args.max_len,
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
            )
        else:
            outs = []
            for i in range(src.size(0)):
                out_i = beam_search_one(
                    model,
                    src[i:i+1],
                    src_len[i:i+1],
                    bos_id=bos_id,
                    eos_id=eos_id,
                    pad_id=pad_id,
                    max_len=args.max_len,
                    beam_size=args.beam_size,
                    len_penalty=args.len_penalty,
                )
                outs.append(out_i)

        for i in range(len(outs)):
            hyp_ids = clean_ids_for_spm(outs[i], pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)
            hyp_text = sp_tgt.decode(hyp_ids)
            hyps.append(hyp_text)

            if batch.index is not None:
                idx = batch.index[i]
                refs.append(refs_by_index[idx])
            else:
                refs.append(refs_in_order[line_ptr])
                line_ptr += 1

    bleu = compute_sacrebleu(hyps, refs, tokenize=args.tokenize)

    tag = args.decode if args.decode == "greedy" else f"beam{args.beam_size}_lp{args.len_penalty}"
    hyp_path = os.path.join(args.out_dir, f"{args.split_name}.hyp.{tag}.txt")
    ref_path = os.path.join(args.out_dir, f"{args.split_name}.ref.txt")
    bleu_path = os.path.join(args.out_dir, f"{args.split_name}.bleu.{tag}.json")

    with open(hyp_path, "w", encoding="utf-8") as f:
        for s in hyps:
            f.write(s.strip() + "\n")
    with open(ref_path, "w", encoding="utf-8") as f:
        for s in refs:
            f.write(s.strip() + "\n")
    with open(bleu_path, "w", encoding="utf-8") as f:
        json.dump(bleu, f, ensure_ascii=False, indent=2)

    sig = bleu["signature"] if bleu.get("signature") else "N/A"
    print(f"{args.split_name} {tag} BLEU = {bleu['bleu']:.2f} | {sig}")
    print("hyp:", hyp_path)
    print("ref:", ref_path)
    print("bleu:", bleu_path)


if __name__ == "__main__":
    main()
