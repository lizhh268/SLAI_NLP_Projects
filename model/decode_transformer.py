# decode_transformer.py
import argparse
import json
import os
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 确保 decode_transformer.py 与 train_transformer_nmt.py 在同一目录
from train_transformer_nmt import (
    TransformerNMT,
    IdsJsonlDataset,
    collate_fn,
    load_refs_from_clean,
    clean_ids_for_spm,
    compute_sacrebleu,
)


@torch.no_grad()
def greedy_decode_batch(
    model: TransformerNMT,
    src: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
) -> List[List[int]]:
    return model.greedy_decode(src, bos_id=bos_id, eos_id=eos_id, max_len=max_len)


@torch.no_grad()
def beam_search_batch(
    model: TransformerNMT,
    src: torch.Tensor,          # [B,S]
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    beam_size: int,
    len_penalty: float,
) -> List[List[int]]:
    """
    朴素 beam search（无 KV-cache），B=64 之类仍可用，但会慢一些。
    返回每条样本不含 BOS/EOS 的 token ids。
    """
    model.eval()
    device = src.device
    B = src.size(0)

    # encode once
    memory, src_pad_mask = model.encode(src)

    # 每条样本维护 beam
    # beam 元素: (tokens_tensor [t], logp, ended)
    beams = [[(torch.empty(0, dtype=torch.long, device=device), 0.0, False)] for _ in range(B)]

    def lp(score: float, length: int, alpha: float) -> float:
        if alpha <= 0.0:
            return score
        return score / (((5.0 + length) / 6.0) ** alpha)

    for _ in range(max_len):
        all_finished = True

        for i in range(B):
            cur = beams[i]
            if all(x[2] for x in cur):
                continue
            all_finished = False

            cand_list = []
            for tokens, logp, ended in cur:
                if ended:
                    cand_list.append((tokens, logp, True))
                    continue

                # 组装当前输入：BOS + tokens
                ys = torch.cat(
                    [torch.tensor([bos_id], device=device, dtype=torch.long), tokens],
                    dim=0
                ).unsqueeze(0)  # [1, t+1]

                logits = model.decode(ys, memory[i:i+1], src_pad_mask[i:i+1])  # [1, t+1, V]
                next_logp = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)  # [V]

                topk_logp, topk_ids = torch.topk(next_logp, k=beam_size)
                for k in range(beam_size):
                    tok = int(topk_ids[k].item())
                    new_logp = logp + float(topk_logp[k].item())
                    new_tokens = torch.cat([tokens, torch.tensor([tok], device=device)], dim=0)
                    new_ended = (tok == eos_id)
                    cand_list.append((new_tokens, new_logp, new_ended))

            # 先按 raw logp 截断
            cand_list.sort(key=lambda x: x[1], reverse=True)
            beams[i] = cand_list[:beam_size]

        if all_finished:
            break

    # 选最优 beam（长度惩罚）
    outputs: List[List[int]] = []
    for i in range(B):
        best = None
        best_score = -1e18
        for tokens, logp, _ in beams[i]:
            # 去掉 eos 之后的部分
            toks = tokens.tolist()
            trimmed = []
            for t in toks:
                if t == eos_id:
                    break
                trimmed.append(t)
            score = lp(logp, max(1, len(trimmed)), len_penalty)
            if score > best_score:
                best_score = score
                best = trimmed
        outputs.append(best if best is not None else [])
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pt or last.pt")
    ap.add_argument("--ids", required=True, help="valid.ids.jsonl or test.ids.jsonl")
    ap.add_argument("--clean", required=True, help="valid.jsonl or test.jsonl with 'en' field")
    ap.add_argument("--spm_tgt_model", required=True)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split_name", default="valid")

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

    model = TransformerNMT(
        src_vocab=int(train_args["src_vocab"]),
        tgt_vocab=int(train_args["tgt_vocab"]),
        d_model=int(train_args["d_model"]),
        n_heads=int(train_args["n_heads"]),
        num_layers_enc=int(train_args["enc_layers"]),
        num_layers_dec=int(train_args["dec_layers"]),
        d_ff=int(train_args["d_ff"]),
        dropout=float(train_args["dropout"]),
        pad_id=pad_id,
        pos_encoding=train_args["pos_encoding"],
        norm=train_args["norm"],
        max_len=4096,
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

        if args.decode == "greedy":
            outs = greedy_decode_batch(model, src, bos_id=bos_id, eos_id=eos_id, max_len=args.max_len)
        else:
            outs = beam_search_batch(
                model, src,
                bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
                max_len=args.max_len, beam_size=args.beam_size, len_penalty=args.len_penalty
            )

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
