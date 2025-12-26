# train_rnn_attn.py
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Data
# -----------------------------

@dataclass
class Batch:
    src: torch.Tensor          # [B, S]
    src_len: torch.Tensor      # [B]
    tgt_in: torch.Tensor       # [B, T]  (BOS ... )
    tgt_out: torch.Tensor      # [B, T]  (... EOS)
    index: Optional[List[int]] # list length B or None

class IdsJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.items.append(obj)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        obj = self.items[i]
        src_ids = obj["src_ids"]
        tgt_ids = obj["tgt_ids"]
        idx = obj.get("index", None)
        return src_ids, tgt_ids, idx

def pad_1d(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lens.max().item())
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out, lens

def collate_fn(batch, pad_id: int, bos_id: int, eos_id: int) -> Batch:
    src_seqs, tgt_seqs, idxs = zip(*batch)

    src, src_len = pad_1d(list(src_seqs), pad_id=pad_id)

    # tgt_in: remove last token, tgt_out: remove first token
    tgt_in_seqs = [t[:-1] for t in tgt_seqs]
    tgt_out_seqs = [t[1:] for t in tgt_seqs]

    tgt_in, _ = pad_1d(tgt_in_seqs, pad_id=pad_id)
    tgt_out, _ = pad_1d(tgt_out_seqs, pad_id=pad_id)

    index_list = None
    if all(x is not None for x in idxs):
        index_list = [int(x) for x in idxs]

    return Batch(
        src=src,
        src_len=src_len,
        tgt_in=tgt_in,
        tgt_out=tgt_out,
        index=index_list,
    )

# -----------------------------
# Attention
# -----------------------------

class Attention(nn.Module):
    """
    alignment: dot | general | additive
    query: [B, H], keys: [B, S, H], mask: [B, S] with True for valid positions
    returns context [B, H], attn [B, S]
    """
    def __init__(self, hidden_size: int, alignment: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.alignment = alignment

        if alignment == "general":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif alignment == "additive":
            self.Wq = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Wk = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)
        elif alignment == "dot":
            pass
        else:
            raise ValueError(f"Unknown alignment: {alignment}")

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor):
        # query: [B,H], keys: [B,S,H], mask: [B,S]
        if self.alignment == "dot":
            scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)  # [B,S]
        elif self.alignment == "general":
            q = self.Wa(query)  # [B,H]
            scores = torch.bmm(keys, q.unsqueeze(2)).squeeze(2)
        else:
            # additive
            q = self.Wq(query).unsqueeze(1)      # [B,1,H]
            k = self.Wk(keys)                    # [B,S,H]
            scores = self.v(torch.tanh(q + k)).squeeze(2)  # [B,S]

        scores = scores.masked_fill(~mask, -1e9)
        attn = F.softmax(scores, dim=1)  # [B,S]
        ctx = torch.bmm(attn.unsqueeze(1), keys).squeeze(1)  # [B,H]
        return ctx, attn

# -----------------------------
# Model
# -----------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int, num_layers: int, rnn_type: str, pad_id: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, src: torch.Tensor, src_len: torch.Tensor):
        # src: [B,S]
        emb = self.emb(src)  # [B,S,E]
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        enc_out_packed, h = self.rnn(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out_packed, batch_first=True)  # [B,S,H]
        return enc_out, h

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int, num_layers: int, rnn_type: str, pad_id: int, dropout: float, alignment: str):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        self.attn = Attention(hidden_size, alignment=alignment)
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM

        # input feeding: concat(emb, ctx)
        self.rnn = rnn_cls(
            input_size=emb_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_size + hidden_size, vocab_size)

    def forward_step(self, y_prev: torch.Tensor, state, enc_out: torch.Tensor, enc_mask: torch.Tensor, ctx_prev: torch.Tensor):
        # y_prev: [B]
        emb = self.emb(y_prev)  # [B,E]
        rnn_in = torch.cat([emb, ctx_prev], dim=-1).unsqueeze(1)  # [B,1,E+H]
        rnn_out, state = self.rnn(rnn_in, state)  # rnn_out: [B,1,H]
        h_t = rnn_out.squeeze(1)                  # [B,H]

        ctx, attn = self.attn(h_t, enc_out, enc_mask)  # [B,H], [B,S]
        logits = self.out(torch.cat([h_t, ctx], dim=-1))  # [B,V]
        return logits, state, ctx, attn

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, emb_size: int, hidden_size: int, num_layers: int,
                 rnn_type: str, pad_id: int, dropout: float, alignment: str):
        super().__init__()
        self.encoder = Encoder(src_vocab, emb_size, hidden_size, num_layers, rnn_type, pad_id, dropout)
        self.decoder = Decoder(tgt_vocab, emb_size, hidden_size, num_layers, rnn_type, pad_id, dropout, alignment)

        # bridge for LSTM hidden state size alignment, keep identity for simplicity
        self.rnn_type = rnn_type

    def forward(self, batch: Batch, teacher_forcing: float, pad_id: int):
        src, src_len = batch.src, batch.src_len
        tgt_in, tgt_out = batch.tgt_in, batch.tgt_out

        enc_out, enc_state = self.encoder(src, src_len)

        B, S, H = enc_out.shape
        enc_mask = (src != pad_id)[:, :S]  # [B,S]

        # init decoder state from encoder final state
        dec_state = enc_state
        ctx = torch.zeros(B, H, device=src.device)

        T = tgt_in.size(1)
        logits_all = []

        y_prev = tgt_in[:, 0]  # BOS
        for t in range(T):
            if t == 0:
                y_prev = tgt_in[:, 0]
            else:
                use_tf = (torch.rand(B, device=src.device) < teacher_forcing)
                y_prev = torch.where(use_tf, tgt_in[:, t], y_prev)

            logits, dec_state, ctx, _ = self.decoder.forward_step(y_prev, dec_state, enc_out, enc_mask, ctx)
            logits_all.append(logits.unsqueeze(1))
            y_prev = logits.argmax(dim=-1)

        logits_all = torch.cat(logits_all, dim=1)  # [B,T,V]
        loss = F.cross_entropy(
            logits_all.reshape(-1, logits_all.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=pad_id,
        )
        return loss

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, src_len: torch.Tensor, max_len: int, bos_id: int, eos_id: int, pad_id: int):
        enc_out, enc_state = self.encoder(src, src_len)
        B, S, H = enc_out.shape
        enc_mask = (src != pad_id)[:, :S]

        dec_state = enc_state
        ctx = torch.zeros(B, H, device=src.device)

        y_prev = torch.full((B,), bos_id, dtype=torch.long, device=src.device)
        outputs = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            logits, dec_state, ctx, _ = self.decoder.forward_step(y_prev, dec_state, enc_out, enc_mask, ctx)
            y_prev = logits.argmax(dim=-1)

            for i in range(B):
                if finished[i]:
                    continue
                tok = int(y_prev[i].item())
                if tok == eos_id:
                    finished[i] = True
                else:
                    outputs[i].append(tok)

            if finished.all():
                break
        return outputs

# -----------------------------
# BLEU helper
# -----------------------------

def load_refs_from_clean(clean_jsonl: str) -> Tuple[List[str], Dict[int, str]]:
    refs_in_order = []
    refs_by_index: Dict[int, str] = {}
    with open(clean_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            en = obj["en"]
            refs_in_order.append(en)
            if "index" in obj:
                refs_by_index[int(obj["index"])] = en
    return refs_in_order, refs_by_index

def compute_sacrebleu(hyp, ref, tokenize="13a"):
    import sacrebleu

    bleu_obj = sacrebleu.corpus_bleu(hyp, [ref], tokenize=tokenize)

    sig = None
    if hasattr(bleu_obj, "signature"):
        sig = bleu_obj.signature
    else:
        # 尝试兼容 sacrebleu 新接口
        try:
            bleu_metric = sacrebleu.metrics.BLEU(tokenize=tokenize)
            sig = bleu_metric.get_signature()
        except Exception:
            sig = None

    return {
        "bleu": float(getattr(bleu_obj, "score", bleu_obj.score)),
        "signature": sig,
        "detail": str(bleu_obj),
    }


# -----------------------------
# Train
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def infer_vocab_size(ids_jsonl: str) -> int:
    mx = 0
    with open(ids_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            mx = max(mx, max(obj["src_ids"]), max(obj["tgt_ids"]))
    return mx + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ids", required=True)
    ap.add_argument("--valid_ids", required=True)
    ap.add_argument("--valid_clean", required=True)
    ap.add_argument("--spm_tgt_model", required=True)

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--src_vocab", type=int, default=0, help="0 means infer from ids")
    ap.add_argument("--tgt_vocab", type=int, default=0, help="0 means infer from ids")
    ap.add_argument("--pad_id", type=int, default=0)
    ap.add_argument("--bos_id", type=int, default=1)
    ap.add_argument("--eos_id", type=int, default=2)

    ap.add_argument("--rnn_type", choices=["gru", "lstm"], default="lstm")
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--emb_size", type=int, default=256)
    ap.add_argument("--hidden_size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--alignment", choices=["dot", "general", "additive"], default="dot")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--teacher_forcing", type=float, default=1.0)

    ap.add_argument("--decode_max_len", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=1)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.src_vocab == 0 or args.tgt_vocab == 0:
        v = infer_vocab_size(args.train_ids)
        if args.src_vocab == 0:
            args.src_vocab = v
        if args.tgt_vocab == 0:
            args.tgt_vocab = v

    train_ds = IdsJsonlDataset(args.train_ids)
    valid_ds = IdsJsonlDataset(args.valid_ids)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, args.pad_id, args.bos_id, args.eos_id),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, args.pad_id, args.bos_id, args.eos_id),
    )

    model = Seq2Seq(
        src_vocab=args.src_vocab,
        tgt_vocab=args.tgt_vocab,
        emb_size=args.emb_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        rnn_type=args.rnn_type,
        pad_id=args.pad_id,
        dropout=args.dropout,
        alignment=args.alignment,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    refs_in_order, refs_by_index = load_refs_from_clean(args.valid_clean)

    import sentencepiece as spm
    sp_tgt = spm.SentencePieceProcessor(model_file=args.spm_tgt_model)

    best_bleu = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch.src = batch.src.to(device)
            batch.src_len = batch.src_len.to(device)
            batch.tgt_in = batch.tgt_in.to(device)
            batch.tgt_out = batch.tgt_out.to(device)

            opt.zero_grad(set_to_none=True)
            loss = model(batch, teacher_forcing=args.teacher_forcing, pad_id=args.pad_id)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[epoch {epoch}] train_loss={avg_loss:.4f}")

        if epoch % args.eval_every == 0:
            model.eval()
            hyps: List[str] = []
            refs: List[str] = []

            line_ptr = 0
            for batch in valid_loader:
                src = batch.src.to(device)
                src_len = batch.src_len.to(device)
                outs = model.greedy_decode(
                    src, src_len,
                    max_len=args.decode_max_len,
                    bos_id=args.bos_id,
                    eos_id=args.eos_id,
                    pad_id=args.pad_id
                )

                for i in range(len(outs)):
                    hyp_text = sp_tgt.decode(outs[i])
                    hyps.append(hyp_text)

                    if batch.index is not None:
                        idx = batch.index[i]
                        refs.append(refs_by_index[idx])
                    else:
                        refs.append(refs_in_order[line_ptr])
                        line_ptr += 1

            bleu = compute_sacrebleu(hyps, refs)
            print(f"[epoch {epoch}] valid_bleu={bleu['bleu']:.2f} | {bleu['signature']}")

            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "valid_bleu": bleu,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))

            if bleu["bleu"] > best_bleu:
                best_bleu = bleu["bleu"]
                torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))

                with open(os.path.join(args.out_dir, "valid.hyp.txt"), "w", encoding="utf-8") as f:
                    for s in hyps:
                        f.write(s.strip() + "\n")
                with open(os.path.join(args.out_dir, "valid.ref.txt"), "w", encoding="utf-8") as f:
                    for s in refs:
                        f.write(s.strip() + "\n")
                with open(os.path.join(args.out_dir, "valid_bleu.json"), "w", encoding="utf-8") as f:
                    json.dump(bleu, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
