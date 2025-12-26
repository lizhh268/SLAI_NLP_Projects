# train_transformer_nmt.py
# PyTorch encoder-decoder Transformer for Zh->En using SentencePiece ids.jsonl
# Supports position encoding ablation: sinusoidal | learned | rope
# Supports norm ablation: layernorm | rmsnorm
# Adds staged ablation mode:
#   Stage 1: compare position encodings with fixed norm (default: layernorm)
#   Stage 2: compare norms under best position encoding

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import argparse
import json
import math
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
    src: torch.Tensor
    src_len: torch.Tensor
    tgt_in: torch.Tensor
    tgt_out: torch.Tensor
    index: Optional[List[int]]

class IdsJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        obj = self.items[i]
        return obj["src_ids"], obj["tgt_ids"], obj.get("index", None)

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

    tgt_in_seqs = [t[:-1] for t in tgt_seqs]
    tgt_out_seqs = [t[1:] for t in tgt_seqs]
    tgt_in, _ = pad_1d(tgt_in_seqs, pad_id=pad_id)
    tgt_out, _ = pad_1d(tgt_out_seqs, pad_id=pad_id)

    index_list = None
    if all(x is not None for x in idxs):
        index_list = [int(x) for x in idxs]

    return Batch(src=src, src_len=src_len, tgt_in=tgt_in, tgt_out=tgt_out, index=index_list)

def infer_vocab_size(ids_jsonl: str) -> int:
    mx = 0
    with open(ids_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            mx = max(mx, max(obj["src_ids"]), max(obj["tgt_ids"]))
    return mx + 1


# -----------------------------
# BLEU helpers
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

def clean_ids_for_spm(ids: List[int], pad_id: int, bos_id: int, eos_id: int) -> List[int]:
    out = []
    for t in ids:
        if t == eos_id:
            break
        if t == pad_id or t == bos_id:
            continue
        out.append(int(t))
    return out


# -----------------------------
# Norm
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return x * self.weight

def make_norm(kind: str, dim: int) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    if kind == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unknown norm: {kind}")


# -----------------------------
# Position encoding
# -----------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        self.pos = nn.Embedding(max_len, dim)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        if T > self.max_len:
            raise ValueError(f"Sequence too long {T} > max_len {self.max_len}")
        idx = torch.arange(T, device=x.device)
        return x + self.pos(idx).unsqueeze(0)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 4096, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_len = max_len
        self._cached = {}

    def get_sin_cos(self, seq_len: int, device, dtype):
        if seq_len > self.max_len:
            raise ValueError(f"RoPE seq_len too long: {seq_len} > {self.max_len}")
        key = (seq_len, device.type, dtype)
        if key in self._cached:
            return self._cached[key]
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        sin = emb.sin().to(dtype)
        cos = emb.cos().to(dtype)
        self._cached[key] = (sin, cos)
        return sin, cos


# -----------------------------
# Attention + Transformer blocks
# -----------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, pos_encoding: str):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.pos_encoding = pos_encoding

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.rope = None
        if pos_encoding == "rope":
            self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, Tq, _ = q.shape
        _, Tk, _ = k.shape

        q = self.q_proj(q).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            sin_q, cos_q = self.rope.get_sin_cos(Tq, q.device, q.dtype)
            sin_k, cos_k = self.rope.get_sin_cos(Tk, k.device, k.dtype)
            q = apply_rope(q, sin_q.view(1, 1, Tq, self.head_dim), cos_q.view(1, 1, Tq, self.head_dim))
            k = apply_rope(k, sin_k.view(1, 1, Tk, self.head_dim), cos_k.view(1, 1, Tk, self.head_dim))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, Tk), -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        out = self.o_proj(out)
        out = self.resid_drop(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.w2(x)
        x = self.drop(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, norm: str, pos_encoding: str):
        super().__init__()
        self.norm1 = make_norm(norm, d_model)
        self.norm2 = make_norm(norm, d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, pos_encoding=pos_encoding)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=None, key_padding_mask=src_key_padding_mask)
        x = x + h
        h = self.ff(self.norm2(x))
        x = x + h
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, norm: str, pos_encoding: str):
        super().__init__()
        self.norm1 = make_norm(norm, d_model)
        self.norm2 = make_norm(norm, d_model)
        self.norm3 = make_norm(norm, d_model)

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, pos_encoding=pos_encoding)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, pos_encoding=pos_encoding)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_attn_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                           attn_mask=tgt_attn_mask, key_padding_mask=tgt_key_padding_mask)
        x = x + h

        h = self.cross_attn(self.norm2(x), memory, memory,
                            attn_mask=None, key_padding_mask=src_key_padding_mask)
        x = x + h

        h = self.ff(self.norm3(x))
        x = x + h
        return x

class TransformerNMT(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int,
        n_heads: int,
        num_layers_enc: int,
        num_layers_dec: int,
        d_ff: int,
        dropout: float,
        pad_id: int,
        pos_encoding: str,
        norm: str,
        max_len: int = 4096,
    ):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.pad_id = pad_id
        self.pos_encoding = pos_encoding

        self.pos_src = None
        self.pos_tgt = None
        if pos_encoding == "sinusoidal":
            self.pos_src = SinusoidalPositionalEncoding(d_model, max_len=max_len)
            self.pos_tgt = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        elif pos_encoding == "learned":
            self.pos_src = LearnedPositionalEmbedding(d_model, max_len=max_len)
            self.pos_tgt = LearnedPositionalEmbedding(d_model, max_len=max_len)
        elif pos_encoding == "rope":
            pass
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, norm=norm, pos_encoding=pos_encoding)
            for _ in range(num_layers_enc)
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, norm=norm, pos_encoding=pos_encoding)
            for _ in range(num_layers_dec)
        ])

        self.final_norm = make_norm(norm, d_model)
        self.lm_head = nn.Linear(d_model, tgt_vocab, bias=False)

    def make_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (x == self.pad_id)

    def make_causal_mask(self, T: int, device, dtype) -> torch.Tensor:
        m = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        out = torch.zeros(T, T, device=device, dtype=dtype)
        out = out.masked_fill(m, -1e9)
        return out

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        src_pad_mask = self.make_padding_mask(src)
        x = self.src_emb(src)
        if self.pos_src is not None:
            x = self.pos_src(x)
        x = self.drop(x)
        for layer in self.enc_layers:
            x = layer(x, src_key_padding_mask=src_pad_mask)
        return x, src_pad_mask

    def decode(self, tgt_in: torch.Tensor, memory: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        tgt_pad_mask = self.make_padding_mask(tgt_in)
        x = self.tgt_emb(tgt_in)
        if self.pos_tgt is not None:
            x = self.pos_tgt(x)
        x = self.drop(x)

        T = tgt_in.size(1)
        causal = self.make_causal_mask(T, device=tgt_in.device, dtype=x.dtype)
        causal = causal.view(1, 1, T, T)

        for layer in self.dec_layers:
            x = layer(
                x, memory,
                tgt_attn_mask=causal,
                tgt_key_padding_mask=tgt_pad_mask,
                src_key_padding_mask=src_pad_mask,
            )

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, src_pad_mask = self.encode(src)
        logits = self.decode(tgt_in, memory, src_pad_mask)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, bos_id: int, eos_id: int, max_len: int) -> List[List[int]]:
        self.eval()
        memory, src_pad_mask = self.encode(src)
        B = src.size(0)

        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        outputs = [[] for _ in range(B)]

        for _ in range(max_len):
            logits = self.decode(ys, memory, src_pad_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1)

            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            for i in range(B):
                if finished[i]:
                    continue
                t = int(next_tok[i].item())
                if t == eos_id:
                    finished[i] = True
                else:
                    outputs[i].append(t)

            if finished.all():
                break
        return outputs


# -----------------------------
# Loss and schedule
# -----------------------------

def label_smoothed_nll_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int, eps: float) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(logp, targets, reduction="none", ignore_index=pad_id)
    if eps <= 0.0:
        return nll.mean()

    smooth = -logp.mean(dim=-1)
    mask = (targets != pad_id)
    nll = nll[mask]
    smooth = smooth[mask]
    loss = (1.0 - eps) * nll + eps * smooth
    return loss.mean()

class WarmupCosineLR:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, peak_lr: float, final_lr: float):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(self.warmup + 1, total_steps)
        self.peak = peak_lr
        self.final = final_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup:
            lr = self.peak * (self.step_num / self.warmup)
        else:
            progress = (self.step_num - self.warmup) / (self.total - self.warmup)
            progress = min(max(progress, 0.0), 1.0)
            lr = self.final + 0.5 * (self.peak - self.final) * (1.0 + math.cos(math.pi * progress))

        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


# -----------------------------
# Train utils
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def run_single_experiment(args) -> float:
    """
    Run one training job and return best BLEU.
    This function is used by both single-run mode and staged ablation mode.
    """
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

    steps_per_epoch = len(train_loader)
    true_total_steps = steps_per_epoch * args.epochs

    if args.total_steps is None or args.total_steps <= 0 or args.total_steps > true_total_steps * 2:
        args.total_steps = true_total_steps

    if args.warmup_steps is None or args.warmup_steps <= 0 or args.warmup_steps > args.total_steps:
        args.warmup_steps = max(200, min(2000, int(args.total_steps * 0.08)))

    print(f"[schedule] steps_per_epoch={steps_per_epoch} total_steps={args.total_steps} warmup_steps={args.warmup_steps}")
    print(f"[config] pos_encoding={args.pos_encoding} norm={args.norm} out_dir={args.out_dir}")

    model = TransformerNMT(
        src_vocab=args.src_vocab,
        tgt_vocab=args.tgt_vocab,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers_enc=args.enc_layers,
        num_layers_dec=args.dec_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_id=args.pad_id,
        pos_encoding=args.pos_encoding,
        norm=args.norm,
        max_len=4096,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    sched = WarmupCosineLR(opt, warmup_steps=args.warmup_steps, total_steps=args.total_steps,
                           peak_lr=args.peak_lr, final_lr=args.final_lr)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16 and device.type == "cuda"))

    refs_in_order, refs_by_index = load_refs_from_clean(args.valid_clean)

    import sentencepiece as spm
    sp_tgt = spm.SentencePieceProcessor(model_file=args.spm_tgt_model)

    best_bleu = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch.src = batch.src.to(device)
            batch.tgt_in = batch.tgt_in.to(device)
            batch.tgt_out = batch.tgt_out.to(device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(args.fp16 and device.type == "cuda")):
                logits = model(batch.src, batch.tgt_in)
                loss = label_smoothed_nll_loss(
                    logits.reshape(-1, logits.size(-1)),
                    batch.tgt_out.reshape(-1),
                    pad_id=args.pad_id,
                    eps=args.label_smoothing,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()

            lr = sched.step()
            global_step += 1
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"[epoch {epoch}] train_loss={avg_loss:.4f} step={global_step} lr={lr:.6g}")

        if epoch % args.eval_every == 0:
            model.eval()
            hyps: List[str] = []
            refs: List[str] = []
            line_ptr = 0

            for batch in valid_loader:
                src = batch.src.to(device)
                outs = model.greedy_decode(src, bos_id=args.bos_id, eos_id=args.eos_id, max_len=args.decode_max_len)

                for i in range(len(outs)):
                    hyp_ids = clean_ids_for_spm(outs[i], pad_id=args.pad_id, bos_id=args.bos_id, eos_id=args.eos_id)
                    hyp_text = sp_tgt.decode(hyp_ids)
                    hyps.append(hyp_text)

                    if batch.index is not None:
                        idx = batch.index[i]
                        refs.append(refs_by_index[idx])
                    else:
                        refs.append(refs_in_order[line_ptr])
                        line_ptr += 1

            bleu = compute_sacrebleu(hyps, refs)
            sig = bleu["signature"] if bleu.get("signature") else "N/A"
            print(f"[epoch {epoch}] valid_bleu={bleu['bleu']:.2f} | {sig}")

            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "global_step": global_step,
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

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(best_bleu)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_ids", required=True)
    ap.add_argument("--valid_ids", required=True)
    ap.add_argument("--valid_clean", required=True)
    ap.add_argument("--spm_tgt_model", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--src_vocab", type=int, default=0)
    ap.add_argument("--tgt_vocab", type=int, default=0)
    ap.add_argument("--pad_id", type=int, default=0)
    ap.add_argument("--bos_id", type=int, default=1)
    ap.add_argument("--eos_id", type=int, default=2)

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--dec_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--pos_encoding", choices=["sinusoidal", "learned", "rope"], default="sinusoidal")
    ap.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="layernorm")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--peak_lr", type=float, default=3e-4)
    ap.add_argument("--final_lr", type=float, default=3e-5)
    ap.add_argument("--warmup_steps", type=int, default=2000)
    ap.add_argument("--total_steps", type=int, default=60000)

    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)

    ap.add_argument("--decode_max_len", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=1)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # Staged ablation options
    ap.add_argument("--run_staged_ablation", action="store_true",
                    help="Run 2-stage ablation: compare pos first, then compare norm under best pos.")
    ap.add_argument("--pos_candidates", type=str, default="sinusoidal,rope",
                    help="Comma-separated pos candidates for stage1, e.g. 'sinusoidal,rope' or 'learned,rope'")
    ap.add_argument("--fixed_norm_for_pos", type=str, default="layernorm", choices=["layernorm", "rmsnorm"],
                    help="Fixed norm used in stage1 for comparing position encodings.")
    ap.add_argument("--norm_candidates", type=str, default="layernorm,rmsnorm",
                    help="Comma-separated norm candidates for stage2, e.g. 'layernorm,rmsnorm'")

    args = ap.parse_args()

    # Two-stage ablation mode
    if args.run_staged_ablation:
        os.makedirs(args.out_dir, exist_ok=True)

        pos_list = parse_csv_list(args.pos_candidates)
        norm_fixed = args.fixed_norm_for_pos
        norm_list = parse_csv_list(args.norm_candidates)

        if len(pos_list) < 2:
            raise ValueError("pos_candidates should contain at least 2 items for stage1 comparison.")
        if len(norm_list) < 2:
            raise ValueError("norm_candidates should contain at least 2 items for stage2 comparison.")

        stage1_results = []
        best_pos = None
        best_pos_bleu = -1.0

        # Stage 1: compare position encodings with fixed norm
        for pos in pos_list:
            sub_args = argparse.Namespace(**vars(args))
            sub_args.pos_encoding = pos
            sub_args.norm = norm_fixed
            sub_args.out_dir = os.path.join(args.out_dir, f"stage1_pos_{pos}_norm_{norm_fixed}")

            bleu = run_single_experiment(sub_args)
            stage1_results.append({
                "pos_encoding": pos,
                "norm": norm_fixed,
                "best_bleu": bleu,
                "out_dir": sub_args.out_dir
            })

            if bleu > best_pos_bleu:
                best_pos_bleu = bleu
                best_pos = pos

        print(f"[stage1] best_pos={best_pos} best_bleu={best_pos_bleu:.4f}")

        # Stage 2: compare norms under best_pos
        stage2_results = []
        best_norm = None
        best_norm_bleu = -1.0

        for norm in norm_list:
            sub_args = argparse.Namespace(**vars(args))
            sub_args.pos_encoding = best_pos
            sub_args.norm = norm
            sub_args.out_dir = os.path.join(args.out_dir, f"stage2_posstar_{best_pos}_norm_{norm}")

            bleu = run_single_experiment(sub_args)
            stage2_results.append({
                "pos_encoding": best_pos,
                "norm": norm,
                "best_bleu": bleu,
                "out_dir": sub_args.out_dir
            })

            if bleu > best_norm_bleu:
                best_norm_bleu = bleu
                best_norm = norm

        summary = {
            "stage1_compare_pos": stage1_results,
            "selected_pos": {
                "pos_encoding": best_pos,
                "fixed_norm": norm_fixed,
                "best_bleu": best_pos_bleu
            },
            "stage2_compare_norm": stage2_results,
            "selected_norm": {
                "pos_encoding": best_pos,
                "norm": best_norm,
                "best_bleu": best_norm_bleu
            },
            "final_choice": {
                "pos_encoding": best_pos,
                "norm": best_norm
            }
        }

        with open(os.path.join(args.out_dir, "ablation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[final] pos_encoding={best_pos} norm={best_norm} best_bleu={best_norm_bleu:.4f}")
        return

    # Default single-run mode
    best_bleu = run_single_experiment(args)
    print(f"[done] best_bleu={best_bleu:.4f}")


if __name__ == "__main__":
    main()
