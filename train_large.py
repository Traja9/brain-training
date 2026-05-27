"""
train_large.py — HPCLM on a large real-world corpus.

Usage:
  python train_large.py                          # uses Ubuntu.txt locally
  python train_large.py --file dataset.txt --chars 200000 --epochs 20
"""

import argparse
import os
import string
import time
import numpy as np

from HPCLM import HPCLM, CharTokenizer

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--file',   default='/home/my-pc/Downloads/Ubuntu.txt')
parser.add_argument('--chars',  type=int, default=50_000)
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

# ── Load & clean ──────────────────────────────────────────────────────────────
RAW  = open(args.file, encoding='utf-8', errors='ignore').read()
TEXT = ''.join(c for c in RAW if c in string.printable)[:args.chars]

os.makedirs('results', exist_ok=True)
log = open('results/training_log.txt', 'w')

def out(msg=''):
    print(msg)
    log.write(msg + '\n')
    log.flush()

out("=" * 64)
out("  HPCLM — Large Dataset Training")
out("=" * 64)
out(f"  File         : {args.file}")
out(f"  Corpus chars : {len(TEXT):,}")
out(f"  Unique chars : {len(set(TEXT))}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tok = CharTokenizer().fit(TEXT)
out(f"  Vocab size   : {tok.vocab_size}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = HPCLM(
    tok,
    embed_dim   = 64,
    l1_dim      = 256,
    l2_dim      = 512,
    l3_dim      = 1024,
    l1_sparsity = 0.05,
    l2_sparsity = 0.04,
    l3_sparsity = 0.03,
    lr          = 0.02,
    seed        = 42,
)
out(f"  Parameters   : {model.param_count():,}")
out()

# ── Training loop with cosine LR decay ────────────────────────────────────────
BASE_LR = 0.02
MIN_LR  = 0.001

ids = tok.encode(TEXT, add_bos=True, add_eos=False)
n   = len(ids)
out(f"  Tokens/epoch : {n:,}")
out(f"  Epochs       : {args.epochs}")
out()
out(f"  {'Epoch':>5}  {'LR':>7}  {'Acc':>6}  {'FE':>12}  {'tok/s':>6}  Sample")
out("  " + "-" * 78)

for epoch in range(1, args.epochs + 1):
    cos_frac = (epoch - 1) / max(args.epochs - 1, 1)
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + np.cos(np.pi * cos_frac))

    model.lr    = lr
    model.L1.lr = lr
    model.L2.lr = lr
    model.L3.lr = lr

    model.reset_context()

    correct = 0
    t0 = time.time()
    for i, tok_id in enumerate(ids):
        next_id = ids[i + 1] if i + 1 < n else -1
        result  = model.step(tok_id, next_token_id=next_id, learn=True)
        if next_id >= 0 and result['top_token'] == next_id:
            correct += 1

    elapsed  = time.time() - t0
    accuracy = correct / max(n - 1, 1)
    fe       = model._fe_history[-1] if model._fe_history else 0.0
    tps      = n / elapsed

    model.reset_context()
    sample = model.generate("The ", max_tokens=40, temperature=0.8, top_k=5, learn=False)
    sample_clean = sample.replace('\n', '↵')[:38]

    out(f"  {epoch:5d}  {lr:7.5f}  {accuracy:6.3f}  {fe:12.2f}  {tps:6.0f}  {sample_clean!r}")

out()

# ── Final generation samples ──────────────────────────────────────────────────
out("── Final Generation ─────────────────────────────────────────────────────")
prompts = ["The ", "To install ", "When the system ", "Ubuntu "]
for p in prompts:
    model.reset_context()
    out_text = model.generate(p, max_tokens=100, temperature=0.7, top_k=5, learn=False)
    out(f"\n  Prompt : {p!r}")
    out(f"  Output : {(p + out_text)!r}")

# ── Summary ───────────────────────────────────────────────────────────────────
out()
s = model.get_state()
out("── Model Summary ────────────────────────────────────────────────────────")
out(f"  Steps trained : {s['steps']:,}")
out(f"  Avg FE (100)  : {s['avg_fe']:.4f}")
out(f"  L1 precision  : {s['l1_precision']:.3f}")
out(f"  L2 precision  : {s['l2_precision']:.3f}")
out(f"  L3 precision  : {s['l3_precision']:.3f}")

log.close()
print("\n  Results saved to results/training_log.txt")
