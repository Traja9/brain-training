"""
train_qhpclm.py — QHPCLM on a large real-world corpus.

Usage:
  python train_qhpclm.py                          # uses Ubuntu.txt locally
  python train_qhpclm.py --file dataset.txt --chars 100000 --epochs 15
"""

import argparse
import os
import string
import time
import numpy as np

from QHPCLM import QHPCLM
from HPCLM import CharTokenizer

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--file',        default='/home/my-pc/Downloads/Ubuntu.txt')
parser.add_argument('--chars',       type=int, default=50_000)
parser.add_argument('--epochs',      type=int, default=15)
parser.add_argument('--hilbert-dim', type=int, default=64, dest='hilbert_dim')
parser.add_argument('--lr',          type=float, default=0.02)
args = parser.parse_args()

# ── Load & clean ──────────────────────────────────────────────────────────────
RAW  = open(args.file, encoding='utf-8', errors='ignore').read()
TEXT = ''.join(c for c in RAW if c in string.printable)[:args.chars]

os.makedirs('results', exist_ok=True)
log = open('results/quantum_training_log.txt', 'w')


def out(msg=''):
    print(msg)
    log.write(msg + '\n')
    log.flush()


out("=" * 64)
out("  QHPCLM — Quantum Large Dataset Training")
out("  Born rule · Unitary evolution · Lindblad decoherence")
out("=" * 64)
out(f"  File         : {args.file}")
out(f"  Corpus chars : {len(TEXT):,}")
out(f"  Unique chars : {len(set(TEXT))}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tok = CharTokenizer().fit(TEXT)
out(f"  Vocab size   : {tok.vocab_size}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = QHPCLM(
    tok,
    hilbert_dim      = args.hilbert_dim,
    decoherence_rate = 0.02,
    lr               = args.lr,
    mem_capacity     = 512,
    walk_nodes       = 32,
    seed             = 42,
)
out(f"  Parameters   : {model.param_count():,}")
out(f"  Hilbert dim  : {args.hilbert_dim}")
out()

# ── Training loop with cosine LR decay ────────────────────────────────────────
BASE_LR = args.lr
MIN_LR  = args.lr * 0.05

ids = tok.encode(TEXT, add_bos=True, add_eos=False)
n   = len(ids)
out(f"  Tokens/epoch : {n:,}")
out(f"  Epochs       : {args.epochs}")
out()
out(f"  {'Epoch':>5}  {'LR':>7}  {'Q-acc':>6}  {'QFE':>7}  {'Purity':>6}  {'tok/s':>6}  Sample")
out("  " + "-" * 82)

for epoch in range(1, args.epochs + 1):
    cos_frac = (epoch - 1) / max(args.epochs - 1, 1)
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + np.cos(np.pi * cos_frac))

    model.lr = lr

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
    qfe      = model._qfe_history[-1] if model._qfe_history else 1.0
    purity   = model._pur_history[-1] if model._pur_history else 1.0
    tps      = n / elapsed

    model.reset_context()
    sample = model.generate("The ", max_tokens=40, temperature=0.9, top_k=8, learn=False)
    sample_clean = sample.replace('\n', '↵')[:35]

    out(f"  {epoch:5d}  {lr:7.5f}  {accuracy:6.3f}  {qfe:7.4f}  {purity:6.4f}  {tps:6.0f}  {sample_clean!r}")

out()

# ── Final generation samples ──────────────────────────────────────────────────
out("── Final Generation ─────────────────────────────────────────────────────")
prompts = ["The ", "To install ", "When the system ", "It is "]
for p in prompts:
    model.reset_context()
    out_text = model.generate(p, max_tokens=120, temperature=0.8, top_k=8, learn=False)
    out(f"\n  Prompt : {p!r}")
    out(f"  Output : {(p + out_text)!r}")

# ── Summary ───────────────────────────────────────────────────────────────────
out()
model.summary()

out()
out("── Quantum Walk ─────────────────────────────────────────────────────────")
model.qwalk.reset(0)
model.qwalk.step(16)
out(f"  Spreading entropy : {model.qwalk.spreading_entropy():.4f}")
out(f"  Quantum advantage : {model.qwalk.quantum_advantage(16):.3f}×")

log.close()
print("\n  Results saved to results/quantum_training_log.txt")
