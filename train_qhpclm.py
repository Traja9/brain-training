"""
train_qhpclm.py — QHPCLM on a large real-world corpus.

Usage:
  python train_qhpclm.py                          # uses Ubuntu.txt locally
  python train_qhpclm.py --file dataset.txt --chars 100000 --epochs 15 --hilbert-dim 64
"""

import argparse
import os
import sys
import string
import time
import numpy as np

from QHPCLM import QHPCLM
from HPCLM import CharTokenizer

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--file',        default=None)
parser.add_argument('--chars',       type=int, default=50_000)
parser.add_argument('--epochs',      type=int, default=15)
parser.add_argument('--hilbert-dim', type=int, default=64, dest='hilbert_dim')
parser.add_argument('--lr',          type=float, default=0.02)
args = parser.parse_args()

# ── Resolve file path (works locally, on Colab, and in GitHub Actions) ────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    args.file,
    os.path.join(_HERE, 'pride_and_prejudice.txt'),
    os.path.join(_HERE, 'dataset.txt'),
    '/content/drive/MyDrive/Brain/pride_and_prejudice.txt',
    '/content/drive/MyDrive/Brain/dataset.txt',
    '/content/dataset.txt',
    '/home/my-pc/Downloads/Ubuntu.txt',
]
_resolved = next((p for p in _CANDIDATES if p and os.path.isfile(p)), None)
if _resolved is None:
    print("ERROR: No dataset file found. Provide one with --file <path>")
    print("  Looked in:", [p for p in _CANDIDATES if p])
    sys.exit(1)
args.file = _resolved

# ── Load & clean ──────────────────────────────────────────────────────────────
RAW  = open(args.file, encoding='utf-8', errors='ignore').read()
TEXT = ''.join(c for c in RAW if c in string.printable)[:args.chars]

# 90 / 10 train-test split
SPLIT      = int(len(TEXT) * 0.9)
TRAIN_TEXT = TEXT[:SPLIT]
TEST_TEXT  = TEXT[SPLIT:]

os.makedirs('results', exist_ok=True)
log = open('results/quantum_training_log.txt', 'w')


def out(msg=''):
    print(msg)
    log.write(msg + '\n')
    log.flush()


out("=" * 70)
out("  QHPCLM — Quantum Large Dataset Training")
out("  Born rule · Unitary evolution · Lindblad decoherence")
out("=" * 70)
out(f"  File         : {args.file}")
out(f"  Total chars  : {len(TEXT):,}  (train {len(TRAIN_TEXT):,} / test {len(TEST_TEXT):,})")
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

# ── Pre-encode train and test splits ──────────────────────────────────────────
train_ids = tok.encode(TRAIN_TEXT, add_bos=True, add_eos=False)
test_ids  = tok.encode(TEST_TEXT,  add_bos=True, add_eos=False)
out(f"  Train tokens : {len(train_ids):,}")
out(f"  Test  tokens : {len(test_ids):,}")
out(f"  Epochs       : {args.epochs}")
out()
out(f"  {'Epoch':>5}  {'LR':>7}  {'Tr-Acc':>7}  {'Te-Acc':>7}  {'QFE':>7}  {'Purity':>6}  {'tok/s':>6}  Sample")
out("  " + "-" * 88)

# ── Training loop with cosine LR decay ────────────────────────────────────────
BASE_LR = args.lr
MIN_LR  = args.lr * 0.05

for epoch in range(1, args.epochs + 1):
    cos_frac = (epoch - 1) / max(args.epochs - 1, 1)
    lr       = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + np.cos(np.pi * cos_frac))
    model.lr = lr

    # ── train pass ────────────────────────────────────────────────────────────
    model.reset_context()
    correct = 0
    t0 = time.time()
    n  = len(train_ids)
    for i, tok_id in enumerate(train_ids):
        next_id = train_ids[i + 1] if i + 1 < n else -1
        result  = model.step(tok_id, next_token_id=next_id, learn=True)
        if next_id >= 0 and result['top_token'] == next_id:
            correct += 1
    elapsed = time.time() - t0
    tr_acc  = correct / max(n - 1, 1)
    qfe     = model._qfe_history[-1] if model._qfe_history else 1.0
    purity  = model._pur_history[-1] if model._pur_history else 1.0
    tps     = n / elapsed

    # ── test pass (no weight updates) ─────────────────────────────────────────
    model.reset_context()
    te_correct = 0
    nt = len(test_ids)
    for i, tok_id in enumerate(test_ids):
        next_id = test_ids[i + 1] if i + 1 < nt else -1
        result  = model.step(tok_id, next_token_id=next_id, learn=False)
        if next_id >= 0 and result['top_token'] == next_id:
            te_correct += 1
    te_acc = te_correct / max(nt - 1, 1)

    # ── sample ────────────────────────────────────────────────────────────────
    model.reset_context()
    sample = model.generate("The ", max_tokens=40, temperature=0.9, top_k=8, learn=False)
    sample_clean = sample.replace('\n', '↵')[:33]
    model.reset_context()

    out(f"  {epoch:5d}  {lr:7.5f}  {tr_acc:7.3f}  {te_acc:7.3f}  {qfe:7.4f}  {purity:6.4f}  {tps:6.0f}  {sample_clean!r}")

out()

# ── Final generation samples ──────────────────────────────────────────────────
out("── Final Generation ─────────────────────────────────────────────────────")
prompts = ["The ", "To install ", "When the system ", "It is "]
for p in prompts:
    model.reset_context()
    out_text = model.generate(p, max_tokens=120, temperature=0.8, top_k=8, learn=False)
    out(f"\n  Prompt : {p!r}")
    out(f"  Output : {(p + out_text)!r}")

# ── Summary + save ────────────────────────────────────────────────────────────
out()
model.summary()

out()
out("── Quantum Walk ─────────────────────────────────────────────────────────")
model.qwalk.reset(0)
model.qwalk.step(16)
out(f"  Spreading entropy : {model.qwalk.spreading_entropy():.4f}")
out(f"  Quantum advantage : {model.qwalk.quantum_advantage(16):.3f}×")

model.save('results/qhpclm_model')
out("  Model saved       → results/qhpclm_model.npz")

log.close()
print("\n  Results saved to results/quantum_training_log.txt")

# ── Interactive inference (terminal only) ─────────────────────────────────────
if sys.stdin.isatty():
    print("\n── Interactive Mode ─────────────────────────────────────────────────────")
    print("  Enter a prompt to generate text.")
    print("  Settings: :temp 0.8   :topk 8   :len 150   quit")
    i_temp, i_topk, i_len = 0.8, 8, 150
    while True:
        try:
            prompt = input("\n  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not prompt:
            continue
        low = prompt.lower()
        if low in ('quit', 'exit', 'q'):
            break
        if low.startswith(':temp '):
            try: i_temp = float(low.split()[1])
            except: pass
            print(f"  temperature → {i_temp}")
            continue
        if low.startswith(':topk '):
            try: i_topk = int(low.split()[1])
            except: pass
            print(f"  top_k → {i_topk}")
            continue
        if low.startswith(':len '):
            try: i_len = int(low.split()[1])
            except: pass
            print(f"  max_tokens → {i_len}")
            continue
        model.reset_context()
        generated = model.generate(prompt, max_tokens=i_len, temperature=i_temp,
                                   top_k=i_topk, learn=False)
        print(f"\n  {prompt}{generated}")
