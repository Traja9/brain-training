"""
train_hybrid_gpu.py — Hybrid_LM with PyTorch GPU decode head.

Same architecture as Hybrid_LM but the backprop decode head runs on CUDA.
Hebbian layers (L1/L2/L3) stay on CPU/NumPy — they are sequential by nature.

Mini-batch strategy:
  1. Run Hebbian forwards sequentially, storing per-token activations.
  2. Stack concat vectors → GPU batch forward/backward through the head.
  3. Apply per-token Hebbian updates (with TP signals from GPU).

Usage:
  python train_hybrid_gpu.py --file dataset.txt --chars 100000 --epochs 40
  python train_hybrid_gpu.py --file dataset.txt --chars 100000 --epochs 40 --batch 256
"""

import argparse, os, sys, string, time
import numpy as np
import torch
import torch.nn as nn

from HPCLM import CharTokenizer, k_wta, row_normalise, kl_from_uniform
from APEX_LM import AdamHebbianLevel, ContextBuffer

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--file',     default=None)
parser.add_argument('--chars',    type=int,   default=50_000)
parser.add_argument('--epochs',   type=int,   default=40)
parser.add_argument('--lr',       type=float, default=0.01)
parser.add_argument('--head-lr',  type=float, default=0.001, dest='head_lr')
parser.add_argument('--l1-dim',   type=int,   default=64,    dest='l1_dim')
parser.add_argument('--l2-dim',   type=int,   default=128,   dest='l2_dim')
parser.add_argument('--l3-dim',   type=int,   default=128,   dest='l3_dim')
parser.add_argument('--head-dim', type=int,   default=64,    dest='head_dim')
parser.add_argument('--batch',    type=int,   default=128,   dest='batch_size')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── File resolution ───────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    args.file,
    os.path.join(_HERE, 'pride_and_prejudice.txt'),
    os.path.join(_HERE, 'dataset.txt'),
    '/kaggle/working/dataset.txt',
    '/content/dataset.txt',
    '/home/my-pc/Downloads/Ubuntu.txt',
]
_resolved = next((p for p in _CANDIDATES if p and os.path.isfile(p)), None)
if _resolved is None:
    print("ERROR: No dataset file found. Provide one with --file <path>")
    sys.exit(1)
args.file = _resolved

# ── Load & clean ──────────────────────────────────────────────────────────────
RAW   = open(args.file, encoding='utf-8', errors='ignore').read()
TEXT  = ''.join(c for c in RAW if c in string.printable)[:args.chars]
SPLIT = int(len(TEXT) * 0.9)
TRAIN_TEXT = TEXT[:SPLIT]
TEST_TEXT  = TEXT[SPLIT:]

os.makedirs('results', exist_ok=True)
log = open('results/hybrid_gpu_training_log.txt', 'w')

_is_colab = 'google.colab' in sys.modules or os.path.exists('/content')

def out(msg=''):
    print(msg); log.write(msg + '\n'); log.flush()

out("=" * 70)
out("  Hybrid_LM GPU — Hebbian CPU Hierarchy + PyTorch GPU Decode Head")
out("=" * 70)
out(f"  Device       : {DEVICE}")
out(f"  File         : {args.file}")
out(f"  Total chars  : {len(TEXT):,}  (train {len(TRAIN_TEXT):,} / test {len(TEST_TEXT):,})")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tok = CharTokenizer().fit(TEXT)
V   = tok.vocab_size
out(f"  Vocab size   : {V}")

# ── Embeddings (NumPy, CPU) ───────────────────────────────────────────────────
EMBED_DIM = 64
rng = np.random.RandomState(42)
E   = (rng.randn(V, EMBED_DIM) * 0.1).astype(np.float32)
row_normalise(E)

# ── Hebbian hierarchy (NumPy, CPU) ────────────────────────────────────────────
L1  = AdamHebbianLevel(EMBED_DIM,    args.l1_dim, 0.05, args.lr, 0.9, 0.999, seed=43)
L2  = AdamHebbianLevel(args.l1_dim,  args.l2_dim, 0.04, args.lr, 0.9, 0.999, seed=44)
L3  = AdamHebbianLevel(args.l2_dim,  args.l3_dim, 0.03, args.lr, 0.9, 0.999, seed=45)
ctx = ContextBuffer(EMBED_DIM, capacity=32, blend=0.4)
CONCAT_DIM = args.l1_dim + args.l2_dim + args.l3_dim

# ── PyTorch decode head (GPU) ─────────────────────────────────────────────────
torch.manual_seed(46)
head = nn.Sequential(
    nn.Linear(CONCAT_DIM, args.head_dim),
    nn.ReLU(),
    nn.Linear(args.head_dim, V),
).to(DEVICE)
head_opt = torch.optim.Adam(head.parameters(), lr=args.head_lr)
ce_loss  = nn.CrossEntropyLoss()

out(f"  Hebbian dims : L1={args.l1_dim} L2={args.l2_dim} L3={args.l3_dim} concat={CONCAT_DIM}")
out(f"  Head dims    : {CONCAT_DIM}→{args.head_dim}(ReLU)→{V}(Softmax)")
out(f"  Batch size   : {args.batch_size}")
out(f"  Epochs       : {args.epochs}")
out()

TP_SCALE = 0.3


def reset_context():
    L1.reset(); L2.reset(); L3.reset(); ctx.reset()


def hebbian_forward(tok_id: int):
    """Run forward pass through all Hebbian levels. Returns concat + per-level state snapshots."""
    x0    = E[tok_id].copy()
    x_ctx = ctx.augment(x0)
    ctx.push(x0)

    L1.forward(x_ctx)
    L2.forward(L1.a)
    L3.forward(L2.a)

    concat = np.concatenate([L1.a, L2.a, L3.a]).astype(np.float32)

    # Snapshot per-level state needed for deferred update
    l1s = (L1.a.copy(), L1._last_x.copy(), L1._last_e.copy())
    l2s = (L2.a.copy(), L2._last_x.copy(), L2._last_e.copy())
    l3s = (L3.a.copy(), L3._last_x.copy(), L3._last_e.copy())

    return concat, l1s, l2s, l3s


def apply_hebbian_updates(batch_l1s, batch_l2s, batch_l3s, inp_grads):
    """Apply deferred Hebbian updates for one mini-batch of tokens."""
    for l1s, l2s, l3s, inp_g in zip(batch_l1s, batch_l2s, batch_l3s, inp_grads):
        l1_a, l1_x, l1_e = l1s
        l2_a, l2_x, l2_e = l2s
        l3_a, l3_x, l3_e = l3s

        # Restore per-level state so update() uses the right snapshot
        L1.a = l1_a;  L1._last_x = l1_x;  L1._last_e = l1_e
        L2.a = l2_a;  L2._last_x = l2_x;  L2._last_e = l2_e
        L3.a = l3_a;  L3._last_x = l3_x;  L3._last_e = l3_e

        # L1 TP update: nudge activation toward lower loss direction
        tp_raw    = l1_a - TP_SCALE * inp_g[:args.l1_dim]
        tp_target = k_wta(np.clip(tp_raw, -1.0, 1.0), L1.k)
        L1.update(tp_target)
        L2.update()
        L3.update()


# ── Mini-batch flush ──────────────────────────────────────────────────────────

def flush_batch(buf_concats, buf_targets, buf_l1s, buf_l2s, buf_l3s, learn):
    if not buf_concats:
        return 0

    x = torch.tensor(np.stack(buf_concats), device=DEVICE)          # (B, CONCAT_DIM)
    y = torch.tensor(buf_targets, dtype=torch.long, device=DEVICE)  # (B,)

    if learn:
        x.requires_grad_(True)
        head_opt.zero_grad()
        logits = head(x)
        loss   = ce_loss(logits, y)
        loss.backward()
        head_opt.step()
        inp_grads = x.grad.cpu().numpy()
        apply_hebbian_updates(buf_l1s, buf_l2s, buf_l3s, inp_grads)
    else:
        with torch.no_grad():
            logits = head(x)

    preds = logits.detach().argmax(dim=1).cpu().numpy()
    correct = int(np.sum(preds == np.array(buf_targets)))
    return correct


# ── Epoch runner ──────────────────────────────────────────────────────────────

def run_epoch(ids, learn=True):
    reset_context()
    n = len(ids)
    total_correct = 0
    valid_steps   = 0

    buf_concats, buf_targets = [], []
    buf_l1s, buf_l2s, buf_l3s = [], [], []

    BATCH = args.batch_size

    for i, tok_id in enumerate(ids):
        next_id = ids[i + 1] if i + 1 < n else -1
        if next_id < 0:
            continue

        concat, l1s, l2s, l3s = hebbian_forward(tok_id)

        buf_concats.append(concat)
        buf_targets.append(next_id)
        buf_l1s.append(l1s)
        buf_l2s.append(l2s)
        buf_l3s.append(l3s)
        valid_steps += 1

        if len(buf_concats) >= BATCH:
            total_correct += flush_batch(buf_concats, buf_targets, buf_l1s, buf_l2s, buf_l3s, learn)
            buf_concats.clear(); buf_targets.clear()
            buf_l1s.clear(); buf_l2s.clear(); buf_l3s.clear()

    # Flush remainder
    total_correct += flush_batch(buf_concats, buf_targets, buf_l1s, buf_l2s, buf_l3s, learn)

    return total_correct / max(valid_steps, 1)


# ── Encode ────────────────────────────────────────────────────────────────────
train_ids = tok.encode(TRAIN_TEXT, add_bos=True, add_eos=False)
test_ids  = tok.encode(TEST_TEXT,  add_bos=True, add_eos=False)

out(f"  Train tokens : {len(train_ids):,}")
out(f"  Test  tokens : {len(test_ids):,}")
out()
out(f"  {'Epoch':>5}  {'LR':>7}  {'HLR':>7}  {'Tr-Acc':>7}  {'Te-Acc':>7}  {'tok/s':>7}")
out("  " + "-" * 55)

# ── Cosine LR decay ───────────────────────────────────────────────────────────
BASE_LR  = args.lr;      MIN_LR  = args.lr      * 0.05
BASE_HLR = args.head_lr; MIN_HLR = args.head_lr * 0.05

for epoch in range(1, args.epochs + 1):
    cos_frac = (epoch - 1) / max(args.epochs - 1, 1)
    lr  = MIN_LR  + 0.5 * (BASE_LR  - MIN_LR)  * (1 + np.cos(np.pi * cos_frac))
    hlr = MIN_HLR + 0.5 * (BASE_HLR - MIN_HLR) * (1 + np.cos(np.pi * cos_frac))

    L1.lr = L2.lr = L3.lr = lr
    for pg in head_opt.param_groups:
        pg['lr'] = hlr

    t0     = time.time()
    tr_acc = run_epoch(train_ids, learn=True)
    te_acc = run_epoch(test_ids,  learn=False)
    elapsed = time.time() - t0
    tps = (len(train_ids) + len(test_ids)) / elapsed

    out(f"  {epoch:5d}  {lr:7.5f}  {hlr:7.5f}  {tr_acc:7.3f}  {te_acc:7.3f}  {tps:7.0f}")

# ── Save ──────────────────────────────────────────────────────────────────────
out()
torch.save(head.state_dict(), 'results/hybrid_gpu_head.pt')
np.savez('results/hybrid_gpu_hebb.npz',
    E=E,
    L1_W=L1.W, L1_V=L1.V,
    L2_W=L2.W, L2_V=L2.V,
    L3_W=L3.W, L3_V=L3.V,
)
out("  Head saved → results/hybrid_gpu_head.pt")
out("  Hebb saved → results/hybrid_gpu_hebb.npz")
log.close()
print("\n  Results → results/hybrid_gpu_training_log.txt")

# ── Interactive inference (terminal only) ─────────────────────────────────────
if sys.stdin.isatty() and not _is_colab:
    print("\n── Interactive Mode ─────────────────────────────────────────────")
    print("  Enter a prompt. Commands: :temp 0.8  :topk 8  :len 150  quit")
    i_temp, i_topk, i_len = 0.8, 8, 150

    def gpu_generate(prompt, max_tokens=100, temperature=0.8, top_k=8):
        reset_context()
        ids = tok.encode(prompt, add_bos=True, add_eos=False)
        for tid in ids:
            hebbian_forward(tid)
        out_ids = []
        last_id = ids[-1]
        special = {CharTokenizer.PAD, CharTokenizer.UNK, CharTokenizer.BOS}

        for _ in range(max_tokens):
            concat, _, _, _ = hebbian_forward(last_id)
            x = torch.tensor(concat[None], device=DEVICE)
            with torch.no_grad():
                logits = head(x)[0].cpu().numpy().astype(np.float64)
            probs = np.exp(logits - logits.max())
            for sid in special:
                probs[sid] = 0.0
            probs[CharTokenizer.EOS] = 0.0
            s = probs.sum()
            probs = probs / s if s > 1e-12 else np.ones(V) / V
            logits2 = np.log(probs + 1e-12) / max(temperature, 1e-6)
            logits2 -= logits2.max()
            probs = np.exp(logits2)
            if top_k > 0:
                threshold = float(np.sort(probs)[-min(top_k, V)])
                probs = np.where(probs >= threshold, probs, 0.0)
            s = probs.sum()
            probs = probs / s if s > 1e-12 else np.ones(V) / V
            next_id = int(np.random.choice(V, p=probs))
            if next_id == CharTokenizer.EOS:
                break
            out_ids.append(next_id)
            last_id = next_id

        return tok.decode(out_ids)

    while True:
        try:
            prompt = input("\n  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print(); break
        if not prompt:
            continue
        low = prompt.lower()
        if low in ('quit', 'exit', 'q'):
            break
        if low.startswith(':temp '):
            try: i_temp = float(low.split()[1])
            except: pass
            print(f"  temperature → {i_temp}"); continue
        if low.startswith(':topk '):
            try: i_topk = int(low.split()[1])
            except: pass
            print(f"  top_k → {i_topk}"); continue
        if low.startswith(':len '):
            try: i_len = int(low.split()[1])
            except: pass
            print(f"  max_tokens → {i_len}"); continue
        generated = gpu_generate(prompt, max_tokens=i_len, temperature=i_temp, top_k=i_topk)
        print(f"\n  {prompt}{generated}")
