"""
train_hybrid_v2.py — Train Hybrid_LM_v2 with mini-batch GPU processing.

Key upgrades over train_hybrid_gpu.py:
  - Trainable embeddings (Adam)
  - TP signal propagated to L1, L2, L3 (not just L1)
  - Deeper head: concat→512→256→vocab
  - Larger dims: L1=256 L2=512 L3=512
  - Longer context: 128 tokens
  - Full dataset by default

Usage:
  python train_hybrid_v2.py --file dataset.txt --epochs 100
  python train_hybrid_v2.py --file dataset.txt --chars 0 --epochs 100   # 0 = full file
"""

import argparse, os, sys, string, time
import numpy as np
import torch

from HPCLM import CharTokenizer, k_wta
from APEX_LM import AdamHebbianLevel, ContextBuffer
from Hybrid_LM_v2 import HybridLM_v2

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--file',     default=None)
parser.add_argument('--chars',    type=int,   default=0,      help='0 = full file')
parser.add_argument('--epochs',   type=int,   default=100)
parser.add_argument('--lr',       type=float, default=0.01)
parser.add_argument('--head-lr',  type=float, default=0.001,  dest='head_lr')
parser.add_argument('--embed-lr', type=float, default=0.001,  dest='embed_lr')
parser.add_argument('--l1-dim',   type=int,   default=256,    dest='l1_dim')
parser.add_argument('--l2-dim',   type=int,   default=512,    dest='l2_dim')
parser.add_argument('--l3-dim',   type=int,   default=512,    dest='l3_dim')
parser.add_argument('--head-dim', type=int,   default=512,    dest='head_dim')
parser.add_argument('--ctx-len',  type=int,   default=128,    dest='ctx_len')
parser.add_argument('--batch',    type=int,   default=128,    dest='batch_size')
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
    '/home/my-pc/Brain/dataset.txt',
]
_resolved = next((p for p in _CANDIDATES if p and os.path.isfile(p)), None)
if _resolved is None:
    print("ERROR: No dataset file found. Use --file <path>")
    sys.exit(1)
args.file = _resolved

# ── Load & clean ──────────────────────────────────────────────────────────────
RAW  = open(args.file, encoding='utf-8', errors='ignore').read()
TEXT = ''.join(c for c in RAW if c in string.printable)
if args.chars > 0:
    TEXT = TEXT[:args.chars]

SPLIT      = int(len(TEXT) * 0.9)
TRAIN_TEXT = TEXT[:SPLIT]
TEST_TEXT  = TEXT[SPLIT:]

os.makedirs('results', exist_ok=True)
_is_colab = 'google.colab' in sys.modules or os.path.exists('/content')
log = open('results/hybrid_v2_training_log.txt', 'w')

def out(msg=''):
    print(msg); log.write(msg + '\n'); log.flush()

out("=" * 70)
out("  Hybrid_LM v2 — Trainable Embeddings + TP-All + Deeper Head")
out("=" * 70)
out(f"  Device       : {DEVICE}")
out(f"  File         : {args.file}")
out(f"  Total chars  : {len(TEXT):,}  (train {len(TRAIN_TEXT):,} / test {len(TEST_TEXT):,})")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tok = CharTokenizer().fit(TEXT)
out(f"  Vocab size   : {tok.vocab_size}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = HybridLM_v2(
    tok,
    l1_dim       = args.l1_dim,
    l2_dim       = args.l2_dim,
    l3_dim       = args.l3_dim,
    head_dim     = args.head_dim,
    lr           = args.lr,
    head_lr      = args.head_lr,
    embed_lr     = args.embed_lr,
    context_len  = args.ctx_len,
)

out(f"  Params       : {model.param_count():,}")
out(f"  Dims         : L1={args.l1_dim} L2={args.l2_dim} L3={args.l3_dim} concat={model.concat_dim}")
out(f"  Head         : {model.concat_dim}→{args.head_dim}→256→{tok.vocab_size}")
out(f"  Context len  : {args.ctx_len}")
out(f"  Batch size   : {args.batch_size}")
out(f"  Epochs       : {args.epochs}")
out()

# ── Encode ────────────────────────────────────────────────────────────────────
train_ids = tok.encode(TRAIN_TEXT, add_bos=True, add_eos=False)
test_ids  = tok.encode(TEST_TEXT,  add_bos=True, add_eos=False)
out(f"  Train tokens : {len(train_ids):,}")
out(f"  Test  tokens : {len(test_ids):,}")
out()

# ── Mini-batch infrastructure ─────────────────────────────────────────────────

def hebbian_forward(tok_id):
    x0    = model.E[tok_id].copy()
    x_ctx = model.ctx.augment(x0)
    model.ctx.push(x0)
    model.L1.forward(x_ctx)
    model.L2.forward(model.L1.a)
    model.L3.forward(model.L2.a)
    concat = np.concatenate([model.L1.a, model.L2.a, model.L3.a]).astype(np.float32)
    l1s = (model.L1.a.copy(), model.L1._last_x.copy(), model.L1._last_e.copy())
    l2s = (model.L2.a.copy(), model.L2._last_x.copy(), model.L2._last_e.copy())
    l3s = (model.L3.a.copy(), model.L3._last_x.copy(), model.L3._last_e.copy())
    x0s = x0.copy()
    return concat, l1s, l2s, l3s, x0s


def apply_updates(batch_l1s, batch_l2s, batch_l3s, batch_x0s, batch_tok_ids, inp_grads):
    l1e = model.l1_dim
    l2e = l1e + model.l2_dim
    for l1s, l2s, l3s, x0, tid, inp_g in zip(
        batch_l1s, batch_l2s, batch_l3s, batch_x0s, batch_tok_ids, inp_grads
    ):
        model.L1.a = l1s[0]; model.L1._last_x = l1s[1]; model.L1._last_e = l1s[2]
        model.L2.a = l2s[0]; model.L2._last_x = l2s[1]; model.L2._last_e = l2s[2]
        model.L3.a = l3s[0]; model.L3._last_x = l3s[1]; model.L3._last_e = l3s[2]

        tp1 = k_wta(np.clip(l1s[0] - model.tp_scale * inp_g[:l1e], -1.0, 1.0), model.L1.k)
        tp2 = k_wta(np.clip(l2s[0] - model.tp_scale * inp_g[l1e:l2e], -1.0, 1.0), model.L2.k)
        tp3 = k_wta(np.clip(l3s[0] - model.tp_scale * inp_g[l2e:], -1.0, 1.0), model.L3.k)
        model.L1.update(tp1)
        model.L2.update(tp2)
        model.L3.update(tp3)

        embed_grad = (1.0 - model.ctx.blend) * (model.L1.W.T @ inp_g[:l1e])
        model._embed_update(tid, embed_grad)


def flush_batch(buf_concats, buf_targets, buf_l1s, buf_l2s, buf_l3s, buf_x0s, buf_ids, learn):
    if not buf_concats:
        return 0

    x = torch.tensor(np.stack(buf_concats), device=DEVICE)
    y = torch.tensor(buf_targets, dtype=torch.long, device=DEVICE)

    if learn:
        x.requires_grad_(True)
        model.head_opt.zero_grad()
        logits = model.head(x)
        loss   = model.ce_loss(logits, y)
        loss.backward()
        model.head_opt.step()
        inp_grads = x.grad.cpu().numpy()
        apply_updates(buf_l1s, buf_l2s, buf_l3s, buf_x0s, buf_ids, inp_grads)
    else:
        with torch.no_grad():
            logits = model.head(x)

    preds   = logits.detach().argmax(dim=1).cpu().numpy()
    correct = int(np.sum(preds == np.array(buf_targets)))
    return correct


def run_epoch(ids, learn=True):
    model.reset_context()
    n = len(ids)
    total_correct = 0
    valid_steps   = 0
    BATCH = args.batch_size

    bufs = [[], [], [], [], [], [], []]  # concats, targets, l1s, l2s, l3s, x0s, ids

    for i, tok_id in enumerate(ids):
        next_id = ids[i + 1] if i + 1 < n else -1
        if next_id < 0:
            continue

        concat, l1s, l2s, l3s, x0 = hebbian_forward(tok_id)
        for b, v in zip(bufs, [concat, next_id, l1s, l2s, l3s, x0, tok_id]):
            b.append(v)
        valid_steps += 1

        if len(bufs[0]) >= BATCH:
            total_correct += flush_batch(*bufs, learn)
            for b in bufs: b.clear()

    total_correct += flush_batch(*bufs, learn)
    return total_correct / max(valid_steps, 1)


# ── Training loop ─────────────────────────────────────────────────────────────
out(f"  {'Epoch':>5}  {'LR':>7}  {'HLR':>7}  {'ELR':>7}  {'Tr-Acc':>7}  {'Te-Acc':>7}  {'tok/s':>7}")
out("  " + "-" * 63)

BASE_LR   = args.lr;      MIN_LR   = args.lr      * 0.05
BASE_HLR  = args.head_lr; MIN_HLR  = args.head_lr * 0.05
BASE_ELR  = args.embed_lr; MIN_ELR = args.embed_lr * 0.05

for epoch in range(1, args.epochs + 1):
    cos_frac = (epoch - 1) / max(args.epochs - 1, 1)
    lr  = MIN_LR  + 0.5 * (BASE_LR  - MIN_LR)  * (1 + np.cos(np.pi * cos_frac))
    hlr = MIN_HLR + 0.5 * (BASE_HLR - MIN_HLR) * (1 + np.cos(np.pi * cos_frac))
    elr = MIN_ELR + 0.5 * (BASE_ELR - MIN_ELR) * (1 + np.cos(np.pi * cos_frac))

    model.L1.lr = model.L2.lr = model.L3.lr = lr
    model.embed_lr = elr
    for pg in model.head_opt.param_groups:
        pg['lr'] = hlr

    t0     = time.time()
    tr_acc = run_epoch(train_ids, learn=True)
    te_acc = run_epoch(test_ids,  learn=False)
    elapsed = time.time() - t0
    tps = (len(train_ids) + len(test_ids)) / elapsed

    out(f"  {epoch:5d}  {lr:7.5f}  {hlr:7.5f}  {elr:7.5f}  {tr_acc:7.3f}  {te_acc:7.3f}  {tps:7.0f}")

    # Save checkpoint every 25 epochs
    if epoch % 25 == 0:
        model.save(f'results/hybrid_v2_ep{epoch}')
        out(f"  [checkpoint saved: epoch {epoch}]")

# ── Final save + generation samples ──────────────────────────────────────────
out()
model.save('results/hybrid_v2_final')
out("  Model saved → results/hybrid_v2_final.pt + hybrid_v2_final_hebb.npz")

out()
out("── Final Generation ─────────────────────────────────────────────────────")
for p in ["The ", "It is ", "She said ", "Mr. Darcy "]:
    model.reset_context()
    generated = model.generate(p, max_tokens=120, temperature=0.8, top_k=8)
    out(f"\n  Prompt : {p!r}")
    out(f"  Output : {(p + generated)!r}")

model.summary()
log.close()
print("\n  Results → results/hybrid_v2_training_log.txt")

# ── Interactive mode (terminal only) ─────────────────────────────────────────
if sys.stdin.isatty() and not _is_colab:
    print("\n── Interactive Mode ─────────────────────────────────────────────")
    print("  Commands: :temp 0.8  :topk 8  :len 150  quit")
    i_temp, i_topk, i_len = 0.8, 8, 150
    while True:
        try:
            prompt = input("\n  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print(); break
        if not prompt: continue
        low = prompt.lower()
        if low in ('quit', 'exit', 'q'): break
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
        model.reset_context()
        generated = model.generate(prompt, max_tokens=i_len, temperature=i_temp, top_k=i_topk)
        print(f"\n  {prompt}{generated}\n")
