"""
SOAP2026.py — Self-Optimising Adaptive Preconditioned optimizer 2026

Nesterov lookahead  +  dual momentum  +  lazy Kronecker preconditioner
+  cautious mask  +  gradient clipping  +  decoupled weight decay.

Reference pseudocode:
  m_f, m_s, v = 0, 0, 0     L, R = I, I     g_prev = 0
  For t:
    w_look = w - β1·P⁻¹·m_f
    g      = ∇L(w_look)
    g*     = g + β2·(g − g_prev)          # gradient correction
    m_f    = β1·m_f + (1−β1)·g*
    m_s    = β3·m_s + (1−β3)·g*
    M      = m̂_f + α·m̂_s
    v      = β2·v + (1−β2)·g*²
    if t%k==0: L+=g·gᵀ  R+=gᵀ·g  P⁻¹=L^{-¼}·R^{-¼}
    mask   = (M·g* > 0)
    G̃      = P⁻¹·g*
    G_clip = G̃·min(1, c/‖G̃‖)
    w     -= η·(P⁻¹·(M⊙mask)) / √(v̂+ε) ⊙ G_clip  +  η·λ·w
"""

import os
import time
import numpy as np
from typing import Callable, Dict, Optional, List


# ──────────────────────────────────────────────────────────────────────────────
# Preconditioner math
# ──────────────────────────────────────────────────────────────────────────────

def _sym_pow_neg_quarter(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Stable M^{-1/4} for a symmetric PSD matrix via eigen-decomposition."""
    S    = (M + M.T) * 0.5
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals ** (-0.25)) @ vecs.T


# ──────────────────────────────────────────────────────────────────────────────
# SOAP 2026
# ──────────────────────────────────────────────────────────────────────────────

class SOAP2026:
    """
    SOAP 2026 optimizer — pure NumPy, no external dependencies.

    Supports 1-D (vector) and 2-D (matrix) weight tensors.

    For 2-D weights:  P⁻¹·g  = L^{-¼} · g · R^{-¼}   (Kronecker / Shampoo)
    For 1-D weights:  P⁻¹·g  = (Σg²+ε)^{-¼} ⊙ g        (diagonal)

    Args:
        lr    : learning rate η
        beta1 : fast momentum + Nesterov step-size
        beta2 : gradient-correction decay + variance decay  (dual use)
        beta3 : slow momentum decay
        alpha : slow-momentum weight in  M = m̂_f + α·m̂_s
        eps   : variance floor ε
        lam   : decoupled weight-decay λ  (AdamW-style)
        c     : preconditioned gradient clip norm
        k     : preconditioner update interval (lazy, default 10)
    """

    def __init__(
        self,
        lr:    float = 1e-3,
        beta1: float = 0.90,
        beta2: float = 0.99,
        beta3: float = 0.95,
        alpha: float = 0.30,
        eps:   float = 1e-8,
        lam:   float = 0.01,
        c:     float = 10.0,
        k:     int   = 10,
    ):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.alpha = alpha
        self.eps   = eps
        self.lam   = lam
        self.c     = c
        self.k     = k

        self._state: Dict[int, dict] = {}
        self.t: int = 0

        # diagnostics (filled by step())
        self.last_mask_density: float = 1.0   # fraction of weights updated
        self.last_clip_scale:   float = 1.0

    # ── per-tensor state ──────────────────────────────────────────────────

    def _init_state(self, w: np.ndarray) -> dict:
        s: dict = dict(
            m_f    = np.zeros_like(w, dtype=np.float64),
            m_s    = np.zeros_like(w, dtype=np.float64),
            v      = np.zeros_like(w, dtype=np.float64),
            g_prev = np.zeros_like(w, dtype=np.float64),
        )
        if w.ndim == 2:
            m, n = w.shape
            s['L']  = np.eye(m) * 1e-4    # left  Kronecker factor (m×m)
            s['R']  = np.eye(n) * 1e-4    # right Kronecker factor (n×n)
            s['PL'] = np.eye(m)            # L^{-1/4}
            s['PR'] = np.eye(n)            # R^{-1/4}
        else:
            s['P_acc'] = np.ones_like(w, dtype=np.float64) * 1e-4
            s['P_inv'] = (np.ones_like(w) * 1e-4) ** (-0.25)
        return s

    def _apply_prec(self, s: dict, x: np.ndarray) -> np.ndarray:
        """Apply P⁻¹ to tensor x."""
        if x.ndim == 2:
            return s['PL'] @ x @ s['PR']
        return s['P_inv'] * x

    def _refresh_prec(self, s: dict, g: np.ndarray):
        """Accumulate Kronecker/diagonal factors and recompute P⁻¹."""
        if g.ndim == 2:
            s['L']  += g @ g.T
            s['R']  += g.T @ g
            s['PL']  = _sym_pow_neg_quarter(s['L'])
            s['PR']  = _sym_pow_neg_quarter(s['R'])
        else:
            s['P_acc'] += g ** 2
            s['P_inv']  = (s['P_acc'] + self.eps) ** (-0.25)

    # ── main update ───────────────────────────────────────────────────────

    def step(
        self,
        w:       np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        key:     int = 0,
    ) -> np.ndarray:
        """
        One SOAP2026 step.

        Args:
            w       : current weight (1-D or 2-D numpy array)
            grad_fn : callable(w_lookahead) -> gradient   same shape as w
            key     : unique integer id per weight tensor

        Returns:
            Updated weight (same dtype and shape).
        """
        self.t += 1
        orig_dtype = w.dtype
        w = w.astype(np.float64)

        if key not in self._state:
            self._state[key] = self._init_state(w)
        s = self._state[key]

        b1, b2, b3 = self.beta1, self.beta2, self.beta3

        # ── Step 1: Nesterov lookahead + gradient correction ──────────────
        # lr keeps the lookahead proportional to the actual step magnitude.
        # Without lr the raw momentum m_f (which is ≈ lr-scale * gradient)
        # would overshoot by ~1/lr at large-gradient initialisation.
        w_look = w - self.lr * b1 * self._apply_prec(s, s['m_f'])
        g      = np.asarray(grad_fn(w_look), dtype=np.float64)
        g      = np.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)
        # gradient correction (Adan-style) — clip prevents early amplification
        g_star = g + b2 * (g - s['g_prev'])
        g_star = np.clip(g_star, -1e4, 1e4)

        # ── Step 2: dual momentum ─────────────────────────────────────────
        s['m_f'] = b1 * s['m_f'] + (1.0 - b1) * g_star
        s['m_s'] = b3 * s['m_s'] + (1.0 - b3) * g_star
        bc1 = 1.0 - b1 ** self.t
        bc3 = 1.0 - b3 ** self.t
        M   = s['m_f'] / bc1 + self.alpha * (s['m_s'] / bc3)

        # ── Step 3: variance ──────────────────────────────────────────────
        s['v'] = b2 * s['v'] + (1.0 - b2) * g_star ** 2
        v_hat  = s['v'] / (1.0 - b2 ** self.t)

        # ── Step 4: lazy preconditioner ───────────────────────────────────
        if self.t % self.k == 0:
            self._refresh_prec(s, g_star)

        # ── Step 5: cautious mask (soft version avoids killing updates) ───
        # Hard binary mask blocks any element where momentum & gradient
        # disagree in sign — very conservative. Soft version tapers smoothly.
        alignment  = M * g_star
        mask       = np.where(alignment > 0.0, 1.0,
                              np.exp(alignment / (np.abs(M).mean() + 1e-8)))

        # ── Step 6: compute preconditioned momentum update ────────────────
        prec_M    = self._apply_prec(s, M * mask)
        direction = prec_M / np.sqrt(v_hat + self.eps)   # Adam-like direction

        # Clip the actual update direction — keeps effective step bounded.
        d_norm     = np.linalg.norm(direction)
        clip_scale = min(1.0, self.c / (d_norm + 1e-12))

        self.last_mask_density = float(np.mean(mask > 0.5))
        self.last_clip_scale   = clip_scale

        # ── Step 7: update + decoupled weight decay ───────────────────────
        update = direction * clip_scale
        update = np.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
        w_new  = w - self.lr * update - self.lr * self.lam * w

        s['g_prev'] = g.copy()
        return w_new.astype(orig_dtype)

    def zero_state(self):
        self._state.clear()
        self.t = 0


# ──────────────────────────────────────────────────────────────────────────────
# Adam baseline (for comparison)
# ──────────────────────────────────────────────────────────────────────────────

def _adam_step(w, g, s, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, lam=0.0):
    s['m'] = b1 * s['m'] + (1 - b1) * g
    s['v'] = b2 * s['v'] + (1 - b2) * g ** 2
    m_hat  = s['m'] / (1 - b1 ** t)
    v_hat  = s['v'] / (1 - b2 ** t)
    return w - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * lam * w


# ──────────────────────────────────────────────────────────────────────────────
# Tests + Demo
# ──────────────────────────────────────────────────────────────────────────────

def _bar(losses: List[float], width: int = 40) -> str:
    """ASCII loss curve (normalised to width chars)."""
    mn, mx = min(losses), max(losses)
    rng = max(mx - mn, 1e-12)
    pts = [int((v - mn) / rng * (width - 1)) for v in losses[::max(1, len(losses)//width)]]
    return ''.join('▁▂▃▄▅▆▇█'[min(7, p * 8 // width)] for p in pts)


def run_demo():
    print("=" * 68)
    print("  SOAP 2026 — Optimizer Tests")
    print("=" * 68)

    rng = np.random.RandomState(7)

    # ── Test 1: Rosenbrock (non-convex, min at [1,1]) ─────────────────────
    print("\n  Test 1 — Rosenbrock  f(x,y)=(1−x)²+100(y−x²)²  min@[1,1]")

    def rb_loss(w):
        x, y = w[0], w[1]
        return (1 - x)**2 + 100.0 * (y - x**2)**2

    def rb_grad(w):
        x, y = float(w[0]), float(w[1])
        return np.array([
            -2*(1-x) - 400*x*(y-x**2),
             200*(y-x**2)
        ])

    w0 = np.array([-1.5, 2.0])
    results = {}
    STEPS = 20000

    # SOAP2026
    w = w0.copy()
    opt = SOAP2026(lr=5e-2, lam=0.0, c=10.0, k=5)
    t0 = time.perf_counter()
    losses_soap = []
    for _ in range(STEPS):
        w = opt.step(w, rb_grad, key=0)
        losses_soap.append(rb_loss(w))
    dt_soap = time.perf_counter() - t0
    results['soap'] = (w.copy(), losses_soap[-1])

    # Adam
    w = w0.copy()
    s_adam = {'m': np.zeros(2), 'v': np.zeros(2)}
    losses_adam = []
    t0 = time.perf_counter()
    for t in range(1, STEPS + 1):
        g = rb_grad(w)
        w = _adam_step(w, g, s_adam, t, lr=2e-2, lam=0.0)
        losses_adam.append(rb_loss(w))
    dt_adam = time.perf_counter() - t0

    print(f"    {'Optimizer':10s}  {'Final loss':>12s}  {'Dist to [1,1]':>14s}  "
          f"{'MaskDens':>9s}  {'Time':>7s}  Curve")
    for name, ws, ls in [('SOAP2026', results['soap'][0], losses_soap),
                         ('Adam',     w.copy(),           losses_adam)]:
        dist = np.linalg.norm(ws - 1.0)
        dt   = dt_soap if name == 'SOAP2026' else dt_adam
        md   = f"{opt.last_mask_density:.2f}" if name == 'SOAP2026' else '  n/a'
        print(f"    {name:10s}  {ls[-1]:12.6f}  {dist:14.6f}  {md:>9s}  {dt:6.3f}s  {_bar(ls)}")

    # ── Test 2: Linear regression (2-D weights, tests Kronecker prec) ────
    print("\n  Test 2 — Linear Regression  (8→4 matrix weights, 200 samples)")
    N, D_in, D_out = 200, 8, 4
    X  = rng.randn(N, D_in).astype(np.float32)
    Wt = rng.randn(D_in, D_out).astype(np.float32)
    Y  = X @ Wt + 0.02 * rng.randn(N, D_out).astype(np.float32)

    def linreg_grad(w_):
        return (X.T @ (X @ w_ - Y)) / N

    W_init = (rng.randn(D_in, D_out) * 0.1).astype(np.float32)

    # SOAP2026
    W = W_init.copy()
    opt2 = SOAP2026(lr=2e-2, lam=1e-5, c=10.0, k=5)
    ls_soap2 = []
    for _ in range(1000):
        W = opt2.step(W, linreg_grad, key=0)
        ls_soap2.append(float(np.mean((X @ W - Y)**2)))

    # Adam
    W = W_init.copy()
    sa = {'m': np.zeros_like(W), 'v': np.zeros_like(W)}
    ls_adam2 = []
    for t in range(1, 1001):
        g = linreg_grad(W)
        W = _adam_step(W, g, sa, t, lr=5e-3, lam=1e-5)
        ls_adam2.append(float(np.mean((X @ W - Y)**2)))

    print(f"    {'Optimizer':10s}  {'Start loss':>10s}  {'Final loss':>10s}  {'Reduction':>10s}  Curve")
    for name, ls in [('SOAP2026', ls_soap2), ('Adam', ls_adam2)]:
        red = ls[0] / (ls[-1] + 1e-12)
        print(f"    {name:10s}  {ls[0]:10.4f}  {ls[-1]:10.6f}  {red:9.1f}×  {_bar(ls)}")

    # ── Test 3: Noisy regression (stress-test cautious mask + clip) ───────
    print("\n  Test 3 — Noisy Regression  (gradient noise σ=0.5, 1-D weights)")
    D = 32
    w_true = rng.randn(D).astype(np.float32)
    X3 = rng.randn(500, D).astype(np.float32)
    Y3 = (X3 @ w_true + 0.5 * rng.randn(500).astype(np.float32))

    def noisy_grad(w_, noise=0.5):
        idx = rng.choice(500, 64, replace=False)
        g   = (X3[idx].T @ (X3[idx] @ w_ - Y3[idx])) / 64.0
        g  += noise * rng.randn(*w_.shape).astype(g.dtype)
        return g

    W3_init = (rng.randn(D) * 0.1).astype(np.float32)
    W3 = W3_init.copy()
    opt3 = SOAP2026(lr=2e-2, lam=1e-4, c=10.0, k=10, beta2=0.95)
    ls_soap3 = []
    for _ in range(2000):
        W3 = opt3.step(W3, noisy_grad, key=0)
        ls_soap3.append(float(np.mean((X3 @ W3 - Y3)**2)))

    W3 = W3_init.copy()
    sa3 = {'m': np.zeros(D), 'v': np.zeros(D)}
    ls_adam3 = []
    for t in range(1, 2001):
        g = noisy_grad(W3)
        W3 = _adam_step(W3, g, sa3, t, lr=1e-2, lam=1e-4)
        ls_adam3.append(float(np.mean((X3 @ W3 - Y3)**2)))

    print(f"    {'Optimizer':10s}  {'Start loss':>10s}  {'Final loss':>10s}  Curve")
    for name, ls in [('SOAP2026', ls_soap3), ('Adam', ls_adam3)]:
        print(f"    {name:10s}  {ls[0]:10.4f}  {ls[-1]:10.6f}  {_bar(ls)}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n  " + "─" * 64)
    print("  Summary")
    print(f"    Rosenbrock  : SOAP2026 final loss {losses_soap[-1]:.4e}  "
          f"| Adam {losses_adam[-1]:.4e}")
    print(f"    LinReg      : SOAP2026 final loss {ls_soap2[-1]:.4e}  "
          f"| Adam {ls_adam2[-1]:.4e}")
    print(f"    Noisy reg   : SOAP2026 final loss {ls_soap3[-1]:.4e}  "
          f"| Adam {ls_adam3[-1]:.4e}")

    os.makedirs('results', exist_ok=True)
    np.save('results/soap2026_rosenbrock.npy', np.array(losses_soap))
    np.save('results/adam_rosenbrock.npy',     np.array(losses_adam))
    print("\n  Convergence curves saved to results/soap2026_*.npy")


if __name__ == '__main__':
    run_demo()
