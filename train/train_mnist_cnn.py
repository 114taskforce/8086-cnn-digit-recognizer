# -*- coding: utf-8 -*-
"""
LeNet-style CNN on binarized 28x28 MNIST, quantized for the 8086 asm program.

Training data (user spec):
  - 500 MNIST train images (50 per class, stratified)
  - each image binarized with TWO thresholds (0.25 / 0.50) -> 1000 samples
  - another 500 train images x 2 thresholds -> 1000 validation (early stop)
  - full 10k test set x 2 thresholds -> 20k test samples

Architecture (28x28 binary input):
  C1: conv 6@5x5 valid  -> 24x24x6  ReLU   (binary input: multiply-free)
  S2: avgpool 2x2       -> 12x12x6         (sum >> 2)
  C3: conv 16@5x5 valid -> 8x8x16   ReLU   (6 input channels, full)
  S4: avgpool 2x2       -> 4x4x16          (sum >> 2)
  F5: flatten 256 -> dense 10, argmax (first max wins)

Fixed-point inference (bit-exact with the asm):
  acc1(int16) = b1 + sum(w1 * bit)          per output, 25 taps
  h1 = clamp(acc1 >> SH1, 0, 127)           int8
  h2 = (h1 4 taps) >> 2                     avgpool
  acc2(int32) = sext(b2) + sum(w2 * h2)     150 taps, int8*int8
  h3 = clamp(acc2 >> SH3, 0, 127)
  h4 = (h3 4 taps) >> 2
  acc3(int32) = sext(b3) + sum(w3 * h4)     256 taps
  out = acc3 >> SH4 -> argmax (first max wins)

Weight file layout (net_weights.bin):
  C1: 6  x [bias i16le][25 x i8le]   = 162 B
  C3: 16 x [bias i16le][150 x i8le]  = 2432 B
  F5: 10 x [bias i16le][256 x i8le]  = 2580 B
Tap order: C1 (dr,dc); C3 (cin,dr,dc); F5 row order (cin,r,c) channel-major.
"""
import os
import struct
import numpy as np

# Optional GPU backend: pip install cupy-cuda12x (RTX 40xx). All tensor math
# goes through xp; RNG / data preprocessing / file export stay on the CPU.
try:
    import cupy as cp
    xp = cp
    _to_np = cp.asnumpy
except ImportError:
    xp = np
    _to_np = np.asarray

from train_mnist_net import load_images, load_labels

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model output")   # final model data: weights + shifts + test sets
DEMO_DIR = os.path.join(BASE, "test demo")       # 28x28 text demos consumed by the DOS program
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

THRS = (0.25, 0.50)     # two binarization thresholds per image
N_PER_CLASS = 500       # 500/class * 10 = 5000 train images
N_VAL_CLASS = 250       # 250/class * 10 = 2500 val images
EPOCHS = 80
PATIENCE = 15
WD = 1e-4
LR = 2e-3
BATCH = 64
SEED = 42
N_TEST_REF = 200        # stratified reference test images written as files

HID1 = 6                # C1 output channels
HID3 = 16               # C3 output channels


# ---------------------------------------------------------------- data ----
def bin28(imgs, thr):
    """(N,28,28) uint8 -> (N,784) float32 {0,1} at given threshold"""
    return (imgs.astype(np.float32) / 255.0 > thr).astype(np.float32).reshape(-1, 784)


def make_dataset(imgs, ys, idx):
    """500 images x 2 thresholds -> (1000, 784) x, (1000,) y"""
    xs = [bin28(imgs[idx], t) for t in THRS]
    return np.concatenate(xs, 0), np.concatenate([ys[idx]] * len(THRS))


def morph(xbits, op):
    """3x3 square dilate('d')/erode('e') on binary 28x28 -> stroke-width variant"""
    g = xbits.reshape(-1, 28, 28) > 0.5
    p = np.pad(g, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros(g.shape, bool) if op == "d" else np.ones(g.shape, bool)
    fn = np.logical_or if op == "d" else np.logical_and
    for dr in range(3):
        for dc in range(3):
            out = fn(out, p[:, dr:dr + 28, dc:dc + 28])
    return out.reshape(-1, 784).astype(np.float32)


def make_train_dataset(imgs, ys, idx):
    """500 imgs x 2 thr x 3 stroke variants (orig/dilate/erode) = 30000 samples"""
    xs, ys_ = [], []
    for t in THRS:
        x0 = bin28(imgs[idx], t)
        xs += [x0, morph(x0, "d"), morph(x0, "e")]
        ys_ += [ys[idx]] * 3
    return np.concatenate(xs, 0), np.concatenate(ys_)


# ------------------------------------------------------------- float NN ----
def im2col_f(x, k):
    """x (N,C,H,W) float -> (N,H-k+1,W-k+1,C*k*k), tap order (cin,dr,dc)"""
    N, C, H, W = x.shape
    Ho, Wo = H - k + 1, W - k + 1
    cols = xp.empty((N, Ho, Wo, C * k * k), np.float32)
    i = 0
    for ci in range(C):
        for dr in range(k):
            for dc in range(k):
                cols[:, :, :, i] = x[:, ci, dr:dr + Ho, dc:dc + Wo]
                i += 1
    return cols


def col2im_f(cols, xshape, k):
    """transpose of im2col_f (overlap-add)"""
    N, C, H, W = xshape
    Ho, Wo = H - k + 1, W - k + 1
    gx = xp.zeros(xshape, np.float32)
    i = 0
    for ci in range(C):
        for dr in range(k):
            for dc in range(k):
                gx[:, ci, dr:dr + Ho, dc:dc + Wo] += cols[:, :, :, i]
                i += 1
    return gx


def pool2(x):
    """(N,C,H,W) -> (N,C,H/2,W/2), 2x2 mean"""
    N, C, H, W = x.shape
    return x.reshape(N, C, H // 2, 2, W // 2, 2).mean(axis=(3, 5))


def pool2_grad(dout):
    """grad of 2x2 mean pool"""
    return xp.repeat(xp.repeat(dout, 2, axis=2), 2, axis=3) / 4.0


class CNN:
    """C1(6) - S2 - C3(16) - S4 - F5(10); layout aligned with the asm"""

    def __init__(self, seed=SEED):
        r = np.random.RandomState(seed)      # RNG on CPU: identical init either way
        self.W1 = xp.asarray(r.randn(HID1, 25) * np.sqrt(2.0 / 25), dtype=np.float32)
        self.b1 = xp.zeros(HID1, dtype=np.float32)
        self.W2 = xp.asarray(r.randn(HID3, 6 * 25) * np.sqrt(2.0 / (6 * 25)), dtype=np.float32)
        self.b2 = xp.zeros(HID3, dtype=np.float32)
        self.W3 = xp.asarray(r.randn(10, 256) * np.sqrt(2.0 / 256), dtype=np.float32)
        self.b3 = xp.zeros(10, dtype=np.float32)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, ximg, cache=None):
        """ximg (N,1,28,28) float {0,1}; returns logits (N,10)"""
        n = ximg.shape[0]
        ximg = xp.asarray(ximg)
        cols1 = im2col_f(ximg, 5)
        c1 = cols1 @ self.W1.T + self.b1                # (N,24,24,6)
        a1 = xp.maximum(c1, 0)
        s2 = pool2(a1.transpose(0, 3, 1, 2)).transpose(0, 2, 3, 1)
        cols2 = im2col_f(s2.transpose(0, 3, 1, 2), 5)   # s2 is NHWC -> NCHW
        c3 = cols2 @ self.W2.T + self.b2                # (N,8,8,16)
        a3 = xp.maximum(c3, 0)
        s4 = pool2(a3.transpose(0, 3, 1, 2)).transpose(0, 2, 3, 1)
        flat = s4.transpose(0, 3, 1, 2).reshape(n, 256)  # channel-major
        logits = flat @ self.W3.T + self.b3
        if cache is not None:
            cache.update(cols1=cols1, cols2=cols2, a1=a1, a3=a3, flat=flat)
        return logits

    def backward(self, ximg, y, cache):
        n = len(y)
        logits = self.forward(ximg, cache)
        z = logits - logits.max(1, keepdims=True)
        p = xp.exp(z); p /= p.sum(1, keepdims=True)
        yoh = xp.zeros_like(p); yoh[xp.arange(n), y] = 1
        g = (p - yoh) / n
        dW3 = g.T @ cache["flat"]
        db3 = g.sum(0)
        dflat = g @ self.W3
        ds4 = dflat.reshape(n, HID3, 4, 4).transpose(0, 2, 3, 1)
        da3 = pool2_grad(ds4.transpose(0, 3, 1, 2)).transpose(0, 2, 3, 1)
        dc3 = da3 * (cache["a3"] > 0)
        dW2 = xp.tensordot(cache["cols2"], dc3, axes=([0, 1, 2], [0, 1, 2])).T
        db2 = dc3.sum((0, 1, 2))
        ds2 = col2im_f(dc3 @ self.W2, (n, HID1, 12, 12), 5).transpose(0, 2, 3, 1)
        da1 = pool2_grad(ds2.transpose(0, 3, 1, 2)).transpose(0, 2, 3, 1)
        dc1 = da1 * (cache["a1"] > 0)
        dW1 = xp.tensordot(cache["cols1"], dc1, axes=([0, 1, 2], [0, 1, 2])).T
        db1 = dc1.sum((0, 1, 2))
        return [dW1, db1, dW2, db2, dW3, db3]

    def predict(self, ximg, chunk=2000):
        if ximg.ndim == 2:
            ximg = ximg.reshape(-1, 1, 28, 28)
        out = xp.empty(len(ximg), np.int64)
        for s in range(0, len(ximg), chunk):
            out[s:s + chunk] = self.forward(ximg[s:s + chunk]).argmax(1)
        return out


def train_cnn(xtr, ytr, xv, yv):
    """mini-batch AdamW + cosine annealing + early stopping"""
    net = CNN()
    m = [xp.zeros_like(p) for p in net.params()]
    v = [xp.zeros_like(p) for p in net.params()]
    b1d, b2d, eps = 0.9, 0.999, 1e-8
    ximg = xtr.reshape(-1, 1, 28, 28)
    ximgv = xv.reshape(-1, 1, 28, 28)
    step, best_va, best_w, bad = 0, -1.0, None, 0
    for ep in range(EPOCHS):
        lr = float(0.5 * LR * (1.0 + np.cos(np.pi * ep / EPOCHS)))   # python float: never mix numpy scalars into xp ops
        perm = np.random.RandomState(SEED + ep).permutation(len(ximg))
        for i in range(0, len(ximg), BATCH):
            idx = xp.asarray(perm[i:i + BATCH])   # indices to xp: cupy rejects numpy operands
            cache = {}
            grads = net.backward(ximg[idx], ytr[idx], cache)
            step += 1
            for pm, g, mm, vv in zip(net.params(), grads, m, v):
                mm *= b1d; mm += (1 - b1d) * g
                vv *= b2d; vv += (1 - b2d) * g * g
                pm -= lr * (mm / (1 - b1d ** step)) / (xp.sqrt(vv / (1 - b2d ** step)) + eps)
                pm -= lr * WD * pm
        va = float((net.predict(ximgv) == yv).mean())
        marker = ""
        if va > best_va:
            best_va, best_w, bad = va, [p.copy() for p in net.params()], 0
            marker = " *"
        else:
            bad += 1
        if ep % 5 == 0 or marker:
            print(f"  epoch {ep+1}: val_acc={va:.4f}{marker} (lr={lr:.5f})")
        if bad >= PATIENCE:
            print(f"  early stop at epoch {ep+1}")
            break
    for p, bw in zip(net.params(), best_w):
        p[...] = bw
    return net, best_va


# --------------------------------------------------------- quantization ----
def im2col_i(x, k):
    N, C, H, W = x.shape
    Ho, Wo = H - k + 1, W - k + 1
    cols = xp.empty((N, Ho, Wo, C * k * k), np.int64)
    i = 0
    for ci in range(C):
        for dr in range(k):
            for dc in range(k):
                cols[:, :, :, i] = x[:, ci, dr:dr + Ho, dc:dc + Wo]
                i += 1
    return cols


def quant_params(net):
    """per-tensor symmetric int8 weights (scale shared by the bias)"""
    qs = []
    for W, b in ((net.W1, net.b1), (net.W2, net.b2), (net.W3, net.b3)):
        k = 127.0 / xp.abs(W).max()
        wq = xp.clip(xp.round(W * k), -127, 127).astype(np.int64)
        bq = xp.clip(xp.round(b * k), -32768, 32767).astype(np.int64)
        qs += [wq, bq]
    return qs


def cnn_forward_int(xbits, qp, SH1, SH3, SH4, chunk=1000, collect=False):
    """bit-exact simulation of the 8086 integer pipeline

    collect=True also returns per-layer max |acc| (SH-independent for C1,
    C3 acc depends on SH1, F5 acc on SH1+SH3) for shift estimation."""
    w1q, b1q, w2q, b2q, w3q, b3q = qp
    preds = []
    mx1 = mx2 = mx3 = 0
    for s in range(0, len(xbits), chunk):
        xb = xp.asarray(xbits[s:s + chunk]).reshape(-1, 1, 28, 28)
        acc1 = im2col_i(xb, 5) @ w1q.T + b1q                 # (n,24,24,6) int64
        assert acc1.min() >= -32768 and acc1.max() <= 32767, "C1 int16 overflow"
        mx1 = max(mx1, int(max(abs(acc1.min()), acc1.max())))
        h1 = xp.clip(acc1 >> SH1, 0, 127)
        h2 = (h1[:, 0::2, 0::2, :] + h1[:, 0::2, 1::2, :] +
              h1[:, 1::2, 0::2, :] + h1[:, 1::2, 1::2, :]) >> 2
        acc2 = im2col_i(h2.transpose(0, 3, 1, 2), 5) @ w2q.T + b2q   # h2 NHWC -> NCHW
        assert acc2.min() >= -(2**31) and acc2.max() <= 2**31 - 1, "C3 int32 overflow"
        mx2 = max(mx2, int(max(abs(acc2.min()), acc2.max())))
        h3 = xp.clip(acc2 >> SH3, 0, 127)
        h4 = (h3[:, 0::2, 0::2, :] + h3[:, 0::2, 1::2, :] +
              h3[:, 1::2, 0::2, :] + h3[:, 1::2, 1::2, :]) >> 2
        flat = h4.transpose(0, 3, 1, 2).reshape(len(xb), 256)
        acc3 = flat @ w3q.T + b3q
        assert acc3.min() >= -(2**31) and acc3.max() <= 2**31 - 1, "F5 int32 overflow"
        mx3 = max(mx3, int(max(abs(acc3.min()), acc3.max())))
        preds.append((acc3 >> SH4).argmax(1))                # first max wins
    if collect:
        return xp.concatenate(preds), (mx1, mx2, mx3)
    return xp.concatenate(preds)


def pack98(x14bin):
    """(N,28,28) {0,1} -> (N,98) uint8, row-major MSB first (bit-exact loop)"""
    n = x14bin.shape[0]
    bits = x14bin.reshape(n, 784) > 0.5
    out = np.zeros((n, 98), np.uint8)
    for i in range(784):
        if bits[:, i].any():
            out[:, i >> 3] |= bits[:, i].astype(np.uint8) << (7 - (i & 7))
    return out


# ----------------------------------------------------------------- main ----
CACHE = os.path.join(OUT, "cnn_model.npz")


def get_trained_net(xtr, ytr, xv, yv):
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        net = CNN()
        net.W1, net.b1 = xp.asarray(d["W1"]), xp.asarray(d["b1"])
        net.W2, net.b2 = xp.asarray(d["W2"]), xp.asarray(d["b2"])
        net.W3, net.b3 = xp.asarray(d["W3"]), xp.asarray(d["b3"])
        va = float((net.predict(xv) == yv).mean())
        print(f"loaded cached model, val_acc={va:.4f}")
        return net, va
    net, best_va = train_cnn(xtr, ytr, xv, yv)
    np.savez(CACHE, W1=_to_np(net.W1), b1=_to_np(net.b1),
                 W2=_to_np(net.W2), b2=_to_np(net.b2), W3=_to_np(net.W3), b3=_to_np(net.b3))
    return net, best_va


def main():
    print("== load MNIST ==")
    tr_x = load_images("train-images-idx3-ubyte.gz")
    tr_y = load_labels("train-labels-idx1-ubyte.gz")
    te_x = load_images("t10k-images-idx3-ubyte.gz")
    te_y = load_labels("t10k-labels-idx1-ubyte.gz")

    # stratified 500 train + 250 val images per class
    rng = np.random.RandomState(SEED)
    tr_ids, va_ids = [], []
    for c in range(10):
        ids = np.nonzero(tr_y == c)[0]
        ids = ids[rng.permutation(len(ids))]
        tr_ids.extend(ids[:N_PER_CLASS].tolist())
        va_ids.extend(ids[N_PER_CLASS:N_PER_CLASS + N_VAL_CLASS].tolist())
    tr_ids, va_ids = np.array(tr_ids), np.array(va_ids)
    xtr, ytr = make_train_dataset(tr_x, tr_y, tr_ids)
    xv, yv = make_dataset(tr_x, tr_y, va_ids)
    xte = np.concatenate([bin28(te_x, t) for t in THRS])
    yte = np.concatenate([te_y] * len(THRS))
    print(f"train {xtr.shape} (500 imgs x {len(THRS)} thr x 3 stroke variants), "
          f"val {xv.shape}, test {xte.shape} (10k x {len(THRS)} thr)")
    print(f"backend: {'cupy (GPU)' if xp is not np else 'numpy (CPU)'}")
    xtr, xv, xte = xp.asarray(xtr), xp.asarray(xv), xp.asarray(xte)   # tensors to GPU once
    ytr, yv, yte = xp.asarray(ytr), xp.asarray(yv), xp.asarray(yte)   # labels too: cupy rejects numpy operands

    print("== training CNN (AdamW + cosine + early stop) ==")
    net, best_va = get_trained_net(xtr, ytr, xv, yv)
    print(f"best val_acc={best_va:.4f}")

    for tag, x, y in (("train", xtr, ytr), ("val", xv, yv),
                      ("test", xte, yte)):
        acc = float((net.predict(x) == y).mean())
        print(f"float {tag}: {acc:.4f}")
    for t, thr in enumerate(THRS):
        acc = float((net.predict(xte[t * 10000:(t + 1) * 10000]) == yte[t * 10000:(t + 1) * 10000]).mean())
        print(f"float test thr={thr}: {acc:.4f}")

    # ---- quantization: data-driven shift estimation + small search ----
    qp = quant_params(net)
    w1q, b1q, w2q, b2q, w3q, b3q = qp
    xprobe, yprobe = xv[:1000], yv[:1000]
    print("== SH estimation + search (1000-sample probe) ==")

    # C1 acc range is SH-independent -> estimate SH1
    _, (mx1, _, _) = cnn_forward_int(xprobe, qp, 1, 1, 1, collect=True)
    sh1_est = max(1, int(round(np.log2(mx1 / 127.0))))
    sh1s = sorted({max(1, sh1_est + d) for d in (-1, 0, 1)})

    # C3 acc range depends on SH1 -> estimate SH3 per SH1
    plan = {}
    for s1 in sh1s:
        _, (_, mx2, _) = cnn_forward_int(xprobe, qp, s1, 1, 1, collect=True)
        sh3_est = max(1, int(round(np.log2(mx2 / 127.0))))
        for s3 in sorted({max(1, sh3_est + d) for d in (-1, 0, 1)}):
            # F5 acc range depends on SH1+SH3 -> estimate SH4 per pair
            _, (_, _, mx3) = cnn_forward_int(xprobe, qp, s1, s3, 1, collect=True)
            sh4_est = max(0, int(np.ceil(np.log2(max(mx3, 1) / 32767.0))))
            for s4 in sorted({max(0, sh4_est + d) for d in (-1, 0, 1)}):
                plan[(s1, s3, s4)] = None
    results = {}
    for key in plan:
        pred = cnn_forward_int(xprobe, qp, *key)
        results[key] = (pred == yprobe).mean()
    ranked = sorted(results, key=results.get, reverse=True)
    print("top5:", [(k, round(float(results[k]), 4)) for k in ranked[:5]])

    print("== fine evaluation on full 20k test ==")
    best = None
    for key in ranked[:3]:
        pred = cnn_forward_int(xte, qp, *key)
        acc = float((pred == yte).mean())
        print(f"SH1,SH3,SH4={key}: quant test acc {acc:.4f}")
        if best is None or acc > best[1]:
            best = (key, acc, pred)
    (SH1, SH3, SH4), qacc, pred_q = best
    pred_q = _to_np(pred_q)             # predictions to host: used by export/demo
    print(f"chosen SH1={SH1} SH3={SH3} SH4={SH4}, quant test acc {qacc:.4f}")

    ptr = cnn_forward_int(xtr, qp, SH1, SH3, SH4)
    qtr = float((ptr == ytr).mean())
    print(f"quant: train {qtr:.4f} | test {qacc:.4f} | gap {qtr - qacc:+.4f}")
    for t, thr in enumerate(THRS):
        acc = (pred_q[t * 10000:(t + 1) * 10000] == te_y).mean()
        print(f"quant test thr={thr}: {acc:.4f}")

    # ---- reference test images (98 B each) + per-image sim ----
    sel = []
    for c in range(10):
        ids = np.nonzero(te_y == c)[0][:N_TEST_REF // 10]
        sel.extend(ids.tolist())
    sel = np.array(sel)
    thr0_bits = bin28(te_x[sel], THRS[0])
    imgs98 = pack98(thr0_bits)
    labels = te_y[sel].astype(np.uint8)

    src = (thr0_bits > 0.5).astype(np.int64)
    chk = np.zeros((len(sel), 784), np.int64)
    for i in range(784):
        chk[:, i] = (imgs98[:, i >> 3].astype(np.int64) >> (7 - (i & 7))) & 1
    assert (chk == src).all(), "pack/unpack roundtrip failed"

    pred200 = _to_np(cnn_forward_int(src, qp, SH1, SH3, SH4))
    assert (pred200 == pred_q[sel]).all(), "reference sim disagrees with full sim"
    ncorr = int((pred200 == labels).sum())

    # ---- write artifacts ----
    w1q, b1q, w2q, b2q, w3q, b3q = (_to_np(a) for a in (w1q, b1q, w2q, b2q, w3q, b3q))
    blob = bytearray()          # net_weights.bin 字节流: C1/C3/F5 顺序拼接, 每通道 [bias i16le][taps i8le]
    for ko in range(HID1):
        blob += struct.pack("<h", int(b1q[ko]))
        blob += struct.pack("<25b", *(int(v) for v in w1q[ko]))
    for ko in range(HID3):
        blob += struct.pack("<h", int(b2q[ko]))
        blob += struct.pack("<150b", *(int(v) for v in w2q[ko]))
    for j in range(10):
        blob += struct.pack("<h", int(b3q[j]))
        blob += struct.pack("<256b", *(int(v) for v in w3q[j]))
    assert len(blob) == HID1 * 27 + HID3 * 152 + 10 * 258, "weight blob size wrong"
    with open(os.path.join(MODEL_DIR, "net_weights.bin"), "wb") as f:
        f.write(blob)

    # MASM 版权重: 同一段字节展开成 DB 文本 (MASM 无 incbin, NN.ASM INCLUDE NN_WDAT.INC)
    wdat = os.path.join(os.path.dirname(BASE), "asm", "NN_WDAT.INC")
    with open(wdat, "w") as f:
        f.write("; generated by train_mnist_cnn.py from model output/net_weights.bin\n")
        f.write("; same bytes as net_weights.bin; w1=C1(6ch) w2=C3(16ch) w3=F5(10),\n")
        f.write("; per channel: [bias i16le][taps i8]  ->  w1 162B, w2 2432B, w3 2580B\n")
        off = 0
        for name, size in (("w1", HID1 * 27), ("w2", HID3 * 152), ("w3", 10 * 258)):
            first = True
            for i in range(off, off + size, 16):
                row = ",".join("0%02Xh" % (b & 0xFF) for b in blob[i:min(i + 16, off + size)])
                f.write(("%-7s DB      %s\n" % (name, row)) if first else ("        DB      %s\n" % row))
                first = False
            off += size
    print("wrote", wdat)
    imgs98.tofile(os.path.join(MODEL_DIR, "test_images.bin"))
    labels.tofile(os.path.join(MODEL_DIR, "test_labels.bin"))
    with open(os.path.join(MODEL_DIR, "nn_cfg.inc"), "w") as f:
        f.write("; generated by train_mnist_cnn.py\n")
        f.write(f"SH1 equ {SH1}\nSH3 equ {SH3}\nSH4 equ {SH4}\n")

    # demo inputs: correctly-classified 0 and 9 from the test set (thr 0.25)
    for digit, name in ((0, "DEMO0.TXT"), (9, "DEMO9.TXT")):
        ids = np.nonzero(te_y == digit)[0]
        ok = ids[pred_q[ids] == digit][0]
        rows = (bin28(te_x[ok:ok + 1], THRS[0]).reshape(28, 28) > 0.5)
        with open(os.path.join(DEMO_DIR, name), "w", newline="\r\n") as f:
            for r in range(28):
                f.write("".join("#" if v else "." for v in rows[r]) + "\n")
        print(f"{name}: test image #{ok}, sim predicts {pred_q[ok]}")

    with open(os.path.join(OUT, "report_cnn.txt"), "w", encoding="utf-8") as f:
        f.write(f"LeNet-style CNN: C1(6@5x5)-S2-C3(16@5x5)-S4-F5(10), 28x28 binary\n")
        f.write(f"thresholds={THRS} train=500 imgs x 2 thr x 3 stroke variants"
                f" (orig/dilate/erode) = 30000 samples\n")
        f.write(f"epochs<={EPOCHS} patience={PATIENCE} wd={WD} lr={LR}\n")
        f.write(f"best val acc: {best_va:.4f}\n")
        f.write(f"float test acc (20k, 2 thr): "
                f"{float((net.predict(xte) == yte).mean()):.4f}\n")
        f.write(f"quant: train {qtr:.4f} test {qacc:.4f} gap {qtr-qacc:+.4f}\n")
        f.write(f"SH1={SH1} SH3={SH3} SH4={SH4}\n")
        f.write(f"reference 200 (thr={THRS[0]}) acc: {ncorr}/200\n")
    print("artifacts written.")
    print(f"reference 200 acc: {ncorr}/200")


if __name__ == "__main__":
    main()
