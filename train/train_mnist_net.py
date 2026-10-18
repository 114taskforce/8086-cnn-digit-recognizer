# -*- coding: utf-8 -*-
"""
Train a small MLP on binarized 14x14 MNIST, quantize weights for 8086 asm.

Network: 196 -> 32 (ReLU) -> 10
Input:   14x14 binarized, packed row-major MSB-first into 25 bytes (196 bits, 4 pad bits = 0)

Anti-overfitting:
  - per-epoch random shift augmentation (dx,dy in {-1,0,1}) on the binary bitmaps
  - AdamW weight decay (decoupled L2)
  - early stopping on a 5k validation split (patience), best-epoch weights kept

Quantized inference (exactly what the 8086 code does):
  acc1(int16) = b1_q + sum( w1_q[int16] * bit )        over 200 bit positions (4 pad bits = 0)
  h8          = clamp(sar(acc1, SH1), 0, 127)          int8 hidden
  acc2(int32) = sext(b2_q) + sum( w2_q[int8] * h8 )    over 32 hidden units
  out_j(int16)= sar(acc2, SH2)                         argmax, first max wins
"""
import os
import struct
import urllib.request
import gzip
import numpy as np

rng0 = np.random.RandomState(42)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "model output")   # final model data: weights + shifts + test sets
OUT = os.path.join(BASE, "out")
os.makedirs(DATA, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

H = 32              # hidden units
THR = 0.25          # binarization threshold (selected by sweep earlier: 0.08/0.15/0.25 -> 0.25 best)
N_VAL = 5000        # validation split for early stopping
N_TEST_EMBED = 200
MAX_EPOCHS = 80
PATIENCE = 10
WD = 1e-4           # weight decay

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def fetch(name):
    path = os.path.join(DATA, name)
    if os.path.exists(path):
        return path
    last = None
    for m in MIRRORS:
        try:
            print("download", m + name)
            req = urllib.request.Request(m + name, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
                f.write(r.read())
            return path
        except Exception as e:
            print("  failed:", e)
            last = e
    raise RuntimeError(f"cannot download {name}: {last}")


def load_images(name):
    with gzip.open(fetch(name), "rb") as f:
        data = f.read()
    n = struct.unpack(">I", data[4:8])[0]
    return np.frombuffer(data, np.uint8, offset=16).reshape(n, 28, 28)


def load_labels(name):
    with gzip.open(fetch(name), "rb") as f:
        data = f.read()
    return np.frombuffer(data, np.uint8, offset=8).astype(np.int64)


def binarize(imgs28, thr):
    """28x28 uint8 -> 14x14 (2x2 mean) -> binary float32 {0,1}"""
    x = imgs28.astype(np.float32).reshape(-1, 14, 14, 2, 2).mean(axis=(3, 4)) / 255.0
    return (x > thr).astype(np.float32)


def pack_bits(x14):
    """(N,14,14) {0,1} -> (N,25) uint8, row-major MSB first, 4 pad bits 0"""
    n = x14.shape[0]
    bits = x14.reshape(n, 196) > 0.5
    out = np.zeros((n, 25), np.uint8)
    for b in range(25):
        for k in range(8):
            i = b * 8 + k
            if i >= 196:
                break
            out[:, b] |= bits[:, i].astype(np.uint8) << (7 - k)
    return out


def shift_augment(x, rng):
    """randomly shift each 14x14 binary image by dx,dy in {-1,0,1} (edge = blank)"""
    n = x.shape[0]
    img = x.reshape(n, 14, 14)
    dx = rng.randint(-1, 2, size=(n, 1, 1))
    dy = rng.randint(-1, 2, size=(n, 1, 1))
    ri = np.arange(14)[None, :, None] - dy      # source row for each output row
    ci = np.arange(14)[None, None, :] - dx
    ok = (ri >= 0) & (ri < 14) & (ci >= 0) & (ci < 14)
    r = np.clip(ri, 0, 13)
    c = np.clip(ci, 0, 13)
    rows = np.arange(n)[:, None, None]
    return np.where(ok, img[rows, r, c], 0).reshape(n, 196)


def forward_float(x, W1, b1, W2, b2):
    h = np.maximum(x @ W1 + b1, 0)
    return h @ W2 + b2


def train(xtr, ytr, xval, yval, epochs=MAX_EPOCHS, hidden=H, seed=42,
          wd=WD, patience=PATIENCE, augment=True):
    """2-layer MLP, AdamW + shift augmentation + early stopping; returns best-epoch weights"""
    r = np.random.RandomState(seed)
    W1 = r.randn(196, hidden) * np.sqrt(2.0 / 196)
    b1 = np.zeros(hidden)
    W2 = r.randn(hidden, 10) * np.sqrt(2.0 / hidden)
    b2 = np.zeros(10)
    mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
    mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
    mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)
    lr, b1d, b2d, eps = 1e-3, 0.9, 0.999, 1e-8
    step = 0
    best_va, best_w, bad = -1.0, None, 0
    for ep in range(epochs):
        lr_ep = 0.5 * lr * (1.0 + np.cos(np.pi * ep / epochs))   # cosine annealing
        if augment is True or (isinstance(augment, float) and augment > 0):
            p = 1.0 if augment is True else augment
            m = r.rand(len(xtr)) < p
            xs = xtr.copy()
            if m.any():
                xs[m] = shift_augment(xtr[m], r)
        else:
            xs = xtr
        perm = r.permutation(len(xs))
        for i in range(0, len(xs), 64):
            idx = perm[i:i + 64]
            x, y = xs[idx], ytr[idx]
            a1 = x @ W1 + b1
            h = np.maximum(a1, 0)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
            yoh = np.zeros_like(p); yoh[np.arange(len(idx)), y] = 1
            g = (p - yoh) / len(idx)
            gW2 = h.T @ g; gb2 = g.sum(0)
            gh = (g @ W2.T) * (a1 > 0)
            gW1 = x.T @ gh; gb1 = gh.sum(0)
            step += 1
            for W, gW, m, v, dec in ((W1, gW1, mW1, vW1, wd), (b1, gb1, mb1, vb1, 0.0),
                                     (W2, gW2, mW2, vW2, wd), (b2, gb2, mb2, vb2, 0.0)):
                m *= b1d; m += (1 - b1d) * gW
                v *= b2d; v += (1 - b2d) * gW * gW
                W -= lr_ep * (m / (1 - b1d ** step)) / (np.sqrt(v / (1 - b2d ** step)) + eps)
                W -= lr_ep * dec * W
        va = (forward_float(xval, W1, b1, W2, b2).argmax(1) == yval).mean()
        marker = ""
        if va > best_va:
            best_va, best_w, bad = va, [W1.copy(), b1.copy(), W2.copy(), b2.copy()], 0
            marker = " *"
        else:
            bad += 1
        print(f"  epoch {ep+1}: val_acc={va:.4f}{marker}")
        if bad >= patience:
            print(f"  early stop at epoch {ep+1} (no improvement for {patience})")
            break
    W1, b1, W2, b2 = best_w
    return W1, b1, W2, b2, best_va


def quantize_sim(xbits, y, W1, b1, W2, b2, SH1, SH2):
    """bit-exact simulation of the 8086 integer pipeline; returns predictions"""
    max1 = np.abs(W1).max()
    k1 = 127.0 / max1
    w1q = np.clip(np.round(W1 * k1), -127, 127).astype(np.int64)   # (196,H)
    b1q = np.clip(np.round(b1 * k1), -32768, 32767).astype(np.int64)
    max2 = np.abs(W2).max()
    k2 = 127.0 / max2
    w2q = np.clip(np.round(W2 * k2), -127, 127).astype(np.int64)   # (H,10)
    b2q = np.clip(np.round(b2 * k2), -32768, 32767).astype(np.int64)

    xb = xbits.astype(np.int64)                                    # (n,196) 0/1
    acc1 = b1q + xb @ w1q                                          # (n,H)
    assert acc1.min() >= -32768 and acc1.max() <= 32767, "layer1 int16 overflow!"
    h8 = np.clip(acc1 >> SH1, 0, 127).astype(np.int64)
    acc2 = b2q + h8 @ w2q                                          # (n,10)
    assert acc2.min() >= -(2 ** 31) and acc2.max() <= 2 ** 31 - 1, "layer2 int32 overflow!"
    out = acc2 >> SH2
    assert out.min() >= -32768 and out.max() <= 32767, "out int16 overflow!"
    pred = out.argmax(axis=1)          # argmax: first max wins (matches asm cmp/jle)
    return pred, (w1q, b1q, w2q, b2q)


def write_weights(path, w1q, b1q, w2q, b2q):
    """layout: per hidden neuron: bias int16le, then 200 int16le weights (196 + 4 zero pad)"""
    with open(path, "wb") as f:
        for h in range(H):
            f.write(struct.pack("<h", int(b1q[h])))
            for i in range(196):
                f.write(struct.pack("<h", int(w1q[i, h])))
            for _ in range(4):
                f.write(struct.pack("<h", 0))
        for j in range(10):
            f.write(struct.pack("<h", int(b2q[j])))
            for h in range(H):
                f.write(struct.pack("<b", int(w2q[h, j])))


def main():
    print("== load MNIST ==")
    tr_x = load_images("train-images-idx3-ubyte.gz")
    tr_y = load_labels("train-labels-idx1-ubyte.gz")
    te_x = load_images("t10k-images-idx3-ubyte.gz")
    te_y = load_labels("t10k-labels-idx1-ubyte.gz")

    xall = binarize(tr_x, THR).reshape(-1, 196)
    xte = binarize(te_x, THR).reshape(-1, 196)
    xtr, ytr = xall[:-N_VAL], tr_y[:-N_VAL]
    xv, yv = xall[-N_VAL:], tr_y[-N_VAL:]
    print(f"train {xtr.shape}, val {xv.shape}, test {xte.shape}")

    print("== training (augment + weight decay + early stop) ==")
    W1, b1, W2, b2, best_va = train(xtr, ytr, xv, yv)
    print(f"best val_acc={best_va:.4f}")

    tr_pred = forward_float(xtr, W1, b1, W2, b2).argmax(1)
    te_pred = forward_float(xte, W1, b1, W2, b2).argmax(1)
    tr_acc = (tr_pred == ytr).mean()
    fte_acc = (te_pred == te_y).mean()
    print(f"float: train {tr_acc:.4f} | test {fte_acc:.4f} | gap {tr_acc - fte_acc:.4f}")

    # ---- quantization: choose shifts by simulation ----
    xte_bits = (xte > 0.5).astype(np.int64)
    xtr_sub = (xtr[:10000] > 0.5).astype(np.int64)
    max1 = np.abs(W1).max(); k1 = 127.0 / max1
    a1max = (xtr_sub @ W1 + b1).max()
    sh1_est = int(round(np.log2(a1max * k1 / 127.0)))
    results = {}
    for SH1 in (sh1_est - 1, sh1_est, sh1_est + 1):
        for SH2 in (5, 6, 7, 8):
            pred, _ = quantize_sim(xte_bits, te_y, W1, b1, W2, b2, max(SH1, 1), SH2)
            results[(max(SH1, 1), SH2)] = (pred == te_y).mean()
    (SH1, SH2) = max(results, key=results.get)
    print("chosen SH1,SH2:", SH1, SH2, "sim acc:", results[(SH1, SH2)])

    pred_q, (w1q, b1q, w2q, b2q) = quantize_sim(xte_bits, te_y, W1, b1, W2, b2, SH1, SH2)
    qte_acc = (pred_q == te_y).mean()
    qtr_acc = (quantize_sim((xtr > 0.5).astype(np.int64), ytr, W1, b1, W2, b2, SH1, SH2)[0] == ytr).mean()
    print(f"quant: train {qtr_acc:.4f} | test {qte_acc:.4f} | gap {qtr_acc - qte_acc:.4f}")

    # ---- embed 200 stratified test images (kept as reference files only) ----
    sel = []
    for c in range(10):
        idx = np.nonzero(te_y == c)[0][:N_TEST_EMBED // 10]
        sel.extend(idx.tolist())
    sel = np.array(sel)
    imgs25 = pack_bits(binarize(te_x[sel], THR))
    labels = te_y[sel].astype(np.uint8)

    # roundtrip sanity: unpack must reproduce the source bits
    src = (binarize(te_x[sel], THR).reshape(-1, 196) > 0.5).astype(np.int64)
    chk = np.zeros((len(sel), 196), np.int64)
    for i in range(196):
        chk[:, i] = (imgs25[:, i >> 3].astype(np.int64) >> (7 - (i & 7))) & 1
    assert (chk == src).all(), "pack/unpack roundtrip failed"

    # simulate the exact 200 embedded images
    bits = np.zeros((len(sel), 196), np.int64)
    for i in range(196):
        bits[:, i] = (imgs25[:, i >> 3].astype(np.int64) >> (7 - (i & 7))) & 1
    acc1 = b1q + bits @ w1q
    h8 = np.clip(acc1 >> SH1, 0, 127)
    acc2 = b2q + h8 @ w2q
    out = acc2 >> SH2
    pred200 = out.argmax(axis=1)
    assert (pred200 == pred_q[sel]).all(), "embedded sim disagrees with full-set sim"

    # ---- write artifacts ----
    write_weights(os.path.join(MODEL_DIR, "net_weights.bin"), w1q, b1q, w2q, b2q)
    imgs25.tofile(os.path.join(MODEL_DIR, "test_images.bin"))
    labels.tofile(os.path.join(MODEL_DIR, "test_labels.bin"))
    with open(os.path.join(MODEL_DIR, "nn_cfg.inc"), "w") as f:
        f.write(f"; generated by train_mnist_net.py\nSH1 equ {SH1}\nSH2 equ {SH2}\n")

    ncorr = int((pred200 == labels).sum())
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"threshold={THR} hidden={H} val_split={N_VAL}\n")
        f.write(f"augment=shift(dx,dy in -1..1) wd={WD} early_stop_patience={PATIENCE}\n")
        f.write(f"best val acc: {best_va:.4f}\n")
        f.write(f"float: train {tr_acc:.4f} test {fte_acc:.4f} gap {tr_acc-fte_acc:+.4f}\n")
        f.write(f"quant: train {qtr_acc:.4f} test {qte_acc:.4f} gap {qtr_acc-qte_acc:+.4f}\n")
        f.write(f"SH1={SH1} SH2={SH2}\n")
        f.write(f"embedded 200 acc: {ncorr}/200\n")
    with open(os.path.join(OUT, "sim_predictions.txt"), "w") as f:
        f.write(" ".join(map(str, pred200)) + "\n")
    print("artifacts written.")
    print(f"embedded 200 acc: {ncorr}/200")


if __name__ == "__main__":
    main()
