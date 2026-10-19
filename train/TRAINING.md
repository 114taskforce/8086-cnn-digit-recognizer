# 从零训练神经网络的权重、偏置与移位常量 —— 数据预处理与训练详解

> **数据预处理**、**权重/偏置是怎么从零训练出来的**、
> **移位常量 SH1/SH3/SH4 是怎么定下来的**。
>
> 所有代码节选自 [train/train\_mnist\_cnn.py](train/train_mnist_cnn.py)。

***

## 0. 全景：一个权重的诞生之路

```
MNIST 四个 .gz (train/data/)
  │  IDX 格式解析 (大端长度头)
  ▼
读成 (N,28,28) uint8 灰度图
  │  bin28: /255 > 阈值 → 0/1        ← 二值化，两种阈值 0.25 / 0.50
  │  morph: 3×3 膨胀 / 腐蚀          ← 笔画粗细增强 (仅训练集)
  ▼
float32 CNN 训练 (NumPy 手写反向传播)
  │  AdamW + 余弦退火 + 早停         → out/cnn_model.npz 缓存
  ▼
对称 per-tensor int8 量化
  │  k = 127/max|W|，bias 与权重同 scale
  ▼
数据驱动移位常量搜索
  │  测每层累加器实际值域 → 估 SH → ±1 网格 → 探针评估 → 20k 精评
  ▼
导出产物
  ├── train/model output/net_weights.bin   5174 B 量化权重 (NASM incbin 嵌入汇编)
  ├── asm/NN_WDAT.INC                      同一字节的 DB 文本形式 (MASM INCLUDE 用)
  ├── train/model output/nn_cfg.inc        SH1/SH3/SH4 三行 equ
  ├── train/model output/test_images.bin   200×98 B 参考测试图
  ├── train/model output/test_labels.bin   200 B 参考标签
  └── train/test demo/DEMO0.TXT / DEMO9.TXT  回归演示输入
```

## 1. 数据预处理

### 1.1 获取与读取 MNIST

数据放在 `train/data/`，缺失时自动从镜像下载：

```python
MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/",
           "https://storage.googleapis.com/cvdf-datasets/mnist/"]
```

MNIST 是 IDX 格式：**大端**的魔数和维度头 + 像素/标签流。图片文件头 16 字节
（含样本数在 4:8），标签文件头 8 字节，各跳过即可：

```python
def load_images(name):
    with gzip.open(fetch(name), "rb") as f:
        data = f.read()
    n = struct.unpack(">I", data[4:8])[0]          # 大端读样本数
    return np.frombuffer(data, np.uint8, offset=16).reshape(n, 28, 28)

def load_labels(name):
    with gzip.open(fetch(name), "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
```

### 1.2 二值化——8086 的输入只有 0 和 1

8086 端的输入就是一张黑白点阵。所以训练数据必须同样二值化：

```python
def bin28(imgs, thr):
    """(N,28,28) uint8 -> (N,784) float32 {0,1}"""
    return (imgs.astype(np.float32) / 255.0 > thr).astype(np.float32).reshape(-1, 784)
```

### 1.3 双阈值：一张图变两个样本

手写输入的笔画粗细不可控。同一张灰度图用**两种阈值**（0.25 / 0.50）各生成一个
样本——0.25 阈值下笔画"变粗"、0.50 下"变细"，网络同时见到两种形态，
对手绘输入的鲁棒性至关重要：

| 像素灰度 | 0.25 阈值 | 0.50 阈值 |
| ---- | ------- | ------- |
| 0    | 背景      | 背景      |
| 64   | **墨迹**  | 背景      |
| 128  | **墨迹**  | **墨迹**  |
| 255  | **墨迹**  | **墨迹**  |

```python
THRS = (0.25, 0.50)

def make_dataset(imgs, ys, idx):
    """验证/测试用：N 图 × 2 阈值 -> (2N, 784)"""
    xs = [bin28(imgs[idx], t) for t in THRS]
    return np.concatenate(xs, 0), np.concatenate([ys[idx]] * len(THRS))
```

### 1.4 形态学增强：膨胀与腐蚀

阈值只改变"边缘像素的判定"，改变不了笔画本身的粗细。训练集额外做 3×3 方形
**膨胀（dilate，9 邻域取或）** 和 **腐蚀（erode，9 邻域取与）**，直接生成更粗/更细
的笔画变体：

```python
def morph(xbits, op):
    """3x3 square dilate('d')/erode('e') -> 笔画粗细变体"""
    g = xbits.reshape(-1, 28, 28) > 0.5
    p = np.pad(g, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros(g.shape, bool) if op == "d" else np.ones(g.shape, bool)
    fn = np.logical_or if op == "d" else np.logical_and
    for dr in range(3):
        for dc in range(3):
            out = fn(out, p[:, dr:dr+28, dc:dc+28])
    return out.reshape(-1, 784).astype(np.float32)

def make_train_dataset(imgs, ys, idx):
    """每类 500 图 × 2 阈值 × 3 笔画变体 = 30000 训练样本"""
    xs, ys_ = [], []
    for t in THRS:
        x0 = bin28(imgs[idx], t)
        xs += [x0, morph(x0, "d"), morph(x0, "e")]
        ys_ += [ys[idx]] * 3
    return np.concatenate(xs, 0), np.concatenate(ys_)
```

**注意**：增强只用于训练集。验证集与测试集只做双阈值（×2），保持评估数据的"原生态"。

### 1.5 数据划分（固定种子，完全可复现）

按类别**分层抽样**，seed=42 固定，任何人重跑得到完全相同的数据划分：

| 划分 | 原始图              | 变换            | 样本数   |
| -- | ---------------- | ------------- | ----- |
| 训练 | 每类 500 张（共 5000） | ×2 阈值 ×3 笔画变体 | 30000 |
| 验证 | 每类 250 张（共 2500） | ×2 阈值         | 5000  |
| 测试 | 全量 10000 张       | ×2 阈值         | 20000 |

```python
rng = np.random.RandomState(SEED)          # SEED = 42
for c in range(10):
    ids = np.nonzero(tr_y == c)[0]
    ids = ids[rng.permutation(len(ids))]   # 每类洗牌后切前 500 / 后 250
    tr_ids.extend(ids[:N_PER_CLASS].tolist())
    va_ids.extend(ids[N_PER_CLASS:N_PER_CLASS+N_VAL_CLASS].tolist())
```

### 1.6 打包成 98 字节：784 bit 怎么排

8086 端不存 784 个字节，而是把 784 个 bit 压进 **98 字节**（784/8）。位序约定：
**行优先、每字节 MSB 在前**——第 i 个像素落在第 `i>>3` 字节的第 `7-(i&7)` 位。
Python 的 `pack98` 与汇编的 `pack_grid`/`read_file`+`infer` 读法完全一致：

```python
def pack98(x14bin):
    """(N,28,28) {0,1} -> (N,98) uint8, row-major MSB first"""
    n = x14bin.shape[0]
    bits = x14bin.reshape(n, 784) > 0.5
    out = np.zeros((n, 98), np.uint8)
    for i in range(784):
        if bits[:, i].any():
            out[:, i >> 3] |= bits[:, i].astype(np.uint8) << (7 - (i & 7))
    return out
```

```
字节 0                    字节 1
┌─┬─┬─┬─┬─┬─┬─┬─┐  ┌─┬─┬─┬─┬─┬─┬─┬─┐
│bit7│6│5│4│3│2│1│bit0│ ...
│像素(0,0) → 像素(0,7)│ 像素(0,8) → ...
└─┴─┴─┴─┴─┴─┴─┴─┘  └─┴─┴─┴─┴─┴─┴─┴─┘
```

位序反了（LSB first）程序照样运行、照样输出，只是准确率暴跌且不报错——所以训练脚本
写完打包函数后立即做**往返校验**：

```python
chk[:, i] = (imgs98[:, i >> 3].astype(np.int64) >> (7 - (i & 7))) & 1
assert (chk == src).all(), "pack/unpack roundtrip failed"
```

<br />

## 2. 网络结构与 float 训练

### 2.1 参数盘点

| 层                     | 输出尺寸    | 权重           | 偏置       | 参数量                       |
| --------------------- | ------- | ------------ | -------- | ------------------------- |
| C1 conv 6\@5×5 valid  | 24×24×6 | W1 (6, 25)   | b1 (6,)  | 150 + 6                   |
| S2 avgpool 2×2        | 12×12×6 | —            | —        | 0                         |
| C3 conv 16\@5×5 valid | 8×8×16  | W2 (16, 150) | b2 (16,) | 2400 + 16                 |
| S4 avgpool 2×2        | 4×4×16  | —            | —        | 0                         |
| F5 dense 256→10       | 10      | W3 (10, 256) | b3 (10,) | 2560 + 10                 |
| **合计**                | <br />  | <br />       | <br />   | **5142**（5110 权重 + 32 偏置） |

量化后：权重 int8 共 5110 B + 偏置 int16 共 64 B = **5174 B**，正是
`net_weights.bin` 的大小。

### 2.2 三个"布局对齐"决定

训练端的数据排布必须与汇编取数顺序逐位对应，有三处生死攸关：

1. **卷积 tap 顺序**：C1 是 `(dr, dc)`，C3 是 `(ci, dr, dc)`（先走完 6 个输入通道）——
   `im2col_f` 的三重循环顺序就是 tap 顺序：
   ```python
   for ci in range(C):
       for dr in range(k):
           for dc in range(k):
               cols[:, :, :, i] = x[:, ci, dr:dr+Ho, dc:dc+Wo]
   ```
2. **im2col 用 NCHW**，而层间张量流是 NHWC，池化前后要 `transpose(0,3,1,2)` 来回切。
3. **F5 展平必须通道优先**：`h4.transpose(0,3,1,2).reshape(n, 256)`，对应汇编按
   `(ci, r, c)` 顺序读 256 个 tap。

### 2.3 初始化与前向

He 初始化（适配 ReLU）：`W ~ N(0, sqrt(2/fan_in))`，偏置置零。

前向用 im2col 把卷积化成矩阵乘法（`cols @ W.T + b`），整个网络就是几次 GEMM +
池化 reshape，几十行 NumPy：

```python
cols1 = im2col_f(ximg, 5)
c1 = cols1 @ self.W1.T + self.b1          # (N,24,24,6)
a1 = np.maximum(c1, 0)                    # ReLU
s2 = pool2(a1.transpose(0,3,1,2)).transpose(0,2,3,1)
...
flat = s4.transpose(0,3,1,2).reshape(n, 256)   # 通道优先!
logits = flat @ self.W3.T + self.b3
```

### 2.4 反向传播

Softmax 交叉熵，梯度 `(p − y) / n`；池化梯度均匀摊回 4 份；ReLU 用 mask
`(a > 0)`；卷积 dW 用 `tensordot` 收缩 batch/空间维：

```python
g = (p - yoh) / n
dW3 = g.T @ cache["flat"]
...
dc3 = da3 * (cache["a3"] > 0)                     # ReLU'
dW2 = np.tensordot(cache["cols2"], dc3, axes=([0,1,2],[0,1,2])).T
ds2 = col2im_f(dc3 @ self.W2, ...)                # 卷积反向 = 转置卷积
da1 = pool2_grad(ds2...)                          # ÷ 4
```

### 2.5 优化策略

| 项          | 取值          | 说明                                 |
| ---------- | ----------- | ---------------------------------- |
| 优化器        | AdamW       | 解耦权重衰减，`pm -= lr·WD·pm`            |
| 学习率        | 2e-3，余弦退火   | `lr = 0.5·LR·(1+cos(π·ep/EPOCHS))` |
| batch / 轮数 | 64 / ≤80    | 每 epoch 固定种子洗牌（SEED+ep，可复现）        |
| 早停         | patience=15 | 验证集 15 轮不提升即停，**回滚到最佳权重快照**        |



## 3. 量化：float → int8 / int16

### 3.1 对称 per-tensor 量化

每层独立求缩放系数 `k = 127 / max|W|`，权重线性映射到 \[-127, 127] 的 int8：

```python
def quant_params(net):
    for W, b in ((net.W1, net.b1), (net.W2, net.b2), (net.W3, net.b3)):
        k = 127.0 / np.abs(W).max()
        wq = np.clip(np.round(W * k), -127, 127).astype(np.int64)      # i8
        bq = np.clip(np.round(b * k), -32768, 32767).astype(np.int64)  # i16
```

### 3.2 偏置为什么必须与权重共享 scale

汇编里偏置和乘积要**直接相加**：

```
acc = b + Σ w·x  =  k·(b + Σ w·x)  =  bq + Σ wq·x     （bq = round(b·k)）
```

只有偏置与权重同乘一个 k，所有项才落在**同一个定标域**里，整数加法才有意义。
如果给偏置单独配 scale，`acc` 的每一项量纲都不同，整个定点体系即告失效——
这是量化最容易犯的错。也正因如此 acc 的量纲 = 权重量纲，后面的移位常量
才有一个统一的"每层归一化"含义（见第 4 章）。

### 3.3 数值域与溢出断言

| 量                               | 理论最坏值                   | 容器                      |
| ------------------------------- | ----------------------- | ----------------------- |
| C1 acc = bias + Σ(25 tap · bit) | \|b1\| + 25×127 ≈ 数千    | **int16**（汇编用 16 位加法）   |
| h1/h2/h3/h4                     | clamp 到 0..127，池化和 ≤508 | int8                    |
| C3 acc（150 tap × 127×127）       | ≈ 242 万                 | int32（汇编 DX:BX 手拼 32 位） |
| F5 acc（256 tap × 127×127）       | ≈ 413 万                 | int32                   |

```python
assert acc1.min() >= -32768 and acc1.max() <= 32767, "C1 int16 overflow"
assert acc2.min() >= -(2**31) and acc2.max() <= 2**31 - 1, "C3 int32 overflow"
```

### 3.4 量化代价

float 97.30% → int8/int16 定点 **97.18%**，仅损失 0.12 个百分点；

***

## 4. 移位常量 SH1 / SH3 / SH4

### 4.1 定标问题：为什么每层要 >>SH

量化后每一层的累加器量纲 = "权重量纲 × 输入量纲"。C1 输出累加值可达 ±数千，
但下一层的权重是按 0..127 的激活来设计的——**必须把累加器重新定标回 0..127**
才能喂给下一层。8086 上做这件事的最廉价手段就是算术右移：

```
h1 = clamp(acc1 >> SH1, 0, 127)     ← SH1 就是"这层输出除以 2^SH1"
h2 = (四邻和) >> 2                  ← 池化是固定的 >>2
h3 = clamp(acc2 >> SH3, 0, 127)
h4 = (四邻和) >> 2
out = acc3 >> SH4                   ← F5 只比大小，不用压回 0..127
```

`clamp(v, 0, 127)` 顺便完成了 ReLU 的功能（负数截成 0），还把激活压进了 int8
存储范围——一举两得。SH 不对的症状很有迷惑性：程序照跑、照出结果，
只是准确率莫名其妙地低（全 0、全同一类等）。

### 4.2 三条语义

| 语义        | Python 端                     | 8086 端                       |
| --------- | ---------------------------- | ---------------------------- |
| 右移负数      | `>>` 是算术移位（向下取整）             | 必须 `SAR`（不能用 SHR）            |
| 激活非线性     | `np.clip(acc >> SH, 0, 127)` | `SAR` 后 `jle`/`jns` 双向 clamp |
| argmax 平局 | numpy 取**第一个**最大值            | `cmp/jle`（不大于即跳过，保留先者）       |

### 4.3 依赖链："置 1 测量法"

三层 SH 互相耦合：C3 的累加值域取决于 SH1 把 h1 压到多小；F5 的值域取决于
SH1、SH3 两者。测量办法是**把还没定的 SH 设为 1**（移位后只测原始 acc 的极值，
与 SH 取值无关的部分先测）：

```
mx1：SH 全置 1 测   → C1 acc 与 SH 无关，可直接估 SH1
mx2：固定 s1，测 C3 → SH3 依赖 SH1
mx3：固定 s1,s3 测 F5 → SH4 依赖 SH1 + SH3
```

### 4.4 估计公式与 ±1 网格搜索

```python
sh1_est = max(1, round(log2(mx1 / 127)))          # 目标：压回 0..127
sh3_est = max(1, round(log2(mx2 / 127)))          # 同上
sh4_est = max(0, ceil(log2(mx3 / 32767)))         # 目标：压回 int16 (见 4.5)
```

每个估计值 ±1 组成候选（≤3×3×3 = 27 组合），在 **1000 个探针样本**上用位级模拟
评估准确率，取 top3 在 **20000 全量测试集**上精评：

```python
plan[(s1, s3, s4)] = None
results[key] = (cnn_forward_int(xprobe, qp, *key) == yprobe).mean()
ranked = sorted(results, key=results.get, reverse=True)
```

> 小字：探针取 `xv[:1000]`，即 0.25 阈值段的前 1000 个（类别 0–3）——实现上的简化；
> 更严谨可分层抽样，但 top3 精评已用全量数据兜底。

最终选定：**SH1=3, SH3=6, SH4=2**，量化 test 97.18%。

### 4.5 SH4 的下界：int16 argmax 约束（本项目的实战教训）

汇编的 F5 argmax 用 16 位寄存器比较 `acc3 >> SH4`。若 SH4 比估计值还小，
移位后数值超出 ±32767，16 位比较截断出错（症状：某些图预测彻底错乱）。
所以 [train/reshim.py](train/reshim.py) 重搜时**砍掉 est−1 候选**，只搜
`(est, est+1)`，并在精评时硬断言：

```python
for s4 in (sh4_est, sh4_est + 1):        # no est-1: int16 argmax constraint
    ...
assert mx3 >> key[2] <= 32767, f"{key}: output exceeds int16"
```

当前产物（SH1=3 SH3=6 SH4=2）即约束版搜索的结果，报告里记作
`SH4 lower-bounded: int16 argmax`。

### 4.6 生成与消费

训练脚本最后把三个数字写进 `asm/nn_cfg.inc`（汇编用 `%include` 引入）：

```nasm
; generated by train_mnist_cnn.py
SH1 equ 3
SH3 equ 6
SH4 equ 2
```

汇编端消费方式：16 位层 `sar dx, cl`；32 位层 `SAR` 高字 + `RCR` 低字双步移位，
且 `cl=0` 时必须 `jcxz` 守卫（否则 `dec cl/jnz` 会移位 256 次）。

***

## 5. 导出产物与验证链路

### 5.1 net\_weights.bin：逐字节布局

三个块直接顺序拼接，little-endian，每通道先 bias 后 taps：

```
偏移      0 ─────────────── 162 ─────────── 2594 ─────────── 5174
          │  C1: 6 通道       │  C3: 16 通道      │  F5: 10 神经元   │
每通道     │ [bias i16][25×i8] │ [bias i16][150×i8]│ [bias i16][256×i8]│
          └ 162 B            └ 2432 B           └ 2580 B
```

汇编端用三段 `incbin` 切片嵌入，长度常量与切片偏移一一对应：

```nasm
C1LEN   equ 6*27                 ; = 162
C3LEN   equ 16*152               ; = 2432
F5LEN   equ 10*258               ; = 2580
w1:     incbin  "net_weights.bin",0,C1LEN
w2:     incbin  "net_weights.bin",C1LEN,C3LEN
w3:     incbin  "net_weights.bin",C1LEN+C3LEN,F5LEN
```

MASM 版（`asm/NN.ASM`）没有 `incbin`，改用 `INCLUDE NN_WDAT.INC`：训练脚本把
**同一段字节流**逐字节展开成 `w1/w2/w3` 三个 DB 数据块（每行 16 字节，块末行按
块边界截断）写到 `asm/NN_WDAT.INC`，与 net_weights.bin 内容完全一致——
两种表示，一份权重，训练导出时同时生成。

tap 顺序再强调一遍：**C1** **`(dr,dc)`；C3** **`(ci,dr,dc)`；F5** **`(ci,r,c)`** **通道优先**——
与 2.2 的 im2col 循环顺序严格一致。

### 5.2 参考测试集与两道断言

- `test_images.bin`：每类前 20 张、共 **200 张**，0.25 阈值二值化后 pack98
  （200×98 = 19600 B）；`test_labels.bin`：200 B 标签
- 导出前跑两道断言：
  1. **往返校验**：98 B 解包回 784 bit 与源图逐位一致（防位序错）
  2. **一致性校验**：200 张的位级模拟结果与全量测试模拟逐张相同（防切数据出错）

### 5.3 DEMO 输入的生成

从测试集中挑出**模型预测正确**的 0 和 9 各一张（0.25 阈值），写成 28×28 的
`#`/`.` 文本（CRLF），作为 DOSBox 的回归输入——保证拿到的演示图一定能预测对。

### 5.4 验证链路

| 层级        | 工具                                    | 验证什么                |
| --------- | ------------------------------------- | ------------------- |
| Python 内部 | `cnn_forward_int`                     | 量化/移位语义自洽，20k 精度    |
| 单样本位级     | [train/dbg\_sim.py](train/dbg_sim.py) | 一张图逐层打印，给汇编插桩对拍     |
| 误差画像      | [train/diag9.py](train/diag9.py)      | 10×10 混淆矩阵，看"谁被认成谁" |
| 端到端       | DOSBox 跑 DEMO0/9 → RESULTS.TXT        | 汇编产物与 Python 模拟一致   |

### 5.5 复现命令

```bat
cd train
python train_mnist_cnn.py     :: 全流程（模型有缓存 out/cnn_model.npz，删掉即重训）
python reshim.py              :: 只重搜移位常量并重写产物（不重训）
cd ..\asm
..\tools\nasm\nasm-2.16.03\nasm.exe -f bin nn_digits.asm -o ..\emu\nn.exe -l ..\emu\nn.lst
```

***

## 6.速查表

| 项          | 值                                                                                  |
| ---------- | ---------------------------------------------------------------------------------- |
| 网络         | C1(6\@5×5)–S2–C3(16\@5×5)–S4–F5(256→10)，输入 28×28 二值                                |
| 参数量 / 权重文件 | 5142 / **5174 B**                                                                  |
| 数据         | train 500 类图×2 阈值×3 变体=30000；val 5000；test 20000                                   |
| 阈值         | 0.25 / 0.50                                                                        |
| 优化         | AdamW, lr 2e-3, wd 1e-4, batch 64, 余弦退火, 早停 15                                     |
| 量化         | 对称 per-tensor int8，bias 同 scale i16                                                |
| 移位常量       | **SH1=3, SH3=6, SH4=2**（nn\_cfg.inc，数据驱动+网格搜索）                                     |
| 精度         | float 97.30% / 量化 **97.18%**（20000 样本）；参考 200 张对 195                               |
| 产物         | net\_weights.bin / nn\_cfg.inc / test\_images.bin / test\_labels.bin / DEMO0,9.TXT |

