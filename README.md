# 8086 手写数字识别（纯汇编 CNN）

在 8086 实模式 DOS 下运行的手写数字识别程序。网络为 LeNet 风格 CNN，用 8086 指令集以 MASM 汇编实现，全部采用 int8/int16 定点运算，无 FPU、无扩展指令。训练在 PC 上用 Python 完成，量化后的权重导出给汇编程序使用，量化模型对 20k 测试样本精度约 97%。

- 输入：28×28 二值文本图（每行 28 字符，`1`/`#`/`*` = 墨迹，其余字符 = 空白）
- 输出：屏幕打印输入图与 `PREDICT: n`，每次运行同时写入 `RESULTS.TXT`

## 网络结构

| 层  | 类型                 | 输出      | 说明                                |
| -- | ------------------ | ------- | --------------------------------- |
| C1 | conv 6\@5×5 valid  | 24×24×6 | 二值输入免乘法，acc int16 → ReLU          |
| S2 | avgpool 2×2        | 12×12×6 | 4 项和 >> 2                         |
| C3 | conv 16\@5×5 valid | 8×8×16  | int8×int8 (imul)，acc int32 → ReLU |
| S4 | avgpool 2×2        | 4×4×16  | 4 项和 >> 2                         |
| F5 | dense 256→10       | 10      | argmax（首个最大值胜出）                   |

定点移位常数 SH1/SH3/SH4 由训练数据驱动估计，量化推理与汇编逐位一致。

## 目录结构

```
train\                 训练脚本与文档
  train_mnist_cnn.py   主训练脚本（下载 MNIST、训练、量化、导出权重）
  reshim.py            SH 常数重搜（违反汇编约束时修复）
  train_mnist_net.py   数据加载（load_images/load_labels）
  TRAINING.md          训练细节文档
asm\                   汇编源码
  NN.ASM               MASM 版（本文档主流程）
  NN_WDAT.INC          (本地生成，不入库) 权重数据，INCLUDE 进 NN.ASM
  nn_digits.asm        NASM 等价版（nasm -f bin 直接出 .COM）
  NN_ARCHITECTURE.md   汇编实现细节（定点方案、寄存器分配、逐层说明）
tool\                  手绘工具
  draw.bat / draw_digits.py   28×28 手绘板，内嵌位级预测实时预览
  input model\         权重备份（nn_cfg.inc 入库；net_weights.bin 不入库）
  handmade\            (使用时生成) h<数字>_<时分>.txt 手绘样本
  README.md            工具使用说明
DOSBox\                本地 DOS 环境（不入库）：DOSBox 0.74-3、MASM/LINK/DEBUG、28×28 样例
train\model output\    (训练生成) net_weights.bin、nn_cfg.inc、测试集
train\test demo\       (训练生成) DEMO0.TXT、DEMO9.TXT
train\out\             (训练生成) report_cnn.txt、cnn_model.npz 缓存
```

## 使用流程

### 第 1 步：训练并导出权重

```
pip install numpy
python train\train_mnist_cnn.py
```

生成：

- `model output\net_weights.bin` — 5174 字节量化权重（C1 162B + C3 2432B + F5 2580B）
- `model output\nn_cfg.inc` — SH1/SH3/SH4 移位常数
- `test demo\DEMO0.TXT`、`test demo\DEMO9.TXT` — 演示样本
- `out\report_cnn.txt` — 精度报告

注意：

- 首次运行需联网下载 MNIST。
- 模型缓存于 `train\out\cnn_model.npz`，重训前删除该文件。

### 第 2 步：生成权重数据文件并用 MASM 编译

把 `asm\NN.ASM` 和 `train\model output\nn_cfg.inc` 复制到 `DOSBox\` 目录（MASM.EXE / LINK.EXE 所在处）。

MASM 没有 NASM 的 `incbin`，还需先把 `train\model output\net_weights.bin` 展开成权重数据文件 `NN_WDAT.INC`，放到 `NN.ASM` 旁边。在项目根目录运行：

```python
# make_wdat.py — net_weights.bin -> NN_WDAT.INC
import os
data = open(os.path.join("train", "model output", "net_weights.bin"), "rb").read()
BLOCKS = [("w1", 6 * 27), ("w2", 16 * 152), ("w3", 10 * 258)]  # C1 162B / C3 2432B / F5 2580B
with open("NN_WDAT.INC", "w") as f:
    off = 0
    for name, size in BLOCKS:
        blk = data[off:off + size]; off += size; first = True
        for i in range(0, size, 16):
            row = ",".join("0%02Xh" % (b & 0xFF) for b in blk[i:i + 16])
            f.write(("%-7s DB      %s" % (name, row)) if first else ("        DB      %s" % row))
            f.write("\n"); first = False
```

把生成的 `NN_WDAT.INC` 也复制到 `DOSBox\` 目录，启动 DOSBox：

```
mount c c:\users\administrator\desktop\8086\dosbox
c:
MASM NN.asm;
LINK NN.obj;
```

MASM 生成 NN.OBJ（附 NN.LST），LINK 生成 NN.EXE。

### 第 3 步：识别

文件模式（推荐）：

```
NN.EXE DEMO0.TXT
NN.EXE handmade\H0_1425.TXT
```

（`DEMO0.TXT` 由训练生成，在 `train\test demo\`，复制到 DOS 目录后使用；`handmade\` 是手绘工具的输出目录）

键盘模式：`NN.EXE` 不带参数，逐行键入 28 行 28 字符，回车提交每行。

- 文件模式忽略 CR/LF/空格/Tab，收满 784 像素即停止，可与手绘工具输出直接配合
- ESC 随时退出；每次运行结束后按任意键继续、ESC 退出
- 每次运行覆盖写 `RESULTS.TXT`（输入图 + 识别结果）

## 手绘样本

`tool\draw.bat` 启动手绘板，鼠标在 28×28 画布上绘制，右侧实时显示 CNN 预测；按数字键 0-9 打标签保存到 `tool\handmade\h<数字>_<时分>.txt`（如 `h0_1425.txt`，DOS 8.3 兼容命名），复制到 DOS 目录即可 `NN.EXE h0_1425.txt` 识别。

## 输入文件格式

28 行 × 28 字符的纯文本，`1`/`#`/`*` 均视为墨迹，其余字符视为空白。参考训练生成的 `train\test demo\DEMO0.TXT`。

## NASM 备选方案

`asm\nn_digits.asm` 为 NASM 语法的等价实现，用 `incbin` 直接内嵌 `net_weights.bin`、`%include nn_cfg.inc`，一条命令出 .COM（无需 LINK，需自备 NASM，在 asm 目录执行）：

```
nasm -f bin nn_digits.asm -o nn.com
```

## 更多文档

- [train/TRAINING.md](train/TRAINING.md) — 数据集、训练与量化细节
- [asm/NN\_ARCHITECTURE.md](asm/NN_ARCHITECTURE.md) — 汇编实现细节
- [tool/README.md](tool/README.md) — 手绘工具使用说明

