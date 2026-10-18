# -*- coding: utf-8 -*-
"""
手绘 28x28 二值数字小工具（配合 8086 汇编 CNN 识别程序 nn.com 使用）

用法:
    python tool\\draw_digits.py    （或双击 tool\\draw.bat）

操作:
    鼠标左键拖动 = 画      鼠标右键拖动 = 擦
    数字键 0-9 = 打标签并保存到 emu\\handmade\\h<标签>_<时分>.txt
    C = 清空画板
    每次修改实时显示识别预测（位级复刻 nn.com 的 CNN 定点运算）

输出格式: 28 行 x 28 字符，'#'=墨迹 '.'=空白，CRLF 换行，
          与 nn.com 的文件输入格式完全一致。
识别用法: 在 DOSBox 中执行  nn.com HANDMADE\\H0_1425.TXT
"""
import os
import re
import struct
import datetime
import tkinter as tk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASM_DIR = os.path.join("input model")
OUT_DIR = os.path.join(ROOT, "tool", "handmade")
N = 28
CELL = 22


def load_model():
    """读取 8086 CNN 程序使用的量化权重与移位参数"""
    cfg = open(os.path.join(ASM_DIR, "nn_cfg.inc")).read()
    sh1 = int(re.search(r"SH1\s+equ\s+(\d+)", cfg).group(1))
    sh3 = int(re.search(r"SH3\s+equ\s+(\d+)", cfg).group(1))
    sh4 = int(re.search(r"SH4\s+equ\s+(\d+)", cfg).group(1))
    data = open(os.path.join(ASM_DIR, "net_weights.bin"), "rb").read()
    off = 0
    b1, w1 = [], []                       # C1: 6 x [bias i16][25 i8]
    for _ in range(6):
        b1.append(struct.unpack_from("<h", data, off)[0]); off += 2
        w1.append(struct.unpack_from("<25b", data, off)); off += 25
    b2, w2 = [], []                       # C3: 16 x [bias i16][150 i8]
    for _ in range(16):
        b2.append(struct.unpack_from("<h", data, off)[0]); off += 2
        w2.append(struct.unpack_from("<150b", data, off)); off += 150
    b3, w3 = [], []                       # F5: 10 x [bias i16][256 i8]
    for _ in range(10):
        b3.append(struct.unpack_from("<h", data, off)[0]); off += 2
        w3.append(struct.unpack_from("<256b", data, off)); off += 256
    return sh1, sh3, sh4, b1, w1, b2, w2, b3, w3


def _clamp8(v):
    return 0 if v < 0 else (127 if v > 127 else v)


def predict(grid, model):
    """与 nn.com 位级一致的 CNN 推理（整数定点，层布局与汇编相同）"""
    sh1, sh3, sh4, b1, w1, b2, w2, b3, w3 = model
    # C1: conv 6@5x5 valid -> 24x24x6, ReLU
    h1 = []
    for ko in range(6):
        bias = b1[ko]
        for r in range(24):
            for c in range(24):
                acc = bias
                for dr in range(5):
                    gr = grid[r + dr]
                    wt = w1[ko]
                    for dc in range(5):
                        if gr[c + dc]:
                            acc += wt[dr * 5 + dc]
                h1.append(_clamp8(acc >> sh1))
    # S2: avgpool 2x2 -> 12x12x6
    h2 = []
    for ko in range(6):
        base0 = ko * 576
        for r in range(12):
            row = base0 + (2 * r) * 24
            for c in range(12):
                p = row + 2 * c
                h2.append((h1[p] + h1[p + 1] + h1[p + 24] + h1[p + 25]) >> 2)
    # C3: conv 16@5x5 valid over 6 channels -> 8x8x16, ReLU
    h3 = []
    for ko in range(16):
        bias = b2[ko]
        wt = w2[ko]
        for r in range(8):
            for c in range(8):
                acc = bias
                for ci in range(6):
                    hb = ci * 144
                    wb = ci * 25
                    for dr in range(5):
                        hr = hb + (r + dr) * 12 + c
                        wr = wb + dr * 5
                        for dc in range(5):
                            acc += wt[wr + dc] * h2[hr + dc]
                h3.append(_clamp8(acc >> sh3))
    # S4: avgpool 2x2 -> 4x4x16
    h4 = []
    for ko in range(16):
        base0 = ko * 64
        for r in range(4):
            row = base0 + (2 * r) * 8
            for c in range(4):
                p = row + 2 * c
                h4.append((h3[p] + h3[p + 1] + h3[p + 8] + h3[p + 9]) >> 2)
    # F5: dense 256 -> 10, argmax first-max-wins
    best, bj = None, 0
    for j in range(10):
        acc = b3[j]
        wt = w3[j]
        for i in range(256):
            acc += wt[i] * h4[i]
        out = acc >> sh4
        if best is None or out > best:      # 首个最大值胜出，与汇编 cmp/jle 一致
            best, bj = out, j
    return bj


def export_txt(grid, label):
    """按 nn.com 输入格式导出，文件名保持 DOS 8.3 兼容"""
    os.makedirs(OUT_DIR, exist_ok=True)
    name = "h%d_%s.txt" % (label, datetime.datetime.now().strftime("%H%M"))
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="\r\n") as f:
        for r in range(N):
            f.write("".join("#" if grid[r][c] else "." for c in range(N)) + "\n")
    return path


class App:
    def __init__(self, root):
        self.model = load_model()
        self.grid = [[0] * N for _ in range(N)]
        self.cells = [[None] * N for _ in range(N)]
        self.brush = 1

        root.title("手绘数字 -> 文本文件 (8086 CNN 识别用)")
        left = tk.Frame(root); left.pack(side="left", padx=8, pady=8)
        right = tk.Frame(root); right.pack(side="left", padx=8, pady=8, fill="y")

        self.cv = tk.Canvas(left, width=N * CELL, height=N * CELL,
                            bg="white", highlightthickness=0)
        self.cv.pack()
        for i in range(N + 1):
            self.cv.create_line(0, i * CELL, N * CELL, i * CELL, fill="#dddddd")
            self.cv.create_line(i * CELL, 0, i * CELL, N * CELL, fill="#dddddd")
        self.cv.bind("<Button-1>", lambda e: self.paint(e, 1))
        self.cv.bind("<B1-Motion>", lambda e: self.paint(e, 1))
        self.cv.bind("<Button-3>", lambda e: self.paint(e, 0))
        self.cv.bind("<B3-Motion>", lambda e: self.paint(e, 0))

        self.pred = tk.Label(right, text="PREDICT: -",
                             font=("Consolas", 24, "bold"), fg="#cc0000")
        self.pred.pack(pady=(0, 10))

        tk.Label(right, text="画笔粗细").pack()
        bf = tk.Frame(right); bf.pack()
        for w, txt in ((0, "细"), (1, "中"), (2, "粗")):
            tk.Button(bf, width=4, text=txt,
                      command=lambda w=w: setattr(self, "brush", w)).pack(side="left", padx=2)

        tk.Button(right, text="清空 (C)", width=14, command=self.clear).pack(pady=6)

        tk.Label(right, text="点击标签 = 保存文件").pack(pady=(6, 0))
        gf = tk.Frame(right); gf.pack()
        for i in range(10):
            tk.Button(gf, width=3, text=str(i),
                      command=lambda i=i: self.save(i)).grid(row=i // 5, column=i % 5, padx=2, pady=1)

        self.status = tk.Label(root, text="", fg="#006600", anchor="w")
        self.status.pack(side="bottom", fill="x")

        root.bind("<Key-c>", lambda e: self.clear())
        for d in "0123456789":
            root.bind(d, lambda e, d=d: self.save(int(d)))

    def paint(self, e, val):
        r, c = e.y // CELL, e.x // CELL
        if not (0 <= r < N and 0 <= c < N):
            return
        for dr in range(-self.brush, self.brush + 1):
            for dc in range(-self.brush, self.brush + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N:
                    self.set_cell(rr, cc, val)
        self.update_pred()

    def set_cell(self, r, c, val):
        if self.grid[r][c] == val:
            return
        self.grid[r][c] = val
        if self.cells[r][c] is not None:
            self.cv.delete(self.cells[r][c])
            self.cells[r][c] = None
        if val:
            x0, y0 = c * CELL + 1, r * CELL + 1
            self.cells[r][c] = self.cv.create_rectangle(
                x0, y0, x0 + CELL - 2, y0 + CELL - 2, fill="black", width=0)

    def update_pred(self):
        if any(self.grid[r][c] for r in range(N) for c in range(N)):
            self.pred.config(text="PREDICT: %d" % predict(self.grid, self.model))
        else:
            self.pred.config(text="PREDICT: -")

    def clear(self):
        for r in range(N):
            for c in range(N):
                self.set_cell(r, c, 0)
        self.update_pred()

    def save(self, label):
        path = export_txt(self.grid, label)
        name = os.path.basename(path).upper()
        self.status.config(text="已保存: %s    识别: DOSBox 里执行 nn.com HANDMADE\\%s"
                           % (path, name))


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
