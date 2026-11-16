#!/usr/bin/env python3
"""QP(양자화 · 가지치기) 노트 "상세 학습" 의 그림(SVG)을 만든다 → assets/fig/qp/*.svg

    STUDY_ROOT=<AI_CAMPUS 의 부모> python3 -I tools/fig_qp.py        # 전부 다시 만들기
    python3 -I tools/fig_qp.py d1_affine                               # 함수 이름 앞부분으로 일부만

그리기 도구(Svg · Plot)는 tools/fig_nn.py 의 것을 그대로 쓴다. 숫자는 강의와 같은 계산으로 직접 구한다 —
활성값 · weight 가 필요한 그림(d1_symasym · d1_perch · d1_clip · d1_results)은 교안의 data/mnist_cnn.pt ·
mnist_sample.npz 를 torch 로 읽는다 (torch 가 없으면 그 그림만 건너뛴다).
"""
import copy, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_nn
from fig_nn import Svg, Plot, INK, MUTED, LINE, BLUE, ORANGE, GREEN, RED, PURPLE, FILL

fig_nn.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'fig', 'qp')
ROOT = os.environ.get('STUDY_ROOT', os.path.expanduser('~/c_study'))
DATA = os.path.join(ROOT, 'AI_CAMPUS', 'QP-day1', 'data')


def mse(a, b): return float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))


_M = {}
def model():
    """SimpleConvNet + MNIST 100장 → 활성값 A · conv weight W · 모델 (강의 1 · 4교시와 같은 계산)"""
    if _M: return _M
    import torch, torch.nn as nn, torch.nn.functional as F

    class SimpleConvNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.conv = nn.Conv2d(1, 30, 5); s.pool = nn.MaxPool2d(2); s.fc1 = nn.Linear(30 * 12 * 12, 100); s.fc2 = nn.Linear(100, 10)

        def forward(s, x):
            x = s.pool(F.relu(s.conv(x))).flatten(1)
            return s.fc2(F.relu(s.fc1(x)))
    cnn = SimpleConvNet(); cnn.load_state_dict(torch.load(os.path.join(DATA, 'mnist_cnn.pt'), map_location='cpu')); cnn.eval()
    d = np.load(os.path.join(DATA, 'mnist_sample.npz'))
    X = torch.from_numpy(d['images'].astype(np.float32) / 255)[:, None]
    with torch.no_grad():
        A = cnn.pool(F.relu(cnn.conv(X))).numpy().ravel()
        logits = cnn(X).numpy()
    _M.update(torch=torch, cnn=cnn, X=X, labels=d['labels'], A=A, W=cnn.conv.weight.detach().numpy(), logits=logits)
    return _M


# ---------------- QP Day 1 ----------------

def d1_why():
    s = Svg(700, 230)
    s.text(350, 22, '모델 크기 = 파라미터 수 × 숫자 하나의 byte', 14, INK, 'middle', True)
    rows = [('SimpleConvNet', 433_890), ('MobileNetV3-Small', 2_542_856), ('YOLOv8n', 3_193_934)]
    cols = [('FP32 · 4 B', 4, BLUE), ('FP16 · 2 B', 2, GREEN), ('INT8 · 1 B', 1, ORANGE)]
    P = Plot(s, 150, 40, 500, 150, (0, 13), (0, 1))
    for i, (name, n) in enumerate(rows):
        y = 52 + i * 50
        s.text(140, y + 20, name, 12.5, INK, 'end', True)
        for j, (lab, b, c) in enumerate(cols):
            mb = n * b / 1e6
            s.rect(150, y + j * 13, P.X(mb) - 150, 11, c, None, 2, 1.2)
            s.text(P.X(mb) + 6, y + j * 13 + 10, f'{mb:.2f} MB', 10.5, MUTED)
    for j, (lab, b, c) in enumerate(cols):
        s.rect(160 + j * 150, 205, 14, 10, c, None, 2, 1.2); s.text(180 + j * 150, 214, lab, 12, INK)
    s.save('d1-why.svg')


def d1_formats():
    s = Svg(700, 256)
    s.text(350, 22, '자료형의 비트 구조 — 부호 · 지수 · 가수 (정수는 값 칸만)', 14, INK, 'middle', True)
    fmts = [('FP32', [(1, '', RED), (8, '지수 8', BLUE), (23, '가수 23', GREEN)], '최대 3.4×10³⁸ · eps 1.19×10⁻⁷'),
            ('FP16', [(1, '', RED), (5, '지수 5', BLUE), (10, '가수 10', GREEN)], '최대 65,504 · eps 0.000977 (유효숫자 ≈ 3자리)'),
            ('BF16', [(1, '', RED), (8, '지수 8', BLUE), (7, '가수 7', GREEN)], 'FP32 와 같은 범위 · 더 거친 간격'),
            ('INT8', [(8, '−128~127', ORANGE)], '256 칸 · 간격 1 · 소수 없음 (2의 보수)'),
            ('UINT8', [(8, '0 ~ 255', PURPLE)], '256 칸 · 음수 없음')]
    u = 13
    for i, (name, parts, note) in enumerate(fmts):
        y = 42 + i * 40
        s.text(60, y + 17, name, 13, INK, 'end', True)
        x = 70
        for n, lab, c in parts:
            s.rect(x, y, n * u, 24, c, None, 3, 1.4)
            if lab: s.text(x + n * u / 2, y + 16, lab, 11, INK, 'middle')
            x += n * u
        s.text(x + 10, y + 16, note, 11, MUTED)
    s.rect(70, 246 - 14, 13, 14, RED, None, 3, 1.4); s.text(90, 244, '= 부호 1비트 · 한 칸 = 1비트', 11.5, MUTED)
    s.save('d1-formats.svg')


def d1_affine():
    s = Svg(700, 250)
    s.text(350, 22, '[−1, 3] 을 UINT8 0 ~ 255 에 겹치기 — scale = 4/255 ≈ 0.01569, zp = 64', 14, INK, 'middle', True)
    P = Plot(s, 60, 50, 580, 1, (-1, 3), (0, 1))
    s.line(60, 70, 640, 70, INK, 1.6); s.text(30, 74, '실수 x', 12, INK, 'middle', True)
    s.line(60, 170, 640, 170, INK, 1.6); s.text(30, 174, '정수 q', 12, INK, 'middle', True)
    sc, zp = 4 / 255, 64
    for xv in [-1, 0, 0.5, 1.234, 3]:
        q = int(np.clip(round(xv / sc) + zp, 0, 255))
        X1, X2 = P.X(xv), 60 + q / 255 * 580
        c = RED if xv == 0 else BLUE
        s.line(X1, 74, X2, 166, c, 1.6, True)
        s.circle(X1, 70, 4, c); s.circle(X2, 170, 4, c)
        s.text(X1, 60, f'{xv:g}', 12, INK, 'middle', True)
        s.text(X2, 190, str(q), 12, INK, 'middle', True)
    s.text(P.X(0) + 8, 120, '0.0 → zp 칸 (정확히)', 11.5, RED)
    s.text(60, 218, 'quantize: q = clip(round(x / scale) + zp, 0, 255)', 12, INK)
    s.text(400, 218, 'dequantize: x̂ = (q − zp) × scale', 12, INK)
    s.text(60, 238, '범위 안: |오차| ≤ scale/2 = 0.00784   ·   범위 밖 4.0 → 255 → 2.996 (잘림 오차 1.004)', 12, MUTED)
    s.save('d1-affine.svg')


def d1_symasym():
    M = model(); A = M['A']
    s = Svg(700, 270)
    s.text(350, 22, 'ReLU 출력(모두 ≥ 0)에 대칭을 쓰면 음수 칸 127개가 버려진다', 14, INK, 'middle', True)
    amax = float(A.max())
    for k, (title, lo, nq, sc, col, note) in enumerate([
            ('비대칭 UINT8 — 0 ~ 2.137 을 256 칸', 0.0, 256, amax / 255, GREEN, f'scale {amax / 255:.5f} · MSE 2.49e-06'),
            ('대칭 INT8 — −2.137 ~ 2.137 을 255 칸', -amax, 255, amax / 127, RED, f'scale {amax / 127:.5f} · MSE 1.26e-05 (5.1 배)')]):
        y0 = 48 + k * 110
        s.text(30, y0, title, 12.5, INK, False, True)
        P = Plot(s, 60, y0 + 10, 600, 60, (-amax, amax), (0, 1))
        hist, edges = np.histogram(A[A > 0], bins=60, range=(0, amax))
        h = np.log10(hist + 1); h = h / h.max()
        for i in range(60):
            s.rect(P.X(edges[i]), y0 + 70 - h[i] * 50, P.X(edges[1]) - P.X(0), h[i] * 50, BLUE, FILL[BLUE], 0, 0.5)
        s.line(60, y0 + 70, 660, y0 + 70, INK, 1.2)
        s.rect(P.X(lo), y0 + 74, P.X(amax) - P.X(lo), 10, col, None, 2, 1.2)
        if lo < 0: s.text(P.X(-amax / 2), y0 + 100, '← 쓰이지 않는 칸 (값이 없음)', 11.5, RED, 'middle')
        s.text(660, y0 + 100, note, 11.5, MUTED, 'end')
    s.text(P.X(0), 262, '0', 11, MUTED, 'middle')
    s.save('d1-symasym.svg')


def d1_perch():
    M = model(); W = M['W']
    sc = np.abs(W.reshape(30, -1)).max(1) / 127
    s = Svg(700, 240)
    s.text(350, 22, 'conv 필터 30개의 알맞은 scale (= max|W| / 127) — per-tensor 는 가장 큰 것 하나로 통일', 14, INK, 'middle', True)
    P = Plot(s, 70, 45, 600, 160, (0, 30), (0, 0.0042))
    P.axes(yt=(0, 0.001, 0.002, 0.003, 0.004), grid=True)
    for i, v in enumerate(sc):
        s.rect(P.X(i) + 3, P.Y(v), 14, P.Y(0) - P.Y(v), BLUE, None, 1, 1.1)
    s.line(70, P.Y(sc.max()), 670, P.Y(sc.max()), RED, 2, dash='6 4')
    s.text(670, P.Y(sc.max()) - 6, f'per-tensor scale {sc.max():.5f}', 11.5, RED, 'end')
    s.text(370, 228, f'필터 번호 0 ~ 29   ·   가장 작은 scale {sc.min():.5f} (2.4배 차이)   ·   per-channel MSE 가 1.91배 작다', 11.5, MUTED, 'middle')
    s.save('d1-perch.svg')


def d1_clip():
    M = model(); A = M['A']; amax = float(A.max())

    def mse_clip(clip, bits):
        qm = 2 ** bits - 1; sc = clip / qm
        return mse(np.clip(np.round(A / sc), 0, qm) * sc, A)
    r = np.linspace(0.05, 1.0, 96)
    s = Svg(700, 300)
    s.text(350, 22, 'clip 범위를 좁히면 — 반올림 오차 ↓ · 잘림 오차 ↑ (MSE 로그 눈금)', 14, INK, 'middle', True)
    P = Plot(s, 90, 45, 550, 190, (0, 1), (-6, -1))
    for t in (0, 0.25, 0.5, 0.75, 1):
        s.line(P.X(t), 45, P.X(t), 235, '#e2e8f0', 1); s.text(P.X(t), 252, f'{t:g}', 12, MUTED, 'middle')
    for t in (-6, -5, -4, -3, -2, -1):
        s.line(90, P.Y(t), 640, P.Y(t), '#e2e8f0', 1); s.text(84, P.Y(t) + 4, f'1e{t}', 11.5, MUTED, 'end')
    s.line(90, 235, 640, 235, INK, 1.3); s.line(90, 45, 90, 235, INK, 1.3)
    for bits, col, dy in ((8, BLUE, -26), (4, ORANGE, -14)):
        ys = [math.log10(mse_clip(c * amax, bits)) for c in r]
        s.poly([(P.X(x), P.Y(max(min(y, -1), -6))) for x, y in zip(r, ys)], col, 2.5)
        k = int(np.argmin(ys)); s.circle(P.X(r[k]), P.Y(ys[k]), 5, col)
        s.text(P.X(r[k]), P.Y(ys[k]) + dy, f'{bits}bit 최적 × {r[k]:.2f}', 11.5, col, 'end' if bits == 8 else 'middle', True)
    p = float(np.percentile(A, 99.9)) / amax
    s.line(P.X(p), 45, P.X(p), 235, PURPLE, 1.6, dash='5 4'); s.text(P.X(p) - 4, 58, '99.9 백분위 (× 0.75)', 11, PURPLE, 'end')
    s.text(365, 272, 'clip / max  (max = 2.137) — 가로 왼쪽일수록 범위를 좁게', 12, INK, 'middle')
    s.text(20, 140, 'MSE', 12, INK)
    s.text(365, 292, '비트가 적을수록(4bit) 칸이 거칠어 → 더 많이 좁히는 것이 이득', 11.5, MUTED, 'middle')
    s.save('d1-clip.svg')


def d1_intmm():
    s = Svg(700, 230)
    s.text(350, 22, '정수 행렬곱 — INT8 × INT8 → INT32 누적 → + INT32 bias → × scale 한 번', 14, INK, 'middle', True)
    boxes = [(20, 'x_q', 'INT8 (100, 100)', BLUE), (20, 'W_q', 'INT8 (10, 100)', BLUE)]
    s.box(10, 50, 110, 46, 'x_q', BLUE, 'INT8 · sx 0.1136')
    s.box(10, 120, 110, 46, 'W_q', BLUE, 'INT8 · sw 0.00205')
    s.box(150, 80, 172, 56, 'acc = x_q @ W_qᵀ', ORANGE, 'INT32 · −64,612 ~ 65,677', 13)
    s.box(350, 80, 168, 56, '+ b_q', ORANGE, 'INT32 = round(b / (sx·sw))', 13)
    s.box(546, 50, 148, 46, '× sx·sw → 실수 y', GREEN, '차이 ≤ 0.10 · 예측 100%', 13)
    s.box(546, 120, 148, 46, '× M → INT8 y_q', PURPLE, 'M = sx·sw/sy ≈ 0.00192', 13)
    s.line(120, 73, 148, 100, MUTED, 2, True); s.line(120, 143, 148, 116, MUTED, 2, True)
    s.line(322, 108, 348, 108, MUTED, 2, True)
    s.line(518, 100, 544, 75, MUTED, 2, True); s.line(518, 116, 544, 141, MUTED, 2, True)
    s.text(350, 195, '⚠ INT16 최대 32,767 도 넘는다 → 누적은 INT32. NumPy int8 @ int8 은 int8 로 조용히 넘침', 12, RED, 'middle')
    s.text(350, 215, '대칭(zp = 0)이라 괄호 안이 전부 정수 — 비대칭이면 zp × (W 행 합) 보정 항이 더 필요', 11.5, MUTED, 'middle')
    s.save('d1-intmm.svg')


def d1_results():
    M = model(); torch = M['torch']; cnn = M['cnn']; X = M['X']; labels = M['labels']

    def fq(W, bits, pc):
        qm = 2 ** (bits - 1) - 1
        sc = (np.abs(W.reshape(W.shape[0], -1)).max(1) / qm).reshape((-1,) + (1,) * (W.ndim - 1)) if pc else np.abs(W).max() / qm
        return (np.clip(np.round(W / sc), -qm, qm) * sc).astype(W.dtype)
    rows = []
    for b in (8, 4, 2):
        for pc in (False, True):
            m = copy.deepcopy(cnn)
            for L in (m.conv, m.fc1, m.fc2): L.weight.data = torch.from_numpy(fq(L.weight.detach().numpy(), b, pc))
            with torch.no_grad(): lg = m(X).numpy()
            rows.append((b, pc, float((lg.argmax(1) == labels).mean()), float(np.abs(lg - M['logits']).max())))
    s = Svg(700, 260)
    s.text(350, 22, 'weight fake quant 6가지 — 정확도 (MNIST 100장, FP32 = 1.00)', 14, INK, 'middle', True)
    P = Plot(s, 70, 45, 600, 150, (0, 6), (0, 1))
    P.axes(yt=(0, 0.25, 0.5, 0.75, 1), grid=True)
    for i, (b, pc, acc, md) in enumerate(rows):
        c = GREEN if pc else BLUE
        s.rect(P.X(i) + 20, P.Y(acc), 60, P.Y(0) - P.Y(acc), c, None, 2, 1.3)
        s.text(P.X(i) + 50, P.Y(acc) - 6, f'{acc:.2f}', 12, INK, 'middle', True)
        s.text(P.X(i) + 50, 214, f'{b}bit {"채널" if pc else "텐서"}', 11.5, INK, 'middle')
        s.text(P.X(i) + 50, 230, f'max_diff {md:.2f}', 10.5, MUTED, 'middle')
    s.text(350, 252, '8bit 는 손실 0 · 4bit 는 정확도 그대로지만 logits 가 1.3 ~ 1.6 흔들림 · 2bit 는 무너짐 (per-channel 이 덜)', 11.5, MUTED, 'middle')
    s.save('d1-results.svg')


# ---------------- QP Day 2 ----------------
# 숫자는 교안 노트북(QP_Day2_lecture.ipynb)을 CPU · 스레드 1 · onnxruntime 1.32 로 직접 실행한 값.
# 지연 시간은 컴퓨터마다 다르다 — 그림에는 "비율" 이 보이도록 그린다.

D2_PERF = [  # (이름, 크기 KB, 정확도, 배치1 mean ms, 배치32 장/초)
    ('FP32', 1070.8, 0.8460, 0.701, 1244.7),
    ('dynamic', 310.2, 0.8445, 6.183, 170.3),
    ('QDQ U8/S8', 321.8, 0.8450, 0.480, 2757.9),
    ('QDQ S8/S8', 322.1, 0.8450, 1.113, 986.9),
    ('QOperator U8/S8', 296.9, 0.8450, 0.342, 3016.6)]

D2_PC_RATIO = [  # per-channel 모델, 활성값 34개의 Percentile scale / MinMax scale (QuantizeLinear 순서)
    1.0, 0.77, 0.86, 0.853, 0.802, 0.76, 0.821, 0.798, 0.665, 0.799, 0.823, 0.79, 0.809, 0.845, 0.687, 0.729, 0.907,
    0.805, 0.807, 0.829, 0.775, 0.795, 0.79, 0.809, 0.76, 0.675, 0.764, 0.762, 0.771, 0.926, 0.867, 1.0, 1.0, 1.0]

D2_GRID = [  # (fmt, act, method, pc, acc, mean_ms) — 미니 프로젝트 16가지 조합
    ('QDQ', 'U8', 'MinMax', False, 0.8460, 0.341), ('QOperator', 'U8', 'MinMax', True, 0.8450, 0.342),
    ('QDQ', 'U8', 'Percentile', True, 0.8435, 0.353), ('QOperator', 'U8', 'Percentile', False, 0.8465, 0.359),
    ('QOperator', 'U8', 'Percentile', True, 0.8435, 0.361), ('QDQ', 'U8', 'Percentile', False, 0.8465, 0.409),
    ('QOperator', 'U8', 'MinMax', False, 0.8460, 0.417), ('QDQ', 'U8', 'MinMax', True, 0.8450, 0.492),
    ('QDQ', 'S8', 'Percentile', True, 0.8440, 1.071), ('QDQ', 'S8', 'MinMax', False, 0.8455, 1.098),
    ('QDQ', 'S8', 'MinMax', True, 0.8450, 1.106), ('QDQ', 'S8', 'Percentile', False, 0.8465, 1.306),
    ('QOperator', 'S8', 'Percentile', True, 0.8435, 4.544), ('QOperator', 'S8', 'MinMax', False, 0.8460, 4.598),
    ('QOperator', 'S8', 'MinMax', True, 0.8450, 4.962), ('QOperator', 'S8', 'Percentile', False, 0.8465, 5.131)]


def d2_ptq():
    s = Svg(700, 250)
    s.text(350, 22, 'PTQ 두 갈래 — 활성값의 scale 을 언제 정하나', 14, INK, 'middle', True)
    s.text(20, 58, 'Dynamic', 14, BLUE, 'start', True)
    s.box(20, 68, 150, 50, 'rn20.onnx', BLUE, 'FP32 1,071 KB')
    s.box(205, 68, 200, 50, 'quantize_dynamic', BLUE, 'weight 만 미리 INT8', 13.5)
    s.box(440, 68, 240, 50, '실행할 때마다 입력 min · max', BLUE, '→ 그때 scale 계산 (보정 X)', 13)
    s.line(170, 93, 203, 93, MUTED, 2, True); s.line(405, 93, 438, 93, MUTED, 2, True)
    s.text(20, 152, 'Static', 14, ORANGE, 'start', True)
    s.box(20, 162, 150, 50, 'rn20_pre.onnx', ORANGE, '+ 보정 500장')
    s.box(205, 162, 200, 50, 'quantize_static', ORANGE, 'CalibrationDataReader', 13.5)
    s.box(440, 162, 240, 50, '활성값 scale · zp 를 파일에', ORANGE, '실행 중에는 계산 없음', 13)
    s.line(170, 187, 203, 187, MUTED, 2, True); s.line(405, 187, 438, 187, MUTED, 2, True)
    s.text(350, 238, 'dynamic = MatMul · Linear 위주(LSTM · Transformer)   ·   static = Conv 위주(CNN) 의 표준', 12, MUTED, 'middle')
    s.save('d2-ptq.svg')


def d2_dynamic():
    s = Svg(700, 250)
    s.text(350, 22, 'dynamic 모델의 Conv 하나 = 실행할 때마다 5단계', 14, INK, 'middle', True)
    steps = [('min · max', '입력 범위 재기'), ('DynamicQuantize', '활성값 → UINT8'), ('ConvInteger', '정수 Conv'),
             ('Cast', 'INT32 → FP32'), ('Mul', '× scale')]
    for i, (a, b) in enumerate(steps):
        x = 12 + i * 138
        s.box(x, 45, 124, 52, a, BLUE if i != 2 else GREEN, b, 13)
        if i: s.line(x - 14, 71, x - 2, 71, MUTED, 2, True)
    P = Plot(s, 150, 120, 500, 80, (0, 300), (0, 2))
    for j, (name, n, ms, c) in enumerate([('FP32', 52, 0.68, ORANGE), ('dynamic', 181, 6.08, BLUE)]):
        y = 125 + j * 38
        s.text(140, y + 19, name, 13, INK, 'end', True)
        s.rect(150, y, P.X(n) - 150, 26, c, None, 3, 1.3)
        s.text(P.X(n) + 8, y + 18, f'노드 {n}개 · 배치 1 {ms} ms', 12.5, INK)
    s.text(350, 222, '노드가 3.5배 → CPU 에서 9배 느림 (배치 32 도 0.10배 처리량) — 크기는 1/3.5, 정확도 0.846 → 0.8445', 12, MUTED, 'middle')
    s.text(350, 242, '⚠ "INT8 이면 빠르다" 가 아니다 — CNN 은 static 으로', 12, RED, 'middle')
    s.save('d2-dynamic.svg')


def d2_bnfold():
    s = Svg(700, 230)
    s.text(350, 22, 'BatchNorm folding — Conv → BN 두 단계를 Conv 하나로', 14, INK, 'middle', True)
    s.box(20, 50, 110, 50, 'x', MUTED)
    s.box(160, 50, 130, 50, 'Conv (W)', BLUE, 'bias 없음')
    s.box(320, 50, 150, 50, 'BN (γ, β, μ, σ²)', PURPLE, '채널마다 곱 · 덧셈', 13.5)
    s.line(130, 75, 158, 75, MUTED, 2, True); s.line(290, 75, 318, 75, MUTED, 2, True)
    s.box(520, 50, 160, 50, "Conv (W', b')", GREEN, '같은 출력 · 차이 9.5e-07', 13.5)
    s.line(470, 75, 518, 75, GREEN, 2.5, True)
    s.text(350, 135, "k = γ / √(σ² + ε)   (채널마다 하나)", 14, INK, 'middle', True)
    s.text(350, 160, "W' = W × k (채널 축)   ·   b' = β − μ × k", 14, INK, 'middle', True)
    s.text(350, 192, 'PyTorch BN 모듈 21개 → ONNX BatchNormalization 노드 0개 (eval 모드 export 가 이미 합침)', 12, MUTED, 'middle')
    s.text(350, 212, "ONNX 첫 Conv weight 와 직접 만든 W' 의 최대 차이 0.0", 12, MUTED, 'middle')
    s.save('d2-bnfold.svg')


def d2_qdq():
    s = Svg(700, 270)
    s.text(350, 22, '같은 static INT8 을 그래프에 담는 두 방식', 14, INK, 'middle', True)
    s.text(20, 52, 'QDQ — 원래 Conv 앞뒤에 Q · DQ 를 끼운다', 13.5, BLUE, 'start', True)
    s.box(20, 62, 90, 32, 'Q', BLUE, None, 14)
    s.box(140, 62, 90, 32, 'DQ', BLUE, None, 14)
    s.box(300, 62, 140, 32, 'Conv', ORANGE, None, 14)
    s.box(470, 62, 90, 32, 'Q', BLUE, None, 14)
    s.box(140, 108, 130, 32, 'INT8 W → DQ', BLUE, None, 13)
    s.line(110, 78, 138, 78, MUTED, 2, True); s.line(230, 78, 298, 78, MUTED, 2, True); s.line(440, 78, 468, 78, MUTED, 2, True)
    s.line(270, 120, 330, 96, MUTED, 1.8, True)
    s.text(580, 74, '노드 145개', 13, INK, 'start', True)
    s.text(580, 92, 'DQ 78 · Q 34', 11.5, MUTED)
    s.text(580, 108, 'Conv 21 · Add 9', 11.5, MUTED)
    s.text(20, 172, 'QOperator — Conv 자체를 정수 연산자로', 13.5, GREEN, 'start', True)
    s.box(20, 182, 110, 34, 'Q (입력 1번)', GREEN, None, 13)
    s.box(160, 182, 170, 34, 'QLinearConv', GREEN, None, 14)
    s.box(360, 182, 150, 34, 'QLinearAdd', GREEN, None, 14)
    s.line(130, 199, 158, 199, MUTED, 2, True); s.line(330, 199, 358, 199, MUTED, 2, True)
    s.text(580, 194, '노드 35개', 13, INK, 'start', True)
    s.text(580, 212, 'QLinearConv 21', 11.5, MUTED)
    s.text(580, 228, 'QLinearAdd 9', 11.5, MUTED)
    s.text(350, 252, '정확도는 둘 다 0.845 · QDQ = TensorRT 등의 표준 입력 · QOperator = ORT CPU 가 그대로 실행', 12, MUTED, 'middle')
    s.save('d2-qdq.svg')


def d2_scales():
    r = D2_PC_RATIO
    s = Svg(700, 250)
    s.text(350, 22, f'Percentile 이 고른 scale ÷ MinMax scale — 활성값 34개 (점선 = 평균 {float(np.mean(D2_PC_RATIO)):.3f})', 14, INK, 'middle', True)
    P = Plot(s, 60, 40, 620, 150, (0, 34), (0, 1.05))
    P.axes(yt=(0, 0.25, 0.5, 0.75, 1.0), grid=True)
    k = int(np.argmin(r))
    for i, v in enumerate(r):
        c = RED if i == k else (MUTED if v >= 0.999 else ORANGE)
        s.rect(P.X(i) + 2.5, P.Y(v), 13, P.Y(0) - P.Y(v), c, None, 1, 1.1)
    m = float(np.mean(r))
    s.line(60, P.Y(m), 680, P.Y(m), BLUE, 2, dash='6 4')
    s.text(P.X(k) + 9, P.Y(r[k]) - 6, f'{r[k]:.3f}', 11.5, RED, 'middle', True)
    s.text(370, 212, '왼쪽 끝 input · 오른쪽 끝 pool · Flatten · logits 는 1.0 (잘리지 않음)   ·   빨강 = layer1.2 Relu (67% 로 축소)', 11.5, MUTED, 'middle')
    s.text(370, 232, 'Entropy 는 34개 모두 MinMax 와 같은 scale (비율 1.000) — 정확도도 MinMax 와 똑같음', 11.5, MUTED, 'middle')
    s.save('d2-scales.svg')


def d2_perf():
    s = Svg(700, 270)
    s.text(350, 22, '5개 모델 성능표 — CPU · 스레드 1 (지연 시간은 컴퓨터마다 다름)', 14, INK, 'middle', True)
    heads = [('크기 KB', 1, 1100), ('배치 1 지연 ms', 3, 6.5), ('배치 32 장/초', 4, 3200)]
    for j, (h, idx, mx) in enumerate(heads):
        x0 = 150 + j * 185
        s.text(x0 + 80, 50, h, 12.5, INK, 'middle', True)
        for i, row in enumerate(D2_PERF):
            y = 62 + i * 38
            if j == 0: s.text(140, y + 18, row[0], 12.5, INK, 'end', True)
            v = row[idx]; w = 115 * v / mx
            c = [ORANGE, BLUE, GREEN, PURPLE, GREEN][i]
            s.rect(x0, y + 4, max(w, 2), 20, c, None, 3, 1.2)
            s.text(x0 + max(w, 2) + 5, y + 19, f'{v:g}', 11.5, INK)
    s.text(350, 262, '정확도: FP32 0.846 · dynamic 0.8445 · 나머지 셋 0.845 — 차이는 2,000장 중 1 ~ 3장', 12, MUTED, 'middle')
    s.save('d2-perf.svg')


def d2_grid():
    s = Svg(700, 290)
    s.text(350, 22, '미니 프로젝트 16가지 PTQ 설정 — 배치 1 지연 시간 (규칙: 손실 1%p 이내 중 가장 빠른 것)', 13.5, INK, 'middle', True)
    P = Plot(s, 70, 45, 600, 172, (0, 5.5), (0, 16))
    P.axes(xt=(0, 1, 2, 3, 4, 5), grid=True)
    for i, (f, a, m, pc, acc, ms) in enumerate(D2_GRID):
        y = 50 + i * 10.4
        c = (GREEN if f == 'QOperator' else BLUE) if a == 'U8' else (RED if f == 'QOperator' else ORANGE)
        s.rect(70, y, P.X(ms) - 70, 8, c, None, 1, 1)
    s.text(P.X(0.45), 58, '★ 선택: QDQ U8 MinMax per-tensor — 0.341 ms · acc 0.846', 11.5, INK, 'start', True)
    s.text(P.X(0.65), 100, '활성값 U8 8개 — 0.34 ~ 0.49 ms (파랑 QDQ · 초록 QOperator)', 11.5, GREEN, 'start', True)
    s.text(P.X(1.45), 146, 'QDQ S8 — 1.07 ~ 1.31 ms', 11.5, ORANGE, 'start', True)
    s.text(P.X(1.45), 170, 'QOperator S8 — 4.5 ~ 5.1 ms', 11.5, RED, 'start', True)
    s.text(370, 256, '가로: 배치 1 지연 시간 (ms) · 빠른 순으로 위에서부터', 12, MUTED, 'middle')
    s.text(370, 276, '16개 정확도 0.8435 ~ 0.8465 (0.3%p = 6장) · 속도를 가르는 것은 활성값 U8 / S8', 12, MUTED, 'middle')
    s.save('d2-grid.svg')

if __name__ == '__main__':
    prefixes = tuple(sys.argv[1:]) or ('d1_',)
    for name, fn in list(globals().items()):
        if callable(fn) and name.startswith(prefixes):
            try: fn()
            except ImportError as e: print('  건너뜀', name, '—', e)
