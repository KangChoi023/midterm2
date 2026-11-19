#!/usr/bin/env python3
"""QP Day 5 (Pruning) 노트 "상세 학습" 의 그림(SVG) → assets/fig/qp/d5-*.svg

    STUDY_ROOT=<AI_CAMPUS 의 부모> python3 -I tools/fig_qp_d5.py        # 전부
    python3 -I tools/fig_qp_d5.py d5_hist                               # 함수 이름 앞부분으로 일부만

그리기 도구(Svg · Plot)는 tools/fig_nn.py 의 것을 쓴다. 숫자는 강의 노트북과 같은 코드를 CPU 로 직접 실행한 값
(정확도 · MACs · 크기 · ORT CPU 지연) — d5_hist 만 교안의 data/resnet20_cifar10.pt 를 torch 로 읽어 다시 계산한다
(torch 가 없으면 그 그림만 건너뛴다). Day 4 그림(tools/fig_qp.py)과 충돌하지 않게 파일을 나눴다.
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_nn
from fig_nn import Svg, Plot, INK, MUTED, LINE, BLUE, ORANGE, GREEN, RED, PURPLE, FILL

fig_nn.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'fig', 'qp')
ROOT = os.environ.get('STUDY_ROOT', os.path.expanduser('~/c_study'))
DATA = os.path.join(ROOT, 'AI_CAMPUS', 'QP-day5', 'data')

# 직접 실행 값 (CPU · torch 2.15 · torch-pruning 1.6.0 · 평가 2,000장)
US = [(0.0, 0.846, 0.846), (0.3, 0.8415, 0.838), (0.5, 0.8125, 0.789), (0.7, 0.482, 0.2945), (0.9, 0.100, 0.100)]
GLOBAL50 = [('fc', 0.058), ('stem.0', 0.125), ('L2.0.sc', 0.135), ('L3.0.sc', 0.226), ('L1.0.c1', 0.268),
            ('L2.1.c1', 0.350), ('L3.0.c1', 0.376), ('L3.1.c2', 0.567), ('L3.2.c1', 0.648), ('L3.2.c2', 0.801)]
FT_ONE = [0.100, 0.632, 0.712, 0.7225]
FT_ITER = [(1, 27.67, 0.8135), (2, 17.34, 0.7465), (3, 10.53, 0.68)]
DEPLOY = [('dense', 1070.8, 0.846, 0.703), ('unstructured 50%', 1070.8, 0.8125, 0.703), ('structured 50% + FT', 276.6, 0.7225, 0.341)]
SWEEP = [(0.0, 41.24, 0.846, 0.842, 0.704), (0.2, 24.89, 0.2415, 0.8195, 0.839),
         (0.4, 14.19, 0.101, 0.7595, 0.629), (0.6, 6.15, 0.100, 0.603, 0.342)]
FP32 = 0.846


def d5_hist():
    import torch
    sd = torch.load(os.path.join(DATA, 'resnet20_cifar10.pt'), map_location='cpu')
    w = np.concatenate([v.abs().flatten().numpy() for k, v in sd.items() if k.endswith('weight') and v.dim() == 4])
    s = Svg(700, 270)
    s.text(350, 22, f'ResNet-20 conv weight |w| 분포 — {len(w):,}개 (세로축 로그)', 14, INK, 'middle', True)
    hi = float(w.max()); cnt, edges = np.histogram(w, bins=60, range=(0, hi))
    P = Plot(s, 70, 40, 600, 170, (0, hi), (0, 5))
    P.axes(xt=[round(x, 2) for x in np.arange(0, hi, 0.1)], yt=(0, 1, 2, 3, 4, 5))
    s.text(680, 228, '|w|', 12, INK, 'end')
    s.text(30, 50, 'log10', 11, MUTED)
    s.text(30, 64, '(개수)', 11, MUTED)
    for c, a, b in zip(cnt, edges[:-1], edges[1:]):
        if c <= 0: continue
        y = math.log10(c + 1)
        s.rect(P.X(a), P.Y(y), max(P.X(b) - P.X(a) - 0.6, 1), P.Y(0) - P.Y(y), BLUE, None, 0, 0.8)
    for t, col, dy in [(0.01, RED, 0), (0.02, ORANGE, 16), (0.05, PURPLE, 32)]:
        frac = float((w < t).mean())
        s.line(P.X(t), 40, P.X(t), 210, col, 1.6, dash='5 3')
        s.text(P.X(0.45), 66 + dy, f'|w| < {t:g} : {frac * 100:.0f}%', 12.5, col, 'start', True)
    s.text(350, 258, '0 근처에 몰린 종 모양 → 작은 것부터 지워도 출력이 거의 안 변한다 (magnitude 기준의 근거)', 12, MUTED, 'middle')
    s.save('d5-hist.svg')


def d5_units():
    s = Svg(700, 300)
    s.text(350, 22, '가지치기 단위 — layer2[0].conv1 (32 × 16 × 3 × 3) 에서 하나를 지우면', 14, INK, 'middle', True)
    rows = [('weight 1개', 'unstructured', 1, '그대로 (0 도 곱함)', MUTED),
            ('2:4 묶음', 'semi-structured', 2, 'Ampere+ 전용 하드웨어만', ORANGE),
            ('filter (출력 채널)', 'structured', 144, '어디서나 빨라짐', GREEN),
            ('입력 채널', 'structured', 288, '어디서나 빨라짐', GREEN),
            ('layer 전체', 'depth', 4608, '빨라짐 (정확도 손해 매우 큼)', BLUE)]
    s.text(150, 50, '단위', 12.5, INK, 'end', True); s.text(165, 50, '지워지는 weight 수 (로그 막대)', 12.5, INK, 'start', True)
    s.text(590, 50, '실제 속도', 12.5, INK, 'middle', True)
    for i, (u, kind, n, sp, c) in enumerate(rows):
        y = 62 + i * 42
        s.text(150, y + 16, u, 13, INK, 'end', True)
        s.text(150, y + 31, kind, 11, MUTED, 'end')
        wbar = 30 + 230 * math.log10(n) / math.log10(4608)
        s.rect(165, y + 4, wbar, 24, c, None, 3, 1.4)
        s.text(165 + wbar + 6, y + 21, f'{n:,}', 12.5, INK, 'start', True)
        s.text(590, y + 21, sp, 12, c if c != MUTED else INK, 'middle')
    s.text(350, 290, '단위가 클수록 정확도 손해 ↑ · 실제 속도 이득 ↑ — 실무에서 속도를 얻는 것은 structured', 12, MUTED, 'middle')
    s.save('d5-units.svg')


def d5_us():
    s = Svg(700, 300)
    s.text(350, 22, 'Unstructured pruning (fine-tuning 없음) — global vs layer 별', 14, INK, 'middle', True)
    P = Plot(s, 60, 50, 280, 190, (0, 0.9), (0, 0.9))
    P.axes(xt=(0, 0.3, 0.5, 0.7, 0.9), yt=(0, 0.2, 0.4, 0.6, 0.8), xl='sparsity')
    s.text(65, 46, '정확도', 12, INK)
    s.line(P.X(0), P.Y(0.1), P.X(0.9), P.Y(0.1), MUTED, 1, dash='3 3')
    s.text(P.X(0.62), P.Y(0.1) - 5, '무작위 0.1', 10.5, MUTED, 'middle')
    for idx, c, lab in [(1, BLUE, 'global'), (2, ORANGE, 'layer 별')]:
        pts = [(P.X(r[0]), P.Y(r[idx])) for r in US]
        s.poly(pts, c, 2.5, dash=None if idx == 1 else '6 4')
        for (x, y), r in zip(pts, US): s.circle(x, y, 4, c)
    s.text(P.X(0.5) + 8, P.Y(0.8125) - 6, '0.8125', 11, BLUE, 'start', True)
    s.text(P.X(0.5) - 8, P.Y(0.789) + 14, '0.789', 11, ORANGE, 'end', True)
    s.text(P.X(0.7) + 8, P.Y(0.482), '0.482', 11, BLUE)
    s.text(P.X(0.7) + 8, P.Y(0.2945) + 4, '0.2945', 11, ORANGE)
    s.line(200, 268, 225, 268, BLUE, 2.5); s.text(230, 272, 'global', 12, INK)
    s.line(285, 268, 310, 268, ORANGE, 2.5, dash='6 4'); s.text(315, 272, 'layer 별', 12, INK)
    # 오른쪽: global 50% 에서 층별 sparsity
    s.text(530, 50, 'global 50% — 층마다 지운 비율', 12.5, INK, 'middle', True)
    for i, (n, v) in enumerate(GLOBAL50):
        y = 62 + i * 21
        s.text(440, y + 13, n, 11, INK, 'end')
        c = GREEN if v < 0.3 else (RED if v > 0.6 else BLUE)
        s.rect(446, y + 2, 200 * v, 15, c, None, 2, 1)
        s.text(450 + 200 * v, y + 14, f'{v:.3f}', 10.5, INK)
    s.text(530, 290, '중요한 fc · stem 은 덜, 마지막 블록은 더 지운다', 11.5, MUTED, 'middle')
    s.save('d5-us.svg')


def d5_sparse():
    s = Svg(700, 300)
    s.text(350, 22, '0 이 90% 여도 — 저장 크기 · 지연은 그대로, 압축 크기만 준다', 14, INK, 'middle', True)
    groups = [('저장 KB', 1109.7, 1109.7, 1200, '{:,.1f}'), ('gzip KB', 1003.3, 191.7, 1200, '{:,.1f}'), ('지연 ms (CPU)', 2.02, 2.17, 2.4, '{:.2f}')]
    for j, (h, a, b, mx, fmt) in enumerate(groups):
        x0 = 40 + j * 140
        s.text(x0 + 55, 52, h, 12.5, INK, 'middle', True)
        for k, (v, c, lab) in enumerate([(a, BLUE, 'dense'), (b, ORANGE, '90%')]):
            hgt = 150 * v / mx
            s.rect(x0 + 12 + k * 50, 215 - hgt, 40, hgt, c, None, 3, 1.3)
            s.text(x0 + 32 + k * 50, 210 - hgt, fmt.format(v), 10.5, INK, 'middle')
            s.text(x0 + 32 + k * 50, 232, lab, 11, MUTED, 'middle')
    s.text(240, 262, 'gzip: 1,003 → 192 KB (5.2배) · 지연은 잡음 수준 (1스레드 · 직접 실행)', 11.5, MUTED, 'middle')
    # 오른쪽: 2:4 mask
    s.text(560, 52, '2:4 semi-structured', 13, INK, 'middle', True)
    w = [0.3, -0.1, 0.05, 0.6, -0.4, 0.02, 0.2, -0.01]
    m = [1, 0, 0, 1, 1, 0, 1, 0]
    for i, (v, k) in enumerate(zip(w, m)):
        x = 440 + i * 30 + (8 if i >= 4 else 0)
        s.rect(x, 70, 28, 30, GREEN if k else MUTED, '#dcfce7' if k else '#f1f5f9', 3, 1.2)
        s.text(x + 14, 90, f'{v:g}', 10.5, INK, 'middle')
        s.text(x + 14, 122, str(k), 13, GREEN if k else MUTED, 'middle', True)
    s.text(560, 145, '4개마다 |w| 큰 2개만 남김 (mask 1)', 11.5, INK, 'middle')
    s.box(450, 160, 220, 46, 'Ampere+ Sparse Tensor Core', GREEN, '최대 약 2배 · T4 미지원', 12.5)
    s.text(560, 232, 'ResNet-20 (stem 제외) → sparsity 0.498', 11.5, INK, 'middle')
    s.text(560, 250, '정확도 0.6485 (unstructured 50% 0.8125)', 11.5, RED, 'middle', True)
    s.save('d5-sparse.svg')


def d5_struct():
    s = Svg(700, 290)
    s.text(350, 22, 'Structured pruning — 같은 keep 으로 conv1 출력 · bn1 · conv2 입력을 함께', 14, INK, 'middle', True)
    s.box(20, 50, 130, 56, 'conv1', BLUE, 'weight[keep]', 14)
    s.box(185, 50, 130, 56, 'bn1', PURPLE, 'γ β μ σ² [keep]', 14)
    s.box(350, 50, 90, 56, 'ReLU', MUTED, None, 14)
    s.box(475, 50, 150, 56, 'conv2', BLUE, 'weight[:, keep]', 14)
    for a, b in [(150, 183), (315, 348), (440, 473)]: s.line(a, 78, b, 78, MUTED, 2, True)
    s.text(85, 124, '출력 32 → 16', 11.5, GREEN, 'middle', True)
    s.text(250, 124, '32 → 16', 11.5, GREEN, 'middle', True)
    s.text(550, 124, '입력 32 → 16 · 출력 32 그대로', 11.5, GREEN, 'middle', True)
    s.rect(15, 40, 620, 95, GREEN, 'none', 8, 1.2)
    s.text(645, 92, '블록 안쪽', 11.5, GREEN, 'start', True)
    s.text(645, 108, '✅ 자유', 11.5, GREEN, 'start', True)
    # 바깥쪽: residual
    s.box(475, 170, 150, 46, 'out (conv2 · bn2)', RED, '채널 C', 12.5)
    s.box(260, 170, 150, 46, 'shortcut(x)', RED, '채널 C', 12.5)
    s.circle(442, 245, 14, RED); s.text(442, 250, '+', 16, RED, 'middle', True)
    s.line(550, 216, 455, 240, MUTED, 1.8, True); s.line(335, 216, 430, 240, MUTED, 1.8, True)
    s.text(40, 190, '블록 바깥쪽 ⚠', 12.5, RED, 'start', True)
    s.text(40, 208, 'out + shortcut(x) 는', 11.5, INK)
    s.text(40, 224, '채널 수가 같아야 한다', 11.5, INK)
    s.text(350, 280, 'conv2 출력만 8채널로 → The size of tensor a (8) must match the size of tensor b (16)', 11.5, RED, 'middle')
    s.save('d5-struct.svg')


def d5_depgraph():
    s = Svg(700, 300)
    s.text(350, 22, 'DependencyGraph — layer1 통로 채널 하나를 자르면 함께 잘리는 곳 (Conv 9 + BN 4)', 13.5, INK, 'middle', True)
    s.rect(20, 120, 660, 26, ORANGE, '#fef3c7', 6, 1.5)
    s.text(350, 138, 'residual 통로 (stem 출력 16채널) — 덧셈으로 이어져 채널 번호가 모두 같다', 12, INK, 'middle', True)
    s.box(25, 50, 90, 46, 'stem.0', RED, '출력', 13)
    s.line(70, 96, 70, 118, RED, 2, True)
    for i in range(3):
        x = 150 + i * 145
        s.box(x, 50, 120, 46, f'layer1.{i}.conv1', BLUE, '입력 (통로 읽기)', 11.5)
        s.line(x + 30, 118, x + 30, 98, BLUE, 2, True)
        s.box(x, 175, 120, 46, f'layer1.{i}.conv2', RED, '출력 (+ 통로)', 11.5)
        s.line(x + 60, 175, x + 60, 148, RED, 2, True)
    s.box(590, 50, 90, 46, 'layer2.0', BLUE, 'conv1 · sc 입력', 11)
    s.line(635, 118, 635, 98, BLUE, 2, True)
    s.text(350, 248, 'BN 4개 = stem.1 · layer1.0~2 bn2 (통로 채널을 정규화)', 12, PURPLE, 'middle')
    s.text(350, 270, 'idxs=[0, 1] 자르면 → stem 출력 14 · layer2[0] 입력 14 · 정확도 0.846 → 0.634', 12, RED, 'middle', True)
    s.text(350, 290, '빨강 = 출력 채널을 자름 · 파랑 = 입력 채널을 자름', 11, MUTED, 'middle')
    s.save('d5-depgraph.svg')


def d5_ft():
    s = Svg(700, 280)
    s.text(350, 22, 'Fine-tuning 회복 — structured 50% (MACs 25%) · 같은 총 3 epoch', 14, INK, 'middle', True)
    P = Plot(s, 70, 45, 360, 180, (0, 3), (0, 0.9))
    P.axes(xt=(0, 1, 2, 3), yt=(0, 0.2, 0.4, 0.6, 0.8), xl='epoch')
    s.text(75, 41, '정확도', 12, INK)
    s.line(P.X(0), P.Y(FP32), P.X(3), P.Y(FP32), MUTED, 1.2, dash='5 4')
    s.text(P.X(3), P.Y(FP32) - 5, 'FP32 0.846', 10.5, MUTED, 'end')
    pts = [(P.X(i), P.Y(v)) for i, v in enumerate(FT_ONE)]
    s.poly(pts, BLUE, 2.5)
    for (x, y), v in zip(pts, FT_ONE):
        s.circle(x, y, 4, BLUE); s.text(x + 6, y + (-7 if v == FT_ONE[-1] else 15), f'{v:.3f}', 10.5, BLUE)
    ip = [(P.X(st), P.Y(a)) for st, _, a in FT_ITER]
    s.poly([(P.X(0), P.Y(FP32))] + ip, ORANGE, 2.3, dash='6 4')
    for (x, y), (st, mac, a) in zip(ip, FT_ITER):
        s.circle(x, y, 4, ORANGE); s.text(x + 6, y + (16 if st == 3 else -7), f'{a:.3f}', 10.5, ORANGE, 'start', True)
    s.line(110, 258, 135, 258, BLUE, 2.5); s.text(140, 262, 'one-shot: 50% 자르고 3 epoch', 11.5, INK)
    s.line(300, 258, 325, 258, ORANGE, 2.3, dash='6 4'); s.text(330, 262, 'iterative: 1/3 자르고 1 epoch × 3', 11.5, INK)
    s.text(570, 60, '같은 MACs 10.53M', 12.5, INK, 'middle', True)
    s.box(470, 75, 200, 46, 'one-shot 0.7225', BLUE, '0.10 → 0.632 → 0.712', 13)
    s.box(470, 135, 200, 46, 'iterative 0.680', ORANGE, 'MACs 27.7 → 17.3 → 10.5', 13)
    s.text(570, 205, '짧은 예산에서는 one-shot 이 나았다', 11.5, RED, 'middle', True)
    s.text(570, 223, '(CPU 직접 실행 · 난수 · 장치마다 다름)', 10.5, MUTED, 'middle')
    s.save('d5-ft.svg')


def d5_deploy():
    s = Svg(700, 250)
    s.text(350, 22, '배포 비교 — 배치 1 ONNX · ORT CPU (스레드 1 · GPU 없음)', 14, INK, 'middle', True)
    heads = [('ONNX KB', 1, 1200, '{:,.1f}'), ('정확도', 2, 1.0, '{:.4f}'), ('지연 ms', 3, 0.8, '{:.3f}')]
    for j, (h, idx, mx, fmt) in enumerate(heads):
        x0 = 220 + j * 160
        s.text(x0 + 55, 52, h, 12.5, INK, 'middle', True)
        for i, row in enumerate(DEPLOY):
            y = 64 + i * 44
            if j == 0: s.text(210, y + 19, row[0], 12.5, INK, 'end', True)
            v = row[idx]; w = 100 * v / mx
            c = [BLUE, ORANGE, GREEN][i]
            s.rect(x0, y + 5, max(w, 2), 22, c, None, 3, 1.2)
            s.text(x0 + max(w, 2) + 5, y + 21, fmt.format(v), 11.5, INK)
    s.text(350, 212, 'unstructured: Conv weight 270,256개 그대로 (0 이 50.1%) → 크기 · 속도 dense 와 같음 (1.00배)', 11.5, INK, 'middle')
    s.text(350, 234, 'structured + FT: Conv weight 67,672개 → 크기 1/4 · 2.06배 (MACs 비율 3.9배만큼은 아님)', 11.5, GREEN, 'middle', True)
    s.save('d5-deploy.svg')


def d5_sweep():
    s = Svg(700, 300)
    s.text(350, 22, '가지치기 비율 곡선 — FT 뒤 정확도 · 점선 = FP32 − 3%p (0.816)', 14, INK, 'middle', True)
    for k, (title, xi, xr, xt) in enumerate([('MACs (M)', 1, (0, 45), (0, 10, 20, 30, 40)), ('지연 ms (ORT CPU)', 4, (0.3, 0.9), (0.3, 0.5, 0.7, 0.9))]):
        P = Plot(s, 60 + k * 340, 45, 270, 190, xr, (0.55, 0.88))
        P.axes(xt=xt, yt=(0.6, 0.7, 0.8), xl=title)
        s.line(P.x0, P.Y(FP32 - 0.03), P.x0 + P.w, P.Y(FP32 - 0.03), RED, 1.3, dash='5 4')
        pts = [(P.X(r[xi]), P.Y(r[3])) for r in SWEEP]
        s.poly(pts, BLUE, 2.3)
        for (x, y), r in zip(pts, SWEEP):
            best = r[0] == 0.2
            s.circle(x, y, 5 if best else 4, GREEN if best else BLUE)
            dx, dy, anc = (-8, 18, 'end') if (k == 1 and r[0] == 0.2) else ((8, 16, 'start') if (k == 1 and r[0] == 0.4) else (8, -6, 'start'))
            s.text(x + dx, y + dy, f'{int(r[0] * 100)}% · {r[3]:.4g}', 10.5, GREEN if best else INK, anc, best)
    s.text(350, 270, 'best_ratio 0.2 — MACs 1.66배 ↓ · 하락 2.65%p · 그러나 ORT CPU 지연 0.704 → 0.839 ms (채널 12 · 25 · 51)', 11.5, INK, 'middle')
    s.text(350, 290, '40% · 60% 는 빨라지지만 (0.629 · 0.342 ms) 정확도 기준 밖', 11.5, MUTED, 'middle')
    s.save('d5-sweep.svg')


if __name__ == '__main__':
    prefixes = tuple(sys.argv[1:]) or ('d5_',)
    for name, fn in list(globals().items()):
        if callable(fn) and name.startswith(prefixes):
            try: fn()
            except ImportError as e: print('  건너뜀', name, '—', e)
