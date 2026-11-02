#!/usr/bin/env python3
"""NN 노트 "상세 학습" 의 그림(SVG)을 만든다 → assets/fig/nn/*.svg

    python3 -I tools/fig_nn.py          # 전부 다시 만들기

글자 그림(```text)은 글꼴에 따라 깨지므로, 그림은 도형 + 진짜 글자인 SVG 로 그린다.
그래프 · 점 그림은 강의와 같은 식 · 같은 난수(seed)로 계산해 그린다. 배경은 흰색(다크 모드에서도 흰 패널로 보임).
d2-perceptron.svg · d2-boundary-and.svg 두 장은 손으로 쓴 SVG 라 이 스크립트가 만들지 않는다.
"""
import math, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'fig', 'nn')
FONT = 'IBM Plex Sans KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif'
INK, MUTED, LINE = '#1e293b', '#64748b', '#94a3b8'
BLUE, ORANGE, GREEN, RED, PURPLE = '#0284c7', '#d97706', '#16a34a', '#dc2626', '#7c3aed'
FILL = {BLUE: '#e0f2fe', ORANGE: '#fef3c7', GREEN: '#dcfce7', RED: '#fee2e2', PURPLE: '#ede9fe'}


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.items = w, h, []

    def add(self, s): self.items.append(s); return self

    def text(self, x, y, s, size=14, color=INK, anchor='start', bold=False, italic=False):
        s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}"'
                        f'{" font-weight=\"700\"" if bold else ""}{" font-style=\"italic\"" if italic else ""}>{s}</text>')

    def line(self, x1, y1, x2, y2, color=MUTED, w=2, arrow=False, dash=None):
        return self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w}"'
                        f'{f" stroke-dasharray=\"{dash}\"" if dash else ""}{" marker-end=\"url(#arr)\"" if arrow else ""}/>')

    def poly(self, pts, color=BLUE, w=2.5, fill='none', dash=None, close=False):
        d = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        tag = 'polygon' if close else 'polyline'
        return self.add(f'<{tag} points="{d}" fill="{fill}" stroke="{color}" stroke-width="{w}"'
                        f'{f" stroke-dasharray=\"{dash}\"" if dash else ""} stroke-linejoin="round"/>')

    def circle(self, x, y, r, color=BLUE, fill=None, w=2):
        return self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill if fill is not None else FILL.get(color, "#fff")}" stroke="{color}" stroke-width="{w}"/>')

    def rect(self, x, y, w, h, color=BLUE, fill=None, r=10, sw=2):
        return self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill if fill is not None else FILL.get(color, "#fff")}" stroke="{color}" stroke-width="{sw}"/>')

    def box(self, x, y, w, h, label, color=BLUE, sub=None, size=15):
        self.rect(x, y, w, h, color)
        if sub:
            self.text(x + w / 2, y + h / 2 - 3, label, size, INK, 'middle', True)
            self.text(x + w / 2, y + h / 2 + 15, sub, 12.5, MUTED, 'middle')
        else:
            self.text(x + w / 2, y + h / 2 + 5, label, size, INK, 'middle', True)
        return self

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" font-family="{FONT}">'
                f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs><rect width="{self.w}" height="{self.h}" fill="#fff"/>')
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name), 'w', encoding='utf8') as f:
            f.write(head + '\n' + '\n'.join(self.items) + '\n</svg>\n')
        print('  ', name)


class Plot:
    """데이터 좌표 → 그림 좌표. 축 · 눈금 · 곡선을 그린다."""
    def __init__(self, svg, x0, y0, w, h, xr, yr):
        self.s, self.x0, self.y0, self.w, self.h, self.xr, self.yr = svg, x0, y0, w, h, xr, yr

    def X(self, x): return self.x0 + (x - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w
    def Y(self, y): return self.y0 + self.h - (y - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def axes(self, xt=(), yt=(), xl='', yl='', grid=True):
        s = self.s
        for t in xt:
            if grid: s.line(self.X(t), self.y0, self.X(t), self.y0 + self.h, '#e2e8f0', 1)
            s.text(self.X(t), self.y0 + self.h + 16, f'{t:g}', 12, MUTED, 'middle')
        for t in yt:
            if grid: s.line(self.x0, self.Y(t), self.x0 + self.w, self.Y(t), '#e2e8f0', 1)
            s.text(self.x0 - 6, self.Y(t) + 4, f'{t:g}', 12, MUTED, 'end')
        ax = self.Y(0) if self.yr[0] <= 0 <= self.yr[1] else self.y0 + self.h
        ay = self.X(0) if self.xr[0] <= 0 <= self.xr[1] else self.x0
        s.line(self.x0, ax, self.x0 + self.w, ax, INK, 1.3)
        s.line(ay, self.y0, ay, self.y0 + self.h, INK, 1.3)
        if xl: s.text(self.x0 + self.w, ax - 6, xl, 13, INK, 'end')
        if yl: s.text(ay + 6, self.y0 + 12, yl, 13, INK)

    def curve(self, f, color=BLUE, n=200, w=2.5, dash=None, xr=None):
        a, b = xr or self.xr
        pts = []
        for i in range(n + 1):
            x = a + (b - a) * i / n
            y = f(x)
            if y is None or not math.isfinite(y): continue
            y = min(max(y, self.yr[0] - 1e9), self.yr[1] + 1e9)
            pts.append((self.X(x), self.Y(min(max(y, self.yr[0]), self.yr[1]))))
        self.s.poly(pts, color, w, dash=dash)


# ---------------- NN Day 2 ----------------

def d2_xor_points():
    s = Svg(620, 270)
    p = Plot(s, 110, 30, 180, 180, (-0.25, 1.25), (-0.25, 1.25))
    p.axes((0, 1), (0, 1), 'x₁', 'x₂', grid=False)
    for (a, b), y in [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]:
        s.circle(p.X(a), p.Y(b), 10, GREEN if y else INK, GREEN if y else '#fff')
        s.text(p.X(a) + (-16 if a == 0 else 16), p.Y(b) + (-14 if b else 26), f'({a},{b}) → {y}', 13, INK, 'end' if a == 0 else 'start')
    # 실패하는 직선 두 개: x₁ + x₂ = 0.5 와 1.5 — 어느 쪽도 1 과 0 을 깨끗이 가르지 못한다
    s.line(p.X(-0.25), p.Y(0.75), p.X(0.75), p.Y(-0.25), RED, 2, dash='6 5')
    s.line(p.X(0.25), p.Y(1.25), p.X(1.25), p.Y(0.25), RED, 2, dash='6 5')
    s.text(340, 50, 'XOR: 두 입력이 다를 때만 1', 15, INK, bold=True)
    s.circle(350, 82, 8, GREEN, GREEN); s.text(366, 87, '출력 1 — (0,1), (1,0)', 13)
    s.circle(350, 110, 8, INK, '#fff'); s.text(366, 115, '출력 0 — (0,0), (1,1)', 13)
    s.text(340, 152, '같은 출력끼리 대각선으로 엇갈려 있어서', 13, MUTED)
    s.text(340, 172, '어떤 직선(빨간 점선)을 그어도', 13, MUTED)
    s.text(340, 192, '한쪽에 1 과 0 이 섞인다', 13, MUTED)
    s.text(340, 222, '→ 선형 분리 불가능', 14, RED, bold=True)
    s.save('d2-xor-points.svg')


def d2_xor_mlp():
    s = Svg(660, 250)
    s.circle(60, 80, 24, BLUE); s.text(60, 86, 'x₁', 15, INK, 'middle', True)
    s.circle(60, 170, 24, BLUE); s.text(60, 176, 'x₂', 15, INK, 'middle', True)
    s.box(230, 50, 120, 56, 'NAND', ORANGE, 's₁')
    s.box(230, 144, 120, 56, 'OR', ORANGE, 's₂')
    s.box(460, 97, 110, 56, 'AND', PURPLE, 'y')
    for y1 in (80, 170):
        for y2 in (78, 172):
            s.line(84, y1, 228, y2, MUTED, 1.8, True)
    s.line(350, 78, 458, 118, MUTED, 1.8, True); s.line(350, 172, 458, 132, MUTED, 1.8, True)
    s.line(570, 125, 620, 125, MUTED, 1.8, True); s.text(632, 130, 'y', 15, INK, 'middle', True)
    s.text(60, 230, '입력층', 13, MUTED, 'middle'); s.text(290, 230, '은닉층 (직선 2개)', 13, MUTED, 'middle')
    s.text(515, 230, '출력층 (조합)', 13, MUTED, 'middle')
    s.text(330, 24, 'XOR = AND( NAND(x), OR(x) )', 15, INK, 'middle', True)
    s.save('d2-xor-mlp.svg')


def d2_xor_space():
    s = Svg(760, 260)
    L = Plot(s, 50, 30, 170, 170, (-0.2, 1.2), (-0.2, 1.2)); L.axes((0, 1), (0, 1), 'x₁', 'x₂', grid=False)
    R = Plot(s, 430, 30, 170, 170, (-0.2, 1.2), (-0.2, 1.2)); R.axes((0, 1), (0, 1), 's₁', 's₂', grid=False)
    pts = [((0, 0), 0, (1, 0)), ((0, 1), 1, (1, 1)), ((1, 0), 1, (1, 1)), ((1, 1), 0, (0, 1))]
    for (a, b), y, (c, d) in pts:
        col = GREEN if y else INK
        s.circle(L.X(a), L.Y(b), 9, col, GREEN if y else '#fff')
        s.circle(R.X(c), R.Y(d), 9 if not y else 11, col, GREEN if y else '#fff')
    s.text(R.X(1) + 16, R.Y(1) + 5, '← (0,1) 과 (1,0) 이', 12.5, GREEN, 'start', True)
    s.text(R.X(1) + 16, R.Y(1) + 22, '    둘 다 여기로', 12.5, GREEN, 'start', True)
    s.line(R.X(0.2), R.Y(1.2), R.X(1.2), R.Y(0.2), RED, 2.2, dash='6 5')
    s.text(R.X(0.62), R.Y(0.6), 's₁ + s₂ = 1.4', 12.5, RED, 'end')
    s.line(250, 115, 405, 115, MUTED, 2, True)
    s.text(328, 100, '은닉층 (NAND, OR)', 13, INK, 'middle', True)
    s.text(328, 138, '좌표를 바꾼다', 13, MUTED, 'middle')
    s.text(135, 238, '원래 공간: 직선 하나로 못 나눔', 13, MUTED, 'middle')
    s.text(515, 238, '새 공간: AND 직선 하나로 나뉨', 13, MUTED, 'middle')
    s.save('d2-xor-space.svg')


def d2_activations():
    s = Svg(720, 250)
    sig = lambda a: 1 / (1 + math.exp(-a))
    fs = [('step', lambda a: 1.0 if a > 0 else 0.0, (-0.3, 1.3), '0 또는 1, 기울기 0'),
          ('sigmoid', sig, (-0.3, 1.3), '0 ~ 1, 부드러운 계단'),
          ('tanh', math.tanh, (-1.3, 1.3), '−1 ~ 1, 0 중심'),
          ('ReLU', lambda a: max(0.0, a), (-0.5, 3.0), '음수는 0, 양수는 그대로')]
    for i, (name, f, yr, note) in enumerate(fs):
        p = Plot(s, 30 + i * 175, 40, 140, 140, (-3, 3), yr)
        p.axes((-2, 0, 2), (), '', '', grid=False)
        if name == 'step':
            s.poly([(p.X(-3), p.Y(0)), (p.X(0), p.Y(0))], BLUE, 3); s.poly([(p.X(0), p.Y(1)), (p.X(3), p.Y(1))], BLUE, 3)
            s.circle(p.X(0), p.Y(1), 4, BLUE, '#fff'); s.circle(p.X(0), p.Y(0), 4, BLUE, BLUE)
        else:
            p.curve(f, BLUE, 120, 3)
        s.text(p.x0 + 70, 26, name, 15, INK, 'middle', True)
        s.text(p.x0 + 70, 215, note, 12, MUTED, 'middle')
        if name in ('sigmoid', 'tanh'):
            for yv in ((0, 1) if name == 'sigmoid' else (-1, 1)):
                s.line(p.X(-3), p.Y(yv), p.X(3), p.Y(yv), LINE, 1, dash='3 4')
    s.text(360, 243, '가로축 = 가중합 a,  세로축 = 뉴런 출력 h(a)', 12.5, MUTED, 'middle')
    s.save('d2-activations.svg')


def d2_vanishing():
    s = Svg(700, 170)
    xs = [30 + i * 134 for i in range(5)]
    vals = ['1', '0.25', '0.0625', '0.0156', '0.0039']
    for i, x in enumerate(xs):
        s.box(x, 50, 84, 50, f'층 {5 - i}' if i else '출력', ORANGE if i else GREEN)
        s.text(x + 42, 126, f'기울기 ≤ {vals[i]}', 12.5, INK if i == 0 else RED, 'middle', i > 0)
        if i < 4: s.line(x + 134, 75, x + 88, 75, MUTED, 2, True); s.text(x + 109, 66, '×0.25', 12, MUTED, 'middle')
    s.text(350, 26, '역전파: 출력 쪽에서 입력 쪽으로 기울기가 층마다 곱해진다', 14, INK, 'middle', True)
    s.text(350, 156, 'sigmoid 기울기는 최대 0.25 → 10층이면 0.25¹⁰ ≈ 9.5 × 10⁻⁷ (거의 0)', 13, MUTED, 'middle')
    s.save('d2-vanishing.svg')


def d2_forward():
    s = Svg(700, 300)
    layers = [(2, 'X', 'N, 2', BLUE), (3, 'Z1', 'N, 3', ORANGE), (2, 'Z2', 'N, 2', ORANGE), (2, 'Y', 'N, 2', GREEN)]
    xs = [70, 250, 430, 610]
    pos = []
    for (k, name, shp, col), x in zip(layers, xs):
        ys = [150 + (j - (k - 1) / 2) * 70 for j in range(k)]
        pos.append(ys)
    for i in range(3):
        for y1 in pos[i]:
            for y2 in pos[i + 1]:
                s.line(xs[i] + 22, y1, xs[i + 1] - 24, y2, LINE, 1.4, True)
    for (k, name, shp, col), x, ys in zip(layers, xs, pos):
        for y in ys: s.circle(x, y, 20, col)
        s.text(x, 40, name, 15, INK, 'middle', True); s.text(x, 58, f'({shp})', 12.5, MUTED, 'middle')
    for i, (w, b, act) in enumerate([('W1 (2, 3)', 'b1 (3,)', 'sigmoid'), ('W2 (3, 2)', 'b2 (2,)', 'sigmoid'), ('W3 (2, 2)', 'b3 (2,)', 'identity')]):
        cx = (xs[i] + xs[i + 1]) / 2
        s.text(cx, 255, w + ' · ' + b, 12.5, ORANGE, 'middle', True)
        s.text(cx, 275, f'A = Z @ W + b → {act}', 12, MUTED, 'middle')
    s.save('d2-forward.svg')


def d2_softmax():
    s = Svg(680, 220)
    a = [0.3, 2.9, 4.0]; e = [math.exp(v) for v in a]; tot = sum(e); y = [v / tot for v in e]
    cols = [('점수 a', [f'{v:g}' for v in a], BLUE), ('exp(a)', [f'{v:.2f}' for v in e], ORANGE),
            ('÷ 합 ' + f'{tot:.2f}', [f'{v:.3f}' for v in y], GREEN)]
    for i, (h, vals, col) in enumerate(cols):
        x = 40 + i * 220
        s.text(x + 70, 30, h, 14, INK, 'middle', True)
        for j, v in enumerate(vals):
            s.box(x, 46 + j * 48, 140, 38, v, col, size=14)
        if i < 2: s.line(x + 146, 112, x + 214, 112, MUTED, 2, True)
    s.text(340, 206, '모두 양수 · 합이 1 · 순서(가장 큰 점수 = 가장 큰 확률)는 그대로', 13, MUTED, 'middle')
    s.save('d2-softmax.svg')


def d2_ce_curve():
    s = Svg(580, 280)
    p = Plot(s, 60, 20, 300, 200, (0, 1), (0, 5))
    p.axes((0, 0.2, 0.4, 0.6, 0.8, 1), (0, 1, 2, 3, 4, 5), '', '손실')
    s.text(p.x0 + p.w / 2, p.y0 + p.h + 36, '정답 클래스에 준 확률 p', 13, INK, 'middle')
    p.curve(lambda x: -math.log(x) if x > 0 else None, RED, 300, 3, xr=(0.007, 1))
    p.curve(lambda x: (1 - x) ** 2, BLUE, 100, 3, dash='7 5')
    for pv in (0.6, 0.1):
        s.circle(p.X(pv), p.Y(-math.log(pv)), 5, RED, RED)
        s.text(p.X(pv) + 8, p.Y(-math.log(pv)) - 6, f'p={pv:g} → {-math.log(pv):.2f}', 12.5, RED, 'start', True)
    s.line(390, 50, 420, 50, RED, 3); s.text(428, 55, 'CE = −log p', 13)
    s.line(390, 78, 420, 78, BLUE, 3, dash='7 5'); s.text(428, 83, '(1 − p)²', 13)
    s.text(390, 120, '정답 확률이 낮을수록', 13, MUTED); s.text(390, 140, 'CE 는 가파르게 커진다', 13, MUTED)
    s.text(390, 160, '→ 확신하며 틀리면 큰 벌점', 13, RED, bold=True)
    s.save('d2-ce-curve.svg')


def d2_linreg():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (50, 1)); T = 2.0 * X + 1.0 + rng.normal(0, 0.1, (50, 1))
    W, b = 2.052, 0.976
    s = Svg(560, 290)
    p = Plot(s, 60, 20, 300, 230, (0, 1), (0.8, 3.3))
    p.axes((0, 0.5, 1), (1, 2, 3), 'x', 't')
    for x, t in zip(X[:, 0], T[:, 0]):
        y = W * x + b
        s.line(p.X(x), p.Y(t), p.X(x), p.Y(y), RED, 1.2)
        s.circle(p.X(x), p.Y(t), 4, BLUE, BLUE, 1)
    p.curve(lambda x: W * x + b, GREEN, 2, 3)
    s.line(390, 60, 420, 60, GREEN, 3); s.text(428, 65, 'y = 2.05x + 0.98', 13)
    s.circle(405, 90, 4, BLUE, BLUE, 1); s.text(428, 95, '데이터 (x, t)', 13)
    s.line(405, 112, 405, 130, RED, 1.5); s.text(428, 126, '잔차 y − t', 13)
    s.text(390, 170, '손실 = 빨간 선 길이의', 13, MUTED); s.text(390, 190, '제곱 평균(의 절반)', 13, MUTED)
    s.text(390, 220, '경사하강은 이 합이 줄도록', 13, MUTED); s.text(390, 240, '직선을 돌리고(W) 옮긴다(b)', 13, MUTED)
    s.save('d2-linreg.svg')


def d2_logistic():
    rng = np.random.default_rng(1)
    X0 = rng.normal([-1, -1], 0.8, (100, 2)); X1 = rng.normal([1, 1], 0.8, (100, 2))
    X = np.vstack([X0, X1]); T = np.array([0] * 100 + [1] * 100).reshape(-1, 1)
    W, b = np.zeros((2, 1)), np.zeros(1)
    for _ in range(300):
        P = 1 / (1 + np.exp(-(X @ W + b))); W -= 0.5 * X.T @ (P - T) / 200; b -= 0.5 * np.mean(P - T, axis=0)
    w1, w2, bb = W[0, 0], W[1, 0], b[0]
    s = Svg(600, 300)
    p = Plot(s, 50, 20, 250, 250, (-4, 4), (-4, 4))
    p.axes((-3, 0, 3), (-3, 0, 3), 'x₁', 'x₂', grid=False)
    for x in X0: s.circle(p.X(x[0]), p.Y(x[1]), 3.5, BLUE, '#fff', 1.4)
    for x in X1: s.circle(p.X(x[0]), p.Y(x[1]), 3.5, ORANGE, ORANGE, 1)
    p.curve(lambda x: -(w1 * x + bb) / w2, INK, 2, 2.5, dash='7 5', xr=(-3.9, 3.9))
    s.circle(330, 50, 5, BLUE, '#fff', 1.4); s.text(344, 55, '클래스 0', 13)
    s.circle(330, 76, 5, ORANGE, ORANGE, 1); s.text(344, 81, '클래스 1', 13)
    s.line(320, 102, 345, 102, INK, 2.5, dash='7 5'); s.text(352, 107, '결정 경계 p = 0.5', 13)
    s.text(320, 140, f'학습 결과 W ≈ ({w1:.2f}, {w2:.2f}), b ≈ {bb:.2f}', 12.5, MUTED)
    s.text(320, 162, 'p = 0.5 ⇔ z = xW + b = 0', 12.5, MUTED)
    s.text(320, 184, '→ 경계는 직선', 12.5, MUTED)
    acc = float(np.mean(((1 / (1 + np.exp(-(X @ W + b)))) > 0.5) == T))
    s.text(320, 214, f'정확도 {acc:.3f} (300 step, lr 0.5)', 13, INK, bold=True)
    s.save('d2-logistic.svg')


def d2_grad_chain():
    s = Svg(680, 200)
    s.box(20, 60, 90, 50, 'x, W, b', BLUE)
    s.box(170, 60, 120, 50, 'z = xW + b', ORANGE)
    s.box(350, 60, 110, 50, 'p = σ(z)', PURPLE)
    s.box(520, 60, 140, 50, 'L = BCE(p, t)', RED)
    for a, b_ in ((110, 168), (290, 348), (460, 518)): s.line(a, 85, b_, 85, MUTED, 2, True)
    s.text(340, 28, '순전파 →', 13, MUTED, 'middle')
    s.text(590, 145, '∂L/∂p = (p − t) / p(1 − p)', 12.5, RED, 'middle')
    s.text(405, 145, '∂p/∂z = p(1 − p)', 12.5, PURPLE, 'middle')
    s.text(230, 145, '∂L/∂z = p − t', 13.5, GREEN, 'middle', True)
    s.text(340, 182, '두 식을 곱하면 p(1 − p) 가 지워져 (예측 − 정답) 만 남는다', 13, INK, 'middle', True)
    s.save('d2-grad-chain.svg')


# ---------------- NN Day 1 ----------------

def d1_learning_loop():
    s = Svg(640, 270)
    s.box(230, 20, 180, 56, '모델', BLUE, 'y = f(x; θ)')
    s.box(440, 110, 180, 56, '손실', RED, 'L(y, 정답)')
    s.box(230, 195, 180, 56, '기울기', PURPLE, '∇L = ∂L/∂θ')
    s.box(20, 110, 180, 56, '갱신', GREEN, 'θ ← θ − η·∇L')
    s.line(410, 60, 470, 108, MUTED, 2, True); s.line(470, 168, 412, 210, MUTED, 2, True)
    s.line(228, 212, 170, 168, MUTED, 2, True); s.line(170, 108, 228, 62, MUTED, 2, True)
    s.text(320, 128, '반복', 15, INK, 'middle', True)
    s.text(320, 150, '(손실이 더 안 줄 때까지)', 12.5, MUTED, 'middle')
    s.save('d1-learning-loop.svg')


def d1_variables():
    s = Svg(640, 210)
    s.text(150, 26, 'C: 변수 = 값이 든 메모리 칸', 14, INK, 'middle', True)
    s.box(60, 50, 80, 50, '3', BLUE); s.text(100, 120, 'a (칸 자체)', 13, MUTED, 'middle')
    s.box(170, 50, 80, 50, '3', BLUE); s.text(210, 120, 'b = a (복사)', 13, MUTED, 'middle')
    s.line(320, 15, 320, 195, '#e2e8f0', 2)
    s.text(480, 26, 'Python: 변수 = 객체에 붙인 이름표', 14, INK, 'middle', True)
    s.box(360, 60, 50, 32, 'a', ORANGE, size=14); s.box(360, 120, 50, 32, 'b', ORANGE, size=14)
    s.box(500, 75, 110, 60, '[1, 2, 3]', GREEN, 'list 객체', 14)
    s.line(410, 76, 498, 98, MUTED, 2, True); s.line(410, 136, 498, 114, MUTED, 2, True)
    s.text(480, 180, 'b = a → 같은 객체를 가리킨다 (복사 아님)', 13, RED, 'middle', True)
    s.save('d1-variables.svg')


def d1_strides():
    s = Svg(660, 250)
    s.text(20, 26, '메모리 (한 줄, 칸마다 8바이트)', 13.5, INK, bold=True)
    for i in range(12):
        col = [BLUE, ORANGE, GREEN][i // 4]
        s.box(20 + i * 52, 38, 48, 40, str(i), col, size=14)
    for r in range(3):
        s.text(20 + r * 208 + 102, 96, f'행 {r}', 12.5, [BLUE, ORANGE, GREEN][r], 'middle', True)
    s.text(20, 136, 'shape (3, 4) 로 읽으면', 13.5, INK, bold=True)
    for r in range(3):
        for c in range(4):
            i = r * 4 + c
            s.box(20 + c * 52, 148 + r * 32, 48, 28, str(i), [BLUE, ORANGE, GREEN][r], size=13)
    s.text(260, 166, 'strides = (32, 8)', 14, INK, bold=True)
    s.text(260, 190, '옆 칸(열 +1) → 8바이트 이동', 13, MUTED)
    s.text(260, 212, '아래 칸(행 +1) → 32바이트 이동 (4칸 × 8)', 13, MUTED)
    s.text(260, 236, 'A[1, 2] → 1×32 + 2×8 = 48바이트 위치 → 값 6', 13, RED, bold=True)
    s.save('d1-strides.svg')


def d1_broadcast():
    s = Svg(660, 200)
    def grid(x, y, rows, cols, vals, col, ghost=False):
        for r in range(rows):
            for c in range(cols):
                v = vals[r][c] if vals else ''
                s.rect(x + c * 40, y + r * 32, 38, 30, col, '#fff' if ghost else None, 4, 1.5 if not ghost else 1)
                s.text(x + c * 40 + 19, y + r * 32 + 20, str(v), 13, MUTED if ghost else INK, 'middle')
    A = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    grid(20, 50, 3, 4, A, BLUE); s.text(100, 36, 'A (3, 4)', 13.5, INK, 'middle', True)
    s.text(196, 102, '+', 22, INK, 'middle', True)
    grid(220, 50, 1, 4, [[10, 20, 30, 40]], ORANGE); s.text(300, 36, 'b (4,)', 13.5, INK, 'middle', True)
    grid(220, 82, 2, 4, [[10, 20, 30, 40]] * 2, ORANGE, ghost=True)
    s.text(300, 168, '행 방향으로 복사된 것처럼', 12.5, MUTED, 'middle')
    s.text(396, 102, '=', 22, INK, 'middle', True)
    grid(420, 50, 3, 4, [[10, 21, 32, 43], [14, 25, 36, 47], [18, 29, 40, 51]], GREEN)
    s.text(500, 36, 'A + b (3, 4)', 13.5, INK, 'middle', True)
    s.text(330, 192, '규칙: 뒤쪽 축부터 맞춰 보고, 크기가 같거나 1 이면 늘려서 계산', 12.5, MUTED, 'middle')
    s.save('d1-broadcast.svg')


def d1_matmul():
    s = Svg(660, 230)
    s.box(20, 60, 150, 110, 'X', BLUE, '(4, 3)  행=샘플, 열=특징')
    s.text(190, 122, '@', 22, INK, 'middle', True)
    s.box(210, 80, 110, 70, 'W', ORANGE, '(3, 2)')
    s.text(340, 122, '+', 22, INK, 'middle', True)
    s.box(360, 98, 80, 36, 'b', PURPLE, '(2,)', 14)
    s.text(460, 122, '=', 22, INK, 'middle', True)
    s.box(480, 60, 150, 110, 'Y', GREEN, '(4, 2)  샘플마다 출력 2개')
    s.text(330, 30, '(N, D) @ (D, M) = (N, M) — 안쪽 D 가 같아야 하고, 계산 뒤 사라진다', 13.5, INK, 'middle', True)
    s.text(330, 205, 'b 는 브로드캐스팅으로 Y 의 모든 행에 더해진다', 13, MUTED, 'middle')
    s.save('d1-matmul.svg')


def d1_dot():
    s = Svg(560, 220)
    O = (60, 180)
    s.line(O[0], O[1], 360, O[1], BLUE, 3, True); s.text(366, O[1] + 5, 'a', 15, BLUE, bold=True)
    bx, by = 260, 50
    s.line(O[0], O[1], bx, by, ORANGE, 3, True); s.text(bx + 8, by - 4, 'b', 15, ORANGE, bold=True)
    s.line(bx, by, bx, O[1], LINE, 1.5, dash='5 4')
    s.line(O[0], O[1] + 14, bx, O[1] + 14, GREEN, 4)
    s.text((O[0] + bx) / 2, O[1] + 34, '‖b‖ cos θ  (b 의 그림자)', 13, GREEN, 'middle', True)
    s.text(95, 168, 'θ', 15, INK, bold=True)
    s.text(380, 70, 'a · b = ‖a‖ × ‖b‖ cos θ', 14, INK, bold=True)
    s.text(380, 100, '같은 방향 → 크다 (cos 1)', 13, MUTED)
    s.text(380, 122, '수직 → 0 (cos 0)', 13, MUTED)
    s.text(380, 144, '반대 방향 → 음수 (cos −1)', 13, MUTED)
    s.save('d1-dot.svg')


def d1_derivative():
    s = Svg(660, 280)
    p = Plot(s, 50, 20, 300, 220, (0, 3), (0, 9))
    p.axes((0, 1, 2, 3), (0, 3, 6, 9), 'x', 'f(x) = x²')
    p.curve(lambda x: x * x, BLUE, 150, 3)
    x0 = 1.5
    p.curve(lambda x: 2 * x0 * (x - x0) + x0 * x0, RED, 2, 2.2, xr=(0.4, 2.8))
    h = 1.0
    p.curve(lambda x: ((x0 + h) ** 2 - x0 ** 2) / h * (x - x0) + x0 * x0, ORANGE, 2, 1.8, dash='6 5', xr=(0.9, 2.95))
    s.circle(p.X(x0), p.Y(x0 * x0), 5, RED, RED)
    s.circle(p.X(x0 + h), p.Y((x0 + h) ** 2), 4.5, ORANGE, '#fff')
    s.text(380, 50, 'x = 1.5 에서의 기울기', 13.5, INK, bold=True)
    s.line(380, 76, 410, 76, RED, 2.2); s.text(416, 81, '접선: 기울기 = 미분값 3', 13)
    s.line(380, 102, 410, 102, ORANGE, 1.8, dash='6 5'); s.text(416, 107, '할선 (x 와 x + 1): 기울기 4', 13)
    s.text(380, 142, 'h 를 줄일수록 할선이 접선에 가까워진다', 13, MUTED)
    s.text(380, 164, '= 미분의 정의 (h → 0 의 극한)', 13, MUTED)
    s.text(380, 196, '수치 미분은 h = 1e-4 와 중앙 차분', 13, MUTED)
    s.text(380, 218, '(f(x+h) − f(x−h)) / 2h 로 근사', 13, MUTED)
    s.save('d1-derivative.svg')


def d1_gradient():
    s = Svg(640, 280)
    p = Plot(s, 40, 20, 240, 240, (-2.2, 2.2), (-2.2, 2.2))
    for r in (0.5, 1.0, 1.5, 2.0):
        s.add(f'<circle cx="{p.X(0):.1f}" cy="{p.Y(0):.1f}" r="{r / 4.4 * 240:.1f}" fill="none" stroke="#cbd5e1" stroke-width="1.3"/>')
    p.axes((), (), 'x₀', 'x₁', grid=False)
    s.circle(p.X(1), p.Y(1), 5, INK, INK)
    s.line(p.X(1), p.Y(1), p.X(1.7), p.Y(1.7), RED, 2.5, True)
    s.line(p.X(1), p.Y(1), p.X(0.3), p.Y(0.3), GREEN, 2.5, True)
    s.text(p.X(1.62), p.Y(1.95), '∇f = (2, 2)', 12.5, RED, 'end', True)
    s.text(p.X(0.25), p.Y(0.05), '−∇f', 12.5, GREEN, 'end', True)
    s.text(320, 50, 'f(x₀, x₁) = x₀² + x₁²  (그릇 모양)', 13.5, INK, bold=True)
    s.text(320, 80, '회색 원 = 높이가 같은 등고선', 13, MUTED)
    s.text(320, 104, '기울기 ∇f → 가장 가파른 오르막', 13, RED)
    s.text(320, 128, '−∇f → 가장 가파른 내리막 (바닥 쪽)', 13, GREEN)
    s.text(320, 160, '점 (1, 1) 에서 ∇f = (2x₀, 2x₁) = (2, 2)', 13, MUTED)
    s.text(320, 184, '경사하강 = −∇f 방향으로 조금씩', 13, INK, bold=True)
    s.save('d1-gradient.svg')


def d1_neglog():
    s = Svg(620, 250)
    p = Plot(s, 60, 20, 300, 190, (0, 1), (0, 5))
    p.axes((0, 0.5, 1), (0, 1, 2, 3, 4, 5), 'p', '−log p')
    p.curve(lambda x: -math.log(x) if x > 0 else None, RED, 300, 3, xr=(0.007, 1))
    s.text(390, 60, '정답에 준 확률 p', 13.5, INK, bold=True)
    s.text(390, 88, 'p → 1 이면 −log p → 0', 13, MUTED)
    s.text(390, 110, '(완벽하면 벌점 없음)', 13, MUTED)
    s.text(390, 140, 'p → 0 이면 −log p → ∞', 13, MUTED)
    s.text(390, 162, '(틀릴수록 큰 벌점)', 13, MUTED)
    s.text(390, 196, 'log(0) = −∞ → 1e-7 을 더한다', 13, RED, bold=True)
    s.save('d1-neglog.svg')


def d1_gd_paths():
    f_grad = lambda p: np.array([2 * p[0], 6 * p[1]])
    s = Svg(660, 300)
    p = Plot(s, 40, 20, 340, 250, (-4.5, 4.5), (-3.2, 3.2))
    for c in (1, 4, 9, 16, 25):
        rx, ry = math.sqrt(c), math.sqrt(c / 3)
        s.add(f'<ellipse cx="{p.X(0):.1f}" cy="{p.Y(0):.1f}" rx="{rx / 9 * 340:.1f}" ry="{ry / 6.4 * 250:.1f}" fill="none" stroke="#cbd5e1" stroke-width="1.2"/>')
    p.axes((), (), 'x', 'y', grid=False)
    for lr, col in ((0.1, GREEN), (0.3, ORANGE)):
        q = np.array([-4.0, 2.0]); pts = [tuple(q)]
        for _ in range(20): q = q - lr * f_grad(q); pts.append(tuple(q))
        s.poly([(p.X(a), p.Y(min(max(b, -3.2), 3.2))) for a, b in pts], col, 2.2)
        for a, b in pts[:8]: s.circle(p.X(a), p.Y(b), 3, col, col, 1)
    s.circle(p.X(-4), p.Y(2), 5, INK, INK); s.text(p.X(-4) + 8, p.Y(2) - 8, '시작 (−4, 2)', 12.5)
    s.text(400, 50, 'f(x, y) = x² + 3y²', 14, INK, bold=True)
    s.line(400, 78, 430, 78, GREEN, 2.5); s.text(436, 83, 'lr 0.1 — 바닥까지 부드럽게', 13)
    s.line(400, 104, 430, 104, ORANGE, 2.5); s.text(436, 109, 'lr 0.3 — y 방향 지그재그', 13)
    s.text(400, 144, 'y 방향이 3배 가파르다', 13, MUTED)
    s.text(400, 166, '→ y 가 학습률 상한을 정한다', 13, MUTED)
    s.text(400, 188, '이 함수는 0 < lr < 1/3 에서 수렴', 13, RED, bold=True)
    s.text(400, 210, '(lr 0.35 면 y 가 커지며 발산)', 13, MUTED)
    s.save('d1-gd-paths.svg')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if callable(fn) and name.startswith(('d1_', 'd2_')): fn()
