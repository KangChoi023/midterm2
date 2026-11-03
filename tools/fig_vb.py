#!/usr/bin/env python3
"""VB(Vision 기본 모델) 노트 "상세 학습" 의 그림(SVG)을 만든다 → assets/fig/vb/*.svg

    python3 -I tools/fig_vb.py          # 전부 다시 만들기

그리기 도구(Svg · Plot)는 tools/fig_nn.py 의 것을 그대로 쓴다. 규칙도 같다 — 도형 + 진짜 글자, 흰 배경,
그래프는 강의와 같은 식으로 계산해 그린다.
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_nn
from fig_nn import Svg, Plot, INK, MUTED, LINE, BLUE, ORANGE, GREEN, RED, PURPLE, FILL

fig_nn.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'fig', 'vb')


def _arrow(s, x1, y1, x2, y2, label=None, dy=-8):
    s.line(x1, y1, x2, y2, MUTED, 2, True)
    if label: s.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, 12, MUTED, 'middle')


# ---------------- VB Day 1 ----------------

def d1_share():
    s = Svg(660, 220)
    s.text(330, 24, 'from_numpy 는 같은 메모리를 보고, torch.tensor 는 복사본을 만든다', 14, INK, 'middle', True)
    s.box(20, 50, 150, 50, 'a (ndarray)', BLUE, size=14)
    s.box(20, 140, 150, 50, 'b = from_numpy(a)', ORANGE, size=13)
    for k, v in enumerate([7, 0, 0]):
        s.rect(250 + k * 46, 95, 46, 46, BLUE, None, 0, 1.5); s.text(250 + k * 46 + 23, 125, str(v), 15, INK, 'middle', True)
    s.text(319, 166, '메모리 하나 — b[0] = 7 이 a 에도 보인다', 12, MUTED, 'middle')
    _arrow(s, 170, 75, 248, 108); _arrow(s, 170, 165, 248, 128)
    s.box(470, 50, 170, 50, 'c = torch.tensor(a)', GREEN, size=13)
    for k, v in enumerate([0, 9, 0]):
        s.rect(486 + k * 46, 120, 46, 46, GREEN, None, 0, 1.5); s.text(486 + k * 46 + 23, 150, str(v), 15, INK, 'middle', True)
    s.line(555, 100, 555, 118, MUTED, 2, True)
    s.text(555, 190, '자기 복사본 — c[1] = 9 는 a 에 안 보인다', 12, MUTED, 'middle')
    s.save('d1-share.svg')


def d1_graph():
    s = Svg(680, 250)
    s.text(340, 22, 'f = x² + 3y² 의 계산 그래프 — 파랑: 순전파 값, 빨강: 역전파 기울기 (출력에서 1 로 시작)', 13.5, INK, 'middle', True)
    s.box(20, 50, 80, 44, 'x = 1', BLUE, size=14); s.box(20, 150, 80, 44, 'y = 2', BLUE, size=14)
    s.circle(180, 72, 24, ORANGE); s.text(180, 78, '²', 18, INK, 'middle', True)
    s.circle(180, 172, 24, ORANGE); s.text(180, 178, '²', 18, INK, 'middle', True)
    s.circle(300, 172, 24, ORANGE); s.text(300, 178, '×3', 15, INK, 'middle', True)
    s.circle(420, 122, 24, ORANGE); s.text(420, 128, '+', 18, INK, 'middle', True)
    s.box(520, 100, 130, 44, 'f = 13', GREEN, size=14)
    for (x1, y1, x2, y2, v, g) in [(100, 72, 156, 72, '1', '2'), (100, 172, 156, 172, '2', '12'), (204, 172, 276, 172, '4', '3'),
                                   (204, 72, 400, 108, '1', '1'), (324, 172, 400, 136, '12', '1'), (444, 122, 518, 122, '13', '1')]:
        s.line(x1, y1, x2, y2, LINE, 2, True)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        s.text(mx, my - 8, v, 13, BLUE, 'middle', True); s.text(mx, my + 20, g, 13, RED, 'middle', True)
    s.text(340, 226, 'x.grad = 1 · 1 · 2x = 2,  y.grad = 1 · 3 · 2y = 12 — f.backward() 가 연쇄 법칙으로 자동 계산', 13, INK, 'middle')
    s.save('d1-graph.svg')


def d1_loop():
    s = Svg(700, 210)
    steps = [('배치 → device', 'xb.to(device)', LINE), ('순전파', 'model(xb)', BLUE), ('손실', 'loss_fn(…, yb)', BLUE),
             ('기울기 비우기', 'opt.zero_grad()', ORANGE), ('역전파', 'loss.backward()', RED), ('갱신', 'opt.step()', GREEN)]
    for i, (a, b, c) in enumerate(steps):
        x = 10 + i * 115
        s.box(x, 50, 100, 64, a, c, b, 13)
        if i < 5: s.line(x + 100, 82, x + 113, 82, MUTED, 2, True)
    s.poly([(660, 118), (660, 150), (60, 150), (60, 120)], MUTED, 2, dash='5,4')
    s.text(360, 168, 'for xb, yb in loader: — 배치마다 반복, 한 바퀴 = 1 epoch', 12.5, MUTED, 'middle')
    s.text(350, 26, '학습 한 step — 모델이 ResNet 이든 YOLO 든 이 모양 그대로', 14, INK, 'middle', True)
    s.text(350, 196, 'NN 과정의 net.gradient(x, t) + optimizer.update(params, grads) 가 zero_grad · backward · step 세 줄로', 12.5, MUTED, 'middle')
    s.save('d1-loop.svg')


def d1_dataloader():
    s = Svg(700, 220)
    s.text(110, 24, 'Dataset — 한 장씩', 13.5, INK, 'middle', True)
    for k in range(6):
        s.rect(20 + k * 30, 40, 26, 26, BLUE, None, 4, 1.5); s.text(33 + k * 30, 58, str(k), 12, INK, 'middle')
    s.text(110, 90, 'ds[i] → (이미지 (1, 28, 28), 레이블)', 12, MUTED, 'middle')
    _arrow(s, 210, 53, 270, 53, 'shuffle')
    s.text(380, 24, '섞인 순서', 13.5, INK, 'middle', True)
    for k, v in enumerate([4, 1, 5, 0, 3, 2]):
        s.rect(290 + k * 30, 40, 26, 26, ORANGE, None, 4, 1.5); s.text(303 + k * 30, 58, str(v), 12, INK, 'middle')
    _arrow(s, 470, 53, 520, 53, 'batch')
    s.text(610, 24, 'DataLoader — 묶음으로', 13.5, INK, 'middle', True)
    for b, grp in enumerate([[4, 1], [5, 0], [3, 2]]):
        s.rect(540, 36 + b * 36, 140, 30, GREEN, None, 6, 1.5)
        s.text(610, 56 + b * 36, f'배치 {b + 1}: {grp[0]}, {grp[1]}', 12, INK, 'middle')
    s.text(350, 170, 'MNIST train 60,000 장 · batch_size=100 · shuffle=True → epoch 당 600 iteration, xb (100, 1, 28, 28) · yb (100,) long', 12.5, INK, 'middle')
    s.text(350, 194, 'NN 과정의 rng.permutation(N) + idx[i : i + 100] 루프를 대신한다 (평가 로더는 shuffle=False)', 12.5, MUTED, 'middle')
    s.save('d1-dataloader.svg')


def d1_hwc():
    s = Svg(700, 262)
    s.text(150, 24, 'OpenCV (H, W, C) uint8 BGR', 13.5, INK, 'middle', True)
    for i in range(3):
        for j in range(3):
            x, y = 40 + j * 72, 40 + i * 40
            for k, c in enumerate([BLUE, GREEN, RED]):
                s.rect(x + k * 22, y, 22, 32, c, None, 0, 1.2)
                s.text(x + k * 22 + 11, y + 21, 'BGR'[k], 11, INK, 'middle', True)
    s.text(150, 178, '한 픽셀의 B, G, R 이 붙어 있다 · 0 ~ 255', 12, MUTED, 'middle')
    _arrow(s, 270, 95, 360, 95, '[:, :, ::-1]', -10)
    s.text(315, 118, 'permute(2, 0, 1)', 12, MUTED, 'middle'); s.text(315, 134, '÷ 255', 12, MUTED, 'middle')
    s.text(520, 24, 'PyTorch (C, H, W) float32 RGB', 13.5, INK, 'middle', True)
    for k, (c, nm) in enumerate([(BLUE, 'B'), (GREEN, 'G'), (RED, 'R')]):
        s.rect(430 + k * 18, 84 - k * 18, 120, 84, c, None, 4, 1.8)
    s.text(526, 80, 'R 판', 13, INK, 'middle', True); s.text(526, 98, '(그 뒤 G · B)', 11.5, MUTED, 'middle')
    s.text(520, 190, '채널별 판 3장 · 0 ~ 1', 12, MUTED, 'middle')
    s.text(350, 222, 'unsqueeze(0) → (1, 3, H, W) 배치 · Normalize: (x − mean.view(3, 1, 1)) / std.view(3, 1, 1)', 13, INK, 'middle', True)
    s.text(350, 248, 'BGR 을 RGB 로 안 바꾸면 오류 없이 정확도만 떨어진다 — 학습 때와 같은 전처리를 추론 때도', 12, RED, 'middle')
    s.save('d1-hwc.svg')


def d1_video():
    s = Svg(700, 230)
    s.text(350, 22, '테스트 영상 — 숫자 띠 (120 × 1280) 위를 320 폭 창이 지나간다 · 6초 × 30 FPS = 180 프레임', 13.5, INK, 'middle', True)
    s.rect(20, 50, 660, 62, LINE, '#f8fafc', 4, 1.5)
    for k, d in enumerate([7, 2, 1, 0, 4, 1]):
        x = 20 + (k + 1) * 82
        s.rect(x, 56, 50, 50, INK, '#0f172a', 3, 1); s.text(x + 25, 89, str(d), 22, '#fff', 'middle', True)
    for k, (x, c) in enumerate([(30, BLUE), (180, ORANGE)]):
        s.rect(x, 44, 165, 74, c, 'none', 4, 3)
    s.text(112, 136, '프레임 0', 12, BLUE, 'middle'); s.text(262, 136, '프레임 i — 창이 오른쪽으로', 12, ORANGE, 'middle')
    s.text(350, 170, '프레임 하나 = (120, 320, 3) uint8 BGR = 115,200 byte → 압축 없이 180장이면 약 20.7 MB → 코덱(mp4v)', 12.5, INK, 'middle')
    s.text(350, 194, 'i % 30 == 15 인 프레임(15, 45, …, 165)에서 숫자가 창 가운데 — 가운데 120 × 120 을 28 × 28 로 줄여 모델에', 12.5, MUTED, 'middle')
    s.text(350, 218, 'VideoCapture → read() → (처리 · 추론) → VideoWriter.write() → release()', 12.5, BLUE, 'middle', True)
    s.save('d1-video.svg')


def d1_preprocess():
    s = Svg(720, 220)
    steps = [('① 흑백', 'cvtColor', LINE), ('② 조명 보정', 'g ÷ 흐린 배경', BLUE), ('③ 반전 · 이진화', 'INV + OTSU', ORANGE),
             ('④ 숫자 자르기', '가장 큰 윤곽', GREEN), ('⑤ 20 × 20', '긴 변 = 20', PURPLE), ('⑥ 28 × 28', '무게중심 (14, 14)', RED)]
    for i, (a, b, c) in enumerate(steps):
        x = 8 + i * 118
        s.box(x, 50, 108, 64, a, c, b, 13)
        if i < 5: s.line(x + 108, 82, x + 116, 82, MUTED, 2, True)
    s.text(360, 26, '종이 사진 → MNIST 모양 — 입력을 학습 데이터 쪽으로 옮긴다', 14, INK, 'middle', True)
    s.text(360, 146, 'MNIST: 검은 배경 · 흰 글씨 · 20 × 20 상자 · 무게중심 가운데  ↔  종이: 흰 배경 · 어두운 글씨 · 그림자 · 치우침', 12.5, INK, 'middle')
    s.text(360, 170, '이 차이(domain shift) 때문에 그냥 28 × 28 로 줄이면 거의 못 맞힌다', 12.5, RED, 'middle', True)
    s.text(360, 194, '반환 (1, 1, 28, 28) float32 0 ~ 1 — 다른 해결: 증강(Day 2) · 전이학습(Day 3)', 12.5, MUTED, 'middle')
    s.save('d1-preprocess.svg')


# ---------------- VB Day 2 ----------------

def d2_receptive():
    s = Svg(700, 250)
    s.text(350, 22, '수용 영역 — 3 × 3 두 층이 보는 입력 = 5 × 5 (5 × 5 한 층과 같다)', 14, INK, 'middle', True)
    c = 26
    for i in range(5):
        for j in range(5):
            inner = 1 <= i <= 3 and 1 <= j <= 3
            s.rect(30 + j * c, 50 + i * c, c, c, BLUE, FILL[BLUE] if not inner else '#bae6fd', 0, 1.2)
    s.text(30 + 2.5 * c, 50 + 5 * c + 20, '입력 5 × 5', 12.5, INK, 'middle')
    _arrow(s, 170, 115, 230, 115, '3 × 3', -10)
    for i in range(3):
        for j in range(3):
            s.rect(245 + j * c, 76 + i * c, c, c, ORANGE, None, 0, 1.2)
    s.text(245 + 1.5 * c, 76 + 3 * c + 20, '1층 출력 3 × 3', 12.5, INK, 'middle')
    _arrow(s, 335, 115, 395, 115, '3 × 3', -10)
    s.rect(410, 102, c, c, GREEN, None, 0, 1.5); s.text(423, 165, '2층 출력 1칸', 12.5, INK, 'middle')
    s.text(470, 70, 'rf ← rf + (K − 1) · jump', 13, INK)
    s.text(470, 92, 'jump ← jump · S', 13, INK)
    s.text(470, 122, '1 → 1 + 2 = 3 → 3 + 2 = 5', 13, BLUE, bold=True)
    s.text(470, 144, '3 × 3 세 층 = 7 × 7', 12.5, MUTED)
    s.text(470, 166, '풀링 · stride 뒤엔 jump 가 커져', 12.5, MUTED)
    s.text(470, 184, '같은 3 × 3 이 더 넓게 퍼진다', 12.5, MUTED)
    s.text(350, 236, '파라미터: 5 × 5 한 층 25C² vs 3 × 3 두 층 18C² (28% 적다) — 게다가 ReLU 가 두 번', 13, INK, 'middle', True)
    s.save('d2-receptive.svg')


def d2_lenet():
    s = Svg(720, 200)
    items = [('입력', '(3, 32, 32)', LINE, 78), ('conv 5×5', '(6, 28, 28)', BLUE, 88), ('pool', '(6, 14, 14)', ORANGE, 78),
             ('conv 5×5', '(16, 10, 10)', BLUE, 94), ('pool', '(16, 5, 5)', ORANGE, 76), ('FC', '400 → 120', PURPLE, 80),
             ('FC', '120 → 84', PURPLE, 74), ('FC', '84 → 10', PURPLE, 70)]
    x = 6
    for i, (a, b, c, w) in enumerate(items):
        s.box(x, 50, w, 64, a, c, b, 13)
        if i < len(items) - 1: s.line(x + w, 82, x + w + 6, 82, MUTED, 2, True)
        x += w + 7
    s.text(360, 28, 'LeNet-5 (1998) — Conv-Pool-Conv-Pool-FC-FC-FC 뼈대, CIFAR-10 용', 14, INK, 'middle', True)
    s.text(360, 142, '파라미터 62,006 (conv 456 + 2,416 · FC 48,120 + 10,164 + 850) — MLP 1,707,274 의 1/27', 13, INK, 'middle')
    s.text(360, 166, '3 epoch test: MLP 0.4994 vs LeNet-5 0.5489 — 구조의 가정(귀납적 편향)이 파라미터보다 중요', 12.5, BLUE, 'middle', True)
    s.save('d2-lenet.svg')


def d2_vgg():
    s = Svg(720, 220)
    s.text(360, 24, 'VGG-mini — [Conv3×3 → BN → ReLU] × 2 → MaxPool 블록 3개, 크기 1/2 마다 채널 2배', 14, INK, 'middle', True)
    items = [('입력', '(3, 32, 32)', LINE, 96), ('블록 1', '(32, 16, 16)', BLUE, 118), ('블록 2', '(64, 8, 8)', ORANGE, 118),
             ('블록 3', '(128, 4, 4)', GREEN, 118), ('Flatten', '2048', LINE, 96), ('Linear', '2048 → 10', PURPLE, 110)]
    x = 8
    for i, (a, b, c, w) in enumerate(items):
        s.box(x, 50, w, 64, a, c, b, 13.5)
        if i < len(items) - 1: s.line(x + w, 82, x + w + 10, 82, MUTED, 2, True)
        x += w + 12
    s.text(360, 142, '파라미터: 블록 10,080 + 55,296 + 221,184 · Linear 20,490 · BN 896 = 307,946', 13, INK, 'middle')
    s.text(360, 166, '3 epoch test 0.7526 (LeNet 0.5489) · MAC 38,653,952 (LeNet 의 약 59배)', 12.5, BLUE, 'middle', True)
    s.text(360, 190, '크기가 절반이면 위치 수 1/4 — 채널을 2배로 늘려도 층마다 연산량이 비슷하게 유지된다', 12.5, MUTED, 'middle')
    s.save('d2-vgg.svg')


def d2_vanish():
    s = Svg(680, 250)
    s.text(340, 22, '첫 층 기울기 크기 (log 눈금) — 그냥 깊게 쌓으면 입력 쪽 기울기가 사라진다', 14, INK, 'middle', True)
    rows = [('plain-8', 3.2e-4, ORANGE), ('plain-20', 1.1e-9, RED), ('plain-20 + BN', 3.4e-1, GREEN), ('마지막 층 (plain-20)', 2.2e-2, LINE)]
    x0, x1 = 190, 640
    X = lambda v: x0 + (math.log10(v) + 10) / 10 * (x1 - x0)
    for e in range(-10, 1, 2):
        s.line(X(10 ** e), 40, X(10 ** e), 200, '#e2e8f0', 1); s.text(X(10 ** e), 216, f'1e{e}' if e else '1', 11.5, MUTED, 'middle')
    for r, (nm, v, c) in enumerate(rows):
        y = 50 + r * 38
        s.text(180, y + 18, nm, 13, INK, 'end')
        s.rect(x0, y + 4, X(v) - x0, 22, c, None, 3, 1.5)
        s.text(X(v) + 6, y + 20, f'{v:.1e}', 12, INK)
    s.text(340, 240, '강의 실행 예 · BN 이 있으면 기울기가 고르게 흐른다 — 그래도 20층의 train 손실(1.785)이 8층(1.699)보다 높다 = 성능 저하', 11.5, MUTED, 'middle')
    s.save('d2-vanish.svg')


def d2_residual():
    s = Svg(700, 230)
    s.text(350, 22, 'Residual block — y = ReLU(F(x) + x), 층은 "바꿀 양" F 만 배운다', 14, INK, 'middle', True)
    s.box(20, 90, 60, 44, 'x', BLUE, size=15)
    s.box(120, 90, 150, 44, 'Conv-BN-ReLU', ORANGE, size=13); s.box(300, 90, 110, 44, 'Conv-BN', ORANGE, size=13)
    s.circle(460, 112, 18, GREEN); s.text(460, 118, '+', 18, INK, 'middle', True)
    s.box(510, 90, 80, 44, 'ReLU', GREEN, size=13); s.box(620, 90, 60, 44, 'y', BLUE, size=15)
    for a, b in ((80, 118), (270, 298), (410, 440), (478, 508), (590, 618)): s.line(a, 112, b, 112, MUTED, 2, True)
    s.poly([(50, 90), (50, 56), (460, 56), (460, 92)], RED, 2.5)
    s.text(255, 48, '지름길 (모양이 같으면 Identity, 다르면 1×1 Conv stride 2 + BN)', 12.5, RED, 'middle', True)
    s.text(265, 160, 'F(x) — 잔차', 12.5, ORANGE, 'middle')
    s.text(350, 192, '역전파: ∂y/∂x = F′(x) + 1 — F′ 가 아무리 작아도 1 이 남아 입력까지 기울기가 흐른다', 13, INK, 'middle', True)
    s.text(350, 216, '바꿀 게 없으면 F = 0 을 배우면 된다 (y = x) — plain 망이 어려워하던 "아무것도 안 하기" 가 쉬워진다', 12.5, MUTED, 'middle')
    s.save('d2-residual.svg')


def d2_resnet20():
    s = Svg(720, 210)
    s.text(360, 24, 'ResNet-20 (CIFAR-10) — 깊이 6n + 2, n = 3', 14, INK, 'middle', True)
    items = [('stem', 'Conv3×3 · 16 · 32²', LINE, 118), ('layer1', 'Block × 3 · 16 · 32²', BLUE, 130), ('layer2', 'Block × 3 · 32 · 16²', ORANGE, 130),
             ('layer3', 'Block × 3 · 64 · 8²', GREEN, 130), ('head', 'GAP + Linear 64→10', PURPLE, 140)]
    x = 8
    for i, (a, b, c, w) in enumerate(items):
        s.box(x, 50, w, 64, a, c, b, 13.5)
        if i < len(items) - 1: s.line(x + w, 82, x + w + 10, 82, MUTED, 2, True)
        x += w + 12
    s.text(360, 140, '가중치 층 = stem 1 + 블록 9 × 2 + Linear 1 = 20 · layer2 · layer3 첫 블록은 stride 2 (크기 1/2, 채널 2배)', 12.5, INK, 'middle')
    s.text(360, 164, '파라미터 272,474 — 거대 FC 대신 GAP + Linear(64 → 10) = 650 개 · MAC 40,813,184', 12.5, INK, 'middle')
    s.text(360, 188, '3 epoch (강의 실행 예): plain-20 train 손실 1.641 · test 0.380 → ResNet-20 1.438 · 0.443 — 성능 저하가 사라짐', 12, BLUE, 'middle', True)
    s.save('d2-resnet20.svg')


def d2_onecycle():
    s = Svg(660, 260)
    total, pct, mx = 1000, 0.3, 0.1
    lo, end = mx / 25, mx / 25 / 1e4
    def lr(t):
        up = pct * total
        if t <= up:
            return lo + (mx - lo) * (1 - math.cos(math.pi * t / up)) / 2
        return end + (mx - end) * (1 + math.cos(math.pi * (t - up) / (total - up))) / 2
    p = Plot(s, 70, 40, 520, 170, (0, total), (0, 0.11))
    p.axes((), (0, 0.05, 0.1), '', '', grid=True)
    p.curve(lr, BLUE, 300, 3)
    for t, lab in ((0, '시작 0.004'), (300, '30% 지점 최대 0.1'), (1000, '끝 4e-07')):
        s.circle(p.X(t), p.Y(lr(t)), 5, RED, RED)
        s.text(p.X(t) + (14 if t < 900 else -10), p.Y(lr(t)) - (22 if t == 0 else 10), lab, 12.5, RED, 'start' if t < 900 else 'end', True)
    s.text(330, 24, 'OneCycleLR (max_lr 0.1) — 작게 → 크게(warm-up) → 거의 0', 14, INK, 'middle', True)
    s.text(590, 232, 'step (배치마다 scheduler.step())', 12, MUTED, 'end')
    s.text(30, 30, 'lr', 13, INK)
    s.save('d2-onecycle.svg')


def d2_augment():
    s = Svg(700, 230)
    s.text(350, 22, '데이터 증강 (train 에만) — 정답이 바뀌지 않는 변형으로 매번 다른 그림', 14, INK, 'middle', True)
    s.rect(30, 50, 130, 130, LINE, '#f1f5f9', 0, 1.5); s.rect(43, 63, 104, 104, BLUE, None, 0, 2)
    s.text(95, 120, '32 × 32', 13, INK, 'middle', True); s.text(95, 200, '사방 4 px 0 패딩 → 40 × 40', 12, MUTED, 'middle')
    _arrow(s, 170, 115, 225, 115, 'RandomCrop', -10)
    s.rect(240, 50, 130, 130, LINE, '#f1f5f9', 0, 1.5); s.rect(257, 58, 104, 104, ORANGE, 'none', 0, 3)
    s.text(309, 200, '무작위 위치에서 32 × 32 → ±4 px 이동', 12, MUTED, 'middle')
    _arrow(s, 380, 115, 440, 115, 'Flip 50%', -10)
    s.box(455, 63, 104, 104, '좌우 반전', GREEN, '정답 그대로', 13)
    _arrow(s, 565, 115, 600, 115)
    s.text(645, 105, 'ToTensor', 12, INK, 'middle'); s.text(645, 125, 'Normalize', 12, INK, 'middle'); s.text(645, 145, '(맨 뒤)', 11.5, MUTED, 'middle')
    s.text(350, 222, 'test 는 원본 그대로 · 숫자 6 을 뒤집으면 9 — 정답이 바뀌는 변형은 금지', 12.5, RED, 'middle', True)
    s.save('d2-augment.svg')


def d2_compare():
    s = Svg(700, 250)
    s.text(350, 22, '파라미터와 MAC 은 따로 — 네 모델 비교 (log 눈금)', 14, INK, 'middle', True)
    rows = [('MLP', 1707274, 1706496, '0.4994'), ('LeNet-5', 62006, 651720, '0.5489'), ('VGG-mini', 307946, 38653952, '0.7526'), ('ResNet-20', 272474, 40813184, '0.8587*')]
    x0, x1 = 130, 600
    X = lambda v: x0 + (math.log10(v) - 4) / 4 * (x1 - x0)
    for e in range(4, 9):
        s.line(X(10 ** e), 40, X(10 ** e), 200, '#e2e8f0', 1); s.text(X(10 ** e), 216, f'1e{e}', 11.5, MUTED, 'middle')
    for r, (nm, pa, ma, acc) in enumerate(rows):
        y = 46 + r * 40
        s.text(120, y + 22, nm, 13, INK, 'end', True)
        s.rect(x0, y + 4, X(pa) - x0, 14, BLUE, None, 2, 1.2); s.rect(x0, y + 20, X(ma) - x0, 14, ORANGE, None, 2, 1.2)
        s.text(X(pa) + 5, y + 16, f'{pa:,}', 11, INK); s.text(X(ma) + 5, y + 32, f'{ma:,}', 11, INK)
        s.text(690, y + 24, acc, 12.5, GREEN, 'end', True)
    s.rect(140, 228, 14, 10, BLUE, None, 2, 1); s.text(160, 237, '파라미터', 12, INK)
    s.rect(240, 228, 14, 10, ORANGE, None, 2, 1); s.text(260, 237, 'MAC (1장)', 12, INK)
    s.text(690, 237, 'test (* 20 epoch + 증강, 나머지 3 epoch)', 11.5, MUTED, 'end')
    s.save('d2-compare.svg')


# ---------------- VB Day 3 ----------------

def d3_latency():
    s = Svg(680, 240)
    s.text(340, 22, 'CPU · 스레드 1 · 배치 1 지연시간 (실행 예) — 실시간 30 FPS 는 한 장 33 ms 안', 14, INK, 'middle', True)
    rows = [('ResNet-20 (32²)', 2.87, BLUE), ('MobileNetV3-S (224²)', 8.8, GREEN), ('ResNet-18 (224²)', 70, ORANGE), ('ResNet-50 (224²)', 134, RED)]
    x0, x1 = 190, 640
    X = lambda v: x0 + v / 140 * (x1 - x0)
    for r, (nm, v, c) in enumerate(rows):
        y = 44 + r * 38
        s.text(180, y + 18, nm, 13, INK, 'end')
        s.rect(x0, y + 4, X(v) - x0, 22, c, None, 3, 1.5); s.text(X(v) + 6, y + 20, f'{v:g} ms', 12, INK)
    s.line(X(33), 36, X(33), 196, RED, 2, dash='5,4'); s.text(X(33) + 4, 206, '33 ms', 12, RED, bold=True)
    s.text(340, 230, 'MAC: ResNet-50 4,089,184,256 vs MobileNetV3-S 56,510,400 (약 72배) — ImageNet top-1 은 76.1% vs 67.7%', 12, MUTED, 'middle')
    s.save('d3-latency.svg')


def d3_bottleneck():
    s = Svg(700, 230)
    s.text(350, 22, 'bottleneck — 채널이 많은 곳에서 바로 3 × 3 을 하지 않고 1 × 1 로 줄여서', 14, INK, 'middle', True)
    def bar(x, y, w, h, c, t):
        s.rect(x, y, w, h, c, None, 4, 1.5); s.text(x + w / 2, y + h + 16, t, 12, INK, 'middle')
    s.text(20, 60, 'basic', 13, INK, bold=True)
    bar(90, 40, 40, 70, BLUE, '256'); s.text(160, 70, '3×3', 12, MUTED, 'middle'); bar(190, 40, 40, 70, BLUE, '256')
    s.text(260, 70, '3×3', 12, MUTED, 'middle'); bar(290, 40, 40, 70, BLUE, '256')
    s.text(210, 140, '2 × 256·256·9 = 1,179,648', 12.5, RED, 'middle', True)
    s.text(380, 60, 'bottleneck', 13, INK, bold=True)
    bar(470, 40, 40, 70, BLUE, '256'); s.text(530, 70, '1×1', 12, MUTED, 'middle'); bar(550, 62, 40, 26, ORANGE, '64')
    s.text(605, 70, '3×3', 12, MUTED, 'middle'); bar(620, 62, 40, 26, ORANGE, '64')
    s.text(580, 140, '16,384 + 36,864 + 16,384 = 69,632', 12.5, GREEN, 'middle', True)
    s.text(580, 160, '(마지막 1×1 로 다시 256)', 11.5, MUTED, 'middle')
    s.text(350, 196, '가중치 1/17 (16.9배 적음) — 1×1 Conv = 위치마다 채널만 섞는 "채널 압축기"', 13, INK, 'middle', True)
    s.text(350, 218, '파일 크기 = 파라미터 × 4 B · 실행 메모리 = + 활성화 · 계산 = MAC (ResNet-50 은 가중치만 약 102 MB)', 12, MUTED, 'middle')
    s.save('d3-bottleneck.svg')


def d3_dsconv():
    s = Svg(700, 250)
    s.text(350, 22, 'Depthwise Separable — 공간 섞기와 채널 섞기를 둘로 쪼갠다 (64 → 128, 3 × 3)', 14, INK, 'middle', True)
    s.text(120, 52, '일반 3 × 3', 13.5, INK, 'middle', True)
    for k in range(4): s.rect(40 + k * 10, 84 - k * 6, 60, 60, BLUE, None, 3, 1.3)
    s.text(150, 100, '× 128 필터', 12.5, INK)
    s.text(120, 160, '필터 하나가 64채널 × 3×3', 12, MUTED, 'middle'); s.text(120, 180, '128·64·9 = 73,728', 13, RED, 'middle', True)
    s.text(470, 52, 'DS = depthwise → pointwise', 13.5, INK, 'middle', True)
    for k in range(4): s.rect(290 + k * 22, 70, 18, 60, ORANGE, None, 3, 1.3)
    s.text(345, 150, 'dw: 채널마다 3×3', 12, INK, 'middle'); s.text(345, 168, '(groups = 64) 64·9 = 576', 12, MUTED, 'middle')
    _arrow(s, 395, 100, 440, 100)
    for k in range(4): s.rect(455 + k * 10, 76 - k * 6, 40, 40, GREEN, None, 3, 1.3)
    s.text(500, 150, 'pw: 1×1 × 128', 12, INK, 'middle'); s.text(500, 168, '(채널만) 64·128 = 8,192', 12, MUTED, 'middle')
    s.text(625, 100, '8,768', 15, GREEN, 'middle', True); s.text(625, 120, '(11.9%)', 12, MUTED, 'middle')
    s.text(350, 210, 'DS / 일반 = 1/Cout + 1/K² = 1/128 + 1/9 = 0.1189 — MAC 75,497,472 → 8,978,432 (32 × 32)', 13, INK, 'middle', True)
    s.text(350, 234, '실제 속도는 덜 빨라진다: MAC 8.4배 ↓ → 실측 4.1배 (depthwise 는 계산 대비 메모리 이동이 많다)', 12, RED, 'middle')
    s.save('d3-dsconv.svg')


def d3_inverted():
    s = Svg(700, 240)
    s.text(350, 22, 'ResNet bottleneck (줄였다 늘리기) vs MobileNetV2 inverted residual (늘렸다 줄이기)', 14, INK, 'middle', True)
    def col(x, y, w, h, c, t, sub):
        s.rect(x, y - h / 2, w, h, c, None, 4, 1.5); s.text(x + w / 2, y + h / 2 + 16, t, 12, INK, 'middle'); s.text(x + w / 2, y + h / 2 + 31, sub, 11, MUTED, 'middle')
    yc = 100
    col(30, yc, 36, 90, BLUE, '256', ''); col(110, yc, 36, 26, ORANGE, '64', '1×1'); col(190, yc, 36, 26, ORANGE, '64', '3×3'); col(270, yc, 36, 90, BLUE, '256', '1×1')
    for a in (66, 146, 226): s.line(a + 4, yc, a + 40, yc, MUTED, 2, True)
    s.text(168, 200, '블록 사이 텐서가 굵다 (256)', 12, MUTED, 'middle')
    col(390, yc, 36, 26, BLUE, '24', ''); col(470, yc, 36, 90, ORANGE, '144', '1×1 ×6'); col(550, yc, 36, 90, ORANGE, '144', 'dw 3×3'); col(630, yc, 36, 26, BLUE, '24', '1×1, 선형')
    for a in (426, 506, 586): s.line(a + 4, yc, a + 40, yc, MUTED, 2, True)
    s.text(528, 200, '블록 사이 텐서가 얇다 (24) → 메모리 절약', 12, MUTED, 'middle')
    s.poly([(408, 86), (408, 52), (648, 52), (648, 86)], RED, 2); s.text(528, 46, '지름길 (stride 1 · cin = cout 일 때)', 11.5, RED, 'middle', True)
    s.text(350, 228, 'depthwise 는 싸서 넓은(144) 곳에서 해도 부담이 적다 · 24 → 24 블록 파라미터 8,832', 12.5, INK, 'middle', True)
    s.save('d3-inverted.svg')


def d3_hswish():
    s = Svg(700, 260)
    p = Plot(s, 60, 36, 380, 180, (-5, 5), (-1, 5))
    p.axes((-4, -3, 0, 3, 4), (0, 2, 4), 'x', '', grid=True)
    relu6 = lambda v: min(max(0, v), 6)
    p.curve(lambda x: max(0, x), LINE, 200, 2.5, dash='6,4')
    p.curve(lambda x: x / (1 + math.exp(-x)), ORANGE, 200, 2.5)
    p.curve(lambda x: x * relu6(x + 3) / 6, BLUE, 200, 3)
    for i, (c, nm) in enumerate(((BLUE, 'h-swish = x · ReLU6(x + 3) / 6'), (ORANGE, 'swish = x · σ(x)'), (LINE, 'ReLU'))):
        s.line(475, 60 + i * 26, 500, 60 + i * 26, c, 3); s.text(506, 65 + i * 26, nm, 12, INK)
    s.text(475, 150, 'x ≤ −3 → 0', 12, MUTED); s.text(475, 170, '−3 ~ 3 → 2차 곡선', 12, MUTED); s.text(475, 190, 'x ≥ 3 → x', 12, MUTED)
    s.text(350, 22, 'h-swish — swish 를 sigmoid(exp) 없이 근사 (장치에서 싸고 양자화에 유리)', 13.5, INK, 'middle', True)
    s.text(350, 250, 'MobileNetV3 = NAS 로 찾은 구조 + SE(채널 중요도) + h-swish', 12.5, MUTED, 'middle')
    s.save('d3-hswish.svg')


def d3_se():
    s = Svg(700, 200)
    s.text(350, 22, 'SE (Squeeze-and-Excitation) — 이미지마다 "지금 중요한 채널" 을 다시 정한다', 14, INK, 'middle', True)
    items = [('특징맵', '(C, H, W)', BLUE, 100), ('squeeze', '채널 평균 → C', ORANGE, 112), ('FC → ReLU', 'C → C/r', PURPLE, 104),
             ('FC → σ', '→ C, 0 ~ 1', PURPLE, 104), ('excitation', '채널마다 곱', GREEN, 112)]
    x = 14
    for i, (a, b, c, w) in enumerate(items):
        s.box(x, 50, w, 64, a, c, b, 13)
        if i < len(items) - 1: s.line(x + w, 82, x + w + 12, 82, MUTED, 2, True)
        x += w + 16
    s.poly([(64, 114), (64, 150), (630, 150), (630, 116)], MUTED, 1.8, dash='5,4')
    s.text(350, 168, '원래 특징맵 × 채널 가중치 → 같은 shape (C, H, W)', 12.5, MUTED, 'middle')
    s.text(350, 190, 'MobileNetV3-Small: InvertedResidual 11개 중 9개에 SE · 파라미터 2,542,856 · MAC 56,510,400 (224)', 12, INK, 'middle')
    s.save('d3-se.svg')


def d3_transfer():
    s = Svg(700, 240)
    s.text(350, 22, '전이학습 — ImageNet 으로 배운 특징 추출기는 가져오고, 분류기 마지막 층만 교체', 14, INK, 'middle', True)
    s.rect(20, 50, 370, 110, BLUE, None, 8, 2)
    s.text(205, 80, 'features (927,008)', 14, INK, 'middle', True)
    s.text(205, 104, '에지 · 질감 · 모양 — 일반 특징', 12.5, MUTED, 'middle')
    s.text(205, 132, '❄ 동결 = feature extraction', 12.5, BLUE, 'middle', True); s.text(205, 150, '같이 학습 = fine-tuning (작은 lr)', 12, MUTED, 'middle')
    _arrow(s, 392, 105, 420, 105)
    s.rect(425, 50, 255, 110, PURPLE, None, 8, 2)
    s.text(552, 76, 'classifier (601,098)', 14, INK, 'middle', True)
    s.text(552, 98, '576→1024 · Hardswish · Dropout', 12, MUTED, 'middle')
    s.text(552, 122, '[3] Linear(1024, 1000)', 12.5, MUTED, 'middle'); s.line(480, 118, 625, 118, RED, 1.5)
    s.text(552, 146, '→ Linear(1024, 10)', 13, RED, 'middle', True)
    s.text(350, 188, '전체 2,542,856 − 1,025,000 + 10,250 = 1,528,106 · 같은 1 epoch (실행 예):', 12.5, INK, 'middle')
    s.text(350, 210, 'feature extraction 0.6607 · fine-tuning 0.8273 · 처음부터 0.5470', 13, GREEN, 'middle', True)
    s.text(350, 232, '전처리는 사전학습 때와 같게 — ImageNet mean · std, 입력 32 → 96', 12, RED, 'middle')
    s.save('d3-transfer.svg')


def d3_bnmomentum():
    s = Svg(640, 250)
    p = Plot(s, 70, 36, 420, 170, (0, 160), (0, 1))
    p.axes((0, 40, 80, 120, 160), (0, 0.5, 1), 'step', '', grid=True)
    p.curve(lambda t: 0.99 ** t, RED, 160, 3); p.curve(lambda t: 0.9 ** t, GREEN, 160, 3)
    s.circle(p.X(160), p.Y(0.99 ** 160), 5, RED, RED); s.text(p.X(160) - 6, p.Y(0.99 ** 160) - 10, '0.20', 12.5, RED, 'end', True)
    s.line(505, 70, 530, 70, RED, 3); s.text(536, 75, 'momentum 0.01', 12.5, INK)
    s.line(505, 96, 530, 96, GREEN, 3); s.text(536, 101, 'momentum 0.1', 12.5, INK)
    s.text(505, 136, '초기값(평균 0 · 분산 1)', 12, MUTED); s.text(505, 154, '이 남아 있는 비율', 12, MUTED); s.text(505, 172, '= (1 − momentum)ᵗ', 12, MUTED)
    s.text(320, 22, 'BN running 통계가 초기값을 벗어나는 속도 — 1,000장 · 배치 64 · 10 epoch = 160 step', 13.5, INK, 'middle', True)
    s.text(320, 244, 'momentum 0.01 이면 eval() 에서 엉뚱한 통계로 정규화 → test 0.10(찍기) — set_bn_momentum(m, 0.1)', 12, RED, 'middle')
    s.save('d3-bnmomentum.svg')


def d3_gradcam():
    s = Svg(720, 230)
    s.text(360, 22, 'Grad-CAM — 클래스 점수를 마지막 합성곱 특징맵 A 로 미분해 "어디를 보고 판단했나"', 14, INK, 'middle', True)
    items = [('① A 저장', 'forward hook (K, h, w)', BLUE), ('② ∂y^c/∂A', 'y^c 만 backward', RED), ('③ α_k', '기울기의 공간 평균', ORANGE),
             ('④ ReLU(Σ α_k A^k)', '점수를 올리는 위치만', GREEN), ('⑤ ÷ max, 키우기', 'bilinear → 원본에 겹침', PURPLE)]
    for i, (a, b, c) in enumerate(items):
        x = 8 + i * 142
        s.box(x, 50, 132, 64, a, c, b, 13)
        if i < 4: s.line(x + 132, 82, x + 140, 82, MUTED, 2, True)
    s.text(360, 146, 'CAM 해상도 = 마지막 특징맵 크기 — ResNet-20 (32 입력) 8 × 8 · MobileNetV3-S (96 입력) 3 × 3', 12.5, INK, 'middle')
    s.text(360, 170, 'cls 를 지정하면 "정답이라면 어디를 봐야 했나" — 오답의 예측 CAM 과 정답 CAM 을 나란히 비교', 12.5, MUTED, 'middle')
    s.text(360, 194, '예측 CAM 이 물체가 아닌 배경 · 가장자리에 몰려 있으면 맥락에 속은 것 → 다양한 배경 데이터를 더한다', 12.5, RED, 'middle')
    s.text(360, 218, '정밀한 위치가 필요하면 분류가 아니라 객체 탐지 (Day 4)', 12, MUTED, 'middle')
    s.save('d3-gradcam.svg')


def d3_compare():
    s = Svg(700, 250)
    s.text(350, 22, 'ResNet-20 (처음부터 20 epoch) vs MobileNetV3-S (전이학습 5 epoch) — 실행 예', 14, INK, 'middle', True)
    rows = [('파라미터', 272474, 1528106, '{:,}'), ('MAC', 40813184, 11057600, '{:,}'), ('지연시간 (ms)', 2.87, 6.18, '{:g}'), ('test', 0.8587, 0.8842, '{:.4f}')]
    for r, (nm, a, b, f) in enumerate(rows):
        y = 46 + r * 46
        s.text(130, y + 26, nm, 13, INK, 'end', True)
        m = max(a, b)
        s.rect(140, y + 6, 360 * a / m, 16, BLUE, None, 2, 1.2); s.text(140 + 360 * a / m + 6, y + 19, f.format(a), 11.5, INK)
        s.rect(140, y + 24, 360 * b / m, 16, GREEN, None, 2, 1.2); s.text(140 + 360 * b / m + 6, y + 37, f.format(b), 11.5, INK)
    s.rect(150, 232, 14, 10, BLUE, None, 2, 1); s.text(170, 241, 'ResNet-20 (32 × 32)', 12, INK)
    s.rect(330, 232, 14, 10, GREEN, None, 2, 1); s.text(350, 241, 'MobileNetV3-S (96 × 96)', 12, INK)
    s.text(690, 100, '파라미터 5.6배인데', 12, MUTED, 'end'); s.text(690, 118, 'MAC 은 1/3.7', 12, MUTED, 'end')
    s.text(690, 180, 'MAC 이 적어도', 12, RED, 'end'); s.text(690, 198, 'CPU 에선 더 느림', 12, RED, 'end')
    s.save('d3-compare.svg')


# ---------------- VB Day 4 ----------------

def d4_iou():
    s = Svg(700, 250)
    s.text(350, 22, 'IoU = 교집합 / 합집합 — 한쪽으로 5 만 밀려도 0.33 으로 떨어진다', 14, INK, 'middle', True)
    def pair(ox, a, b, title, val, col):
        k = 10
        s.rect(ox + a[0] * k, 50 + a[1] * k, (a[2] - a[0]) * k, (a[3] - a[1]) * k, BLUE, None, 0, 2.5)
        s.rect(ox + b[0] * k, 50 + b[1] * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k, ORANGE, None, 0, 2.5)
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        if x2 > x1 and y2 > y1: s.rect(ox + x1 * k, 50 + y1 * k, (x2 - x1) * k, (y2 - y1) * k, col, '#cfe8d5', 0, 1.5)
        s.text(ox + 80, 185, title, 12.5, MUTED, 'middle'); s.text(ox + 80, 210, val, 15, col, 'middle', True)
    pair(30, (0, 0, 10, 10), (5, 5, 15, 15), 'A (0,0,10,10) · B (5,5,15,15)', 'IoU = 25 / 175 = 0.1429', GREEN)
    pair(260, (0, 0, 10, 10), (0, 0, 10, 10), '완전히 같음', 'IoU = 100 / 100 = 1', GREEN)
    pair(490, (5, 0, 15, 10), (0, 0, 10, 10), '가로로 5 만 밀림', 'IoU = 50 / 150 = 0.3333', GREEN)
    s.text(350, 240, '안 겹치면 교집합 0 → IoU 0 · 교집합 폭 · 높이가 음수면 0 으로 자른다 (np.clip)', 12, RED, 'middle')
    s.save('d4-iou.svg')


def d4_nms():
    s = Svg(700, 260)
    s.text(350, 22, 'NMS — 점수 순으로 하나 남기고, 많이 겹친 박스(IoU > 0.5)를 지운다', 14, INK, 'middle', True)
    boxes = [(100, 100, 200, 300, 0.95, 1), (110, 105, 205, 310, 0.90, 0), (300, 120, 380, 290, 0.85, 1), (95, 90, 190, 290, 0.60, 0), (305, 110, 390, 300, 0.40, 0)]
    k = 0.55
    def draw(ox, only_keep):
        for (a, b, c, d, sc, keep) in sorted(boxes, key=lambda q: q[5]):
            if only_keep and not keep: continue
            col = GREEN if keep else RED
            s.rect(ox + (a - 80) * k, 40 + (b - 80) * k, (c - a) * k, (d - b) * k, col, 'none', 0, 2.5 if keep else 1.5)
            s.text(ox + (a - 80) * k + 3 + (0 if keep else 24), 40 + (b - 80) * k + 14 + (0 if keep else 0), f'{sc:.2f}', 11, col)
    draw(30, False); draw(385, True)
    s.text(150, 245, 'NMS 전 — 박스 5개 (IoU 0.80 · 0.78 · 0.75 로 겹침)', 12, MUTED, 'middle')
    s.text(500, 245, 'NMS 후 — keep = [0, 2], 사람 2명', 12.5, GREEN, 'middle', True)
    _arrow(s, 335, 130, 375, 130)
    s.save('d4-nms.svg')


def d4_ap():
    s = Svg(700, 300)
    s.text(350, 22, '손 계산 AP — 정답 4개 · 탐지 5개 (TP, TP, FP, TP, FP)', 14, INK, 'middle', True)
    x0, y0, W, H = 60, 40, 300, 210
    X = lambda r: x0 + r * W
    Y = lambda p: y0 + H - p * H
    s.line(x0, y0 + H, x0 + W, y0 + H, INK, 1.5); s.line(x0, y0, x0, y0 + H, INK, 1.5)
    for v in (0, 0.5, 1):
        s.text(x0 - 6, Y(v) + 4, f'{v:g}', 11, MUTED, 'end'); s.text(X(v), y0 + H + 16, f'{v:g}', 11, MUTED, 'middle')
    s.text(x0 + W / 2, y0 + H + 34, 'recall', 12, MUTED, 'middle'); s.text(24, y0 - 6, 'precision', 12, MUTED)
    for a, b, p in ((0, 0.5, 1.0), (0.5, 0.75, 0.75)):
        s.rect(X(a), Y(p), (b - a) * W, p * H, BLUE, '#dbe7f7', 0, 1.5)
    pts = [(0.25, 1.0), (0.5, 1.0), (0.5, 0.6667), (0.75, 0.75), (0.75, 0.6)]
    s.poly([(X(r), Y(p)) for r, p in pts], ORANGE, 2, 'none')
    for r, p in pts: s.circle(X(r), Y(p), 4, ORANGE, ORANGE)
    s.text(X(0.62), Y(0.75) + 16, '0.75', 11, INK, 'middle')
    rows = [('순위', '판정', '누적 P', 'R'), ('1', 'TP', '1.00', '0.25'), ('2', 'TP', '1.00', '0.50'), ('3', 'FP', '0.67', '0.50'), ('4', 'TP', '0.75', '0.75'), ('5', 'FP', '0.60', '0.75')]
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            s.text(400 + j * 70, 70 + i * 24, c, 13, INK if i else MUTED, 'start', i == 0)
    s.text(400, 225, 'envelope(오른쪽 최댓값) 아래 넓이', 12.5, INK)
    s.text(400, 247, 'AP = 0.25×1 + 0.25×1 + 0.25×0.75', 12.5, BLUE)
    s.text(400, 272, '   = 0.6875', 15, BLUE, 'start', True)
    s.save('d4-ap.svg')


def d4_yolo():
    s = Svg(700, 250)
    s.text(350, 22, 'YOLOv8 — 640×640 입력, stride 3가지에서 한 번에 예측 → 8,400 칸', 14, INK, 'middle', True)
    s.box(20, 40, 100, 60, '입력', BLUE, '640×640×3', 14)
    s.box(150, 40, 110, 60, 'Backbone', GREEN, 'CSP', 14)
    s.box(290, 40, 110, 60, 'Neck', ORANGE, 'FPN + PAN', 14)
    s.box(430, 40, 120, 60, 'Head', PURPLE, 'anchor-free', 14)
    s.box(580, 40, 100, 60, '출력', RED, '(1, 84, 8400)', 14)
    for a, b in ((120, 150), (260, 290), (400, 430), (550, 580)): _arrow(s, a, 70, b, 70)
    for i, (st, n) in enumerate(((8, 80), (16, 40), (32, 20))):
        y = 130 + i * 32
        s.text(95, y + 16, f'stride {st}', 13, INK, 'end', True)
        s.rect(105, y, n * 2.2, 22, [GREEN, ORANGE, PURPLE][i], None, 2, 1.5)
        s.text(115 + n * 2.2, y + 16, f'{n}×{n} = {n * n:,} 칸', 12.5, INK)
    s.text(560, 160, '6,400 + 1,600 + 400', 13, INK, 'middle'); s.text(560, 184, '= 8,400', 16, RED, 'middle', True)
    s.text(350, 240, '84 = 박스 4 (cx cy w h) + 클래스 80 (sigmoid 점수)', 12, MUTED, 'middle')
    s.save('d4-yolo.svg')



if __name__ == '__main__':
    prefixes = tuple(sys.argv[1:]) or ('d1_', 'd2_', 'd3_', 'd4_')
    for name, fn in list(globals().items()):
        if callable(fn) and name.startswith(prefixes): fn()
