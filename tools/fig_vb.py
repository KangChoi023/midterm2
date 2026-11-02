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


if __name__ == '__main__':
    prefixes = tuple(sys.argv[1:]) or ('d1_', 'd2_', 'd3_')
    for name, fn in list(globals().items()):
        if callable(fn) and name.startswith(prefixes): fn()
