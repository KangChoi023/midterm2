#!/usr/bin/env python3
"""QP Day 4 (QAT) 노트 "상세 학습" 의 그림(SVG) → assets/fig/qp/d4-*.svg

    python3 -I tools/fig_qp_d4.py            # Day 4 그림 전부
    python3 -I tools/fig_qp_d4.py d4_ste     # 함수 이름 앞부분으로 일부만

그리기 도구(Svg · Plot)는 tools/fig_nn.py 의 것을 쓴다 (fig_qp.py 와 같은 방식).
fake quant · STE 그림은 강의와 같은 식(scale 0.25, −4 ~ 3)으로 직접 계산한다.
정확도 숫자는 교안 노트북(QP-day4/QP_Day4_lecture.ipynb)의 정답 코드를 CPU 에서 직접 실행한 값
(PyTorch · ModelOpt 0.48 · ORT · 평가용 2,000장) — GPU / TensorRT 는 없어서 7 · 8교시 지연은 ORT CPU.
"""
import math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_nn
from fig_nn import Svg, Plot, INK, MUTED, BLUE, ORANGE, GREEN, RED, PURPLE

fig_nn.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'fig', 'qp')

FP32 = 0.846
D4_BITS = [(8, 0.8435, 98, 0.0072), (6, 0.844, None, None), (4, 0.8355, 12, 0.1275), (3, 0.730, 6, 0.3022), (2, 0.1205, 3, 0.7822)]
D4_HAND_W3 = [0.728, 0.8215, 0.8335, 0.8375]          # PTQ, epoch 1 · 2 · 3
D4_MO_W3 = [0.728, 0.8185, 0.8245, 0.832]
D4_DUEL = [  # (method, acc_int8, acc_w3a8, train_min, latency_ms ORT CPU)
    ('FP32', 0.846, None, 0.0, 0.713),
    ('ORT PTQ', 0.842, None, 0.0, 1.178),
    ('ModelOpt PTQ', 0.843, 0.728, 0.0, 1.675),
    ('수작업 QAT', 0.849, 0.8375, 9.98, None),
    ('ModelOpt QAT', 0.843, 0.832, 1.96, 1.583)]


def fq(x, s=0.25, lo=-4, hi=3):
    return min(max(round(x / s), lo), hi) * s


def d4_bits():
    s = Svg(700, 270)
    s.text(350, 22, 'weight-only PTQ — 비트를 줄이면 칸이 줄어든다 (활성값은 FP32)', 14, INK, 'middle', True)
    P = Plot(s, 70, 45, 320, 170, (0, 5), (0, 0.9))
    P.axes(yt=(0, 0.2, 0.4, 0.6, 0.8), grid=True)
    s.line(70, P.Y(FP32), 390, P.Y(FP32), MUTED, 1.3, dash='5 4')
    s.text(388, P.Y(FP32) - 5, 'FP32 0.846', 11, MUTED, 'end')
    for i, (b, a, _, _) in enumerate(D4_BITS):
        c = GREEN if a > 0.8 else (ORANGE if a > 0.5 else RED)
        x = P.X(i) + 14
        s.rect(x, P.Y(a), 36, P.Y(0) - P.Y(a), c, None, 2, 1.2)
        s.text(x + 18, P.Y(a) - 6, f'{a:.3f}', 11.5, INK, 'middle', True)
        s.text(x + 18, 233, f'{b}비트', 12, INK, 'middle')
    s.text(230, 256, '정확도 (ResNet-20 · 평가 2,000장)', 12, MUTED, 'middle')
    s.text(560, 55, 'layer1.0.conv1 채널 0 (weight 144개)', 12.5, INK, 'middle', True)
    s.text(470, 80, '비트', 12, MUTED, 'middle', True); s.text(540, 80, '칸 (±qmax)', 12, MUTED, 'middle', True)
    s.text(625, 80, '실제 값 · 오차', 12, MUTED, 'middle', True)
    y = 104
    for b, a, n, e in D4_BITS:
        if n is None: continue
        q = 2 ** (b - 1) - 1
        c = RED if b <= 3 else INK
        s.text(470, y, f'{b}', 12.5, c, 'middle', True)
        s.text(540, y, f'{2 * q + 1}칸 (±{q})', 12, c, 'middle')
        s.text(625, y, f'{n}개 · {e:.3f}', 12, c, 'middle')
        y += 26
    s.text(560, 222, '오차 = ‖ŵ − w‖ / ‖w‖', 11.5, MUTED, 'middle')
    s.text(560, 244, '3비트 → 채널마다 값 7개뿐', 12.5, RED, 'middle', True)
    s.save('d4-bits.svg')


def d4_fakequant():
    s = Svg(700, 300)
    s.text(350, 22, 'fake quant (scale 0.25 · −4 ~ 3) — forward 는 계단, 값은 FP32', 14, INK, 'middle', True)
    P = Plot(s, 60, 45, 330, 200, (-1.6, 1.6), (-1.2, 1.2))
    P.axes(xt=(-1.5, -1, -0.5, 0.5, 1, 1.5), yt=(-1, -0.5, 0.5, 1), xl='x', yl='x̂')
    P.curve(lambda x: x, MUTED, 2, 1.2, dash='4 4')
    pts, prev = [], None
    for i in range(641):
        x = -1.6 + 3.2 * i / 640
        y = fq(x)
        if prev is not None and y != prev:
            pts.append((P.X(x), P.Y(prev)))
        pts.append((P.X(x), P.Y(y))); prev = y
    s.poly(pts, BLUE, 2.5)
    s.line(P.X(-1.0), 45, P.X(-1.0), 245, RED, 1.2, dash='3 3'); s.line(P.X(0.75), 45, P.X(0.75), 245, RED, 1.2, dash='3 3')
    s.text(P.X(-1.0) - 4, 60, '−1.0', 11, RED, 'end'); s.text(P.X(0.75) + 4, 60, '0.75', 11, RED)
    s.text(225, 290, '빨간 점선 밖 = clamp — 끝 칸에 잘림 (5.0 → 0.75)', 11.5, RED, 'middle')
    s.text(550, 55, '같은 x 의 세 가지 값', 12.5, INK, 'middle', True)
    rows = [('x', [-1.2, -0.3, 0.0, 0.13, 0.9, 5.0], INK), ('q = clamp(round(x/s))', [-4, -1, 0, 1, 3, 3], ORANGE),
            ('x̂ = q · s', [-1.0, -0.25, 0.0, 0.25, 0.75, 0.75], BLUE)]
    for k, (lab, vals, c) in enumerate(rows):
        y = 85 + k * 48
        s.text(420, y, lab, 12, c, 'start', True)
        for j, v in enumerate(vals):
            s.text(430 + j * 44, y + 20, f'{v:g}', 12, c, 'middle' if False else 'start')
    s.text(550, 238, 'x̂ 는 FP32 지만 8개 값 중 하나', 12, MUTED, 'middle')
    s.text(550, 258, 'weight 는 FP32 로 학습 · 저장', 12, MUTED, 'middle')
    s.save('d4-fakequant.svg')


def d4_ste():
    s = Svg(700, 300)
    s.text(350, 22, 'backward 세 가지 — round 0 · STE 1 · clamp 범위 밖 0', 14, INK, 'middle', True)
    P = Plot(s, 50, 50, 300, 150, (-2.2, 2.2), (-0.3, 1.4))
    P.axes(xt=(-2, -1, 1, 2), yt=(1,), xl='x', yl='∂x̂/∂x')
    s.poly([(P.X(-2.2), P.Y(0)), (P.X(2.2), P.Y(0))], RED, 3)
    s.poly([(P.X(-2.2), P.Y(0.04)), (P.X(-1.0), P.Y(0.04)), (P.X(-1.0), P.Y(1)), (P.X(0.75), P.Y(1)),
            (P.X(0.75), P.Y(0.04)), (P.X(2.2), P.Y(0.04))], GREEN, 2.5)
    s.text(P.X(-0.1), P.Y(1) - 8, 'STE: 범위 안 1', 12, GREEN, 'middle', True)
    s.text(P.X(1.35), P.Y(0) - 22, '범위 밖 0', 11.5, GREEN, 'middle')
    s.text(P.X(-1.6), P.Y(0) + 18, 'torch.round: 어디서나 0', 11.5, RED, 'middle', True)
    for x, g in [(-2.0, 0), (-0.4, 1), (0.3, 1), (0.6, 1), (2.0, 0)]:
        s.circle(P.X(x), P.Y(g), 4.5, ORANGE)
    s.text(200, 235, '점 = 실습 3-2 의 x [−2.0, −0.4, 0.3, 0.6, 2.0] → 기울기 [0, 1, 1, 1, 0]', 11, ORANGE, 'middle')
    s.text(530, 55, 'toy 실험 — w 3개 · SGD lr 0.1 · 50번', 12.5, INK, 'middle', True)
    s.box(390, 70, 280, 58, 'round 버전: loss 0.1875', RED, 'w [0.31, −0.77, 0.20] 그대로', 13)
    s.box(390, 145, 280, 58, 'STE 버전: loss 0', GREEN, 'w → [0.41, −0.62, 0.40]', 13)
    s.text(530, 225, 'x̂ = [0.5, −0.5, 0.5] = target', 12, GREEN, 'middle', True)
    s.text(350, 270, 'forward 는 둘 다 계단 — 다른 것은 backward 뿐. QAT 는 이것을 수만 개 weight 에 cross-entropy 로', 11.5, MUTED, 'middle')
    s.save('d4-ste.svg')


def d4_qat():
    s = Svg(700, 300)
    s.text(350, 22, '직접 만드는 QAT (W3A8) — 네 단계', 14, INK, 'middle', True)
    steps = [('1 BN folding', 'BN 21 → 0 · 0.846', BLUE), ('2 QConv2d', '입력 + weight FQ', PURPLE),
             ('3 보정 = PTQ', 'W3A8 0.728', ORANGE), ('4 fine-tuning', '3 epoch · lr 1e-3', GREEN)]
    for i, (a, b, c) in enumerate(steps):
        x = 15 + i * 172
        s.box(x, 40, 155, 52, a, c, b, 13.5)
        if i < 3: s.line(x + 155, 66, x + 170, 66, MUTED, 2, True)
    P = Plot(s, 80, 120, 560, 130, (0, 3), (0.70, 0.86))
    P.axes(xt=(0, 1, 2, 3), yt=(0.72, 0.76, 0.80, 0.84), grid=True)
    s.line(80, P.Y(FP32), 640, P.Y(FP32), MUTED, 1.3, dash='5 4'); s.text(638, P.Y(FP32) - 5, 'FP32 0.846', 11, MUTED, 'end')
    s.line(80, P.Y(0.728), 640, P.Y(0.728), ORANGE, 1.3, dash='5 4'); s.text(638, P.Y(0.728) - 5, 'PTQ 0.728', 11, ORANGE, 'end')
    pts = [(P.X(i), P.Y(a)) for i, a in enumerate(D4_HAND_W3)]
    s.poly(pts, GREEN, 2.5)
    for i, a in enumerate(D4_HAND_W3):
        s.circle(P.X(i), P.Y(a), 4.5, GREEN)
        if i: s.text(P.X(i), P.Y(a) + 18, f'{a:.4f}', 11.5, GREEN, 'middle', True)
    s.text(360, 285, 'epoch (0 = QAT 전 PTQ) — 첫 epoch 에 대부분 회복 · +11.0%p', 12, INK, 'middle')
    s.save('d4-qat.svg')


def d4_modelopt():
    s = Svg(700, 280)
    s.text(350, 22, 'mtq.quantize(model, INT8_DEFAULT_CFG, forward_loop) — 삽입 + 보정을 한 줄로', 13.5, INK, 'middle', True)
    s.rect(30, 45, 300, 165, BLUE, None, 10, 1.6)
    s.text(180, 66, 'QuantConv2d (Conv 21개 · Linear 1개도)', 12.5, INK, 'middle', True)
    s.box(45, 80, 130, 44, 'input_quantizer', GREEN, '8비트 · 텐서 하나', 12)
    s.box(185, 80, 130, 44, 'weight_quantizer', GREEN, '8비트 · axis 0', 12)
    s.box(115, 140, 130, 44, 'output_quantizer', MUTED, '꺼짐', 12)
    s.text(180, 202, '모두 대칭 (zero_point 0) · max 보정', 11.5, MUTED, 'middle')
    s.text(510, 60, '삽입 110개 → 켜진 것 45개', 13, INK, 'middle', True)
    rows = [('입력 (Conv · Linear)', 22, GREEN), ('weight (Conv · Linear)', 22, GREEN), ('pool 입력', 1, GREEN),
            ('output · BN 양자화기', 65, MUTED)]
    for i, (lab, n, c) in enumerate(rows):
        y = 78 + i * 30
        s.text(470, y + 14, lab, 11.5, INK, 'end')
        s.rect(478, y + 2, n * 2.6, 18, c, None, 2, 1.2)
        s.text(482 + n * 2.6, y + 15, f'{n}', 11.5, INK)
    s.text(510, 215, 'INT8 PTQ 0.843 · W3A8 PTQ 0.728', 12.5, BLUE, 'middle', True)
    s.text(350, 245, 'BN 은 접지 않고 QuantBatchNorm2d 로 남는다 → 배포 때 TensorRT 가 접는다', 12, MUTED, 'middle')
    s.text(350, 267, 'make_cfg(3): config 를 깊은 복사 → "*weight_quantizer" 의 num_bits = 3', 12, ORANGE, 'middle')
    s.save('d4-modelopt.svg')


def d4_curve():
    s = Svg(700, 285)
    s.text(350, 22, 'W3A8 QAT 정확도 곡선 — ModelOpt vs 직접 만든 것 (epoch 0 = PTQ)', 14, INK, 'middle', True)
    P = Plot(s, 80, 45, 420, 180, (0, 3), (0.70, 0.86))
    P.axes(xt=(0, 1, 2, 3), yt=(0.72, 0.76, 0.80, 0.84), grid=True)
    s.line(80, P.Y(FP32), 500, P.Y(FP32), MUTED, 1.3, dash='5 4'); s.text(498, P.Y(FP32) - 5, 'FP32 0.846', 11, MUTED, 'end')
    s.line(80, P.Y(0.728), 500, P.Y(0.728), ORANGE, 1.3, dash='5 4'); s.text(498, P.Y(0.728) + 15, 'PTQ W3A8 0.728', 11, ORANGE, 'end')
    for data, c in [(D4_HAND_W3, GREEN), (D4_MO_W3, BLUE)]:
        s.poly([(P.X(i), P.Y(a)) for i, a in enumerate(data)], c, 2.5)
        for i, a in enumerate(data): s.circle(P.X(i), P.Y(a), 4, c)
    s.text(518, 92, 'ep', 12, MUTED, 'middle', True)
    s.text(555, 92, '직접', 12, GREEN, 'middle', True); s.text(640, 92, 'ModelOpt', 12, BLUE, 'middle', True)
    for i in range(4):
        y = 116 + i * 24
        s.text(518, y, str(i), 12, MUTED, 'middle')
        s.text(555, y, f'{D4_HAND_W3[i]:.4f}', 12, GREEN, 'middle')
        s.text(640, y, f'{D4_MO_W3[i]:.4f}', 12, BLUE, 'middle')
    s.text(600, 228, 'INT8: 0.843 → 0.843 (이득 0)', 12, RED, 'middle', True)
    s.text(290, 245, 'epoch', 12, MUTED, 'middle')
    s.text(350, 270, '저비트(W3A8)는 QAT 로 +10%p 넘게 회복 · INT8 은 PTQ 와 같다 — QAT 가 필요한 곳을 숫자로 판단', 11.5, INK, 'middle')
    s.save('d4-curve.svg')


def d4_deploy():
    s = Svg(700, 270)
    s.text(350, 22, 'QAT 모델 배포 — ModelOpt → QDQ ONNX → (보정 없이) TensorRT INT8', 14, INK, 'middle', True)
    s.box(15, 45, 190, 56, 'ModelOpt QAT 모델', BLUE, 'mo8 · 0.843', 13)
    s.box(255, 45, 190, 56, 'QDQ ONNX', GREEN, 'export_b1 · 1,135 KB', 13)
    s.box(495, 45, 190, 56, 'TensorRT 엔진', ORANGE, 'fp16 + int8 · 보정 없음', 13)
    s.line(205, 73, 253, 73, MUTED, 2, True); s.line(445, 73, 493, 73, MUTED, 2, True)
    s.text(350, 128, 'rn20_qat.onnx 노드 (직접 실행)', 12.5, INK, 'middle', True)
    rows = [('QuantizeLinear', 45, GREEN), ('DequantizeLinear', 45, GREEN), ('Conv', 21, BLUE), ('BatchNormalization', 21, PURPLE)]
    for i, (lab, n, c) in enumerate(rows):
        y = 140 + i * 24
        s.text(250, y + 14, lab, 11.5, INK, 'end')
        s.rect(258, y + 2, n * 4, 17, c, None, 2, 1.2)
        s.text(262 + n * 4, y + 15, str(n), 11.5, INK)
    s.text(560, 150, 'trt_check', 12, INK, 'middle', True)
    s.text(560, 170, 'qdq 90 · uint8 0 · nonzero_zp 0', 11.5, GREEN, 'middle')
    s.text(560, 190, '→ ok True', 12, GREEN, 'middle', True)
    s.text(560, 215, 'ORT 정확도 0.843 (= PyTorch)', 11.5, INK, 'middle')
    s.text(350, 258, 'QDQ ONNX 의 weight 는 아직 FP32 → 파일은 FP32(1,074 KB) 정도 · 작아지는 것은 엔진 · BN 은 TensorRT 가 접는다', 11, MUTED, 'middle')
    s.save('d4-deploy.svg')


def d4_duel():
    s = Svg(700, 300)
    s.text(350, 22, 'PTQ vs QAT 대결 — 같은 ResNet-20 · 같은 평가 2,000장 (지연 = ORT CPU · 배치 1)', 13.5, INK, 'middle', True)
    heads = [('INT8 정확도', 1, 1.0, '{:.4f}', 0.80), ('W3A8 정확도', 2, 1.0, '{:.4f}', 0.70), ('학습 분', 3, 11, '{:.1f}', 0), ('지연 ms', 4, 1.8, '{:.2f}', 0)]
    for j, (h, idx, mx, fmt, base) in enumerate(heads):
        x0 = 140 + j * 140
        s.text(x0 + 55, 52, h, 12.5, INK, 'middle', True)
        for i, row in enumerate(D4_DUEL):
            y = 62 + i * 38
            if j == 0: s.text(130, y + 18, row[0], 12.5, INK, 'end', True)
            v = row[idx]
            if v is None:
                s.text(x0 + 4, y + 19, '—', 12, MUTED); continue
            w = max(80 * (v - base) / (mx - base), 2)
            c = [MUTED, ORANGE, ORANGE, GREEN, GREEN][i]
            s.rect(x0, y + 4, w, 20, c, None, 3, 1.2)
            s.text(x0 + w + 5, y + 19, fmt.format(v), 11.5, INK)
    s.text(350, 262, 'QAT 이득: INT8 +0.6%p → 필요 없음 · W3A8 +10.95%p → 필요 (기준 1%p)', 12.5, INK, 'middle', True)
    s.text(350, 284, '정확도 막대는 0.80 · 0.70 부터 · CPU 에서는 INT8 이 FP32 보다 느리다 — TensorRT GPU 에서 다시 잰다', 11, MUTED, 'middle')
    s.save('d4-duel.svg')


if __name__ == '__main__':
    prefixes = tuple(sys.argv[1:]) or ('d4_',)
    for name, fn in list(globals().items()):
        if callable(fn) and name.startswith(prefixes):
            fn()
