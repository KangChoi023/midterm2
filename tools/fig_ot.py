#!/usr/bin/env python3
"""OT(ONNX · TensorRT) 노트 "상세 학습" 의 그림(SVG)을 만든다 → assets/fig/ot/*.svg

    python3 -I tools/fig_ot.py          # 전부 다시 만들기
    python3 -I tools/fig_ot.py d3_      # 함수 이름 앞부분으로 일부만

그리기 도구(Svg · Plot)는 tools/fig_nn.py 의 것을 그대로 쓴다. 규칙도 같다 — 도형 + 진짜 글자, 흰 배경,
숫자는 강의 노트북을 실제로 실행한 값(notes/cell_outputs.ot*.json) 이나 직접 계산한 값.
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_nn
from fig_nn import Svg, Plot, INK, MUTED, LINE, BLUE, ORANGE, GREEN, RED, PURPLE, FILL

fig_nn.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'fig', 'ot')


def _arrow(s, x1, y1, x2, y2, label=None, dy=-8, size=12):
    s.line(x1, y1, x2, y2, MUTED, 2, True)
    if label: s.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, size, MUTED, 'middle')


def _node(s, x, y, w, h, op, sub=None, color=BLUE):
    s.rect(x, y, w, h, color, None, 6, 1.6)
    s.text(x + w / 2, y + (h / 2 + 5 if not sub else h / 2 - 2), op, 13, INK, 'middle', True)
    if sub: s.text(x + w / 2, y + h / 2 + 14, sub, 10.5, MUTED, 'middle')


# ---------------- OT Day 1 ----------------

def d1_path():
    s = Svg(700, 270)
    s.text(350, 22, '학습하는 곳 vs 실행하는 곳 — ONNX 는 "계산 순서 + 가중치" 를 한 파일로', 14, INK, 'middle', True)
    s.rect(15, 40, 300, 150, BLUE, None, 10, 2)
    s.text(165, 62, '학습 (Colab · PyTorch)', 13.5, INK, 'middle', True)
    for i, t in enumerate(['① forward  ② loss  ③ backward', '④ step  ⑤ 반복 — 정답 y 필요', '.pt = 가중치 숫자만 (state_dict)',
                           '계산 순서는 파이썬 클래스 코드에']):
        s.text(30, 88 + i * 22, t, 12, INK if i < 2 else MUTED)
    s.rect(385, 40, 300, 150, GREEN, None, 10, 2)
    s.text(535, 62, '실행 (현장 장치 · 다른 언어)', 13.5, INK, 'middle', True)
    for i, t in enumerate(['전처리 (/255 · 정규화 · 모양) — ONNX 밖', 'forward 만 — ONNX 안', '결과 해석 (argmax · 이름) — ONNX 밖',
                           'Python · PyTorch 가 없을 수 있다']):
        s.text(400, 88 + i * 22, t, 12, RED if i == 1 else INK if i < 3 else MUTED, 'start', i == 1)
    _arrow(s, 318, 115, 382, 115, 'export', -8)
    xs = [30, 165, 310, 455]
    labs = [('PyTorch', BLUE), ('model.onnx', ORANGE), ('ONNX Runtime', GREEN), ('TensorRT', RED)]
    for i, (n, c) in enumerate(labs):
        s.rect(xs[i], 212, 120, 34, c, None, 6, 1.5); s.text(xs[i] + 60, 234, n, 13, INK, 'middle', True)
        if i: _arrow(s, xs[i - 1] + 120, 229, xs[i] - 2, 229)
    s.text(582, 234, '(Day 4 · 5)', 11.5, MUTED); s.text(370, 262, '(Day 1 · 3)', 11.5, MUTED, 'middle')
    s.text(195, 262, '설계도 — 속도는 실행기가 정한다', 11.5, MUTED, 'middle')
    s.save('d1-path.svg')


def d1_tiny():
    s = Svg(700, 270)
    s.text(350, 22, 'nn.Linear(3, 2) → ONNX 노드 하나 (Gemm) — y = x·Wᵀ + b', 14, INK, 'middle', True)
    s.rect(20, 95, 110, 40, LINE, '#f8fafc', 6, 1.5); s.text(75, 112, 'input', 12.5, INK, 'middle', True)
    s.text(75, 127, 'float32[1, 3]', 11, MUTED, 'middle')
    s.rect(200, 64, 150, 102, BLUE, None, 6, 1.6); s.text(275, 88, 'Gemm', 14, INK, 'middle', True)
    s.text(275, 106, 'transB = 1', 11, MUTED, 'middle')
    s.text(275, 132, 'B ⟨2×3⟩ = weight', 11.5, ORANGE, 'middle'); s.text(275, 152, 'C ⟨2⟩ = bias', 11.5, ORANGE, 'middle')
    s.rect(420, 95, 110, 40, LINE, '#f8fafc', 6, 1.5); s.text(475, 112, 'output', 12.5, INK, 'middle', True)
    s.text(475, 127, 'float32[1, 2]', 11, MUTED, 'middle')
    _arrow(s, 130, 115, 198, 115); _arrow(s, 350, 115, 418, 115)
    s.text(560, 80, 'W = [[1, 2, 3],', 12.5, INK); s.text(560, 98, '      [0, −1, 1]]', 12.5, INK)
    s.text(560, 118, 'b = [0.5, −0.5]', 12.5, INK)
    s.text(560, 146, 'x = [1, 1, 1] →', 12.5, INK)
    s.text(560, 164, '[6.5, −0.5]', 13, RED, 'start', True)
    s.text(20, 200, 'export = 더미 입력 (1, 3) 으로 한 번 실행하며 지나간 연산을 녹화 — 값은 상관없고 shape · dtype 이 중요', 12, INK)
    s.text(20, 222, '파일 크기: 파라미터 8개 × 4 B = 32 B + 그래프 · 이름 · 버전 217 B = tiny.onnx 249 B', 12, INK)
    s.text(20, 244, 'weight 는 (출력 2, 입력 3) 으로 저장 → Gemm 이 transB = 1 로 전치해서 곱한다', 12, MUTED)
    s.save('d1-tiny.svg')


def d1_ort():
    s = Svg(700, 280)
    s.text(350, 22, 'ONNX Runtime 실행 4단계 — 입력은 딕셔너리, 출력은 리스트', 14, INK, 'middle', True)
    steps = [('① 열기', 'InferenceSession(path,', 'providers=[CPU…])'), ('② 이름', 'sess.get_inputs()[0]', '.name · .shape'),
             ('③ 실행', 'sess.run(None,', '{name: float32 배열})[0]'), ('④ 비교', 'np.allclose(out, ref,', 'atol=1e-5)')]
    for i, (t, a, b) in enumerate(steps):
        x = 15 + i * 172
        s.rect(x, 42, 158, 86, [BLUE, ORANGE, GREEN, PURPLE][i], None, 8, 1.8)
        s.text(x + 79, 64, t, 13.5, INK, 'middle', True)
        s.text(x + 79, 88, a, 11.5, INK, 'middle'); s.text(x + 79, 106, b, 11.5, INK, 'middle')
        if i: _arrow(s, x - 14, 85, x - 2, 85)
    s.text(20, 156, '틀리면 나오는 메시지', 13, INK, 'start', True)
    rows = [('float64 배열 (np.ones 기본)', 'INVALID_ARGUMENT : Unexpected input data type', '.astype(np.float32)'),
            ('입력 이름 틀림 {"x": …}', 'INVALID_ARGUMENT : Invalid input name', 'get_inputs()[0].name'),
            ('배치 4 를 넣음 (export 는 1)', 'Got invalid dimensions … Got: 4 Expected: 1', 'Day 2 dynamic_axes'),
            ('torch 텐서를 그대로', 'NumPy 만 받는다', '.cpu().numpy()')]
    for i, (a, b, c) in enumerate(rows):
        y = 180 + i * 22
        s.text(20, y, a, 11.5, INK); s.text(230, y, b, 11.5, RED); s.text(548, y, '→ ' + c, 11.5, GREEN)
    s.text(350, 272, 'PyTorch 와 차이 4.77e-07 (난수 20개) — float 끝자리 차이는 정상이라 == 가 아니라 allclose', 11.5, MUTED, 'middle')
    s.save('d1-ort.svg')


def d1_cnn():
    s = Svg(700, 250)
    s.text(350, 22, 'SimpleConvNet 의 ONNX 그래프 — 노드 7개, 가중치 있는 노드 3개 (파라미터 433,890)', 14, INK, 'middle', True)
    nodes = [('Conv', '30×1×5×5', BLUE, '(1,30,24,24)'), ('Relu', '', LINE, ''), ('MaxPool', '2×2', LINE, '(1,30,12,12)'),
             ('Flatten', '', LINE, '(1,4320)'), ('Gemm', '100×4320', BLUE, '(1,100)'), ('Relu', '', LINE, ''), ('Gemm', '10×100', BLUE, '(1,10)')]
    x = 20
    s.text(x, 70, 'input', 12, INK, 'start', True); s.text(x, 86, '(1,1,28,28)', 10.5, MUTED)
    x = 92
    for i, (op, w, c, shp) in enumerate(nodes):
        _node(s, x, 52, 76, 44 if not w else 50, op, w or None, c if c != LINE else PURPLE)
        if shp: s.text(x + 38, 122, shp, 10.5, MUTED, 'middle')
        if i < 6: _arrow(s, x + 76, 77, x + 84, 77)
        x += 85
    s.text(20, 158, '파라미터 = Conv 30×25 + 30 = 780 · Gemm 4320×100 + 100 = 432,100 · Gemm 100×10 + 10 = 1,010 → 433,890', 12, INK)
    s.text(20, 180, '파일 ≈ 433,890 × 4 B = 1.74 MB — 가중치의 99.6% 가 fc1 (Flatten 뒤 첫 Gemm)', 12, INK)
    s.text(20, 206, '전처리: (img.astype(float32) / 255)[None, None] → ORT 와 PyTorch 일치율 1.000 · 정확도 1.000 (100장)', 12, GREEN)
    s.text(20, 228, '/255 를 빼면? 오류 없이 정확도만 조용히 떨어진다 (1.00 → 0.92) — 가장 찾기 어려운 버그', 12, RED)
    s.save('d1-cnn.svg')


def d1_preprocess():
    s = Svg(700, 270)
    s.text(350, 22, '사진 한 장 → 모델 입력 (1, 3, 224, 224) — 전처리 6단계는 ONNX 밖의 코드', 14, INK, 'middle', True)
    st = [('① imread', '(H, W, 3) uint8', 'BGR 순서'), ('② RGB · resize', '(224, 224, 3)', 'resize 는 (W, H)'),
          ('③ /255 · 정규화', '(x − MEAN) / STD', '범위 −2.12 ~ 2.62'), ('④ transpose', '(3, 224, 224)', 'HWC → CHW'),
          ('⑤ 배치 축', '(1, 3, 224, 224)', 'x[None] float32'), ('⑥ run · softmax', '(1000,) → top-5', 'exp(l − max) / 합')]
    for i, (t, a, b) in enumerate(st):
        x = 12 + (i % 3) * 228; y = 44 + (i // 3) * 92
        s.rect(x, y, 212, 76, [BLUE, BLUE, ORANGE, GREEN, GREEN, PURPLE][i], None, 8, 1.6)
        s.text(x + 106, y + 24, t, 13.5, INK, 'middle', True)
        s.text(x + 106, y + 45, a, 12, INK, 'middle'); s.text(x + 106, y + 64, b, 11, MUTED, 'middle')
        if i % 3 < 2: _arrow(s, x + 212, y + 38, x + 226, y + 38)
    s.text(20, 238, 'MEAN = [0.485, 0.456, 0.406] · STD = [0.229, 0.224, 0.225] (ImageNet 학습 때 값) — np.float32 로 만들어야 결과도 float32', 11.5, INK)
    s.text(20, 258, '⚠ reshape(3, 224, 224) 는 모양만 바꿈 — 축 순서는 transpose(2, 0, 1) · BGR 그대로 · 정규화 누락은 오류 없이 다른 답', 11.5, RED)
    s.save('d1-preprocess.svg')


def d1_bnfold():
    s = Svg(700, 230)
    s.text(350, 22, 'SafetyNet — PyTorch 의 BatchNorm 은 ONNX 에서 Conv 안으로 합쳐진다', 14, INK, 'middle', True)
    s.text(20, 60, 'PyTorch', 13, BLUE, 'start', True)
    seq = ['Conv', 'BN', 'ReLU', 'Pool', 'Conv', 'BN', 'ReLU', 'Pool', 'GAP', 'Flatten', 'Linear']
    x = 90
    for op in seq:
        c = RED if op == 'BN' else BLUE
        s.rect(x, 44, 50, 26, c, None, 4, 1.4); s.text(x + 25, 62, op, 11.5, INK, 'middle', op == 'BN'); x += 54
    s.text(20, 120, 'ONNX', 13, GREEN, 'start', True)
    seq2 = ['Conv*', 'Relu', 'MaxPool', 'Conv*', 'Relu', 'MaxPool', 'GlobalAvgPool', 'Flatten', 'Gemm']
    x = 90
    for op in seq2:
        w = 84 if op == 'GlobalAvgPool' else 58
        s.rect(x, 104, w, 26, GREEN, None, 4, 1.4); s.text(x + w / 2, 122, op, 11.5, INK, 'middle'); x += w + 6
    s.text(20, 160, 'Conv* = BN 을 접어 넣은 Conv: W′ = W · γ/√(σ²+ε), b′ = (b − μ) · γ/√(σ²+ε) + β — 실행 모드에선 BN 이 고정된 곱셈 · 덧셈', 11.5, INK)
    s.text(20, 182, '노드 11 → 9 · 파라미터 1,491 (BN 48개 포함) · safety.onnx 7,502 B · 난수 10개 최대 차이 8.20e-08', 11.5, INK)
    s.text(20, 206, '⚠ export 전에 safety.eval() — 학습 모드면 BN 이 배치 통계를 써서 PyTorch 기준값이 달라진다', 11.5, RED)
    s.save('d1-bnfold.svg')


# ---------------- OT Day 2 ----------------

def d2_graph():
    s = Svg(700, 290)
    s.text(350, 22, 'ONNX 파일 = 상자 안의 상자 — 화살표는 저장되지 않고 "이름이 같으면" 이어진다', 14, INK, 'middle', True)
    s.rect(10, 38, 680, 140, LINE, '#f8fafc', 10, 1.5); s.text(22, 56, 'ModelProto — ir_version · opset_import (17) · producer', 11.5, MUTED)
    s.rect(22, 64, 656, 106, BLUE, '#f0f9ff', 8, 1.5); s.text(34, 82, 'GraphProto (m.graph) — 목록 네 개', 11.5, BLUE, 'start', True)
    for i, (n, c, d) in enumerate((('input', GREEN, '1개 · 사진만'), ('node', BLUE, '7개 · 연산'), ('initializer', ORANGE, '6개 · 가중치'), ('output', GREEN, '1개 · logits'))):
        x = 36 + i * 160
        s.rect(x, 92, 148, 62, c, None, 6, 1.5); s.text(x + 74, 116, n, 13.5, INK, 'middle', True); s.text(x + 74, 138, d, 11.5, MUTED, 'middle')
    y = 200
    s.text(20, y, 'node 하나 = op_type + input 이름 목록 + output 이름 목록', 12.5, INK, 'start', True)
    _node(s, 20, y + 14, 120, 44, 'Conv', None, BLUE)
    s.text(150, y + 30, "output: ['/conv/Conv_output_0']", 11, MUTED)
    _node(s, 390, y + 14, 120, 44, 'Relu', None, PURPLE)
    s.text(520, y + 30, "input: ['/conv/Conv_output_0']", 11, MUTED)
    _arrow(s, 140, y + 44, 388, y + 44, '같은 이름 → 화살표', 16, 11.5)
    s.text(20, y + 80, "Conv 의 input = ['input', 'conv.weight', 'conv.bias'] — 데이터 1개 + initializer 이름 2개 · 바깥에서 넣는 것은 사진뿐", 11.5, INK)
    s.save('d2-graph.svg')


def d2_tools():
    s = Svg(700, 260)
    s.text(350, 22, 'ONNX 파일을 다루는 세 도구 — 셋 다 "결과가 맞는지" 는 말해 주지 않는다', 14, INK, 'middle', True)
    cols = [('opset', '연산 사전의 판 번호', BLUE, ['같은 GELU 도', 'opset 13 → 노드 9개', '(Erf · Div · Add · Mul …)', 'opset 20 → Gemm · Gelu 2개', '두 파일 결과 차이 0']),
            ('checker', '규칙에 맞는지 검사', ORANGE, ['Relu 입력 이름을 "nope" 로', '→ checker 가 끊긴 화살표 지적', '→ ORT 도 세션 생성 실패', '앞 노드 output 이름으로', '되돌리면 통과']),
            ('shape_inference', '중간 shape 채우기', GREEN, ['value_info 0 → 6개', '[1, 30, 24, 24] (Conv)', '[1, 30, 12, 12] (MaxPool)', '[1, 4320] (Flatten)', '결과는 새 모델 mi 에'])]
    for i, (t, d, c, lines) in enumerate(cols):
        x = 15 + i * 228
        s.rect(x, 40, 214, 188, c, None, 8, 1.8)
        s.text(x + 107, 64, t, 14, INK, 'middle', True); s.text(x + 107, 84, d, 11.5, MUTED, 'middle')
        for j, l in enumerate(lines): s.text(x + 14, 112 + j * 22, l, 11.5, INK)
    s.text(350, 250, '맞춤법 검사기 · 도면 치수처럼 "형식" 만 본다 → 결과는 늘 PyTorch 와 allclose 로 따로 확인', 11.5, RED, 'middle')
    s.save('d2-tools.svg')


def d2_dynamic():
    s = Svg(700, 270)
    s.text(350, 22, 'dynamic_axes — 축에 숫자 대신 이름을 붙여 "크기가 바뀔 수 있다" 고 약속', 14, INK, 'middle', True)
    s.rect(15, 42, 320, 92, LINE, '#f8fafc', 8, 1.5)
    s.text(30, 64, '고정 (Day 1)', 13, INK, 'start', True); s.text(30, 88, 'input [1, 1, 28, 28]', 12.5, INK)
    s.text(30, 112, '배치 4 → Got: 4 Expected: 1 ✘', 12, RED)
    s.rect(365, 42, 320, 92, GREEN, None, 8, 1.5)
    s.text(380, 64, 'dynamic_axes={"input": {0: "batch"}, …}', 12, INK, 'start', True)
    s.text(380, 88, "input ['batch', 1, 28, 28] → ['batch', 10]", 12.5, INK)
    s.text(380, 112, '배치 1 · 4 · 8 · 100 모두 ✔ (100장 정확도 1.0)', 12, GREEN)
    s.text(20, 162, '높이 · 너비까지 열면? {0: "batch", 2: "height", 3: "width"} — 약속일 뿐, 모델이 못 받는 크기는 실행 중 오류', 12.5, INK, 'start', True)
    steps = [('32×32', LINE), ('Conv 5×5', BLUE), ('28×28', LINE), ('MaxPool', BLUE), ('14×14', LINE), ('Flatten', BLUE), ('5,880', RED), ('Gemm ✘', RED)]
    x = 20
    for t, c in steps:
        w = 74
        s.rect(x, 178, w, 30, c, None, 5, 1.4); s.text(x + w / 2, 198, t, 12, INK, 'middle', c == RED); x += w + 10
    s.text(20, 234, 'Conv · Pool 은 어떤 크기든 되지만 fc1 은 정확히 30 × 12 × 12 = 4,320 개를 원한다 (32×32 면 30 × 14 × 14 = 5,880)', 11.5, MUTED)
    s.text(20, 254, '오류: GEMM: Dimension mismatch, W: {100,4320} K: 5880 — 열어도 되는 축은 모델이 정말 처리할 수 있는 축뿐', 11.5, RED)
    s.save('d2-dynamic.svg')


def d2_traps():
    s = Svg(700, 300)
    s.text(350, 22, '조용히 틀리는 export 3가지 — 더미 입력으로 "실행된 길" 만 녹화된다', 14, INK, 'middle', True)
    rows = [('① 파이썬 if', 'if x.mean() > 0: x*2 else x-1', '더미 ones → x*2 길만 녹화', '음수 입력: ONNX [−6,−4,−2] ≠ 원래 [−4,−3,−2]', 'torch.where(조건, x*2, x−1)', 'Where'),
            ('② NumPy · float()', 'm = float(np.mean(x.numpy()))', '더미 [5,5,5] → m = 5 상수', '[1,2,3]: ONNX [−4,−3,−2] ≠ 원래 [−1,0,1]', 'x − x.mean(dim=1, keepdim=True)', 'ReduceMean'),
            ('③ 숫자로 쓴 크기', 'x.view(1, −1)', 'dynamic 이어도 1 로 고정', '배치 4 → (1, 64) — 실행 오류', 'x.flatten(1) · view(x.size(0), −1)', 'Flatten')]
    s.text(20, 52, '원인', 12, MUTED, 'start', True); s.text(250, 52, '증상 (에러 없음!)', 12, MUTED, 'start', True); s.text(500, 52, '고치기 → 새 노드', 12, MUTED, 'start', True)
    for i, (t, code, why, bad, fix, node) in enumerate(rows):
        y = 64 + i * 66
        s.rect(15, y, 670, 58, [ORANGE, PURPLE, BLUE][i], None, 8, 1.4)
        s.text(25, y + 20, t, 13, INK, 'start', True); s.text(25, y + 40, code, 11, MUTED)
        s.text(250, y + 20, why, 11.5, INK); s.text(250, y + 40, bad, 11, RED)
        s.text(500, y + 20, fix, 11, GREEN, 'start', True); s.text(500, y + 40, '→ ' + node + ' 노드', 11, MUTED)
    s.text(20, 280, '확인법: verify() 에 더미와 "다른" 입력(음수 · 다른 사진 · 배치 4) 을 넣고, TracerWarning 한 줄을 읽는다 — 더미 하나면 늘 ✅', 11.5, INK)
    s.save('d2-traps.svg')


def d2_slim():
    s = Svg(700, 250)
    s.text(350, 22, 'onnxslim — 같은 계산을 더 적은 노드로 (ViewNet, x.view(x.size(0), −1))', 14, INK, 'middle', True)
    rows = [('접기 끔', ['Identity', 'Identity', 'Conv', 'BN', 'Relu', 'Shape', 'Const', 'Gather', 'Const', 'Const', 'Unsq', 'Const', 'Unsq', 'Concat', 'Reshape', 'Gemm'], 16),
            ('export 기본', ['Conv', 'Relu', 'Shape', 'Const', 'Gather', 'Const', 'Unsq', 'Const', 'Concat', 'Reshape', 'Gemm'], 11),
            ('onnxslim 후', ['Conv', 'Relu', 'Reshape', 'Gemm'], 4)]
    for i, (t, ops, n) in enumerate(rows):
        y = 46 + i * 52
        s.text(16, y + 20, t, 12, INK, 'start', True); s.text(16, y + 36, f'{n}개', 12, BLUE, 'start', True)
        x = 100
        for op in ops:
            c = RED if op in ('Shape', 'Gather', 'Unsq', 'Concat', 'Const', 'Identity') else ORANGE if op == 'BN' else GREEN
            s.rect(x, y + 6, 34, 30, c, None, 4, 1.2); s.text(x + 17, y + 25, op[:6], 8.5, INK, 'middle'); x += 36
    s.text(20, 210, '빨강 = 배치 크기를 매번 읽어 shape 를 조립하는 노드 → 미리 계산해 접기(constant folding) · BN 은 Conv 에 합침', 11.5, INK)
    s.text(20, 232, '단순화 전 · 후 배치 4 결과 차이 0.0 — 정확도가 오르는 게 아니라 같은 계산을 짧게 적을 뿐 · 결과는 onnx.save 로 저장', 11.5, MUTED)
    s.save('d2-slim.svg')


# ---------------- OT Day 3 ----------------

def d3_knobs():
    s = Svg(700, 270)
    s.text(350, 22, 'ONNX 파일은 그대로, 실행하는 쪽 손잡이만 바꾼다 — 한 번에 하나씩', 14, INK, 'middle', True)
    s.rect(270, 44, 160, 64, ORANGE, None, 8, 2); s.text(350, 72, 'mnv3s.onnx', 14, INK, 'middle', True); s.text(350, 92, '그대로', 11.5, MUTED, 'middle')
    knobs = [('스레드', 'intra_op_num_threads', '3교시', BLUE, 20, 44), ('최적화 레벨', 'graph_optimization_level', '4교시', GREEN, 20, 120),
             ('장치', 'providers = [CUDA, CPU]', '5교시', RED, 480, 44), ('배치', '입력 (b, 3, 224, 224)', '6교시', PURPLE, 480, 120)]
    for t, a, c, col, x, y in knobs:
        s.rect(x, y, 200, 58, col, None, 8, 1.6); s.text(x + 100, y + 22, f'{t} ({c})', 13, INK, 'middle', True); s.text(x + 100, y + 42, a, 11, MUTED, 'middle')
        _arrow(s, x + 200 if x < 300 else x, y + 29, 270 if x < 300 else 430, 76 if y < 100 else 100)
    s.text(20, 206, '원칙 ① 한 번에 하나만 바꾼다  ② 조건을 표의 열로 적는다 (모델 · 실행기 · 스레드 · 레벨 · 배치 · 장치)', 11.5, INK)
    s.text(20, 228, '        ③ 바꾼 뒤에도 결과는 같아야 한다 (allclose) — 빨라졌는데 답이 달라지면 소용없다', 11.5, INK)
    s.text(20, 254, 'providers 는 희망 목록 — 설치됨 [Azure, CPU] · 실제 사용 [CPU] → GPU 가 없으면 조용히 CPU 로, get_providers() 로 확인', 11.5, MUTED)
    s.save('d3-knobs.svg')


def d3_threads():
    s = Svg(700, 270)
    s.text(350, 22, '스레드 · 최적화 레벨을 하나씩 바꿔 잰 결과 (mnv3s.onnx, 배치 1, CPU 4코어)', 14, INK, 'middle', True)
    p = Plot(s, 60, 50, 240, 160, (0, 3.2), (0, 3.0))
    th = [(1, 2.16, 2.44), (2, 1.83, 2.02), (4, 1.87, 2.01)]
    s.line(60, 210, 300, 210, INK, 1.3)
    for v in (0, 1, 2, 3): s.text(54, p.Y(v) + 4, str(v), 11, MUTED, 'end')
    for i, (t, m, p95) in enumerate(th):
        x = 80 + i * 75
        s.rect(x, p.Y(m), 28, p.Y(0) - p.Y(m), BLUE, None, 0, 1.2); s.rect(x + 30, p.Y(p95), 28, p.Y(0) - p.Y(p95), ORANGE, None, 0, 1.2)
        s.text(x + 14, p.Y(m) - 4, f'{m}', 10.5, INK, 'middle'); s.text(x + 44, p.Y(p95) - 4, f'{p95}', 10.5, INK, 'middle')
        s.text(x + 29, 226, f'스레드 {t}', 11.5, INK, 'middle')
    s.text(180, 246, '평균(파랑) · p95(주황) ms — 1→2 는 빨라지고 2→4 는 그대로', 11, MUTED, 'middle')
    q = Plot(s, 400, 50, 280, 160, (0, 4), (0, 4))
    s.line(400, 210, 680, 210, INK, 1.3)
    for v in (0, 1, 2, 3, 4): s.text(394, q.Y(v) + 4, str(v), 11, MUTED, 'end')
    lv = [('DISABLE', 3.38, '1.00'), ('BASIC', 2.99, '1.13'), ('EXTENDED', 2.79, '1.21'), ('ALL', 1.58, '2.14')]
    for i, (n, m, sp) in enumerate(lv):
        x = 412 + i * 68
        s.rect(x, q.Y(m), 44, q.Y(0) - q.Y(m), GREEN if n == 'ALL' else LINE, None, 0, 1.2)
        s.text(x + 22, q.Y(m) - 18, f'{m}', 11, INK, 'middle'); s.text(x + 22, q.Y(m) - 5, f'×{sp}', 10.5, RED if n == 'ALL' else MUTED, 'middle', n == 'ALL')
        s.text(x + 22, 226, n, 10.5, INK, 'middle')
    s.text(540, 246, '평균 ms · speedup = DISABLE ÷ 그 레벨 (ALL 이 기본값)', 11, MUTED, 'middle')
    s.text(350, 264, '코어 수 = 보통 상한 · 스레드를 늘려도 작은 모델은 나누는 비용 때문에 거의 그대로 — 결과(출력)는 레벨과 무관하게 같다', 11, INK, 'middle')
    s.save('d3-threads.svg')


def d3_fusion():
    s = Svg(700, 276)
    s.text(350, 22, '최적화 레벨마다 ORT 가 고친 그래프 — mnist_cnn_dyn.onnx (optimized_model_filepath 로 저장)', 14, INK, 'middle', True)
    rows = [('DISABLE', ['Conv', 'Relu', 'MaxPool', 'Flatten', 'Gemm', 'Relu', 'Gemm'], '그대로'),
            ('BASIC', ['Conv', 'Relu', 'MaxPool', 'Flatten', 'Gemm', 'Relu', 'Gemm'], '상수 접기'),
            ('EXTENDED', ['FusedConv', 'MaxPool', 'Flatten', 'FusedGemm', 'Gemm'], '노드 합치기'),
            ('ALL', ['Conv', 'ReorderOutput', 'MaxPool', 'Flatten', 'FusedGemm', 'Gemm'], '레이아웃')]
    for i, (n, ops, d) in enumerate(rows):
        y = 44 + i * 46
        s.text(16, y + 22, n, 12.5, INK, 'start', True); s.text(16, y + 37, f'{len(ops)}개', 11, BLUE)
        x = 100
        for op in ops:
            w = 78 if len(op) > 7 else 60
            c = RED if op.startswith('Fused') else ORANGE if op.startswith('Reorder') else BLUE
            s.rect(x, y + 6, w, 30, c, None, 4, 1.3); s.text(x + w / 2, y + 26, op, 10.5, INK, 'middle', c != BLUE); x += w + 5
        s.text(x + 6, y + 26, d, 10.5, MUTED)
    s.text(20, 244, '합치기 = 중간 결과를 메모리에 썼다 다시 읽는 왕복이 줄어든다 · 레이아웃 = CPU 명령에 맞는 데이터 배치', 11, INK)
    s.text(20, 264, 'Fused* · Reorder* 는 com.microsoft 도메인 = ORT 전용 연산 — 저장한 최적화 그래프는 TensorRT · 다른 CPU 에 쓰지 않는다', 11, RED)
    s.save('d3-fusion.svg')


def d3_gpu():
    s = Svg(700, 210)
    s.text(350, 22, 'GPU 실행 = 복사 → 계산 → 복사 — 계산이 빨라도 나르는 시간이 생긴다', 14, INK, 'middle', True)
    s.rect(20, 50, 180, 70, BLUE, None, 8, 1.8); s.text(110, 78, 'CPU 메모리', 13.5, INK, 'middle', True); s.text(110, 100, 'NumPy 입력 · 출력', 11.5, MUTED, 'middle')
    s.rect(500, 50, 180, 70, RED, None, 8, 1.8); s.text(590, 78, 'GPU (CUDA EP)', 13.5, INK, 'middle', True); s.text(590, 100, '계산은 빠름', 11.5, MUTED, 'middle')
    _arrow(s, 200, 70, 498, 70, '① 입력 복사 (Host → Device)', -8)
    s.line(498, 104, 200, 104, MUTED, 2, True); s.text(350, 122, '③ 결과 복사 (Device → Host)', 12, MUTED, 'middle')
    s.text(590, 140, '② 계산', 12, RED, 'middle', True)
    s.text(20, 166, '배치 1: 계산이 작아 복사 · 첫 실행 준비(수백 ms) 비중이 크다 → 이득 작음 · 배치가 크면 계산이 커져 GPU 가 크게 유리', 11.5, INK)
    s.text(20, 188, '이 환경(CPU 4코어): 배치 1 = 1.66 ms · 배치 32 = 20.33 ms (장당 0.64 ms) — GPU 결과는 CPU 와 차이 < 1e-3 이어야', 11.5, MUTED)
    s.save('d3-gpu.svg')


def d3_batch():
    s = Svg(700, 270)
    s.text(350, 22, '배치를 키우면 — 지연은 늘고, 처리량은 어느 지점에서 멈춘다 (mnv3s · CPU)', 14, INK, 'middle', True)
    data = [(1, 1.64, 611), (4, 3.67, 1090), (16, 9.21, 1737), (64, 43.89, 1458)]
    X = lambda i: 90 + i * 130
    p = Plot(s, 70, 50, 420, 170, (0, 3), (0, 50))
    s.line(70, 220, 500, 220, INK, 1.3); s.line(70, 50, 70, 220, RED, 1.3); s.line(500, 50, 500, 220, GREEN, 1.3)
    for v in (0, 10, 20, 30, 40, 50): s.text(64, p.Y(v) + 4, str(v), 10.5, RED, 'end')
    for v in (0, 500, 1000, 1500, 2000): s.text(506, 220 - v / 2000 * 170 + 4, str(v), 10.5, GREEN)
    s.text(70, 44, '지연 ms', 11.5, RED, 'middle', True); s.text(500, 44, '처리량 장/초', 11.5, GREEN, 'middle', True)
    s.line(70, p.Y(33.3), 500, p.Y(33.3), RED, 1, dash='4,4'); s.text(76, p.Y(33.3) - 4, '30 FPS 예산 33.3 ms', 10.5, RED)
    s.poly([(X(i), p.Y(l)) for i, (b, l, t) in enumerate(data)], RED, 2.5)
    s.poly([(X(i), 220 - t / 2000 * 170) for i, (b, l, t) in enumerate(data)], GREEN, 2.5)
    for i, (b, l, t) in enumerate(data):
        s.circle(X(i), p.Y(l), 4, RED, RED); s.circle(X(i), 220 - t / 2000 * 170, 4, GREEN, GREEN)
        s.text(X(i), 236, f'배치 {b}', 11.5, INK, 'middle')
        s.text(X(i) + 8, p.Y(l) - 7, f'{l}', 10.5, RED); s.text(X(i), 220 - t / 2000 * 170 - 8, f'{t}', 10.5, GREEN, 'middle')
    x0 = 545
    s.text(x0, 70, '처리량 = 배치 × 1000 ÷ 지연', 11.5, INK, 'start', True)
    s.text(x0, 92, '장당 = 지연 ÷ 배치', 11.5, INK)
    s.text(x0, 114, '1.64 → 0.58 ms/장 (배치 16)', 11, MUTED)
    s.text(x0, 142, '카메라 수 = 처리량 ÷ 30', 11.5, INK, 'start', True)
    s.text(x0, 164, '배치 1: 20대', 11, MUTED); s.text(x0, 184, '배치 16: 57대 (지연 9 ms)', 11, MUTED)
    s.text(350, 262, '실시간 카메라 → 배치 1 · p95 / 녹화 영상 · 여러 카메라 → 처리량이 가장 큰 배치 (여기선 16, 64 는 오히려 감소)', 11.5, INK, 'middle')
    s.save('d3-batch.svg')


def d3_profile():
    s = Svg(700, 250)
    s.text(350, 22, 'profiling — 노드별 시간 비율 (20번 실행의 dur 합, 단위 µs)', 14, INK, 'middle', True)
    mn = [('Conv', 42.4), ('Mul', 18.7), ('HardSigmoid', 14.8), ('ReorderOutput', 9.5), ('ReorderInput', 5.7), ('GlobalAvgPool', 5.1)]
    cn = [('FusedGemm', 30.4), ('Conv', 20.0), ('ReorderOutput', 16.2)]
    s.text(20, 52, 'MobileNetV3-S (노드 142개)', 12.5, BLUE, 'start', True)
    for i, (n, v) in enumerate(mn):
        y = 62 + i * 22
        s.text(140, y + 14, n, 11, INK, 'end'); s.rect(146, y + 2, v * 4.2, 16, BLUE if i == 0 else LINE, None, 0, 1); s.text(150 + v * 4.2, y + 14, f'{v}%', 10.5, INK)
    s.text(420, 52, 'SimpleConvNet', 12.5, GREEN, 'start', True)
    for i, (n, v) in enumerate(cn):
        y = 62 + i * 22
        s.text(530, y + 14, n, 11, INK, 'end'); s.rect(536, y + 2, v * 3.6, 16, GREEN if i == 0 else LINE, None, 0, 1); s.text(540 + v * 3.6, y + 14, f'{v}%', 10.5, INK)
    s.text(420, 150, 'fc1 (100 × 4320) 이 가중치의 99.6%', 11, MUTED)
    s.text(20, 212, 'Mul · HardSigmoid = HardSwish · SE 블록 조각 · Reorder* = 4교시 ALL 의 레이아웃 바꾸기 비용', 11, MUTED)
    s.text(20, 234, '병목은 모델마다 다르다 → 손댈 곳이 정해진다 (Conv 를 빠르게: TensorRT · INT8 양자화 — Day 4 · 다음 과목)', 11.5, INK)
    s.save('d3-profile.svg')


if __name__ == '__main__':
    prefixes = tuple(sys.argv[1:]) or ('d1_', 'd2_', 'd3_', 'd4_', 'd5_')
    for name, fn in list(globals().items()):
        if callable(fn) and name.startswith(prefixes): fn()
