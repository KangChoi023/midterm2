## Day 4 — QAT: 학습하면서 양자화에 적응시키기

### 1교시 · PTQ 의 한계: 비트를 줄이면

**키워드**: `weight-only PTQ`, `bits = 8 · 6 · 4 · 3 · 2`, `채널별 대칭`, `qmax = 2^(bits−1) − 1`, `칸 수`, `4비트 아래 붕괴`, `QAT 가 필요한 경우`, `rn20_ptq8.onnx`

**개요**: Day 2 · 3 에서 ResNet-20 은 8비트 PTQ 로 0.846 → 0.84x — 거의 손실이 없었다. 그런데 메모리 · 전력을 더 줄이려고 weight 를 4 · 3 · 2비트로 내리면? 채널별 대칭 양자화 `quant_weight(w, bits)` 로 모든 Conv · Linear weight 만 바꿔(활성값은 FP32) 재면 8비트 **0.8435** · 6비트 0.844 · 4비트 **0.8355** · 3비트 **0.730** · 2비트 **0.1205** — 4비트 아래에서 급격히 무너진다. 3비트면 한 채널이 쓸 수 있는 값이 7개(−3 ~ 3)뿐이기 때문. 오늘의 기준선은 8비트 PTQ(Day 3 의 TensorRT 용 S8 대칭 `make_trt_int8`): FP32 0.846 → **0.842** · 1,074 → 323 KB. 저비트 · 작은 모델 · depthwise 처럼 PTQ 가 무너지는 곳에서 쓰는 처방이 **QAT** — 이미 학습된 모델에 양자화를 끼워 넣고 짧게 fine-tuning 한다.

**한눈에 보기**
PTQ 의 한계 — 비트를 줄이면 무너진다
> 칸 수 = 2^bits 라서 3비트 대칭이면 채널마다 값이 7개뿐 — 반올림 오차가 weight 크기의 30% 에 이르러 PTQ 로는 정확도가 무너지고, 학습 중에 양자화를 겪게 하는 QAT 가 필요해진다
- **weight-only PTQ — 8비트 0.8435, 4비트 0.8355, 3비트 0.730, 2비트 0.1205**: FP32 0.846, 활성값은 FP32 그대로인데도
- **원인은 weight 칸 수**: 채널 0 이 쓰는 값 98 · 12 · 6 · 3개 → 상대 오차 0.007 · 0.127 · 0.302 · 0.782
- **8비트는 PTQ 로 충분**: ORT PTQ INT8 0.842 (−0.4%p), 1,074 → 323 KB
- **QAT 가 필요한 경우 — 저비트 · 작은 모델 · depthwise**: weight 하나의 역할이 크고 범위 차이가 크다
- **⚠ outlier 모델은 QAT 로도 잘 안 고쳐진다**: BadQuantNet 0.30 → 0.50 — mixed precision · 채널 균등화가 낫다

**핵심 용어**
- **QAT(Quantization-Aware Training, 양자화 인식 학습)**: 학습 forward 에 양자화(fake quant)를 끼워 넣어, 모델이 양자화 오차를 미리 겪으며 weight 를 조정하게 하는 방법
- **weight-only PTQ**: weight 만 저비트로 바꾸고 활성값은 FP32 로 두는 PTQ — 비트 수의 영향만 따로 볼 때
- **비트 다이얼**: 같은 모델을 비트 수만 바꿔 정확도를 재는 실험 (8 → 2비트)
- **fine-tuning(미세 조정)**: 이미 학습된 모델을 작은 learning rate · 짧은 epoch 로 조금 더 학습

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `quant_weight()` | `quant_weight(w, bits)` | 출력 채널별 대칭 quantize → dequantize (이 교시에서 만듦) |
| `torch.unique()` | `len(torch.unique(q[0]))` | 채널 하나가 실제로 쓰는 서로 다른 값 개수 |
| `export_b1()` | `export_b1(copy.deepcopy(rn20), "rn20_b1.onnx")` | 배치 1 고정 ONNX 로 (TensorRT 엔진용, 제공) |
| `onnx_acc()` | `onnx_acc("rn20_ptq8.onnx")` | ONNX 모델 평가용 2,000장 정확도 (한 장씩, 제공) |
| `make_trt_int8()` | `make_trt_int8("rn20_b1_pre.onnx", "rn20_ptq8.onnx")` | Day 3 의 TensorRT 용 S8 대칭 PTQ (제공) |

**핵심 내용**

#### 저비트 PTQ 의 붕괴
> 칸 수 = 2^bits — 3비트 대칭이면 채널마다 7개 값(−3 ~ 3)뿐이라 반올림 오차가 weight 크기의 30% 에 이른다
- **weight-only PTQ** → 8비트 **0.8435** · 6비트 0.844 · 4비트 **0.8355** · 3비트 **0.730** · 2비트 **0.1205** (FP32 0.846)
- **칸 수 · 오차** → layer1.0.conv1 채널 0 이 쓰는 값 98 · 12 · 6 · 3개 (8 · 4 · 3 · 2비트) · 상대 오차 0.007 · 0.127 · 0.302 · 0.782
- **8비트 기준선** → ORT PTQ INT8 **0.842** (−0.4%p) · 1,074 → **323 KB** — 8비트는 PTQ 로 충분
- ⚠ 활성값은 FP32 그대로인데도 무너진다 — 원인은 weight 칸 수

#### QAT 가 필요한 경우
> 학습 중에 양자화를 미리 겪게 해서, weight 가 "양자화된 뒤에도 잘 맞는 자리" 로 옮겨 가게 한다
- **저비트 (4 · 3 · 2비트)** → 칸이 16 · 8 · 4개 — PTQ 오차가 너무 크다
- **작은 모델 · depthwise Conv** → weight 하나의 역할이 크고, 채널마다 weight 9개 · 범위 차이가 크다 (QP D1·5교시)
- **순서** → FP32 학습 완료 → (BN folding) → FakeQuant 삽입 → 보정 → 짧은 fine-tuning → QDQ ONNX → TensorRT INT8
- ⚠ outlier 모델(Day 3 BadQuantNet)은 QAT 로도 잘 안 고쳐진다 (강사 실험 0.30 → 0.50) — mixed precision · 채널 균등화가 낫다

**기본 사용형태**
```python
def quant_weight(w, bits):
    qmax = 2 ** (bits - 1) - 1                                   # 3비트 → 3
    amax = w.abs().flatten(1).amax(1).view(-1, *[1] * (w.dim() - 1))   # 채널별 |w| 최댓값
    scale = (amax / qmax).clamp(min=1e-8)
    return torch.clamp(torch.round(w / scale), -qmax, qmax) * scale

m = copy.deepcopy(rn20)
with torch.no_grad():
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mod.weight.copy_(quant_weight(mod.weight, 3))
acc_torch(m)                                                     # 0.730
```

**연결**
- [QP D1·4교시 대칭 vs 비대칭](#/qp/1/4) — qmax = 2^(bits−1) − 1 · zero_point 0 인 대칭 식 그대로 (비트만 바꿈)
- [QP D1·5교시 Granularity](#/qp/1/5) — 채널별 scale: `amax(1)` 를 `(-1, 1, 1, 1)` 로 펴서 채널마다 나눈다
- [QP D3·4교시 TensorRT INT8 개념](#/qp/3/4) — 기준선 `make_trt_int8` (S8 · 대칭 · reduce_range)
- [QP D3·3교시 Mixed Precision](#/qp/3/3) — outlier 모델의 처방은 QAT 가 아니라 범인만 FP32

**상세 학습**

#### 이 교시에서 답하는 질문
Day 2 · 3 의 결론은 "ResNet-20 은 8비트 PTQ 로 충분" 이었다. 그렇다면 비트를 더 줄여도 될까? 이 교시는 weight 비트 수만 8 → 2 로 내려 가며 정확도를 재서 **PTQ 가 어디서 무너지는지** 보고, 그 문제를 푸는 방법(QAT)이 무엇인지 큰 그림을 잡는다.

#### 개념 1 · 비트를 줄이면 칸이 줄어든다
- 대칭 양자화: q = clamp(round(w / s), −qmax, qmax), ŵ = q · s, s = 채널별 |w| 최댓값 / qmax.
- qmax = 2^(bits−1) − 1 → 8비트 127 · 4비트 7 · 3비트 3 · 2비트 1. 칸 간격 s 가 그만큼 커진다.

| bits | qmax | 쓸 수 있는 값 | 채널 0 실제 값 수 | 상대 오차 ‖ŵ − w‖/‖w‖ |
|---|---|---|---|---|
| 8 | 127 | 255 | 98 | 0.0072 |
| 4 | 7 | 15 | 12 | 0.1275 |
| 3 | 3 | 7 | 6 | 0.3022 |
| 2 | 1 | 3 | 3 | 0.7822 |

(layer1.0.conv1 의 채널 0 = weight 144개 · 직접 실행.) 2비트면 weight 가 −s · 0 · +s 셋 중 하나 — 정보가 거의 남지 않는다.

##### 🔎 왜 이렇게 되나
- **비유**: 255칸 자에서 7칸 자로 바꾸면, 같은 길이를 훨씬 성기게 잴 수밖에 없다
- **① 칸이 줄어드는 이유**: qmax = 2^(bits−1) − 1 — 8비트 127, 4비트 7, 3비트 3, 2비트 1
- **② 오차가 커지는 이유**: 같은 범위를 적은 칸으로 — 칸 간격 s 가 커진다
- **③ 숫자로**: 상대 오차 8비트 0.0072 → 4비트 0.1275
- ⚠ 비트를 반으로 줄이면 오차는 반이 아니라 수십 배로 커진다

<<CODE 1-c1 | QAT 의 흐름>>
<<CODE 1-1 | quant_weight — 채널별 대칭 weight 양자화>>

#### 개념 2 · 비트 다이얼 결과
![weight-only PTQ 정확도 — 8 · 6 · 4비트는 거의 그대로, 3비트 0.730, 2비트 0.12 로 붕괴](assets/fig/qp/d4-bits.svg)

| 모델 | 정확도 | 크기 |
|---|---|---|
| FP32 | 0.846 | 1,074 KB |
| weight-only 8 · 6 · 4비트 | 0.8435 · 0.844 · 0.8355 | — |
| weight-only 3 · 2비트 | **0.730** · **0.1205** | — |
| ORT PTQ INT8 (W8A8, `rn20_ptq8.onnx`) | 0.842 | 323 KB |

(직접 실행 · 평가용 2,000장.) 교안 해설은 "8비트 PTQ 손실 0.3%p 정도" — 이 실행은 0.4%p (Day 3 교안 수정으로 `reduce_range=True` · weight 7비트 ±64 가 들어간 설정).

##### 🔎 왜 이렇게 되나
- **① 4비트까지 버티는 이유**: weight 만 양자화 — ResNet-20 은 이 정도 오차를 견딘다 (0.8355)
- **② 3비트부터 무너지는 이유**: 쓸 수 있는 값이 7개 — 0.730, 2비트는 0.1205 (거의 찍기)
- **③ PTQ 의 한계**: 학습이 끝난 weight 를 반올림만 하므로 큰 오차를 되돌릴 방법이 없다
- ⚠ 이 손실을 학습으로 되찾는 것이 QAT (4교시)

<<CODE 1-2 | weight-only PTQ 를 8 · 6 · 4 · 3 · 2비트로>>
<<CODE 1-c2 | Day 3 방식 ORT PTQ INT8 기준선>>

#### 한 번에 정리
- 칸 수 = 2^bits — 3비트 대칭이면 채널당 값 7개.
- weight-only PTQ: 4비트까지 0.83 대, 3비트 0.730, 2비트 0.12 — 4비트 아래에서 붕괴.
- 8비트 PTQ 는 0.842 (−0.4%p) — QAT 가 필요 없다.
- QAT = 이미 학습된 모델 + 양자화를 끼운 짧은 fine-tuning.
- ⚠ 비트마다 `copy.deepcopy(rn20)` 로 새 복사본 — 같은 모델을 계속 양자화하면 오차가 겹친다.

---

### 2교시 · Fake Quantization: 학습 그래프 안의 양자화

**키워드**: `fake quant`, `quantize → dequantize`, `x̂ 는 FP32 · 값은 2^bits 개`, `FakeQuant(nn.Module)`, `per_channel (weight · 대칭 · 매번 scale)`, `활성값 (텐서 하나 · 비대칭 · 보정 범위 고정)`, `register_buffer`, `round 기울기 0`

**개요**: 학습은 FP32 로 해야 기울기가 흐른다. 그래서 QAT 는 진짜 정수 연산 대신 **양자화했다가 바로 되돌린 FP32 값** — fake quant: q = clamp(round(x/s) + zp, qmin, qmax), x̂ = (q − zp) · s. x̂ 는 FP32 지만 2^bits 개의 값만 가지므로 모델은 "양자화된 세상" 을 본다. weight 는 FP32 로 계속 학습 · 저장되고, 진짜 INT8 은 배포할 때 만든다. 이것을 모듈 `FakeQuant` 로 만든다 — weight(per_channel)는 채널별 대칭 · scale 을 매번 다시 계산, 활성값은 텐서 하나 · 비대칭 · 보정 모드에서 잰 범위 lo ~ hi 고정. 문제: `torch.round` 의 기울기는 0 이라 loss 0.1875 인데 w.grad = [0, 0, 0] — 이대로는 학습이 멈춘다.

**한눈에 보기**
Fake Quantization — 학습 그래프 안의 양자화
> x̂ = (clamp(round(x/s) + zp) − zp) · s — 값은 FP32 지만 2^bits 개 칸 위에만 놓여, 학습 중에 모델이 양자화 오차를 그대로 겪는다
- **예 — s 0.25 · 범위 −4 ~ 3**: [−1.2, −0.3, 0, 0.13, 0.9, 5.0] → [−1.0, −0.25, 0, 0.25, 0.75, 0.75]
- **weight 는 FP32 로 학습 · 저장**: 진짜 INT8 은 배포(ONNX · TensorRT) 때 만든다
- **weight FakeQuant — 채널별 대칭, scale 은 매번 계산**: 3비트면 채널당 7개 값
- **활성값 FakeQuant — 텐서 하나 · 비대칭, 보정으로 lo ~ hi 고정**: −0.5 ~ 2.0 → s 0.0098, zp 51
- **round 의 기울기는 0**: loss 0.1875 인데 w.grad = [0, 0, 0] — 학습 신호가 끊긴다 → STE (3교시)
- **⚠ 범위 밖 값은 끝 칸으로 잘린다 (5.0 → 0.75)**: scale 의 amax 는 `x.detach()` — scale 은 학습 대상이 아니다

**핵심 용어**
- **Fake quantization(가짜 양자화)**: quantize → dequantize 를 한 번에 — 값은 FP32 인데 2^bits 개 칸 위에만 놓인다
- **보정 모드(calib)**: 활성값 FakeQuant 가 값을 바꾸지 않고 최솟값 · 최댓값(lo · hi)만 기록하는 상태
- **buffer(버퍼)**: `register_buffer` 로 등록한 텐서 — 학습되지 않지만 state_dict 에 저장되고 `.to(DEV)` 로 같이 옮겨진다
- **계단 함수**: round 처럼 평평한 구간이 이어지는 함수 — 평평한 곳의 미분은 0

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `fake_quant()` | `fake_quant(x, scale, zp, qmin, qmax)` | quantize → dequantize 한 번에 (이 교시에서 만듦) |
| `FakeQuant` | `FakeQuant(bits=8, per_channel=False)` | weight · 활성값용 fake quant 모듈 (이 교시에서 만듦) |
| `register_buffer()` | `self.register_buffer("lo", torch.tensor(0.0))` | 학습 안 되는 상태 텐서 등록 |
| `torch.clamp()` | `torch.clamp(torch.round(x / s) + zp, qmin, qmax)` | 범위 밖 값을 끝 칸으로 |
| `torch.minimum()` | `self.lo = torch.minimum(self.lo, x.detach().min())` | 보정 중 최솟값 갱신 (최댓값은 `torch.maximum`) |

**핵심 내용**

#### Fake quantization
> x̂ = (clamp(round(x/s) + zp, qmin, qmax) − zp) · s — FP32 값이지만 2^bits 개 칸 위에만 놓인다
- **forward** → 모델이 양자화 오차를 그대로 겪는다 (계단 모양 출력)
- **저장** → weight 는 FP32 로 학습 · 저장 — 진짜 INT8 은 배포(ONNX · TensorRT) 때 만든다
- **예** → s 0.25 · −4 ~ 3: [−1.2, −0.3, 0, 0.13, 0.9, 5.0] → [−1.0, −0.25, 0, 0.25, 0.75, 0.75]
- ⚠ 범위 밖 값은 끝 칸으로 잘린다 (5.0 → 0.75) — clamp

#### FakeQuant 모듈 — weight 와 활성값이 다르다
> weight = 채널별 대칭 · scale 을 매 forward 마다 계산 / 활성값 = 텐서 하나 · 비대칭 · 보정으로 정한 lo ~ hi 고정
- **weight (per_channel)** → s = 채널별 |w| 최댓값 / qmax, zp 0, −qmax ~ qmax — 3비트면 채널당 7개 값
- **활성값 보정 모드** → `calib=True` 이면 lo · hi 만 갱신하고 x 를 그대로 돌려준다
- **활성값** → 범위 min(lo, 0) ~ max(hi, 0) · s = (hi − lo)/255 · zp = round(−lo/s) — 예: −0.5 ~ 2.0 → s 0.0098 · zp 51, 3.0 → 2.0
- ⚠ scale 계산의 amax 는 `x.detach()` — scale 은 학습 대상이 아니다

#### round 의 기울기 = 0
> 계단의 평평한 곳은 미분 0 → `torch.round` 를 지난 기울기는 모두 0 → weight 가 한 걸음도 움직이지 않는다
- **예** → w [0.31, −0.77, 0.20] → x̂ [0.25, −0.75, 0.25] · target [0.5, −0.5, 0.5] → loss **0.1875** · w.grad **[0, 0, 0]**
- **왜** → loss 는 0 이 아닌데 기울기가 0 — 학습 신호가 끊겼다
- **해결** → 3교시 STE (backward 에서 round 를 없는 것처럼)
- ⚠ PyTorch `torch.round` 는 짝수 쪽 반올림 — −1.5 → −2, 0.5 → 0

**기본 사용형태**
```python
def fake_quant(x, scale, zp, qmin, qmax):
    return (torch.clamp(torch.round(x / scale) + zp, qmin, qmax) - zp) * scale

class FakeQuant(nn.Module):
    def forward(self, x):
        if self.per_channel:                                   # weight
            qmax = 2 ** (self.bits - 1) - 1
            amax = x.detach().abs().flatten(1).amax(1).view(-1, *[1] * (x.dim() - 1)).clamp(min=1e-8)
            return fake_quant(x, amax / qmax, 0.0, -qmax, qmax)
        if self.calib:                                         # 보정: 범위만 기록
            self.lo = torch.minimum(self.lo, x.detach().min())
            self.hi = torch.maximum(self.hi, x.detach().max()); return x
        lo, hi = min(float(self.lo), 0.0), max(float(self.hi), 1e-8)
        qmax = 2 ** self.bits - 1; s = (hi - lo) / qmax; zp = round(-lo / s)
        return fake_quant(x, s, zp, 0, qmax)                   # 활성값: 비대칭
```

**연결**
- [QP D1·3교시 Affine 양자화](#/qp/1/3) — quantize · dequantize 식 그대로, 오늘은 둘을 한 번에 이어 붙였다
- [QP D1·8교시 미니 프로젝트](#/qp/1/8) — `fake_quant_weight` 로 weight 를 양자화 → 되돌린 FP32 를 넣은 같은 생각
- [VB D1·3교시 autograd](#/vb/1/3) — 기울기가 계산 그래프를 거꾸로 흐르는 원리, round 에서 끊긴다
- [NN D3·4교시 활성화 계층](#/nn/3/4) — ReLU 의 backward 처럼 계층마다 "기울기를 어떻게 넘길지" 정한다

**상세 학습**

#### 이 교시에서 답하는 질문
1교시에서 "학습 중에 양자화를 겪게 하자" 고 했다. 그런데 학습은 FP32 기울기로 하는데 INT8 연산을 어떻게 끼워 넣을까? 이 교시는 **양자화 → 역양자화를 붙인 FP32 연산(fake quant)** 을 함수 · 모듈로 만들고, 그대로는 기울기가 0 이 되는 문제를 확인한다.

#### 개념 1 · fake quant 의 모양
![fake quant (scale 0.25, −4 ~ 3) — forward 는 계단, 범위 밖은 끝 칸 · round 의 기울기는 0, STE 는 범위 안 1](assets/fig/qp/d4-fakequant.svg)

| 기호 | 뜻 | 예 (3비트 대칭) |
|---|---|---|
| s | scale (칸 간격) | 0.25 |
| zp | zero-point (대칭이면 0) | 0 |
| qmin ~ qmax | 정수 범위 | −4 ~ 3 |
| x̂ | 되돌린 FP32 값 | −1.0 ~ 0.75 사이 8칸 |

x = linspace(−1.2, 1.2, 9) → x̂ = [−1.0, −1.0, −0.5, −0.25, 0, 0.25, 0.5, 0.75, 0.75] (직접 실행 · −0.6/0.25 = −2.4 → −2).

##### 🔎 왜 이렇게 되나
- **비유**: 리허설에서 실제 무대(INT8)의 조명 조건을 흉내 내며 연습하는 것
- **① fake 인 이유**: 양자화 → 역양자화를 바로 해서 값은 FP32 지만 칸 위에만 놓인다
- **② 학습 그래프에 넣는 이유**: 모델이 학습 중에 양자화 오차를 미리 겪고 그에 맞게 weight 를 고친다
- **③ 예**: 3비트 s 0.25 → x̂ 는 −1.0 ~ 0.75 사이 8칸
- ⚠ fake quant 모델은 FP32 로 계산한다 — 실제로 빨라지는 것은 배포(INT8 엔진) 때

<<CODE 2-c1 | fake quant 의 식>>
<<CODE 2-1 | fake_quant — 양자화 → 역양자화를 한 번에>>

#### 개념 2 · 보정 범위로 정하는 활성값 scale
- 보정 데이터로 lo = −0.5, hi = 2.0 → s = 2.5 / 255 = 0.0098, zp = round(0.5 / 0.0098) = 51.
- 입력 [−0.5, 0, 1.0, 3.0] → [−0.5, 0.0, 1.0, 2.0] — 0 은 정확히 0 (zp 가 정수), 범위 밖 3.0 은 2.0.
- 범위는 0 을 꼭 포함(min(lo, 0) · max(hi, 0)) — ReLU 출력의 0 을 오차 없이 표현하려고 (QP D1·4교시).

##### 🔎 왜 이렇게 되나
- **① 보정으로 정하는 이유**: 활성값 범위는 입력에 따라 다르다 — 데이터를 흘려 lo · hi 를 본다
- **② 0 을 포함하는 이유**: ReLU 출력의 0 을 오차 없이 — zp 가 정수라 0 은 정확히 0
- **③ 예**: lo −0.5, hi 2.0 → s = 2.5 / 255 = 0.0098, zp = 51
- ⚠ 범위 밖 값은 끝 칸에 붙는다 — 3.0 → 2.0

<<CODE 2-2 | FakeQuant 모듈 — weight 는 채널별, 활성값은 보정 범위>>
<<CODE 2-c2 | round 의 기울기는 0 — w 가 움직이지 않는다>>

#### 한 번에 정리
- fake quant = quantize → dequantize — 값은 FP32, 칸은 2^bits 개.
- weight: 채널별 대칭 · scale 매번 / 활성값: 텐서 하나 · 비대칭 · 보정 범위 고정.
- weight 는 FP32 로 학습 · 저장, 진짜 INT8 은 배포 때.
- ⚠ `torch.round` 의 기울기는 0 — 이대로는 학습이 안 된다 (3교시).

---

### 3교시 · STE (Straight-Through Estimator)

**키워드**: `STE`, `forward = round · backward = 1`, `torch.autograd.Function`, `@staticmethod forward · backward`, `RoundSTE.apply`, `clamp 밖 기울기 0`, `toy_train`, `loss 0.1875 → 0`

**개요**: 아이디어는 단순하다 — forward 는 round, backward 는 round 가 없는 것처럼 위에서 온 기울기를 **그대로 통과**(∂L/∂x = ∂L/∂y). PyTorch 에서는 `torch.autograd.Function` 을 상속해 forward · backward 를 따로 정의한다(`RoundSTE.apply(x)`). fake quant 에는 clamp 도 있으므로 **범위 밖으로 잘린 값은 기울기 0** — 범위 안의 값만 학습된다: x = [−2.0, −0.4, 0.3, 0.6, 2.0] (3비트 · 범위 −1.0 ~ 0.75) → 기울기 [0, 1, 1, 1, 0]. 작은 실험: w 3개를 target [0.5, −0.5, 0.5] 에 맞추도록 SGD 50번 — round 버전은 제자리(loss 0.1875), STE 버전은 w → [0.41, −0.62, 0.40] 로 움직여 양자화 결과가 정확히 목표 칸에 들어간다(loss 0).

**한눈에 보기**
STE — 계단을 곧게 통과하는 기울기
> forward 는 y = round(x) 그대로, backward 는 ∂L/∂x = ∂L/∂y — 반올림이 없었던 것처럼 기울기를 넘겨 양자화된 모델도 학습할 수 있게 한다
- **backward 는 항등**: 위 기울기 [1, 2, 3] → x.grad [1, 2, 3]
- **`torch.autograd.Function` 으로 구현**: `@staticmethod forward` · `backward` → `.apply(x)`
- **clamp 범위 밖은 기울기 0**: 3비트 s 0.25, x [−2.0, −0.4, 0.3, 0.6, 2.0] → [0, 1, 1, 1, 0] — 이미 잘린 값은 움직여도 그대로
- **toy 실험 — round 는 제자리, STE 는 loss 0**: w 가 [0.41, −0.62, 0.40] 으로 옮겨 x̂ = [0.5, −0.5, 0.5]
- **⚠ STE 는 근사다**: 그래서 lr 을 작게 · 짧게 학습하고, 범위는 보정으로 먼저 정한다

**핵심 용어**
- **STE(Straight-Through Estimator)**: 미분이 0 · 정의 안 되는 연산(round)을 backward 에서 항등 함수처럼 취급해 기울기를 그대로 넘기는 근사
- **torch.autograd.Function**: forward · backward 를 직접 정의하는 PyTorch 연산 — `.apply(x)` 로 호출
- **ctx(context)**: forward 에서 backward 로 값을 넘기는 상자 (STE 는 쓸 것이 없음)
- **clamp 기울기**: 범위 안 1 · 범위 밖 0 — 잘린 값은 더 밀어도 출력이 안 바뀌므로

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.autograd.Function` | `class RoundSTE(torch.autograd.Function): …` | forward · backward 를 직접 정의한 연산 |
| `RoundSTE.apply()` | `RoundSTE.apply(x / scale)` | forward 는 round, backward 는 기울기 그대로 (이 교시에서 만듦) |
| `toy_train()` | `loss, w = toy_train(fake_quant, steps=50)` | w 3개를 3비트 fake quant 로 target 에 맞추는 SGD 실험 (제공) |
| `torch.optim.SGD()` | `torch.optim.SGD([w], lr=0.1)` | 텐서 목록을 학습하는 경사 하강법 |

**핵심 내용**

#### STE
> forward: y = round(x) · backward: ∂L/∂x = ∂L/∂y — 계단을 "곧게 통과(straight-through)" 하는 것처럼 기울기를 넘긴다
- **forward** → 반올림 그대로 — 모델은 계속 양자화된 값을 본다
- **backward** → 기울기 1 (항등) — [0.2, 0.7, −1.6] · 위 기울기 [1, 2, 3] → x.grad [1, 2, 3]
- **구현** → `torch.autograd.Function` 의 `@staticmethod forward(ctx, x)` · `backward(ctx, g)` → `.apply(x)`
- ⚠ 근사다 — 진짜 미분(0)이 아니라 "round 가 없었다면" 의 기울기. 그래서 lr 을 작게 · 짧게 학습한다

#### clamp 와 함께 — 범위 밖은 학습 안 됨
> fake_quant 의 round 만 STE 로 바꾸면, clamp 범위 안은 기울기 1 · 범위 밖은 0
- **예** → 3비트 s 0.25 → 범위 −1.0 ~ 0.75 · x [−2.0, −0.4, 0.3, 0.6, 2.0] → 기울기 **[0, 1, 1, 1, 0]**
- **왜 0** → 이미 끝 칸에 잘린 값은 조금 움직여도 출력이 그대로
- **toy 실험** → round: loss **0.1875** · w 제자리 / STE: loss **0** · w [0.41, −0.62, 0.40] → x̂ [0.5, −0.5, 0.5]
- ⚠ 범위를 너무 좁게 잡으면 많은 값이 잘려 학습 신호가 줄어든다 — 그래서 보정으로 범위를 먼저 정한다

**기본 사용형태**
```python
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, g):
        return g                                              # 기울기 그대로 통과

def fake_quant(x, scale, zp, qmin, qmax):                     # STE 버전 (같은 이름으로 다시)
    return (torch.clamp(RoundSTE.apply(x / scale) + zp, qmin, qmax) - zp) * scale
```

**연결**
- [VB D1·3교시 autograd](#/vb/1/3) — `backward()` 가 각 연산의 backward 를 거꾸로 부르는 구조, 오늘은 그 하나를 직접 정의
- [NN D3·3교시 연쇄 법칙](#/nn/3/3) — ∂L/∂x = ∂L/∂y · ∂y/∂x 에서 ∂y/∂x 를 1 로 바꾸는 것이 STE
- [NN D3·4교시 활성화 계층](#/nn/3/4) — ReLU backward 의 "0 이하 기울기 0" 과 clamp 밖 기울기 0 이 같은 모양

**상세 학습**

#### 이 교시에서 답하는 질문
2교시의 fake quant 는 forward 는 맞지만 backward 에서 기울기가 0 이었다. 양자화된 출력의 loss 로 FP32 weight 를 학습하려면 기울기를 어떻게든 흘려야 한다. 이 교시는 STE 로 그 문제를 풀고, 작은 실험으로 "STE 면 weight 가 목표 칸으로 옮겨 간다" 를 확인한다.

#### 개념 1 · 세 가지 backward
![round 는 기울기 0, STE 는 1, clamp 범위 밖은 0 — toy 실험에서 round 는 제자리, STE 는 목표 칸 도착](assets/fig/qp/d4-ste.svg)

| 연산 | forward | backward | 결과 |
|---|---|---|---|
| `torch.round` | 계단 | 0 | 학습 안 됨 |
| STE round | 계단 | 1 (그대로) | 학습됨 |
| clamp 범위 밖 | 끝 칸 | 0 | 잘린 값은 학습 안 됨 |

##### 🔎 왜 이렇게 되나
- **비유**: 계단 위에서 공을 굴리면 평평한 디딤판에서 멈춘다 — STE 는 계단을 경사로로 "가정" 하고 굴린다
- **① round 의 기울기가 0 인 이유**: 계단 함수는 거의 모든 곳이 평평 — 조금 바꿔도 출력이 안 변한다
- **② STE 의 속임수**: forward 는 계단 그대로, backward 는 기울기를 1 로 그대로 통과
- **③ 범위 밖은 0**: clamp 로 잘린 값은 움직여도 출력이 안 바뀌므로 기울기 0
- ⚠ STE 는 근사 — 정확한 미분이 아니지만 실제로 잘 동작한다

<<CODE 3-c1 | STE 의 forward · backward>>
<<CODE 3-1 | RoundSTE — 반올림 · 기울기는 그대로>>
<<CODE 3-2 | STE 버전 fake_quant — 범위 밖만 기울기 0>>

#### 개념 2 · toy 실험 — w 3개를 목표 칸으로
- 설정: w0 = [0.31, −0.77, 0.20], 3비트 fake quant(s 0.25), target [0.5, −0.5, 0.5], SGD lr 0.1 · 50번.
- 처음 x̂ = [0.25, −0.75, 0.25] → 칸 하나씩 어긋남 → loss 3 × 0.25² = 0.1875.
- STE: 기울기 2(x̂ − target) 가 w 로 그대로 흘러 w 가 칸 경계를 넘는다 → w [0.41, −0.62, 0.40] → x̂ = [0.5, −0.5, 0.5] (0.41/0.25 = 1.64 → 2).
- 칸에 들어가면 x̂ = target 이라 기울기 0 → 거기서 멈춘다. QAT 는 이것을 수만 개 weight 에 대해 cross-entropy 로 하는 것.

##### 🔎 왜 이렇게 되나
- **① round 로는 안 움직이는 이유**: 기울기 0 — 50번 학습해도 w 가 그대로
- **② STE 로 움직이는 이유**: 기울기 2(x̂ − target) 가 w 로 흘러 칸 경계를 넘는다 — 0.41/0.25 = 1.64 → 2
- **③ 멈추는 이유**: 칸에 들어가면 x̂ = target 이라 기울기 0
- ⚠ 처음 loss = 3 × 0.25² = 0.1875 — 칸 하나씩 어긋난 상태

<<CODE 3-c2 | toy 실험 — w 3개를 목표 칸으로 학습>>

#### 한 번에 정리
- STE = forward 는 round, backward 는 기울기 1 (`torch.autograd.Function`).
- clamp 때문에 범위 밖 값은 기울기 0 — 잘린 값은 학습 안 됨.
- STE 덕분에 "양자화된 결과" 의 loss 로 FP32 weight 를 학습할 수 있다.
- ⚠ STE 는 근사 — 작은 lr · 짧은 fine-tuning 이 기본.

---

### 4교시 · 직접 만드는 QAT

**키워드**: `BN folding 먼저`, `fold() · fold_all()`, `QConv2d · QLinear`, `입력 FakeQuant + weight FakeQuant`, `quantize_model()`, `calibrate() = PTQ`, `train_qat()`, `lr 1e-3 · cosine · 3 epoch`, `W3A8 0.728 → 0.8375`

**개요**: 직접 만드는 QAT 의 순서 — (1) **BN folding 먼저**: 배포할 때는 BN 이 Conv 에 접힌 weight 가 양자화되므로, QAT 도 접힌 weight 로 학습해야 배포와 같은 숫자가 된다(BN 21개 → 0 · 정확도 0.846 그대로 · 출력 차이 4.8e-06). (2) Conv · Linear 를 **입력(활성값) FakeQuant + weight FakeQuant** 가 붙은 `QConv2d` · `QLinear` 로 바꿈(21 + 1개, ONNX 의 QDQ 위치와 같음). (3) 보정용 500장으로 활성값 범위를 잡음 → 이 상태가 PTQ: W3A8 **0.728** (W8A8 0.844). (4) 학습용 10,000장 · 3 epoch · lr 1e-3(처음 학습의 1/100) · cosine 감소 · SGD momentum 0.9 · weight decay 5e-4 로 fine-tuning → **QAT 0.8375**(epoch 1 · 2 · 3 = 0.8215 · 0.8335 · 0.8375), PTQ 대비 **+11.0%p** 회복 · FP32 와 0.85%p 차이. 대부분의 회복은 첫 epoch 에 일어난다.

**한눈에 보기**
직접 만드는 QAT — BN folding → 보정 → 짧은 학습
> BN 을 접은 뒤 Conv · Linear 22개에 FakeQuant 를 붙이고, 보정(= PTQ)으로 범위를 잡은 다음 작은 lr 로 3 epoch 만 학습(= QAT)한다
- **BN 을 먼저 접는다**: 21개 → 0, 출력 차이 4.8e-06 — fake quant 가 본 weight = 배포 때 양자화되는 weight
- **QConv2d = 입력 Q/DQ + weight Q/DQ**: ONNX 의 QDQ 위치와 같다
- **보정 결과(= PTQ) — W3A8 0.728**: W4A8 0.8345 · W8A8 0.844
- **설정 — lr 1e-3 · cosine · 3 epoch**: 처음 학습 lr 의 1/100, SGD momentum 0.9 · weight decay 5e-4
- **W3A8 PTQ 0.728 → QAT 0.8375 (+11.0%p)**: 첫 epoch 에 대부분 회복 (0.8215 · 0.8335 · 0.8375)
- **⚠ 보정 전 lo · hi 는 1e9 · −1e9 로**: 0 으로 두면 범위가 잘못 잡힌다 · QAT 결과는 ±0.5%p 쯤 흔들린다

**핵심 용어**
- **BN folding(접기)**: W' = W · k, b' = β − μ · k (k = γ/√(σ² + ε)) — BN 을 앞 Conv 의 weight · bias 에 흡수
- **QConv2d · QLinear**: forward 에서 입력과 weight 를 각각 FakeQuant 한 뒤 conv · linear 를 하는 층
- **W3A8**: weight 3비트 · 활성값(Activation) 8비트
- **cosine 감소(CosineAnnealingLR)**: learning rate 를 cos 곡선으로 처음 값 → 0 까지 줄이는 스케줄

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `fold()` | `fold(conv, bn)` | Conv + BN → bias 있는 Conv 하나 (제공) |
| `fold_all()` | `rn20_f = fold_all(rn20)` | stem · 블록 conv1/bn1 · conv2/bn2 · shortcut 을 모두 접은 복사본 (제공) |
| `QConv2d` | `QConv2d(cin, cout, k, stride, pad, bias=True).setup(wbits, abits)` | 입력 · weight FakeQuant 를 붙인 Conv (제공, `QLinear` 도 같은 모양) |
| `quantize_model()` | `quantize_model(rn20, wbits=3, abits=8)` | fold_all → Conv · Linear 를 QConv2d · QLinear 로 (재귀) |
| `calibrate()` | `calibrate(model)` | 활성값 FakeQuant 범위를 보정용 500장으로 |
| `train_qat()` | `train_qat(model, epochs=3, lr=1e-3, on_epoch=None)` | QAT fine-tuning → 걸린 초 |
| `CosineAnnealingLR()` | `torch.optim.lr_scheduler.CosineAnnealingLR(opt, 전체 step 수)` | lr 을 cos 곡선으로 0 까지 |
| `setattr()` | `setattr(parent, name, q)` | 부모 모듈의 자식을 새 층으로 바꿔 끼우기 |

**핵심 내용**

#### BN folding 을 먼저
> QAT 전에 접어야 "fake quant 가 본 weight = 배포 때 양자화되는 weight" — 접힌 W' = W · γ/√(σ² + ε)
- **식** → k = γ/√(σ² + ε), W' = W · k, b' = β − μ · k (원래 bias 가 있으면 + b · k)
- **확인** → BN **21개 → 0** · 정확도 0.846 그대로 · 출력 최대 차이 **4.8e-06**
- **왜** → BN 이 접히면 채널별 weight 범위가 k 배씩 바뀐다 — 안 접고 학습하면 배포 때 범위가 달라진다
- ⚠ ModelOpt 는 BN 을 남긴 채 QAT (5 · 6교시) — 그때는 배포 도구가 접은 뒤의 범위 차이를 감수

#### QConv2d → 보정 → fine-tuning
> Conv · Linear 22개를 FakeQuant 2개가 붙은 층으로 바꾸고, 보정(=PTQ) 다음 짧게 학습(=QAT)
- **QConv2d** → `F.conv2d(self.aq(x), self.wq(self.weight), self.bias, …)` — 입력 Q/DQ + weight Q/DQ (ONNX 의 QDQ 위치와 같음)
- **보정 = PTQ** → 활성값 FakeQuant 만 `calib=True` · lo 1e9 · hi −1e9 로 초기화 → `forward_loop` → W3A8 **0.728** · W4A8 0.8345 · W8A8 0.844
- **설정** → lr **1e-3** (처음 학습의 1/100) · cosine · **3 epoch** · SGD momentum 0.9 · weight decay 5e-4 (VB Day 2 와 같음)
- ⚠ 보정 전 lo · hi 를 1e9 · −1e9 로 — 0 으로 두면 범위가 0 을 넘지 못해 잘못 잡힌다

#### 직접 QAT 결과
> W3A8: PTQ 0.728 → QAT **0.8375** (+11.0%p) — FP32 0.846 과 0.85%p 차이
- **epoch 별** → 0.8215 · 0.8335 · 0.8375 — 첫 epoch 에 대부분 회복
- **학습 비용** → 10,000장 × 3 epoch = 237 step · 이 실행 CPU 555초 (다른 작업과 겹친 값 — 같은 조건 INT8 QAT 는 44초 · 교안: CPU 약 1.5분)
- **8교시에서** → 비용(학습 분)도 대결 표에 함께 적는다
- ⚠ QAT 뒤 정확도는 학습 순서 · 하드웨어에 따라 ±0.5%p 쯤 흔들린다 — 교안 예시 "0.73 → 0.83"

**기본 사용형태**
```python
def quantize_model(model, wbits=8, abits=8):
    m = fold_all(model)
    def swap(parent):
        for name, ch in parent.named_children():
            if isinstance(ch, nn.Conv2d):
                q = QConv2d(ch.in_channels, ch.out_channels, ch.kernel_size, ch.stride, ch.padding, bias=True).setup(wbits, abits)
            elif isinstance(ch, nn.Linear):
                q = QLinear(ch.in_features, ch.out_features).setup(wbits, abits)
            else:
                swap(ch); continue
            q.weight.data = ch.weight.data.clone(); q.bias.data = ch.bias.data.clone()
            setattr(parent, name, q)
    swap(m); return m.to(DEV)

def calibrate(model):
    fqs = [f for f in model.modules() if isinstance(f, FakeQuant) and not f.per_channel]
    for f in fqs: f.calib = True; f.lo.fill_(1e9); f.hi.fill_(-1e9)
    forward_loop(model)
    for f in fqs: f.calib = False
    return model
```

**연결**
- [QP D2·3교시 양자화 전처리](#/qp/2/3) — BN folding 식 (Day 2 는 ONNX export 가 해 주던 일을 오늘은 직접)
- [VB D2·5교시 BatchNorm](#/vb/2/5) — γ · β · running_mean · running_var 의 뜻
- [VB D2·7교시 학습 기법](#/vb/2/7) — 학습률 스케줄 · 증강(뒤집기 · 4픽셀 이동), 오늘 `batches()` 가 같은 증강
- [NN D4·6교시 규제](#/nn/4/6) — weight decay 5e-4

**상세 학습**

#### 이 교시에서 답하는 질문
2 · 3교시에서 fake quant 와 STE 라는 부품을 만들었다. 이제 ResNet-20 전체에 끼워 실제로 학습해야 한다. **어디에** 끼우고(Conv · Linear 의 입력과 weight), **무엇을 먼저** 하고(BN folding · 보정), **얼마나** 학습하나(lr 1e-3 · 3 epoch)? 이 교시는 W3A8 에서 PTQ 0.728 을 QAT 로 0.8375 까지 되살린다.

#### 개념 1 · 네 단계
![직접 만드는 QAT — BN folding → QConv2d → 보정(PTQ 0.728) → fine-tuning(QAT 0.8375)](assets/fig/qp/d4-qat.svg)

| 단계 | 하는 일 | 결과 |
|---|---|---|
| 1 BN folding | `fold_all` — BN 21개를 Conv 에 흡수 | 0.846 그대로 |
| 2 층 바꾸기 | Conv 21 → `QConv2d`, Linear 1 → `QLinear` | FakeQuant 44개 (입력 22 · weight 22) |
| 3 보정 | 활성값 22개 lo · hi ← 보정 500장 | W3A8 PTQ 0.728 |
| 4 fine-tuning | 학습 10,000장 · 3 epoch · lr 1e-3 cosine | W3A8 QAT 0.8375 |

##### 🔎 왜 이렇게 되나
- **① BN 을 먼저 접는 이유**: 배포 때 실행되는 숫자(W')로 fake quant 해야 학습한 scale 이 그대로 쓰인다
- **② 층을 바꾸는 이유**: QConv2d · QLinear 가 입력 · weight 에 FakeQuant 를 붙인다 — 44개
- **③ 보정 뒤 학습**: 보정으로 활성값 범위를 먼저 정하고(PTQ 0.728), 짧게 재학습(QAT 0.8375)
- ⚠ 순서를 바꾸면(보정 전 학습) 활성값 범위가 정해지지 않아 학습이 불안정하다

<<CODE 4-c1 | QConv2d · QLinear — FakeQuant 를 붙인 계층>>
<<CODE 4-c2 | fold_all — BN 을 Conv 에 접기>>
<<CODE 4-1 | quantize_model · calibrate — 계층 바꾸기와 보정>>

#### 개념 2 · 학습 설정의 이유
| 설정 | 값 | 이유 |
|---|---|---|
| learning rate | 1e-3 (처음 학습의 1/100) · cosine 감소 | 이미 학습된 weight 를 조금만 옮긴다 |
| epoch | 3 (79 step × 3 = 237 step) | 대부분의 회복은 첫 epoch 에 |
| optimizer | SGD momentum 0.9 · weight decay 5e-4 | VB Day 2 학습과 같은 설정 |
| 데이터 | 학습용 10,000장 (보정 · 평가와 겹치지 않음) | 평가 점수가 부풀지 않게 |

| W/A | PTQ | QAT (3 epoch) |
|---|---|---|
| W8A8 | 0.844 | (8교시) |
| W4A8 | 0.8345 | — |
| W3A8 | **0.728** | **0.8375** |

(직접 실행 · 평가 2,000장.)

##### 🔎 왜 이렇게 되나
- **비유**: 다 지은 집의 가구만 살짝 옮기는 것 — 처음부터 다시 짓지 않는다
- **① 학습률이 작은 이유**: 이미 학습된 weight 를 양자화 칸에 맞게 조금만 옮긴다 — 처음 학습의 1/100
- **② 3 epoch 인 이유**: 대부분의 회복은 첫 epoch 에 — 0.728 → 0.8215
- **③ 데이터를 분리하는 이유**: 보정 · 평가와 겹치지 않는 학습용 10,000장 — 평가 점수가 부풀지 않게
- ⚠ 학습률이 크면 이미 좋은 weight 가 망가져 오히려 나빠진다

<<CODE 4-2 | train_qat — SGD · cosine 로 짧게 재학습>>

#### 한 번에 정리
- 순서: BN folding → QConv2d · QLinear → 보정(= PTQ) → 짧은 fine-tuning(= QAT).
- W3A8: PTQ 0.728 → QAT 0.8375 (+11.0%p) — 3 epoch · lr 1e-3 로 충분.
- QAT 학습 루프는 평소 학습과 같다 — 모델 안의 FakeQuant 가 STE 로 기울기를 넘긴다.
- ⚠ 학습 비용(분)도 결과의 일부 — 8교시 대결 표에 함께 적는다.

---

### 5교시 · NVIDIA ModelOpt PTQ

**키워드**: `nvidia-modelopt 0.48`, `mtq.quantize()`, `INT8_DEFAULT_CFG`, `forward_loop`, `TensorQuantizer`, `input_quantizer · weight_quantizer`, `켜진 양자화기 45개`, `make_cfg()`, `quant_cfg 목록 형식`, `대칭 → TensorRT 바로 통과`

**개요**: ModelOpt(`nvidia-modelopt`)는 NVIDIA 의 모델 최적화 도구 — 4교시에 직접 만든 일을 한 줄로 한다: `mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=forward_loop)` = 양자화기 삽입 + 보정. Conv · Linear 마다 `TensorQuantizer` 가 붙는다(`input_quantizer` · `weight_quantizer` · `output_quantizer`). INT8 기본 설정은 weight 채널별 8비트 · 입력 텐서별 8비트 · **대칭** · max 보정. ResNet-20 에서 양자화기 110개가 들어가고 **켜진 것은 45개**(Conv 21 + Linear 1 의 입력 · weight 44 + pool 입력 1), INT8 PTQ 정확도 **0.843**. 비트를 바꾸려면 config 를 **깊은 복사**해 `*weight_quantizer` 의 `num_bits` 만 바꾼다(`make_cfg(3)`) → W3A8 PTQ **0.728** — 직접 만든 W3A8(0.728)과 같다(같은 원리). 다른 점: ModelOpt 는 **BN 을 접지 않고 그대로 둔다**(배포 때 TensorRT 가 처리).

**한눈에 보기**
NVIDIA ModelOpt PTQ — 한 줄로 양자화기 삽입 + 보정
> `mtq.quantize(model, config, forward_loop)` 가 Conv · Linear 에 양자화기를 넣고 보정까지 한다 — 4교시에서 직접 만든 것과 같은 원리, 대칭이라 TensorRT 조건도 그대로 만족
- **직접 만든 것 ↔ ModelOpt**: FakeQuant ↔ TensorQuantizer · quantize_model ↔ 자동 삽입 · calibrate ↔ forward_loop
- **켜진 양자화기 45개**: Conv 21 + Linear 1 의 입력 · weight 44 + pool 입력 1 — INT8 PTQ 0.843
- **config 를 깊은 복사해 num_bits 만 3 으로 → W3A8 0.728**: 직접 만든 결과와 같다
- **weight · 입력 모두 대칭 (zp 0)**: Day 3 의 TensorRT INT8 조건 통과 — 4교시 직접 만든 활성값은 비대칭이라 못 넣음
- **⚠ 깊은 복사 없이 바꾸면 `mtq.INT8_DEFAULT_CFG` 자체가 바뀐다**: BN 은 접지 않고 남는다 (배포 때 TensorRT 가 접음)

**핵심 용어**
- **ModelOpt(TensorRT Model Optimizer)**: NVIDIA 의 PyTorch 모델 양자화 · 가지치기 · 증류 도구 — `modelopt.torch.quantization` 을 `mtq` 로
- **TensorQuantizer**: ModelOpt 의 fake quant 모듈 — 직접 만든 `FakeQuant` 에 해당 (STE 포함)
- **forward_loop**: 보정 데이터를 모델에 흘려 주는 함수 — ModelOpt 가 이 안에서 범위(amax)를 잰다
- **quant_cfg**: "어떤 이름의 양자화기를 몇 비트 · 어느 축으로" 를 적은 설정 — 0.48 은 목록, 옛 버전은 사전

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `mtq.quantize()` | `mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=forward_loop)` | 양자화기 삽입 + 보정 → 양자화된 모델 |
| `mtq.INT8_DEFAULT_CFG` | `copy.deepcopy(mtq.INT8_DEFAULT_CFG)` | weight 채널별 8비트 · 입력 텐서별 8비트 · 대칭 설정 |
| `forward_loop()` | `forward_loop(model)` | 보정용 500장을 100장씩 흘리기 (제공) |
| `make_cfg()` | `make_cfg(wbits=3)` | weight 비트만 바꾼 config (이 교시에서 만듦) |
| `Module.named_modules()` | `[(n, m) for n, m in mo8.named_modules() if n.endswith("_quantizer")]` | 모든 하위 모듈을 (이름, 모듈) 로 |
| `TensorQuantizer.is_enabled` | `sum(m.is_enabled for n, m in qs)` | 켜진 양자화기 세기 |

**핵심 내용**

#### mtq.quantize — 한 줄로 PTQ
> mtq.quantize(model, config, forward_loop) = Conv · Linear 에 양자화기 삽입 + forward_loop 로 보정 — 4교시 quantize_model + calibrate
- **직접 만든 것 ↔ ModelOpt** → FakeQuant ↔ TensorQuantizer · quantize_model ↔ 자동 삽입 · calibrate ↔ forward_loop (max 보정)
- **양자화기 수** → 삽입 110개 · 켜진 것 **45개** = Conv 21 + Linear 1 의 입력 · weight (44) + pool 입력 (1) · output · BN 양자화기는 꺼짐
- **INT8 PTQ** → **0.843** (ORT PTQ 0.842 와 같은 수준)
- ⚠ BN 을 접지 않는다 (`QuantBatchNorm2d` 로 남음) — 접는 것은 배포 때 TensorRT 가 한다

#### config 로 비트 바꾸기
> config 를 깊은 복사 → `*weight_quantizer` 항목의 `num_bits` 만 3 으로 → W3A8
- **0.48 형식** → `cfg["quant_cfg"]` = 목록 `[{"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}}, …]`
- **옛 형식** → 사전 `qc["*weight_quantizer"]["num_bits"]` — 두 형식을 다 받게 쓴다
- **W3A8 PTQ** → ModelOpt **0.728** = 직접 만든 0.728 (같은 원리 · 같은 max 보정)
- ⚠ 깊은 복사 없이 바꾸면 `mtq.INT8_DEFAULT_CFG` 자체가 바뀌어 다음 양자화가 모두 3비트가 된다

#### 대칭 양자화 → TensorRT 조건 통과
> ModelOpt 는 weight · 입력 모두 대칭(zero_point 0) — Day 3 의 TensorRT INT8 조건(S8 · 대칭)을 그대로 만족
- **직접 만든 것과의 차이** → 4교시 활성값은 비대칭(UINT8 · zp ≠ 0) — TensorRT 에 바로 못 넣는다
- **7교시에서** → ModelOpt 모델을 `torch.onnx.export` → QDQ ONNX · `trt_check` 합격
- **기본 설정 표시** → `TensorQuantizer(8 bit fake per-tensor amax=2.13e+00 …)` · weight 는 `axis=0 amax=[…](16)`
- ⚠ GPU 가 없어도 `DEV = "cpu"` 로 ModelOpt fake quant · QAT 가 돈다 — TensorRT 엔진만 GPU 가 필요

**기본 사용형태**
```python
import modelopt.torch.quantization as mtq

mo8 = mtq.quantize(copy.deepcopy(rn20).to(DEV), mtq.INT8_DEFAULT_CFG, forward_loop=forward_loop)
qs = [(n, m) for n, m in mo8.named_modules() if n.endswith("_quantizer")]
n_on = sum(m.is_enabled for n, m in qs)                      # 45

def make_cfg(wbits=8):
    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)                 # 꼭 깊은 복사
    qc = cfg["quant_cfg"]
    if isinstance(qc, list):                                  # ModelOpt 0.48
        for e in qc:
            if e.get("quantizer_name") == "*weight_quantizer":
                e["cfg"]["num_bits"] = wbits
    else:                                                     # 옛 버전
        qc["*weight_quantizer"]["num_bits"] = wbits
    return cfg
mo3 = mtq.quantize(copy.deepcopy(rn20).to(DEV), make_cfg(3), forward_loop=forward_loop)
```

**연결**
- [QP D2·4교시 Static 양자화](#/qp/2/4) — ORT 의 `quantize_static` + `CalibrationDataReader` 와 같은 자리 (ModelOpt 는 PyTorch 모델에)
- [QP D3·4교시 TensorRT INT8 개념](#/qp/3/4) — TensorRT 는 S8 · 대칭만 받는다 → ModelOpt 의 기본이 대칭인 이유
- [QP D1·8교시 미니 프로젝트](#/qp/1/8) — `copy.deepcopy` 로 원본을 지키며 양자화한 같은 습관

**상세 학습**

#### 이 교시에서 답하는 질문
4교시의 QAT 는 원리를 보여 주지만, 실제 배포에서는 BN 처리 · ONNX 내보내기 · TensorRT 조건까지 맞춰야 한다. 이 일을 대신해 주는 NVIDIA 의 도구가 ModelOpt 다. 이 교시는 직접 만든 부품이 ModelOpt 의 무엇에 해당하는지 짝을 짓고, PTQ 결과가 같은지 확인한다.

#### 개념 1 · 직접 만든 것 ↔ ModelOpt
![mtq.quantize — Conv · Linear 에 input · weight 양자화기 삽입 + forward_loop 보정, 켜진 양자화기 45개](assets/fig/qp/d4-modelopt.svg)

| 직접 만든 것 (4교시) | ModelOpt |
|---|---|
| `FakeQuant` | `TensorQuantizer` (`input_quantizer` · `weight_quantizer`) |
| `quantize_model` (Conv 바꿔 끼우기) | `mtq.quantize` 가 Conv · Linear 를 `QuantConv2d` · `QuantLinear` 로 |
| `calibrate` | `forward_loop` 로 보정 (max calibration) |
| BN 먼저 접기 | BN 그대로 (배포 때 TensorRT 가 처리) |
| 활성값 비대칭 (zp ≠ 0) | 모두 대칭 (zp 0) |
| 범위 · 비트 = 생성자 인자 | config (`INT8_DEFAULT_CFG`) |

##### 🔎 왜 이렇게 되나
- **① 같은 구조인 이유**: ModelOpt 의 TensorQuantizer = 직접 만든 FakeQuant, mtq.quantize = 층 바꾸기 + 보정
- **② BN 을 접지 않는 이유**: 배포 때 TensorRT 가 빌드 중에 접는다
- **③ 쓰는 이유**: 배포에 필요한 대칭 QDQ ONNX · 체크포인트 · TensorRT 연결이 갖춰져 있다
- ⚠ 원리를 알아야 도구 설정이 틀렸을 때 알아챈다 — 4교시를 직접 만든 이유

<<CODE 5-c1 | ModelOpt 두 줄 사용법>>
<<CODE 5-1 | ModelOpt INT8 PTQ · 켜진 양자화기 수>>

#### 개념 2 · 양자화기 110개 중 45개
- `QuantConv2d` 21 · `QuantLinear` 1 · `QuantBatchNorm2d` 21 · pool 1 개마다 input · output 양자화기 (+ Conv · Linear 는 weight).
- 기본 설정은 `"*"` 를 먼저 끄고(`enable: False`) weight · input 양자화기만 켠다 → output 양자화기 · BN 양자화기는 꺼짐.
- 켜진 45개 = 입력 22 + weight 22 + `pool.input_quantizer` 1.

| 설정 | 직접 만든 PTQ | ModelOpt PTQ |
|---|---|---|
| W8A8 | 0.844 | 0.843 |
| W3A8 | 0.728 | 0.728 |

(직접 실행.) W3A8 이 같은 0.728 — weight 는 둘 다 채널별 대칭 max 이고, 활성값 8비트 오차는 작기 때문.

##### 🔎 왜 이렇게 되나
- **① 110개 중 45개인 이유**: 기본 설정은 먼저 모두 끄고 weight · input 양자화기만 켠다
- **② 45 = 입력 22 + weight 22 + pool 1**: output · BN 양자화기는 꺼져 있다
- **③ 요약을 보는 이유**: 어느 양자화기가 켜졌는지 확인해야 직접 만든 것과 비교할 수 있다
- ⚠ 설정 이름만 믿지 말고 켜진 양자화기 수를 직접 센다

<<CODE 5-c2 | 양자화기 요약 보기>>
<<CODE 5-2 | make_cfg — weight 비트만 바꾼 config (W3A8)>>

#### 한 번에 정리
- `mtq.quantize(model, config, forward_loop)` = 양자화기 삽입 + 보정 — 실습 4-1 과 같은 일.
- 켜진 양자화기 45개 (Conv · Linear 입력 · weight + pool 입력) · INT8 PTQ 0.843.
- config 로 비트 · 축을 바꾼다 — 깊은 복사 필수 · 버전별 형식(목록 / 사전) 주의.
- ModelOpt 는 대칭 양자화 → QDQ ONNX 가 TensorRT 조건을 바로 통과 (7교시).
- ⚠ ModelOpt 는 BN 을 접지 않는다 — 배포 도구(TensorRT)가 접는다.

---

### 6교시 · ModelOpt QAT: 정확도 곡선 · 체크포인트

**키워드**: `QAT = PTQ 모델을 평소처럼 학습`, `양자화기 안에 STE`, `train_qat 재사용`, `hist_df (epoch 0 = PTQ)`, `W3A8 0.728 → 0.832`, `INT8 0.843 → 0.843`, `mto.save · mto.restore`, `state_dict 만으로는 복원 불가`

**개요**: ModelOpt 의 QAT 는 특별한 함수가 없다 — `mtq.quantize` 로 PTQ 한 모델을 **그대로 평소처럼 학습**하면 된다(양자화기 `TensorQuantizer` 안에 STE 가 들어 있다). 그래서 4교시 `train_qat` 를 그대로 쓴다. W3A8(5-2 의 `mo3`)을 QAT 하며 epoch 마다 정확도를 기록하면 0.728 → 0.8185 → 0.8245 → **0.832**(+10.4%p) — 직접 만든 QAT(0.8375)와 비슷하다. INT8(`mo8`)은 0.843 → **0.843** — 이득 0, 8비트는 PTQ 로 충분. 체크포인트: 양자화기가 들어간 모델은 `torch.save(state_dict)` 만으로는 구조(어디에 어떤 양자화기)를 복원할 수 없다 → `mto.save(model, path)` / `mto.restore(새 FP32 모델, path)` 가 양자화 구조까지 저장 · 복원한다(복원 정확도 0.843 그대로).

**한눈에 보기**
ModelOpt QAT — PTQ 모델을 그대로 학습
> mtq.quantize 한 모델에 평소 학습 루프를 돌리면 QAT — TensorQuantizer 안의 STE 가 기울기를 넘긴다, 저장은 mto 로 양자화 구조까지 함께
- **W3A8 — 0.728 → 0.832 (+10.4%p)**: epoch 별 0.8185 · 0.8245 · 0.832 — 직접 만든 0.8375 와 비슷
- **INT8 은 이득 0 (0.843 → 0.843)**: 8비트는 PTQ 로 충분
- **mto.save / mto.restore**: FP32 구조 + 양자화기 배치 + amax 를 함께 — 복원 후 정확도 0.843 그대로
- **state_dict 만으로는 안 된다**: 새 FP32 모델에는 `weight_quantizer` 같은 하위 모듈이 없어 키가 맞지 않음
- **⚠ 학습은 짧게(3 epoch) · lr 1e-3**: 복원할 때도 같은 ModelOpt 버전 · 같은 FP32 구조

**핵심 용어**
- **정확도 곡선(learning curve)**: epoch 마다 잰 정확도를 이은 선 — epoch 0 은 QAT 전(= PTQ) 값
- **mto(modelopt.torch.opt)**: ModelOpt 의 저장 · 복원 도구 — 모델에 적용한 변환(양자화) 기록까지 함께 저장
- **체크포인트(checkpoint)**: 학습한 모델을 다시 불러 쓰기 위해 저장한 파일 (`rn20_mo_qat8.pth`)
- **QAT 이득**: QAT 정확도 − PTQ 정확도 (%p) — 학습 비용을 들일 가치가 있는지 판단하는 숫자

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `mto.save()` | `mto.save(mo8, "rn20_mo_qat8.pth")` | 양자화 구조 + weight 저장 |
| `mto.restore()` | `mto.restore(load_rn20().to(DEV), "rn20_mo_qat8.pth")` | 빈 FP32 모델에 양자화 구조를 다시 씌우고 weight 복원 |
| `load_rn20()` | `load_rn20()` | 학습된 ResNet-20 FP32 (eval 모드, 제공) |
| `on_epoch` | `train_qat(mo3, on_epoch=lambda ep, m: hist.append({"epoch": ep + 1, "acc": acc_torch(m)}))` | epoch 끝마다 부르는 함수 — 곡선 기록 |

**핵심 내용**

#### ModelOpt QAT — PTQ 모델을 그대로 학습
> mtq.quantize 한 모델 + 평소 학습 루프 = QAT — TensorQuantizer 안의 STE 가 기울기를 넘긴다
- **W3A8 곡선** → epoch 0 · 1 · 2 · 3 = **0.728** · 0.8185 · 0.8245 · **0.832** (+10.4%p)
- **INT8** → 0.843 → **0.843** (이득 0) — 8비트는 PTQ 로 충분
- **직접 만든 QAT 와** → W3A8 0.8375 vs 0.832 — 같은 원리라 비슷 (BN 을 접었느냐 정도의 차이)
- ⚠ 학습은 짧게(3 epoch) · lr 1e-3 — 길게 하면 시간만 들고 과적합 위험

#### 체크포인트 — mto.save / mto.restore
> 양자화된 모델 = FP32 구조 + 양자화기 배치 + 범위(amax) — `mto` 는 이 셋을 함께 저장한다
- **저장** → `mto.save(mo8, "rn20_mo_qat8.pth")` → 1,174 KB
- **복원** → `mto.restore(load_rn20().to(DEV), 경로)` — 빈 FP32 모델에 양자화 구조까지 다시 씌운다 → 정확도 **0.843** 그대로
- **왜 state_dict 만으로는 안 되나** → 새 FP32 모델에는 `weight_quantizer` 같은 하위 모듈이 없어 키가 맞지 않는다
- ⚠ 복원할 때도 같은 ModelOpt 버전 · 같은 FP32 구조(`ResNet()`)가 필요

**기본 사용형태**
```python
hist = [{"epoch": 0, "acc": acc_mo_ptq3}]
t_mo_qat3 = train_qat(mo3, on_epoch=lambda ep, m: hist.append({"epoch": ep + 1, "acc": acc_torch(m)}))
hist_df = pd.DataFrame(hist)                                  # epoch 0 ~ 3

import modelopt.torch.opt as mto
t_mo_qat8 = train_qat(mo8)
mto.save(mo8, "rn20_mo_qat8.pth")
restored = mto.restore(load_rn20().to(DEV), "rn20_mo_qat8.pth")
acc_torch(restored)                                           # 0.843 = acc_torch(mo8)
```

**연결**
- [QP D4·4교시 직접 만드는 QAT](#/qp/4/4) — 같은 `train_qat` · 같은 설정, 모델만 ModelOpt 로
- [VB D1·5교시 학습 루프 · 평가 · 저장](#/vb/1/5) — `torch.save(state_dict)` · `load_state_dict`, 양자화 모델은 `mto` 로
- [NN D4·5교시 과적합](#/nn/4/5) — epoch 마다 정확도를 그려 보는 같은 습관

**상세 학습**

#### 이 교시에서 답하는 질문
5교시에 ModelOpt 로 PTQ 를 했다. QAT 는 어떻게 할까 — 별도의 함수가 있을까? 그리고 학습한 양자화 모델은 어떻게 저장했다가 다시 불러올까? 이 교시는 "PTQ 모델을 그대로 학습하면 QAT" 임을 곡선으로 확인하고, 양자화 구조까지 저장하는 `mto.save / restore` 를 쓴다.

#### 개념 1 · 정확도 곡선
![W3A8 QAT 정확도 곡선 — 직접 만든 것 0.8375 · ModelOpt 0.832, 둘 다 첫 epoch 에 대부분 회복 · INT8 은 이득 0](assets/fig/qp/d4-curve.svg)

| epoch | 직접 만든 QAT | ModelOpt QAT |
|---|---|---|
| 0 (PTQ) | 0.728 | 0.728 |
| 1 | 0.8215 | 0.8185 |
| 2 | 0.8335 | 0.8245 |
| 3 | **0.8375** | **0.832** |

| 모델 | PTQ | QAT | 이득 |
|---|---|---|---|
| ModelOpt W3A8 | 0.728 | 0.832 | +10.4%p |
| ModelOpt INT8 | 0.843 | 0.843 | 0 |

(직접 실행 · CPU · 학습 시간 W3A8 75초 · INT8 42초.)

##### 🔎 왜 이렇게 되나
- **① 저비트에서 이득이 큰 이유**: PTQ 손실이 큰 W3A8 은 학습으로 되찾을 것이 많다 — +10.4%p
- **② INT8 에서 이득이 0 인 이유**: PTQ 가 이미 FP32 와 거의 같다 — 되찾을 손실이 없다
- **③ 직접 만든 것과 비슷한 이유**: 같은 원리(fake quant + STE + 짧은 재학습) — 0.8375 vs 0.832
- ⚠ QAT 는 학습 시간이 든다 — PTQ 로 충분하면 쓰지 않는다

<<CODE 6-1 | ModelOpt QAT W3A8 정확도 곡선>>

#### 개념 2 · 양자화 모델 저장
| 방법 | 저장되는 것 | 복원 |
|---|---|---|
| `torch.save(model.state_dict())` | weight · 버퍼 값만 | 같은 구조의 모델이 먼저 있어야 함 — 양자화기 없는 FP32 모델엔 안 맞음 |
| `mto.save(model, path)` | 값 + ModelOpt 변환 기록(양자화 config · 위치) | `mto.restore(빈 FP32 모델, path)` 가 구조까지 다시 만든다 |

##### 🔎 왜 이렇게 되나
- **① state_dict 로 부족한 이유**: 값만 저장 — 양자화기가 어디 붙었는지 구조 정보가 없다
- **② mto.save**: 값 + 양자화 config · 위치 기록
- **③ 복원**: mto.restore(빈 FP32 모델, path) 가 구조까지 다시 만든다
- ⚠ 양자화기 없는 FP32 모델에 state_dict 를 그대로 load 하면 키가 안 맞는다

<<CODE 6-2 | INT8 QAT · 체크포인트 저장 · 복원>>

#### 한 번에 정리
- ModelOpt QAT = PTQ 모델을 평소처럼 학습 (양자화기 안에 STE).
- W3A8 은 QAT 로 크게 회복(0.728 → 0.832), INT8 은 PTQ 와 같다(0.843).
- 체크포인트는 `mto.save` / `mto.restore` — 양자화 구조까지 저장.
- ⚠ 복원은 빈 FP32 모델(`load_rn20()`)에 — 이미 양자화한 모델에 또 씌우지 않는다.

---

### 7교시 · QAT 모델 배포: QDQ ONNX · TensorRT INT8

**키워드**: `torch.onnx.export → QuantizeLinear · DequantizeLinear`, `explicit 경로`, `Q/DQ 45쌍`, `trt_check ok`, `보정 없이 INT8 엔진`, `trtexec --int8 --fp16`, `BN 은 ONNX 에 남음`, `QDQ ONNX ≈ FP32 크기`

**개요**: ModelOpt 로 양자화한 모델을 `torch.onnx.export`(여기서는 `export_b1`, 배치 1 고정) 하면 양자화기가 **QuantizeLinear · DequantizeLinear** 로 바뀐 **QDQ ONNX** 가 나온다 — Day 3 의 explicit 경로. `rn20_qat.onnx`: Q 45 · DQ 45 (켜진 양자화기 45개와 같음), `trt_check` → qdq 90 · uint8 0 · nonzero_zp 0 → **ok True**(ModelOpt 는 대칭), ORT 정확도 **0.843** = PyTorch. scale 이 이미 들어 있으므로 TensorRT 빌드에 **보정이 필요 없다** — `build_once(…, fp16=True, int8=True)` 는 `trtexec --onnx=rn20_qat.onnx --int8 --fp16` 과 같다. Netron 으로 보면 Conv 마다 입력 쪽 Q→DQ · weight 쪽 Q→DQ(채널별 scale)가 붙고, BN(21개)은 Conv 뒤에 남아 있어 TensorRT 가 빌드할 때 접는다. QDQ ONNX 의 weight 는 아직 FP32 라 파일이 FP32 와 비슷(1,135 KB vs 1,074 KB) — 진짜로 작아지는 것은 **엔진**이다.

**한눈에 보기**
QAT 모델 배포 — QDQ ONNX → TensorRT INT8
> ModelOpt 의 TensorQuantizer 는 export 때 Q · DQ 노드 쌍이 되고 scale 이 상수로 들어가, 보정 없이 TensorRT INT8 플래그만 켜면 엔진이 된다
- **Q 45 · DQ 45 = 켜진 양자화기 45개**: Conv 21 · BatchNormalization 21 · Gemm 1
- **trt_check 합격**: uint8 0 · nonzero_zp 0 — 대칭이라 따로 맞출 필요 없음, ORT 정확도 0.843
- **TensorRT 는 `int8=True, fp16=True` 만**: calibrator 없음 — Day 3 와 같은 `build_once`
- **GPU 가 없으면 ORT CPU 로**: 0.843 · 1.60 ms · 1,135 KB (weight 는 FP32 + Q/DQ, 엔진에서 INT8 로 작아짐)
- **⚠ ORT CPU 에서는 FP32(0.71 ms)보다 느리다**: BN 미접기 · S8 활성값 · Q/DQ 때문 — 속도는 TensorRT GPU 에서 · scale 은 Constant 노드

**핵심 용어**
- **QDQ ONNX**: QuantizeLinear → DequantizeLinear 쌍으로 양자화 위치 · scale 을 표시한 ONNX (QP D2·5교시)
- **explicit 양자화**: scale 이 들어 있는 QDQ ONNX 를 TensorRT 에 넣는 방식 — Q/DQ 있는 곳만 INT8 (QP D3·4교시)
- **보정 없는 빌드**: QAT · PTQ 로 정한 scale 을 그대로 쓰므로 TensorRT 가 다시 범위를 재지 않음
- **Netron**: ONNX 그래프를 그림으로 보는 도구 — Q/DQ 위치 확인

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `export_b1()` | `export_b1(copy.deepcopy(mo8), "rn20_qat.onnx")` | ModelOpt 모델 → QDQ ONNX (CPU 로 옮겨 배치 1, 제공) |
| `Counter()` | `Counter(n.op_type for n in onnx.load(p).graph.node)` | 노드 종류별 개수 (Q · DQ 세기) |
| `trt_check()` | `trt_check("rn20_qat.onnx")` | TensorRT INT8 조건 점검 — Constant 노드 scale 까지 보도록 확장 (제공) |
| `build_once()` | `build_once("rn20_qat.onnx", "rn20_qat.engine", fp16=True, int8=True)` | 엔진이 없을 때만 빌드 (OT D5, 제공) |
| `TRTInfer()` | `TRTInfer("rn20_qat.engine")(X_ev[:1])` | 엔진 실행기 (OT D5, 제공 · GPU 일 때) |
| `bench_fn()` | `bench_fn(lambda: infer(X_ev[:1]), n=300, warm=30)["mean"]` | 배치 1 평균 지연 ms (제공) |

**핵심 내용**

#### QAT 모델 → QDQ ONNX
> ModelOpt 의 TensorQuantizer 는 export 때 QuantizeLinear · DequantizeLinear 쌍이 된다 — scale 은 상수로, zero_point 는 0
- **노드 수** → Q **45** · DQ **45** = 켜진 양자화기 45개 · Conv 21 · BatchNormalization 21 · Gemm 1 · scale/zp 상수(Constant) 90
- **trt_check** → qdq 90 · uint8 0 · nonzero_zp 0 → **ok True** (대칭이라 Day 3 처럼 따로 맞출 필요 없음)
- **정확도** → ORT **0.843** = PyTorch `acc_mo_qat8` 0.843
- ⚠ scale 이 initializer 가 아니라 **Constant 노드**로 들어간다 — 그래서 오늘 `trt_check` 는 Constant 까지 본다

#### 보정 없이 TensorRT INT8
> scale 이 이미 ONNX 안에 → BuilderFlag.INT8 + FP16 만 켜면 된다 (calibrator 없음) — Day 3 와 같은 함수
- **GPU** → `TRTInfer(build_once("rn20_qat.onnx", "rn20_qat.engine", fp16=True, int8=True))` (명령으로는 `trtexec --onnx=rn20_qat.onnx --int8 --fp16`)
- **GPU 없을 때** → 같은 ONNX 를 ORT CPU 로: 0.843 · **1.60 ms** · 1,135 KB (이 실행)
- **크기** → QDQ ONNX 의 weight 는 FP32 + Q/DQ → FP32(1,074 KB) 정도 · 엔진에서 INT8 로 작아진다
- ⚠ ORT CPU 에서는 이 모델이 FP32(0.71 ms)보다 느리다 — BN 이 안 접혀 있고 S8 활성값 · Q/DQ 변환 때문. 속도는 TensorRT GPU 에서 본다

**기본 사용형태**
```python
export_b1(copy.deepcopy(mo8), "rn20_qat.onnx")
cnt = Counter(n.op_type for n in onnx.load("rn20_qat.onnx").graph.node)
n_q, n_dq = cnt["QuantizeLinear"], cnt["DequantizeLinear"]       # 45, 45
trt_check("rn20_qat.onnx")                                       # ok True
if TRT_OK:
    infer = TRTInfer(build_once("rn20_qat.onnx", "rn20_qat.engine", fp16=True, int8=True))
else:
    s = make_session("rn20_qat.onnx"); infer = lambda x, s=s: s.run(None, {"input": x})[0]
mean_ms = bench_fn(lambda: infer(X_ev[:1]), n=300, warm=30)["mean"]
```

**연결**
- [QP D3·4교시 TensorRT INT8 개념](#/qp/3/4) — implicit vs explicit, 오늘은 QAT 가 만든 QDQ ONNX 로 explicit
- [QP D3·5교시 INT8 엔진 빌드](#/qp/3/5) — `build_engine` 에 `BuilderFlag.INT8` 한 줄, 같은 함수
- [QP D2·5교시 QDQ vs QOperator](#/qp/2/5) — Q → DQ 쌍의 그래프 모양
- [OT D5·1교시 TRTInfer](#/ot/5/1) — 엔진 실행 클래스

**상세 학습**

#### 이 교시에서 답하는 질문
6교시에서 QAT 한 모델은 아직 PyTorch 안의 fake quant 다. 실제 장치에서 INT8 로 돌리려면 TensorRT 엔진이 필요하다. QAT 로 정한 scale 을 어떻게 TensorRT 에 전달할까? 이 교시는 QDQ ONNX 로 내보내 scale 을 그대로 실어 나르고, 보정 없이 INT8 엔진을 만드는 길을 확인한다.

#### 개념 1 · QDQ ONNX 의 모양
![ModelOpt QAT 모델 → QDQ ONNX(Q/DQ 45쌍 · trt_check ok) → 보정 없는 TensorRT INT8 엔진](assets/fig/qp/d4-deploy.svg)

| 확인할 것 | rn20_qat.onnx (직접 실행) |
|---|---|
| Conv 입력 쪽 | Q → DQ (텐서 하나 scale) |
| Conv weight 쪽 | Q → DQ (채널별 scale, axis 0) |
| BN | Conv 뒤에 그대로 (21개) — TensorRT 가 빌드 때 접는다 |
| zero_point | 모두 0 (int8) → `trt_check` ok |
| 파일 크기 | 1,135 KB (FP32 1,074 KB) |

##### 🔎 왜 이렇게 되나
- **① 입력은 per-tensor, weight 는 per-channel 인 이유**: 활성값은 관례적으로 scale 하나, weight 는 채널마다 (axis 0)
- **② BN 이 남아 있는 이유**: ModelOpt 는 BN 을 접지 않았다 — TensorRT 가 빌드 때 접는다
- **③ zero_point 가 모두 0 인 이유**: 대칭 int8 — TensorRT 가 받는 형식, trt_check 통과
- ⚠ 파일 크기가 FP32 보다 크다 (1,135 KB) — weight 가 FP32 + Q/DQ 정보로 저장된다, INT8 은 엔진에서

<<CODE 7-1 | QAT 모델을 QDQ ONNX 로 · Q/DQ 수 · 정확도>>

#### 개념 2 · Day 3 PTQ 경로와 비교
| | Day 3 PTQ (`make_trt_int8`) | Day 4 QAT (ModelOpt) |
|---|---|---|
| scale 정하는 곳 | ORT `quantize_static` 보정 | ModelOpt 보정 + QAT 학습 |
| 입력 | FP32 ONNX | PyTorch 모델 |
| QDQ ONNX | 보정 결과로 생성 | `torch.onnx.export` 가 생성 |
| TensorRT 빌드 | INT8 + FP16 · 보정 없음 | INT8 + FP16 · 보정 없음 (같음) |
| BN | ONNX export 때 접힘 | ONNX 에 남음 → TensorRT 가 접음 |

엔진 크기 · 지연은 GPU(TensorRT)에서만 잴 수 있다 — 이 실행은 GPU 가 없어 ORT CPU 값(0.843 · 1.60 ms · 1,135 KB).

##### 🔎 왜 이렇게 되나
- **① 같은 점**: TensorRT 빌드는 둘 다 INT8 + FP16, 보정 없음 — scale 이 이미 QDQ 에 있다
- **② 다른 점**: scale 을 정하는 곳 — PTQ 는 ORT 보정, QAT 는 ModelOpt 보정 + 학습
- **③ 그래서 경로가 하나로 모인다**: 어떤 방법이든 QDQ ONNX 를 만들면 같은 TensorRT 빌드
- ⚠ QAT 엔진도 배포 장치에서 다시 빌드 · 측정한다

<<CODE 7-2 | 보정 없이 INT8 엔진 — 정확도 · 지연 · 크기>>

#### 한 번에 정리
- ModelOpt 모델 → `torch.onnx.export` → QDQ ONNX (scale 포함 · zero_point 0).
- TensorRT 빌드는 보정 없이 INT8 + FP16 — Day 3 와 같은 함수.
- QDQ ONNX 는 FP32 크기, 작아지는 것은 엔진.
- ⚠ Netron 으로 Q/DQ 위치를 확인하고, 속도는 배포 장치(TensorRT GPU)에서 잰다.

---

### 8교시 · 미니 프로젝트: PTQ vs QAT 대결

**키워드**: `5가지 방법`, `duel_df`, `acc_int8 · acc_w3a8 · train_min · latency_ms`, `deploy_lat()`, `gain = max(QAT) − max(PTQ)`, `need_qat (> 1%p)`, `ptq_vs_qat.csv · qp_baseline.csv`, `qat_curve.png`

**개요**: 같은 ResNet-20 을 **5가지 방법**(FP32 · ORT PTQ · ModelOpt PTQ · 수작업 QAT · ModelOpt QAT)으로 양자화해 한 표(`duel_df`)에 놓고, "QAT 가 필요한 경우" 를 숫자로 결론 낸다. 이 실행 결과: INT8 — ORT PTQ 0.842 · ModelOpt PTQ 0.843 · 수작업 QAT **0.849** · ModelOpt QAT 0.843 → QAT 이득 **+0.6%p** → **필요 없음**(기준 1%p). W3A8 — ModelOpt PTQ 0.728 · 수작업 QAT 0.8375 · ModelOpt QAT 0.832 → 이득 **+10.95%p** → **필요**. QAT 의 비용(학습 분 · GPU)과 이득(%p)을 함께 적어야 결론이 선다. 지연(배치 1)은 GPU 면 TensorRT 엔진, 아니면 ORT CPU — 이 실행(ORT CPU)에서는 INT8 모델이 FP32(0.71 ms)보다 느렸다(1.18 ~ 1.68 ms). 배포는 ModelOpt → QDQ ONNX → TensorRT, 저비트(W3A8)는 실행 커널이 있는지까지 확인한다.

**한눈에 보기**
PTQ vs QAT 대결 — QAT 가 필요한지 숫자로 판정
> 5가지 방법을 INT8 · W3A8 정확도 · 학습 분 · 지연으로 같은 표에 두고, gain = max(QAT) − max(PTQ) 가 1%p 를 넘을 때만 QAT 를 고른다
- **INT8 — gain +0.6%p → QAT 불필요**: 수작업 QAT 0.849 − ModelOpt PTQ 0.843 (2,000장에서 12장, 잡음 수준)
- **W3A8 — gain +10.95%p → QAT 필요**: 수작업 QAT 0.8375 − PTQ 0.728
- **학습 비용도 함께**: 수작업 9.98분 · ModelOpt 1.96분 (CPU), PTQ 는 0
- **배포라면 ModelOpt QAT**: 대칭 QDQ ONNX → TensorRT 로 바로 이어진다 (수작업은 비대칭 · 내보내기 없음)
- **qp_baseline.csv 에 Day4 5행**: `drop_duplicates(subset=["day", "model"], keep="last")`
- **⚠ W3A8 을 실제로 빠르게 돌리려면 3비트 커널 · 장치 지원이 필요**: 없으면 메모리만 줄고 계산은 8비트

**핵심 용어**
- **대결 표(duel table)**: 방법마다 같은 자(정확도 2종 · 학습 분 · 지연)로 잰 한 표
- **QAT 이득(gain)**: INT8 = max(두 QAT) − max(두 PTQ) · W3A8 = max(두 QAT) − ModelOpt PTQ
- **판정 기준**: 이득 > 0.01 (1%p) 이면 "QAT 필요"
- **실행 커널**: 장치가 그 비트(예: 3비트)로 실제 계산하는 코드 — 없으면 저비트 모델도 빠르지 않다

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `deploy_lat()` | `deploy_lat("rn20_qat.onnx")` | GPU 면 TensorRT(QDQ 는 INT8+FP16), 아니면 ORT CPU — 배치 1 평균 ms (제공) |
| `duel_df` | `pd.DataFrame([{"method", "acc_int8", "acc_w3a8", "train_min", "latency_ms"}, …])` | 5행 대결 표 |
| `DataFrame.set_index()` | `d_ = duel_df.set_index("method")` | 방법 이름으로 행 꺼내기 |
| `plt.axhline()` | `plt.axhline(acc_mo_ptq3, ls="--")` | 기준선(PTQ · FP32) 그리기 |
| `plt.savefig()` | `plt.savefig("qat_curve.png", dpi=120)` | 그림 파일로 저장 |

**핵심 내용**

#### PTQ vs QAT 대결 표
> 5가지 방법 × (INT8 정확도 · W3A8 정확도 · 학습 분 · 지연) — 같은 모델 · 같은 평가 2,000장 · 같은 자
- **INT8** → FP32 0.846 · ORT PTQ 0.842 · ModelOpt PTQ 0.843 · 수작업 QAT **0.849** · ModelOpt QAT 0.843
- **W3A8** → ModelOpt PTQ **0.728** · 수작업 QAT **0.8375** · ModelOpt QAT 0.832 (ORT PTQ · FP32 는 NaN)
- **학습 분 (INT8 + W3A8)** → 수작업 9.98 · ModelOpt 1.96 (CPU · 이 실행) · PTQ 는 0
- ⚠ 수작업 W3A8 학습 555초는 다른 작업과 겹쳐 느려진 값 — 같은 조건의 다른 QAT 는 42 ~ 75초. 학습 분은 장치 · 부하에 따라 크게 다르다

#### QAT 가 필요한가 — 숫자로 판정
> gain = max(QAT) − max(PTQ) · gain > 1%p 면 QAT 필요 — INT8 +0.6%p 불필요 · W3A8 +10.95%p 필요
- **INT8** → 0.849 − 0.843 = **+0.6%p** → need_qat False (8비트는 PTQ 로 충분)
- **W3A8** → 0.8375 − 0.728 = **+10.95%p** → need_qat True
- **배포라면** → ModelOpt QAT — 정확도는 비슷하지만 대칭 QDQ ONNX → TensorRT 로 바로 이어진다 (수작업은 활성값 비대칭 · 내보내기 없음)
- ⚠ 0.6%p 는 평가 2,000장에서 12장 차이 — 잡음 수준일 수 있다 (수작업 QAT 0.849 가 FP32 0.846 보다 높은 것도 그 정도)

#### 저장 · 곡선 · 결론
> ptq_vs_qat.csv (대결 표) · qp_baseline.csv 에 Day4 5행 추가 · qat_curve.png (W3A8 곡선 + PTQ · FP32 기준선)
- **qp_baseline.csv** → day="Day4" · model="rn20_<method>" · quant="INT8 (<method>)" · acc = acc_int8 · mean_ms = latency_ms · 나머지 NaN
- **중복** → `pd.concat` 뒤 `drop_duplicates(subset=["day", "model"], keep="last")`
- **outlier 모델에 QAT** → 강사 실험 ModelOpt QAT 0.25 → 0.50 — 범위 문제는 mixed precision · 채널 균등화가 낫다 (Day 3)
- ⚠ W3A8 을 실제로 빠르게 돌리려면 3비트 실행 커널 · 장치 지원이 필요 — 없으면 메모리만 줄고 계산은 8비트

**기본 사용형태**
```python
b = base_df.set_index("model").acc
duel_df = pd.DataFrame([
    {"method": "FP32", "acc_int8": b["FP32"], "acc_w3a8": np.nan, "train_min": 0, "latency_ms": deploy_lat("rn20_b1_pre.onnx")},
    {"method": "ORT PTQ", "acc_int8": b["ORT PTQ INT8"], "acc_w3a8": np.nan, "train_min": 0, "latency_ms": deploy_lat("rn20_ptq8.onnx")},
    {"method": "ModelOpt PTQ", "acc_int8": acc_mo_ptq8, "acc_w3a8": acc_mo_ptq3, "train_min": 0, "latency_ms": deploy_lat("rn20_mo_ptq8.onnx")},
    {"method": "수작업 QAT", "acc_int8": acc_qat8, "acc_w3a8": acc_qat3, "train_min": (t_qat8 + t_qat3) / 60, "latency_ms": np.nan},
    {"method": "ModelOpt QAT", "acc_int8": acc_mo_qat8, "acc_w3a8": acc_mo_qat3, "train_min": (t_mo_qat8 + t_mo_qat3) / 60, "latency_ms": deploy_lat("rn20_qat.onnx")}])
d_ = duel_df.set_index("method")
gain = {"INT8": max(d_.acc_int8["수작업 QAT"], d_.acc_int8["ModelOpt QAT"]) - max(d_.acc_int8["ORT PTQ"], d_.acc_int8["ModelOpt PTQ"]),
        "W3A8": max(d_.acc_w3a8["수작업 QAT"], d_.acc_w3a8["ModelOpt QAT"]) - d_.acc_w3a8["ModelOpt PTQ"]}
need_qat = {k: bool(v > 0.01) for k, v in gain.items()}
```

**연결**
- [QP D3·8교시 미니 프로젝트](#/qp/3/8) — 같은 "같은 자로 재고 → 먼저 정한 규칙으로 판정" 형식 (어제는 엔진 4개)
- [QP D2·7교시 성능 측정](#/qp/2/7) — `qp_baseline.csv` 를 처음 만든 곳, 오늘 Day4 5행을 더한다
- [QP D3·3교시 Mixed Precision](#/qp/3/3) — outlier 모델의 처방 (QAT 보다 낫다)
- [VB D3·4교시 MobileNet V1 · V2 · V3](#/vb/3/4) — depthwise Conv 모델 — 저비트에서 QAT 효과가 큰 경우

**상세 학습**

#### 이 교시에서 답하는 질문
오늘 PTQ 2가지(ORT · ModelOpt)와 QAT 2가지(수작업 · ModelOpt)를 만들었다. 결국 "QAT 를 해야 하나?" 에 답해야 한다. 이 교시는 다섯 방법을 같은 자로 재서 한 표에 놓고, QAT 이득을 계산해 **INT8 은 불필요 · W3A8 은 필요** 를 숫자로 결론 낸다.

#### 개념 1 · 과제 구성 (100점)
| 과제 | 내용 | 배점 |
|---|---|---|
| 1 | 수작업 QAT INT8 학습 (`acc_qat8`) + ModelOpt PTQ INT8 → `rn20_mo_ptq8.onnx` | 20 |
| 2 | 대결 표 `duel_df` (정확도 2종 · 학습 분 · 지연) | 20 |
| 3 | QAT 이득 `gain` · "QAT 필요?" `need_qat` | 20 |
| 4 | `ptq_vs_qat.csv` · `qp_baseline.csv` 저장 | 20 |
| 5 | 정확도 곡선 `qat_curve.png` | 10 |
| 6 | 결론 쓰기 (마크다운, 강사 채점) | 10 |

정답 보기 셀 없음 · 채점 셀이 자동 점수와 정답 열람 기록을 보여 준다.

<<CODE 과제 1 | 과제 1 — 수작업 QAT INT8 · ModelOpt PTQ INT8 ONNX>>

#### 개념 2 · 대결 표 (이 실행)
![PTQ vs QAT 대결 — INT8 은 다섯 방법이 0.84 대로 비슷, W3A8 은 QAT 가 PTQ 보다 +10%p · CPU 에서는 INT8 이 느림](assets/fig/qp/d4-duel.svg)

| method | acc_int8 | acc_w3a8 | train_min | latency_ms (ORT CPU) |
|---|---|---|---|---|
| FP32 | 0.846 | — | 0 | 0.71 |
| ORT PTQ | 0.842 | — | 0 | 1.18 |
| ModelOpt PTQ | 0.843 | 0.728 | 0 | 1.68 |
| 수작업 QAT | 0.849 | 0.8375 | 9.98 | — |
| ModelOpt QAT | 0.843 | 0.832 | 1.96 | 1.58 |

(직접 실행 · CPU · 배치 1 · n=300.) QAT 이득: INT8 **+0.6%p** (False) · W3A8 **+10.95%p** (True). 교안의 의도는 T4 GPU + TensorRT — 거기서는 INT8 엔진이 FP32 보다 빠르다. 이 표의 지연은 "x86 CPU 에서는 S8 활성값 QDQ 가 느리다(QP D2·5교시)" 만 보여 준다.

##### 🔎 왜 이렇게 되나
- **① 수작업 QAT INT8 이 가장 높은 이유**: 0.849 — 하지만 FP32 와의 차이가 잡음 수준
- **② 학습 시간**: 수작업 9.98분 vs ModelOpt 1.96분 — 같은 epoch 라도 구현 효율이 다르다
- **③ CPU 지연**: ORT CPU 에서는 INT8 이 FP32 보다 느리다 — GPU · TensorRT 에서 다시 잰다
- ⚠ INT8 정확도 차이(0.842 ~ 0.849)는 평가 2,000장에서 몇 장 — 순위를 매기지 않는다

<<CODE 과제 2 | 과제 2 — PTQ vs QAT 대결 표>>
<<CODE 과제 3 | 과제 3 — QAT 이득과 판정>>
<<CODE 과제 4 | 과제 4 — CSV 저장>>
<<CODE 과제 5 | 과제 5 — QAT 정확도 곡선 그림>>

#### 개념 3 · 과제 6 의 생각할 거리
- **QAT 가 필요한 경우 / PTQ 로 충분한 경우**: W3A8 처럼 PTQ 손실이 큰 저비트 → QAT (+10.95%p) · INT8 → PTQ (+0.6%p 는 잡음 수준).
- **수작업 vs ModelOpt**: 정확도는 비슷(0.8375 vs 0.832) — 배포에는 ModelOpt (대칭 QDQ ONNX · `mto` 체크포인트 · TensorRT 연결).
- **outlier 모델에 QAT**: 강사 실험 0.25 → 0.50 — 원인이 범위(outlier)면 학습으로 잘 안 풀린다 → 범인 layer FP32 · 채널 균등화.
- **W3A8 을 빠르게**: 3비트 weight 를 실제로 계산하는 커널(저비트 GEMM) · 장치 지원이 있어야 한다.

##### 🔎 왜 이렇게 되나
- **① QAT 가 필요한 경우**: PTQ 손실이 큰 저비트 (W3A8 +10.95%p)
- **② PTQ 로 충분한 경우**: INT8 (+0.6%p 는 잡음 수준) — 학습 시간을 아낀다
- **③ outlier 모델**: QAT 로 0.25 → 0.50 — 원인이 범위면 학습으로 잘 안 풀린다 → 범인 layer FP32 · 채널 균등화 (Day 3)
- ⚠ 먼저 PTQ 로 손실을 재고, 한도를 넘을 때만 QAT 를 꺼낸다

#### 한 번에 정리
- 같은 모델 · 같은 자로 5가지 방법을 비교 — INT8 은 PTQ 로 충분(+0.6%p), W3A8 은 QAT 필요(+10.95%p).
- QAT 의 비용(학습 분 · GPU)과 이득(%p)을 함께 적어야 결론이 선다.
- 배포는 ModelOpt → QDQ ONNX → TensorRT (보정 없이 INT8 + FP16).
- ⚠ 저비트는 실행 커널이 있는지까지 확인 — 지연은 배포 장치에서 다시 잰다.
