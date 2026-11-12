## Day 1 — 양자화의 수학: 숫자를 INT8 에 담기

### 1교시 · 왜 경량화인가: 크기 · 메모리 · 속도 · 전력

**키워드**: `모델 크기 = 파라미터 수 × byte`, `FP32 4 · FP16 2 · INT8 1`, `메모리 대역폭`, `경량화 4가지`, `크기 · 속도 · 정확도 함께 기록`

**개요**: 현장 장치(Jetson · Raspberry Pi · MCU)는 Colab 보다 메모리가 작고 메모리에서 숫자를 읽어 오는 속도(대역폭)도 느리다. 추론 한 번마다 모든 가중치를 읽어야 하므로 숫자 하나가 4 byte → 1 byte 가 되면 읽을 양이 1/4 로 준다. 오늘은 VB · OT 에서 쓴 SimpleConvNet 의 **진짜 학습된 가중치**(MNIST 100장 정확도 1.00)로 하루 종일 실험한다. 이 과목 6일: Day 1 양자화의 수학(NumPy) → Day 2 ONNX Runtime PTQ → Day 3 PTQ 디버깅 · TensorRT INT8 → Day 4 QAT → Day 5 Pruning → Day 6 통합 경량화(Pruning + 증류 + INT8).

**핵심 용어**
- **경량화(model compression)**: 정확도를 크게 잃지 않으면서 모델의 크기 · 메모리 · 연산량을 줄이는 기술
- **양자화(quantization)**: 숫자를 더 적은 비트의 자료형(FP32 → INT8)으로 바꿔 담는 것
- **가지치기(pruning)**: 중요하지 않은 가중치 · 채널을 0 으로 만들거나 없애는 것 (Day 5)
- **지식 증류(knowledge distillation)**: 큰 모델(선생)의 출력을 작은 모델(학생)이 따라 배우게 하는 것 (Day 6)
- **메모리 대역폭(memory bandwidth)**: 메모리에서 초당 읽어 올 수 있는 byte 수 — 작은 장치에서 추론 속도를 묶는 병목

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Tensor.numel()` | `sum(p.numel() for p in cnn.parameters())` | 텐서 원소 개수 → 파라미터 수 |
| `torchvision.models.mobilenet_v3_small()` | `mobilenet_v3_small(weights=None)` | 구조만 만들어 파라미터 수 세기 (내려받기 없음) |

**핵심 내용**

#### 모델 크기 = 파라미터 수 × 숫자 하나의 byte
> 저장 크기(byte) = 파라미터 개수 × 자료형 byte 수 (FP32 4 · FP16 2 · INT8 1 · INT4 0.5)
- **SimpleConvNet** → 433,890 개 × 4 B = **1.736 MB** (FP32) → INT8 0.434 MB = **1/4**
- **예** → MobileNetV3-Small 2,542,856 개: FP32 10.17 · FP16 5.09 · **INT8 2.54 MB** / YOLOv8n 3,193,934 개: 12.78 · 6.39 · 3.19 MB
- **비트를 줄이면** → 크기는 비트에 비례해 줄지만 담을 수 있는 서로 다른 값도 2^비트 로 준다 (8비트 256 · 4비트 16 · 2비트 4)
- ⚠ MB = 10⁶ byte 로 계산 (강의 기준) — 2²⁰ 이 아님

#### 경량화 4가지와 기록 원칙
> 양자화 · 가지치기 · 지식 증류 · 효율 구조(MobileNet 같은 가벼운 설계) — 이 과목은 앞의 셋
- **작아지면 좋아지는 것** → 저장 공간 · 메모리 · 읽는 시간(속도) · 전력
- **기록 원칙** → 경량화 결과는 항상 **크기 · 속도 · 정확도** 세 가지를 한 표에
- ⚠ "크기 1/4 = 속도 4배" 아님 — 속도는 하드웨어 · 런타임의 INT8 연산 지원에 달림 (Day 2 · 3 에서 측정)
- ⚠ "INT8 = 정확도 1/4" 아님 — 잘 하면 손실 1%p 안팎. **얼마나 잃는지 재는 것**이 이 과목의 핵심

**기본 사용형태**
```python
n_params = sum(p.numel() for p in cnn.parameters())   # 433,890
BYTES = {"FP32": 4, "FP16": 2, "INT8": 1}
size_mb = {dt: n_params * b / 1e6 for dt, b in BYTES.items()}   # 1.736 · 0.868 · 0.434
ratio = size_mb["FP32"] / size_mb["INT8"]               # 4.0
```

**연결**
- [OT D4·6교시 FP16 으로 더 빠르게](#/ot/4/6) — FP16 은 이미 해 본 "절반 크기" 경량화, 오늘은 INT8 로 1/4
- [OT D1·6교시 MNIST CNN 내보내기](#/ot/1/6) — 오늘 쓰는 SimpleConvNet · `mnist_cnn.pt` 가 바로 그 모델
- [OT D5·8교시 다음 과목](#/ot/5/8) — "INT8 이면 성능표가?" 라는 예고가 이 과목

**상세 학습**

#### 이 교시에서 답하는 질문
OT 에서는 모델을 ONNX · TensorRT 로 옮겨 **같은 숫자를 더 빨리** 계산했다(FP16 까지). 하지만 숫자 자체는 여전히 FP32 · FP16 이었다. 현장 장치에서 진짜 병목은 "가중치를 메모리에서 읽어 오는 양" 이다. 이 교시는 모델 크기를 숫자로 세어 보고, 숫자 하나를 1 byte 로 줄이면(INT8) 무엇이 좋아지고 무엇을 확인해야 하는지 정한다.

#### 개념 1 · 크기 = 파라미터 수 × byte
![세 모델의 FP32 · FP16 · INT8 크기 막대 — 모두 정확히 1/2 · 1/4](assets/fig/qp/d1-why.svg)

SimpleConvNet 의 파라미터를 층별로 세면:

| 층 | shape | 개수 |
|---|---|---|
| conv.weight | (30, 1, 5, 5) | 750 |
| conv.bias | (30,) | 30 |
| fc1.weight | (100, 4320) | **432,000** |
| fc1.bias | (100,) | 100 |
| fc2.weight | (10, 100) | 1,000 |
| fc2.bias | (10,) | 10 |
| 합 | | **433,890** |

fc1 하나가 99.6% 를 차지한다 — 크기를 줄이려면 큰 층이 중요하다. 비트별 크기: 16비트 0.868 · 8비트 0.434 · 4비트 0.217 · 2비트 0.108 MB.

#### 개념 2 · 경량화 4가지
| 방법 | 무엇을 줄이나 | 이 과목 |
|---|---|---|
| 양자화 | 숫자 하나의 비트 수 (FP32 → INT8) | Day 1 ~ 4 |
| 가지치기 | 가중치 · 채널 개수 | Day 5 |
| 지식 증류 | 큰 모델 → 작은 모델로 지식 옮기기 | Day 6 |
| 효율 구조 | 처음부터 가볍게 설계 (MobileNet) | 다루지 않음 |

#### 코드로 확인
```python
n_params = sum(p.numel() for p in cnn.parameters())
fp32_mb = n_params * 4 / 1e6
BYTES = {"FP32": 4, "FP16": 2, "INT8": 1}
size_mb = {m: {dt: PARAMS[m] * b / 1e6 for dt, b in BYTES.items()} for m in PARAMS}
ratio = size_mb["SimpleConvNet"]["FP32"] / size_mb["SimpleConvNet"]["INT8"]
```
- **무엇을 하나**: 파라미터 수를 세고 자료형별 크기 표와 압축률을 만든다.
- **어떤 개념**: 개념 1.
- **출력**: `파라미터 433,890 개 → FP32 1.736 MB` · MobileNetV3-Small INT8 `2.54MB` · `4.0 배 작아진다` (직접 실행)

#### 한 번에 정리
- 크기 = 파라미터 수 × byte. FP32 4 · FP16 2 · INT8 1 → INT8 은 정확히 1/4.
- 경량화 4가지 중 양자화 · 가지치기 · 증류를 6일 동안.
- 결과는 크기 · 속도 · 정확도를 함께.
- ⚠ 크기 1/4 ≠ 속도 4배 — 속도는 직접 잰다.

---

### 2교시 · 숫자 표현: FP32 · FP16 · INT8 · UINT8

**키워드**: `부호 · 지수 · 가수`, `np.finfo`, `np.iinfo`, `FP16 최대 65,504`, `wrap-around`, `round → clip → astype`

**개요**: 양자화를 하려면 각 자료형이 **어떤 범위**의 숫자를 **어떤 간격**으로 담는지 알아야 한다. 부동소수점(FP32 · FP16)은 "지수 + 가수" 라 범위가 넓고 0 근처가 촘촘하다. 정수(INT8 · UINT8)는 256 칸 · 간격 1 이고 소수가 없다 — 그래서 실수를 담으려면 **scale** 이 필요하다(3교시). 실수를 그냥 `astype(np.int8)` 하면 소수점이 버려지고 범위 밖은 한 바퀴 돌아 엉뚱한 값이 된다.

**핵심 용어**
- **FP16(half precision)**: 부호 1 · 지수 5 · 가수 10 비트 부동소수점 — 최대 65,504, 유효숫자 약 3자리
- **BF16(bfloat16)**: 부호 1 · 지수 8 · 가수 7 — FP32 와 같은 범위, 간격은 더 거침
- **eps(machine epsilon)**: 1 과 그다음 표현 가능한 수의 간격 — FP32 1.19×10⁻⁷, FP16 0.000977
- **wrap-around(넘침 되돌이)**: 정수 범위를 넘은 값이 반대쪽 끝부터 다시 세어지는 현상 (200 → −56)
- **UINT8**: 부호 없는 8비트 정수 0 ~ 255

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `np.finfo()` | `np.finfo(np.float16).max` | 부동소수점 자료형의 최댓값 · eps · tiny |
| `np.iinfo()` | `np.iinfo(np.int8).min` | 정수 자료형의 최솟값 · 최댓값 |
| `ndarray.astype()` | `x.astype(np.float16)` | 자료형 바꾸기 (정수로는 소수점 버림 · 넘침 주의) |
| `np.clip()` | `np.clip(x, -128, 127)` | 범위 밖 값을 끝값으로 자르기 |

**핵심 내용**

#### 부동소수점 vs 정수 — 범위와 간격
> 부동소수점 = 지수(범위) + 가수(정밀도), 정수 = 일정한 간격 1 의 256 칸
- **FP32** → 1 · 8 · 23 비트, 최대 3.4×10³⁸, eps 1.19×10⁻⁷
- **FP16** → 1 · 5 · 10 비트, **최대 65,504** → 70000 은 `inf`, 1000.1 은 1000 (차이 −0.0998)
- **INT8 · UINT8** → −128 ~ 127 · 0 ~ 255, 둘 다 **256 개** 값, 간격 1
- **간격 모양** → 부동소수점은 0 근처 촘촘 · 큰 수 성김 / 정수 + scale 은 어디서나 같은 간격
- ⚠ FP16 범위를 넘으면 오류 없이 `inf` — 큰 활성값이 있는 모델은 FP16 에서 조용히 망가질 수 있다

#### 실수 → 정수 변환은 round → clip → astype
> 가장 가까운 정수로 반올림 → 범위 안으로 자르기 → 형 변환, 이 순서만 안전
- **astype 만** → `[0.3, 1.7, −1.6, 200, −300]` → `[0, 1, −1, −56, −44]` (소수 버림 + wrap-around)
- **올바른 순서** → `np.clip(np.round(v), -128, 127).astype(np.int8)` → `[0, 2, −2, 127, −128]`
- **곱셈** → `np.int8(100) * np.int8(100)` = **16** (10000 이 넘침) → INT32 로 곱하면 10000 (7교시)
- ⚠ `np.uint8` 끼리 빼기 → 음수가 255 근처 큰 수. 계산 전에 `astype(np.int32)` 로 넓힌다

**기본 사용형태**
```python
np.iinfo(np.int8).min, np.iinfo(np.int8).max      # -128, 127
np.finfo(np.float16).max                          # 65504.0
x16 = x.astype(np.float16)                        # 70000 → inf
safe = np.clip(np.round(v), -128, 127).astype(np.int8)
```

**연결**
- [ARCH D2·4교시 정수의 구조](#/arch/2/4) — 2의 보수 · 8비트 범위 −128 ~ 127, wrap-around 의 원리
- [PY D1·3교시 연산자와 비트 연산](#/py/1/3) — 비트로 숫자를 담는 감각
- [OT D4·6교시 FP16 으로 더 빠르게](#/ot/4/6) — FP16 의 유효숫자 3자리가 정확도 차이로 보였던 이유

**상세 학습**

#### 이 교시에서 답하는 질문
1교시에서 "INT8 이면 1/4" 를 확인했다. 그런데 INT8 에는 소수가 없고 −128 ~ 127 밖에 못 담는다. 가중치 0.0123 같은 실수를 어떻게 넣을까? 먼저 각 자료형의 **범위와 간격**을 숫자로 확인하고, 실수를 정수로 바꿀 때 생기는 두 가지 사고(소수 버림 · 넘침)를 막는 순서를 익힌다.

#### 개념 1 · 비트 구조
![FP32 · FP16 · BF16 · INT8 · UINT8 의 부호 · 지수 · 가수 비트 칸](assets/fig/qp/d1-formats.svg)

| 자료형 | 비트 (부호 · 지수 · 가수) | 최대 | 1 근처 간격(eps) |
|---|---|---|---|
| FP32 | 1 · 8 · 23 | 3.4×10³⁸ | 1.19×10⁻⁷ |
| FP16 | 1 · 5 · 10 | 65,504 | 0.000977 |
| INT8 | 8 (2의 보수) | 127 (최소 −128) | 1 |
| UINT8 | 8 | 255 (최소 0) | 1 |

FP16 의 가수 10비트 → 유효숫자 약 3자리. 그래서 1000.1 을 담으면 1000 이 된다(1000 근처 간격이 0.5).

#### 개념 2 · 정수 변환의 두 가지 사고
| 입력 | `astype(np.int8)` 만 | round → clip → astype |
|---|---|---|
| 0.3 | 0 | 0 |
| 1.7 | 1 (버림) | 2 |
| −1.6 | −1 (버림) | −2 |
| 200 | **−56** (200 − 256) | 127 |
| −300 | **−44** (−300 + 256) | −128 |

#### 코드로 확인
```python
x = np.array([0.1, 1.0, 1000.1, 70000.0], np.float32)
x16 = x.astype(np.float16)
err = x16.astype(np.float32) - x
fp16_max = float(np.finfo(np.float16).max)
safe = np.clip(np.round(v), -128, 127).astype(np.int8)
```
- **무엇을 하나**: FP16 으로 바꿨을 때의 차이와 안전한 INT8 변환.
- **어떤 개념**: 개념 1 · 2.
- **출력**: `1000.1 → 1000` (차이 −0.0998) · `70000 → inf` · `FP16 최댓값: 65504.0` · `[0, 2, −2, 127, −128]` (직접 실행)

#### 한 번에 정리
- 부동소수점은 범위 넓고 0 근처 촘촘, 정수는 256 칸 간격 1.
- FP16 최대 65,504 — 넘으면 inf.
- 정수 변환은 round → clip → astype, 곱 · 누적은 INT32.
- ⚠ uint8 끼리 빼면 음수가 큰 양수로 — 넓혀서 계산.

---

### 3교시 · Affine 양자화: scale · zero-point · quantize · dequantize

**키워드**: `scale`, `zero_point`, `quantize`, `dequantize`, `범위에 0 포함`, `오차 ≤ scale/2`, `clipping 오차`

**개요**: 양자화는 실수 구간 xmin ~ xmax 를 정수 구간 qmin ~ qmax 에 **겹쳐 놓는 것**이다. 필요한 숫자는 두 개 — **scale**(정수 1칸이 실수로 얼마인가)과 **zero_point**(실수 0.0 이 몇 번 칸에 놓이나). 예: −1 ~ 3 을 UINT8 0 ~ 255 에 → scale 4/255 ≈ 0.01569, zp 64. 범위 안 값의 왕복 오차는 반 칸(scale/2) 이하, 범위 밖 값은 끝 칸에 붙어 잘린 만큼 틀린다.

**핵심 용어**
- **scale(스케일)**: 정수 한 칸의 실수 크기 = (xmax − xmin) / (qmax − qmin)
- **zero-point(영점, zp)**: 실수 0.0 이 놓이는 정수 칸 = qmin − round(xmin / scale) — 항상 정수
- **quantize(양자화)**: 실수 → 정수, q = clip(round(x / scale) + zp, qmin, qmax)
- **dequantize(역양자화)**: 정수 → 실수, x̂ = (q − zp) × scale
- **반올림 오차(rounding error)**: 범위 안 값이 가장 가까운 칸으로 옮겨지며 생기는 오차 — 최대 scale/2
- **잘림 오차(clipping error)**: 범위 밖 값이 끝 칸에 붙어 생기는 오차 — 잘린 거리만큼

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `calc_scale_zp()` | `s, zp = calc_scale_zp(-1.0, 3.0)` | 범위 → scale · zero_point (기본 UINT8 0 ~ 255) |
| `quantize()` | `q = quantize(x, s, zp)` | 실수 → 정수 (INT32 배열로 돌려줌) |
| `dequantize()` | `x_hat = dequantize(q, s, zp)` | 정수 → 실수 |

**핵심 내용**

#### scale 과 zero-point
> scale = (xmax − xmin) / (qmax − qmin), zp = qmin − round(xmin / scale) — 실수 범위를 정수 칸에 겹치는 두 숫자
- **0 포함** → 먼저 xmin = min(xmin, 0), xmax = max(xmax, 0) — 0 이 정수 칸에 정확히 놓여야 ReLU 출력 · padding 의 0 이 0 으로 돌아옴
- **예** → −1 ~ 3 → scale = 4/255 = 0.015686, zp = 0 − round(−63.75) = **64** / 0.5 ~ 3 → 0 ~ 3 으로 넓혀 scale 3/255, zp 0
- **모두 0 인 텐서** → xmax = xmin 이면 나눗셈 피해 (1.0, qmin)
- ⚠ zp 는 반드시 **정수** (그리고 qmin ~ qmax 안) — 실수 zp 면 0 이 정확히 표현되지 않는다

#### quantize · dequantize 와 두 가지 오차
> q = clip(round(x / scale) + zp, qmin, qmax), x̂ = (q − zp) × scale
- **예** → `[−1, 0, 0.5, 1.234, 3]` → q `[0, 64, 96, 143, 255]` → x̂ `[−1.0039, 0, 0.502, 1.2392, 2.9961]`
- **반올림 오차** → 범위 안 난수 10,000 개의 최대 오차 0.0078431 ≤ scale/2 = 0.0078431, **0.0 → 0.0 정확히**
- **잘림 오차** → 범위 밖 4.0 · 5.0 → 255 → 2.996 (오차 1.004 · 2.004), −2.0 → 0 → −1.004 — scale/2 의 **127배 이상**
- **비트 수** → 8비트 scale 0.0157 · 최대 오차 0.0078 / 4비트 0.267 · 0.133 / 2비트 1.333 · 0.667
- ⚠ quantize 결과를 바로 `np.uint8` 로 두고 `q − zp` → wrap-around. 계산은 INT32, 저장할 때만 8비트

**기본 사용형태**
```python
def calc_scale_zp(xmin, xmax, qmin=0, qmax=255):
    xmin, xmax = min(xmin, 0.0), max(xmax, 0.0)       # 0 포함
    if xmax == xmin: return 1.0, qmin
    scale = (xmax - xmin) / (qmax - qmin)
    zp = int(np.clip(qmin - round(xmin / scale), qmin, qmax))
    return scale, zp

def quantize(x, scale, zp, qmin=0, qmax=255):
    return np.clip(np.round(np.asarray(x) / scale) + zp, qmin, qmax).astype(np.int32)

def dequantize(q, scale, zp):
    return (np.asarray(q, np.float64) - zp) * scale
```

**연결**
- [NN D1·3교시 NumPy ndarray](#/nn/1/3) — `np.round` · `np.clip` · `astype` 를 배열 전체에 한 번에
- [VB D1·6교시 디지털 이미지](#/vb/1/6) — 픽셀 0 ~ 255 를 `/255` 로 0 ~ 1 로 — 거꾸로 하면 scale = 1/255 인 양자화
- [OT D1·7교시 전처리 6단계](#/ot/1/7) — 정규화 `(x − mean) / std` 와 같은 "빼고 나누기" 꼴

**상세 학습**

#### 이 교시에서 답하는 질문
2교시에서 INT8 에는 소수가 없다는 것을 봤다. 그러면 0.0123 같은 가중치를 어떻게 담을까? 답은 **눈금 바꾸기**다 — 실수 범위를 256 칸으로 나누고, 실수 대신 "몇 번째 칸인가" 만 저장한다. 이 교시는 그 눈금을 정하는 두 숫자(scale · zp)와 왕복 함수 두 개를 만들고, 왕복하면 얼마나 틀리는지 잰다.

#### 개념 1 · 실수 범위를 정수 칸에 겹치기
비유: 0 ~ 4 m 길이를 255 칸 자로 잰다. 한 칸 = 4/255 m(scale). 자의 0 눈금이 64번째 칸에 있다(zp).

![−1 ~ 3 을 0 ~ 255 에 겹친 그림, 0.0 은 64 칸으로](assets/fig/qp/d1-affine.svg)

| 기호 | 뜻 | 예 (−1 ~ 3 → UINT8) |
|---|---|---|
| xmin, xmax | 실수 범위 (0 포함) | −1, 3 |
| qmin, qmax | 정수 범위 | 0, 255 |
| scale | 한 칸의 실수 크기 | 4/255 = 0.015686 |
| zp | 0.0 의 칸 번호 | 64 |

#### 개념 2 · 왕복 오차 = 반올림 + 잘림
- 범위 안: round 는 많아야 **반 칸** 틀린다 → |x − x̂| ≤ scale/2.
- 범위 밖: clip 이 끝 칸에 붙인다 → 잘린 거리만큼 틀린다 (4.0 → 2.996, 오차 1.004).
- 비트를 줄이면 칸이 적어 scale 이 커지고 최대 오차도 커진다:

| 비트 | 칸 | scale | zp | 최대 오차 |
|---|---|---|---|---|
| 8 | 256 | 0.0157 | 64 | 0.0078 |
| 6 | 64 | 0.0635 | 16 | 0.0317 |
| 4 | 16 | 0.2667 | 4 | 0.1333 |
| 2 | 4 | 1.3333 | 1 | 0.6666 |

#### 코드로 확인
```python
s, zp = calc_scale_zp(-1.0, 3.0)
q = quantize(np.array([-1.0, 0.0, 0.5, 1.234, 3.0]), s, zp)
x_rand = np.random.uniform(-1, 3, 10_000)
x_hat = dequantize(quantize(x_rand, s, zp), s, zp)
max_err = np.abs(x_hat - x_rand).max()
x_out_hat = dequantize(quantize(np.array([4.0, 5.0, -2.0]), s, zp), s, zp)
```
- **무엇을 하나**: scale · zp 를 구하고 범위 안 · 밖 값을 왕복시켜 오차를 잰다.
- **어떤 개념**: 개념 1 · 2.
- **출력**: `scale = 0.015686  zero_point = 64` · `q : [0 64 96 143 255]` · `최대 오차 0.00784 ≤ scale/2 = 0.00784 ? True` · `0.0 → 0.0` · 되돌린 값 `[2.9961 2.9961 −1.0039]` (직접 실행)

#### 한 번에 정리
- scale = 범위 / 칸 수, zp = 0.0 이 놓이는 정수 칸 (항상 정수).
- quantize = clip(round(x / scale) + zp), dequantize = (q − zp) × scale.
- 범위 안 오차 ≤ scale/2, 범위 밖은 잘린 만큼.
- ⚠ 범위에 0 을 넣지 않으면 0 이 정확히 안 돌아와 오차가 계속 쌓인다.

---

### 4교시 · 대칭 vs 비대칭: ReLU 출력과 weight

**키워드**: `비대칭(asymmetric)`, `대칭(symmetric)`, `zp = 0`, `−127 ~ 127`, `ReLU 출력 → 비대칭`, `weight → 대칭`

**개요**: 비대칭은 데이터 범위 xmin ~ xmax 를 그대로 쓰고 zp 가 데이터에 따라 정해진다(UINT8 0 ~ 255). 대칭은 −max|x| ~ +max|x| 로 범위를 0 중심으로 잡아 **zp = 0** 이다(INT8 −127 ~ 127). 대칭은 계산이 간단하지만(7교시) 한쪽으로 치우친 데이터에서는 칸을 낭비한다. 실험: ReLU 뒤 활성값(모두 0 이상)은 비대칭이 **5.1배** 정확, 0 중심으로 퍼진 conv weight 는 대칭이 겨우 1.2배 손해.

**핵심 용어**
- **비대칭 양자화(asymmetric)**: 범위 xmin ~ xmax · zp 있음 · 보통 UINT8 — 활성값에 (ONNX Runtime 기본 U8)
- **대칭 양자화(symmetric)**: 범위 ±max|x| · zp = 0 · INT8 −127 ~ 127 — weight 에 (TensorRT 는 활성값도 대칭)
- **MSE(평균 제곱 오차)**: 양자화 전후 값 차이의 제곱 평균 — 오차를 숫자 하나로 비교

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `sym_scale()` | `st = sym_scale(x, bits=8)` | 대칭 scale = max\|x\| / (2^(bits−1) − 1) |
| `sym_quantize()` | `q = sym_quantize(x, st)` | round(x / scale) → clip(−qmax, qmax) → INT32 |
| `mse()` | `mse(x_hat, x)` | 평균 제곱 오차 (환경 셀이 제공) |

**핵심 내용**

#### 대칭 양자화
> scale = max|x| / 127, q = clip(round(x / scale), −127, 127), x̂ = q × scale — zp 가 없는 양자화
- **예** → `[−2, −0.5, 0, 1]` → scale 2/127 = 0.01575 → q `[−127, −32, 0, 64]` → x̂ `[−2, −0.504, 0, 1.008]`
- **왜 −128 을 안 쓰나** → −128 ~ 127 은 0 기준으로 비대칭 — −127 ~ 127 로 정확히 0 중심
- **장점** → zp = 0 이라 정수 곱셈 식이 짧다 (7교시)
- ⚠ 대칭 역양자화에 zp 를 빼지 않는다 — q × scale 만

#### 데이터 모양에 맞춰 고르기
> 한쪽으로 치우친 값 → 비대칭, 0 중심으로 퍼진 값 → 대칭
- **ReLU 뒤 활성값** → 432,000 개 · 0 ~ 2.137 · 0 인 값 51.8% → 비대칭 scale 0.00838 MSE 2.49e-06 / 대칭 scale 0.01683 MSE 1.26e-05 → **5.1배**
- **conv weight** → −0.482 ~ 0.387 → 비대칭 MSE 9.35e-07 / 대칭 1.12e-06 → **1.20배** (거의 손해 없음)
- **치우침이 커지면** → weight + 0.1 → 1.25배, + 0.3 → 2.55배, + 0.5 → 3.95배
- ⚠ 관례: weight = 대칭 INT8, 활성값 = 비대칭 UINT8 (ONNX Runtime) 또는 대칭 INT8 (TensorRT)

**기본 사용형태**
```python
def sym_scale(x, bits=8):
    return float(np.abs(x).max() / (2 ** (bits - 1) - 1))

def sym_quantize(x, scale, bits=8):
    qmax = 2 ** (bits - 1) - 1
    return np.clip(np.round(np.asarray(x) / scale), -qmax, qmax).astype(np.int32)

st = sym_scale(W); W_hat = sym_quantize(W, st) * st        # 대칭 왕복
```

**연결**
- [NN D3·4교시 활성화 계층: ReLU](#/nn/3/4) — ReLU 출력이 모두 0 이상이라 비대칭이 맞는 이유
- [VB D1·4교시 nn.Module](#/vb/1/4) — `cnn.conv.weight.detach().numpy()` 로 weight 꺼내기

**상세 학습**

#### 이 교시에서 답하는 질문
3교시의 양자화는 범위를 데이터에 맞추고 zp 를 따로 두었다(비대칭). 그런데 zp 가 있으면 정수 곱셈 식에 보정 항이 생겨 계산이 길어진다(7교시). zp 를 0 으로 고정하면(대칭) 간단해지지만 손해는 없을까? 이 교시는 **실제 활성값과 weight** 로 두 방식의 오차를 재서 "어디에 어느 방식" 을 정한다.

#### 개념 1 · 두 방식 비교
| 방식 | 범위 | zero_point | 정수 범위 (8비트) | 쓰는 곳 (관례) |
|---|---|---|---|---|
| 비대칭 | xmin ~ xmax | 데이터에 따라 | UINT8 0 ~ 255 | 활성값 (ONNX Runtime U8) |
| 대칭 | −max\|x\| ~ +max\|x\| | 항상 0 | INT8 −127 ~ 127 | weight (TensorRT 는 활성값도) |

#### 개념 2 · ReLU 출력에 대칭을 쓰면 칸 절반이 빈다
활성값은 모두 0 이상이다. 대칭으로 잡으면 −2.137 ~ 0 의 127칸에는 값이 하나도 없고, 실제 값은 나머지 절반 칸에만 놓인다 → scale 이 2배(0.01683 vs 0.00838) → MSE 약 4~5배.

![ReLU 활성값 분포 위에 비대칭(0 ~ max)과 대칭(±max) 범위를 겹친 그림](assets/fig/qp/d1-symasym.svg)

| 데이터 | 범위 | 비대칭 MSE | 대칭 MSE | 대칭 / 비대칭 |
|---|---|---|---|---|
| ReLU 뒤 활성값 A (432,000 개) | 0 ~ 2.137 | 2.49e-06 | 1.26e-05 | **5.07** |
| conv weight W (750 개) | −0.482 ~ 0.387 | 9.35e-07 | 1.12e-06 | **1.20** |

#### 코드로 확인
```python
with torch.no_grad():
    A = cnn.pool(F.relu(cnn.conv(X))).numpy().ravel()     # (432,000,)
s_a, zp_a = calc_scale_zp(A.min(), A.max())
mse_asym = mse(dequantize(quantize(A, s_a, zp_a), s_a, zp_a), A)
s_s = sym_scale(A)
mse_sym = mse(sym_quantize(A, s_s) * s_s, A)
```
- **무엇을 하나**: 같은 활성값을 두 방식으로 왕복해 MSE 를 비교한다.
- **어떤 개념**: 개념 2.
- **출력**: `0 인 비율 51.8%` · `비대칭 scale 0.00838 MSE 2.49e-06` · `대칭 scale 0.01683 MSE 1.26e-05 → 5.1 배` (직접 실행)

#### 한 번에 정리
- 비대칭: 범위 그대로 · zp 있음 · 치우친 값(ReLU 출력)에 유리.
- 대칭: ±max|x| · zp = 0 · 0 중심 값(weight)에 적합, 정수 연산이 단순.
- 관례: weight 대칭 INT8 · 활성값 비대칭 UINT8(ORT) 또는 대칭 INT8(TensorRT).
- ⚠ 대칭은 −127 ~ 127 (−128 은 안 씀).

---

### 5교시 · Granularity: per-tensor vs per-channel

**키워드**: `granularity`, `per-tensor`, `per-channel`, `출력 채널 = 0번 축`, `(C, 1, 1, 1) broadcast`, `outlier channel`

**개요**: scale 을 텐서 전체에 **하나**(per-tensor) 쓸 수도, **출력 채널마다 하나씩**(per-channel) 쓸 수도 있다. conv weight (30, 1, 5, 5) 라면 필터 30개가 각자 scale 을 갖는다. per-tensor scale 은 가장 큰 필터에 맞춰지므로 작은 필터가 거친 눈금을 쓰게 된다. 실험: conv weight 는 per-channel 이 **1.91배**, fc1 weight 는 **2.63배** 정확. weight 는 per-channel 이 표준(ONNX Runtime `per_channel=True`, TensorRT 기본).

**핵심 용어**
- **granularity(입도)**: scale 하나가 담당하는 범위의 단위 — 텐서 전체 · 채널 · 그룹
- **per-tensor**: 텐서 전체에 scale 1개
- **per-channel**: 출력 채널(0번 축)마다 scale 1개 — conv (30, 1, 5, 5) → 30개, fc1 (100, 4320) → 100개
- **outlier channel(튀는 채널)**: 다른 채널보다 값이 유난히 큰 채널 — per-tensor scale 을 혼자 키운다
- **group-wise 양자화**: 채널을 다시 32 · 64 · 128 개씩 묶어 묶음마다 scale — 4비트 LLM 의 비결

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `per_channel_scale()` | `sc = per_channel_scale(W)` | 0번 축마다 max\|W\| / qmax → (C, 1, …) 모양 |
| `ndarray.reshape()` | `W.reshape(30, -1)` · `sc_w.reshape(30, 1, 1, 1)` | 채널별로 펴기 · broadcast 모양 맞추기 |

**핵심 내용**

#### per-tensor vs per-channel
> per-channel = 출력 채널(0번 축)마다 자기 max|W| 로 scale 을 따로 — 작은 필터도 촘촘한 눈금
- **per-tensor** → scale 0.003798 하나 = 가장 큰 필터의 scale
- **per-channel** → scale 30개, 0.001612 ~ 0.003798 (2.4배 차이) — 가장 큰 것 = per-tensor scale
- **예** → conv MSE 1.12e-06 → 5.86e-07 (**1.91배**), fc1 (100, 4320) **2.63배**
- **튀는 필터** → 필터 0 을 ×3 · ×10 · ×30 하면 차이 5.6 · 18.0 · 23.3배 — per-tensor 는 29개가 함께 손해
- ⚠ scale 개수만큼 파일이 커진다 (FP32 4 B × 채널 수 — 8교시에서 548 B)

#### 채널 scale 의 broadcast 모양
> (30,) 을 (30, 1, 1, 1) 로 바꿔야 W (30, 1, 5, 5) 와 채널끼리 짝지어 나뉜다
- **채널별 max** → `np.abs(W.reshape(30, -1)).max(1)` → (30,)
- **Conv · Linear 공통** → 0번 축 = 출력 채널 → conv (30, 1, 1, 1), fc1 (100, 1)
- ⚠ `W / sc_w` 를 (30,) 그대로 하면 마지막 축(5)과 맞추려다 오류 또는 엉뚱한 broadcast
- ⚠ 활성값은 per-tensor — 다음 층에서 여러 채널을 더하므로 채널마다 scale 이 다르면 정수 덧셈이 안 된다

**기본 사용형태**
```python
def per_channel_scale(W, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(W.reshape(W.shape[0], -1)).max(1) / qmax          # (C,)
    return s.reshape((-1,) + (1,) * (W.ndim - 1))                 # (C, 1, 1, 1) 또는 (C, 1)

sc = per_channel_scale(W)
W_pc = np.clip(np.round(W / sc), -127, 127) * sc                  # per-channel 왕복
```

**연결**
- [NN D1·4교시 벡터화](#/nn/1/4) — broadcast 규칙: 뒤에서부터 축을 맞추므로 (30, 1, 1, 1) 이 필요
- [VB D1·4교시 nn.Module](#/vb/1/4) — Conv2d weight 의 (out, in, kH, kW) 순서

**상세 학습**

#### 이 교시에서 답하는 질문
4교시까지는 텐서 하나에 scale 하나였다. 그런데 conv 필터 30개는 크기가 제각각이다. 가장 큰 필터에 맞춘 scale 하나로 30개를 모두 재면, 값이 작은 필터는 256칸 중 일부만 쓰게 된다. 이 교시는 **scale 을 몇 개 둘 것인가**(granularity)를 정하고, 채널마다 scale 을 두면 얼마나 좋아지는지 잰다.

#### 개념 1 · 필터마다 알맞은 scale 이 다르다
![conv 필터 30개의 scale 막대와 per-tensor scale 점선](assets/fig/qp/d1-perch.svg)

| | per-tensor | per-channel |
|---|---|---|
| scale 개수 (conv) | 1 | 30 |
| scale 값 | 0.003798 | 0.001612 ~ 0.003798 |
| conv MSE | 1.12e-06 | 5.86e-07 (1.91배 ↓) |
| fc1 정확도 비 | 1 | 2.63배 |

#### 개념 2 · 튀는 채널 하나의 피해
필터 0 만 K 배로 키운 실험:

| K | per-tensor MSE | per-channel MSE | 차이 |
|---|---|---|---|
| 1 | 1.12e-06 | 5.86e-07 | 1.9배 |
| 3 | 4.12e-06 | 7.38e-07 | 5.6배 |
| 10 | 4.43e-05 | 2.47e-06 | 18.0배 |
| 30 | 4.11e-04 | 1.77e-05 | 23.3배 |

#### 코드로 확인
```python
st_w = sym_scale(W)                                       # per-tensor
sc_w = np.abs(W.reshape(30, -1)).max(1) / 127             # (30,)
sc4 = sc_w.reshape(30, 1, 1, 1)
W_pc = np.clip(np.round(W / sc4), -127, 127) * sc4
W_pt = np.clip(np.round(W / st_w), -127, 127) * st_w
print(mse(W_pt, W) / mse(W_pc, W))
```
- **무엇을 하나**: 두 granularity 로 왕복해 MSE 를 비교한다.
- **어떤 개념**: 개념 1.
- **출력**: `shape (30,) | 가장 작은 scale 0.001612 | 가장 큰 scale 0.003798` · `per-channel 이 1.91 배 정확` · fc1 `2.63 배` (직접 실행)

#### 한 번에 정리
- per-tensor: scale 1개, 가장 큰 채널에 맞춰져 작은 채널이 손해.
- per-channel: 0번 축마다 scale — weight 의 표준.
- 채널 scale 은 (C, 1, 1, 1) 로 만들어 broadcast.
- ⚠ 활성값은 per-tensor (채널끼리 더해야 하므로).

---

### 6교시 · Clipping 과 Outlier: 범위를 어디까지 잡을까

**키워드**: `MinMax`, `clipping`, `Percentile 99.9`, `반올림 오차 vs 잘림 오차`, `outlier`, `calibration`

**개요**: 지금까지는 범위를 최솟값 ~ 최댓값(MinMax)으로 잡았다. 그런데 아주 드문 큰 값 하나 때문에 범위가 넓어지면 나머지 99.9% 값이 모두 거친 간격을 쓴다. 범위를 일부러 좁히는 것이 **clipping** 이다. 좁히면 반올림 오차는 줄고 잘림 오차는 는다 — 최적점은 데이터 분포와 비트 수에 따라 다르므로 **재서 고른다**. 활성값의 범위는 보정(calibration) 데이터를 흘려 정한다(Day 2).

**핵심 용어**
- **MinMax**: 본 값의 최솟값 ~ 최댓값을 범위로 — 가장 단순, outlier 에 약함
- **clipping(범위 자르기)**: 범위를 일부러 좁혀 드문 큰 값을 끝 칸에 붙이는 것
- **Percentile(백분위) 보정**: 99.9% 같은 백분위 값을 범위 끝으로
- **outlier(이상치)**: 센서 노이즈 · 특이한 입력 때문에 섞인 아주 큰 값
- **calibration(보정)**: 대표 데이터를 모델에 흘려 활성값 범위를 정하는 과정 (Day 2 PTQ)
- **Entropy 보정**: 원래 분포와 양자화 분포의 차이(KL divergence)가 가장 작은 범위를 고르는 방법

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `np.percentile()` | `np.percentile(A, 99.9)` | 백분위 값 (그보다 큰 값이 0.1%) |
| `mse_clip()` | `mse_clip(A, clip, bits=8)` | 0 ~ clip 범위 · 2^bits 칸으로 왕복한 MSE (이 교시에서 만듦) |
| `np.linspace()` | `np.linspace(0.05, 1.0, 96) * A.max()` | 시험할 clip 후보 96개 |

**핵심 내용**

#### 오차 = 반올림 오차 + 잘림 오차
> 범위를 좁히면 칸이 촘촘해져 반올림 오차 ↓, 대신 범위 밖 값이 늘어 잘림 오차 ↑ — 그 합이 최소인 곳이 최적
- **분포** → 활성값 A 최대 2.137, 99.9 백분위 **1.607** (최대의 0.75배) — 범위를 25% 줄여도 잘리는 값은 0.1% (432개)
- **8비트 최적** → clip = max × **0.97** (MSE 2.11e-06, MinMax 2.49e-06) — 거의 MinMax
- **4비트 최적** → clip = max × **0.71** (MSE 5.47e-04, MinMax 7.99e-04) — 비트가 적을수록 더 좁게
- ⚠ "Percentile 이 항상 MinMax 보다 좋다" 아님 — 8비트 · outlier 없음이면 99.9% 의 MSE 2.63e-05 가 MinMax 2.49e-06 보다 **10배 나쁨**

#### outlier 하나와 Percentile
> 큰 값 하나가 MinMax 범위를 20배로 늘리면 나머지 431,999 개가 모두 20배 거친 눈금을 쓴다
- **실험** → A[0] = max × 20 = 42.74 → MinMax 범위 42.74 / Percentile 99.9 범위 1.607
- **전체 MSE** → MinMax 1.10e-03 < Percentile 3.94e-03 (outlier 를 잘라 크게 틀리므로)
- **나머지 431,999 개** → MinMax 1.10e-03 vs Percentile 2.63e-05 → Percentile 이 **42배** 정확
- **outlier 자체** → 42.74 → MinMax 42.74 (정확) · Percentile 1.61 (잘림)
- ⚠ 무엇으로 비교할지(전체 MSE · 대부분 값 · 모델 정확도)를 먼저 정한다 — 지표에 따라 결론이 뒤집힌다

**기본 사용형태**
```python
p999 = np.percentile(A, 99.9)                       # 1.6073
def mse_clip(x, clip, bits=8):
    qmax = 2 ** bits - 1; s = clip / qmax
    return mse(np.clip(np.round(x / s), 0, qmax) * s, x)
clips = np.linspace(0.05, 1.0, 96) * A.max()
best8 = clips[np.argmin([mse_clip(A, c, 8) for c in clips])]
```

**연결**
- [NN D1·7교시 확률 기초](#/nn/1/7) — 분포 · 백분위 · 히스토그램
- [PY D4·3교시 Matplotlib](#/py/4/3) — 로그 눈금 히스토그램 · `semilogy`
- [OT D3·2교시 시간을 재는 법](#/ot/3/2) — 평균 대신 p95 를 쓰는 것과 같은 "드문 값" 다루기

**상세 학습**

#### 이 교시에서 답하는 질문
3 ~ 5교시는 범위를 MinMax 로 잡았다. 그런데 활성값 분포를 보면 큰 값은 아주 드물다(로그 눈금으로 봐야 보일 정도). 드문 큰 값 몇 개 때문에 99.9% 의 값이 손해를 본다면, 범위를 일부러 좁히는 게 낫지 않을까? 이 교시는 **범위를 어디까지 잡을까**를 MSE 곡선으로 재서 답한다 — 그리고 그 답이 비트 수 · outlier · 비교 지표에 따라 달라진다는 것을 본다.

#### 개념 1 · 좁히기의 득실
![clip/max 에 따른 8비트 · 4비트 MSE 곡선과 최적점, 99.9 백분위 점선](assets/fig/qp/d1-clip.svg)

- 오른쪽 끝(clip = max)이 MinMax. 왼쪽으로 갈수록 범위가 좁다.
- 8비트는 칸이 촘촘해서 반올림 오차가 원래 작다 → 좁혀서 얻는 게 적고 잘리는 손해가 빨리 커진다 → 최적 0.97.
- 4비트는 칸이 16개뿐이라 반올림 오차가 크다 → 좁히는 이득이 커서 최적 0.71.

| 백분위 | clip | 8비트 MSE | 4비트 MSE |
|---|---|---|---|
| 99 | 1.174 | 6.49e-04 | 8.70e-04 |
| 99.9 | 1.607 | 2.63e-05 | **5.64e-04** |
| 99.99 | 1.867 | 3.26e-06 | 6.37e-04 |
| 100 (MinMax) | 2.137 | **2.49e-06** | 7.99e-04 |

#### 개념 2 · 보정 방법 (Day 2 미리 보기)
| 방법 | 범위 정하는 법 | 오늘의 실험 |
|---|---|---|
| MinMax | 본 값의 최솟값 ~ 최댓값 | 3 · 4교시 |
| Percentile | 99.9% 같은 백분위 | 문제 6-1 · 6-3 |
| Entropy | 원래 분포와의 KL 차이 최소 | Day 2 |
| (MSE 최소) | 오차가 가장 작은 clip 탐색 | 문제 6-2 |

보정 데이터는 실제 현장 이미지와 비슷한 것을 **수백 장** — 너무 적거나 한쪽으로 치우치면 실제 범위를 못 본다.

#### 코드로 확인
```python
A_out = A.copy(); A_out[0] = A.max() * 20
def roundtrip(x, clip):
    s = clip / 255
    return np.clip(np.round(x / s), 0, 255) * s
r_mm = roundtrip(A_out, A_out.max())
r_pc = roundtrip(A_out, np.percentile(A_out, 99.9))
print(mse(r_mm[1:], A_out[1:]) / mse(r_pc[1:], A_out[1:]))
```
- **무엇을 하나**: outlier 하나를 넣고 MinMax · Percentile 범위의 MSE 를 "전체" 와 "나머지" 로 나눠 잰다.
- **어떤 개념**: 개념 1 · outlier.
- **출력**: `범위 MinMax 42.7 | Percentile 1.607` · `all : MinMax 1.10e-03 | Percentile 3.94e-03` · `rest : … 나머지 값은 Percentile 이 42 배 정확` (직접 실행)

#### 한 번에 정리
- 오차 = 반올림(범위 안) + 잘림(범위 밖). 좁히면 앞 ↓ 뒤 ↑.
- 최적 범위는 분포와 비트 수에 따라 — 8비트 × 0.97, 4비트 × 0.71.
- outlier 하나에 MinMax 는 나머지 전체의 정밀도를 잃는다.
- ⚠ 비교 지표를 먼저 정한다 (전체 MSE 와 나머지 MSE 의 결론이 반대).

---

### 7교시 · 정수 연산 추론: INT8 × INT8 → INT32

**키워드**: `y = s_w s_x (W_q x_q + b_q)`, `INT32 누적`, `bias 는 INT32`, `requantize`, `M = s_x s_w / s_y`, `zp 보정 항`

**개요**: 지금까지는 양자화한 값을 다시 실수로 되돌렸다. 진짜 INT8 추론은 곱셈 · 덧셈을 **정수로** 한다. 대칭(zp = 0)이면 y = Wx + b ≈ s_w s_x (W_q x_q + b/(s_w s_x)) — 괄호 안은 모두 정수이고 실수 곱은 마지막 한 번뿐이다. SimpleConvNet 의 fc2 (100 → 10) 를 정수로 계산하면 FP32 와 최대 0.10 차이 · 예측 **100% 일치**.

**핵심 용어**
- **INT32 누적(accumulator)**: INT8 곱의 합을 담는 32비트 정수 — 100개만 더해도 INT16(32,767)을 넘는다
- **bias 양자화**: bias 를 scale s_x · s_w 의 INT32 로 — b_q = round(b / (s_x s_w))
- **requantize(재양자화)**: INT32 결과를 실수로 되돌리지 않고 M = s_x s_w / s_y 를 곱해 다음 층의 INT8 로 바꾸는 것
- **고정소수점 곱(fixed-point multiply)**: 하드웨어가 M 을 "정수 곱 + 비트 시프트" 로 계산하는 방법

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `np.int32()` | `xq.astype(np.int32) @ Wq.astype(np.int32).T` | 넓힌 뒤 곱해 넘침 없이 누적 |
| `np.array_equal()` | `np.array_equal(acc_fix, acc)` | 두 정수 배열이 완전히 같은지 |

**핵심 내용**

#### 정수 행렬곱 식
> y = s_w s_x (W_q x_q + b_q), b_q = round(b / (s_w s_x)) — 괄호 안은 정수, 실수 곱은 마지막 한 번
- **곱** → INT8 × INT8, **누적 · bias** → INT32, **마지막** → × s_x s_w
- **예** → H (100, 100) sx 0.1136 · W2 (10, 100) sw 0.00205 → acc −64,612 ~ 65,677 (INT16 초과)
- **bias** → b2 `[0.0486, −0.029, …]` → bq `[209, −124, 160, −273, −137, 473, 190, 9, 96, −81]`
- **결과** → logits −15.04 ~ 15.39 중 최대 차이 **0.1006**, 예측 일치율 **1.00**
- ⚠ NumPy `int8 @ int8` 결과는 **int8** — 조용히 넘친다. 곱하기 전에 INT32 로 넓힌다
- ⚠ bias 를 INT8 로 하지 않는다 — scale s_x s_w 가 아주 작아 bq 가 127 을 넘는다(473)

#### requantize 와 비대칭의 보정 항
> 다음 층으로 넘길 때는 y_q = clip(round((acc + b_q) × M)), M = s_x s_w / s_y
- **예** → sy = max|y| / 127 = 0.1212 → M = **0.001924** → y_q × sy 와 FP32 최대 차이 0.139 (< 2·sy)
- **하드웨어** → M 도 정수 곱 + 비트 시프트로 → 실수 연산이 완전히 사라짐
- **비대칭이면** → (x_q − zp) 때문에 보정 항 **zp × (W 행의 합)** 을 빼야 같아짐 (ZP 20 · 64 로 확인)
- ⚠ 그래서 weight 는 대칭 — W 쪽 zp 까지 있으면 보정 항이 더 늘어난다

**기본 사용형태**
```python
sx, sw = sym_scale(H), sym_scale(W2)
xq = sym_quantize(H, sx).astype(np.int8); Wq = sym_quantize(W2, sw).astype(np.int8)
acc = xq.astype(np.int32) @ Wq.astype(np.int32).T          # INT32 누적
bq = np.round(b2 / (sx * sw)).astype(np.int32)              # bias 도 INT32
y_int = (acc + bq) * sx * sw                                # 실수 곱은 한 번
sy = sym_scale(y_fp); M = sx * sw / sy
yq = np.clip(np.round((acc + bq) * M), -127, 127).astype(np.int8)   # requantize
```

**연결**
- [NN D3·5교시 Affine 계층](#/nn/3/5) — y = xW + b 를 정수로 계산하는 것이 오늘의 식
- [NN D1·5교시 선형대수](#/nn/1/5) — 행렬곱 `@` 과 전치 `.T`
- [OT D4·7교시 엔진 레이어 들여다보기](#/ot/4/7) — TensorRT 가 고르는 INT8 커널이 하는 일이 바로 이것 (Day 3)

**상세 학습**

#### 이 교시에서 답하는 질문
3 ~ 6교시는 "양자화 → 역양자화" 로 오차만 쟀다. 계산은 여전히 실수로 했다. 그런데 INT8 이 빠른 진짜 이유는 **정수끼리 곱하고 더하는 회로**가 실수 회로보다 작고 빠르기 때문이다. 이 교시는 Affine 층 y = Wx + b 를 정수만으로 계산하는 식을 세우고, 그때 생기는 두 가지 함정(누적 넘침 · bias 크기)과 다음 층으로 넘기는 방법(requantize)을 확인한다.

#### 개념 1 · 식을 정수로 바꾸기
W ≈ s_w W_q, x ≈ s_x x_q 를 넣으면

y ≈ (s_w W_q)(s_x x_q) + b = s_w s_x (W_q x_q + b / (s_w s_x))

| 기호 | 뜻 | 자료형 | 예 (fc2) |
|---|---|---|---|
| x_q | 입력 H 의 정수 | INT8 | (100, 100), sx 0.1136 |
| W_q | weight 의 정수 | INT8 | (10, 100), sw 0.00205 |
| W_q x_q | 곱의 합 (누적) | **INT32** | −64,612 ~ 65,677 |
| b_q | bias / (s_w s_x) | **INT32** | −273 ~ 473 |
| s_w s_x | 마지막에 한 번 곱할 실수 | FP32 | 0.000233 |

![INT8 곱 → INT32 누적 → + INT32 bias → × scale, 또는 × M 으로 INT8 출력](assets/fig/qp/d1-intmm.svg)

#### 개념 2 · 왜 INT32 인가
INT8 곱 하나는 최대 127 × 127 = 16,129. 100개를 더하면 최대 약 161만 — INT16(32,767)로도 부족하다. 실제 acc 최댓값 65,677 이 이를 보여 준다. INT32(약 21억)면 충분하다.

#### 코드로 확인
```python
acc = xq.astype(np.int32) @ Wq.astype(np.int32).T
bq = np.round(b2 / (sx * sw)).astype(np.int32)
y_int = (acc + bq) * sx * sw
max_err = np.abs(y_int - y_fp).max()
agree = (y_int.argmax(1) == y_fp.argmax(1)).mean()
```
- **무엇을 하나**: fc2 를 정수로 계산하고 FP32 결과와 비교한다.
- **어떤 개념**: 개념 1 · 2.
- **출력**: `acc (100, 10) int32 | 범위 −64612 ~ 65677 | INT16 최대 32,767` · `logits 범위 −15.04 ~ 15.39 | 최대 차이 0.1006 | 예측 일치율 1.00` · `M = 0.001924` (직접 실행)

#### 한 번에 정리
- 대칭이면 y = s_w s_x (W_q x_q + b_q) — 괄호 안은 전부 정수.
- 곱은 INT8, 누적 · bias 는 INT32, 마지막에 scale 한 번 (또는 M 으로 requantize).
- 비대칭 zp 는 보정 항 zp × (W 행 합) 을 만든다 → weight 는 대칭.
- ⚠ `int8 @ int8` 은 int8 로 넘친다 — 먼저 넓힌다.

---

### 8교시 · 미니 프로젝트: 손으로 만드는 INT8 SimpleConvNet · 채점

**키워드**: `fake quantization`, `my_quant.py`, `weight-only`, `6가지 조합`, `INT8 크기 = weight + scale + bias`, `손실 1%p 이내`

**개요**: 오늘 만든 함수로 SimpleConvNet 의 weight 3개(conv · fc1 · fc2)를 양자화하고 **비트 수(8 · 4 · 2) × granularity(per-tensor · per-channel)** 6가지 조합의 정확도를 비교한다. 양자화 → 역양자화한 weight 를 다시 PyTorch 로 실행하는 방법을 **fake quantization** 이라 한다(Day 4 QAT 의 바탕). 결과: 8비트는 손실 0, 4비트는 정확도는 그대로지만 logits 가 1.3 ~ 1.6 흔들림, 2비트는 무너짐(per-tensor 0.25 · per-channel 0.52). 과제 6개(100점) 중 1 ~ 5 는 채점 셀이 자동 채점, 6 은 강사 채점 · 정답 보기 셀 없음.

**핵심 용어**
- **fake quantization(가짜 양자화)**: 양자화 → 역양자화한 weight 로 기존 프레임워크(PyTorch)에서 실행 — 정확도 영향을 빠르게 확인
- **weight-only 양자화**: weight 만 양자화하고 활성값 · 계산은 실수로 두는 것
- **예측 일치율(agreement)**: FP32 모델과 같은 클래스를 예측한 비율
- **max_diff**: FP32 logits 와의 최대 절대 차이 — 정확도보다 먼저 흔들림을 보여 줌

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `fake_quant_weight()` | `fake_quant_weight(W, bits=8, per_channel=False)` | 대칭 양자화 → 역양자화한 weight (dtype 은 입력과 같게) |
| `evaluate()` | `evaluate(bits, per_channel)` | 모델 복사 → weight 3개 교체 → acc · agree · max_diff 사전 |
| `int8_bytes()` | `int8_bytes(per_channel)` | INT8 파일 크기 = weight 1 B + scale 4 B + bias 4 B |
| `copy.deepcopy()` | `m = copy.deepcopy(cnn)` | 원래 모델을 건드리지 않고 복사본만 바꾸기 |
| `importlib.reload()` | `importlib.reload(my_quant)` | 고쳐 저장한 `my_quant.py` 를 다시 읽기 |

**핵심 내용**

#### fake quantization 과 6가지 조합 결과
> fake quant = weight 를 양자화 → 역양자화해 다시 실수로 넣고 그대로 실행 — 정수 회로 없이 정확도 영향만 확인
- **8비트** → per-tensor · per-channel 모두 acc **1.00**, max_diff 0.067
- **4비트** → per-tensor 1.00 (max_diff 1.59) · per-channel 0.99 (max_diff 1.32)
- **2비트** → per-tensor **0.25** · per-channel **0.52** — 서로 다른 값이 3개(−1 · 0 · 1 × scale)뿐
- **비트가 적을수록** → per-channel 의 가치가 커진다 (8비트 차이 없음 → 2비트 0.25 vs 0.52)
- ⚠ 4비트 "정확도 그대로" 를 100장만 보고 믿지 않는다 — max_diff 가 1 이 넘으면 다른 데이터에서 예측이 뒤집힐 수 있다

#### 실제 INT8 파일 크기
> 크기 = weight(1 B × 개수) + scale(FP32 4 B × 층 또는 채널 수) + bias(INT32 4 B × 개수)
- **per-tensor** → 433,750 + 3 × 4 + 140 × 4 = **434,322 B** → FP32 1,735,560 B 의 1/**3.996**
- **per-channel** → scale 이 30 + 100 + 10 = 140 개 → **434,870 B** → 1/**3.991** (늘어난 크기 548 B)
- **왜 정확히 4 가 아닌가** → bias 가 INT32(4 B) 그대로 + scale 이 추가
- ⚠ 오늘은 weight 만 — 진짜 INT8 추론(7교시)을 하려면 **활성값**도 양자화해야 하고, 그 범위는 calibration 으로 (Day 2)

**기본 사용형태**
```python
m = copy.deepcopy(cnn)                                       # 원본 보존
for layer in (m.conv, m.fc1, m.fc2):
    W = layer.weight.detach().numpy()
    layer.weight.data = torch.from_numpy(fake_quant_weight(W, bits, per_channel))
with torch.no_grad():
    logits = m(X).numpy()
acc = (logits.argmax(1) == labels).mean()
```

**연결**
- [OT D1·8교시 미니 프로젝트](#/ot/1/8) — 같은 SimpleConvNet 을 혼자 4단계로 내보냈던 과제
- [OT D4·8교시 실행기 4가지 성능표](#/ot/4/8) — 크기 · 속도 · 정확도를 한 표에 놓고 고르는 방식 그대로
- [PY D3·4교시 JSON 과 CSV](#/py/3/4) — 결과표를 pandas DataFrame · CSV 로

**상세 학습**

#### 이 교시에서 답하는 질문
1 ~ 7교시에서 양자화의 부품(scale · zp · 대칭 · per-channel · clip · 정수 곱)을 하나씩 만들었다. 이제 이것을 **모델 전체**에 적용하면 정확도는 얼마나 떨어지나? 비트를 어디까지 줄일 수 있고, per-channel 은 언제 꼭 필요한가? 그리고 파일은 정말 1/4 이 되나? 이 교시는 6가지 설정을 한 표에 놓고 배포 설정을 고른다.

#### 개념 1 · 과제 구성 (100점)
| 과제 | 내용 | 배점 |
|---|---|---|
| 1 | `my_quant.py` — 오늘의 함수 4개(calc_scale_zp · quantize · dequantize · sym_scale)를 `%%writefile` 로 파일에 | 20 |
| 2 | `fake_quant_weight(W, bits, per_channel)` | 20 |
| 3 | 6가지 조합 평가 `evaluate` | 20 |
| 4 | INT8 모델 크기 `int8_bytes` (weight + scale + bias) | 20 |
| 5 | 결과표 · 배포 설정 고르기 (손실 ≤ 1%p 중 비트 최소 → max_diff 최소) | 10 |
| 6 | 결과 해석 (마크다운, 강사 채점) | 10 |

`my_quant.py` 는 Day 2 에서 ONNX Runtime 결과와 숫자를 비교할 때 `import my_quant` 로 다시 쓴다. 제출은 채점 셀 결과가 보이는 상태로 노트북 공유.

#### 개념 2 · 결과표
![6가지 조합의 정확도 막대와 max_diff](assets/fig/qp/d1-results.svg)

| bits | granularity | acc | agree | max_diff | conv weight 서로 다른 값 |
|---|---|---|---|---|---|
| 8 | per-tensor | 1.00 | 1.00 | 0.067 | 195 |
| 8 | per-channel | 1.00 | 1.00 | 0.067 | |
| 4 | per-tensor | 1.00 | 1.00 | 1.585 | 14 |
| 4 | per-channel | 0.99 | 0.99 | 1.322 | |
| 2 | per-tensor | 0.25 | 0.25 | 15.61 | 3 |
| 2 | per-channel | 0.52 | 0.52 | 13.39 | |

conv weight MSE (per-tensor → per-channel): 8비트 1.12e-06 → 5.86e-07 · 4비트 3.99e-04 → 1.90e-04 · 2비트 1.98e-02 → 9.38e-03.

#### 개념 3 · 과제 6 의 생각할 거리
- 8비트는 per-tensor 도 칸이 촘촘해 차이가 없지만, 2비트는 칸이 3개뿐이라 scale 하나가 큰 필터에 맞춰지면 작은 필터의 값이 거의 전부 0 이 된다.
- 4비트의 max_diff 1.3 ~ 1.6 은 logits 가 크게 흔들린다는 뜻 — 100장은 표본이 작다. 더 많은 데이터 · 예측 일치율로 확인해야 한다.
- 압축률 3.996 · 3.991 — bias INT32 + scale 때문. per-channel 로 늘어나는 크기는 scale 137개 × 4 B = 548 B.
- 진짜 INT8 추론에는 활성값 양자화가 더 필요 → 범위는 보정 데이터로(6교시 · Day 2).
- 🔎 4비트 LLM 이 되는 비결: group-wise 양자화 — 채널 하나를 다시 32 · 64 · 128 개씩 묶어 묶음마다 scale.

#### 한 번에 정리
- fake quantization = 양자화 → 역양자화한 weight 로 기존 프레임워크에서 실행.
- 8비트 weight-only 는 손실 0, 비트를 줄일수록 per-channel 이 중요.
- 크기 = weight + scale + bias → 정확히 1/4 은 아님 (3.99배).
- ⚠ 정확도 · 출력 차이 · 크기를 한 표에 놓고 결정 — 정확도 하나로 고르지 않는다.
