## Day 2 — ONNX Runtime PTQ: 학습 없이 INT8 로

### 1교시 · PTQ 개관: ResNet-20 을 ONNX 로, 평가 함수 만들기

**키워드**: `PTQ`, `dynamic vs static`, `보정 데이터`, `ResNet-20`, `rn20.onnx`, `ort_acc()`, `기준 정확도 0.846`

**개요**: Day 1 은 NumPy 로 양자화를 손으로 했다. 오늘은 ONNX Runtime 의 양자화 도구로 **학습 없이(PTQ)** 모델 전체를 INT8 로 바꾼다. 재료는 VB Day 2 에서 학습한 CIFAR-10 **ResNet-20**(파라미터 272,474개) — 보정용 500장 · 평가용 2,000장(클래스마다 200장). PTQ 는 두 가지: 활성값 scale 을 **실행할 때마다** 재는 dynamic, **보정 데이터로 미리** 정하는 static. 먼저 기준을 잡는다: PyTorch 정확도 = ONNX 정확도 = **0.846**. 이 과목 6일: Day 1 양자화의 수학 → **Day 2 ORT PTQ** → Day 3 PTQ 디버깅 · TensorRT INT8 → Day 4 QAT → Day 5 Pruning → Day 6 통합 경량화.

**한눈에 보기**
PTQ — 학습 없이 INT8 로
> 학습이 끝난 FP32 모델을 다시 학습하지 않고(보정 데이터만으로) INT8 로 바꾸는 사후 양자화 — 먼저 같은 평가 함수로 FP32 기준을 잡는다
- **dynamic — 모델만 있으면 된다**: weight 만 미리 INT8, 활성값 scale 은 실행 중 계산 (LSTM · Transformer 의 Linear 에 알맞음)
- **static — 보정 데이터로 활성값 범위까지 미리**: Conv 위주 CNN 에 알맞음 — `quantize_static` + `CalibrationDataReader`
- **기준 — ONNX FP32 정확도 0.846**: PyTorch 와 같은 값, 출력 최대 차이 3.1e-06, 파일 1,071 KB
- **평가 2,000장이라 0.001 = 2장**: 작은 차이는 장수로 생각한다
- **⚠ 보정 데이터에 정답 라벨은 필요 없다 — 범위만 잰다**: export 때 `dynamic_axes` 를 빼면 배치 250 평가가 실패

**핵심 용어**
- **PTQ(Post-Training Quantization, 학습 후 양자화)**: 다 학습한 모델을 다시 학습하지 않고 INT8 로 바꾸는 것
- **Dynamic 양자화**: weight 만 미리 INT8, 활성값 scale 은 실행할 때마다 그 입력의 min · max 로 계산 — 보정 데이터 불필요
- **Static 양자화**: 보정 데이터를 흘려 활성값 범위를 미리 재고 scale · zero_point 를 파일에 저장 — CNN 의 표준
- **보정 데이터(calibration data)**: 활성값 범위를 재려고 모델에 흘리는 대표 입력 (수백 장, 정답 라벨 불필요)
- **기준 정확도(baseline)**: 양자화 전 FP32 정확도 — 모든 INT8 모델을 이것과 비교

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.onnx.export()` | `torch.onnx.export(model, torch.zeros(1, 3, 32, 32), "rn20.onnx", input_names=["input"], output_names=["logits"], dynamic_axes=…, opset_version=17)` | ResNet-20 을 배치 가변 ONNX 로 |
| `ort_acc()` | `ort_acc(path, X=None, y=None, bs=250)` | ONNX 파일 정확도 — 오늘 모든 모델을 이 함수로 (이 교시에서 만듦) |
| `size_kb()` | `size_kb("rn20.onnx")` | 파일 크기 KB (환경 셀 제공) |

**핵심 내용**

#### PTQ — 학습 없이 INT8 로
> 학습이 끝난 FP32 모델 → (보정 데이터만으로) INT8 모델. 다시 학습하지 않는다 (다시 학습하는 것은 Day 4 QAT)
- **dynamic** → weight 미리 INT8 · 활성값 scale 은 실행 중 계산 · 준비물 = 모델만 · LSTM · Transformer 의 Linear 에 알맞음
- **static** → weight · 활성값 scale 모두 미리 · 준비물 = 모델 + 보정 데이터 · Conv 위주 CNN 에 알맞음
- **ORT 함수** → `quantize_dynamic` / `quantize_static` + `CalibrationDataReader`
- ⚠ 보정 데이터에는 정답 라벨이 필요 없다 — 범위만 잰다

#### 기준을 먼저 잡는다 — ort_acc()
> 같은 평가 데이터 · 같은 함수로 FP32 와 INT8 을 재야 비교가 된다
- **PyTorch FP32** → 평가 2,000장 정확도 **0.846** (`torch_acc`)
- **ONNX FP32** → `rn20.onnx` 1,071 KB · PyTorch 출력과 최대 차이 3.1e-06 · 정확도 **0.846** (같음)
- **1장 = 0.05%p** → 평가 2,000장이라 정확도 0.001 차이 = 2장
- ⚠ `dynamic_axes` 를 빼면 배치 1 로 고정 — `ort_acc` 의 배치 250 실행이 실패한다

**기본 사용형태**
```python
torch.onnx.export(model, torch.zeros(1, 3, 32, 32), "rn20.onnx",
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}}, opset_version=17, **EXPORT_KW)

def ort_acc(path, X=None, y=None, bs=250):
    if X is None: X, y = X_ev, y_ev
    s = ort.InferenceSession(path, providers=CPU)
    pred = np.concatenate([s.run(None, {"input": X[i:i + bs]})[0].argmax(1) for i in range(0, len(X), bs)])
    return float((pred == y).mean())
```

**연결**
- [VB D2·6교시 ResNet: Residual block](#/vb/2/6) — 오늘의 모델 ResNet-20 을 처음 만든 곳
- [OT D2·5교시 dynamic_axes](#/ot/2/5) — 배치 크기를 가변으로 내보내는 방법
- [QP D1·8교시 미니 프로젝트](#/qp/1/8) — 어제는 weight 만 fake quant, 오늘은 활성값까지 진짜 INT8

**상세 학습**

#### 이 교시에서 답하는 질문
Day 1 은 양자화 식(scale · zp · per-channel · clipping)을 NumPy 로 하나씩 만들었지만, 실제 INT8 모델 파일은 만들지 않았다. 활성값의 범위는 "보정 데이터로 정한다" 고만 했다. 오늘은 ONNX Runtime 이 이 일을 자동으로 하게 한다. 그 전에 **무엇과 비교할지**(기준 정확도)와 **어떻게 잴지**(`ort_acc`)를 먼저 고정한다.

#### 개념 1 · PTQ 의 두 갈래
![dynamic 은 실행 중에, static 은 보정 데이터로 미리 scale 을 정한다](assets/fig/qp/d2-ptq.svg)

| | Dynamic | Static |
|---|---|---|
| weight | 미리 INT8 | 미리 INT8 |
| 활성값 scale | 실행할 때마다 그 입력의 min · max | 보정 데이터로 미리 정해 파일에 |
| 준비물 | 모델만 | 모델 + 보정 데이터 수백 장 |
| 잘 맞는 모델 | MatMul · Linear 위주 (LSTM · Transformer) | Conv 위주 (CNN) |
| ORT 함수 | `quantize_dynamic` | `quantize_static` + `CalibrationDataReader` |

#### 개념 2 · 오늘의 데이터와 모델
- ResNet-20: stem(Conv 16) → layer1(16) × 3블록 → layer2(32) × 3 → layer3(64) × 3 → 평균 풀링 → fc(10). Conv 21개 · BN 21개.
- 전처리: uint8 (N, 32, 32, 3) → /255 → (x − MEAN) / STD → (N, 3, 32, 32) — VB Day 2 와 같다. 보정 · 평가 · 배포가 모두 **같은 `preprocess()`** 를 써야 한다.
- 평가용 2,000장 정확도 0.846 은 VB Day 2 의 전체 테스트 정확도와 이미지가 달라 조금 다를 수 있다.

<<CODE 1-c1 | ResNet-20 정의 · CIFAR-10 데이터 준비>>
<<CODE 1-c2 | PyTorch 기준 정확도 0.846>>
<<CODE 1-1 | rn20.onnx 로 내보내기 (배치 축 열기) · 출력 비교>>
<<CODE 1-2 | ort_acc — ONNX 모델 정확도 함수>>

#### 한 번에 정리
- PTQ = 학습 없이 양자화. dynamic(실행 중 scale) vs static(보정 데이터로 미리 scale).
- 기준: PyTorch = ONNX = 0.846 (평가 2,000장, 1장 = 0.05%p).
- `ort_acc()` 는 오늘 모든 모델을 같은 조건으로 재는 도구.
- ⚠ 배치 가변(`dynamic_axes`)으로 내보내야 배치 단위 평가 · 처리량 측정이 된다.

---

### 2교시 · Dynamic 양자화: 한 줄로 INT8, 그런데 왜 느릴까

**키워드**: `quantize_dynamic`, `QuantType.QInt8`, `크기 1/3.5`, `bench()`, `warm-up · 평균 · p95`, `ConvInteger`, `노드 52 → 181`, `9배 느림`

**개요**: `quantize_dynamic` 한 줄이면 weight 가 INT8 이 되어 파일이 1,071 → 310 KB(1/3.5)로 줄고 정확도는 0.846 → 0.8445 로 거의 그대로다. 그런데 속도를 재 보면 CPU 배치 1 에서 **FP32 보다 9배 느리다**. 노드 개수를 세 보면 이유가 보인다 — Conv 하나가 실행할 때마다 "범위 재기 → 양자화 → 정수 Conv → Cast → scale 곱" 5단계가 되어 노드가 52 → 181개로 늘었다. 결론: CNN 은 static 으로(4교시). 속도는 OT Day 3 의 `bench()`(warm-up 후 평균 · p95)로 잰다.

**한눈에 보기**
Dynamic 양자화 — 한 줄로 INT8, 그런데 느리다
> weight 만 미리 INT8 로 두고 활성값은 실행할 때마다 그 입력의 min · max 로 양자화하는 가장 쉬운 PTQ — CNN 에서는 실행마다 붙는 단계 때문에 오히려 느려질 수 있다
- **크기 1,071 → 310 KB, 정확도 0.846 → 0.8445**: −0.15%p = 3장
- **노드 52 → 181**: Conv 마다 범위 재기 → DynamicQuantizeLinear → ConvInteger → Cast → Mul 5단계
- **배치 1 약 9배 느림**: FP32 0.68 ms vs dynamic 6.08 ms · 배치 32 도 0.10배
- **속도는 bench() 로 — warm-up 후 평균 · p95 · fps**: 스레드 1 고정, fps = 1000 / mean × 배치
- **⚠ "INT8 이면 무조건 빠르다" 가 아니다 — 측정이 먼저**: 크기가 1/4 이 아닌 이유는 bias · scale · 그래프 정보

**핵심 용어**
- **ConvInteger**: 정수 입력 · 정수 weight 로 Conv 를 하고 INT32 를 내는 ONNX 연산자 (scale 은 따로 곱해야 함)
- **DynamicQuantizeLinear**: 실행 중에 입력의 min · max 로 scale · zp 를 계산해 UINT8 로 바꾸는 연산자
- **warm-up**: 측정 전에 몇 번 먼저 실행해 버리는 것 — 첫 실행의 메모리 할당 · 최적화 시간을 빼려고
- **p95**: 측정값을 줄 세웠을 때 95% 지점 — 가끔 느린 실행까지 반영한 지연 시간
- **처리량(fps)**: 초당 처리 장수 = 1000 / 평균 ms × 배치 크기

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `quantize_dynamic()` | `quantize_dynamic("rn20.onnx", "rn20_dyn.onnx", weight_type=QuantType.QInt8)` | weight INT8 + 활성값 실행 중 양자화 |
| `QuantType` | `QuantType.QInt8` · `QuantType.QUInt8` | 양자화 자료형 (S8 · U8) |
| `bench()` | `bench(sess, x, n=100, warm=10)` | warm-up 후 n 번 → `{"mean", "p95", "fps"}` (이 교시에서 만듦) |
| `make_session()` | `make_session(path, threads=1)` | 스레드 1개로 고정한 세션 (제공) |
| `op_count()` | `op_count(path)` | 노드 종류별 개수 `Counter` (예제 2-A 제공) |

**핵심 내용**

#### Dynamic 양자화
> weight 만 미리 INT8, 활성값은 실행할 때마다 그 입력의 min · max 로 양자화 — 보정 데이터가 필요 없는 가장 쉬운 PTQ
- **크기** → 1,071 KB → **310 KB** (1/3.5)
- **정확도** → 0.846 → **0.8445** (−0.15%p = 3장)
- **잘 맞는 곳** → LSTM · Transformer 의 MatMul · Linear (계산량 대비 노드가 적다)
- ⚠ 크기 1/4 이 아닌 이유 — bias · scale 은 FP32/INT32 로 남고 그래프 정보도 있다

#### 왜 느린가 — 노드 52 → 181
> dynamic Conv = 범위 재기 → DynamicQuantizeLinear → ConvInteger → Cast → Mul, 실행할 때마다 5단계
- **노드 수** → FP32 52개(Conv 21 · Relu 19 · Add 9 …) → dynamic **181개**(Mul 44 · Add 31 · Cast 22 · Reshape 21 · ConvInteger 21 · DynamicQuantizeLinear 20)
- **배치 1** → FP32 0.68 ms vs dynamic 6.08 ms → **약 9배 느림** (이 컴퓨터 · 스레드 1)
- **배치 32** → FP32 1,512 장/초 vs dynamic 158 장/초 → 배치를 키워도 0.10배
- ⚠ "INT8 이면 무조건 빠르다" ✗ — 측정이 먼저

#### 속도 재기 — bench()
> warm 번 버리고 n 번 재서 평균 · p95 · fps
- **warm-up** → 첫 실행은 메모리 할당 · 최적화가 섞여 수십 배 느릴 수 있다
- **fps** → 1000 / mean × 배치 크기 (배치 4 면 × 4)
- **공정한 비교** → `make_session` 으로 스레드 1개 고정
- ⚠ 한 번만 재면 첫 실행 준비 시간이 섞인다

**기본 사용형태**
```python
with quiet():
    quantize_dynamic("rn20.onnx", "rn20_dyn.onnx", weight_type=QuantType.QInt8)

def bench(sess, x, n=100, warm=10):
    name = sess.get_inputs()[0].name
    for _ in range(warm): sess.run(None, {name: x})
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); sess.run(None, {name: x}); t.append((time.perf_counter() - t0) * 1000)
    m = float(np.mean(t))
    return {"mean": m, "p95": float(np.percentile(t, 95)), "fps": 1000 / m * x.shape[0]}
```

**연결**
- [OT D3·2교시 시간을 재는 법](#/ot/3/2) — `bench()` 를 처음 만든 곳 (warm-up · 평균 · p95 · FPS)
- [OT D3·3교시 스레드 수](#/ot/3/3) — `SessionOptions.intra_op_num_threads` 로 스레드 고정
- [OT D3·7교시 profiling](#/ot/3/7) — 노드 종류별로 어디가 느린지 보는 방법

**상세 학습**

#### 이 교시에서 답하는 질문
1교시에서 PTQ 두 갈래를 봤다. 가장 쉬운 dynamic 은 정말 "한 줄로 작고 빠른" 모델을 줄까? 크기와 정확도는 기대대로지만, 속도를 재 보면 반대 결과가 나온다. 이 교시는 **측정이 왜 먼저인지**를 보여 주고, 그 원인을 그래프의 노드 수로 설명한다.

#### 개념 1 · dynamic Conv 의 5단계
![dynamic 모델의 Conv 하나는 실행할 때마다 5단계 — 노드 52 → 181](assets/fig/qp/d2-dynamic.svg)

- FP32 Conv 는 노드 1개. dynamic 은 입력을 그때그때 양자화하고(DynamicQuantizeLinear), 정수 Conv(ConvInteger) 결과 INT32 를 실수로 바꾸고(Cast), 두 scale 의 곱을 곱한다(Mul).
- 노드 하나마다 실행 준비 비용이 있다. ResNet-20 처럼 Conv 하나의 계산량이 작은 모델은 그 비용이 계산 절약보다 크다.
- LSTM · Transformer 의 큰 MatMul 은 계산량이 커서 절약이 비용을 이긴다 → dynamic 이 맞는 곳.

<<CODE 2-1 | quantize_dynamic 한 줄 — 크기 1/3 · 정확도>>
<<CODE 2-c1 | op_count — Conv 가 ConvInteger + 변환 노드로>>

#### 개념 2 · 지연 시간과 처리량
| 모델 | 배치 1 mean | 배치 1 p95 | 배치 1 fps | 배치 32 fps |
|---|---|---|---|---|
| FP32 | 0.676 ms | 0.71 ms | 1,480 | 1,512 |
| dynamic | 6.077 ms | 9.09 ms | 165 | 158 |

지연 시간은 컴퓨터 · 실행할 때마다 달라진다 — 비율(약 9배)을 기억한다.

<<CODE 2-2 | bench — warm-up · 평균 · p95 · FPS>>
<<CODE 2-c2 | 배치 크기를 바꿔 dynamic 이 왜 느린지 보기>>

#### 한 번에 정리
- `quantize_dynamic` = weight INT8 + 활성값 실행 중 양자화. 보정 데이터 불필요.
- 크기 1/3.5 · 정확도 거의 그대로 · 그러나 CNN 에서는 노드 3.5배 → 약 9배 느림.
- `bench()` · `make_session()` · `op_count()` 는 오늘 계속 쓴다.
- ⚠ INT8 = 빠름이 아니다. 측정은 warm-up 후 여러 번.

---

### 3교시 · 양자화 전처리: BatchNorm folding · quant_pre_process

**키워드**: `BatchNorm folding`, `W' = W·γ/√(σ²+ε)`, `b' = β − μ·γ/√(σ²+ε)`, `fold_bn()`, `BatchNormalization 0개`, `quant_pre_process`, `value_info 0 → 52`

**개요**: 양자화 전에 그래프를 정리해 두면 정확도와 속도가 모두 좋아진다. 가장 중요한 정리는 **BatchNorm folding** — BN 의 채널별 곱셈 · 덧셈을 바로 앞 Conv 의 weight · bias 에 미리 합쳐, Conv → BN 두 단계를 Conv 하나로 만든다(양자화할 텐서가 줄고 오차도 준다). 직접 `fold_bn()` 을 만들어 보면 원래 출력과 차이 9.5e-07. 그런데 PyTorch exporter 는 eval 모드에서 이미 BN 을 합쳐 내보낸다(ONNX 에 BatchNormalization 0개). 남은 정리는 `quant_pre_process` — 중간 텐서 모양 정보(value_info)를 0 → 52개로 채워 준다. 양자화 입력은 항상 `*_pre.onnx`.

**한눈에 보기**
양자화 전처리 — BatchNorm folding · quant_pre_process
> BN 을 앞 Conv 의 weight 와 bias 에 접어 넣으면(W' = W × k, b' = β − μ × k) 양자화할 텐서가 줄고 오차도 한 번 덜 생긴다 — 그 뒤 모양 정보를 채운 파일을 양자화 입력으로 쓴다
- **k = γ / √(σ² + ε), 채널마다**: (16,) 을 `k[:, None, None, None]` 으로 weight (16, 3, 3, 3) 에 맞춘다
- **접은 결과가 같은지 확인**: Conv+BN 출력과 `F.conv2d(x, W', b')` 의 최대 차이 9.5e-07
- **PyTorch exporter 는 eval 모드에서 이미 접는다**: BN 모듈 21개 → ONNX BatchNormalization 0개
- **quant_pre_process — value_info 0 → 52**: 노드 수 · 정확도는 그대로, 양자화 입력은 항상 `rn20_pre.onnx`
- **⚠ eval 모드 전용**: train 모드로 export 하면 BN 이 남는다 — 접을 때는 running 평균 · 분산

**핵심 용어**
- **BatchNorm folding(BN 접기)**: BN 의 γ · β · μ · σ² 를 앞 Conv 의 W · b 에 미리 합치는 것 — 추론 전용(eval)
- **value_info**: ONNX 그래프의 중간 텐서마다 붙은 모양 · 자료형 정보
- **quant_pre_process**: 양자화 전 그래프 정리(모양 추론 · 단순화) — 양자화 도구가 텐서 모양을 알 수 있게
- **initializer**: ONNX 파일 안에 저장된 상수 텐서(weight · bias · scale · zp)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `fold_bn()` | `Wf, bf = fold_bn(model.stem[0], model.stem[1])` | Conv + BN → (W', b') NumPy float32 (이 교시에서 만듦) |
| `quant_pre_process()` | `quant_pre_process("rn20.onnx", "rn20_pre.onnx")` | 양자화 전처리 — 모양 정보 채우기 |
| `graph.value_info` | `len(onnx.load(p).graph.value_info)` | 중간 텐서 모양 정보 목록 |
| `F.conv2d()` | `F.conv2d(x, W, b, padding=1)` | 함수형 Conv — folding 결과 검증 |

**핵심 내용**

#### BatchNorm folding
> BN(Wx) = γ(Wx − μ)/√(σ²+ε) + β = **W'x + b'** — W' = W × k, b' = β − μ × k, k = γ/√(σ²+ε) (채널마다)
- **왜** → 양자화할 텐서가 줄고(BN 출력이 없어짐) 반올림 오차도 한 번 덜 생긴다 · 연산도 1단계 줄어 빠름
- **k 의 모양** → (16,) → weight (16, 3, 3, 3) 에 맞춰 `k[:, None, None, None]`
- **검증** → stem 의 Conv+BN 출력과 `F.conv2d(x, W', b')` 의 최대 차이 **9.5e-07**
- ⚠ eval 모드 전용 — 학습 중 BN 은 배치마다 μ · σ² 가 바뀐다 (running 값으로 접는다)

#### 이미 접혀 있다 · quant_pre_process
> PyTorch exporter 는 eval 모드에서 BN 을 Conv 에 자동으로 합친다 → 남은 일은 모양 정보 채우기
- **확인** → PyTorch BN 모듈 21개 → ONNX BatchNormalization 노드 **0개** · ONNX 첫 Conv weight 와 직접 만든 W' 차이 **0.0**
- **quant_pre_process** → value_info **0 → 52** · 노드 52 → 52 · 정확도 0.846 그대로
- **규칙** → 양자화 입력은 항상 `rn20_pre.onnx`
- ⚠ train 모드로 export 하면 BN 이 남는다 — 내보내기 전 `model.eval()`

**기본 사용형태**
```python
def fold_bn(conv, bn):
    k = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
    W = conv.weight.detach().numpy() * k[:, None, None, None]
    b = bn.bias.detach().numpy() - bn.running_mean.detach().numpy() * k
    if conv.bias is not None: b = b + conv.bias.detach().numpy() * k
    return W.astype(np.float32), b.astype(np.float32)

quant_pre_process("rn20.onnx", "rn20_pre.onnx")
```

**연결**
- [VB D2·5교시 깊이의 문제 · BatchNorm](#/vb/2/5) — ResNet 의 BN 이 하는 일
- [NN D4·4교시 Batch Normalization](#/nn/4/4) — γ · β · running mean/var 를 NumPy 로 만든 곳
- [OT D2·4교시 opset · checker · shape_inference](#/ot/2/4) — 모양 정보(value_info)를 채우는 shape inference
- [OT D3·4교시 그래프 최적화 레벨](#/ot/3/4) — 노드 합치기(fusion) 의 같은 생각

**상세 학습**

#### 이 교시에서 답하는 질문
static 양자화(4교시)를 하기 전에 그래프를 어떻게 준비해야 할까? Conv 뒤의 BN 을 그대로 두면 Conv 출력과 BN 출력을 각각 양자화해야 하고, 오차도 두 번 생긴다. 이 교시는 BN 을 Conv 에 접는 식을 직접 유도 · 구현하고, 실제로 ONNX 파일이 이미 그렇게 되어 있는지 확인한 뒤, 남은 전처리(`quant_pre_process`)를 한다.

#### 개념 1 · BN 을 Conv 에 접는 식
![Conv → BN 두 단계를 Conv(W', b') 하나로](assets/fig/qp/d2-bnfold.svg)

$$\text{BN}(Wx) = \gamma \frac{Wx - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta = \underbrace{\left(\frac{\gamma}{\sqrt{\sigma^2+\epsilon}} W\right)}_{W'} x + \underbrace{\left(\beta - \frac{\gamma \mu}{\sqrt{\sigma^2+\epsilon}}\right)}_{b'}$$

| 기호 | 뜻 | 코드 |
|---|---|---|
| γ · β | BN 이 학습한 크기 · 이동 | `bn.weight` · `bn.bias` |
| μ · σ² | 학습 중 모은 평균 · 분산 | `bn.running_mean` · `bn.running_var` |
| ε | 0 으로 나누기 방지 | `bn.eps` (1e-5) |
| k | 채널마다 곱할 값 γ/√(σ²+ε) | 모양 (out_ch,) |

<<CODE 3-1 | fold_bn — BN 을 Conv 의 W' · b' 로 접기>>
<<CODE 3-c1 | ONNX 에는 이미 BN 노드가 없다>>

#### 개념 2 · quant_pre_process 가 하는 일
- 양자화 도구는 텐서마다 모양을 알아야 per-channel 축 · 노드 묶기를 정할 수 있다. 내보낸 직후의 `rn20.onnx` 는 value_info 가 0개.
- `quant_pre_process` 는 shape inference + 그래프 단순화를 해서 `rn20_pre.onnx` 로 저장 → value_info 52개.
- 이 모델은 BN 이 이미 접혀 노드 수는 52 그대로. 정확도도 그대로(0.846).

<<CODE 3-2 | quant_pre_process — 중간 shape 정보 채우기>>

#### 한 번에 정리
- BN folding: W' = W·γ/√(σ²+ε), b' = β − μ·γ/√(σ²+ε).
- 양자화할 텐서가 줄고 정확도 · 속도 모두 좋아진다.
- PyTorch exporter 는 eval 모드에서 이미 접는다 → ONNX 에 BatchNormalization 0개.
- ⚠ 양자화 전에는 항상 `quant_pre_process` → `*_pre.onnx` 를 입력으로.

---

### 4교시 · Static 양자화: CalibrationDataReader · scale 검산

**키워드**: `quantize_static`, `CalibrationDataReader`, `get_next() · rewind()`, `QDQ · U8/S8 · per-channel`, `정확도 0.846 → 0.845`, `scale 0.016141 · zp 123`, `어두운 보정 → 0.684`

**개요**: static 양자화는 **보정 데이터**를 모델에 흘려 활성값 텐서마다 범위를 재고, 그 범위로 scale · zero_point 를 미리 정해 파일에 넣는다. 보정 데이터는 `CalibrationDataReader` 를 상속한 클래스(`get_next()` 가 `{"input": 배치}` 를 하나씩, 끝나면 `None`)로 건넨다. 결과: 정확도 0.846 → **0.845**(−0.1%p), 크기 1,071 → 322 KB. 파일 속 입력 scale · zp(0.016141 · 123)는 Day 1 의 `my_quant.calc_scale_zp` 로 계산한 값과 **똑같다**. 보정 데이터를 일부러 어둡게 하면 정확도가 **0.684** 로 무너진다 — 보정 데이터는 "많이" 보다 "현장 입력과 비슷하게".

**한눈에 보기**
Static 양자화 — 보정 데이터로 범위를 미리 고정
> 보정 데이터로 활성값 텐서마다 범위를 미리 재서 scale · zp 를 파일에 고정한다 — 실행 중에는 계산하지 않아 빠르다
- **정확도 0.846 → 0.845 (−2장), 크기 322 KB**: QDQ · 활성값 U8 · weight S8 · per-channel
- **CalibrationDataReader — `get_next()` 가 배치를 하나씩, 끝나면 None**: 보정 500장 ÷ 50 = 10 배치
- **보정은 수백 장이면 충분**: 10장 0.8445 · 500장 0.845
- **파일 속 scale 은 Day 1 식 그대로**: 입력 범위 −1.9895 ~ 2.1265 → scale 0.016141, zp 123
- **어두운 사진으로 보정하면 0.845 → 0.684**: 범위가 좁게 잡혀 실제 밝은 값이 모두 잘린다 — 낮 · 밤을 섞어 보정
- **⚠ reader 는 quantize_static 마다 새로 만든다**: None 을 안 돌려주면 끝나지 않음 · 보정 · 평가 · 배포 전처리는 같게

**핵심 용어**
- **CalibrationDataReader**: ORT 가 보정 데이터를 받는 틀 — `get_next()` (다음 배치 사전, 끝나면 None) · `rewind()` (처음으로)
- **MinMax 보정**: 보정 데이터에서 본 최솟값 ~ 최댓값을 범위로 (기본값)
- **QuantizeLinear (Q)**: 실수 → 정수 노드, 입력 = (x, scale, zero_point)
- **분포 불일치(distribution shift)**: 보정 데이터와 실제 입력의 값 분포가 다른 것 — 범위가 엉뚱하게 잡혀 clipping 이 생긴다

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `CalibrationDataReader` | `class CifarReader(CalibrationDataReader): …` | 보정 데이터 건네는 틀 (상속) |
| `CifarReader()` | `CifarReader(X_cal, bs=50)` | 50장 × 10 배치를 차례로 (이 교시에서 만듦) |
| `quantize_static()` | `quantize_static("rn20_pre.onnx", "rn20_int8.onnx", CifarReader(X_cal), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)` | 보정 → 활성값 scale 고정한 INT8 |
| `QuantFormat` | `QuantFormat.QDQ` · `QuantFormat.QOperator` | 그래프에 담는 방식 (5교시) |
| `my_quant.calc_scale_zp()` | `calc_scale_zp(X_cal.min(), X_cal.max(), 0, 255)` | Day 1 함수로 ORT 의 scale · zp 검산 |

**핵심 내용**

#### Static 양자화
> 보정 데이터로 활성값 텐서마다 범위를 미리 재서 scale · zp 를 파일에 고정 — 실행 중에는 계산하지 않는다
- **설정** → QDQ 형식 · 활성값 U8 · weight S8 · per-channel
- **결과** → 정확도 0.846 → **0.845** (−0.10%p = 2장) · 크기 1,071 → **322 KB** (약 1/3.3) · QuantizeLinear 34개
- **보정 장수** → 10장 0.8445 · 500장 0.845 — 이 모델은 수백 장이면 충분 (강의 실측 50장 0.843 · 100장 0.845)
- ⚠ 보정 · 평가 · 배포 모두 같은 `preprocess()` — 정규화 빠짐 · RGB/BGR 뒤바뀜이면 범위가 엉뚱하게

#### CalibrationDataReader
> `get_next()` 가 `{"입력 이름": 배치}` 를 하나씩 돌려주고, 다 쓰면 `None` — ORT 는 None 이 나올 때까지 부른다
- **구현** → `__init__` 에 X · bs · `self.i = 0` / `get_next` 는 `X[i:i+bs]` 후 i += bs / `rewind` 는 i = 0
- **오늘** → 보정 500장 ÷ 50 = **10 배치**, 배치 모양 (50, 3, 32, 32)
- ⚠ None 을 안 돌려주면 보정이 끝나지 않는다 · 사전 키가 ONNX 입력 이름(`input`)과 다르면 오류
- ⚠ reader 는 한 번 쓰면 끝까지 읽힌 상태 — `quantize_static` 마다 **새로** 만들어 넘긴다

#### 검산 · 보정 데이터의 분포
> 파일 속 scale · zp 는 Day 1 식 그대로: scale = (max − min)/255, zp = round(−min/scale)
- **입력 텐서** → 보정 범위 −1.9895 ~ 2.1265 → ORT scale **0.016141** · zp **123** = `my_quant` 결과와 같음
- **어두운 보정(밝기 × 0.2)** → 정확도 0.845 → **0.684** (−16.2%p)
- **왜** → 어두운 이미지는 정규화 후 값이 음수 쪽에 몰려 범위가 좁게 잡힘 → 실제 밝은 값이 모두 잘림(clipping)
- ⚠ 밤에도 쓰는 카메라라면 낮 · 밤 이미지를 섞어 보정한다

**기본 사용형태**
```python
class CifarReader(CalibrationDataReader):
    def __init__(self, X, bs=50):
        self.X, self.bs, self.i = X, bs, 0
    def get_next(self):
        if self.i >= len(self.X): return None
        b = {"input": self.X[self.i:self.i + self.bs]}; self.i += self.bs; return b
    def rewind(self):
        self.i = 0

quantize_static("rn20_pre.onnx", "rn20_int8.onnx", CifarReader(X_cal),
                quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8, per_channel=True)
```

**연결**
- [QP D1·3교시 Affine 양자화](#/qp/1/3) — scale · zero-point 식, 오늘 ORT 와 같은 답을 낸 `calc_scale_zp`
- [QP D1·6교시 Clipping 과 Outlier](#/qp/1/6) — 어두운 보정의 정확도 붕괴 = 모델 전체에서 본 clipping 오차
- [PY D3·1교시 클래스 기초](#/py/3/1) — `CalibrationDataReader` 상속 · `__init__` · 메서드

**상세 학습**

#### 이 교시에서 답하는 질문
2교시의 dynamic 은 활성값 범위를 실행할 때마다 재느라 느렸다. 범위를 **미리** 재 두면 실행 중에는 계산이 필요 없다. 그럼 범위는 무엇으로 재나? — 보정 데이터. 이 교시는 보정 데이터를 ORT 에 건네는 방법(`CalibrationDataReader`), static 양자화 결과, 그리고 ORT 가 정한 scale 이 Day 1 식과 같은지를 확인한다. 마지막으로 보정 데이터가 실제와 다르면 어떻게 되는지 본다.

#### 개념 1 · 보정의 흐름
1. `quantize_static` 이 모델의 활성값 텐서마다 기록 장치를 단다.
2. reader 의 `get_next()` 를 None 이 나올 때까지 불러 배치를 흘린다 (500장 = 10 배치).
3. 텐서마다 본 값의 min · max(MinMax 보정)로 scale · zp 계산 → QuantizeLinear 노드의 initializer 로 저장.
4. weight 는 보정 없이 바로 per-channel 대칭 INT8.

<<CODE 4-1 | CifarReader — 보정 데이터를 50장씩 내주는 리더>>
<<CODE 4-2 | quantize_static — QDQ · U8/S8 · per-channel>>
<<CODE 4-c1 | 입력 scale · zp 를 Day 1 my_quant 로 검산>>

#### 개념 2 · 보정 데이터의 분포가 정확도를 정한다
| 보정 데이터 | 정확도 | 차이 |
|---|---|---|
| 정상 500장 | 0.845 | −0.10%p |
| 정상 10장 | 0.8445 | −0.15%p |
| 어둡게(× 0.2) 500장 | 0.684 | −16.2%p |

장수보다 **분포**가 중요하다. 어두운 보정은 범위를 좁게 잡아 실제 밝은 입력을 잘라 버린다 (Day 1 6교시의 잘림 오차).

<<CODE 4-c2 | 어두운 사진으로 보정하면 정확도가 무너진다>>
<<CODE 4-c3 | 보정 장수를 바꿔 정확도 보기>>

#### 한 번에 정리
- static = 보정 데이터로 활성값 범위를 미리 재서 scale 고정 — CNN 의 표준.
- `CalibrationDataReader` = `get_next()`(끝나면 None) · `rewind()` 두 함수.
- 파일 속 scale · zp 는 Day 1 식 그대로 — 검산 가능.
- ⚠ 보정 데이터는 현장 입력과 같은 분포로 (장수는 수백 장이면 충분).

---

### 5교시 · QDQ vs QOperator: 그래프 모양 · U8 / S8

**키워드**: `QDQ`, `QOperator`, `QLinearConv`, `노드 145 vs 35`, `INT8 initializer → DequantizeLinear`, `per-channel scale (16,)`, `U8/S8 빠름 · S8/S8 느림`

**개요**: 같은 static INT8 이라도 ONNX 그래프에 담는 방식이 두 가지다. **QDQ** 는 원래 Conv 를 그대로 두고 앞뒤에 QuantizeLinear · DequantizeLinear 를 끼운다(노드 145개, 읽기 쉽고 TensorRT 등의 표준 입력). **QOperator** 는 Conv 자체를 QLinearConv 같은 정수 연산자로 바꾼다(노드 35개, ORT CPU 가 그대로 실행). 정확도는 둘 다 0.845. QDQ 의 weight 는 INT8 initializer → DequantizeLinear → Conv 로 연결되고 scale 이 채널 수(16개)만큼 — Day 1 의 대칭 per-channel 그대로다. 그리고 x86 CPU 는 **활성값 U8 × weight S8** 조합이 빠르다: 활성값까지 S8 로 하면 FP32 보다도 느려졌다.

**한눈에 보기**
QDQ vs QOperator · U8 vs S8 — 파일 형식과 자료형
> QDQ 는 원래 그래프에 Quantize · Dequantize 노드를 끼우고, QOperator 는 Conv 자체를 정수 연산자(QLinearConv)로 바꾼다 — 같은 scale 이라 정확도는 같고, 속도는 활성값 자료형이 가른다
- **QDQ 145노드 vs QOperator 35노드**: QDQ 는 원래 Conv 가 보여 읽기 쉽고 TensorRT 등 다른 런타임의 표준
- **둘 다 정확도 0.845**: 같은 scale · zp
- **weight 는 INT8 initializer + per-channel scale**: q × scale 로 되돌리면 3교시 W' 와 최대 오차 0.00146 ≤ scale/2
- **x86 은 활성값 U8 × weight S8 이 빠르다**: 배치 1 QDQ U8/S8 0.386 ms (FP32 의 1.92배 빠름) · S8/S8 1.310 ms (3.4배 느림)
- **⚠ QDQ 의 Conv 는 FP32 계산이 아니다**: 런타임이 Q-Conv-DQ 를 묶어 정수 연산으로 — U8/S8 결론은 x86 CPU 의 것, 배포 장치에서 다시

**핵심 용어**
- **QDQ 형식**: 연산자 앞뒤에 Q(실수 → 정수) · DQ(정수 → 실수) 노드를 끼운 그래프 — 런타임이 Q-Conv-DQ 를 묶어(fuse) 정수 연산으로 실행
- **QOperator 형식**: QLinearConv · QLinearAdd 처럼 정수 입출력 연산자로 바꾼 그래프
- **DequantizeLinear (DQ)**: 정수 → 실수 노드, (q − zp) × scale
- **QLinearConv**: 입력 · weight · 출력의 scale/zp 를 모두 받아 정수로 Conv 하는 ONNX 연산자
- **U8/S8**: 활성값 부호 없는 8비트(UINT8) × weight 부호 있는 8비트(INT8) — x86 INT8 명령이 최적화된 조합

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `QuantFormat.QOperator` | `quantize_static(…, quant_format=QuantFormat.QOperator, …)` | QLinearConv 형식으로 |
| `QuantType.QInt8` | `activation_type=QuantType.QInt8` | 활성값을 S8 로 (S8/S8 모델) |
| `Counter.most_common()` | `c_qdq.most_common(5)` | 많은 노드 종류부터 5개 |

**핵심 내용**

#### QDQ vs QOperator
> QDQ = 원래 그래프 + Q · DQ 노드 / QOperator = Conv 자체를 정수 연산자(QLinearConv)로
- **QDQ** → 노드 **145개** (DQ 78 · Q 34 · Conv 21 · Add 9) — 원래 Conv 가 보여 읽기 쉬움 · TensorRT 등 다른 런타임의 표준
- **QOperator** → 노드 **35개** (QLinearConv 21 · QLinearAdd 9 · Q 1 …) — ORT CPU 에서 바로 실행
- **정확도** → 둘 다 **0.845** (같은 scale · zp)
- ⚠ QDQ 의 Conv 는 "FP32 로 계산" 이 아니다 — 런타임이 Q-Conv-DQ 를 묶어 정수 연산으로 바꿔 실행

#### weight 는 INT8 initializer + per-channel scale
> 첫 Conv: INT8 weight (16, 3, 3, 3) → DequantizeLinear(scale (16,), zp 0) → Conv
- **찾는 법** → `Conv.input[1]` = weight 를 만드는 DQ 의 출력 → 그 DQ 의 input[0] · input[1] = INT8 weight · scale
- **역양자화** → 대칭(zp = 0)이라 q × scale → 3교시 W' 와 최대 오차 **0.00146** ≤ 채널 scale/2 최대 0.00148
- **크기** → weight 는 1 B, 하지만 bias · scale 은 4 B → 파일 322 KB (1/3.3)
- ⚠ scale 을 `[:, None, None, None]` 로 채널 축에 맞춰야 한다 (Day 1 5교시)

#### 활성값 U8 vs S8
> x86 CPU 의 INT8 명령은 "부호 없는 8비트 × 부호 있는 8비트" 에 최적화 — 활성값 U8 · weight S8 이 빠르다
- **배치 1 (이 컴퓨터)** → FP32 0.742 ms · QDQ U8/S8 **0.386 ms (1.92배 빠름)** · QDQ S8/S8 1.310 ms (FP32 의 0.57배) · QOperator U8/S8 0.487 ms
- **S8/S8** → U8/S8 보다 **3.4배 느림** — 빠른 경로를 못 쓴다
- **정확도** → S8/S8 도 0.845 — 차이는 속도뿐
- ⚠ 이 결론은 x86 CPU 의 것 — ARM · GPU(TensorRT)는 배포 장치에서 다시 잰다

**기본 사용형태**
```python
quantize_static("rn20_pre.onnx", "rn20_qop.onnx", CifarReader(X_cal),
                quant_format=QuantFormat.QOperator, activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8, per_channel=True)
quantize_static("rn20_pre.onnx", "rn20_s8.onnx", CifarReader(X_cal),
                quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,      # S8/S8
                weight_type=QuantType.QInt8, per_channel=True)
c_qdq, c_qop = op_count("rn20_int8.onnx"), op_count("rn20_qop.onnx")
```

**연결**
- [QP D1·5교시 Granularity](#/qp/1/5) — per-channel scale 과 broadcast 모양
- [QP D1·7교시 정수 연산 추론](#/qp/1/7) — QLinearConv 안에서 일어나는 INT8 × INT8 → INT32 → requantize
- [OT D2·3교시 가중치 · 속성 꺼내기](#/ot/2/3) — initializer 를 `numpy_helper.to_array` 로 꺼내는 법
- [OT D4·6교시 FP16 으로 더 빠르게](#/ot/4/6) — 자료형을 바꿀 때 속도 · 정확도를 함께 재는 방식

**상세 학습**

#### 이 교시에서 답하는 질문
4교시에서 만든 `rn20_int8.onnx` 는 QDQ 형식이었다. 그런데 같은 양자화를 다른 모양의 그래프로 담을 수도 있다. 두 형식은 무엇이 다르고 언제 무엇을 쓰나? 그리고 자료형 조합(U8/S8 vs S8/S8)은 정확도 말고 무엇을 바꾸나? 이 교시는 그래프를 열어 노드를 세고, weight 가 실제로 어떻게 저장되어 있는지 꺼내 보고, 속도를 잰다.

#### 개념 1 · 두 형식의 그래프
![같은 Conv 를 두 방식으로 — QDQ 는 Q · DQ 를 끼우고, QOperator 는 QLinearConv 하나로](assets/fig/qp/d2-qdq.svg)

| | QDQ | QOperator |
|---|---|---|
| 모양 | 원래 Conv 앞뒤에 Q · DQ | Conv → QLinearConv |
| 노드 (ResNet-20) | 145개 | 35개 |
| 장점 | 원래 그래프가 보여 읽기 쉬움 · TensorRT 등의 표준 입력 | 노드가 적고 ORT CPU 에서 바로 빠름 |
| 실행 | 런타임이 Q-Conv-DQ 를 묶어 정수 연산으로 | 그대로 실행 |

<<CODE 5-1 | QOperator 형식으로 만들어 노드 구성 비교>>
<<CODE 5-c1 | weight 를 DQ 의 scale 로 되돌려 오차 확인>>

#### 개념 2 · 자료형 조합과 속도
| 모델 | 배치 1 mean | FP32 대비 |
|---|---|---|
| FP32 (`rn20_pre.onnx`) | 0.742 ms | 1.00배 |
| QDQ U8/S8 | 0.386 ms | 1.92배 빠름 |
| QDQ S8/S8 | 1.310 ms | 0.57배 (느려짐) |
| QOperator U8/S8 | 0.487 ms | 1.52배 빠름 |

지연 시간은 잴 때마다 흔들린다 — 7교시 표에서는 QOperator 가 0.342 ms 로 QDQ U8/S8(0.480 ms)보다 빨랐다. 흔들리지 않는 결론은 "U8/S8 ≫ S8/S8".

<<CODE 5-2 | 활성값 S8 모델 · 형식별 배치 1 지연 비교>>

#### 한 번에 정리
- QDQ = 원래 그래프 + Q/DQ (TensorRT 등의 표준 입력), QOperator = QLinearConv 같은 정수 연산자 (ORT CPU 용).
- weight = INT8 initializer + per-channel scale + zp 0 — Day 1 의 대칭 per-channel 그대로.
- x86 CPU 에서는 활성값 U8 · weight S8 이 빠르다 (S8/S8 은 3.4배 느림).
- ⚠ 형식 · 자료형은 배포 장치에서 측정해 고른다.

---

### 6교시 · Calibration 방법 비교: MinMax · Entropy · Percentile

**키워드**: `calibrate_method`, `CalibrationMethod.MinMax · Entropy · Percentile`, `per-channel True / False`, `6개 모델 차이 0.30%p`, `act_scales()`, `Percentile 범위 평균 18% 축소`, `Entropy = MinMax`

**개요**: Day 1 6교시에서 범위를 어디까지 잡을지(clipping) 직접 실험했다. ORT 의 `calibrate_method` 가 그 선택이다: MinMax(본 값 전부) · Percentile(드문 큰 값을 잘라냄, 기본 99.999%) · Entropy(원래 분포와 양자화 분포의 KL 차이가 가장 작은 범위). 3가지 × per-channel 2가지 = 6개 모델을 만들면 정확도는 0.8435 ~ 0.8465 — **차이 0.30%p(6장)** 로 거의 같다. 하지만 실제로 고른 scale 을 열어 보면 Percentile 은 활성값 34개의 범위를 평균 18% 줄였고(가장 많이 잘린 텐서는 67% 로), Entropy 는 이 실행에서 34개 모두 MinMax 와 같은 scale 이었다. 정확도만 보지 말고 **scale 도 열어 본다**.

**한눈에 보기**
Calibration 방법 — MinMax · Percentile · Entropy
> 보정 방법은 Day 1 의 clipping 선택 — 본 값 전체(MinMax), 드문 큰 값 잘라내기(Percentile), 분포 차이 최소(Entropy) 중 어디까지를 범위로 잡을지 정한다
- **MinMax — 기본값, outlier 에 약하다**: 본 값의 최솟값 ~ 최댓값
- **Percentile — 범위를 평균 0.819배로**: 가장 많이 줄인 곳은 Relu 출력 0.665배, 입력 · logits 는 1.0
- **Entropy — 이 실행에서는 MinMax 와 같은 범위**: 34개 scale 비율 모두 1.000 (onnxruntime 1.32)
- **정확도 차이는 0.30%p (6장)**: 양자화에 강한 모델은 방법 차이가 작다
- **⚠ 같은 이름의 방법도 버전 · 설정에 따라 다르게 동작한다**: scale 을 열어 확인

**핵심 용어**
- **calibrate_method**: 보정 데이터에서 활성값 범위를 정하는 방법 (`quantize_static` 의 인자)
- **Percentile 보정**: 값의 아주 큰 쪽 일부(기본 0.001%)를 잘라 범위를 좁히는 방법
- **Entropy 보정**: 원래 값 분포와 양자화한 분포의 차이(KL divergence)가 가장 작은 범위를 고르는 방법
- **KL divergence**: 두 확률 분포가 얼마나 다른지 재는 값 (같으면 0)
- **per-tensor (pt) / per-channel (pc)**: weight scale 을 텐서 하나에 1개 / 출력 채널마다 1개

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `CalibrationMethod` | `calibrate_method=CalibrationMethod.Percentile` | 보정 방법 고르기 (MinMax · Entropy · Percentile) |
| `act_scales()` | `act_scales("rn20_MinMax_pc.onnx")` | QuantizeLinear 마다 `{입력 텐서 이름: scale}` (이 교시에서 만듦) |
| `pd.DataFrame()` | `pd.DataFrame(rows)` | 사전 목록 → 결과표 |

**핵심 내용**

#### 보정 방법 3가지
> calibrate_method = Day 1 의 clipping 선택 — 범위를 어디까지 잡나
- **MinMax** → 본 값의 최솟값 ~ 최댓값 (기본값, outlier 에 약함)
- **Percentile** → 아주 드문 큰 값을 잘라냄 (기본 99.999%)
- **Entropy** → 원래 분포 vs 양자화 분포의 KL 차이가 가장 작은 범위
- ⚠ 6개 모델 정확도 0.8435 ~ 0.8465 — 차이 0.30%p = 2,000장 중 6장, 양자화에 강한 모델은 방법 차이가 작다

#### 실제로 고른 scale 비교
> 정확도가 같아도 고른 범위는 다를 수 있다 — Percentile scale ÷ MinMax scale 로 비교
- **Percentile / MinMax** → 평균 **0.819** · 최소 **0.665** (layer1.2 의 Relu 출력 — 범위를 67% 로)
- **Entropy / MinMax** → 34개 모두 **1.000** — 이 실행(onnxruntime 1.32)에서 Entropy 는 MinMax 와 같은 범위 · 같은 정확도
- **잘리지 않은 텐서** → input · 평균 풀링 · Flatten · logits 의 Q 는 비율 1.0
- ⚠ 같은 이름의 방법이 버전 · 설정에 따라 기대와 다르게 동작할 수 있다 — scale 을 열어 확인

**기본 사용형태**
```python
for meth in [CalibrationMethod.MinMax, CalibrationMethod.Entropy, CalibrationMethod.Percentile]:
    for pc in [False, True]:
        p = f"rn20_{meth.name}_{'pc' if pc else 'pt'}.onnx"
        quantize_static("rn20_pre.onnx", p, CifarReader(X_cal), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        per_channel=pc, calibrate_method=meth)

def act_scales(path):
    g = onnx.load(path).graph
    ini = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    return {n.input[0]: float(ini[n.input[1]]) for n in g.node if n.op_type == "QuantizeLinear"}
```

**연결**
- [QP D1·6교시 Clipping 과 Outlier](#/qp/1/6) — 반올림 오차 vs 잘림 오차, Percentile 을 손으로 만든 곳
- [QP D1·5교시 Granularity](#/qp/1/5) — per-tensor vs per-channel
- [NN D1·7교시 확률 기초](#/nn/1/7) — 분포 · 백분위

**상세 학습**

#### 이 교시에서 답하는 질문
4교시는 범위를 MinMax 로 정했다. Day 1 에서 봤듯 드문 큰 값 때문에 범위가 넓어지면 나머지 값들이 손해를 본다. ORT 에도 범위를 좁히는 방법이 있을까? 있다면 ResNet-20 에서는 얼마나 차이가 날까? 이 교시는 3가지 방법 × 2가지 granularity 를 정확도로 비교하고, 정확도가 비슷할 때 **실제로 고른 범위가 같은지**까지 확인한다.

#### 개념 1 · 6개 모델 정확도
| method | per_channel | acc |
|---|---|---|
| MinMax | False | 0.8460 |
| MinMax | True | 0.8450 |
| Entropy | False | 0.8460 |
| Entropy | True | 0.8450 |
| Percentile | False | 0.8465 |
| Percentile | True | 0.8435 |

모두 손실 1%p 이내, 최고 − 최저 = 0.30%p. 평가 1장 = 0.05%p 라 이 차이는 몇 장의 우연일 수 있다.

<<CODE 6-1 | MinMax · Entropy · Percentile × per-channel 6개 모델 정확도>>

#### 개념 2 · 고른 범위는 다르다
![Percentile 이 활성값 34개의 범위를 얼마나 줄였나 — 평균 0.819, 최소 0.665](assets/fig/qp/d2-scales.svg)

- Percentile 은 Relu · Conv 출력의 범위를 대부분 0.7 ~ 0.9 배로 줄였다 → 범위 안 값은 더 촘촘해지고, 드문 큰 값은 잘린다.
- 그래도 정확도는 그대로 — ResNet-20 은 양자화에 강한 모델. 방법 차이가 큰 모델(YOLOv8n 등)은 Day 3 에서.
- Entropy 는 이 실행에서 MinMax 와 scale 이 똑같았다 → "Entropy 를 썼다" 는 이름만 믿지 말고 결과 scale 을 확인.

<<CODE 6-2 | act_scales — 방법마다 고른 활성값 scale 비교>>

#### 한 번에 정리
- `calibrate_method` = Day 1 의 clipping 선택 (MinMax · Percentile · Entropy).
- Percentile 은 드문 큰 값을 잘라 범위를 평균 18% 줄였다.
- 양자화에 강한 모델은 방법 차이가 작다 (0.30%p).
- ⚠ 정확도만 보지 말고 실제로 고른 scale 도 열어 본다.

---

### 7교시 · 성능 측정: 크기 · 정확도 · 지연 시간 · qp_baseline.csv

**키워드**: `같은 조건`, `스레드 1`, `지연 시간(배치 1) vs 처리량(배치 32)`, `perf_df`, `qp_baseline.csv`, `day 열`, `drop_duplicates`

**개요**: 오늘 만든 5개 모델(FP32 · dynamic · QDQ U8/S8 · QDQ S8/S8 · QOperator U8/S8)을 **같은 조건**(스레드 1 · 같은 입력 · warm-up)으로 재서 한 표에 모은다. 지연 시간(배치 1)과 처리량(배치 32)을 따로 잰다. 결과: static INT8(U8/S8)은 크기 약 1/3 · 처리량 약 2.2 ~ 2.4배 · 정확도 손실 0.1%p, dynamic 은 오히려 느림. 이 표를 `qp_baseline.csv` 에 `day` 열을 붙여 쌓는다 — OT 과정의 `ot_baseline.csv` 처럼, Day 3 이후 TensorRT INT8 · QAT · Pruning 행이 이어진다.

**한눈에 보기**
성능 측정 — 크기 · 정확도 · 지연 · 처리량을 한 표에
> 스레드 1 · 같은 입력 · warm-up 후로 조건을 맞춰 크기 · 정확도 · 지연(배치 1) · 처리량(배치 32)을 재고, qp_baseline.csv 에 Day 별로 쌓는다
- **static INT8 U8/S8 — 처리량 2.2배**: 1,245 → 2,758 장/초, 크기 322 KB, 정확도 −0.1%p
- **QOperator U8/S8 — 2.4배, 배치 1 0.342 ms**: 가장 빠르다
- **dynamic — FP32 의 0.14배**: 170 장/초, 배치 1 6.18 ms
- **qp_baseline.csv 에 이어 붙이기**: day · model · quant · size_kb · acc · mean_ms · p95_ms · fps_b32 — (day, model) 중복 제거
- **⚠ 지연은 잴 때마다 흔들린다 (0.386 ↔ 0.480 ms)**: 결정은 여유를 두고 · `index=False` 를 빼면 `Unnamed: 0` 열

**핵심 용어**
- **지연 시간(latency)**: 한 장(배치 1)을 처리하는 데 걸리는 시간 — 실시간 응답에 중요
- **처리량(throughput)**: 초당 처리 장수(배치 32) — 많은 이미지를 한꺼번에 처리할 때 중요
- **qp_baseline.csv**: 이 과목의 성능 기록표 — Day 마다 행을 이어 붙여 Day 6 까지 비교
- **중복 제거(drop_duplicates)**: 같은 (day, model) 행이 다시 들어오면 마지막 것만 남기기 — 셀을 여러 번 실행해도 표가 불어나지 않게

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `DataFrame.insert()` | `out.insert(0, "day", "Day2")` | 맨 앞에 day 열 추가 |
| `pd.concat()` | `pd.concat([pd.read_csv("qp_baseline.csv"), out])` | 기존 표 아래에 이어 붙이기 |
| `DataFrame.drop_duplicates()` | `drop_duplicates(subset=["day", "model"], keep="last")` | 같은 행은 마지막 것만 |
| `DataFrame.to_csv()` | `to_csv("qp_baseline.csv", index=False)` | CSV 로 저장 (행 번호 빼고) |

**핵심 내용**

#### 같은 조건으로 한 표에
> 크기 · 정확도 · 지연 시간(배치 1) · 처리량(배치 32) — 스레드 1, 같은 입력, warm-up 후
- **static INT8 U8/S8** → 크기 1,071 → 322 KB · 배치 32 처리량 1,245 → **2,758 장/초 (2.2배)** · 정확도 −0.1%p
- **QOperator U8/S8** → 297 KB · **3,017 장/초 (2.4배)** · 배치 1 0.342 ms (가장 빠름)
- **dynamic** → 310 KB · 170 장/초 (FP32 의 0.14배) · 배치 1 6.18 ms
- ⚠ 지연 시간은 잴 때마다 흔들린다 (5교시 0.386 ms ↔ 7교시 0.480 ms) — 결정은 여유를 두고

#### qp_baseline.csv 에 쌓기
> `day` 열을 앞에 붙이고, 파일이 있으면 이어 붙인 뒤 (day, model) 중복을 지우고 저장
- **열** → day · model · quant · size_kb · acc · mean_ms · p95_ms · fps_b32
- **오늘** → Day2 행 5개 (8교시에 `rn20_best.onnx` 1개 추가)
- **다음** → Day 3 TensorRT INT8 · Day 4 QAT · Day 5 Pruning 행이 이어진다
- ⚠ `index=False` 를 빼면 다음에 읽을 때 `Unnamed: 0` 열이 생긴다

**기본 사용형태**
```python
rows = []
for p, q in MODELS:
    s = make_session(p)
    b1, b32 = bench(s, X_ev[:1], n=200, warm=20), bench(s, X_ev[:32], n=30, warm=5)
    rows.append({"model": p, "quant": q, "size_kb": round(size_kb(p), 1), "acc": ort_acc(p),
                 "mean_ms": round(b1["mean"], 3), "p95_ms": round(b1["p95"], 3), "fps_b32": round(b32["fps"], 1)})
perf_df = pd.DataFrame(rows)

out = perf_df.copy(); out.insert(0, "day", "Day2")
if os.path.exists("qp_baseline.csv"):
    out = pd.concat([pd.read_csv("qp_baseline.csv"), out]).drop_duplicates(subset=["day", "model"], keep="last")
out.to_csv("qp_baseline.csv", index=False)
```

**연결**
- [OT D3·6교시 배치 크기: 지연 vs 처리량](#/ot/3/6) — 배치 1 과 배치 32 를 따로 재는 이유
- [OT D4·8교시 실행기 4가지 성능표](#/ot/4/8) — 크기 · 속도 · 정확도를 한 표에 놓고 고르는 방식
- [PY D3·4교시 JSON 과 CSV](#/py/3/4) — CSV 저장 · 읽기

**상세 학습**

#### 이 교시에서 답하는 질문
2 ~ 6교시에서 모델을 여러 개 만들었지만 숫자가 교시마다 흩어져 있고, 잰 조건도 조금씩 달랐다. 어떤 모델을 배포할지 고르려면 **같은 조건으로 한 표**에 놓아야 한다. 그리고 그 표를 Day 3 ~ 6 의 결과와도 비교하려면 파일로 쌓아 두어야 한다. 이 교시는 성능표를 만들고 `qp_baseline.csv` 에 저장한다.

#### 개념 1 · 5개 모델 성능표
![5개 모델의 크기 · 배치 1 지연 · 배치 32 처리량 막대](assets/fig/qp/d2-perf.svg)

| model | quant | size_kb | acc | mean_ms | p95_ms | fps_b32 |
|---|---|---|---|---|---|---|
| rn20.onnx | FP32 | 1070.8 | 0.8460 | 0.701 | 0.806 | 1244.7 |
| rn20_dyn.onnx | dynamic | 310.2 | 0.8445 | 6.183 | 9.022 | 170.3 |
| rn20_int8.onnx | QDQ U8/S8 | 321.8 | 0.8450 | 0.480 | 0.658 | 2757.9 |
| rn20_s8.onnx | QDQ S8/S8 | 322.1 | 0.8450 | 1.113 | 1.379 | 986.9 |
| rn20_qop.onnx | QOperator U8/S8 | 296.9 | 0.8450 | 0.342 | 0.370 | 3016.6 |

(직접 실행 · CPU 스레드 1 — 컴퓨터마다 ms 는 다르고 비율이 비슷하다)

<<CODE 7-1 | 5개 모델 성능표 — 크기 · 정확도 · 지연 · 처리량>>
<<CODE 7-2 | qp_baseline.csv 에 쌓기>>

#### 개념 2 · 지연 시간과 처리량을 따로
- 배치 1 은 "한 장이 얼마나 빨리 나오나"(현장 카메라 실시간 판정), 배치 32 는 "한꺼번에 얼마나 많이"(저장된 영상 일괄 처리).
- 배치를 키우면 노드 준비 비용이 여러 장에 나눠져 처리량이 오른다 — INT8 의 이점은 배치 32 처리량에서 더 또렷했다(2.2 ~ 2.4배).

<<CODE 7-c1 | 정확도 vs 처리량 그림>>

#### 한 번에 정리
- 비교는 같은 조건 (스레드 1 · 같은 입력 · warm-up) 으로.
- 지연 시간(배치 1)과 처리량(배치 32)을 따로.
- static INT8 U8/S8 = 크기 약 1/3 · 처리량 약 2배 · 정확도 손실 거의 없음. dynamic 은 느림.
- ⚠ 결과는 `qp_baseline.csv` 에 `index=False` 로 쌓는다 — Day 6 까지 이어진다.

---

### 8교시 · 미니 프로젝트: 최적 PTQ 설정 찾기 · 채점

**키워드**: `make_ptq()`, `evaluate()`, `16가지 조합 그리드`, `손실 1%p 이내 → 가장 빠른 것`, `rn20_best.onnx`, `클래스별 정확도`, `과제 6 해석`

**개요**: 과제 — ResNet-20 을 현장 PC(x86 CPU)에 배포한다. **정확도 손실 1%p 이내**에서 배치 1 지연 시간이 가장 짧은 PTQ 설정을 찾아 `rn20_best.onnx` 로 저장한다. PTQ 설정 4가지 축 — 형식(QDQ · QOperator) × 활성값(U8 · S8) × 보정 방법(MinMax · Percentile) × granularity(pc · pt) = **16가지**를 모두 만들고 재서 규칙으로 고른다. 16개 정확도는 0.8435 ~ 0.8465 로 모두 통과 — 속도를 가른 것은 **활성값 U8 / S8**(U8 0.34 ~ 0.49 ms, QOperator S8 은 4.5 ~ 5.1 ms). 전체 정확도와 함께 **클래스별 정확도**도 확인한다. 과제 6개(100점) 중 1 ~ 5 는 자동 채점, 6 은 강사 채점 · 정답 보기 셀 없음.

**한눈에 보기**
최적 PTQ 설정 찾기 — 규칙을 먼저, 그리드로
> 형식 × 활성값 자료형 × 보정 방법 × granularity 16가지를 모두 재고, 미리 정한 규칙(손실 ≤ 1%p → 가장 빠른 것)으로 고른다
- **16개 모두 0.8435 ~ 0.8465**: 정확도로는 못 가른다 → 속도가 결정
- **속도는 활성값 자료형이 가른다**: U8 0.34 ~ 0.49 ms · QDQ S8 1.07 ~ 1.31 · QOperator S8 4.5 ~ 5.1 ms
- **이 실행의 선택 — QDQ · U8 · MinMax · per-tensor**: 0.341 ms · acc 0.846
- **클래스별 정확도도 확인**: FP32 cat 0.695 (가장 낮음) · best 는 클래스마다 −0.5 ~ +0.5%p (200장 중 1장)
- **⚠ 1 · 2위 차이 0.001 ms 는 측정 흔들림보다 작다**: 배포 장치에서 다시 재고, 놓치면 안 되는 클래스는 하락 한도를 규칙에

**핵심 용어**
- **그리드 탐색(grid search)**: 설정 후보를 모든 조합으로 만들어 재 보고 고르는 방법
- **선택 규칙**: 먼저 정한 기준 — "손실 한도 안에서 → 가장 빠른 것"
- **클래스별 정확도**: 클래스마다 따로 잰 정확도 — 전체 정확도가 같아도 특정 클래스만 나빠졌는지 확인
- **측정 흔들림**: 같은 모델도 잴 때마다 지연 시간이 달라지는 것 — 차이가 작으면 순위가 바뀔 수 있다

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `make_ptq()` | `make_ptq("QDQ", "U8", "MinMax", True)` | 설정 하나로 INT8 모델 → 경로 `pj_{fmt}_{act}_{method}_{pc\|pt}.onnx` |
| `evaluate(path)` | `evaluate("pj_QDQ_U8_MinMax_pt.onnx")` | `{"acc", "size_kb", "mean_ms", "p95_ms"}` (스레드 1 · 배치 1 · n=100) |
| `itertools.product()` | `itertools.product(["QDQ", "QOperator"], ["U8", "S8"], ["MinMax", "Percentile"], [False, True])` | 16가지 조합 |
| `shutil.copy()` | `shutil.copy(best.path, "rn20_best.onnx")` | 고른 파일을 이름 바꿔 복사 |
| `class_acc()` | `class_acc("rn20_best.onnx")` | 클래스별 정확도 길이 10 배열 |

**핵심 내용**

#### PTQ 설정 4가지 축과 선택 규칙
> 형식 × 활성값 자료형 × 보정 방법 × granularity = 2 × 2 × 2 × 2 = 16가지 — 규칙(손실 ≤ 1%p → 가장 빠른 것)을 먼저 정하고 그리드로
- **정확도** → 16개 모두 0.8435 ~ 0.8465 (손실 ≤ 0.25%p) — 정확도로는 못 가른다
- **속도** → 활성값 U8 8개 0.34 ~ 0.49 ms · QDQ S8 1.07 ~ 1.31 ms · QOperator S8 4.5 ~ 5.1 ms
- **이 실행의 선택** → QDQ · U8 · MinMax · per-tensor — 0.341 ms · acc 0.846 (2위 QOperator U8 MinMax pc 0.342 ms)
- ⚠ 1 · 2위 차이 0.001 ms 는 측정 흔들림보다 작다 — 결정은 여유를 두고, 배포 장치에서 다시 잰다

#### 클래스별 정확도
> 전체 정확도가 같아도 특정 클래스만 나빠질 수 있다 — 클래스 k 마다 `(pred[y == k] == k).mean()`
- **FP32** → cat 0.695 (가장 낮음) · automobile 0.945 (가장 높음)
- **best vs FP32** → 클래스마다 −0.5 ~ +0.5%p (= 200장 중 1장) — 한쪽으로 몰린 하락 없음
- **현장에서** → 놓치면 안 되는 클래스가 있다면 그 클래스의 하락 한도를 규칙에 넣는다
- ⚠ 클래스당 200장 → 1장 = 0.5%p. 작은 차이는 우연일 수 있다

**기본 사용형태**
```python
def make_ptq(fmt, act, method, pc):
    F_ = {"QDQ": QuantFormat.QDQ, "QOperator": QuantFormat.QOperator}
    A_ = {"U8": QuantType.QUInt8, "S8": QuantType.QInt8}
    M_ = {"MinMax": CalibrationMethod.MinMax, "Percentile": CalibrationMethod.Percentile}
    p = f"pj_{fmt}_{act}_{method}_{'pc' if pc else 'pt'}.onnx"
    quantize_static("rn20_pre.onnx", p, CifarReader(X_cal), quant_format=F_[fmt], activation_type=A_[act],
                    weight_type=QuantType.QInt8, per_channel=pc, calibrate_method=M_[method])
    return p

ok = grid_df[grid_df.acc >= fp32_acc - 0.01 - 1e-9]
best = ok.sort_values("mean_ms").iloc[0]
```

**연결**
- [OT D3·8교시 현장 카메라 4대용 설정 정하기](#/ot/3/8) — 조건(스레드 · 배치)을 그리드로 재서 규칙으로 고른 같은 형식의 과제
- [QP D1·8교시 미니 프로젝트](#/qp/1/8) — 어제는 비트 × granularity 6가지, 오늘은 PTQ 설정 16가지
- [VB D2·7교시 학습 기법 · 혼동 행렬](#/vb/2/7) — 클래스별로 성능을 보는 방법

**상세 학습**

#### 이 교시에서 답하는 질문
2 ~ 7교시에서 PTQ 설정의 부품(형식 · 자료형 · 보정 방법 · granularity)을 하나씩 바꿔 봤다. 실제 배포에서는 이 4가지를 **한꺼번에** 정해야 한다. 어떤 조합이 가장 좋을까? 이 교시는 규칙을 먼저 정하고(손실 1%p 이내 → 가장 빠른 것), 16가지를 모두 재서 고르고, 고른 모델이 특정 클래스에서 나빠지지 않았는지까지 확인한다.

#### 개념 1 · 과제 구성 (100점)
| 과제 | 내용 | 배점 |
|---|---|---|
| 1 | `make_ptq(fmt, act, method, pc)` — 설정 하나로 INT8 모델 | 20 |
| 2 | `evaluate(path)` — 정확도 · 크기 · 지연 시간 | 20 |
| 3 | 16가지 조합 그리드 → `grid_df` (약 2분) | 20 |
| 4 | 규칙대로 선택 → `rn20_best.onnx` · `qp_baseline.csv` 에 추가 | 20 |
| 5 | 클래스별 정확도 비교 — `worst_cls` · `worst_drop` | 10 |
| 6 | 결과 해석 (마크다운, 강사 채점) | 10 |

정답 보기 셀 없음 — 힌트만 보고 완성. 채점 셀이 자동 채점 점수와 정답 열람 기록을 보여 준다. 과제 4 의 csv 행: `quant` = `"{fmt} {act}/S8 {method} {pc|pt}"`, `fps_b32` 는 배치 32 로 bench.

<<CODE 과제 1 | 과제 1 — make_ptq: 설정 4가지로 INT8 모델 만들기>>
<<CODE 과제 2 | 과제 2 — evaluate: 정확도 · 크기 · 지연>>

#### 개념 2 · 16가지 결과
![16가지 PTQ 설정의 배치 1 지연 시간 — 활성값 U8 이 빠르다](assets/fig/qp/d2-grid.svg)

| 묶음 | 개수 | 배치 1 mean | 정확도 |
|---|---|---|---|
| QDQ · QOperator × U8 | 8 | 0.34 ~ 0.49 ms | 0.8435 ~ 0.8465 |
| QDQ × S8 | 4 | 1.07 ~ 1.31 ms | 0.8440 ~ 0.8465 |
| QOperator × S8 | 4 | 4.54 ~ 5.13 ms | 0.8435 ~ 0.8465 |

(직접 실행 · 이 컴퓨터 기준. U8 8개 안의 순위는 다시 재면 바뀔 수 있다)

<<CODE 과제 3 | 과제 3 — 16가지 조합 그리드>>
<<CODE 과제 4 | 과제 4 — 손실 1%p 안에서 가장 빠른 설정 · 저장>>

#### 개념 3 · 과제 6 의 생각할 거리
- **왜 빨랐나**: 5교시 — x86 INT8 명령은 U8 × S8 조합이 빠른 경로. 형식은 QDQ · QOperator 모두 ORT CPU 에서 정수 연산으로 실행된다.
- **정확도 차이를 믿을 수 있나**: 16개 차이 0.30%p = 6장. 1장 = 0.05%p 라 순위를 정하기엔 표본이 작다 → 손실 한도를 넘는지만 본다.
- **Jetson(ARM CPU · TensorRT)에 그대로?**: 안 된다. CPU 명령 · 런타임이 다르다 — TensorRT 는 QDQ 형식을 읽어 자기 방식으로 실행 → 배포 장치에서 지연 시간 · 정확도를 다시 잰다 (Day 3).
- **밤에도 쓰는 카메라**: 예제 4-B — 어두운 이미지만으로 보정하면 0.684. 낮 · 밤 이미지를 현장 비율대로 섞어 보정 데이터를 만든다.

<<CODE 과제 5 | 과제 5 — 클래스별 정확도 비교>>

#### 한 번에 정리
- PTQ 설정 = 형식 · 활성값 자료형 · 보정 방법 · granularity 4가지.
- 규칙(손실 한도 → 속도)을 먼저 정하고 그리드로 고른다.
- 이 CPU 에서 속도를 가른 것은 활성값 U8 / S8.
- ⚠ 지연 시간은 흔들린다 → 결정은 여유를 두고 배포 장치에서 다시 잰다. 클래스별 정확도도 함께 본다.
