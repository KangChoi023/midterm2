## Day 3 — PTQ 디버깅과 TensorRT INT8

### 1교시 · 정확도가 떨어지는 이유: BadQuantNet 관찰

**키워드**: `양자화에 민감한 곳`, `BadQuantNet`, `같은 FP32 함수`, `outlier 채널`, `forward hook`, `최대 / 중앙값 비율`, `per-tensor 활성값`, `2칸 vs 228칸`

**개요**: Day 2 설정 그대로 INT8 로 바꾸면 ResNet-20 은 0.846 → 0.8445 로 멀쩡한데, 강사가 준 **BadQuantNet** 은 0.846 → **0.2955** 로 무너진다. 두 모델은 FP32 출력 차이가 3.8e-06 — **같은 함수**다. 다른 것은 중간 텐서의 범위: 블록마다 bn1 출력(ReLU 직전)의 채널별 최댓값을 forward hook 으로 재 보면, 블록 3개(layer1.1 · layer2.1 · layer3.1)에서 채널 하나가 다른 채널보다 **250 ~ 390배** 크다. 활성값은 **per-tensor**(텐서 하나에 scale 하나)라 그 outlier 채널에 scale 이 맞춰지고, 나머지 15채널은 256칸 중 **2칸**만 쓴다. 오늘의 질문: "어디가 · 왜 망가졌나 → 어떻게 고치나 → TensorRT 로 어떻게 배포하나".

**한눈에 보기**
양자화에 민감한 곳 — outlier 채널이 정확도를 무너뜨린다
> 같은 8비트라도 범위가 넓거나 · 채널마다 다르거나 · 정보가 적은 텐서에 오차가 몰린다 — 활성값은 per-tensor 라 채널 하나가 크면 나머지가 몇 칸에 뭉개진다
- **FP32 는 같은데 INT8 은 0.8445 vs 0.2955**: ResNet-20 과 BadQuantNet — 출력 차이 3.8e-06 인 같은 FP32 모델
- **범인은 채널 3 의 최대/중앙값 250 ~ 390배**: layer1.1 247.9 · layer2.1 344.8 · layer3.1 387.2 (나머지 1.4 ~ 1.9)
- **나머지 15채널이 256칸 중 2칸만 쓴다**: layer1.1 scale 1.81 (outlier 포함) vs 0.0104 (제외) → 2칸 vs 228칸
- **민감 지점 — 첫 Conv · 마지막 출력 · depthwise · outlier 채널**: FP16 유지 · per-channel · FP32 로 남기기 · 채널 균등화
- **⚠ weight 는 per-channel 로 피하지만 활성값은 per-tensor**: 그래서 더 위험하고, 오차는 뒤로 갈수록 누적된다

**핵심 용어**
- **양자화 민감 지점**: 첫 Conv · 마지막 분류기 · 탐지 head · depthwise Conv · SiLU/Hardswish 뒤 · outlier 채널 — INT8 오차가 크게 나는 곳
- **outlier 채널**: 다른 채널보다 값이 수백 배 큰 채널 하나 — 텐서 전체 scale 을 독차지한다
- **forward hook(순전파 훅)**: 모듈이 실행될 때마다 그 입력 · 출력을 받아 보는 함수 — 중간 텐서를 꺼낼 때 쓴다
- **오차 누적**: 앞 layer 의 양자화 오차가 다음 layer 의 입력이 되어 뒤로 갈수록 커지는 것

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `export_onnx()` | `export_onnx(model, "bad.onnx", batch=None)` | 배치 가변(None) 또는 고정(숫자) ONNX 로 (제공) |
| `ptq_ort()` | `ptq_ort(src, dst, exclude=(), quantize=None, X=None)` | Day 2 static PTQ 설정 그대로 (QDQ · U8/S8 · per-channel · MinMax, 제공) |
| `register_forward_hook()` | `h = mod.bn1.register_forward_hook(lambda m_, inp, out: …)` | 모듈 출력 가로채기 → 끝나면 `h.remove()` |
| `np.delete()` | `np.delete(A, ch, axis=1)` | 채널 하나 빼기 (outlier 제외) |
| `np.unique()` | `len(np.unique(np.round(normal / s)))` | 실제로 쓰인 정수 칸 수 세기 |

**핵심 내용**

#### 양자화에 민감한 곳
> 범위가 넓거나 · 채널마다 다르거나 · 정보가 적은 텐서 — 같은 8비트라도 오차가 몰리는 위치가 정해져 있다
- **첫 Conv · 마지막 출력** → 입력 정보를 처음 줄이는 곳 · 범위가 다른 값을 한 텐서에 담는 곳 → FP16 유지
- **depthwise Conv** → 채널마다 weight 가 몇 개뿐 → per-channel 필수
- **outlier 채널** → 채널 하나가 수백 배 → 그 layer 를 FP32 로 (3교시) · 채널 균등화
- ⚠ 오차는 **누적**된다 — 앞에서 생긴 오차가 뒤로 갈수록 커진다

#### outlier 채널 — 같은 FP32, 다른 INT8
> 활성값은 per-tensor 라 scale = 텐서 최댓값 / 255 — 채널 하나가 크면 나머지 채널은 몇 칸에 뭉개진다
- **정확도** → ResNet-20 INT8 0.8445 · BadQuantNet INT8 **0.2955** (FP32 는 둘 다 0.846 · 출력 차이 3.8e-06)
- **최대 / 중앙값 비율** → layer1.1 **247.9** · layer2.1 **344.8** · layer3.1 **387.2** (모두 채널 3) · 나머지 블록 1.4 ~ 1.9
- **예** → layer1.1: scale 1.81 (outlier 포함) vs 0.0104 (제외) → 나머지 15채널이 **2칸 vs 228칸** / 256
- ⚠ weight 는 per-channel 로 피할 수 있지만 활성값은 per-tensor — 그래서 더 위험하다 (QP D1·5교시의 "튀는 필터" 가 활성값에서)

**기본 사용형태**
```python
acts, hs = {}, []
for name, mod in bad.named_modules():
    if isinstance(mod, BasicBlock):
        hs.append(mod.bn1.register_forward_hook(
            lambda m_, inp, out, name=name: acts.__setitem__(name, out.detach().numpy())))
with torch.no_grad():
    bad(torch.from_numpy(X_cal[:100]))
for h in hs: h.remove()                                  # hook 은 꼭 지운다
cm = np.abs(acts["layer1.1"]).max(axis=(0, 2, 3))        # 채널별 최댓값 (16,)
ratio, ch = cm.max() / np.median(cm), int(cm.argmax())
```

**연결**
- [QP D1·5교시 Granularity](#/qp/1/5) — per-tensor 는 범위가 가장 큰 채널에 scale 이 맞춰진다 (weight 에서 본 같은 문제)
- [QP D1·6교시 Clipping 과 Outlier](#/qp/1/6) — outlier 를 자르느냐 · 범위에 넣느냐의 trade-off
- [QP D2·4교시 Static 양자화](#/qp/2/4) — `ptq_ort` 가 그대로 쓰는 Day 2 설정
- [VB D2·6교시 ResNet: Residual block](#/vb/2/6) — BasicBlock 의 conv1 → bn1 → ReLU → conv2 구조

**상세 학습**

#### 이 교시에서 답하는 질문
Day 2 는 "PTQ 하면 정확도가 거의 그대로" 였다. 그런데 실제 모델에서는 INT8 로 바꾸자마자 정확도가 무너지는 일이 흔하다. 왜 그럴까? 이 교시는 FP32 로는 똑같은 두 모델(ResNet-20 · BadQuantNet)을 같은 설정으로 양자화해 결과가 갈리는 것을 보고, 그 원인이 **중간 텐서의 범위**(outlier 채널)임을 숫자로 확인한다.

#### 개념 1 · BadQuantNet — 같은 함수, 다른 범위
- 강사가 ResNet-20 의 일부 블록에서 "bn1 출력의 채널 3 을 300배 키우고, 다음 conv2 의 그 채널 weight 를 1/300 로" 바꾼 모델. ReLU 는 양수 배를 통과시키므로 최종 출력은 그대로다.
- FP32 출력 최대 차이 3.8e-06 · 정확도 둘 다 0.846. 그러나 conv1 → ReLU → conv2 사이 텐서의 범위가 수백 배 넓어졌다.

| 모델 | FP32 | INT8 (Day 2 설정) |
|---|---|---|
| ResNet-20 | 0.846 | 0.8445 |
| BadQuantNet | 0.846 | 0.2955 |

<<CODE 1-1 | ResNet-20 vs BadQuantNet — FP32 같고 INT8 은 무너진다>>
<<CODE 1-2 | forward hook 으로 블록별 bn1 출력 · outlier 채널 비율>>

#### 개념 2 · 왜 scale 을 독차지하나
![bn1 출력 채널별 최댓값 비율 — 블록 3개에서 250 ~ 390배, layer1.1 의 나머지 채널은 2칸만 쓴다](assets/fig/qp/d3-outlier.svg)

| 기호 | 뜻 | layer1.1 값 |
|---|---|---|
| A | ReLU 후 활성값 (100, 16, 32, 32) | 최댓값 ≈ 462 (채널 3) |
| s_all = A.max() / 255 | 텐서 전체로 잡은 scale | 1.8113 |
| s_wo | outlier 를 뺀 15채널의 최댓값 / 255 | 0.0104 (최댓값 ≈ 2.66) |
| 칸 수 | 15채널 값 / scale 을 반올림한 서로 다른 정수 개수 | 2 vs 228 |

나머지 채널의 값 0 ~ 2.7 은 1.81 짜리 칸에서 0 또는 1 로만 표현된다 → 정보가 사라진다.

<<CODE 1-c1 | outlier 채널 하나가 scale 을 독차지 — 나머지가 쓰는 칸 수>>

#### 한 번에 정리
- 같은 FP32 함수라도 중간 텐서 범위가 다르면 INT8 결과가 크게 갈린다 (0.8445 vs 0.2955).
- 원인: 블록 3개 bn1 출력의 outlier 채널(250 ~ 390배) — per-tensor 활성값 scale 을 독차지.
- 문제는 모델 전체가 아니라 **특정 layer** → 2교시에서 숫자로 찾는다.
- ⚠ forward hook 은 쓰고 나서 `h.remove()` — 안 지우면 이후 실행마다 계속 불린다.

---

### 2교시 · Layer 별 오차 분석: SQNR · cosine · 민감도

**키워드**: `SQNR`, `20·log10(‖x‖/‖x−x̂‖)`, `1% 오차 = 40 dB`, `cosine similarity`, `qdq_loss_debug`, `_DequantizeLinear_Output`, `누적 비교`, `단독 민감도`, `범인 3블록`

**개요**: 어느 layer 가 범인인지 찾으려면 FP32 텐서와 INT8 텐서가 "얼마나 비슷한지" 재는 지표가 필요하다 — **SQNR**(신호 대 양자화 잡음, dB)과 **cosine similarity**. 찾는 방법은 둘: (1) **누적 비교** — FP32 · INT8 모델을 끝까지 돌려 같은 위치의 중간 텐서를 비교(`qdq_loss_debug`). layer1.1 에서 35 → 9.6 dB 로 떨어진 뒤 계속 낮아 "어디서부터" 무너졌는지 보인다. (2) **단독 민감도** — 블록 하나만 INT8 로 바꾼 모델을 블록마다 만들어 최종 logits 비교. 범인 3개(layer1.1 10.4 · layer2.1 7.7 · layer3.1 3.5 dB)만 20 dB 아래로 또렷하다 — 1교시 outlier 위치와 정확히 같다.

**한눈에 보기**
Layer 별 오차 분석 — SQNR 로 범인 찾기
> SQNR = 20·log10(‖x‖ / ‖x − x̂‖) dB 로 layer 마다 FP32 와 INT8 출력을 비교한다 — 누적 비교로 "어디서부터", 단독 민감도로 "누가" 범인인지 찾는다
- **8비트 이상적이면 약 40 dB, 20 dB 아래면 의심**: 1% 오차 = 40.0 dB · 10% 노이즈 = 20.2 dB
- **cosine 은 방향만 본다**: 크기를 3배 해도 1 — 0.99 아래면 의심
- **누적 — layer1.1 에서 9.6 dB 로 무너진다**: stem 37.8 · layer1.0 34.3 → 그 뒤로는 모두 낮음
- **단독 — 범인 셋**: layer1.1 10.4 · layer2.1 7.5 · layer3.1 3.6 dB, 나머지 32 ~ 49 dB
- **⚠ 누적만으로는 뒤쪽 범인이 안 보인다**: 이미 망가진 입력을 받기 때문 · 분모 1e-12, float64 로 계산

**핵심 용어**
- **SQNR(Signal-to-Quantization-Noise Ratio)**: 20·log10(‖x‖ / ‖x − x̂‖) dB — 신호 크기 대비 오차 크기, 클수록 좋음
- **cosine similarity**: x·x̂ / (‖x‖‖x̂‖) — 두 텐서의 방향이 같은 정도 (1 = 같은 방향, −1 = 반대)
- **누적 비교**: 모델 전체를 INT8 로 돌리고 중간 텐서를 비교 — 앞의 오차가 섞여 보인다
- **단독 민감도(sensitivity)**: 블록 하나만 INT8, 나머지 FP32 로 최종 출력 비교 — 그 블록의 자기 책임만 보인다
- **qdq_loss_debug**: ORT 의 중간 텐서 비교 도구 모음 (출력 꺼내기 · 값 모으기)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `sqnr()` | `sqnr(x, y)` | SQNR dB (이 교시에서 만듦) |
| `cosine()` | `cosine(x, y)` | cosine similarity (이 교시에서 만듦) |
| `np.linalg.norm()` | `np.linalg.norm(x - y)` | 벡터 크기 ‖x‖ = √(Σx²) |
| `modify_model_output_intermediate_tensors()` | `modify_model_output_intermediate_tensors("bad_pre.onnx", "bad_aug_f.onnx")` | 모든 중간 텐서를 출력으로 꺼낸 모델 만들기 |
| `collect_activations()` | `collect_activations("bad_aug_f.onnx", CifarReader(X_cal[:100]))` | 텐서 이름 → 배치별 값 목록 |
| `nodes_to_quantize` | `ptq_ort(src, dst, quantize=[노드 이름…])` | 이 노드들만 양자화 (단독 민감도) |

**핵심 내용**

#### SQNR · cosine
> SQNR = 20·log10(‖x‖ / ‖x − x̂‖) dB — 오차가 신호의 1/10 이면 20 dB, 1/100 이면 40 dB
- **읽는 법** → 8비트 이상적 양자화 ≈ 40 dB 안팎 · **20 dB 아래면 의심** · 0 dB 근처 = 신호만큼 오차
- **예** → 1% 오차 = 40.0 dB · 노이즈 10% = 20.2 dB · 부호 반대 cos = −1.00
- **cosine** → 크기는 무시하고 방향만 (3배 해도 1) — 0.99 아래면 의심
- ⚠ 0 으로 나누지 않게 분모에 1e-12, 계산은 float64 로 펴서 (`np.ravel(x).astype(np.float64)`)

#### 누적 비교 vs 단독 민감도
> 누적 = "어디서부터" 무너지나 · 단독 = "누가" 범인인가 — 앞에서 무너지면 뒤는 모두 나빠 보이므로 둘 다 본다
- **누적** → stem 37.8 · layer1.0 34.3 → **layer1.1 9.6** → … → layer3.2 1.8 dB (layer1.1 뒤로는 모두 낮음)
- **단독** → 범인 layer1.1 **10.4** · layer2.1 **7.5** · layer3.1 **3.6** dB · 나머지 32 ~ 49 dB
- **INT8 쪽 텐서** → 이름 뒤에 `_DequantizeLinear_Output` 이 붙은 것 = 양자화 → 역양자화를 거쳐 다음 layer 가 실제로 받는 값
- ⚠ 누적 그래프만으로는 layer2.1 · layer3.1 이 범인인지 안 보인다 — 이미 망가진 입력을 받았기 때문

**기본 사용형태**
```python
def sqnr(x, y):
    x = np.ravel(x).astype(np.float64); y = np.ravel(y).astype(np.float64)
    return float(20 * np.log10(np.linalg.norm(x) / (np.linalg.norm(x - y) + 1e-12)))

def cosine(x, y):
    x = np.ravel(x).astype(np.float64); y = np.ravel(y).astype(np.float64)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
```

**연결**
- [QP D1·3교시 Affine 양자화](#/qp/1/3) — 양자화 오차(MSE)를 처음 잰 곳, SQNR 은 그 오차를 신호 크기와 비교한 dB 값
- [QP D2·5교시 QDQ vs QOperator](#/qp/2/5) — `_DequantizeLinear_Output` 이 나오는 QDQ 그래프 모양
- [OT D4·3교시 polygraphy run](#/ot/4/3) — 두 런타임 출력을 비교하는 같은 생각 (5교시에서 다시)

**상세 학습**

#### 이 교시에서 답하는 질문
1교시에서 "특정 블록의 outlier 채널" 을 봤지만, 실제 모델에서는 어디를 봐야 할지 모른다. 모든 layer 를 숫자 하나로 점검할 지표와, 그 지표로 범인을 찾는 절차가 필요하다. 이 교시는 지표(SQNR · cosine) 두 개를 만들고, 누적 비교와 단독 민감도로 범인 블록을 찾는다.

#### 개념 1 · 지표 두 개
| 지표 | 식 | 좋은 값 | 의심 |
|---|---|---|---|
| SQNR | 20·log10(‖x‖ / ‖x − x̂‖) dB | 클수록 (8비트 ≈ 40 dB) | 20 dB 아래 |
| cosine | x·x̂ / (‖x‖‖x̂‖) | 1 에 가까울수록 | 0.99 아래 |

- ‖x‖ = √(x₁² + x₂² + …) — 텐서 전체를 한 줄로 편 벡터의 길이.
- 오차가 신호의 1% → ‖x‖/‖x − x̂‖ = 100 → 20·log10(100) = 40 dB.

<<CODE 2-1 | sqnr · cosine — 두 오차 지표>>
<<CODE 2-c1 | FP32 · INT8 블록 출력을 꺼내 누적 SQNR>>
<<CODE 2-c2 | 블록 출력 SQNR 그림>>

#### 개념 2 · 두 가지 찾는 법
![누적 비교는 layer1.1 이후 모두 낮고, 단독 민감도는 범인 3블록만 20 dB 아래](assets/fig/qp/d3-sens.svg)

| 블록 | 누적 SQNR | 단독 SQNR |
|---|---|---|
| stem | 37.84 | 43.99 |
| layer1.0 | 34.32 | 41.74 |
| **layer1.1** | **9.56** | **10.36** |
| layer1.2 | 10.61 | 45.91 |
| layer2.0 | 9.48 | 40.66 |
| **layer2.1** | 6.43 | **7.52** |
| layer2.2 | 6.45 | 46.27 |
| layer3.0 | 5.02 | 41.99 |
| **layer3.1** | 1.86 | **3.57** |
| layer3.2 | 1.77 | 49.17 |
| fc | — | 31.95 |

누적은 보정용 100장, 단독은 평가용 500장의 logits 로 쟀다. 단독 민감도는 블록마다 모델을 하나씩 만들어야 해서 느리지만 범인이 또렷하다.

<<CODE 2-2 | 블록 하나만 양자화 — 단독 민감도 · 범인 찾기>>
<<CODE 2-c3 | 단독 민감도 그림>>

#### 한 번에 정리
- SQNR(dB, 클수록 좋음, 1% 오차 = 40 dB) · cosine(1 = 같은 방향).
- 누적 비교 = 어디서부터 · 단독 민감도 = 누가.
- BadQuantNet 범인 = layer1.1 · layer2.1 · layer3.1 = 1교시 outlier 위치.
- ⚠ INT8 모델에서 비교할 텐서는 `…_DequantizeLinear_Output` (이름 그대로인 텐서는 양자화 전 값).

---

### 3교시 · Mixed Precision: 범인만 FP32 로

**키워드**: `nodes_to_exclude`, `mixed precision`, `conv1 · Relu · conv2`, `생산자 · 소비자`, `block_nodes()`, `0 · 1 · 3블록 → 0.2955 · 0.6290 · 0.8440`, `채널 균등화 · SmoothQuant`

**개요**: 범인을 찾았으면 그 부분만 양자화하지 않고 FP32 로 남긴다 — `quantize_static(..., nodes_to_exclude=[노드 이름])`. 대부분 INT8 · 일부 FP32 인 모델이 **mixed precision**. 어떤 노드를 빼야 할까? outlier 텐서는 conv1 의 출력 → ReLU → conv2 의 입력으로 흐르므로 **세 노드를 함께** 빼야 한다(conv2 만 빼면 0.2905 그대로). 범인 블록을 나쁜 것부터 0 · 1 · 3개 빼면 정확도 0.2955 → 0.6290 → **0.8440**(FP32 0.846 에 0.2%p 차이로 회복), 지연 시간 0.355 → 0.436 → 0.559 ms(FP32 0.679 ms 보다는 여전히 빠름), 크기 322 → 535 → 598 KB.

**한눈에 보기**
Mixed Precision — 범인만 FP32 로
> 대부분은 INT8 로 두고 범인 노드만 `nodes_to_exclude` 로 FP32 에 남겨, 정확도 · 속도 · 크기를 표로 보고 결정한다
- **3블록(9노드) 제외 → 0.8440**: FP32 0.846 에 0.2%p — 0.559 ms · 598 KB 로 FP32(0.679 ms · 1,074 KB)보다 빠르고 작다
- **1블록만 → 0.6290, 0블록 → 0.2955**: 빼는 블록 수에 따라 단계적으로 회복
- **outlier 텐서가 지나는 길 전체를 뺀다**: 생산자 conv1 · 중간 Relu · 소비자 conv2 — conv2 만 빼면 0.2905 그대로
- **다른 처방 — 채널 균등화**: 큰 채널 × 1/k, 다음 weight × k (ReLU 는 양수 배 통과) — LLM 의 SmoothQuant 와 같은 아이디어
- **⚠ 빼는 곳마다 Q · DQ 변환이 생겨 느려진다**: "그래도 FP32 보다 빠른가" 를 꼭 잰다 · 노드 이름은 그래프에서 확인

**핵심 용어**
- **Mixed precision(혼합 정밀도)**: 대부분은 INT8, 민감한 layer 만 FP32 · FP16 으로 남긴 모델
- **nodes_to_exclude**: `quantize_static` 에 넘기는 "양자화하지 않을 노드 이름" 목록
- **생산자 · 소비자**: 문제 텐서를 만드는 노드(conv1)와 받아 쓰는 노드(Relu · conv2) — 텐서가 어디서든 양자화되면 문제가 남는다
- **채널 균등화(cross-layer equalization)**: outlier 채널을 1/k 로 줄이고 다음 weight 를 k 배 — FP32 함수는 그대로, 범위만 고르게

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `block_nodes()` | `block_nodes("layer1.1")` | 블록의 conv1 · Relu · conv2 노드 이름 3개 (이 교시에서 만듦) |
| `nodes_to_exclude` | `ptq_ort(src, dst, exclude=[노드 이름…])` | 이 노드들은 FP32 로 남기기 |
| `bench_fn()` | `bench_fn(lambda: s.run(None, {"input": x1}), n=200, warm=20)` | 함수 하나를 warm-up 후 n 번 → `{"mean", "p95", "fps"}` (제공) |

**핵심 내용**

#### Mixed precision
> 대부분 INT8 + 범인 노드만 FP32 — `nodes_to_exclude` 한 줄로, 정확도와 속도를 표로 보고 결정한다
- **0블록** → 정확도 0.2955 · 0.355 ms · 322 KB (가장 빠르지만 붕괴)
- **1블록 (layer3.1)** → 0.6290 · 0.436 ms · 535 KB
- **3블록 (9노드)** → **0.8440** · 0.559 ms · 598 KB — FP32(0.846) 에 0.2%p 차이로 회복 · FP32(0.679 ms · 1,074 KB)보다 빠르고 작다
- ⚠ 빼는 곳마다 FP32 ↔ INT8 변환(Q · DQ)이 생겨 느려진다 — "그래도 FP32 보다 빠른가" 를 꼭 잰다

#### 텐서가 지나는 길 전체를 뺀다
> outlier 텐서 = conv1 출력 → Relu → conv2 입력 — 생산자(conv1) · 중간(Relu) · 소비자(conv2) 를 함께 FP32 로
- **block_nodes** → `/layer1/layer1.1/conv1/Conv` · `/layer1/layer1.1/Relu` · `/layer1/layer1.1/conv2/Conv`
- **빼면 안 되는 것** → shortcut Conv · 블록 끝 `Relu_1` (outlier 가 지나지 않는다)
- **예** → conv2 만 FP32 → **0.2905** (그대로) · 세 노드 → 0.8440
- ⚠ 노드 이름은 그래프마다 다르다 — `onnx.load(...).graph.node` 에서 확인하고 쓴다

#### 다른 처방 — 채널 균등화
> outlier 채널 × 1/k, 다음 weight × k — ReLU 는 양수 배를 통과시키므로 FP32 출력은 같고 범위만 고르게
- **BadQuantNet** → 강사가 "채널 × 300, 다음 weight × 1/300" 으로 만든 모델 — 반대로 하면 원래대로
- **LLM 에서** → SmoothQuant 가 같은 아이디어 (활성값의 큰 채널을 weight 쪽으로 옮김)
- **Day 4 에서** → 학습으로 범위에 적응시키는 QAT
- ⚠ 오늘 실습은 "범인을 FP32 로 남기기" — 균등화는 개념만

**기본 사용형태**
```python
def block_nodes(block, graph=g_bad):
    p = f"/{block.split('.')[0]}/{block}/"              # "/layer1/layer1.1/"
    want = {p + "conv1/Conv", p + "Relu", p + "conv2/Conv"}
    return [n.name for n in graph.node if n.name in want]

ex = [n for b in ["layer3.1", "layer2.1", "layer1.1"] for n in block_nodes(b)]   # 9개
ptq_ort("bad_pre.onnx", "bad_mix3.onnx", exclude=ex)
```

**연결**
- [QP D2·4교시 Static 양자화](#/qp/2/4) — `quantize_static` 의 기본 인자들, 오늘은 `nodes_to_exclude` 를 더한다
- [QP D2·7교시 성능 측정](#/qp/2/7) — 크기 · 정확도 · 지연 시간을 한 표로 재는 방법
- [OT D4·6교시 FP16](#/ot/4/6) — TensorRT 에서는 뺀 노드가 FP32 대신 FP16 으로 (4교시)

**상세 학습**

#### 이 교시에서 답하는 질문
2교시에서 범인 블록 3개를 찾았다. 이제 고쳐야 한다. 가장 쉬운 처방은 그 부분만 양자화하지 않는 것 — 그런데 정확히 **어느 노드**를 빼야 하고, 빼면 속도 · 크기는 얼마나 손해일까? 이 교시는 빼는 노드를 정하는 원칙(텐서가 지나는 길 전체)과, 빼는 블록 수에 따른 정확도 · 지연 · 크기 trade-off 를 표로 확인한다.

#### 개념 1 · 어떤 노드를 빼나
- BasicBlock: `conv1 → (BN 접힘) → Relu → conv2 → (BN) → + shortcut → Relu_1`.
- outlier 채널은 bn1 출력(= ONNX 에서 conv1 출력, BN 이 접혀 있음)에 있다 → conv1 의 출력 · Relu · conv2 의 입력이 모두 큰 범위.
- conv2 만 빼면 conv1 출력 쪽에서 이미 Q/DQ 가 붙어 양자화된다 → 0.2905.

<<CODE 3-1 | block_nodes — 블록의 conv1 · Relu · conv2 노드 이름>>
<<CODE 3-c1 | conv2 만 빼면? (outlier 는 conv1 출력)>>

#### 개념 2 · 몇 블록을 빼나
![범인 블록을 0 · 1 · 3개 빼면 정확도는 회복, 지연 · 크기는 조금씩 늘어난다](assets/fig/qp/d3-mixed.svg)

| 제외 | 노드 | 정확도 | 배치 1 ms | 크기 KB |
|---|---|---|---|---|
| FP32 | — | 0.846 | 0.679 | 1,074 |
| 0블록 | 0 | 0.2955 | 0.355 | 322 |
| 1블록 (layer3.1) | 3 | 0.6290 | 0.436 | 535 |
| 3블록 | 9 | 0.8440 | 0.559 | 598 |

(CPU · 스레드 1 · 직접 실행. 지연 시간은 컴퓨터마다 다르다 — 순서와 비율을 본다.) 단독 민감도가 가장 낮은 블록(layer3.1, 3.6 dB)부터 뺐다. 크기가 늘어난 것은 FP32 로 남긴 conv weight 가 4 byte 그대로이기 때문.

<<CODE 3-2 | 범인 0 · 1 · 3 블록을 FP32 로 — 정확도 · 지연 · 크기>>

#### 한 번에 정리
- `nodes_to_exclude` 로 범인 노드만 FP32 → mixed precision.
- 문제 텐서의 생산자 · 소비자를 함께 뺀다 (conv1 · Relu · conv2).
- 3블록 제외 → 정확도 회복(0.8440), FP32 보다 여전히 빠르고 작다.
- ⚠ 빼는 만큼 Q/DQ 변환이 늘어 느려진다 — 표로 보고 결정.

---

### 4교시 · TensorRT INT8 개념: implicit vs explicit (QDQ)

**키워드**: `implicit · calibrator (deprecated)`, `explicit · QDQ ONNX`, `S8 · 대칭 · zero_point 0`, `trt_check()`, `make_trt_int8()`, `reduce_range=True · 7비트 weight`, `ActivationSymmetric`, `BuilderFlag.INT8`, `INT8 + FP16 fallback`

**개요**: TensorRT 로 INT8 엔진을 만드는 길은 둘 — **implicit**(FP32 ONNX + calibrator 클래스, TensorRT 가 빌드 중에 범위를 재고 INT8 위치도 정함 · TensorRT 10 에서 deprecated)과 **explicit**(scale 이 이미 들어 있는 **QDQ ONNX** 를 넣음 · Q/DQ 가 있는 곳만 INT8 · 권장). 조건: TensorRT INT8 은 **부호 있는 INT8(S8) · 대칭(zero_point = 0)** 만 받는다. Day 2 의 x86 최적 설정(활성값 U8 · 비대칭)은 `trt_check` 에서 불합격(uint8 zp 68개 · 0 아닌 zp 26개). `make_trt_int8` — QInt8 + `ActivationSymmetric` · `WeightSymmetric` + `reduce_range=True`(weight 7비트 ±64, Day 2 4교시) — 로 만들면 합격 · 정확도 0.8420. 빌드는 OT Day 4 `build_engine` 에 `BuilderFlag.INT8` 한 줄. FP16 을 함께 켜면 양자화하지 않은 layer 는 FP16.

**한눈에 보기**
TensorRT INT8 — explicit(QDQ) 과 S8 · 대칭 조건
> implicit 은 TensorRT 가 빌드 중에 보정 · INT8 위치를 정하고, explicit 은 QDQ ONNX 의 Q/DQ 가 있는 곳만 INT8 로 돈다 — TensorRT 10 은 explicit 권장
- **explicit 은 mixed precision 이 쉽다**: Q/DQ 를 빼면 끝 (3교시 그대로) — Day 4 QAT 모델도 같은 길
- **TensorRT INT8 은 zp = 0 · 부호 있는 INT8 만**: Day 2 의 활성값 U8(비대칭)은 그대로 못 넣는다
- **trt_check — Day 2 모델은 uint8 68 · nonzero_zp 26 → 불합격**: QInt8 + 대칭으로 다시 만들면 합격, ORT 정확도 0.8420
- **reduce_range=True → weight |q| ≤ 64 (7비트)**: bias 는 FP32 로 두고 TensorRT 가 처리
- **⚠ TensorRT 엔진 입력은 배치 고정이 다루기 쉽다**: `export_onnx(..., batch=1)` · reader 도 bs=1

**핵심 용어**
- **implicit 양자화**: FP32 ONNX + calibrator(`IInt8EntropyCalibrator2`) 로 TensorRT 가 빌드 중 범위를 정하는 방식 — TensorRT 10 deprecated
- **explicit 양자화**: Q/DQ 노드가 이미 들어간 ONNX 를 넣는 방식 — INT8 위치 · scale 을 우리가 정한다 (QAT 모델도 같은 길)
- **INT8 + FP16 fallback**: INT8 · FP16 플래그를 함께 켜서, Q/DQ 없는 layer 는 FP16 으로 실행
- **ActivationSymmetric · WeightSymmetric**: ORT `extra_options` — 활성값 · weight 를 대칭(zp = 0)으로 양자화
- **reduce_range=True**: weight 를 8비트 대신 7비트(±64)로 — VNNI 없는 CPU 에서 INT8 곱의 합이 16비트를 넘치지 않게 (QP D2·4교시)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `trt_check()` | `trt_check("rn20_int8.onnx")` | TensorRT 에 넣을 수 있는 QDQ 인지 점검 → `{"qdq", "uint8", "nonzero_zp", "ok"}` (예제 4-A 제공) |
| `make_trt_int8()` | `make_trt_int8(src, dst, reader, exclude=())` | TensorRT 용 S8 · 대칭 · `reduce_range` QDQ ONNX (이 교시에서 만듦) |
| `extra_options` | `extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False}` | `quantize_static` 의 세부 설정 |
| `BuilderFlag.INT8` | `config.set_flag(trt.BuilderFlag.INT8)` | 엔진 빌드에서 INT8 켜기 |
| `build_engine()` | `build_engine(onnx_path, engine_path, fp16=False, int8=False)` | OT D4 빌드 함수 + INT8 인자 (이 교시에서 완성) |

**핵심 내용**

#### implicit vs explicit
> implicit = TensorRT 가 빌드 중에 보정 · INT8 위치 결정 / explicit = QDQ ONNX 의 Q/DQ 가 있는 곳만 INT8 — 오늘은 explicit
- **범위를 정하는 곳** → implicit: TensorRT 빌드 중 / explicit: ONNX Runtime · ModelOpt 등 바깥 도구
- **mixed precision** → implicit: 어렵다 / explicit: Q/DQ 를 빼면 끝 (3교시 그대로)
- **TensorRT 10** → implicit deprecated · explicit 권장 (Day 4 QAT 모델도 같은 길)
- ⚠ `trtexec --onnx=model_qdq.onnx --int8 --fp16` = INT8 + FP16 플래그와 같은 뜻

#### TensorRT 의 조건 — S8 · 대칭
> TensorRT INT8 은 zero_point = 0 · 부호 있는 INT8 만 — Day 2 의 활성값 U8(비대칭)은 그대로 못 넣는다
- **Day 2 모델** → `trt_check` = qdq 112 · **uint8 68 · nonzero_zp 26** · ok False
- **make_trt_int8** → QInt8 + 대칭 → qdq 128 · uint8 0 · nonzero_zp 0 · **ok True** · ORT 정확도 **0.8420** (FP32 0.846)
- **reduce_range=True** → weight 최댓값 |q| = **64** (7비트) — 교안 검증 셀이 `≤ 64` 로 확인
- **QuantizeBias False** → bias 는 FP32 로 두고 TensorRT 가 알아서 처리
- ⚠ TensorRT 엔진 입력은 배치 고정이 다루기 쉽다 → `export_onnx(..., batch=1)` · reader 도 `bs=1`

**기본 사용형태**
```python
def make_trt_int8(src, dst, reader, exclude=()):
    with quiet():
        quantize_static(src, dst, reader, quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        nodes_to_exclude=list(exclude),
                        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                                       "QuantizeBias": False})
    return dst

# build_engine 안 — FP16 플래그 바로 아래
if int8:
    config.set_flag(trt.BuilderFlag.INT8)
```

**연결**
- [QP D1·4교시 대칭 vs 비대칭](#/qp/1/4) — zero_point = 0 인 대칭 양자화, 오늘 TensorRT 의 조건
- [QP D2·5교시 QDQ vs QOperator](#/qp/2/5) — U8 / S8 활성값, x86 CPU 는 U8 이 빠르다 (그래서 Day 2 는 U8)
- [OT D4·4교시 build_engine()](#/ot/4/4) — 오늘 INT8 한 줄을 더하는 원래 빌드 함수

**상세 학습**

#### 이 교시에서 답하는 질문
1 ~ 3교시는 ONNX Runtime CPU 로 INT8 을 실행했다. 실제 배포 장치(GPU · Jetson)에서는 TensorRT 를 쓴다(OT 과목). 그럼 지금까지 만든 INT8 ONNX 를 TensorRT 에 그대로 넣을 수 있을까? 이 교시는 TensorRT INT8 의 두 경로와, explicit 경로에서 지켜야 할 조건(S8 · 대칭)을 확인하고 조건에 맞는 ONNX 를 만든다.

#### 개념 1 · 두 경로
![implicit 은 TensorRT 가 보정, explicit 은 QDQ ONNX 를 넣는다 — Day 2 모델은 trt_check 불합격](assets/fig/qp/d3-trt.svg)

| | implicit | explicit (QDQ) |
|---|---|---|
| 입력 | FP32 ONNX + calibrator 클래스 | QDQ ONNX (scale 포함) |
| 범위를 정하는 곳 | TensorRT 빌드 중 | 바깥 도구 (ORT 등) |
| INT8 위치 | TensorRT 가 빠른 쪽으로 | Q/DQ 있는 곳만 |
| TensorRT 10 | deprecated | 권장 |

<<CODE 4-c1 | trt_check — S8 · zero-point 0 인지 검사>>

#### 개념 2 · trt_check 가 보는 것
- Q/DQ 노드의 입력은 (x, scale, zero_point) — `input[2]` 가 zero_point initializer.
- 그 자료형이 uint8 이거나 값이 0 이 아니면 TensorRT 파서가 "zero point must be 0" 류 오류로 빌드를 멈춘다.

| 모델 | qdq | uint8 | nonzero_zp | ok |
|---|---|---|---|---|
| rn20_int8.onnx (Day 2 설정) | 112 | 68 | 26 | False |
| rn20_trt_int8.onnx (make_trt_int8) | 128 | 0 | 0 | True |

<<CODE 4-1 | make_trt_int8 — S8 대칭 QDQ · 배치 1 모델>>
<<CODE 4-2 | build_engine 에 INT8 플래그 한 줄>>

#### 한 번에 정리
- TensorRT 10 INT8 = explicit(QDQ ONNX) 권장 · implicit calibrator 는 deprecated.
- 조건: S8 · 대칭(zp = 0) → `activation_type=QInt8` + `ActivationSymmetric` · `WeightSymmetric` (+ `reduce_range=True` → weight 7비트).
- 빌드: `BuilderFlag.INT8` 한 줄, FP16 도 켜면 뺀 layer 는 FP16.
- ⚠ Day 2 의 U8 모델은 x86 CPU 용 — TensorRT 에 넣기 전에 `trt_check`.

---

### 5교시 · INT8 엔진 빌드: FP32 · FP16 · INT8 비교

**키워드**: `build_once()`, `엔진 3개`, `정확도 · 지연 · 크기 한 표`, `run_acc()`, `Polygraphy`, `tolerance`, `argmax 일치율`, `layer_precisions()`, `ProfilingVerbosity.DETAILED`

**개요**: 같은 ResNet-20 으로 엔진 세 개 — FP32(`rn20_b1_pre.onnx`, 플래그 없음) · FP16(같은 ONNX + FP16) · INT8(`rn20_trt_int8.onnx` + INT8 + FP16) — 를 만들어 **정확도 · 지연 시간 · 크기**를 한 표로 비교한다. 엔진 검증은 두 단계: (1) 같은 입력에서 ORT INT8 과 출력을 비교 — 런타임마다 연산 순서 · 융합이 달라 숫자가 똑같지 않으므로 **허용 오차(tolerance)**와 **예측 일치율**을 본다(Polygraphy 와 같은 생각). (2) 엔진의 layer 마다 실제 실행 정밀도(INT8 · FP16 · FP32)를 센다. GPU 가 없는 이 실행(ORT CPU): FP32 0.8460 · 0.70 ms · 1.05 MB vs INT8 0.8420 · 1.28 ms · 0.32 MB — x86 CPU 는 활성값 S8 이 느려 INT8 이 오히려 느리다.

**한눈에 보기**
INT8 엔진 빌드 — FP32 · FP16 · INT8 을 같은 자로
> 같은 모델로 FP32 · FP16 · INT8 엔진을 만들어 정확도 · 배치 1 지연 · 파일 크기를 한 표에 두고, 출력 비교와 layer 정밀도로 검증한다
- **INT8 엔진 = QDQ ONNX + int8 · fp16 플래그**: Q/DQ 없는 곳은 FP16
- **이 실행(ORT CPU) — INT8 0.8420 · 1.28 ms · 0.32 MB**: FP32 0.8460 · 0.70 ms · 1.05 MB, GPU 가 없어 FP16 은 NaN
- **작은 모델 · 배치 1 은 INT8 이득이 작게 보인다**: 복사 · 실행 준비 시간이 대부분 (8교시 YOLOv8n 에서 더 잘 보임)
- **검증 ① 출력 비교 — 최대 차이 + 예측 일치율**: 300장, GPU 면 일치 ≥ 0.95 → ✅
- **검증 ② layer_precisions**: 엔진 정보의 `Int8` · `Half` 글자로 layer 마다 실제 정밀도를 센다
- **⚠ layer 별 정보는 빌드할 때 `ProfilingVerbosity.DETAILED`**: 숫자는 장치마다 다르다

**핵심 용어**
- **허용 오차(tolerance)**: 두 런타임 출력을 비교할 때 "이 정도 차이는 같다" 고 보는 범위 (`--atol` · `--rtol`)
- **argmax 일치율**: 두 모델의 예측 클래스가 같은 비율 — 최대 차이와 함께 본다
- **Polygraphy**: 같은 입력을 여러 런타임에 넣어 출력을 비교하는 NVIDIA 도구
- **layer 융합**: TensorRT 가 Conv + BN + ReLU (+ Add) 를 한 layer 로 합치는 것 — 엔진 layer 수 < ONNX 노드 수

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `build_once()` | `build_once(onnx_path, engine_path, fp16=False, int8=False)` | 엔진이 없을 때만 빌드 (제공) |
| `run_acc()` | `run_acc(infer, X=X_ev, y=y_ev)` | 배치 1 로 한 장씩 정확도 (이 교시에서 만듦) |
| `polygraphy run` | `polygraphy run rn20_trt_int8.onnx --onnxrt --trt --int8 --fp16 --atol 0.1 --rtol 0.1` | 명령줄로 ORT · TRT 출력 비교 |
| `layer_precisions()` | `layer_precisions("rn20_int8.engine")` | 엔진 layer 별 실행 정밀도 `Counter` (예제 5-A 제공) |
| `create_engine_inspector()` | `eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)` | 엔진 layer 정보 JSON |

**핵심 내용**

#### 엔진 세 개 — 같은 자로 잰다
> 같은 모델 → FP32 · FP16 · INT8 엔진 → 정확도 · 배치 1 지연 · 파일 크기를 한 표에
- **INT8 엔진** → QDQ ONNX + `int8=True, fp16=True` (Q/DQ 없는 곳은 FP16)
- **이 실행 (ORT CPU)** → FP32 0.8460 · 0.70 ms · 1.05 MB / INT8 0.8420 · **1.28 ms** · **0.32 MB** / FP16 NaN
- **작은 모델 · 배치 1** → GPU 에서도 데이터 복사 · 실행 준비 시간이 대부분이라 INT8 이득이 작게 보인다 (8교시 YOLOv8n 에서 더 잘 보임)
- ⚠ GPU 가 없으면 FP16 행은 NaN · FP32 · INT8 은 ORT CPU 값 — 숫자는 장치마다 다르다

#### 엔진 검증 — 출력 비교 · layer 정밀도
> ① 같은 입력에서 ORT INT8 과 비교 (최대 차이 + 예측 일치율) ② layer 마다 실제로 INT8 로 도는지 확인
- **출력 비교** → 처음 300장 · `max_abs = |a − b|.max()` · `agree = (a.argmax(1) == b.argmax(1)).mean()`
- **이 실행** → 같은 ORT 끼리라 최대 차이 0.0000 · 일치 1.000 (GPU 이면 일치 ≥ 0.95 → ✅, 아래면 ⚠ — 구현이 틀린 것이 아니라 두 런타임 차이이므로 해설의 점검 순서를 따라간다)
- **layer_precisions** → 엔진 정보의 출력 자료형 글자(`Int8 …` · `Half …`)로 INT8 · FP16 · FP32 를 센다
- ⚠ layer 별 정보를 보려면 빌드할 때 `ProfilingVerbosity.DETAILED` (실습 4-2)

**기본 사용형태**
```python
def run_acc(infer, X=X_ev, y=y_ev):
    return float(np.mean([infer(X[i:i + 1]).argmax() == y[i] for i in range(len(X))]))

ENG = [("FP32", "rn20_b1_pre.onnx", False, False), ("FP16", "rn20_b1_pre.onnx", True, False),
       ("INT8", "rn20_trt_int8.onnx", True, True)]
infer = TRTInfer(build_once(src, f"rn20_{prec.lower()}.engine", fp16=f16, int8=i8))   # GPU
```

**연결**
- [OT D4·3교시 polygraphy run](#/ot/4/3) — 두 런타임 출력 비교 · `--atol` · `--rtol`
- [OT D4·7교시 엔진 레이어 들여다보기](#/ot/4/7) — engine inspector 로 융합 · 정밀도 보기
- [OT D5·1교시 TRTInfer](#/ot/5/1) — 오늘 엔진 실행에 그대로 쓰는 클래스
- [QP D2·5교시 QDQ vs QOperator](#/qp/2/5) — x86 CPU 에서 활성값 S8 이 느린 이유

**상세 학습**

#### 이 교시에서 답하는 질문
4교시에서 TensorRT 조건에 맞는 QDQ ONNX 를 만들었다. 실제로 엔진을 만들면 FP32 · FP16 보다 얼마나 작고 빠를까? 그리고 그 엔진이 "제대로" INT8 로 계산하는지 어떻게 확인할까? 이 교시는 엔진 세 개를 같은 조건으로 재고, 출력 비교 · layer 정밀도 두 가지로 엔진을 검증한다.

#### 개념 1 · 비교표
| 엔진 | ONNX | 플래그 | 정확도 | 배치 1 ms | 크기 MB |
|---|---|---|---|---|---|
| FP32 | rn20_b1_pre.onnx | — | 0.8460 | 0.698 | 1.05 |
| FP16 | rn20_b1_pre.onnx | FP16 | (GPU 필요) | — | — |
| INT8 | rn20_trt_int8.onnx | INT8 + FP16 | 0.8420 | 1.280 | 0.32 |

(GPU 없음 → ORT CPU · 평가 2,000장 · 직접 실행.) CPU 에서 INT8 이 느린 것은 활성값을 S8 로 만들었기 때문 — Day 2 5교시에서 본 "x86 은 U8 × S8 이 빠른 경로" 와 같다. TensorRT GPU 에서는 S8 이 표준이라 결과가 다르다 → 배포 장치에서 다시 잰다.

<<CODE 5-1 | FP32 · FP16 · INT8 엔진 — 정확도 · 지연 · 크기>>
<<CODE 5-c1 | polygraphy 로 ORT 와 TensorRT INT8 비교>>

#### 개념 2 · 왜 숫자가 똑같지 않나
- 런타임마다 연산 순서 · layer 융합 · FP16 fallback 이 달라 INT8 끼리도 출력이 조금씩 다르다.
- 그래서 "같다" 의 기준 = 최대 차이가 허용 오차 안 + 예측(argmax) 대부분 일치. logits 크기가 약 19.5 이니 차이 몇 이하는 예측을 바꾸지 않는 경우가 많다.

<<CODE 5-2 | ORT INT8 vs 엔진 INT8 logits 차이 · 예측 일치>>
<<CODE 5-c2 | 엔진 레이어별 정밀도 세기>>

#### 한 번에 정리
- 같은 QDQ ONNX → INT8 + FP16 플래그 → INT8 엔진.
- FP32 · FP16 · INT8 을 정확도 · 지연 · 크기 한 표로.
- 엔진 검증 = 출력 비교(tolerance · 예측 일치) + layer 정밀도.
- ⚠ 작은 모델 · 배치 1 · CPU 에서는 INT8 이 빠르지 않을 수 있다 — 측정값으로 판단.

---

### 6교시 · 탐지 모델 양자화: YOLOv8n · Detect head

**키워드**: `output0 (1, 84, 8400)`, `박스 0~640 + 점수 0~1`, `scale 5.166`, `점수 0칸 → 사람 0명`, `Detect head 후처리`, `/model.22/ · cv2 · cv3`, `YoloReader`, `보정 43장 / 평가 127장`, `IoU 0.966`

**개요**: YOLOv8n 출력 `output0` (1, 84, 8400) 은 한 텐서에 **박스 4줄(0 ~ 640 픽셀)**과 **클래스 점수 80줄(0 ~ 1)**을 함께 담는다(마지막 Concat). 이 텐서를 대칭 INT8 scale 하나로 양자화하면 scale = **5.166** — 점수 0 ~ 1 은 칸 하나의 0.19배라 모두 0 으로 반올림되어 사람을 **한 명도** 못 찾는다(FP32 2명 → 0명). 그래서 Detect head 의 **후처리 부분**(DFL · Sigmoid · 박스 디코딩 · Concat 등 24노드)은 양자화하지 않고, 무거운 Conv(cv2 · cv3 갈래)는 INT8 로 둔다 → 다시 2명 · FP32 박스와 IoU 중앙값 **0.966**. 데이터는 Penn-Fudan 170장을 보정 43장 / 평가 127장으로 **겹치지 않게** 나눈다.

**한눈에 보기**
탐지 모델 양자화 — YOLOv8n Detect head 의 함정
> output0 은 박스(0 ~ 640)와 점수(0 ~ 1)가 섞인 텐서 — scale 하나로 양자화하면 점수가 0칸이 되어 사람을 0명 찾는다, 그래서 head 후처리만 FP16 으로 남긴다
- **scale = 656 / 127 = 5.166**: 점수 0.9 / 5.166 = 0.17 → 반올림 0
- **전체 양자화 → 사람 2명이 0명**: trt_check 는 합격 — 형식은 맞아도 결과가 틀릴 수 있다
- **head 후처리 24노드만 제외**: `/model.22/` 의 Reshape · Concat · Sigmoid · Softmax … — 무거운 cv2 · cv3 Conv 는 INT8
- **결과 — 사람 2명, 짝 IoU 중앙값 0.966**: 크기 12.26 → 3.43 MB
- **⚠ 범위가 다른 값이 섞이는 출력은 대표 민감 지점**: 보정 43장 · 평가 127장은 같은 사진을 쓰지 않는다

**핵심 용어**
- **Detect head**: YOLOv8 의 마지막 모듈(`/model.22/`) — cv2(박스) · cv3(클래스) Conv 갈래 + 후처리(DFL · 디코딩 · Concat)
- **DFL(Distribution Focal Loss) 디코딩**: 박스 변마다 16칸 확률(Softmax)의 기댓값으로 거리를 구하는 부분
- **보정 / 평가 분리**: 보정에 쓴 사진을 평가에 쓰지 않는 것 — 학습 / 검증 분리와 같은 원칙
- **짝 IoU**: INT8 박스마다 가장 많이 겹치는 FP32 박스와의 IoU — 두 모델의 박스가 같은지 보는 기능 점검

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `YoloReader` | `YoloReader(CAL_IDX)` | 사진을 한 장씩 `{"images": (1, 3, 640, 640)}` 로 내보내는 보정 reader (예제 6-A 제공) |
| `ort_infer()` | `ort_infer("yolov8n_int8.onnx")` | ONNX → `x` 를 받아 output0 을 돌려주는 함수 (TRTInfer 와 같은 모양, 제공) |
| `iou_matrix()` | `iou_matrix(b8, b32).max(1)` | 박스 묶음 두 개의 IoU 표 (N, M) (제공) |
| `re.match()` | `re.match(r"/model.22/cv[23]\.\d/", name)` | 노드 이름이 무거운 Conv 갈래인지 판별 |

**핵심 내용**

#### 범위가 다른 값을 한 텐서에 — Detect head 의 함정
> output0 = 박스(0 ~ 640) + 점수(0 ~ 1) — scale 하나(5.166)로 양자화하면 점수가 0칸 → 사람 0명
- **식** → 대칭 INT8 scale = max|x| / 127 ≈ 656 / 127 = **5.166** · 점수 0.9 / 5.166 = 0.17 → 반올림 **0**
- **전체 양자화 (6-1)** → `score_levels = 1 / scale` = **0.19칸** · 사람 FP32 2명 → INT8 **0명** (trt_check 는 합격 — 형식은 맞아도 결과가 틀림)
- **처방** → 후처리만 제외 · 무거운 Conv 는 INT8
- ⚠ 분류기의 마지막 출력 · 탐지 head 는 "범위가 다른 값이 섞이는" 대표 민감 지점 (1교시 표)

#### head 후처리만 FP16 — 무거운 Conv 는 INT8
> `/model.22/` 로 시작하되 cv2 · cv3 갈래 Conv 가 아닌 노드 24개를 `exclude` 로
- **제외 24노드** → Reshape 8 · Concat 4 · Slice 2 · Sub 2 · Add 2 · Sigmoid · Transpose · Softmax · dfl Conv · Div · Mul
- **결과 (6-2)** → 사람 2명 · FP32 박스와 짝 IoU 중앙값 **0.966** · trt_check 합격
- **크기** → yolov8n.onnx 12.26 MB → yolov8n_int8.onnx **3.43 MB**
- ⚠ 보정 43장(인덱스 4의 배수) · 평가 127장 — 같은 사진을 두 번 쓰지 않는다

**기본 사용형태**
```python
CAL_IDX = [i for i in range(len(imgs)) if i % 4 == 0]          # 43장
EVAL_IDX = [i for i in range(len(imgs)) if i % 4 != 0]         # 127장

g_y = onnx.load("yolov8n_pre.onnx").graph
head = [n.name for n in g_y.node
        if n.name.startswith("/model.22/") and not re.match(r"/model.22/cv[23]\.\d/", n.name)]
make_trt_int8("yolov8n_pre.onnx", "yolov8n_int8.onnx", YoloReader(CAL_IDX), exclude=head)
yolo_int8 = ort_infer("yolov8n_int8.onnx")
```

**연결**
- [OT D5·2교시 YOLOv8n 엔진 · 전처리 · 후처리](#/ot/5/2) — output0 (1, 84, 8400) · letterbox · decode 를 처음 연결한 곳
- [VB D4·6교시 YOLO 구조](#/vb/4/6) — 출력 텐서에 박스와 클래스 점수가 함께 있는 이유
- [VB D4·3교시 IoU · NMS](#/vb/4/3) — 짝 IoU 에 쓰는 IoU 식
- [QP D2·4교시 Static 양자화](#/qp/2/4) — `CifarReader` 와 같은 구조의 `YoloReader`

**상세 학습**

#### 이 교시에서 답하는 질문
1 ~ 5교시는 분류 모델(ResNet-20)이었다. 침입 감지 카메라(OT Day 5)는 탐지 모델 YOLOv8n 을 쓴다. 탐지 모델도 그대로 INT8 로 바꾸면 될까? 이 교시는 YOLOv8n 을 통째로 양자화하면 무엇이 깨지는지(출력 텐서의 범위 문제), 어느 노드를 빼야 고쳐지는지 확인한다.

#### 개념 1 · 한 텐서에 두 가지 범위
![output0 을 scale 하나로 양자화하면 점수는 칸 하나의 0.19배 → 0, head 후처리를 빼면 다시 찾는다](assets/fig/qp/d3-head.svg)

| 기호 | 뜻 | 값 |
|---|---|---|
| output0 | (1, 84, 8400) — 앵커 8,400개 × (박스 4 + 점수 80) | |
| out_scale | output0 을 만드는 DequantizeLinear 의 scale | 5.166 |
| score_levels | 1 / out_scale — 점수 0 ~ 1 에 쓸 수 있는 칸 수 | 0.19 |

칸이 1 개도 안 되므로 모든 점수가 0 → conf 0.25 를 넘는 박스가 없다.

<<CODE 6-c1 | Penn-Fudan · YOLOv8n · 탐지 함수 준비>>
<<CODE 6-c2 | 보정 43장 / 평가 127장 · YoloReader>>
<<CODE 6-1 | YOLOv8n 전체 INT8 — 사람 0명, output0 scale 이 너무 크다>>

#### 개념 2 · 무엇을 빼고 무엇을 남기나
- **남김(INT8)**: cv2.i(박스 갈래) · cv3.i(클래스 갈래)의 Conv — 계산량의 대부분.
- **뺌(FP16 · ORT 에서는 FP32)**: DFL(Reshape · Transpose · Softmax · 1×1 conv) · Sigmoid · 앵커 디코딩(Slice · Sub · Add · Div · Mul) · Concat.
- 노드 수: FP32 233개 → head 제외 INT8 732개 (Q 218 · DQ 281 이 더해짐).

<<CODE 6-2 | Detect head 후처리를 빼고 INT8 — 사람을 다시 찾는다>>

#### 한 번에 정리
- 탐지 출력 = 범위가 다른 값(박스 · 점수)을 한 텐서에 → 통째 양자화하면 점수 0.
- Detect head 후처리(24노드)만 빼고 무거운 Conv 는 INT8.
- 보정 43장 / 평가 127장 — 겹치지 않게.
- ⚠ `trt_check` 합격 ≠ 결과가 맞음 — 사람 수 · 박스로 기능을 점검한다.

---

### 7교시 · 정확도 검증: 보정 / 평가 분리 · mAP

**키워드**: `AP@0.5`, `mAP@0.5:0.95`, `conf 0.01`, `eval_map()`, `보정 / 평가 분리`, `make_infer()`, `같은 detect · 같은 AP 함수`, `AP50 −0.5%p · mAP −2.3%p`

**개요**: 탐지 모델 정확도는 **AP@0.5**(IoU 0.5 기준 AP)와 **mAP@0.5:0.95**(IoU 0.5 ~ 0.95, 10단계 평균)로 잰다 — 박스를 낮은 conf(0.01)까지 모두 받아 precision-recall 곡선의 면적을 구한다(VB Day 4 함수). 검증 규칙 3가지: 평가는 **보정에 쓰지 않은 사진**, FP32 · FP16 · INT8 을 **같은 함수 · 같은 사진**, 기준은 **FP32 대비 몇 %p**. 평가 127장: FP32 AP50 0.9719 · mAP 0.8375 → INT8 0.9666 · 0.8149 (−0.53 · −2.26%p — 교안 "AP50 1%p 안팎 · mAP 2~3%p"). 보정 43장으로 재면 INT8 − FP32 차이가 AP50 −1.49%p 로 평가 세트와 전혀 다르게 나온다 — 어느 사진으로 쟀는지가 결과를 바꾸므로 보정에 쓴 사진은 평가에서 뺀다. `make_infer` 하나로 엔진(TensorRT) · ORT 추론기를 같은 모양으로 바꿔 끼운다.

**한눈에 보기**
탐지 정확도 검증 — AP50 · mAP, 보정과 평가 분리
> conf 0.01 까지 박스를 남겨 정답과 짝지어 AP50 · mAP 를 재고, 보정에 쓰지 않은 사진으로만 평가한다
- **FP32 AP50 0.9719 · mAP 0.8375 → INT8 0.9666 · 0.8149**: AP50 −0.53%p, mAP −2.26%p
- **mAP 가 더 예민하다**: 박스가 조금만 어긋나도 IoU 0.9 · 0.95 기준에서 놓친다
- **보정 사진으로 재면 결론이 뒤집힌다**: 보정 43장에서는 AP50 −1.49%p · mAP −1.47%p
- **같은 detect · 같은 AP 함수 · 같은 사진, 엔진만 바꾼다**: `make_infer`
- **⚠ conf 0.25 로 자르면 AP 가 낮게 나온다**: 평가는 0.01 · 127장이라 1%p 는 사진 몇 장 차이일 수 있다

**핵심 용어**
- **AP@0.5 (AP50)**: IoU ≥ 0.5 를 맞힌 것으로 보고 구한 precision-recall 곡선의 면적
- **mAP@0.5:0.95**: IoU 기준 0.5, 0.55, …, 0.95 의 AP 10개 평균 — 박스 위치 정밀도까지 본다 (더 예민)
- **낙관 편향**: 보정에 쓴 사진으로 평가하면 그 사진에 맞춘 범위로 재는 셈이라 결과가 좋게 나올 수 있는 것 (이 실행에서는 오히려 낮게 나옴 — 어느 쪽이든 믿을 수 없는 숫자)
- **같은 모양의 추론기**: `x → output0` 함수 — ORT · TensorRT 어느 쪽이든 detect 에 그대로 꽂힌다

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `eval_map()` | `eval_map(infer, EVAL_IDX)` | `(AP50, mAP50-95)` (이 교시에서 만듦) |
| `make_infer()` | `make_infer("INT8", "yolov8n_int8.onnx", fp16=True, int8=True)` | GPU 면 엔진 TRTInfer, 아니면 ORT (FP16 은 None) (이 교시에서 만듦) |
| `evaluate_ap()` | `evaluate_ap(dets, gts, iou_thr=0.5)` | AP (VB D4 함수, 제공) |
| `map_50_95()` | `map_50_95(dets, gts)` | IoU 0.5 ~ 0.95 AP 평균 (제공) |

**핵심 내용**

#### 탐지 정확도 — AP50 · mAP
> 사진마다 conf 0.01 까지 박스 (N, 5) → 정답 박스와 짝지어 PR 곡선 → 면적 = AP · IoU 10단계 평균 = mAP
- **FP32 (평가 127장)** → AP50 **0.9719** · mAP **0.8375**
- **INT8 head 제외** → AP50 **0.9666** (−0.53%p) · mAP **0.8149** (−2.26%p)
- **더 예민한 쪽** → mAP — 박스 위치가 조금만 어긋나도 IoU 0.9 · 0.95 기준에서 놓친다
- ⚠ conf 0.25 로 자르면 PR 곡선의 낮은 점수 쪽이 잘려 AP 가 낮게 나온다 — 평가는 0.01

#### 보정 / 평가 분리
> 보정에 쓴 사진으로 재면 그 사진에 맞춘 범위로 재는 셈 — 평가는 보정에 쓰지 않은 사진으로
- **평가 127장** → INT8 − FP32 = AP50 **−0.53%p** · mAP −2.26%p
- **보정 43장** → INT8 − FP32 = AP50 **−1.49%p** · mAP −1.47%p (INT8 0.9422 / 0.8086 · FP32 0.9571 / 0.8233) — 세트를 바꾸면 차이가 뒤집힌다
- **규칙** → 같은 detect · 같은 AP 함수 · 같은 사진, 엔진만 바꾼다 (`make_infer`)
- ⚠ 평가 사진이 127장뿐 — 1%p 안팎의 차이는 사진 몇 장 차이일 수 있다

**기본 사용형태**
```python
def eval_map(infer, idx):
    dets = []
    for i in idx:
        b, s = detect(imgs[i], infer, conf=0.01)
        dets.append(np.concatenate([b, s[:, None]], 1))      # (N, 5)
    g = [gts[i] for i in idx]
    return evaluate_ap(dets, g), map_50_95(dets, g)

def make_infer(prec, onnx_path, fp16=False, int8=False):
    if TRT_OK:
        return TRTInfer(build_once(onnx_path, f"yolov8n_{prec.lower().replace('-', '_')}.engine", fp16=fp16, int8=int8))
    return None if prec == "FP16" else ort_infer(onnx_path)
```

**연결**
- [VB D4·4교시 Precision · Recall · AP · mAP](#/vb/4/4) — 오늘 쓰는 AP 함수를 만든 곳
- [OT D5·3교시 ORT vs TensorRT FP32 vs FP16](#/ot/5/3) — 같은 사진 · 같은 함수로 런타임만 바꿔 비교한 같은 방법
- [QP D2·4교시 Static 양자화](#/qp/2/4) — 보정 데이터는 현장 분포로 · 오늘은 평가와도 분리

**상세 학습**

#### 이 교시에서 답하는 질문
6교시에서 사진 한 장으로 "사람을 다시 찾는다" 는 것을 확인했다. 하지만 배포를 결정하려면 숫자 하나로 정확도를 말해야 한다. 탐지 모델의 정확도는 무엇으로, 어떤 사진으로 재야 할까? 이 교시는 AP50 · mAP 를 재는 함수와, 보정 사진으로 재면 왜 안 되는지를 숫자로 확인한다.

#### 개념 1 · AP 를 구하는 순서
1. 평가 사진마다 `detect(conf=0.01)` → 박스 + 점수 (N, 5).
2. 모든 박스를 점수 높은 순으로 줄 세우고, 정답 박스와 IoU ≥ 기준이면 TP (정답 하나에 하나만).
3. 누적 TP 로 precision · recall → PR 곡선 → 면적 = AP.
4. 기준을 0.5 하나로 = AP50, 0.5 ~ 0.95 10단계 평균 = mAP.

<<CODE 7-1 | eval_map — AP50 · mAP, 보정 / 평가 세트 비교>>

#### 개념 2 · 네 가지 측정
| 모델 | 세트 | AP50 | mAP |
|---|---|---|---|
| FP32 | 평가 127장 | 0.9719 | 0.8375 |
| INT8 (head 제외) | 평가 127장 | 0.9666 | 0.8149 |
| INT8 (head 제외) | 보정 43장 | 0.9422 | 0.8086 |
| FP32 | 보정 43장 | 0.9571 | 0.8233 |

(직접 실행 · ORT CPU.) 세트끼리 절대값은 사진 난이도가 달라 비교하지 않는다. 같은 세트 안의 **INT8 − FP32 차이**를 본다: 평가 세트 AP50 −0.53%p · 보정 세트 −1.49%p — 사진 43장 · 127장의 차이로도 결론(mAP 2%p 한도 통과 여부)이 바뀔 수 있다. 그래서 평가 세트를 미리 정해 두고 바꾸지 않는다.

<<CODE 7-2 | make_infer — FP16 · INT8 추론기를 같은 모양으로>>

#### 한 번에 정리
- 탐지 정확도 = AP50 · mAP@0.5:0.95 (conf 0.01 까지 받아서).
- 평가는 보정에 쓰지 않은 사진, 같은 detect · 같은 AP 함수, FP32 대비 %p.
- YOLOv8n INT8(head 제외): AP50 −0.5%p · mAP −2.3%p — mAP 가 더 예민.
- ⚠ 보정 사진으로 재면 숫자가 달라진다 (이 실행 AP50 −1.49%p vs 평가 −0.53%p) — 평가 세트는 따로.

---

### 8교시 · 미니 프로젝트: INT8 배포 심사서

**키워드**: `엔진 4개 (FP32 · FP16 · INT8 · INT8-mixed)`, `mixed_nodes 31개`, `measure()`, `review_df`, `추천 규칙`, `deploy_review.csv`, `qp_baseline.csv`, `짝 IoU · 사람 수 차이`, `배포 장치에서 다시 측정`

**개요**: 상황 — 침입 감지 카메라(OT Day 5)를 GPU 장치에 배포한다. YOLOv8n 엔진 4개(FP32 · FP16 · INT8(head 제외) · INT8-mixed(head + 첫 Conv + 마지막 출력 Conv 6개 제외, 31노드))를 **같은 자**(지연 · FPS · AP50 · mAP · 크기)로 재서 심사표를 만들고, **규칙을 먼저 정해** 하나를 추천한다: AP50 ≥ FP32 − 0.01 이고 mAP ≥ FP32 − 0.02 인 엔진 중 가장 빠른 것. 추천 엔진은 FP32 와 박스를 직접 비교(짝 IoU · 사람 수)해 기능 점검까지 한다. 이 실행(ORT CPU)에서는 INT8(head 제외)이 mAP 한도(−2%p)를 넘어 탈락하고, INT8-mixed 는 통과하지만 S8 활성값 때문에 느려 **FP32 추천** — GPU(TensorRT)에서는 결과가 달라질 수 있으므로 배포 장치에서 다시 재야 한다.

**한눈에 보기**
INT8 배포 심사서 — 먼저 정한 규칙으로 고른다
> 엔진 4개를 지연 · FPS · AP50 · mAP · 크기로 재고, AP50 ≥ FP32 − 0.01 · mAP ≥ FP32 − 0.02 를 만족하는 것 중 가장 빠른 것을 고른다
- **INT8 은 mAP 0.8149 < 0.8175 로 탈락**: INT8-mixed 는 AP50 0.9734 · mAP 0.8279 로 통과
- **INT8-mixed = head 24 + 첫 Conv + 출력 Conv 6개 = 31노드 제외**: AP50 +0.69%p · mAP +1.30%p, 크기 3.43 → 3.51 MB
- **이 실행의 추천은 FP32 (29.9 ms)**: ORT CPU 에서 S8 INT8 은 79 ~ 90 ms 로 느리다 — GPU 가 없어 FP16 은 NaN
- **기능 점검 — 사람 수 · 짝 IoU**: 평가 20장에서 FP32 와 비교 — 형식이 맞아도 0명인 사고를 잡는다
- **⚠ TensorRT GPU 에서는 순위가 바뀔 수 있다**: 배포 장치에서 다시 빌드 · 측정

**핵심 용어**
- **배포 심사서**: 후보 엔진들을 같은 기준으로 잰 표 + 규칙에 따른 추천 + 근거
- **INT8-mixed**: head 후처리에 더해 민감한 첫 Conv · 마지막 출력 Conv 까지 FP16 으로 남긴 INT8 모델
- **선택 규칙**: 정확도 손실 한도를 먼저 정하고, 통과한 것 중 가장 빠른 것
- **기능 점검**: mAP 외에 대표 장면의 박스 · 사람 수를 직접 비교하는 확인

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `measure()` | `measure("INT8", "yolov8n_int8.onnx", True, True)` | `{"prec", "backend", "latency_ms", "fps", "AP50", "mAP", "size_mb"}` |
| `DataFrame.sort_values()` | `review_df[review_df["pass"]].sort_values("latency_ms").iloc[0].prec` | 통과한 엔진 중 가장 빠른 것 |
| `DataFrame.itertuples()` | `for r in review_df.itertuples(): …` | 행마다 qp_baseline 행 만들기 |

**핵심 내용**

#### 배포 심사 — 같은 자, 먼저 정한 규칙
> 엔진 4개를 지연 · FPS · AP50 · mAP · 크기로 재고 → AP50 ≥ FP32 − 0.01 · mAP ≥ FP32 − 0.02 중 가장 빠른 것
- **FP32** → 29.9 ms · AP50 0.9719 · mAP 0.8375 · 12.26 MB
- **INT8 / INT8-mixed** → 79.2 / 90.1 ms · AP50 0.9666 / **0.9734** · mAP **0.8149 ✗** / 0.8279 · 3.43 / 3.51 MB
- **이 실행의 추천** → INT8 은 mAP 0.8149 < 0.8175(FP32 − 0.02) 로 탈락 · FP32 · INT8-mixed 통과 → 가장 빠른 **FP32** (FP16 은 GPU 없음 → NaN)
- ⚠ ORT CPU 에서 S8 활성값 INT8 은 느리다 — TensorRT GPU 에서는 순위가 바뀔 수 있다 → 배포 장치에서 다시 빌드 · 측정

#### INT8-mixed · 기능 점검
> mixed = head 24 + 첫 Conv `/model.0/conv/Conv` + 출력 Conv 6개(`cv2.i.2` · `cv3.i.2`) = 31노드 제외
- **정확도** → INT8 대비 AP50 +0.69%p · mAP +1.30%p — mAP 한도를 다시 통과
- **대가** → 크기 3.43 → 3.51 MB · 이 CPU 에서 79 → 90 ms
- **기능 점검 (과제 5)** → 평가 처음 20장에서 FP32 vs 추천 엔진: 짝 IoU 중앙값 · 사람 수 차이 합 (FP32 추천이라 1.000 · 0)
- ⚠ mAP 만으로는 부족 — 6-1 처럼 형식이 맞아도 사람 0명인 사고를 기능 점검이 잡는다

**기본 사용형태**
```python
def measure(prec, onnx_path, fp16=False, int8=False):
    f = make_infer(prec, onnx_path, fp16, int8)
    if f is None:
        return {"prec": prec, "backend": BACKEND, **{k: np.nan for k in ["latency_ms", "fps", "AP50", "mAP", "size_mb"]}}
    x = to_input(imgs[EVAL_IDX[0]])[0]
    r = bench_fn(lambda: f(x), n=200 if TRT_OK else 10, warm=20 if TRT_OK else 2)
    a, m = eval_map(f, EVAL_IDX)
    return {"prec": prec, "backend": BACKEND, "latency_ms": r["mean"], "fps": r["fps"],
            "AP50": a, "mAP": m, "size_mb": size_kb(onnx_path) / 1024}   # GPU 면 엔진 파일 크기

f32 = review_df.set_index("prec").loc["FP32"]
review_df["pass"] = (review_df.AP50 >= f32.AP50 - 0.01) & (review_df.mAP >= f32.mAP - 0.02)
recommend = review_df[review_df["pass"]].sort_values("latency_ms").iloc[0].prec
```

**연결**
- [QP D2·8교시 미니 프로젝트](#/qp/2/8) — 어제는 CPU 용 PTQ 설정 16가지, 오늘은 GPU 배포용 엔진 4개 · 같은 "규칙 → 선택" 형식
- [OT D5·6교시 결과 만들기: 영상 · 이벤트 · 성능표](#/ot/5/6) — 침입 감지 카메라 프로젝트, 오늘 엔진을 꽂을 곳
- [OT D4·8교시 실행기 4가지 성능표](#/ot/4/8) — 엔진 여러 개를 한 표로 비교한 과제

**상세 학습**

#### 이 교시에서 답하는 질문
오늘 배운 것(범인 찾기 · mixed precision · TensorRT 조건 · head 제외 · 분리 평가)을 하나의 결정으로 묶는다. "어떤 엔진을 배포할까?" — 느낌이 아니라 표와 규칙으로 답하고, 그 결정이 실제 장면에서도 맞는지(박스 · 사람 수) 확인한다.

#### 개념 1 · 과제 구성 (100점)
| 과제 | 내용 | 배점 |
|---|---|---|
| 1 | INT8-mixed 모델 (`mixed_nodes` 31개 제외) | 20 |
| 2 | `measure(...)` — 지연 · FPS · AP50 · mAP · 크기 | 20 |
| 3 | 엔진 4개 심사표 `review_df` | 20 |
| 4 | 규칙대로 추천 → `deploy_review.csv` · `qp_baseline.csv` (day="Day3", acc = AP50) | 20 |
| 5 | FP32 vs 추천 엔진 박스 비교 (짝 IoU 중앙값 · 사람 수 차이 합) | 10 |
| 6 | 심사서 의견 (마크다운, 강사 채점) | 10 |

정답 보기 셀 없음 · GPU 가 없으면 FP16 행은 비우고 ORT CPU 값으로 채점.

<<CODE 과제 1 | 과제 1 — INT8-mixed: 첫 Conv · 출력 Conv 도 제외>>
<<CODE 과제 2 | 과제 2 — measure: 지연 · AP · 크기>>

#### 개념 2 · 심사표 (이 실행)
![엔진 4개 심사표 — ORT CPU 에서 INT8 은 mAP 한도로 탈락, 통과한 둘 중 가장 빠른 FP32 추천](assets/fig/qp/d3-review.svg)

| 엔진 | 지연 ms | FPS | AP50 | mAP | 크기 MB | pass |
|---|---|---|---|---|---|---|
| FP32 | 29.9 | 33.4 | 0.9719 | 0.8375 | 12.26 | ✓ (추천) |
| FP16 | — | — | — | — | — | GPU 필요 |
| INT8 | 79.2 | 12.6 | 0.9666 | 0.8149 | 3.43 | ✗ (mAP) |
| INT8-mixed | 90.1 | 11.1 | 0.9734 | 0.8279 | 3.51 | ✓ |

(ORT CPU · 스레드 자동 · n=10 · 직접 실행. 정확도는 Day 3 교안 수정(`reduce_range=True`) 뒤 다시 잰 값, 지연 시간은 이전 실행 값 — 7비트 weight 는 속도를 바꾸지 않는다.) 교안의 의도는 T4 GPU + TensorRT — 거기서는 INT8 이 빠른 경로라 추천이 INT8 계열이 될 수 있다. 이 표의 지연 시간은 "CPU 에서는 S8 INT8 이 느리다" 는 것만 보여 준다.

<<CODE 과제 3 | 과제 3 — 엔진 4개 심사표>>
<<CODE 과제 4 | 과제 4 — 규칙대로 추천 · 저장>>

#### 개념 3 · 과제 6 의 생각할 거리
- **INT8 vs INT8-mixed 차이가 잡음보다 큰가**: mAP 차 1.3%p 가 합격 / 탈락을 갈랐다 — 평가 127장에서 박스 몇 개 — 사진을 늘리거나 여러 번 나눠 재서 흔들림을 본다.
- **전체 양자화했다면**: 6-1 — 형식(trt_check)은 합격인데 사람 0명. 심사서에 "대표 장면 사람 수 비교" 같은 기능 점검 항목을 넣는다.
- **Jetson 에 그대로?**: 엔진은 GPU 종류 · TensorRT 버전에 묶인다 → QDQ ONNX 를 가져가 장치에서 다시 빌드 · 측정한다.

<<CODE 과제 5 | 과제 5 — FP32 vs 추천 엔진 박스 비교>>

#### 한 번에 정리
- 배포 심사 = 같은 자(지연 · FPS · AP50 · mAP · 크기)로 재고, 먼저 정한 규칙으로 고른다.
- INT8-mixed = head + 첫 Conv + 출력 Conv 6개 제외 → INT8 이 놓친 mAP 한도를 다시 통과.
- mAP 에 더해 박스 · 사람 수 기능 점검을 붙인다.
- ⚠ 엔진 · 지연 시간은 장치에 묶인다 — 배포 장치(GPU · Jetson)에서 다시 빌드 · 측정.
