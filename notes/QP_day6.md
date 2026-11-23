## Day 6 — 통합 경량화 파이프라인

### 1교시 · Knowledge Distillation: soft target · temperature · KD loss

**키워드**: `Knowledge Distillation`, `teacher / student`, `soft target`, `dark knowledge`, `temperature T`, `KD loss`, `T²`, `F.kl_div`

**개요**: 작은 모델(학생)을 정답(hard label)만으로 가르치지 않고, 큰 모델(선생님)이 낸 **출력 분포**까지 따라 하게 만드는 기법이다. 선생님은 "고양이 0.6 · 개 0.3 · 사슴 0.05" 처럼 클래스 사이의 닮음(dark knowledge)을 알려 준다. logits 를 temperature T 로 나눈 뒤 softmax 하면 작은 확률이 잘 보인다. 선생님은 학습하지 않고(`eval()` · `no_grad`), 배포 때는 학생만 쓴다.

**한눈에 보기**
Knowledge Distillation — 선생님의 soft target 을 따라 배우기
> 큰(또는 원본) 모델의 출력 분포를 temperature 로 부드럽게 펴서 작은 모델이 정답과 함께 따라 하게 한다 — loss = α·CE(학생, 정답) + (1 − α)·T²·KL(선생님‖학생)
- **softmax(z / T) — T 를 키우면 2 · 3등이 드러난다**: logits [4, 2, 0] → T=1 0.867 · 0.117 · 0.016 / T=4 0.506 · 0.307 · 0.186
- **T=1 이면 hard label 과 거의 같다**: 추가 정보가 없다 — T 가 너무 크면 모두 1/클래스 수에 가까워져 다시 사라짐
- **α 는 정답 비중, T² 는 기울기 보정**: 오늘 α 0.5 · T 4 — soft 항의 기울기가 1/T² 로 작아지는 것을 되돌린다
- **예 — CE 0.6216 → KD loss 0.7202**: 학생과 선생님이 같으면 soft 항은 0
- **⚠ 선생님은 학습하지 않는다**: `eval()` + `torch.no_grad()`

**핵심 용어**
- **Knowledge Distillation(지식 증류)**: 큰 모델(teacher)의 출력을 작은 모델(student)이 따라 하도록 학습하는 기법
- **soft target**: `softmax(teacher_logits / T)` — 선생님이 낸 확률 분포 (정답 하나짜리 hard label 의 반대)
- **dark knowledge**: 1등이 아닌 클래스에 준 확률 속의 정보 ("고양이는 개와 닮았다")
- **temperature(T)**: logits 를 나누는 수 — 클수록 분포가 평평해진다 (보통 2 ~ 8)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `soft()` | `soft(logits, T=1.0)` | `F.softmax(logits / T, dim=1)` — temperature softmax |
| `kd_loss()` | `kd_loss(s_logits, t_logits, y, T=4.0, alpha=0.5)` | α·CE + (1 − α)·T²·KL |
| `F.kl_div()` | `F.kl_div(log_p_student, p_teacher, reduction="batchmean")` | 두 분포의 KL 발산 (첫 인자는 **log 확률**) |

**핵심 내용**

#### Temperature softmax — soft target
> softmax(z / T) — T 를 키우면 분포가 평평해져 2 · 3등 클래스의 확률이 드러난다
- **식** → `p_i = exp(z_i / T) / Σ exp(z_j / T)`
- **예** → logits [4, 2, 0]: T=1 → 0.867 · 0.117 · 0.016 (1등 외 0.133) / T=4 → 0.506 · 0.307 · 0.186 (0.494) / T=10 → 0.402 · 0.329 · 0.269 (0.598)
- **왜** → T=1 에서는 hard label 과 거의 같아 추가 정보가 없다
- ⚠ T 가 너무 크면 모두 1/클래스 수에 가까워져 정보가 다시 사라진다

#### KD loss
> loss = α·CE(학생, 정답) + (1 − α)·T²·KL(선생님‖학생) — 정답과 선생님을 함께 따라 한다
- **α** → 정답(CE) 비중 (오늘 0.5) · **T** → 오늘 4
- **T²** → soft 항의 기울기가 1/T² 로 작아지는 것을 되돌리는 보정
- **예** → logits s = [[2, .5, −1], [.1, .2, .3]] · t = [[1, 2, 0], [0, 0, 3]] · y = [0, 2] 이면 CE = 0.6216, T=4 · α=0.5 → loss = 0.7202
- ⚠ 선생님은 학습하지 않는다 (`eval()` + `torch.no_grad()`) · 학생이 선생님과 같으면 soft 항은 0 → loss = α·CE

**기본 사용형태**
```python
def soft(logits, T=1.0):
    return F.softmax(logits / T, dim=1)

def kd_loss(s_logits, t_logits, y, T=4.0, alpha=0.5):
    ce = F.cross_entropy(s_logits, y)
    kl = F.kl_div(F.log_softmax(s_logits / T, dim=1), soft(t_logits, T), reduction="batchmean")
    return alpha * ce + (1 - alpha) * T * T * kl
```

**연결**
- [QP D5·6교시 Fine-tuning · iterative pruning](#/qp/5/6) — 자른 모델을 정답(CE)만으로 다시 학습하던 것 → 오늘은 선생님을 추가
- [NN D3·5교시 Affine 계층](#/nn/3/5) — logits → softmax → CE 의 출발점

**상세 학습**

#### 이 교시에서 답하는 질문
Day 5 에서 가지치기한 모델은 정답(CE)만으로 다시 학습했다. 선생님(원본)이 이미 있는데 정답만 보여 주는 것은 아깝다. 선생님의 "무엇과 헷갈리는지"까지 넘겨 주면 같은 예산으로 더 잘 회복할 수 있다 — 그 장치가 KD loss 다.

#### 개념 1 · soft target 이 주는 정보
| | hard label | soft target (T=4) |
|---|---|---|
| 내용 | "고양이" 1개 | 고양이 0.6 · 개 0.3 · 사슴 0.05 … |
| 알려 주는 것 | 정답 | 정답 + 클래스 사이의 닮음 |

<<CODE 1-c2 | 선생님 logits 를 T 로 나누면 분포가 부드러워진다>>
<<CODE 1-1 | soft · T 별 1등 외 확률>>

#### 개념 2 · KD loss 의 두 항
| 기호 | 뜻 | 오늘의 값 |
|---|---|---|
| α | CE 항의 비중 | 0.5 |
| T | temperature | 4 |
| T² | KL 기울기 보정 | 16 |
| KL | 선생님 분포 → 학생 분포의 차이 | `F.kl_div(학생 log_softmax(·/T), 선생님 softmax(·/T), "batchmean")` |

<<CODE 1-c1 | KD loss 식>>
<<CODE 1-2 | kd_loss — CE + T² · KL>>

#### 한 번에 정리
KD = 학생이 정답(CE)과 선생님의 soft target(KL)을 함께 따라 한다. T 가 클수록 1등 외 확률이 드러난다. 학습 때만 선생님이 필요하고 배포 비용은 늘지 않는다. ⚠ `F.kl_div` 의 첫 인자는 log 확률 — 확률을 넣으면 값이 틀린다.

### 2교시 · Pruning + KD: 가지치기한 모델을 선생님에게 배우게 하기

**키워드**: `Pruning + KD`, `원본 = 선생님`, `train_kd`, `같은 예산 비교`, `KD 이득`, `rn20_p50_kd.pth`

**개요**: 가지치기에서는 선생님이 공짜로 생긴다 — **자르기 전의 원본**이 선생님이다. 원본 ResNet-20(0.846, MACs 41.2M)을 채널 50% 자르면 정확도 ≈ 0.10 · MACs 10.5M. 이 학생을 같은 데이터 10,000장 · 3 epoch · lr 1e-2 로 학습하되 **loss 만** 바꿔 비교한다 (fine-tuning: CE / KD: α·CE + (1 − α)·T²·KL). KD 는 step 마다 선생님 forward 가 한 번 더 들어 학습이 조금 느리다.

**한눈에 보기**
Pruning + KD — 자르기 전 원본이 선생님
> 따로 큰 모델을 학습할 필요 없이, 가지치기 전 원본 모델을 선생님으로 삼아 자른 모델의 회복을 KD 로 한다
- **Day 5 fine-tuning 0.10 → 0.7225**: 50% · 3 epoch
- **같은 예산으로 KD 0.75**: fine-tuning 0.72 보다 높다 (강의 요약 값, 실행마다 조금 다름)
- **덜 자를수록 KD 이득은 작다**: 20% 모델은 둘 다 0.78 초과
- **⚠ 선생님 정확도는 그대로여야 한다**: `acc_torch(rn20) == FP32_ACC` — 선생님이 학습되면 잘못된 구현

**핵심 용어**
- **Pruning + KD**: structured pruning 으로 자른 모델을 원본 모델이 가르쳐 회복시키는 조합
- **같은 예산 비교**: 같은 학생 · 같은 데이터 · 같은 epoch · lr — 달라지는 것은 loss 뿐

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `train_kd()` | `train_kd(student, teacher, epochs=EPOCHS, lr=1e-2, T=4.0, alpha=0.5, on_epoch=None)` | `finetune` 과 같은 구조 + 선생님 · `kd_loss`, 걸린 초를 돌려준다 |
| `prune_channels()` | `prune_channels(rn20, 0.5)` | 채널 50% structured pruning (Day 5) |
| `copy.deepcopy()` | `p50_kd = copy.deepcopy(p50)` | 같은 출발점의 복사본 만들기 |

**핵심 내용**

#### Pruning + KD
> 자르기 전 원본이 선생님 — 따로 큰 모델을 학습할 필요 없이 가지치기 회복에 KD 를 쓴다
- **Day 5** → 50% · 3 epoch fine-tuning: 0.10 → 0.7225
- **오늘 (같은 예산)** → 50% 모델 fine-tuning 0.72 → KD 0.75 (강의 요약 값, 실행마다 조금 다를 수 있음)
- **덜 자를수록** → KD 이득은 작아진다 (20% 모델은 둘 다 0.78 초과)
- ⚠ 선생님 `rn20` 의 정확도는 그대로여야 한다 (`acc_torch(rn20) == FP32_ACC`) — 선생님이 학습되면 잘못된 구현

**기본 사용형태**
```python
p50 = prune_channels(rn20, 0.5)                  # 선생님 = 자르기 전의 rn20
p50_kd = copy.deepcopy(p50)
t_kd = train_kd(p50_kd, rn20)                    # 선생님은 eval · no_grad, loss = kd_loss
p50_ft = copy.deepcopy(p50); t_ft = finetune(p50_ft)
```

**연결**
- [QP D5·4교시 Structured pruning 직접 하기](#/qp/5/4) — `prune_channels` 의 원리
- [QP D5·6교시 Fine-tuning · iterative pruning](#/qp/5/6) — 비교 기준 (CE 만)

**상세 학습**

#### 이 교시에서 답하는 질문
1교시의 KD loss 를 실제 가지치기 모델에 적용하면 정말 더 잘 회복하나? 공정하게 비교하려고 학생 · 데이터 · epoch 를 모두 고정하고 loss 만 바꾼다.

#### 개념 1 · 선생님이 공짜인 이유
원본 ResNet-20 ──prune 50%──▶ 학생. 자르기 전 모델이 그대로 남아 있으므로 선생님을 따로 학습할 필요가 없다.

<<CODE 2-c1 | Pruning + KD 의 흐름>>
<<CODE 2-1 | train_kd — 50% 모델을 선생님에게 배우게>>
<<CODE 2-2 | 같은 예산 fine-tuning vs KD>>
<<CODE 2-c2 | 20% 모델로도 비교>>

#### 한 번에 정리
같은 예산이면 KD 가 fine-tuning 보다 같거나 낫다. KD 는 학습 때만 선생님이 필요하고, 배포되는 것은 학생뿐이다. ⚠ 학습 시간은 선생님 forward 만큼 늘어난다.

### 3교시 · 순서의 문제: Prune → KD → INT8 → TensorRT

**키워드**: `Prune → KD → INT8 → TensorRT`, `구조 → 회복 → 양자화 → 엔진`, `to_int8`, `need_qat`, `채널별 amax`, `compress()`

**개요**: 배운 기법을 한 모델에 차례로 적용한다. 순서는 ① Prune(구조 확정) → ② KD(정확도 회복) → ③ INT8(PTQ, 손실이 1%p 를 넘으면 QAT) → ④ QDQ ONNX → TensorRT. 양자화를 먼저 하고 가지치기를 하면, 채널 수는 줄었는데 양자화기의 채널별 scale(amax)은 예전 개수 그대로라 forward 에서 **크기가 안 맞는다는 RuntimeError** 가 난다 (실습 3-2).

**한눈에 보기**
경량화 순서 — Prune → KD → INT8 → TensorRT
> 양자화 범위(scale)는 최종 weight · 활성값으로 정해야 하므로 구조를 먼저 바꾸고(prune), 회복한 뒤(KD), 마지막에 양자화하고 엔진을 만든다
- **구조가 먼저**: 채널 수가 바뀌면 weight · 양자화 범위가 모두 다시 정해져야 한다
- **회복이 다음**: scale 은 회복이 끝난 weight 로 정해야 맞다
- **PTQ 먼저, QAT 는 손실이 1%p 를 넘을 때만**: Day 4 의 판정 규칙
- **⚠ 양자화 → 가지치기 순서면 오류**: 출력 채널은 줄었는데 `weight_quantizer.amax` 개수는 그대로 → `RuntimeError: size of tensor …`

**핵심 용어**
- **경량화 순서**: 구조(prune) → 회복(KD) → 양자화(INT8) → 엔진(TensorRT)
- **need_qat**: PTQ 손실이 1%p(0.01)를 넘으면 QAT 를 더한다 (Day 4 기준)
- **채널별 scale(amax)**: weight 양자화기가 채널 수만큼 들고 있는 값

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `to_int8()` | `to_int8(model)` | 복사본을 ModelOpt INT8 PTQ (`mtq.quantize(…, INT8_DEFAULT_CFG, forward_loop)`) |
| `compress()` | `compress(model, teacher, ratio=0.5, epochs=EPOCHS, out="compressed_int8.onnx")` | prune → KD → INT8 → QDQ ONNX 한 번에 (제공) |
| `export_b1()` | `export_b1(copy.deepcopy(q), "rn20_p50_kd_int8.onnx")` | batch=1 고정 ONNX (TensorRT 용) |
| `trt_check()` | `trt_check("rn20_p50_kd_int8.onnx")` | TensorRT 로 빌드 가능한지 · QDQ 노드 수 |

**핵심 내용**

#### 순서: 구조 → 회복 → 양자화 → 엔진
> 양자화 범위(scale)는 최종 weight · 활성값으로 정해야 하므로 양자화는 마지막
- **왜 구조 먼저** → 채널 수가 바뀌면 weight · 양자화 범위가 모두 다시 정해져야 한다
- **왜 회복 다음** → scale 은 회복이 끝난 weight 로 정해야 맞다
- **PTQ 먼저** → QAT 는 PTQ 손실이 1%p 를 넘을 때만
- ⚠ 양자화 → 가지치기 순서: conv1 weight 의 출력 채널은 줄었는데 `weight_quantizer.amax` 개수는 그대로 → `RuntimeError: size of tensor …`

**기본 사용형태**
```python
def compress(model, teacher, ratio=0.5, epochs=EPOCHS, out="compressed_int8.onnx"):
    m = prune_channels(model, ratio)       # ① 구조를 먼저 확정
    train_kd(m, teacher, epochs=epochs)    # ② 정확도 회복 (KD)
    q = to_int8(m)                         # ③ 양자화 — PTQ (손실이 크면 QAT)
    export_b1(copy.deepcopy(q), out)       # ④ QDQ ONNX → TensorRT
    return m, q
```

**연결**
- [QP D4·5교시 NVIDIA ModelOpt PTQ](#/qp/4/5) — `mtq.quantize` 사용법
- [QP D4·7교시 QAT 모델 배포: QDQ ONNX · TensorRT INT8](#/qp/4/7) — QDQ ONNX 내보내기
- [QP D3·4교시 TensorRT INT8 개념](#/qp/3/4) — QDQ(explicit) 방식

**상세 학습**

#### 이 교시에서 답하는 질문
Day 3 ~ 5 에서 PTQ · QAT · 가지치기를 따로 배웠다. 한 모델에 모두 적용할 때 어느 것을 먼저 해야 하나? 순서를 바꾸면 무엇이 깨지나?

#### 개념 1 · 순서의 이유
| 단계 | 하는 일 | 앞 단계가 바뀌면 |
|---|---|---|
| ① Prune | 채널 수 확정 | — |
| ② KD | weight 회복 | prune 뒤에야 회복할 대상이 정해진다 |
| ③ INT8 | scale 결정 | 최종 weight 가 아니면 scale 이 어긋난다 |
| ④ 엔진 | QDQ ONNX → TensorRT | — |

<<CODE 3-c1 | 순서: Prune → KD → INT8 → TensorRT>>
<<CODE 3-1 | to_int8 — KD 모델을 ModelOpt INT8 로 · ONNX>>
<<CODE 3-2 | 양자화 먼저 → 가지치기 하면 오류>>
<<CODE 3-c2 | compress — 네 단계를 한 함수로>>

#### 한 번에 정리
순서 = 구조 → 회복 → 양자화 → 엔진. 가지치기 + KD 모델도 8비트는 PTQ 로 충분했지만(need_qat False), 항상 재 보고 판정한다. ⚠ 양자화 뒤에 채널을 자르면 채널별 scale 이 맞지 않아 깨진다.

### 4교시 · 탐지 모델 경량화: YOLOv8n INT8 · INT8-mixed 준비

**키워드**: `YOLOv8n`, `재학습 없이 가능한 것`, `INT8 PTQ (QDQ)`, `mixed precision`, `Detect head 제외`, `배포 영상으로 보정`, `CalibrationDataReader`, `엔진 3종`

**개요**: 오후에는 분류(ResNet-20)에서 탐지(YOLOv8n)로 넘어가 OT Day 5 침입 감지 파이프라인에 경량화한 엔진을 넣는다. 재학습 없이 할 수 있는 것은 FP16 · INT8 PTQ · INT8 mixed precision. QAT · structured pruning · KD 는 COCO 학습 데이터와 수 시간이 필요해 오늘은 하지 않는다 (12월 프로젝트에서 자기 데이터로). 보정 데이터는 **실제로 돌릴 영상**(people.mp4)의 앞부분 프레임(0 ~ 479, 10프레임마다 48장)이고, 평가는 480 프레임 이후 — 겹치지 않는다.

**한눈에 보기**
탐지 모델 경량화 — 재학습 없이 되는 것부터
> 보정 데이터만 있으면 되는 FP16 · INT8 PTQ · mixed precision 은 YOLOv8n 에 바로 쓰고, 학습 데이터가 필요한 QAT · pruning · KD 는 나중으로 미룬다
- **FP16 은 플래그 하나, INT8 PTQ 는 보정 데이터만**: OT Day 5 · QP Day 3 방법 그대로
- **QAT · pruning · KD 는 COCO 학습 데이터 · 수 시간**: 오늘은 하지 않는다
- **INT8 = head 후처리 제외, INT8-mixed = + 첫 Conv + 출력 Conv 6개**: 무거운 Conv 는 INT8, 예민한 곳은 FP16
- **확인 — qdq 수 mixed < INT8, 사람 수 FP16 과 ±2 이내**: 프레임 530 기준
- **⚠ 보정 프레임(0 ~ 479)과 평가 구간(480 ~)이 겹치면 평가가 부풀려진다**

**핵심 용어**
- **INT8-mixed**: Detect head 후처리 + 첫 Conv + 마지막 출력 Conv 6개까지 FP16 으로 남긴 INT8
- **배포 영상 보정**: 보정 데이터를 배포 환경의 장면에서 뽑는다 (평가 구간과 분리)
- **엔진 3종**: FP16(기준) · INT8(head 제외) · INT8-mixed

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `read_frames()` | `read_frames(idx)` | 영상에서 번호가 idx 인 프레임들을 읽는다 (제공) |
| `FrameReader` | `class FrameReader(CalibrationDataReader)` | 보정 프레임을 한 장씩 `{"images": …}` 로 돌려주는 reader |
| `make_yolo_int8()` | `make_yolo_int8("yolov8n_int8.onnx", FrameReader(cal_frames), exclude=head)` | YOLOv8n static PTQ (exclude = FP 로 남길 노드) |
| `make_infer()` | `make_infer("yolov8n_int8.onnx", fp16=True, int8=True)` | 엔진을 만들어 `infer(x)` 함수로 (제공) |

**핵심 내용**

#### 재학습 없이 되는 것 / 안 되는 것
> 보정 데이터만 있으면 되는 기법(FP16 · INT8 PTQ · mixed)은 오늘, 학습 데이터가 필요한 기법(QAT · pruning · KD)은 나중에
- **FP16** → 플래그 하나 (OT Day 5) · **INT8 PTQ** → 보정 데이터만 (Day 3) · **mixed** → 예민한 층을 더 제외
- **QAT · pruning · KD** → COCO 학습 데이터 · 수 시간 → 오늘은 하지 않음
- ⚠ 보정 프레임(0 ~ 479)과 평가 구간(480 ~)이 겹치면 평가가 부풀려진다

#### INT8 / INT8-mixed 엔진
> 무거운 Conv 는 INT8, 예민한 곳(head 후처리 · 첫 Conv · 출력 Conv)은 FP16
- **INT8** → `exclude = head` (Detect head 중 cv2 · cv3 갈래 Conv 를 뺀 노드)
- **INT8-mixed** → `exclude = head + ["/model.0/conv/Conv"] + last_convs` (출력 Conv 6개)
- **확인** → `trt_check` 의 qdq 수: mixed < INT8 · 프레임 530 의 사람 수가 FP16 과 ±2 이내

**기본 사용형태**
```python
CAL_IDX = list(range(0, EVAL_START, 10))          # 48장
cal_frames = read_frames(CAL_IDX)

class FrameReader(CalibrationDataReader):
    def __init__(self, frames):
        self.it = iter([{"images": to_input(f)[0]} for f in frames])
    def get_next(self):
        return next(self.it, None)                # 다 쓰면 None
```

**연결**
- [QP D3·6교시 탐지 모델 양자화: YOLOv8n · Detect head](#/qp/3/6) — head 를 왜 제외하나
- [QP D3·3교시 Mixed Precision: 범인만 FP32 로](#/qp/3/3) — exclude 로 일부를 남기는 방식
- [OT D5·2교시 YOLOv8n 엔진 만들기](#/ot/5/2) — 엔진 · 전 / 후처리

**상세 학습**

#### 이 교시에서 답하는 질문
분류 모델에서 쓴 기법을 탐지 모델에 어디까지 쓸 수 있나? 학습 데이터가 없을 때 할 수 있는 것과 보정 데이터는 어디서 가져와야 하나?

#### 개념 1 · 재학습 필요 여부
| 기법 | 재학습 | 오늘 YOLOv8n |
|---|---|---|
| FP16 | 불필요 | ✅ 기준 엔진 |
| INT8 PTQ (QDQ) | 불필요 (보정만) | ✅ |
| INT8 mixed | 불필요 | ✅ |
| QAT · pruning · KD | 필요 | ❌ |

<<CODE 4-c1 | YOLOv8n · people.mp4 · 침입 감지 함수 준비>>
<<CODE 4-1 | 배포 영상 프레임으로 보정 데이터 · FrameReader>>
<<CODE 4-2 | INT8 · INT8-mixed 모델과 엔진 3종>>

#### 한 번에 정리
재학습 없이: FP16 · INT8 PTQ · mixed. 보정은 배포할 장면의 영상으로, 평가 구간과 분리. 엔진 3종 = FP16 · INT8 · INT8-mixed. ⚠ 학습이 필요한 기법은 학습 데이터가 있어야 한다.

### 5교시 · 침입 감지 파이프라인 교체: 엔진만 바꾼다

**키워드**: `infer 를 인자로`, `read → pre → infer → post`, `time_stages`, `run_intrusion`, `intrusion_*.mp4`, `events_*.csv`, `Amdahl`

**개요**: OT Day 5 의 파이프라인은 추론기(`infer`)를 인자로 받는 구조라 엔진만 바꿔 끼우면 된다. 프레임 시간 = read(CPU) + pre(CPU) + **infer(GPU, 양자화로 빨라지는 곳은 여기뿐)** + post(CPU). 단계별 시간을 재고(`time_stages`), 엔진마다 영상 · 이벤트 CSV 를 만든다(`run_intrusion`).

**한눈에 보기**
침입 감지 파이프라인 — 엔진만 교체
> read → pre → infer → post → 그리기 · 저장 구조에서 바뀌는 곳은 infer 한 곳 — FP16 · INT8 · INT8-mixed 엔진을 갈아 끼워 영상과 이벤트를 각각 만든다
- **빨라지는 단계는 infer 뿐**: read · pre · post 는 CPU 그대로
- **산출물 — 영상 3개 + 이벤트 CSV 3개**: `intrusion_{fp16,int8,int8_mixed}.mp4` · `events_*.csv`
- **같은 함수 · 같은 영상으로**: 엔진만 다르게 해야 비교가 된다 (OT Day 5 와 같은 구조)
- **⚠ 추론이 빨라질수록 나머지 단계의 비율이 커진다**: 6교시 Amdahl

**핵심 용어**
- **엔진 교체**: 전 · 후처리는 그대로 두고 `infer` 만 바꾼다
- **infer_share**: 프레임 시간 중 추론이 차지하는 비율
- **warm-up**: 측정 전에 몇 번 먼저 돌려 초기 지연을 빼는 것

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `time_stages()` | `time_stages(cap, infer, conf=0.25)` | `{"read","pre","infer","post"}` 각 ms |
| `run_intrusion()` | `run_intrusion(source, out_path, infer, start=0, max_frames=None, tag=None)` | 영상 → 프레임마다 처리 · 저장 · 로그 → DataFrame |
| `time.perf_counter()` | `t0 = time.perf_counter()` | 단계 사이 시간 재기 |
| `cv2.VideoWriter()` | `cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))` | 결과 영상 저장 |

**핵심 내용**

#### 엔진만 교체하는 구조
> `read → pre → infer(엔진) → post → 그리기 · 저장` — 바뀌는 곳은 infer 한 곳
- **빨라지는 단계** → infer 만 (read · pre · post 는 CPU)
- **산출물** → `intrusion_{fp16,int8,int8_mixed}.mp4` · `events_*.csv` 3종
- ⚠ 추론이 빨라질수록 나머지 단계의 비율이 커진다 → 6교시 Amdahl

**기본 사용형태**
```python
for name, f in ENG.items():
    tag = name.lower().replace("-", "_")
    LOGS[name] = run_intrusion(VIDEO, f"intrusion_{tag}.mp4", f, start=SEG[0], tag=tag)
```

**연결**
- [OT D5·4교시 시간 나눠 재기](#/ot/5/4) — 전처리 · 추론 · 후처리 시간
- [OT D5·5교시 미니 프로젝트 ① 이식: 추론기만 교체](#/ot/5/5) — 같은 구조의 첫 사용

**상세 학습**

#### 이 교시에서 답하는 질문
엔진 3종을 만들었는데 실제 파이프라인에 어떻게 꽂고, 양자화는 프레임 시간 중 어느 부분을 줄이나?


#### 개념 1 · 엔진만 바꾸는 파이프라인 — 바뀌는 곳은 infer 한 곳
프레임 한 장은 **영상 읽기 → 전처리 → 추론 → 후처리** 네 단계를 지난다. 추론 함수(infer)를 인자로 받게 만들어 두면 FP16 · INT8 · INT8-mixed 엔진을 바꿔 끼우는 것만으로 같은 파이프라인을 돌릴 수 있다 (OT Day 5 와 같은 구조). 양자화가 줄이는 것은 이 중 **추론** 하나뿐이라, 단계별 시간을 나눠 재야 실제 이득이 보인다.

<<CODE 5-c1 | 프레임 한 장의 네 단계>>
<<CODE 5-1 | time_stages — 네 단계 시간 · 엔진별 평균>>
<<CODE 5-2 | run_intrusion — 엔진만 바꿔 침입 감지 · 이벤트>>

#### 한 번에 정리
infer 를 인자로 받으면 엔진 교체만으로 끝난다. 프레임 시간 = read + pre + infer + post. ⚠ 양자화가 줄이는 것은 infer 뿐.

### 6교시 · 결과 비교: 이벤트 일치율 · 병목의 이동

**키워드**: `GT 없는 평가`, `기준 엔진 FP16`, `프레임 일치율`, `이벤트 일치`, `agreement`, `match_events`, `Amdahl 의 법칙`

**개요**: 현장 영상에는 정답(GT)이 없어 mAP 를 못 잰다. 대신 **기준 엔진(FP16)과 얼마나 같은 결정을 내리는가**를 본다 — 프레임 수준(n_people · n_inside · alarm 이 같은 프레임 비율)과 이벤트 수준(FP16 이벤트마다 겹치는 이벤트가 있나 · 시작 시각 차이). 박스 몇 개가 달라도 경보 · 이벤트가 같으면 서비스에는 같은 결과다. 또 추론이 빨라질수록 병목이 전 · 후처리로 옮겨간다.

**한눈에 보기**
결과 비교 — 이벤트 일치율 · 병목의 이동
> 정답(GT)이 없는 현장 영상은 기준 엔진과 같은 결정을 내리는지(프레임 · 이벤트 일치율)로 평가하고, Amdahl 식으로 추론 가속이 전체 속도에 주는 효과를 계산한다
- **프레임 수준 — 사람 수(엄격) · 구역 안 · 경보**: 서비스에 영향을 주는 것은 경보
- **이벤트 수준 — 재현율 + 시작 시각 차이**: 경보가 늦게 켜지면 mAP 가 비슷해도 문제
- **Amdahl — 전체 향상 = 1 / ((1 − p) + p / k)**: p = 0.5 면 추론 2배 → 1.33배 · 10배 → 1.82배, 최대 2배
- **실제 p 는 내 실행의 추론 비율로 다시 계산**: `stage_df.infer_share["FP16"]`
- **⚠ 지표는 서비스가 정한다**: 사람 수 일치율이 낮아도 경보 · 이벤트는 거의 같을 수 있다

**핵심 용어**
- **프레임 일치율**: 기준 로그와 값이 같은 프레임의 비율 (n_people 가 가장 엄격, alarm 이 서비스에 중요)
- **이벤트 재현율**: FP16 이벤트 중 겹치는 이벤트가 있는 비율
- **Amdahl 의 법칙**: 일부(비율 p)만 k배 빨라지면 전체는 `1 / ((1 − p) + p / k)`

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `agreement()` | `agreement(ref, log)` | `{"n_people","n_inside","alarm"}` 일치 프레임 비율 |
| `match_events()` | `match_events(ref, ev, tol=0.5)` | 기준 이벤트마다 matched · start_diff_s |

**핵심 내용**

#### GT 없는 현장 평가 — 일치율
> 기준 엔진과 같은 결정을 내리나 — 프레임 수준 + 이벤트 수준
- **프레임** → n_people(엄격) · n_inside · alarm(서비스에 영향)
- **이벤트** → 재현율 + 시작 시각 차이
- ⚠ 사람 수 일치율이 낮아도 경보 · 이벤트는 거의 같을 수 있다 — 반대로 mAP 가 비슷해도 이벤트 시작이 늦으면 문제 (지표는 서비스가 정한다)

#### Amdahl — 병목의 이동
> 전체 속도 향상 = 1 / ((1 − p) + p / k) — p 는 추론이 차지하는 비율, k 는 추론 속도 향상
- **예** → p = 0.5 이면 추론 2배 → 1.33배 · 4배 → 1.60배 · 10배 → 1.82배 (최대 1 / (1 − p) = 2배)
- ⚠ 실제 p 는 `stage_df.infer_share["FP16"]` — 내 실행 값으로 다시 계산한다 (위 p = 0.5 는 계산 예)

**기본 사용형태**
```python
p = float(stage_df.infer_share["FP16"])
amdahl = lambda k: 1 / ((1 - p) + p / k)       # 추론 k배 → 프레임 amdahl(k)배
```

**연결**
- [QP D3·7교시 정확도 검증: 보정 / 평가 분리 · mAP](#/qp/3/7) — GT 가 있을 때의 평가 (mAP)
- [OT D5·6교시 결과 만들기: 영상 · 이벤트 · 성능표](#/ot/5/6) — 이벤트 CSV 의 출처

**상세 학습**

#### 이 교시에서 답하는 질문
정답이 없는 영상에서 INT8 엔진이 쓸 만한지 어떻게 판단하나? 추론이 빨라졌는데 전체 FPS 가 기대만큼 안 오르는 이유는?


#### 개념 1 · 정답 없는 평가 — 기준 엔진과의 일치율
현장 영상에는 정답(GT)이 없어 mAP 를 못 잰다. 대신 **FP16 엔진을 기준**으로, INT8 엔진이 프레임마다 같은 결정(사람 수 · 구역 안 · 경보)을 내렸는지(프레임 일치율), FP16 의 경보 구간을 놓치지 않았는지(이벤트 재현율)를 본다. 추론이 빨라질수록 전 · 후처리가 차지하는 비율이 커진다 (Amdahl).

<<CODE 6-1 | agreement — FP16 기준 프레임 일치율>>
<<CODE 6-2 | match_events — 이벤트 수준 재현율>>
<<CODE 6-c1 | 단계별 시간 누적 막대 — 병목의 이동>>

#### 한 번에 정리
GT 가 없으면 기준 엔진과의 일치율(프레임 · 이벤트)로 평가한다. 추론이 빨라지면 병목은 전 · 후처리로 옮겨간다. ⚠ 추론 속도만 보고 전체 속도를 약속하지 않는다.

### 7교시 · 배포 판단: 최종 qp_baseline · 경량화 결정표

**키워드**: `qp_baseline.csv`, `경량화 결정표`, `Jetson Orin`, `Raspberry Pi 5`, `MCU (Cortex-M)`, `재학습 없는 것 먼저`, `실제 장치에서 다시 재기`

**개요**: 6일 동안의 결과를 `qp_baseline.csv` 한 표에 모으고(오늘 5행: 분류 2 + 침입 감지 3), 장치에 따라 기법을 고른다. 선택 순서 — ① 목표를 숫자로(지연 · 크기 · 정확도 하한) ② 재학습 없이: FP16 → INT8 PTQ → mixed (목표를 만족하면 끝) ③ 부족하면 재학습: structured pruning → KD → INT8 PTQ → (손실 1%p 초과 시) QAT ④ 항상 실제 장치 · 파이프라인에서 다시 잰다 (MACs ≠ 지연, 추론 ≠ 프레임 전체).

**한눈에 보기**
배포 판단 — 장치가 기법을 정한다
> GPU 가 있으면 TensorRT INT8, CPU 만 있으면 INT8 + pruning, MCU 는 전부 INT8 — 목표를 숫자로 먼저 정하고, 재학습 없는 기법부터 차례로 시도한다
- **재학습 없이 — FP16 → INT8 PTQ → mixed precision**: 보정 데이터만 있으면 된다
- **재학습 — structured pruning → KD → INT8 PTQ → QAT**: QAT 는 PTQ 손실이 1%p 를 넘을 때만
- **목표를 만족하면 거기서 끝**: 더 줄이는 것이 목적이 아니다
- **최종 qp_baseline.csv**: Day 2 ~ 6 결과를 같은 열로 모아 결정표를 만든다
- **⚠ MACs ≠ 지연, 추론 ≠ 프레임 전체**: 실제 장치에서 다시 잰다

**핵심 용어**
- **경량화 결정표**: 장치별 런타임 · 정밀도 · pruning · KD 추천표 (`decision_table.csv`)
- **qp_baseline.csv**: 크기 · 지연 · p95 · FPS · 정확도(또는 일치율)를 누적한 결과표

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `recommend()` | `recommend(dev)` | 장치 정보 → `{"runtime","precision","pruning","kd"}` |
| `pd.read_csv()` | `final_df = pd.read_csv("qp_baseline.csv")` | 누적 결과표 읽기 |

**핵심 내용**

#### 장치가 기법을 정한다
> GPU 가 있으면 TensorRT INT8, CPU 만 있으면 INT8 + pruning, MCU 는 전부 INT8
| 장치 | 런타임 | 쓸 수 있는 것 |
|---|---|---|
| Jetson Orin | NVIDIA GPU · TensorRT | FP16 · INT8(QDQ) · structured pruning · (2:4 sparsity) |
| Raspberry Pi 5 | ARM CPU · ONNX Runtime / TFLite | INT8(static PTQ) · structured pruning — CPU 는 MACs 가 곧 시간 · FP16 이득 없음 |
| MCU (Cortex-M) | TFLite Micro · 수백 KB | 전부 INT8 · 아주 작은 모델 — pruning · KD 필수 |
- ⚠ MACs ≠ 지연, 추론 ≠ 프레임 전체 → 실제 장치에서 다시 잰다

#### 기법 선택 순서
> 재학습 없는 것 먼저 → 부족하면 prune → KD → INT8 → (1%p 초과) QAT
- **재학습 없이** → FP16 → INT8 PTQ → mixed precision
- **재학습** → structured pruning → KD → INT8 PTQ → QAT
- ⚠ 목표를 숫자로 먼저 — 목표를 만족하면 거기서 끝

**기본 사용형태**
```python
final_df = pd.read_csv("qp_baseline.csv")
display(final_df[final_df.day == "Day6"])         # 오늘 5행
```

**연결**
- [QP D2·7교시 성능 측정: qp_baseline.csv](#/qp/2/7) — 표의 시작
- [QP D3·8교시 미니 프로젝트: INT8 배포 심사서](#/qp/3/8) — 규칙 → 선택 형식

**상세 학습**

#### 이 교시에서 답하는 질문
결과가 쌓였다. 이제 "내 장치에는 무엇을 쓰나?" 를 어떤 순서로 정하나?


#### 개념 1 · 결과는 한 표에, 선택은 장치 순서로
6일 동안 `qp_baseline.csv` 에 쌓은 크기 · 정확도 · 지연을 한 표로 보고, 장치 조건(NVIDIA GPU 유무 · 메모리)에 따라 런타임과 정밀도를 고른다. 재학습이 없는 방법(FP16 → INT8 PTQ)부터 시도하고, 부족할 때 pruning · KD · QAT 로 넘어간다.

<<CODE 7-1 | 최종 qp_baseline.csv>>
<<CODE 7-2 | recommend — 장치별 경량화 결정표>>

#### 한 번에 정리
결과는 한 표에, 선택은 "재학습 없는 것 → 재학습" 순서로, 장치가 기법을 정한다. ⚠ 항상 실제 장치에서 다시 잰다.

### 8교시 · 미니 프로젝트: 평가 대비 예제 2문항 (쉬움 · 중간)

**키워드**: `평가 대비`, `static PTQ 비교표`, `structured pruning 30%`, `ptq_df`, `prune_df`, `eval_practice.csv`, `소스 기준 부분 점수`

**개요**: 파트 2 평가(ONNX · TensorRT + Quantization · Pruning)의 실습과 같은 수준의 예제 2문항을 혼자 푼다. 과제 1(쉬움, 30점): ONNX 모델 static PTQ 후 크기 · 정확도 · 지연 비교표 `ptq_df`. 과제 2(중간, 40점): structured pruning 30% + 3 epoch 학습 후 결과표 `prune_df`. 과제 3(20점): `eval_practice.csv` · 요약 `answer`. 과제 4(10점): 마크다운 결론. 제출은 소스 기준 · 기능별 부분 점수 — 끝까지 못 해도 실행되는 곳까지 제출한다. 정답 보기 셀은 없다.

**한눈에 보기**
평가 대비 — 비교표 + 숫자로 쓴 결론
> 쉬움 = static PTQ 비교표, 중간 = pruning + 학습 결과표 — 마지막은 표의 숫자로 쓴 한 줄 결론
- **과제 1 — static PTQ**: `eval_fp32.onnx` · `eval_int8.onnx` 의 크기 · 정확도 · 지연
- **과제 2 — 30% 가지치기 + 3 epoch**: `prune_channels(·, 0.3)` → finetune 또는 train_kd → MACs · 정확도 · 지연
- **과제 3 — answer 사전**: ptq_size_x · ptq_acc_drop_pp · prune_macs_x · prune_acc_drop_pp · pass_3pp
- **과제 4 — 결론 4문항**: INT8 배포 판단 · CPU 에서 QDQ INT8 이 느린 이유 · 3%p 기준 · 양자화가 마지막인 이유
- **⚠ 끝까지 못 해도 만든 곳까지 실행되는 상태로 제출**: 부분 점수

**핵심 용어**
- **ptq_df**: 열 model · size_kb · acc · latency_ms, 행 FP32 · INT8
- **prune_df**: 열 model · macs_m · params · acc · latency_ms, 행 FP32 · pruned 30% · pruned 30% + tuned
- **기준**: 30% 모델은 FP32 대비 3%p 이내 (Day 5 기준)

**핵심 내용**

#### 평가 실습 — 표 + 수치 한 줄 결론
> 쉬움 = static PTQ 비교표 · 중간 = pruning + 학습 결과표 · 마지막은 숫자로 쓴 결론
- **과제 1** → static PTQ → `eval_fp32.onnx` · `eval_int8.onnx` → 크기 · 정확도 · 지연
- **과제 2** → `prune_channels(·, 0.3)` → 3 epoch(`finetune` 또는 `train_kd`) → MACs · 정확도 · 지연
- **과제 3** → `answer = {ptq_size_x, ptq_acc_drop_pp, prune_macs_x, prune_acc_drop_pp, pass_3pp}`
- **과제 4 결론 4문항** → INT8 배포 판단 · CPU 에서 QDQ INT8 이 느린 이유 · 30% 모델의 3%p 기준 · 양자화가 마지막인 이유(3-2 오류)와 INT8 엔진 배포 전 확인 수치(6 · 7교시)
- ⚠ 끝까지 못 해도 만든 곳까지 실행되는 상태로 제출 — 부분 점수

**기본 사용형태**
```python
p30 = prune_channels(rn20, 0.3)
p30_tuned = copy.deepcopy(p30)
finetune(p30_tuned)                                # 또는 train_kd(p30_tuned, rn20)
```

**연결**
- [QP D2·8교시 미니 프로젝트: 최적 PTQ 설정 찾기](#/qp/2/8) — 과제 1 의 바탕
- [QP D5·8교시 미니 프로젝트: 가지치기 비율 곡선](#/qp/5/8) — 과제 2 의 바탕

**상세 학습**

#### 이 교시에서 답하는 질문
이틀 뒤 평가에서는 필답 + 실습을 본다. 지금까지의 도구를 어떤 순서로 써서 "표 + 결론" 한 세트를 만드나?


#### 개념 1 · 평가 실습 — 비교표 두 개와 숫자 결론
과제 1 은 static PTQ 전후(FP32 · INT8)의 크기 · 정확도 · 지연 표, 과제 2 는 채널 30% 가지치기 전 · 직후 · 재학습 뒤의 MACs · 정확도 · 지연 표를 만든다. 마지막에 두 표를 숫자 요약(`answer`)으로 정리한다.

<<CODE 과제 1 | 과제 1 — static PTQ 후 FP32 vs INT8>>
<<CODE 과제 2 | 과제 2 — 채널 30% + 3 epoch 학습>>
<<CODE 과제 3 | 과제 3 — 저장 · 요약>>

#### 한 번에 정리
쉬움: static PTQ → 크기 · 정확도 · 지연 표. 중간: structured pruning → 학습 → 결과표. 표 + 숫자 한 줄 결론까지가 한 세트. ⚠ 못 끝내도 실행되는 상태로.
