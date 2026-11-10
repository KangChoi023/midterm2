## Day 5 — 탐지 모델 TensorRT · 종합 프로젝트: 위험구역 침입 감지

### 1교시 · Day 4 복습 · 함수 → 클래스 TRTInfer

**키워드**: `TRTInfer 클래스`, `__init__ = 준비(한 번)`, `__call__ = 실행(여러 번)`, `build_once`, `GPU 버퍼 · 주소`, `execute_async_v3`

**개요**: 어제 `trt_runner()` 는 안쪽 함수 `run` 이 엔진 · 버퍼를 붙잡아 두는 방식이었다. 오늘은 같은 일을 **클래스**로 바꾼다. 준비물(엔진 · Context · 입출력 이름 · GPU 버퍼 · 주소)은 `__init__` 에서 **한 번**, 실행(복사 → 실행 → 기다리기 → NumPy)은 `__call__` 에서 **여러 번**. 이렇게 하면 `infer(x)` 처럼 함수 부르듯 쓸 수 있어 Day 5 프로젝트에서 그대로 가져다 쓴다. `build_once` 는 엔진 파일이 이미 있으면 다시 만들지 않는다(빌드가 오래 걸리므로).

**핵심 용어**
- **TRTInfer**: TensorRT 엔진 파일을 열어 `infer(x)` 로 실행해 주는 클래스 — 준비는 `__init__`, 실행은 `__call__`
- **`__call__`**: 객체 뒤에 `()` 를 붙여 부르면 실행되는 특별 메서드 — `mnv3(x1)` 처럼 함수처럼 씀
- **`build_once`**: `.engine` 이 없을 때만 ONNX → 엔진으로 빌드하는 헬퍼

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `TRTInfer()` | `TRTInfer("mnv3s_fp32.engine")` | 엔진 파일 → 함수처럼 쓰는 추론기 |
| `build_once()` | `build_once("a.onnx", "a.engine", fp16=True)` | 엔진 파일이 없을 때만 빌드 |
| `get_tensor_mode()` | `engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT` | 텐서 이름이 입력인지 출력인지 구분 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 1-1 | `TRTInfer` 클래스 정의 — `__init__` · `__call__` · `__repr__` |
| 1-2 | mnv3s 를 export → 엔진 → `TRTInfer` 로 실행, ORT 결과와 최대 차이 확인 (입력 `(1, 3, 224, 224)` → 출력 `(1, 1000)`) |

**핵심 내용**

#### 함수 → 클래스: 준비 한 번, 실행 여러 번
> `TRTInfer(engine_path)` — 비싼 준비(엔진 읽기 · 버퍼 · 주소)는 만들 때 한 번, 이후 `infer(x)` 는 복사 → 실행 → 기다리기 → NumPy 만
- **`__init__`** → 엔진 읽기 · Context · 입출력 이름 · GPU 버퍼 · 주소 등록 (한 번)
- **`__call__(x)`** → 입력 복사 → `execute_async_v3` → `synchronize` → 출력을 NumPy 로
- **왜** → 준비물이 `self` 에 모여 읽기 쉽고, 프레임마다 준비를 다시 하지 않음
- ⚠ 입력은 `float32` · 연속 메모리(`np.ascontiguousarray`) — 아니면 값이 깨지거나 오류

#### build_once: 엔진은 있으면 다시 만들지 않는다
> 엔진 빌드는 수십 초 걸린다 → 파일이 있으면 건너뛴다
- **핵심 성질** → 빌드 결과(`.engine`)는 만든 GPU · TensorRT 버전에서만 유효
- ⚠ ONNX 를 바꿨는데 같은 이름의 오래된 엔진이 남아 있으면 옛 모델이 실행됨 — 파일을 지우고 다시 빌드

**기본 사용형태**
```python
mnv3 = TRTInfer("mnv3s_fp32.engine")        # 준비 (한 번)
y = mnv3(x1)                                # 실행 (여러 번) → NumPy (1, 1000)
```

**연결**
- [OT D1·4교시 ONNX Runtime 으로 실행](#/ot/1/4) — 같은 입력으로 두 실행기를 비교하는 습관
- [VB D5·5교시 ONNX export · ONNX Runtime](#/vb/5/5) — mnv3s ONNX 의 출처

**상세 학습**

#### 이 교시에서 답하는 질문
Day 4 의 `trt_runner()` 는 동작하지만 엔진 · 버퍼가 안쪽 함수에 숨어 있어 확인하기 어렵다. 프로젝트에서는 같은 추론기를 프레임마다 수천 번 부른다 → 준비를 한 번만 하고 `infer(x)` 로 쓰는 모양이 필요하다.

#### 개념 1 · 준비 단계와 실행 단계
| 구분 | 어디서 | 무엇을 | 몇 번 |
|---|---|---|---|
| 준비 | `__init__` | 엔진 · Context · 이름 · 버퍼 · 주소 | 1번 |
| 실행 | `__call__` | 복사 → 실행 → 대기 → NumPy | 프레임마다 |

#### 코드로 확인
```python
mnv3 = TRTInfer("mnv3s_fp32.engine")
ref = ort.InferenceSession("mnv3s_b1.onnx", providers=CPU).run(None, {"input": x1})[0]
print(mnv3, "| 최대 차이", f"{np.abs(mnv3(x1) - ref).max():.1e}")
```
- **무엇을 하나**: TensorRT 결과와 ORT 결과가 같은지 최대 차이로 확인
- **어떤 개념**: 개념 1 — 같은 추론기 모양으로 두 실행기를 바꿔 끼움
- **출력**: 교안 기대값 — `TRTInfer(in=['input'] (1, 3, 224, 224), out=['logits'])` · 출력 shape `(1, 1000)` (GPU 가 있는 Colab 에서 확인)

#### 한 번에 정리
- `TRTInfer` = 준비(`__init__`) 한 번 + 실행(`__call__`) 여러 번 → `infer(x)` 로 사용
- `build_once` = 엔진은 오래 걸리니 있으면 재사용
- ⚠ 엔진은 GPU · TensorRT 버전에 묶임 — 다른 장치에서는 `.onnx` 로 다시 빌드

---

### 2교시 · YOLOv8n 엔진 만들기 · VB 전처리 · 후처리 연결

**키워드**: `yolov8n_fp16.engine`, `(1, 3, 640, 640) → (1, 84, 8400)`, `letterbox`, `decode_yolo`, `detect(frame, infer)`, `infer 바꿔 끼우기`

**개요**: VB Day 5 에서 export 한 `yolov8n.onnx`(입력 `images (1, 3, 640, 640)`, 출력 `output0 (1, 84, 8400)`)를 **FP16 엔진**으로 바꾼다. 앞의 `letterbox` 와 뒤의 `decode_yolo`(NMS 포함)는 VB 함수를 그대로 쓴다 — 엔진이 바뀌어도 앞뒤 코드는 그대로다. `detect(frame, infer)` 는 추론기를 **인자로 받아서** ORT · TensorRT 어느 쪽이든 끼울 수 있다.

**핵심 용어**
- **`output0 (1, 84, 8400)`**: YOLOv8n 의 원시 출력 — 84 = 박스 4 + 클래스 80, 8400 = 후보 위치 수
- **letterbox**: 비율을 유지한 채 640×640 에 맞추고 남는 곳을 회색 패딩으로 채우는 전처리 (VB Day 4)
- **`decode_yolo`**: 원시 출력 → 신뢰도 필터 → NMS → 박스 · 점수 (VB 함수)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `detect()` | `detect(frame, infer, conf=0.25, iou_thr=0.7)` | letterbox → `infer` → decode → 원래 좌표 박스 · 점수 |
| `iou_matrix()` | `iou_matrix(b_trt, b_ort).max(1)` | 두 박스 집합의 IoU 행렬 — 짝 IoU 비교용 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 2-1 | ONNX 노드 수 출력 → FP16 엔진 빌드 → `TRTInfer` 로 읽기, 출력 `(1, 84, 8400)` · 파일 크기(MB) 확인 |
| 2-2 | `detect` 정의 → ORT CPU · TensorRT FP16 로 같은 사진 탐지 → 짝 IoU 비교 (거의 같은 박스) |

**핵심 내용**

#### YOLOv8n 엔진의 입출력
> 입력 `images (1, 3, 640, 640)` → 출력 `output0 (1, 84, 8400)` — ONNX 때와 모양이 같다
- **84** → 박스 4(중심 x, y · 폭 · 높이) + 클래스 점수 80 (COCO)
- **8400** → 격자 세 겹의 위치 수 = 80² + 40² + 20² = 6400 + 1600 + 400
- **출력 크기** → 84 × 8400 = 705,600 개 float
- **Day 4 분류와 다른 점** → 출력이 곧 답이 아니라 **후처리(decode · NMS)** 가 필요
- ⚠ 엔진 출력도 "원시 텐서" — 박스로 읽으려면 VB 의 `decode_yolo` 가 있어야 함

#### detect(frame, infer): 추론기만 바꿔 끼운다
> letterbox → `infer(x)` → `decode_yolo` → letterbox 되돌리기 → 화면 안으로 자르기
- **전처리** → BGR→RGB · CHW · `/255` · `[None]` (Day 1 의 6단계 중 정규화 대신 `/255` 만)
- **좌표 되돌리기** → `(boxes − [pl, pt, pl, pt]) / r` — 패딩을 빼고 축소 비율로 나눔
- **infer** → `sess.run(...)` 을 감싼 `ort_infer` 든 `yolo16` 이든 `x → 출력` 모양이면 됨
- ⚠ 엔진마다 전처리를 다르게 하면 "실행기 비교" 가 아니라 "전처리 비교" 가 됨 — 같은 `detect` 를 쓴다

**기본 사용형태**
```python
ort_infer = lambda x: sess_cpu.run(None, {"images": x})[0]   # ORT 도 '같은 모양' 의 함수로
b_ort, s_ort = detect(frame200, ort_infer)
b_trt, s_trt = detect(frame200, yolo16)                      # 엔진만 다름
```

**연결**
- [VB D4·6교시 YOLO 구조: grid · 출력 텐서](#/vb/4/6) — 84 채널 · 8400 위치의 의미
- [VB D5·5교시 ONNX export · ONNX Runtime](#/vb/5/5) — 같은 `yolov8n.onnx` 와 `letterbox` · `decode_yolo`
- [OT D1·7교시 이미지 한 장 추론](#/ot/1/7) — 전처리는 ONNX(엔진) 밖이라는 원칙

**상세 학습**

#### 이 교시에서 답하는 질문
Day 4 에서는 분류 모델(출력 `(1, 1000)`)로 엔진을 만들었다. 탐지 모델은 출력이 `(1, 84, 8400)` 이고 앞(letterbox) · 뒤(decode · NMS) 코드가 필요하다. 이미 VB 에서 만든 앞뒤 코드를 그대로 두고 **가운데 추론기만** TensorRT 로 바꾸려면 어떻게 하나?

#### 개념 1 · 앞 · 가운데 · 뒤
| 구간 | 코드 | 바뀌나? |
|---|---|---|
| 앞 | `letterbox` · `/255` · CHW | 안 바뀜 (VB) |
| 가운데 | `infer(x)` | **ORT ↔ TensorRT** |
| 뒤 | `decode_yolo` · 좌표 되돌리기 | 안 바뀜 (VB) |

#### 코드로 확인
```python
build_once(YOLO_ONNX, "yolov8n_fp16.engine", fp16=True)
yolo16 = TRTInfer("yolov8n_fp16.engine")
print(yolo16, "| 출력", tuple(yolo16.buf[yolo16.outputs[0]].shape))
```
- **무엇을 하나**: FP16 엔진을 만들고 출력 모양을 확인
- **어떤 개념**: 개념 1 — 가운데 구간만 엔진으로 교체
- **출력**: 교안 기대값 — 입력 `images (1, 3, 640, 640)` · 출력 `(1, 84, 8400)` (파일 크기 MB 는 GPU 에서 확인)

#### 한 번에 정리
- 엔진 입출력은 ONNX 와 같다 — `(1, 3, 640, 640)` → `(1, 84, 8400)`
- `detect(frame, infer)` = 앞뒤는 VB 그대로, `infer` 만 교체
- ⚠ 사진 한 장이 같아 보여도 전체 성능은 같은지 다음 교시에서 숫자로 확인

---

### 3교시 · 정확도 확인: ORT vs TensorRT FP32 vs FP16

**키워드**: `AP50`, `mAP50-95`, `conf=0.01`, `짝 IoU`, `FP16 정확도`, `같은 데이터 · 같은 함수`

**개요**: 사진 한 장이 같아 보여도 전체 성능이 같은지는 숫자로 본다. VB Day 4 의 Penn-Fudan **170장** · 정답 박스 · `evaluate_ap` 로 세 실행기(ORT CPU · TRT FP32 · TRT FP16)를 같은 방법으로 평가한다. 정확도는 **전체 지표(AP)** 와 **박스 단위(짝 IoU)** 두 가지로 본다. 교안의 기대: FP32 는 ORT 와 AP50 차이 < 0.01, FP16 은 < 0.03 — FP16 으로 바꿔도 정확도는 거의 그대로.

**핵심 용어**
- **AP50**: IoU 0.5 기준 평균 정밀도 (VB Day 4)
- **mAP50-95**: IoU 0.5 ~ 0.95 를 0.05 간격으로 평균한 더 엄격한 지표
- **짝 IoU**: 두 실행기가 낸 박스를 서로 짝지어 본 IoU — 박스 단위 일치도

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `evaluate_ap()` | `evaluate_ap(dets, gts)` | 탐지 결과 · 정답 → AP50 (VB 함수) |
| `map_50_95()` | `map_50_95(dets, gts)` | IoU 임계값 10개 평균 mAP |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 3-1 | FP32 엔진 빌드 → `INFERS` 딕셔너리로 세 실행기를 170장에 돌려 AP50 · mAP50-95 · 박스 수 · 시간 표 |
| 3-2 | 사진 20장에서 FP16 박스 ↔ FP32 박스 짝 IoU (중앙값 > 0.95 기대) |

**핵심 내용**

#### 정확도 비교 규칙: 같은 사진 · 같은 detect · 같은 evaluate_ap
> 실행기 하나만 바꾸고 나머지는 모두 같게 → 차이는 실행기(정밀도)에서만 생긴다
- **AP 는 낮은 conf 로** → `conf=0.01` — 낮은 점수 박스까지 남겨야 recall 곡선이 끝까지 감
- **전체 지표 + 박스 단위** → AP 차이(전체)와 짝 IoU(박스별)를 함께 봄
- **기대 결과** → FP32 ≈ ORT (차이 < 0.01), FP16 ≈ ORT (차이 < 0.03)
- ⚠ `conf=0.25` 로 AP 를 재면 낮은 점수 박스가 잘려 AP 가 낮게 나옴 (실제 성능 아님)
- ⚠ FP32 엔진과 ORT 의 AP 가 다르면 대개 입력 dtype · BGR/RGB 실수 — 같은 사진 한 장의 출력부터 비교

**기본 사용형태**
```python
INFERS = {"ORT CPU": ort_infer, "TRT FP32": yolo32, "TRT FP16": yolo16}
for name, f in INFERS.items():
    dets = []
    for im in imgs:
        b, sc = detect(im, f, conf=0.01)                       # 낮은 conf — AP 용
        dets.append(np.concatenate([b, sc[:, None]], 1))
    print(name, evaluate_ap(dets, gts))
```

**연결**
- [VB D4·4교시 평가 지표: Precision · Recall · AP · mAP](#/vb/4/4) — AP 가 어떻게 계산되나
- [VB D4·8교시 직접 만든 mAP 로 탐지기 평가](#/vb/4/8) — 같은 170장 · `evaluate_ap`
- [OT D1·4교시 ONNX Runtime 으로 실행](#/ot/1/4) — "같은 입력이면 같은 출력" 확인 습관의 확장판

**상세 학습**

#### 이 교시에서 답하는 질문
2교시에서 사진 한 장은 거의 같은 박스가 나왔다. 하지만 한 장이 같다고 170장 전체가 같다는 보장은 없다. FP16 은 숫자를 절반 크기로 쓰므로 정확도가 떨어졌는지 같은 평가 함수로 확인한다.

#### 개념 1 · 두 가지 눈금
| 눈금 | 무엇을 보나 | 도구 |
|---|---|---|
| 전체 | 모든 사진 · 모든 박스의 성능 | AP50 · mAP50-95 |
| 박스 단위 | 같은 사람의 박스가 얼마나 겹치나 | 짝 IoU |

#### 코드로 확인
```python
pair_iou = []
for im in imgs[:20]:
    b32, _ = detect(im, yolo32); b16, _ = detect(im, yolo16)
    if len(b32) and len(b16):
        pair_iou += list(iou_matrix(b16, b32).max(1))
print(np.median(pair_iou), np.mean(np.array(pair_iou) >= 0.95))
```
- **무엇을 하나**: FP16 박스마다 가장 많이 겹치는 FP32 박스의 IoU 를 모음
- **어떤 개념**: 개념 1 — 박스 단위 눈금
- **출력**: 교안 기대값 — 짝 IoU 중앙값 > 0.95 (실제 숫자는 GPU 에서 확인)

#### 한 번에 정리
- 같은 데이터 · 같은 함수로 세 실행기를 비교 → FP32 = ORT, FP16 ≈ ORT
- 정확도는 AP(전체) + 짝 IoU(박스별) 두 눈금으로
- ⚠ AP 는 `conf=0.01` — `0.25` 로 재면 낮게 나옴

---

### 4교시 · 시간 나눠 재기: 전처리 · 추론 · 후처리 · 그리기

**키워드**: `네 조각`, `pre · infer · post · draw`, `FPS = 1000 ÷ 합`, `병목`, `warm-up`, `time.perf_counter`

**개요**: 엔진은 빨라졌다. 그런데 영상 **한 프레임**을 처리하는 시간도 그만큼 빨라졌을까? 한 프레임을 네 조각 — **전처리**(letterbox · 정규화) · **추론** · **후처리**(decode · 발 위치 · 구역) · **그리기** — 으로 나눠 ORT CPU 와 TensorRT FP16 을 비교한다. 추론이 빨라지면 CPU 에서 하는 나머지 세 조각이 **새 병목**이 된다.

**핵심 용어**
- **병목(bottleneck)**: 전체 시간 중 가장 큰 비중을 차지하는 조각 — 이것을 줄여야 전체가 빨라짐
- **warm-up**: 측정 전에 몇 번 미리 실행해 첫 실행 준비 시간을 제외하는 것
- **`time.perf_counter()`**: 짧은 시간 측정용 정밀 타이머

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `time_stages()` | `time_stages(frame, infer, conf=0.25)` | 한 프레임을 pre · infer · post · draw 네 조각(ms)으로 측정 |
| `time.perf_counter()` | `t0 = time.perf_counter()` | 정밀 시각 (차이 × 1000 = ms) |
| `idxmax()` | `share.loc[rt].idxmax()` | 값이 가장 큰 열 이름 → 병목 조각 이름 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 4-1 | `time_stages` 정의 → 영상 50프레임 (앞 5장 warm-up) 평균 → `df_st` (조각별 ms · total · fps) |
| 4-2 | 조각별 비중(%) → 가장 오래 걸린 조각(병목) 이름 · 누적 막대 그래프 |

**핵심 내용**

#### 한 프레임 = 네 조각의 합
> 프레임 시간 = 전처리 + 추론 + 후처리 + 그리기, **FPS = 1000 ÷ 프레임 시간(ms)**
- **식** → `total = pre + infer + post + draw` · `fps = 1000 / total`
- **예** → 프레임당 50 ms 이면 1000 ÷ 50 = 20 FPS
- **추론만 빨라지면** → 줄어드는 건 `infer` 한 조각뿐 — 나머지 세 조각은 그대로
- ⚠ 추론 시간만 보고 "FPS 가 이만큼 나온다" 고 말하면 틀림 — 실제는 네 조각의 합 (영상 읽기 · 저장은 이 표에 없지만 실제 FPS 에는 들어감)
- ⚠ warm-up 없이 첫 프레임부터 평균 → TensorRT 첫 실행 준비 시간이 섞임

#### 새 병목: 추론이 빨라지면 CPU 쪽이 커진다
> 가장 큰 조각을 찾아야(`idxmax`) 다음에 무엇을 고칠지 안다
- **비중(%)** → `조각 ÷ total × 100` — `share.loc[rt].idxmax()` 가 병목
- **TensorRT FP16 에서** → `infer` 비중이 작아지고 전처리 · 후처리 · 그리기(CPU) 비중이 상대적으로 커짐
- **다음 개선** → 병목 조각을 줄이기 (예: 전처리 GPU 로, 그리기 간소화) — 5일 안에서는 찾는 데까지
- ⚠ "빠른 엔진 = 빠른 프로그램" 아님 — 전체 속도는 가장 느린 조각이 정한다

**기본 사용형태**
```python
t0 = time.perf_counter()
x = preprocess_frame(frame)               # pre
t1 = time.perf_counter()
out = infer(x)                            # infer
t2 = time.perf_counter()
ms_infer = (t2 - t1) * 1000               # 구간 차이 × 1000 = ms
```

**연결**
- [VB D5·6교시 성능 측정법: baseline 성능표](#/vb/5/6) — 평균 · p95 · FPS 로 표를 만드는 법
- [VB D5·7교시 위험구역 침입 감지](#/vb/5/7) — 네 조각이 들어 있는 파이프라인

**상세 학습**

#### 이 교시에서 답하는 질문
추론이 몇 배 빨라졌다는 사실이 영상 처리 전체가 몇 배 빨라진다는 뜻은 아니다. 실제 시간이 어디에 쓰이는지 조각으로 나눠서 봐야 다음에 무엇을 고칠지 알 수 있다.

#### 개념 1 · 전체 속도는 느린 조각이 정한다
예 (설명용 숫자): 네 조각이 10 + 80 + 5 + 5 = 100 ms 라면 추론이 80 ms 로 가장 크다. 추론을 8 ms 로 줄이면 합은 10 + 8 + 5 + 5 = 28 ms — 추론은 10배 빨라졌지만 **전체는 100 ÷ 28 ≈ 3.6 배**.

| 상황 | 합 | FPS (1000 ÷ 합) |
|---|---|---|
| 추론 80 ms | 100 ms | 10 |
| 추론 8 ms | 28 ms | 약 35.7 |

#### 코드로 확인
```python
share = df_st[parts].div(df_st["total"], axis=0) * 100
bottleneck = {rt: share.loc[rt].idxmax() for rt in share.index}
print("가장 오래 걸린 조각:", bottleneck)
```
- **무엇을 하나**: 조각별 시간을 전체로 나눠 비중(%)으로 바꾸고 가장 큰 조각을 찾음
- **어떤 개념**: 개념 1 — 병목 찾기
- **출력**: 교안 기대값 — 실행기마다 `idxmax` 조각 이름 (GPU 에서 확인)

#### 한 번에 정리
- 프레임 시간 = 네 조각의 합, FPS = 1000 ÷ 합
- 추론이 빨라지면 CPU 쪽(전처리 · 후처리 · 그리기)이 새 병목
- ⚠ 설명 예의 숫자는 가정 — 실제 값은 `df_st` 로 측정

---

### 5교시 · 미니 프로젝트 ① 이식: 추론기만 교체

**키워드**: `process_frame`, `run_intrusion`, `debounce`, `streak >= n_consec`, `infer 하나만 교체`, `발 위치 · in_zone`, `프레임 로그 CSV`

**개요**: VB Day 5 의 위험구역 침입 감지기를 TensorRT 로 옮긴다. 바꾸는 곳은 탐지에 넘기는 **`infer` 하나** — 발 위치 · 구역 판정 · 3프레임 연속 경보(debounce) · 그리기는 VB 그대로. 배점(총 100): ① `process_frame` 20 · ② `run_intrusion` 20 · ③ 이벤트 요약 20 · ④ 성능 비교 20(③④는 6교시) · ⑤ 팀 발표 20(7교시, 강사 채점).

**핵심 용어**
- **debounce**: 구역 안에 누가 있는 프레임이 **연속** `n_consec` 번일 때만 경보 — 깜빡임 · 오탐 방지
- **streak**: 구역 안에 누군가 있는 연속 프레임 수 (없으면 0 으로 리셋)
- **프레임 로그 CSV**: 프레임마다 `frame · time_s · n_people · n_inside · alarm · ms` 를 적은 파일

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `process_frame()` | `process_frame(frame, infer, streak, poly, n_consec, conf, fps)` | 프레임 한 장: 탐지 → 발 위치 → 구역 → debounce → 그리기 → `(그린 프레임, info, 새 streak)` |
| `run_intrusion()` | `run_intrusion(source, out_path, infer, max_frames=None)` | 영상 전체를 읽으며 `process_frame` 반복 · 결과 영상 · 로그 CSV 저장 |
| `cv2.VideoWriter()` | `cv2.VideoWriter(path, fourcc, fps, (W, H))` | 프레임을 영상 파일로 저장 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 과제 1 | `process_frame(frame530, yolo16, streak=2)` → info · 새 streak 확인 |
| 과제 2 | `run_intrusion(VIDEO, "intrusion_trt.mp4", yolo16)` — 영상 전체 · 최근 30프레임 평균 FPS 표시 · 로그 CSV |

**핵심 내용**

#### process_frame: 프레임 한 장의 다섯 단계
> 탐지 → 발 위치 → 구역 안? → debounce → 그리기 — `infer` 만 넘겨받아 실행기에 무관
- **발 위치** → 박스 아래 중앙 `((x1+x2)/2, y2)` — 사람이 서 있는 땅 위 점
- **debounce** → `streak = streak + 1 if inside.any() else 0` → `alarm = streak >= n_consec`
- **예** → `streak=2` 로 들어와 이번 프레임에 구역 안에 사람이 있으면 새 streak = 3 → `n_consec=3` 이므로 경보 (alarm = 1)
- **반환** → `(vis, info, streak)` — `streak` 를 호출한 쪽이 들고 다음 프레임에 다시 넘김
- ⚠ `streak` 를 함수 안에서만 두면 프레임마다 0 으로 돌아감 — 반드시 반환해서 이어 받기

#### run_intrusion: 영상 루프 + 저장
> `VideoCapture` 로 읽고 → `process_frame` → `VideoWriter` 로 쓰고 → 로그를 CSV 로
- **영상 여는 곳은 한 곳** → `source` 인자 — 나중에 파일 대신 카메라 번호만 주면 됨
- **ms 기록** → 프레임마다 처리 시간을 로그에 남김 (6교시 성능표의 재료)
- **`max_frames`** → 앞부분만 돌려 ORT 와 비교할 때 (6교시)
- ⚠ `cap.read()` 가 `ok=False` 면 영상 끝 — `break` 하지 않으면 오류
- ⚠ `fourcc="mp4v"` 로 저장한 영상은 Colab 에서 바로 재생이 안 될 수 있음 → 6교시 `to_h264`

**기본 사용형태**
```python
vis, info, st = process_frame(frame530, yolo16, streak=2)
log_trt = run_intrusion(VIDEO, "intrusion_trt.mp4", yolo16)   # 추론기만 yolo16
```

**연결**
- [VB D5·7교시 종합 미니 프로젝트: 위험구역 침입 감지](#/vb/5/7) — `in_zone` · `foot_points` · debounce 의 원형
- [OT D1·8교시 미니 프로젝트](#/ot/1/8) — 4단계를 혼자 해 보는 같은 구성
- [OT D5·2교시 detect(frame, infer)](#/ot/5/2) — `infer` 를 인자로 받는 설계

**상세 학습**

#### 이 교시에서 답하는 질문
2 · 3교시에서 `detect` 가 어떤 `infer` 든 받는다는 것을 확인했다. 그렇다면 VB 의 전체 침입 감지기에서도 `infer` 만 `yolo16` 으로 바꾸면 나머지 코드를 건드리지 않고 TensorRT 로 옮길 수 있다.

#### 개념 1 · 바뀌는 것 한 곳
| 부분 | VB (ONNX Runtime) | OT (TensorRT) |
|---|---|---|
| 탐지기 | `detect_onnx` | `detect(frame, yolo16)` |
| 발 위치 · 구역 · debounce · 그리기 | 그대로 | 그대로 |
| 결과 | 영상 · 로그 | 영상 · 로그 (같은 형식) |

#### 코드로 확인
```python
streak = streak + 1 if inside.any() else 0
alarm = int(streak >= n_consec)
```
- **무엇을 하나**: 구역 안 프레임이 연속될 때만 경보 켬
- **어떤 개념**: process_frame — debounce
- **출력**: 직접 계산 — `streak=2` 에서 안에 사람 있음 → streak 3 → `n_consec=3` 이면 alarm 1, 사람 없음 → streak 0 → alarm 0

#### 한 번에 정리
- 바뀐 것은 `infer` 하나 — `yolo16` 을 넘겼을 뿐
- `process_frame` = 5단계 · `streak` 를 반환해 이어 받음, `run_intrusion` = 영상 루프 + 저장 + 로그
- ⚠ debounce 의 streak 는 호출 사이에 유지해야 함

---

### 6교시 · 미니 프로젝트 ② 결과 만들기: 영상 · 이벤트 · 성능표

**키워드**: `to_h264`, `find_events`, `intrusion_events.csv`, `ot_baseline.csv`, `추론만 vs 파이프라인 전체`, `bench_fn`

**개요**: 결과물 네 가지 — **결과 영상**(`intrusion_trt_h264.mp4`) · **프레임 로그**(`intrusion_trt_log.csv`) · **이벤트 요약**(`intrusion_events.csv`) · **성능표**(`ot_baseline.csv`). 이벤트는 경보가 연속으로 켜진 구간(`find_events`, VB Day 5 와 같음)이고, 성능표는 ① **추론만**(ORT CPU · TRT FP32 · TRT FP16 을 `bench_fn`) ② **파이프라인 전체**(ORT 는 앞 200프레임만, TRT FP16 은 전체) 다섯 줄을 `ot_baseline.csv` 에 이어 붙인다. 두 배수가 다른 이유가 4교시의 다음 병목이다.

**핵심 용어**
- **이벤트**: 경보가 연속으로 켜진 구간 — `(시작 초, 끝 초)`
- **추론만 vs 파이프라인**: 같은 성능표에 두 종류를 모두 기록 — 배수가 다른 이유 = 4교시의 병목
- **`ot_baseline.csv`**: 과목 내내 쌓는 성능 기준표 — 다음 과목(Quantization · Pruning)의 비교 기준

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `find_events()` | `find_events(alarm, fps)` | 0/1 경보 배열 → `(시작 초, 끝 초)` 목록 |
| `to_h264()` | `to_h264("a.mp4", "a_h264.mp4")` | `ffmpeg` 로 H.264 변환 → 브라우저 재생 가능 |
| `bench_fn()` | `bench_fn(lambda: f(x640), n=100, warm=5)` | 함수 한 번 실행 시간의 mean · p50 · p95 · fps |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 그림 셀 | `to_h264` 변환 · 시간별 "구역 안 사람 수" 와 경보 구간(빨강) 그래프 · 영상 재생 |
| 과제 3 | `find_events(alarm, V_FPS)` → `intrusion_events.csv` (`event · start_s · end_s · duration_s`) |
| 과제 4 | ORT CPU · TRT FP32 · TRT FP16 추론만 3줄 + 파이프라인 2줄 = 5줄을 `ot_baseline.csv` 에 이어 붙임 |

**핵심 내용**

#### find_events: 경보 구간 요약
> `alarm` 0/1 배열을 훑어 켜지는 순간(start)과 꺼지는 순간(마지막 켜진 프레임)을 구간으로 묶음
- **켜짐** → `a and start is None` → `start = i`
- **꺼짐** → `not a and start is not None` → `(start / fps, (i − 1) / fps)` 저장
- **끝까지 켜진 경우** → 반복이 끝난 뒤 `start is not None` 이면 마지막 프레임까지로 닫음
- **예** → fps = 10 에서 alarm = `[0, 1, 1, 1, 0, 0, 1]` → 구간 `(0.1, 0.3)` 과 `(0.6, 0.6)` (끝에서 닫힘)
- ⚠ 끝 프레임은 "경보가 꺼진 프레임" 이 아니라 **마지막으로 켜져 있던 프레임** (`i − 1`) — CSV 의 `duration_s` 는 `끝 − 시작 + 1/fps`

#### 추론만 vs 파이프라인 전체
> 같은 모델이라도 "추론만" 의 배수와 "영상 처리 전체" 의 배수는 다르다
- **추론만** → `bench_fn` 으로 letterbox 된 입력 한 장 (`n=30` ORT · `n=100` TRT, `warm=5`)
- **파이프라인** → `run_intrusion` 의 프레임당 시간(읽기 · 저장 포함) → `mean · p50 · p95 · fps`
- **다섯 줄** → `model` 열이 `yolov8n (추론만)` × 3 + `intrusion pipeline` × 2, `precision` 열로 FP32 / FP16 구분
- **왜 배수가 다른가** → 파이프라인에는 4교시의 CPU 쪽 조각(전처리 · 후처리 · 그리기 · 영상 읽기 · 저장)이 모두 들어 있어서
- ⚠ Day 4 의 `ot_baseline.csv` 에 `precision` 열이 없으면 `FP32` 로 채운 뒤 이어 붙임

**기본 사용형태**
```python
events = find_events(alarm, V_FPS)           # [(시작 초, 끝 초), ...]
pd.concat([old, final], ignore_index=True).to_csv("ot_baseline.csv", index=False)
```

**연결**
- [VB D5·7교시 위험구역 침입 감지](#/vb/5/7) — `find_events` 와 CSV 의 원형
- [VB D5·6교시 성능 측정법: baseline 성능표](#/vb/5/6) — `ot_baseline.csv` 의 가까운 조상
- [OT D5·4교시 시간 나눠 재기](#/ot/5/4) — 배수가 다른 이유(병목)

**상세 학습**

#### 이 교시에서 답하는 질문
5교시에서 만든 영상 · 로그를 사람이 읽는 결과(경보가 몇 번 · 언제 · 얼마나)와 성능 숫자(추론만 · 파이프라인)로 정리한다. 이 두 가지가 7교시 발표의 근거가 된다.

#### 개념 1 · 이벤트 = 켜진 구간
프레임 로그의 `alarm` 열(0/1)을 구간으로 묶으면 "몇 번 · 언제 · 얼마나" 가 된다.

| alarm (fps = 10) | 이벤트 |
|---|---|
| 0 1 1 1 0 0 1 | (0.1 s, 0.3 s) · (0.6 s, 0.6 s) |

#### 코드로 확인
```python
events = find_events(alarm, V_FPS)
for k, (a, b) in enumerate(events):
    print(k + 1, round(a, 1), round(b, 1), round(b - a + 1 / V_FPS, 1))
```
- **무엇을 하나**: 경보 구간마다 시작 · 끝 · 길이(초) 출력
- **어떤 개념**: 개념 1 — 이벤트 요약
- **출력**: 교안 기대값 — `intrusion_events.csv` 표 (영상에 따라 달라짐, GPU · 영상으로 확인)

#### 한 번에 정리
- 프레임 로그 → `find_events` → 이벤트 CSV
- 성능표에는 추론만과 파이프라인 전체를 모두 — 배수 차이의 원인 = 다음 병목
- ⚠ 이벤트의 끝은 마지막으로 켜진 프레임, 길이는 `+ 1/fps`

---

### 7교시 · 팀 발표 (과제 5 · 20점)

**키워드**: `3분 발표`, `경보 장면`, `최종 성능표`, `가장 오래 걸린 단계`, `숫자를 조건과 함께`

**개요**: 팀별 3분 발표 + 다른 팀에 질문 하나씩. 세 장면 — ① **결과 영상 1분**(경보가 울린 장면 하나: 누가 · 언제 · 왜 경보인지, 6교시 영상 · `intrusion_events.csv`) ② **최종 성능표 1분**(ORT CPU vs TensorRT FP16: 추론만 · 파이프라인 FPS, FP16 의 AP — 과제 4 표 · 3-1 표) ③ **가장 오래 걸린 단계 1분**(네 조각 중 가장 큰 것과 이유, 다음에 고칠 것 — 4-2 그래프). 채점(20점): 숫자를 조건과 함께 말했는가 5 · FP16 정확도 판단 근거 5 · 병목과 다음 개선안 5 · 질문에 대한 답 5.

**핵심 용어**
- **조건과 함께 말하기**: "FPS 가 몇" 이 아니라 "ORT CPU · 파이프라인 전체 · 배치 1 에서 몇 FPS"
- **판단 근거**: FP16 이 괜찮다는 말은 AP 차이 · 짝 IoU 로 뒷받침

**핵심 내용**

#### 발표 3분: 세 장면
> 무엇이 → 얼마나 → 왜 — 경보 장면 · 성능표 · 병목
| 장면 | 보여 줄 것 | 근거 |
|---|---|---|
| 1 · 결과 영상 (1분) | 경보 장면 하나 — 누가 · 언제 · 왜 | 영상 · `intrusion_events.csv` |
| 2 · 최종 성능표 (1분) | ORT CPU vs TRT FP16 — 추론만 · 파이프라인 FPS · FP16 의 AP | 과제 4 표 · 3-1 표 |
| 3 · 병목 (1분) | 네 조각 중 가장 큰 것 · 이유 · 다음에 고칠 것 | 4-2 그래프 |
- ⚠ 숫자만 읽지 말고 **조건**(실행기 · 정밀도 · 추론만/전체)을 함께 말해야 점수
- ⚠ "빨라졌다" 로 끝내지 말고 FP16 정확도(AP · 짝 IoU) 판단 근거를 붙임

**연결**
- [OT D5·3교시 정확도 확인](#/ot/5/3) — FP16 정확도 판단 근거
- [OT D5·4교시 시간 나눠 재기](#/ot/5/4) — 병목 그래프
- [OT D5·6교시 결과 만들기](#/ot/5/6) — 성능표 · 이벤트 CSV

**상세 학습**

#### 이 교시에서 답하는 질문
앞 6교시의 결과를 3분 안에 다른 사람이 이해하게 말해야 한다. 자기가 만든 숫자를 **조건과 함께** 말하고 **다음에 무엇을 고칠지** 제안하는 것이 이 교시의 목표다.

#### 개념 1 · 발표 메모 틀
| 칸 | 내용 |
|---|---|
| 경보 장면 | 누가 · 언제 · 왜 경보인지 |
| 성능표 핵심 숫자 | 조건 포함 |
| 가장 오래 걸린 단계 | 이유 · 다음 개선 |

#### 한 번에 정리
- 결과 영상 · 성능표 · 병목, 세 장면을 각각 1분
- 숫자는 조건과 함께, FP16 은 AP · 짝 IoU 로 근거 제시
- ⚠ 질문에도 답해야 한다 (5점)

---

### 8교시 · 5일 종합 퀴즈 · 제출 · 다음 과목

**키워드**: `5일 종합 퀴즈 30문항`, `trt_이름.zip`, `ot_baseline.csv`, `5일 지도`, `Quantization · Pruning`, `Jetson 에서 다시 빌드`

**개요**: 17:00 5일 종합 필기 퀴즈 30문항(Google Forms), 제출 파일 `trt_이름.zip`(네 파일 + 노트북 `.ipynb` 별도) 만들기, 5일 정리. 제출 파일: `ot_baseline.csv` · `intrusion_events.csv` · `intrusion_trt_log.csv` · `intrusion_trt_h264.mp4`. 다음 과목: **Quantization · Pruning**(INT8 이면 성능표가 어떻게 바뀔까 — `ot_baseline.csv` 가 비교 기준) · **현장 안전 모니터링 프로젝트**(`run_intrusion` 의 `source` 만 카메라로, Jetson 에서는 `.onnx` 를 가져가 **그 장치에서 다시 빌드**) · **YOLOv8 Edge 모델**(직접 학습한 YOLO 를 같은 경로 ONNX → 엔진 → `TRTInfer` 로).

**핵심 용어**
- **5일 지도**: Day 1 ONNX 첫걸음 → Day 2 ONNX 속 들여다보기 → Day 3 ONNX Runtime 으로 빠르게 → Day 4 TensorRT 첫걸음 → Day 5 탐지 · 종합 프로젝트
- **다시 빌드**: 엔진은 GPU · TensorRT 버전에 묶이므로 다른 장치(Jetson)에서는 `.onnx` 를 들고 가 거기서 다시 만든다

**핵심 내용**

#### ONNX · TensorRT 5일 지도
> 내보내기 → 들여다보기 → ONNX Runtime 으로 빠르게 → TensorRT → 탐지 · 종합 프로젝트
| 일차 | 주제 | 쌓인 도구 |
|---|---|---|
| Day 1 | ONNX 첫걸음 | `torch.onnx.export` · Netron · `InferenceSession` · `allclose` |
| Day 2 | ONNX 속 들여다보기 | `describe_onnx` · opset · `dynamic_axes` · `verify` · `onnxslim` |
| Day 3 | ONNX Runtime 으로 빠르게 | `bench` · 스레드 · 최적화 레벨 · CUDA EP · 배치 · profiling |
| Day 4 | TensorRT 첫걸음 | `polygraphy` · `build_engine` · `trt_runner` · FP16 · 엔진 인스펙터 |
| Day 5 | 탐지 · 종합 프로젝트 | `TRTInfer` · `detect(frame, infer)` · AP 비교 · 네 조각 시간 · 침입 감지 |

#### 셀프 체크: 5일 동안 만든 것
> 준비(한 번)와 실행(여러 번)을 나눠 `infer` 만 바꿔 끼우고, 정확도 · 병목을 숫자로 말할 수 있으면 끝
- **클래스** → `TRTInfer` 로 엔진 준비와 실행을 나눈다
- **입출력** → YOLOv8n `(1, 3, 640, 640)` → `(1, 84, 8400)` 을 설명한다
- **비교** → 같은 데이터 · 함수로 FP32 · FP16 의 AP 를 비교하고, 네 조각 시간으로 병목을 찾는다
- ⚠ 노트북 · `ot_baseline.csv` · 엔진을 만든 `.onnx` 는 Drive 에 저장해 둘 것

**연결**
- [OT D1·1교시 왜 ONNX 인가](#/ot/1/1) — 5일의 출발점
- [OT D5·5교시 추론기만 교체](#/ot/5/5) — 이식의 핵심
- [VB D5·8교시 발표 · 최종 필기 퀴즈 · baseline 제출](#/vb/5/8) — 이전 과목의 baseline

**상세 학습**

#### 이 교시에서 답하는 질문
5일 동안 쌓은 도구가 한 경로로 이어진다 — PyTorch → ONNX → 엔진 → `TRTInfer` → 프로젝트. 이 경로를 한 번에 떠올리고, 다음 과목에서 같은 `ot_baseline.csv` 와 비교하는 것이 이 교시의 목표다.

#### 개념 1 · 하나의 경로
| 단계 | 어디서 배웠나 |
|---|---|
| PyTorch → ONNX | Day 1 · 2 |
| ONNX Runtime 으로 측정 | Day 3 |
| ONNX → 엔진 → 실행 | Day 4 |
| 엔진 → 탐지 프로젝트 | Day 5 |

#### 한 번에 정리
- 제출: `trt_이름.zip` + 노트북, 비교 기준은 `ot_baseline.csv`
- 다음: INT8 · 카메라 입력 · 직접 학습한 YOLO — 모두 같은 경로를 재사용
- ⚠ 엔진 파일은 장치에 묶임 — Jetson 에서는 `.onnx` 로 다시 빌드
