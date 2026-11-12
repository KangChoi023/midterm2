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
어제 `trt_runner()` 는 안쪽 함수 `run` 에 엔진 · 버퍼를 붙잡아 두는 방식이었다. 오늘은 이것을 **클래스**로 바꾼다. "준비물(엔진 · Context · 버퍼)" 이 `self` 에 들어가 읽기 쉽고, 오늘 프로젝트와 이후 과목에서 그대로 가져다 쓴다. 엔진은 오래 걸리니 이미 있으면 다시 만들지 않는 `build_once` 도 준비한다.

#### 개념 1 · 준비 단계와 실행 단계
![TRTInfer 클래스 — __init__ 은 한 번(엔진 · Context · 이름 · 버퍼 · 주소), __call__ 은 프레임마다(복사 → 실행 → 기다리기 → NumPy)](assets/fig/ot/d5-class.svg)

| 메서드 | 언제 | 하는 일 |
|---|---|---|
| `__init__(engine_path)` | 한 번 | 엔진 읽기 → Context → 입력 · 출력 이름 나누기 → GPU 버퍼 `self.buf[이름]` → 주소 알려 주기, `self.input_shape` 기록 |
| `__call__(x)` | 프레임마다 | 입력 버퍼에 `copy_` → `execute_async_v3` → `synchronize` → 출력 `.cpu().numpy()` |
| `__repr__` | 출력할 때 | `TRTInfer(in=['input'] (1, 3, 224, 224), out=['logits'])` |

`__call__` 이 있으니 객체를 **함수처럼** 부른다: `mnv3(x1)`. 그래서 ORT 세션을 감싼 `lambda x: sess.run(…)[0]` 과 같은 자리에 넣을 수 있다 (2교시).

#### 개념 2 · build_once
`build_once(onnx_path, engine_path, fp16=False)` — 엔진 파일이 이미 있으면 건너뛰고, 없으면 Day 4 의 `build_engine` 으로 만든다. YOLOv8n 엔진 빌드는 몇 분 걸릴 수 있어 노트북을 다시 실행할 때 시간을 아낀다. 단, GPU · TensorRT 버전이 바뀐 장치에서는 파일을 지우고 다시 빌드해야 한다.

#### 코드로 확인
```python
class TRTInfer:
    def __init__(self, engine_path):
        self.engine = RUNTIME.deserialize_cuda_engine(open(engine_path, "rb").read())
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.buf = {n: torch.empty(tuple(self.engine.get_tensor_shape(n)), dtype=torch.float32,
                                   device="cuda") for n in names}
        for n in names:
            self.context.set_tensor_address(n, self.buf[n].data_ptr())
        self.input_shape = tuple(self.engine.get_tensor_shape(self.inputs[0]))
    def __call__(self, x):
        self.buf[self.inputs[0]].copy_(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return self.buf[self.outputs[0]].cpu().numpy()
```
- **무엇을 하나**: Day 4 의 `trt_runner` 를 클래스로 — 준비는 `__init__`, 실행은 `__call__`.
- **어떤 개념**: 개념 1.
- **출력**: 메서드 `['__init__', '__call__', '__repr__']` (직접 실행), mnv3s 엔진으로 `input_shape (1, 3, 224, 224)` · 출력 `(1, 1000)` · ORT CPU 와 같은 답 (교안 기대값)

#### 한 번에 정리
- `TRTInfer(engine)` — `__init__`: 엔진 · Context · 이름 · GPU 버퍼 · 주소 (한 번), `__call__(x)`: 복사 → 실행 → 기다리기 → NumPy.
- 객체를 함수처럼 부를 수 있어 ORT 세션과 바꿔 끼울 수 있다.
- `build_once` — 엔진이 있으면 다시 만들지 않는다.
- ⚠ 장치가 바뀌면 엔진 파일을 지우고 다시 빌드 (build_once 가 옛 엔진을 그대로 쓰지 않게).

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
VB Day 5 에서 export 한 것과 같은 `yolov8n.onnx` (입력 `images (1, 3, 640, 640)`, 출력 `output0 (1, 84, 8400)`) 를 **FP16 엔진**으로 바꾼다. 앞의 letterbox 와 뒤의 `decode_yolo` (NMS 포함) 는 VB 함수를 그대로 쓰고, 가운데 **추론기만** 바꿔 끼우는 `detect(frame, infer)` 를 만든다.

#### 개념 1 · 앞 · 가운데 · 뒤
![detect(frame, infer) — letterbox → infer(ORT · TRT FP32 · TRT FP16) → decode_yolo → 원래 좌표](assets/fig/ot/d5-detect.svg)

| 부분 | 코드 | 출처 |
|---|---|---|
| 앞 | `letterbox(frame)` → BGR→RGB · CHW · `/255` · float32 · 배치 축 | VB Day 4 |
| 가운데 | `out = infer(x)` — (1, 84, 8400) | **바꿔 끼우는 곳** |
| 뒤 | `decode_yolo(out, conf, iou_thr)` (전치 · conf 필터 · xywh→xyxy · NMS) → `(boxes − 패딩) / r` → clip | VB Day 4 · 5 |

#### 개념 2 · 같은 모양의 함수
| infer | 만드는 법 |
|---|---|
| ORT CPU | `ort_infer = lambda x: sess_cpu.run(None, {"images": x})[0]` |
| TRT FP32 | `yolo32 = TRTInfer("yolov8n_fp32.engine")` |
| TRT FP16 | `yolo16 = TRTInfer("yolov8n_fp16.engine")` |

셋 다 `x → (1, 84, 8400)` 이므로 `detect(frame, 아무거나)` 로 부를 수 있다. 실행기 비교가 **추론기만 바꾼 비교**가 되어 전처리 · 후처리 차이가 섞이지 않는다.

확인: 프레임 200 에서 ORT CPU 와 TRT FP16 의 사람 수 차이 ≤ 1, 짝 IoU 중앙값 > 0.9 (초록 굵은 박스와 빨강 얇은 박스가 겹쳐 보이면 같은 결과).

#### 코드로 확인
```python
def detect(frame, infer, conf=0.25, iou_thr=0.7):
    lb, r, (pl, pt) = letterbox(frame)
    x = (lb[..., ::-1].transpose(2, 0, 1)[None] / 255.0).astype(np.float32)     # BGR→RGB · CHW · 0~1
    out = infer(x)
    boxes, scores = decode_yolo(out, conf, iou_thr)
    boxes = (boxes - [pl, pt, pl, pt]) / r                                      # letterbox 되돌리기
    H, W = frame.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W); boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    return boxes, scores
b_ort, _ = detect(frame200, ort_infer)
b_trt, _ = detect(frame200, yolo16)
```
- **무엇을 하나**: 같은 detect 에 추론기만 바꿔 넣어 ORT CPU 와 TensorRT FP16 결과를 비교.
- **어떤 개념**: 개념 1 · 2.
- **출력**: `yolo16.inputs = ['images']`, 입력 `(1, 3, 640, 640)` → 출력 `(1, 84, 8400)`, 두 실행기의 사람 수 차이 ≤ 1 · 짝 IoU 중앙값 > 0.9 (교안 기대값)

#### 한 번에 정리
- `build_once(YOLO_ONNX, "yolov8n_fp16.engine", fp16=True)` → `TRTInfer` — 출력 (1, 84, 8400) 그대로.
- `detect(frame, infer)` — letterbox → infer → decode_yolo → 원래 좌표.
- infer 만 바꿔 끼운다 — 앞뒤는 VB 그대로.
- ⚠ 엔진마다 전처리를 다르게 하면 실행기 비교가 아니라 전처리 비교가 된다.

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
사진 한 장이 같아 보여도 **전체 성능**이 같은지는 숫자로 봐야 한다. VB Day 4 의 Penn-Fudan 170장 · 정답 박스와 `evaluate_ap` 로 ORT CPU · TRT FP32 · TRT FP16 의 AP50 · mAP50-95 를 비교하고, 박스 하나하나도 FP32 와 FP16 이 같은지 본다.

#### 개념 1 · 두 가지 눈금
| 눈금 | 무엇을 보나 | 기준 (교안) |
|---|---|---|
| 전체 지표 — AP50 · mAP50-95 | 170장 전체의 찾기 · 위치 정확도 | ORT CPU AP50 > 0.5, AP50 차이 FP32 < 0.01 · FP16 < 0.03 |
| 박스 단위 — 짝 IoU | 20장에서 FP16 박스마다 FP32 박스와의 최대 IoU | 중앙값 > 0.95 |

참고로 VB Day 4 에서 YOLOv8n 640 의 Penn-Fudan AP50 은 0.9692, mAP50-95 는 0.8357 (ORT · PyTorch CPU). FP32 엔진은 이와 같고, FP16 은 거의 그대로여야 한다.

#### 개념 2 · 비교 규칙
- **같은 사진 · 같은 detect · 같은 evaluate_ap** — 바뀌는 건 infer 하나.
- AP 는 **낮은 conf (0.01)** 로 잰다. conf 0.25 로 자르면 낮은 점수 박스가 빠져 recall 곡선이 짧아지고 AP 가 낮게 나온다 (VB Day 4: 평가용 conf vs 운영 conf).
- FP32 엔진과 ORT 의 AP 가 다르면 대개 입력 dtype · 채널 순서(BGR/RGB) 실수 — 같은 사진으로 출력을 먼저 비교한다.

#### 코드로 확인
```python
INFERS = {"ORT CPU": ort_infer, "TRT FP32": yolo32, "TRT FP16": yolo16}
ap_rows = []
for name, f in INFERS.items():
    dets = []
    for im in imgs:
        b, s = detect(im, f, conf=0.01)                     # AP 는 낮은 conf 로
        dets.append(np.concatenate([b, s[:, None]], 1))
    ap_rows.append({"runtime": name, "AP50": evaluate_ap(dets, gts), "mAP50-95": map_50_95(dets, gts)})
df_ap = pd.DataFrame(ap_rows)
```
- **무엇을 하나**: 세 실행기로 170장을 탐지해 AP50 · mAP50-95 를 같은 함수로.
- **어떤 개념**: 개념 1 · 2.
- **출력**: 세 줄 표 — FP32 = ORT (차이 < 0.01), FP16 ≈ ORT (차이 < 0.03) (교안 기대값)

```python
pair_iou = []
for im in imgs[:20]:
    b32, _ = detect(im, yolo32); b16, _ = detect(im, yolo16)
    if len(b32) and len(b16):
        pair_iou += list(iou_matrix(b16, b32).max(1))
pair_iou = np.array(pair_iou)
```
- **무엇을 하나**: FP16 박스마다 FP32 박스와의 최대 IoU.
- **어떤 개념**: 개념 1.
- **출력**: 박스 10개 이상, 짝 IoU 중앙값 > 0.95 — 박스 단위로도 거의 같다 (교안 기대값)

#### 한 번에 정리
- 같은 사진 · 같은 detect · 같은 evaluate_ap 로 세 실행기 비교 → FP32 = ORT, FP16 ≈ ORT.
- 정확도는 전체 지표(AP) + 박스 단위(짝 IoU) 두 가지로.
- FP16 의 최대 차이가 커도 AP · 박스가 같으면 쓸 수 있다.
- ⚠ AP 는 conf 0.01 로 — 0.25 로 재면 AP 가 낮게 나온다.

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
엔진은 빨라졌다. 그런데 영상 **한 프레임**을 처리하는 시간도 그만큼 빨라졌을까? 한 프레임을 네 조각(전처리 · 추론 · 후처리 · 그리기)으로 나눠 ORT CPU 와 TensorRT FP16 을 비교하면, 추론이 빨라질수록 **다른 조각이 새 병목**이 되는 것이 보인다.

#### 개념 1 · 전체 속도는 느린 조각이 정한다
![한 프레임 = 전처리 + 추론 + 후처리 + 그리기 — 추론을 80 ms 에서 8 ms 로 줄여도 프레임은 100 ms 에서 28 ms](assets/fig/ot/d5-stages.svg)

| 조각 | 구간 | 어디서 |
|---|---|---|
| pre | letterbox · 색 · 축 · /255 | CPU |
| infer | `infer(x)` | ORT: CPU / TRT: GPU (복사 포함) |
| post | decode_yolo · 좌표 되돌리기 · 발 위치 · 구역 판정 | CPU |
| draw | `draw_result` — 구역 · 박스 · 문구 | CPU |

프레임 시간 = 네 조각의 합, **FPS = 1000 ÷ 합**.

| 설명용 숫자 | 합 | FPS | 추론 비중 |
|---|---|---|---|
| 10 + **80** + 5 + 5 | 100 ms | 10 | 80% |
| 10 + **8** + 5 + 5 | 28 ms | 35.7 | 29% |

추론을 10배 빠르게 해도 프레임은 3.6배만 빨라진다 — 이제 전처리 10 ms 가 1등. 다음엔 CPU 쪽(전처리 · 그리기)을 줄인다.

#### 개념 2 · time_stages 로 재기
- 조각 경계마다 `time.perf_counter()` 를 찍어 차이 × 1000 = ms.
- 영상 50프레임을 앞 5장 warm-up 하고 평균 → `df_st` (조각별 ms · total · fps).
- `share = 조각 ÷ total × 100` → `bottleneck = share.idxmax()` — ORT CPU 는 infer, TRT FP16 은 다른 조각이 1등이 되는지 본다.

#### 코드로 확인
```python
def time_stages(frame, infer, conf=0.25):
    t0 = time.perf_counter()
    lb, r, (pl, pt) = letterbox(frame)
    x = (lb[..., ::-1].transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    t1 = time.perf_counter()
    out = infer(x)
    t2 = time.perf_counter()
    boxes, scores = decode_yolo(out, conf, 0.7)
    boxes = (boxes - [pl, pt, pl, pt]) / r
    inside = [in_zone(p) for p in foot_points(boxes)]
    t3 = time.perf_counter()
    draw_result(frame, boxes, inside, alarm=any(inside))
    t4 = time.perf_counter()
    return dict(zip(["pre", "infer", "post", "draw"],
                    [(b - a) * 1000 for a, b in [(t0, t1), (t1, t2), (t2, t3), (t3, t4)]]))
```
- **무엇을 하나**: 한 프레임을 네 조각 ms 로.
- **어떤 개념**: 개념 2.
- **출력**: ORT CPU 와 TRT FP16 의 조각별 ms · total · fps 표, 네 조각 합 = total, 가장 오래 걸린 조각이 실행기마다 다름 (교안 기대값 — 숫자는 GPU 마다 다름)

#### 한 번에 정리
- 프레임 시간 = 전처리 + 추론 + 후처리 + 그리기, FPS = 1000 ÷ 합.
- 추론이 빨라지면 CPU 쪽(전처리 · 후처리 · 그리기)이 새 병목.
- warm-up 후 50프레임 평균, 비율로 병목 찾기.
- ⚠ 추론 시간만 보고 "FPS 가 이만큼" 이라 발표하지 않는다 — 영상 읽기 · 저장까지 합친 파이프라인 FPS 는 6교시.

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
VB Day 5 의 위험구역 침입 감지기를 TensorRT 로 옮긴다. 바꾸는 곳은 탐지에 넘기는 **infer 하나** — 발 위치 · 구역 판정 · 3프레임 연속 경보(debounce) · 그리기는 VB 그대로다. 프레임 한 장을 처리하는 `process_frame` 과 영상 전체를 도는 `run_intrusion` 으로 나눈다.

#### 개념 1 · 바뀌는 것 한 곳
![침입 감지 파이프라인 — detect(infer=yolo16) · 발 위치 · 구역 · debounce · 그리기, 그리고 결과 파일 네 개](assets/fig/ot/d5-project.svg)

| 단계 | 코드 | VB 와 비교 |
|---|---|---|
| ① 탐지 | `detect(frame, infer)` | **infer = yolo16** (VB 는 ONNX Runtime) |
| ② 발 위치 | `foot_points(boxes)` → `((x1+x2)/2, y2)` | 같음 |
| ③ 구역 | `in_zone(p, GRASS)` → `pointPolygonTest ≥ 0` | 같음 |
| ④ debounce | `streak = streak + 1 if inside.any() else 0`, `alarm = int(streak >= n_consec)` | 같음 (n_consec = 3) |
| ⑤ 그리기 | `draw_result(frame, boxes, inside, alarm, poly, fps)` | + 최근 30프레임 평균 FPS |

#### 개념 2 · process_frame 과 run_intrusion
| 함수 | 받는 것 | 돌려주는 것 |
|---|---|---|
| `process_frame(frame, infer, streak, …)` | 프레임 한 장 + 직전까지의 연속 수 | (그린 프레임, `{n_people, n_inside, alarm}`, 새 streak) |
| `run_intrusion(source, out_path, infer, …, max_frames=None)` | 영상 경로(나중에 카메라 번호) | 프레임 로그 목록 + 결과 영상 + `…_log.csv` |

- 영상을 여는 곳은 `cv2.VideoCapture(source)` **한 곳** — 프로젝트에서 source 만 카메라로 바꾼다.
- 로그 한 줄: `frame · time_s · n_people · n_inside · alarm · ms` (ms = 프레임 처리 시간, 6교시 성능표의 재료).
- 화면 오른쪽 위 FPS = 1000 ÷ 최근 30프레임 평균 ms.
- `max_frames` 를 주면 그 프레임까지만 — 6교시 ORT CPU 비교용 (느려서 200프레임만).

#### 코드로 확인
```python
def process_frame(frame, infer, streak, poly=GRASS, n_consec=3, conf=0.25, fps=None):
    boxes, scores = detect(frame, infer, conf=conf)
    feet = foot_points(boxes)
    inside = np.array([in_zone(p, poly) for p in feet], bool)
    streak = streak + 1 if inside.any() else 0
    alarm = int(streak >= n_consec)
    vis = draw_result(frame, boxes, inside, alarm, poly, fps)
    info = {"n_people": len(boxes), "n_inside": int(inside.sum()), "alarm": alarm}
    return vis, info, streak
```
- **무엇을 하나**: 과제 1 — 프레임 한 장의 다섯 단계.
- **어떤 개념**: 개념 1 · 2.
- **출력**: `process_frame(frame530, yolo16, streak=2)` → 구역 안 사람이 있으면 streak 3 → 경보 1 (교안 기대값)

```python
def run_intrusion(source, out_path, infer, poly=GRASS, n_consec=3, conf=0.25, max_frames=None):
    cap = cv2.VideoCapture(source)                         # 영상을 여는 곳은 여기 한 곳
    fps_v = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (W, H))
    log, ms, streak, i = [], [], 0, 0
    while max_frames is None or i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        vis, info, streak = process_frame(frame, infer, streak, poly, n_consec, conf,
                                          1000 / np.mean(ms[-30:]) if ms else None)
        writer.write(vis)
        ms.append((time.perf_counter() - t0) * 1000)
        log.append({"frame": i, "time_s": round(i / fps_v, 1), **info, "ms": round(ms[-1], 2)})
        i += 1
    cap.release(); writer.release()
    return log                                             # + _log.csv 저장 (강의 코드)
```
- **무엇을 하나**: 과제 2 — 영상 루프 · 저장 · 로그.
- **어떤 개념**: 개념 2.
- **출력**: `intrusion_trt.mp4` + `intrusion_trt_log.csv`, 전체 프레임 처리 (교안 기대값)

#### 한 번에 정리
- `process_frame` = 탐지(infer) → 발 위치 → 구역 → debounce → 그리기.
- `run_intrusion` = 영상 루프 + 저장 + 로그 CSV.
- 바뀐 것은 infer 하나 — `yolo16` 을 넘겼을 뿐 나머지는 VB 그대로.
- ⚠ streak 를 프레임마다 돌려받아 다음 프레임에 넘겨야 debounce 가 이어진다.

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
파이프라인이 돌았으니 결과물을 만든다 — **결과 영상 · 이벤트 요약 · 최종 성능표**. 특히 성능표에는 "추론만" 과 "파이프라인 전체" 를 둘 다 넣고, 두 배수가 왜 다른지 4교시의 병목으로 설명한다.

#### 개념 1 · 이벤트 = 켜진 구간
`find_events(alarm, fps)` 는 VB Day 5 와 같다 — alarm 이 0 → 1 이 되는 프레임이 시작, 1 → 0 직전 프레임이 끝, 영상 끝까지 켜져 있으면 마지막 프레임이 끝.

| 열 | 식 |
|---|---|
| start_s | 시작 프레임 ÷ fps |
| end_s | 끝 프레임 ÷ fps (경보가 켜져 있던 **마지막** 프레임) |
| duration_s | end − start + 1 ÷ fps (끝 프레임도 경보이므로 한 프레임 시간을 더함) |

예: `alarm = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]`, fps = 10 → `[(0.2, 0.4), (0.7, 0.8), (1.0, 1.0)]`, 길이 0.3 · 0.2 · 0.1 초. 결과는 `intrusion_events.csv` (`event · start_s · end_s · duration_s`).

#### 개념 2 · 추론만 vs 파이프라인 전체
| 줄 | 무엇을 잼 | 실행기 |
|---|---|---|
| yolov8n (추론만) | letterbox 된 입력 한 장의 `infer(x)` — `bench_fn` | ORT CPU · TRT FP32 · TRT FP16 |
| intrusion pipeline | `run_intrusion` 의 (전체 시간 ÷ 프레임 수) — 읽기 · 저장 포함, p50 · p95 는 로그의 ms | ORT CPU (200프레임) · TRT FP16 (전체) |

다섯 줄을 `ot_baseline.csv` 에 이어 붙인다 (Day 4 파일이 있으면 그 아래, precision 없는 옛 줄은 FP32). 추론만의 배수(TRT FP16 ÷ ORT CPU)가 파이프라인의 배수보다 크다 — 파이프라인에는 빨라지지 않은 CPU 조각과 영상 읽기 · 저장이 남아 있기 때문 (4교시).

결과 영상은 브라우저에서 재생되도록 `ffmpeg … -c:v libx264 -pix_fmt yuv420p` 로 H.264 변환(`intrusion_trt_h264.mp4`)하고, 구역 안 사람 수와 경보 구간을 시간축 그래프로 겹쳐 본다.

#### 코드로 확인
```python
events = find_events(alarm, V_FPS)
with open("intrusion_events.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["event", "start_s", "end_s", "duration_s"])
    for k, (a, b) in enumerate(events):
        w.writerow([k + 1, round(a, 1), round(b, 1), round(b - a + 1 / V_FPS, 1)])
```
- **무엇을 하나**: 과제 3 — 경보 배열을 이벤트 표로.
- **어떤 개념**: 개념 1.
- **출력**: 이벤트마다 한 줄 — VB Day 5 강사 실행은 795프레임 중 경보 224프레임 · 이벤트 5번 (교안 기대값)

```python
lb, _, _ = letterbox(frame200)
x640 = (lb[..., ::-1].transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
rows = []
for name, f, prec in [("ORT CPU", ort_infer, "FP32"), ("TRT FP32", yolo32, "FP32"), ("TRT FP16", yolo16, "FP16")]:
    r = bench_fn(lambda: f(x640), n=30 if name == "ORT CPU" else 100, warm=5)
    rows.append({"model": "yolov8n (추론만)", "provider": name, "precision": prec,
                 "threads": 0, "level": "ALL", "batch": 1, **r})
log_ort = run_intrusion(VIDEO, "intrusion_ort200.mp4", ort_infer, max_frames=200)
```
- **무엇을 하나**: 과제 4 — 추론만 3줄 + 파이프라인 2줄을 같은 열로.
- **어떤 개념**: 개념 2.
- **출력**: `파이프라인: ORT CPU ○ FPS → TensorRT FP16 ○ FPS (○배)` — 추론만의 배수보다 작다 (교안 기대값 — 숫자는 GPU 마다 다름)

#### 한 번에 정리
- 프레임 로그 → `find_events` → 이벤트 CSV — 경보가 몇 번 · 언제 · 얼마나.
- 추론만 vs 파이프라인 전체 — 같은 성능표에 둘 다.
- 배수가 다른 이유 = 4교시의 다음 병목 (CPU 조각 · 읽기 · 저장).
- ⚠ duration 은 끝 − 시작에 한 프레임 시간을 더한다 — 경보 한 프레임짜리 이벤트도 0초가 아니다.

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
5일 동안 만든 것을 팀별 3분에 보여 준다. 무엇을 어떤 순서로 말해야 숫자가 설득력을 가질까? 세 장면과 채점 기준을 정리한다.

#### 개념 1 · 발표 3분 = 세 장면
| 장면 | 보여 줄 것 | 노트북에서 |
|---|---|---|
| 1 · 결과 영상 (1분) | 경보가 울린 장면 하나 — 누가 · 언제 · 왜 경보인지 | 6교시 영상 · `intrusion_events.csv` |
| 2 · 최종 성능표 (1분) | ORT CPU vs TRT FP16 — 추론만 · 파이프라인 FPS, FP16 의 AP | 과제 4 표 · 3-1 표 |
| 3 · 가장 오래 걸린 단계 (1분) | 네 조각 중 가장 큰 것과 이유, 다음에 고칠 것 | 4-2 그래프 |

#### 개념 2 · 채점 기준 (20점)
| 기준 | 점수 |
|---|---|
| 숫자를 **조건과 함께** 말했는가 — "FPS 가 몇" 이 아니라 "ORT CPU · 파이프라인 전체 · 배치 1 에서 몇 FPS" | 5 |
| FP16 정확도 판단 근거 (AP 차이 · 짝 IoU) | 5 |
| 병목과 다음 개선안 | 5 |
| 질문 · 답변 | 5 |

발표 메모 틀: ① 무엇을 만들었나 (한 문장) ② 결과 영상의 경보 한 장면 ③ 성능표 두 줄 (추론만 · 파이프라인) + FP16 AP ④ 병목 조각과 다음 개선 (전처리 GPU 로 옮기기 · INT8 · 입력 크기 줄이기 등).

#### 한 번에 정리
- 세 장면: 결과 영상 → 성능표 → 병목.
- 숫자는 조건과 함께 (실행기 · 추론만/파이프라인 · 배치).
- FP16 은 "빨라졌다" 와 "정확도가 그대로다" 를 함께 말한다.
- ⚠ 추론만 FPS 를 파이프라인 FPS 처럼 말하지 않는다.

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
5일 종합 퀴즈와 제출로 마무리하면서, 5일 동안 쌓은 도구가 **하나의 경로**로 이어진다는 것을 정리한다. 그리고 그 경로가 다음 과목에서 어떻게 그대로 쓰이는지 본다.

#### 개념 1 · 하나의 경로
![ONNX · TensorRT 5일 — Day 별 도구와 다음 과목 연결](assets/fig/ot/d5-map.svg)

PyTorch 모델 → `torch.onnx.export` (Day 1) → 검사 · 고치기 · 배치 자유 (Day 2) → ONNX Runtime 측정 · 손잡이 (Day 3) → TensorRT 엔진 · FP16 (Day 4) → `TRTInfer` 로 파이프라인에 끼우기 (Day 5).

| 일차 | 쌓인 도구 |
|---|---|
| Day 1 | `torch.onnx.export` · Netron · `InferenceSession` · `allclose` |
| Day 2 | `describe_onnx` · opset · `dynamic_axes` · `verify` · `onnxslim` |
| Day 3 | `bench` · 스레드 · 최적화 레벨 · CUDA EP · 배치 · profiling |
| Day 4 | `polygraphy` · `build_engine` · `trt_runner` · FP16 · 엔진 인스펙터 |
| Day 5 | `TRTInfer` · `detect(frame, infer)` · AP 비교 · 네 조각 시간 · 침입 감지 |

#### 개념 2 · 제출과 셀프 체크
- 제출: `trt_이름.zip` = `ot_baseline.csv` · `intrusion_events.csv` · `intrusion_trt_log.csv` · `intrusion_trt_h264.mp4` (+ 노트북 `.ipynb` 는 따로 다운로드).
- 셀프 체크: ① 엔진 준비(한 번)와 실행(여러 번)을 클래스로 나눈다 ② YOLOv8n 엔진 입출력 (1, 3, 640, 640) → (1, 84, 8400) ③ infer 를 인자로 받아 ORT · TensorRT 를 바꿔 끼운다 ④ 같은 데이터 · 함수로 FP32 · FP16 AP 비교 ⑤ 네 조각으로 병목 찾기 ⑥ 추론기만 바꿔 결과 영상 · 이벤트 · 성능표.

#### 개념 3 · 다음 과목으로
| 과목 | 이어받는 것 |
|---|---|
| Quantization · Pruning (11/16~) | INT8 이면 성능표가 어떻게 바뀔까? — `ot_baseline.csv` 가 비교 기준 |
| 프로젝트 · 현장 안전 모니터링 (11/25~) | `run_intrusion` 의 source 만 카메라로 · Jetson 에서는 `.onnx` 를 가져가 다시 빌드 |
| YOLOv8 Edge (12/11~) | 직접 학습한 YOLO 를 같은 경로 (ONNX → 엔진 → TRTInfer) 로 |

#### 한 번에 정리
- 5일 = 하나의 경로: export → 검사 · 고치기 → ORT 측정 → TensorRT 엔진 → TRTInfer 파이프라인.
- 성능표 `ot_baseline.csv` 가 다음 과목의 비교 기준.
- Drive 에 남길 것: 노트북 · `ot_baseline.csv` · 엔진을 만든 `.onnx`.
- ⚠ 엔진 파일이 아니라 `.onnx` 를 보관한다 — 엔진은 장치마다 다시 빌드.

