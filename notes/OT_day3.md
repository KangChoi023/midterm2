## Day 3 — ONNX Runtime 으로 빠르게: 제대로 재고, 하나씩 바꿔 비교하기

### 1교시 · Day 2 복습 · 오늘의 모델 · GPU 확인

**키워드**: `mnv3s.onnx`, `mnist_cnn_dyn.onnx`, `providers = 희망 목록`, `get_providers()`, `한 번에 하나만`

**개요**: 오늘은 ONNX 파일은 그대로 두고 **실행하는 쪽 손잡이**(스레드 · 최적화 레벨 · 장치 · 배치)를 바꿔 가며 속도를 잰다. 잴 모델 두 개 — `mnv3s.onnx`(MobileNetV3-small, 입력 `(배치, 3, 224, 224)`, opset 17, 배치 자유)와 `mnist_cnn_dyn.onnx`(SimpleConvNet, `(배치, 1, 28, 28)`). 실험 원칙 3개: ① 한 번에 하나만 바꾼다 ② 조건을 표의 열로 적는다 ③ 바꾼 뒤에도 결과는 같아야 한다(`allclose`).

**핵심 용어**
- **providers**: 어디서 계산할지 적는 우선순위 목록 — `["CUDAExecutionProvider", "CPUExecutionProvider"]`
- **dynamic_axes**: 배치 축을 숫자 대신 이름(`batch`)으로 열어 두는 export 옵션 (Day 2)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `get_providers()` | `sess.get_providers()` | 세션이 **실제로** 쓰는 실행기 목록 |
| `get_available_providers()` | `ort.get_available_providers()` | 설치된 실행기 목록 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 1-1 | 두 모델을 `dynamic_axes` 로 export, `x1 (1,3,224,224)` · `xm (1,1,28,28)` 입력 준비, PyTorch 와 `allclose` 확인 |
| 1-2 | CUDA 를 앞에 둔 세션을 만들어 `used[0] == "CUDAExecutionProvider"` 로 `GPU_OK` 판정 |

**핵심 내용**

#### providers 는 희망 목록
> 세션은 목록 앞에서부터 쓸 수 있는 실행기를 고른다 — 진짜 쓰는 것은 `get_providers()` 로 확인
- **설치됨 vs 실제 사용** → `get_available_providers()` 와 `sess.get_providers()` 는 다를 수 있음
- **CPU 를 맨 뒤에** → CUDA 가 안 되면 조용히 CPU 로 돌아감 (그래서 확인이 필요)
- ⚠ CUDA 를 적었다고 GPU 인 것이 아님 — 라이브러리가 안 맞으면 CPU 로 폴백

#### 실험 원칙 3가지
> 스레드와 레벨을 동시에 바꾸면 무엇 때문에 빨라졌는지 모른다
- **한 번에 하나** → 손잡이 하나만 바꿔 비교
- **조건을 열로** → 모델 · 실행기 · 스레드 · 레벨 · 배치 · 장치
- **결과 동일** → 빨라졌는데 답이 달라지면 소용없음 (`np.allclose`)

**기본 사용형태**
```python
probe = ort.InferenceSession("mnist_cnn_dyn.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
GPU_OK = probe.get_providers()[0] == "CUDAExecutionProvider"
```

**연결**
- [OT D2·배치 자유 export](#/ot/2/1) — `dynamic_axes` 로 배치 축을 열어 둔 파일을 오늘 계속 씀
- [OT D1·6교시 ONNX Runtime 실행](#/ot/1/6) — `InferenceSession` · `run` 의 기본형

**상세 학습**

#### 이 교시에서 답하는 질문
Day 1~2 에서는 "파일을 만들고 읽는" 법을 배웠다. 이제 "같은 파일을 더 빠르게 돌리려면 무엇을 바꾸나" 가 질문이다. 바꿀 수 있는 것은 손잡이 다섯 개(재는 법 · 스레드 · 최적화 레벨 · 장치 · 배치)이고, 잘못 바꾸면 비교가 무의미해지므로 원칙부터 정한다.

#### 개념 1 · 오늘의 입력 크기
| 모델 | 입력 | 원소 수 |
|---|---|---|
| mnv3s | (1, 3, 224, 224) | 3 × 224 × 224 = **150,528** |
| mnist_cnn_dyn | (1, 1, 28, 28) | 28 × 28 = **784** |
float32 이므로 mnv3s 입력 한 장 = 150,528 × 4 B = **602,112 B**(약 0.6 MB).

#### 코드로 확인
```python
used = probe.get_providers()
print("설치됨:", ort.get_available_providers())
print("실제 사용:", used)
```
- **무엇을 하나**: 희망 목록과 실제 실행기를 비교
- **어떤 개념**: providers 는 희망 목록
- **출력**: 환경마다 다름 (GPU 없으면 `['CPUExecutionProvider']`)

#### 한 번에 정리
- 오늘 = 파일 고정 · 실행 손잡이만 변경
- 원칙: 하나씩 · 조건을 열로 · 결과 동일 확인
- ⚠ providers 는 희망일 뿐 — `get_providers()` 로 확인

---

### 2교시 · 시간을 재는 법: warm-up · 평균 · p95 · FPS → bench()

**키워드**: `warm-up`, `time.perf_counter`, `평균 · p50 · p95`, `FPS = 1000 ÷ 평균(ms)`, `bench()`

**개요**: 한 번만 재면 첫 실행(준비 비용)이 섞여 믿을 수 없다. **warm-up 을 버리고 → 여러 번 재서 → 평균 · p50 · p95 · FPS** 로 요약한다. 이 과정을 함수 `bench(sess, x, n=100, warm=10)` 로 묶어 오늘 내내 쓴다.

**핵심 용어**
- **warm-up**: 재기 전에 몇 번 돌려 버리는 준비 실행
- **p95**: 100번 중 95번이 이 값 이하 — 가끔 튀는 느린 실행을 보는 지표
- **FPS**: 초당 처리 장 수 = 1000 ÷ 평균(ms) × 배치

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `time.perf_counter()` | `t0 = time.perf_counter()` | 짧은 시간을 정밀하게 재는 시계 |
| `np.percentile()` | `np.percentile(t, 95)` | p50 · p95 계산 |
| `bench()` | `bench(sess, x, n=100, warm=10)` | warm-up → n번 측정 → mean · p50 · p95 · fps |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 2-1 | 같은 일을 5번 재서 첫 실행이 가장 느린지 확인 |
| 2-2 | warm-up 10번 버리고 100번 측정 → 평균 · p50 · p95 · FPS |
| 2-3 | 위를 `bench()` 로 묶어 MobileNetV3 · SimpleConvNet 비교 |
| 바꿔보기 | 같은 모델을 PyTorch(`no_grad`, CPU)로 재서 ORT 와 비교 |

**핵심 내용**

#### warm-up — 첫 실행은 버린다
> 첫 실행에는 메모리 할당 · 커널 준비 같은 1회성 비용이 섞인다
- **핵심** → 재지 않는 warm-up 10번 → 측정 100번
- **예** → 첫 실행이 나머지 평균의 몇 배인지 `first5[0] / mean(first5[1:])` 로 확인
- ⚠ `time.time()` 으로 한 번만 재기 · 측정 중 다른 셀 실행

#### 평균 · p50 · p95 · FPS
> FPS = 1000 ÷ 평균(ms) × 배치 — 평균만 보지 말고 p95 로 "가끔 튀는 지연"을 본다
- **p50** → 중앙값 (보통의 한 번)
- **p95** → 느린 쪽 5% 의 경계 — 실시간 시스템의 기준
- **예** → 평균 20 ms → FPS = 1000 ÷ 20 = **50**
- ⚠ 사람마다 Colab 장비가 달라 숫자는 다름 — **같은 노트북 안에서 조건만 바꿔** 비교

**새로 만드는 함수** (이 교시에서 만드는 함수)
| 함수 | 기본형 | 용도 |
|---|---|---|
| `bench()` | `bench(sess, x, n=100, warm=10)` | 오늘의 측정 도구 (위 표와 동일) |

**기본 사용형태**
```python
def bench(sess, x, n=100, warm=10):
    name = sess.get_inputs()[0].name
    for _ in range(warm): sess.run(None, {name: x})
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); sess.run(None, {name: x})
        t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    return {"mean": t.mean(), "p50": np.percentile(t, 50),
            "p95": np.percentile(t, 95), "fps": 1000 / t.mean() * x.shape[0]}
```

**연결**
- [VB D5·6교시 ORT vs PyTorch 속도](#/vb/5/6) — 어제까지의 "ORT 가 더 빠르다" 를 bench 방식으로 다시 확인
- [OT D4·TensorRT 측정](#/ot/4/1) — 같은 `bench` 방식을 엔진에 적용

**상세 학습**

#### 이 교시에서 답하는 질문
"빨라졌다" 를 주장하려면 믿을 수 있는 숫자가 필요하다. 한 번 재면 첫 실행 · 우연한 지연이 섞인다. 반복 측정 + 요약 지표 + 함수화로 해결한다.

#### 개념 1 · 지표의 쓰임
| 지표 | 질문 | 계산 |
|---|---|---|
| mean | 평균적으로 얼마? | `t.mean()` |
| p50 | 보통은? | `np.percentile(t, 50)` |
| p95 | 나쁜 날은? | `np.percentile(t, 95)` |
| fps | 초당 몇 장? | 1000 ÷ mean × 배치 |
p95 ≥ p50 은 항상 성립 (백분위는 커질수록 값이 같거나 커짐).

#### 개념 2 · 30 FPS 의 예산
30 FPS = 한 장당 1000 ÷ 30 ≈ **33.3 ms**. 평균이 이보다 작아도 p95 가 크면 가끔 프레임을 놓친다.

#### 코드로 확인
```python
mean, p50, p95 = times.mean(), np.percentile(times, 50), np.percentile(times, 95)
print(f"평균 {mean:.2f} | p50 {p50:.2f} | p95 {p95:.2f} | {1000/mean:.0f} FPS")
```
- **무엇을 하나**: 100번 측정값을 한 줄로 요약
- **어떤 개념**: 개념 1
- **출력**: 환경마다 다름 — `p95 ≥ p50`, `fps × mean = 1000` 은 항상

#### 한 번에 정리
- warm-up 버리기 → 100번 → mean · p50 · p95 · fps
- 배치 b 장이면 FPS 에 × b
- ⚠ 평균만 보고 p95 를 안 보면 가끔 튀는 지연을 놓침

---

### 3교시 · 스레드 수: SessionOptions · 한 번에 하나만

**키워드**: `SessionOptions`, `intra_op_num_threads`, `코어 수`, `make_session`, `옵션 → 세션 순서`

**개요**: 첫 번째 손잡이 = 연산 하나를 몇 명(스레드)이 나눠 계산하나. `SessionOptions` 를 만들어 값을 정하고 **그 다음에** `InferenceSession(path, so, providers)` 로 세션을 만든다. 기본값 0 = 자동. 이후 손잡이를 인자로 받는 `make_session` 을 오늘의 세션 공장으로 쓴다.

**핵심 용어**
- **스레드(thread)**: 계산을 나눠 맡는 일꾼 — 코어 = 작업대
- **intra_op_num_threads**: 연산 하나를 나눠 계산할 스레드 수 (0 = 자동)
- **오버서브스크립션**: 코어보다 스레드가 많아 서로 기다리는 상태

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `ort.SessionOptions()` | `so = ort.SessionOptions()` | 세션 실행 옵션 묶음 만들기 |
| `os.cpu_count()` | `os.cpu_count()` | 이 컴퓨터의 CPU 코어 수 |
| `make_session()` | `make_session(path, threads=0, level="ALL", providers=None)` | 손잡이를 인자로 받는 세션 공장 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 3-1 | 스레드 1 · 2 · 4 로 세션을 만들어 bench → `df_th` 표 |
| 3-2 | p95 가 가장 작은 스레드 수 고르기, 코어보다 많은 경우 비교 |
| make_session | 스레드 · 레벨 · providers · 저장 경로 · profile 을 인자로 받는 함수 |

**핵심 내용**

#### 스레드 수 — 코어 수가 보통 상한
> 스레드 = 일꾼, 코어 = 작업대 — 일꾼이 작업대보다 많으면 기다리느라 오히려 느려질 수 있다
- **기본값 0** → ORT 가 자동으로 정함
- **고르는 기준** → 실시간이면 평균이 아니라 **p95 가 가장 작은 것**
- **순서** → `so.intra_op_num_threads = k` → `InferenceSession(path, so, ...)` 
- ⚠ 세션을 만든 **뒤에** 옵션을 바꾸면 적용되지 않음
- ⚠ 스레드를 늘리면 무조건 빨라진다고 생각 — 작은 연산은 나누는 비용이 더 큼

#### make_session — 손잡이를 한 곳에
> 앞으로는 `make_session(path, threads, level, providers, save_opt, profile)` 한 함수로 세션을 만든다
- **threads** → 이 교시 · **level** → 4교시 · **providers** → 5교시 · **profile** → 7교시
- **LEVELS 딕셔너리** → 문자열 `"ALL"` 을 `ort.GraphOptimizationLevel.ORT_ENABLE_ALL` 로 바꿔 줌

**기본 사용형태**
```python
so = ort.SessionOptions()
so.intra_op_num_threads = 2
s = ort.InferenceSession("mnv3s.onnx", so, providers=CPU)
print(bench(s, x1))
```

**연결**
- [OT D3·2교시 bench](#/ot/3/2) — 스레드마다 같은 `bench` 로 측정
- [OS D2·프로세스와 스레드](#/os/2/1) — 스레드 · 코어 개념의 출처

**상세 학습**

#### 이 교시에서 답하는 질문
2교시에서 재는 도구를 만들었으니 첫 손잡이를 돌려 본다. 스레드는 가장 값싼 손잡이지만 "많을수록 좋다" 는 오해를 깨는 좋은 첫 실험이다.

#### 개념 1 · 하나씩 바꾸는 실험 표
| threads | mean | p95 |
|---|---|---|
| 1 | ? | ? |
| 2 | ? | ? |
| 4 | ? | ? |
값은 장비마다 다르므로 직접 실행해 채운다. 다른 열(레벨 · 배치)은 고정.

#### 코드로 확인
```python
best_threads = int(df_th.loc[df_th["p95"].idxmin(), "threads"])
```
- **무엇을 하나**: p95 가 가장 작은 행의 `threads`
- **어떤 개념**: 스레드 — p95 로 고르기
- **출력**: 환경마다 다름

#### 한 번에 정리
- `SessionOptions` → `intra_op_num_threads` → `InferenceSession`
- 코어 수가 보통 최선, 넘으면 p95 부터 나빠짐
- ⚠ 옵션은 세션 만들기 **전에**

---

### 4교시 · 그래프 최적화 레벨: 노드 합치기 · 레이아웃

**키워드**: `graph_optimization_level`, `DISABLE · BASIC · EXTENDED · ALL`, `FusedConv`, `FusedGemm`, `optimized_model_filepath`

**개요**: ORT 는 세션을 만들 때 그래프를 스스로 고친다. 얼마나 고칠지가 레벨이다. **BASIC**: 상수 접기 등 → **EXTENDED**: 노드 합치기(Conv+Relu → `FusedConv`, Gemm+Relu → `FusedGemm`) → **ALL**: 데이터 배치(레이아웃)까지 CPU 에 맞게 (기본값). 합치면 메모리 왕복이 줄어 빨라지고, **결과는 같다**.

**핵심 용어**
- **노드 합치기(fusion)**: 연달아 있는 연산 여러 개를 하나로 묶기
- **FusedConv / FusedGemm**: ORT 전용 합쳐진 연산 (`com.microsoft` domain)
- **레이아웃 최적화**: CPU 명령에 맞게 데이터 배치를 바꾸는 것

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `ort.GraphOptimizationLevel` | `ort.GraphOptimizationLevel.ORT_ENABLE_ALL` | 최적화 레벨 상수 |
| `optimized_model_filepath` | `so.optimized_model_filepath = "opt.onnx"` | 최적화된 그래프를 파일로 저장 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 4-1 | 네 레벨로 `mnist_cnn_dyn` 을 저장하고 노드 목록 비교 — DISABLE 은 **7개**, EXTENDED 에 `FusedConv` · `FusedGemm` |
| 4-2 | 네 레벨의 mnv3s 속도 → `speedup = DISABLE 평균 ÷ 평균` |
| 바꿔보기 | 네 레벨 출력을 DISABLE 과 `allclose` 비교 |

**핵심 내용**

#### 최적화 레벨 4단계
> DISABLE → BASIC(상수 접기) → EXTENDED(노드 합치기) → ALL(레이아웃, 기본값)
- **EXTENDED** → Conv+Relu → `FusedConv`, Gemm+Relu → `FusedGemm`
- **왜 빨라지나** → 연산 사이의 메모리 읽고 쓰기 왕복이 줄어서
- **결과** → 레벨을 올려도 출력은 같다 (`allclose`)
- ⚠ 저장한 최적화 그래프는 그 장치 전용 — 다른 CPU 에 배포 금지
- ⚠ `FusedConv` 가 든 파일은 ORT 전용 — TensorRT · 다른 실행기가 읽지 못함

#### speedup 계산
> speedup = 기준(DISABLE) 평균 ÷ 해당 레벨 평균 — 기준 자신은 항상 1.0
- **예** → DISABLE 40 ms, ALL 20 ms 라면 speedup = 40 ÷ 20 = **2.0 배**

**기본 사용형태**
```python
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
so.optimized_model_filepath = "cnn_opt_EXTENDED.onnx"
ort.InferenceSession("mnist_cnn_dyn.onnx", so, providers=CPU)
```

**연결**
- [OT D2·onnxslim](#/ot/2/6) — 어제의 상수 접기 → ORT 는 합치기 · 레이아웃까지
- [OT D4·TensorRT 엔진 빌드](#/ot/4/2) — TensorRT 도 같은 방식으로 노드를 합침

**상세 학습**

#### 이 교시에서 답하는 질문
스레드는 "같은 일을 나눠 하기" 였다. 레벨은 "일 자체를 줄이기" — 그래프를 고쳐 연산 수를 줄인다. 눈에 보이게 하려고 고친 그래프를 파일로 저장해 연다.

#### 개념 1 · SimpleConvNet 의 합치기
| 레벨 | 노드 | 비고 |
|---|---|---|
| DISABLE | 7개 | 원본 그대로 |
| EXTENDED | Conv+Relu → FusedConv, Gemm+Relu → FusedGemm | 노드 수 감소 |
`describe_onnx` 의 domain 열이 `com.microsoft` 이면 ORT 전용 연산.

#### 코드로 확인
```python
for lv in ["BASIC", "EXTENDED", "ALL"]:
    out = make_session("mnv3s.onnx", level=lv).run(None, {"input": x1})[0]
    print(lv, np.allclose(out, base, atol=1e-4))
```
- **무엇을 하나**: 레벨을 올려도 답이 같은지 확인
- **어떤 개념**: 실험 원칙 3
- **출력**: 세 줄 모두 `True` 기대 (교안 의도)

#### 한 번에 정리
- 레벨 4단계, 기본 ALL · 합치기 = 메모리 왕복 감소
- 결과는 같고 속도만 다름
- ⚠ 저장한 최적화 파일은 배포용이 아님

---

### 5교시 · CPU → GPU: CUDAExecutionProvider

**키워드**: `CUDAExecutionProvider`, `복사 · 계산 · 복사`, `GPU warm-up`, `배치와 GPU`, `onnxruntime-gpu`

**개요**: 세 번째 손잡이 = 어디서 계산하나. `providers` 맨 앞에 `CUDAExecutionProvider` 를 두면 GPU 에서 실행한다. 코드는 한 단어지만 **CPU ↔ GPU 사이 데이터 복사**와 **첫 실행 준비** 비용이 새로 생긴다. 그래서 배치 1 에서는 이득이 작고, 배치가 클수록 GPU 가 유리하다. `GPU_OK` 가 False 면 이 교시 셀은 "건너뜀".

**핵심 용어**
- **CUDAExecutionProvider**: GPU(NVIDIA CUDA)에서 실행하는 실행기
- **복사 비용**: 입력을 GPU 로, 출력을 CPU 로 옮기는 시간

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `CUDAExecutionProvider` | `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` | GPU 로 실행 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 5-1 | GPU 세션을 만들고 첫 5번의 시간 (첫 실행이 매우 느림) |
| 5-2 | CPU · GPU × 배치 1 · 32 비교표 |
| 5-3 | GPU vs CPU 출력 최대 차이 · 예측 일치율 (`atol=1e-3`) |

**핵심 내용**

#### GPU 실행 — 복사 · 계산 · 복사
> 입력 CPU→GPU 복사 → GPU 계산 → 출력 GPU→CPU 복사 — 계산이 빨라도 복사 · 준비 비용이 붙는다
- **배치 1** → 일이 작아 이득이 작음
- **배치 32** → 일이 많아 GPU 가 크게 유리
- **결과** → 장치가 달라도 같은 모델 (최대 차이 < 0.001)
- ⚠ `onnxruntime` 과 `onnxruntime-gpu` 를 둘 다 설치 — 꼬여서 GPU 가 안 보임 (CPU 판 삭제)
- ⚠ GPU 첫 실행 시간을 성능으로 기록 — warm-up 먼저
- ⚠ 배치 1 하나만 재고 "GPU 가 별로" 결론 · 큰 배치로 GPU 메모리 초과 시 `Failed to allocate memory`

**기본 사용형태**
```python
gpu_sess = make_session("mnv3s.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(gpu_sess.get_providers()[0])      # CUDAExecutionProvider 여야 진짜 GPU
```

**연결**
- [OT D3·6교시 배치 크기](#/ot/3/6) — 배치를 키울수록 GPU 가 유리한 이유의 연장
- [OT D4·TensorRT](#/ot/4/1) — GPU 에서 더 빠르게: 엔진으로 컴파일
- [VB D5·5교시 ONNX export · ONNX Runtime](#/vb/5/5) — CPU 실행의 출발점

**상세 학습**

#### 이 교시에서 답하는 질문
CPU 에서 손잡이를 돌려 봤으니 장치 자체를 바꾸면 어떨까. "GPU = 무조건 빠름" 은 아니며, 일의 크기와 복사 비용의 균형이라는 점을 확인한다.

#### 개념 1 · 배치별 GPU 의 입장
| 배치 | 일의 양 | GPU 의 입장 |
|---|---|---|
| 1 | 입력 1장 = 150,528 원소 | 복사 · 준비 비용 비중이 큼 |
| 32 | 1장의 32배 = 4,816,896 원소 | 병렬 계산이 비용을 상쇄 |
배치 32 입력 원소 수 = 32 × 150,528 = **4,816,896**.

#### 코드로 확인
```python
print(f"최대 차이 {np.abs(out_gpu - out_cpu).max():.1e} | 일치율 {(out_gpu.argmax(1) == out_cpu.argmax(1)).mean():.2f}")
```
- **무엇을 하나**: GPU 와 CPU 출력 비교
- **어떤 개념**: 장치가 바뀌어도 결과는 같다
- **출력**: 교안 기준 `최대 차이 < 1e-3`

#### 한 번에 정리
- providers 맨 앞에 CUDA → `get_providers()` 로 확인
- GPU = 계산 빠름 + 복사 · 첫 실행 비용 → 배치가 클수록 유리
- ⚠ `onnxruntime` · `onnxruntime-gpu` 동시 설치 금지

---

### 6교시 · 배치 크기: 지연 vs 처리량

**키워드**: `지연(latency)`, `처리량(throughput)`, `imgs_per_s = 배치 × 1000 ÷ 지연`, `장당 시간`, `카메라 수 = 처리량 ÷ FPS`

**개요**: 배치를 1 → 64 로 늘리며 두 숫자를 따로 본다. **지연** = 한 번 `run` 이 끝날 때까지 걸린 ms(실시간 카메라는 프레임마다 바로 답이 필요). **처리량** = 1초에 처리하는 장 수(녹화 일괄 처리 · 여러 카메라). 배치가 커지면 지연은 늘고, 장당 시간 · 처리량은 어느 지점에서 멈춘다.

**핵심 용어**
- **지연(latency)**: 한 번 `run` 의 시간 (ms)
- **처리량(throughput)**: 초당 처리 장 수 = 배치 × 1000 ÷ 지연
- **장당 시간**: 지연 ÷ 배치

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `DataFrame.idxmax()` | `df_b.imgs_per_s.idxmax()` | 값이 가장 큰 행의 인덱스 (최대 처리량 배치) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 6-1 | 배치 1 · 4 · 16 · 64 의 `latency_ms` · `imgs_per_s` · `ms_per_img` 표 |
| 6-2 | 지연 · 처리량 그래프(배치 축 log₂) + 처리량 최대 배치 |
| 6-3 | `처리량 // 30` = 30 FPS 카메라를 몇 대 감당하나 |

**핵심 내용**

#### 지연 vs 처리량
> 처리량 = 배치 × 1000 ÷ 지연(ms) — 배치를 키우면 처리량은 늘 수 있지만 지연도 같이 늘어난다
- **실시간 카메라** → 배치 1 · p95 (프레임마다 바로 답)
- **일괄 처리 · 여러 카메라** → 처리량이 큰 배치
- **예** → 배치 8, 지연 40 ms → 처리량 = 8 × 1000 ÷ 40 = **200 장/초**, 장당 5 ms
- ⚠ 배치 b 의 시간을 "한 장 시간" 으로 착각 — 장당 = 지연 ÷ b
- ⚠ 처리량만 보고 배치를 키움 — 지연이 예산(30 FPS 면 33.3 ms)을 넘으면 실시간 불가

#### 카메라 몇 대?
> 감당 카메라 수 = 처리량 ÷ 카메라 FPS (`//` 는 몫)
- **예** → 200 장/초 ÷ 30 = 6.67 → `//` 로 **6대**
- **신경망에서 / OT 에서** → Day 3 8교시 · 카메라 4대 = 초당 120장

**기본 사용형태**
```python
for b in [1, 4, 16, 64]:
    xb = np.random.rand(b, 3, 224, 224).astype(np.float32)
    r = bench(cpu_sess, xb, n=20, warm=3)
    print(b, r["mean"], r["fps"], r["mean"] / b)    # 지연 · 처리량 · 장당 ms
```

**연결**
- [OT D3·5교시 CPU → GPU](#/ot/3/5) — 배치를 키울수록 GPU 가 유리한 이유
- [OT D3·8교시 카메라 4대](#/ot/3/8) — 처리량 ÷ 30 계산을 실제 설정 고르기에 사용

**상세 학습**

#### 이 교시에서 답하는 질문
5교시에서 배치를 바꾸니 결과가 크게 달라졌다. 배치는 "속도" 하나로 말할 수 없고 지연과 처리량 두 숫자가 엇갈린다. 목적에 맞는 숫자를 고르는 법을 배운다.

#### 개념 1 · 지연 · 처리량 · 장당 시간의 관계
| 이름 | 식 | 배치 8, 지연 40 ms 일 때 |
|---|---|---|
| 지연 | `mean` | 40 ms |
| 처리량 | 배치 × 1000 ÷ 지연 | 8 × 1000 ÷ 40 = 200 장/초 |
| 장당 시간 | 지연 ÷ 배치 | 40 ÷ 8 = 5 ms |
`ms_per_img × imgs_per_s = 1000` 이 항상 성립 (5 × 200 = 1000).

#### 코드로 확인
```python
cams_b1 = int(df_b.imgs_per_s.iloc[0] // 30)
cams_best = int(df_b.imgs_per_s.max() // 30)
```
- **무엇을 하나**: 배치 1 일 때와 최대 처리량일 때 감당 카메라 수
- **어떤 개념**: 카메라 수 = 처리량 ÷ FPS
- **출력**: 환경마다 다름 (`cams_best >= cams_b1`)

#### 한 번에 정리
- 지연 = 한 번 시간 · 처리량 = 배치 × 1000 ÷ 지연
- 실시간 → 배치 1 · p95 / 일괄 · 여러 카메라 → 큰 배치
- ⚠ 지연이 예산(33.3 ms @30FPS)을 넘으면 처리량이 높아도 실시간 불가

---

### 7교시 · profiling: 어디가 느린가

**키워드**: `enable_profiling`, `end_profiling()`, `cat == "Node"`, `op_name`, `dur (µs)`

**개요**: 지금까지는 전체 시간만 쟀다. 더 빠르게 하려면 **어느 노드(연산)가 시간을 먹는지** 알아야 한다. 세션에 `enable_profiling=True` → `run` 여러 번 → **`end_profiling()`** 으로 JSON 파일을 닫고 → `cat == "Node"` 인 기록의 `op_name` 별 `dur` 합을 비율로 계산한다. 병목은 모델마다 다르다 — MobileNetV3 = `Conv`, SimpleConvNet = `FusedGemm`.

**핵심 용어**
- **profiling**: 노드마다 걸린 시간을 파일로 남기는 기능
- **병목(bottleneck)**: 전체 시간에서 가장 큰 비중을 차지하는 곳
- **dur**: 노드 한 번의 시간 — 단위 **µs** (1000 µs = 1 ms)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `end_profiling()` | `path = ps.end_profiling()` | 기록을 닫고 JSON 파일 경로를 돌려줌 |
| `collections.Counter()` | `agg[op_name] += dur` | op 별로 시간 합산 |
| `profile_top()` | `profile_top(path, x, k=3, n=20)` | 시간 비율 상위 k 개 연산 표 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 7-1 | profiling 세션으로 20번 실행 → JSON 에서 노드 기록만 추림 (`len % 20 == 0`) |
| 7-2 | `op_name` 별 `ms_per_run` · `share_%` 표 (합 100%) |
| 7-3 | `profile_top()` 으로 두 모델의 Top 3 비교 |

**핵심 내용**

#### profiling 5단계
> 켜기 → 여러 번 실행 → `end_profiling()` → JSON 읽기 → op 별 `dur` 합산
- **노드 기록 추리기** → `e.get("cat") == "Node"` 이고 이름이 `_kernel_time` 으로 끝남
- **비율** → `share_% = op 의 dur 합 ÷ 전체 dur 합 × 100` (전부 더하면 100%)
- **1회당 시간** → `dur 합 ÷ 실행 횟수(20) ÷ 1000` ms
- ⚠ `end_profiling()` 을 부르지 않고 파일을 찾음 — 파일이 안 닫혀 비어 있음
- ⚠ profiling 켠 세션으로 속도를 잼 — 기록하느라 느려짐 (재는 세션 · 기록하는 세션 분리)
- ⚠ `dur` 를 ms 로 착각 — µs

**기본 사용형태**
```python
ps = make_session("mnv3s.onnx", profile=True)
for _ in range(20): ps.run(None, {"input": x1})
events = json.load(open(ps.end_profiling()))
node_ev = [e for e in events if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")]
```

**연결**
- [OT D3·4교시 최적화 레벨](#/ot/3/4) — `FusedGemm` 이 Top 에 나오는 이유
- [OT D4·TensorRT](#/ot/4/1) — 병목을 알았으니 엔진 · 양자화로 줄이기

**상세 학습**

#### 이 교시에서 답하는 질문
레벨 · 스레드 · 배치로 전체 시간은 줄였지만 "어디를 더 손대야 하나" 는 모른다. 병목이 없는 곳을 고쳐 봐야 소용없으므로 노드별 시간을 본다.

#### 개념 1 · 비율 계산 예
Conv 600 µs, Relu 100 µs, Gemm 300 µs 라면 합 1000 µs.
| op | 시간(µs) | 비율 |
|---|---|---|
| Conv | 600 | 60% |
| Gemm | 300 | 30% |
| Relu | 100 | 10% |
합 = 100%. 1위 = Conv → 먼저 줄일 곳.

#### 코드로 확인
```python
agg = collections.Counter()
for e in node_ev: agg[e["args"]["op_name"]] += e["dur"]
total = sum(agg.values())
```
- **무엇을 하나**: op 별 시간 합과 전체 합
- **어떤 개념**: 비율 계산
- **출력**: 환경마다 다름 — `share_%` 의 합은 항상 100

#### 한 번에 정리
- profiling = 노드별 시간 JSON → op 별 합산 → 비율
- 병목은 모델마다 다름 (MobileNetV3 = Conv, SimpleConvNet = FusedGemm)
- ⚠ `end_profiling()` 필수 · `dur` 는 µs

---

### 8교시 · 미니 프로젝트: 현장 카메라 4대용 설정 정하기

**키워드**: `bench_config`, `grid 표`, `chosen_batch`, `ot_baseline.csv`, `카메라 4대 = 120장/초`

**개요**: 현장 카메라 4대(각 30 FPS = **초당 120장**)를 CPU 한 대로 MobileNetV3 분류하려 한다. 오늘 배운 손잡이로 실험하고 결과를 **`ot_baseline.csv`** 로 남긴 뒤 추천 설정을 쓴다. 채점은 숫자가 아니라 **방법과 일관성**(예: '최선' 이 표에서 정말 가장 작은 p95 인가). 이 CSV 에 Day 4 TensorRT 결과를 같은 열로 이어 붙인다.

| 과제 | 내용 | 배점 |
|---|---|---|
| 1 | `bench_config` — 설정 한 줄 → 측정 한 줄 | 20 |
| 2 | 스레드 {1, 2} × 레벨 {DISABLE, ALL} 실험표, p95 최소 = best | 20 |
| 3 | best 설정에서 배치 1 · 4 · 8, 120장/초 감당하는 최소 지연 배치 | 20 |
| 4 | best 설정 profiling Top 3 | 20 |
| 5 | `ot_baseline.csv` 저장 · p95 막대그래프 | 10 |
| 6 | 추천 설정과 근거 (마크다운) | 10 |

**핵심 용어**
- **ot_baseline.csv**: 조건 열(model · provider · threads · level · batch) + 결과 열(mean · p50 · p95 · fps)의 성능표
- **grid**: 손잡이 조합을 모두 재서 만든 표

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `bench_config()` | `bench_config(path, threads=0, level="ALL", batch=1, n=50)` | 설정을 받아 조건 + 결과 한 줄(dict) 돌려줌 |
| `pd.concat()` | `pd.concat([grid, batch_tbl], ignore_index=True)` | 표 이어 붙이기 |
| `DataFrame.idxmin()` | `grid["p95"].idxmin()` | 값이 가장 작은 행의 인덱스 (best 고르기) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 8-1 | `make_session` + `bench` 로 조건 + 결과 한 딕셔너리 |
| 8-2 | 2 × 2 = 4 가지 조합 grid → p95 최소 행 best |
| 8-3 | `fps >= 120` 인 배치 중 `mean` 최소 → `chosen_batch` (없으면 None) |
| 8-4 | best 설정으로 profiling → `top3` |
| 8-5 | grid + batch_tbl → CSV 저장 · 막대그래프 |

**핵심 내용**

#### 설정 고르기 = 조건 열 + 결과 열 한 줄
> 표가 쌓이면 결정의 근거가 된다 — 목적에 따라 기준이 다르다
- **실시간** → p95 · 지연 / **여러 카메라** → 처리량(fps)
- **카메라 4대 조건** → 4 × 30 = **120장/초** 이상 (`fps >= 120`)
- **고르는 순서** → 감당 가능한 것만 걸러(`ok`) → 그 중 지연 최소
- ⚠ 감당하는 배치가 하나도 없으면 `None` — 추천은 "스레드 · 장치 변경" 쪽으로

#### 감당이 아슬아슬할 때 (과제 6)
> CPU 한 대에서는 손잡이로, 그래도 모자라면 장치 변경
- **CPU 에서** → 스레드 · 최적화 레벨 조정, 배치 조정 (profiling 의 Top 연산 줄이기: Day 4 · Quantization)
- **장치 변경** → GPU (5교시) · TensorRT (Day 4)

**기본 사용형태**
```python
def bench_config(path, threads=0, level="ALL", batch=1, n=50):
    s_ = make_session(path, threads=threads, level=level)
    x_ = np.random.rand(batch, 3, 224, 224).astype(np.float32)
    return {"model": os.path.basename(path), "provider": s_.get_providers()[0],
            "threads": threads, "level": level, "batch": batch, **bench(s_, x_, n=n, warm=5)}
```

**연결**
- [OT D3·6교시 배치 크기](#/ot/3/6) — 처리량 ÷ FPS 로 카메라 수 계산
- [OT D4·TensorRT 첫걸음](#/ot/4/1) — `ot_baseline.csv` 에 TensorRT 결과를 이어 붙임
- [OT D3·7교시 profiling](#/ot/3/7) — 과제 4 의 Top 3

**상세 학습**

#### 이 교시에서 답하는 질문
오늘 배운 손잡이 다섯 개를 한 번에 쓰는 실전 문제 — "정말 이 설정이 최선인가?" 를 숫자와 표로 답한다. 숫자는 사람마다 다르므로 **방법**(하나씩 바꾸기 · 조건 열 · 같은 기준)이 평가 대상이다.

#### 개념 1 · 요구 처리량
| 항목 | 값 |
|---|---|
| 카메라 수 | 4 |
| 카메라당 FPS | 30 |
| 필요 처리량 | 4 × 30 = **120 장/초** |
| 한 장 예산 (배치 1) | 1000 ÷ 120 ≈ **8.3 ms** |

#### 개념 2 · 실험 설계
| 단계 | 바꾸는 것 | 고정 |
|---|---|---|
| 과제 2 | 스레드 · 레벨 (4가지) | 배치 1 |
| 과제 3 | 배치 (1 · 4 · 8) | best 의 스레드 · 레벨 |
단계마다 한 종류만 바꿔 원인을 알 수 있다.

#### 코드로 확인
```python
ok = batch_tbl[batch_tbl["fps"] >= 120]
chosen_batch = int(ok.loc[ok["mean"].idxmin(), "batch"]) if len(ok) else None
```
- **무엇을 하나**: 120장/초를 감당하는 배치 중 지연 최소 선택
- **어떤 개념**: 개념 1
- **출력**: 환경마다 다름 (`None` 일 수 있음)

#### 한 번에 정리
- 요구 = 4 × 30 = 120 장/초 · 필터(`fps >= 120`) 후 지연 최소
- 표(`ot_baseline.csv`)가 근거 · Day 4 에서 이어 붙임
- ⚠ 과제 6 은 숫자로 근거 + "한 번에 하나만" 원칙으로 레벨 vs 스레드 영향 설명
