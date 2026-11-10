## Day 4 — TensorRT 첫걸음: ONNX 를 엔진으로 바꾸기

### 1교시 · Day 3 복습 · TensorRT 란 · 설치 확인

**키워드**: `설계도 · 기계 · 완성품`, `.engine`, `compute capability`, `배치 1 고정 ONNX`, `ONNX 를 배포`

**개요**: ONNX 는 설계도, ONNX Runtime(ORT)은 설계도를 읽어 바로 돌리는 기계, **TensorRT 엔진은 "이 GPU 전용 완성품"** 이다. TensorRT 는 빌드할 때 이 GPU 에서 커널 후보를 직접 재 보고 가장 빠른 것을 골라 `.engine` 파일로 저장한다. 그래서 엔진은 만든 GPU 종류 · TensorRT 버전에서만 동작한다. 오늘은 배치 1 로 고정한 `mnv3s_b1.onnx` · `mnist_cnn_b1.onnx` 를 쓴다.

**핵심 용어**
- **TensorRT**: NVIDIA GPU(Jetson 포함) 전용 추론 최적화 · 실행 도구
- **엔진(engine)**: TensorRT 가 빌드해 저장한 실행 파일 (`.engine`) — GPU 세대 · 버전에 묶임
- **compute capability**: GPU 세대 번호 — T4 = (7, 5) → `sm_75`, Jetson Orin = 8.7
- **정답(ref)**: 오늘 비교 기준 — ORT CPU 출력

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.cuda.get_device_capability()` | `torch.cuda.get_device_capability(0)` | GPU 세대 번호 `(주, 부)` |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 1-1 | TensorRT 버전 · GPU 이름 · compute capability 를 `sm_75` 꼴로 출력 |
| 1-2 | `dynamic_axes` 없이 export → 배치 1 고정 ONNX 두 개 + ORT CPU 출력을 `ref` 로 저장 |

**핵심 내용**

#### ORT vs TensorRT
> ONNX 한 개를 두 방식으로 실행 — ORT 는 세션 만들 때 가볍게 최적화, TensorRT 는 빌드 때 오래 재 보고 결과를 파일로 저장
| | ONNX Runtime | TensorRT |
|---|---|---|
| 최적화 시점 | 세션 만들 때마다 (1초 이내) | 빌드 때 한 번 (수십 초 ~ 분) |
| 방법 | 정해진 규칙으로 노드 합치기 | 규칙 + 이 GPU 에서 커널을 직접 재 보고 선택 |
| 실행 장치 | CPU · GPU · 여러 장치 | NVIDIA GPU 전용 |
| 결과 | 메모리 안 (파일 없음) | `.engine` 파일 |
- ⚠ 엔진은 다른 GPU(A100 · Jetson) · 다른 TensorRT 버전으로 가져가면 못 읽는다 → **배포할 것은 `.onnx`**, 도착한 장치에서 다시 빌드

#### 배치 1 고정 ONNX
> `dynamic_axes` 를 안 주면 배치 축이 1 로 굳는다 — 오늘 TensorRT 빌드가 이것만 받는다
- **굳는 곳** → 입력 shape `[1, 3, 224, 224]`, 출력 `(1, 1000)`
- ⚠ 배치가 자유로운 ONNX(`["batch", ...]`)를 그대로 빌드하면 optimization profile 오류

**기본 사용형태**
```python
print("TensorRT 버전:", trt.__version__)
cc = torch.cuda.get_device_capability(0)
print(cc, f"sm_{cc[0]}{cc[1]}")      # T4 → (7, 5) sm_75
```

**연결**
- [OT D3·1교시 ONNX Runtime 성능](#/ot/3/1) — ORT 가 하던 최적화를 TensorRT 가 이어받는다
- [OT D1·2교시 첫 ONNX 만들기](#/ot/1/2) — `torch.onnx.export` 가 배치 1 고정을 만드는 원리

**상세 학습**

#### 이 교시에서 답하는 질문
Day 3 의 ORT 는 CPU 에서도 GPU 에서도 돌았지만 "NVIDIA GPU 에서 최고 속도" 는 아니었다. TensorRT 는 이 GPU 한 대에 맞춰 미리 빌드해 두는 방식으로 속도를 더 끌어낸다. 대신 결과물(엔진)이 이식되지 않는다.

#### 개념 1 · 설계도 → 기계 → 완성품
ONNX(설계도)는 어디서나 읽힌다. ORT 는 그 자리에서 읽고 바로 실행하는 기계. TensorRT 는 이 GPU 에 맞춰 조립을 끝낸 완성품이라 빠르지만 다른 GPU 에서는 쓸 수 없다.

#### 코드로 확인
```python
cc = torch.cuda.get_device_capability(0)
print("compute capability:", cc, "→", f"sm_{cc[0]}{cc[1]}")
```
- **무엇을 하나**: GPU 세대 번호 확인
- **어떤 개념**: 개념 1 — 엔진이 묶이는 조건
- **출력**: GPU 에 따라 다름 (교안 해설: T4 는 `(7, 5)` → `sm_75`)

#### 한 번에 정리
- ONNX = 설계도 · ORT = 읽어서 바로 실행 · 엔진 = 이 GPU 전용 완성품
- GPU 세대 · TensorRT 버전이 바뀌면 다시 빌드 → 배포는 `.onnx`
- ⚠ 엔진 파일을 복사해 다른 장치에서 쓰려 하면 읽지 못한다

---

### 2교시 · 명령 한 줄로 엔진 만들기: polygraphy convert

**키워드**: `Polygraphy`, `polygraphy convert`, `--fp16`, `종료 코드 0`, `[W] 경고 vs [E] 오류`

**개요**: Polygraphy 는 NVIDIA 가 만든 TensorRT 도우미 도구다 (Jetson · 서버의 `trtexec` 는 pip 판 TensorRT 에 없어 Colab 에서는 Polygraphy 로 같은 일을 한다). `polygraphy convert 모델.onnx -o 엔진.engine` 한 줄이 파싱 → 최적화 → 커널 고르기 → 저장을 해 준다. 노트북에서는 환경 셀의 `sh("명령", log="파일")` 로 실행한다.

**핵심 용어**
- **Polygraphy**: TensorRT 빌드 · 비교 · 점검을 명령 한 줄로 해 주는 도구
- **종료 코드(return code)**: 명령 성공 여부 — 0 이 성공

**새로 나온 명령**
| 명령 | 기본형 | 용도 |
|---|---|---|
| `polygraphy convert` | `polygraphy convert m.onnx -o m.engine` | ONNX → 엔진 빌드 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 2-1 | `mnv3s_b1.onnx` → `mnv3s_fp32_cli.engine`, 종료 코드 · 걸린 시간 출력 |
| 2-2 | 로그에서 `[W]` · `[E]` 줄을 세고 ONNX · 엔진 파일 크기(MB) 비교 |
| 바꿔보기 | `--fp16` 만 붙여 FP16 엔진 크기 비교 |

**핵심 내용**

#### 엔진 빌드 = 오래 걸리는 한 번의 작업
> 빌드 = ONNX 파싱 → 최적화 → **레이어마다 후보 커널을 이 GPU 에서 돌려 보고 가장 빠른 것 선택** → 파일 저장
- **왜 오래 걸리나** → 커널을 실제로 재 보기 때문 (수십 초 ~ 분)
- **성공 판정** → 종료 코드 `0` — 로그의 `[W]` 는 경고일 뿐, `[E]` 가 오류
- **옵션** → `--fp16` 만 붙이면 FP16 엔진
- ⚠ 빌드 셀을 여러 번 누르면 빌드가 겹쳐 더 느려지고 GPU 메모리가 모자랄 수 있다 — 한 번 누르고 기다리기

**기본 사용형태**
```python
rc = sh("polygraphy convert mnv3s_b1.onnx -o mnv3s_fp32_cli.engine", log="build_fp32.log")
log = open("build_fp32.log").read().splitlines()
warn = [l for l in log if l.startswith("[W]")]
err  = [l for l in log if l.startswith("[E]")]
print("경고", len(warn), "오류", len(err))
```

**연결**
- [OT D4·4교시 Python 으로 엔진 빌드](#/ot/4/4) — 이 명령이 안에서 하던 일을 코드 여섯 단계로
- [OT D1·4교시 ONNX Runtime 으로 실행](#/ot/1/4) — 빌드 없이 바로 실행하던 방식과 대비

**상세 학습**

#### 이 교시에서 답하는 질문
엔진은 어떻게 만드는가? 가장 짧은 길은 명령 한 줄이다. 4교시에서 같은 일을 파이썬으로 풀어 쓰기 전에, 먼저 "넣으면 엔진이 나온다" 를 체험한다.

#### 개념 1 · polygraphy convert
입력은 `.onnx`, 출력은 `.engine`. 옵션 없으면 FP32. 시간이 오래 걸리는 것은 정상이며, 로그의 마지막 줄에 빌드 시간이 남는다.

#### 코드로 확인
```python
log = open("build_fp32.log").read().splitlines()
err = [l for l in log if l.startswith("[E]")]
print("오류", len(err), "줄")
print({f: f"{os.path.getsize(f)/1e6:.1f} MB" for f in ["mnv3s_b1.onnx", "mnv3s_fp32_cli.engine"]})
```
- **무엇을 하나**: 오류 줄 수와 파일 크기 확인
- **어떤 개념**: 개념 1 — 성공 판정은 `[E]` 와 종료 코드
- **출력**: 교안 기대값 — 오류 0 줄 (파일 크기는 실행 환경마다 다름)

#### 한 번에 정리
- `polygraphy convert 모델.onnx -o 엔진.engine [--fp16]`
- 로그는 `[E]` · `[W]` · 빌드 시간을 본다
- ⚠ 엔진을 다른 GPU 로 가져가면 못 읽음 → ONNX 를 가져가 다시 빌드

---

### 3교시 · 명령 한 줄로 비교하기: polygraphy run --onnxrt --trt

**키워드**: `polygraphy run`, `PASSED / FAILED`, `atol · rtol`, `|a − b| ≤ atol + rtol·|b|`, `FP16 은 느슨하게`

**개요**: 엔진이 빨라도 답이 같아야 쓴다. `polygraphy run 모델.onnx --onnxrt --trt` 는 같은 무작위 입력을 ORT 와 TensorRT 에 넣고 출력을 비교해 PASSED / FAILED 를 알려 준다. 허용 오차는 `--atol` · `--rtol` 로 정한다.

**핵심 용어**
- **atol / rtol**: 절대 · 상대 허용 오차 — 두 출력이 "같다" 고 보는 기준

**새로 나온 명령**
| 명령 | 기본형 | 용도 |
|---|---|---|
| `polygraphy run` | `polygraphy run m.onnx --onnxrt --trt --atol 1e-3 --rtol 1e-3` | ORT 와 TensorRT 출력 비교 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 3-1 | FP32 엔진을 엄격한 기준(1e-3)으로 비교 → PASSED |
| 3-2 | 로그에서 판정 · 최대 차이를 읽는 `verdict()` 로 FP32 · FP16(엄격 · 느슨) 세 로그 비교 |

**핵심 내용**

#### 허용 오차 판정
> 두 출력 a, b 가 같다 ⇔ `|a − b| ≤ atol + rtol · |b|`
- **식** → atol 은 0 근처 값용 고정 여유, rtol 은 값 크기에 비례하는 여유
- **FP32 엔진** → 엄격하게(1e-3) — 사실상 같은 답이어야 함
- **FP16 엔진** → 정밀도가 낮아 엄격 기준에서는 거의 FAILED → 느슨하게(5e-2) + **실제 정확도로 판단**
- **실행 구조** → `--trt` 를 주면 안에서 엔진을 새로 빌드하므로 또 수십 초
- ⚠ 무작위 입력 한 번의 PASSED 로 "완벽" 결론 금지 — 6교시처럼 실제 데이터로 확인
- ⚠ `--onnxrt` 를 빠뜨리면 비교 대상이 없어 실행만 한다

**기본 사용형태**
```python
sh("polygraphy run mnv3s_b1.onnx --onnxrt --trt --atol 1e-3 --rtol 1e-3", log="cmp_fp32.log", tail=6)
text = open("cmp_fp32.log").read()
print("PASSED" if re.search(r"\bPASSED \| Runtime", text) else "FAILED")
```

**연결**
- [OT D1·4교시 allclose 로 PyTorch 와 비교](#/ot/1/4) — 같은 `atol`/`rtol` 판정을 손으로 했던 것
- [OT D4·5교시 코드로 최대 차이 확인](#/ot/4/5) — PASSED 를 파이썬으로 직접 재현

**상세 학습**

#### 이 교시에서 답하는 질문
빌드한 엔진이 원래 모델과 같은 답을 내는가? 속도만 보고 쓰면 위험하다. 같은 입력을 두 실행기에 넣어 비교하는 것을 명령 하나로 자동화한다.

#### 개념 1 · atol · rtol 판정식
기호: `a` = TensorRT 출력, `b` = ORT 출력(기준). 예: `b = 100` 이고 `atol = rtol = 1e-3` 이면 허용 차이는 `1e-3 + 1e-3 × 100 = 0.101`. `b = 0` 이면 `1e-3` 만 허용 — 그래서 atol 이 필요하다.

#### 개념 2 · FP16 에는 FP16 의 기준
FP16 은 유효숫자가 3자리 안팎이라 1e-3 같은 엄격한 기준을 넘기 쉽다. 기준을 풀고, 최종 판단은 분류 정확도(6교시)로 한다.

#### 코드로 확인
```python
def verdict(log_path):
    text = open(log_path).read()
    result = "PASSED" if re.search(r"\bPASSED \| Runtime", text) else "FAILED"
    m = re.findall(r"Absolute Difference.*?max=([0-9.eE+-]+)", text)
    return result, (float(m[-1]) if m else None)
```
- **무엇을 하나**: 로그에서 판정과 최대 절대 차이를 꺼낸다
- **어떤 개념**: 개념 1 — 실제 최대 차이가 atol 보다 얼마나 작은지
- **출력**: 교안 기대값 — FP32 는 `PASSED`

#### 한 번에 정리
- `polygraphy run … --onnxrt --trt [--fp16] --atol … --rtol …`
- 판정 식 `|a − b| ≤ atol + rtol·|b|` · FP32 엄격 / FP16 느슨 + 정확도
- ⚠ FP16 에 FP32 와 같은 `atol` → 거의 항상 FAILED

---

### 4교시 · Python 으로 엔진 빌드 → build_engine()

**키워드**: `Logger → Builder → Network → OnnxParser → Config → build_serialized_network`, `explicit batch`, `WORKSPACE`, `DETAILED`, `build_engine()`

**개요**: `polygraphy convert` 가 안에서 하던 일을 코드 **여섯 단계**로 직접 한다. ① Logger ② Builder ③ Network(빈 설계판) ④ OnnxParser 로 채움 ⑤ Config(작업 메모리 · FP16 등) ⑥ `build_serialized_network` → 파일 저장. 마지막에 함수 `build_engine(onnx, engine, fp16=False)` 로 묶어 이후 교시에서 그대로 쓴다.

**핵심 용어**
- **Builder / Network**: 엔진을 만드는 공장 / 파싱 결과를 담는 빈 설계판
- **OnnxParser**: ONNX 를 읽어 Network 를 채우는 도구
- **BuilderConfig**: 빌드 옵션 모음 (작업 메모리 · 정밀도 · 프로파일링)
- **직렬화(serialize)**: 엔진을 바이트 덩어리(blob)로 만들어 파일에 쓰는 일

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `trt.Logger()` | `trt.Logger(trt.Logger.WARNING)` | TensorRT 메시지 받는 곳 |
| `trt.Builder()` | `trt.Builder(LOGGER)` | 엔진 공장 |
| `builder.create_network()` | `builder.create_network(0)` | 빈 Network (0 = explicit batch) |
| `trt.OnnxParser()` | `trt.OnnxParser(network, LOGGER)` | ONNX 파서 |
| `parser.parse_from_file()` | `parser.parse_from_file("m.onnx")` | ONNX 를 읽어 Network 채움 (성공 True) |
| `builder.create_builder_config()` | `builder.create_builder_config()` | 빌드 옵션 객체 |
| `config.set_memory_pool_limit()` | `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)` | 빌드 작업 메모리 상한 (1 GB) |
| `config.set_flag()` | `config.set_flag(trt.BuilderFlag.FP16)` | FP16 허용 |
| `builder.build_serialized_network()` | `builder.build_serialized_network(network, config)` | 엔진 빌드 → blob |
| `build_engine()` | `build_engine("m.onnx", "m.engine", fp16=False)` | 위 여섯 단계를 묶은 오늘의 함수 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 4-1 | 파싱 후 성공 여부 · 레이어 수 · 입력 이름 / shape `(1, 3, 224, 224)` 확인 |
| 4-2 | Config 설정 → 빌드 → `mnv3s_fp32.engine` 저장 |
| 4-3 | 단계를 `build_engine(onnx, engine, fp16)` 로 묶어 `cnn_fp32.engine` 생성 |

**핵심 내용**

#### 엔진 빌드 여섯 단계
> `Logger → Builder → Network → OnnxParser → Config → build_serialized_network` — convert 명령과 같은 일
- **① Logger** → WARNING 이상만 보이게
- **③ Network** → `create_network(0)` = 배치를 shape 에 그대로 적는 explicit batch
- **④ 파싱** → 결과(True/False)를 반드시 확인, 실패 시 `parser.get_error(i)` 로 이유
- **⑤ Config** → `WORKSPACE` 1 GB · `profiling_verbosity = DETAILED`(7교시 인스펙터용) · FP16 플래그
- ⚠ 파싱 실패를 확인 않고 빌드하면 빈 설계판 → blob 이 `None`
- ⚠ Builder · Network · Config 를 전역으로 재사용하면 이전 설계가 섞인다 → **빌드마다 새로** (그래서 `build_engine` 이 안에서 만든다)

**기본 사용형태**
```python
builder = trt.Builder(LOGGER)
network = builder.create_network(0)
parser = trt.OnnxParser(network, LOGGER)
parser.parse_from_file("mnv3s_b1.onnx")
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
blob = builder.build_serialized_network(network, config)
open("mnv3s_fp32.engine", "wb").write(blob)
```

**연결**
- [OT D4·2교시 polygraphy convert](#/ot/4/2) — 같은 일을 하던 명령 한 줄
- [OT D4·6교시 FP16](#/ot/4/6) — `fp16=True` 로 같은 함수를 재사용
- [OT D5·1교시 TRTInfer 클래스](#/ot/5/1) — 실행 쪽(trt_runner)은 클래스로 발전

**상세 학습**

#### 이 교시에서 답하는 질문
명령 한 줄은 편하지만 다른 코드 속에서 쓰기 어렵다. 파이썬 함수로 만들어 두면 이후 어떤 ONNX 든 같은 방식으로 엔진을 만들 수 있다.

#### 개념 1 · 여섯 단계 = 공장 비유
Logger(작업 일지) → Builder(공장) → Network(빈 설계판) → Parser(설계도를 판에 옮겨 적는 사람) → Config(공장 설정) → 빌드(완성품 출고). 출고물 blob 을 파일에 쓰면 엔진이다.

#### 개념 2 · 메모리 상한 계산
`1 << 30` = 2³⁰ B = 1,073,741,824 B ≈ 1 GiB. 빌드할 때 커널을 재 보는 작업 공간의 상한이다.

#### 코드로 확인
```python
def build_engine(onnx_path, engine_path, fp16=False):
    builder = trt.Builder(LOGGER); network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("파싱 실패")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if fp16: config.set_flag(trt.BuilderFlag.FP16)
    open(engine_path, "wb").write(builder.build_serialized_network(network, config))
```
- **무엇을 하나**: 파싱 확인 → Config → 빌드 → 저장을 한 함수로
- **어떤 개념**: 개념 1 — 여섯 단계
- **출력**: 교안 기대값 — `4-1` 입력 shape `(1, 3, 224, 224)`, 엔진 파일 생성

#### 한 번에 정리
- 여섯 단계: Logger · Builder · Network · OnnxParser · Config · build_serialized_network
- `build_engine(onnx, engine, fp16=False)` — convert 와 같은 일
- ⚠ 파싱 결과 확인 · 빌드마다 새 Builder

---

### 5교시 · Python 으로 엔진 실행 → trt_runner()

**키워드**: `Runtime → deserialize_cuda_engine`, `ExecutionContext`, `set_tensor_address`, `execute_async_v3`, `synchronize`, `GPU 주소만 받는다`, `trt_runner()`

**개요**: ORT 는 NumPy 를 그대로 받았지만 **TensorRT 는 GPU 메모리 주소만** 받는다. GPU 메모리는 PyTorch CUDA 텐서로 준비하고 `data_ptr()` 주소를 알려 준다. 순서: ① Runtime 으로 엔진 읽기 ② Context ③ 입출력 이름 · shape ④ CUDA 버퍼 + CPU→GPU 복사 ⑤ 주소 알려 주기 · 실행 · 기다리기 ⑥ GPU→CPU. 이를 `trt_runner(engine_path)` 로 묶는다 — 버퍼는 한 번만 준비하고 `run(x)` 만 여러 번.

**핵심 용어**
- **Runtime**: 엔진 파일을 읽는 도구 (빌드의 Builder 에 해당)
- **ExecutionContext**: 실행 한 번에 필요한 상태 (입출력 주소 등) — 엔진 하나로 여러 개 가능
- **비동기 실행**: `execute_async_v3` 는 바로 돌아옴 → `torch.cuda.synchronize()` 로 끝까지 기다려야 함

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `trt.Runtime()` | `trt.Runtime(LOGGER)` | 엔진을 읽는 도구 |
| `deserialize_cuda_engine()` | `RUNTIME.deserialize_cuda_engine(bytes)` | 파일 바이트 → 엔진 |
| `engine.create_execution_context()` | `engine.create_execution_context()` | 실행 Context |
| `engine.get_tensor_name()` | `engine.get_tensor_name(i)` | 입출력 텐서 이름 (`num_io_tensors` 개) |
| `engine.get_tensor_mode()` | `engine.get_tensor_mode(name)` | INPUT / OUTPUT 구분 |
| `context.set_tensor_address()` | `context.set_tensor_address(name, buf.data_ptr())` | 텐서 이름에 GPU 주소 연결 |
| `context.execute_async_v3()` | `context.execute_async_v3(torch.cuda.current_stream().cuda_stream)` | 비동기 실행 |
| `torch.cuda.synchronize()` | `torch.cuda.synchronize()` | GPU 작업 끝까지 기다림 |
| `trt_runner()` | `run = trt_runner("m.engine"); y = run(x)` | 위 단계를 묶은 오늘의 함수 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 5-1 | 엔진 입출력 표 — 이름은 export 때 붙인 `input` · `logits`, 입력 `(1, 3, 224, 224)` |
| 5-2 | CUDA 버퍼 · 주소 · 실행 · 대기 · 결과 → ORT CPU 와 최대 차이 비교 |
| 5-3 | `trt_runner` — 버퍼를 안쪽 `run(x)` 가 붙잡아 재사용 |
| 5-4 | ORT CPU · ORT CUDA · TRT FP32 를 `bench` / `bench_fn` 으로 재고 `speedup` 열 |

**핵심 내용**

#### TensorRT 실행 = 주소 알려 주기
> 엔진 읽기 → Context → GPU 버퍼 → `set_tensor_address` → `execute_async_v3` → `synchronize` → 결과 가져오기
- **ORT 와 차이** → ORT 는 NumPy 입력, TensorRT 는 GPU 주소(`data_ptr()`)
- **입력 dtype** → `np.random.rand` 는 float64 → 반드시 `.astype(np.float32)`
- **버퍼** → 한 번만 만들고 `copy_` 로 덮어쓰기 (매번 새로 만들면 느림)
- ⚠ NumPy 배열을 그대로 `set_tensor_address` 에 넣으면 안 됨 — CUDA 텐서의 `data_ptr()`
- ⚠ 실행 뒤 `synchronize` 없이 시간을 재면 끝나기 전 시간이라 엄청 빨라 보임

#### trt_runner — 준비는 한 번, 실행은 여러 번
> `trt_runner(path)` 가 엔진 · Context · 버퍼를 준비하고, 안쪽 `run(x)` 가 복사 → 실행 → 대기 → NumPy 반환
- **엔진 붙잡기** → `run.engine, run.context = eng, ctx` (사라져 메모리 오류 나지 않게)
- **Day 5 에서** → 클래스 `TRTInfer` 로 발전

**기본 사용형태**
```python
RUNTIME = trt.Runtime(LOGGER)
eng = RUNTIME.deserialize_cuda_engine(open("mnv3s_fp32.engine", "rb").read())
ctx = eng.create_execution_context()
ctx.set_tensor_address("input", d_in.data_ptr())
ctx.set_tensor_address("logits", d_out.data_ptr())
ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()
```

**연결**
- [OT D1·5교시 입출력 이름 붙이기](#/ot/1/5) — `input_names` · `output_names` 가 엔진 텐서 이름이 된다
- [OT D5·1교시 TRTInfer 클래스](#/ot/5/1) — `trt_runner` 를 클래스로
- [OT D3·2교시 bench 성능 측정](#/ot/3/2) — 같은 `bench` 방식으로 TensorRT 도 잼

**상세 학습**

#### 이 교시에서 답하는 질문
엔진 파일을 만들었으니 이제 실행해야 한다. 그런데 ORT 처럼 `run(None, {...})` 한 줄이 아니라, GPU 메모리를 직접 준비하고 주소를 건네야 한다. 왜, 어떻게?

#### 개념 1 · 왜 GPU 주소인가
TensorRT 는 GPU 안에서만 돌기 때문에 입력이 이미 GPU 메모리에 있어야 한다. CPU(NumPy) → GPU 복사는 사용자가 하고, 어디에 있는지 주소(`data_ptr()`)만 알려 준다.

#### 개념 2 · 비동기 실행과 기다리기
`execute_async_v3` 는 일을 시켜 놓고 바로 돌아온다. 결과를 읽거나 시간을 재기 전에 `synchronize()` 가 필요하다.

#### 코드로 확인
```python
def run(x):
    d_i.copy_(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return d_o.cpu().numpy()
```
- **무엇을 하나**: 같은 GPU 버퍼에 입력을 덮어쓰고 실행 → 결과를 NumPy 로
- **어떤 개념**: 개념 1 · 2
- **출력**: 교안 기대값 — 출력 shape `(1, 1000)`, ORT CPU 와 최대 차이 < 1e-3

#### 한 번에 정리
- Runtime → 엔진 → Context → CUDA 버퍼 → 주소 → 실행 → `synchronize` → CPU 로
- `trt_runner(engine)` → `run(x)` — 버퍼는 한 번만
- ⚠ GPU 주소만 · float32 입력 · 실행 뒤 대기

---

### 6교시 · FP16 으로 더 빠르게: 속도 · 정확도

**키워드**: `FP16 = 2 B`, `최대 65504`, `eps ≈ 0.001`, `Tensor Core`, `build_engine(fp16=True)`, `입출력은 FP32 그대로`

**개요**: FP16 은 숫자 하나를 16비트(2바이트)로 저장한다. 메모리가 절반이고 T4 의 Tensor Core 가 FP16 을 훨씬 빨리 계산한다. 대신 표현 범위와 정밀도가 줄어든다. 빌드 Config 에 FP16 플래그 한 줄이면 되고(`build_engine(..., fp16=True)`), **속도 · 파일 크기 · 오차 · 실제 정확도**를 함께 확인한다.

**핵심 용어**
- **FP16(half precision)**: 16비트 부동소수점 — 범위 ±65504, 유효숫자 약 3자리
- **Tensor Core**: NVIDIA GPU 의 행렬 곱 전용 연산기 — FP16 에서 매우 빠름
- **eps**: 1 근처에서 표현 가능한 가장 작은 간격 (FP16 ≈ 0.000977, FP32 ≈ 1.19e-07)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `np.float16()` | `float(np.float16(0.1))` | FP16 으로 저장한 값 확인 |
| `np.finfo()` | `np.finfo(np.float16).max` | 자료형의 최대값 · eps |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 6-1 | 0.1 · 3.14159265 · 1000.3 · 70000 을 FP16 으로 저장해 오차 · 범위 확인 |
| 6-2 | FP16 엔진 빌드 → FP32 와 파일 크기 · 속도 · ORT 와 최대 차이 비교 |
| 6-3 | MNIST 100장 정확도: ORT CPU · TRT FP32 · TRT FP16 |

**핵심 내용**

#### FP16 — 절반 크기, 절반 정밀도
> 16비트 부동소수점: 메모리 1/2 · Tensor Core 로 빠름 · 대신 범위 65504 · 유효숫자 약 3자리
| 값 | FP16 으로 저장하면 |
|---|---|
| 0.1 | 0.0999755859375 |
| 3.14159265 | 3.140625 |
| 1000.3 | 1000.5 (1000 근처 간격 0.5) |
| 70000 | 65504 를 넘어 `inf` |
- **엔진의 입출력** → 여전히 FP32 (안쪽 계산만 FP16) — 입력을 float16 으로 바꿔 넣으면 안 됨
- **판단 방법** → 파일 ↓ · 속도 ↑ · 최대 차이 ↑ → **정확도(MNIST)로 판정**
- ⚠ "FP16 은 무조건 2배 빠르다" 아님 — 복사 · 작은 레이어 · FP32 로 남은 레이어 때문에 모델마다 다름 → 재 보고 말하기
- ⚠ 속도만 기록하지 말고 성능표에 정확도 열을 함께

**기본 사용형태**
```python
build_engine("mnv3s_b1.onnx", "mnv3s_fp16.engine", fp16=True)
run16 = trt_runner("mnv3s_fp16.engine")
diff16 = float(np.abs(run16(x1) - ref).max())
```

**연결**
- [OT D4·3교시 polygraphy 허용 오차](#/ot/4/3) — FP16 은 느슨한 기준으로
- [OT D5·3교시 FP16 정확도 AP](#/ot/5/3) — 탐지 모델에서도 같은 질문
- [VB D5·5교시 ONNX export · ORT](#/vb/5/5) — 속도 비교의 출발점

**상세 학습**

#### 이 교시에서 답하는 질문
더 빠르게 하려면? TensorRT 의 가장 큰 효과 중 하나는 정밀도를 낮추는 것이다. 정밀도를 낮추면 얼마나 빨라지고, 답은 얼마나 틀어지는가?

#### 개념 1 · FP16 의 범위와 간격
FP16 은 최대 65504 (이를 넘으면 `inf`). 1 근처 간격 eps = 2⁻¹⁰ = 0.0009765625. 값이 클수록 간격이 벌어져 512 ~ 1024 구간은 0.5 간격 → 1000.3 이 1000.5 로 저장된다. 0.1 의 저장 오차는 약 2.4e-05.

#### 개념 2 · 신경망은 왜 견디나
신경망은 작은 오차에 둔감해서 분류 결과(top-1)는 대개 그대로다. 그래서 숫자 오차(max_diff)와 정확도를 **둘 다** 본다.

#### 코드로 확인
```python
for v in [0.1, 3.14159265, 1000.3, 70000.0]:
    print(f"{v:>12} → FP16 {float(np.float16(v))!r}")
```
- **무엇을 하나**: FP16 저장 값 확인
- **어떤 개념**: 개념 1
- **출력**: `0.0999755859375` · `3.140625` · `1000.5` · `inf` (70000 은 범위 초과)

#### 한 번에 정리
- FP16 = 2 B · 최대 65504 · 유효숫자 약 3자리 · Tensor Core 로 빠름
- `build_engine(..., fp16=True)` — Config 한 줄
- 파일 ↓ · 속도 ↑ · 오차 ↑ → 정확도로 판단
- ⚠ FP16 엔진도 입출력은 FP32 · 속도는 재서 확인

---

### 7교시 · 왜 빨라졌나: 엔진 레이어 들여다보기

**키워드**: `엔진 인스펙터`, `ONNX 노드 vs 엔진 레이어`, `레이어 합치기(fusion)`, `이름의 " + "`, `profiling_verbosity = DETAILED`

**개요**: 빌드할 때 `DETAILED` 로 해 두었기 때문에 엔진 안의 레이어 목록을 꺼내 볼 수 있다. **ONNX 노드 수 vs 엔진 레이어 수**를 비교하면 무엇이 합쳐졌는지 보이고, 이름에 ` + ` 가 들어간 레이어가 합쳐진 레이어다. 빨라진 이유: ① 레이어 합치기(메모리 왕복 ↓) ② 이 GPU 에서 재 보고 고른 커널 ③ FP16(Tensor Core).

**핵심 용어**
- **레이어 합치기(layer fusion)**: Conv + BN + 활성화처럼 연속 연산을 하나의 커널로 묶는 최적화
- **엔진 인스펙터(engine inspector)**: 엔진 안 레이어 정보를 JSON 으로 꺼내는 도구

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `create_engine_inspector()` | `engine.create_engine_inspector()` | 인스펙터 만들기 |
| `get_engine_information()` | `insp.get_engine_information(trt.LayerInformationFormat.JSON)` | 레이어 정보 (JSON 문자열) |
| `polygraphy inspect model` | `polygraphy inspect model m.engine --show layers` | 명령으로 레이어 보기 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 7-1 | ONNX 노드 수 vs 엔진 레이어 수, 앞 6개 이름 출력 |
| 7-2 | 이름에 ` + ` 가 있는 합쳐진 레이어 · 가장 많이 합친 레이어 · `Reformat` 개수 |
| 7-3 | 레이어 종류(`LayerType`) 빈도, polygraphy 로 같은 목록 |

**핵심 내용**

#### 레이어 합치기 (fusion)
> 연속된 연산(Conv · BN · ReLU 등)을 하나의 커널로 → 메모리를 오가는 횟수가 줄어 빨라진다
- **보는 법** → 엔진 레이어 이름의 ` + ` = 합쳐진 ONNX 노드들 (`n.split(" + ")`)
- **노드 수 비교** → ONNX 노드 수보다 엔진 레이어 수가 적은 것이 보통 (실제 개수는 교안 실행값)
- **`Reformat` 레이어** → 데이터 배치 형식 변환용 — 빨라진 이유가 아니라 부수 비용
- **빨라진 이유 3가지** → ① 합치기 ② 이 GPU 에서 재 보고 고른 커널 ③ FP16(Tensor Core)
- ⚠ `DETAILED` 없이 빌드한 엔진을 인스펙트하면 이름만 나오고 `LayerType` 이 없다 (그래서 `build_engine` 에 넣어 둠)
- ⚠ 레이어 수가 적을수록 무조건 빠르다고 단정 금지 — 결국 bench 로 확인

**기본 사용형태**
```python
insp = run32.engine.create_engine_inspector()
info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
names = [l["Name"] if isinstance(l, dict) else l for l in info["Layers"]]
fused = [n for n in names if " + " in n]
```

**연결**
- [OT D1·3교시 Netron 으로 그래프 보기](#/ot/1/3) — ONNX 쪽 노드 · 엔진 쪽 레이어를 비교
- [OT D2·2교시 onnxslim 그래프 줄이기](#/ot/2/2) — 그래프 수준 최적화와 대비
- [NN D4·4교시 BatchNorm](#/nn/4/4) — BN 이 Conv 로 흡수되는 원리

**상세 학습**

#### 이 교시에서 답하는 질문
TensorRT 엔진이 빠른 건 알았다. 정확히 무엇이 달라졌길래? 엔진 안을 열어 보면 ONNX 노드가 합쳐져 있다.

#### 개념 1 · 메모리 왕복이 줄어든다
레이어마다 결과를 GPU 메모리에 쓰고 다시 읽으면 시간이 걸린다. Conv → BN → ReLU 를 한 커널로 묶으면 중간 결과를 메모리에 쓰지 않고 바로 다음 계산으로 넘긴다.

#### 코드로 확인
```python
fused = [n for n in names if " + " in n]
parts = [len(n.split(" + ")) for n in fused]
print("합쳐진 레이어", len(fused), "개 · 여기에 들어간 ONNX 노드", sum(parts), "개")
```
- **무엇을 하나**: 합쳐진 레이어와 그 안의 노드 수를 센다
- **어떤 개념**: 개념 1
- **출력**: 교안 기대값 — 합쳐진 레이어가 있고 ONNX 노드 수보다 엔진 레이어 수가 적음 (정확한 숫자는 실행 환경에서)

#### 한 번에 정리
- 인스펙터 → JSON → 레이어 이름 · 종류. 이름의 ` + ` = 합쳐진 레이어
- 빨라진 이유 = 합치기 · GPU 에서 고른 커널 · FP16
- ⚠ `DETAILED` 빌드 필요

---

### 8교시 · 미니 프로젝트: 실행기 4가지 성능표

**키워드**: `ORT CPU · ORT CUDA · TRT FP32 · TRT FP16`, `실행기 = 같은 모양 함수(x → 점수)`, `bench_fn`, `speedup`, `ot_baseline.csv`

**개요**: 같은 `mnv3s_b1.onnx` 를 네 가지 실행기로 돌려 **속도 + 정확도를 한 표**로 만들고 Day 3 의 `ot_baseline.csv` 에 이어 붙인다. 실행기를 모두 "x 를 받아 점수를 돌려주는 같은 모양의 함수" 로 맞추면 반복문 하나로 비교된다. 과제 6개(배점 20·20·20·20·10·10): 4행 측정표 · 정확도 열 · MNIST 정확도 · 속도 비교 · CSV 이어 붙이기 · 정리 · 추천.

**핵심 용어**
- **성능표**: 조건(provider · precision · batch) + 속도(mean · p95 · fps) + 정확도(max_diff · top1_same)로 구성
- **speedup**: ORT CPU 평균 ÷ 각 평균 (CPU 기준 몇 배 빠른가)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `bench_fn()` | `bench_fn(lambda: f(x1), n=100, warm=10)` | 함수 하나의 지연 시간 측정 (mean · p50 · p95 · fps) |
| `DataFrame.idxmin()` | `cmp["p95"].idxmin()` | 최솟값 행 이름 (가장 빠른 실행기) |
| `pd.concat()` | `pd.concat([old, cmp], ignore_index=True)` | 성능표 이어 붙이기 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 과제 1 | `RUN` 네 실행기를 같은 `x1` 로 `bench_fn` → 10열 표 `cmp` |
| 과제 2 | 각 출력 vs `ref` → `max_diff` · `top1_same` 열 |
| 과제 3 | SimpleConvNet 로 MNIST 100장 정확도 `mnist_acc` |
| 과제 4 | `speedup` 열 · p95 최소인 `fastest` |
| 과제 5 | Day 3 줄에 `precision="FP32"` 를 채우고 `ot_baseline.csv` 에 이어 붙임 + p95 막대 |
| 과제 6 | 현장 카메라용 추천 · FP16 오차 판단 근거 · Jetson 에는 `.onnx` 를 가져가야 하는 이유 |

**핵심 내용**

#### 같은 모양 함수로 맞춘 비교
> 실행기를 모두 `f(x) → 출력` 으로 맞추면 `for name, f in RUN.items()` 한 반복문으로 속도 · 정확도를 같이 잰다
- **측정 조건 열** → `model · provider · precision · threads(0) · level("ALL") · batch(1)` — Day 3 열 이름과 같아 이어 붙임
- **정확도 열** → `max_diff`(ORT CPU 와 최대 절대 차이) · `top1_same`
- **speedup** → `cpu_mean / cmp["mean"]` — ORT CPU 는 정확히 1.0
- **fastest** → p95 가 가장 작은 실행기 (평균이 아니라 꼬리 지연)
- ⚠ 속도만 쓰면 반쪽 — FP16 은 `max_diff` 가 FP32 보다 크므로 `mnist_acc` 같은 정확도로 판정
- ⚠ 엔진 파일을 Jetson 에 복사하지 말고 `.onnx` 를 가져가 그 장치에서 다시 빌드

**기본 사용형태**
```python
rows = []
for name, f in RUN.items():
    r = bench_fn(lambda: f(x1), n=100, warm=10)
    rows.append({"provider": name, "precision": PREC[name], **r})
cmp = pd.DataFrame(rows)
cmp["speedup"] = cmp.loc[cmp.provider == "ORT CPU", "mean"].iloc[0] / cmp["mean"]
```

**연결**
- [OT D3·6교시 배치 크기 · 성능표](#/ot/3/6) — 이어 붙이는 `ot_baseline.csv` 의 출발점
- [OT D5·6교시 최종 성능표](#/ot/5/6) — 같은 CSV 에 YOLOv8n 줄을 이어 붙임
- [OT D4·6교시 FP16](#/ot/4/6) — 표의 `TRT FP16` 행

**상세 학습**

#### 이 교시에서 답하는 질문
오늘 배운 네 실행기 중 현장 카메라용으로 무엇을 쓸까? 감이 아니라 같은 입력 · 같은 측정 · 속도와 정확도를 한 표에 놓고 숫자로 고른다.

#### 개념 1 · 성능표의 세 부분
조건(무엇을 어떤 설정으로) · 속도(mean · p50 · p95 · fps) · 정확도(max_diff · top1_same · MNIST 정확도). 조건이 없는 숫자는 비교할 수 없고, 정확도가 없는 속도는 쓸 수 없다.

#### 개념 2 · p95 로 고르는 이유
평균은 가끔 느린 요청에 가려진다. p95(상위 5 %를 뺀 최악) 가 현장 카메라처럼 매 프레임 제때 처리해야 하는 일에 더 믿을 만하다.

#### 코드로 확인
```python
cpu_mean = cmp.loc[cmp.provider == "ORT CPU", "mean"].iloc[0]
cmp["speedup"] = cpu_mean / cmp["mean"]
fastest = cmp.loc[cmp["p95"].idxmin(), "provider"]
```
- **무엇을 하나**: CPU 대비 배수 · p95 최소 실행기
- **어떤 개념**: 개념 1 · 2
- **출력**: 교안 기대값 — ORT CPU 의 speedup 은 1.0 (나머지 값은 GPU 에서 실행해야 나옴)

#### 한 번에 정리
- 실행기를 같은 모양 함수로 → 한 반복문으로 비교
- 성능표 = 조건 + 속도(mean · p95 · fps) + 정확도 · 배포는 `.onnx`
- `ot_baseline.csv` → Day 5 에서 YOLOv8n 줄을 이어 붙임
- ⚠ 정확도 없는 속도표는 반쪽
