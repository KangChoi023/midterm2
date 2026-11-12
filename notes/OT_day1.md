## Day 1 — ONNX 첫걸음: 내보내고, 보고, 실행하기

### 1교시 · 왜 ONNX 인가: 학습하는 곳 vs 실행하는 곳

**키워드**: `학습 환경 vs 실행 환경`, `.pt = 가중치만`, `ONNX = 계산 순서 + 가중치`, `추론 = forward 만`, `전처리는 ONNX 밖`

**개요**: 지금까지 모델은 Colab(PyTorch)에서 만들고 Colab 에서 썼다. 현장 카메라 옆 작은 장치에는 Python · PyTorch 가 없을 수 있다. 실행하는 쪽에는 **계산 순서**(무엇을 곱하고 더하고 어떤 함수를 거치나)와 **가중치**(학습으로 얻은 숫자) 두 가지만 있으면 된다. ONNX 는 이 둘을 **파일 하나**에 담는 공용 형식(설계도)이다. 이 과목 5일: Day 1 내보내기 · 실행 → Day 2 그래프 읽기 · 배치 자유롭게 → Day 3 ONNX Runtime 성능 측정 → Day 4 TensorRT → Day 5 YOLOv8n 침입 감지.

**핵심 용어**
- **ONNX(Open Neural Network Exchange)**: 계산 순서(그래프) + 가중치를 한 파일에 담는 모델 공용 형식
- **추론(inference)**: 학습된 모델에 입력을 넣어 결과를 얻는 일 — forward 만 한다
- **state_dict**: `.pt` 에 저장되는 가중치 딕셔너리 (`weight` · `bias` 이름별 텐서)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.save()` | `torch.save(tiny.state_dict(), "tiny.pt")` | 가중치만 파일로 저장 |
| `torch.load()` | `torch.load("tiny.pt")` | `.pt` 를 열어 `state_dict` 확인 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 1-1 | `nn.Linear(3, 2)` 의 `.pt` 를 열어 키 `weight (2, 3)` · `bias (2,)` 확인 → 숫자 8개뿐 |
| 1-2 | 학습 5단계 중 추론에 필요한 단계 번호 `[0, 1, 5]` 고르기 |

**핵심 내용**

#### .pt 와 ONNX 의 차이
> `.pt` = 숫자(가중치)만, ONNX = 숫자 + "무엇을 어떤 순서로 계산하나"
- **`.pt`** → `weight (2, 3)` + `bias (2,)` = 8개 숫자. 쓰려면 `nn.Linear(3, 2)` 를 만드는 파이썬 코드가 반드시 있어야 함
- **ONNX** → 계산 순서(그래프)까지 저장 → 파이썬 · PyTorch 없이 실행기(ONNX Runtime)만 있으면 됨
- ⚠ "ONNX 로 바꾸면 무조건 빨라진다" 아님 — ONNX 는 **형식**, 속도는 **실행기**가 정함 (Day 3 · 4 에서 측정)

#### 추론에 필요한 것 — ONNX 에 담기는 것
> 학습 5단계(forward → loss → backward → step …) 중 ONNX 에는 **forward 만** 담긴다
- **담김** → forward (입력 → 출력 계산)
- **안 담김** → loss · backward · step (정답 `y` 가 필요하고 가중치를 바꾸는 학습 단계)
- **ONNX 밖** → 전처리(`/255` · 정규화 · 크기) · 결과 해석(argmax · 클래스 이름) → 실행 쪽 코드가 해야 함 (7교시)
- ⚠ ONNX 파일로 학습을 이어 할 수 없다 — 학습은 PyTorch 에서

**기본 사용형태**
```python
tiny = nn.Linear(3, 2)
torch.save(tiny.state_dict(), "tiny.pt")
sd = torch.load("tiny.pt")
for k in sd: print(k, tuple(sd[k].shape))   # weight (2, 3) / bias (2,)
```

**연결**
- [VB D5·5교시 ONNX export · ONNX Runtime](#/vb/5/5) — 어제 한 번 해 본 export 를 오늘은 처음부터 한 줄씩
- [VB D1·6교시 디지털 이미지](#/vb/1/6) — 전처리가 ONNX 밖에 있다는 것의 출발점

**상세 학습**

#### 이 교시에서 답하는 질문
VB 에서 만든 모델은 Colab 안에서만 돌았다. 현장 카메라 옆 작은 장치에는 Python · PyTorch 가 없을 수 있다. 그러면 "파일 하나만 들고 가서 돌리는" 방법이 필요하다. 이 교시는 `.pt` 파일을 열어 **무엇이 빠져 있는지** 확인하고, ONNX 가 그 빈칸(계산 순서)을 어떻게 채우는지, 그리고 이번 과목 5일의 길을 정리한다.

#### 개념 1 · 학습하는 곳 vs 실행하는 곳
![학습 환경(PyTorch, 5단계)과 실행 환경(전처리 · forward · 결과 해석)의 차이, PyTorch → ONNX → ONNX Runtime → TensorRT 의 길](assets/fig/ot/d1-path.svg)

| | 학습하는 곳 | 실행하는 곳 |
|---|---|---|
| 장소 | Colab · GPU 서버 | 현장 장치 · 다른 언어 프로그램 |
| 하는 일 | forward → loss → backward → step 반복 | 전처리 → **forward** → 결과 해석 |
| 필요한 것 | 데이터 · 정답 · PyTorch | 계산 순서 + 가중치 + 전처리 규칙 |

이번 과목 5일: Day 1 내보내기 · 실행 → Day 2 ONNX 속 들여다보기 · export 오류 고치기 → Day 3 ONNX Runtime 측정 (bench · 스레드 · GPU) → Day 4 TensorRT 엔진 · FP16 → Day 5 YOLOv8n TensorRT 침입 감지.

#### 개념 2 · `.pt` 에는 숫자만 있다
`nn.Linear(3, 2)` 의 `state_dict` 를 열면 `weight (2, 3)` + `bias (2,)` = **숫자 8개**뿐이다. "입력에 weight 를 곱하고 bias 를 더한다" 는 **계산 순서는 파이썬 클래스 코드에만** 있다. 그래서 `.pt` 를 쓰려면 모델 코드 + PyTorch 를 같이 옮겨야 한다.

| 파일 | 담긴 것 | 실행에 필요한 것 |
|---|---|---|
| `.pt` (state_dict) | 가중치 숫자 | 모델 클래스 코드 + PyTorch |
| `.onnx` | 계산 순서(그래프) + 가중치 | ONNX Runtime · TensorRT 등 실행기만 |

#### 개념 3 · 추론 = forward 만, 전처리는 밖
학습 5단계 중 추론에 필요한 것은 **0 준비 · 1 forward · 5 결과 해석** (`infer_steps = [0, 1, 5]`). loss · backward · step 은 정답이 필요하고 가중치를 바꾸는 학습 단계라 ONNX 에 들어가지 않는다.

자주 하는 오해 세 가지:
- "ONNX 로 바꾸면 무조건 빨라진다" → 아니다. ONNX 는 **형식(설계도)**, 빠르기는 **실행기**가 정한다 (Day 3 · 4 에서 측정).
- "ONNX 로 학습을 이어 할 수 있다" → 이 과목의 ONNX 는 forward 만 담는다.
- "전처리도 모델에 들어 있다" → 보통 아니다. `/255` · 정규화 · 크기 맞추기는 실행 쪽 코드가 한다.

#### 코드로 확인
```python
tiny = nn.Linear(3, 2)
torch.save(tiny.state_dict(), "tiny.pt")
sd = torch.load("tiny.pt")
for k in sd:
    print(k, tuple(sd[k].shape))
```
- **무엇을 하나**: `.pt` 안에 무엇이 들었는지 열어 본다.
- **어떤 개념**: 개념 2 — 숫자만 있다.
- **출력**: `weight (2, 3)` · `bias (2,)` (직접 실행)

#### 한 번에 정리
- `.pt` = 가중치만 (계산 순서는 파이썬 코드에), ONNX = 계산 순서 + 가중치 한 파일.
- 추론에 필요한 건 전처리 → forward → 결과 해석, ONNX 에는 forward 만.
- 실행 장치에 전달할 것 = ONNX 파일 + 전처리 규칙 + 클래스 이름.
- ⚠ ONNX = 형식일 뿐 속도 보장 아님.

---

### 2교시 · 첫 ONNX 만들기: nn.Linear(3, 2) 내보내기

**키워드**: `torch.onnx.export`, `더미 입력`, `tracing(녹화)`, `eval()`, `파일 크기 ≈ 파라미터 × 4 B`

**개요**: 가장 중요한 한 줄 `torch.onnx.export(모델, 더미 입력, 파일 이름)`. 더미 입력으로 모델을 **한 번 실행하며 연산을 녹화**해 그래프로 저장한다. 값은 상관없고 **shape · dtype** 이 중요. 손으로 확인되도록 가중치를 `W = [[1, 2, 3], [0, -1, 1]]`, `b = [0.5, -0.5]` 로 직접 정한다. 순서: ① 모델 만들기 → ② 더미 입력 → ③ export → ④ 파일 확인.

**핵심 용어**
- **더미 입력(dummy input)**: export 때 한 번 흘려 보내는 가짜 입력 — shape · dtype 만 의미 있음
- **tracing**: 더미 입력으로 실행하며 거친 연산을 기록하는 방식
- **EXPORT_KW**: 환경 셀이 만든 export 옵션 딕셔너리 — 매번 `**EXPORT_KW` 로 붙임

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.onnx.export()` | `torch.onnx.export(model, dummy, "x.onnx", **EXPORT_KW)` | PyTorch 모델 → ONNX 파일 |
| `Module.eval()` | `tiny.eval()` | 실행 모드 (Dropout · BN 고정) |
| `torch.randn()` | `torch.randn(1, 3)` | 더미 입력 만들기 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 2-1 | 더미 입력 `torch.randn(1, 3)` → shape `(1, 3)`, dtype `float32` |
| 2-2 | `tiny` → `tiny.onnx` 로 export |
| 2-3 | 파라미터 8개 × 4 B = 32 B, 파일은 249 B |

**핵심 내용**

#### export = 더미 입력으로 한 번 실행하며 녹화
> `torch.onnx.export(모델, 더미 입력, 파일 이름)` — 더미 입력이 지나간 연산을 그래프로 저장
- **더미 입력** → 값은 아무거나, **shape · dtype 이 입력 규격으로 굳음** `(1, 3)` float32
- **4단계** → ① 모델(`eval()`) → ② 더미 입력 → ③ export → ④ 파일 확인
- **eval()** → 습관으로 export 전에 (Dropout · BN 이 있으면 PyTorch 쪽 비교 기준이 달라짐)
- ⚠ 더미 입력을 `torch.randn(3)` 처럼 배치 축 없이 만들면 → 배치 축 없는 입력만 받는 파일이 됨
- ⚠ float64 더미 → ONNX 입력도 float64 로 굳음

#### 파일 크기 = 파라미터 수 × 4 B + 그래프 설명
> float32 숫자 하나 = 4 byte
- **예** → `Linear(3, 2)`: 파라미터 8 × 4 = 32 B, `tiny.onnx` = 249 B (나머지 = 그래프 · 이름 · shape)
- **바꿔보기** → `Linear(3, OUT)`: OUT = 5 → 20개 · 80 B · 파일 297 B / OUT = 100 → 400개 · 1,600 B · 파일 1,821 B

**기본 사용형태**
```python
tiny = nn.Linear(3, 2).eval()
dummy = torch.randn(1, 3)                      # (배치, 입력 3) float32
torch.onnx.export(tiny, dummy, "tiny.onnx", **EXPORT_KW)
print(os.path.getsize("tiny.onnx"), "B")       # 249
```

**연결**
- [VB D5·5교시 ONNX export · ONNX Runtime](#/vb/5/5) — `input_names` · `dynamic_axes` 가 붙은 export
- [NN D3·5교시 Affine · Softmax-with-Loss](#/nn/3/5) — `y = xWᵀ + b` 가 그래프의 Gemm 이 된다

**상세 학습**

#### 이 교시에서 답하는 질문
오늘 가장 중요한 한 줄은 `torch.onnx.export` 다. 결과를 손으로 확인할 수 있도록 가중치를 직접 정한 `nn.Linear(3, 2)` 를 내보내며, export 가 **무엇을 녹화하고 무엇을 고정하는지**, 파일 크기로 내용을 어떻게 짐작하는지 본다.

#### 개념 1 · export 4단계
| 단계 | 코드 | 왜 |
|---|---|---|
| ① 모델 만들기 | `tiny.weight.copy_(…)` (`torch.no_grad()` 안) · `tiny.eval()` | 손 계산과 비교할 수 있게 |
| ② 더미 입력 | `dummy = torch.randn(1, 3)` | 입력 **shape · dtype** 을 알려 줌 (값은 상관없음) |
| ③ 내보내기 | `torch.onnx.export(tiny, dummy, "tiny.onnx")` | 한 번 실행하며 연산을 녹화 |
| ④ 파일 확인 | `os.path.getsize("tiny.onnx")` | 크기로 내용 짐작 |

#### 개념 2 · 숫자를 직접 정한 모델
![nn.Linear(3, 2) 가 ONNX 의 Gemm 노드 하나가 되는 모습과 손 계산, 파일 크기 구성](assets/fig/ot/d1-tiny.svg)

| 기호 | 값 |
|---|---|
| W | [[1, 2, 3], [0, −1, 1]] (2×3) |
| b | [0.5, −0.5] |
| y = x·Wᵀ + b | x = [1, 1, 1] → [1+2+3+0.5, 0−1+1−0.5] = **[6.5, −0.5]** |

가중치를 직접 넣을 때는 `torch.no_grad()` 로 기록을 끈다. `eval()` 은 Dropout · BN 이 있는 모델에서 실행 모드로 바꾸는 습관 — tiny 에는 영향이 없지만 4교시 비교 기준이 학습 모드로 계산되는 실수를 막는다.

#### 개념 3 · export = 녹화, 파일 크기 = 가중치 + 설명
- export 는 더미 입력으로 forward 를 **한 번 실행**하며 지나간 연산을 기록한다 → 더미 입력의 shape `(1, 3)` 이 파일에 **고정**된다.
- 더미를 `torch.randn(3)` 처럼 배치 축 없이 만들면, 실행할 때도 배치 축 없는 입력만 받는 파일이 된다.
- 파일 크기 = 파라미터 8개 × 4 B (float32) = **32 B** + 그래프 · 이름 · 버전 정보 **217 B** = **249 B**.

| 바꿔보기 `nn.Linear(3, OUT)` | 파라미터 | 가중치 바이트 |
|---|---|---|
| OUT = 2 | 3×2 + 2 = 8 | 32 B |
| OUT = 5 | 3×5 + 5 = 20 | 80 B |
| OUT = 100 | 3×100 + 100 = 400 | 1,600 B |

#### 코드로 확인
```python
tiny = nn.Linear(3, 2)
with torch.no_grad():
    tiny.weight.copy_(torch.tensor([[1., 2., 3.], [0., -1., 1.]]))
    tiny.bias.copy_(torch.tensor([0.5, -0.5]))
tiny.eval()
dummy = torch.randn(1, 3)
torch.onnx.export(tiny, dummy, "tiny.onnx", **EXPORT_KW)
```
- **무엇을 하나**: 손 계산 가능한 모델을 만들어 ONNX 로 내보낸다 (`EXPORT_KW` = 강의 공통 설정 `opset_version=17, dynamo=False`).
- **어떤 개념**: 개념 1 · 2.
- **출력**: `tiny([1, 1, 1]) = [[6.5 -0.5]]`, 더미 `torch.Size([1, 3]) torch.float32`, 노드 `['Gemm']` (직접 실행)

```python
n_params = sum(p.numel() for p in tiny.parameters())
print(f"파라미터 {n_params} 개 × 4 B = {n_params * 4} B  |  tiny.onnx {os.path.getsize('tiny.onnx')} B")
```
- **무엇을 하나**: 파라미터 바이트와 실제 파일 크기 비교.
- **어떤 개념**: 개념 3.
- **출력**: `파라미터 8 개 × 4 B = 32 B  |  tiny.onnx 249 B` — 나머지 217 B 는 그래프 · 이름 · 버전 정보 (직접 실행)

#### 한 번에 정리
- `torch.onnx.export(model, dummy, "x.onnx")` — 더미 입력으로 한 번 실행하며 녹화.
- 더미 입력은 값이 아니라 shape · dtype 이 중요 — `(1, 3)` float32 가 파일에 고정된다.
- 파일 크기 ≈ 파라미터 수 × 4 B + 그래프 설명.
- ⚠ 배치 축 없는 더미(`randn(3)`), `eval()` 누락, 파일 이름 오타가 흔한 실수.

---

### 3교시 · Netron 으로 보기: 노드 · 화살표 · 가중치

**키워드**: `Netron`, `노드 = 연산`, `Gemm`, `initializer`, `transB`, `onnx.load`

**개요**: Netron 은 ONNX 를 그림으로 보여 주는 뷰어(상자 = 연산 노드, 화살표 = 데이터 흐름, 상자 옆 `B` · `C` = 가중치). `tiny.onnx` 는 **Gemm 상자 1개**: `B` = weight ⟨2×3⟩, `C` = bias ⟨2⟩, 입출력 `float32[1,3] → float32[1,2]`. 같은 내용을 `onnx.load(path).graph.node / .initializer` 로 코드에서도 꺼낼 수 있다. 층을 쌓으면(`Linear → ReLU → Linear`) `Gemm → Relu → Gemm`.

**핵심 용어**
- **노드(node)**: 그래프의 연산 하나 — `op_type` 이 이름 (`Gemm` · `Relu` · `Conv` …)
- **initializer**: 노드에 붙는 가중치 상수 (weight · bias)
- **Gemm**: General Matrix Multiply — 행렬곱 + 덧셈. `nn.Linear` 가 이것으로 변환됨
- **Netron**: ONNX 그래프 뷰어 (netron.app)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `onnx.load()` | `m = onnx.load("tiny.onnx")` | ONNX 파일 읽기 |
| `onnx.helper.printable_graph()` | `onnx.helper.printable_graph(m.graph)` | 그래프를 글로 보기 |
| `numpy_helper.to_array()` | `onnx.numpy_helper.to_array(m.graph.initializer[0])` | 가중치를 NumPy 배열로 |
| `show_netron()` | `show_netron("tiny.onnx", port=8081)` | Colab 안에 Netron 창 열기 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 3-1 | 노드 종류 목록 `['Gemm']` |
| 3-2 | PyTorch `weight (2, 3)` 와 ONNX `B` 가 같은 숫자 · 같은 모양 |
| 3-3 | 2층 MLP 예측 `['Gemm', 'Relu', 'Gemm']` |

**핵심 내용**

#### Netron 읽는 법 — 상자 · 화살표 · B · C
> 상자 = 노드(연산), 화살표 = 데이터 흐름, 상자 옆 B · C = 가중치(initializer)
- **`nn.Linear`** → `Gemm` 1개: `B` = weight ⟨2×3⟩, `C` = bias ⟨2⟩
- **`nn.ReLU`** → `Relu` — 가중치 없음, initializer 안 붙음
- **입력 이름** → 자동 이름 `onnx::Gemm_0` — 5교시에서 `input` 으로 바꿈
- ⚠ 같은 `port` 로 Netron 을 두 번 열면 이전 창이 바뀜 → 파일마다 `8081`, `8082` …
- ⚠ 창이 빈 화면이면 서드파티 쿠키 차단 → netron.app 에 파일을 끌어다 놓기

#### Gemm 의 transB — weight 는 (출력, 입력) 으로 저장
> PyTorch 는 weight 를 (out, in) = (2, 3) 으로 저장하고 `x @ Wᵀ` 로 계산 → Gemm 에 `transB = 1`
- **예** → `Linear(3, 4) → ReLU → Linear(4, 2)`: 노드 `['Gemm', 'Relu', 'Gemm']`, initializer 4개 (weight · bias × 2)
- ⚠ `initializer[0]` 이 항상 weight 라고 가정하지 말 것 — 큰 모델은 순서가 다름, `.name` 으로 찾기 (Day 2)

**기본 사용형태**
```python
m = onnx.load("tiny.onnx")
print([n.op_type for n in m.graph.node])                    # ['Gemm']
W = onnx.numpy_helper.to_array(m.graph.initializer[0])      # (2, 3)
print(onnx.helper.printable_graph(m.graph))                 # 그래프를 글로
```

**연결**
- [NN D3·5교시 Affine · Softmax-with-Loss](#/nn/3/5) — `Gemm` = 행렬곱 + 편향, 오늘 그래프의 한 상자
- [NN D3·4교시 활성화 계층: ReLU · Sigmoid](#/nn/3/4) — `Relu` 상자
- [OT D1·6교시 MNIST CNN 내보내기](#/ot/1/6) — 같은 읽기를 큰 모델에

**상세 학습**

#### 이 교시에서 답하는 질문
내보낸 파일 안에 정말 계산 순서와 가중치가 들어 있나? **Netron** 으로 그림으로 보고, 같은 내용을 `onnx.load` 로 코드로도 꺼낸다. 층을 하나 더 쌓으면 그림이 어떻게 바뀌는지 **미리 예측**하고 확인한다.

#### 개념 1 · Netron 읽는 법
| 그림에서 | 뜻 | tiny.onnx 에서 |
|---|---|---|
| 상자 | 노드(연산) | `Gemm` 1개 (General Matrix Multiply = 행렬곱 + 덧셈) |
| 화살표 | 데이터 흐름 | input → Gemm → output |
| 상자 옆 B · C | 가중치 (initializer) | B = weight ⟨2×3⟩, C = bias ⟨2⟩ |
| 입출력 표시 | dtype · shape | `float32[1,3]` → `float32[1,2]` |

B 를 클릭하면 1, 2, 3, 0, −1, 1 이 그대로 보인다. Colab 에서 Netron 창이 안 뜨면 파일을 내려받아 netron.app 에 끌어다 놓으면 된다. 파일마다 다른 port (8081, 8082, …) 를 쓴다.

#### 개념 2 · Gemm 의 transB — weight 는 (출력, 입력)
PyTorch 의 `nn.Linear` 는 weight 를 **(출력 2, 입력 3)** 모양으로 저장한다. 계산은 x·Wᵀ 이므로 Gemm 노드는 속성 **transB = 1** (B 를 전치해서 곱함) 을 갖는다. PyTorch weight `(2, 3)` = ONNX B `(2, 3)` — 같은 숫자, 같은 모양.

#### 개념 3 · 층을 쌓으면 상자가 늘어난다
| 모델 | 예측한 노드 | 가중치 있는 노드 |
|---|---|---|
| `Linear(3, 2)` | Gemm | 1 |
| `Linear(3, 4) → ReLU → Linear(4, 2)` | **Gemm → Relu → Gemm** | 2 (Relu 는 가중치 없는 상자) |

#### 코드로 확인
```python
m = onnx.load("tiny.onnx")
print([n.op_type for n in m.graph.node])
W_onnx = onnx.numpy_helper.to_array(m.graph.initializer[0])
print("PyTorch weight", tuple(tiny.weight.shape), "| ONNX B", W_onnx.shape)
print(W_onnx)
```
- **무엇을 하나**: 노드 종류와 첫 initializer(weight) 를 NumPy 로 꺼낸다.
- **어떤 개념**: 개념 1 · 2.
- **출력**: `['Gemm']`, `PyTorch weight (2, 3) | ONNX B (2, 3)`, `[[1. 2. 3.] [0. -1. 1.]]` (직접 실행)

```python
mlp = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
torch.onnx.export(mlp, torch.randn(1, 3), "mlp.onnx", **EXPORT_KW)
predict = ["Gemm", "Relu", "Gemm"]
print(predict == [n.op_type for n in onnx.load("mlp.onnx").graph.node])
```
- **무엇을 하나**: 2층 MLP 의 노드 순서를 예측하고 실제와 비교.
- **어떤 개념**: 개념 3.
- **출력**: `예측 성공: ['Gemm', 'Relu', 'Gemm']` (직접 실행)

#### 한 번에 정리
- Netron: 상자 = 노드(연산), 화살표 = 데이터 흐름, B · C = 가중치(initializer).
- `nn.Linear` → `Gemm` (transB = 1), `nn.ReLU` → `Relu`.
- 코드로는 `onnx.load(path).graph.node` · `.initializer`.
- ⚠ `initializer[0]` 이 항상 weight 라고 가정하지 않는다 — 큰 모델은 이름(`.name`)으로 찾는다 (Day 2).

---

### 4교시 · ONNX Runtime 으로 실행: PyTorch 와 같은가?

**키워드**: `ONNX Runtime`, `InferenceSession`, `get_inputs`, `run(None, {이름: 배열})`, `np.allclose`

**개요**: ONNX Runtime(ORT)은 ONNX 설계도를 읽어 바로 실행하는 기계 — PyTorch 없이 `onnxruntime` 만 설치하면 된다. 실행 4단계: ① `InferenceSession(path, providers)` 로 열기 → ② `get_inputs()[0].name` 으로 입력 이름 → ③ `run(None, {이름: float32 NumPy 배열})` → ④ `np.allclose` 로 PyTorch 와 비교. **입력 = 딕셔너리, 출력 = 리스트.**

**핵심 용어**
- **ONNX Runtime(ORT)**: ONNX 모델을 실행하는 추론 엔진
- **provider**: 실행 장치 (`CPUExecutionProvider` · `CUDAExecutionProvider` · `TensorrtExecutionProvider`) — 리스트 순서 = 우선순위
- **allclose**: 오차 허용 범위 안에서 "같다" 판정

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `ort.InferenceSession()` | `ort.InferenceSession("tiny.onnx", providers=["CPUExecutionProvider"])` | ONNX Runtime 세션 열기 |
| `sess.get_providers()` | `sess.get_providers()` | 쓰이는 실행 장치 목록 |
| `sess.get_inputs()` | `sess.get_inputs()[0].name` | 입력 이름 · shape · type |
| `sess.run()` | `sess.run(None, {name: x})` | 실행 → 출력 리스트 |
| `np.allclose()` | `np.allclose(out, ref, atol=1e-5)` | 오차 허용 같음 비교 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 4-1 | 세션 열고 `['CPUExecutionProvider']` 확인 |
| 4-2 | 입력 이름 `onnx::Gemm_0`, shape `[1, 3]`, type `tensor(float)` |
| 4-3 | `x = [1, 1, 1]` → `[[6.5, -0.5]]` (손계산과 일치) |
| 4-4 | 난수 입력 20개로 PyTorch 와 비교 → 모두 같다 |

**핵심 내용**

#### ORT 실행 4단계
> `InferenceSession` 열기 → 입력 이름 → `run(None, {이름: float32 배열})[0]` → `allclose` 로 비교
- **입력** → `{이름: NumPy float32 배열}` 딕셔너리 (텐서 X)
- **출력** → 리스트 — 출력이 하나여도 `[0]` 으로 꺼냄. `run` 첫 인자 `None` = 출력 전부
- **예** → x = [1, 1, 1] → `[[6.5, -0.5]]` = 손계산 1+2+3+0.5, 0−1+1−0.5
- ⚠ `torch` 텐서를 넣으면 오류 → `.numpy()` (GPU 면 `.cpu().numpy()`)
- ⚠ `np.ones(...)` · `np.array([...])` 는 float64 → `INVALID_ARGUMENT : Unexpected input data type` → `.astype(np.float32)`
- ⚠ 입력 이름이 틀리면 `Required inputs (['onnx::Gemm_0']) are missing`

#### == 대신 allclose
> float32 는 계산 순서가 다르면 끝자리가 조금 달라진다 — `np.allclose(a, b, atol=1e-5)` 로 판정
- **차이 크기** → 1e-7 정도는 정상 (`safety` 최대 차이 8.2e-08)
- ⚠ export 후에는 **항상** PyTorch 결과와 비교

**기본 사용형태**
```python
sess = ort.InferenceSession("tiny.onnx", providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out = sess.run(None, {in_name: x1.numpy()})[0]      # (1, 2)
print(np.allclose(out, tiny(x1).detach().numpy(), atol=1e-5))
```

**연결**
- [VB D5·5교시 ONNX export · ONNX Runtime](#/vb/5/5) — 최대 차이 < 1e-4 로 비교하던 방식
- [OT D1·5교시 이름 붙이기 · 입력 shape](#/ot/1/5) — 이상한 입력 이름을 `input` 으로

**상세 학습**

#### 이 교시에서 답하는 질문
ONNX 파일을 만들었으니 이제 **PyTorch 없이** 실행해 본다. ONNX Runtime(ORT) 은 설계도를 읽어 바로 실행하는 기계다 — 실행 장치에는 `onnxruntime` 만 설치하면 된다. 그리고 결과가 PyTorch 와 "같다" 는 것을 어떻게 판정하는지 배운다.

#### 개념 1 · ORT 실행 4단계
![ONNX Runtime 실행 4단계와 흔한 에러 메시지 · 고치는 법](assets/fig/ot/d1-ort.svg)

| 단계 | 코드 | 돌려받는 것 |
|---|---|---|
| ① 열기 | `sess = ort.InferenceSession("tiny.onnx", providers=["CPUExecutionProvider"])` | 세션 |
| ② 입력 이름 | `inp = sess.get_inputs()[0]` → `inp.name` · `inp.shape` · `inp.type` | `onnx::Gemm_0` · `[1, 3]` · `tensor(float)` |
| ③ 실행 | `outs = sess.run(None, {in_name: x1.numpy()})` | **리스트** (출력 1개면 길이 1) |
| ④ 결과 | `y_ort = outs[0]` | `[[6.5, -0.5]]` — 손 계산과 같다 |

`None` = "출력 전부 달라". 입력은 **{이름: NumPy 배열}** 딕셔너리, 출력은 **리스트**.

#### 개념 2 · == 대신 allclose
두 프로그램은 같은 식을 다른 순서 · 다른 라이브러리로 계산하므로 float32 의 **끝자리**가 다를 수 있다. 그래서 `==` 대신 `np.allclose(out, ref, atol=1e-5)` 로 판정한다.

| 비교 | 결과 (직접 실행) |
|---|---|
| 난수 입력 20개, PyTorch vs ORT | 최대 차이 **4.77e-07**, 모두 allclose ✔ |
| 판정 기준 | 차이 ≈ 1e-7 은 정상 (float32 유효숫자 약 7자리) |

#### 개념 3 · 에러 메시지로 원인 찾기
| 실수 | 메시지 | 고치기 |
|---|---|---|
| `np.ones((1, 3))` — NumPy 기본은 float64 | `INVALID_ARGUMENT : Unexpected input data type` | `.astype(np.float32)` |
| 입력 이름 틀림 `{"x": …}` | `INVALID_ARGUMENT : Invalid input name` | `get_inputs()[0].name` |
| torch 텐서를 그대로 | NumPy 만 받음 | `.numpy()` (GPU 면 `.cpu().numpy()`) |
| 결과를 `outs` 그대로 씀 | 리스트라 연산 오류 | `outs[0]` |

#### 코드로 확인
```python
sess = ort.InferenceSession("tiny.onnx", providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
outs = sess.run(None, {in_name: x1.numpy()})
print(type(outs), len(outs), outs[0])
```
- **무엇을 하나**: ORT 로 tiny.onnx 를 실행한다.
- **어떤 개념**: 개념 1.
- **출력**: `<class 'list'> 1 [[ 6.5 -0.5]]` — 입력 이름은 자동으로 붙은 `onnx::Gemm_0` (직접 실행)

```python
max_diff, same_all = 0.0, True
for _ in range(20):
    xr = torch.randn(1, 3)
    with torch.no_grad():
        ref = tiny(xr).numpy()
    out = sess.run(None, {in_name: xr.numpy()})[0]
    max_diff = max(max_diff, float(np.abs(out - ref).max()))
    same_all = same_all and np.allclose(out, ref, atol=1e-5)
```
- **무엇을 하나**: 난수 입력 20개로 PyTorch 와 ORT 를 비교.
- **어떤 개념**: 개념 2.
- **출력**: `최대 차이 4.77e-07 | 모두 같다: True` (직접 실행)

#### 한 번에 정리
- ORT 4단계: `InferenceSession` → `get_inputs()[0].name` → `run(None, {name: float32 배열})[0]` → `allclose`.
- 입력 = 딕셔너리, 출력 = 리스트.
- export 후에는 항상 PyTorch 와 비교한다 — 차이 ≈ 1e-7 은 정상.
- ⚠ float64 · torch 텐서 · 이름 오타가 가장 흔한 세 가지 에러.

---

### 5교시 · 이름 붙이기 · 입력 shape (N, C, H, W)

**키워드**: `input_names`, `output_names`, `(N, C, H, W)`, `shape 고정`, `Invalid rank`

**개요**: 자동 이름(`onnx::Gemm_0`) 대신 `input_names=["input"]`, `output_names=["output"]` 을 붙이면 실행 코드가 단순해진다(리스트로 넘길 것). 이미지 입력은 4개 축 **(N, C, H, W)** = 장 수 · 채널 · 높이 · 너비, 1장이어도 `N = 1`. 입력 shape 는 export 때의 더미 입력으로 **고정**된다 — 배치 4개를 넣으면 `Got: 4 Expected: 1`. 자유롭게 만드는 `dynamic_axes` 는 Day 2.

**핵심 용어**
- **NCHW**: PyTorch · ONNX 이미지 텐서 순서 (OpenCV 는 `H, W, C`)
- **고정 shape**: export 때의 더미 입력 shape 가 그대로 입력 규격이 됨

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.onnx.export()` | `torch.onnx.export(…, input_names=["input"], output_names=["output"])` | 입출력에 이름 붙이기 (리스트로) |
| `sess.get_outputs()` | `[o.name for o in sess.get_outputs()]` | 출력 이름 · shape |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 5-1 | `tiny_named.onnx` → 입력 `['input']`, 출력 `['output']` |
| 5-2 | 흑백 `(1, 1, 28, 28)` · 컬러 `(1, 3, 224, 224)` · 8장 `(8, 3, 224, 224)` |
| 5-3 | 배치 4 입력 → 오류 `index: 0 Got: 4 Expected: 1` |

**핵심 내용**

#### 입출력 이름 붙이기
> `input_names=["input"]`, `output_names=["output"]` — 이름을 붙이면 `run(None, {"input": x})` 로 단순해짐
- **인자** → 문자열이 아니라 **리스트** (`"input"` ✗, `["input"]` ✓)
- **예** → 이름 전 `onnx::Gemm_0`, 이름 후 `input` / `output`
- ⚠ `input_names="input"` (문자열) → 글자 단위로 쪼개져 오류

#### 이미지 텐서 (N, C, H, W)
> N 장 수 · C 채널 · H 높이 · W 너비 — 1장이어도 N = 1
- **예** → 흑백 1장 `(1, 1, 28, 28)` · 컬러 1장 `(1, 3, 224, 224)` · 컬러 8장 `(8, 3, 224, 224)`
- **shape 고정** → 더미가 `(1, 3)` 이면 배치 1개만 받음 (`Got: 4 Expected: 1`)
- ⚠ 사진 1장을 `(3, 224, 224)` 로 넣으면 배치 축이 없어 `Invalid rank` → `x[None]`
- ⚠ OpenCV `(H, W, 3)` 는 순서가 다름 → 7교시 `transpose(2, 0, 1)`

**기본 사용형태**
```python
torch.onnx.export(tiny, dummy, "tiny_named.onnx",
                  input_names=["input"], output_names=["output"], **EXPORT_KW)
s = ort.InferenceSession("tiny_named.onnx", providers=["CPUExecutionProvider"])
s.run(None, {"input": x1.numpy()})
```

**연결**
- [VB D1·6교시 디지털 이미지](#/vb/1/6) — (H, W, C) 배열 vs (C, H, W) 텐서
- [VB D5·5교시 ONNX export · ONNX Runtime](#/vb/5/5) — `dynamic_axes` 로 배치 자유롭게 (Day 2 에서 직접)
- [OT D1·7교시 이미지 한 장 추론](#/ot/1/7) — NCHW 로 바꾸는 전처리

**상세 학습**

#### 이 교시에서 답하는 질문
4교시의 입력 이름 `onnx::Gemm_0` 은 export 가 자동으로 붙인 이름이라 매번 찾아봐야 했다. 이름을 직접 붙이면 실행 코드가 단순해진다. 또 지금까지 입력은 숫자 3개였지만 이미지는 **4개 축 (N, C, H, W)** 을 가진다. 그리고 그 shape 가 export 때 **고정**된다는 것을 에러로 확인한다.

#### 개념 1 · 입출력 이름 = 실행 코드와의 약속
- `input_names=["input"], output_names=["output"]` → `sess.run(None, {"input": x})` 로 부른다.
- 이름은 **리스트**로 준다 (`input_names="input"` 은 실수).
- 출력 이름은 관례적으로 분류 모델이면 `logits`.

#### 개념 2 · 이미지 텐서 (N, C, H, W)
| 축 | 뜻 | 예 |
|---|---|---|
| N | 장 수 (배치) — 1장이어도 1 | 1 · 8 |
| C | 채널 — 흑백 1, 컬러 3 | 1 · 3 |
| H · W | 높이 · 너비 | 28×28 · 224×224 |

| 입력 | shape |
|---|---|
| MNIST 1장 | `(1, 1, 28, 28)` |
| 컬러 224 사진 1장 | `(1, 3, 224, 224)` |
| 컬러 224 사진 8장 | `(8, 3, 224, 224)` |

OpenCV 이미지는 `(H, W, C)` 라 순서가 다르다 → 7교시 `transpose(2, 0, 1)`. 1장을 `(3, 224, 224)` 로 넣으면 배치 축이 없어 `Invalid rank` → `x[None]` 또는 `np.expand_dims(x, 0)`.

#### 개념 3 · shape 는 더미 입력에서 굳는다
2교시 더미가 `(1, 3)` 이었으므로 `tiny_named.onnx` 는 **배치 1개만** 받는다. 4개를 한 번에 넣으면:

`INVALID_ARGUMENT : Got invalid dimensions for input: input … index: 0 Got: 4 Expected: 1`

배치를 자유롭게 하는 법(`dynamic_axes`)은 Day 2 5교시.

#### 코드로 확인
```python
torch.onnx.export(tiny, dummy, "tiny_named.onnx",
                  input_names=["input"], output_names=["output"], **EXPORT_KW)
sess_n = ort.InferenceSession("tiny_named.onnx", providers=["CPUExecutionProvider"])
print([i.name for i in sess_n.get_inputs()], [o.name for o in sess_n.get_outputs()])
print(sess_n.run(None, {"input": x1.numpy()})[0])
```
- **무엇을 하나**: 이름을 붙여 다시 내보내고 이름으로 실행.
- **어떤 개념**: 개념 1.
- **출력**: `['input'] ['output']`, `[[ 6.5 -0.5]]` (직접 실행)

```python
x4 = np.random.rand(4, 3).astype(np.float32)
try:
    sess_n.run(None, {"input": x4})
except Exception as e:
    print(str(e)[:220])
```
- **무엇을 하나**: 배치 1 로 굳은 파일에 배치 4 를 넣어 본다.
- **어떤 개념**: 개념 3.
- **출력**: `Got invalid dimensions for input: input … index: 0 Got: 4 Expected: 1` (직접 실행)

#### 한 번에 정리
- `input_names=["input"], output_names=["logits"]` — 이름을 붙이면 실행 코드가 단순해진다.
- 이미지 텐서 = (N, C, H, W), 1장이어도 N = 1.
- 입력 shape 는 export 때의 더미 입력으로 고정된다 (자유롭게 = Day 2).
- ⚠ 이름은 리스트로, OpenCV (H, W, C) 는 그대로 넣지 않는다.

---

### 6교시 · MNIST CNN 내보내기: 같은 4단계를 큰 모델에

**키워드**: `SimpleConvNet`, `433,890 파라미터`, `Conv · Relu · MaxPool · Flatten · Gemm`, `/255 누락`, `조용히 틀림`

**개요**: VB Day 1 의 SimpleConvNet(433,890 파라미터, 학습 끝난 `mnist_cnn.pt`, MNIST test 0.984)을 `tiny` 와 똑같은 4단계로 내보낸다. 더미는 `torch.zeros(1, 1, 28, 28)`, 이름 `input` / `logits`. 그래프는 `Conv, Relu, MaxPool, Flatten, Gemm, Relu, Gemm` — 가중치 있는 노드 3개, 파일 1.74 MB. 100장 예측이 PyTorch 와 100% 일치. **전처리(`/255`)를 빼먹으면 오류 없이 조용히 정확도만 떨어진다.**

**핵심 용어**
- **일치율(agreement)**: ORT 예측 == PyTorch 예측 인 비율
- **조용한 오류**: 에러 없이 결과만 틀려지는 버그 — 전처리 불일치가 대표

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Tensor.argmax()` | `int(out.argmax())` | 가장 큰 점수의 클래스 번호 |
| `np.load()` | `np.load("mnist_sample.npz")` | 샘플 이미지 · 라벨 읽기 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 6-1 | `mnist_cnn.onnx` export, 파일 ≈ 1.74 MB |
| 6-2 | 노드 `['Conv', 'Relu', 'MaxPool', 'Flatten', 'Gemm', 'Relu', 'Gemm']`, 가중치 노드 3 |
| 6-3 | 100장 ORT vs PyTorch 일치율 1.0, 정확도 ≥ 0.95 |

**핵심 내용**

#### 큰 모델도 4단계는 같다
> eval → export(이름 · 1장 shape) → Netron/코드 확인 → ORT 실행 · 일치율
- **그래프** → `Conv · Relu · MaxPool · Flatten · Gemm · Relu · Gemm` (7노드), 가중치 노드 3
- **initializer** → 6개: `(30,1,5,5)` `(30,)` `(100,4320)` `(100,)` `(10,100)` `(10,)`
- **파일 크기** → 433,890 × 4 B ≈ 1.74 MB
- **더미 입력** → `zeros` 도 됨 (값이 아니라 shape 를 녹화)
- ⚠ `F.relu` · `x.flatten(1)` 도 그래프에 노드(`Relu` · `Flatten`)로 남는다

#### 전처리가 다르면 조용히 틀린다
> 학습 때 입력은 0~1 — 실행 쪽에서 `/255` 를 빼면 에러 없이 정확도만 하락
- **입력 준비** → `(img.astype(np.float32) / 255)[None, None]` → `(1, 1, 28, 28)`
- ⚠ `uint8` 그대로 넣으면 `Unexpected input data type`
- ⚠ 가장 찾기 어려운 버그 — 컬러 사진(7교시)에서는 더 크게 틀림

**기본 사용형태**
```python
torch.onnx.export(cnn, torch.zeros(1, 1, 28, 28), "mnist_cnn.onnx",
                  input_names=["input"], output_names=["logits"], **EXPORT_KW)
x = (img.astype(np.float32) / 255)[None, None]            # (1, 1, 28, 28)
pred = int(sess.run(None, {"input": x})[0].argmax())
```

**연결**
- [NN D5·5교시 SimpleConvNet 으로 MNIST 학습](#/nn/5/5) — 오늘 내보내는 모델의 원래 학습 (VB D1·8교시 미니 프로젝트와 같은 구조)
- [NN D5·4교시 Convolution · Pooling 계층](#/nn/5/4) — `Conv` 상자 안의 계산
- [OT D1·7교시 이미지 한 장 추론](#/ot/1/7) — 전처리를 더 엄격하게

**상세 학습**

#### 이 교시에서 답하는 질문
tiny 로 연습한 4단계를 **진짜 모델**에 적용한다. VB Day 1 의 SimpleConvNet (433,890 파라미터, MNIST test 0.984) 을 내보내 그래프를 읽고, 100장으로 PyTorch 와 같은 답을 내는지 확인한다. 그리고 **전처리가 학습 때와 다르면 오류 없이 조용히 틀린다**는 가장 위험한 함정을 직접 본다.

#### 개념 1 · 큰 모델도 4단계는 같다
![SimpleConvNet 의 ONNX 그래프 — 노드 7개와 각 단계 출력 shape, 파라미터 계산, /255 누락의 영향](assets/fig/ot/d1-cnn.svg)

| 단계 | SimpleConvNet 에서 |
|---|---|
| ① export | 더미 `torch.zeros(1, 1, 28, 28)`, 이름 `input` / `logits` |
| ② 열어보기 | 노드 7개: Conv · Relu · MaxPool · Flatten · Gemm · Relu · Gemm |
| ③ ORT 실행 | 입력 `[1, 1, 28, 28]`, 출력 `logits [1, 10]` |
| ④ 비교 | 100장 일치율 **1.000**, 정확도 **1.000** |

#### 개념 2 · 가중치 노드 세기
| 노드 | initializer | 파라미터 |
|---|---|---|
| Conv | `conv.weight (30, 1, 5, 5)` · `conv.bias (30,)` | 750 + 30 = 780 |
| Gemm (fc1) | `fc1.weight (100, 4320)` · `fc1.bias (100,)` | 432,000 + 100 = 432,100 |
| Gemm (fc2) | `fc2.weight (10, 100)` · `fc2.bias (10,)` | 1,000 + 10 = 1,010 |
| 합계 | 가중치 있는 노드 **3개** | **433,890** → × 4 B = **1.74 MB** = 파일 크기 |

4320 = 30 채널 × 12 × 12 (Conv 5×5 로 28 → 24, MaxPool 로 24 → 12). Relu · MaxPool · Flatten 은 가중치가 없다.

#### 개념 3 · 전처리가 다르면 조용히 틀린다
학습 때 입력은 `0 ~ 1` (/255) 이었다. 실행 쪽에서 `/255` 를 빼먹으면 **오류 메시지는 하나도 없이** 정확도만 떨어진다 (실행 예: 1.00 → 0.92). 흑백 숫자는 쉬워서 이 정도지만, 7교시 컬러 사진에서는 훨씬 크게 틀린다.

| 실수 | 증상 |
|---|---|
| uint8 그대로 넣기 | `Unexpected input data type` — 에러라서 바로 앎 |
| `/255` · 정규화 누락 | 에러 없이 틀린 답 — **가장 찾기 어려운 버그** |
| `cnn.eval()` 없이 비교 | 이 모델은 BN · Dropout 이 없어 같지만, BN 모델(7 · 8교시)은 PyTorch 쪽이 달라짐 |

#### 코드로 확인
```python
g = onnx.load("mnist_cnn.onnx").graph
ops = [n.op_type for n in g.node]
n_weighted = sum(op in ("Conv", "Gemm") for op in ops)
print(ops, "| 가중치 노드:", n_weighted)
print([(t.name, tuple(t.dims)) for t in g.initializer])
```
- **무엇을 하나**: 노드 종류 · 가중치 노드 수 · initializer 이름과 모양.
- **어떤 개념**: 개념 2.
- **출력**: `['Conv', 'Relu', 'MaxPool', 'Flatten', 'Gemm', 'Relu', 'Gemm'] | 가중치 노드: 3`, `('conv.weight', (30, 1, 5, 5)) … ('fc2.bias', (10,))` (직접 실행)

```python
sess_cnn = ort.InferenceSession("mnist_cnn.onnx", providers=["CPUExecutionProvider"])
pred_ort, pred_torch = [], []
for img in digits:
    x = (img.astype(np.float32) / 255)[None, None]          # (1, 1, 28, 28)
    pred_ort.append(int(sess_cnn.run(None, {"input": x})[0].argmax()))
    with torch.no_grad():
        pred_torch.append(int(cnn(torch.from_numpy(x)).argmax()))
agree = (np.array(pred_ort) == np.array(pred_torch)).mean()
```
- **무엇을 하나**: 100장을 ORT 와 PyTorch 로 분류해 일치율과 정확도.
- **어떤 개념**: 개념 1 · 3.
- **출력**: `ORT–PyTorch 일치율 1.000 | 정확도 1.000` (직접 실행)

#### 한 번에 정리
- 큰 모델도 export(이름 · 1장 shape) → Netron → ORT run → 일치율.
- 가중치 노드 3개 (Conv · Gemm · Gemm), 파라미터 433,890 × 4 B = 1.74 MB.
- 실행 쪽 전처리는 학습 때와 똑같이 — `/255` 누락은 오류 없이 1.00 → 0.92.
- ⚠ `[None, None]` 으로 N · C 축 추가, `.astype(np.float32)` 먼저.

---

### 7교시 · 이미지 한 장 추론: 전처리 6단계 · MobileNetV3

**키워드**: `전처리 6단계`, `BGR → RGB`, `cv2.resize(W, H)`, `ImageNet MEAN · STD`, `transpose(2, 0, 1)`, `softmax · top-5`

**개요**: VB Day 5 의 `mnv3s.onnx`(MobileNetV3-Small, ImageNet 1000 클래스, 입력 `input` 224×224)로 실제 사진을 분류한다. 사진 한 장이 입력 텐서가 되기까지 6단계: ① `imread`(H, W, C · BGR) → ② BGR→RGB · `resize(224, 224)` → ③ `/255` 후 `(x − MEAN) / STD` → ④ `transpose(2, 0, 1)` → ⑤ `[None]` 배치 축 → ⑥ `run` · softmax · top-5. `preprocess` · `classify` 두 함수로 묶어 8교시에서 그대로 쓴다.

**핵심 용어**
- **전처리(preprocessing)**: 사진 → 모델 입력 텐서로 바꾸는 규칙 — ONNX 밖 코드
- **MEAN · STD**: ImageNet 정규화 값 `[0.485, 0.456, 0.406]` · `[0.229, 0.224, 0.225]`
- **top-5**: 확률 높은 5개 클래스

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `cv2.imread()` | `cv2.imread("butterfly.jpg")` | 사진 읽기 → `(H, W, 3)` BGR `uint8` |
| `cv2.cvtColor()` | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | BGR → RGB |
| `cv2.resize()` | `cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)` | 크기 맞추기 — 인자 순서 **(W, H)** |
| `ndarray.transpose()` | `x.transpose(2, 0, 1)` | `(H, W, C)` → `(C, H, W)` |
| `preprocess()` | `preprocess(img_bgr, size=224)` | 6단계를 묶은 함수 → `(1, 3, size, size)` float32 |
| `classify()` | `classify(sess, img_bgr, k=5)` | 전처리 → 실행 → softmax → top-k `(이름, 확률)` |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 7-1 | BGR→RGB + 224 리사이즈 → `(224, 224, 3)` `uint8` |
| 7-2 | `/255` · 정규화 → `float32`, 범위 −2.12 ~ 2.62 (butterfly) |
| 7-3 | `transpose` · `[None]` → `(1, 3, 224, 224)` float32 |
| 7-4 | softmax · top-5 (확률 합 = 1) |

**핵심 내용**

#### 전처리 6단계
> imread → BGR→RGB · resize(W, H) → /255 · 정규화 → HWC→CHW → 배치 축 → run · softmax
| 단계 | 코드 | shape · dtype |
|---|---|---|
| ① 읽기 | `cv2.imread` | (356, 493, 3) uint8 BGR |
| ② 색 · 크기 | `cvtColor` · `resize(224, 224)` | (224, 224, 3) uint8 RGB |
| ③ 정규화 | `/255` · `(x − MEAN) / STD` | (224, 224, 3) float32 |
| ④ 축 순서 | `transpose(2, 0, 1)` | (3, 224, 224) |
| ⑤ 배치 축 | `[None]` | (1, 3, 224, 224) |
| ⑥ 실행 | `run` → softmax → top-5 | (1000,) 확률 |
- ⚠ `cv2.resize(img, (H, W))` 아님 — **(W, H)**. 정사각형이면 티가 안 나서 더 위험
- ⚠ `x.reshape(3, 224, 224)` 는 모양만 바꿈 → 축 순서는 `transpose`
- ⚠ `MEAN` 을 `np.array([...])` 로 쓰면 float64 → 결과가 float64 → ORT 오류 (`np.float32` 지정)

#### softmax · top-5
> `p = exp(logits − max) / Σ` — 점수를 확률로, `argsort` 로 큰 순서
- **왜 `max` 를 빼나** → `exp` 가 너무 커지는 overflow 방지 (NN 과정의 그 방법)
- **함수로 묶기** → `preprocess` · `classify` — 어떤 사진에도 사용
- ⚠ `argmax` 만 보지 말고 top-5 와 확률로 모델의 확신 정도를 볼 것
- ⚠ 전처리를 하나라도 빼면 오류 없이 틀림: BGR 그대로 · 정규화 없음 · `/255` 없음 (실험 셀로 비교)

**기본 사용형태**
```python
def preprocess(img_bgr, size=224):
    img = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)
    x = (img.astype(np.float32) / 255 - MEAN) / STD
    return x.transpose(2, 0, 1)[None]                      # (1, 3, size, size) float32

logits = sess.run(None, {"input": preprocess(cv2.imread("butterfly.jpg"))})[0][0]
```

**연결**
- [VB D1·6교시 디지털 이미지: 픽셀 · 채널 · 정규화](#/vb/1/6) — 정규화 · 텐서 변환의 원래 학습
- [NN D2·4교시 softmax](#/nn/2/4) — 점수 → 확률, `max` 빼기
- [OT D1·8교시 미니 프로젝트](#/ot/1/8) — `preprocess(size=64)` 로 재사용

**상세 학습**

#### 이 교시에서 답하는 질문
MNIST 는 `/255` 하나로 끝났지만 실제 컬러 사진은 전처리가 여러 단계다. VB Day 5 의 `mnv3s.onnx` (ImageNet 1000 클래스 MobileNetV3-Small, 입력 `input ['batch', 3, 224, 224]`) 로 사진을 분류하면서, **사진 한 장이 모델 입력이 되기까지의 6단계**와 각 단계에서 틀리기 쉬운 곳을 정리한다.

#### 개념 1 · 전처리 6단계
![사진 한 장을 (1, 3, 224, 224) 입력으로 만드는 전처리 6단계](assets/fig/ot/d1-preprocess.svg)

| 단계 | 코드 | 모양 · 타입 |
|---|---|---|
| ① 읽기 | `cv2.imread(path)` | (H, W, 3) uint8, **BGR** |
| ② 색 · 크기 | `cv2.cvtColor(…, BGR2RGB)` → `cv2.resize(…, (224, 224), INTER_AREA)` | (224, 224, 3) uint8 |
| ③ 값 | `/ 255` → `(x − MEAN) / STD` | float32, 범위 −2.12 ~ 2.62 |
| ④ 축 순서 | `x.transpose(2, 0, 1)` | (3, 224, 224) |
| ⑤ 배치 축 | `x[None]` | (1, 3, 224, 224) |
| ⑥ 실행 · 해석 | `sess.run` → softmax → top-5 | (1000,) 확률 |

#### 개념 2 · OpenCV 와 PyTorch 의 약속 차이
| | OpenCV | PyTorch · ONNX 모델 |
|---|---|---|
| 색 순서 | BGR | RGB |
| 축 순서 | (H, W, C) | (N, C, H, W) |
| 크기 인자 | `resize(img, (W, H))` | — |
| 값 | 0 ~ 255 uint8 | 정규화된 float32 |

#### 개념 3 · 정규화와 softmax
- **정규화**: ImageNet 학습 때 쓴 값 MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225] (채널별). `np.array(…, np.float32)` 로 만들어야 결과도 float32 — float64 면 ORT 오류.
- **softmax**: `e = exp(logits − logits.max())`, `prob = e / e.sum()` — max 를 빼는 건 exp 가 넘치지 않게 (값은 같다). 합이 1 인지 확인.
- **top-5**: `np.argsort(prob)[::-1][:5]` — argmax 하나만 보면 모델이 얼마나 확신하는지 모른다.

전처리를 일부러 틀리게 해 보면 (강의 바꿔보기): BGR 그대로 · 정규화 없음 · `/255` 없음 → 셋 다 **오류 없이** top-1 이 바뀌거나 확신도가 떨어진다. 가장 크게 망가뜨리는 건 `/255` 누락 (값이 255배 커짐).

#### 코드로 확인
```python
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img224 = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
x = (img224.astype(np.float32) / 255 - MEAN) / STD
x_in = x.transpose(2, 0, 1)[None]
print(img224.shape, x.dtype, f"{x.min():.2f} ~ {x.max():.2f}", x_in.shape)
```
- **무엇을 하나**: 전처리 ②~⑤.
- **어떤 개념**: 개념 1 · 2.
- **출력**: `(224, 224, 3)`, `float32 -2.12 ~ 2.62`, `(1, 3, 224, 224)` (직접 실행)

```python
def classify(sess, img_bgr, k=5):
    logits = sess.run(None, {"input": preprocess(img_bgr)})[0][0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    top = np.argsort(p)[::-1][:k]
    return [(CLASSES[i], float(p[i])) for i in top]
```
- **무엇을 하나**: 전처리 → 실행 → softmax → top-k 를 함수 하나로 (8교시에서 그대로 씀).
- **어떤 개념**: 개념 3.
- **출력**: 사진마다 `(클래스 이름, 확률)` k개 — `prob.sum() ≈ 1`, 확률 내림차순 (교안 검증 셀)

#### 한 번에 정리
- 전처리 6단계: imread → RGB · resize(W, H) → /255 · 정규화 → HWC→CHW → 배치 축 → run · softmax.
- 전처리는 ONNX 밖 — 학습 때와 같은 순서 · 같은 값이어야 한다.
- `preprocess` · `classify` 로 묶어 두면 어떤 사진에도 쓴다.
- ⚠ `reshape(3, 224, 224)` 는 축 순서를 바꾸지 않는다 — `transpose(2, 0, 1)`.

---

### 8교시 · 미니 프로젝트: 혼자서 4단계 · 채점

**키워드**: `SafetyNet`, `BatchNorm 이 Conv 에 합쳐짐`, `onnx_report`, `배포 = ONNX + 전처리 + 클래스 이름`, `GlobalAveragePool`

**개요**: 오늘의 4단계(eval → export → Netron·`onnx_report` → ORT 실행·allclose)를 처음 보는 모델 **SafetyNet**(64×64 컬러 → safe · warning · danger 3분류, 가중치는 seed 7 연습용)에 혼자 적용한다. 과제 6개(내보내기 20 · 그래프 요약 20 · 실행 비교 20 · `onnx_report` 20 · 사진 3장 분류 10 · 해석 10). 핵심 발견: **eval 모드로 export 하면 BN 이 Conv 에 합쳐져** Netron 에 `BatchNormalization` 이 없다.

**핵심 용어**
- **BN 융합(fusion)**: eval 모드 export 때 `BatchNorm` 의 계산이 앞 `Conv` 의 가중치에 합쳐지는 것
- **onnx_report**: 파일 크기 · 노드 수 · 연산 종류 · 입출력 정보를 딕셔너리로 돌려주는 점검 함수
- **opset**: ONNX 연산자 버전 — 실행기마다 지원 범위가 다름 (Day 2)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `onnx_report()` | `onnx_report("safety.onnx")` | 파일 · 입력 · 출력 정보 딕셔너리 |
| `os.path.getsize()` | `os.path.getsize(path)` | 파일 크기 (byte) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 8-1 | `safety.eval()` 후 `safety.onnx` export (7,502 B) |
| 8-2 | 노드 9개, 가중치 노드 3, 파라미터 1,491 |
| 8-3 | 난수 10개 최대 차이 ≈ 8.2e-08 → 같다 |
| 8-4 | `onnx_report` 딕셔너리 5키 |
| 8-5 | 사진 3장을 `size=64` 로 전처리 → `(1, 3, 64, 64)` 분류 |

**핵심 내용**

#### 처음 보는 모델에 4단계
> `eval()` → `export`(이름 · 1장 shape) → `onnx_report` · Netron → ORT 실행 · `allclose`
- **SafetyNet 결과** → 노드 `['Conv', 'Relu', 'MaxPool', 'Conv', 'Relu', 'MaxPool', 'GlobalAveragePool', 'Flatten', 'Gemm']` 9개, 가중치 노드 3, 파일 7,502 B (7.3 KB)
- **파라미터** → Conv 224 + BN 16 + Conv 1,168 + BN 32 + Linear 51 = **1,491**
- **BN 이 없는 이유** → eval 모드에서 BN 의 평균 · 분산이 고정 상수라 앞 Conv 에 합쳐짐 → 그래프가 PyTorch 코드와 똑같이 생기지 않을 수 있음
- ⚠ `eval()` 을 빼고 비교하면 PyTorch 쪽이 배치 통계로 계산 → 최대 차이 ≈ 0.58 로 어긋남 (실행 확인)
- ⚠ 최대 차이가 0 이 아니어도(≈ 1e-7) float 오차라 "같다" 가 맞음

#### onnx_report — 모든 모델을 먼저 살펴보는 함수
> 경로 → `{size_kb, n_nodes, ops, inputs, outputs}`
- **safety.onnx** → `size_kb 7.3` · `n_nodes 9` · `ops ['Conv', 'Flatten', 'Gemm', 'GlobalAveragePool', 'MaxPool', 'Relu']` · `inputs [('input', [1, 3, 64, 64])]` · `outputs [('logits', [1, 3])]`
- **입출력** → `onnx.load` 가 아니라 ORT 세션의 `get_inputs()` · `get_outputs()` 로 얻음

#### 배포 = ONNX 파일 + 전처리 규칙 + 클래스 이름
> 파일 하나만 옮기면 안 된다 — 셋 다 전달
- **전달 항목** → ① `.onnx` ② 전처리 규칙(크기 · RGB · 정규화 값 · CHW) ③ 클래스 이름 목록
- ⚠ 오류 없이 틀리는 경우를 잡으려면 배포 전에 PyTorch 와 같은 입력으로 결과 비교
- ⚠ 실행기마다 지원 연산(opset)이 달라 Day 2 에서 직접 만난다

**기본 사용형태**
```python
def onnx_report(path):
    model = onnx.load(path)
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return {"size_kb": round(os.path.getsize(path) / 1024, 1),
            "n_nodes": len(model.graph.node),
            "ops": sorted(set(n.op_type for n in model.graph.node)),
            "inputs": [(i.name, i.shape) for i in s.get_inputs()],
            "outputs": [(o.name, o.shape) for o in s.get_outputs()]}
```

**연결**
- [OT D1·4교시 ONNX Runtime 으로 실행](#/ot/1/4) — `allclose` 비교
- [OT D1·7교시 이미지 한 장 추론](#/ot/1/7) — `preprocess(size=64)` 재사용
- [NN D4·4교시 Batch Normalization](#/nn/4/4) — BN 의 평균 · 분산이 eval 때 고정되는 이유
- [VB D5·6교시 성능 측정법](#/vb/5/6) — Day 3 의 성능 측정으로 이어짐

**상세 학습**

#### 이 교시에서 답하는 질문
오늘 반복한 4단계를 **처음 보는 모델**에 혼자 적용한다. 현장 안전 카메라용으로 가정한 작은 CNN **SafetyNet** (64×64 컬러 → safe · warning · danger) 이다. 가중치는 seed 로 고정한 연습용이라 정확도가 아니라 **배포 과정**이 과제다. 그리고 그래프가 PyTorch 코드와 똑같이 생기지 않는 이유(BN 합치기)를 찾는다.

#### 개념 1 · 과제와 배점
| 과제 | 내용 | 배점 |
|---|---|---|
| 1 | `safety.eval()` → `safety.onnx` (이름 input / logits, 1장 64×64) | 20 |
| 2 | 그래프 요약 — 노드 종류 · 가중치 노드 수 · 파라미터 수 | 20 |
| 3 | ORT 실행 · 난수 10개로 PyTorch 와 최대 차이 | 20 |
| 4 | `onnx_report(path)` — 파일 · 노드 · 입출력을 딕셔너리로 | 20 |
| 5 | 사진 3장 분류 (`preprocess(size=64)`) | 10 |
| 6 | 결과 해석 (BN · eval · 차이 · 전달할 정보) | 10 |

#### 개념 2 · BN 은 어디로 갔나
![SafetyNet 의 PyTorch 층과 ONNX 노드 비교 — BatchNorm 이 Conv 안으로 합쳐진다](assets/fig/ot/d1-bnfold.svg)

실행 모드의 BatchNorm 은 고정된 μ · σ² · γ · β 로 하는 **곱셈 · 덧셈**일 뿐이다. 바로 앞 Conv 의 가중치에 미리 곱해 넣을 수 있다:

| 기호 | 뜻 |
|---|---|
| W′ = W · γ / √(σ² + ε) | 합친 Conv 가중치 |
| b′ = (b − μ) · γ / √(σ² + ε) + β | 합친 Conv bias |

그래서 PyTorch 층 11개(Conv · BN · ReLU · Pool × 2 + GAP · Flatten · Linear)가 ONNX 노드 **9개**가 된다 — `BatchNormalization` 노드가 없다. 파라미터 1,491 = Conv 224 + BN 16 + Conv 1,168 + BN 32 + Linear 51.

`eval()` 을 빼면 PyTorch 쪽 BN 이 **배치 통계**를 써서 기준값이 달라지고 과제 3 비교가 실패한다.

#### 개념 3 · onnx_report — 모든 모델을 먼저 살펴보는 함수
| 파일 | size_kb | n_nodes | ops | inputs → outputs |
|---|---|---|---|---|
| tiny_named.onnx | 0.2 | 1 | Gemm | `input [1, 3]` → `output [1, 2]` |
| mnist_cnn.onnx | 1695.9 | 7 | Conv · Flatten · Gemm · MaxPool · Relu | `input [1, 1, 28, 28]` → `logits [1, 10]` |
| safety.onnx | 7.3 | 9 | Conv · Flatten · Gemm · GlobalAveragePool · MaxPool · Relu | `input [1, 3, 64, 64]` → `logits [1, 3]` |

#### 개념 4 · 배포 = 파일 + 전처리 규칙 + 클래스 이름
ONNX 파일만 넘기면 받는 사람은 입력을 어떻게 만들지 모른다. 함께 전달할 것: ① 전처리 규칙 (64×64 · RGB · /255 · MEAN/STD · NCHW) ② 클래스 이름 순서 (`["safe", "warning", "danger"]`) ③ 입출력 이름 · shape (`input [1, 3, 64, 64]` → `logits [1, 3]`).

#### 코드로 확인
```python
safety.eval()
torch.onnx.export(safety, torch.randn(1, 3, 64, 64), "safety.onnx",
                  input_names=["input"], output_names=["logits"], **EXPORT_KW)
g_s = onnx.load("safety.onnx").graph
ops = [n.op_type for n in g_s.node]
print(ops, sum(op in ("Conv", "Gemm") for op in ops), sum(p.numel() for p in safety.parameters()))
```
- **무엇을 하나**: 과제 1 · 2 — 내보내고 그래프 요약.
- **어떤 개념**: 개념 2.
- **출력**: `safety.onnx 7,502 B`, `['Conv', 'Relu', 'MaxPool', 'Conv', 'Relu', 'MaxPool', 'GlobalAveragePool', 'Flatten', 'Gemm']`, 가중치 노드 3, 파라미터 1491 (직접 실행)

```python
def onnx_report(path):
    model = onnx.load(path)
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return {"size_kb": round(os.path.getsize(path) / 1024, 1),
            "n_nodes": len(model.graph.node),
            "ops": sorted(set(n.op_type for n in model.graph.node)),
            "inputs": [(i.name, i.shape) for i in s.get_inputs()],
            "outputs": [(o.name, o.shape) for o in s.get_outputs()]}
```
- **무엇을 하나**: 과제 4 — 파일 하나를 요약하는 함수.
- **어떤 개념**: 개념 3.
- **출력**: 위 표의 세 줄 (직접 실행). 과제 3 최대 차이 `8.20e-08 | allclose: True`

#### 한 번에 정리
- 처음 보는 모델도 eval → export(이름 · shape) → Netron · `onnx_report` → ORT 실행 · allclose.
- eval 모드 export 는 Conv 뒤 BN 을 합친다 — 노드 11 → 9, 그래프가 코드와 다르게 생길 수 있다.
- 배포 = ONNX 파일 + 전처리 규칙 + 클래스 이름 — 셋 다 전달.
- ⚠ 최대 차이가 0 이 아니어도(8.2e-08) float32 끝자리 차이라 "같다" 로 판정한다.

