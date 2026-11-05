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
VB 에서 만든 모델은 Colab 안에서만 돌았다. 현장 장치로 옮기려면 "파일 하나만 들고 가서 돌리는" 방법이 필요하다. `.pt` 는 숫자뿐이라 모델 코드까지 따로 옮겨야 하고, ONNX 는 계산 순서까지 한 파일에 들어 있다.

#### 개념 1 · .pt 에는 숫자만 있다
`nn.Linear(3, 2)` 는 `weight` 6개(2×3) + `bias` 2개 = **숫자 8개**. `torch.save(model.state_dict())` 는 이 8개만 저장한다. "입력에 `weight` 를 곱하고 `bias` 를 더한다" 는 순서는 파이썬 클래스 안에만 있다.

#### 개념 2 · 추론 = 전처리 → forward → 결과 해석
| 단계 | 어디에 있나 |
|---|---|
| 전처리 (`/255` · 정규화 · 모양) | ONNX 밖 (실행 쪽 코드) |
| forward | **ONNX 안** |
| 결과 해석 (argmax · 이름) | ONNX 밖 |

#### 코드로 확인
```python
sd = torch.load("tiny.pt")
for k in sd:
    print(k, tuple(sd[k].shape))
```
- **무엇을 하나**: `.pt` 안에 뭐가 들었는지 열어 본다
- **어떤 개념**: 개념 1 — 숫자만 있다
- **출력**: `weight (2, 3)` · `bias (2,)`

#### 한 번에 정리
- `.pt` = 가중치만 · ONNX = 계산 순서 + 가중치 · 추론에 필요한 건 forward
- 전처리 · 결과 해석은 ONNX 에 없다 — 따로 챙겨서 전달
- ⚠ ONNX = 형식일 뿐 속도 보장 아님

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
1교시에서 "계산 순서까지 파일에 담는다" 고 했다. 그 파일을 실제로 만드는 방법이 `torch.onnx.export` 한 줄이다. 입력 모양을 어떻게 정하는지, 파일 크기가 어디서 오는지 확인한다.

#### 개념 1 · 숫자를 직접 정해 둔 모델
`W = [[1, 2, 3], [0, -1, 1]]`, `b = [0.5, -0.5]`, `y = x Wᵀ + b`.
| 기호 | 뜻 |
|---|---|
| x | 입력 (1, 3) |
| W | 가중치 (2, 3) = (출력, 입력) |
| b | 편향 (2,) |
예: x = [1, 1, 1] → y₁ = 1+2+3+0.5 = **6.5**, y₂ = 0−1+1−0.5 = **−0.5**

#### 개념 2 · 더미 입력 = shape 를 알려 주는 용도
export 는 값이 아니라 **모양**을 녹화한다. `torch.zeros(1, 3)` 도 같다. 첫 `1` = 배치, `3` = `in_features`.

#### 코드로 확인
```python
n_params = sum(p.numel() for p in tiny.parameters())
print(n_params, n_params * 4, os.path.getsize("tiny.onnx"))
```
- **무엇을 하나**: 파라미터 수 × 4 B 와 실제 파일 크기 비교
- **어떤 개념**: 개념 2 — 크기의 출처
- **출력**: `8 32 249`

#### 한 번에 정리
- export = 더미 입력으로 녹화 · 더미는 shape · dtype 이 중요
- 파일 크기 ≈ 파라미터 × 4 B + 그래프 설명
- ⚠ 배치 축 없는 더미 · float64 더미 · 파일 이름 오타(`FileNotFoundError`)

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
export 한 파일이 정말 내가 만든 모델인가? 눈으로 확인하는 도구가 Netron 이고, 코드로 확인하는 방법이 `onnx.load` 이다.

#### 개념 1 · 노드와 initializer
| 구성 | 코드 | 예 (`tiny.onnx`) |
|---|---|---|
| 노드 | `m.graph.node` | `Gemm` 1개 |
| 가중치 | `m.graph.initializer` | `B (2, 3)`, `C (2,)` |
| 입력 | `m.graph.input` | `onnx::Gemm_0` `[1, 3]` |

#### 개념 2 · 층을 쌓으면 상자가 늘어난다
`Linear(3,4) → ReLU → Linear(4,2)` 를 export 하면 위에서 아래로 계산 순서대로 `Gemm · Relu · Gemm`.

#### 코드로 확인
```python
mlp = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
torch.onnx.export(mlp, torch.randn(1, 3), "mlp.onnx", **EXPORT_KW)
g = onnx.load("mlp.onnx").graph
print([n.op_type for n in g.node], len(g.initializer))
```
- **무엇을 하나**: 2층 MLP 의 상자 순서와 가중치 개수 확인
- **어떤 개념**: 개념 2
- **출력**: `['Gemm', 'Relu', 'Gemm'] 4`

#### 한 번에 정리
- Netron: 상자 = 노드 · B C = 가중치 · `Linear → Gemm(transB=1)` · `ReLU → Relu`
- 코드로는 `onnx.load(path).graph.node / .initializer`
- ⚠ `initializer[0]` = weight 라는 가정은 작은 모델에서만 맞음

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
파일을 만들었으니 실제로 돌려 보고, PyTorch 와 같은 답이 나오는지 확인해야 한다. 이 확인이 "배포 전 검증" 의 기본이다.

#### 개념 1 · 입력은 딕셔너리, 출력은 리스트
`sess.run(None, {"이름": 배열})` → `[출력0, 출력1, …]`. 입력 이름은 `get_inputs()` 로 확인.

#### 개념 2 · 에러 메시지로 원인 찾기
| 실수 | 메시지 (실행 확인) |
|---|---|
| float64 배열 | `Unexpected input data type. Actual: (tensor(double)) , expected: (tensor(float))` |
| 이름 틀림 | `Required inputs (['onnx::Gemm_0']) are missing from input feed (['x'])` |

#### 코드로 확인
```python
sess = ort.InferenceSession("tiny.onnx", providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
print(sess.run(None, {name: np.array([[1, 1, 1]], np.float32)}))
```
- **무엇을 하나**: ORT 로 `tiny.onnx` 실행
- **어떤 개념**: 개념 1
- **출력**: `[array([[ 6.5, -0.5]], dtype=float32)]`

#### 한 번에 정리
- ORT 4단계: 열기 → 이름 → `run` → `allclose`
- 입력 = 딕셔너리(float32 NumPy) · 출력 = 리스트
- ⚠ `==` 로 비교하지 말고 `np.allclose`

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
4교시의 `onnx::Gemm_0` 같은 이름은 불편하고, 입력 shape 가 어디서 정해지는지도 모호했다. 이름을 붙이고 shape 규칙을 확인한다.

#### 개념 1 · 이름 = 실행 코드와의 약속
export 때 붙인 이름이 `run` 의 딕셔너리 키가 된다. 배포할 때 이름도 함께 전달해야 한다.

#### 개념 2 · shape 는 더미 입력에서 굳는다
`(1, 3)` 더미 → 입력 `[1, 3]`. 4개를 한 번에 넣으면 `index: 0 Got: 4 Expected: 1`. 오늘은 **1장씩** 넣는다.

#### 코드로 확인
```python
x4 = np.random.rand(4, 3).astype(np.float32)
try: s.run(None, {"input": x4})
except Exception as e: print(str(e)[:120])
```
- **무엇을 하나**: 배치 4 를 넣어 오류 확인
- **어떤 개념**: 개념 2
- **출력**: `Got invalid dimensions for input: input … index: 0 Got: 4 Expected: 1`

#### 한 번에 정리
- 이름은 리스트로 · 이미지는 (N, C, H, W) · 1장도 N = 1
- 입력 shape 는 export 더미로 고정 (자유롭게 = Day 2)
- ⚠ 배치 축 빼먹으면 `Invalid rank`

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
`tiny` 에서 되던 4단계가 Conv · Pool 이 있는 실제 모델에서도 그대로인가? 그리고 숫자 말고 이미지가 들어갈 때 어떤 함정이 생기나?

#### 개념 1 · 가중치 노드 세기
| 노드 | 파라미터 |
|---|---|
| Conv (30, 1, 5, 5) | 30·1·25 + 30 = 780 |
| Gemm (4320 → 100) | 4320·100 + 100 = 432,100 |
| Gemm (100 → 10) | 100·10 + 10 = 1,010 |
| 합계 | **433,890** |

#### 개념 2 · 전처리 일치가 "같은 모델" 의 조건
ONNX 는 forward 만 담는다. `/255` 가 빠지면 그래프는 같은데 입력 분포가 달라져 결과가 다르다.

#### 코드로 확인
```python
g = onnx.load("mnist_cnn.onnx").graph
ops = [n.op_type for n in g.node]
print(ops, sum(op in ("Conv", "Gemm") for op in ops))
```
- **무엇을 하나**: 노드 종류와 가중치 노드 수
- **어떤 개념**: 개념 1
- **출력**: `['Conv', 'Relu', 'MaxPool', 'Flatten', 'Gemm', 'Relu', 'Gemm'] 3`

#### 한 번에 정리
- 4단계 불변: eval → export → 확인 → 실행 · 비교
- 파일 ≈ 파라미터 × 4 B (1.74 MB) · 가중치 노드 3
- ⚠ 전처리를 학습 때와 다르게 하면 오류 없이 틀린다

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
숫자 입력과 달리 사진은 읽기 · 색 순서 · 크기 · 축 순서 · 정규화 · 배치 축까지 맞춰야 모델이 읽는다. 그 규칙(전처리)은 ONNX 에 없으므로 실행 쪽 코드가 학습 때와 똑같이 만들어야 한다.

#### 개념 1 · OpenCV 와 PyTorch 의 약속 차이
| 항목 | OpenCV | 모델 입력 |
|---|---|---|
| 색 순서 | BGR | RGB |
| 축 순서 | (H, W, C) | (C, H, W) |
| 크기 인자 | (W, H) | — |
| 값 범위 | 0 ~ 255 uint8 | 0~1 → 정규화 float32 |

#### 개념 2 · 정규화
`x' = (x/255 − MEAN) / STD` — 채널마다 평균 · 표준편차로 맞춘다. 결과 범위 예: butterfly −2.12 ~ 2.62.

#### 코드로 확인
```python
img = cv2.imread("butterfly.jpg")
x = preprocess(img)
print(img.shape, "→", x.shape, x.dtype)
```
- **무엇을 하나**: 전처리 전 · 후 shape 확인
- **어떤 개념**: 개념 1 · 2
- **출력**: `(356, 493, 3) → (1, 3, 224, 224) float32`

#### 한 번에 정리
- 6단계: 읽기 → RGB·resize → 정규화 → CHW → 배치 → run·softmax
- 전처리는 ONNX 밖 · 학습 때와 같은 순서 · 같은 값
- ⚠ `resize` 는 (W, H) · MEAN 은 float32

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
오늘 배운 4단계를 아무 모델에나 적용할 수 있는가? 처음 보는 BN 포함 모델로 내보내기 · 점검 · 실행 · 비교를 혼자 해 보고, 배포에 무엇을 같이 넘겨야 하는지 정리한다.

#### 개념 1 · 4단계 체크리스트
| 단계 | 한 일 | 확인 값 (SafetyNet) |
|---|---|---|
| ① eval + export | `input` / `logits`, `(1, 3, 64, 64)` | 7,502 B |
| ② 열어 보기 | `onnx_report` · Netron | 노드 9 · 가중치 노드 3 |
| ③ ORT 실행 | 세션 · `run` | 출력 `(1, 3)` |
| ④ 비교 | `allclose` | 최대 차이 ≈ 8e-08 |

#### 개념 2 · BN 은 어디로 갔나
eval 모드 BN 은 `y = (x − 평균) / √분산 · γ + β` 가 상수뿐이라 Conv 의 weight · bias 로 흡수된다. 그래서 `Conv → BN → ReLU` 가 `Conv → Relu` 로 그려진다.

#### 코드로 확인
```python
g = onnx.load("safety.onnx").graph
ops = [n.op_type for n in g.node]
print(len(ops), "BatchNormalization" in ops, sum(p.numel() for p in safety.parameters()))
```
- **무엇을 하나**: 노드 수 · BN 존재 여부 · 파라미터 수
- **어떤 개념**: 개념 2
- **출력**: `9 False 1491`

#### 한 번에 정리
- 4단계: eval → export → onnx_report·Netron → ORT·allclose
- eval export 는 BN 을 Conv 에 합침 · 배포 = 파일 + 전처리 + 클래스 이름
- ⚠ `eval()` 빼먹으면 비교가 크게 어긋남 (≈ 0.58)
