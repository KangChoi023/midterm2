## Day 2 — ONNX 속 들여다보기: 코드로 읽고, 오류를 고치기

### 1교시 · Day 1 복습 · 4단계 다시 · 오늘의 지도

**키워드**: `4단계`, `eval → export → 열어보기 → run·allclose`, `노드 7 · 가중치 6`, `오늘의 지도`

**개요**: 어제의 4단계를 `mnist_cnn.onnx` 로 혼자 다시 한다(복습 퀴즈 20문항 후 문제 1-1 · 1-2). 오늘은 4단계 중 **"열어보기"를 Netron 대신 코드로** 하고, 어제 막혔던 4가지(그래프 구조 · opset · 배치 크기 · export 가 조용히 틀리는 경우)를 푼다. 기억할 숫자: `mnist_cnn.onnx` = **노드 7개 · 가중치 텐서 6개**.

**핵심 용어**
- **4단계**: ① `eval()` ② `export`(이름 · shape) ③ 열어보기 ④ `run` · `allclose`
- **노드(node)**: 그래프의 연산 하나 — Conv · Relu · MaxPool · Flatten · Gemm · Relu · Gemm = 7개
- **가중치 텐서(initializer)**: `conv.weight` · `conv.bias` · `fc1.weight` · `fc1.bias` · `fc2.weight` · `fc2.bias` = 6개

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Module.eval()` | `cnn.eval()` | 실행 모드 (복습) |
| `torch.onnx.export()` | `torch.onnx.export(cnn, torch.zeros(1, 1, 28, 28), "mnist_cnn.onnx", input_names=["input"], output_names=["logits"], **EXPORT_KW)` | 내보내기 (복습) |

**핵심 내용**

#### 4단계 복습 — 오늘은 3단계(열어보기)를 코드로
> `eval()` → `export`(이름 · shape) → 열어보기 → `run` · `allclose`
- **1단계 확인** → 예측과 PyTorch 가 `np.allclose(atol=1e-5)` 로 같음, 입력 `[('input', [1, 1, 28, 28])]`
- **노드 7 · 가중치 6** → 어제 Netron 으로 센 값을 오늘 `len(graph.node)` · `len(graph.initializer)` 로 꺼냄
- ⚠ 숫자를 모르고 코드를 돌리면 결과 확인이 안 됨 — 7 · 6 을 외워 두기

**기본 사용형태**
```python
cnn.eval()
torch.onnx.export(cnn, torch.zeros(1, 1, 28, 28), "mnist_cnn.onnx",
                  input_names=["input"], output_names=["logits"], **EXPORT_KW)
sess_cnn = ort.InferenceSession("mnist_cnn.onnx", providers=["CPUExecutionProvider"])
y_ort = sess_cnn.run(None, {"input": x0})[0]
```

**연결**
- [OT D1·6교시 MNIST CNN 내보내기](#/ot/1/6) — 같은 4단계를 큰 모델에
- [OT D2·2교시 ONNX 파일 = 그래프](#/ot/2/2) — 노드 7 · 가중치 6 을 코드로 꺼냄

**상세 학습**

#### 이 교시에서 답하는 질문
Day 1 은 "내보내고 · 보고 · 실행" 까지였다. 보는 일을 Netron 그림에만 맡기면 모델이 커질 때 확인할 방법이 없다. 오늘은 파일 속을 **코드로** 읽고, 오류가 났을 때 **스스로 고치는** 법을 배운다.

#### 개념 1 · 오늘의 지도
| 어제 막힌 것 | 오늘 푸는 교시 |
|---|---|
| 파일 속이 어떻게 생겼나 | 2 · 3교시 (그래프 · 가중치 · 속성) |
| 연산 사전 버전(opset) · 파일 검사 | 4교시 |
| 배치 4를 넣으면 오류 | 5교시 (`dynamic_axes`) |
| 오류 없이 결과가 틀림 · 미지원 연산 | 6 · 7교시 |

#### 코드로 확인
```python
same = np.allclose(y_ort, y_pt, atol=1e-5)
print("예측", y_ort.argmax(), "| 정답", labels[0], "| PyTorch 와 같다:", same)
```
- **무엇을 하나**: ORT 와 PyTorch 결과 비교
- **어떤 개념**: 4단계의 마지막 — `allclose`
- **출력**: `same = True` (직접 실행)

#### 한 번에 정리
- 4단계: eval → export → 열어보기 → run·allclose
- 기억할 숫자: 노드 7 · 가중치 텐서 6
- ⚠ 오늘 내내 쓰는 규칙 — 어떤 변경이든 **마지막은 `allclose`**

---

### 2교시 · ONNX 파일 = 그래프: input · node · initializer · output

**키워드**: `ModelProto`, `GraphProto`, `input · node · initializer · output`, `이름이 같으면 화살표`, `topological order`

**개요**: `onnx.load` 한 결과는 상자 속 상자다. 가장 바깥 `ModelProto` 안에 `GraphProto`(`model.graph`)가 있고, 그 안에 **네 목록**(`input` · `node` · `initializer` · `output`)이 있다. 화살표는 파일에 따로 저장되지 않는다 — 노드는 입력 이름 목록과 출력 이름 목록만 갖고, **앞 노드의 output 이름 = 뒤 노드의 input 이름** 이면 이어진다. `mnist_cnn.onnx`: input 1 · output 1 · node 7 · initializer 6.

**핵심 용어**
- **ModelProto / GraphProto**: 모델(버전 정보 포함) / 그 안의 계산 그래프
- **initializer**: 그래프 안에 저장된 가중치 — 바깥에서 넣어 주는 `input` 이 **아님**
- **topological order**: 노드를 "계산 순서대로" 저장해야 한다는 규칙 (앞 노드 출력이 뒤 노드 입력)
- **value_info**: 중간 화살표의 shape 정보 (4교시)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `onnx.load()` | `m = onnx.load("mnist_cnn.onnx"); g = m.graph` | 파일 읽기 → 모델 · 그래프 |
| `shape_of()` | `shape_of(g.input[0])` | 입출력 shape 를 리스트로 (직접 만든 도우미) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 보기 | `ir_version` 9 · `opset` 20 · `producer` pytorch (내 환경 값 — PyTorch 버전에 따라 다름) |
| 2-1 | `(input, output, node, initializer) = (1, 1, 7, 6)` |
| 2-2 | `input [1, 1, 28, 28]` → `logits [1, 10]` |
| 2-3 | 노드마다 데이터 이름 / 가중치 이름 구분 |
| 2-4 | `input` 이름에서 시작해 화살표를 따라가면 노드 순서와 같다 |

**핵심 내용**

#### ONNX 파일의 상자 구조
> `ModelProto` → `model.graph`(GraphProto) → `input` · `node` · `initializer` · `output` 네 목록
- **input** → 바깥에서 넣는 것 = 사진 1개 (`input [1, 1, 28, 28]`)
- **node** → 연산 7개: `Conv · Relu · MaxPool · Flatten · Gemm · Relu · Gemm`
- **initializer** → 가중치 6개 (파일 안에 저장된 숫자)
- **output** → `logits [1, 10]`
- ⚠ 이름은 **단수형** `g.input` · `g.node` (`g.inputs` · `g.nodes` 는 AttributeError)

#### 화살표 = 이름 맞추기
> 노드는 `input` 이름 목록 + `output` 이름 목록만 가진다 — 같은 이름이면 이어진 화살표
- **예** → `Conv` 의 output `/conv/Conv_output_0` 가 `Relu` 의 input
- **데이터 vs 가중치** → 노드 입력 중 initializer 이름이면 가중치, 아니면 데이터. Conv · Gemm = 데이터 1 + 가중치 2, 나머지 = 데이터 1 + 가중치 0
- **순서** → 화살표를 따라가도 저장 순서와 같다 (topological order)
- ⚠ `print(m)` 은 가중치 숫자 전부를 찍어 화면이 멈출 수 있음 — 필요한 부분만 꺼내기

**기본 사용형태**
```python
m = onnx.load("mnist_cnn.onnx"); g = m.graph
print(len(g.input), len(g.output), len(g.node), len(g.initializer))   # 1 1 7 6
def shape_of(vi):
    return [d.dim_value if d.dim_value else d.dim_param for d in vi.type.tensor_type.shape.dim]
for n in g.node: print(n.op_type, list(n.input), "→", list(n.output))
```

**연결**
- [OT D1·3교시 Netron 으로 보기](#/ot/1/3) — Netron 의 상자 · 화살표를 코드로
- [OT D2·3교시 가중치 · 속성 꺼내기](#/ot/2/3) — initializer 값을 NumPy 로
- [OT D2·5교시 dynamic_axes](#/ot/2/5) — `shape_of` 의 `dim_param` 쪽이 쓰이는 곳

**상세 학습**

#### 이 교시에서 답하는 질문
Netron 그림은 눈으로만 본다. 같은 정보를 코드로 꺼내면 모델이 커져도 **세고 비교하고 자동 검사**할 수 있다. 그러려면 먼저 파일이 어떤 구조인지 알아야 한다.

#### 개념 1 · 네 목록
| 목록 | 개수 | 내용 |
|---|---|---|
| `g.input` | 1 | `input [1, 1, 28, 28]` |
| `g.node` | 7 | 연산 (계산 순서대로) |
| `g.initializer` | 6 | `conv.weight` · `conv.bias` · `fc1.weight` · `fc1.bias` · `fc2.weight` · `fc2.bias` |
| `g.output` | 1 | `logits [1, 10]` |

#### 개념 2 · 노드 표 (직접 실행)
| # | op_type | input → output |
|---|---|---|
| 0 | Conv | `input`, `conv.weight`, `conv.bias` → `/conv/Conv_output_0` |
| 1 | Relu | `/conv/Conv_output_0` → `/Relu_output_0` |
| 2 | MaxPool | `/Relu_output_0` → `/pool/MaxPool_output_0` |
| 3 | Flatten | → `/Flatten_output_0` |
| 4 | Gemm | `/Flatten_output_0`, `fc1.weight`, `fc1.bias` → `/fc1/Gemm_output_0` |
| 5 | Relu | → `/Relu_1_output_0` |
| 6 | Gemm | `/Relu_1_output_0`, `fc2.weight`, `fc2.bias` → `logits` |

#### 코드로 확인
```python
name = g.input[0].name
path = []
while name != g.output[0].name:
    node = next(n for n in g.node if name in n.input)   # 이 이름을 받아 가는 노드
    path.append(node.op_type)
    name = node.output[0]                                # 다음 화살표 이름
print(" → ".join(path))
```
- **무엇을 하나**: 이름만 따라가며 그래프를 걷는다
- **어떤 개념**: 개념 2 — 이름이 같으면 화살표
- **출력**: `Conv → Relu → MaxPool → Flatten → Gemm → Relu → Gemm`

#### 한 번에 정리
- `onnx.load` → `model.graph` → input · node · initializer · output
- 화살표 = 이름 일치 · 가중치는 input 이 아니라 initializer
- ⚠ `g.nodes`(복수) 오타 · `print(m)` 로 가중치 전체 출력

---

### 3교시 · 가중치 · 속성 꺼내기: 그래프 요약 표 describe_onnx()

**키워드**: `numpy_helper.to_array`, `get_attribute_value`, `attrs_of`, `describe_onnx`, `433,890`, `Conv-BN 합치기`

**개요**: 2교시에 이름까지 읽었다. 이번에는 이름 뒤의 **숫자(가중치)** 와 노드의 **설정값(속성 attribute)** 을 꺼내고, 모두를 표 한 장으로 요약하는 `describe_onnx(path)` 를 만든다. 오늘부터 모든 모델은 이 표로 먼저 본다 (= 코드로 만든 Netron). 가중치 합계 433,890 = Day 1 의 파라미터 수.

**핵심 용어**
- **속성(attribute)**: 노드의 설정값 — Conv 의 `kernel_shape` · `strides` · `pads` 등
- **`.size` vs `.shape`**: 원소 **개수**(모든 축 곱) vs 모양
- **Conv-BN 합치기**: eval export 때 BatchNorm 이 Conv 의 weight · bias 안으로 녹는 것

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `numpy_helper.to_array()` | `onnx.numpy_helper.to_array(t)` | initializer → NumPy |
| `onnx.helper.get_attribute_value()` | `onnx.helper.get_attribute_value(a)` | 속성 하나의 값 꺼내기 |
| `attrs_of()` | `attrs_of(node)` | 노드의 속성 딕셔너리 (직접 만든 도우미) |
| `describe_onnx()` | `describe_onnx("mnist_cnn.onnx")` | 노드별 요약 표 (DataFrame) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 3-1 | 가중치 6개 shape · 합계 433,890 · PyTorch 와 `allclose` |
| 3-2 | Conv `kernel_shape [5, 5]` · MaxPool `kernel_shape [2, 2]` · `strides [2, 2]` |
| 3-3 | `describe_onnx` 완성 — 노드 7, `params` 합계 433,890 |
| 바꿔보기 | SafetyNet: PyTorch 1,491 vs ONNX 1,443 → 차이 48 |

**핵심 내용**

#### 가중치 꺼내기 — initializer → NumPy
> `numpy_helper.to_array(t)` — 이름 `t.name`, 값 `ndarray`
- **mnist_cnn 가중치** → `conv.weight (30, 1, 5, 5)` 750 · `conv.bias (30,)` 30 · `fc1.weight (100, 4320)` 432,000 · `fc1.bias` 100 · `fc2.weight (10, 100)` 1,000 · `fc2.bias` 10
- **합계** → 750 + 30 + 432,000 + 100 + 1,000 + 10 = **433,890** (Day 1 의 파라미터 수와 같음)
- ⚠ 개수는 `.size` — `.shape` 는 모양

#### 속성 꺼내기 — attrs_of
> `{a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}`
- **Conv** → `kernel_shape [5, 5]` · `strides [1, 1]` · `pads [0, 0, 0, 0]` · `dilations [1, 1]` · `group 1`
- **MaxPool** → `kernel_shape [2, 2]` · `strides [2, 2]` (= `nn.MaxPool2d(2)`)
- **Gemm** → `transB 1` (weight 를 뒤집어 곱함 — Day 1 3교시) · **Flatten** → `axis 1`
- ⚠ `a.ints` · `a.i` 를 직접 읽지 말고 도우미 함수 사용 (속성 종류마다 필드가 다름)

#### describe_onnx — 코드로 만든 Netron
> 노드 한 줄 = Netron 상자 하나: `op_type` · `data_in` · `out` · `params` · `attrs`
- **mnist_cnn** → `params = [780, 0, 0, 0, 432100, 0, 1010]` (합 433,890)
- **SafetyNet 은 다름** → PyTorch 1,491 vs ONNX 1,443, 차이 **48** = BN 의 γ · β (8 + 8 + 16 + 16)
- **왜** → Conv-BN 합치기로 BN 숫자가 Conv 에 녹아 따로 안 셈 (running_mean · var 는 원래 학습 파라미터 아님)
- ⚠ `print(df)` 는 열이 잘림 — `display(df)`

**기본 사용형태**
```python
def attrs_of(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

def describe_onnx(path):
    model = onnx.load(path); gr = model.graph
    W_ = {t.name: onnx.numpy_helper.to_array(t) for t in gr.initializer}
    rows = [{"op_type": n.op_type,
             "data_in": [x for x in n.input if x and x not in W_],
             "out": list(n.output),
             "params": sum(W_[x].size for x in n.input if x in W_),
             "attrs": attrs_of(n)} for n in gr.node]
    return pd.DataFrame(rows)
```

**연결**
- [OT D1·8교시 미니 프로젝트 (BN 융합)](#/ot/1/8) — BN 이 사라진 이유
- [NN D4·4교시 Batch Normalization](#/nn/4/4) — γ · β 가 무엇인지
- [OT D2·7교시 onnxslim](#/ot/2/7) — `describe_onnx` 로 전후 비교
- [OT D2·8교시 미니 프로젝트](#/ot/2/8) — 최종 파일을 이 표로 확인

**상세 학습**

#### 이 교시에서 답하는 질문
2교시의 이름만으로는 "모델이 몇 개의 숫자를 갖고 어떤 설정으로 계산하는가" 를 모른다. 이 교시는 숫자와 설정을 꺼내 **PyTorch 코드와 ONNX 파일이 같은 모델인지 숫자로 대조**하는 도구를 만든다.

#### 개념 1 · 가중치 합계 대조
| 텐서 | shape | 개수 |
|---|---|---|
| conv.weight | (30, 1, 5, 5) | 750 |
| conv.bias | (30,) | 30 |
| fc1.weight | (100, 4320) | 432,000 |
| fc1.bias | (100,) | 100 |
| fc2.weight | (10, 100) | 1,000 |
| fc2.bias | (10,) | 10 |
| **합계** | | **433,890** |

`4320 = 30 × 12 × 12` — Conv 5×5 로 28 → 24, MaxPool 2 로 24 → 12.

#### 개념 2 · SafetyNet 의 "사라진 48개" (직접 실행)
| 노드 | Conv | Conv | Gemm | 합 |
|---|---|---|---|---|
| ONNX params | 224 | 1,168 | 51 | **1,443** |
| PyTorch | Conv 224 + BN 16 + Conv 1,168 + BN 32 + Linear 51 | | | **1,491** |

차이 1,491 − 1,443 = **48** (= BN γ · β 8 + 8 + 16 + 16).

#### 코드로 확인
```python
df = describe_onnx("mnist_cnn.onnx")
print("노드", len(df), "| 가중치 합계", f"{df.params.sum():,}")
print(df.params.tolist())
```
- **무엇을 하나**: 노드 표를 만들어 합계 확인
- **어떤 개념**: 개념 1 — 합계 대조
- **출력**: `노드 7 | 가중치 합계 433,890` · `[780, 0, 0, 0, 432100, 0, 1010]`

#### 한 번에 정리
- 가중치 = `numpy_helper.to_array` · 속성 = `get_attribute_value` · 표 = `describe_onnx`
- 모양이 아니라 **`.size`** 로 개수 · 합계가 파라미터 수와 같은지 확인
- ⚠ ONNX 가중치 수 ≠ PyTorch 파라미터 수일 수 있음 (BN 융합)

---

### 4교시 · opset · checker · shape_inference

**키워드**: `opset`, `opset_version=`, `Erf vs Gelu`, `checker.check_model`, `shape_inference.infer_shapes`, `value_info`

**개요**: ONNX 파일을 다룰 때 쓰는 세 도구. **opset** = 파일이 쓰는 "연산 사전"의 판 번호(같은 계산도 판에 따라 노드가 달라짐), **checker** = 규칙에 맞게 만들어졌는지 검사(맞춤법 검사기), **shape_inference** = 중간 화살표마다 shape 를 계산해 적기(도면에 치수 적기). 셋 다 **"결과가 맞는지"는 말해 주지 않는다** → 마지막은 `allclose`.

**핵심 용어**
- **opset(operator set)**: 연산 사전의 판 번호 — `opset_version=` 로 정함. 높을수록 새 연산 지원, 실행기가 모르는 판이면 오류
- **checker**: 그래프 형식 검사 — 끊긴 화살표 · 순서 위반을 찾음
- **shape_inference**: 중간 텐서의 shape 를 `graph.value_info` 에 채움

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.onnx.export()` | `torch.onnx.export(model, x, path, opset_version=13, **EXPORT_KW)` | 판 번호 지정 |
| `onnx.checker.check_model()` | `onnx.checker.check_model(model)` | 형식 검사 (문제 있으면 예외) |
| `onnx.shape_inference.infer_shapes()` | `mi = onnx.shape_inference.infer_shapes(m)` | 중간 shape 채우기 (새 모델 반환) |
| `Model.SerializeToString()` | `ort.InferenceSession(bad.SerializeToString(), …)` | 모델 객체를 바이트로 (파일 없이 세션) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 4-1 | GELU 모델을 opset 13 · 20 으로 export → 노드 9개(`Erf` 로 풀어 씀) vs 2개(`Gelu` 한 단어) |
| (확인) | 두 파일 결과 최대 차이가 거의 0 → 같은 계산 |
| checker | Relu 입력 이름을 `"nope"` 로 바꾸면 끊긴 화살표 |
| 4-2 | 이름을 이전 노드 output 으로 되돌려 checker 통과 |
| 4-3 | `value_info` 0 → 6개, `/pool/MaxPool_output_0 = [1, 30, 12, 12]` |

**핵심 내용**

#### opset — 연산 사전의 판 번호
> 같은 계산도 사전의 판에 따라 노드를 다르게 적는다
- **예** → GELU: opset 13 = `['Gemm', 'Constant', 'Div', 'Erf', 'Constant', 'Add', 'Mul', 'Constant', 'Mul']` **9개** / opset 20 = `['Gemm', 'Gelu']` **2개**
- **결과는 같음** → 두 파일 출력 차이 ≈ 0 (연산 표기만 다름)
- **기본값** → `mnist_cnn.onnx` opset 은 내 환경에서 20 (PyTorch 버전마다 다름 — 8교시에 17 로 못 박음)
- ⚠ opset 을 너무 높이면 **배포할 실행기가 모른다** (`… opset … is not supported` 류) — 실행기의 지원 판부터 확인

#### checker — 고장 난 파일 찾기
> 규칙(모든 입력이 앞에서 만들어진 이름이어야 함, 계산 순서로 정렬)을 검사
- **실험** → `bad.graph.node[1].input[0] = "nope"` (Relu 의 입력 이름을 존재하지 않는 값으로)
- **checker** → `ValidationError: Nodes in a graph must be topologically sorted, however input 'nope' …`
- **ORT** → `INVALID_ARGUMENT : Invalid model. Node input 'nope' is not a graph input, initializer, or output of a previous node.`
- **고치기** → `bad.graph.node[1].input[0] = bad.graph.node[0].output[0]` 로 이름을 이어 주면 통과
- ⚠ checker 통과 = **형식이 맞다**일 뿐 — 결과가 맞는지는 `allclose`

#### shape_inference — 중간 shape 채우기
> `mi = infer_shapes(m)` → `mi.graph.value_info` 에 화살표별 shape
- **전 → 후** → `len(m.graph.value_info)` 0 → **6**
- **예** → `/conv/Conv_output_0 [1, 30, 24, 24]` → `/pool/MaxPool_output_0 [1, 30, 12, 12]` → `/Flatten_output_0 [1, 4320]` → `/fc1/Gemm_output_0 [1, 100]`
- **6개인 이유** → 화살표 7개 중 input · output 은 이미 shape 가 있음
- ⚠ `infer_shapes(m)` 만 하고 `m.graph.value_info` 를 보면 0 — **결과는 새 모델(`mi`)**

**기본 사용형태**
```python
for v in [13, 20]:
    torch.onnx.export(gelu_net, torch.randn(1, 4), f"gelu_op{v}.onnx", opset_version=v, **EXPORT_KW)

onnx.checker.check_model(onnx.load("mnist_cnn.onnx"))      # 문제 없으면 아무 말 없음
mi = onnx.shape_inference.infer_shapes(m)
shapes = {vi.name: shape_of(vi) for vi in mi.graph.value_info}
```

**연결**
- [OT D2·2교시 ONNX 파일 = 그래프](#/ot/2/2) — checker 가 보는 "순서 · 이름"
- [OT D2·7교시 미지원 연산](#/ot/2/7) — opset 을 올려서 해결되는 경우
- [OT D1·6교시 MNIST CNN 내보내기](#/ot/1/6) — Day 1 의 shape 24 → 12 → 4320 을 코드로 얻음
- [OT D2·8교시 미니 프로젝트](#/ot/2/8) — checker → onnxslim → shape_inference 순서

**상세 학습**

#### 이 교시에서 답하는 질문
파일이 열리고 실행된다고 올바른 파일일까? 실행기마다 아는 연산 사전이 다르고, 사람이 고친 파일은 깨질 수 있다. 세 도구가 각각 **무엇을 보장하고 무엇은 보장하지 않는지** 구분한다.

#### 개념 1 · 세 도구 비교
| 도구 | 하는 일 | 비유 | 보장 못 하는 것 |
|---|---|---|---|
| opset | 연산 사전의 판 번호 | 사전 몇 판 | 실행기가 그 판을 아는지 |
| checker | 형식 · 순서 검사 | 맞춤법 검사기 | 결과가 맞는지 |
| shape_inference | 중간 shape 채우기 | 도면에 치수 | 결과가 맞는지 |

#### 개념 2 · 같은 GELU, 다른 모습
| opset | 노드 수 | 노드 |
|---|---|---|
| 13 | 9 | Gemm, Constant, Div, **Erf**, Constant, Add, Mul, Constant, Mul |
| 20 | 2 | Gemm, **Gelu** |

#### 코드로 확인
```python
mi = onnx.shape_inference.infer_shapes(m)
shapes = {vi.name: shape_of(vi) for vi in mi.graph.value_info}
print("추론 전", len(m.graph.value_info), "→ 추론 후", len(shapes))
print(shapes["/pool/MaxPool_output_0"])
```
- **무엇을 하나**: 중간 화살표 shape 를 계산해 채움
- **어떤 개념**: 개념 1 — shape_inference
- **출력**: `추론 전 0 → 추론 후 6` · `[1, 30, 12, 12]`

#### 한 번에 정리
- opset = 사전 판 · checker = 규칙 검사 · shape_inference = shape 채우기
- 셋 모두 결과 정확도는 모름 → `allclose`
- ⚠ opset 을 높게 쓰면 실행기가 못 읽음 · `infer_shapes` 결과는 새 모델

---

### 5교시 · 배치 크기 자유롭게: dynamic_axes

**키워드**: `dynamic_axes`, `{0: "batch"}`, `고정 shape`, `Got: 4 Expected: 1`, `dim_param`, `Flatten → Gemm`

**개요**: Day 1 5교시 — 고정 shape 파일에 배치 4를 넣으면 `Got: 4 Expected: 1`. export 때 `dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}` 로 "이 축은 크기가 바뀔 수 있다"고 알려 주면 shape 의 첫 축이 숫자 1 대신 **이름 `'batch'`** 가 되어 1장 · 4장 · 100장을 모두 받는다. 단, dynamic 은 **약속일 뿐** — 모델이 처리 못 하는 크기는 실행 중에 오류.

**핵심 용어**
- **dynamic_axes**: `{텐서 이름: {축 번호: "축 이름"}}` — 크기가 바뀔 수 있는 축을 지정
- **dim_param**: 이름 붙은 축(`'batch'`). 숫자 축은 `dim_value`
- **배치 추론**: 여러 장을 한 번에 `run` — 1장씩 반복보다 빠르다 (정확한 측정은 Day 3)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.onnx.export()` | `torch.onnx.export(…, dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, **EXPORT_KW)` | 크기가 바뀌는 축 지정 |
| `sess.get_inputs()` | `sess_dyn.get_inputs()[0].shape` | `['batch', 1, 28, 28]` |
| `ndarray.argmax()` | `pred = out.argmax(1)` | 줄(사진)마다 가장 큰 칸 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 5-1 | 입력 `['batch', 1, 28, 28]` · 출력 `['batch', 10]`, 배치 1 · 4 · 8 → `(1, 10)` `(4, 10)` `(8, 10)` |
| 5-2 | MNIST 100장을 한 번에 → 정확도 1.0 (PyTorch 예측과 100장 모두 같음) |
| 바꿔보기 | 1장씩 100번 vs 100장 1번 시간 비교 (느낌만 — 값은 장비마다 다름) |
| 5-3 | 높이 · 너비도 dynamic 으로 열고 32×32 를 넣으면 `Gemm` 에서 실패 |

**핵심 내용**

#### dynamic_axes — 배치 축 열기
> `dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}` — 축 0 의 크기를 이름 `"batch"` 로 풀어 줌
- **전** → 고정 파일에 배치 4 → `Got: 4 Expected: 1` (오류)
- **후** → `get_inputs()[0].shape = ['batch', 1, 28, 28]`, 배치 1 · 4 · 8 모두 실행
- **100장 한 번에** → `X = (digits / 255)[:, None]` → `(100, 1, 28, 28)`, `argmax(1)` 로 사진마다 예측 → 정확도 1.0
- ⚠ 키는 `input_names` / `output_names` 와 **같은 이름**이어야 적용됨 · 값은 `{0: "batch"}` 처럼 이름을 줄 것 (리스트 `[0]` 은 이름이 자동으로 붙어 읽기 어려움)
- ⚠ `argmax()` 만 쓰면 전체에서 하나 — 사진마다는 `argmax(1)`

#### dynamic 은 약속일 뿐 — 모델이 처리 못 하는 크기
> Conv · Pool 은 어떤 크기든 되지만, Flatten 뒤 Gemm 은 **정확히 4320 개**를 원한다
- **실험** → `{0: "batch", 2: "height", 3: "width"}` 로 열고 `(1, 1, 32, 32)` 를 넣음
- **오류** → `FusedGemm … GEMM: Dimension mismatch, W: {100,4320} K: 5880 N:100`
- **5880** → 32×32 → Conv 5×5 = 28×28 → Pool = 14×14 → 30 × 14 × 14 = 5,880 ≠ 4,320
- **실패 지점** → `Gemm` (fc1)
- ⚠ 열린 축이라고 아무 크기나 되는 게 아님 — 모델 구조가 허락하는 범위

**기본 사용형태**
```python
torch.onnx.export(cnn, torch.zeros(1, 1, 28, 28), "mnist_cnn_dyn.onnx",
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, **EXPORT_KW)
sess_dyn = ort.InferenceSession("mnist_cnn_dyn.onnx", providers=["CPUExecutionProvider"])
X = (digits.astype(np.float32) / 255)[:, None]               # (100, 1, 28, 28)
pred = sess_dyn.run(None, {"input": X})[0].argmax(1)
```

**연결**
- [OT D1·5교시 이름 붙이기 · 입력 shape](#/ot/1/5) — 고정 shape 의 `Got: 4 Expected: 1`
- [OT D2·2교시 ONNX 파일 = 그래프](#/ot/2/2) — `shape_of` 의 `dim_param` 쪽이 `'batch'`
- [OT D2·6교시 조용히 틀리는 export](#/ot/2/6) — `view(1, -1)` 이 dynamic 을 깨뜨림
- [VB D5·5교시 ONNX export · ONNX Runtime](#/vb/5/5) — 같은 `dynamic_axes`
- [VB D5·6교시 성능 측정법](#/vb/5/6) — 배치 효과를 정확히 재는 법 (OT Day 3 에서 이어짐)

**상세 학습**

#### 이 교시에서 답하는 질문
현장에서는 사진이 1장씩 올 때도, 한꺼번에 100장 올 때도 있다. 고정 shape 파일은 한 가지 배치 크기만 받는다. **내보낼 때 어느 축이 바뀔 수 있는지** 알려 주는 것이 `dynamic_axes`.

#### 개념 1 · 고정 vs 동적
| | 고정 | dynamic_axes |
|---|---|---|
| 입력 shape | `[1, 1, 28, 28]` | `['batch', 1, 28, 28]` |
| 출력 shape | `[1, 10]` | `['batch', 10]` |
| 배치 4 | 오류 `Got: 4 Expected: 1` | `(4, 10)` |

#### 개념 2 · 열어도 되는 축 / 안 되는 축
| 축 | 이 모델에서 |
|---|---|
| 배치 (0) | 열어도 안전 — 모든 연산이 배치별로 독립 |
| 높이 · 너비 (2, 3) | 실행은 시작하지만 Flatten → Gemm 에서 4320 이 안 맞아 실패 |

#### 코드로 확인
```python
sess_dyn = ort.InferenceSession("mnist_cnn_dyn.onnx", providers=["CPUExecutionProvider"])
print("입력", sess_dyn.get_inputs()[0].shape, "| 출력", sess_dyn.get_outputs()[0].shape)
for b in [1, 4, 8]:
    print("배치", b, "→", sess_dyn.run(None, {"input": np.random.rand(b, 1, 28, 28).astype(np.float32)})[0].shape)
```
- **무엇을 하나**: 열린 배치 축에 1 · 4 · 8장을 넣어 본다
- **어떤 개념**: 개념 1 — 동적 shape
- **출력**: `입력 ['batch', 1, 28, 28] | 출력 ['batch', 10]` · `(1, 10)` `(4, 10)` `(8, 10)`

#### 한 번에 정리
- `dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}` → 첫 축이 이름 `'batch'`
- 여러 장은 한 번에 `run` (`argmax(1)` 로 사진마다)
- ⚠ dynamic = 약속 — 모델이 감당 못 하면 실행 중 오류 (Flatten → Gemm)

---

### 6교시 · 조용히 틀리는 export 3가지: if · NumPy · 고정 배치

**키워드**: `tracing`, `TracerWarning`, `torch.where`, `x.mean(dim=…)`, `flatten(1)`, `verify()`, `export_warn()`

**개요**: 지금까지 export 는 늘 성공했고 결과도 맞았다. 그런데 **export 가 성공했는데 결과가 틀리는** 경우가 있다 — 오류 메시지가 없어 가장 위험. 원인은 하나: export 는 **더미 입력으로 실제 실행된 길만 녹화**한다(Day 1 2교시). 그래서 ① 파이썬 `if` ② NumPy · `.item()` · `float()` 로 꺼낸 값 ③ 숫자로 쓴 배치 크기 가 **그 순간의 결과로 굳는다**. 확인은 `verify()` (여러 입력으로 비교) + `export_warn()` (TracerWarning 만 보기).

**핵심 용어**
- **TracerWarning**: tracing 이 "이 값은 굳을 수 있다" 고 알리는 경고 — 이 교시의 3가지가 전부 경고 한 줄로만 알려 줌
- **verify()**: `{이름: 입력}` 여러 경우에서 PyTorch vs ONNX 최대 차이를 표로 (✅/❌)
- **export_warn()**: export 하며 `TracerWarning` 만 골라 출력 (환경 셀이 경고를 숨겼기 때문)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `verify()` | `verify(model, "x.onnx", {"음수": x_neg, …})` | 여러 입력으로 PyTorch · ONNX 비교 |
| `export_warn()` | `export_warn(model, dummy, "x.onnx", input_names=["input"], …)` | export + TracerWarning 출력 |
| `torch.where()` | `torch.where(x.mean() > 0, x * 2, x - 1)` | 조건 분기를 그래프 안에 (`Where` 노드) |
| `Tensor.mean()` | `x.mean(dim=1, keepdim=True)` | 평균을 torch 연산으로 (`ReduceMean` 노드) |
| `Tensor.flatten()` | `x.flatten(1)` | 배치 축은 두고 나머지만 펴기 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| ① Gate | `if x.mean() > 0` → 양수 입력 ✅ · 음수 입력 ❌ (최대 차이 2.00) |
| 6-1 | `torch.where` 로 고침 → 둘 다 ✅, 그래프에 `Where` |
| ② Center | `float(np.mean(…))` → 더미 `[5,5,5]` ✅ · `[1,2,3]` ❌ (3.00) |
| 6-2 | `x - x.mean(dim=1, keepdim=True)` → 둘 다 ✅, `ReduceMean` |
| ③ Hard | `x.view(1, -1)` → 배치 4 실패 |
| 6-3 | `x.flatten(1)` → 배치 1 · 4 모두 ✅ (최대 차이 ≈ 6e-08) |

**핵심 내용**

#### ① 파이썬 if — 한쪽 길만 녹화된다
> 더미 입력이 지나간 쪽 분기만 그래프에 남고, `if` 자체는 사라진다
- **Gate** → `if x.mean() > 0: x * 2 else: x - 1`, 더미 `ones` (양수) → 그래프는 `['Constant', 'Mul']` (곱하기만)
- **결과** → 양수 `[1,2,3]` 최대 차이 0 ✅ / 음수 `[-3,-2,-1]` 최대 차이 **2.00** ❌ (ONNX 는 음수에도 ×2)
- **고치기** → `torch.where(x.mean() > 0, x * 2, x - 1)` → 두 길을 모두 계산해 고름 (`ReduceMean · Greater · Mul · Sub · Where`)
- ⚠ `if self.training:` 처럼 **모델 설정**에 따른 분기는 괜찮음 (입력과 무관하게 항상 같은 길). 문제는 **입력 값**에 따른 분기

#### ② NumPy · .item() · float() — 그때의 값이 상수로 굳는다
> torch 밖으로 나가 계산한 값은 그래프에 연결이 없어 **더미 입력일 때의 숫자**가 박힘
- **Center** → `float(np.mean(x.detach().numpy()))` 후 `x - m`, 더미 5 → 그래프 `['Constant', 'Sub']` (`x - 5`)
- **결과** → `[5,5,5]` ✅ / `[1,2,3]` 최대 차이 **3.00** ❌ (진짜는 평균 2 를 빼야 하는데 5 를 뺌)
- **고치기** → `x - x.mean(dim=1, keepdim=True)` (`ReduceMean` 노드 — 매번 계산)
- ⚠ `.item()` · `float()` · `.numpy()` · `int()` 가 보이면 의심

#### ③ 배치 크기를 숫자로 고정
> `x.view(1, -1)` 의 `1` 이 그래프에 숫자로 박힘 — `dynamic_axes` 를 줘도 실행 중 오류
- **Hard** → `self.fc(x.view(1, -1))`, 입력 `(N, 4, 4)` → `(1, 16N)`, 그래프 `['Constant', 'Reshape', 'Gemm']`
- **결과** → 배치 1 ✅ (≈ 6e-08) / 배치 4 ❌ (PyTorch 쪽도 `mat1 and mat2 shapes cannot be multiplied (1x64 and 16x2)`)
- **고치기** → `x.flatten(1)` 또는 `x.view(x.size(0), -1)` → `['Flatten', 'Gemm']`, 배치 4 → `(4, 2)` ✅
- ⚠ 모양을 숫자 리터럴로 쓰기 전에 "배치가 바뀌어도 맞나?" 확인

#### 확인하는 법 — 한 입력으로는 늘 ✅
> `verify(model, path, cases)` 에 **일부러 다른 경우** (음수 · 어두운 사진 · 배치 4)를 넣기
- **더미 한 개로만 비교하면** 녹화된 길과 같은 입력이라 항상 ✅
- **TracerWarning** → `Converting a tensor to a Python boolean …` (①), `Converting a tensor to a NumPy array …` (②) — "경고일 뿐" 하고 넘기지 말 것
- ⚠ 확인 기준은 PyTorch 쪽의 **의도**(여러 경우에서 맞는 값)

**기본 사용형태**
```python
export_warn(gate, torch.ones(1, 3), "gate.onnx", input_names=["input"], output_names=["output"])
cases = {"양수": np.array([[1., 2., 3.]], np.float32), "음수": np.array([[-3., -2., -1.]], np.float32)}
verify(gate, "gate.onnx", cases)          # ✅ 양수 / ❌ 음수 (최대 차이 2.00)
```

**연결**
- [OT D1·2교시 첫 ONNX 만들기](#/ot/1/2) — "더미 입력으로 한 번 실행하며 녹화"의 결과
- [OT D2·5교시 dynamic_axes](#/ot/2/5) — ③ 이 dynamic 을 무력화
- [OT D2·8교시 미니 프로젝트](#/ot/2/8) — 세 가지가 한 모델에 모두 들어 있음
- [NN D3·5교시 Affine · Softmax-with-Loss](#/nn/3/5) — 배치 축을 유지하는 `flatten` 의 의미

**상세 학습**

#### 이 교시에서 답하는 질문
오류가 나는 건 차라리 쉽다. 가장 위험한 건 **export 도 성공하고 더미 입력 비교도 ✅ 인데 다른 입력에서 틀리는** 파일이다. 왜 그런지, 어떻게 찾고 어떻게 고치는지.

#### 개념 1 · 녹화되는 길 vs 굳는 값
| 코드 | 녹화되는 것 | 굳는 것 | 고치는 법 |
|---|---|---|---|
| `if x.mean() > 0` | 더미가 지난 분기 | 분기 선택 | `torch.where` |
| `float(np.mean(…))` | (그래프 밖) | 더미 때의 평균 숫자 | `x.mean(dim=…)` |
| `x.view(1, -1)` | `Reshape` 의 `1` | 배치 크기 | `flatten(1)` |

#### 개념 2 · 결과 표 (직접 실행)
| 모델 | 입력 | 최대 차이 | 판정 |
|---|---|---|---|
| Gate | 양수 | 0 | ✅ |
| Gate | 음수 | 2.00 | ❌ |
| GateFix | 양수 · 음수 | 0 | ✅ |
| Center | 더미 `[5,5,5]` | 0 | ✅ |
| Center | `[1,2,3]` | 3.00 | ❌ |
| CenterFix | 둘 다 | 0 | ✅ |
| Hard | 배치 1 / 4 | ≈ 6e-08 / 실패 | ✅ / ❌ |
| HardFix | 배치 1 / 4 | ≈ 6e-08 / ≈ 6e-08 | ✅ / ✅ |

#### 코드로 확인
```python
class GateFix(nn.Module):
    def forward(self, x):
        return torch.where(x.mean() > 0, x * 2, x - 1)

export_warn(GateFix(), torch.ones(1, 3), "gate_fix.onnx", input_names=["input"], output_names=["output"])
ok_gate = verify(GateFix(), "gate_fix.onnx", cases_gate)
```
- **무엇을 하나**: `if` 를 `torch.where` 로 바꾸고 두 경우 비교
- **어떤 개념**: 개념 1 — 두 길 모두 그래프에
- **출력**: `['ReduceMean', 'Constant', 'Greater', 'Constant', 'Mul', 'Constant', 'Sub', 'Where']` · 양수 · 음수 모두 ✅ → `True`

#### 한 번에 정리
- export = 더미 입력으로 실행된 길만 녹화 → if · NumPy 값 · 숫자 크기는 굳는다
- 고치기: `torch.where` · torch 연산(`mean` 등) · `flatten(1)` / `view(x.size(0), -1)`
- ⚠ 더미 하나로만 `verify` 하면 늘 ✅ — 음수 · 어두운 사진 · 배치 4 를 넣어라

---

### 7교시 · 미지원 연산 고치기 · onnxslim 단순화

**키워드**: `UnsupportedOperatorError`, `aten::median`, `torch.sort`, `onnxslim.slim`, `상수 접기(constant folding)`, `do_constant_folding`

**개요**: 6교시는 **조용히** 틀렸다. 이번엔 반대로 export 가 **오류를 내며 멈추는** 경우 — 메시지를 읽고 바꾸면 되니 오히려 쉽다. `torch.median` 은 ONNX 에 연산이 없어 `aten::median` 오류 → 정렬 후 가운데 고르기(`torch.sort`) 로 대체. 이어서 `onnxslim` 으로 "계산하면 늘 같은 값"인 노드들을 미리 계산해 접어 **11개 노드 → 4개**.

**핵심 용어**
- **미지원 연산**: PyTorch 에는 있는데 ONNX opset 에 대응 연산이 없는 것 (`aten::연산 이름`)
- **onnxslim**: 그래프를 단순화하는 도구 — 상수 접기 · 불필요 노드 제거
- **constant folding**: 입력과 무관하게 늘 같은 값이 나오는 계산을 미리 해 둠 (`do_constant_folding=True` 가 export 기본값)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `onnxslim.slim()` | `slim_model = onnxslim.slim("viewnet.onnx")` | 그래프 단순화 (모델 객체 반환) |
| `onnx.save()` | `onnx.save(slim_model, "viewnet_slim.onnx")` | 모델 객체를 파일로 |
| `torch.sort()` | `torch.sort(x, dim=1).values[:, k:k + 1]` | 정렬 후 가운데 고르기 (median 대체) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 7-1 | export 오류 메시지에서 연산 이름 `err_op = "aten::median"` 읽기 |
| 7-2 | `torch.sort` 로 대체 → 원래 모델과 `verify` 모두 ✅ (그래프에 `TopK`) |
| 7-3 | ViewNet `onnxslim.slim` → 11개 → 4개 `['Conv', 'Relu', 'Reshape', 'Gemm']`, 전후 최대 차이 0 |
| 바꿔보기 | `do_constant_folding=False` 로 내보낸 뒤 slim — 내 환경 16개 → 4개 |

**핵심 내용**

#### 미지원 연산 — 메시지 읽고 바꾸기
> 오류 메시지: `Exporting the operator 'aten::median' to ONNX opset version N is not supported`
- **읽는 법** → 따옴표 안이 **PyTorch 연산 이름**(`aten::median`), N 이 시도한 opset
- **고치는 두 길** → ① `opset_version` 을 올려 보기 (4교시) ② 최신 판에도 없으면 **지원되는 연산의 조합**으로 바꾸기
- **median 예** → `torch.median(x, dim=1)` 대신 `torch.sort(x, dim=1).values[:, k:k + 1]` (`k = x.shape[1] // 2`, 5개면 가운데 3번째) — 그래프에선 `TopK`
- **검증** → 원래 모델(`med_net`)과 비교: 배치 1 · 배치 6(튀는 값 포함) 모두 ✅ (≈ 3e-08)
- ⚠ 구조를 바꿨으면 **반드시 원래 모델과 `verify`** — 같은 가중치를 `load_state_dict` 로 복사

#### onnxslim — 같은 계산, 더 적은 노드
> `slim(path)` → 모델 객체 → `onnx.save` 로 저장해야 파일이 됨
- **ViewNet** (`x.view(x.size(0), -1)` — 배치 크기를 매번 읽어 shape 를 조립) → 11개 `['Conv', 'Relu', 'Shape', 'Constant', 'Gather', 'Constant', 'Unsqueeze', 'Constant', 'Concat', 'Reshape', 'Gemm']`
- **slim 후** → 4개 `['Conv', 'Relu', 'Reshape', 'Gemm']`, `Shape` 없음, 전후 결과 최대 차이 **0**
- **BN 은?** → eval export 가 이미 Conv 에 합쳐 둠 (원래 BatchNorm 이 있던 모델)
- **접기 끄기** → `do_constant_folding=False` 면 내 환경 16개(`Identity` · `BatchNormalization` 등 남음), slim 하면 다시 4개
- ⚠ `slim(...)` 결과를 저장하지 않고 원래 파일을 다시 쓰는 실수 · 단순화 ≠ 정확도 향상 (같은 계산을 짧게 적을 뿐)

**기본 사용형태**
```python
slim_model = onnxslim.slim("viewnet.onnx")
onnx.save(slim_model, "viewnet_slim.onnx")
before = [n.op_type for n in onnx.load("viewnet.onnx").graph.node]
after  = [n.op_type for n in onnx.load("viewnet_slim.onnx").graph.node]
print(f"{len(before)}개 → {len(after)}개 :", after)      # 11개 → 4개
```

**연결**
- [OT D2·4교시 opset · checker · shape_inference](#/ot/2/4) — opset 을 올려서 풀리는 경우
- [OT D2·3교시 describe_onnx](#/ot/2/3) — 전후 노드 표 비교
- [OT D2·6교시 조용히 틀리는 export](#/ot/2/6) — 구조를 바꾼 뒤 `verify` 하는 습관
- [OT D1·8교시 미니 프로젝트 (BN 융합)](#/ot/1/8) — 가중치 노드가 합쳐진 사례

**상세 학습**

#### 이 교시에서 답하는 질문
6교시와 달리 이번 문제는 export 가 소리 내어 실패한다. 메시지에서 **무엇이 막혔는지** 읽는 법, 막힌 연산을 **우회하는 법**, 마지막에 파일을 **가볍게 다듬는 법**.

#### 개념 1 · 오류 메시지 읽기
| 부분 | 뜻 |
|---|---|
| `UnsupportedOperatorError` | 지원 안 되는 연산 |
| `'aten::median'` | PyTorch 연산 이름 (이걸 대체해야 함) |
| `opset version N` | 시도한 판 — 더 높은 판에 있는지 확인 |

#### 개념 2 · onnxslim 전 / 후
| | 노드 수 | 노드 |
|---|---|---|
| 전 | 11 | Conv, Relu, **Shape, Constant, Gather, Constant, Unsqueeze, Constant, Concat**, Reshape, Gemm |
| 후 | 4 | Conv, Relu, Reshape, Gemm |

배치 크기를 읽어 shape 를 만드는 7개 노드가 접혀 `Reshape` 하나가 됨.

#### 코드로 확인
```python
x4 = np.random.rand(4, 1, 28, 28).astype(np.float32)
r0 = ort.InferenceSession("viewnet.onnx", providers=["CPUExecutionProvider"]).run(None, {"input": x4})[0]
r1 = ort.InferenceSession("viewnet_slim.onnx", providers=["CPUExecutionProvider"]).run(None, {"input": x4})[0]
print("단순화 전 · 후 최대 차이:", np.abs(r0 - r1).max())
```
- **무엇을 하나**: 단순화 전후 결과 비교 (배치 4)
- **어떤 개념**: 개념 2 — 결과는 그대로
- **출력**: `0.0`

#### 한 번에 정리
- 미지원 연산: 메시지의 `aten::연산` 확인 → opset 올리기 또는 지원 연산 조합 → 원래 모델과 `verify`
- `onnxslim.slim` → `onnx.save` · 11 → 4 · 결과 동일
- ⚠ 구조 바꾼 뒤 비교 안 하기 · slim 결과를 저장 안 하기

---

### 8교시 · 미니 프로젝트: 고장 난 BrightNet 고쳐 배포하기

**키워드**: `BrightNet`, `고장 3곳`, `torch.where · 평균 dim=(1,2,3) · flatten(1)`, `opset 17`, `brightnet_final.onnx`, `checker → onnxslim → shape_inference`

**개요**: PyTorch 에서는 잘 돌지만 ONNX 로 내보내면 이상한 현장 카메라용 분류기 **BrightNet**(64×64 컬러 0~1 → safe · warning · danger)을 고쳐 배포용 `brightnet_final.onnx` 를 만든다. 의도: ① 평균 밝기 < 0.3 이면 2배 ② 평균을 빼 보정 ③ CNN. 고장 3곳이 6교시의 세 가지와 똑같다 — `if` · `float(np.mean(…))` · `x.view(1, -1)`. 과제 6개(고장 찾기 10 · 고치기 30 · 배포용 export 20 · 점검·단순화 20 · 최종 확인 10 · 해석 10, 해석은 강사 채점). 정확도가 아니라 **배포 과정**이 과제 (가중치는 seed 연습용).

**핵심 용어**
- **intended()**: 사진을 **한 장씩** 넣은 PyTorch 결과를 쌓는 채점 기준 ("의도")
- **배포용 파일**: 배치 자유 · opset 17 · 이름 `input` / `logits` · 검사 · 단순화 · shape 채움까지 마친 최종 `.onnx`
- **인계 정보**: `.onnx` 와 함께 opset · 입력 shape · 전처리 규칙을 전달

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.where()` | `torch.where(mean < 0.3, x * 2, x)` | 사진마다 어두우면 2배 |
| `Tensor.mean()` | `x.mean(dim=(1, 2, 3), keepdim=True)` | 사진마다 평균 `(N, 1, 1, 1)` |
| `verify()` | `verify(bright, "brightnet_broken.onnx", cases_b)` | 고장 찾기 (6교시 도구) |
| `onnxslim.slim()` | `onnxslim.slim("brightnet_fixed.onnx")` | 단순화 |
| `onnx.shape_inference.infer_shapes()` | `infer_shapes(slim_b)` | 중간 shape 채움 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 과제 1 (8-1) | 그대로 export → `failed = ["다른 밝은 사진", "어두운 사진"]` |
| 과제 2 (8-2) | `BrightNetFixed` 로 고장 3곳 수정 → 의도와 최대 차이 ≈ 1.5e-08 |
| 과제 3 (8-3) | `brightnet_fixed.onnx` — 배치 자유 · opset 17 · `input`/`logits` |
| 과제 4 (8-4) | checker → slim → shape_inference → `brightnet_final.onnx` (노드 16 → 14, `value_info` 27) |
| 과제 5 (8-5) | 배치 1 · 배치 8 모두 의도와 같음 (≈ 1e-07), 분류 결과 |

**핵심 내용**

#### 고장 찾기 — 더미만 ✅
> 더미(`orange.jpg`, 밝기 0.459)로 export 한 파일을 세 경우로 `verify`
- **결과** → 더미 그대로 ✅ (≈ 1.5e-08) / 다른 밝은 사진(butterfly) ❌ **3.86e-03** / 어두운 사진 ❌ **6.87e-03**
- **이유** → ① 더미는 밝아서 `if` 의 **"2배 안 하는 길"** 만 녹화됨(어두운 사진이 와도 2배 안 함) ② 평균이 더미의 숫자로 굳음 → 다른 사진도 같은 값을 뺌
- **경고** → `TracerWarning` 2개 (Python boolean 변환 · NumPy 변환) — ③ `view(1, -1)` 은 경고 없이 숨어 있음 (배치 8 에서 실패)
- ⚠ 밝기 값: butterfly 0.402 · orange 0.459 · baboon 0.496, 0.25 배 어두운 사진 0.101 · 0.115 · 0.124

#### 고치기 3곳 — 6교시 방법 그대로
> `BrightNetFixed` — forward 만 다시 쓰고 `load_state_dict` 로 같은 가중치
- **① if → `torch.where`** → `mean = x.mean(dim=(1,2,3), keepdim=True)`, `x = torch.where(mean < 0.3, x * 2, x)`
- **② NumPy → torch 연산** → `x = x - x.mean(dim=(1,2,3), keepdim=True)`
- **③ `view(1, -1)` → `flatten(1)`**
- **왜 `dim=(1,2,3)`?** → `x.mean()` 은 배치 전체 평균 한 개 — 밝은 사진과 어두운 사진이 섞인 배치 8 에서 **사진마다** 판단하려면 사진별 평균이 필요 (채널 · 높이 · 너비만 줄이고 배치 축은 남김, `keepdim` 으로 브로드캐스트)
- **확인** → 배치 6 의 고친 모델 vs 의도(한 장씩) 최대 차이 ≈ 1.5e-08

#### 배포용 만들기 순서
> 고치기 → export(batch · opset · 이름) → checker → onnxslim → shape_inference → 최종 비교
- **export** → `dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}`, `opset_version=17`, `input_names=["input"]`, `output_names=["logits"]` (TracerWarning 없음)
- **원본 16노드** → `ReduceMean · Mul · Less · Where · ReduceMean · Sub · Conv · Relu · MaxPool · Conv · Relu · MaxPool · Flatten · Gemm` (Mul · Less 의 상수 2개 때문에 `params` 가 1씩 나옴)
- **최종** → `brightnet_final.onnx` 노드 14, `value_info` 27개, opset 17, checker 통과
- **최종 비교** → 배치 1 (사진 6장 각각) · 배치 8 (6장 + 앞 2장) 모두 의도와 최대 차이 ≈ 9.7e-08 → 같다
- **분류 결과** → butterfly `safe` · orange `danger` · baboon `danger` · 어두운 버전도 `safe` / `danger` / `danger`
- ⚠ 가중치는 연습용이라 분류 결과 자체에 의미는 없음 — **고친 파일이 의도와 같은가**가 평가

#### 인계할 때 적어 줄 정보
> `.onnx` 만 넘기지 말고 함께
- **opset** → 17 (배포 장치가 아는 판)
- **입력 shape** → `[batch, 3, 64, 64]` float32, 값 범위 0 ~ 1, 이름 `input` / `logits`
- **전처리** → RGB · 64×64 resize · `/255` · CHW · 클래스 이름 `safe · warning · danger`
- ⚠ checker · shape_inference · onnxslim 은 **형식 · shape · 단순화** 만 해 주고 **결과가 맞는지는 보장 못 함** → 마지막은 `allclose`

**기본 사용형태**
```python
class BrightNetFixed(BrightNet):
    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)          # (N, 1, 1, 1)
        x = torch.where(mean < 0.3, x * 2, x)
        x = x - x.mean(dim=(1, 2, 3), keepdim=True)
        return self.fc(self.features(x).flatten(1))

export_warn(fixed, dummy_b, "brightnet_fixed.onnx", input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
final = onnx.shape_inference.infer_shapes(onnxslim.slim("brightnet_fixed.onnx"))
onnx.save(final, "brightnet_final.onnx"); onnx.checker.check_model(onnx.load("brightnet_final.onnx"))
```

**연결**
- [OT D1·8교시 미니 프로젝트 (SafetyNet)](#/ot/1/8) — 어제의 4단계 프로젝트
- [OT D2·6교시 조용히 틀리는 export](#/ot/2/6) — 고장 3곳의 원리
- [OT D2·5교시 dynamic_axes](#/ot/2/5) — 배치 자유
- [OT D2·4교시 opset · checker · shape_inference](#/ot/2/4) — 점검 도구
- [OT D2·7교시 onnxslim](#/ot/2/7) — 단순화

**상세 학습**

#### 이 교시에서 답하는 질문
오늘 배운 것을 한 모델에 모두 쓴다. "PyTorch 에서는 되는데 ONNX 로는 이상하다" 를 **재현 → 원인 → 수정 → 배포용 파일 → 최종 검증**의 순서로 끝까지 해 보고, 배포 담당자에게 무엇을 넘길지 정리한다.

#### 개념 1 · 고장 3곳 대응표
| 고장 | BrightNet 원래 코드 | 증상 | 수정 |
|---|---|---|---|
| ① 입력 값 분기 | `if x.mean() < 0.3: x = x * 2` | 어두운 사진도 ×2 안 함 | `torch.where(mean < 0.3, x * 2, x)` |
| ② NumPy 값 | `m = float(np.mean(x.detach().numpy()))` | 더미의 평균이 굳음 | `x.mean(dim=(1,2,3), keepdim=True)` |
| ③ 고정 배치 | `x.view(1, -1)` | 배치 8 실패 | `x.flatten(1)` |

#### 개념 2 · 배포 파일 체크리스트
| 단계 | 도구 | 확인 값 |
|---|---|---|
| export | `export_warn` | TracerWarning 없음, 16 노드 |
| 검사 | `checker` | 통과 |
| 단순화 | `onnxslim.slim` | 16 → 14 노드 |
| shape 채움 | `infer_shapes` | `value_info` 27개 |
| 최종 비교 | `ort` vs `intended` | 배치 1 · 8 모두 ≈ 1e-07 |

#### 코드로 확인
```python
dummy_b = torch.from_numpy(photos["orange.jpg"])
export_warn(bright, dummy_b, "brightnet_broken.onnx", input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
cases_b = {"더미 그대로": photos["orange.jpg"], "다른 밝은 사진": photos["butterfly.jpg"], "어두운 사진": dark["butterfly_dark"]}
verify(bright, "brightnet_broken.onnx", cases_b)
```
- **무엇을 하나**: 고장 난 모델을 세 경우로 비교
- **어떤 개념**: 개념 1 — 더미만 ✅
- **출력**: ✅ 더미(≈ 1.5e-08) · ❌ 다른 밝은 사진 `3.86e-03` · ❌ 어두운 사진 `6.87e-03`

#### 한 번에 정리
- 고장 찾기는 여러 경우(더미 · 다른 사진 · 어두운 사진 · 배치)로 · 기준은 "원래 코드"가 아니라 **의도**(한 장씩)
- 배포 파일: 고치기 → export(batch · opset · 이름) → checker → onnxslim → shape_inference → 최종 비교
- 인계: `.onnx` + opset + 입력 shape + 전처리 규칙
- ⚠ checker · slim · shape_inference 통과 ≠ 결과가 맞음
