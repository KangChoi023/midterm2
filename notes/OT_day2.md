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
Day 1 은 "내보내고 · 보고 · 실행" 까지였다. 보는 일을 Netron 그림에만 맡기면 모델이 커질 때 확인할 방법이 없고, 어제는 배치 4 를 넣자 오류가 났다. 오늘은 파일 속을 **코드로** 읽고, export 가 실패하거나 **조용히 틀릴 때** 스스로 고치는 법을 배운다. 먼저 어제의 4단계를 혼자 다시 해 보며 오늘 확인할 숫자(노드 7 · 가중치 텐서 6)를 예측해 둔다.

#### 개념 1 · 4단계 다시
| 단계 | mnist_cnn 에서 | 확인 |
|---|---|---|
| 0 실행 모드 | `cnn.eval()` | `not cnn.training` |
| 1 내보내기 | 더미 `torch.zeros(1, 1, 28, 28)`, 이름 input / logits | `onnx_report` 입력 `[1, 1, 28, 28]` |
| 2 열어보기 | 오늘은 **코드로** (2 · 3교시) | 노드 7 · initializer 6 |
| 3 실행 | `sess.run(None, {"input": x0})` | 예측 7 = 정답 7 |
| 4 확인 | `np.allclose(y_ort, y_pt, atol=1e-5)` | True |

가중치 텐서 6개 = Conv · Gemm · Gemm 이 각각 weight + bias 를 하나씩 가진다 (3 × 2).

#### 개념 2 · 오늘의 지도 — 어제 막혔던 것 4가지
| 어제 막힌 것 | 오늘 푸는 교시 | 도구 |
|---|---|---|
| 파일 속이 어떻게 생겼나 | 2 · 3교시 | `graph.input · node · initializer · output`, `describe_onnx` |
| 연산 사전 버전 · 고장 난 파일 | 4교시 | opset · checker · shape_inference |
| 배치 4 를 넣으면 오류 | 5교시 | `dynamic_axes` |
| 오류 없이 결과가 틀림 · export 실패 | 6 · 7교시 | `verify` · `torch.where` · onnxslim |

#### 코드로 확인
```python
cnn.eval()
torch.onnx.export(cnn, torch.zeros(1, 1, 28, 28), "mnist_cnn.onnx",
                  input_names=["input"], output_names=["logits"], **EXPORT_KW)
sess_cnn = ort.InferenceSession("mnist_cnn.onnx", providers=["CPUExecutionProvider"])
x0 = (digits[0].astype(np.float32) / 255)[None, None]
y_ort = sess_cnn.run(None, {"input": x0})[0]
with torch.no_grad():
    y_pt = cnn(torch.from_numpy(x0)).numpy()
print("예측", y_ort.argmax(), "| 정답", labels[0], "| 같다:", np.allclose(y_ort, y_pt, atol=1e-5))
```
- **무엇을 하나**: 어제의 4단계를 한 셀에.
- **어떤 개념**: 개념 1.
- **출력**: `예측 7 | 정답 7 | 같다: True`, 예측한 노드 7 · 가중치 텐서 6 도 맞음 (직접 실행)

#### 한 번에 정리
- 4단계: eval → export(이름 · shape) → 열어보기 → run · allclose.
- mnist_cnn.onnx = 노드 7개 · 가중치 텐서 6개 — 오늘 확인 셀의 기준 숫자.
- 오늘은 "열어보기" 를 코드로, 그리고 export 오류 고치기.
- ⚠ 4단계의 0번(eval)을 빼면 BN · Dropout 모델은 비교 기준부터 틀어진다.

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
Netron 의 그림은 파일 안에 어떤 모양으로 들어 있을까? `onnx.load` 로 읽은 객체를 열어 보면 **상자 안의 상자** 구조이고, 화살표는 따로 저장되어 있지 않다. 이 교시는 네 가지 목록을 꺼내고, **이름을 따라** 화살표를 직접 이어 본다.

#### 개념 1 · 상자 안의 상자
![ONNX 파일의 구조 — ModelProto 안의 GraphProto, 그 안의 input · node · initializer · output 네 목록, 이름이 같으면 이어지는 화살표](assets/fig/ot/d2-graph.svg)

| 상자 | 꺼내는 법 | mnist_cnn.onnx |
|---|---|---|
| 모델 (ModelProto) | `m = onnx.load(path)` | `ir_version` (파일 형식 버전) · `opset_import[0].version` = 17 · `producer_name` = pytorch |
| 그래프 (GraphProto) | `g = m.graph` | 아래 네 목록 |
| `g.input` | 바깥에서 넣는 것 | 1개 — `input [1, 1, 28, 28]` |
| `g.node` | 연산 | 7개 |
| `g.initializer` | 가중치 | 6개 |
| `g.output` | 결과 | 1개 — `logits [1, 10]` |

shape 는 축마다 `dim_value` (숫자) 또는 `dim_param` (이름 — 5교시의 `'batch'`) 중 하나에 들어 있다 → `shape_of(vi)` 는 숫자가 있으면 숫자, 없으면 이름.

#### 개념 2 · 화살표 = 이름 맞추기
노드는 **op_type + input 이름 목록 + output 이름 목록**만 가진다. 앞 노드의 output 이름 = 뒤 노드의 input 이름이면 화살표다.

| 노드 | 데이터 입력 | 가중치 입력 (initializer) |
|---|---|---|
| Conv | `input` | `conv.weight` · `conv.bias` |
| Relu | `/conv/Conv_output_0` | — |
| MaxPool | `/Relu_output_0` | — |
| Flatten | `/pool/MaxPool_output_0` | — |
| Gemm | `/Flatten_output_0` | `fc1.weight` · `fc1.bias` |

모든 노드는 데이터 화살표 1개 + 가중치 0 또는 2개. 가중치는 `input` 이 아니라 `initializer` 에 있으니, **바깥에서 넣는 것은 사진뿐**이다.

#### 코드로 확인
```python
n_in, n_out, n_node, n_init = len(g.input), len(g.output), len(g.node), len(g.initializer)
def shape_of(vi):
    return [d.dim_value if d.dim_value else d.dim_param for d in vi.type.tensor_type.shape.dim]
for vi in list(g.input) + list(g.output):
    print(vi.name, shape_of(vi))
```
- **무엇을 하나**: 네 목록의 길이와 입출력 shape.
- **어떤 개념**: 개념 1.
- **출력**: `input 1 · output 1 · node 7 · initializer 6`, `input [1, 1, 28, 28]`, `logits [1, 10]` (직접 실행)

```python
name = g.input[0].name
path = []
while name != g.output[0].name:
    node = next(n for n in g.node if name in n.input)       # 이 이름을 받아 가는 노드
    path.append(node.op_type)
    name = node.output[0]                                    # 다음 화살표 이름
print(" → ".join(path))
```
- **무엇을 하나**: 입력 이름에서 출발해 이름을 따라 출력까지 걸어간다.
- **어떤 개념**: 개념 2.
- **출력**: `Conv → Relu → MaxPool → Flatten → Gemm → Relu → Gemm` — 노드 순서와 같다 (직접 실행)

#### 한 번에 정리
- `onnx.load(path).graph` 안에 input · node · initializer · output 네 목록.
- 노드 = op_type + input 이름들 + output 이름들. 이름이 같으면 화살표.
- 가중치는 initializer — 바깥에서 넣는 것은 사진뿐.
- ⚠ `g.inputs` · `g.nodes` (복수형) 는 AttributeError, `print(m)` 은 가중치까지 찍혀 화면이 멈춘다.

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
2교시에서 이름까지 읽었다. 이번엔 이름 뒤의 **숫자(가중치)** 와 노드의 **설정값(속성 · attribute)** 을 꺼내고, 모든 것을 표 한 장으로 요약하는 `describe_onnx` 를 만든다. 이 표로 SafetyNet 을 보면 ONNX 의 가중치 수가 PyTorch 파라미터 수와 **다르다** — 왜일까?

#### 개념 1 · 가중치와 속성 꺼내기
| 꺼낼 것 | 도구 | 예 |
|---|---|---|
| 가중치 | `onnx.numpy_helper.to_array(initializer)` | `fc1.weight (100, 4320) float32`, 합계 433,890 |
| 속성 | `onnx.helper.get_attribute_value(attribute)` | Conv `kernel_shape [5, 5]` · `strides [1, 1]` · `pads [0, 0, 0, 0]` |
| | | MaxPool `kernel_shape [2, 2]` · `strides [2, 2]` |
| | | Flatten `axis 1` |

PyTorch 코드 `nn.Conv2d(1, 30, 5)` · `nn.MaxPool2d(2)` 와 그대로 대응한다. 개수는 `.shape` 가 아니라 `.size` (모든 축 곱).

#### 개념 2 · describe_onnx — 코드로 만든 Netron
| op_type | data_in | params | attrs |
|---|---|---|---|
| Conv | input | 780 | kernel 5×5 |
| Relu | … | 0 | — |
| MaxPool | … | 0 | kernel 2×2 · stride 2 |
| Flatten | … | 0 | axis 1 |
| Gemm | … | 432,100 | transB 1 |
| Relu | … | 0 | — |
| Gemm | … | 1,010 | transB 1 |

노드 7 · 가중치 합계 **433,890** = Day 1 의 파라미터 수. 오늘부터 모든 모델은 이 표로 먼저 본다 (8교시 · Day 3 이후).

#### 개념 3 · SafetyNet 의 "사라진 48개"
| | 개수 |
|---|---|
| PyTorch 파라미터 | 1,491 |
| ONNX 가중치 합계 | 1,443 |
| 차이 | **48** = BN 의 γ · β (8 + 8 + 16 + 16) |

Day 1 에서 본 Conv-BN 합치기 때문에 BN 의 숫자가 Conv 의 weight · bias 안으로 녹아 들어가 따로 세어지지 않는다. running_mean · var 는 원래 학습 파라미터가 아니라 PyTorch 쪽에서도 세지 않는다.

#### 코드로 확인
```python
W = {t.name: onnx.numpy_helper.to_array(t) for t in g.initializer}
print(sum(a.size for a in W.values()), np.allclose(W["conv.weight"], cnn.conv.weight.detach().numpy()))
def attrs_of(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
for n in g.node:
    print(n.op_type.ljust(8), attrs_of(n))
```
- **무엇을 하나**: 가중치 합계 · PyTorch 와 같은지, 노드마다 속성.
- **어떤 개념**: 개념 1.
- **출력**: `433890 True`, `Conv {'dilations': [1, 1], 'group': 1, 'kernel_shape': [5, 5], 'pads': [0, 0, 0, 0], 'strides': [1, 1]}`, `MaxPool {… 'kernel_shape': [2, 2], … 'strides': [2, 2]}`, `Flatten {'axis': 1}` (직접 실행)

```python
def describe_onnx(path):
    gr = onnx.load(path).graph
    W_ = {t.name: onnx.numpy_helper.to_array(t) for t in gr.initializer}
    rows = [{"op_type": n.op_type,
             "data_in": [x for x in n.input if x and x not in W_],
             "out": list(n.output),
             "params": sum(W_[x].size for x in n.input if x in W_),
             "attrs": attrs_of(n)} for n in gr.node]
    return pd.DataFrame(rows)
```
- **무엇을 하나**: 노드마다 종류 · 입출력 · 가중치 수 · 속성을 한 줄씩 표로.
- **어떤 개념**: 개념 2.
- **출력**: 7행, `params = [780, 0, 0, 0, 432100, 0, 1010]` (직접 실행)

#### 한 번에 정리
- 가중치 = `numpy_helper.to_array(initializer)`, 속성 = `helper.get_attribute_value(attribute)`.
- `describe_onnx(path)` = Netron 을 코드로 — 노드마다 종류 · 입출력 · 가중치 수 · 속성.
- ONNX 가중치 수 ≠ PyTorch 파라미터 수일 수 있다 (SafetyNet 1,491 vs 1,443 — Conv-BN 합치기).
- ⚠ `a.ints` · `a.i` 를 직접 읽지 말고 도우미 함수로 — 속성 종류마다 필드가 다르다.

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
ONNX 파일을 다룰 때 쓰는 세 가지 도구 — **opset**(연산 사전의 판 번호) · **checker**(맞춤법 검사기) · **shape_inference**(도면에 치수 적기) 를 하나씩 써 본다. 같은 GELU 도 판에 따라 노드가 9개 또는 2개가 되고, 일부러 망가뜨린 파일을 checker 가 잡아내는 것을 본다.

#### 개념 1 · 세 도구 비교
![opset · checker · shape_inference 세 도구가 하는 일과 실습 결과](assets/fig/ot/d2-tools.svg)

| 도구 | 하는 일 | 비유 | 하지 않는 것 |
|---|---|---|---|
| opset | 파일이 쓰는 연산 사전의 판 번호 (`opset_version=`) | 사전 몇 판으로 썼나 | — |
| checker | 규칙에 맞게 만들어졌나 (끊긴 화살표 · 순서) | 맞춤법 검사기 | 결과가 맞는지 |
| shape_inference | 중간 화살표마다 shape 계산해 `value_info` 에 적기 | 도면에 치수 적기 | 결과가 맞는지 |

#### 개념 2 · 같은 GELU, 다른 모습
| opset | 노드 | 개수 |
|---|---|---|
| 13 | Gemm · Constant · Div · **Erf** · Constant · Add · Mul · Constant · Mul | 9 |
| 20 | Gemm · **Gelu** | 2 |

GELU(x) = 0.5 · x · (1 + erf(x / √2)) — opset 13 에는 Gelu 라는 단어가 없어 Erf 로 풀어 쓴다. 두 파일의 결과 차이는 0. 판 번호를 너무 높이면 배포할 실행기가 모르는 연산이 생긴다 (`opset … is not supported`) → 실행기가 아는 판부터 확인. 강의 기본값은 17.

#### 개념 3 · checker 와 shape_inference
- **checker**: Relu 의 입력 이름을 존재하지 않는 `"nope"` 로 바꾸면 화살표가 끊긴다 → checker 가 오류, ORT 도 세션 생성 실패. 앞 노드(Conv)의 output 이름 `/conv/Conv_output_0` 으로 되돌리면 통과 · 결과도 원래대로.
- **shape_inference**: `value_info` 0개 → **6개**. 결과는 **새 모델**(`mi`) 에 있다.

| 중간 이름 | shape |
|---|---|
| /conv/Conv_output_0 | [1, 30, 24, 24] |
| /pool/MaxPool_output_0 | [1, 30, 12, 12] |
| /Flatten_output_0 | [1, 4320] |
| /fc1/Gemm_output_0 | [1, 100] |

#### 코드로 확인
```python
gelu_net = nn.Sequential(nn.Linear(4, 4), nn.GELU()).eval()
for v in [13, 20]:
    torch.onnx.export(gelu_net, torch.randn(1, 4), f"gelu_op{v}.onnx", opset_version=v, dynamo=False)
    ops = [n.op_type for n in onnx.load(f"gelu_op{v}.onnx").graph.node]
    print(f"opset {v}: {len(ops)}개", ops)
```
- **무엇을 하나**: 같은 모델을 opset 13 · 20 으로 내보내 노드 비교.
- **어떤 개념**: 개념 2.
- **출력**: `opset 13: 9개 ['Gemm', 'Constant', 'Div', 'Erf', …]`, `opset 20: 2개 ['Gemm', 'Gelu']` (직접 실행)

```python
mi = onnx.shape_inference.infer_shapes(m)
shapes = {vi.name: shape_of(vi) for vi in mi.graph.value_info}
for k, v in shapes.items():
    print(" ", k.ljust(26), v)
```
- **무엇을 하나**: 중간 shape 를 채운 새 모델에서 value_info 를 읽는다.
- **어떤 개념**: 개념 3.
- **출력**: 추론 전 0 → 추론 후 6, `/pool/MaxPool_output_0 [1, 30, 12, 12]`, `/Flatten_output_0 [1, 4320]` (직접 실행)

#### 한 번에 정리
- opset = 연산 사전 판 번호 — 같은 GELU 도 9개 vs 2개.
- checker = 규칙 검사 (끊긴 화살표), shape_inference = 중간 shape 채우기 (24 → 12 → 4320).
- 셋 다 "결과가 맞는지" 는 말해 주지 않는다 → allclose.
- ⚠ `infer_shapes(m)` 의 결과는 새 모델 — `m.graph.value_info` 를 보면 여전히 0개.

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
어제 5교시: 고정 shape 파일에 배치 4 를 넣으면 `Got: 4 Expected: 1`. 오늘은 "이 축은 크기가 바뀔 수 있다" 고 export 때 알려 준다. 그러면 100장을 **한 번에** 넣을 수 있다. 그런데 높이 · 너비까지 열면 어떻게 될까?

#### 개념 1 · 고정 vs 동적
![dynamic_axes 로 배치 축 열기, 높이 · 너비를 열었을 때 Gemm 에서 실패하는 이유](assets/fig/ot/d2-dynamic.svg)

| | export 인자 | 입력 shape | 배치 4 |
|---|---|---|---|
| 고정 | (없음) | `[1, 1, 28, 28]` | ✘ |
| 동적 | `dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}` | `['batch', 1, 28, 28]` → `['batch', 10]` | ✔ (1 · 4 · 8 · 100) |

- 키는 `input_names` · `output_names` 와 **같은 이름**이어야 적용된다.
- `{0: "batch"}` 처럼 축 번호와 이름을 준다 (`[0]` 리스트도 되지만 이름이 자동으로 붙어 읽기 어렵다).
- 2교시의 `shape_of` 가 `dim_param` 을 읽는 이유가 이것 — 이름 붙은 축은 `dim_value` 가 0 이다.

#### 개념 2 · 1장씩 100번 vs 100장 1번
`X = (digits / 255)[:, None]` → `(100, 1, 28, 28)` 을 `run` 한 번에 넣으면 정확도 1.0, PyTorch 예측과 100장 모두 같다. 시간도 1장씩 100번보다 짧다 — 함수 호출 · 메모리 준비 같은 고정 비용을 한 번만 내기 때문 (Day 3 6교시에서 지연 vs 처리량으로 제대로 잰다).

#### 개념 3 · 열어도 되는 축 / 안 되는 축
`{0: "batch", 2: "height", 3: "width"}` 로 높이 · 너비까지 열면 입력 shape 는 `['batch', 1, 'height', 'width']` 가 된다. 하지만 **dynamic 은 약속일 뿐**:

| 단계 | 28×28 (학습 크기) | 32×32 |
|---|---|---|
| Conv 5×5 | 24×24 | 28×28 |
| MaxPool 2 | 12×12 | 14×14 |
| Flatten | 30 × 12 × 12 = **4,320** | 30 × 14 × 14 = **5,880** |
| Gemm (fc1, 100×4320) | ✔ | ✘ `GEMM: Dimension mismatch, W: {100,4320} K: 5880` |

Conv · Pool 은 어떤 크기든 처리하지만 Flatten 뒤 Gemm 은 정확히 4,320개를 원한다 → `failed_op = "Gemm"`.

#### 코드로 확인
```python
torch.onnx.export(cnn, torch.zeros(1, 1, 28, 28), "mnist_cnn_dyn.onnx",
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, **EXPORT_KW)
sess_dyn = ort.InferenceSession("mnist_cnn_dyn.onnx", providers=["CPUExecutionProvider"])
print(sess_dyn.get_inputs()[0].shape, sess_dyn.get_outputs()[0].shape)
X = (digits.astype(np.float32) / 255)[:, None]            # (100, 1, 28, 28)
print((sess_dyn.run(None, {"input": X})[0].argmax(1) == labels).mean())
```
- **무엇을 하나**: 배치 축을 연 파일로 100장을 한 번에.
- **어떤 개념**: 개념 1 · 2.
- **출력**: `['batch', 1, 28, 28] ['batch', 10]`, 배치 1 · 4 · 8 → `(1, 10)` · `(4, 10)` · `(8, 10)`, 100장 정확도 `1.0` (직접 실행)

#### 한 번에 정리
- `dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}` → 첫 축이 이름 `'batch'`.
- 100장을 한 번에 run — 1장씩 100번보다 빠르다.
- dynamic 은 약속일 뿐 — 모델이 처리 못 하는 크기는 실행 중 오류 (32×32 → 5,880 ≠ 4,320).
- ⚠ dynamic_axes 의 키 이름이 input_names 와 다르면 조용히 적용되지 않는다.

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
지금까지 export 는 늘 성공했고 결과도 맞았다. 그런데 **export 가 성공했는데 결과가 틀리는** 경우가 있다. 오류 메시지가 없어 가장 위험하다. 원인은 하나 — export 는 **더미 입력으로 실제 실행된 길만** 녹화한다 (Day 1 2교시). 세 가지 함정을 직접 만들고, 여러 입력으로 비교하는 `verify` 로 잡아낸다.

#### 개념 1 · 녹화되는 길 vs 굳는 값
![조용히 틀리는 export 세 가지 — 파이썬 if, NumPy 값, 숫자로 쓴 크기와 고치는 법](assets/fig/ot/d2-traps.svg)

| 함정 | 원래 코드 | export 때 일어나는 일 | 다른 입력에서 |
|---|---|---|---|
| ① 파이썬 if | `if x.mean() > 0: x*2 else x-1` | 더미 `ones` → `x*2` 길만 녹화 | 음수 `[−3,−2,−1]` → ONNX `[−6,−4,−2]`, 원래는 `[−4,−3,−2]` |
| ② NumPy · `float()` | `m = float(np.mean(x.numpy()))` | 더미 `[5,5,5]` → `m = 5` 가 **상수** | `[1,2,3]` → ONNX `[−4,−3,−2]`, 원래는 `[−1,0,1]` |
| ③ 숫자로 쓴 크기 | `x.view(1, -1)` | dynamic 이어도 배치 1 로 고정 | 배치 4 → `(1, 64)` 가 되어 실행 오류 |

①②는 `TracerWarning` 한 줄로만 알려 준다. 환경 셀에서 경고를 숨겼기 때문에 `export_warn` 으로 이 경고만 골라 본다.

#### 개념 2 · 고치는 법 = torch 연산으로
| 함정 | 고친 코드 | 새로 생기는 노드 | verify |
|---|---|---|---|
| ① | `torch.where(x.mean() > 0, x * 2, x - 1)` | ReduceMean · Greater · Mul · Sub · **Where** — 두 길 모두 그래프에 | 양수 · 음수 차이 0 |
| ② | `x - x.mean(dim=1, keepdim=True)` | **ReduceMean** · Sub — 매번 평균 계산 | 차이 0 |
| ③ | `self.fc(x.flatten(1))` | **Flatten** · Gemm | 배치 1 · 4 차이 ≤ 1.5e-08, 배치 4 → `(4, 2)` |

`if self.training:` 같은 **모델 설정**에 따른 분기는 괜찮다 — 입력 값과 상관없이 항상 같은 길이니까. 문제는 **입력 값**에 따른 분기다.

#### 개념 3 · verify — 더미와 다른 입력으로
`verify(model, path, cases)` 는 `{이름: 입력}` 마다 PyTorch 와 ORT 결과를 비교해 ✅ / ❌ 와 최대 차이를 찍는다. PyTorch 자체가 실패하는 경우(③의 배치 4)도 표시한다. **더미 하나로만** 돌리면 녹화된 길과 같은 입력이라 늘 ✅ — 일부러 다른 경우(음수 · 어두운 사진 · 배치 4)를 넣는다.

#### 코드로 확인
```python
class GateFix(nn.Module):
    def forward(self, x):
        return torch.where(x.mean() > 0, x * 2, x - 1)
gate_fix = GateFix()
export_warn(gate_fix, torch.ones(1, 3), "gate_fix.onnx", input_names=["input"], output_names=["output"])
cases_gate = {"양수 [1,2,3]": np.array([[1., 2., 3.]], np.float32),
              "음수 [-3,-2,-1]": np.array([[-3., -2., -1.]], np.float32)}
ok_gate = verify(gate_fix, "gate_fix.onnx", cases_gate)
```
- **무엇을 하나**: if 를 `torch.where` 로 바꿔 두 길을 모두 그래프에 넣고 양수 · 음수로 검증.
- **어떤 개념**: 개념 1 · 2 (①).
- **출력**: `['ReduceMean', 'Constant', 'Greater', 'Constant', 'Mul', 'Constant', 'Sub', 'Where']`, ✅ 양수 차이 0, ✅ 음수 차이 0 (직접 실행)

```python
class CenterFix(nn.Module):
    def forward(self, x):
        return x - x.mean(dim=1, keepdim=True)
class HardFix(Hard):
    def forward(self, x):
        return self.fc(x.flatten(1))                 # (N, 4, 4) → (N, 16)
```
- **무엇을 하나**: ② NumPy 평균 → torch 평균, ③ `view(1, -1)` → `flatten(1)`.
- **어떤 개념**: 개념 2 (② · ③).
- **출력**: `center_fix.onnx ['Constant', 'ReduceMean', 'Sub']` 두 경우 차이 0 · `hard_fix.onnx ['Flatten', 'Gemm']` 배치 4 차이 1.49e-08 (직접 실행)

#### 한 번에 정리
- export = 더미 입력으로 실행된 길만 녹화 — if · NumPy 값 · 숫자로 쓴 크기는 그 순간의 결과로 굳는다.
- 고치는 법: `torch.where` · torch 연산(`mean`) · `flatten(1)` / `view(x.size(0), -1)`.
- 확인법: 여러 경우의 입력으로 `verify` + `TracerWarning` 읽기.
- ⚠ verify 를 더미 하나로 돌리면 늘 ✅ — 일부러 다른 입력을 넣는다.

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
6교시의 문제는 "조용히" 틀렸다. 이번엔 반대로 export 가 **오류를 내며 멈추는** 경우다. 오류가 나니 오히려 쉽다 — 메시지를 읽고 지원되는 연산으로 바꾸면 된다. 그리고 export 된 그래프에 남은 군더더기를 **onnxslim** 으로 줄인다.

#### 개념 1 · 미지원 연산 — 메시지 읽고 바꾸기
`MedianAlarm` 은 센서 5개 값의 **중앙값**(튀는 값에 강함)으로 경보를 낸다. `torch.median` 을 쓰면 export 가 멈춘다:

| 단계 | 내용 |
|---|---|
| 메시지 읽기 | `aten::median` — `aten` = PyTorch 내부 연산 이름, 이 opset 의 ONNX 사전에 대응이 없음 |
| 선택지 | ① opset 올리기 (그 판에 생겼다면) ② 지원 연산의 조합으로 바꾸기 |
| 바꾸기 | 정렬 후 가운데: `torch.sort(x, dim=1).values[:, k:k+1]`, `k = x.shape[1] // 2` → ONNX 의 **TopK** |
| 검증 | 바꾼 모델이 아니라 **원래 모델(torch.median)** 과 verify — 배치 1 · 배치 6(튀는 값 ×50) 모두 ✅ |

#### 개념 2 · onnxslim — 같은 계산, 더 적은 노드
![ViewNet 의 노드 수 — 상수 접기를 끈 export 16개, 기본 export 11개, onnxslim 후 4개](assets/fig/ot/d2-slim.svg)

`x.view(x.size(0), -1)` 는 배치 크기를 매번 읽어(Shape → Gather) shape 를 조립(Unsqueeze → Concat)한 뒤 Reshape 한다. onnxslim 은 이런 부분을 미리 계산해 접고(**constant folding**), 쓸모없는 노드를 지운다.

| ViewNet | 노드 수 | 노드 |
|---|---|---|
| `do_constant_folding=False` | 16 | Identity × 2 · Conv · **BatchNormalization** · Relu · Shape · Gather · Unsqueeze · Concat · Reshape · Gemm … |
| export 기본 (접기 켬) | 11 | Conv · Relu · Shape · Constant · Gather · Constant · Unsqueeze · Constant · Concat · Reshape · Gemm |
| onnxslim 후 | **4** | Conv · Relu · Reshape · Gemm |

단순화 전 · 후 배치 4 결과 차이 **0.0** — 정확도가 오르는 게 아니라 같은 계산을 짧게 적을 뿐이다.

#### 코드로 확인
```python
class MedianAlarmFix(MedianAlarm):
    def forward(self, x):
        k = x.shape[1] // 2
        med = torch.sort(x, dim=1).values[:, k:k + 1]        # (N, 1)
        return self.fc(med)
med_fix = MedianAlarmFix().eval()
med_fix.load_state_dict(med_net.state_dict())
ok_m = verify(med_net, "median_fix.onnx", cases_m)          # ← 원래 모델과 비교
```
- **무엇을 하나**: 중앙값을 정렬 + 가운데 고르기로 바꾸고, 원래 모델과 비교.
- **어떤 개념**: 개념 1.
- **출력**: 노드에 `TopK` 등장, ✅ 배치 1 · ✅ 배치 6 튀는 값 (직접 실행)

```python
slim_model = onnxslim.slim("viewnet.onnx")
onnx.save(slim_model, "viewnet_slim.onnx")
before = [n.op_type for n in onnx.load("viewnet.onnx").graph.node]
after = [n.op_type for n in onnx.load("viewnet_slim.onnx").graph.node]
print(f"{len(before)}개 → {len(after)}개 :", after)
```
- **무엇을 하나**: 단순화하고 저장, 노드 수 비교.
- **어떤 개념**: 개념 2.
- **출력**: `11개 → 4개 : ['Conv', 'Relu', 'Reshape', 'Gemm']`, 전 · 후 최대 차이 `0.0` (직접 실행)

#### 한 번에 정리
- 미지원 연산: 메시지의 `aten::연산` 확인 → opset 올리기 또는 지원 연산 조합 → 원래 모델과 verify.
- `onnxslim.slim(path)` → `onnx.save` — 상수 접기 · 불필요 노드 제거 (ViewNet 11 → 4).
- 단순화 전후도 allclose 로 확인.
- ⚠ slim 결과는 모델 객체 — 저장하지 않으면 원래 파일 그대로다.

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
현장 카메라용 분류기 **BrightNet** 은 PyTorch 에서는 잘 돌지만 ONNX 로 내보내면 이상하게 동작한다. 오늘 배운 모든 도구로 고장을 찾고, 고치고, 배포용 최종 파일 `brightnet_final.onnx` 를 만든다. (가중치는 seed 로 고정한 연습용 — 정확도가 아니라 **배포 과정**이 과제.)

#### 개념 1 · BrightNet 의 의도와 고장 3곳
의도 — 사진 한 장마다: ① 어두우면(평균 < 0.3) 2배 밝게 → ② 평균을 빼서 밝기 보정 → ③ CNN → safe · warning · danger. 입력 64×64 컬러, 0 ~ 1.

| 의도 | 원래 코드 (고장) | 6교시 함정 | 고친 코드 |
|---|---|---|---|
| ① 어두우면 2배 | `if x.mean() < 0.3: x = x * 2` | 파이썬 if | `torch.where(mean < 0.3, x * 2, x)` |
| ② 평균 빼기 | `x - float(np.mean(x.numpy()))` | NumPy 값이 상수로 | `x - x.mean(dim=(1, 2, 3), keepdim=True)` |
| ③ 펴기 | `x.view(1, -1)` | 숫자로 쓴 크기 | `x.flatten(1)` |

**왜 `dim=(1, 2, 3)`?** `x.mean()` 은 배치 전체의 평균 하나다. 배치 8 에 밝은 사진과 어두운 사진이 섞여 있으면 사진마다 판단할 수 없다 → 채널 · 높이 · 너비만 줄여 사진별 평균 `(N, 1, 1, 1)`, `keepdim` 으로 브로드캐스트.

#### 개념 2 · 고장 찾기 — 더미만 ✅
| 경우 (더미 = orange.jpg, 밝기 0.459) | 결과 |
|---|---|
| 더미 그대로 | ✅ |
| 다른 밝은 사진 (butterfly 0.402) | ❌ — 평균이 더미의 숫자로 굳음 |
| 어두운 사진 (×0.25 → 0.101) | ❌ — 더미는 밝아서 "2배 안 하는 길" 만 녹화됨 |

→ `failed = ["다른 밝은 사진", "어두운 사진"]`. ③ `view(1, -1)` 은 경고 없이 숨어 있다가 배치 8 에서 터진다. 채점 기준은 "원래 코드" 가 아니라 **의도** — 사진을 한 장씩 넣은 PyTorch 결과(`intended`).

#### 개념 3 · 배포 파일 만들기 순서
| 순서 | 도구 | 결과 |
|---|---|---|
| 고치기 | `BrightNetFixed` + `load_state_dict` (같은 가중치) | 배치 6 vs 의도 최대 차이 1.49e-08 |
| export | `dynamic_axes` batch · `opset_version=17` · input / logits | 노드 16, TracerWarning 없음, 19.5 KB |
| ① checker | `onnx.checker.check_model` | 통과 |
| ② 단순화 | `onnxslim.slim` | 노드 16 → **14** |
| ③ shape 채우기 | `onnx.shape_inference.infer_shapes` | 중간 shape **27개** |
| ④ 다시 checker · 저장 | `onnx.save(final, "brightnet_final.onnx")` | opset 17 |
| 최종 비교 | 배치 1 (6장 각각) · 배치 8 (6장 + 2장 반복) vs `intended` | 둘 다 최대 차이 **9.7e-08** |

인계할 때 적어 줄 정보: opset 17 · 입력 `input ['batch', 3, 64, 64]` float32 0 ~ 1 RGB · 출력 `logits` 클래스 순서 `["safe", "warning", "danger"]`.

#### 코드로 확인
```python
class BrightNetFixed(BrightNet):
    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)              # (N, 1, 1, 1) — 사진마다 평균
        x = torch.where(mean < 0.3, x * 2, x)                    # ① if → 두 길을 모두
        x = x - x.mean(dim=(1, 2, 3), keepdim=True)              # ② NumPy → torch 연산
        x = self.features(x)
        x = x.flatten(1)                                         # ③ 배치 축은 그대로
        return self.fc(x)
```
- **무엇을 하나**: 고장 3곳을 6교시 방법으로 고친 forward.
- **어떤 개념**: 개념 1.
- **출력**: 고친 모델 (배치 6) vs 의도 (한 장씩) 최대 차이 `1.49e-08` (직접 실행)

```python
onnx.checker.check_model(onnx.load("brightnet_fixed.onnx"))                     # ①
slim_b = onnxslim.slim("brightnet_fixed.onnx")                                  # ②
final = onnx.shape_inference.infer_shapes(slim_b)                               # ③
onnx.save(final, "brightnet_final.onnx")
onnx.checker.check_model(onnx.load("brightnet_final.onnx"))                     # ④
```
- **무엇을 하나**: 검사 → 단순화 → shape 채우기 → 저장 → 다시 검사.
- **어떤 개념**: 개념 3.
- **출력**: `노드 16 → 14 | 중간 shape 27개 | opset 17`, 최종 배치 1 · 배치 8 최대 차이 `9.7e-08`, 자동 채점 과제 1~5 만점 (직접 실행)

#### 한 번에 정리
- 고장 찾기는 여러 경우의 입력(더미 · 다른 사진 · 어두운 사진 · 배치)으로 — 더미 하나로는 늘 ✅.
- 기준은 원래 코드가 아니라 의도(한 장씩 넣은 결과).
- 배포 파일 = 고치기 → export(batch · opset · 이름) → checker → onnxslim → shape_inference → 최종 비교.
- ⚠ 평균은 `dim=(1, 2, 3), keepdim=True` — 배치 전체 평균이면 섞인 배치에서 사진마다 판단이 틀린다.

