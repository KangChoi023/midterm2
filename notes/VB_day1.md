## Day 1 — NumPy 에서 PyTorch 로, 그리고 영상 데이터

### 1교시 · NN 복습 퀴즈 · mynn 에서 PyTorch 로

**키워드**: `mynn ↔ PyTorch 대응`, `torch.Tensor`, `autograd`, `nn.Module`, `nn.Sequential`, `(N, C, H, W)`, `433,890`, `CrossEntropyLoss = softmax 포함`, `logits`, `Colab T4 GPU`

**개요**: NN 과정에서는 원리를 보려고 모든 것을 NumPy 로 직접 만들었다. ① 그 부품(Affine · Conv · Pooling · 손실 · 옵티마이저 · 미니배치 루프)은 PyTorch 에 **이미 빠르고 검증된 형태로** 모두 들어 있다 — 이번 과정은 그 부품으로 더 큰 모델 · 실제 이미지 · 영상으로 간다. ② NN Day 5 의 SimpleConvNet 은 `nn.Sequential` 에 계층을 늘어놓기만 하면 되고 파라미터 수(**433,890**)도 똑같다. ③ `CrossEntropyLoss` 는 softmax 를 포함하므로 모델 끝은 logits 그대로.

**핵심 용어**
- **텐서(Tensor)**: `np.ndarray` 의 PyTorch 판 — GPU 에 올릴 수 있고 연산 기록(autograd)을 남긴다
- **autograd**: 순전파 연산을 기록했다가 `loss.backward()` 한 줄로 역전파를 자동 계산하는 기능
- **nn.Module**: 모델 · 계층의 부모 클래스 — `parameters()` 로 학습할 값을 자동으로 모아 준다
- **logits**: softmax 를 거치기 전 출력 — `nn.CrossEntropyLoss` 가 softmax 를 포함하므로 모델 끝에 softmax 를 붙이지 않는다
- **cuDNN**: NVIDIA GPU 용 딥러닝 연산 라이브러리 — PyTorch 의 합성곱이 GPU 에서 내부적으로 쓴다

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `nn.Linear()` | `nn.Linear(in, out)` | 완전연결층 (NN 의 Affine) |
| `nn.Conv2d()` | `nn.Conv2d(C_in, C_out, k)` | 합성곱 층 (NN 의 Convolution) |
| `nn.MaxPool2d()` | `nn.MaxPool2d(2)` | 최대 풀링 |
| `nn.ReLU()` | `nn.ReLU()` | ReLU 활성화 |
| `nn.Flatten()` | `nn.Flatten()` | (N, C, H, W) → (N, C·H·W) 로 펼치기 |
| `nn.Sequential()` | `nn.Sequential(layer, ...)` | 계층을 순서대로 묶기 |
| `nn.CrossEntropyLoss()` | `nn.CrossEntropyLoss()` | softmax + 교차 엔트로피 + 평균 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 이번 과정의 5일 | 과정 지도 그림 — Neural Network 과정에서 Vision 기본 모델 과정으로 이어지는 5일 로드맵 |
| NN Day 5 의 SimpleConvNet 을 PyTorch 로 다시 쓰기 | `nn.Sequential` 로 SimpleConvNet 을 만들고 (2, 1, 28, 28) 을 계층마다 통과시켜 shape 출력, 파라미터 합 **433,890** 확인 |

**핵심 내용**

#### mynn.py ↔ PyTorch 대응표
> NumPy 로 손수 짠 것 = PyTorch 한 줄 — 원리는 같고 부품만 바뀐다
| NN 과정 (mynn · NumPy) | PyTorch | 달라지는 점 |
|---|---|---|
| `np.ndarray` | `torch.Tensor` | GPU · 연산 기록 |
| 계층마다 `backward()` | `loss.backward()` | 계산 그래프를 따라 자동 미분 |
| `Affine(784, 100)` | `nn.Linear(784, 100)` | 같은 Y = XW + b (W 는 (out, in) 으로 저장해 x @ W.T + b) |
| `Convolution(1, 30, 5)` | `nn.Conv2d(1, 30, 5)` | im2col · col2im 은 내부 처리 (GPU 는 cuDNN) |
| `Pooling(2, 2, 2)` · `Relu()` | `nn.MaxPool2d(2)` · `nn.ReLU()` | 같음 |
| `SoftmaxWithLoss()` | `nn.CrossEntropyLoss()` | logits 를 그대로 넣는다 |
| `Sequential([...])` | `nn.Sequential(...)` · `nn.Module` | `parameters()` 로 자동 수집 |
| `Adam(lr=0.001)` + `update()` | `torch.optim.Adam(model.parameters(), lr=1e-3)` + `step()` | `zero_grad()` 단계가 추가 |
| `rng.permutation` 미니배치 루프 | `DataLoader(batch_size=100, shuffle=True)` | 섞기 · 자르기 · 병렬 로딩 |
| `np.savez` | `torch.save(model.state_dict(), path)` | 이름 → 텐서 딕셔너리 |
- **손으로 만든 경험** → PyTorch 코드 한 줄 한 줄이 "내가 짰던 그 부분"으로 읽힌다
- **이미지 텐서 모양** → NN 과정과 같은 **(N, C, H, W)**

#### SimpleConvNet 을 PyTorch 로
> Conv(1→30, 5×5) → ReLU → Pool(2×2) → Flatten → Linear(4320→100) → ReLU → Linear(100→10)
| 층 | 출력 shape | 파라미터 |
|---|---|---|
| 입력 | (N, 1, 28, 28) | — |
| Conv2d(1, 30, 5) | (N, 30, 24, 24) | 30·1·5·5 + 30 = **780** |
| ReLU · MaxPool2d(2) | (N, 30, 12, 12) | 0 |
| Flatten | (N, 4320) | 0 (30·12·12 = 4320) |
| Linear(4320, 100) | (N, 100) | 4320·100 + 100 = **432,100** |
| Linear(100, 10) | (N, 10) | 100·10 + 10 = **1,010** |
| 합계 | | **433,890** (NumPy 판과 같음) |
- **NumPy 판 기록** → 1 epoch test **0.9810**
- **`nn.Flatten()`** → NumPy 판에서 Affine 이 내부에서 하던 펼치기를 계층 하나로
- ⚠ 모델 끝에 softmax 를 붙이면 안 된다 — `CrossEntropyLoss` 안에 이미 있다 (두 번 적용)
##### 자세히
```text
 (N,1,28,28) ─Conv2d(1,30,5)─► (N,30,24,24) ─ReLU─► ─MaxPool2d(2)─► (N,30,12,12)
   ─Flatten─► (N,4320) ─Linear─► (N,100) ─ReLU─► ─Linear─► (N,10) logits
                                                    └─► CrossEntropyLoss (softmax inside)
```

#### 이번 과정의 5일 · Colab 준비
> PyTorch 전환 → CNN 대표 구조 → 경량 · 전이학습 → 객체 탐지 → 세그멘테이션 · 배포
| Day | 만드는 것 |
|---|---|
| 1 | PyTorch 전환 · 이미지 · 영상 파이프라인 → 종이 손글씨 인식 |
| 2 | CNN 대표 구조 LeNet · VGG · ResNet → CIFAR-10 |
| 3 | 경량 CNN · 전이학습 · Grad-CAM → 현장 안전 분류기 |
| 4 | 객체 탐지 — IoU · NMS · mAP · YOLO 구조 |
| 5 | 세그멘테이션 · 포즈 · ONNX · 성능 측정 → 침입 감지 |
- **환경 셀이 `device : cpu`** → 런타임 › 런타임 유형 변경 › **T4 GPU** (오늘은 CPU 로도 돌지만 8교시 학습이 몇 배 느림)
- **GPU 할당 실패** → 무료 Colab 은 사용자가 많으면 못 받을 수 있음 → 그대로 CPU 로 진행
- ⚠ 런타임이 끊긴 뒤 `NameError` → 맨 위 환경 셀부터 다시 실행

**기본 사용형태**
```python
net = nn.Sequential(
    nn.Conv2d(1, 30, 5),   # (N, 1, 28, 28) → (N, 30, 24, 24)
    nn.ReLU(),
    nn.MaxPool2d(2),       # → (N, 30, 12, 12)
    nn.Flatten(),          # → (N, 4320)
    nn.Linear(4320, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
)
n_params = sum(p.numel() for p in net.parameters())   # 433890
```

**연결**
- [NN D5·5교시 SimpleConvNet 으로 MNIST 학습](#/nn/5/5) — 같은 구조를 NumPy 로 만들었던 곳
- [NN D3·5교시 Affine · Softmax-with-Loss 계층](#/nn/3/5) — `nn.Linear` · `CrossEntropyLoss` 가 대신하는 계층
- [NN D4·8교시 미니 프로젝트: 미니 프레임워크 mynn.py](#/nn/4/8) — Sequential · Trainer 의 원조

**연관 문제**
<<QA 1>>

### 2교시 · Tensor · GPU

**키워드**: `torch.Tensor`, `float32 기본`, `axis → dim`, `keepdims → keepdim`, `from_numpy = 공유`, `torch.tensor = 복사`, `reshape · view · permute · unsqueeze`, `.to(device)`, `synchronize()`

**개요**: Tensor 는 ndarray 에 세 가지(**device** · **requires_grad** · 기본 float32)를 더한 것이다. ① 함수 이름은 NumPy 와 거의 같고 축 인자만 `axis` → `dim`, `keepdims` → `keepdim`. ② `from_numpy` 는 같은 메모리를 **공유**하고 `torch.tensor` 는 **복사**한다 — NN Day 1 "b = a 는 복사가 아니다" 와 같은 이야기. ③ GPU 는 수천 개 코어로 행렬곱을 한꺼번에 계산하지만, 데이터와 모델이 **같은 device** 에 있어야 한다.

**핵심 용어**
- **device**: 텐서가 놓인 곳 — `cpu` 또는 `cuda`(GPU 메모리)
- **requires_grad**: 이 텐서로 한 연산을 기록해 나중에 미분할지 여부 (3교시)
- **메모리 공유(share)**: `from_numpy` 가 같은 메모리를 보게 하는 것 — 한쪽을 고치면 다른 쪽도 바뀐다
- **비동기(asynchronous) 실행**: GPU 가 명령만 받고 바로 돌아오는 것 — 시간을 재려면 `torch.cuda.synchronize()` 로 끝날 때까지 기다린다

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.tensor()` | `torch.tensor(data)` | 값을 **복사**해 텐서 만들기 |
| `torch.from_numpy()` | `torch.from_numpy(a)` | ndarray 와 **메모리 공유** |
| `torch.zeros()` | `torch.zeros(2, 1, 28, 28)` | 0 으로 채운 텐서 |
| `torch.arange()` | `torch.arange(24)` | 연속 정수 텐서 |
| `torch.randn()` | `torch.randn(n, n, device=dev)` | 표준정규 난수 텐서 |
| `torch.exp()` | `torch.exp(x)` | 원소별 e^x |
| `Tensor.reshape()` | `t.reshape(2, -1)` | 모양 바꾸기 (원소 순서 그대로) |
| `Tensor.permute()` | `t.permute(2, 0, 1)` | 축 순서 바꾸기 |
| `Tensor.unsqueeze()` | `t.unsqueeze(0)` | 크기 1 인 축 넣기 |
| `Tensor.to()` | `t.to(device)` | device · dtype 옮기기 |
| `Tensor.numpy()` | `t.numpy()` | ndarray 로 (CPU · 기록 없는 텐서만) |
| `Tensor.item()` | `t.item()` | 원소 1개 텐서 → 파이썬 숫자 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | 같은 2×3 배열을 NumPy · Tensor 로 만들어 dtype(float64 vs float32) · device 비교, `sum(dim=0)` · `@` · `.numpy()` · `.item()` |
| from_numpy 는 메모리를 공유하고, torch.tensor 는 복사한다 | `a = zeros(3)` 에서 `b = from_numpy(a)`, `c = tensor(a)` → `b[0] = 7`, `c[1] = 9` 후 a = [7, 0, 0], c = [0, 9, 0] |
| GPU 로 옮기기 — .to(device) | 2048×2048 행렬곱을 CPU · GPU 로 10번 평균 시간 비교 (준비 운동 1회, `synchronize()`) |

**핵심 내용**

#### Tensor = ndarray + device + requires_grad
> 함수 이름은 NumPy 와 거의 같다 — `axis → dim` · `keepdims → keepdim` · `np.exp → torch.exp`
| 속성 · 이름 | NumPy | PyTorch |
|---|---|---|
| 기본 실수형 | float64 | **float32** |
| 축 인자 | `axis=1, keepdims=True` | `dim=1, keepdim=True` (`axis` 도 대부분 받음) |
| 최댓값 | `Y.max(axis=1, keepdims=True)` | `Y.max(dim=1, keepdim=True).values` |
| 놓인 곳 | (CPU 만) | `t.device` — cpu / cuda |
- **`max(dim=…)`** → (값, 인덱스) 두 개를 돌려줌 → 값은 **`.values`**
- **softmax 순전파** → `Y = X @ W + b` → `Y − Y.max` (overflow 방지) → `exp` → `/ 합` — NN Day 2 와 같은 식, 이름만 바뀜
- ⚠ `float64` 배열을 `from_numpy` 로 그대로 모델에 넣으면 dtype 오류 → **`.float()`**

#### from_numpy 는 공유, torch.tensor 는 복사
> NN Day 1 "b = a 는 복사가 아니다" 와 같은 뿌리
| 호출 | 메모리 | 한쪽을 고치면 |
|---|---|---|
| `torch.from_numpy(a)` | **공유** | 서로 보인다 |
| `torch.tensor(a)` | **복사** | 영향 없음 |
- **예** → `b = from_numpy(a)`, `c = tensor(a)` 에서 `b[0] = 7`, `c[1] = 9` → **a = [7, 0, 0]** · c = [0, 9, 0]
- **from_numpy 장점** → 복사가 없어 빠름 (큰 이미지 배치)
##### 자세히
```text
 a (ndarray) ──┐
               ├──► [ 7  0  0 ]   same memory
 b = from_numpy(a) ┘
 c = tensor(a)  ──► [ 0  9  0 ]   its own copy
```

#### shape 바꾸기
> reshape · view = 모양만, permute = 축 순서, unsqueeze = 크기 1 축
| 함수 | 하는 일 | 쓰는 곳 |
|---|---|---|
| `reshape` · `view` | 원소 순서 그대로, 모양만 | Flatten |
| `permute(2, 0, 1)` | 축 순서 변경 (원소 배치가 달라짐) | 이미지 HWC → CHW (6교시) |
| `unsqueeze(0)` / `squeeze(0)` | 크기 1 축 넣기 / 빼기 | 이미지 1장 → 배치 (1, C, H, W) |
- **`x = arange(24).reshape(2, 3, 4)`** → `x.permute(0, 2, 1)` 은 (2, 4, 3), `p[1, 3, 2] == x[1, 2, 3] == 23`
- **`x.reshape(2, -1)`** → (2, 12) — `-1` 은 "나머지 자동"

#### GPU 로 옮기기 — 같은 device 에
> GPU 는 수천 개 코어로 행렬곱을 한꺼번에 — 데이터와 모델이 같은 device 에 있어야 계산된다
- **2048×2048 행렬곱** → 2·2048³ ≈ **17.2 GFLOP**, 텐서 하나 **16 MiB** (float32)
- **시간 재기** → 첫 실행은 초기화가 섞이므로 준비 운동 1회 후 10번 평균
- **결과 꺼내기** → GPU 텐서는 `.cpu().numpy()`
- ⚠ GPU 는 **비동기** — `torch.cuda.synchronize()` 없이 재면 실제보다 훨씬 빠르게 나온다
- ⚠ 모델은 GPU, 데이터는 CPU → `Expected all tensors to be on the same device` → 둘 다 **`.to(device)`**
- ⚠ GPU 텐서는 바로 `.numpy()` 불가 → **`.cpu().numpy()`**

**기본 사용형태**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
X = torch.from_numpy(X_np).float()        # float64 → float32 (메모리 공유)
layer = nn.Linear(784, 10).to(device)
Y = layer(X.to(device))                   # 같은 device
print(Y.cpu().detach().numpy().shape)     # GPU → CPU → ndarray
```

**연결**
- [NN D1·2교시 Python 속성 (C 개발자용)](#/nn/1/2) — `b = a` 는 복사가 아니다
- [NN D1·3교시 NumPy ndarray](#/nn/1/3) — dtype · axis · keepdims
- [VB D1·3교시 autograd: 역전파를 자동으로](#/vb/1/3) — `requires_grad` 가 쓰이는 곳

**연관 문제**
<<QA 2>>

### 3교시 · autograd: 역전파를 자동으로

**키워드**: `requires_grad=True`, `계산 그래프`, `loss.backward()`, `.grad 에 누적`, `zero_grad()`, `torch.no_grad()`, `detach()`, `Xᵀ(y − t)/N`, `zero_grad → backward → step`

**개요**: NN Day 3 에서는 계층마다 `backward()` 를 손으로 짜고 gradient check 로 확인했다. ① PyTorch 는 `requires_grad=True` 인 텐서로 한 연산을 **계산 그래프**로 기록해 두었다가 `loss.backward()` 한 번에 연쇄 법칙을 거꾸로 적용한다. ② 결과는 잎(leaf) 텐서의 `.grad` 에 **더해진다** — 그래서 매번 비워야 한다. ③ autograd 결과는 손으로 유도한 Xᵀ(y − t)/N 과 같고, 학습 한 step 은 `zero_grad() → backward() → step()`.

**핵심 용어**
- **계산 그래프(computational graph)**: 연산을 노드로 기록한 구조 — 각 노드가 미분에 필요한 값을 저장 (NN 의 `self.x`, `self.mask` 와 같은 일)
- **잎 텐서(leaf)**: 사용자가 직접 만든 `requires_grad=True` 텐서 — `.grad` 가 여기에 쌓인다
- **기울기 누적**: `backward()` 를 부를 때마다 `.grad` 에 더해지는 성질 — 버그가 아니라 기능 (큰 배치를 나눠 모을 때)
- **no_grad**: 연산 기록을 끄는 블록 — 추론 · 직접 갱신에서 사용

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Tensor.backward()` | `loss.backward()` | 역전파 — 잎의 `.grad` 에 더하기 |
| `Tensor.grad` | `x.grad` | 쌓인 기울기 (속성) |
| `Tensor.zero_grad` | `x.grad.zero_()` | 기울기를 0 으로 비우기 |
| `torch.no_grad()` | `with torch.no_grad():` | 기록을 끄는 블록 (추론 · 직접 갱신) |
| `Tensor.detach()` | `t.detach().numpy()` | 그래프에서 떼어낸 텐서 |
| `F.one_hot()` | `F.one_hot(t, K)` | 정수 레이블 → 원-핫 |
| `F.cross_entropy()` | `F.cross_entropy(logits, t)` | softmax + 교차 엔트로피 + 평균 |
| `torch.optim.SGD()` | `torch.optim.SGD([q], lr=0.1)` | 확률적 경사하강법 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | f = x² + 3y², (1, 2) → `f.backward()` 로 x.grad = **2**, y.grad = **12** · 누적 실험 2 → 4 → (비운 뒤) 2 · Affine + softmax + CE 에서 autograd dW 와 손 유도 Xᵀ(y − t)/N 비교 · 직접 경사하강 vs `optim.SGD` |

**핵심 내용**

#### autograd — 기록했다가 거꾸로
> requires_grad=True → 연산 기록 → `backward()` → 잎의 `.grad` 에 누적
- **f = x² + 3y², (x, y) = (1, 2)** → f = **13**, df/dx = 2x = **2**, df/dy = 6y = **12**
- **`f.backward()`** → `x.grad = 2`, `y.grad = 12` — f 가 **스칼라**여야 바로 호출 가능 (손실은 언제나 스칼라)
- **Affine + softmax + CE** → autograd 의 dW = 손으로 유도한 **Xᵀ(y − t)/N** (NN Day 3 의 "(y − t)x")
- ⚠ 그래프가 붙은 텐서를 `.numpy()` → `Can't call numpy() on Tensor that requires grad` → **`.detach().numpy()`**
##### 자세히
```text
 x ─┐                         backward from 1:
    (²) ─┐                    df/dx = 1 · 2x  = 2
 y ─(²)─(×3)─(+) ── f = 13    df/dy = 1 · 3 · 2y = 12
 forward saves x, y  →  backward multiplies local derivatives (chain rule)
```

#### 기울기는 더해진다 — 매번 비운다
> `backward()` 를 부를 때마다 `.grad` 에 **더해짐**
| 호출 순서 (x = 1, f = x²) | `x.grad` |
|---|---|
| 1번째 `backward` | 2 |
| 비우지 않고 2번째 | **4** (더해짐) |
| `grad.zero_()` 후 3번째 | 2 |
- **누적이 쓸모 있는 곳** → 큰 배치를 여러 번에 나눠 기울기를 모을 때 → 그래서 학습 루프는 매번 **직접** 비운다
- ⚠ `zero_grad()` 를 빼먹으면 기울기가 계속 더해져 학습이 폭주
- ⚠ 평가 · 추론에서 기록을 안 끄면 메모리 낭비 → **`with torch.no_grad():`**

#### 직접 경사하강 vs optimizer
> `opt.zero_grad() → loss.backward() → opt.step()` = NN 의 `optimizer.update(params, grads)`
- **f = p₀² + 3p₁², 시작 (−4, 2), lr 0.1, 100 step** → 매 step p₀ × **0.8**, p₁ × **0.4** → 끝 위치 ≈ (0, 0)
- **직접 갱신** → `with torch.no_grad(): p -= lr * p.grad` 후 `p.grad.zero_()`
- **`optim.SGD([q], lr=0.1)`** → 위 직접 루프와 같은 결과
- ⚠ 직접 갱신 `p -= lr * p.grad` 는 **`no_grad` 안에서** (없으면 leaf 변수 in-place 오류)

**기본 사용형태**
```python
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
f = x ** 2 + 3 * y ** 2
f.backward()
print(x.grad.item(), y.grad.item())     # 2.0 12.0

opt = torch.optim.SGD([q], lr=0.1)
opt.zero_grad(); loss.backward(); opt.step()   # 학습 한 step
```

**연결**
- [NN D3·3교시 연쇄 법칙](#/nn/3/3) — backward 가 하는 일 (국소 미분의 곱)
- [NN D3·5교시 Affine · Softmax-with-Loss 계층](#/nn/3/5) — 손으로 짠 dW = Xᵀ · dout
- [VB D1·5교시 학습 루프 · 평가 · 저장](#/vb/1/5) — `zero_grad → backward → step` 이 쓰이는 곳

**연관 문제**
<<QA 3>>

### 4교시 · nn.Module · Dataset · DataLoader

**키워드**: `nn.Module`, `__init__ = 부품`, `forward = 연결`, `parameters()`, `79,510`, `Dataset (__len__ · __getitem__)`, `DataLoader`, `transforms.ToTensor`, `600 iteration`, `long 레이블`

**개요**: 모델과 데이터를 PyTorch 의 표준 틀에 담는다. ① 모델은 `nn.Module` 을 상속한 클래스 — `__init__` 에서 계층을 만들고 `forward` 에서 데이터가 지나가는 길을 쓴다, 파라미터는 `parameters()` 로 자동 수집(NN 의 `Sequential.params()`). ② 데이터는 `Dataset` 이 한 장씩 주고 `DataLoader` 가 섞어서 `batch_size` 개씩 쌓아 준다(NN 의 `rng.permutation` 루프). ③ `CrossEntropyLoss` 를 풀어 쓰면 log_softmax 의 정답 칸 평균 — NN Day 2 와 같은 팬시 인덱싱.

**핵심 용어**
- **nn.Module 상속 클래스**: `__init__` 에서 `self.` 에 계층을 붙이고 `forward(x)` 를 정의한 모델 — `model(x)` 로 호출
- **Dataset**: `len(ds)` 와 `ds[i]` (i 번째 이미지, 레이블) 만 알면 되는 객체
- **DataLoader**: 인덱스를 섞고(`shuffle`) 배치로 묶어 텐서 하나로 쌓아 주는 반복자
- **iteration / epoch**: 배치 1번 처리 / 전체 데이터 1바퀴 — 60,000 ÷ 100 = **600**
- **transforms.ToTensor**: uint8 (H, W) 0~255 이미지를 float (1, H, W) 0~1 텐서로 바꾸는 변환

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `nn.Module` | `class M(nn.Module):` | 모델 부모 클래스 |
| `Module.parameters()` | `model.parameters()` | 학습할 값 전체 |
| `F.relu()` | `F.relu(x)` | 함수형 ReLU |
| `F.log_softmax()` | `F.log_softmax(logits, dim=1)` | log(softmax) |
| `torchvision.datasets.MNIST()` | `datasets.MNIST("data", train=True, download=True, transform=t)` | MNIST Dataset |
| `transforms.ToTensor()` | `transforms.ToTensor()` | uint8 (H, W) 0~255 → float (1, H, W) 0~1 |
| `DataLoader()` | `DataLoader(ds, batch_size=100, shuffle=True)` | 섞기 · 배치 · 로딩 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 데이터는 Dataset → DataLoader | MNIST train 60,000 · test 10,000, 한 장 (1, 28, 28) float32 0~1 확인 → 배치 100 로더에서 (100, 1, 28, 28) · (100,) 꺼내기 · CE 를 log_softmax 로 풀어 같은 값 확인 · 직접 만든 Dataset |

**핵심 내용**

#### 모델 클래스의 모양
> `__init__` = 부품 만들기, `forward` = 연결하기, `parameters()` 는 자동 수집
- **MLP** → `fc1 = Linear(784, 100)`, `fc2 = Linear(100, 10)` → 784·100 + 100 + 100·10 + 10 = **79,510**
- **`forward` 첫 줄 `x.flatten(1)`** → (N, 1, 28, 28) → (N, 784), **배치 축은 남긴다**
- **마지막** → logits 그대로 — softmax 는 손실 함수 안에서
- **`self.` 에 붙인 계층** → 그 파라미터가 `model.parameters()` 에 자동으로 모임 (NN `Sequential.params()`)
##### 자세히
```text
 class MLP(nn.Module):
   __init__  : self.fc1 = Linear(784,100), self.fc2 = Linear(100,10)   parts
   forward(x): x.flatten(1) → fc1 → relu → fc2 → logits                wiring
 model(x)  calls forward;  model.parameters()  collects fc1.W, fc1.b, fc2.W, fc2.b
```

#### Dataset → DataLoader
> Dataset 은 한 장씩, DataLoader 는 섞어서 묶음으로
| 단계 | 하는 일 | NN 과정에서 |
|---|---|---|
| `Dataset[i]` | (이미지 i, 레이블 i) 한 쌍 | `x_train[i]` |
| `shuffle=True` | epoch 마다 순서 섞기 | `rng.permutation(N)` |
| `batch_size=100` | 100개씩 텐서 하나로 | 슬라이스 `idx[i:i+100]` |
- **학습 로더** → `batch_size=100, shuffle=True` → epoch 당 **600** iteration
- **평가 로더** → `batch_size=1000, shuffle=False` → **10** iteration
- **직접 만든 Dataset** → `__getitem__` 에서 `torch.from_numpy(X[i]).float() / 255` → `unsqueeze(0)` (채널 축) → `(x, int(y[i]))`
- ⚠ 평가 로더는 `shuffle=False` (순서가 의미 있을 때)

#### CrossEntropyLoss 를 풀어 쓰면
> loss = −log_softmax(logits)[i, 정답 i] 의 평균
- **`-F.log_softmax(logits, dim=1)[torch.arange(N), yb].mean()`** → `nn.CrossEntropyLoss()(logits, yb)` 와 같은 값
- **팬시 인덱싱** → 행마다 정답 칸 하나씩 (NN Day 3 `y[np.arange(N), t]`)
- ⚠ 레이블은 **long(int64)** 정수 — float 로 주면 오류

**기본 사용형태**
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)
    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
xb, yb = next(iter(train_loader))     # (100, 1, 28, 28), (100,)
```

**연결**
- [NN D3·7교시 미니배치 SGD](#/nn/3/7) — `rng.permutation` 으로 짰던 루프
- [NN D2·6교시 출력층과 손실 함수](#/nn/2/6) — CE = −log(정답 확률)
- [Python D3·1교시 클래스 기초 — dict 를 졸업하기](#/py/3/1) — 클래스 · `__init__` · `self`

**연관 문제**
<<QA 4>>

### 5교시 · 학습 루프 · 평가 · 저장

**키워드**: `train() / eval()`, `to device → forward → loss → zero_grad → backward → step`, `no_grad`, `argmax(dim=1)`, `loss.item()`, `state_dict`, `torch.save / load_state_dict`, `map_location`

**개요**: 모든 PyTorch 학습 코드는 **같은 모양**이다 — 모델이 ResNet 이든 YOLO 든 바뀌는 건 모델 · 데이터 · 손실 함수뿐. ① 학습: `train()` → 배치마다 (device 로 → 순전파 → 손실 → zero_grad → backward → step). ② 평가: `eval()` + `no_grad()` + `argmax` — 두 가지는 하는 일이 다르다. ③ 저장은 모델 객체가 아니라 `state_dict()`(이름 → 파라미터)만, 불러오기는 같은 클래스로 새 모델을 만든 뒤 `load_state_dict`.

**핵심 용어**
- **학습 모드 / 평가 모드**: `model.train()` / `model.eval()` — Dropout · BatchNorm 동작이 바뀐다
- **state_dict**: 계층 이름 → 파라미터 텐서 딕셔너리 — 저장 · 불러오기의 표준 단위
- **map_location**: 저장 파일을 어느 device 로 읽을지 — GPU 에서 저장한 것을 CPU 에서 열 때 필요
- **정확도(accuracy)**: 맞은 수 / 전체 — `(pred == yb).sum().item()` 을 모아 나눔

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Module.train()` | `model.train()` | 학습 모드 |
| `Module.eval()` | `model.eval()` | 평가 모드 |
| `torch.optim.Adam()` | `torch.optim.Adam(model.parameters(), lr=1e-3)` | Adam 옵티마이저 |
| `Optimizer.zero_grad()` | `opt.zero_grad()` | 기울기 비우기 |
| `Optimizer.step()` | `opt.step()` | 파라미터 갱신 |
| `Tensor.argmax()` | `logits.argmax(dim=1)` | 가장 큰 칸의 인덱스 = 예측 |
| `torch.save()` | `torch.save(model.state_dict(), "mlp.pt")` | 파라미터 저장 |
| `torch.load()` | `torch.load("mlp.pt", map_location=device)` | 저장 파일 읽기 |
| `Module.load_state_dict()` | `model.load_state_dict(sd)` | 파라미터 채우기 |
| `torch.manual_seed()` | `torch.manual_seed(0)` | 난수 고정 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | `train_one_epoch` · `evaluate` 를 만들어 MLP 를 Adam 으로 몇 epoch 학습 → 정확도 출력 → `state_dict` 저장 · 새 모델에 불러와 같은 정확도 확인 |

**핵심 내용**

#### 학습 루프 한 바퀴
> 배치를 device 로 → 순전파 → 손실 → `zero_grad` → `backward` → `step`
| 단계 | PyTorch | NN 과정 |
|---|---|---|
| 배치 → device | `xb, yb = xb.to(device), yb.to(device)` | (없음 — CPU) |
| 순전파 | `logits = model(xb)` | `net.predict(x)` |
| 손실 | `loss = loss_fn(logits, yb)` | `SoftmaxWithLoss.forward` |
| 기울기 비우기 | `opt.zero_grad()` | (매번 새 grads 딕셔너리) |
| 역전파 | `loss.backward()` | `net.gradient(x, t)` |
| 갱신 | `opt.step()` | `optimizer.update(params, grads)` |
- **`train_one_epoch` 첫 줄 `model.train()`** → 평가 뒤에 이어 학습해도 안전
- **손실 기록** → `loss.item() * len(yb)` 를 모아 샘플 수로 나눔 (배치 크기가 달라도 정확한 평균)
- ⚠ `loss` 텐서를 그대로 리스트에 모음 → 그래프가 붙어 메모리 누수 → **`loss.item()`**
- ⚠ 옵티마이저를 만든 **뒤에** `model.to(device)` → 순서 지킬 것 (모델을 먼저 옮긴다)
##### 자세히
```text
 for xb, yb in loader:
     xb, yb → device ─► model(xb) ─► loss_fn ─► zero_grad ─► backward ─► step
                         forward     scalar     clear .grad   fill .grad   W -= …
```

#### 평가 — eval() 과 no_grad() 는 다른 일
> `model.eval()` = Dropout · BN 동작 전환, `torch.no_grad()` = 기울기 기록 끄기
- **예측** → `pred = model(xb).argmax(dim=1)` — softmax 없이 (순서 불변)
- **정확도** → `(pred == yb).sum().item()` 을 모아 전체 수로 나눔
- ⚠ 평가 뒤 학습을 이어 가는데 `model.train()` 을 안 부름 → Dropout · BN 이 꺼진 채 학습

#### 저장과 불러오기 — state_dict
> 모델 객체가 아니라 파라미터 딕셔너리만 저장
- **저장** → `torch.save(model.state_dict(), "mlp.pt")` — `{"fc1.weight": (100, 784), "fc1.bias": (100,), …}`
- **불러오기** → 같은 클래스로 새 모델 → `load_state_dict(torch.load(..., map_location=device))`
- **확인** → 불러온 모델의 정확도가 저장 전과 **같아야** 한다
- **`fc1.weight` shape (100, 784)** → PyTorch 는 W 를 (out, in) 으로 저장

**기본 사용형태**
```python
def train_one_epoch(model, loader, loss_fn, opt):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(yb); n += len(yb)
    return total / n

def evaluate(model, loader):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
            n += len(yb)
    return correct / n
```

**연결**
- [NN D3·8교시 미니 프로젝트: MNIST 2층 MLP](#/nn/3/8) — 직접 짠 epoch 루프
- [NN D4·4교시 Batch Normalization](#/nn/4/4) — 학습 · 추론 모드가 필요한 이유 (train_flg)
- [VB D1·8교시 미니 프로젝트: 종이 손글씨 · 영상 숫자 인식](#/vb/1/8) — 같은 두 함수를 CNN 에 그대로 사용

**연관 문제**
<<QA 5>>

### 6교시 · 디지털 이미지: 픽셀 · 채널 · 정규화

**키워드**: `(H, W, C) uint8 BGR`, `(N, C, H, W) float32 RGB`, `img[:, :, ::-1]`, `permute(2, 0, 1)`, `흑백 0.299R + 0.587G + 0.114B`, `uint8 overflow`, `cv2.add 포화`, `crop → resize → ToTensor → Normalize`, `cv2.resize (W, H)`

**개요**: MNIST 는 이미 다듬어진 28×28 흑백이었다. 실제 사진은 크고, 컬러이고, uint8 이다. ① OpenCV · NumPy 는 **(H, W, C) uint8 BGR**, PyTorch 는 **(C, H, W) float32 RGB 0~1** 에 채널별 평균 · 표준편차 정규화, 모델에는 배치 축을 붙인 (N, C, H, W). ② 흑백은 세 채널의 가중 평균이고, uint8 끼리 계산하면 255 를 넘어 **overflow** — 계산은 float 로. ③ 사진 → 입력 텐서: crop → resize → ToTensor → Normalize → unsqueeze(0), 그리고 **학습 때와 추론 때의 전처리가 같아야** 한다.

**핵심 용어**
- **채널(channel)**: 같은 크기의 밝기 판 — 컬러 사진은 B · G · R 3장의 겹침
- **uint8 overflow**: 0~255 정수끼리 더해 255 를 넘으면 0 부터 다시 세는 현상 (300 → 44)
- **포화(saturation)**: 255 를 넘는 값을 255 에서 멈추게 하는 연산 (`cv2.add`)
- **정규화(normalize)**: `(x − mean) / std` 를 채널마다 — NN Day 1 표준화의 이미지 판
- **center crop**: 가운데 정사각형만 잘라 내기 — 비율을 깨지 않고 정사각 입력을 만들 때

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `cv2.imread()` | `cv2.imread(path)` | 이미지 읽기 (BGR, uint8) |
| `cv2.cvtColor()` | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | 색 공간 변환 |
| `cv2.resize()` | `cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)` | 크기 바꾸기 — 크기는 **(W, H)** |
| `cv2.add()` | `cv2.add(a, b)` | 255 에서 멈추는 덧셈 |
| `ndarray.astype()` | `a.astype(np.float32)` | 자료형 바꾸기 |
| `zipfile.ZipFile()` | `zipfile.ZipFile(path).extractall("data")` | zip 풀기 |
| `Tensor.float()` | `t.float()` | float32 로 |
| `Tensor.view()` | `mean.view(3, 1, 1)` | 모양 바꾸기 (브로드캐스트용) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | 종이 손글씨 사진 30장(d숫자_번호.jpg)을 풀어 한 장 읽기 → B · G · R 채널 따로 보기 · 흑백 직접 계산 vs `cvtColor` · uint8 덧셈 overflow · HWC → CHW 텐서 · crop → resize → 정규화로 (1, 3, 224, 224) |

**핵심 내용**

#### 두 세계의 약속 — OpenCV vs PyTorch
> OpenCV: (H, W, C) uint8 BGR → PyTorch: (N, C, H, W) float32 RGB 0~1 (+ 정규화)
| 항목 | OpenCV · NumPy | PyTorch |
|---|---|---|
| 모양 | (H, W, C) — 한 픽셀의 B, G, R 이 붙어 있음 | (C, H, W) — 채널별 판 3장 → 배치 (N, C, H, W) |
| 값 | uint8 0~255 | float32 0~1 (+ 정규화) |
| 채널 순서 | **BGR** | **RGB** |
- **BGR → RGB** → `rgb = img[:, :, ::-1]` (마지막 축을 거꾸로)
- **HWC → CHW** → `permute(2, 0, 1)` — 새 0번 = 옛 2번(C), 새 1번 = 옛 0번(H), 새 2번 = 옛 1번(W)
- **`torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255`** = `transforms.ToTensor()(rgb)` (음수 stride 때문에 `.copy()`)
- ⚠ BGR 을 RGB 로 안 바꿔 사전학습 모델에 넣으면 **오류 없이 정확도만 조용히 떨어진다**
##### 자세히
```text
 OpenCV (H, W, 3)            PyTorch (3, H, W)
 pixel(0,0) = [B G R]        R ┌──────┐
 pixel(0,1) = [B G R]  ──►   G │plane │  ──► unsqueeze(0) ──► (1, 3, H, W)
 …                           B └──────┘
       [:, :, ::-1]  then  permute(2, 0, 1)  then  / 255
```

#### 흑백 = 채널의 가중 평균
> gray = 0.299 R + 0.587 G + 0.114 B — 가중치 합 1, 사람 눈은 **초록**에 가장 민감
- **`cv2.cvtColor(img, BGR2GRAY)`** → 직접 계산과 같다 — 단 직접 계산은 **float32 로 바꾼 뒤**
- **파란 잉크** → B 채널에서 덜 어둡고 R 채널에서 가장 어둡다

#### uint8 overflow — 계산은 float 로
> 0~255 정수끼리 더해 255 를 넘으면 256 을 빼고 다시 센다
| 계산 | 결과 |
|---|---|
| `uint8(200) + 100` | **44** (300 − 256) |
| `uint8(250) + 100` | **94** (350 − 256) |
| `float32 + 100` | 300 · 350 |
| `cv2.add(p, 100)` | 255 · 255 (포화) |
- ⚠ 이미지 계산은 **float 로 바꾼 뒤** → 필요하면 다시 uint8

#### 입력 만들기 파이프라인
> crop → resize → ToTensor → Normalize → unsqueeze(0)
- **가운데 정사각형 crop** → `s = min(h, w)`, `top = (h − s)//2`, `left = (w − s)//2`
- **resize** → `cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)` (줄일 때 INTER_AREA)
- **텐서로** → `permute(2, 0, 1).float().div(255).unsqueeze(0)` → **(1, 3, 224, 224)**
- **정규화 (ImageNet 값)** → mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] → `(x − mean.view(3, 1, 1)) / std.view(3, 1, 1)`
- ⚠ `view(3)` 그대로면 마지막 축(W) 과 맞추려다 오류 → **`view(3, 1, 1)`**
- ⚠ `cv2.resize(img, (H, W))` — 크기는 **(W, H)**
- ⚠ 학습 때 쓴 정규화를 추론 때도 **똑같이**

**기본 사용형태**
```python
img = cv2.imread("data/paper_digits/d3_0.jpg")      # BGR uint8 (H, W, 3)
rgb = img[:, :, ::-1]
h, w = rgb.shape[:2]; s = min(h, w)
crop = rgb[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s]
x = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
x = torch.from_numpy(x.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)   # (1, 3, 224, 224)
x = (x - mean.view(3, 1, 1)) / std.view(3, 1, 1)
```

**연결**
- [NN D1·7교시 확률 기초](#/nn/1/7) — 표준화 `(x − μ) / σ` 를 채널마다
- [NN D3·8교시 미니 프로젝트: MNIST 2층 MLP](#/nn/3/8) — `/ 255` 정규화
- [VB D1·7교시 영상 파일 파이프라인](#/vb/1/7) — 영상의 프레임 하나가 곧 이 이미지

**연관 문제**
<<QA 6>>

### 7교시 · 영상 파일 파이프라인

**키워드**: `프레임`, `FPS 30`, `180 프레임`, `VideoCapture`, `VideoWriter`, `fourcc mp4v`, `읽기 → 처리 → 쓰기 루프`, `(W, H)`, `release()`, `H.264 변환`

**개요**: 영상은 **같은 크기 이미지(프레임)의 연속**이다. ① 카메라든 파일이든 `cv2.VideoCapture` 로 열어 한 프레임씩 꺼내는 루프 — 열 때 주소만 바꾸면 카메라에서도 같은 코드가 돈다. ② 오늘은 MNIST 숫자가 지나가는 테스트 영상을 직접 만들고(`VideoWriter`), 다시 읽어 처리한 뒤 결과 영상을 저장한다. ③ 길이(초) = 프레임 수 / FPS, 원본 그대로면 용량이 커서 코덱으로 압축한다.

**핵심 용어**
- **프레임(frame)**: 영상 속 이미지 한 장 — 오늘은 (120, 320, 3) uint8 BGR
- **FPS(frames per second)**: 1초에 보여 주는 프레임 수 — 오늘 **30**
- **코덱(fourcc)**: 프레임을 압축하는 방식의 4글자 이름 — 오늘 `mp4v`
- **H.264**: 브라우저에서 재생되는 대표 코덱 — `mp4v` 를 ffmpeg 로 바꿔 재생

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `cv2.VideoCapture()` | `cv2.VideoCapture("a.mp4")` | 영상 파일(또는 카메라) 열기 |
| `VideoCapture.read()` | `ok, frame = cap.read()` | 프레임 1장 — 끝이면 `ok=False` |
| `VideoCapture.get()` | `cap.get(cv2.CAP_PROP_FPS)` | 프레임 수 · FPS · 크기 읽기 |
| `VideoCapture.release()` | `cap.release()` | 닫기 |
| `cv2.VideoWriter()` | `cv2.VideoWriter(path, fourcc, FPS, (W, H))` | 영상 쓰기 준비 |
| `cv2.VideoWriter_fourcc()` | `cv2.VideoWriter_fourcc(*"mp4v")` | 코덱 코드 만들기 |
| `VideoWriter.write()` | `writer.write(frame)` | 프레임 1장 쓰기 |
| `cv2.rectangle()` | `cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)` | 사각형 그리기 |
| `cv2.putText()` | `cv2.putText(img, text, (x, y), font, scale, color, 2)` | 글자 그리기 |
| `time.perf_counter()` | `time.perf_counter()` | 정밀 시간 재기 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | MNIST test 0~5번 숫자로 가로 띠 (120, 1280) 를 만들고 창이 지나가는 6초 · 30 FPS 영상 digits.mp4 저장 → 다시 열어 프레임 수 · FPS · 크기 확인 → 가운데 상자를 그려 out.mp4 저장 |
| 결과 영상 보기 | ffmpeg 로 H.264 로 바꿔 노트북 안에서 재생 (없으면 프레임 몇 장을 그림으로) |

**핵심 내용**

#### 영상 = 프레임의 연속
> 길이(초) = 프레임 수 / FPS — 오늘 6초 × 30 FPS = **180 프레임**
| 값 | 숫자 |
|---|---|
| 숫자 | MNIST test 0~5번 (7, 2, 1, 0, 4, 1) 을 96×96 으로 키움 |
| 띠 shape | (120, 1280) = 160 × (6 + 2) 칸 |
| 프레임 | 120 × 320 × 3 = **115,200** byte |
| 숫자가 가운데 오는 프레임 | 15, 45, 75, 105, 135, 165 (30 프레임마다) |
- **압축 없이 저장하면** → 180 × 115,200 ≈ **20.7 MB** → 코덱이 필요한 이유
- **`mp4v`** → 브라우저에서 재생 안 되는 경우가 많음 → `ffmpeg -vcodec libx264 -pix_fmt yuv420p` 로 H.264 변환 (없으면 `i % 30 == 15` 프레임을 그림으로)
- **카메라** → `VideoCapture(0)` 처럼 주소만 바꾸면 같은 루프 (프로젝트 과목에서)
##### 자세히
```text
 strip (120 × 1280):  [ 7 ][ 2 ][ 1 ][ 0 ][ 4 ][ 1 ]  …
 window 320 wide slides left ──►  frame 0, 1, 2, … 179
 frame i % 30 == 15  →  a digit is centered
```

#### 읽기 → 처리 → 쓰기 루프
> `while True: ok, frame = cap.read(); if not ok: break; 처리; writer.write(frame)` → `release()`
- **프레임 → 모델 입력** → 흑백 → 가운데 정사각형(120×120) → 28×28 `INTER_AREA` → `float / 255` → `[None, None]` = (1, 1, 28, 28)
- **처리 시간** → `perf_counter()` 로 프레임당 ms → 30 FPS 실시간이면 프레임당 **33 ms** 안
- **가운데 창 상자** → `cv2.rectangle(frame, (100, 0), (219, 119), 색, 2)`
- ⚠ `VideoWriter(..., (H, W))` → 오류 없이 **빈 영상** (크기가 다른 프레임은 조용히 버림) — 크기는 **(W, H)**
- ⚠ `release()` 를 안 함 → 파일이 덜 써져 재생 불가
- ⚠ `cap.read()` 의 `ok` 를 확인하지 않음 → 끝에서 `frame` 이 `None` 이라 다음 줄 오류
- ⚠ OpenCV 색은 **BGR** 순서 — (0, 255, 0) 은 초록, (0, 165, 255) 는 주황
##### 자세히
```text
 VideoCapture ──► read() ──► ok? ──no──► break ──► release() both
                    │ yes
                    ▼
               process frame (draw, infer) ──► VideoWriter.write(frame)
```

**기본 사용형태**
```python
cap = cv2.VideoCapture("digits.mp4")
writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 120))   # (W, H)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.rectangle(frame, (100, 0), (219, 119), (0, 200, 255), 2)
    writer.write(frame)
cap.release(); writer.release()
```

**연결**
- [VB D1·6교시 디지털 이미지: 픽셀 · 채널 · 정규화](#/vb/1/6) — 프레임 하나 = HWC uint8 BGR 이미지
- [VB D1·8교시 미니 프로젝트: 종이 손글씨 · 영상 숫자 인식](#/vb/1/8) — 이 루프에 모델 추론을 끼운다

**연관 문제**
<<QA 7>>

### 8교시 · 미니 프로젝트: 종이 손글씨 · 영상 숫자 인식

**키워드**: `domain shift`, `그냥 줄이기 vs 전처리`, `전처리 6단계`, `조명 보정 divide`, `Otsu 이진화`, `contour · boundingRect`, `20×20 + 무게중심 (14, 14)`, `프레임별 추론`, `신뢰도 · 투표`

**개요**: 오늘 배운 것을 모두 엮는다. ① NN Day 5 의 SimpleConvNet 을 `nn.Module` 로 쓰고 2 epoch 학습해 MNIST test 0.98 이상. ② 그런데 종이 사진을 그냥 28×28 로 줄여 넣으면 거의 못 맞힌다 — MNIST 는 검은 배경 · 흰 글씨 · 20×20 상자 · 무게중심이 가운데인 데이터이기 때문(**domain shift**). ③ 입력을 학습 데이터 모양으로 맞추는 **전처리 6단계**로 해결하고, 같은 모델로 7교시 영상을 프레임마다 인식한다.

**핵심 용어**
- **도메인 이동(domain shift)**: 학습 데이터와 실제 입력의 분포가 다른 문제 — "실험실에선 되는데 현장에서 안 된다"의 가장 흔한 원인
- **Otsu 이진화**: 밝기 분포로 기준값을 자동으로 정하는 흑/백 나누기
- **윤곽(contour) · bounding box**: 덩어리의 테두리 · 그것을 감싸는 최소 직사각형
- **무게중심(moment)**: 픽셀 밝기의 평균 위치 — MNIST 는 (14, 14) 에 오도록 정렬됨
- **신뢰도(confidence)**: softmax 최대 확률 — 낮으면 예측을 믿지 않는 기준으로

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `cv2.GaussianBlur()` | `cv2.GaussianBlur(g, (0, 0), 25)` | 가우시안 흐리기 (σ=25 면 배경 추정) |
| `cv2.divide()` | `cv2.divide(g, bg, scale=255)` | 배경으로 나눠 조명 평평하게 |
| `cv2.threshold()` | `cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)` | 반전 + Otsu 이진화 |
| `cv2.morphologyEx()` | `cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)` | 가는 줄 · 점 지우기 (열림) |
| `cv2.findContours()` | `cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` | 덩어리 윤곽 찾기 |
| `cv2.boundingRect()` | `cv2.boundingRect(c)` | 윤곽의 (x, y, w, h) |
| `cv2.moments()` | `cv2.moments(canvas)` | 무게중심 계산용 모멘트 |
| `cv2.warpAffine()` | `cv2.warpAffine(img, M, (28, 28))` | 평행이동 등 변환 |
| `torch.softmax()` | `torch.softmax(logits, dim=1)` | 확률로 바꾸기 (신뢰도) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 과제 1 — SimpleConvNet | 1교시 구조를 `nn.Module` 클래스로 (계층 이름 conv · fc1 · fc2) — 파라미터 433,890, (2, 1, 28, 28) → (2, 10) |
| 과제 2 — 학습 | 5교시 `train_one_epoch` · `evaluate` 그대로, Adam 0.001 · 배치 100 · 2 epoch → test 0.98 이상 (GPU epoch 당 약 10초) |
| 사진을 그냥 넣으면? | 종이 사진을 흑백 → 28×28 → ÷255 만 해서 넣고 예측 — 흰 배경이라 거의 못 맞힘 |
| 과제 3 — preprocess | 6단계 전처리 → (1, 1, 28, 28) float32 0~1, 무게중심 (14, 14) ± 1 · 긴 변 17~22 |
| 과제 4 — 30장 정확도 비교 | naive 와 preprocess 로 30장 예측 (정답 = 파일 이름 두 번째 글자) → 전처리 0.8 이상이고 +0.3 이상 |
| 과제 5 — 영상 숫자 인식 | digits.mp4 180 프레임을 인식해 글자를 넣은 digits_pred.mp4 저장, 가운데 프레임 6장 예측이 모두 정답 |

**핵심 내용**

#### 같은 모델, 다른 세상 — domain shift
> 모델이 배운 세상(MNIST)과 입력(종이 사진)이 다르면 아무리 좋은 모델도 틀린다
| 성질 | MNIST | 종이 사진 |
|---|---|---|
| 배경 · 글씨 | 검은 배경 · 흰 글씨 | 흰 종이 · 어두운 글씨 |
| 크기 · 위치 | 20×20 상자, 무게중심 가운데 | 작고 한쪽에 치우침 |
| 잡음 | 없음 | 조명 · 그림자 · 줄 · 흐림 |
- **해결 방향 ①** → 입력을 학습 데이터 쪽으로 옮기기 = **전처리** (오늘)
- **해결 방향 ②** → 모델이 실제 데이터를 보고 배우게 = 데이터 증강(Day 2) · 전이학습(Day 3)
- **과제 배점** → SimpleConvNet 20 · 학습 25 · preprocess 25 · 30장 비교 10 · 영상 10 · 해석 10 (강사)

#### 전처리 6단계 — 종이 사진 → MNIST 모양
> 흑백 → 조명 보정 → 반전 · 이진화 → 자르기 → 20×20 → 무게중심
| 단계 | 코드 | 이유 |
|---|---|---|
| ① 흑백 | `cvtColor(BGR2GRAY)` | 모델 입력 1채널 |
| ② 조명 보정 | `bg = GaussianBlur(g, (0,0), 25)`; `divide(g, bg, scale=255)` | 아주 크게 흐린 이미지 = 조명 · 그림자만 남은 배경 → 나누면 밝기가 평평 |
| ③ 흐리기 · 반전 · 이진화 | `GaussianBlur((5,5))` + `threshold(BINARY_INV + OTSU)` + `MORPH_OPEN (3×3)` | 흰 종이 → 검은 배경, 글씨만 255 (기준값은 Otsu 가 자동) |
| ④ 숫자만 자르기 | `max(contours, key=cv2.contourArea)` → `boundingRect` | 줄 · 얼룩 · 여백 제거 |
| ⑤ 20×20 에 맞추기 | `scale = 20.0 / max(w, h)`, 비율 유지 resize | MNIST 규칙 |
| ⑥ 28×28 가운데 + 무게중심 | `cx = m10/m00`, `cy = m01/m00`, `M = [[1,0,14−cx],[0,1,14−cy]]`, `warpAffine` | MNIST 규칙 |
- **반환** → (1, 1, 28, 28) float32 0~1
- **그냥 줄이기(naive)** → 흑백 → `resize(28, 28)` → `/255` 만 — 흰 배경이라 MNIST 와 반대
- ⚠ 조명 보정 두 줄을 지우면 그림자 낀 사진이 무너진다 (이진화 기준이 조명에 끌려감)
- ⚠ 전처리 후에도 틀리는 사진 → 변환 그림을 보고 원인(획 끊김 · 모양) 추측
##### 자세히
```text
 photo ─gray─► ─÷ blurred bg─► flat light ─INV+OTSU─► white digit on black
   ─largest contour─► crop ─long side 20─► 20×20 ─center of mass → (14,14)─► 28×28
```

#### 영상 숫자 인식 — 프레임 루프 + 추론
> 7교시 루프 안에서: 프레임 → 입력 변환 → `predict` → 결과 그리기 → write
- **`predict`** → `eval()` + `no_grad()` + `softmax(...)[0].cpu()` → (예측, 신뢰도)
- **그리기** → 신뢰도 > **0.9** 면 초록 상자, 아니면 주황 + `putText(f"{pred} ({conf:.2f})")`
- **채점** → 가운데 프레임 `30k + 15` 의 예측을 정답 [7, 2, 1, 0, 4, 1] 과 비교
- **두 숫자 사이 프레임** → 신뢰도가 낮고 예측이 흔들림 → 실제 서비스라면 **신뢰도 문턱 · 연속 프레임 다수결(투표)**

**기본 사용형태**
```python
def predict(model, x):
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(x.to(device)), dim=1)[0].cpu()
    return int(p.argmax()), float(p.max())

g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
bg = cv2.GaussianBlur(g, (0, 0), 25)
g = cv2.divide(g, bg, scale=255)                       # 조명 평평하게
_, bw = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

**연결**
- [VB D1·5교시 학습 루프 · 평가 · 저장](#/vb/1/5) — `train_one_epoch` · `evaluate` 를 그대로 사용
- [VB D1·7교시 영상 파일 파이프라인](#/vb/1/7) — 프레임 루프에 추론을 끼운다
- [NN D5·5교시 SimpleConvNet 으로 MNIST 학습](#/nn/5/5) — NumPy 로 만든 같은 구조 (433,890)
- [VB D2·7교시 학습 기법: 데이터 증강 · 학습률 스케줄 · 혼동 행렬 · 연산량](#/vb/2/7) — 반대 방향의 해결 (증강)

**연관 문제**
<<QA 8>>
