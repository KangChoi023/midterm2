## Day 1 — NumPy 에서 PyTorch 로, 그리고 영상 데이터

### 1교시 · NN 복습 퀴즈 · mynn 에서 PyTorch 로

**키워드**: `mynn ↔ PyTorch 대응`, `nn.Module`, `nn.Sequential`, `(N, C, H, W)`, `433,890`, `CrossEntropyLoss`, `Colab GPU`

**개요**: NN 과정에서 NumPy 로 직접 만든 부품(Affine · Conv · Pooling · 손실 · 옵티마이저 · 미니배치 루프)은 PyTorch 에 이미 모두 들어 있다. 이번 과정은 그 부품으로 더 큰 모델 · 실제 이미지 · 영상으로 간다. NN Day 5 의 SimpleConvNet 은 `nn.Sequential` 로 계층을 늘어놓기만 하면 되고 파라미터 수(433,890)도 똑같다.

**핵심 용어**
- **텐서(Tensor)**: `np.ndarray` 의 PyTorch 판 — GPU 에 올릴 수 있고 연산 기록(autograd)을 남긴다
- **autograd**: 순전파 연산을 기록했다가 `loss.backward()` 한 줄로 역전파를 자동 계산하는 기능
- **nn.Module**: 모델 · 계층의 부모 클래스 — `parameters()` 로 학습할 값을 자동으로 모아 준다
- **logits**: softmax 를 거치기 전 출력 — `nn.CrossEntropyLoss` 가 softmax 를 포함하므로 모델 끝에 softmax 를 붙이지 않는다

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

**핵심 내용**

#### mynn.py ↔ PyTorch 대응표
> NumPy 로 손수 짠 것 = PyTorch 한 줄
| NN 과정 (mynn · NumPy) | PyTorch | 달라지는 점 |
|---|---|---|
| `np.ndarray` | `torch.Tensor` | GPU · 연산 기록 |
| 계층마다 `backward()` | `loss.backward()` | 계산 그래프를 따라 자동 미분 |
| `Affine(784, 100)` | `nn.Linear(784, 100)` | 같은 Y = XW + b (저장은 W 가 (out, in)) |
| `Convolution(1, 30, 5)` | `nn.Conv2d(1, 30, 5)` | im2col 은 내부 처리 |
| `Pooling(2, 2, 2)` | `nn.MaxPool2d(2)` | 같음 |
| `SoftmaxWithLoss()` | `nn.CrossEntropyLoss()` | logits 를 그대로 넣는다 |
| `Adam(lr=0.001)` + `update()` | `torch.optim.Adam(model.parameters(), lr=1e-3)` + `step()` | `zero_grad()` 단계가 추가 |
| `rng.permutation` 미니배치 루프 | `DataLoader(batch_size=100, shuffle=True)` | 섞기 · 자르기 · 병렬 로딩 |
| `np.savez` | `torch.save(model.state_dict(), path)` | 이름 → 텐서 딕셔너리 |

#### SimpleConvNet 을 PyTorch 로
> Conv(1→30, 5×5) → ReLU → Pool(2×2) → Affine(4320→100) → ReLU → Affine(100→10)
| 층 | 출력 shape | 파라미터 |
|---|---|---|
| 입력 | (N, 1, 28, 28) | — |
| Conv2d(1, 30, 5) | (N, 30, 24, 24) | 30·1·5·5 + 30 = 780 |
| ReLU · MaxPool2d(2) | (N, 30, 12, 12) | 0 |
| Flatten | (N, 4320) | 0 (30·12·12 = 4320) |
| Linear(4320, 100) | (N, 100) | 4320·100 + 100 = 432,100 |
| Linear(100, 10) | (N, 10) | 100·10 + 10 = 1,010 |
| **합계** | | **433,890** (NumPy 판과 같음) |
- 1 epoch test 정확도 0.9810 (NumPy 판 기록)
- ⚠ 모델 끝에 softmax 를 붙이면 안 된다 — `CrossEntropyLoss` 안에 이미 있다 (두 번 적용)
- ⚠ Colab 환경 셀이 `device : cpu` 면 런타임 유형을 **T4 GPU** 로 — 오늘 실습은 CPU 로도 돌지만 8교시 학습이 몇 배 느리다
- ⚠ 런타임이 끊긴 뒤 `NameError` → 맨 위 환경 셀부터 다시 실행

##### 자세히
- 이미지 텐서의 모양 `(N, C, H, W)` 는 NN 과정과 같다 (배치 · 채널 · 높이 · 너비).
- 이번 과정 5일: Day 1 PyTorch 전환 · 이미지 · 영상 → Day 2 CNN 대표 구조 (LeNet · VGG · ResNet) → Day 3 경량 CNN · 전이학습 · Grad-CAM → Day 4 객체 탐지 (IoU · NMS · mAP · YOLO) → Day 5 세그멘테이션 · 포즈 · ONNX · 성능 측정.

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
- [NN D5·5교시 SimpleConvNet](#/nn/5/5) — 같은 구조를 NumPy 로 만들었던 곳
- [NN D3·5교시 Affine 계층](#/nn/3/5) — `nn.Linear` 가 대신하는 계층

### 2교시 · Tensor · GPU

**키워드**: `torch.Tensor`, `float32`, `dim`, `from_numpy`(공유), `torch.tensor`(복사), `reshape·view·permute·unsqueeze`, `.to(device)`

**개요**: Tensor 는 ndarray 에 `device`(cpu/cuda) 와 `requires_grad` 를 더한 것. 함수 이름은 NumPy 와 거의 같고 축 인자만 `axis` → `dim`, `keepdims` → `keepdim`. 기본 실수형은 float32 (NumPy 는 float64). GPU 는 행렬곱을 한꺼번에 계산하지만 데이터와 모델이 **같은 device** 에 있어야 한다.

**핵심 용어**
- **device**: 텐서가 놓인 곳 — `cpu` 또는 `cuda`(GPU 메모리)
- **requires_grad**: 이 텐서로 한 연산을 기록해 나중에 미분할지 여부 (3교시)
- **메모리 공유(share)**: `from_numpy` 가 같은 메모리를 보게 하는 것 — 한쪽을 고치면 다른 쪽도 바뀐다

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

**핵심 내용**

#### NumPy → PyTorch 이름 바꾸기
> `axis→dim` · `keepdims→keepdim` · `np.exp→torch.exp` · `max(...)` 는 `.values` 가 붙는다
| NumPy | PyTorch |
|---|---|
| `Y.max(axis=1, keepdims=True)` | `Y.max(dim=1, keepdim=True).values` |
| `e.sum(axis=1, keepdims=True)` | `e.sum(dim=1, keepdim=True)` |
| `a.dtype` 기본 float64 | `t.dtype` 기본 float32 |
- **softmax 순전파**: `Y = X @ W + b` → `Y - Y.max` (오버플로 방지) → `exp` → `/ 합` — NN Day 2 와 같은 식

#### from_numpy vs tensor
| 호출 | 메모리 | 한쪽을 고치면 |
|---|---|---|
| `torch.from_numpy(a)` | 공유 | 서로 보인다 |
| `torch.tensor(a)` | 복사 | 영향 없음 |
- 예: `a = np.zeros(3)`, `b = from_numpy(a)`, `c = tensor(a)` 에서 `b[0]=7`, `c[1]=9` → `a = [7, 0, 0]` · `c = [0, 9, 0]`

#### shape 바꾸기
| 함수 | 하는 일 | 쓰는 곳 |
|---|---|---|
| `reshape` · `view` | 원소 순서 그대로, 모양만 | Flatten |
| `permute(2, 0, 1)` | 축 순서 변경 (배치가 달라짐) | HWC → CHW |
| `unsqueeze(0)` / `squeeze(0)` | 크기 1 축 넣기 / 빼기 | 이미지 1장 → 배치 |
- `x = arange(24).reshape(2,3,4)` → `x.permute(0,2,1)` 은 (2,4,3), `p[1,3,2] == x[1,2,3] == 23`; `x.reshape(2,-1)` 은 (2,12) — `-1` 은 "나머지 자동"

#### GPU 속도 재기
- 2048×2048 행렬곱 = 2·2048³ ≈ 17.2 GFLOP, 텐서 하나 16 MiB (float32)
- 첫 실행은 초기화 시간이 섞이므로 준비 운동 1회 후 `reps=10` 평균
- ⚠ GPU 는 비동기 — `torch.cuda.synchronize()` 없이 재면 실제보다 훨씬 빠르게 나온다
- ⚠ `float64` 배열을 `from_numpy` 로 그대로 모델에 넣으면 dtype 오류 → `.float()`
- ⚠ 모델은 GPU, 데이터는 CPU → `Expected all tensors to be on the same device` → 둘 다 `.to(device)`
- ⚠ GPU 텐서는 바로 `.numpy()` 불가 → `.cpu().numpy()`

**기본 사용형태**
```python
X = torch.from_numpy(X_np).float()        # float64 → float32
layer = nn.Linear(784, 10).to(device)
Y = layer(X.to(device))
print(Y.cpu().detach().numpy().shape)
```

**연결**
- [NN D1·3교시 NumPy ndarray](#/nn/1/3) — `b = a` 는 복사가 아니라는 이야기와 같은 뿌리
- [VB D1·3교시 autograd](#/vb/1/3) — `requires_grad` 가 쓰이는 곳

### 3교시 · autograd: 역전파를 자동으로

**키워드**: `requires_grad`, `계산 그래프`, `backward()`, `.grad 누적`, `zero_grad()`, `no_grad()`, `detach()`

**개요**: NN Day 3 에서는 계층마다 `backward()` 를 손으로 짜고 gradient check 로 확인했다. PyTorch 는 `requires_grad=True` 인 텐서로 한 연산을 그래프로 기록해 두었다가 `loss.backward()` 한 번에 연쇄 법칙을 거꾸로 적용한다. 결과는 잎(leaf) 텐서의 `.grad` 에 **더해진다**. 학습 한 step = `zero_grad() → backward() → step()`.

**핵심 용어**
- **계산 그래프(computational graph)**: 연산을 노드로 기록한 구조 — 각 노드가 미분에 필요한 값을 저장
- **잎 텐서(leaf)**: 사용자가 직접 만든 `requires_grad=True` 텐서 — `.grad` 가 여기에 쌓인다
- **기울기 누적**: `backward()` 를 부를 때마다 `.grad` 에 더해지는 성질 — 버그가 아니라 기능

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

**핵심 내용**

#### 손으로 미분 vs autograd
> f = x² + 3y² 에서 (x, y) = (1, 2) → f = 13, df/dx = 2x = 2, df/dy = 6y = 12
- **`f.backward()`** → `x.grad = 2`, `y.grad = 12` (f 가 **스칼라**여야 바로 호출 가능 — 손실은 언제나 스칼라)
- **Softmax-with-Loss + Affine** → `dW = Xᵀ (y − t) / N` — 손으로 유도한 식과 autograd 결과가 같다

#### 기울기 누적
| 호출 순서 | `x.grad` (x=1, f=x²) |
|---|---|
| 1번째 `backward` | 2 |
| 비우지 않고 2번째 | **4** (더해짐) |
| `grad.zero_()` 후 3번째 | 2 |
- 누적은 큰 배치를 여러 번에 나눠 기울기를 모을 때 쓰는 기능 → 그래서 학습 루프는 매번 직접 비운다
- ⚠ `zero_grad()` 를 빼먹으면 기울기가 계속 더해져 학습이 폭주
- ⚠ 그래프가 붙은 텐서를 `.numpy()` → `Can't call numpy() on Tensor that requires grad` → `.detach().numpy()`
- ⚠ 평가 · 추론에서 기록을 안 끄면 메모리 낭비 → `with torch.no_grad():`
- ⚠ 직접 갱신 `p -= lr * p.grad` 는 `no_grad` 안에서 (없으면 leaf 변수 in-place 오류)

#### 직접 경사하강 vs optimizer
- f = p₀² + 3p₁², 시작 (−4, 2), lr = 0.1, 100 step → 매 step p₀ 는 ×0.8, p₁ 은 ×0.4 → 끝 위치는 거의 (0, 0)
- `opt = SGD([q], lr=0.1)` 의 `zero_grad → backward → step` 이 위 직접 루프와 같다 (NN 의 `optimizer.update(params, grads)` = `step()`)

**기본 사용형태**
```python
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
f = x ** 2 + 3 * y ** 2
f.backward()
print(x.grad.item(), y.grad.item())     # 2.0 12.0
```

**연결**
- [NN D3·5교시 Affine 계층](#/nn/3/5) — 손으로 짠 dW = Xᵀ · dY
- [VB D1·5교시 학습 루프](#/vb/1/5) — `zero_grad → backward → step` 이 쓰이는 곳

### 4교시 · nn.Module · Dataset · DataLoader

**키워드**: `nn.Module`, `__init__ / forward`, `parameters()`, `Dataset`, `DataLoader`, `transforms.ToTensor`, `600 iteration`

**개요**: 모델은 `nn.Module` 을 상속한 클래스 — `__init__` 에서 계층을 만들고 `forward` 에서 데이터가 지나가는 길을 쓴다. 데이터는 `Dataset`(`__len__` · `__getitem__`) 이 한 장씩 주고 `DataLoader` 가 섞어서 `batch_size` 개씩 쌓아 준다. MNIST 는 배치 100 이면 epoch 당 600 iteration.

**핵심 용어**
- **Dataset**: `len(ds)` 와 `ds[i]` (i 번째 이미지, 레이블) 만 알면 되는 객체
- **DataLoader**: 인덱스를 섞고(`shuffle`) 배치로 묶어 텐서 하나로 쌓아 주는 반복자
- **iteration / epoch**: 배치 1번 처리 / 전체 데이터 1바퀴 — 60000 ÷ 100 = 600

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

**핵심 내용**

#### 모델 클래스의 모양
> `__init__` = 부품 만들기, `forward` = 연결하기, `parameters()` 는 자동 수집
- MLP: `fc1 = Linear(784, 100)`, `fc2 = Linear(100, 10)` → 파라미터 784·100+100 + 100·10+10 = **79,510**
- `forward` 첫 줄 `x.flatten(1)` → (N, 1, 28, 28) → (N, 784), **배치 축은 남긴다**
- 마지막은 logits 그대로 — softmax 는 손실 함수 안에서

#### Dataset → DataLoader
| 단계 | 하는 일 | NN 과정에서 |
|---|---|---|
| `Dataset[i]` | (이미지 i, 레이블 i) 한 쌍 | `x_train[i]` |
| `shuffle=True` | 에폭마다 순서 섞기 | `rng.permutation(N)` |
| `batch_size=100` | 100개씩 텐서 하나로 | 슬라이스 `idx[i:i+100]` |
- MNIST train 60,000 / test 10,000, 한 장은 (1, 28, 28) float32 0~1
- 학습 `batch_size=100` → 600 iteration/epoch, 평가 `batch_size=1000, shuffle=False` → 10 iteration
- 직접 만든 Dataset: `__getitem__` 에서 `torch.from_numpy(X[i]).float()/255` 후 `unsqueeze(0)` (채널 축) → `(x, int(y[i]))`

#### CrossEntropyLoss 를 풀어 쓰면
> `loss = −log_softmax(logits)[i, 정답 i]` 의 평균
- `-F.log_softmax(logits, dim=1)[torch.arange(N), yb].mean()` 이 `nn.CrossEntropyLoss()(logits, yb)` 와 같다 (팬시 인덱싱 — NN Day 2 와 같은 방법)
- ⚠ 레이블은 **long(int64)** 정수 — float 로 주면 오류
- ⚠ 평가 로더는 `shuffle=False` (순서가 의미 있을 때)

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
- [NN D2·6교시 출력층과 손실 함수](#/nn/2/6) — 팬시 인덱싱으로 정답 칸 고르기

### 5교시 · 학습 루프 · 평가 · 저장

**키워드**: `train()/eval()`, `zero_grad→backward→step`, `no_grad`, `argmax`, `state_dict`, `torch.save/load`

**개요**: 모든 PyTorch 학습 코드는 같은 모양이다 — 모델이 ResNet 이든 YOLO 든 바뀌는 건 모델 · 데이터 · 손실 함수뿐. 평가는 `eval()` + `no_grad()` + `argmax`. 저장은 모델 객체가 아니라 `state_dict()`(파라미터)만.

**핵심 용어**
- **학습 모드 / 평가 모드**: `model.train()` / `model.eval()` — Dropout · BatchNorm 동작이 바뀐다
- **state_dict**: 계층 이름 → 파라미터 텐서 딕셔너리 — 저장 · 불러오기의 표준 단위

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

#### 평가와 저장
- **평가**: `model.eval()` + `with torch.no_grad():` + `pred = model(xb).argmax(dim=1)` + `(pred == yb).sum().item()` → 정확도 = 맞은 수 / 전체
- `eval()` 과 `no_grad()` 는 **하는 일이 다르다** — 전자는 Dropout · BN 동작, 후자는 기울기 기록
- **저장**: `torch.save(model.state_dict(), "mlp.pt")` → 불러오기: 같은 클래스로 새 모델 → `load_state_dict(torch.load(..., map_location=device))` → 정확도가 저장 전과 같아야 한다
- ⚠ 평가 뒤 학습을 이어 가는데 `model.train()` 을 안 부름 → Dropout · BN 이 꺼진 채 학습 (`train_one_epoch` 첫 줄에 두면 안전)
- ⚠ `loss` 텐서를 그대로 리스트에 모음 → 그래프가 붙어 메모리 누수 → `loss.item()`
- ⚠ 옵티마이저를 만든 **뒤에** `model.to(device)` → 순서 지킬 것 (모델을 먼저 옮긴다)

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
- [NN D3·8교시 MNIST 2층 MLP](#/nn/3/8) — 직접 짠 에폭 루프
- [VB D1·8교시 미니 프로젝트](#/vb/1/8) — 같은 두 함수를 CNN 에 그대로 사용

### 6교시 · 디지털 이미지: 픽셀 · 채널 · 정규화

**키워드**: `HWC ↔ CHW`, `BGR → RGB`, `uint8 overflow`, `흑백 가중치`, `crop→resize→ToTensor→Normalize`, `cv2.resize (W, H)`

**개요**: 사진이 모델 입력 텐서가 되기까지의 변환. OpenCV · NumPy 는 `(H, W, C)` uint8 **BGR**, PyTorch 는 `(C, H, W)` float32 RGB 0~1 에 평균 · 표준편차 정규화, 모델에는 배치 축을 붙인 `(N, C, H, W)`. 학습 때의 전처리와 추론 때의 전처리가 같아야 한다.

**핵심 용어**
- **채널(channel)**: 같은 크기의 밝기 판 — 컬러 사진은 B · G · R 3장의 겹침
- **uint8 overflow**: 0~255 정수끼리 더해 255 를 넘으면 0 부터 다시 세는 현상
- **정규화(normalize)**: `(x − mean) / std` 를 채널마다 — NN Day 1 표준화의 이미지 판
- **포화(saturation)**: 255 를 넘는 값을 255 에서 멈추게 하는 연산 (`cv2.add`)

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

**핵심 내용**

#### 두 세계의 약속
| | OpenCV · NumPy | PyTorch |
|---|---|---|
| 모양 | (H, W, C) | (C, H, W) → 배치 (N, C, H, W) |
| 값 | uint8 0~255 | float32 0~1 (+ 정규화) |
| 채널 순서 | **BGR** | **RGB** |
- `rgb = img[:, :, ::-1]` — 마지막 축을 거꾸로 (B,G,R → R,G,B)
- `torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255` = `transforms.ToTensor()(rgb)` (음수 stride 때문에 `.copy()`)
- `permute(2, 0, 1)`: 새 0번 = 옛 2번(C), 새 1번 = 옛 0번(H), 새 2번 = 옛 1번(W)

#### 흑백 = 채널의 가중 평균
> gray = 0.299 R + 0.587 G + 0.114 B  (가중치 합 1, 사람 눈은 초록에 가장 민감)
- `cv2.cvtColor(img, BGR2GRAY)` 와 직접 계산이 같다 — 단 직접 계산은 **float32 로 바꾼 뒤**
- 파란 잉크는 B 채널에서 덜 어둡고 R 채널에서 가장 어둡다

#### uint8 overflow
| 계산 | 결과 |
|---|---|
| `uint8(200) + 100` | **44** (300 − 256) |
| `uint8(250) + 100` | **94** (350 − 256) |
| `float32 + 100` | 300 · 350 |
| `cv2.add(p, 100)` | 255 · 255 (포화) |
- ⚠ 이미지 계산은 float 로 바꾼 뒤 → 필요하면 다시 uint8

#### 입력 만들기 파이프라인
> crop → resize → ToTensor → Normalize → unsqueeze(0)
- 가운데 정사각형 crop: `s = min(h, w)`, `top = (h − s)//2`, `left = (w − s)//2`
- `cv2.resize(crop, (224, 224), INTER_AREA)` → `permute(2,0,1).float().div(255).unsqueeze(0)` → (1, 3, 224, 224)
- **정규화** (ImageNet 값): mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] → `(x − mean.view(3,1,1)) / std.view(3,1,1)`
- ⚠ `view(3)` 그대로면 마지막 축(W) 과 맞추려다 오류 → `view(3, 1, 1)`
- ⚠ BGR 을 RGB 로 안 바꿔 사전학습 모델에 넣으면 **오류 없이 정확도만 조용히 떨어진다**
- ⚠ `cv2.resize(img, (H, W))` — 크기는 (W, H)
- ⚠ 학습 때 쓴 정규화를 추론 때도 똑같이

**기본 사용형태**
```python
img = cv2.imread("data/paper_digits/d3_0.jpg")      # BGR uint8 (H, W, 3)
rgb = img[:, :, ::-1]
h, w = rgb.shape[:2]; s = min(h, w)
crop = rgb[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s]
x = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
x = torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)   # (1, 3, 224, 224)
```

**연결**
- [NN D1·7교시 확률 기초 (표준화)](#/nn/1/7) — `(x − μ) / σ` 를 채널마다 한 것
- [VB D1·7교시 영상 파일](#/vb/1/7) — 영상의 프레임 하나가 곧 이 이미지
- [VB D1·8교시 종이 손글씨](#/vb/1/8) — 전처리를 직접 만드는 곳

### 7교시 · 영상 파일 파이프라인

**키워드**: `프레임`, `FPS`, `VideoCapture`, `VideoWriter`, `fourcc/mp4v`, `읽기→처리→쓰기 루프`, `release()`

**개요**: 영상 = 같은 크기 이미지(프레임)의 연속. 카메라든 파일이든 `cv2.VideoCapture` 로 열어 한 프레임씩 꺼내는 루프이고, 열 때 주소만 바꾸면 카메라에서도 같은 코드가 돈다. 결과는 `VideoWriter` 로 저장한다. 길이(초) = 프레임 수 / FPS.

**핵심 용어**
- **프레임(frame)**: 영상 속 이미지 한 장 — 오늘은 (120, 320, 3) uint8 BGR
- **FPS**: 1초에 보여 주는 프레임 수 — 오늘 30
- **코덱(fourcc)**: 프레임을 압축하는 방식의 4글자 이름 — 오늘 `mp4v`

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

**핵심 내용**

#### 오늘의 테스트 영상
> MNIST test 0~5번 숫자 (7, 2, 1, 0, 4, 1) 를 96×96 으로 키워 가로 띠로 이어 붙이고, 창이 왼쪽으로 지나간다
| 값 | 숫자 |
|---|---|
| 띠 shape | (120, 1280) = 160 × (6 + 2) 칸 |
| 프레임 | 120 × 320 × 3 = 115,200 byte |
| 길이 | 6초 × 30 FPS = **180 프레임** |
| 숫자가 가운데 오는 프레임 | 15, 45, 75, 105, 135, 165 (30 프레임마다) |
- 원본 그대로 저장하면 180 × 115,200 ≈ 20.7 MB → 코덱 압축이 필요한 이유
- `mp4v` 는 브라우저에서 재생이 안 되는 경우가 많다 → `ffmpeg -vcodec libx264 -pix_fmt yuv420p` 로 H.264 로 바꿔 재생 (ffmpeg 없으면 `i % 30 == 15` 프레임 몇 장을 그림으로)

#### 읽기 → 처리 → 쓰기 루프
> `while True: ok, frame = cap.read(); if not ok: break; 처리; writer.write(frame)`
- **프레임 → 모델 입력**: 흑백 → 가운데 정사각형(120×120) → 28×28 `INTER_AREA` → `float/255` → `[None, None]` = (1, 1, 28, 28)
- 처리 시간 `t0 = perf_counter()` 로 프레임당 ms 를 재면 실시간 가능 여부(30 FPS = 33 ms/프레임) 가늠
- 영상에 상자 `cv2.rectangle(frame, (100, 0), (219, 119), 색, 2)` — 가운데 창
- ⚠ `VideoWriter(..., (H, W))` → 오류 없이 **빈 영상** (조용히 버림) — 크기는 **(W, H)**
- ⚠ `release()` 를 안 함 → 파일이 덜 써져 재생 불가
- ⚠ `cap.read()` 의 `ok` 를 확인하지 않음 → 끝에서 `frame` 이 `None` 이라 다음 줄 오류
- ⚠ OpenCV 색은 **BGR** 순서 — (0, 255, 0) 은 초록, (0, 165, 255) 는 주황

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
- [VB D1·6교시 디지털 이미지](#/vb/1/6) — 프레임 하나 = HWC uint8 BGR 이미지
- [VB D1·8교시 영상 숫자 인식](#/vb/1/8) — 이 루프에 모델 추론을 끼운다

### 8교시 · 미니 프로젝트: 종이 손글씨 · 영상 숫자 인식

**키워드**: `domain shift`, `전처리 6단계`, `조명 보정(divide)`, `Otsu 이진화`, `contour/boundingRect`, `20×20 + 무게중심`, `프레임별 추론`

**개요**: MNIST 로 99% 가까이 맞히는 SimpleConvNet 도 종이 사진을 그냥 28×28 로 줄여 넣으면 거의 못 맞힌다 — MNIST 는 검은 배경 · 흰 글씨 · 20×20 상자 · 무게중심이 가운데인 데이터이기 때문. 입력을 학습 데이터 모양으로 맞추는 **전처리**로 해결하고, 같은 모델로 7교시 영상을 프레임마다 인식한다.

**핵심 용어**
- **도메인 이동(domain shift)**: 학습 데이터와 실제 입력의 분포가 다른 문제 — "실험실에선 되는데 현장에서 안 된다"의 가장 흔한 원인
- **Otsu 이진화**: 밝기 분포로 기준값을 자동으로 정하는 흑/백 나누기
- **bounding box**: 덩어리를 감싸는 최소 직사각형
- **무게중심(moment)**: 픽셀 밝기의 평균 위치 — MNIST 는 (14, 14) 에 오도록 정렬됨

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

**핵심 내용**

#### 과제 구성 (배점 90 + 강사 10)
| 과제 | 내용 | 배점 |
|---|---|---|
| 1 | SimpleConvNet 을 `nn.Module` 로 (계층 이름 `conv`·`fc1`·`fc2`, 433,890) | 20 |
| 2 | 2 epoch 학습, test ≥ 0.98 | 25 |
| 3 | 사진 전처리 함수 `preprocess` | 25 |
| 4 | 30장 정확도 — 그냥 줄이기 vs 전처리 (전처리 ≥ 0.8 이고 +0.3 이상) | 10 |
| 5 | 영상 프레임 인식 · 결과 영상 (180 프레임, 가운데 6장 모두 정답) | 10 |
| 6 | 결과 해석 (마크다운, 강사 채점) | 10 |

#### SimpleConvNet 클래스 (과제 1 빈칸)
- `fc1 = nn.Linear(4320, 100)` · `x = self.pool(F.relu(self.conv(x)))` (→ (N, 30, 12, 12)) · `return self.fc2(x)`

#### 전처리 6단계 (과제 3)
> 종이 사진 → MNIST 모양: 흑백 → 조명 보정 → 반전·이진화 → 자르기 → 20×20 → 무게중심
| 단계 | 코드 | 이유 |
|---|---|---|
| ① 흑백 | `cvtColor(BGR2GRAY)` | 모델 입력 1채널 |
| ② 조명 보정 | `bg = GaussianBlur(g, (0,0), 25)`; `divide(g, bg, scale=255)` | 아주 크게 흐린 이미지 = 조명 · 그림자만 남은 배경 → 원본을 배경으로 나눠 평평하게 |
| ③ 흐리기 · 반전 · 이진화 | `GaussianBlur((5,5))` + `threshold(THRESH_BINARY_INV + THRESH_OTSU)` + `MORPH_OPEN (3×3)` | 흰 종이 → 검은 배경, 글씨만 255 |
| ④ 숫자만 자르기 | `max(contours, key=cv2.contourArea)` → `boundingRect` | 줄 · 얼룩 · 여백 제거 |
| ⑤ 20×20 에 맞추기 | `scale = 20.0 / max(w, h)`, 비율 유지 resize | MNIST 규칙 |
| ⑥ 28×28 가운데 + 무게중심 | `cx = m10/m00`, `cy = m01/m00`, `M = [[1,0,14−cx],[0,1,14−cy]]`, `warpAffine` | MNIST 규칙 |
- 반환: `(1, 1, 28, 28)` float32 0~1 — 채점은 무게중심이 (14, 14) ± 1, 긴 변 17~22 인지 확인
- 그냥 줄이기(`naive`) = 흑백 → `resize(28, 28)` → `/255` 만 — 흰 배경이라 MNIST 와 반대

#### 영상 인식 (과제 5)
- 프레임마다 `predict(cnn, 프레임 → 입력)` → 신뢰도 > 0.9 면 초록 상자, 아니면 주황 + `putText(f"{pred} ({conf:.2f})")` → `writer.write`
- 숫자가 가운데 온 프레임 `centers = [30k + 15]` 의 예측을 정답 `[7, 2, 1, 0, 4, 1]` 과 비교
- 두 숫자 사이 프레임은 신뢰도가 낮고 예측이 흔들린다 → 실제 서비스라면 **신뢰도 문턱 · 연속 프레임 다수결(투표)** 로 처리
- ⚠ `predict` 는 `eval()` + `no_grad()` + `softmax(...)[0].cpu()`
- ⚠ 조명 보정 두 줄을 지우면 그림자 낀 사진이 무너진다 (이진화 기준이 조명에 끌려감)
- ⚠ 전처리 후에도 틀리는 사진은 변환 그림을 보고 원인(획 끊김 · 모양) 추측

##### 자세히
- 과정의 방향: 입력을 학습 데이터 쪽으로 옮기는 것이 전처리, 반대로 **모델이 실제 데이터를 보고 배우게** 하는 것이 Day 2 의 데이터 증강과 Day 3 의 전이학습.

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
- [VB D1·5교시 학습 루프](#/vb/1/5) — `train_one_epoch` · `evaluate` 를 그대로 사용
- [VB D1·7교시 영상 파일](#/vb/1/7) — 프레임 루프에 추론을 끼운다
- [NN D5·5교시 SimpleConvNet](#/nn/5/5) — NumPy 로 만든 같은 구조 (433,890)
