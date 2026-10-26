## Day 3 — 역전파 (backpropagation)

### 1교시 · 복습 퀴즈 · 수치 미분의 한계

**키워드**: `수치 미분 O(P²)`, `역전파 O(P)`, `2P 번 순전파`, `159,020`

**개요**: 수치 미분은 파라미터 하나마다 손실을 두 번 계산 — step 당 순전파 2P 번, 순전파 비용도 P 에 비례하니 O(P²). 784-100-10 망은 step 당 손실 159,020번. 역전파는 순전파 1번 + 역방향 1번(순전파의 2~3배)으로 모든 기울기를 한꺼번에 — O(P). 수치 미분은 학습이 아니라 검증용.

**핵심 용어**
- **역전파(backpropagation)**: 출력에서 입력 방향으로 기울기를 한 번에 전달해 모든 파라미터의 기울기를 구하는 방법

**핵심 내용**

#### 비용 비교
> 수치 미분 ∝ 2P × P = O(P²) · 역전파 ∝ O(P)
| 망 | 파라미터 P | 수치 미분 1 step 손실 계산 |
|---|---|---|
| 784-100-10 | 79,510 | 2P = **159,020번** |
- **파라미터 10배** → 수치 미분 시간 100배
- **수치 미분의 쓰임** → 역전파가 맞는지 검증(gradient check, 6교시)

**연결**
- [NN D1·6교시 미분과 기울기](#/nn/1/6) — numerical_gradient

### 2교시 · 계산 그래프

**키워드**: `계산 그래프`, `노드 · 엣지`, `순전파 = 값`, `역전파 = 기울기`, `국소 계산`, `사과 · 귤 쇼핑`

**개요**: 계산 과정을 노드(연산)와 엣지(값)로 그린다. 순전파는 왼쪽 → 오른쪽으로 값을, 역전파는 오른쪽 → 왼쪽으로 기울기를 나른다. 역전파 값 = "그 값이 1 오르면 출력이 얼마나 오르나". 노드는 자기 입력과 위에서 온 기울기만으로 계산(국소 계산) — 노드 종류마다 forward · backward 를 한 번만 만들면 된다.

**핵심 용어**
- **계산 그래프(computational graph)**: 계산을 노드(연산)와 엣지(값)로 그린 그림
- **국소 계산(local computation)**: 노드가 전체를 몰라도 자기 입력과 위에서 내려온 기울기만으로 자기 몫을 계산하는 것

**핵심 내용**

#### 사과 · 귤 쇼핑
> 사과 100원 × 2개 + 귤 150원 × 3개, 소비세 1.1 → 최종 715원
| 값 | 순전파 | 역전파 ∂(최종)/∂(값) |
|---|---|---|
| 사과 가격 100 | | 2.2 (2개 × 1.1) |
| 사과 개수 2 | | 110 |
| 귤 가격 150 | | 3.3 |
| 귤 개수 3 | | 165 |
| 합계 | 200 + 450 = 650 | 1.1 |
| 소비세 1.1 | 650 × 1.1 = 715 | 650 |
- **사과 가격 기울기 2.2** → 사과가 1원 오르면 최종 금액 2.2원 오름
- **출력에서 1 로 시작** → 노드마다 국소 미분을 곱하며 거꾸로

#### 국소 계산의 이점
- **노드 하나는 단순** → 종류별 forward/backward 만 구현해 조립
- **순전파 때 중간값 저장** → 역전파 때 재계산 안 함
- **역전파 한 번** → 모든 입력의 기울기가 동시에 (수치 미분은 입력마다 따로)

### 3교시 · 연쇄 법칙

**키워드**: `연쇄 법칙`, `덧셈 노드 = 복사`, `곱셈 노드 = 바꿔 곱함`, `dout`, `MulLayer`, `forward 에서 저장`

**개요**: 합성 함수의 미분은 단계별 미분의 곱(연쇄 법칙) — 역전파는 이것을 그래프 위에서 실행하는 것. 덧셈 노드는 기울기를 그대로 복사, 곱셈 노드는 상대편 입력을 곱한다. 계층 = forward(값 계산 + 필요한 값 저장) + backward(dout 받아 입력 쪽 기울기 반환).

**핵심 용어**
- **연쇄 법칙(chain rule)**: ∂z/∂x = ∂z/∂t · ∂t/∂x — 합성 함수의 미분은 각 단계 미분의 곱
- **dout**: 위(출력 쪽)에서 내려온 기울기 ∂L/∂(이 노드의 출력)

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `MulLayer` | `forward(x, y)` → `x * y` · `backward(dout)` → `(dout * y, dout * x)` | 곱셈 노드 계층 |
| `AddLayer` | `forward(x, y)` → `x + y` · `backward(dout)` → `(dout, dout)` | 덧셈 노드 계층 |

**핵심 내용**

#### 연쇄 법칙
> z = (x + y)², t = x + y → ∂z/∂x = 2t · 1
- **x = 3, y = 1** → t = 4, z = 16, ∂z/∂x = ∂z/∂y = **8**

#### 두 가지 기본 노드
| 노드 | 순전파 | 역전파 |
|---|---|---|
| 덧셈 z = x + y | x + y | dx = dout, dy = dout — **그대로 복사** |
| 곱셈 z = x · y | x · y | dx = dout · y, dy = dout · x — **서로 바꿔 곱함** |
- **곱셈 노드는 순전파 입력이 필요** → forward 에서 `self.x`, `self.y` 저장 — 오늘 모든 계층의 공통 구조
- ⚠ `dx = dout * self.x` (자기 자신을 곱함) → 틀림. 곱셈 노드는 **상대편**을 곱한다
- ⚠ 같은 MulLayer 객체를 두 곳에 재사용 → 저장값이 덮어써짐 → 노드마다 객체 하나
- ⚠ backward 는 forward 의 **반대 순서**로

**기본 사용형태**
```python
class MulLayer:
    def forward(self, x, y):
        self.x, self.y = x, y        # 역전파용 저장
        return x * y
    def backward(self, dout):
        return dout * self.y, dout * self.x   # 상대편을 곱한다
```

### 4교시 · 활성화 계층: ReLU · Sigmoid

**키워드**: `Relu 계층`, `mask`, `스위치`, `Sigmoid 계층`, `self.out`, `y(1 − y)`

**개요**: 활성화 함수를 계층으로. ReLU 는 forward 에서 0 이하였던 자리(mask)를 기억해 backward 때 그 자리 기울기를 0 으로 막는 스위치. Sigmoid 는 forward 출력 y 를 저장해 backward 에 dout · y(1 − y).

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Relu` | `forward`: `mask = (x <= 0)`, 그 자리 0 · `backward`: `dx[mask] = 0` | ReLU 계층 |
| `Sigmoid` | `forward`: `out = 1/(1+exp(-x))` 저장 · `backward`: `dout * out * (1 - out)` | Sigmoid 계층 |

**핵심 내용**

#### ReLU = 스위치
> 순전파 때 막힌(≤ 0) 자리는 역전파 때도 막힌다(0), 통과한 자리는 그대로
- **mask** → `(x <= 0)` 불리언 배열 — Day 1 마스크 인덱싱

#### Sigmoid = 출력 하나로 기울기
> ∂L/∂x = dout · y(1 − y) — × (−1) → exp → +1 → 1/x 네 노드를 연쇄 법칙으로 정리한 결과
- **sigmoid'(0)** = 0.25
- ⚠ `Relu.forward` 에서 `x[mask] = 0` 으로 입력을 직접 바꾸면 앞 계층 출력(원본)이 망가짐 → `x.copy()` 에 작업
- ⚠ `Sigmoid.backward` 는 저장한 `self.out` 으로 — `sigmoid(dout)` 아님
- ⚠ forward 없이 backward → `self.mask` · `self.out` 이 None

**기본 사용형태**
```python
class Relu:
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy(); out[self.mask] = 0
        return out
    def backward(self, dout):
        dx = dout.copy(); dx[self.mask] = 0
        return dx

class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x)); return self.out
    def backward(self, dout):
        return dout * self.out * (1 - self.out)
```

**연결**
- [NN D2·4교시 활성화 함수](#/nn/2/4) — relu_grad · sigmoid_grad 가 이 backward

### 5교시 · Affine · Softmax-with-Loss 계층

**키워드**: `Affine`, `dX = dout Wᵀ`, `dW = Xᵀ dout`, `db = Σ dout`, `기울기 shape = 자기 shape`, `SoftmaxWithLoss`, `(Y − T) / N`

**개요**: Affine 은 Y = XW + b 계층 — 역전파는 "기울기는 자기 자신과 shape 가 같다"로 유도: dX = dout Wᵀ, dW = Xᵀ dout, db = 행 방향 합. Softmax-with-Loss 는 softmax + CE 를 한 계층으로 묶어 기울기가 깔끔하게 (Y − T)/N.

**핵심 용어**
- **Affine 계층**: Y = XW + b 를 계산하는 완전연결 계층
- **Softmax-with-Loss 계층**: softmax 출력과 cross-entropy 손실을 묶은 마지막 계층

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Affine` | `forward`: `X @ W + b` · `backward`: `dX = dout @ W.T`, `dW = X.T @ dout`, `db = dout.sum(axis=0)` | 완전연결 계층 |
| `SoftmaxWithLoss` | `forward(X, t)` → 손실 · `backward(dout=1)` → `(y - onehot(t)) / N` | 출력 + 손실 계층 |
| `cross_entropy_error()` | `-np.sum(np.log(y[np.arange(N), t] + 1e-12)) / N` | 레이블 · one-hot 모두 받는 CE |

**핵심 내용**

#### Affine 역전파 — shape 로 유도
> 기울기는 자기 자신과 shape 가 같다
| 기호 | shape | 기울기 |
|---|---|---|
| X | (N, D) | dX = dout @ Wᵀ — (N, M)(M, D) |
| W | (D, M) | dW = Xᵀ @ dout — (D, N)(N, M) |
| b | (M,) | db = dout.sum(axis=0) |
| Y, dout | (N, M) | |
- **편향은 왜 합** → 브로드캐스팅으로 N 개 행에 같은 b 가 N 번 쓰였으니 기울기도 N 개를 모두 더함 (덧셈 노드 "복사"의 반대)
- ⚠ shape 만 보고 `dW = dout @ self.X` 처럼 대충 바꾸면 — 정사각일 때 오류 없이 틀림
- ⚠ `db = dout.mean(axis=0)` → 틀림. 평균(/N)은 SoftmaxWithLoss 가 이미 했다

#### Softmax-with-Loss = (Y − T) / N
> softmax 미분과 log 미분이 상쇄 — Day 2 의 "(예측 − 정답)" 이 다시
- **정답이 레이블(정수)이어도** → `y[np.arange(N), t]` 로 행마다 정답 확률 하나씩 (팬시 인덱싱)
- ⚠ `/ N` 을 빼먹으면 손실은 평균인데 기울기는 합 → 배치 크기만큼 학습률이 커진 셈
- ⚠ 레이블 t 가 float 이면 인덱싱 오류 → `astype(int)`

**기본 사용형태**
```python
class Affine:
    def __init__(self, W, b): self.W, self.b = W, b
    def forward(self, X):
        self.X = X
        return X @ self.W + self.b
    def backward(self, dout):
        self.dW = self.X.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

class SoftmaxWithLoss:
    def forward(self, X, t):
        self.t, self.y = t, softmax(X)
        return cross_entropy_error(self.y, t)
    def backward(self, dout=1):
        N = len(self.t)
        dx = self.y.copy(); dx[np.arange(N), self.t] -= 1
        return dx / N
```

**연결**
- [NN D2·8교시 로지스틱 회귀](#/nn/2/8) — 손으로 유도한 Xᵀ(P − T)/N 이 Affine + 손실 계층의 역전파

### 6교시 · 계층 조립과 Gradient check

**키워드**: `TwoLayerNet`, `OrderedDict`, `reversed`, `gradient check`, `상대 오차`, `1e-7`, `1e-12`, `버그 찾기`

**개요**: 계층을 OrderedDict 에 쌓아(Affine1 → Relu1 → Affine2 → SoftmaxWithLoss) 순서대로 forward, 반대 순서로 backward. 역전파 기울기가 맞는지 수치 미분과 상대 오차로 비교(gradient check) — shape 가 맞아도 값이 틀릴 수 있으니 새 계층마다 확인.

**핵심 용어**
- **gradient check**: 역전파로 구한 기울기를 수치 미분과 비교해 구현을 검증하는 것
- **상대 오차(relative error)**: ‖a − b‖ / (‖a‖ + ‖b‖) — 값의 크기와 무관하게 비교

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `TwoLayerNet` | `TwoLayerNet(784, 100, 10)` · `.predict` · `.loss` · `.accuracy` · `.gradient` | 계층으로 조립한 2층 신경망 |
| `rel_error()` | `np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))` | 상대 오차 |
| `failing()` | `[k for k in g_bp if rel_error(g_bp[k], g_num[k]) > tol]` | 기준을 넘는 파라미터 이름 목록 |

**핵심 내용**

#### TwoLayerNet 의 gradient()
> ① loss() 로 순전파(값 저장) → ② lastLayer.backward(1) → ③ layers 를 reversed 로 backward → ④ 각 Affine 의 dW · db 모으기
- **params 와 Affine 은 같은 배열 객체** → 갱신은 반드시 **제자리** `params[k] -= lr * g`
- ⚠ gradient() 에서 순전파를 빼먹으면 이전 배치의 저장값으로 엉뚱한 기울기
- ⚠ `reversed(self.layers.values())` 가 안 되는 버전 → `reversed(list(...))`

#### 상대 오차 판정
| 상대 오차 | 판정 |
|---|---|
| < 1e-7 | ✅ 구현이 맞다 |
| 1e-7 ~ 1e-4 | ⚠ 의심 — ReLU 의 0 근처 또는 손실 식이 미세하게 다름 |
| > 1e-2 | ❌ 버그 |
- **784-10-10 (7,960개), 샘플 3장** → 수치 미분도 몇 초 — gradient check 는 작은 망 · 작은 배치로

#### 왜 log 에 1e-12 인가
- **Day 2 의 1e-7** → 역전파 (Y − T) 는 eps 없는 손실의 정확한 미분 → 수치 미분과 약 5 × 10⁻⁷ 차이 → 기준 초과
- **1e-12** 로 줄이면 통과

#### 버그 찾기 — shape 가 맞아도 틀린다
- **버그 ① `db = dout.mean(axis=0)`** → db 가 틀림
- **버그 ② `dX = dout @ self.W`** (`.T` 빠짐) → W2 (10, 10) 정사각이라 오류 없이 실행 → **앞쪽 파라미터(W1, b1)**에서 드러남
- **오류는 그 계층 앞쪽 파라미터에서 드러난다** — dX 가 앞으로 흘러가므로

**기본 사용형태**
```python
def gradient(self, X, t):
    self.loss(X, t)                                   # ① 순전파
    dout = self.lastLayer.backward(1)                 # ②
    for layer in reversed(list(self.layers.values())):
        dout = layer.backward(dout)                   # ③
    return {"W1": self.layers["Affine1"].dW, "b1": self.layers["Affine1"].db,
            "W2": self.layers["Affine2"].dW, "b2": self.layers["Affine2"].db}

def rel_error(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))
```

### 7교시 · 미니배치 SGD

**키워드**: `미니배치`, `SGD`, `batch size`, `iteration`, `epoch`, `셔플`, `제자리 갱신`, `나선 데이터`

**개요**: 전체 데이터로 기울기를 구하면 느리다 — 무작위 일부(미니배치)로 기울기를 추정해 자주 갱신하는 것이 SGD. epoch 마다 셔플 → batch 단위 역전파 · 갱신. 곡선은 잡음으로 들쭉날쭉하지만 내려간다. 은닉층 ReLU 의 꺾인 직선 조각들이 모여 나선을 따라가는 곡선 경계가 된다.

**핵심 용어**
- **미니배치(mini-batch)**: 한 번 갱신에 쓰는 무작위 샘플 묶음
- **SGD(확률적 경사 하강법)**: 미니배치 기울기로 파라미터를 갱신하는 경사하강
- **iteration(step)**: 미니배치 하나로 기울기 → 갱신 1회
- **일반화(generalization)**: 학습에 안 쓴 데이터(test)에서도 잘 맞히는 능력

**핵심 내용**

#### 용어와 숫자
| 용어 | 나선 데이터 | MNIST |
|---|---|---|
| batch size | 30 | 100 |
| epoch 당 iteration | 240 ÷ 30 = 8 | 60,000 ÷ 100 = 600 |
- **셔플** → `rng.permutation` — 같은 순서면 미니배치 구성이 고정되어 기울기가 치우침
- **train 정확도** → 학습 상태 · **test 정확도** → 일반화

#### 흔한 실수
- ⚠ `params[k] = params[k] - lr * g` → 새 배열 → Affine 과 연결이 끊겨 학습 안 됨 → **`-=`**
- ⚠ `idx[i:i + batch_size]` → 시작을 `i * batch_size` 로 곱해야
- ⚠ 셔플한 인덱스로 X 만 자르고 t 는 원래 순서 → 입력 · 정답 어긋남
- ⚠ test 데이터로 학습 금지 — 평가 전용
- **손실 곡선이 중간에 들쭉날쭉** → 미니배치 잡음 + 큰 학습률 → Day 4 옵티마이저

**기본 사용형태**
```python
iters_per_epoch = len(X_train) // batch_size
for epoch in range(epochs):
    idx = rng.permutation(len(X_train))
    for i in range(iters_per_epoch):
        batch = idx[i * batch_size:(i + 1) * batch_size]
        grads = net.gradient(X_train[batch], t_train[batch])
        for k in net.params:
            net.params[k] -= lr * grads[k]           # 제자리 갱신
```

**연결**
- [NN D4·1교시 SGD 의 약점](#/nn/4/1)

### 8교시 · 미니 프로젝트: MNIST 2층 MLP

**키워드**: `MNIST`, `(N, 784)`, `/ 255`, `TwoLayerNet(784, 100, 10)`, `학습 전 손실 2.30`, `test ≥ 0.95`

**개요**: 오늘 만든 계층 · 역전파 · 미니배치 SGD 로 손글씨 0~9 분류. 28×28 흑백 6만 장 학습, 처음 보는 1만 장 평가. lr 0.1 · batch 100 · 10 epoch(6,000 iteration)로 test 95% 이상.

**핵심 용어**
- **MNIST**: 28×28 흑백 손글씨 숫자 데이터셋 — 학습 60,000장 · 평가 10,000장

**핵심 내용**

#### 전처리
> (60000, 28, 28) uint8 → (60000, 784) float, 0~1
- **펼치기** → `reshape(-1, 784)` · **정규화** → `/ 255.0` (값이 크면 학습 불안정) · **레이블** → 정수 (SoftmaxWithLoss 가 받음)

#### 학습 전 손실 = −log 0.1 ≈ 2.30
- 가중치 × 0.01 로 아주 작음 → 점수 ≈ 0 → softmax 가 10개 클래스에 거의 균등(0.1) → CE = −log 0.1

#### 학습 설정
- **lr 0.1, batch 100, 10 epoch** → iteration 마다 미니배치 손실(`network.lastLayer.loss`), epoch 마다 train · test 정확도 기록
- **Colab CPU** → 30초 ~ 1분

**기본 사용형태**
```python
X_train = x_train.reshape(-1, 784) / 255.0
X_test  = x_test.reshape(-1, 784) / 255.0
t_train, t_test = y_train.astype(int), y_test.astype(int)
network = TwoLayerNet(784, 100, 10)
loss0 = network.loss(X_train[:100], t_train[:100])    # ≈ 2.30
```

**연결**
- [NN D4·8교시 미니 프레임워크 mynn](#/nn/4/8) — 이 망을 개선 기법으로 다시
- [NN D5·5교시 SimpleConvNet 으로 MNIST](#/nn/5/5) — 같은 데이터를 CNN 으로
