## Day 3 — 역전파 (backpropagation)

### 1교시 · 복습 퀴즈 · 수치 미분의 한계

**키워드**: `수치 미분 O(P²)`, `역전파 O(P)`, `step 당 순전파 2P 번`, `159,020`, `차수(order)`, `gradient check 용`

**개요**: Day 2 까지는 기울기를 손으로 유도하거나 수치 미분으로 구했다. ① 수치 미분은 파라미터 **하나마다** 손실을 두 번 계산해서, 파라미터가 P 개면 step 한 번에 순전파 2P 번 — 순전파 비용도 P 에 비례하니 전체는 **O(P²)**. ② 784-100-10 망은 step 한 번에 손실 **159,020**번 → 학습이 사실상 불가능. ③ 오늘 만들 **역전파**는 순전파 1번 + 역방향 1번으로 모든 기울기를 한꺼번에 — O(P). 수치 미분은 버리지 않고 역전파가 맞는지 확인하는 "정답지"로 쓴다.

**핵심 용어**
- **수치 미분(numerical gradient)**: 파라미터 하나를 ±h 만큼 움직여 (L(W + h) − L(W − h)) / 2h 로 기울기를 근사하는 것 (Day 1)
- **역전파(backpropagation)**: 출력에서 입력 방향으로 기울기를 한 번에 전달해 모든 파라미터의 기울기를 구하는 방법
- **시간 복잡도 O(·)**: 입력 크기가 커질 때 계산량이 늘어나는 차수 — O(P²) 는 P 가 10배면 시간 100배
- **gradient check**: 역전파 기울기를 수치 미분과 비교해 구현을 검증하는 것 (6교시)

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 수치 미분은 왜 느린가 | Day 2 방식(함수 + 딕셔너리)으로 784-h-10 망을 만들고, h = 1 · 2 · 4 일 때 전체 수치 미분 시간을 재서 784-100-10 의 시간을 추정 |

**핵심 내용**

#### 수치 미분은 왜 느린가
> step 당 순전파 2P 번 × 순전파 1번 비용 ∝ P → 수치 미분 1 step ∝ **O(P²)**
- **파라미터 하나** → +h 로 1번, −h 로 1번 → 순전파 **2번**
- **784-100-10 (P = 79,510)** → step 당 손실 계산 **159,020번** → 수십 ~ 100 초
- **파라미터 10배** → 시간 **100배** — 컴퓨터가 빨라져도 차수의 차이는 줄지 않는다
| 망 | 파라미터 P | 수치 미분 1 step 손실 계산 (2P) |
|---|---|---|
| 784-4-10 | 3,190 | 6,380 |
| 784-10-10 | 7,960 | 15,920 |
| 784-100-10 | **79,510** | **159,020** |

#### 역전파 = 순전파 1번 + 역방향 1번
> 모든 기울기를 한꺼번에 — 비용은 순전파의 2~3배, **O(P)**
- **784-100-10, 배치 100** → 역전파 1 step 수 ms vs 수치 미분 수십 초 → **수만 배** 차이 (CPU 마다 다름)
- **로그 그래프** → 수치 미분은 P², 역전파는 P 에 비례 → 두 직선의 기울기가 다름
- **수치 미분의 쓰임** → 느리지만 구현이 단순해 틀리기 어려움 → 역전파 **검증용** (6교시)
##### 자세히
```text
 numerical   for each of P params:  L(W+h), L(W−h)   →  2P forwards × O(P)  =  O(P²)
 backprop    1 forward  +  1 backward                →  ≈ 3 forwards        =  O(P)
```

**기본 사용형태**
```python
def count_params(params): return sum(p.size for p in params.values())
def n_loss_evals(P):      return 2 * P            # 파라미터마다 +h, −h 두 번
n_loss_evals(79_510)                              # 159020
```

**연결**
- [NN D1·6교시 미분과 기울기](#/nn/1/6) — numerical_gradient (중앙 차분)
- [NN D2·8교시 로지스틱 회귀](#/nn/2/8) — "왜 Day 3 에서 역전파가 필요한가" 표

**연관 문제**
<<QA 1>>

### 2교시 · 계산 그래프

**키워드**: `계산 그래프`, `노드 = 연산`, `엣지 = 값`, `순전파 = 값`, `역전파 = 기울기`, `출력에서 1 로 시작`, `국소 계산`, `사과 · 귤 쇼핑 715`

**개요**: 복잡한 식도 작은 계산의 연결로 쪼개면, 기울기도 조각마다 구해 이어 붙일 수 있다. ① 계산 과정을 **노드(연산) + 엣지(값)** 로 그리고, 순전파는 왼쪽 → 오른쪽으로 **값**을, 역전파는 오른쪽 → 왼쪽으로 **기울기**를 나른다. ② 역전파로 얻는 값의 의미는 "그 값이 1 오르면 출력이 얼마나 오르나". ③ 노드는 자기 입력과 위에서 온 기울기만으로 계산(**국소 계산**) → 노드 종류마다 forward · backward 를 한 번만 만들어 조립하면 된다.

**핵심 용어**
- **계산 그래프(computational graph)**: 계산을 노드(연산)와 엣지(값)로 그린 그림
- **순전파(forward)**: 왼쪽 → 오른쪽으로 값을 계산해 나르는 것
- **역전파(backward)**: 오른쪽 → 왼쪽으로 기울기 ∂(출력)/∂(값) 을 나르는 것 — 출력에서 **1** 로 시작
- **국소 미분(local derivative)**: 노드 하나의 출력을 그 입력으로 미분한 값 (곱셈 노드 x·y 의 x 쪽 = y)
- **국소 계산(local computation)**: 노드가 전체를 몰라도 자기 입력과 위에서 내려온 기울기만으로 자기 몫을 계산하는 것

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 예제 — 사과 · 귤 쇼핑 | 순전파로 200 · 450 · 650 · **715** 를 계산, 역전파로 각 값의 기울기를 구하고 수치 미분과 비교 |

**핵심 내용**

#### 사과 · 귤 쇼핑 — 값은 앞으로, 기울기는 뒤로
> 사과 100원 × 2개 + 귤 150원 × 3개, 소비세 1.1 → 최종 **715**원
| 값 | 순전파 | 역전파 ∂(최종)/∂(값) | 계산 |
|---|---|---|---|
| 소비세 1.1 | 650 × 1.1 = 715 | **650** | 곱셈 → 상대편 650 |
| 합계 650 | 200 + 450 | **1.1** | 1 × 1.1 |
| 사과 합 200 · 귤 합 450 | 100 × 2 · 150 × 3 | 1.1 · 1.1 | 덧셈 → 그대로 |
| 사과 가격 100 | | **2.2** | 1.1 × 2 |
| 사과 개수 2 | | **110** | 1.1 × 100 |
| 귤 가격 150 | | **3.3** | 1.1 × 3 |
| 귤 개수 3 | | **165** | 1.1 × 150 |
- **사과 가격 기울기 2.2** → 사과가 1원 오르면 최종 금액 **2.2**원 오름 (2개 × 세금 1.1)
- **출발** → ∂(최종)/∂(최종) = **1** 에서 시작해 노드마다 국소 미분을 곱하며 거꾸로
- **수치 미분과 비교** → `numerical_gradient(total, v)` 가 같은 값 → 역전파가 맞다
##### 자세히
```text
 apple  100 ─┐
             (×) ── 200 ─┐
 count    2 ─┘           (+) ── 650 ─┐
 orange 150 ─┐           │           (×) ── 715
             (×) ── 450 ─┘           │
 count    3 ─┘                tax 1.1┘

 backward:  1 ──► ×tax: 1.1 ──► +: 1.1, 1.1 ──► ×count: 2.2 (apple), 3.3 (orange)
```

#### 국소 계산 — 노드는 자기 일만 한다
> × tax 노드는 앞에서 사과가 몇 개였는지 몰라도 된다 — 자기 입력(650, 1.1)과 위의 기울기만 있으면 됨
- **노드 하나는 단순** → 노드 **종류마다** forward · backward 를 한 번만 구현해 조립 (3~5교시)
- **순전파 때 중간값 저장** → 역전파 때 다시 계산하지 않음
- **역전파 한 번** → 모든 입력의 기울기가 **동시에** (수치 미분은 입력마다 따로)

**기본 사용형태**
```python
apple_price  = 100 * 2                  # 순전파
all_price    = apple_price + 150 * 3
price        = all_price * 1.1          # 715

dprice = 1                              # 역전파: 출력에서 1
dall   = dprice * 1.1                   # 곱셈 노드 → 상대편
dapple = dall * 1 * 2                   # 덧셈 → 그대로, 곱셈 → 상대편 (2.2)
```

**연결**
- [NN D1·6교시 미분과 기울기](#/nn/1/6) — "1 오르면 얼마나 오르나" = 편미분

**연관 문제**
<<QA 2>>

### 3교시 · 연쇄 법칙

**키워드**: `연쇄 법칙`, `국소 미분의 곱`, `덧셈 노드 = 그대로 복사`, `곱셈 노드 = 서로 바꿔 곱함`, `dout`, `MulLayer`, `AddLayer`, `forward 에서 저장`

**개요**: 2교시의 "그대로 보낸다", "상대편을 곱한다" 규칙이 어디서 왔는지 본다. ① 합성 함수의 미분은 단계별 미분의 곱(**연쇄 법칙**) — 역전파는 이것을 그래프 위에서 실행하는 것. ② 기본 노드 두 개: 덧셈은 기울기를 **복사**, 곱셈은 **상대편 입력**을 곱한다. ③ 계층 = forward(값 계산 + 역전파에 필요한 값 **저장**) + backward(dout 을 받아 입력 쪽 기울기 반환) — 오늘 만드는 모든 계층의 공통 구조.

**핵심 용어**
- **연쇄 법칙(chain rule)**: ∂z/∂x = ∂z/∂t · ∂t/∂x — 합성 함수의 미분은 각 단계 미분의 곱
- **dout**: 위(출력 쪽)에서 내려온 기울기 ∂L/∂(이 노드의 출력)
- **계층(layer)**: forward · backward 메서드를 가진 클래스 — 노드 하나(또는 여러 개)를 묶은 부품
- **덧셈 노드**: z = x + y. 국소 미분이 1, 1 → dout 을 그대로 양쪽에
- **곱셈 노드**: z = x · y. 국소 미분이 y, x → dout 에 **상대편**을 곱함

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `MulLayer` | `forward(x, y)` → `x * y` · `backward(dout)` → `(dout * y, dout * x)` | 곱셈 노드 계층 |
| `AddLayer` | `forward(x, y)` → `x + y` · `backward(dout)` → `(dout, dout)` | 덧셈 노드 계층 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | z = (x + y)², x = 3, y = 1 → 역전파 dz/dx = dz/dy = **8** 을 수치 미분과 비교 |
| 두 가지 기본 노드 | `MulLayer` 두 개로 사과만 (100 × 2) × 1.1 = **220**, 역전파 → 사과 가격 2.2 · 개수 110 · 소비세 200 |

**핵심 내용**

#### 연쇄 법칙 — 역전파의 원리
> z = (x + y)² 를 t = x + y, z = t² 로 쪼개면 ∂z/∂x = ∂z/∂t · ∂t/∂x = 2t · 1
- **x = 3, y = 1** → t = **4**, z = **16**, ∂z/∂x = ∂z/∂y = 8 · 1 = **8**
- **역전파** → 출력에서 1 을 출발시켜 노드를 지날 때마다 그 노드의 **국소 미분**을 곱한다
- **Day 2 의 손 유도** → ½(y − t)² 의 "바깥 미분 × 안쪽 미분" 이 바로 연쇄 법칙
##### 자세히
```text
 x = 3 ─┐                          dz/dx = 8 × 1 = 8
        (+) ── t = 4 ── (²) ── z = 16
 y = 1 ─┘                          dz/dy = 8 × 1 = 8
 backward:   1 ──► ×2t = 8 ──► ×1, ×1
```

#### 두 가지 기본 노드
> 덧셈은 **그대로 복사**, 곱셈은 **서로 바꿔 곱함**
| 노드 | 순전파 | 국소 미분 | 역전파 |
|---|---|---|---|
| 덧셈 z = x + y | x + y | 1, 1 | dx = dout, dy = dout |
| 곱셈 z = x · y | x · y | y, x | dx = dout · **y**, dy = dout · **x** |
- **곱셈 노드는 순전파 입력이 필요** → forward 에서 `self.x`, `self.y` 저장
- **"forward 에서 저장 → backward 에서 사용"** → 4 · 5교시 모든 계층의 공통 패턴
- ⚠ `dx = dout * self.x` (자기 자신을 곱함) → 틀림. 곱셈 노드는 **상대편**을 곱한다
- ⚠ 같은 MulLayer 객체를 두 곳에 재사용 → 두 번째 forward 가 저장값을 덮어써 첫 노드의 역전파가 틀어짐 → **노드마다 객체 하나**
- ⚠ backward 는 forward 의 **반대 순서**로 — 같은 순서면 아직 계산되지 않은 dout 을 쓴다

**기본 사용형태**
```python
class MulLayer:
    def forward(self, x, y):
        self.x, self.y = x, y              # 역전파용 저장
        return x * y
    def backward(self, dout):
        return dout * self.y, dout * self.x   # 상대편을 곱한다

class AddLayer:
    def forward(self, x, y): return x + y
    def backward(self, dout): return dout * 1, dout * 1
```

**연결**
- [NN D2·7교시 선형 회귀](#/nn/2/7) — 바깥 미분 × 안쪽 미분으로 (y − t)x 를 유도
- [Python D3·1교시 클래스 기초](#/py/3/1) — self 에 값을 저장하는 객체

**연관 문제**
<<QA 3>>

### 4교시 · 활성화 계층: ReLU · Sigmoid

**키워드**: `Relu 계층`, `mask = (x <= 0)`, `스위치`, `x.copy()`, `Sigmoid 계층`, `self.out`, `dout · y(1 − y)`, `sigmoid'(0) = 0.25`

**개요**: Day 2 의 활성화 함수를 **계층**으로 만든다 — 배열의 원소 하나하나가 독립된 노드. ① ReLU 는 forward 에서 0 이하였던 자리(**mask**)를 기억해, backward 때 그 자리 기울기를 0 으로 막는 **스위치**. ② Sigmoid 는 네 노드(×(−1) → exp → +1 → 1/x)를 연쇄 법칙으로 정리하면 dout · y(1 − y) — 순전파 출력 **y 하나만** 저장하면 된다. ③ 계층의 backward 는 수치 미분과 비교해 확인한다.

**핵심 용어**
- **Relu 계층**: forward 에서 mask = (x <= 0) 저장, backward 에서 mask 자리 기울기 0
- **mask**: 순전파 때 0 이하였던 자리를 True 로 표시한 불리언 배열 (Day 1 마스크 인덱싱)
- **Sigmoid 계층**: forward 에서 출력 out 저장, backward 에서 dout · out · (1 − out)
- **원소별(element-wise) 계층**: 각 원소가 따로 계산되는 계층 — 입력 · 출력 · 기울기 shape 가 모두 같다

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Relu` | `forward`: `mask = (x <= 0)`, 그 자리 0 · `backward`: `dx[mask] = 0` | ReLU 계층 |
| `Sigmoid` | `forward`: `out = 1/(1+exp(-x))` 저장 · `backward`: `dout * out * (1 - out)` | Sigmoid 계층 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| ReLU | `[[1, −0.5], [−2, 3]]` → out `[[1, 0], [0, 3]]`, backward(1) `[[1, 0], [0, 1]]`, 입력 배열이 안 바뀐 것 확인 |
| Sigmoid | 입력 [1, −2] → y = [0.7311, 0.1192], backward(1) = [**0.1966**, **0.1050**], x = 0 이면 **0.25** |

**핵심 내용**

#### ReLU 계층 = 스위치
> 순전파 때 막힌(x ≤ 0) 자리는 역전파 때도 막히고(0), 통과한 자리는 기울기를 그대로 통과
- **forward** → `mask = (x <= 0)` 저장 → `out = x.copy()` 의 mask 자리 0
- **backward** → `dx = dout.copy()` 의 mask 자리 0 — 나머지는 dout 그대로 (국소 미분 1)
- **예** → 입력 `[[1, −0.5], [−2, 3]]` → mask `[[F, T], [T, F]]` → backward(1) = `[[1, 0], [0, 1]]`
- ⚠ forward 에서 `x[self.mask] = 0` 으로 **입력을 직접** 바꾸면 바깥 원본(앞 계층 출력)이 망가짐 → `x.copy()` 에 작업 (Day 1 가변 객체)
##### 자세히
```text
 x     [ 1.0  -0.5 ]      mask  [ F  T ]      out   [ 1  0 ]
       [-2.0   3.0 ]            [ T  F ]            [ 0  3 ]
 dout  [ 1  1 ]     ──►   dx    [ 1  0 ]    closed switch → gradient 0
       [ 1  1 ]                 [ 0  1 ]
```

#### Sigmoid 계층 = 출력 하나로 기울기
> ∂L/∂x = dout · y²e⁻ˣ = dout · **y(1 − y)** — Day 2 sigmoid_grad 와 같은 식
- **네 노드로 쪼개기** → × (−1) → exp → + 1 → 1/x, 국소 미분을 곱해 정리
- **forward** → `self.out` 저장 (backward 에서 exp 를 다시 계산하지 않음)
- **x = 1 · −2 · 0** → 기울기 0.1966 · 0.1050 · **0.25** (최대)
- ⚠ backward 에서 `sigmoid(dout)` 같은 식 → 틀림. 저장한 **`self.out`** 으로 계산
- ⚠ forward 없이 backward → `self.mask` · `self.out` 이 None → 오류. 항상 forward → backward
##### 자세히
```text
 x ──(× −1)──► −x ──(exp)──► e⁻ˣ ──(+1)──► 1+e⁻ˣ ──(1/x)──► y
 backward:  dout · (−1/(1+e⁻ˣ)²) · 1 · e⁻ˣ · (−1)  =  dout · y² e⁻ˣ  =  dout · y(1 − y)
```

#### 계층 backward 를 확인하는 법
> L = Σ (forward(x) ⊙ dout) 을 수치 미분하면 backward(dout) 와 같아야 한다
- **이유** → 이 L 의 ∂L/∂(출력) 이 정확히 dout → 계층 하나만 떼어 시험할 수 있음
- ⚠ 수치 미분이 forward 를 여러 번 불러 `self.out` 이 바뀜 → 비교 전에 forward 를 다시 한 번

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
- [NN D1·2교시 Python 속성](#/nn/1/2) — 가변 객체라 `.copy()` 가 필요
- [NN D1·3교시 NumPy ndarray](#/nn/1/3) — 불리언 마스크 인덱싱

**연관 문제**
<<QA 4>>

### 5교시 · Affine · Softmax-with-Loss 계층

**키워드**: `Affine`, `기울기 shape = 자기 shape`, `dX = dout Wᵀ`, `dW = Xᵀ dout`, `db = Σ dout`, `SoftmaxWithLoss`, `(Y − T) / N`, `y[np.arange(N), t]`, `1e-12`

**개요**: 신경망의 나머지 두 부품. ① **Affine** 은 Day 2 가설 Y = XW + b 그대로의 계층 — 역전파 식은 "기울기는 자기 자신과 shape 가 같다" 하나로 유도된다: dX = dout Wᵀ, dW = Xᵀ dout, db = 행 방향 합. ② **Softmax-with-Loss** 는 softmax + cross-entropy 를 한 계층으로 묶은 것 — 복잡한 미분이 상쇄되어 기울기가 **(Y − T)/N**. ③ Day 2 에서 손으로 유도한 Xᵀ(P − T)/N 이 바로 이 두 계층의 역전파였다.

**핵심 용어**
- **Affine 계층**: Y = XW + b 를 계산하는 완전연결(fully connected) 계층
- **Softmax-with-Loss 계층**: softmax 출력과 cross-entropy 손실을 묶은 마지막 계층
- **레이블(label) 정답**: one-hot 대신 정답 번호(정수)만 둔 배열 — `[2, 0]`
- **팬시 인덱싱 `y[np.arange(N), t]`**: 행마다 정답 자리의 확률 하나씩을 고르는 것

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Affine` | `forward`: `X @ W + b` · `backward`: `dX = dout @ W.T`, `dW = X.T @ dout`, `db = dout.sum(axis=0)` | 완전연결 계층 |
| `SoftmaxWithLoss` | `forward(X, t)` → 손실 · `backward(dout=1)` → `(y - onehot(t)) / N` | 출력 + 손실 계층 |
| `cross_entropy_error()` | `-np.sum(np.log(y[np.arange(N), t] + 1e-12)) / N` | 레이블 · one-hot 모두 받는 CE |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| Affine 역전파 — shape 로 유도하기 | N, D, M = 2, 3, 4 무작위 값으로 dX (2, 3) · dW (3, 4) · db (4,) 를 구해 모두 수치 미분과 일치 확인 |
| Softmax-with-Loss — 기울기가 | 점수 2행 · 정답 `[2, 0]` → 레이블로도 one-hot 으로도 CE **0.8850**, Affine + SoftmaxWithLoss 역전파를 수치 미분과 비교 |

**핵심 내용**

#### Affine 역전파 — shape 로 유도한다
> 기울기는 자기 자신과 shape 가 같다 — (N, D) 나 (D, M) 을 만드는 행렬곱은 하나뿐
| 기호 | shape | 기울기 | 행렬곱 shape |
|---|---|---|---|
| X | (N, D) | dX = dout @ Wᵀ | (N, M)(M, D) |
| W | (D, M) | dW = Xᵀ @ dout | (D, N)(N, M) |
| b | (M,) | db = dout.sum(axis=0) | 행 방향 합 |
| Y, dout | (N, M) | — | — |
- **편향은 왜 합** → 브로드캐스팅으로 같은 b 가 N 개 행에 **N 번** 쓰였으니 기울기도 N 개를 모두 더함 (덧셈 노드 "복사"의 반대 방향)
- **dX 는 앞 계층으로** → 반환값 · **dW, db 는 저장** → `self.dW`, `self.db` (갱신에 씀)
- **W, b 는 params 와 같은 배열 객체** → 바깥에서 제자리 갱신하면 계층도 바로 반영
- ⚠ shape 만 보고 `dW = dout @ self.X` 처럼 대충 바꾸면 — 정사각일 때 **오류 없이 틀림** (6교시 버그 ②)
- ⚠ `db = dout.mean(axis=0)` → 틀림. 평균(/N)은 SoftmaxWithLoss 가 이미 했다 (6교시 버그 ①)

#### Softmax-with-Loss = (Y − T) / N
> softmax 의 복잡한 미분과 log 의 미분이 상쇄 — Day 2 의 "(예측 − 정답)" 이 다시
- **forward** → softmax → cross-entropy 손실 (y, t 저장)
- **backward(dout=1)** → `dx = y.copy(); dx[np.arange(N), t] -= 1; dx / N` — 정답 자리에서만 1 을 뺌
- **예** → 정답 확률 0.737 인 행 → 정답 칸 기울기 (0.737 − 1)/2 = **−0.132** (키우는 쪽), 나머지는 + (줄이는 쪽)
- **정답이 레이블이어도** → `y[np.arange(N), t]` 로 행마다 정답 확률 하나 · one-hot 이면 `argmax` 로 레이블로 바꿔 같은 코드
- **log 에 1e-12** → Day 2 의 1e-7 대신 (이유는 6교시)
- ⚠ `/ N` 을 빼먹으면 손실은 평균인데 기울기는 합 → 배치 크기만큼 학습률이 커진 셈
- ⚠ 레이블 t 가 float 이면 인덱싱 오류 → `astype(int)`
##### 자세히
```text
 X ──► Affine1 ──► Relu ──► Affine2 ──► SoftmaxWithLoss ──► L
 ◄─ dX ◄────────── dx ◄──── dX ◄──────── (Y − T)/N ◄──────── 1
       dW = Xᵀ·dout             dW = Xᵀ·dout
```
- Day 2 로지스틱 회귀의 dW = Xᵀ(P − T)/N 은 dout = (P − T)/N 을 넣은 Xᵀ · dout

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
- [NN D2·6교시 출력층과 손실 함수](#/nn/2/6) — softmax · CE · one-hot
- [NN D1·5교시 선형대수](#/nn/1/5) — 브로드캐스팅으로 b 가 N 번 더해짐

**연관 문제**
<<QA 5>>

### 6교시 · 계층 조립과 Gradient check

**키워드**: `TwoLayerNet`, `OrderedDict`, `predict · loss · accuracy · gradient`, `reversed`, `gradient check`, `상대 오차 < 1e-7`, `1e-7 vs 1e-12`, `버그 찾기`, `앞쪽 파라미터에서 드러남`

**개요**: 부품이 다 모였다. ① 계층을 **OrderedDict** 에 쌓아(Affine1 → Relu1 → Affine2, 마지막 SoftmaxWithLoss) 순서대로 forward, **반대 순서로** backward 하면 2층 신경망 TwoLayerNet. ② 역전파 기울기가 맞는지 수치 미분과 **상대 오차**로 비교한다(gradient check) — shape 가 맞아도 값이 틀릴 수 있으니 새 계층마다 확인. ③ 일부러 넣은 버그 두 개로 "오류는 그 계층 **앞쪽** 파라미터에서 드러난다"를 확인한다.

**핵심 용어**
- **OrderedDict**: 넣은 순서를 기억하는 딕셔너리 — 계층 순서 = forward 순서
- **TwoLayerNet**: Affine1 → Relu1 → Affine2 → SoftmaxWithLoss 로 조립한 2층 신경망
- **gradient check**: 역전파로 구한 기울기를 수치 미분과 비교해 구현을 검증하는 것
- **상대 오차(relative error)**: ‖a − b‖ / (‖a‖ + ‖b‖) — 기울기 크기와 무관하게 비교

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `TwoLayerNet` | `TwoLayerNet(784, 100, 10)` · `.predict` · `.loss` · `.accuracy` · `.gradient` | 계층으로 조립한 2층 신경망 |
| `rel_error()` | `np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))` | 상대 오차 |
| `failing()` | `[k for k in g_bp if rel_error(g_bp[k], g_num[k]) > tol]` | 기준을 넘는 파라미터 이름 목록 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| TwoLayerNet — 계층을 OrderedDict 에 쌓는다 | 784-10-10 망 · MNIST 3장으로 gradient() → W1 (784, 10) · b1 (10,) · W2 (10, 10) · b2 (10,) shape 확인 |
| 상대 오차로 비교하기 | 파라미터 7,960개 수치 미분(몇 초)과 역전파를 비교 → 모두 1e-7 미만 통과 |
| 왜 1e-12 인가 | log 에 1e-7 을 더한 Day 2 손실로 바꾸면 모든 상대 오차가 약 **5e-7** — 버그가 아니라 손실 식의 차이 |
| 버그 찾기 | 정상 → `[]`, 버그 ① (db 평균) → `["b2"]` (상대 오차 0.5), 버그 ② (`.T` 빠짐) → `["W1", "b1"]` |
| 역전파는 얼마나 빠른가 | 784-100-10 · 배치 100 역전파 1 step 시간(ms)을 1교시 수치 미분 추정(초)과 비교 → 수만 배 |

**핵심 내용**

#### TwoLayerNet — 계층을 쌓아 조립
> params(W1 b1 W2 b2) + layers(OrderedDict) + lastLayer(SoftmaxWithLoss)
- **predict(X)** → layers 를 순서대로 forward → 점수 (softmax 전)
- **loss(X, t)** → lastLayer.forward(predict(X), t)
- **accuracy(X, t)** → `predict(X).argmax(axis=1) == t` 의 평균 (softmax 없이 argmax)
- **gradient(X, t)** → ① loss() 로 순전파(값 저장) → ② `lastLayer.backward(1)` → ③ layers 를 **reversed** 로 backward → ④ 각 Affine 의 dW · db 모으기
- **params 와 Affine 은 같은 배열 객체** → 갱신은 반드시 **제자리** `params[k] -= lr * g` (7교시)
- ⚠ gradient() 에서 순전파를 빼먹으면 계층에 남은 **이전 배치**의 저장값으로 엉뚱한 기울기
- ⚠ `reversed(self.layers.values())` 가 안 되는 파이썬 버전 → `reversed(list(...))`
##### 자세히
```text
 params  {"W1", "b1", "W2", "b2"}  ◄── same array objects ──►  Affine1.W, Affine1.b, Affine2.W, Affine2.b
 forward   Affine1 → Relu1 → Affine2 → SoftmaxWithLoss
 backward  Affine1 ← Relu1 ← Affine2 ← SoftmaxWithLoss.backward(1)
```

#### 상대 오차로 판정한다
> 기울기 크기가 파라미터마다 달라서 절댓값 차이 대신 ‖a − b‖ / (‖a‖ + ‖b‖)
| 상대 오차 | 판정 |
|---|---|
| < **1e-7** | ✅ 구현이 맞다 |
| 1e-7 ~ 1e-4 | ⚠ 의심 — ReLU 의 꺾이는 점(0) 근처이거나 손실 식이 미세하게 다름 |
| > **1e-2** | ❌ 버그 |
- **784-10-10 (7,960개), 샘플 3장** → 수치 미분 15,920번 순전파 = 몇 초 — gradient check 는 **작은 망 · 작은 배치**로
- **왜 log 에 1e-12** → 역전파 (Y − T) 는 eps 없는 손실의 정확한 미분 → 1e-7 을 더한 손실의 수치 미분과 약 **5 × 10⁻⁷** 차이 → 기준 초과 → **1e-12** 로 줄이면 통과
- ⚠ 큰 망 · 큰 배치로 gradient check → 수치 미분이 너무 오래 걸림

#### 버그 찾기 — shape 가 맞아도 틀린다
> 오류는 그 계층의 **앞쪽** 파라미터에서 드러난다 — 틀린 dX 가 앞으로 흘러가므로
| Affine2 에 넣은 버그 | 실행 | 걸리는 파라미터 |
|---|---|---|
| ① `db = dout.mean(axis=0)` | 정상 실행 | **b2** 만 (상대 오차 0.5) |
| ② `dX = dout @ self.W` (`.T` 빠짐) | W2 (10, 10) 정사각이라 오류 없음 | **W1, b1** (앞 계층 전부) |
- **버그 ②의 W2 · b2** → 자기 dW · db 는 맞음 — 틀린 것은 앞으로 보내는 dX
- **새 계층을 만들 때마다** → gradient check (Day 4 BatchNorm · Day 5 Convolution 도)

**기본 사용형태**
```python
def gradient(self, X, t):
    self.loss(X, t)                                   # ① 순전파
    dout = self.lastLayer.backward(1)                 # ②
    for layer in reversed(list(self.layers.values())):
        dout = layer.backward(dout)                   # ③ 반대 순서로
    return {"W1": self.layers["Affine1"].dW, "b1": self.layers["Affine1"].db,
            "W2": self.layers["Affine2"].dW, "b2": self.layers["Affine2"].db}

def rel_error(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))
def failing(g_bp, g_num, tol=1e-7):
    return [k for k in g_bp if rel_error(g_bp[k], g_num[k]) > tol]
```

**연결**
- [NN D1·6교시 미분과 기울기](#/nn/1/6) — numerical_gradient 가 정답지
- [NN D4·8교시 미니 프로젝트: 미니 프레임워크 mynn.py](#/nn/4/8) — 계층을 쌓는 Sequential 로 일반화
- [NN D5·4교시 Convolution · Pooling 계층 → mynn.py](#/nn/5/4) — 새 계층도 gradient check

**연관 문제**
<<QA 6>>

### 7교시 · 미니배치 SGD

**키워드**: `미니배치`, `SGD`, `batch size`, `iteration`, `epoch`, `iters_per_epoch = N // batch_size`, `셔플 rng.permutation`, `제자리 갱신 -=`, `train · test`, `나선 데이터`

**개요**: 기울기를 빠르게 구할 수 있게 됐으니 이제 학습이다. ① Day 2 처럼 데이터 **전체**로 기울기를 구하면 MNIST 6만 장은 너무 느리다 → 무작위 일부(**미니배치**)로 기울기를 추정해 자주 갱신하는 것이 **SGD**. ② epoch 마다 셔플 → batch 단위로 역전파 · 갱신. 손실 곡선은 잡음으로 들쭉날쭉하지만 평균적으로 내려간다. ③ 3 클래스 나선 데이터에서, 은닉층 ReLU 의 꺾인 직선 조각들이 모여 곡선 경계가 되는 것을 본다.

**핵심 용어**
- **미니배치(mini-batch)**: 한 번 갱신에 쓰는 무작위 샘플 묶음
- **SGD(확률적 경사 하강법, Stochastic Gradient Descent)**: 미니배치 기울기로 파라미터를 갱신하는 경사하강
- **batch size**: 미니배치 하나의 샘플 수
- **iteration(step)**: 미니배치 하나로 기울기 → 갱신 1회
- **epoch**: 학습 데이터 전체를 한 바퀴 — iteration 수 = N // batch_size
- **일반화(generalization)**: 학습에 안 쓴 데이터(test)에서도 잘 맞히는 능력

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 3 클래스 나선 데이터 | `make_spiral()` 로 세 갈래 나선 300점 → train 240 · test 60, TwoLayerNet(2, 100, 3) 을 batch 30 · lr 1.0 · 300 epoch 로 학습하며 결정 경계가 굽어지는 과정을 그림 |

**핵심 내용**

#### 미니배치 SGD — 일부로 추정해 자주 갱신
> 미니배치 기울기 = 전체 기울기의 **잡음 섞인 추정** — 곡선은 들쭉날쭉해도 평균적으로 내려간다
| 용어 | 나선 데이터 | MNIST (8교시) |
|---|---|---|
| batch size | 30 | 100 |
| epoch 당 iteration | 240 ÷ 30 = **8** | 60,000 ÷ 100 = **600** |
| 학습률 · epoch | 1.0 · 300 | 0.1 · 10 |
- **셔플** → 매 epoch `rng.permutation` — 같은 순서면 미니배치 구성이 고정되어 기울기가 한쪽으로 치우침
- **train 정확도** → 학습 상태 · **test 정확도** → 일반화 (처음 보는 데이터)
- **손실이 중간(50~100 epoch)에 들쭉날쭉** → 미니배치 잡음 + 큰 학습률(1.0) → Day 4 옵티마이저(Momentum · Adam)
##### 자세히
```text
 for epoch:
     idx = permutation(N)                      shuffle
     for i in range(N // B):
         batch = idx[i*B : (i+1)*B]            same batch for X and t
         grads = net.gradient(X[batch], t[batch])
         params[k] -= lr * grads[k]            in-place
```

#### 나선 데이터 — 꺾인 직선 조각이 곡선 경계로
> 직선 경계(Day 2 로지스틱 회귀)로는 못 푸는 세 갈래 나선을 은닉층 100개 ReLU 망이 푼다
- **epoch 1** → 거의 한 클래스로 칠해짐 → 학습이 진행될수록 경계가 굽어 나선을 따라감
- **ReLU 하나** → 꺾인 직선 하나 → **100개**가 모이면 곡선처럼 보이는 경계
- **결정 경계 그리기** → 격자 200 × 200 = **40,000**점을 `predict(grid).argmax(axis=1)` 로 칠함

#### 흔한 실수
> 손실이 전혀 줄지 않으면 갱신 · 인덱스 · 정답 짝부터 확인
- ⚠ `params[k] = params[k] - lr * g` → **새 배열** → Affine 과 연결이 끊겨 학습 안 됨 → **`-=`** (또는 `[...] =`)
- ⚠ `idx[i:i + batch_size]` → 시작을 **`i * batch_size`** 로 곱해야 (안 곱하면 거의 같은 샘플만 반복)
- ⚠ 셔플한 인덱스로 X 만 자르고 t 는 원래 순서 → 입력 · 정답 어긋남 → 같은 `batch` 로 둘 다
- ⚠ test 데이터로 학습 금지 — test 는 **평가 전용**

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
- [NN D1·8교시 경사하강법](#/nn/1/8) — 전체 기울기 경사하강 → 미니배치로
- [NN D4·1교시 복습 퀴즈 · SGD 의 약점](#/nn/4/1) — 들쭉날쭉한 경로를 옵티마이저로 개선
- [NN D2·4교시 활성화 함수](#/nn/2/4) — 비선형(ReLU)이 경계를 꺾는다

**연관 문제**
<<QA 7>>

### 8교시 · 미니 프로젝트: MNIST 2층 MLP

**키워드**: `MNIST`, `(60000, 28, 28) uint8`, `reshape(-1, 784)`, `/ 255.0`, `TwoLayerNet(784, 100, 10)`, `학습 전 손실 log 10 ≈ 2.30`, `lr 0.1 · batch 100 · 10 epoch`, `6,000 iteration`, `test ≥ 0.95`, `int8 양자화`

**개요**: 오늘 만든 계층 · 역전파 · 미니배치 SGD 로 손글씨 숫자 0~9 를 분류한다. ① 28×28 흑백 이미지를 784 칸 한 줄로 펼치고 0~1 로 정규화. ② 학습 전 손실이 **log 10 ≈ 2.30** 인지 확인(초기화가 정상이라는 신호). ③ lr 0.1 · batch 100 · 10 epoch(6,000 iteration)로 학습해 처음 보는 1만 장에서 **95% 이상**. 마지막으로 가중치를 int8 로 줄여 보는 온디바이스 맛보기.

**핵심 용어**
- **MNIST**: 28×28 흑백 손글씨 숫자 데이터셋 — 학습 **60,000**장 · 평가 **10,000**장, 값 0~255 (uint8)
- **정규화(normalization)**: 입력을 0~1 같은 작은 범위로 맞추는 것 — 값이 크면 학습이 불안정
- **학습 곡선(learning curve)**: iteration 별 손실 · epoch 별 train/test 정확도 그래프
- **양자화(quantization)**: float 가중치를 int8 같은 작은 정수로 바꿔 메모리를 줄이는 것

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 과제 1 — 전처리 | (60000, 28, 28) uint8 → X_train (60000, 784) float 0~1, 레이블은 정수 |
| 과제 2 — 네트워크 생성 · 학습 전 손실 | TwoLayerNet(784, 100, 10) 의 처음 100장 손실 ≈ **2.3026** (= −log 0.1) |
| 과제 3 — 학습 루프 | lr 0.1 · batch 100 · 10 epoch, iteration 마다 `network.lastLayer.loss` · epoch 마다 train/test 정확도 기록 → test 0.95 이상 |
| 과제 4 — 학습 곡선 | 왼쪽 미니배치 손실 6,000점, 오른쪽 train · test 정확도와 목표선 0.95 |
| 틀린 숫자 보기 | test 1만 장 중 틀린 약 **300**장의 앞 16장을 정답 t · 예측 p 와 함께 표시 |
| 온디바이스 맛보기 — 가중치를 int8 로 줄이면? | 가중치를 −127 ~ 127 정수로 바꿔 추론 → 메모리 float64 약 **621 KB** → int8 약 **78 KB** (1/8), 정확도 변화 확인 |

**핵심 내용**

#### 전처리 — 펼치고 0~1 로
> (60000, 28, 28) uint8 → (60000, 784) float, 0~1
- **펼치기** → `reshape(-1, 784)` — 한 장 = 784 칸 행 하나 (행 = 샘플)
- **정규화** → `/ 255.0` — 0~255 그대로면 가중합이 커져 학습 불안정
- **레이블** → 정수 배열 그대로 (SoftmaxWithLoss 가 레이블을 받음)
- ⚠ `/ 255` 로 나눠도 정수 배열에 다시 넣으면 0 · 1 로 잘림 — 새 float 배열로 받는다

#### 학습 전 손실 = log 10 ≈ 2.30
> 가중치 × 0.01 → 점수 ≈ 0 → softmax 가 10개 클래스에 거의 균등(**0.1**) → CE = −log 0.1 = **2.3026**
- **2.30 근처** → 초기화 · 손실 구현이 정상이라는 신호
- **크게 다르면** → 전처리(/255 빠짐) · 가중치 크기 · 손실 식부터 의심

#### 학습과 결과 읽기
> lr 0.1 · batch 100 · 10 epoch = 600 × 10 = **6,000** iteration, Colab CPU 30초 ~ 1분
- **기준 결과 (seed 0)** → train **0.9747**, test **0.9694** (CPU 마다 넷째 자리는 다를 수 있음)
- **train > test 조금** → 정상 — 차이가 크게 벌어지면 **과적합** (Day 4)
- **수치 미분으로 했다면** → 6,000 step × 수십 초 → 며칠 (50초면 약 **3.5일**) — 역전파는 그래프를 한 번 거꾸로 지나며 모든 기울기를 재사용
- **틀린 숫자 약 300장** → 사람이 봐도 헷갈리는 글씨가 많다
- **미니배치 손실 기록** → `gradient()` 안의 순전파가 계산해 둔 `network.lastLayer.loss` 를 그대로

#### 온디바이스 맛보기 — int8 양자화
> scale = max|W| / 127 → W 를 −127 ~ 127 정수로, 추론 때 × scale 로 되돌려 쓴다
- **메모리** → float64(8 바이트) → int8(1 바이트) = **1/8** (79,510개 → 약 621 KB → 78 KB)
- **정확도** → 셀에서 float 결과와 직접 비교 (얼마나 떨어지는지가 이 실험의 질문) — Vision 과정 이후 Quantization 의 핵심 아이디어
- **(선택) 저장** → `np.savez` 로 학습한 params 를 Drive 에 — Day 4 에서 이어 쓰기

**기본 사용형태**
```python
X_train = x_train.reshape(-1, 784) / 255.0
X_test  = x_test.reshape(-1, 784) / 255.0
t_train, t_test = y_train.astype(int), y_test.astype(int)
network = TwoLayerNet(784, 100, 10)
loss0 = network.loss(X_train[:100], t_train[:100])    # ≈ 2.30

def quantize(W):
    scale = np.abs(W).max() / 127
    return np.round(W / scale).astype(np.int8), scale
```

**연결**
- [NN D4·8교시 미니 프로젝트: 미니 프레임워크 mynn.py](#/nn/4/8) — 이 망을 개선 기법으로 다시
- [NN D5·5교시 SimpleConvNet 으로 MNIST 학습](#/nn/5/5) — 같은 데이터를 CNN 으로
- [NN D2·5교시 행렬로 보는 순전파](#/nn/2/5) — 79,510개 · 메모리 계산
- [NN D4·5교시 과적합](#/nn/4/5) — train 과 test 의 차이

**연관 문제**
<<QA 8>>
