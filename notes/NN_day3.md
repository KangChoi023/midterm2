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

#### 수치 미분의 비용
> 파라미터 하나마다 손실을 두 번(+h, −h) 계산 → step 당 순전파 2P 번 × 순전파 비용 ∝ P = **O(P²)**
- **`n_loss_evals(P)`** → 2P — 784-100-10(P = 79,510)이면 step 당 손실 **159,020번** (수십 ~ 100 초)
- **시간 복잡도 O(P²)** → 파라미터 10배면 시간 **100배** — 컴퓨터가 빨라져도 이 차이는 줄지 않는다
- **`count_params`** → 모든 W · b 원소 수의 합 = P
- **쓰임** → 느리지만 틀리기 어려워 역전파의 **정답지**(gradient check)로 남긴다

#### 역전파
> 출력에서 입력 방향으로 기울기를 한 번에 전달해 **모든 파라미터의 기울기**를 구하는 방법 — 순전파 1번 + 역방향 1번
- **비용** → 순전파의 2~3배, **O(P)**
- **예** → 784-100-10 · 배치 100: 역전파 1 step 수 ms vs 수치 미분 수십 초 → **수만 배** 차이 (CPU 마다 다름)
- **원리** → 계산 그래프 · 연쇄 법칙 (2 · 3교시)

**기본 사용형태**
```python
def count_params(params): return sum(p.size for p in params.values())
def n_loss_evals(P):      return 2 * P            # 파라미터마다 +h, −h 두 번
n_loss_evals(79_510)                              # 159020
```

**연결**
- [NN D1·6교시 미분과 기울기](#/nn/1/6) — numerical_gradient (중앙 차분)
- [NN D2·8교시 로지스틱 회귀](#/nn/2/8) — "왜 Day 3 에서 역전파가 필요한가" 표

**상세 학습**

#### 이 교시에서 답하는 질문
Day 2 까지는 기울기를 손으로 유도하거나(층 하나일 때) Day 1 의 수치 미분으로 구했다. 그런데 층이 쌓이면 손 유도는 어렵고, 수치 미분은 너무 느리다. 이 교시는 수치 미분이 **왜, 얼마나** 느린지 직접 재 보고, 오늘 만들 **역전파**가 그 문제를 어떻게 푸는지 미리 본다.

#### 개념 1 · 수치 미분은 파라미터 수의 제곱으로 느려진다
수치 미분은 파라미터 하나를 +h, −h 로 움직여 손실을 두 번 계산한다. 파라미터가 P 개면 한 step 에 순전파를 **2P 번** 해야 한다. 그런데 순전파 한 번의 비용도 대부분 행렬곱이라 P 에 비례한다.

> 수치 미분 1 step 비용 ∝ 2P × P = **O(P²)**

![수치 미분과 역전파의 비용 — 784-100-10 망은 수치 미분이면 step 당 손실을 159,020번 계산한다](assets/fig/nn/d3-cost.svg)

- O(P²) 는 "파라미터가 10배면 시간 100배" 라는 뜻이다. 컴퓨터가 빨라져도 이 **차수(order)** 의 차이는 줄지 않는다.
- 784-100-10 망은 step 한 번에 수십 ~ 100 초. 학습은 수천 step 이 필요하니 사실상 불가능하다.

#### 개념 2 · 역전파는 한 번에 모든 기울기를 구한다
역전파는 순전파 1번으로 중간값을 저장해 두고, 출력에서 입력 방향으로 **한 번** 거꾸로 지나가며 모든 파라미터의 기울기를 구한다. 비용은 순전파의 2~3배, 즉 **O(P)**. 784-100-10 · 배치 100 이면 수 ms — 수치 미분과 **수만 배** 차이다.

수치 미분은 버리지 않는다. 느리지만 구현이 단순해 틀리기 어렵기 때문에, 역전파가 맞는지 확인하는 **정답지(gradient check, 6교시)** 로 쓴다.

#### 코드로 확인
```python
def count_params(params): return sum(p.size for p in params.values())
def n_loss_evals(P):      return 2 * P          # 파라미터마다 +h, -h 두 번
n_loss_evals(count_params(init_params(100)))
```
- **무엇을 하나**: 784-h-10 망의 파라미터 수를 세고, 수치 미분 한 step 에 필요한 손실 계산 횟수를 구한다.
- **어떤 개념**: 개념 1 — 2P 번.
- **출력**: h = 1 · 2 · 4 이면 P = 805 · 1,600 · 3,190, h = 100 이면 P = 79,510 → `159020`

#### 한 번에 정리
- 수치 미분 = 파라미터마다 손실 2번 → O(P²) → 큰 망에서는 불가능.
- 역전파 = 순전파 1번 + 역방향 1번 → O(P), 수만 배 빠르다.
- 수치 미분은 역전파 검증용 정답지로 남긴다.

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

#### 계산 그래프
> 계산을 **노드(연산)** 와 **엣지(값)** 로 그린 그림 — 순전파는 값을 왼 → 오, 역전파는 기울기를 오 → 왼으로 나른다
- **역전파 값의 뜻** → "그 값이 1 오르면 최종 출력이 얼마나 오르나" = ∂(출력)/∂(그 값)
- **출발** → 출력에서 **1** 로 시작해 노드마다 국소 미분을 곱하며 거꾸로
- **예** → 사과 100원 × 2 + 귤 150원 × 3, 소비세 1.1 → 715원 · 사과 가격의 기울기 **2.2** (2개 × 1.1), 소비세의 기울기 **650**
- **검증** → `numerical_gradient` 와 같은 값이면 역전파가 맞다

#### 국소 계산
> 노드는 전체를 몰라도 **자기 입력과 위에서 내려온 기울기만으로** 자기 몫을 계산한다
- **왜 중요** → 노드 종류마다 forward · backward 를 한 번만 만들면 아무리 큰 망도 조립된다 (3 ~ 5교시)
- **중간값 저장** → 순전파 때 저장해 두면 역전파 때 다시 계산하지 않는다
- **동시에** → 역전파 한 번으로 모든 입력의 기울기 (수치 미분은 입력마다 따로)

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

**상세 학습**

#### 이 교시에서 답하는 질문
역전파는 어떻게 "한 번에" 모든 기울기를 구할까? 이 교시는 그 바탕인 **계산 그래프** — 계산을 작은 노드의 연결로 그린 그림 — 를 사과 · 귤 쇼핑 예로 이해한다.

#### 개념 1 · 계산 그래프 — 값은 앞으로, 기울기는 뒤로
복잡한 식도 "작은 계산의 연결" 로 쪼갤 수 있다. 계산을 **노드(연산)** 와 **엣지(값)** 로 그린 것이 계산 그래프다.
- **순전파**: 왼쪽에서 오른쪽으로 값을 계산해 나른다.
- **역전파**: 오른쪽에서 왼쪽으로 **기울기** ∂(최종 출력)/∂(그 값) 을 나른다. 출력에서 **1** 로 시작해, 노드를 지날 때마다 그 노드의 국소 미분을 곱한다.

![사과 · 귤 쇼핑의 계산 그래프 — 파랑은 순전파 값, 빨강 ∂ 는 최종 금액에 대한 기울기](assets/fig/nn/d3-shopping.svg)

역전파로 얻은 값의 뜻: "그 값이 1 오르면 최종 금액이 얼마나 오르나". 사과 가격의 기울기가 **2.2** 라면 사과가 1원 오를 때 최종 금액이 2.2원 오른다 (2개 × 세금 1.1). 소비세의 기울기는 **650** (세금 전 합계).

#### 개념 2 · 국소 계산 — 노드는 자기 일만 한다
마지막 × 소비세 노드는 앞에서 사과가 몇 개였는지 몰라도 된다. 자기 입력(650, 1.1)과 위에서 내려온 기울기만으로 자기 몫을 계산해 넘긴다. 이것을 **국소 계산**이라 한다.
- 신경망이 아무리 커도 노드 하나가 하는 일은 단순하다 → **노드 종류마다 forward · backward 를 한 번만** 만들어 조립하면 된다 (3 ~ 5교시).
- 중간값을 순전파 때 저장해 두면 역전파 때 다시 계산하지 않는다.
- 역전파 한 번으로 **모든 입력의 기울기가 동시에** 나온다 — 수치 미분은 입력마다 따로 계산해야 했다.

#### 코드로 확인
```python
apple_price  = 100 * 2;  orange_price = 150 * 3
all_price    = apple_price + orange_price
price        = all_price * 1.1                       # 순전파
dprice = 1                                           # 역전파: 출력에서 1
dall_price = dprice * 1.1;   dtax = dprice * all_price   # 곱셈 노드 → 상대편
dapple = dall_price * 1 * 2;  dapple_num = dall_price * 1 * 100
```
- **무엇을 하나**: 그래프를 따라 값을 계산하고, 출력에서 1 을 출발시켜 기울기를 거꾸로 구한다.
- **어떤 개념**: 개념 1 — 덧셈 노드는 그대로(× 1), 곱셈 노드는 상대편 값을 곱한다.
- **출력**: price **715**, dapple **2.2**, dapple_num **110**, dtax **650** — `numerical_gradient` 결과와 같다

#### 한 번에 정리
- 계산 그래프: 순전파는 값을 앞으로, 역전파는 기울기를 뒤로.
- 역전파 값 = "그 값이 1 오르면 출력이 얼마나 오르나".
- 노드는 국소 계산만 한다 → 노드별 forward/backward 를 만들어 조립.

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

#### 연쇄 법칙
> 합성 함수의 미분은 단계별 미분의 곱: ∂z/∂x = ∂z/∂t · ∂t/∂x — 역전파는 이것을 그래프 위에서 실행하는 것
- **역전파 절차** → 출력에서 1 을 출발시켜 노드를 지날 때마다 그 노드의 **국소 미분**을 곱한다
- **예** → z = (x + y)², x = 3, y = 1 → t = 4, z = 16, ∂z/∂x = 2t · 1 = **8**
- **Day 2 에서** → ½(y − t)² 의 "바깥 미분 × 안쪽 미분" 이 연쇄 법칙이었다

#### 덧셈 · 곱셈 노드
> 덧셈 노드는 기울기를 **그대로 복사**, 곱셈 노드는 **상대편 입력을 곱해** 보낸다
- **덧셈 z = x + y** → dx = dout, dy = dout (`AddLayer`)
- **곱셈 z = x · y** → dx = dout · y, dy = dout · x (`MulLayer`) — forward 에서 `self.x`, `self.y` 를 저장해야
- **계층의 모양** → forward(값 계산 + 필요한 값 저장) + backward(dout 받아 입력 쪽 기울기 반환) — 오늘 모든 계층 공통
- ⚠ 곱셈 노드에서 자기 자신을 곱하면(`dout * self.x`) 틀림 — **상대편**
- ⚠ 같은 계층 객체를 두 곳에 재사용하면 저장값이 덮어써진다 — 노드마다 객체 하나, backward 는 반대 순서

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

**상세 학습**

#### 이 교시에서 답하는 질문
2교시에서 "덧셈 노드는 그대로 보낸다, 곱셈 노드는 상대편을 곱한다" 는 규칙을 썼다. 이 규칙은 어디서 왔을까? 이 교시는 역전파의 수학적 원리, **연쇄 법칙**을 보고, 두 기본 노드를 **계층(layer) 클래스**로 만든다.

#### 개념 1 · 연쇄 법칙 — 단계별 미분의 곱
합성 함수의 미분은 각 단계 미분의 곱이다.

> z = (x + y)² 를 t = x + y, z = t² 로 나누면 ∂z/∂x = ∂z/∂t · ∂t/∂x = 2t · 1

x = 3, y = 1 이면 t = 4, z = 16, ∂z/∂x = 8. 역전파는 이 연쇄 법칙을 그래프 위에서 실행하는 것이다 — 출력에서 1 을 출발시켜, 노드를 지날 때마다 그 노드의 **국소 미분**을 곱한다. Day 2 에서 ½(y − t)² 를 미분할 때 쓴 "바깥 미분 × 안쪽 미분" 이 바로 이것이었다.

#### 개념 2 · 덧셈 노드와 곱셈 노드
![덧셈 노드는 기울기를 그대로 복사하고, 곱셈 노드는 상대편 입력을 곱해 보낸다](assets/fig/nn/d3-nodes.svg)

| 노드 | 국소 미분 | 역전파 (위에서 온 기울기 dout) |
|---|---|---|
| 덧셈 z = x + y | ∂z/∂x = 1, ∂z/∂y = 1 | dx = dout, dy = dout — **그대로 복사** |
| 곱셈 z = x · y | ∂z/∂x = y, ∂z/∂y = x | dx = dout · y, dy = dout · x — **서로 바꿔 곱함** |

곱셈 노드는 역전파 때 순전파의 입력 x, y 가 필요하다. 그래서 계층 클래스는 **forward 에서 입력을 저장**해 두고 backward 에서 쓴다. 이 "forward 에서 저장 → backward 에서 사용" 패턴이 오늘 만드는 모든 계층의 공통 구조다.

#### 코드로 확인
```python
class MulLayer:
    def forward(self, x, y):
        self.x, self.y = x, y            # 역전파에 쓰려고 저장
        return x * y
    def backward(self, dout):
        return dout * self.y, dout * self.x    # 상대편 입력을 곱한다
```
- **무엇을 하나**: 곱셈 노드를 forward(값 + 저장) · backward(기울기 반환) 두 메서드로 만든다.
- **어떤 개념**: 개념 2 — 곱셈 노드는 상대편을 곱한다.
- **출력**: 사과만 (100 × 2) × 1.1 = **220**, backward → 사과 가격 **2.2**, 개수 **110**, 소비세 **200**

#### 한 번에 정리
- 연쇄 법칙: 합성 함수의 미분 = 단계별 국소 미분의 곱 → 역전파의 원리.
- 덧셈 노드는 그대로 복사, 곱셈 노드는 서로 바꿔 곱함.
- 계층 = forward(값 + 저장) + backward(dout → 입력 쪽 기울기).
- ⚠ 같은 계층 객체를 두 곳에 쓰면 저장값이 덮어써진다 — 노드마다 객체 하나.

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
> 순전파 때 0 이하였던 자리(mask)는 역전파 때도 기울기를 막고(0), 양수였던 자리는 그대로 통과시킨다
- **`Relu.forward`** → `mask = (x <= 0)` 저장, `out = x.copy()` 의 mask 자리 0
- **`Relu.backward`** → `dx = dout.copy()` 의 mask 자리 0 (나머지 국소 미분 1)
- **예** → x = [[1, −0.5], [−2, 3]] → backward(1) = [[1, 0], [0, 1]]
- ⚠ 입력 x 를 직접 바꾸면 앞 계층 출력(원본)이 망가진다 → `x.copy()`

#### Sigmoid 계층
> ∂L/∂x = dout · y(1 − y) — 순전파 출력 y 하나만 있으면 기울기가 나온다 (Day 2 `sigmoid_grad` 와 같은 식)
- **`Sigmoid.forward`** → `self.out` 저장 (backward 에서 exp 를 다시 계산하지 않음)
- **유도** → ×(−1) → exp → +1 → 1/x 네 노드의 국소 미분을 곱해 정리
- **예** → x = 1, −2, 0 에서 기울기 0.1966, 0.1050, **0.25**(최대)
- ⚠ forward 없이 backward 를 부르면 `self.mask` · `self.out` 이 None — 항상 forward → backward

#### 계층 하나 검증하기
> L = Σ (forward(x) ⊙ dout) 을 수치 미분하면 backward(dout) 와 같아야 한다
- **왜** → 이 L 의 ∂L/∂(출력) 이 정확히 dout 이라, 계층 하나만 떼어 시험할 수 있다
- ⚠ 수치 미분이 forward 를 여러 번 불러 저장값이 바뀜 → 비교 전에 forward 를 한 번 더

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

**상세 학습**

#### 이 교시에서 답하는 질문
Day 2 의 활성화 함수(ReLU · sigmoid)를 역전파가 되는 **계층**으로 만든다. 이제 숫자 하나가 아니라 배열 전체를 다룬다 — 배열의 원소 하나하나가 각자 독립된 노드라고 생각하면 된다.

#### 개념 1 · ReLU 계층 = 전기 스위치
ReLU 의 미분은 x > 0 이면 1, 아니면 0 이다. 그래서 역전파는 **순전파 때 막혔던(0 이하) 자리는 기울기도 막고, 통과한 자리는 그대로 보낸다**. 어느 자리가 막혔는지 기억하려고 forward 에서 True/False 배열 `mask = (x <= 0)` 을 저장한다.

![ReLU 계층 — 순전파에서 0 이 된 자리(빨강)는 역전파에서도 기울기가 0](assets/fig/nn/d3-relu.svg)

#### 개념 2 · Sigmoid 계층 — 출력 하나로 기울기를 낸다
sigmoid y = 1/(1 + e⁻ˣ) 를 계산 그래프로 쪼개면 ×(−1) → exp → +1 → 1/x 네 노드다. 각 노드의 국소 미분을 곱해 정리하면

> ∂L/∂x = dout · y(1 − y)

Day 2 의 `sigmoid_grad` 와 같은 식이다. 역전파에 필요한 것이 순전파 출력 y **하나뿐**이므로 forward 에서 `self.out` 만 저장하면 된다.

#### 개념 3 · 계층 하나를 떼어 검증하는 법
L = Σ (forward(x) ⊙ dout) 라는 손실을 만들면, 이 L 을 출력으로 미분한 값이 정확히 dout 이 된다. 그래서 이 L 을 수치 미분한 결과가 backward(dout) 와 같으면 그 계층의 역전파가 맞다. 망 전체를 만들기 전에 계층마다 확인할 수 있다.

#### 코드로 확인
```python
class Relu:
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy(); out[self.mask] = 0
        return out
    def backward(self, dout):
        dx = dout.copy(); dx[self.mask] = 0
        return dx
```
- **무엇을 하나**: 0 이하인 자리를 기억해 순전파 · 역전파 모두 그 자리를 0 으로 만든다.
- **어떤 개념**: 개념 1 — 스위치.
- **출력**: x = [[1, −0.5], [−2, 3]] → out [[1, 0], [0, 3]], backward(1) [[1, 0], [0, 1]], 입력 x 는 그대로

```python
class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x)); return self.out
    def backward(self, dout):
        return dout * self.out * (1 - self.out)
```
- **무엇을 하나**: 순전파 출력을 저장하고, 역전파에서 그것으로 y(1 − y) 를 곱한다.
- **어떤 개념**: 개념 2.
- **출력**: x = [1, −2] → y = [0.7311, 0.1192], backward(1) = [**0.1966**, **0.1050**], x = 0 이면 **0.25**

#### 한 번에 정리
- ReLU 계층: forward 에서 mask 저장, backward 에서 mask 자리 0 — 스위치.
- Sigmoid 계층: forward 에서 출력 저장, backward 는 dout · y(1 − y).
- 계층 검증: L = Σ forward(x) ⊙ dout 의 수치 미분 = backward(dout).
- ⚠ `x[mask] = 0` 으로 입력을 직접 바꾸면 앞 계층 출력이 망가진다 → `x.copy()`.

**연관 문제**
<<QA 4>>

### 5교시 · Affine · Softmax-with-Loss 계층

**키워드**: `Affine`, `기울기 shape = 자기 shape`, `dX = dout Wᵀ`, `dW = Xᵀ dout`, `db = Σ dout`, `SoftmaxWithLoss`, `(Y − T) / N`, `y[np.arange(N), t]`, `1e-12`

**개요**: 신경망의 나머지 두 부품. ① **Affine** 은 Day 2 가설 Y = XW + b 그대로의 계층 — 역전파 식은 "기울기는 자기 자신과 shape 가 같다" 하나로 유도된다: dX = dout Wᵀ, dW = Xᵀ dout, db = 행 방향 합. ② **Softmax-with-Loss** 는 softmax + cross-entropy 를 한 계층으로 묶은 것 — 복잡한 미분이 상쇄되어 기울기가 **(Y − T)/N**. ③ Day 2 에서 손으로 유도한 Xᵀ(P − T)/N 이 바로 이 두 계층의 역전파였다.

**핵심 용어**
- **Affine 계층**: Y = XW + b 를 계산하는 완전연결(fully connected) 계층
- **Softmax-with-Loss 계층**: softmax 출력과 cross-entropy 손실을 묶은 마지막 계층
- **레이블(label) 정답**: one-hot 대신 정답 번호(정수)만 둔 배열 — `[2, 0]`
- **팬시 인덱싱(fancy indexing)**: 인덱스 배열로 원소를 고르는 것 — `y[np.arange(N), t]` 는 행마다 정답 자리의 확률 하나씩

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

#### Affine 계층의 역전파
> 순전파 Y = X @ W + b, 역전파는 "기울기는 자기 자신과 shape 가 같다" 하나로 유도된다
- **dX = dout @ Wᵀ** → (N, M)(M, D) = (N, D) — 앞 계층으로 보내는 값(반환)
- **dW = Xᵀ @ dout · db = dout.sum(axis=0)** → 저장해 두고 갱신에 쓴다 (`self.dW`, `self.db`)
- **편향이 합인 이유** → 브로드캐스팅으로 같은 b 가 N 개 행에 쓰였으니 기울기도 N 개를 더한다
- **`Affine` 의 W, b** → params 와 같은 배열 객체 — 바깥에서 제자리 갱신하면 계층에도 반영
- ⚠ shape 만 보고 식을 바꾸면 정사각 행렬일 때 오류 없이 틀린다 · db 를 평균으로 쓰면 틀림

#### Softmax-with-Loss 계층
> softmax 와 cross-entropy 를 한 계층으로 묶으면 역전파가 **(Y − T) / N** — 복잡한 미분이 상쇄된다
- **`SoftmaxWithLoss.backward`** → `y.copy()` 의 정답 자리에서 1 을 빼고 N 으로 나눈다
- **`cross_entropy_error`** → 정답이 레이블이든 one-hot 이든 팬시 인덱싱 `y[np.arange(N), t]` 로 정답 확률을 골라 −log 평균 (log 에 1e-12)
- **Day 2 에서** → 로지스틱 회귀의 Xᵀ(P − T)/N 은 dout = (P − T)/N 을 넣은 Affine 의 Xᵀ · dout
- **예** → 정답 확률 0.737 인 행(N = 2) → 정답 칸 기울기 (0.737 − 1)/2 = **−0.132**
- ⚠ `/ N` 을 빠뜨리면 기울기가 배치 크기만큼 커진다 · 레이블은 정수(`astype(int)`)

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

**상세 학습**

#### 이 교시에서 답하는 질문
신경망의 나머지 두 부품을 만든다. **Affine** 은 Day 2 의 가설 Y = XW + b 그대로의 계층이고, **Softmax-with-Loss** 는 출력층 softmax 와 손실 cross-entropy 를 한 계층으로 묶은 것이다. 두 계층의 역전파 식이 어디서 나오는지 본다.

#### 개념 1 · Affine 역전파 — shape 로 유도한다
위에서 내려온 기울기 dout = ∂L/∂Y 는 Y 와 같은 (N, M) 이다. 기억할 원칙은 하나: **기울기는 자기 자신과 shape 가 같다**.

![Affine 역전파 — dX · dW · db 의 식과 shape](assets/fig/nn/d3-affine.svg)

- dX 는 X 와 같은 (N, D) 여야 하므로 dout (N, M) 에 Wᵀ (M, D) 를 곱하는 방법뿐이다 → **dX = dout @ Wᵀ**.
- dW 는 (D, M) 이어야 하므로 Xᵀ (D, N) 에 dout (N, M) → **dW = Xᵀ @ dout**.
- 편향 b 는 순전파에서 브로드캐스팅으로 N 개 행 모두에 더해졌다. 같은 b 가 N 번 쓰였으니 기울기도 N 개를 **모두 더한다** → **db = dout.sum(axis=0)**.
- dX 는 앞 계층으로 돌려주고, dW · db 는 계층에 저장해 두었다가 갱신에 쓴다.

#### 개념 2 · Softmax-with-Loss — 기울기가 (Y − T) / N
분류 신경망의 끝은 언제나 softmax → cross-entropy 다. 둘을 한 계층으로 묶어 역전파하면 softmax 의 복잡한 미분과 log 의 미분이 깔끔하게 상쇄되어

> ∂L/∂X = (Y − T) / N  (Y = softmax 출력, T = one-hot 정답)

Day 2 의 선형 회귀 · 로지스틱 회귀에서 본 "(예측 − 정답)" 이 다시 나온다. 잘 설계된 출력층 + 손실의 짝은 모두 이 모양이 되도록 만들어져 있다. 실제 계산은 softmax 출력을 복사해 **정답 자리에서만 1 을 빼고** N 으로 나누면 된다 — 정답 칸은 음수(확률을 키우는 쪽), 나머지는 양수(줄이는 쪽).

#### 코드로 확인
```python
class Affine:
    def forward(self, X):
        self.X = X
        return X @ self.W + self.b
    def backward(self, dout):
        self.dW = self.X.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T
```
- **무엇을 하나**: 순전파에서 X 를 저장하고, 역전파에서 dW · db 를 저장한 뒤 dX 를 돌려준다.
- **어떤 개념**: 개념 1 — shape 로 유도한 세 식.
- **출력**: N, D, M = 2, 3, 4 → dX (2, 3), dW (3, 4), db (4,) — 모두 수치 미분과 일치

```python
def backward(self, dout=1):                      # SoftmaxWithLoss
    N = len(self.t)
    dx = self.y.copy(); dx[np.arange(N), self.t] -= 1
    return dx / N
```
- **무엇을 하나**: softmax 출력에서 행마다 정답 칸만 1 을 빼고 배치 크기로 나눈다.
- **어떤 개념**: 개념 2 — (Y − T)/N.
- **출력**: 점수 [[0.3, 2.9, 4.0], [1.0, 2.0, 0.5]], 정답 [2, 0] → 손실 **0.8850**, 첫 행 정답 칸 기울기 (0.737 − 1)/2 = **−0.132**

#### 한 번에 정리
- Affine: dX = dout Wᵀ, dW = Xᵀ dout, db = 행 방향 합 — 기울기 shape = 자기 shape.
- Softmax-with-Loss: (Y − T)/N — Day 2 의 "(예측 − 정답)".
- Day 2 의 손 유도 기울기 = Affine + Softmax-with-Loss 계층의 역전파.
- ⚠ db 를 평균으로 쓰면 틀림 — /N 은 Softmax-with-Loss 가 이미 했다.

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

#### TwoLayerNet — 계층을 조립한 신경망
> Affine1 → Relu1 → Affine2 를 OrderedDict 에 쌓고, 끝에 SoftmaxWithLoss — 순서대로 forward, 반대 순서로 backward
- **predict · loss · accuracy** → 점수 계산 · 손실 · argmax 정확도
- **gradient()** → ① loss 로 순전파(값 저장) ② `lastLayer.backward(1)` ③ layers 를 reversed 로 backward ④ 각 Affine 의 dW · db 모으기
- **params** → Affine 과 같은 배열 객체 → 갱신은 반드시 제자리 `-=`
- ⚠ gradient() 에서 순전파를 빼면 이전 배치의 저장값으로 엉뚱한 기울기 · `reversed(list(...))`

#### gradient check — 상대 오차
> 역전파 기울기를 수치 미분과 비교해 구현을 검증한다 — 크기와 무관하게 `rel_error` = ‖a − b‖ / (‖a‖ + ‖b‖)
- **판정** → < **1e-7** 통과 · 1e-7 ~ 1e-4 의심 · > **1e-2** 버그
- **작게 검사** → 784-10-10(7,960개) · 샘플 3장 — 수치 미분이 몇 초면 된다
- **log 에 1e-12 인 이유** → 1e-7 을 더한 손실은 (Y − T) 와 미세하게 달라 상대 오차 약 **5×10⁻⁷** → 기준 초과
- ⚠ 큰 망 · 큰 배치로 검사하면 수치 미분이 너무 오래 걸린다

#### shape 가 맞아도 틀리는 버그
> 오류는 그 계층의 **앞쪽** 파라미터에서 드러난다 — 틀린 dX 가 앞으로 흘러가므로 (`failing()` 으로 찾기)
- **db 를 평균으로** → **b2** 만 틀림 (상대 오차 0.5)
- **dX 에서 `.T` 를 빠뜨림** → W2 가 (10, 10) 정사각이라 오류 없이 실행 → 앞쪽 **W1, b1** 이 틀림
- **원칙** → 새 계층을 만들 때마다 gradient check (Day 4 BatchNorm · Day 5 Convolution)

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

**상세 학습**

#### 이 교시에서 답하는 질문
부품이 다 모였다. 계층을 쌓아 2층 신경망 **TwoLayerNet** 을 조립하고, 역전파로 구한 기울기가 정말 맞는지 수치 미분과 비교하는 **gradient check** 를 한다. shape 가 맞아도 값이 틀리는 버그를 어떻게 찾는지도 본다.

#### 개념 1 · 계층을 쌓아 조립한다
![TwoLayerNet — 순전파는 왼쪽부터, 역전파는 SoftmaxWithLoss 에서 시작해 반대 순서로](assets/fig/nn/d3-net.svg)

- 계층을 넣은 순서를 기억하는 딕셔너리 **OrderedDict** 에 Affine1 → Relu1 → Affine2 를 쌓고, 끝에 SoftmaxWithLoss 를 둔다.
- **gradient()** 는 네 단계다: ① loss() 로 순전파(계층마다 값 저장) ② `lastLayer.backward(1)` ③ layers 를 **반대 순서(reversed)** 로 backward ④ 각 Affine 의 dW · db 를 모은다.
- params 딕셔너리와 Affine 계층은 **같은 배열 객체**를 가리킨다 → 갱신은 반드시 제자리(`-=`).

#### 개념 2 · gradient check — 상대 오차로 판정
기울기 값의 크기는 파라미터마다 달라서 차이의 절댓값보다 **상대 오차**로 비교한다.

> rel_error(a, b) = ‖a − b‖ / (‖a‖ + ‖b‖)

| 상대 오차 | 판정 |
|---|---|
| < 1e-7 | 통과 — 구현이 맞다 |
| 1e-7 ~ 1e-4 | 의심 — ReLU 의 꺾이는 점 근처이거나 손실 식이 미세하게 다름 |
| > 1e-2 | 버그 |

- 수치 미분이 느리므로 **작은 망(784-10-10, 파라미터 7,960개) · 샘플 3장**으로 검사한다.
- log 에 1e-7 을 더한 Day 2 손실로 검사하면 모든 상대 오차가 약 5×10⁻⁷ 이 나온다. 역전파 (Y − T) 는 eps 없는 손실의 정확한 미분이라 생기는 차이 — 버그가 아니다. 그래서 오늘은 **1e-12** 를 쓴다.

#### 개념 3 · shape 가 맞아도 틀리는 버그
- **db = dout.mean(axis=0)** (합 대신 평균): b2 만 틀린다 (상대 오차 0.5).
- **dX = dout @ self.W** (`.T` 빠뜨림): W2 가 (10, 10) 정사각이라 오류 없이 실행된다. 그런데 Affine2 의 dX 가 틀려 그 **앞쪽(Affine1) 의 W1, b1 이 모두 틀린다**.
- 교훈: 오류는 그 계층의 **앞쪽** 파라미터에서 드러난다. 새 계층을 만들 때마다 gradient check.

#### 코드로 확인
```python
def gradient(self, X, t):
    self.loss(X, t)                                   # ① 순전파
    dout = self.lastLayer.backward(1)                 # ②
    for layer in reversed(list(self.layers.values())):
        dout = layer.backward(dout)                   # ③ 반대 순서
    return {"W1": self.layers["Affine1"].dW, "b1": self.layers["Affine1"].db,
            "W2": self.layers["Affine2"].dW, "b2": self.layers["Affine2"].db}
```
- **무엇을 하나**: 순전파로 값을 저장한 뒤, 손실 계층부터 거꾸로 backward 를 불러 각 Affine 의 기울기를 모은다.
- **어떤 개념**: 개념 1 — 순서대로 forward, 반대로 backward.
- **출력**: 784-10-10, MNIST 3장 → W1 (784, 10), b1 (10,), W2 (10, 10), b2 (10,)

```python
def rel_error(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))
def failing(g_bp, g_num, tol=1e-7):
    return [k for k in g_bp if rel_error(g_bp[k], g_num[k]) > tol]
```
- **무엇을 하나**: 역전파와 수치 미분의 상대 오차를 재고, 기준을 넘는 파라미터 이름만 고른다.
- **어떤 개념**: 개념 2 · 3.
- **출력**: 정상 `[]`, db 버그 `['b2']`, dX 버그 `['W1', 'b1']`

#### 한 번에 정리
- TwoLayerNet: OrderedDict 로 쌓아 순서대로 forward, reversed 로 backward.
- gradient check: 상대 오차 < 1e-7 이면 통과. 작은 망 · 작은 배치로.
- shape 가 맞아도 값이 틀릴 수 있다 — 오류는 앞쪽 파라미터에서 드러난다.

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

#### 미니배치 SGD
> 전체 데이터 대신 무작위 일부(**미니배치**)로 기울기를 추정해 자주 갱신하는 경사하강 — 기울기는 잡음 섞인 추정이라 손실이 들쭉날쭉하지만 평균적으로 내려간다
- **batch size · iteration · epoch** → 한 번 갱신에 쓰는 샘플 수 · 갱신 1회 · 데이터 한 바퀴 (iteration = N // batch_size)
- **셔플** → epoch 마다 `rng.permutation` — 같은 순서면 미니배치 구성이 고정되어 기울기가 치우친다
- **train · test 정확도** → 학습 상태 · 일반화(처음 보는 데이터)
- **예** → MNIST 60,000장 · batch 100 → epoch 당 **600** iteration
- ⚠ `params[k] = params[k] - lr * g` 는 새 배열 → 계층과 연결이 끊김 → **`-=`**
- ⚠ 배치 인덱스는 `idx[i * batch_size:(i + 1) * batch_size]`, X 와 t 를 같은 batch 로

#### ReLU 망의 곡선 경계
> ReLU 뉴런 하나는 꺾인 직선 하나 — 은닉층에 많이 모이면 곡선 경계가 된다
- **예** → 3 클래스 나선 데이터(직선으로 못 나눔)를 TwoLayerNet(2, 100, 3) 이 학습하며 경계가 나선을 따라 굽는다
- **Day 4 에서** → 큰 학습률(1.0)에서 손실이 들쭉날쭉한 문제를 옵티마이저(Momentum · Adam)로

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

**상세 학습**

#### 이 교시에서 답하는 질문
기울기를 빠르게 구할 수 있게 됐으니 이제 학습이다. Day 2 는 데이터 **전체**로 기울기를 구했지만, MNIST 6만 장을 매 step 전부 쓰면 느리다. 이 교시는 무작위 일부로 기울기를 추정하는 **미니배치 SGD** 와, 은닉층이 곡선 경계를 만드는 모습을 본다.

#### 개념 1 · 미니배치 SGD
무작위로 고른 일부(미니배치)로 기울기를 추정해 자주 갱신하는 경사하강을 **확률적 경사 하강법(SGD)** 이라 한다.

![미니배치 SGD 의 흐름 — 셔플 → 배치로 자르기 → 기울기 · 갱신을 epoch 마다 반복](assets/fig/nn/d3-minibatch.svg)

| 용어 | 뜻 | 나선 데이터 | MNIST |
|---|---|---|---|
| batch size | 한 번 갱신에 쓰는 샘플 수 | 30 | 100 |
| iteration | 미니배치 하나로 기울기 → 갱신 1회 | | |
| epoch | 학습 데이터 전체를 한 바퀴 | 240 ÷ 30 = 8 iteration | 60,000 ÷ 100 = 600 iteration |

- 매 epoch **셔플**한다. 같은 순서로 반복하면 미니배치 구성이 고정되어 기울기 추정이 한쪽으로 치우칠 수 있다.
- 미니배치 기울기는 전체 기울기의 **잡음 섞인 추정**이라 손실 곡선이 들쭉날쭉하지만, 평균적으로는 내려간다.
- 학습은 train 데이터로, "처음 보는 데이터에도 통하는지(일반화)" 는 학습에 쓰지 않은 **test** 정확도로 확인한다.

#### 개념 2 · ReLU 망은 곡선 경계를 만든다
세 갈래로 꼬인 나선 데이터는 Day 2 의 로지스틱 회귀(직선 하나)로는 풀 수 없다. 은닉층 100개의 ReLU 망은 학습이 진행될수록 경계가 굽어지며 나선을 따라간다. ReLU 뉴런 하나는 꺾인 직선 하나를 만들고, 100개가 모이면 곡선처럼 보이는 경계가 된다 (Day 2 "은닉층은 공간을 바꾼다"). 중간에 손실이 들쭉날쭉한 것은 미니배치 잡음 + 큰 학습률(1.0) 때문이고, Day 4 의 옵티마이저가 이 문제를 다룬다.

#### 코드로 확인
```python
iters_per_epoch = len(X_tr) // batch_size
for epoch in range(epochs):
    idx = rng.permutation(len(X_tr))
    for i in range(iters_per_epoch):
        batch = idx[i * batch_size:(i + 1) * batch_size]
        grads = net.gradient(X_tr[batch], t_tr[batch])
        for k in net.params:
            net.params[k] -= lr * grads[k]           # 제자리 갱신
```
- **무엇을 하나**: epoch 마다 섞고, batch_size 씩 잘라 기울기를 구해 파라미터를 고친다.
- **어떤 개념**: 개념 1 — 미니배치 SGD.
- **출력**: 나선 데이터 batch 30 이면 epoch 당 **8** iteration, 300 epoch 동안 결정 경계가 나선을 따라 굽는다 (그림으로 확인)

#### 한 번에 정리
- 미니배치 SGD: epoch 마다 셔플 → batch 단위로 기울기 · 갱신.
- 미니배치 기울기는 잡음 섞인 추정 → 곡선은 들쭉날쭉하지만 내려간다.
- train 정확도는 학습 상태, test 정확도는 일반화.
- ⚠ `params[k] = params[k] - lr * g` 는 새 배열이라 계층과 연결이 끊긴다 → `-=`.

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

#### MNIST 2층 MLP 학습 순서
> 전처리 → 학습 전 손실 확인 → 미니배치 SGD → train · test 정확도로 평가
- **전처리** → (60000, 28, 28) uint8 → `reshape(-1, 784)` · 정규화 `/ 255.0` (0 ~ 1) · 레이블은 정수
- **학습 전 손실 ≈ log 10 = 2.30** → 작은 가중치 → 10개 클래스 확률이 거의 균등(0.1) → 초기화 · 손실 구현이 정상이라는 신호
- **설정** → lr 0.1 · batch 100 · 10 epoch = **6,000** iteration
- **예** → 기준 결과 train **0.9747**, test **0.9694** — 학습 곡선(손실 · 정확도)으로 확인 (CPU 마다 넷째 자리는 다를 수 있음)
- ⚠ train 이 test 보다 크게 높으면 과적합 (Day 4)

#### 양자화 맛보기
> float 가중치를 −127 ~ 127 정수(int8)로 바꿔 저장하고, 추론 때 scale 을 곱해 되돌려 쓴다
- **식** → scale = max|W| / 127, 저장 = round(W / scale)
- **효과** → float64 → int8 = 메모리 **1/8** (784-100-10 약 621 KB → 78 KB)
- **Vision 과정에서** → Quantization 의 핵심 아이디어

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

**상세 학습**

#### 이 교시에서 답하는 질문
오늘 만든 계층 · 역전파 · 미니배치 SGD 로 손글씨 숫자 0~9 를 분류한다. 28×28 흑백 6만 장으로 학습하고 처음 보는 1만 장으로 평가해 **test 95% 이상**을 목표로 한다.

#### 개념 1 · 전처리 — 모델이 받는 모양으로
- **펼치기**: (60000, 28, 28) 이미지를 한 장 = 784 칸 한 줄로 → `reshape(-1, 784)`.
- **정규화**: 0~255 를 255 로 나눠 0~1 로. 값이 크면 가중합이 커져 학습이 불안정하다.
- **레이블**: 정수 배열 그대로 — SoftmaxWithLoss 가 레이블을 받는다.

#### 개념 2 · 학습 전 손실로 구현을 점검한다
가중치를 0.01 배로 아주 작게 시작하면 출력 점수가 거의 0 이라 softmax 확률이 10개 클래스에 거의 균등(각 0.1)하다. 그러면 손실은 −log 0.1 = **log 10 ≈ 2.30**. 학습 전 손실이 이 근처면 초기화와 손실 구현이 정상이라는 신호다. 크게 다르면 전처리(/255 빠짐) · 가중치 크기 · 손실 식부터 의심한다.

#### 개념 3 · 학습과 결과 읽기
- 설정: 학습률 0.1, 배치 100, 10 epoch = 600 × 10 = **6,000** iteration (Colab CPU 30초 ~ 1분).
- 기준 결과(seed 0): train **0.9747**, test **0.9694**. train 이 test 보다 조금 높은 것은 정상 — 차이가 크게 벌어지면 과적합(Day 4).
- 같은 학습을 수치 미분으로 했다면 6,000 step × 수십 초 → 며칠. 역전파는 그래프를 한 번 거꾸로 지나며 기울기를 재사용하기 때문에 빠르다.
- **양자화 맛보기**: 가중치를 −127 ~ 127 정수(int8)로 바꾸면 메모리가 1/8(float64 대비)이 된다 — Vision 과정 이후 Quantization 의 아이디어.

#### 코드로 확인
```python
X_train = x_train.reshape(-1, 784) / 255.0
t_train = y_train.astype(int)
network = TwoLayerNet(784, 100, 10)
loss0 = network.loss(X_train[:100], t_train[:100])
```
- **무엇을 하나**: 이미지를 펼쳐 0~1 로 바꾸고, 학습 전 손실을 계산한다.
- **어떤 개념**: 개념 1 · 2.
- **출력**: X_train (60000, 784), 학습 전 손실 ≈ **2.30** (= log 10 ≈ 2.3026)

```python
for epoch in range(10):
    idx = rng.permutation(len(X_train))
    for i in range(600):
        batch = idx[i * 100:(i + 1) * 100]
        grads = network.gradient(X_train[batch], t_train[batch])
        for k in network.params: network.params[k] -= 0.1 * grads[k]
```
- **무엇을 하나**: 7교시 루프를 MNIST 에 적용한다 (6,000 iteration).
- **어떤 개념**: 개념 3.
- **출력**: 기준 결과 train **0.9747**, test **0.9694** (CPU 마다 넷째 자리는 다를 수 있음)

#### 한 번에 정리
- 전처리: 펼치기(784) · /255 · 정수 레이블.
- 학습 전 손실 ≈ 2.30 이면 정상.
- lr 0.1 · batch 100 · 10 epoch → test 약 0.97.
- 역전파 덕분에 수치 미분이면 며칠 걸릴 학습이 1분.

**연관 문제**
<<QA 8>>
