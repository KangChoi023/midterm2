## Day 4 — 학습 기법

### 1교시 · 복습 퀴즈 · SGD 의 약점

**키워드**: `SGD`, `비등방성`, `지그재그`, `f = x²/20 + y²`, `∇f = (x/10, 2y)`, `학습률 상한 η < 1`, `update(params, grads)`, `Momentum · AdaGrad · Adam`

**개요**: Day 3 까지 쓴 SGD 는 기울기 반대로 일정 비율만큼 움직인다. ① 단순하지만 손실 지형의 경사가 **방향마다 크게 다르면**(비등방성) 기울기가 최소점을 향하지 않아 지그재그로 간다. ② 가장 가파른 방향이 학습률 **상한**을 정하고, 완만한 방향은 그 학습률로는 **느리다** — 둘 다 만족시킬 수 없다. ③ 해결 아이디어 두 가지: 관성(Momentum), 파라미터마다 다른 학습률(AdaGrad · RMSProp), 둘 다(Adam) → 2교시.

**핵심 용어**
- **SGD(확률적 경사 하강법)**: W ← W − η ∂L/∂W — 기울기 반대 방향으로 일정 비율만큼
- **비등방성(anisotropy)**: 손실 지형의 경사가 방향마다 크게 다른 것 — 실제 신경망은 수만 차원에서 훨씬 심하다
- **옵티마이저(optimizer)**: 기울기로 파라미터를 갱신하는 규칙. 오늘은 모두 `update(params, grads)` 로 제자리 갱신
- **지그재그**: 가파른 방향으로 부호를 바꾸며 왔다 갔다 하는 경로 — 완만한 방향으로는 조금씩만 전진

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `SGD` | `SGD(lr=0.01).update(params, grads)` | `params[k] -= lr * grads[k]` |
| `run_optimizer()` | `run_optimizer(opt, steps=30, start=(-7, 2))` | 옵티마이저로 f 위의 경로를 기록 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 방향마다 경사가 다른 지형 | (−7, 2) 에서 SGD 30 step 경로를 등고선 위에 그림, 학습률 0.1 · 0.95 · 1.0 · 1.05 의 끝점 비교 |

**핵심 내용**

#### 비등방성 — SGD 가 지그재그로 가는 이유
> 손실 지형의 경사가 **방향마다 크게 다른 것** — f = x²/20 + y² 는 y 방향이 x 방향보다 **20배** 가파르다 (∇f = (x/10, 2y))
- **핵심 성질** → 기울기가 최소점이 아니라 **가파른 쪽**을 가리킨다 — (−7, 2) 에서 ∇f = (−0.7, 4)
- **SGD(확률적 경사 하강법)** → W ← W − ηg 를 모든 방향에 같은 η 로 → 가파른 y 는 튀고(**지그재그**), 완만한 x 는 기어간다
- **`run_optimizer()`** → (−7, 2) 에서 옵티마이저로 30 step 경로를 기록해 등고선 위에 그린다
- **신경망에서** → 파라미터 수만 개 공간에서 비등방성이 훨씬 심하다

#### 학습률의 상한과 속도
> 가장 가파른 방향이 학습률 **상한**(여기선 η < 1)을 정하고, 그 학습률로는 완만한 방향이 느리다 — 하나의 η 로 둘 다 만족 못 함
- **식** → y ← (1 − 2η)y, x ← (1 − η/10)x — |1 − 2η| < 1 이어야 y 가 줄어든다
- **예** → 30 step 후 lr 0.1 → x **−5.18** (느림) · 0.95 → (−0.35, 0.08) · 1.0 → y 가 ±2 를 영원히 오감 · 1.05 → y **34.9** 발산
- **Day 1 에서** → x² + 3y² 의 상한 1/3 과 같은 원리 (상한 = 2 / 가장 큰 곡률)
- ⚠ 학습률만 바꿔서는 해결 안 됨 → 방향마다 다르게 움직이는 **옵티마이저** 필요

#### 옵티마이저 — 해결 아이디어 두 가지
> 기울기로 파라미터를 갱신하는 규칙 — 오늘은 모두 `update(params, grads)` 로 params 배열을 **제자리** 갱신
- **관성** → 부호가 바뀌는 성분은 상쇄, 일관된 방향은 가속 → Momentum
- **파라미터별 학습률** → 많이 움직인 방향은 작게 → AdaGrad · RMSProp
- **둘 다** → Adam (2교시)
- **`SGD`** → `params[k] -= self.lr * grads[k]` — 비교 기준

**기본 사용형태**
```python
class SGD:
    def __init__(self, lr=0.01): self.lr = lr
    def update(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]      # 제자리 갱신

def grad_f(p): return np.array([p[0] / 10, 2 * p[1]])   # f = x²/20 + y²
```

**연결**
- [NN D1·8교시 미니 프로젝트: 경사하강법으로 최소점 찾기](#/nn/1/8) — 같은 현상 (x² + 3y², 상한 1/3)
- [NN D3·7교시 미니배치 SGD](#/nn/3/7) — 제자리 갱신 · 들쭉날쭉한 손실

**상세 학습**

#### 이 교시에서 답하는 질문
Day 3 까지는 SGD(기울기 반대로 일정 비율만큼 이동) 하나로 학습했다. 2층 MLP 에서는 잘 됐지만, 망이 깊어지고 손실 지형이 복잡해지면 SGD 는 느리고 불안정해진다. 이 교시는 SGD 가 **어떤 지형에서, 왜** 약한지 작은 2차원 함수로 직접 보고, 2교시에서 만들 옵티마이저들이 무엇을 고치려는지 정한다.

#### 개념 1 · 비등방성 — 기울기는 최소점을 가리키지 않는다
산 비탈을 생각해 보자. 동서 방향은 완만하고 남북 방향은 가파른 골짜기라면, 서 있는 자리에서 "가장 가파른 내리막"은 골짜기 바닥(목적지)이 아니라 **가까운 벽 쪽**을 가리킨다. 이렇게 방향마다 경사가 크게 다른 것을 **비등방성(anisotropy)** 이라 한다.

> f(x, y) = x²/20 + y², ∇f = (x/10, 2y) — y 방향이 x 방향보다 20배 가파르다

| 위치 | 기울기 ∇f | 최소점 (0, 0) 방향 |
|---|---|---|
| (−7, 2) | (−0.7, 4) — 거의 y 쪽 | (7, −2) — 거의 x 쪽 |

그래서 SGD 는 y 방향으로 크게 튀며 부호를 바꾸고(**지그재그**), 정작 가야 할 x 방향으로는 조금씩만 전진한다.

![f = x²/20 + y² 위에서 다섯 옵티마이저의 30 step 경로 — SGD 는 지그재그, Momentum 은 관성으로 x 를 가로지르고, AdaGrad · Adam 은 방향마다 걸음을 조절한다](assets/fig/nn/d4-optimizers.svg)

#### 개념 2 · 학습률은 두 방향을 동시에 만족시킬 수 없다
SGD 한 step 을 방향별로 쓰면 각 좌표가 일정한 비율로 곱해진다.

| 방향 | 한 step 후 | 줄어들 조건 |
|---|---|---|
| y | y ← (1 − 2η) y | \|1 − 2η\| < 1 → **η < 1** |
| x | x ← (1 − η/10) x | η 가 작으면 1 − η/10 ≈ 1 → 아주 느림 |

가파른 y 가 학습률의 **상한**을 정하고, 그 상한 안의 학습률로는 완만한 x 가 느리다. Day 1 미니 프로젝트(x² + 3y², 상한 1/3)와 같은 원리다 — 상한은 "2 / 가장 큰 곡률".

| lr | 30 step 후 (x, y) | 무슨 일 |
|---|---|---|
| 0.1 | (−5.18, 0.002) | y 는 금방 0, x 는 −7 → −5.18 밖에 못 감 |
| 0.95 | (−0.35, 0.08) | y 가 부호를 바꾸며 줄고, x 도 꽤 감 |
| 1.0 | (−0.30, 2) | y ← −y — ±2 를 영원히 오감 |
| 1.05 | (−0.25, 34.9) | y 가 매번 1.1 배 → 발산 |

#### 코드로 확인
```python
def grad_f(p):                         # p = [x, y]
    return np.array([p[0] / 10.0, 2.0 * p[1]])

def run_optimizer(opt, steps=30, start=(-7.0, 2.0)):
    params = {"p": np.array(start, dtype=float)}
    path = [params["p"].copy()]
    for _ in range(steps):
        opt.update(params, {"p": grad_f(params["p"])})
        path.append(params["p"].copy())
    return np.array(path)                  # (steps + 1, 2)
```
- **무엇을 하나**: 옵티마이저 객체 하나를 받아 (−7, 2) 에서 30 step 움직이며 위치를 기록한다. 옵티마이저는 `update(params, grads)` 만 있으면 무엇이든 끼울 수 있다.
- **어떤 개념**: 개념 1 — 기울기 (x/10, 2y) 로 움직인 경로.
- **출력**: `(31, 2)` 배열 — 그림의 경로 한 줄

```python
class SGD:
    def __init__(self, lr=0.01): self.lr = lr
    def update(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]      # 제자리 갱신

for lr in [0.1, 0.95, 1.0, 1.05]:
    print(lr, run_optimizer(SGD(lr=lr))[-1])
```
- **무엇을 하나**: 학습률 네 개로 SGD 를 돌려 30 step 후의 끝점을 비교한다.
- **어떤 개념**: 개념 2 — 상한 η < 1 과 느린 x 방향.
- **출력**: `0.1 [-5.178 0.002]` · `0.95 [-0.35 0.085]` · `1.0 [-0.297 2.]` · `1.05 [-0.251 34.899]`

#### 한 번에 정리
- 비등방성: 방향마다 경사가 다르면 기울기가 최소점이 아니라 가파른 쪽을 가리킨다 → 지그재그.
- 가장 가파른 방향이 학습률 상한을 정하고, 그 학습률로는 완만한 방향이 느리다.
- 해결: 관성(Momentum) · 파라미터별 학습률(AdaGrad · RMSProp) · 둘 다(Adam).
- ⚠ 학습률만 조정해서는 이 문제를 풀 수 없다.

**연관 문제**
<<QA 1>>

### 2교시 · 옵티마이저 구현

**키워드**: `Momentum`, `속도 v`, `AdaGrad`, `누적 h`, `RMSProp`, `지수 이동 평균`, `Adam`, `편향 보정`, `1e-7`, `상태는 첫 update 에서 zeros_like`, `MultiLayerNet`

**개요**: 네 옵티마이저를 SGD 와 같은 `update(params, grads)` 형태로 만든다. ① **Momentum** 은 속도에 기울기를 쌓아 관성으로, **AdaGrad** 는 기울기 제곱을 누적해 많이 움직인 파라미터의 학습률을 줄이고, **RMSProp** 은 그 누적을 최근 위주로, **Adam** 은 둘을 합치고 편향을 보정한다. ② 상태(v, h, m)는 파라미터마다 필요해서 첫 update 때 `zeros_like` 로 만든다. ③ MNIST 5층에서 SGD 가 가장 느리고 나머지는 약 0.97 — 실무 기본값은 **Adam(lr 0.001)**.

**핵심 용어**
- **Momentum**: 속도 v 에 기울기를 누적해 관성으로 움직임 — 부호가 바뀌는 성분은 상쇄, 일관된 성분은 가속
- **AdaGrad**: 기울기 제곱을 누적(h)해 많이 움직인 파라미터의 학습률을 줄임 — h 가 계속 커져 결국 멈춤
- **RMSProp**: h 를 **지수 이동 평균**으로 — 오래된 기울기는 잊어서 멈추지 않음
- **Adam**: Momentum(1차 모멘트 m) + RMSProp(2차 모멘트 v) + 편향 보정
- **편향 보정(bias correction)**: m, v 가 0 에서 시작해 처음 몇 step 동안 0 쪽으로 치우친 것을 1 − βᵗ 로 나눠 바로잡는 것
- **지수 이동 평균(EMA)**: 새 값 = ρ × 옛 값 + (1 − ρ) × 지금 값 — 최근 값일수록 비중이 큼

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Momentum` | `Momentum(lr=0.01, momentum=0.9)` | v ← αv − ηg, W ← W + v |
| `AdaGrad` | `AdaGrad(lr=0.01)` | h ← h + g², W ← W − ηg/(√h + 1e-7) |
| `RMSProp` | `RMSProp(lr=0.01, decay=0.99)` | h ← ρh + (1 − ρ)g² |
| `Adam` | `Adam(lr=0.001, beta1=0.9, beta2=0.999)` | m · v 이동 평균 + 편향 보정 |
| `MultiLayerNet` | `MultiLayerNet(784, [100]*4, 10, weight_init_std="he", use_batchnorm=…, use_dropout=…, weight_decay_lambda=…)` | 오늘의 실험용 다층망 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | 1교시 지형에서 SGD(0.95) · Momentum(0.1) · AdaGrad(1.5) · RMSProp(0.1) · Adam(0.3) 경로를 한 그림에 비교 |
| 실제 신경망에서 — MNIST 5층 MLP | `MultiLayerNet` 784-100×4-10 을 다섯 옵티마이저로 2000 iteration 학습 → 손실 곡선 · test 정확도 비교 |

**핵심 내용**

#### Momentum — 관성
> v ← αv − ηg, W ← W + v (α = **0.9**) — 공이 굴러가듯 속도 v 에 기울기를 쌓는다
- **핵심 성질** → 부호가 바뀌는 y 성분은 서로 상쇄, 일관된 x 성분은 속도가 쌓여 가속
- **지형에서** → 처음엔 y 로 튀지만 x 방향으로 빠르게 가로지른다 (최소점을 지나쳤다 돌아옴)
- **상태 v** → 파라미터마다 하나 — 첫 update 에서 `np.zeros_like`
- ⚠ 옵티마이저 객체를 여러 모델에 재사용 → 상태(v, h, m)가 섞임 → 모델마다 새로

#### AdaGrad · RMSProp — 파라미터별 학습률
> h ← h + g², W ← W − ηg / (√h + 1e-7) — 많이 움직인(기울기가 컸던) 파라미터일수록 걸음이 작아진다
- **AdaGrad** → h 가 계속 커져 학습률이 결국 0 → **멈춤**
- **RMSProp** → h ← ρh + (1 − ρ)g² (ρ = **0.99**) — **지수 이동 평균**으로 오래된 기울기를 잊어 멈추지 않는다
- **지형에서** → y 방향은 처음 g 가 커서 h 가 빨리 커짐 → 지그재그가 사라진다
- **1e-7** → h = 0 일 때 0 으로 나누기 방지

#### Adam — 관성 + 파라미터별 학습률
> m ← β₁m + (1 − β₁)g (방향), v ← β₂v + (1 − β₂)g² (크기), W ← W − η m̂ / (√v̂ + 1e-7)
- **편향 보정** → m, v 가 0 에서 시작해 0 쪽으로 치우침 → m̂ = m / (1 − β₁ᵗ) — t = 1 이면 m = 0.1g 를 0.1 로 나눠 m̂ = g
- **기본값** → β₁ **0.9**, β₂ **0.999**, lr **0.001** — 실무 기본 옵티마이저
- **예** → 지형에서 lr 0.3 의 첫 step: (−7, 2) → (**−6.7, 1.7**) — 두 방향 모두 정확히 lr 만큼 (g 의 크기와 무관)
- ⚠ `self.t += 1` 을 파라미터 for **안**에 넣으면 t 가 파라미터 수만큼 커져 보정이 틀어짐

#### MNIST 5층에서 비교
> `MultiLayerNet` 784-100×4-10 (He) · 2000 iteration — SGD 가 가장 느리고 나머지는 약 0.97
- **test** → SGD(0.01) 0.9378 · Momentum(0.01) 0.9644 · AdaGrad(0.01) 0.9726 · RMSProp(0.001) **0.9745** · Adam(0.001) 0.9740
- **`MultiLayerNet`** → Day 3 TwoLayerNet 을 은닉층 여러 개로 + 오늘의 옵션(초기화 · BN · Dropout · weight decay)
- **실무** → Adam 으로 시작, 마지막 성능을 짜낼 때 SGD + Momentum 재시도
- ⚠ `params[k] = params[k] - …` → 새 배열로 계층과 연결이 끊김 → `-=` · `+=`

**기본 사용형태**
```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr, self.beta1, self.beta2, self.m, self.v, self.t = lr, beta1, beta2, None, None, 0
    def update(self, params, grads):
        if self.m is None:
            self.m = {k: np.zeros_like(p) for k, p in params.items()}
            self.v = {k: np.zeros_like(p) for k, p in params.items()}
        self.t += 1                                          # for 밖에서 한 번
        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)
```

**연결**
- [NN D3·6교시 계층 조립과 Gradient check](#/nn/3/6) — params 와 계층이 같은 배열 객체
- [NN D4·8교시 미니 프로젝트: 미니 프레임워크 mynn.py](#/nn/4/8) — Adam 으로 5층 MLP 학습

**상세 학습**

#### 이 교시에서 답하는 질문
1교시에서 SGD 의 문제를 두 가지로 정리했다 — 지그재그(방향이 흔들림)와 방향마다 다른 속도. 이 교시는 이를 고치는 네 옵티마이저를 SGD 와 똑같은 `update(params, grads)` 모양으로 만들고, 작은 지형과 실제 MNIST 망에서 비교한다.

#### 개념 1 · Momentum — 관성으로 흔들림을 상쇄
언덕을 굴러가는 공을 떠올리자. 공은 지금 경사만 보지 않고 **이전에 달리던 속도**를 유지한다. Momentum 은 속도 v 에 기울기를 계속 쌓는다.

> v ← αv − ηg, W ← W + v (α = 0.9)

| 성분 | 매 step 기울기 | 속도 v 에서 |
|---|---|---|
| y (지그재그) | +, −, +, − (부호가 바뀜) | 서로 **상쇄** → 흔들림 감소 |
| x (일관된 방향) | 매번 같은 부호 | 계속 **쌓임** → 가속 |

#### 개념 2 · AdaGrad · RMSProp — 파라미터마다 다른 학습률
많이 움직인 파라미터는 "이미 충분히 조정됐다" 고 보고 걸음을 줄인다. 기울기 제곱을 누적한 h 로 나눈다.

> h ← h + g², W ← W − η g / (√h + 1e-7)

| 기호 | 뜻 |
|---|---|
| g | 이번 기울기 ∂L/∂W |
| h | 지금까지 g² 의 누적 — 파라미터마다 따로 |
| 1e-7 | h = 0 일 때 0 으로 나누기 방지 |

AdaGrad 는 h 가 계속 커지기만 해서 결국 학습이 멈춘다. **RMSProp** 은 h 를 지수 이동 평균(h ← 0.99h + 0.01g²)으로 바꿔 오래된 기울기를 잊는다 — 최근 기울기만 반영하므로 멈추지 않는다.

#### 개념 3 · Adam — 둘을 합치고 편향을 보정
Adam 은 Momentum 의 m(방향)과 RMSProp 의 v(크기)를 함께 쓴다.

> m ← β₁m + (1 − β₁)g, v ← β₂v + (1 − β₂)g², W ← W − η · m̂ / (√v̂ + 1e-7)

m, v 는 0 에서 시작하므로 처음 몇 step 은 0 쪽으로 치우친다. t = 1 이면 m = 0.1g 뿐이다. 그래서 **편향 보정** m̂ = m / (1 − β₁ᵗ) 로 나눈다 → 1 − 0.9 = 0.1 로 나눠 m̂ = g 로 되돌린다. v 도 같다. 결과적으로 첫 step 은 m̂ / √v̂ = g / |g| = ±1 → **모든 방향이 정확히 η 만큼** 움직인다 (기울기 크기와 무관).

기본값 β₁ 0.9 · β₂ 0.999 · lr 0.001 — 실무에서 가장 먼저 쓰는 옵티마이저다.

#### 개념 4 · 실제 신경망에서 — MNIST 5층
작은 지형 하나로는 "어느 것이 최고" 라고 할 수 없다. 784-100×4-10 망(`MultiLayerNet`, He 초기화)을 2000 iteration 학습해 비교한다 (강의 실행 값).

| SGD (0.01) | Momentum (0.01) | AdaGrad (0.01) | RMSProp (0.001) | Adam (0.001) |
|---|---|---|---|---|
| 0.9378 | 0.9644 | 0.9726 | **0.9745** | 0.9740 |

SGD 만 눈에 띄게 느리고 나머지는 약 0.97 이다.

#### 코드로 확인
```python
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum, self.v = lr, momentum, None
    def update(self, params, grads):
        if self.v is None:                                  # 첫 update 때 shape 를 알게 됨
            self.v = {k: np.zeros_like(p) for k, p in params.items()}
        for k in params:
            self.v[k] = self.momentum * self.v[k] - self.lr * grads[k]
            params[k] += self.v[k]
```
- **무엇을 하나**: 파라미터마다 속도 v 를 두고 매 step 0.9 배로 줄인 뒤 −ηg 를 더해 이동한다.
- **어떤 개념**: 개념 1 — 관성. 상태 v 를 `__init__` 이 아니라 첫 update 에서 만드는 이유는 그때 비로소 파라미터 shape 를 알기 때문이다.
- **출력**: 지형에서 lr 0.1, 첫 step (−7, 2) → (−6.93, 1.6) · 30 step 후 (0.90, 0.09) — 최소점을 지나쳤다 돌아오는 중

```python
self.t += 1                                                 # for 밖에서 한 번
for k in params:
    self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
    self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k] ** 2
    m_hat = self.m[k] / (1 - self.beta1 ** self.t)        # 편향 보정
    v_hat = self.v[k] / (1 - self.beta2 ** self.t)
    params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)
```
- **무엇을 하나**: Adam 의 update 본체 — m · v 를 이동 평균으로 갱신하고 편향 보정 후 이동한다.
- **어떤 개념**: 개념 3 — t 는 step 수라서 파라미터 for 밖에서 한 번만 늘린다.
- **출력**: 지형에서 lr 0.3, 첫 step (−7, 2) → (**−6.7, 1.7**) — 기울기는 (−0.7, 4) 로 6배 차이지만 두 방향 모두 0.3 씩

```python
optimizers = {"SGD": SGD(lr=0.95), "Momentum": Momentum(lr=0.1), "AdaGrad": AdaGrad(lr=1.5),
              "RMSProp": RMSProp(lr=0.1), "Adam": Adam(lr=0.3)}
paths = {name: run_optimizer(opt) for name, opt in optimizers.items()}
```
- **무엇을 하나**: 다섯 옵티마이저로 1교시 지형을 30 step 씩 움직여 경로를 모은다 (1교시 그림).
- **어떤 개념**: 개념 1 ~ 3 의 차이를 한 그림에서.
- **출력**: 30 step 후 원점까지 거리 — SGD 0.36 · Momentum 0.90 · AdaGrad **0.12** · RMSProp 0.70 · Adam 0.35

#### 한 번에 정리
- Momentum: 속도에 기울기를 쌓아 흔들림은 상쇄, 일관된 방향은 가속.
- AdaGrad: g² 누적으로 파라미터별 학습률 (결국 멈춤) → RMSProp: 이동 평균으로 잊어서 계속.
- Adam = Momentum + RMSProp + 편향 보정. 실무 기본값 lr 0.001.
- MNIST 5층: SGD 0.9378, 나머지 약 0.97.
- ⚠ 옵티마이저 상태(v, h, m)는 모델마다 새로 · Adam 의 t 는 for 밖에서.

**연관 문제**
<<QA 2>>

### 3교시 · 가중치 초기화

**키워드**: `대칭성`, `0 초기화 금지`, `활성화 분포`, `포화`, `신호 소멸`, `분산 ≈ n × Var(W) × Var(x)`, `Xavier √(1/n)`, `He √(2/n)`, `n = 입력 노드 수`

**개요**: 학습은 초기 가중치에서 출발한다. Day 3 의 "0.01 × 정규분포" 는 2층에선 괜찮았지만 깊어지면 문제가 된다. ① 모두 같은 값이면 뉴런들이 같은 값 · 같은 기울기로 함께 움직여 100개가 사실상 1개(**대칭**). ② 너무 크면 sigmoid 가 0 · 1 에 몰려 기울기 소실, 너무 작으면 층을 지날수록 신호가 사라진다. ③ 해결: 입력 노드 수 n 에 맞춘 표준편차 — sigmoid 는 **Xavier √(1/n)**, ReLU 는 **He √(2/n)**.

**핵심 용어**
- **대칭성(symmetry)**: 같은 가중치로 시작한 뉴런들이 같은 기울기를 받아 끝까지 같아지는 현상 — 무작위 초기화로 깬다
- **활성화 분포**: 층 출력 값들의 히스토그램 — 좋은 초기화는 층을 지나도 적당히 퍼진 분포를 유지
- **포화(saturation)**: sigmoid 출력이 0 · 1 근처에 몰려 기울기가 거의 0 인 상태
- **Xavier 초기화**: std = √(1/n_in) — sigmoid · tanh 용
- **He 초기화**: std = √(2/n_in) — ReLU 용 (음수 절반이 0 이 되어 분산이 반으로 줄므로 2배)

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `xavier_std()` | `np.sqrt(1.0 / n_in)` | Xavier 표준편차 |
| `he_std()` | `np.sqrt(2.0 / n_in)` | He 표준편차 |
| `activation_stats()` | `activation_stats(weight_init_std, act, n_layers=5)` | 층마다 활성화 분포 기록 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 모두 같은 값(0)으로 시작하면? | 모든 가중치를 같은 값으로 두고 300 iteration 학습 → 은닉 뉴런 100개의 가중치가 끝까지 똑같은 것 확인 |
| 활성화 값의 분포 | 입력 1000개(100차원)를 100 노드 층 5개에 통과 — sigmoid · ReLU × std 1 · 0.01 · Xavier · He 조합의 층별 히스토그램과 표준편차 |

**핵심 내용**

#### 대칭성 — 같은 값 초기화 금지
> 같은 가중치로 시작한 뉴런은 같은 출력 → 같은 기울기 → 같이 갱신 → **영원히 같다** (100개가 사실상 1개)
- **모두 0.01** → 300 iteration 학습 후에도 은닉 뉴런의 가중치 열이 모두 같음
- **모두 0** → 더 나쁨: ReLU 출력이 0 → 기울기 자체가 **0**
- **편향** → 0 이어도 괜찮다 (가중치가 이미 서로 다르므로)
- ⚠ 가중치는 반드시 **무작위**로 — 대칭을 깨야 뉴런이 서로 다른 특징을 배운다

#### 활성화 분포 — 너무 크거나 너무 작으면
> 층 출력 분산 ≈ n × Var(W) × Var(x) — Var(W) 가 1/n 보다 크면 커지고, 작으면 층마다 줄어든다
- **너무 크면(sigmoid + std 1)** → 약 **82%** 가 0.1 미만 · 0.9 초과 = **포화** → 기울기 소실
- **너무 작으면(ReLU + std 0.01)** → 층별 표준편차 0.058 → 0.004 → 0.0003 → 신호 소멸
- **`activation_stats()`** → 입력 1000개(100차원)를 100 노드 층 5개에 통과시켜 층마다 출력을 기록 (히스토그램 · 표준편차)
- ⚠ sigmoid + std 0.01 → 모두 0.5 근처 → 뉴런이 같은 값이라 표현력 제한

#### Xavier · He 초기화
> 입력 노드 수 n 에 맞춘 표준편차 — **Xavier** √(1/n) (sigmoid · tanh), **He** √(2/n) (ReLU)
- **`xavier_std()` · `he_std()`** → n = 100 이면 0.1 · 0.141
- **He 가 2배인 이유** → ReLU 가 음수 절반을 0 으로 만들어 분산이 ½ → 2배로 보상
- **예** → ReLU 층별 표준편차: Xavier 0.58 → 0.18 (줄어듦) · He 0.82 → **1.00** (유지) · MNIST 4층 test: std 0.01 **0.11** · Xavier 0.92 · He **0.94**
- ⚠ n 은 **입력 노드 수**(이전 층 뉴런 수) · sigmoid 망엔 Xavier, ReLU 망엔 He 로 짝을 맞추기

**기본 사용형태**
```python
def xavier_std(n_in): return np.sqrt(1.0 / n_in)     # sigmoid · tanh
def he_std(n_in):     return np.sqrt(2.0 / n_in)     # ReLU

W = he_std(784) * rng.standard_normal((784, 100))   # n_in = 이전 층 뉴런 수
b = np.zeros(100)                                    # 편향은 0 이어도 됨
```

**연결**
- [NN D2·4교시 활성화 함수](#/nn/2/4) — sigmoid 기울기 최대 0.25 · 기울기 소실
- [NN D1·7교시 확률 기초](#/nn/1/7) — 정규분포 난수 · 분산
- [NN D3·8교시 미니 프로젝트: MNIST 2층 MLP](#/nn/3/8) — std 0.01 로 시작했던 2층

**상세 학습**

#### 이 교시에서 답하는 질문
학습은 초기 가중치에서 출발한다. Day 3 의 "0.01 × 정규분포" 는 2층에서는 잘 됐다. 그런데 같은 초기화로 4층을 학습하면 test 0.11 — 찍기 수준이다. 이 교시는 초기값이 **왜** 학습의 성패를 가르는지, 그리고 얼마로 시작해야 하는지 답한다.

#### 개념 1 · 대칭성 — 모두 같은 값이면 뉴런이 하나
은닉 뉴런 100개를 모두 같은 가중치로 시작하면 어떻게 될까?
1. 같은 입력 × 같은 가중치 → **같은 출력**
2. 같은 출력 → 역전파에서 **같은 기울기**
3. 같은 기울기 → 같이 갱신 → 다음 step 에도 같다 → **영원히 같다**

100개가 사실상 1개인 셈이다. 모두 0 이면 더 나빠서, ReLU 출력이 0 이 되어 기울기 자체가 0 이다. 그래서 가중치는 반드시 **무작위**로 시작해 대칭을 깬다 (편향은 0 이어도 된다 — 가중치가 이미 다르므로).

#### 개념 2 · 활성화 분포 — 층을 지나며 커지거나 사라진다
층 출력 a = Σ xᵢWᵢ 는 n 개 항의 합이다. 서로 독립이면 분산이 더해져

> Var(a) ≈ n × Var(W) × Var(x)

| Var(W) | 층을 지날 때 분산 | 결과 |
|---|---|---|
| 1/n 보다 큼 | 계속 커짐 | sigmoid 가 0 · 1 에 몰림 = **포화** → 기울기 소실 |
| 1/n 보다 작음 | 계속 줄어듦 | 신호가 0 으로 **소멸** |
| 약 1/n | 유지 | 층마다 적당히 퍼진 분포 |

![활성화 분포 — 입력 1000개를 100 노드 층 5개에 통과시킨 층별 히스토그램. sigmoid + std 1 은 0 · 1 에 몰리고, ReLU + std 0.01 은 사라지고, ReLU + He 는 유지된다](assets/fig/nn/d4-activations.svg)

#### 개념 3 · Xavier · He 초기화
분산을 유지하려면 Var(W) = 1/n, 즉 표준편차 √(1/n) — 이것이 **Xavier** 초기화다 (n = 입력 노드 수). ReLU 는 음수 절반을 0 으로 만들어 분산이 ½ 로 줄므로, 2배로 보상한 √(2/n) 이 **He** 초기화다.

| 초기화 | 표준편차 | 짝 | n = 100 일 때 |
|---|---|---|---|
| Xavier | √(1/n) | sigmoid · tanh | 0.1 |
| He | √(2/n) | ReLU | 0.141 |

ReLU 층별 표준편차 — std 0.01: 0.058 → 0.004 → 0.0003 (소멸) · Xavier: 0.58 → 0.18 (줄어듦) · He: 0.82 → 1.00 (**유지**).

#### 코드로 확인
```python
def xavier_std(n_in): return np.sqrt(1.0 / n_in)
def he_std(n_in):     return np.sqrt(2.0 / n_in)

def activation_stats(weight_init_std, act, n_layers=5, seed=0):
    rng_ = np.random.default_rng(seed)
    x = rng_.standard_normal((1000, 100))
    outs = []
    for _ in range(n_layers):
        W = init_scale(weight_init_std, 100) * rng_.standard_normal((100, 100))
        a = x @ W
        x = 1 / (1 + np.exp(-a)) if act == "sigmoid" else np.maximum(0, a)
        outs.append(x)
    return outs
```
- **무엇을 하나**: 정규분포 입력 1000개를 100 노드 층 5개에 통과시키며 층마다 활성화를 모은다. `init_scale` 은 숫자 · "xavier" · "he" 를 표준편차로 바꾼다.
- **어떤 개념**: 개념 2 · 3 — 초기화마다 층별 분포가 어떻게 변하나.
- **출력**: sigmoid + std 1 의 1층은 **82%** 가 0.1 미만 · 0.9 초과 · ReLU + He 층별 std 0.82 → 1.00

```python
for w in [0.01, "xavier", "he"]:
    net = MultiLayerNet(784, [100, 100, 100, 100], 10, weight_init_std=w)
    h = fit(net, SGD(lr=0.01), X_train, t_train, iters=2000, batch_size=128, eval_every=None)
    print(w, net.accuracy(X_test, t_test))
```
- **무엇을 하나**: 초기화만 바꾼 같은 4층 MLP 를 SGD 로 2000 iteration 학습한다.
- **어떤 개념**: 개념 3 — 초기화가 학습의 성패를 가른다.
- **출력**: std 0.01 **0.11** (학습 안 됨) · Xavier 0.92 · He **0.94** (강의 실행 값)

#### 한 번에 정리
- 같은 값 초기화 = 대칭 → 뉴런 100개가 1개. 반드시 무작위.
- 출력 분산 ≈ n × Var(W) × Var(x) → 크면 포화, 작으면 소멸.
- sigmoid 엔 Xavier √(1/n), ReLU 엔 He √(2/n) — n 은 입력 노드 수.
- ⚠ 2층에서 되던 std 0.01 이 4층에서는 test 0.11.

**연관 문제**
<<QA 3>>

### 4교시 · Batch Normalization

**키워드**: `BatchNorm`, `열마다 평균 0 · 분산 1`, `γ · β`, `ε = 1e-7`, `running 평균 · 분산`, `train_flg`, `momentum 0.9`, `Affine → BN → ReLU`, `dx 의 열 합 = 0`

**개요**: 초기화는 출발점에서만 분포를 맞춘다 — 학습이 진행되면 가중치가 바뀌어 분포가 다시 틀어진다. ① **BN** 은 매 미니배치마다 층 출력을 **열(특성)마다** 평균 0 · 분산 1 로 맞춘 뒤, 학습되는 γ · β 로 늘이고 옮긴다. ② 추론 때는 미니배치가 없을 수도(사진 한 장) 있으니 학습 중 모아 둔 **running 평균 · 분산**을 쓴다 → `train_flg` 가 필요한 이유. ③ 효과: 초기화에 둔감해지고 큰 학습률에서도 안정적 → 학습이 빠르다.

**핵심 용어**
- **배치 정규화(Batch Normalization, BN)**: 미니배치의 열마다 x̂ = (x − μ) / √(σ² + ε) 로 정규화하고 y = γx̂ + β
- **γ · β(scale · shift)**: 정규화 후 다시 늘이고 옮기는 학습 파라미터 (처음 **1**, **0**)
- **running 평균 · 분산**: 학습 중 배치 통계의 이동 평균 — running = 0.9 · running + 0.1 · 배치값, 추론 때 사용
- **train_flg**: True = 배치 통계로 정규화 + running 갱신, False = running 통계로 정규화 (갱신 없음)
- **ε(epsilon)**: 분산이 0 일 때 0 으로 나누지 않으려는 작은 값 **1e-7**

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `BatchNorm` | `BatchNorm(gamma, beta, momentum=0.9, eps=1e-7)` · `forward(x, train_flg)` · `backward(dout)` | BN 계층 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | 평균 3 · 표준편차 2 근처 미니배치 (5, 4) → 출력 열 평균 = β, 열 표준편차 = γ 확인 · N(5, 2²) 배치 50번으로 running 이 (5, 4) 에 가까워짐 · 한 장 학습 모드 → 출력 0 · BN 유무 × 초기화 std 실험 |

**핵심 내용**

#### 배치 정규화(Batch Normalization)
> 미니배치의 **열(특성)마다** x̂ = (x − μ) / √(σ² + ε) 로 평균 0 · 분산 1 을 맞춘 뒤 y = γx̂ + β
- **γ · β** → 학습되는 늘이기 · 옮기기 (처음 1, 0) — 출력 열 평균 = **β**, 표준편차 = **γ**
- **ε(epsilon)** → 분산 0 일 때 0 으로 나누기 방지 **1e-7**
- **`BatchNorm`** → `BatchNorm(gamma, beta, momentum=0.9, eps=1e-7)` · 넣는 자리는 보통 **Affine → BN → ReLU**
- ⚠ `x.mean()` (축 없음) → 전체 평균 하나 → 특성마다 **`axis=0`**
- ⚠ γ, β 를 옵티마이저 params 에 넣어야 학습된다

#### 학습 · 추론 모드 — running 평균 · 분산
> 학습: 배치 통계로 정규화 + running 갱신 · 추론: 모아 둔 running 통계로 (`train_flg` 로 구분)
- **running 갱신** → running = 0.9 · running + 0.1 · 배치값 — 학습 모드에서만
- **예** → N(5, 2²) 배치 50번 → running_mean **5.03**, running_var **3.83** (참값 5, 4 에 근접)
- **왜 필요** → 학습 모드에 한 장을 넣으면 평균이 자기 자신 → 출력 항상 **0**
- ⚠ predict 는 **`train_flg=False`** — True 면 배치 구성에 따라 결과가 바뀐다

#### BN 의 backward · 효과
> dβ = Σ dout, dγ = Σ x̂ · dout (행 방향 합) — dx 의 열 합은 항상 **0**
- **dx 열 합 = 0** → BN 바로 앞 Affine 의 편향 기울기는 0 (1e-16 수준) → gradient check 오차가 커 보여도 버그 아님
- **효과** → 초기화에 둔감 · 큰 학습률에서도 안정 → 학습이 빠르다
- **예** → 784-100×5-10 · 1000장 · std 0.01: BN 없음 **0.117** (학습 안 됨) vs BN 있음 train **1.000**

**기본 사용형태**
```python
def forward(self, x, train_flg=True):
    if train_flg:
        mu = x.mean(axis=0); xc = x - mu
        var = (xc ** 2).mean(axis=0); std = np.sqrt(var + self.eps)
        xn = xc / std
        self.xc, self.xn, self.std = xc, xn, std
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
        self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * var
    else:
        xn = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
    return self.gamma * xn + self.beta
```

**연결**
- [NN D3·5교시 Affine · Softmax-with-Loss 계층](#/nn/3/5) — 편향 기울기 = 행 방향 합
- [NN D3·6교시 계층 조립과 Gradient check](#/nn/3/6) — 참값이 0 이면 상대 오차가 무의미
- [NN D5·5교시 SimpleConvNet 으로 MNIST 학습](#/nn/5/5) — CNN 에서도 Conv → BN → ReLU

**상세 학습**

#### 이 교시에서 답하는 질문
3교시의 초기화는 **출발점**에서만 분포를 맞춘다. 학습이 진행되면 가중치가 바뀌고, 층 출력의 분포도 다시 틀어진다. 그렇다면 매 step 마다 분포를 강제로 맞춰 주면 어떨까? 이것이 **배치 정규화(BN)** 다.

#### 개념 1 · BN 의 식 — 열마다 평균 0 · 분산 1, 그리고 γ · β
미니배치 x 의 shape 는 (N, D) — N 개 샘플, D 개 특성. BN 은 **열(특성)마다** 따로 정규화한다.

> μ = mean(x, axis=0), σ² = var(x, axis=0), x̂ = (x − μ) / √(σ² + ε), y = γ x̂ + β

| 기호 | 뜻 | 처음 값 |
|---|---|---|
| μ, σ² | 이 미니배치의 열별 평균 · 분산 | — |
| ε | 분산 0 일 때 0 으로 나누기 방지 | 1e-7 |
| γ, β | 학습되는 늘이기 · 옮기기 — "평균 0 · 분산 1 이 최선이 아니면 스스로 바꿔라" | 1, 0 |

정규화하면 출력 열 평균 = **β**, 표준편차 = **γ** 가 된다.

![BatchNorm — 미니배치의 열마다 평균 0 · 분산 1 로 맞춘 뒤 γ · β 로 늘이고 옮긴다. 학습 때는 running 평균 · 분산을 갱신하고, 추론 때는 그것을 쓴다](assets/fig/nn/d4-bn.svg)

#### 개념 2 · 학습 모드와 추론 모드
추론 때는 미니배치가 없을 수도 있다 (사진 한 장). 한 장으로 평균을 내면 자기 자신이라 x − μ = 0 → 출력이 항상 β 다. 그래서 학습 중에 배치 통계의 이동 평균을 모아 두었다가 추론 때 쓴다.

| 모드 | 정규화에 쓰는 μ, σ² | running 갱신 |
|---|---|---|
| 학습 (`train_flg=True`) | 이 배치의 통계 | running = 0.9 · running + 0.1 · 배치값 |
| 추론 (`train_flg=False`) | running 평균 · 분산 | 하지 않음 |

#### 개념 3 · backward 와 효과
- dβ = Σ dout, dγ = Σ x̂ · dout — 행 방향 합 (Affine 의 db 와 같은 이유: N 개 행에 같은 β 가 쓰였다).
- dx 는 열 합이 항상 0 이다 → BN 바로 앞 Affine 의 편향 기울기는 0 (1e-16 수준). gradient check 상대 오차가 커 보여도 참값이 0 이라 생기는 일이지 버그가 아니다.
- 효과: 784-100×5-10 · 1000장 · std 0.01 에서 BN 이 없으면 0.117 (학습 안 됨), 있으면 train 1.000. std 1.0 에서도 BN 이 없으면 발산하지만 BN 이 있으면 학습된다 → **초기화에 둔감**, 큰 학습률에서도 안정.

#### 코드로 확인
```python
def forward(self, x, train_flg=True):
    if train_flg:
        mu = x.mean(axis=0); xc = x - mu
        var = np.mean(xc ** 2, axis=0); std = np.sqrt(var + self.eps)
        xn = xc / std
        self.xc, self.xn, self.std = xc, xn, std             # backward 용
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
    else:
        xn = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
    return self.gamma * xn + self.beta
```
- **무엇을 하나**: 학습 모드면 배치 통계로 정규화하고 running 을 갱신, 추론 모드면 running 으로 정규화한 뒤 γ · β 를 적용한다.
- **어떤 개념**: 개념 1 · 2.
- **출력**: 평균 3 · 표준편차 2 근처의 (5, 4) 배치, γ = [1, 2, 0.5, 1.5], β = [0, 1, −1, 0.5] → 출력 열 평균 `[0, 1, -1, 0.5]` (= β), 열 표준편차 `[1, 2, 0.5, 1.5]` (= γ)

```python
bn_r = BatchNorm(np.ones(1), np.zeros(1)); r_ = np.random.default_rng(0)
for _ in range(50):
    bn_r.forward(r_.normal(5.0, 2.0, (100, 1)))      # N(5, 2²) 배치 100개짜리 50번
x1 = np.array([[5.0]])
bn_r.forward(x1, train_flg=False), bn_r.forward(x1, train_flg=True)
```
- **무엇을 하나**: running 통계가 참값에 다가가는지 보고, 한 장을 두 모드로 넣어 비교한다.
- **어떤 개념**: 개념 2 — 추론에 running 이 필요한 이유.
- **출력**: running_mean **5.03**, running_var **3.83** (참값 5, 4) · 추론 −0.015 vs 학습 모드 **0**

#### 한 번에 정리
- BN: 열마다 (x − μ)/√(σ² + ε) 후 γx̂ + β — Affine → BN → ReLU.
- 학습은 배치 통계 + running 갱신, 추론은 running 통계 (`train_flg=False`).
- dβ = Σdout, dγ = Σx̂·dout, dx 의 열 합 = 0.
- 효과: 초기화에 둔감 · 큰 학습률에서도 안정.
- ⚠ `axis=0` 을 빼먹으면 특성별이 아니라 전체 평균 하나.

**연관 문제**
<<QA 4>>

### 5교시 · 과적합

**키워드**: `과적합`, `train ↑ test 정체`, `test 손실 증가`, `과신`, `300장 · 6층 0.786`, `데이터 늘리기 0.926`, `모델 줄이기 0.802`, `조기 종료`

**개요**: 과적합은 학습 데이터를 **외워** 처음 보는 데이터에서 성능이 떨어지는 현상 — 데이터가 적고 모델이 클수록 잘 생긴다. ① MNIST **300장**으로 784-100×6-10(약 **13만** 파라미터)을 학습하면 train 1.0, test 0.786 에서 멈춘다. ② 정확도는 멈춘 것 같지만 **test 손실은 계속 오른다** — 틀리는 샘플에 점점 더 확신하며 틀리기 때문. ③ 해결: 데이터 늘리기(가장 확실) > 모델 줄이기 / 규제(6교시) / 조기 종료.

**핵심 용어**
- **과적합(overfitting)**: train 성능은 오르는데 test 성능이 멈추거나 떨어지는 현상
- **과소적합(underfitting)**: 학습 데이터조차 잘 못 맞히는 상태 (모델이 너무 작거나 규제가 너무 셈)
- **과신(overconfidence)**: 가중치가 커져 틀리는 샘플에도 높은 확률을 주는 상태 — CE 손실이 커짐
- **조기 종료(early stopping)**: test(검증) 손실이 오르기 시작할 때 학습을 멈추는 것

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | 300장으로 784-100×6-10 을 SGD(0.1) 200 epoch 학습 → train · test 정확도와 손실 곡선 (test 손실이 오름) |
| 무엇이 과적합을 줄이나 | 같은 모델을 3000장(20 epoch)으로, 그리고 300장에 작은 모델 784-100-10 으로 학습해 test 비교 |

**핵심 내용**

#### 과적합(overfitting)
> 학습 데이터를 **외워** train 성능은 오르는데 test 성능이 멈추거나 떨어지는 현상 — 데이터가 적고 모델이 클수록
- **신호** → train 정확도 ↑, test 정확도 정체, **test 손실 증가**
- **예** → MNIST 300장 · 784-100×6-10 (약 **130,010**개) · 200 epoch → train **1.000**, test **0.786**
- **과신** → 가중치가 커져 틀리는 샘플에도 높은 확률 → CE 는 확신하며 틀릴수록 큰 벌점 → test 손실이 오름
- ⚠ 정확도만 보면 놓친다 — 과신은 **손실 곡선**에서 먼저 드러남
- ⚠ train 정확도만 보고 "잘 학습됐다" 금지 → 학습에 쓰지 않은 데이터로 평가

#### 과적합 줄이기
> 데이터를 늘리는 것이 가장 확실 — 그다음 모델 줄이기 · 규제 · 조기 종료
- **데이터 10배(3000장)** → test **0.926** (iteration 수를 맞추려 20 epoch)
- **모델 줄이기(1층 784-100-10)** → 0.802 — 외울 능력 자체를 줄임 (너무 줄이면 **과소적합**: 학습 데이터조차 못 맞힘)
- **조기 종료** → 검증 손실이 오르기 시작하는 epoch 에서 멈춤 · 규제(6교시)는 0.80 ~ 0.81
- ⚠ test 를 보며 epoch · 모델을 고르면 test 가 "본 데이터"가 된다 → validation (7교시)

**기본 사용형태**
```python
net = MultiLayerNet(784, [100] * 6, 10)                    # 약 13만 파라미터
h = fit(net, SGD(lr=0.1), X300, t300, epochs=200, X_val=X_test, t_val=t_test, eval_every=5)
# h 의 train/test 정확도 · test 손실 곡선을 함께 그려 본다
```

**연결**
- [NN D2·6교시 출력층과 손실 함수](#/nn/2/6) — CE 는 확신하며 틀릴수록 큰 벌점
- [NN D2·8교시 미니 프로젝트: 로지스틱 회귀 분류기](#/nn/2/8) — 오래 학습하면 W 가 커져 과신
- [NN D3·8교시 미니 프로젝트: MNIST 2층 MLP](#/nn/3/8) — train 과 test 의 차이

**상세 학습**

#### 이 교시에서 답하는 질문
1 ~ 4교시로 학습은 빠르고 안정적이 되었다. 그런데 "학습 데이터를 잘 맞힌다" 와 "처음 보는 데이터를 잘 맞힌다" 는 다르다. 이 교시는 학습을 잘 할수록 오히려 test 성능이 나빠질 수 있는 현상 — **과적합** — 을 보고, 무엇이 그것을 줄이는지 비교한다.

#### 개념 1 · 과적합 — 외워 버린 모델
시험 범위의 문제와 답을 통째로 외운 학생은 같은 문제는 100점이지만 처음 보는 문제는 못 푼다. 모델도 같다. **데이터가 적고 모델이 크면** 외울 능력은 크고 외울 양은 적어서 과적합이 잘 생긴다.

MNIST **300장**으로 784-100×6-10 (파라미터 784·100 + 100 + 5·(100·100 + 100) + 100·10 + 10 = **130,010**개)을 200 epoch 학습하면:
- train 정확도 **1.000** (모두 외움)
- test 정확도 **0.786** 에서 멈춤
- test **손실은 계속 오른다**

![과적합의 신호 — 정확도는 train 1.0 · test 0.786 에서 벌어지고, test 손실은 오히려 오른다](assets/fig/nn/d4-overfit.svg)

#### 개념 2 · 왜 test 손실이 오르나 — 과신
학습 데이터를 완벽히 맞히려고 가중치가 점점 커진다 → softmax 출력이 0 · 1 에 가까워진다(확신) → 틀리는 test 샘플에도 높은 확신. 교차 엔트로피는 **확신하며 틀릴수록** 큰 벌점이라 (Day 2 −log) 정확도는 그대로인데 손실은 오른다. 그래서 과적합은 정확도보다 **손실 곡선**에서 먼저 보인다.

#### 개념 3 · 무엇이 과적합을 줄이나
| 방법 | test | 이유 |
|---|---|---|
| 300장 · 6층 (기준) | 0.786 | 과적합 |
| 데이터 10배 (3000장, 20 epoch) | **0.926** | 가장 확실 — 하지만 데이터는 비싸다 |
| 모델 줄이기 (1층 784-100-10) | 0.802 | 외울 능력 자체를 줄임 |
| 규제 (6교시) | 0.80 ~ 0.81 | 모델은 그대로, 학습에 제약 |

3000장을 20 epoch 만 한 것은 iteration 수를 300장 · 200 epoch 와 맞추기 위해서다. 그 밖에 **조기 종료**(검증 손실이 오르기 시작하면 멈춤)도 있다. 반대로 모델을 너무 줄이거나 규제를 너무 세게 하면 학습 데이터조차 못 맞히는 **과소적합**이 된다.

#### 코드로 확인
```python
X300, t300 = X_train[:300], t_train[:300]
net_plain = MultiLayerNet(784, [100] * 6, 10)
h_plain = fit(net_plain, SGD(lr=0.1), X300, t300, epochs=200, X_val=X_test, t_val=t_test, eval_every=5)
gap_plain = h_plain["train_acc"][-1] - h_plain["val_acc"][-1]
best_loss_epoch = h_plain["epoch"][int(np.argmin(h_plain["val_loss"]))]
```
- **무엇을 하나**: 300장으로 6층 MLP 를 200 epoch 학습하고, train − test 간극과 test 손실이 가장 낮았던 epoch 을 구한다.
- **어떤 개념**: 개념 1 · 2 — 간극과, test 손실이 최저를 찍고 다시 오르는 지점(= 조기 종료 시점).
- **출력**: train **1.000**, test **0.786**, 간극 약 0.21 (강의 실행 값)

```python
net_3000 = MultiLayerNet(784, [100] * 6, 10)
h_3000 = fit(net_3000, SGD(lr=0.1), X_train[:3000], t_train[:3000], epochs=20, X_val=X_test, t_val=t_test, eval_every=20)
net_small = MultiLayerNet(784, [100], 10)
h_small = fit(net_small, SGD(lr=0.1), X300, t300, epochs=200, X_val=X_test, t_val=t_test, eval_every=200)
```
- **무엇을 하나**: 데이터를 10배로 늘린 경우와 모델을 1층으로 줄인 경우를 학습한다.
- **어떤 개념**: 개념 3.
- **출력**: 3000장 · 6층 test **0.926** · 300장 · 1층 test **0.802**

#### 한 번에 정리
- 과적합: train ↑, test 정체, test 손실 ↑ — 데이터 적음 + 모델 큼.
- test 손실이 오르는 이유는 과신 (CE 는 확신하며 틀리면 큰 벌점).
- 줄이는 법: 데이터 늘리기 > 모델 줄이기 · 규제 · 조기 종료.
- ⚠ 정확도만 보지 말고 손실 곡선을 본다.

**연관 문제**
<<QA 5>>

### 6교시 · 규제: weight decay · Dropout

**키워드**: `규제`, `weight decay`, `L2`, `½λΣ‖W‖²`, `기울기 + λW`, `(1 − ηλ)W`, `Dropout`, `mask`, `추론 × (1 − ratio)`, `앙상블`, `λ 너무 크면 과소적합`

**개요**: 모델 크기는 그대로 두고 학습에 **제약**을 걸어 과적합을 줄인다(규제). ① **weight decay** 는 손실에 ½λΣ‖W‖² 벌점을 더해(기울기 + λW) 가중치가 커지지 못하게 → 과신이 줄어 test 손실이 내려간다. ② **Dropout** 은 학습 때 은닉 뉴런을 무작위로 꺼 매번 다른 작은 망을 학습시키고, 추론 땐 모두 켜고 (1 − ratio) 배 → 여러 망의 평균(앙상블) 효과. ③ 300장 · 6층에서 0.786 → weight decay 0.810, Dropout 0.800.

**핵심 용어**
- **규제(regularization)**: 모델은 그대로 두고 학습에 제약을 걸어 과적합을 줄이는 방법
- **weight decay(L2 규제)**: 손실에 ½λΣ‖W‖² 를 더해 가중치가 커지지 못하게 함 — 매 step W 를 (1 − ηλ) 배로
- **λ(lambda)**: 벌점의 세기 — 하이퍼파라미터 (7교시에서 고름)
- **Dropout**: 학습 때 은닉 뉴런을 비율 ratio 만큼 무작위로 끄는(출력 0) 기법
- **앙상블(ensemble)**: 여러 모델의 예측을 평균해 더 견고하게 만드는 것 — Dropout 의 추론이 그 근사

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `l2_penalty()` | `0.5 * lam * sum(np.sum(W ** 2) for W in 가중치들)` | L2 벌점 (편향 · γ · β 제외) |
| `Dropout` | `Dropout(dropout_ratio=0.5)` · `forward(x, train_flg)` · `backward(dout)` | 드롭아웃 계층 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| weight decay | `l2_penalty` 로 ½λΣ‖W‖² 를 계산해 `MultiLayerNet.loss` 결과(손실 + 벌점)와 일치 확인, λ = 0.01 로 300장 · 6층 200 epoch 학습 |
| Dropout | Dropout 계층의 학습 · 추론 출력과 backward 확인, ratio 0.2 로 같은 실험 → 규제 없음 · weight decay 와 곡선 비교 |

**핵심 내용**

#### weight decay (L2 규제)
> L' = L + ½λΣ‖W‖², 기울기 ∂L/∂W + **λW** — 가중치가 커지지 못하게 벌점을 주는 **규제**
- **규제(regularization)** → 모델은 그대로 두고 학습에 제약 — weight decay · Dropout
- **갱신** → W ← **(1 − ηλ)W** − η∂L/∂W — η 0.1 · 벌점 세기 λ(lambda) 0.01 이면 매 step W × 0.999
- **`l2_penalty()`** → 0.5 · λ · Σ W² — **W 에만** (편향 · γ · β 제외)
- **예** → 300장 · 6층, λ 0.01: test **0.810**, test 손실 1.12 → **0.65** (과신 감소)
- ⚠ λ 가 너무 크면(0.1) 과소적합 — train 0.767, test 0.641
- ⚠ 손실에만 벌점을 더하고 기울기에 λW 를 빼먹으면 → gradient check 로 잡힌다

#### Dropout
> 학습 때 은닉 뉴런을 비율 ratio 만큼 무작위로 끄고(mask), 추론 땐 모두 켜고 **(1 − ratio)** 배
- **왜** → 특정 뉴런에 기대지 못해 견고한 특징을 배운다 · 매번 다른 작은 망 = **앙상블** 효과
- **`Dropout`** → forward: `x * mask` (mask = rand > ratio) · backward: `dout * mask` (ReLU 의 mask 와 같다)
- **예** → ratio 0.2: 학습 때 켜진 비율 약 0.8 = 추론 출력 0.8 · 300장 · 6층 test **0.800** (초반은 느림: 25 epoch train 0.53)
- ⚠ 추론 때 마스크 금지 (`train_flg=False`) · 출력층 뒤가 아니라 **은닉층 뒤에만**

**기본 사용형태**
```python
def l2_penalty(net, lam):
    return 0.5 * lam * sum(np.sum(net.params[k] ** 2) for k in net.params if k.startswith("W"))

class Dropout:
    def __init__(self, dropout_ratio=0.5, seed=0):
        self.ratio, self.rng, self.mask = dropout_ratio, np.random.default_rng(seed), None
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = self.rng.random(x.shape) > self.ratio    # True = 살림
            return x * self.mask
        return x * (1.0 - self.ratio)
    def backward(self, dout):
        return dout * self.mask
```

**연결**
- [NN D3·4교시 활성화 계층: ReLU · Sigmoid](#/nn/3/4) — Dropout 역전파 = ReLU 의 mask
- [NN D2·8교시 미니 프로젝트: 로지스틱 회귀 분류기](#/nn/2/8) — W 가 커지며 생기는 과신

**상세 학습**

#### 이 교시에서 답하는 질문
5교시에서 데이터를 늘리는 것이 가장 확실했지만 데이터는 비싸다. 모델을 줄이면 표현력도 준다. 이 교시는 **모델은 그대로 두고 학습에 제약을 거는** 방법 — 규제 — 두 가지를 만든다.

#### 개념 1 · weight decay — 큰 가중치에 벌점
과신은 가중치가 커지면서 생겼다. 그렇다면 손실에 "가중치가 크면 벌점" 을 더하자.

> L' = L + ½ λ Σ‖W‖², ∂L'/∂W = ∂L/∂W + λW

갱신식을 정리하면 W ← W − η(∂L/∂W + λW) = **(1 − ηλ) W** − η ∂L/∂W — 매 step 가중치를 조금씩 줄인다(decay). η 0.1 · λ 0.01 이면 매 step × 0.999.

| 기호 | 뜻 |
|---|---|
| λ | 벌점의 세기 — 하이퍼파라미터 (7교시에서 고름) |
| ½ | 미분하면 2 가 나와서 상쇄되도록 |
| W | 가중치만 — 편향 · γ · β 에는 걸지 않는다 |

λ 가 너무 크면(0.1) 학습 데이터조차 못 맞혀 과소적합 (train 0.767, test 0.641).

#### 개념 2 · Dropout — 무작위로 끄고, 추론 땐 평균
학습 때마다 은닉 뉴런을 비율 ratio 만큼 무작위로 끈다(출력 0). 매번 다른 작은 망을 학습하는 셈이고, 특정 뉴런에 기댈 수 없으니 더 견고한 특징을 배운다.

![Dropout — 학습 때는 뉴런을 무작위로 끄고(×), 추론 때는 모두 켠 뒤 (1 − ratio) 배 한다](assets/fig/nn/d4-dropout.svg)

| 단계 | 계산 | 이유 |
|---|---|---|
| 학습 forward | x · mask (mask = rand > ratio) | 무작위로 끄기 |
| 학습 backward | dout · mask | 꺼진 뉴런으로는 기울기도 안 흐름 (ReLU 의 mask 와 같다) |
| 추론 | x · (1 − ratio) | 학습 때 평균 (1 − ratio) 만 켜져 있었으므로 크기를 맞춤 = 여러 망의 평균(앙상블) |

#### 개념 3 · 결과 — 정확도는 조금, 손실은 크게
300장 · 6층 · 200 epoch (강의 실행 값):

| 설정 | train | test | test 손실 |
|---|---|---|---|
| 규제 없음 | 1.000 | 0.786 | 1.12 (증가) |
| weight decay λ 0.01 | 1.000 | **0.810** | **0.65** (감소) |
| Dropout 0.2 | 1.000 | 0.800 | 1.33 |

weight decay 는 과신을 막아 test 손실을 크게 낮췄다. Dropout 은 정확도는 올렸지만 손실 개선은 작고, 초반 학습이 느리다 (25 epoch 에 train 0.53).

#### 코드로 확인
```python
def l2_penalty(net, lam):
    return 0.5 * lam * sum(np.sum(net.params[f"W{i}"] ** 2) for i in range(1, net.n_layers + 1))

net_w = MultiLayerNet(784, [100] * 6, 10, weight_decay_lambda=0.01)
L_with = net_w.loss(X300, t300)
L_ce = net_w.lastLayer.forward(net_w.predict(X300), t300)
np.isclose(L_with, L_ce + l2_penalty(net_w, 0.01))
```
- **무엇을 하나**: W1 ~ W7 의 제곱합으로 벌점을 계산하고, `MultiLayerNet.loss` 가 CE + 벌점과 같은지 확인한다.
- **어떤 개념**: 개념 1 — 벌점은 W 에만.
- **출력**: `True`

```python
class Dropout:
    def __init__(self, dropout_ratio=0.5, seed=0):
        self.ratio, self.rng, self.mask = dropout_ratio, np.random.default_rng(seed), None
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = self.rng.random(x.shape) > self.ratio     # True = 살림
            return x * self.mask
        return x * (1.0 - self.ratio)
    def backward(self, dout):
        return dout * self.mask
```
- **무엇을 하나**: 학습 때는 무작위 마스크를 곱하고 저장, 추론 때는 (1 − ratio) 배, 역전파는 마스크를 곱한다.
- **어떤 개념**: 개념 2.
- **출력**: ratio 0.2 · 1로 채운 (1000, 100) 입력 → 학습 출력 평균 **0.799** · 추론 출력 **0.8** — 평균이 같다

#### 한 번에 정리
- weight decay: 손실 + ½λΣW², 기울기 + λW → 매 step (1 − ηλ)W. W 에만.
- Dropout: 학습 땐 무작위로 끄고(mask), 추론 땐 (1 − ratio) 배 = 앙상블.
- 300장 · 6층: 0.786 → weight decay 0.810 (손실 0.65), Dropout 0.800.
- ⚠ λ 가 너무 크면 과소적합 · 추론 때 마스크를 쓰면 안 된다.

**연관 문제**
<<QA 6>>

### 7교시 · 하이퍼파라미터 탐색

**키워드**: `하이퍼파라미터`, `train / validation / test`, `test 는 마지막에 한 번`, `로그 스케일 10ᵘ`, `랜덤 탐색 > 격자`, `넓게 짧게 → 좁게 길게`, `최고 val 0.885 (lr 0.242)`

**개요**: 학습률 · λ · 은닉층 크기처럼 **사람이 정하는 값**이 하이퍼파라미터다. ① test 로 고르면 test 가 "본 데이터"가 되어 성능을 부풀려 보고하게 된다 → 학습 데이터 일부를 떼어 **validation** 으로 고른다. ② 학습률처럼 자릿수가 중요한 값은 지수를 균등하게 뽑는 **로그 스케일**, 그리고 격자보다 **랜덤 탐색**. ③ "넓게 짧게 → 좁게 길게" 를 2~3번 반복하고, 정해지면 train + validation 으로 다시 학습해 test 로 **한 번만** 평가.

**핵심 용어**
- **하이퍼파라미터(hyperparameter)**: 학습으로 찾지 않고 사람이 정하는 값 — 학습률, λ, 층 크기, 배치 크기
- **검증 데이터(validation)**: 하이퍼파라미터를 고르는 데 쓰는, 학습에 쓰지 않은 데이터
- **로그 스케일(log scale)**: 값 대신 지수를 균등하게 뽑기 — η = 10ᵘ, u ~ U(−4, 0)
- **랜덤 탐색(random search)**: 조합을 무작위로 뽑아 시도 — 격자(grid) 탐색보다 효율적

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `log_uniform()` | `10 ** rng.uniform(low, high)` | 지수를 균등하게 뽑기 — 0.0001 ~ 1 을 자릿수마다 고르게 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | MNIST 학습 데이터 1000장을 섞어 앞 20%(**200**장)를 validation, 나머지 **800**장을 train 으로 |
| 로그 스케일 랜덤 탐색 | lr · λ 를 로그 스케일로 **30**번 뽑아 784-100-100-10 을 20 epoch 학습(약 15초) → val 정확도를 lr 축에 산점도로 |

**핵심 내용**

#### 하이퍼파라미터와 검증 데이터
> 학습으로 찾지 않고 사람이 정하는 값(학습률 · λ · 층 크기) — **validation** 으로 고르고, test 는 마지막에 **한 번**
- **세 갈래** → train: W · b 학습 · validation: 하이퍼파라미터 고르기 · test: 최종 보고
- **예** → MNIST 1000장을 섞어 **200**장 validation, **800**장 train
- **최종** → 정해지면 train + validation 으로 다시 학습 → test 로 한 번 평가
- ⚠ 탐색 중 test 정확도를 보고 고르면 성능을 부풀려 보고하게 된다

#### 로그 스케일 · 랜덤 탐색
> 값 대신 **지수**를 균등하게 뽑는다 (η = 10ᵘ, u ~ U(−4, 0)) — 0.001 vs 0.01 차이가 0.001 vs 0.002 보다 중요하므로
- **`log_uniform()`** → `10 ** rng.uniform(low, high)` — 0.0001 ~ 1 의 각 자릿수에 약 25%씩
- **랜덤 > 격자** → 한 축만 중요할 때 3 × 3 격자는 그 축의 값 3개, 랜덤 9번은 **9개**를 본다
- **예** → 30번 시도, 최고 val **0.885** (lr **0.242**) — lr 0.1 ~ 0.6 이 좋고 λ 영향은 작음
- ⚠ `rng.uniform(1e-4, 1)` → 약 **90%** 가 0.1 ~ 1 에 몰림 · 처음부터 긴 epoch 금지 (넓게 짧게 → 좁게 길게)

**기본 사용형태**
```python
def log_uniform(rng, low, high): return 10 ** rng.uniform(low, high)

rng = np.random.default_rng(0)
for trial in range(30):
    lr  = log_uniform(rng, -4, 0)          # 0.0001 ~ 1
    lam = log_uniform(rng, -6, -2)
    # 짧게(20 epoch) 학습 → validation 정확도 기록
```

**연결**
- [NN D4·5교시 과적합](#/nn/4/5) — test 를 보며 고르면 안 되는 이유
- [NN D1·7교시 확률 기초](#/nn/1/7) — 균등분포 난수

**상세 학습**

#### 이 교시에서 답하는 질문
오늘 정한 값만 해도 학습률, λ, dropout 비율, 층 크기, 배치 크기 — 이렇게 **사람이 정하는 값**을 하이퍼파라미터라 한다. 이 교시는 이 값들을 **무엇으로 평가해서**, **어떻게 뽑아서** 고르는지 답한다.

#### 개념 1 · 데이터 세 갈래 — test 는 마지막에 한 번
test 정확도를 보며 하이퍼파라미터를 고르면, test 에 맞춰 고른 셈이라 test 가 더 이상 "처음 보는 데이터" 가 아니다. 보고한 성능이 실제보다 부풀려진다. 그래서 학습 데이터 일부를 떼어 **검증(validation) 데이터** 로 쓴다.

| 데이터 | 용도 | 몇 번 보나 |
|---|---|---|
| train | 파라미터(W, b) 학습 | 매 epoch |
| validation | 하이퍼파라미터 고르기 | 시도마다 |
| test | 최종 성능 보고 | **마지막에 한 번** |

하이퍼파라미터가 정해지면 train + validation 을 합쳐 다시 학습하고 test 로 한 번 평가한다.

#### 개념 2 · 로그 스케일과 랜덤 탐색
학습률은 0.001 과 0.002 의 차이보다 0.001 과 0.01 의 차이가 훨씬 중요하다 — **자릿수**가 중요하다. 그래서 값 대신 지수를 균등하게 뽑는다.

> η = 10ᵘ, u ~ U(−4, 0) → 0.0001 ~ 1 의 각 자릿수에 약 25%씩

그냥 `uniform(0.0001, 1)` 로 뽑으면 약 90% 가 0.1 ~ 1 에 몰려 작은 학습률은 거의 시도되지 않는다.

또 격자(grid)보다 **랜덤 탐색**이 낫다. 두 하이퍼파라미터 중 하나만 중요할 때, 3 × 3 격자 9번은 중요한 축에서 서로 다른 값을 3개만 보지만 랜덤 9번은 9개를 본다.

![격자 탐색과 랜덤 탐색 — 같은 9번 시도에서 중요한 축의 서로 다른 값이 3개 대 9개](assets/fig/nn/d4-search.svg)

탐색은 "넓은 범위 · 짧은 epoch → 좁힌 범위 · 긴 epoch" 를 2 ~ 3번 반복한다.

#### 코드로 확인
```python
rng7 = np.random.default_rng(0)
p = rng7.permutation(1000)
X_val7, t_val7 = X_train[:1000][p[:200]], t_train[:1000][p[:200]]
X_tr7, t_tr7 = X_train[:1000][p[200:]], t_train[:1000][p[200:]]
```
- **무엇을 하나**: 학습 데이터 1000장을 섞어 앞 200장을 validation, 나머지 800장을 train 으로 나눈다. test 10000장은 건드리지 않는다.
- **어떤 개념**: 개념 1.
- **출력**: train `(800, 784)` · validation `(200, 784)` — 겹치는 인덱스 0개

```python
def log_uniform(rng_, low, high):
    return 10 ** rng_.uniform(low, high)

for trial in range(30):
    lr, lam = log_uniform(rng7, -4, 0), log_uniform(rng7, -6, -1)
    net = MultiLayerNet(784, [100, 100], 10, weight_decay_lambda=lam, seed=trial)
    h = fit(net, SGD(lr=lr), X_tr7, t_tr7, epochs=20, batch_size=100, seed=trial,
            X_val=X_val7, t_val=t_val7, eval_every=20)
    results.append((h["val_acc"][-1], lr, lam, h["train_acc"][-1]))
```
- **무엇을 하나**: lr · λ 를 로그 스케일로 30번 뽑아 짧게(20 epoch) 학습하고 validation 정확도를 기록한다.
- **어떤 개념**: 개념 2 — 넓게 짧게 훑는 첫 단계.
- **출력**: 최고 val **0.885** (lr **0.242**) · lr 0.1 ~ 0.6 이 0.84 ~ 0.885 · lr < 0.001 은 0.1 근처 · λ 영향은 작음 (강의 실행 값)

#### 한 번에 정리
- 하이퍼파라미터는 validation 으로 고르고, test 는 마지막에 한 번.
- 자릿수가 중요한 값은 로그 스케일(10ᵘ)로 뽑는다.
- 격자보다 랜덤 탐색 · 넓게 짧게 → 좁게 길게.
- ⚠ `uniform(1e-4, 1)` 은 90% 가 0.1 ~ 1 에 몰린다.

**연관 문제**
<<QA 7>>

### 8교시 · 미니 프로젝트: 미니 프레임워크 mynn.py

**키워드**: `mynn.py`, `%%writefile`, `Layer`, `Sequential`, `params() "0.W"`, `복사하지 않는다`, `Trainer.fit`, `save_params · load_params`, `BN running 저장`, `784-100×4-10 + BN + He + Adam`, `110,610`, `BN folding`

**개요**: Day 3 · 4 의 부품을 **재사용 가능한 모듈 하나**(mynn.py)로 정리한다 — PyTorch · Keras 가 하는 일의 핵심. ① 계층을 리스트로 쌓고(**Sequential**), 파라미터를 한곳에 모으고(`params()` — 복사하지 않고 그대로), 학습 루프를 한 번만 작성한다(**Trainer**). ② 784-100×4-10 + BN + He + Adam 으로 **5 epoch** 만에 test 0.975 를 넘는다(Day 3 은 10 epoch 에 0.9694). ③ 학습 기법은 추론 비용을 늘리지 않는다 — BN 은 앞 Affine 에 접어 넣을 수 있다(BN folding).

**핵심 용어**
- **Sequential**: 계층 리스트를 순서대로 forward · 반대로 backward 하는 모델 컨테이너
- **Trainer**: 셔플 → 배치 → gradient → update 학습 루프와 history 기록을 맡는 클래스
- **buffers**: 학습되지 않지만 저장해야 하는 값 — BN 의 running 평균 · 분산
- **`%%writefile`**: 셀 내용을 파일로 저장하는 Colab(IPython) 명령 (`-a` 는 끝에 덧붙이기)
- **BN folding**: 추론 때 고정된 BN 을 앞의 Affine 에 합쳐 계층 수를 줄이는 최적화 (TensorRT 등)

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Sequential` | `Sequential([Affine(784, 100, "he", 1), BatchNorm(100), Relu(), …], weight_decay_lambda)` | params() · grads() · predict() · loss() · gradient() |
| `Trainer` | `Trainer(model, Adam(0.001), batch_size=100).fit(X, t, epochs, X_val, t_val)` | 셔플 → 배치 → gradient → update |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| mynn.py ① — 함수 · 계층 · 옵티마이저 | 주어진 코드 — Affine · Relu · Sigmoid · BatchNorm · Dropout · 옵티마이저 5종을 `%%writefile mynn.py` 로 저장 |
| 과제 1 — Sequential | `params()` 를 `"0.W"` 꼴 키로 모으고, `grads()` 에 λW 를 더하고, gradient check — `0.b` 만 상대 오차가 커 보임(참값 0) |
| 과제 2 — Trainer.fit · load_params | Day 3 7교시 루프를 `fit` 으로, 저장된 값을 제자리 복사하는 `load_params` |
| 과제 3 — 5층 MLP 구성 · 학습 | 784 → 100 × 4 → 10, 은닉층마다 Affine(He) → BatchNorm → Relu, Adam 0.001 · 배치 100 · 5 epoch → test 0.97 이상 |
| 과제 4 — 저장하고 다시 불러오기 | 저장 → 다른 seed 로 만든 같은 구조 모델에 불러와 정확도가 똑같은지 확인, Drive 에 mynn.py 복사 |
| 온디바이스 관점 — 학습 기법은 추론 비용을 늘리지 않는다 | BN 을 앞 Affine 에 접어 넣어 13계층 → **9계층**으로 줄여도 test 정확도가 같은 것 확인 |

**핵심 내용**

#### 미니 프레임워크 mynn.py
> 계층을 리스트로 쌓고(**Sequential**), 파라미터를 한곳에 모으고, 학습 루프를 한 번만(**Trainer**) — PyTorch · Keras 의 핵심
- **`Sequential`** → `params()` 가 `{"0.W", "1.gamma", …}` 로 계층 배열을 **복사하지 않고 그대로** 모음 · `grads()` 는 W 에 λW 를 더함
- **`Trainer`** → `fit(X, t, epochs, X_val, t_val)` = 셔플 → 배치 → gradient → update + history
- **`%%writefile`** → 셀 내용을 mynn.py 로 저장 (`-a` 는 덧붙이기) · save_params · load_params 는 **buffers**(BN running)까지
- ⚠ `params()` 가 복사본을 주면 옵티마이저가 복사본만 바꿔 학습이 안 됨
- ⚠ `load_params` 는 **제자리 복사** `p[...] = 저장값` — 새 배열 대입은 연결이 끊김

#### 5층 MLP — 오늘의 기법을 모두 합치면
> 784-100×4-10, 은닉층마다 Affine(He) → BatchNorm → Relu + Adam(0.001) — **5 epoch** 에 test **0.9757**
- **Day 3 와 비교** → 784-100-10 · std 0.01 · SGD · 10 epoch → 0.9694
- **파라미터 110,610** → Affine 109,810 + BN γ · β 4층 × 200
- **기여** → He(신호 유지) · BN(안정) · Adam(파라미터별 학습률) · 깊은 층(표현력)

#### BN folding — 학습 기법은 추론 비용을 늘리지 않는다
> 추론 때 고정된 BN 은 선형 변환 → 앞 Affine 에 합친다: W' = W·γ/√(σ² + ε), b' = (b − μ)·γ/√(σ² + ε) + β
- **예** → 13계층 → **9계층**, 정확도 동일 — TensorRT 같은 추론 엔진이 실제로 하는 최적화
- **나머지 기법** → Adam · He · weight decay 는 학습 때만, Dropout 은 추론 때 곱하기 하나

**기본 사용형태**
```python
import mynn
layers = []
for i, (n_in, n_out) in enumerate([(784, 100), (100, 100), (100, 100), (100, 100)]):
    layers += [mynn.Affine(n_in, n_out, "he", i + 1), mynn.BatchNorm(n_out), mynn.Relu()]
layers.append(mynn.Affine(100, 10, "he", 5))
model = mynn.Sequential(layers)
trainer = mynn.Trainer(model, mynn.Adam(0.001), batch_size=100)
trainer.fit(X_train, t_train, epochs=5, X_val=X_test, t_val=t_test)
```

**연결**
- [NN D3·6교시 계층 조립과 Gradient check](#/nn/3/6) — TwoLayerNet 을 Sequential 로 일반화
- [NN D5·4교시 Convolution · Pooling 계층 → mynn.py](#/nn/5/4) — mynn.py 에 CNN 계층 추가
- [NN D5·7교시 모델 비용 계산](#/nn/5/7) — 추론 비용 · 메모리

**상세 학습**

#### 이 교시에서 답하는 질문
Day 3 · 4 에서 계층, 옵티마이저, 초기화, BN, Dropout, weight decay 를 각각 만들었다. 이것들을 매번 노트북에 다시 붙여 넣지 않고 **재사용 가능한 모듈 하나**(mynn.py)로 묶으면, 새 모델을 몇 줄로 조립할 수 있다. PyTorch · Keras 가 하는 일의 핵심이 바로 이것이다.

#### 개념 1 · 프레임워크의 세 부품
| 부품 | 하는 일 |
|---|---|
| 계층 (Affine · Relu · BatchNorm · Dropout …) | 모두 같은 규약: `params` · `grads` 딕셔너리, `forward(x, train_flg)`, `backward(dout)` — BN 은 `buffers`(running 통계)도 |
| `Sequential(layers)` | 계층 리스트를 순서대로 forward, 반대로 backward · `params()` 로 파라미터를 한곳에 |
| `Trainer(model, optimizer)` | 셔플 → 배치 → gradient → update 학습 루프를 한 번만 작성 |

핵심은 `params()` 가 계층의 배열을 **복사하지 않고 그대로** 모은다는 점이다. 그래서 옵티마이저가 제자리 갱신(`-=`)하면 계층의 가중치가 바로 바뀐다 (Day 3 의 "같은 배열 객체" 원리).

![mynn.py 의 구조 — Sequential.params() 는 계층의 배열을 복사 없이 모아 옵티마이저에 넘긴다](assets/fig/nn/d4-mynn.svg)

#### 개념 2 · 오늘의 기법을 모두 합친 5층 MLP
784 → 100 × 4 → 10, 은닉층마다 Affine(He) → BatchNorm → Relu (13계층), Adam 0.001 · 배치 100.

| 항목 | Day 3 | Day 4 |
|---|---|---|
| 모델 | 784-100-10 | 784-100×4-10 + BN |
| 초기화 · 옵티마이저 | std 0.01 · SGD 0.1 | He · Adam 0.001 |
| 학습 | 10 epoch | **5** epoch |
| test | 0.9694 | **0.9757** |
| 파라미터 | 79,510 | **110,610** |

파라미터 110,610 = Affine (784·100 + 100) + 3 × (100·100 + 100) + (100·10 + 10) = 109,810 + BN γ · β 4층 × 200 = 800.

#### 개념 3 · BN folding — 학습 기법은 추론 비용을 늘리지 않는다
추론 때 BN 의 μ, σ², γ, β 는 모두 고정값이라 y = γ(x − μ)/√(σ² + ε) + β 는 그냥 선형 변환이다. 바로 앞 Affine 에 합칠 수 있다.

> W' = W · γ / √(σ² + ε), b' = (b − μ) · γ / √(σ² + ε) + β

13계층 → 9계층, 정확도 동일. Adam · He · weight decay 는 학습 때만 쓰이고, Dropout 은 추론 때 곱하기 하나 — 추론 모델은 여전히 행렬곱 + ReLU 다.

#### 코드로 확인
```python
def params(self):
    """{'0.W': 배열, '0.b': 배열, '1.gamma': …} — 계층의 배열을 그대로(같은 객체) 모은다"""
    return {f"{i}.{name}": p for i, layer in enumerate(self.layers) for name, p in layer.params.items()}

def grads(self):
    g = {}
    for i, layer in enumerate(self.layers):
        for name, d in layer.grads.items():
            if name == "W" and self.weight_decay_lambda:
                d = d + self.weight_decay_lambda * layer.params["W"]     # weight decay λW
            g[f"{i}.{name}"] = d
    return g
```
- **무엇을 하나**: `Sequential` 의 두 메서드 — 파라미터와 기울기를 "계층번호.이름" 키로 모은다. 기울기에는 W 일 때만 λW 를 더한다.
- **어떤 개념**: 개념 1 — 복사 없이 모으기 · 6교시 weight decay.
- **출력**: 5층 MLP 의 키 `0.W, 0.b, 1.gamma, 1.beta, 3.W, …, 12.b` — gradient check 에서 `0.b` 만 상대 오차가 커 보임 (BN 앞 편향, 참값 0)

```python
sizes = [784, 100, 100, 100, 100, 10]
layers = []
for i in range(5):
    layers.append(mynn.Affine(sizes[i], sizes[i + 1], weight_init="he", seed=i + 1))
    if i < 4:                                                    # 은닉층
        layers += [mynn.BatchNorm(sizes[i + 1]), mynn.Relu()]
model = mynn.Sequential(layers)
trainer = mynn.Trainer(model, mynn.Adam(lr=0.001), batch_size=100, seed=0)
history = trainer.fit(X_train, t_train, epochs=5, X_val=X_test, t_val=t_test)
```
- **무엇을 하나**: 5층 MLP 를 조립하고 Adam 으로 5 epoch 학습한다.
- **어떤 개념**: 개념 2.
- **출력**: 계층 **13**개 · 파라미터 **110,610**개 · 5 epoch 후 test **0.9757** (강의 실행 값)

```python
def load_params(model, path):
    data = np.load(path)
    for k, p in model.state().items():
        p[...] = data[k]                  # 제자리 복사 — 계층 · 옵티마이저와의 연결 유지
```
- **무엇을 하나**: 저장한 .npz 에서 파라미터와 BN running 통계를 읽어 기존 배열에 **제자리 복사**한다.
- **어떤 개념**: 개념 1 — `p = data[k]` 처럼 새 배열을 대입하면 계층과 연결이 끊긴다.
- **출력**: 다른 seed 로 만든 같은 구조 모델이 불러온 뒤 원래 모델과 **같은 정확도**

#### 한 번에 정리
- 계층 규약 + Sequential(params · grads) + Trainer(루프) = 미니 프레임워크.
- params() 는 복사 없이 모은다 → 옵티마이저의 제자리 갱신이 계층에 바로 반영.
- He + BN + Adam → 5 epoch 에 test 0.9757 (Day 3 은 10 epoch 0.9694).
- BN folding: 추론 때 BN 을 Affine 에 합쳐 13 → 9계층, 정확도 동일.
- ⚠ 저장 · 불러오기는 BN running 통계까지, 불러오기는 제자리 복사로.

**연관 문제**
<<QA 8>>
