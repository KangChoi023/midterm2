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

#### 지그재그의 원인 — 기울기는 최소점을 향하지 않는다
> f(x, y) = x²/20 + y², ∇f = (x/10, 2y) — y 방향이 x 방향보다 **20배** 가파르다
- **(−7, 2) 에서 기울기** → (−0.7, 4) → 거의 **y 쪽**만 가리킴 → 최소점 (0, 0) 방향이 아님
- **y 방향** → 매 step y ← (1 − 2η)y — 가파르니 금방 튐
- **x 방향** → 매 step x ← (1 − η/10)x — 완만하니 아주 느림
- **실제 신경망** → 파라미터 수만 개 공간에서 이런 비등방성이 훨씬 심함
##### 자세히
```text
 y
  2 ┤ ●╲    ╱╲
    │    ╲╱    ╲    ╱╲
  0 ┤───────────╲╱────╲╱──── ★ (0, 0)
 -2 ┤
    └────────────────────────► x
   -7                         0
 steep in y → bounces · flat in x → creeps
```

#### 학습률을 바꿔 봐도 — 상한과 속도의 줄다리기
> 가장 가파른 y 방향이 상한 **η < 1** 을 정하고, 그 학습률로는 x 방향이 느리다
| lr | 30 step 후 (x, y) | 무슨 일 |
|---|---|---|
| 0.1 | (**−5.18**, 0.002) | y 는 금방 0, x 는 −7 → −5.18 밖에 못 감 |
| 0.95 | (−0.35, 0.08) | y 는 부호를 바꾸며(지그재그) 줄고 x 도 어느 정도 감 |
| 1.0 | (−0.30, **2**) | y ← −y → ±2 를 영원히 오감 |
| 1.05 | (−0.25, **34.9**) | \|1 − 2.1\| = 1.1 배씩 커져 **발산** |
- **어떤 학습률도** → 두 방향을 동시에 만족시키지 못함 → 방향마다 다르게 움직이는 옵티마이저가 필요
- **Day 1 x² + 3y²** → 상한 1/3 이었던 것과 같은 원리 (상한 = 2 / 가장 큰 곡률)

#### 해결 아이디어 두 가지
> 관성으로 지그재그를 상쇄하거나, 파라미터마다 학습률을 따로 조절한다
- **관성(속도)** → 부호가 바뀌는 성분은 상쇄, 일관된 방향은 가속 → **Momentum**
- **파라미터별 학습률** → 많이 움직인 방향은 작게 → **AdaGrad · RMSProp**
- **둘 다** → **Adam**
- **공통 형태** → `update(params, grads)` 로 params 딕셔너리 배열을 **제자리** 갱신 (Day 3 7교시)

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

#### 갱신식 한 장
> 모두 `update(params, grads)` — g = ∂L/∂W, 1e-7 은 0 으로 나누기 방지
| 옵티마이저 | 갱신식 | 직관 |
|---|---|---|
| SGD | W ← W − ηg | 기울기 반대로 |
| Momentum | v ← αv − ηg, W ← W + v (α = **0.9**) | 공이 굴러가듯 관성 |
| AdaGrad | h ← h + g², W ← W − ηg / (√h + 1e-7) | 많이 움직인 파라미터는 천천히 (결국 멈춤) |
| RMSProp | h ← ρh + (1 − ρ)g² (ρ = **0.99**), 나머지 AdaGrad 와 같음 | 최근 기울기만 반영 → 멈추지 않음 |
| Adam | m ← β₁m + (1 − β₁)g, v ← β₂v + (1 − β₂)g², m̂ = m/(1 − β₁ᵗ), v̂ = v/(1 − β₂ᵗ), W ← W − η m̂ / (√v̂ + 1e-7) | 관성 + 파라미터별 학습률 |
- **Adam 기본값** → β₁ = **0.9**, β₂ = **0.999**, lr = **0.001**
- **편향 보정이 필요한 이유** → t = 1 이면 m = 0.1g 뿐 → 1 − 0.9¹ = 0.1 로 나눠 m̂ = g 로 되돌림
##### 자세히
```text
 SGD        W ── −ηg ──►
 Momentum   v = 0.9·v − ηg      same sign → speeds up · flipping sign → cancels
 AdaGrad    h = h + g²          big h → small step (forever shrinking)
 RMSProp    h = 0.99·h + 0.01·g²  (forgets old g)
 Adam       m (direction)  +  v (scale)  +  bias correction
```

#### 지형 위에서 본 차이
> 같은 1교시 지형 — 옵티마이저마다 경로 모양이 다르다
- **Momentum** → 처음엔 SGD 처럼 y 로 튀지만 관성이 x 방향 속도를 쌓아 빠르게 가로지름 (최소점을 지나쳤다 돌아옴)
- **AdaGrad** → y 방향은 처음 기울기가 커서 h 가 빨리 커짐 → 학습률이 금방 줄어 지그재그가 사라짐
- **Adam** → 관성 + 파라미터별 학습률 → 부드러운 곡선으로 접근
- **이 함수 하나로** → "어느 옵티마이저가 최고" 라고 할 수 없다 → 실제 신경망에서 비교

#### MNIST 5층 MLP (2000 iteration)
> SGD 가 가장 느리고, 나머지는 약 0.97 — 실무 기본값은 Adam
| SGD (0.01) | Momentum (0.01) | AdaGrad (0.01) | RMSProp (0.001) | Adam (0.001) |
|---|---|---|---|---|
| 0.9378 | 0.9644 | 0.9726 | **0.9745** | 0.9740 |
- **실무** → Adam(lr ≈ **0.001**) 기본, 마지막 성능을 짜낼 때 SGD + Momentum 재시도
- **`MultiLayerNet`** → Day 3 TwoLayerNet 을 은닉층 여러 개로 일반화 + 오늘의 옵션(초기화 · BN · Dropout · weight decay)
- ⚠ 상태를 `__init__` 에서 만들면 파라미터 shape 를 모름 → 첫 update 에서 `zeros_like`
- ⚠ `params[k] = params[k] - …` → 새 배열로 Affine 과 연결이 끊김 → `-=` · `+=`
- ⚠ Adam 의 `self.t += 1` 을 파라미터 for **안**에 넣으면 t 가 파라미터 수만큼 커져 편향 보정이 틀어짐
- ⚠ 옵티마이저 객체를 여러 모델에 재사용 → 상태(v, h, m)가 섞임 → 모델마다 새로

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

#### 같은 값 초기화 = 대칭 — 100개가 1개
> 같은 입력 × 같은 가중치 → 같은 출력 → 같은 기울기 → 같이 갱신 → 영원히 같다
- **모두 0.01 (같은 값)** → 은닉 뉴런 100개가 사실상 **1개**
- **모두 0** → 더 나쁨: ReLU 출력이 0 → 기울기 자체가 **0**
- **편향은 0 이어도 괜찮다** → 가중치가 이미 서로 다르므로
- ⚠ 모든 가중치를 0(또는 같은 값)으로 초기화 → 대칭이 깨지지 않음 → 반드시 **무작위**

#### 활성화 분포 — 너무 크거나 너무 작으면
> 출력 분산 ≈ n × Var(W) × Var(x) → **Var(W) = 1/n** 이면 층을 지나도 분산이 유지
| 조합 | 층별 활성화 | 결과 |
|---|---|---|
| sigmoid + std 1 | 약 **82%** 가 0.1 미만 · 0.9 초과 | 포화 → 기울기 소실 |
| sigmoid + std 0.01 | 모두 0.5 근처 | 뉴런이 같은 값 → 표현력 제한 |
| ReLU + std 0.01 | 표준편차 0.058 → 0.004 → 0.0003 … | 층마다 **신호 소멸** |
| ReLU + Xavier | 0.58 → 0.18 | 점점 줄어듦 |
| ReLU + He | 0.82 → **1.00** | **유지** |
- **Xavier** → std = √(1/n) — n = 100 이면 **0.1**
- **He** → std = √(2/n) — ReLU 가 음수 절반을 0 으로 만들어 분산이 ½ → 2배로 보상
##### 자세히
```text
 too large (sigmoid, std 1)    ▇▁▁▁▁▁▁▁▇     piled at 0 and 1 → gradient ≈ 0
 too small (ReLU, std 0.01)    ▇▁▁▁▁▁▁▁▁     everything near 0 → signal vanishes
 He (ReLU)                     ▇▆▅▄▃▂▂▁▁     spread kept layer after layer
```

#### MNIST 4층에서 확인 (SGD 0.01, 2000 iteration)
> 초기화만 바꿨는데 학습이 되느냐 마느냐가 갈린다
| 초기화 | std 0.01 | Xavier | He |
|---|---|---|---|
| test | **0.11** (학습 안 됨) | 0.92 | **0.94** |
- **std 0.01** → 신호가 사라져 10% (찍기 수준)
- ⚠ He 의 n 은 **입력 노드 수(이전 층 뉴런 수)** — 출력 노드 수 아님
- ⚠ sigmoid 망에 He, ReLU 망에 Xavier — 짝을 맞추기 (틀려도 학습은 되지만 느리거나 불안정)

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

**연관 문제**
<<QA 3>>

### 4교시 · Batch Normalization

**키워드**: `BatchNorm`, `열마다 평균 0 · 분산 1`, `γ · β`, `ε = 1e-7`, `running 평균 · 분산`, `train_flg`, `momentum 0.9`, `Affine → BN → ReLU`, `dx 의 열 합 = 0`

**개요**: 초기화는 출발점에서만 분포를 맞춘다 — 학습이 진행되면 가중치가 바뀌어 분포가 다시 틀어진다. ① **BN** 은 매 미니배치마다 층 출력을 **열(특성)마다** 평균 0 · 분산 1 로 맞춘 뒤, 학습되는 γ · β 로 늘이고 옮긴다. ② 추론 때는 미니배치가 없을 수도(사진 한 장) 있으니 학습 중 모아 둔 **running 평균 · 분산**을 쓴다 → `train_flg` 가 필요한 이유. ③ 효과: 초기화에 둔감해지고 큰 학습률에서도 안정적 → 학습이 빠르다.

**핵심 용어**
- **배치 정규화(Batch Normalization, BN)**: 미니배치의 열마다 x̂ = (x − μ) / √(σ² + ε) 로 정규화하고 y = γx̂ + β
- **γ(scale) · β(shift)**: 정규화 후 다시 늘이고 옮기는 학습 파라미터 (처음 **1**, **0**)
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

#### 정규화 식 — 열마다 평균 0 · 분산 1, 그리고 γ · β
> μ = mean(x, axis=0), σ² = var(x, axis=0), x̂ = (x − μ) / √(σ² + ε), y = γx̂ + β
- **열(특성)마다** → (N, D) 미니배치의 D 개 열 각각 따로 — `axis=0`
- **γ · β** → "평균 0 · 분산 1 이 최선이 아니면 네트워크가 스스로 늘이고 옮겨라" — 출력 열 평균 = **β**, 표준편차 = **γ**
- **넣는 자리** → 보통 **Affine → BN → ReLU**
- ⚠ `x.mean()` (축 없음) → 전체 평균 하나 → 특성마다 **`axis=0`**
- ⚠ γ, β 를 옵티마이저 params 에 안 넣으면 학습 안 됨 (`MultiLayerNet` 은 `gamma{i}`, `beta{i}`)
##### 자세히
```text
 x (N, D)          per column                    output
 col j: 3.1 5.2 …  ──►  (x − μⱼ) / √(σⱼ² + ε)  ──►  γⱼ · x̂ + βⱼ
                         mean 0, var 1              mean βⱼ, std γⱼ
```

#### 학습 · 추론 모드 — running 통계
> 학습: 배치 통계로 정규화 + running 갱신 · 추론: 모아 둔 running 통계로 (배치 구성과 무관)
- **running 갱신** → `running = 0.9 · running + 0.1 · 배치값` — **학습 모드에서만**
- **N(5, 2²) 배치 50번** → running_mean ≈ 5, running_var ≈ 4 (참값에 근접)
- **학습 모드에 한 장** → 평균이 자기 자신 → 출력 항상 **0** → 추론에 running 이 꼭 필요한 이유
- ⚠ 추론 때 `train_flg=True` → 배치 구성에 따라 결과가 바뀌고 한 장이면 0 → predict 는 **`train_flg=False`**
- ⚠ running 통계를 추론 때 갱신 → 갱신은 학습 모드에서만

#### backward 요점
> dβ = Σ dout, dγ = Σ x̂ · dout (행 방향 합) — dx 는 열 합이 **0**
- **dβ** → β 가 N 개 행에 더해졌으니 행 방향 합 (Affine 의 db 와 같은 이유)
- **dγ** → 곱셈 노드 → 상대편 x̂ 를 곱해 행 방향 합
- **dx 의 열 합 = 0** → BN 바로 앞 Affine 의 편향 기울기는 0 (1e-16 수준) → gradient check 상대 오차가 커 보여도 **버그 아님** (참값이 0)

#### BN 의 효과 — 초기화에 둔감
> MNIST 1000장, 784-100×5-10, SGD(0.01), 20 epoch — 초기화 std 를 바꿔 BN 유무 비교
- **std 0.01 · BN 없음** → 전혀 학습 안 됨 (**0.117**) · **BN 있음** → train **1.000**
- **std 1.0 · BN 없음** → 발산(overflow) · BN 있음 → 학습됨
- **큰 학습률에서도 안정** → 학습이 빠르다
- **train 과 test 차이** → 1000장만 써서 생긴 과적합 (5교시)

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

#### 과적합의 신호 — 손실 곡선을 본다
> train 정확도 ↑, test 정확도 정체, **test 손실 증가**
- **300장 · 6층 (약 130,010개)** → train **1.000**, test **0.786** 에서 멈춤
- **test 손실이 오르는 이유** → 학습 데이터를 완벽히 맞히려 가중치가 커짐 → 틀리는 test 샘플에 더 확신 → CE 는 확신하며 틀릴수록 큰 벌점 (Day 2)
- **데이터 적음 + 모델 큼** → 외울 수 있는 능력은 크고 외울 양은 적다
- ⚠ 정확도만 보면 놓친다 — 과신은 **손실 곡선**에서 먼저 드러남
- ⚠ train 정확도만 보고 "잘 학습됐다" → 반드시 학습에 쓰지 않은 데이터로 평가
##### 자세히
```text
 accuracy               loss
 1.0 ┤ ─────── train    │        ╱ test   (rises: overconfident)
     │╱ ─ ─ ─  test     │ ╲    ╱
 0.8 ┤                  │  ╲__╱
     └───────► epoch    │     ╲______ train
                        └───────► epoch
```

#### 무엇이 과적합을 줄이나
> 데이터를 늘리는 것이 가장 확실 — 하지만 데이터는 비싸다
| 방법 | test | 비고 |
|---|---|---|
| 300장 · 6층 | 0.786 | 과적합 |
| 데이터 10배 (3000장) | **0.926** | 가장 확실 — 데이터는 비싸다 |
| 모델 줄이기 (1층 784-100-10) | 0.802 | 외울 능력 자체를 줄임 |
| 규제 (6교시) | 0.80 ~ 0.81 | 모델 그대로, 학습에 제약 |
- **3000장은 20 epoch** → iteration 수를 300장 · 200 epoch 와 맞추려고
- **조기 종료** → 검증 손실이 오르기 시작하는 epoch 에서 멈춤
- ⚠ test 를 보며 epoch · 모델을 고르면 test 가 "본 데이터"가 된다 → **validation** (7교시)

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

#### weight decay — 가중치에 벌점
> L' = L + ½λΣ‖W‖², ∂L'/∂W = ∂L/∂W + **λW**
- **갱신** → W ← W − η(∂L/∂W + λW) = **(1 − ηλ)W** − η∂L/∂W — 매 step 가중치를 조금씩 줄임(decay)
- **η = 0.1, λ = 0.01** → 매 step W × **0.999**
- **효과** → 가중치가 커지지 못함 → 과신 감소 → test 손실이 계속 **내려감**
- **W 에만** → 편향 · γ · β 에는 걸지 않는다
- ⚠ 손실에만 ½λΣW² 를 더하고 기울기에 λW 를 빼먹으면(또는 반대) → gradient check 로 잡힘
- ⚠ λ 가 너무 크면(0.1) 학습 데이터조차 못 맞힘 → **과소적합** (train 0.767, test 0.641)

#### Dropout — 무작위로 끄고, 추론 땐 평균
> 학습: mask = rand > ratio, x · mask · 추론: x · (1 − ratio) · 역전파: dout · mask
- **특정 뉴런에 기대지 못함** → 더 견고한 특징을 배운다
- **추론 때 (1 − ratio) 배** → 학습 때 평균 (1 − ratio) 만 켜져 있었으므로 → 출력 크기를 맞춤 (여러 망의 평균 = 앙상블)
- **역전파** → ReLU mask 와 같다 — 꺼진 뉴런으로는 기울기도 안 흐름
- **학습 속도** → 초반이 느림 (25 epoch train **0.53**) — 결국 test 정확도 상승
- ⚠ 추론 때 마스크 사용 금지 — `train_flg=False` 면 마스크 없이 (1 − ratio) 배
- ⚠ 출력층 뒤에 넣지 않기 — **은닉층 활성화 뒤에만**
##### 자세히
```text
 train (ratio 0.5)          inference
 ●──●  ●──●                 ●──●──●──●
  ╲ ╳  ╳ ╱   random off      all on, output × (1 − ratio)
 ●  ○  ●  ○  ○ = dropped     = average of many small nets
```

#### 300장 · 6층, 200 epoch 결과
> 규제는 test 정확도를 조금, test 손실을 크게 바꾼다
| 설정 | train | test | test 손실 |
|---|---|---|---|
| 규제 없음 | 1.000 | 0.786 | 1.12 (증가) |
| weight decay λ = 0.01 | 1.000 | **0.810** | **0.65** (감소) |
| Dropout 0.2 | 1.000 | 0.800 | 1.33 |
- **weight decay** → 과신을 막아 test 손실이 내려감
- **Dropout** → 정확도는 오르지만 손실 개선은 작음

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

#### 데이터 세 갈래
> test 에 맞춰 고른 순간 test 는 더 이상 "처음 보는 데이터"가 아니다
| 데이터 | 용도 | 몇 번 보나 |
|---|---|---|
| train | 파라미터(W, b) 학습 | 매 epoch |
| validation | 하이퍼파라미터 고르기 | 시도마다 |
| test | 최종 성능 보고 | **마지막에 한 번** |
- **최종** → 하이퍼파라미터가 정해지면 train + validation 을 합쳐 다시 학습 → test 로 한 번 평가
- ⚠ 탐색 중에 test 정확도를 보고 고르기 → **validation 만**

#### 로그 스케일 · 랜덤 탐색
> 0.001 과 0.002 의 차이보다 0.001 과 0.01 의 차이가 중요 → 지수를 균등하게
- **로그 스케일** → η = 10ᵘ, u ~ U(−4, 0) → 0.0001 ~ 1 을 자릿수마다 고르게 (각 자릿수에 약 25%)
- **랜덤 > 격자** → 둘 중 하나만 중요할 때 3 × 3 격자는 그 축을 **3개**만, 랜덤 9번은 **9개**를 본다
- **넓게 → 좁게** → 넓은 범위는 짧은 epoch 으로 걸러 내고, 좁힌 뒤 길게 — 2~3번 반복
- ⚠ `rng.uniform(1e-4, 1)` → 거의 모든 시도가 0.1 ~ 1 에 몰림 (90%) → **지수를** 뽑기
- ⚠ 처음부터 긴 epoch 으로 탐색 → 시간 낭비
##### 자세히
```text
 grid 3×3                 random 9
 important axis →         important axis →
 ●   ●   ●                 ●     ●   ●
 ●   ●   ●                   ●  ●      ●
 ●   ●   ●                ●    ●   ●
 3 distinct values         9 distinct values
```

#### 이 실험의 결과 읽기
> lr 이 결정적, λ 영향은 작다
- **lr 0.1 ~ 0.6** → val **0.84 ~ 0.885**
- **lr < 0.001** → 20 epoch 으로는 거의 학습 안 됨 (val 0.1 근처)
- **λ** → 1e-6 이든 1e-2 든 비슷 (이 범위에서)
- **최고** → val **0.885** (lr **0.242**)
- **다음 단계** → lr 을 10⁻¹·³ ~ 10⁻⁰·² 로 좁히고 epoch 을 늘려 λ 를 세밀하게

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

#### mynn.py 구조 — 프레임워크의 핵심
> 계층을 리스트로 쌓고, 파라미터를 한곳에 모으고, 학습 루프를 한 번만
| 구성 | 역할 |
|---|---|
| 계층 Affine · Relu · Sigmoid · BatchNorm · Dropout | 각자 `params` · `grads` dict, `forward(x, train_flg)`, `backward(dout)` (BN 은 `buffers` 도) |
| 옵티마이저 5종 | 2교시와 같음 — `update(params, grads)` |
| `Sequential(layers, weight_decay_lambda)` | `params()` · `grads()` · `predict()` · `loss()` · `gradient()` |
| `Trainer(model, optimizer, batch_size, seed)` | `fit(X, t, epochs, X_val, t_val)` — history 기록 |
| `save_params` · `load_params` | 파라미터 + BN running 통계를 .npz 로 저장 / 제자리 복사로 불러오기 |
- **`params()`** → `{"0.W": …, "1.gamma": …}` (계층 번호.이름) — 계층의 배열을 **복사하지 않고 그대로** 모음
- **같은 배열 객체** → 옵티마이저가 제자리 갱신하면 계층에 바로 반영 · params 는 루프 밖에서 한 번 모으면 됨
- **`grads()`** → 이름이 `"W"` 면 **λW** 를 더함 (weight decay)
- ⚠ `params()` 가 복사본을 주면 옵티마이저가 복사본만 바꿔 **학습이 안 됨**
- ⚠ `load_params` 가 새 배열을 대입하면 계층과 연결이 끊김 → **제자리 복사** (`p[...] = 저장값`)
##### 자세히
```text
 Sequential.params()  {"0.W", "0.b", "1.gamma", "1.beta", "3.W", …}
        │  same arrays (no copy)
        ▼
 layers [ Affine | BatchNorm | Relu | Affine | BatchNorm | Relu | … | Affine ]
          0        1           2      3        4           5          12
 optimizer.update(params, grads)  →  changes the layers directly
```

#### 5층 MLP — Day 3 과 비교
> 은닉층마다 Affine(He) → BatchNorm → Relu, 마지막 Affine — 13계층
| 항목 | Day 3 | Day 4 |
|---|---|---|
| 모델 | 784-100-10 | 784-100×4-10 + BN |
| 초기화 | std 0.01 | He |
| 옵티마이저 | SGD 0.1 | Adam 0.001 |
| 학습 | 10 epoch | **5** epoch |
| test | 0.9694 | **0.9757** |
| 파라미터 | 79,510 | **110,610** |
- **파라미터 110,610** → Affine 109,810 + BN γ · β 4층 × 200 = 800
- **기여한 기법** → He(신호 유지) · BN(안정 · 빠른 학습) · Adam(파라미터별 학습률) · 깊은 층(표현력)
- **save · load** → BN 의 running 평균 · 분산까지 저장해야 추론 결과가 같다 (추론은 running 을 씀)

#### 온디바이스 관점 — 학습 기법은 추론 비용을 늘리지 않는다
> 추론 때 모델은 여전히 행렬곱 + ReLU
- **Adam · He · weight decay** → 학습 때만 쓰임
- **Dropout** → 추론 땐 (1 − ratio) 곱하기 하나
- **BN** → 고정된 γ(x − μ)/σ + β = 선형 변환 → 앞 Affine 에 접어 넣기: W' = W·γ/√(σ² + ε), b' = (b − μ)·γ/√(σ² + ε) + β
- **BN folding** → **13계층 → 9계층**, 정확도 동일 — TensorRT 같은 추론 엔진이 실제로 하는 최적화

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

**연관 문제**
<<QA 8>>
