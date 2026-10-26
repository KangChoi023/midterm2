## Day 4 — 학습 기법

### 1교시 · 복습 퀴즈 · SGD 의 약점

**키워드**: `SGD`, `비등방성`, `지그재그`, `x²/20 + y²`, `update(params, grads)`

**개요**: SGD 는 W ← W − η∇L — 단순하지만 방향마다 경사가 크게 다른 지형(비등방성)에서는 기울기가 최소점을 향하지 않아 지그재그. 가장 가파른 방향이 학습률 상한을 정하고, 완만한 방향은 느리다. 해결: 관성(Momentum), 파라미터별 학습률(AdaGrad · RMSProp), 둘 다(Adam).

**핵심 용어**
- **SGD(확률적 경사 하강법)**: W ← W − η ∂L/∂W
- **비등방성(anisotropy)**: 손실 지형의 경사가 방향마다 크게 다른 것
- **옵티마이저(optimizer)**: 기울기로 파라미터를 갱신하는 규칙. 모두 `update(params, grads)` 로 제자리 갱신

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `SGD` | `SGD(lr=0.01).update(params, grads)` | `params[k] -= lr * grads[k]` |
| `run_optimizer()` | `run_optimizer(opt, steps=30, start=(-7, 2))` | 옵티마이저로 f 위의 경로를 기록 |

**핵심 내용**

#### 지그재그의 원인
> f(x, y) = x²/20 + y², ∇f = (x/10, 2y) — y 가 x 보다 20배 가파르다
- **기울기는 가파른 y 쪽을 주로 가리킴** → 최소점 (0, 0) 을 향하지 않는다
- **학습률 상한** → y 방향 때문에 η < 1 · **x 방향은 느림**

**연결**
- [NN D1·8교시 경사하강법](#/nn/1/8) — 같은 현상 (x² + 3y², 경계 1/3)

### 2교시 · 옵티마이저 구현

**키워드**: `Momentum`, `AdaGrad`, `RMSProp`, `Adam`, `편향 보정`, `상태는 첫 update 에서`

**개요**: 네 옵티마이저를 SGD 와 같은 update(params, grads) 형태로. 상태(속도 v, 누적 h, 모멘트 m)는 파라미터마다 필요해 첫 update 때 zeros_like 로 만든다. MNIST 5층에서 SGD 가 가장 느리고 나머지는 약 0.97. 실무 기본값은 Adam(lr 0.001).

**핵심 용어**
- **Momentum**: 속도 v 에 기울기를 누적해 관성으로 움직임 — 부호가 바뀌는 성분은 상쇄, 일관된 성분은 가속
- **AdaGrad**: 기울기 제곱을 누적(h)해 많이 움직인 파라미터의 학습률을 줄임
- **RMSProp**: h 를 지수 이동 평균으로 — 오래된 기울기는 잊는다
- **Adam**: Momentum(1차 모멘트 m) + RMSProp(2차 모멘트 v) + 편향 보정

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Momentum` | `Momentum(lr=0.01, momentum=0.9)` | v ← αv − ηg, W ← W + v |
| `AdaGrad` | `AdaGrad(lr=0.01)` | h ← h + g², W ← W − ηg/(√h + 1e-7) |
| `RMSProp` | `RMSProp(lr=0.01, decay=0.99)` | h ← ρh + (1 − ρ)g² |
| `Adam` | `Adam(lr=0.001, beta1=0.9, beta2=0.999)` | m · v 이동 평균 + 편향 보정 |
| `MultiLayerNet` | `MultiLayerNet(784, [100]*4, 10, weight_init_std="he", use_batchnorm=…, use_dropout=…, weight_decay_lambda=…)` | 오늘의 실험용 다층망 |

**핵심 내용**

#### 갱신식 한 장
| 옵티마이저 | 갱신식 (g = ∂L/∂W) | 직관 |
|---|---|---|
| SGD | W ← W − ηg | 기울기 반대로 |
| Momentum | v ← αv − ηg, W ← W + v (α = 0.9) | 공이 굴러가듯 관성 |
| AdaGrad | h ← h + g², W ← W − ηg / (√h + 1e-7) | 많이 움직인 파라미터는 천천히 (h 가 계속 커져 결국 멈춤) |
| RMSProp | h ← ρh + (1 − ρ)g² | 최근 기울기만 반영 |
| Adam | m ← β₁m + (1 − β₁)g, v ← β₂v + (1 − β₂)g², m̂ = m/(1 − β₁ᵗ), v̂ = v/(1 − β₂ᵗ), W ← W − η m̂ / (√v̂ + 1e-7) | 관성 + 파라미터별 학습률 |

#### MNIST 5층 (2000 iteration)
| SGD | Momentum | AdaGrad | RMSProp | Adam |
|---|---|---|---|---|
| 0.9378 | 0.9644 | 0.9726 | 0.9745 | 0.9740 |
- **실무** → Adam(lr ≈ 0.001) 기본, 마지막 성능을 짜낼 때 SGD + Momentum 재시도

#### 함정
- ⚠ 상태를 `__init__` 에서 만들면 shape 를 모름 → 첫 update 에서 `zeros_like`
- ⚠ `params[k] = params[k] - …` → 새 배열로 Affine 과 연결이 끊김 → `-=` · `+=`
- ⚠ Adam 의 `self.t += 1` 을 파라미터 for 안에 넣으면 t 가 파라미터 수만큼 커져 편향 보정이 틀어짐
- ⚠ 옵티마이저 객체를 여러 모델에 재사용 → 상태가 섞임 → 모델마다 새로

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

### 3교시 · 가중치 초기화

**키워드**: `대칭성`, `0 초기화 금지`, `활성화 분포`, `포화`, `Xavier √(1/n)`, `He √(2/n)`

**개요**: 같은 값으로 초기화하면 뉴런들이 같은 값 · 같은 기울기로 함께 움직여 100개가 사실상 1개(대칭). 너무 크면 sigmoid 가 포화(기울기 소실), 너무 작으면 신호가 사라진다. 입력 노드 수 n 에 맞춘 표준편차 — sigmoid 는 Xavier √(1/n), ReLU 는 He √(2/n).

**핵심 용어**
- **대칭성(symmetry)**: 같은 가중치로 시작한 뉴런들이 같은 기울기를 받아 끝까지 같아지는 현상
- **Xavier 초기화**: std = √(1/n_in) — sigmoid · tanh 용
- **He 초기화**: std = √(2/n_in) — ReLU 용 (음수 절반이 0 이 되므로 2배)

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `xavier_std()` | `np.sqrt(1.0 / n_in)` | Xavier 표준편차 |
| `he_std()` | `np.sqrt(2.0 / n_in)` | He 표준편차 |
| `activation_stats()` | `activation_stats(weight_init_std, act, n_layers=5)` | 층마다 활성화 분포 기록 |

**핵심 내용**

#### 같은 값 초기화 = 대칭
- **모두 0.01** → 은닉 뉴런이 같은 값 · 같은 기울기 → 100개가 1개
- **모두 0** → 더 나쁨: ReLU 출력 0 → 기울기 자체가 0
- **편향은 0 이어도 괜찮다** (가중치가 이미 다르므로)

#### 활성화 분포로 보는 좋은 초기화
> 출력 분산 ≈ n × Var(W) × Var(x) → Var(W) = 1/n 이면 층을 지나도 유지
| 문제 | 증상 | 결과 |
|---|---|---|
| 너무 큼 (sigmoid, std 1) | 활성화가 0 · 1 에 몰림 | 기울기 소실 |
| 너무 작음 (std 0.01) | 0.5 근처(sigmoid) · 0 근처(ReLU) | 표현력 제한 · 신호 소멸 |

#### MNIST 4층 (SGD 0.01, 2000 iteration)
| 초기화 | 0.01 | Xavier | He |
|---|---|---|---|
| test | 0.11 | 0.92 | 0.94 |
- ⚠ He 의 n 은 **입력 노드 수(이전 층 뉴런 수)** — 출력 노드 수 아님
- ⚠ sigmoid 망에 He, ReLU 망에 Xavier — 짝을 맞추기 (틀려도 학습은 되지만 느리거나 불안정)

**연결**
- [NN D2·4교시 활성화 함수](#/nn/2/4) — 기울기 소실
- [NN D1·7교시 확률 기초](#/nn/1/7) — 정규분포 난수

### 4교시 · Batch Normalization

**키워드**: `BatchNorm`, `평균 0 · 분산 1`, `γ · β`, `running 평균`, `train_flg`, `Affine → BN → ReLU`

**개요**: 초기화는 출발점에서만 분포를 맞춘다. BN 은 매 미니배치마다 층 출력의 열(특성)별로 평균 0 · 분산 1 로 맞춘 뒤 γ · β 로 늘이고 옮긴다. 학습 땐 배치 통계, 추론 땐 학습 중 모아 둔 running 평균. 효과: 초기화에 둔감, 빠르고 안정적인 학습.

**핵심 용어**
- **배치 정규화(Batch Normalization, BN)**: 미니배치의 열마다 (x − μ) / √(σ² + ε) 로 정규화하고 y = γx̂ + β
- **γ(scale) · β(shift)**: 정규화 후 다시 늘이고 옮기는 학습 파라미터 (처음 1, 0)
- **running 평균 · 분산**: 학습 중 배치 통계의 이동 평균 — 추론 때 사용

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `BatchNorm` | `BatchNorm(gamma, beta, momentum=0.9, eps=1e-7)` · `forward(x, train_flg)` · `backward(dout)` | BN 계층 |

**핵심 내용**

#### 정규화 식
> μ = mean(x, axis=0), σ² = var(x, axis=0), x̂ = (x − μ) / √(σ² + ε), y = γx̂ + β
- **"평균 0 · 분산 1 이 최선이 아니면"** → 네트워크가 γ · β 로 스스로 조정
- **추론** → 미니배치가 없을 수도(한 장) → running 평균 · 분산으로
- **running 갱신** → `running = 0.9 · running + 0.1 · 배치값` — 학습 모드에서만

#### backward 요점
- **dβ = Σ dout**, **dγ = Σ x̂ · dout** (행 방향 합)
- **dx 의 열 합 = 0** → BN 바로 앞 Affine 의 편향 기울기는 0 (상대 오차가 커 보여도 버그 아님)

#### BN 의 효과 (1000장, 20 epoch)
- **init std 1.0 · BN 없음** → 발산(overflow) · **BN 있음** → 학습됨
- **std 0.01 처럼 작아도** → BN 이 있으면 학습됨 — 초기화에 둔감

#### 함정
- ⚠ 추론 때 `train_flg=True` → 배치 구성에 따라 결과가 바뀌고, 한 장이면 출력 0 → predict 는 `train_flg=False`
- ⚠ `x.mean()` (축 없음) → 전체 평균 하나 → 특성마다 `axis=0`
- ⚠ γ, β 를 params 에 안 넣으면 학습 안 됨

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

### 5교시 · 과적합

**키워드**: `과적합`, `train ↑ test 정체`, `test 손실 증가`, `과신`, `데이터 늘리기`, `모델 줄이기`

**개요**: 과적합은 학습 데이터를 외워 처음 보는 데이터에서 성능이 떨어지는 현상 — 데이터가 적고 모델이 클수록. MNIST 300장으로 6층(약 13만 파라미터)을 학습하면 train 1.0, test 0.786 에서 멈추고 test 손실은 계속 오른다(확신하며 틀림).

**핵심 용어**
- **과적합(overfitting)**: train 성능은 오르는데 test 성능이 멈추거나 떨어지는 현상
- **과소적합(underfitting)**: 학습 데이터조차 잘 못 맞히는 상태

**핵심 내용**

#### 과적합의 신호
> train 정확도 ↑, test 정확도 정체, **test 손실 증가**
- **test 손실이 오르는 이유** → 가중치가 커지며 틀리는 샘플에 더 확신 → CE 는 확신하며 틀릴수록 큰 벌점
- ⚠ 정확도만 보면 놓친다 — 과신은 손실 곡선에서 먼저 드러남

#### 무엇이 과적합을 줄이나
| 방법 | test | 비고 |
|---|---|---|
| 300장 · 6층 | 0.786 | 과적합 |
| 데이터 10배 (3000장) | 0.926 | 가장 확실 — 하지만 데이터는 비싸다 |
| 모델 줄이기 (1층) | 0.802 | 외울 능력 자체를 줄임 |
| 규제 (6교시) | 0.80 ~ 0.81 | 모델 그대로, 학습에 제약 |
- ⚠ test 를 보며 epoch · 모델을 고르면 test 가 "본 데이터"가 된다 → validation (7교시)

### 6교시 · 규제: weight decay · Dropout

**키워드**: `규제`, `weight decay`, `L2`, `½λΣW²`, `+ λW`, `Dropout`, `mask`, `(1 − ratio)`, `앙상블`

**개요**: 모델 크기는 그대로 두고 학습에 제약을 거는 것이 규제. weight decay 는 손실에 ½λΣ‖W‖² 벌점(기울기 + λW)을 더해 가중치를 작게 → 과신 감소. Dropout 은 학습 때 은닉 뉴런을 무작위로 꺼 매번 다른 작은 망을 학습, 추론 땐 모두 켜고 (1 − ratio) 배.

**핵심 용어**
- **규제(regularization)**: 모델은 그대로 두고 학습에 제약을 걸어 과적합을 줄이는 방법
- **weight decay(L2 규제)**: 손실에 ½λΣ‖W‖² 를 더해 가중치가 커지지 못하게 함
- **Dropout**: 학습 때 은닉 뉴런을 무작위로 끄는 기법 — 앙상블 효과

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `l2_penalty()` | `0.5 * lam * sum(np.sum(W ** 2) for W in 가중치들)` | L2 벌점 (편향 · γ · β 제외) |
| `Dropout` | `Dropout(dropout_ratio=0.5)` · `forward(x, train_flg)` · `backward(dout)` | 드롭아웃 계층 |

**핵심 내용**

#### weight decay
> L' = L + ½λΣ‖W‖², ∂L'/∂W = ∂L/∂W + λW
- **갱신** → W ← (1 − ηλ)W − η∂L/∂W — 매 step 가중치를 (1 − ηλ) 배로 조금씩 줄임
- **W 에만** → 편향 · γ · β 에는 걸지 않는다
- ⚠ 손실에만 더하고 기울기에 λW 를 빼먹으면(또는 반대) → gradient check 로 잡힘
- ⚠ λ 가 너무 크면(0.1) 과소적합

#### Dropout
> 학습: mask = rand > ratio, x · mask · 추론: x · (1 − ratio) · 역전파: dout · mask
- **특정 뉴런에 기대지 못함** → 견고한 특징
- **추론 때 (1 − ratio) 배** → 학습 때 평균 (1 − ratio) 만 켜져 있었으므로 (여러 망의 평균 = 앙상블)
- **역전파** → ReLU mask 와 같다 — 꺼진 뉴런으로는 기울기도 안 흐름
- ⚠ 추론 때 마스크 사용 금지 · 출력층 뒤에 넣지 않기 (은닉층 활성화 뒤에만)

#### 300장 · 6층, 200 epoch
| | train | test | test 손실 |
|---|---|---|---|
| 규제 없음 | 1.000 | 0.786 | 1.12 (증가) |
| weight decay λ = 0.01 | 1.000 | 0.810 | 0.65 (감소) |
| Dropout 0.2 | 1.000 | 0.800 | 1.33 |
- **Dropout** → 초반 학습이 느림(25 epoch train 0.53) · 결국 test 정확도 상승

**기본 사용형태**
```python
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

### 7교시 · 하이퍼파라미터 탐색

**키워드**: `하이퍼파라미터`, `train / validation / test`, `로그 스케일`, `랜덤 탐색`, `넓게 → 좁게`

**개요**: 학습률 · λ · 은닉층 크기처럼 사람이 정하는 값이 하이퍼파라미터. test 로 고르면 성능을 부풀리게 되므로 validation 으로 고른다. 지수를 균등하게 뽑는 로그 스케일 + 격자보다 랜덤 탐색, "넓게 짧게 → 좁게 길게".

**핵심 용어**
- **하이퍼파라미터(hyperparameter)**: 학습으로 찾지 않고 사람이 정하는 값 — 학습률, λ, 층 크기, 배치 크기
- **검증 데이터(validation)**: 하이퍼파라미터를 고르는 데 쓰는, 학습에 쓰지 않은 데이터

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `log_uniform()` | `10 ** rng.uniform(low, high)` | 지수를 균등하게 뽑기 — 0.0001 ~ 1 을 자릿수마다 고르게 |

**핵심 내용**

#### 데이터 세 갈래
| 데이터 | 용도 | 몇 번 보나 |
|---|---|---|
| train | 파라미터(W, b) 학습 | 매 epoch |
| validation | 하이퍼파라미터 고르기 | 시도마다 |
| test | 최종 성능 보고 | 마지막에 한 번 |

#### 탐색 방법
- **로그 스케일** → η = 10ᵘ, u ~ U(−4, 0) — 0.001 vs 0.01 차이가 중요
- **랜덤 > 격자** → 하나만 중요할 때 3×3 격자는 그 축을 3개만, 랜덤 9번은 9개를 본다
- **결과(30번, 20 epoch)** → lr 0.1 ~ 0.6 이 val 0.84 ~ 0.885, lr < 0.001 은 거의 학습 안 됨, λ 영향은 작음 → 최고 val 0.885 (lr 0.242)
- ⚠ `rng.uniform(1e-4, 1)` → 거의 모두 0.1 ~ 1 에 몰림 → 지수를 뽑기
- ⚠ 처음부터 긴 epoch 금지 — 넓은 범위는 짧게 걸러 내고 좁힌 뒤 길게

### 8교시 · 미니 프로젝트: 미니 프레임워크 mynn.py

**키워드**: `mynn.py`, `Layer`, `Sequential`, `params()`, `Trainer`, `save · load`, `BN folding`

**개요**: Day 3 · 4 의 부품을 재사용 가능한 모듈 하나로 — 계층을 리스트로 쌓고, 파라미터를 한곳에 모으고, 학습 루프를 한 번만. PyTorch · Keras 가 하는 일의 핵심. 784-100×4-10 + BN + He + Adam 으로 5 epoch 만에 test 0.975 를 넘는다.

**핵심 용어**
- **Sequential**: 계층 리스트를 순서대로 forward · 반대로 backward 하는 모델 컨테이너
- **BN folding**: 추론 때 고정된 BN 을 앞의 Affine 에 합쳐 계층 수를 줄이는 최적화 (TensorRT 등)

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Sequential` | `Sequential([Affine(784, 100, "he", 1), BatchNorm(100), Relu(), …], weight_decay_lambda)` | params() · grads() · predict() · loss() · gradient() |
| `Trainer` | `Trainer(model, Adam(0.001), batch_size=100).fit(X, t, epochs, X_val, t_val)` | 셔플 → 배치 → gradient → update |

**핵심 내용**

#### mynn.py 구조
- **계층** → 각자 `params` · `grads` dict, `forward(x, train_flg)`, `backward(dout)` (BN 은 running 통계를 `buffers` 에)
- **`Sequential.params()`** → `{"0.W": …, "1.gamma": …}` — 계층의 배열을 **복사하지 않고 그대로** 모음 → 옵티마이저가 제자리 갱신하면 계층에 바로 반영
- **`grads()`** → 이름이 "W" 면 λW 를 더함 (weight decay)
- ⚠ params() 가 복사본을 주면 옵티마이저가 복사본만 바꿔 학습이 안 됨

#### 5층 MLP
- **구성** → 은닉층마다 Affine(He) → BatchNorm → Relu, 마지막 Affine
- **save · load** → BN 의 running 평균 · 분산까지 저장해야 추론 결과가 같다

#### Day 3 vs Day 4
| | Day 3 | Day 4 |
|---|---|---|
| 모델 | 784-100-10 | 784-100×4-10 + BN |
| 초기화 | std 0.01 | He |
| 옵티마이저 | SGD 0.1 | Adam 0.001 |
| 학습 | 10 epoch | 5 epoch |
| test | 0.9694 | 0.9757 |
| 파라미터 | 79,510 | 110,610 |

#### 온디바이스 관점
> 학습 기법은 추론 비용을 늘리지 않는다
- **Adam · He · weight decay** → 학습 때만
- **Dropout** → 추론 땐 (1 − ratio) 곱하기 · **BN** → 고정된 선형 변환 → 앞 Affine 에 접어 넣기(BN folding)

**연결**
- [NN D5·4교시 Convolution · Pooling 계층](#/nn/5/4) — mynn.py 에 CNN 계층 추가
