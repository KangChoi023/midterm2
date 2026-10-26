## Day 1 — Python · NumPy와 AI 수학

### 1교시 · 과정 안내 · Colab 사용법

**키워드**: `AI ⊃ 머신러닝 ⊃ 딥러닝`, `학습 = 최적화`, `argmin`, `학습의 고리`, `파라미터`

**개요**: 학습이란 손실(Loss)을 가장 작게 만드는 파라미터를 찾는 최적화 — θ* = argmin L(θ). 5일 동안 이 한 문장을 직접 코드로 만든다. 신경망이 찾는 "규칙"은 결국 숫자(파라미터)의 모음.

**핵심 용어**
- **머신러닝(machine learning)**: 규칙을 사람이 쓰지 않고 입력 · 정답 쌍(데이터)에서 찾게 하는 방법
- **딥러닝(deep learning)**: 층(layer)을 여러 겹 쌓은 신경망으로 학습하는 머신러닝
- **파라미터(parameter, θ)**: 학습으로 찾는 숫자 — 가중치 W 와 편향 b
- **손실(loss, L)**: 예측과 정답의 차이를 숫자 하나로 나타낸 값. 작을수록 좋다

**핵심 내용**

#### 학습 = 최적화
> θ* = argmin_θ L(θ) — 손실을 최소로 만드는 파라미터 찾기
- **전통 프로그래밍** → 규칙(F = 1.8C + 32)을 사람이 코드로
- **머신러닝** → 입력 · 정답 쌍을 주면 컴퓨터가 규칙의 숫자(w, b)를 찾음

#### 학습의 고리 — 5일의 지도
> 모델 → 손실 → 기울기 → 갱신 → 반복
| 단계 | 하는 일 | 만드는 날 |
|---|---|---|
| 모델 y = f(x; θ) | 입력에 파라미터를 곱하고 더해 예측 | Day 2 순전파 |
| 손실 L | 예측과 정답의 차이 | Day 2 MSE · Cross-Entropy |
| 기울기 ∂L/∂θ | 어느 쪽으로 바꾸면 손실이 줄어드나 | Day 1 미분 · Day 3 역전파 |
| 갱신 θ ← θ − η∇L | 기울기 반대로 조금 이동 | Day 1 경사하강 · Day 4 옵티마이저 |
- **Day 5** → 합성곱 · 풀링으로 CNN

#### Colab 실수 방지
- **원본에서 작업** → 저장 안 됨 → 파일 › Drive 에 사본 저장
- **런타임 끊김 후 NameError** → 맨 위 환경 셀부터 다시
- **`%timeit 식`** → 여러 번 돌려 평균 실행 시간

**연결**
- [Python D1·1교시 Colab 시작하기](#/py/1/1) — 셀 실행 순서 · `!` · `%`

### 2교시 · Python 속성 (C 개발자용)

**키워드**: `이름표`, `동적 타입`, `is vs ==`, `가변 · 불변`, `참조의 배열`, `.copy()`

**개요**: C 의 `int a = 3;` 은 값이 든 메모리 칸, Python 의 `a = 3` 은 heap 의 int 객체에 붙인 이름표. 타입은 객체에 있다(동적 타입). 리스트는 값이 아니라 참조의 배열이라 숫자 계산이 느리다 → NumPy 가 필요한 이유.

**핵심 용어**
- **가변 객체(mutable)**: 제자리에서 바뀌는 객체 — list · dict · ndarray. `b = a` 로 공유된다
- **불변 객체(immutable)**: 바뀌지 않는 객체 — int · float · str · tuple. `y = y + 1` 은 새 객체를 만들어 이름표를 옮긴다

**핵심 내용**

#### C ↔ Python 대응
| C | Python | 핵심 차이 |
|---|---|---|
| `int a = 3;` | `a = 3` | 이름표, 타입은 객체에 |
| `int arr[5];` | `arr = [0] * 5` | 길이 가변, 타입 섞임 |
| `struct` | `class` | 메서드를 함께, `self` = C 의 this |
| `{ }` | 들여쓰기 | 들여쓰기 자체가 문법 |
| 포인터 | 참조 | `b = a` 는 같은 객체 |

#### 같은 객체인가 vs 같은 값인가
> `is` = 같은 객체(id 비교), `==` = 같은 값
- **`b = a`** → `a is b` True — b.append 하면 a 도 바뀜
- **`c = a.copy()`** → `a is c` False, `a == c` True
- **`2 ** 100`** → Python int 는 크기 제한 없음 (C 의 overflow 없음)

#### C 개발자가 자주 하는 실수
- ⚠ `b = a` 를 복사로 착각 → 리스트 · ndarray 는 `.copy()`
- ⚠ `7 / 2` = 3.5 (항상 실수) · 정수 몫은 `7 // 2` = 3
- ⚠ `a[-1]` 은 마지막 원소(오류 아님) · 탭과 스페이스를 섞으면 IndentationError

##### 자세히
- 리스트에 숫자 하나 더할 때마다 "참조 따라가기 → 타입 확인 → 덧셈 → 새 객체 만들기" 를 반복 → 느림. ndarray 는 C 배열처럼 값을 연속으로 둬 이 과정을 없앤다

**연결**
- [C D2·1교시 포인터 기초](#/c/2/1) — 참조 = 주소를 가리키는 것
- [Python D2·1교시 리스트와 튜플 — 참조와 복사](#/py/2/1)

### 3교시 · NumPy ndarray

**키워드**: `ndarray`, `ndim · shape · size · dtype`, `strides`, `view vs copy`, `axis`, `keepdims`, `브로드캐스팅`

**개요**: ndarray = 같은 dtype 숫자가 연속 메모리에 놓인 다차원 배열 + shape · strides 라는 "읽는 규칙". 신경망의 입력 · 가중치 · 출력은 전부 ndarray. axis 는 없어지는 축, 브로드캐스팅은 오른쪽 축부터 같거나 1 이면 결합.

**핵심 용어**
- **ndarray**: NumPy 다차원 배열. 같은 dtype 의 값이 연속 메모리에
- **shape**: 각 축의 크기 튜플 — `np.zeros((3, 4))` 는 `(3, 4)`
- **strides**: 각 축으로 한 칸 움직일 때 건너뛰는 바이트 수 — int64 (3, 4) 는 `(32, 8)`
- **view**: 메모리를 공유하는 다른 모양의 배열. 슬라이스 · reshape 결과
- **axis**: 집계할 때 **없어지는 축**. `sum(axis=0)` 은 axis 0 이 사라져 열마다 합
- **브로드캐스팅(broadcasting)**: shape 를 오른쪽 끝부터 비교해 같거나 한쪽이 1 이면 1 인 쪽을 늘려 연산하는 규칙

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `np.array()` | `np.array([1, 2, 3], dtype=np.float32)` | 리스트 → ndarray |
| `np.zeros()` | `np.zeros((3, 4))` | 0 으로 채운 배열 (`np.ones` 는 1) |
| `np.arange()` | `np.arange(0, 10, 2)` | 범위 배열 (stop 미포함) |
| `np.linspace()` | `np.linspace(0, 1, 5)` | 구간을 n 등분한 점 (끝 포함) |
| `ndarray.reshape()` | `A.reshape(4, 3)` · `A.reshape(-1)` | 모양 바꾸기 — 대부분 view |
| `np.random.default_rng()` | `rng = np.random.default_rng(42)` | seed 고정 난수 생성기 |

**핵심 내용**

#### ndarray 의 정체 — 한 줄 메모리 + 읽는 규칙
> `A[i, j]` 의 위치 = i × strides[0] + j × strides[1] 바이트 (C 2차원 배열 주소 계산과 같다)
- **`np.arange(12).reshape(3, 4)`** → 12개를 한 줄로 놓고 shape (3, 4) · strides (32, 8)
- **reshape · 슬라이스** → view — `B = A.reshape(4, 3); B[0,0] = 100` 하면 `A[0,0]` 도 100
- **독립 사본** → `.copy()` · 공유 확인 `np.shares_memory(A, B)`

#### 인덱싱 · 슬라이싱
- `A[1, 2]` 한 칸 · `A[0]` 0행 · `A[:, 1]` 1열 · `A[1:, 2:]` 부분 행렬
- **`A[A > 6]`** → 불리언 마스크 (Day 3 ReLU 역전파에서 사용)

#### axis — 없어지는 축
> (3, 4) 에서 `sum(axis=0)` → (4,) 열마다, `sum(axis=1)` → (3,) 행마다
- **`keepdims=True`** → 크기 1 로 남겨 `(3, 1)` — 다시 브로드캐스팅할 때
- **신경망** → 배치 축(axis 0)으로 평균을 내는 일이 많다 (손실 평균, BatchNorm)

#### 브로드캐스팅
> 오른쪽 축부터: 같거나 한쪽이 1 → 1 인 쪽이 늘어난다
| 식 | 결과 |
|---|---|
| (3, 4) + (4,) | (3, 4) — 모든 행에 더함 |
| (3, 1) + (1, 4) | (3, 4) |
| (3, 4) + (3,) | ❌ 4 와 3 이 다름 |
- **신경망 `X @ W + b`** → 편향 b(H,) 가 모든 샘플(행)에 더해지는 것이 브로드캐스팅
- ⚠ `np.array([1, 2, 3])` 은 int64 — 0.5 를 대입하면 0 으로 잘림 → `dtype=float`
- ⚠ (3,) 와 (3, 1) 은 다른 shape — 1차원 벡터는 행도 열도 아니다

**기본 사용형태**
```python
import numpy as np
rng = np.random.default_rng(42)
A = np.arange(12).reshape(3, 4)      # shape (3, 4)
A.sum(axis=0)                        # (4,)  열마다
A.mean(axis=1, keepdims=True)        # (3, 1)
A + np.array([10, 20, 30, 40])       # (3, 4) + (4,) 브로드캐스팅
B = A[1:, 2:].copy()                 # 독립 사본
```

**연결**
- [C D1·4교시 자료형과 메모리 표현](#/c/1/4) — 연속 메모리와 주소 계산
- [Python D4·1교시 NumPy 기초](#/py/4/1) — 처음 배운 ndarray

### 4교시 · 벡터화

**키워드**: `벡터화`, `BLAS`, `@`, `행 · 열 내적`, `(N, D) @ (D, H)`, `원소별 곱 *`

**개요**: 벡터화 = Python for 루프 대신 NumPy 연산 한 번으로 배열 전체를 처리. 내부가 C · BLAS 라 수백 배 빠르다. 행렬곱 C = A @ B 의 한 칸은 A 의 행과 B 의 열의 내적.

**핵심 용어**
- **벡터화(vectorization)**: 반복문을 NumPy 내부(C · BLAS)로 넘겨 배열 전체를 한 번에 계산하는 것
- **행렬곱(matrix multiplication)**: C[i][j] = Σ_k A[i][k] · B[k][j]. A 의 열 수 = B 의 행 수여야 한다

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `A @ B` | `C = A @ B` | 행렬곱 (`np.dot(A, B)` 와 같음) |
| `np.sum()` | `A.sum(axis=0)` | 합계 (axis 로 기준 축 지정) |
| `np.mean()` | `A.mean(axis=1, keepdims=True)` | 평균 |
| `np.argmax()` | `scores.sum(axis=1).argmax()` | 최댓값의 인덱스 |

**핵심 내용**

#### 행렬곱 한 칸 = 행 · 열 내적
> (N, D) @ (D, H) → (N, H) — 안쪽 D 가 같아야 하고 그 축은 사라진다
- **3중 for 루프** → `C[i][j] += A[i][k] * B[k][j]`
- **200×200** → for 루프 수 초 vs `@` 수 ms — 수백~수천 배

#### 왜 빠른가
| 순수 Python for | NumPy `@` |
|---|---|
| 곱셈마다 인터프리터가 해석 | 컴파일된 BLAS 함수 한 번 호출 |
| 원소마다 타입 확인 · 객체 생성 | 같은 dtype 이라 확인 불필요 |
| 참조 따라 메모리 여기저기 | 연속 메모리를 캐시 · SIMD 로 |

#### 함정
- ⚠ `A * B` 는 **원소별 곱**, 행렬곱은 `A @ B` — shape 가 같으면 `*` 도 오류 없이 돌아 버그를 찾기 어렵다
- ⚠ `matmul: … mismatch in its core dimension` → 안쪽 차원이 다르다 → shape 부터 출력

**기본 사용형태**
```python
scores = np.array([[90, 70, 80], [60, 95, 85], [75, 80, 70], [88, 92, 95], [50, 65, 60]])
scores.mean(axis=0)                            # 과목별 평균 (3,)
scores.sum(axis=1)                             # 학생별 총점 (5,)
scores.sum(axis=1).argmax()                    # 1등 학생 번호 → 3
scores - scores.mean(axis=1, keepdims=True)    # 학생별 평균을 뺀 점수
```

### 5교시 · 선형대수

**키워드**: `내적`, `노름`, `cos θ`, `선형 변환`, `회전 행렬`, `X @ W + b`

**개요**: 내적은 두 벡터가 얼마나 같은 방향인가를 재고, 뉴런의 가중합 w · x 도 내적. 행렬곱은 공간을 회전 · 확대 · 찌그러뜨리는 선형 변환 — 신경망 한 층 XW 는 데이터를 새 공간으로 옮긴다.

**핵심 용어**
- **내적(dot product)**: a · b = Σ aᵢbᵢ = ‖a‖‖b‖cos θ. 같은 방향이면 크고, 직각이면 0, 반대면 음수
- **노름(norm)**: 벡터의 길이 ‖a‖ = √(a · a)
- **선형 변환(linear transformation)**: 행렬곱으로 공간을 바꾸는 것. 행렬의 열 = 기준 벡터 e₁, e₂ 가 옮겨 가는 곳

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `np.dot()` | `np.dot(a, b)` | 내적 (1차원끼리는 스칼라) |
| `np.linalg.norm()` | `np.linalg.norm(a)` | 벡터 길이 |
| `ndarray.T` | `P @ R.T` | 전치 (행 ↔ 열) |

**핵심 내용**

#### 내적 = 닮음 측정기
> 뉴런 하나 = 입력 x 가 가중치 패턴 w 와 얼마나 닮았는지 재는 장치
- `a = [1, 2, 2]`, `b = [2, 0, 1]` → a · b = 4, ‖a‖ = 3

#### 행렬곱 = 선형 변환
- **회전** → R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]] · **확대** → S = [[sx, 0], [0, sy]]
- **점이 행으로 쌓인 P (N, 2)** → 변환은 `P @ M.T`
- ⚠ `M @ P` 는 shape 가 안 맞는다 · (3,) 끼리 `@` 는 스칼라 — 행렬이 필요하면 `reshape(1, 3)`

#### 신경망 한 층 미리 보기
> X (N, D) @ W (D, H) + b (H,) = Y (N, H)
- **입력 4샘플 × 특징 3 → 출력 2** → W (3, 2), b (2,), Y (4, 2)

**기본 사용형태**
```python
X = rng.normal(size=(4, 3))   # 샘플 4 × 특징 3
W = rng.normal(size=(3, 2))
b = np.zeros(2)
Y = X @ W + b                 # (4, 2)
```

**연결**
- [NN D2·5교시 행렬로 보는 순전파](#/nn/2/5) — 이 한 층을 쌓는다

### 6교시 · 미분과 기울기

**키워드**: `순간 변화율`, `중앙 차분`, `h = 1e-4`, `편미분`, `기울기(gradient)`, `−∇f`, `numerical_gradient`

**개요**: 미분 = h → 0 의 순간 변화율, 컴퓨터에서는 중앙 차분 (f(x+h) − f(x−h)) / 2h. 편미분을 모은 벡터가 기울기 ∇f 이고, 함수값이 가장 빨리 커지는 방향 — 그 반대(−∇f)로 가면 가장 빨리 작아진다(경사하강의 원리).

**핵심 용어**
- **수치 미분(numerical differentiation)**: 아주 작은 h 로 미분을 근사 — 중앙 차분이 오차가 작다
- **편미분(partial derivative)**: 변수 하나만 움직이고 나머지는 고정한 미분
- **기울기(gradient, ∇f)**: 편미분을 모은 벡터. 함수값이 가장 빨리 커지는 방향

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `numerical_diff()` | `numerical_diff(f, x, h=1e-4)` | 중앙 차분으로 f'(x) 근사 |
| `numerical_gradient()` | `numerical_gradient(f, x, h=1e-4)` | 임의 shape 배열 x 에 대한 기울기 — Day 3 gradient check 의 정답지 |

**핵심 내용**

#### 수치 미분
> f'(x) ≈ (f(x+h) − f(x−h)) / 2h, h ≈ 1e-4
- **(x²)' at 3** → 6 · **sin'(0)** → 1
- ⚠ h 를 1e-15 처럼 너무 작게 하면 부동소수점 반올림 오차가 커진다 (`np.float32(1e-50)` = 0.0)

#### 기울기 = 가장 가파른 오르막
> f = x₀² + x₁² → ∇f = (2x₀, 2x₁) — −∇f 화살표가 모두 최소점 (0, 0) 을 향한다

#### numerical_gradient 의 동작
- **원소마다** → `x[idx] = orig + h` → f, `x[idx] = orig - h` → f, `(fxh1 - fxh2) / (2*h)`, **`x[idx] = orig` 로 되돌리기**
- ⚠ **정수 배열**이면 `orig + h` 가 정수로 잘려 기울기 0 → `np.array([3.0, 4.0])`
- ⚠ 되돌리기를 빼먹으면 다음 원소의 기울기가 틀린다
- ⚠ 기울기 방향으로 **더하면** 손실이 커진다 — 경사하강은 **빼기**

**기본 사용형태**
```python
def numerical_diff(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient(f, x, h=1e-4):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + h; fxh1 = f(x)
        x[idx] = orig - h; fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = orig                       # 되돌리기!
        it.iternext()
    return grad
```

**연결**
- [NN D3·1교시 수치 미분의 한계](#/nn/3/1) — 파라미터가 많으면 왜 느린가
- [NN D3·6교시 계층 조립과 Gradient check](#/nn/3/6) — 이 함수로 역전파 검증

### 7교시 · 확률 기초

**키워드**: `평균 · 분산 · 표준편차`, `균등분포`, `정규분포`, `seed`, `표준화`, `log(0)`, `exp overflow`

**개요**: 신경망은 난수로 가중치를 초기화하고(Day 4) 확률로 답을 낸다(Day 2 Softmax). 균등 · 정규분포 난수, seed 로 재현성, 표준화, 그리고 log · exp 의 함정.

**핵심 용어**
- **균등분포(uniform) U(a, b)**: 구간 안 모든 값이 같은 확률 — 평평한 히스토그램 (Xavier uniform 초기화)
- **정규분포(normal) N(μ, σ²)**: 평균 근처가 가장 많은 종 모양. ±1σ 약 68%, ±2σ 약 95% (He normal 초기화)
- **표준화(standardization)**: (x − μ) / σ → 평균 0, 표준편차 1

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `rng.uniform()` | `rng.uniform(-1, 1, 10_000)` | 균등분포 난수 |
| `rng.normal()` | `rng.normal(0, 1, 10_000)` | 정규분포 난수 |
| `np.std()` | `x.std()` · `x.std(ddof=1)` | 표준편차 (모집단 N · 표본 N−1) |
| `np.log()` | `np.log(p + 1e-7)` | 자연로그 — 확률 → 손실 |
| `np.exp()` | `np.exp(x)` | 지수 — 점수 → 확률 |

**핵심 내용**

#### 분포와 재현성
- **seed 고정** → `rng = np.random.default_rng(42)` — 안 하면 실행마다 결과가 달라 비교 불가
- **`x.std()`** → N 으로 나눔(모집단), 표본은 `ddof=1`

#### log 와 exp — Day 2 예고편
> 정답 확률이 작을수록 −log p 가 커진다 → 손실
- ⚠ `np.log(0)` = −inf → 이후 전부 nan → 작은 eps(1e-7)를 더한다
- ⚠ `np.exp(1000)` = inf (overflow) → Day 2 Softmax 에서 최댓값을 빼서 해결

**연결**
- [NN D2·6교시 출력층과 손실 함수](#/nn/2/6) — Softmax · Cross-Entropy
- [NN D4·3교시 가중치 초기화](#/nn/4/3) — 균등 · 정규 난수의 쓰임

### 8교시 · 미니 프로젝트: 경사하강법으로 최소점 찾기

**키워드**: `경사하강`, `p ← p − η∇f`, `학습률 η`, `수렴 · 진동 · 발산`, `(1 − 6η)`

**개요**: f(x, y) = x² + 3y² 의 최소점 (0, 0) 을 경사하강으로 찾는다. 학습률이 작으면 느리고, 적당하면 수렴, 크면 진동 · 발산 — 가장 가파른 방향이 학습률의 상한을 정한다.

**핵심 용어**
- **경사하강법(gradient descent)**: p ← p − η∇f(p) 를 반복해 함수의 최소점을 찾는 방법
- **학습률(learning rate, η)**: 한 번에 기울기 반대 방향으로 이동하는 크기

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `gradient_descent()` | `gradient_descent(grad, init, lr, steps)` | init 에서 steps 번 갱신, 출발점 포함 경로 (steps+1, 2) 반환. 노름 > 1e6 이면 발산으로 멈춤 |

**핵심 내용**

#### 한 걸음 계산
> ∇f = (2x, 6y)
- **p₀ = (−4, 2), η = 0.1** → ∇f = (−8, 12) → p₁ = (−4, 2) − (−0.8, 1.2) = (−3.2, 0.8)
- **y 가 먼저 빨리 줄어듦** → y 방향(3y²)이 더 가파르다
- **`gradient_descent()`** → 이 걸음을 steps 번 반복하며 경로를 모아 등고선 위에 그린다

#### 학습률과 수렴 조건
> 매 스텝 y ← (1 − 6η) y, x ← (1 − 2η) x — 곱하는 수의 절댓값이 1 보다 작아야 수렴
| η | y 에 곱하는 수 1 − 6η | 결과 |
|---|---|---|
| 0.01 | 0.94 | 수렴하지만 느림 (30스텝에 못 닿음) |
| 0.1 | 0.4 | 빠르게 수렴 |
| 0.3 | −0.8 | 부호가 바뀌며 지그재그로 수렴 |
| 0.35 | −1.1 | 발산 |
- **경계** → |1 − 6η| < 1 → **0 < η < 1/3**
- **실제 신경망** → 방향이 수백만 개라 손으로 못 푼다 → Day 4 Momentum · Adam 이 방향마다 걸음 크기를 자동 조절

**기본 사용형태**
```python
f      = lambda p: p[0] ** 2 + 3 * p[1] ** 2
grad_f = lambda p: np.array([2 * p[0], 6 * p[1]])

def gradient_descent(grad, init, lr, steps):
    p = np.array(init, dtype=float)
    history = [p.copy()]
    for _ in range(steps):
        p = p - lr * grad(p)
        history.append(p.copy())
        if np.linalg.norm(p) > 1e6:          # 발산
            break
    return np.array(history)

h = gradient_descent(grad_f, [-4.0, 2.0], lr=0.1, steps=100)   # h[-1] ≈ (0, 0)
```

**연결**
- [NN D2·7교시 선형 회귀를 경사하강으로](#/nn/2/7) — 같은 갱신을 파라미터에
- [NN D4·2교시 옵티마이저 구현](#/nn/4/2) — 학습률 문제의 해결
