## Day 5 — CNN 원리

### 1교시 · 복습 퀴즈 · 완전연결층의 한계

**키워드**: `완전연결층(Affine)`, `펼치면 위치 관계를 잃음`, `픽셀 뒤섞기 0.9436 vs 0.9455`, `파라미터 폭발`, `224×224×3 → 1000 = 1.5억`, `지역성`, `가중치 공유`, `1,792`

**개요**: 지금까지의 MLP 는 이미지를 784 칸 벡터로 펼쳐서 봤다. ① Affine 은 784 개 입력에 각각 따로 가중치를 두므로 "어느 픽셀이 어느 픽셀 옆인지"를 쓰지 않는다 — 픽셀 순서를 모든 이미지에서 똑같이 뒤섞어도 정확도가 같다. ② 큰 이미지에서는 파라미터가 **폭발**한다 — 스마트폰 카메라 크기면 첫 층만 1억 5천만 개(600 MB). ③ 합성곱은 **지역성 + 가중치 공유** 두 아이디어로 둘 다 해결한다.

**핵심 용어**
- **완전연결층(fully connected, Affine)**: 모든 입력과 모든 출력이 각자 가중치로 연결된 층
- **지역성(locality)**: 출력 하나가 입력의 작은 영역(예 3×3)만 보는 것 — 이웃 관계를 구조적으로 사용
- **가중치 공유(weight sharing)**: 같은 필터를 이미지의 모든 위치에 쓰는 것 — 파라미터 수가 이미지 크기와 무관, 어디에 있든 같은 특징을 검출
- **순열(permutation)**: 순서를 바꾼 배열 — 고정 순열로 픽셀을 섞으면 사람 눈엔 노이즈

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `fc_weights()` | `n_in * n_out` | 완전연결층 가중치 수 |
| `conv_params()` | `FN * C * F * F + FN` | 합성곱 층 파라미터 수 (필터 + 편향) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 이미지를 펼치면 잃는 것 | 고정 순열로 픽셀을 뒤섞은 MNIST 와 원본으로 같은 MLP(784-100-10, Adam, 1 epoch) 학습 → **0.9436 vs 0.9455** |
| 파라미터 폭발 | `fc_weights` · `conv_params` 로 28×28 → 100, 224×224×3 → 1000, 3×3 RGB → 64 비교 |

**핵심 내용**

#### 완전연결층의 한계 — 위치 관계를 잃는다
> 이미지를 784칸 벡터로 펼치면 위아래로 붙은 두 픽셀이 **28칸** 떨어진다 — 완전연결층(Affine)은 이웃 관계를 쓰지 않는다
- **증거** → 고정 순열(permutation)로 모든 이미지의 픽셀을 똑같이 뒤섞어도 MLP 정확도 그대로 (원본 **0.9455** vs 뒤섞음 **0.9436**)
- **그래서 MLP 는** → 숫자 모양에 대한 "상식" 을 6만 장에서 처음부터 배워야 한다
- ⚠ 사람 눈엔 노이즈인 그림인데 MLP 에겐 원본과 같다 — 공간 구조를 안 쓴다는 뜻

#### 파라미터 폭발
> 완전연결층 가중치 = 입력 수 × 출력 수 — 이미지가 커지면 감당할 수 없다
- **`fc_weights()`** → 28×28 → 100 은 78,400 · 224×224×3 → 1000 은 **150,528,000** (float32 약 **600 MB**)
- **`conv_params()`** → FN·C·F·F + FN — 3×3 RGB 필터 64개 = **1,792** (이미지 크기와 무관)
- **온디바이스에서** → 첫 층만 600 MB 면 불가능

#### 합성곱의 두 아이디어
> **지역성** + **가중치 공유** — 위치 관계를 쓰면서 파라미터 수를 이미지 크기와 무관하게
- **지역성(locality)** → 출력 하나는 입력의 작은 영역(3×3)만 본다 → 이웃 관계를 구조적으로 사용
- **가중치 공유(weight sharing)** → 같은 필터를 모든 위치에 → 파라미터가 적고, 어디에 있든 같은 특징을 검출

**기본 사용형태**
```python
def fc_weights(n_in, n_out):      return n_in * n_out
def conv_params(C, FN, F):        return FN * C * F * F + FN

fc_weights(224 * 224 * 3, 1000)   # 150528000
conv_params(3, 64, 3)             # 1792
```

**연결**
- [NN D2·5교시 행렬로 보는 순전파](#/nn/2/5) — 파라미터 수 = 모델 크기 = 메모리
- [NN D3·8교시 미니 프로젝트: MNIST 2층 MLP](#/nn/3/8) — reshape(-1, 784) 로 펼쳤던 입력

**상세 학습**

#### 이 교시에서 답하는 질문
Day 3 · 4 의 MLP 는 MNIST 에서 0.97 을 넘었다. 그런데 이 모델은 이미지를 784칸짜리 **줄**로 펼쳐서 봤다. 이 교시는 펼치면 무엇을 잃는지, 큰 이미지에서는 왜 MLP 를 쓸 수 없는지 확인하고, 오늘 배울 **합성곱**이 그 두 문제를 어떤 아이디어로 푸는지 미리 본다.

#### 개념 1 · 펼치면 위치 관계를 잃는다
28×28 이미지를 `reshape(-1, 784)` 로 펼치면 한 줄(28칸)씩 이어 붙는다. 위아래로 바로 붙은 두 픽셀 a, b 는 펼친 벡터에서 **28칸** 떨어진다.

![이미지를 펼치면 위아래로 붙은 a, b 가 펼친 벡터에서 28칸 떨어진다](assets/fig/nn/d5-flatten.svg)

완전연결층(Affine)은 784개 입력마다 따로 가중치를 둔다. "몇 번 칸이 몇 번 칸 옆인지" 라는 정보를 전혀 쓰지 않는다. 그 증거가 **픽셀 뒤섞기 실험**이다. 모든 이미지의 픽셀 순서를 똑같은 순열로 섞으면 사람 눈엔 노이즈가 되지만, MLP 정확도는 원본 0.9455, 뒤섞음 0.9436 으로 거의 같다. MLP 는 "획은 이어져 있다" 같은 상식 없이 6만 장에서 모든 것을 처음부터 배운다.

#### 개념 2 · 파라미터 폭발
완전연결층의 가중치 수 = 입력 수 × 출력 수.

| 층 | 계산 | 파라미터 |
|---|---|---|
| FC 28×28 → 100 | 784 × 100 | 78,400 |
| FC 224×224×3 → 1000 | 150,528 × 1000 | **150,528,000** (float32 약 600 MB) |
| Conv 3×3, RGB → 필터 64개 | 64 × 3 × 3 × 3 + 64 | **1,792** |

스마트폰 사진 크기면 첫 층 하나만 600 MB 다. 합성곱 층은 필터 크기와 개수로만 정해지므로 이미지가 224×224 든 28×28 이든 1,792 개다.

#### 개념 3 · 합성곱의 두 아이디어
| 아이디어 | 뜻 | 효과 |
|---|---|---|
| 지역성 | 출력 하나는 입력의 작은 영역(예 3×3)만 본다 | 이웃 관계를 구조적으로 사용 |
| 가중치 공유 | 같은 필터를 이미지의 모든 위치에 쓴다 | 파라미터 수가 이미지 크기와 무관 · 어디에 있든 같은 특징 검출 |

#### 코드로 확인
```python
perm = np.random.default_rng(0).permutation(784)       # 모든 이미지에 같은 순열
X_perm_train, X_perm_test = X_train[:, perm], X_test[:, perm]
for name, A, B in [("original", X_train, X_test), ("permuted", X_perm_train, X_perm_test)]:
    m = mynn.Sequential([mynn.Affine(784, 100, seed=1), mynn.Relu(), mynn.Affine(100, 10, seed=2)])
    mynn.Trainer(m, mynn.Adam(lr=0.001), batch_size=100, seed=0).fit(A, t_train, epochs=1, verbose=False)
    mlp_acc[name] = m.accuracy(B, t_test)
```
- **무엇을 하나**: 픽셀을 고정 순열로 뒤섞은 MNIST 와 원본으로 같은 MLP 를 1 epoch 학습해 비교한다.
- **어떤 개념**: 개념 1 — MLP 는 위치 관계를 쓰지 않는다.
- **출력**: 원본 **0.9455** vs 뒤섞음 **0.9436** (강의 실행 값) — 차이 0.01 미만

```python
def fc_weights(n_in, n_out):  return n_in * n_out
def conv_params(C, F, FN):    return FN * C * F * F + FN      # 필터 FN 개 + 편향 FN 개

fc_weights(28 * 28, 100), fc_weights(224 * 224 * 3, 1000), conv_params(3, 3, 64)
```
- **무엇을 하나**: 완전연결층과 합성곱 층의 파라미터 수를 센다.
- **어떤 개념**: 개념 2.
- **출력**: `(78400, 150528000, 1792)`

#### 한 번에 정리
- 펼치면 이웃 픽셀이 28칸 떨어진다 — MLP 는 위치 관계를 안 쓴다 (뒤섞어도 정확도 같음).
- 큰 이미지에서 완전연결층은 파라미터가 폭발한다 (224×224×3 → 1000 = 1.5억).
- 합성곱 = 지역성 + 가중치 공유 → 둘 다 해결.
- ⚠ 합성곱 층의 파라미터 수는 이미지 크기와 무관하다.

**연관 문제**
<<QA 1>>

### 2교시 · 합성곱 연산

**키워드**: `합성곱`, `필터(커널)`, `cross-correlation`, `패딩 P`, `스트라이드 S`, `OH = (H + 2P − FH)/S + 1`, `(N, C, H, W)`, `(FN, C, FH, FW)`, `채널을 더해 한 장`, `Sobel`

**개요**: 합성곱은 작은 **필터**를 이미지 위에서 미끄러뜨리며 겹치는 칸끼리 곱해 모두 더한 값을 출력에 적는 연산이다(신경망에서는 필터를 뒤집지 않는 cross-correlation). ① 패딩은 둘레에 0 을 덧대 크기를 유지하고, 스트라이드는 건너뛰며 크기를 줄인다 — 출력 크기 공식 하나로 계산. ② 필터 하나는 입력의 **모든 채널**을 덮어 곱한 뒤 더해 출력 **한 장**, 필터 FN 개면 FN 장. ③ CNN 은 Sobel 같은 필터를 사람이 설계하지 않고 데이터로 **학습**한다.

**핵심 용어**
- **합성곱(convolution)**: 필터를 슬라이딩하며 원소별 곱의 합을 계산하는 연산
- **필터(filter, 커널)**: 학습되는 작은 가중치 블록 (C, FH, FW)
- **패딩(padding, P)**: 입력 둘레에 0 을 P 줄 덧대기 — 가장자리 정보를 살리고 출력 크기 유지
- **스트라이드(stride, S)**: 필터를 한 번에 S 칸씩 이동 — 출력이 작아진다
- **특징 맵(feature map)**: 필터 하나의 출력 — 그 필터가 반응한 위치
- **채널(channel)**: 이미지의 겹 수 — 컬러 R · G · B = 3, 중간층은 수십

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `conv_output_size()` | `(H + 2 * P - F) // S + 1` | 출력 크기 |
| `conv_naive()` | `conv_naive(x, W, b, stride=1, pad=0)` | 4중 for 루프 합성곱 (배치 · 다채널 · 다중 필터) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 다채널 · 다중 필터 · 배치 | `conv_naive` 로 배치 · 다채널 · 패딩 · 스트라이드 합성곱, Sobel 가로 · 세로 에지 필터 (2, 1, 3, 3) 를 숫자 이미지에 적용해 특징 맵 2장 |

**핵심 내용**

#### 합성곱(convolution)
> 작은 **필터**를 미끄러뜨리며 겹치는 칸끼리 곱해 모두 더한 값을 출력에 적는 연산 — out[i, j] = Σ in[i·S + y, j·S + x] · K[y, x] + b
- **예** → 4×4 입력 · 3×3 필터 → 2×2 출력, 왼쪽 위 칸 = **15**
- **cross-correlation** → 수학의 합성곱은 필터를 뒤집지만 신경망은 안 뒤집음 (필터가 학습되니 차이 없음)
- **Sobel 가로 에지** → [[−1, −2, −1], [0, 0, 0], [1, 2, 1]] — 위아래 밝기 차이에 반응 · CNN 은 이런 필터를 **학습**한다
- **특징 맵(feature map)** → 필터 하나의 출력 — 그 필터가 반응한 위치가 크다

#### 출력 크기 — 패딩과 스트라이드
> OH = (H + 2P − FH) // S + 1
- **패딩(padding, P)** → 둘레에 0 을 덧대 가장자리 정보 보존 · 크기 유지 — 28, 3×3, P1 → **28**
- **스트라이드(stride, S)** → S 칸씩 건너뛰기 → 출력이 작아짐 — 28, 3×3, S2, P1 → **14**
- **`conv_output_size()`** → 28, 5×5 → 24 · 224, 7×7, S2, P3 → 112 · 7, 3×3, S2 → 3
- ⚠ 나눗셈은 정수 `//`

#### 다채널 · 다중 필터 · 배치 — shape 규칙
> 입력 (N, C, H, W) · 필터 (FN, C, FH, FW) · 출력 (N, FN, OH, OW)
- **채널(channel)** → 필터 하나가 모든 채널을 덮어 곱한 뒤 **더해서** 출력 한 장 · 필터(커널)마다 편향 하나
- **필터 FN 개** → 출력 FN 장 (출력 채널 = 필터 수)
- **`conv_naive()`** → 4중 for 루프 합성곱 — 2채널 × 2×2 를 모두 1 로 하면 2·2·2 + 1 = **9**
- ⚠ `np.pad` 는 높이 · 너비만: `[(0,0), (0,0), (P,P), (P,P)]`

**기본 사용형태**
```python
def conv_output_size(H, F, S=1, P=0): return (H + 2 * P - F) // S + 1

def conv_naive(x, W, b, stride=1, pad=0):
    N, C, H, Wd = x.shape; FN, _, FH, FW = W.shape
    OH, OW = (H + 2*pad - FH) // stride + 1, (Wd + 2*pad - FW) // stride + 1
    xp = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)])
    out = np.zeros((N, FN, OH, OW))
    for n in range(N):
        for f in range(FN):
            for i in range(OH):
                for j in range(OW):
                    win = xp[n, :, i*stride:i*stride+FH, j*stride:j*stride+FW]   # (C, FH, FW)
                    out[n, f, i, j] = np.sum(win * W[f]) + b[f]
    return out
```

**연결**
- [NN D1·3교시 NumPy ndarray](#/nn/1/3) — 4차원 배열 · 슬라이싱 · np.pad
- [NN D1·5교시 선형대수](#/nn/1/5) — 곱해서 더하기 = 내적

**상세 학습**

#### 이 교시에서 답하는 질문
1교시에서 "작은 필터를 모든 위치에 쓴다" 는 아이디어를 봤다. 이 교시는 그것을 실제 계산으로 만든다 — 합성곱 한 칸은 어떻게 계산되나, 출력 크기는 얼마인가, 컬러(여러 채널) 이미지와 여러 필터는 어떻게 다루나.

#### 개념 1 · 합성곱 — 미끄러뜨리며 곱해서 더하기
작은 **필터**(커널)를 이미지 위에 올리고, 겹치는 칸끼리 곱해 모두 더한 값을 출력 한 칸에 적는다. 필터를 한 칸씩 옮기며 반복한다.

> out[i, j] = Σ in[i·S + y, j·S + x] · K[y, x] + b

![4 × 4 입력과 3 × 3 필터의 합성곱 — 왼쪽 위 영역 × 필터 = 15, 옮기면 16 · 6 · 15](assets/fig/nn/d5-conv.svg)

- 수학의 합성곱은 필터를 뒤집지만 신경망은 뒤집지 않는다(**cross-correlation**). 필터 값이 학습되므로 차이가 없다.
- 필터 하나의 출력을 **특징 맵**이라 한다. 필터와 비슷한 무늬가 있는 자리에서 값이 크다.
- 예: **Sobel 가로 에지** 필터 [[−1, −2, −1], [0, 0, 0], [1, 2, 1]] 는 위아래 밝기 차이(가로 경계)에 반응한다. CNN 은 이런 필터를 사람이 설계하지 않고 데이터로 학습한다.

#### 개념 2 · 출력 크기 — 패딩과 스트라이드
> OH = (H + 2P − FH) // S + 1

| 기호 | 뜻 |
|---|---|
| H, FH | 입력 · 필터 높이 |
| P (패딩) | 둘레에 0 을 덧대는 줄 수 — 가장자리 정보를 살리고 크기 유지 |
| S (스트라이드) | 필터를 한 번에 옮기는 칸 수 — 크면 출력이 작아짐 |

![패딩 P = 1 은 크기를 유지하고, 스트라이드 S = 2 는 2칸씩 건너뛰어 출력을 줄인다](assets/fig/nn/d5-padstride.svg)

| H | F | S | P | 출력 | 의미 |
|---|---|---|---|---|---|
| 28 | 5 | 1 | 0 | 24 | 패딩이 없으면 줄어든다 |
| 28 | 3 | 1 | 1 | 28 | 3×3 + P1 = 크기 유지 (same) |
| 28 | 3 | 2 | 1 | 14 | 스트라이드 2 = 절반 |
| 224 | 7 | 2 | 3 | 112 | |

#### 개념 3 · 다채널 · 다중 필터 · 배치
컬러 이미지는 R · G · B 3장(**채널**)이다. 필터 하나는 모든 채널을 덮는 (C, FH, FW) 블록이고, 채널마다 곱한 결과를 **모두 더해** 출력 한 장을 만든다. 필터가 FN 개면 출력도 FN 장이다.

![다채널 합성곱 — 필터 하나가 모든 채널을 덮어 곱한 뒤 더해 한 장, 필터 FN 개면 FN 장](assets/fig/nn/d5-channels.svg)

| 배열 | shape |
|---|---|
| 입력 | (N, C, H, W) |
| 필터 | (FN, C, FH, FW) |
| 편향 | (FN,) — 필터마다 하나 |
| 출력 | (N, FN, OH, OW) |

#### 코드로 확인
```python
def conv_naive(x, W, b, stride=1, pad=0):
    N, C, H, Wd = x.shape; FN, _, FH, FW = W.shape
    OH, OW = conv_output_size(H, FH, stride, pad), conv_output_size(Wd, FW, stride, pad)
    xp = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)])
    out = np.zeros((N, FN, OH, OW))
    for n in range(N):
        for f in range(FN):
            for i in range(OH):
                for j in range(OW):
                    window = xp[n, :, i * stride:i * stride + FH, j * stride:j * stride + FW]
                    out[n, f, i, j] = np.sum(window * W[f]) + b[f]
    return out
```
- **무엇을 하나**: 배치 · 필터 · 출력 위치를 4중 루프로 돌며, (C, FH, FW) 영역과 필터를 곱해 모두 더한다.
- **어떤 개념**: 개념 1 ~ 3 — 패딩은 높이 · 너비에만, 채널은 합쳐서 한 장.
- **출력**: 모두 1 인 (1, 2, 3, 3) 입력 · (1, 2, 2, 2) 필터 · 편향 1 → shape `(1, 1, 2, 2)`, 모든 값 **9** (= 2채널 × 2 × 2 + 1)

```python
sobel_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)     # 가로 에지
sobel_v = sobel_h.T                                                         # 세로 에지
W_sobel = np.stack([sobel_h, sobel_v])[:, None, :, :]                      # (2, 1, 3, 3)
edges = conv_naive(X4_train[:1], W_sobel, np.zeros(2), stride=1, pad=1)
```
- **무엇을 하나**: 손으로 설계한 에지 필터 2개로 숫자 이미지를 합성곱한다.
- **어떤 개념**: 개념 1 — 필터 = 특징 검출기, 개념 2 — P1 이라 크기 유지.
- **출력**: `edges.shape` = `(1, 2, 28, 28)` — 가로획 · 세로획이 밝게 나온 특징 맵 2장

#### 한 번에 정리
- 합성곱 = 필터를 미끄러뜨리며 겹치는 칸끼리 곱해 더하기 (신경망은 뒤집지 않음).
- 출력 크기 OH = (H + 2P − FH) // S + 1 — 패딩은 유지, 스트라이드는 축소.
- 필터 하나는 모든 채널을 덮어 더해 한 장, 필터 FN 개 → 출력 FN 장.
- ⚠ `np.pad` 는 `[(0,0), (0,0), (P,P), (P,P)]` — 배치 · 채널 축엔 덧대지 않는다.

**연관 문제**
<<QA 2>>

### 3교시 · im2col: 합성곱을 행렬곱으로

**키워드**: `im2col`, `영역 하나 = 행 하나`, `필터 하나 = 열 하나`, `col @ col_W + b`, `reshape · transpose(0, 3, 1, 2)`, `루프 FH·FW 번`, `메모리 약 18.4배`, `속도 백 배 이상`

**개요**: 4중 for 루프 합성곱은 느리다. ① **im2col** 은 필터가 닿는 영역 하나하나를 **행 하나**로 펼친 큰 행렬을 만든다 — 필터는 **열 하나**로 펴면 합성곱이 Affine 과 같은 `X @ W + b` 행렬곱 **한 번**. ② 결과를 reshape · transpose 로 (N, FN, OH, OW) 로 되돌린다. ③ 겹치는 영역을 중복 복사하니 **메모리는 더** 쓰지만 **속도는 백 배 이상** — CPU · GPU · NPU 가 가장 잘하는 행렬곱으로 바꾸는 교환(trade-off).

**핵심 용어**
- **im2col(image to column)**: 합성곱 영역을 행으로 펼쳐 합성곱을 행렬곱으로 바꾸는 변환
- **col**: im2col 결과 (N·OH·OW, C·FH·FW) — 행 하나 = 출력 한 칸이 보는 영역
- **col_W**: 필터를 펼친 (C·FH·FW, FN) — 열 하나 = 필터 하나
- **trade-off(교환)**: 하나를 얻는 대신 다른 하나를 내주는 관계 — 여기선 속도 ↔ 메모리

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `im2col()` | `im2col(x, FH, FW, stride=1, pad=0)` | (N, C, H, W) → (N·OH·OW, C·FH·FW) |
| `col2im()` | `col2im(col, input_shape, FH, FW, stride, pad)` | im2col 의 반대 — 겹치는 자리는 더함 (역전파용) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | `im2col` 결과 shape 확인 → 행렬곱 합성곱이 `conv_naive` 와 같은 값인지 비교, MNIST 20장 · 필터 30개(5×5)로 두 방법 시간 측정 |

**핵심 내용**

#### im2col — 합성곱을 행렬곱 한 번으로
> 필터가 닿는 **영역 하나 = 행 하나**(col), **필터 하나 = 열 하나**(col_W) → col @ col_W + b — Affine 과 같은 식
- **`im2col()`** → (N, C, H, W) → (N·OH·OW, C·FH·FW) — 4×4 입력 · 3×3 이면 (4, 9), 첫 행 = 왼쪽 위 영역
- **되돌리기** → (N·OH·OW, FN) → `reshape(N, OH, OW, FN).transpose(0, 3, 1, 2)` → (N, FN, OH, OW)
- **예** → MNIST 1장 · 5×5 필터 30개: col (576, 25) @ col_W (25, 30) → (1, 30, 24, 24)
- **`col2im()`** → im2col 의 반대 — 겹치는 자리는 **더한다** (4교시 역전파용)
- ⚠ col_W 는 `W.reshape(FN, -1).T` — `.T` 를 빠뜨리면 (FN, C·FH·FW)

#### 구현 요령 · 속도 ↔ 메모리 trade-off
> 영역을 하나씩 도는 대신 필터 안 위치 (y, x) 를 돌며 모든 출력 위치를 한꺼번에 슬라이싱 — 루프는 FH·FW 번
- **루프 횟수** → 5×5 면 **25**번 (출력 위치 수와 무관) — Day 1 벡터화
- **trade-off(교환)** → 겹치는 영역을 중복 복사해 메모리 약 FH·FW / S² 배 (MNIST 5×5: **18.4배**), 대신 행렬곱 한 번 → **백 배 이상** 빠름
- **실제 라이브러리** → 비슷한 방법 — 행렬곱은 CPU · GPU · NPU 가 가장 잘하는 연산

**기본 사용형태**
```python
col   = im2col(x, FH, FW, stride, pad)          # (N·OH·OW, C·FH·FW)
col_W = W.reshape(FN, -1).T                     # (C·FH·FW, FN)
out   = col @ col_W + b                         # (N·OH·OW, FN)
out   = out.reshape(N, OH, OW, FN).transpose(0, 3, 1, 2)   # (N, FN, OH, OW)
```

**연결**
- [NN D1·4교시 벡터화](#/nn/1/4) — 루프를 행렬곱 한 번으로
- [NN D3·5교시 Affine · Softmax-with-Loss 계층](#/nn/3/5) — 같은 X @ W + b

**상세 학습**

#### 이 교시에서 답하는 질문
2교시의 `conv_naive` 는 파이썬 4중 루프라 느리다. MNIST 6만 장을 학습시키기엔 너무 느리다. 이 교시는 합성곱을 **행렬곱 한 번**으로 바꾸는 방법 — im2col — 을 만들고, 그 대가로 무엇을 내주는지 본다.

#### 개념 1 · im2col — 영역은 행으로, 필터는 열로
합성곱의 출력 한 칸은 "영역 (C·FH·FW 개 값)" 과 "필터 (같은 개수)" 의 내적이다. 그렇다면
1. 필터가 닿는 **영역 하나를 행 하나**로 펼쳐 쌓으면 → col (N·OH·OW, C·FH·FW)
2. **필터 하나를 열 하나**로 펼치면 → col_W (C·FH·FW, FN)
3. col @ col_W + b = (N·OH·OW, FN) — Day 3 Affine 의 `X @ W + b` 와 같은 식이다.

![im2col — 4 × 4 입력의 3 × 3 영역 4개를 행 4개로 펼치고, 필터는 열로 펴서 행렬곱 한 번](assets/fig/nn/d5-im2col.svg)

| 단계 | shape (MNIST 1장, 5×5 필터 30개) |
|---|---|
| 입력 x | (1, 1, 28, 28) |
| `im2col(x, 5, 5)` → col | (576, 25) — 출력 24×24 = 576칸 |
| `W.reshape(30, -1).T` → col_W | (25, 30) |
| `col @ col_W + b` | (576, 30) |
| `.reshape(1, 24, 24, 30).transpose(0, 3, 1, 2)` | (1, 30, 24, 24) |

마지막 transpose 는 (N, OH, OW, FN) 의 채널 축 FN 을 앞으로 옮겨 (N, FN, OH, OW) 로 만든다.

#### 개념 2 · 구현 요령 — 루프는 필터 크기만큼
영역을 하나씩(576번) 도는 대신, 필터 안의 위치 (y, x) 를 돈다. "모든 영역의 (y, x) 번째 값" 은 슬라이싱 `img[:, :, y : y + S·OH : S, x : x + S·OW : S]` 한 번으로 얻는다. 루프는 5×5 = **25**번뿐이고 출력 크기와 무관하다 (Day 1 벡터화).

#### 개념 3 · 속도 ↔ 메모리 trade-off
이웃 영역은 서로 겹치므로 같은 픽셀이 col 의 여러 행에 **중복 복사**된다. 메모리는 약 FH·FW / S² 배가 된다 — MNIST 5×5, S = 1 이면 25 × 576 / 784 ≈ **18.4배**. 대신 CPU · GPU · NPU 가 가장 잘하는 행렬곱 한 번이 되어 **백 배 이상** 빨라진다. 실제 라이브러리도 비슷한 방법을 쓰고, 메모리가 빠듯한 온디바이스에서는 이 교환 비율이 설계 기준이 된다.

#### 코드로 확인
```python
def im2col(x, FH, FW, stride=1, pad=0):
    N, C, H, W = x.shape
    OH, OW = conv_output_size(H, FH, stride, pad), conv_output_size(W, FW, stride, pad)
    img = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)])
    col = np.zeros((N, C, FH, FW, OH, OW))
    for y in range(FH):
        for x_ in range(FW):
            col[:, :, y, x_, :, :] = img[:, :, y:y + stride * OH:stride, x_:x_ + stride * OW:stride]
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
```
- **무엇을 하나**: 필터 안 위치 (y, x) 마다 모든 출력 위치의 값을 한꺼번에 잘라 담고, 축을 바꿔 (영역, 값) 행렬로 편다.
- **어떤 개념**: 개념 1 · 2.
- **출력**: `im2col(np.arange(16.).reshape(1, 1, 4, 4), 3, 3)` → shape `(4, 9)`, 첫 행 `[0, 1, 2, 4, 5, 6, 8, 9, 10]` (왼쪽 위 3×3 영역)

```python
def conv_im2col(x, W, b, stride=1, pad=0):
    FN, C, FH, FW = W.shape; N, _, H, Wd = x.shape
    OH, OW = conv_output_size(H, FH, stride, pad), conv_output_size(Wd, FW, stride, pad)
    col = im2col(x, FH, FW, stride, pad)          # (N·OH·OW, C·FH·FW)
    col_W = W.reshape(FN, -1).T                   # (C·FH·FW, FN)
    out = col @ col_W + b                         # (N·OH·OW, FN)
    return out.reshape(N, OH, OW, FN).transpose(0, 3, 1, 2)
```
- **무엇을 하나**: im2col → 행렬곱 → 모양 되돌리기로 합성곱을 계산한다.
- **어떤 개념**: 개념 1 — Affine 과 같은 식.
- **출력**: `conv_naive` 와 같은 값 · MNIST 20장 · 필터 30개에서 **백 배 이상** 빠름 (시간은 CPU 마다 다름) · `col_ratio` ≈ **18.4**

#### 한 번에 정리
- im2col: 영역 하나 = 행 하나, 필터 하나 = 열 하나 → col @ col_W + b.
- 결과는 reshape(N, OH, OW, FN).transpose(0, 3, 1, 2) 로 되돌린다.
- 루프는 FH·FW 번 — 슬라이싱으로 모든 위치를 한 번에.
- 메모리 약 18.4배 ↔ 속도 백 배 이상 (trade-off).
- ⚠ col_W 는 `W.reshape(FN, -1).T` — `.T` 를 잊지 말 것.

**연관 문제**
<<QA 3>>

### 4교시 · Convolution · Pooling 계층 → mynn.py

**키워드**: `Convolution 계층`, `역전파 = Affine 그대로`, `col2im 은 더한다`, `transpose(0, 2, 3, 1)`, `Max Pooling`, `arg_max`, `파라미터 없음`, `layer_check`, `mynn_conv.py`

**개요**: 합성곱을 Day 4 mynn 규약(params · grads · forward · backward) 계층으로 만든다. ① Convolution forward 는 `col @ col_W + b` — Day 3 Affine 의 `X @ W + b` 와 같으니 **backward 도 Affine 그대로**, 마지막에 **col2im** 으로 이미지 모양으로 되돌린다(겹친 자리는 **더함**). ② **Max Pooling** 은 2×2 영역 최댓값만 남겨 크기를 절반으로 — 파라미터가 없고, 역전파는 최댓값이었던 자리로만(ReLU mask 처럼). ③ 둘 다 gradient check 를 통과시키고 mynn.py 에 추가한다.

**핵심 용어**
- **Convolution 계층**: im2col + Affine + reshape 로 만든 합성곱 계층 (mynn 규약)
- **col2im**: im2col 의 반대 — 여러 영역에 복사되어 쓰인 픽셀의 기울기를 모두 **더한다**
- **Max Pooling**: 영역(예 2×2, stride 2)마다 최댓값만 남기는 다운샘플링. 채널 수는 그대로
- **arg_max**: 순전파 때 영역 안에서 최댓값이었던 위치 — 역전파 때 기울기를 보낼 자리
- **이동 불변성(translation invariance)**: 입력이 조금 움직여도 출력이 크게 안 바뀌는 성질 — 풀링이 일부 제공

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Convolution` | `Convolution(in_channels, out_channels, filter_size, stride=1, pad=0, weight_init="he", seed)` | 합성곱 계층 (mynn 규약) |
| `Pooling` | `Pooling(pool_h=2, pool_w=2, stride=2)` | Max Pooling 계층 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| Max Pooling | `%%writefile mynn_conv.py` 로 Convolution · Pooling 을 완성 → mynn.py = Day 4 + Day 5 로 합쳐 다시 import → `layer_check` 로 dx · dW · db 를 수치 미분과 비교 |

**핵심 내용**

#### Convolution 계층의 역전파
> forward 가 col @ col_W + b 이므로 backward 도 **Affine 식 그대로** — 마지막에 **col2im** 으로 이미지 모양으로
- **`Convolution`** → dW = (col.T @ dout).T.reshape(FN, C, FH, FW) · db = dout.sum(axis=0) · dcol = dout @ col_W.T
- **`col2im()`** → 한 픽셀이 여러 영역(col 행)에 복사되어 쓰였으니 기울기를 **모두 더한다** (복사 노드의 역전파 = 더하기)
- **dout 펼치기** → (N, FN, OH, OW) → `transpose(0, 2, 3, 1)` → `reshape(-1, FN)`
- ⚠ transpose 를 빠뜨리면 채널과 위치가 섞임 · col2im 은 `=` 가 아니라 **`+=`**

#### Max Pooling
> 영역(2×2, stride 2)마다 최댓값만 남기는 다운샘플링 — 크기 절반, 채널 수 그대로, 파라미터 **없음**
- **`Pooling`** → forward 에서 최댓값 위치 **arg_max** 저장 → backward 는 그 자리로만 dout, 나머지 0 (ReLU mask 처럼)
- **예** → (N, 30, 24, 24) → (N, 30, 12, 12)
- **이동 불변성(translation invariance)** → 영역 안에서 조금 움직여도 최댓값은 그대로 → 작은 이동에 둔감
- ⚠ `Sequential.params()` 에 Pooling 은 안 나온다 — 학습할 값이 없다

#### 새 계층은 gradient check
> L = Σ (forward(x) ⊙ dout) 의 수치 미분과 backward 를 비교 (Day 3 4교시 방법)
- **`layer_check`** → dx · dW · db 상대 오차 — Convolution(S1 P1 · S2 P0) · Pooling 모두 통과
- **mynn.py** → `%%writefile mynn_conv.py` 로 두 계층을 만들어 Day 4 mynn.py 뒤에 합친다

**기본 사용형태**
```python
def forward(self, x, train_flg=False):
    FN, C, FH, FW = self.params["W"].shape
    N, _, H, W = x.shape
    OH, OW = conv_output_size(H, FH, self.stride, self.pad), conv_output_size(W, FW, self.stride, self.pad)
    self.x, self.col = x, im2col(x, FH, FW, self.stride, self.pad)
    self.col_W = self.params["W"].reshape(FN, -1).T
    out = self.col @ self.col_W + self.params["b"]
    return out.reshape(N, OH, OW, FN).transpose(0, 3, 1, 2)

def backward(self, dout):
    FN, C, FH, FW = self.params["W"].shape
    dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
    self.grads["b"] = dout.sum(axis=0)
    self.grads["W"] = (self.col.T @ dout).T.reshape(FN, C, FH, FW)
    dcol = dout @ self.col_W.T
    return col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
```

**연결**
- [NN D3·5교시 Affine · Softmax-with-Loss 계층](#/nn/3/5) — 같은 역전파 식
- [NN D3·4교시 활성화 계층: ReLU · Sigmoid](#/nn/3/4) — Pooling 역전파 = mask
- [NN D4·8교시 미니 프로젝트: 미니 프레임워크 mynn.py](#/nn/4/8) — 같은 계층 규약에 추가

**상세 학습**

#### 이 교시에서 답하는 질문
합성곱을 학습시키려면 역전파가 필요하다. 이 교시는 합성곱과 풀링을 Day 4 mynn 의 **계층 규약**(params · grads · forward · backward)에 맞춘 계층으로 만들고, gradient check 로 검증해 mynn.py 에 추가한다. 핵심 질문: 합성곱의 역전파를 새로 유도해야 할까? — 아니다.

#### 개념 1 · Convolution 역전파 = Affine 역전파 + col2im
forward 가 `col @ col_W + b` 이므로 backward 도 Day 3 Affine 의 식을 그대로 쓴다 (X 자리에 col, W 자리에 col_W).

| 기울기 | Affine (Day 3) | Convolution |
|---|---|---|
| dW | `X.T @ dout` | `(col.T @ dout).T.reshape(FN, C, FH, FW)` |
| db | `dout.sum(axis=0)` | 같음 (dout 을 (N·OH·OW, FN) 으로 펼친 뒤) |
| dX | `dout @ W.T` | `dcol = dout @ col_W.T` → **col2im** → (N, C, H, W) |

**col2im** 은 im2col 의 반대다. 단, 한 입력 픽셀이 여러 영역(여러 col 행)에 복사되어 쓰였으므로 그 자리들에서 온 기울기를 **모두 더한다** — Day 3 "값이 여러 곳에 쓰이면 역전파 때 기울기를 더한다" 와 같은 원리다. 예를 들어 4×4 입력 · 3×3 필터에서 가운데 픽셀은 영역 4개에 쓰여 기울기 4개가 더해진다.

#### 개념 2 · Max Pooling — 최댓값만, 기울기도 그 자리로만
2×2 영역마다 최댓값만 남겨 크기를 절반으로 줄인다. 채널 수는 그대로이고 학습할 파라미터는 **없다**.

![Max Pooling — 2 × 2 영역마다 최댓값만 남기고, 역전파 때는 그 최댓값 자리로만 기울기를 보낸다](assets/fig/nn/d5-pool.svg)

- forward 에서 최댓값이 있던 위치(**arg_max**)를 저장한다.
- backward 는 arg_max 자리로만 dout 을 보내고 나머지는 0 — ReLU 의 mask 와 같은 방식이다.
- 영역 안에서 조금 움직여도 최댓값은 그대로라 작은 이동에 둔감해진다 (**이동 불변성**의 일부).

#### 개념 3 · 새 계층은 gradient check
Day 3 4교시 방법: L = Σ (forward(x) ⊙ dout) 를 수치 미분한 값과 backward(dout) 를 비교한다. 이 L 의 ∂L/∂(출력) 이 정확히 dout 이라 계층 하나만 떼어 시험할 수 있다.

#### 코드로 확인
```python
def backward(self, dout):                                        # Convolution
    FN, C, FH, FW = self.params["W"].shape
    dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)            # (N·OH·OW, FN)
    self.grads["b"] = dout.sum(axis=0)
    self.grads["W"] = (self.col.T @ dout).T.reshape(FN, C, FH, FW)
    dcol = dout @ self.col_W.T                                    # Affine 의 dX 와 같은 식
    return col2im(dcol, self.x_shape, FH, FW, self.stride, self.pad)
```
- **무엇을 하나**: dout 을 forward 의 반대 순서로 (N·OH·OW, FN) 로 펼친 뒤 Affine 식으로 dW · db · dcol 을 구하고, col2im 으로 입력 모양으로 되돌린다.
- **어떤 개념**: 개념 1.
- **출력**: `col2im(np.ones((4, 9)), (1, 1, 4, 4), 3, 3)` → `[[1, 2, 2, 1], [2, 4, 4, 2], [2, 4, 4, 2], [1, 2, 2, 1]]` — 각 픽셀이 몇 개 영역에 쓰였는지 (더하기)

```python
def forward(self, x, train_flg=True):                           # Pooling
    N, C, H, W = x.shape
    OH = conv_output_size(H, self.pool_h, self.stride, self.pad)
    OW = conv_output_size(W, self.pool_w, self.stride, self.pad)
    self.x_shape = x.shape
    col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad).reshape(-1, self.pool_h * self.pool_w)
    self.arg_max = col.argmax(axis=1)                            # 역전파 때 기울기를 보낼 자리
    out = col[np.arange(col.shape[0]), self.arg_max]
    return out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
```
- **무엇을 하나**: im2col 로 영역을 행으로 편 뒤 (채널별로 한 행씩) 행마다 최댓값과 그 위치를 고른다.
- **어떤 개념**: 개념 2.
- **출력**: 그림의 4×4 입력 → `[[4, 5], [6, 3]]` · backward([[1, 2], [3, 4]]) → 최댓값 자리 4곳에만 1 · 2 · 3 · 4, 나머지 0

```python
rng4 = np.random.default_rng(0)
x4 = rng4.normal(size=(2, 3, 7, 7))
checks = {"conv s1 p1": layer_check(mynn.Convolution(3, 4, 3, stride=1, pad=1, seed=1), x4.copy(), rng4),
          "conv s2 p0": layer_check(mynn.Convolution(3, 2, 3, stride=2, pad=0, seed=2), x4.copy(), rng4),
          "pool 2×2":   layer_check(mynn.Pooling(2, 2, stride=2), x4[:, :, :6, :6].copy(), rng4)}
```
- **무엇을 하나**: 패딩 · 스트라이드 설정을 바꾼 합성곱 2개와 풀링을 수치 미분과 비교한다.
- **어떤 개념**: 개념 3.
- **출력**: dx · W · b 상대 오차 모두 **1e-11 미만** (numpy 로 직접 실행 — 약 1e-12) → 통과

#### 한 번에 정리
- Convolution backward = Affine backward(col, col_W) + col2im.
- col2im 은 겹치는 자리의 기울기를 더한다 (`+=`).
- Max Pooling: 최댓값만 남기고 arg_max 저장, 역전파는 그 자리로만 — 파라미터 없음.
- 새 계층은 gradient check 로 검증한 뒤 mynn.py 에 추가.
- ⚠ backward 에서 `transpose(0, 2, 3, 1)` 를 빠뜨리면 채널과 위치가 섞인다.

**연관 문제**
<<QA 4>>

### 5교시 · SimpleConvNet 으로 MNIST 학습

**키워드**: `SimpleConvNet`, `Conv → ReLU → Pool → Affine → ReLU → Affine`, `단계별 shape`, `(N, 1, 28, 28)`, `30·12·12 = 4,320`, `433,890`, `1 epoch 0.981`, `계층 규약의 힘`

**개요**: Day 4 의 Sequential · Trainer 를 **한 줄도 고치지 않고** 계층 리스트에 Convolution · Pooling 만 넣으면 CNN 이 된다 — 계층 규약을 만들어 둔 보람. ① 입력은 (N, 784) 가 아니라 (N, 1, 28, 28) 이미지 모양 그대로. ② Pooling 출력 (N, 30, 12, 12) 은 Affine 이 내부에서 펼쳐 받는다. ③ **1 epoch** 만에 test **0.981** — 공간 구조를 쓰는 모델이 더 적은 학습으로 더 잘 맞힌다.

**핵심 용어**
- **CNN(합성곱 신경망, convolutional neural network)**: 합성곱 · 풀링 계층으로 공간 구조를 쓰는 신경망
- **SimpleConvNet**: Conv(1→30, 5×5) → ReLU → Pool(2×2) → Affine(4320→100) → ReLU → Affine(100→10) 구조의 작은 CNN
- **계층 규약**: params · grads · forward(x, train_flg) · backward(dout) — 이것만 지키면 어떤 계층이든 Sequential 에 끼울 수 있음

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | SimpleConvNet 을 Sequential 로 조립 → 배치 100 단계별 shape 확인 → Adam 0.001 · 배치 100 · 1 epoch 학습 → test 0.981, day5_cnn.npz 로 저장 |

**핵심 내용**

#### CNN 조립 — SimpleConvNet
> Conv(1→30, 5×5) → ReLU → Pool(2×2) → Affine(4320→100) → ReLU → Affine(100→10) — 합성곱 · 풀링으로 공간 구조를 쓰는 신경망(CNN)
- **단계별 shape (배치 100)** → (100, 1, 28, 28) → (100, 30, 24, 24) → (100, 30, 12, 12) → (100, 100) → (100, 10)
- **4,320** → 30 × 12 × 12 — Affine 이 (N, C, H, W) 를 내부에서 펼쳐 받는다
- **파라미터** → 780 + 432,100 + 1,010 = **433,890**
- ⚠ 입력은 (N, 784) 가 아니라 **(N, 1, 28, 28)** — 채널 축 1

#### 계층 규약의 힘
> params · grads · forward · backward 만 지키면 어떤 계층이든 Sequential 에 끼운다 — 프레임워크는 한 줄도 안 고친다
- **Day 4 MLP 와 차이** → 계층 리스트 앞에 Convolution · Relu · Pooling 이 들어간 것뿐 (Trainer 그대로)
- **예** → Adam 0.001 · 1 epoch → test **0.981** (Day 3 MLP 10 epoch 0.9694 · Day 4 5층 MLP 5 epoch 0.9757)
- **PyTorch · Keras** → 같은 원리 (nn.Sequential, model.fit)

**기본 사용형태**
```python
X4_train = X_train.reshape(-1, 1, 28, 28)            # 이미지 모양 그대로
cnn = mynn.Sequential([
    mynn.Convolution(1, 30, 5, seed=1), mynn.Relu(), mynn.Pooling(2, 2, 2),
    mynn.Affine(30 * 12 * 12, 100, seed=2), mynn.Relu(), mynn.Affine(100, 10, seed=3)])
mynn.Trainer(cnn, mynn.Adam(lr=0.001), batch_size=100).fit(X4_train, t_train, epochs=1)
```

**연결**
- [NN D4·8교시 미니 프로젝트: 미니 프레임워크 mynn.py](#/nn/4/8) — Sequential · Trainer 를 그대로
- [NN D3·8교시 미니 프로젝트: MNIST 2층 MLP](#/nn/3/8) — 같은 데이터, MLP 결과

**상세 학습**

#### 이 교시에서 답하는 질문
합성곱 · 풀링 계층이 준비됐다. 이제 CNN 을 만들어 MNIST 를 학습시킨다. 질문은 두 가지 — 계층들을 어떤 순서로 쌓고 shape 가 어떻게 변하나, 그리고 Day 4 의 Sequential · Trainer 를 고쳐야 하나?

#### 개념 1 · SimpleConvNet 의 구조와 shape
> Conv(1→30, 5×5) → ReLU → Pool(2×2) → Affine(4320→100) → ReLU → Affine(100→10) → Softmax

![SimpleConvNet — 입력 (1, 28, 28) 에서 Conv · Pool 을 거쳐 Affine 2개로 10개 점수](assets/fig/nn/d5-cnn.svg)

| 단계 | shape (배치 100) | 계산 |
|---|---|---|
| input | (100, 1, 28, 28) | `reshape(-1, 1, 28, 28)` — 이미지 모양 그대로 |
| conv → ReLU | (100, 30, 24, 24) | (28 − 5)/1 + 1 = 24, 필터 30개 |
| pool | (100, 30, 12, 12) | 24 / 2 |
| fc1 → ReLU | (100, 100) | 30 × 12 × 12 = 4,320 입력 |
| fc2 | (100, 10) | 점수 → SoftmaxWithLoss |

Affine 은 (N, C, H, W) 입력을 받으면 내부에서 (N, 4320) 으로 펼친다 (Day 4 에서 만들어 둠).

#### 개념 2 · 계층 규약의 힘
Day 4 MLP 와 다른 점은 계층 리스트 앞에 `Convolution, Relu, Pooling` 이 들어간 것뿐이다. Sequential 의 gradient() 는 계층이 무엇이든 반대 순서로 backward 를 부르고, Trainer 는 그대로 셔플 → 배치 → 갱신을 한다. **프레임워크는 한 줄도 고치지 않는다** — PyTorch 의 nn.Sequential, Keras 의 model.fit 이 같은 원리다.

| 모델 | 학습 | test |
|---|---|---|
| Day 3 784-100-10 MLP | SGD 10 epoch | 0.9694 |
| Day 4 5층 MLP + BN + He | Adam 5 epoch | 0.9757 |
| Day 5 SimpleConvNet | Adam **1** epoch | **0.981** |

공간 구조를 쓰는 모델이 더 적은 학습으로 더 잘 맞힌다.

#### 코드로 확인
```python
cnn = mynn.Sequential([
    mynn.Convolution(1, 30, 5, stride=1, pad=0, seed=1),
    mynn.Relu(),
    mynn.Pooling(2, 2, stride=2),
    mynn.Affine(30 * 12 * 12, 100, seed=2),
    mynn.Relu(),
    mynn.Affine(100, 10, seed=3),
])
```
- **무엇을 하나**: SimpleConvNet 을 계층 리스트로 조립한다.
- **어떤 개념**: 개념 1 · 2.
- **출력**: `cnn.count_params()` = **433,890** (Conv 780 + Affine 432,100 + Affine 1,010)

```python
trainer5 = mynn.Trainer(cnn, mynn.Adam(lr=0.001), batch_size=100, seed=0)
history5 = trainer5.fit(X4_train, t_train, epochs=1, X_val=X4_test, t_val=t_test)
mynn.save_params(cnn, "day5_cnn.npz")
```
- **무엇을 하나**: Day 4 Trainer 그대로 1 epoch(600 step) 학습하고 가중치를 저장한다. 입력은 `X4_train` = (60000, 1, 28, 28).
- **어떤 개념**: 개념 2.
- **출력**: test **0.981** (강의 실행 값, 학습 시간은 CPU 마다 다름)

#### 한 번에 정리
- Conv → ReLU → Pool → Affine → ReLU → Affine, shape 28 → 24 → 12 → 4,320 → 100 → 10.
- 계층 규약 덕분에 Sequential · Trainer 를 고치지 않고 CNN 이 된다.
- 1 epoch 에 test 0.981 — MLP 보다 적은 학습으로 더 높다.
- ⚠ 입력은 (N, 1, 28, 28) — (N, 784) 로 넣으면 Convolution 이 shape 오류.

**연관 문제**
<<QA 5>>

### 6교시 · 학습 결과 해석

**키워드**: `1층 필터 30장`, `상관계수 0.86~0.999`, `변화량 약 28%`, `무작위 필터도 쓸모`, `특징 맵`, `부품 → 모양 → 물체`, `2 px 이동 0.9119 vs 0.7975`, `3 px 0.748`, `데이터 증강`

**개요**: CNN 이 무엇을 배웠는지 **그려 보고 숫자로** 확인한다. ① 1 epoch 후 1층 필터는 초기값에서 조금만 바뀐다(상관 0.86~0.999) — 무작위 5×5 필터도 국소 패턴에 반응하고, 학습의 대부분은 뒤쪽 Affine 이 그 반응을 **조합**하는 데 쓰였다. ② 특징 맵 30장은 필터마다 반응한 위치(가로획 · 경계 · 배경)를 보여 준다. ③ CNN 은 가중치 공유 · 풀링 덕분에 2 px 이동에 MLP 보다 강하지만, 풀링이 한 번뿐이라 3 px 에서는 크게 떨어진다 → 데이터 증강.

**핵심 용어**
- **필터 시각화**: 1층 필터 (30, 1, 5, 5) 를 5×5 그림 30장으로 그리는 것
- **상관계수(correlation)**: 두 배열이 얼마나 같은 모양인지 −1 ~ 1 로 — 1 이면 같은 무늬
- **특징 맵(feature map)**: 필터 하나를 통과한 출력 — 그 필터가 반응한 위치가 밝게
- **데이터 증강(augmentation)**: 학습 때 이미지를 일부러 옮기고 돌려 이동 · 회전에 강하게 만드는 기법

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | 학습 전 · 후 1층 필터 30장과 변화량을 그림, 필터별 상관계수 계산 · test 첫 숫자를 Conv → ReLU 에 통과시킨 특징 맵 30장 |
| MLP 와 비교 — 위치가 조금 바뀌면? | test 이미지를 2 px · 3 px 옮겨 Day 4 MLP(day4_mlp.npz)와 CNN 정확도 비교 |

**핵심 내용**

#### 1층 필터 — 생각보다 조금 바뀐다
> 1 epoch (600 step) 후 필터별 학습 전후 **상관계수 0.86 ~ 0.999**, 바뀐 양은 처음 크기의 약 **28%**
- **필터 시각화** → (30, 1, 5, 5) 를 5×5 그림 30장으로 — 학습 전 · 후 · 차이
- **왜** → 무작위 5×5 필터도 국소 패턴에 반응 — 학습의 대부분은 뒤쪽 Affine 이 그 반응을 **조합**하는 데 쓰였다
- **더 오래 · 더 깊게** → Sobel 같은 뚜렷한 에지 검출기가 나타난다 (Vision 과정)
- ⚠ "그럴 것이다" 가 아니라 **그려 보고 숫자로** 확인하는 습관

#### 특징 맵 — 부품 검출 결과
> 필터 30개 → 특징 맵 30장 — 맵마다 밝게 반응하는 부분(가로획 · 경계 · 배경)이 다르다
- **뒤쪽 Affine** → 30장의 부품 검출 결과를 조합해 숫자 판단
- **층이 깊어지면** → 부품 → 모양 → 물체 로 점점 추상적인 특징

#### 위치가 조금 바뀌면 — CNN vs MLP
> 가중치 공유 + 풀링 → 작은 이동에 덜 민감
- **예** → 2 px 이동: CNN **0.9119** vs Day 4 MLP 0.7975
- **3 px 이동** → CNN 도 **0.748** — 풀링이 한 번(2×2)뿐이라 한계
- **데이터 증강(augmentation)** → 학습 때 이미지를 일부러 옮기고 돌려 이동 · 회전에 강하게
- ⚠ 파라미터는 CNN(433,890)이 MLP(110,610)보다 많다 — 어디에 몰렸는지는 7교시

**기본 사용형태**
```python
W0, W1 = W_init.reshape(30, -1), cnn.layers[0].params["W"].reshape(30, -1)
corr = [np.corrcoef(a, b)[0, 1] for a, b in zip(W0, W1)]     # 0.86 ~ 0.999
fmap = cnn.layers[1].forward(cnn.layers[0].forward(X4_test[:1]))   # (1, 30, 24, 24)
shifted = np.roll(X4_test, 2, axis=3)                         # 예: 오른쪽으로 2 px
```

**연결**
- [NN D5·2교시 합성곱 연산](#/nn/5/2) — Sobel 에지 필터
- [NN D4·5교시 과적합](#/nn/4/5) — 데이터를 늘리는 것(증강)이 가장 확실

**상세 학습**

#### 이 교시에서 답하는 질문
CNN 은 0.981 을 냈다. 그런데 **무엇을** 배웠을까? 이 교시는 학습된 필터와 특징 맵을 그려 보고, "CNN 은 위치 변화에 강하다" 는 말이 사실인지 숫자로 확인한다.

#### 개념 1 · 1층 필터 — 생각보다 조금 바뀐다
1층 필터 (30, 1, 5, 5) 를 5×5 그림 30장으로 그려 학습 전 · 후를 비교한다. 1 epoch(600 step) 후 필터별 학습 전후 상관계수는 **0.86 ~ 0.999**, 바뀐 양은 처음 크기의 약 **28%** 다 — 대부분 모양이 그대로다.

| 관찰 | 해석 |
|---|---|
| 무작위 필터인데도 정확도 0.98 | 무작위 5×5 필터도 이웃 픽셀 밝기 차이(국소 패턴)에 반응한다 |
| 필터는 조금만 바뀜 | 학습의 대부분은 뒤쪽 Affine 이 그 반응을 **조합**하는 법을 배우는 데 쓰였다 |
| 변화량의 빨강 · 파랑이 한쪽으로 몰림 | 데이터가 필터를 경계 · 획 방향으로 밀기 시작한 흔적 |

더 오래, 더 깊게 학습하면 2교시 Sobel 같은 뚜렷한 에지 검출기가 나타난다 (Vision 과정).

#### 개념 2 · 특징 맵 — 부품 검출 결과
test 첫 숫자를 Conv → ReLU 에 통과시키면 특징 맵 30장 (24×24) 이 나온다. 맵마다 밝게 반응한 부분이 다르다 — 어떤 맵은 가로획, 어떤 맵은 오른쪽 경계, 어떤 맵은 배경. 뒤쪽 Affine 은 이 30장의 "부품 검출 결과" 를 조합해 숫자를 판단한다. 층이 깊어지면 부품 → 모양 → 물체로 점점 추상적인 특징을 잡는다.

#### 개념 3 · 위치가 조금 바뀌면 — CNN vs MLP
test 이미지를 오른쪽으로 몇 픽셀 옮겨 정확도를 잰다 (강의 실행 값).

| 모델 | 파라미터 | 원본 | 2 px 이동 |
|---|---|---|---|
| Day 4 MLP | 110,610 | 0.9757 | 0.7975 |
| CNN | 433,890 | 0.9810 | **0.9119** |

같은 필터를 모든 위치에 쓰고(가중치 공유) 풀링이 작은 이동을 흡수하므로 CNN 이 훨씬 덜 떨어진다. 하지만 3 px 이면 CNN 도 **0.748** — 풀링이 2×2 한 번뿐이라 한계가 있다. 실제 모델은 풀링을 여러 번 쓰고 **데이터 증강**(학습 때 일부러 옮기고 돌린 이미지를 섞기)으로 보완한다.

#### 코드로 확인
```python
W_before = mynn.Convolution(1, 30, 5, seed=1).params["W"]      # 같은 seed = 학습 전 필터
W_after = cnn.layers[0].params["W"]
dW = W_after - W_before
corr = [np.corrcoef(W_before[i].ravel(), W_after[i].ravel())[0, 1] for i in range(30)]
np.linalg.norm(dW) / np.linalg.norm(W_before)
```
- **무엇을 하나**: 같은 seed 로 학습 전 필터를 다시 만들어, 필터마다 학습 전후 상관계수와 전체 변화량 비율을 구한다.
- **어떤 개념**: 개념 1.
- **출력**: 상관계수 **0.86 ~ 0.999**, 변화량 비율 약 **0.28** (강의 실행 값)

```python
x1 = X4_test[:1]
fmap = cnn.layers[1].forward(cnn.layers[0].forward(x1, False), False)     # Conv → ReLU
shift_acc = {"cnn": [], "mlp": []}
for s in range(4):
    xs = np.roll(X4_test, s, axis=3)                                       # 너비 방향으로 s px
    shift_acc["cnn"].append(cnn.accuracy(xs, t_test))
    shift_acc["mlp"].append(mlp4.accuracy(xs.reshape(-1, 784), t_test))
```
- **무엇을 하나**: 첫 test 이미지의 특징 맵을 뽑고, test 전체를 0 ~ 3 px 옮겨 CNN 과 Day 4 MLP 의 정확도를 잰다.
- **어떤 개념**: 개념 2 · 3.
- **출력**: `fmap.shape` = `(1, 30, 24, 24)` · 2 px 에서 CNN **0.9119** vs MLP 0.7975 · 3 px 에서 CNN 0.748

#### 한 번에 정리
- 1층 필터는 1 epoch 동안 조금만 바뀐다 (상관 0.86 ~ 0.999) — 학습의 대부분은 Affine 의 조합.
- 특징 맵 = 필터마다 반응한 위치, Affine 이 이를 조합해 판단.
- CNN 은 작은 이동에 강하다 (2 px: 0.91 vs 0.80) — 3 px 부터는 증강이 필요.
- ⚠ "그럴 것이다" 가 아니라 그려 보고 숫자로 확인.

**연관 문제**
<<QA 6>>

### 7교시 · 모델 비용 계산

**키워드**: `파라미터 = 저장 공간`, `MAC = 시간 · 전력`, `활성화 메모리`, `× 4 바이트`, `파라미터는 Affine 에, 연산은 Conv 에`, `865,000 MAC`, `경량 CNN 20,490`, `depthwise separable`, `양자화 1/4`

**개요**: 온디바이스에서는 모델을 정확도만으로 고르지 않는다. ① **메모리**(파라미터 · 활성화)와 **연산량**(MAC — 시간 · 전력)을 층마다 공식으로 계산한다. ② SimpleConvNet 은 "파라미터는 Affine 에, 연산은 Conv 에" — 파라미터 433,890 개 중 432,100 개가 Affine1, Conv 는 780 개로 MAC 432,000 번. ③ 파라미터를 줄이면 저장 공간, MAC 을 줄이면 속도 · 전력 — **다른 문제**다. 경량화: Affine 입력 줄이기 · 양자화 · 층 합치기.

**핵심 용어**
- **MAC(multiply-accumulate)**: 곱하고 더하기 한 번. 연산량(시간 · 전력)의 단위 — 출력 값 하나마다 필터 크기만큼
- **활성화 메모리**: 실행 중 각 층 출력을 담는 메모리
- **경량 CNN**: 작은 필터 · 풀링을 여러 번 써서 Affine 입력을 줄인 CNN
- **depthwise separable conv**: 합성곱을 채널별 합성곱 + 1×1 합성곱으로 쪼개 MAC 을 줄이는 방법 (MobileNet)
- **양자화(quantization)**: float32 를 int8 등으로 줄여 메모리를 1/4 로

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `conv_cost()` | `conv_cost((C, H, W), FN, F, S=1, P=0)` | (params, MACs, 출력 shape) |
| `pool_cost()` | `pool_cost((C, H, W), P=2)` | (0, 0, (C, H//P, W//P)) |
| `affine_cost()` | `affine_cost(in_shape, n_out)` | (n_in·n_out + n_out, n_in·n_out, (n_out,)) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | 비용 함수로 SimpleConvNet 층별 파라미터 · MAC · 출력 shape 표 → 합계 433,890 · 865,000 |
| 경량 설계 — 줄일 곳이 보인다 | `cost_table` 로 경량 CNN(3×3 Conv 두 번 + 풀링 두 번) · 224×224 Conv 한 층의 비용을 SimpleConvNet 과 비교 |

**핵심 내용**

#### 모델 비용 — 파라미터 · MAC · 활성화
> 파라미터 → 저장 공간(× 4 바이트), **MAC**(곱하고 더하기 한 번) → 시간 · 전력, **활성화 메모리** → 실행 중 메모리
- **`conv_cost()`** → 파라미터 FN·C·FH·FW + FN · MAC FN·OH·OW·C·FH·FW · 활성화 FN·OH·OW
- **`affine_cost()`** → 파라미터 n_in·n_out + n_out · MAC n_in·n_out (파라미터 하나당 MAC 하나)
- **`pool_cost()`** → 파라미터 0 · MAC 0, 크기만 절반 (ReLU 도 0)
- **왜 Conv 는 MAC 이 많나** → 적은 가중치를 모든 출력 위치에서 다시 쓰므로

#### SimpleConvNet 의 비용
> "파라미터는 Affine 에, 연산은 Conv 에"
- **파라미터** → 합계 433,890 (float32 1.74 MB) 중 Affine 4320→100 이 432,100 = **99.6%**
- **MAC** → 합계 **865,000** — Conv 780개 파라미터로 432,000 (절반)
- **Affine 이 큰 이유** → Pooling 출력 4,320 개를 100 개에 모두 연결

#### 경량 설계
> Affine 에 들어가는 특징을 줄이면 파라미터 급감 — 하지만 저장 공간과 속도는 **다른 문제**
- **경량 CNN** → 3×3 Conv 16 · Pool · 3×3 Conv 32 · Pool · Affine 10 → 파라미터 **20,490** (21배 ↓), MAC 1,031,744 (오히려 ↑)
- **큰 이미지** → 224×224×3 의 3×3×64 Conv 한 층: 파라미터 1,792 · MAC **86,704,128** · 활성화 12.8 MB → MobileNet 의 **depthwise separable conv**
- **그 밖에** → int8 **양자화**(메모리 1/4) · BN folding(층 수 감소)
- **평가 과제 모델** → Conv 8 · Pool · Conv 16 · Pool · Affine 10 → 파라미터 **9,098**, MAC **290,080**

**기본 사용형태**
```python
def conv_cost(in_shape, FN, F, S=1, P=0):
    C, H, W = in_shape
    OH, OW = conv_output_size(H, F, S, P), conv_output_size(W, F, S, P)
    params = FN * C * F * F + FN
    macs   = FN * OH * OW * C * F * F
    return params, macs, (FN, OH, OW)

def pool_cost(in_shape, P=2):
    C, H, W = in_shape
    return 0, 0, (C, H // P, W // P)

def affine_cost(in_shape, n_out):
    n_in = int(np.prod(in_shape))
    return n_in * n_out + n_out, n_in * n_out, (n_out,)
```

**연결**
- [NN D2·5교시 행렬로 보는 순전파](#/nn/2/5) — 파라미터 수 = 모델 크기
- [NN D3·8교시 미니 프로젝트: MNIST 2층 MLP](#/nn/3/8) — int8 양자화 맛보기
- [NN D4·8교시 미니 프로젝트: 미니 프레임워크 mynn.py](#/nn/4/8) — BN folding

**상세 학습**

#### 이 교시에서 답하는 질문
6교시에서 CNN 의 파라미터(433,890)가 MLP(110,610)보다 많았다. 그런데 합성곱은 파라미터가 적다고 하지 않았나? 이 교시는 모델의 **비용** — 저장 공간 · 계산 시간 · 실행 메모리 — 을 층마다 공식으로 계산해, 비용이 어디에 몰려 있고 무엇을 줄여야 하는지 본다. 온디바이스(스마트폰 · Jetson)에서는 정확도만큼 중요한 기준이다.

#### 개념 1 · 비용 세 가지와 공식
| 지표 | 의미 | Convolution | Affine |
|---|---|---|---|
| 파라미터 | 저장 공간 (float32 × 4 바이트) | FN·C·FH·FW + FN | n_in·n_out + n_out |
| MAC | 곱하고 더하기 횟수 = 시간 · 전력 (1장) | FN·OH·OW·C·FH·FW | n_in·n_out |
| 활성화 | 실행 중 층 출력 메모리 | FN·OH·OW | n_out |

Pooling · ReLU 는 파라미터 0, MAC 0 (비교 연산은 무시). 합성곱은 가중치를 공유해 파라미터가 적지만, 같은 가중치를 **모든 출력 위치에서 다시 쓰므로** MAC 이 많다. Affine 은 파라미터 하나당 MAC 하나다.

#### 개념 2 · SimpleConvNet — 파라미터는 Affine 에, 연산은 Conv 에
![SimpleConvNet 의 층별 파라미터와 MAC — 파라미터는 Affine 4320→100 에 몰리고, MAC 은 Conv 와 Affine 이 절반씩](assets/fig/nn/d5-cost.svg)

| 층 | 파라미터 | MAC | 출력 |
|---|---|---|---|
| Conv 1→30, 5×5 | 780 | 432,000 | (30, 24, 24) |
| Pool 2×2 | 0 | 0 | (30, 12, 12) |
| Affine 4320→100 | **432,100** | 432,000 | (100,) |
| Affine 100→10 | 1,010 | 1,000 | (10,) |
| 합계 | 433,890 (1.74 MB) | 865,000 | |

Conv MAC = 30 × 24 × 24 × 1 × 5 × 5 = 432,000. Affine1 이 파라미터의 99.6% 인 이유는 Pooling 출력 4,320 개를 100 개에 모두 연결했기 때문이다.

#### 개념 3 · 경량 설계 — 저장과 속도는 다른 문제
| 모델 | 파라미터 | MAC (1장) | float32 |
|---|---|---|---|
| SimpleConvNet | 433,890 | 865,000 | 1.74 MB |
| 경량 CNN (3×3 Conv 16 · Pool · 3×3 Conv 32 · Pool · Affine 10) | **20,490** | 1,031,744 | 0.08 MB |
| 224×224×3 입력, 3×3×64 Conv 한 층 | 1,792 | **86,704,128** | 활성화 12.8 MB |

- 경량 CNN 은 풀링을 두 번 써서 Affine 입력을 32 × 7 × 7 = 1,568 로 줄였다 → 파라미터 21배 ↓. 그런데 MAC 은 오히려 1.2배 ↑ — 파라미터(저장)와 MAC(속도)은 **따로 줄여야** 한다.
- 큰 이미지에서는 파라미터보다 MAC 과 활성화 메모리가 병목 → MobileNet 의 **depthwise separable conv** (채널별 합성곱 + 1×1 합성곱).
- 그 밖에 int8 **양자화**(메모리 1/4, Day 3) · BN folding(층 수 감소, Day 4).

#### 코드로 확인
```python
def conv_cost(in_shape, FN, F, S=1, P=0):
    C, H, W = in_shape
    OH, OW = conv_output_size(H, F, S, P), conv_output_size(W, F, S, P)
    return FN * C * F * F + FN, FN * OH * OW * C * F * F, (FN, OH, OW)

def pool_cost(in_shape, P=2):
    C, H, W = in_shape
    return 0, 0, (C, H // P, W // P)

def affine_cost(in_shape, n_out):
    n_in = int(np.prod(in_shape))
    return n_in * n_out + n_out, n_in * n_out, (n_out,)
```
- **무엇을 하나**: 층 하나의 (파라미터, MAC, 출력 shape) 를 돌려준다. 출력 shape 를 다음 층 입력으로 넘기며 이어 계산한다.
- **어떤 개념**: 개념 1.
- **출력**: `conv_cost((1, 28, 28), 30, 5)` → `(780, 432000, (30, 24, 24))` · `affine_cost((30, 12, 12), 100)` → `(432100, 432000, (100,))`

```python
simple_spec = [("conv", 30, 5, 1, 0), ("pool", 2), ("affine", 100), ("affine", 10)]
lite_spec = [("conv", 16, 3, 1, 1), ("pool", 2), ("conv", 32, 3, 1, 1), ("pool", 2), ("affine", 10)]
P_simple, M_simple = show(cost_table((1, 28, 28), simple_spec), "SimpleConvNet")
P_lite, M_lite = show(cost_table((1, 28, 28), lite_spec), "경량 CNN")
```
- **무엇을 하나**: 층 목록(spec)으로 비용표를 만들어 두 모델을 비교한다.
- **어떤 개념**: 개념 2 · 3.
- **출력**: SimpleConvNet 433,890 · 865,000 / 경량 CNN **20,490** · 1,031,744 → 파라미터 21배 감소, MAC 1.2배

#### 한 번에 정리
- 파라미터 = 저장, MAC = 시간 · 전력, 활성화 = 실행 메모리 — 층마다 공식으로 계산.
- SimpleConvNet: 파라미터의 99.6% 는 Affine1, MAC 의 절반은 Conv.
- 경량화: Affine 입력 줄이기(풀링) · depthwise separable · 양자화 · BN folding.
- ⚠ 파라미터를 줄여도 MAC 은 늘 수 있다 — 저장과 속도는 다른 문제.

**연관 문제**
<<QA 7>>

### 8교시 · 평가 · 과정 정리

**키워드**: `평가 비중 50 · 30 · 20`, `필답 30문항`, `CNN 실습 평가`, `gradient check (새 설정)`, `저장 · 불러오기 test ≥ 0.98`, `경량 CNN 비용표`, `5일 정리`, `학습의 고리`

**개요**: 5일 과정의 평가와 정리. ① 평가는 일차별 미니 프로젝트 **50%** · 필답 **30%**(객관식 · shape 계산 · 역전파 유도) · CNN 실습 **20%**. ② 실습은 새 설정의 Convolution · Pooling gradient check, 학습한 CNN 저장 → 새 모델로 불러와 test ≥ 0.98, 경량 CNN 비용표를 층마다 직접 계산, 결과 발표. ③ 5일 동안 첫날의 **학습의 고리**(모델 → 손실 → 기울기 → 갱신)를 처음부터 직접 만들었다.

**핵심 용어**
- **필답 평가**: Google Forms 30문항 — 객관식 · shape 계산 · 역전파 유도
- **CNN 실습 평가**: 노트북 과제 1~4 + 채점 셀
- **학습의 고리**: 모델 → 손실 → 기울기 → 갱신 — 5일 과정 전체의 뼈대 (Day 1)

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 과제 1 — gradient check | 입력 (2, 2, 9, 9) · 채널 2 · 필터 3개 · 4×4 · S2 · P1 Convolution 과 3×3 Max Pooling(S3) 을 `layer_check` 로 검사 |
| 과제 2 — 저장 · 불러오기 · 정확도 | day5_cnn.npz 를 다른 seed 로 만든 SimpleConvNet 에 불러와 test 0.98 이상 확인, mynn.py 와 가중치를 Drive 에 복사 |
| 과제 3 — 경량 CNN 비용표 | `cost_table` 없이 층마다 `conv_cost` · `pool_cost` · `affine_cost` 를 직접 호출 → 파라미터 **9,098** · MAC **290,080** |

**핵심 내용**

#### 평가 구성
> 미니 프로젝트 50% · **필답 평가** 30% · **CNN 실습 평가** 20%
- **필답** → 30문항 — 객관식 · shape 계산 · 역전파 유도
- **CNN 실습** → gradient check(입력 (2, 2, 9, 9), 필터 3개 4×4 S2 P1 · 3×3 Pooling S3) 30 · 저장/불러오기 test ≥ 0.98 30 · 비용표 30 · 발표 10
- **발표 정리** → 내 CNN 과 Day 3 · 4 MLP 비교 · 필터 관찰 하나 · 온디바이스라면 무엇을 줄일지(비용표 근거)

#### 5일 정리 — 학습의 고리
> 모델 → 손실 → 기울기 → 갱신 — 그 고리를 처음부터 직접 만들고, 빠르고 안정적으로
- **Day 1 · 2** → NumPy · 미분 · 경사하강 / 뉴런 · 순전파 · 손실
- **Day 3 · 4** → 역전파로 기울기를 빠르게 / 옵티마이저 · 초기화 · BN · 규제 → mynn.py
- **Day 5** → 합성곱 · 풀링 → CNN, 모델 비용 → 다음은 Vision 과정 (깊은 CNN · 증강 · 양자화)

**기본 사용형태**
```python
p, m = 0, 0
for cost in [conv_cost((1, 28, 28), 8, 3, 1, 1), pool_cost((8, 28, 28)),
             conv_cost((8, 14, 14), 16, 3, 1, 1), pool_cost((16, 14, 14)),
             affine_cost((16, 7, 7), 10)]:
    p, m = p + cost[0], m + cost[1]
print(p, m)                                     # 9098 290080
```

**연결**
- [NN D1·1교시 과정 안내 · Colab 사용법](#/nn/1/1) — 첫날의 "학습 = argmin L(θ)" 와 학습의 고리
- [NN D3·6교시 계층 조립과 Gradient check](#/nn/3/6) — gradient check 방법

**상세 학습**

#### 이 교시에서 답하는 질문
5일 과정의 마지막 교시다. 평가가 어떻게 구성되는지, 그리고 5일 동안 만든 것이 하나의 그림으로 어떻게 이어지는지 정리한다.

#### 개념 1 · 평가 구성
| 평가 | 비중 | 방법 |
|---|---|---|
| 미니 프로젝트 (Day 1 ~ 4) | 50% | 각 노트북 채점 셀 |
| 필답 | 30% | 30문항 — 객관식 · shape 계산 · 역전파 유도 |
| CNN 실습 | 20% | gradient check 30 · 저장/불러오기 30 · 비용표 30 · 발표 10 |

실습 과제는 4 · 5 · 7교시를 새 설정으로 다시 해 보는 것이다.
- **gradient check**: 입력 (2, 2, 9, 9), 채널 2 · 필터 3개 · 4×4 · S2 · P1 합성곱 (출력 (2, 3, 4, 4)) 과 3×3 Max Pooling S3 (출력 (2, 2, 3, 3)).
- **저장 · 불러오기**: day5_cnn.npz 를 다른 seed 로 만든 SimpleConvNet 에 불러와 test ≥ 0.98.
- **비용표**: Conv 8 (3×3, P1) · Pool · Conv 16 · Pool · Affine 10 을 층마다 직접 계산 → 파라미터 **9,098**, MAC **290,080**.

#### 개념 2 · 5일 정리 — 학습의 고리
Day 1 에 본 **학습의 고리**: 모델 → 손실 → 기울기 → 갱신. 5일 동안 이 고리를 처음부터 직접 만들고, 빠르고 안정적으로 돌게 했다.

| Day | 만든 것 | 고리에서 |
|---|---|---|
| 1 | NumPy · 미분 · 경사하강 | 도구 · 갱신 |
| 2 | 뉴런 · 순전파 · 손실 | 모델 · 손실 |
| 3 | 계산 그래프 · 역전파 → MNIST | 기울기 (빠르게) |
| 4 | 옵티마이저 · 초기화 · BN · 규제 → mynn.py | 갱신 · 안정적 학습 |
| 5 | 합성곱 · 풀링 → CNN, 모델 비용 | 더 좋은 모델 · 온디바이스 |

다음 Vision 과정에서는 깊은 CNN · 데이터 증강 · 양자화 · TensorRT 로 이어진다.

#### 코드로 확인
```python
c1 = conv_cost((1, 28, 28), 8, 3, 1, 1)
p1 = pool_cost(c1[2])
c2 = conv_cost(p1[2], 16, 3, 1, 1)
p2 = pool_cost(c2[2])
a1 = affine_cost(p2[2], 10)
eval_params = c1[0] + c2[0] + a1[0]
eval_macs = c1[1] + c2[1] + a1[1]
```
- **무엇을 하나**: 평가 과제 3 — 경량 CNN 의 비용을 층마다 직접 호출해 더한다.
- **어떤 개념**: 7교시 개념 1 — 앞 층의 출력 shape 가 다음 층 입력.
- **출력**: shape (8, 28, 28) → (8, 14, 14) → (16, 14, 14) → (16, 7, 7) → (10,) · 파라미터 **9,098** · MAC **290,080** · float32 35.5 KB

```python
rng8 = np.random.default_rng(5)
x8 = rng8.normal(size=(2, 2, 9, 9))
eval_conv = layer_check(mynn.Convolution(2, 3, 4, stride=2, pad=1, seed=7), x8.copy(), rng8)
eval_pool = layer_check(mynn.Pooling(3, 3, stride=3), x8.copy(), rng8)
```
- **무엇을 하나**: 평가 과제 1 — 새 설정의 합성곱 · 풀링을 gradient check 한다.
- **어떤 개념**: 4교시 개념 3.
- **출력**: 상대 오차 모두 약 **1e-12** (numpy 로 직접 실행) → 통과

#### 한 번에 정리
- 평가: 미니 프로젝트 50 · 필답 30 · CNN 실습 20.
- 실습 = gradient check · 저장/불러오기 test ≥ 0.98 · 비용표(9,098 · 290,080) · 발표.
- 5일 = 학습의 고리(모델 → 손실 → 기울기 → 갱신)를 직접 만들고 빠르고 안정적으로.
- ⚠ 비용표는 `cost_table` 없이 층마다 직접 — 앞 층 출력 shape 를 다음 층에 넘긴다.

**연관 문제**
<<QA 8>>
