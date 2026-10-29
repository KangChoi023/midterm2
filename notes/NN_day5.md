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

#### 펼치면 잃는 것 — 위치 관계
> 위아래로 붙은 두 픽셀이 펼치면 **28칸** 떨어진다 — Affine 은 그 관계를 모른다
- **고정 순열로 뒤섞은 MNIST** → MLP 정확도 그대로 (0.9436 vs 0.9455) — 위치 정보를 안 쓴다는 증거
- **그래서 MLP 는** → 숫자 모양에 대한 "상식"을 6만 장에서 처음부터 배워야 함
##### 자세히
```text
 image 28×28              flattened 784
 row 0  [ a  … ]          [ a … (28 cells) … b … ]
 row 1  [ b  … ]           ▲                 ▲
 a and b are neighbors     index 0           index 28  → 28 apart
```

#### 파라미터 폭발
> 완전연결층 가중치 = 입력 수 × 출력 수 — 이미지가 커지면 감당 불가
| 층 | 파라미터 |
|---|---|
| FC 28×28 → 100 | 78,400 |
| FC 224×224×3 → 1000 | **150,528,000** (float32 약 **600 MB**) |
| Conv 3×3, RGB → 64 필터 | **1,792** (= 64 × 3 × 3 × 3 + 64) |
- **온디바이스 불가능** → 첫 층만 600 MB
- **Conv 는 이미지 크기와 무관** → 224×224 든 28×28 이든 1,792

#### 합성곱의 두 아이디어
> 지역성 + 가중치 공유
| 아이디어 | 뜻 | 효과 |
|---|---|---|
| 지역성 | 출력 하나는 입력의 작은 영역(3×3)만 본다 | 이웃 관계를 구조적으로 사용 |
| 가중치 공유 | 같은 필터를 모든 위치에 쓴다 | 파라미터 수가 이미지 크기와 무관 · 어디서든 같은 특징 검출 |

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

#### 합성곱 = 미끄러뜨리며 곱해서 더하기
> out[i, j] = Σ in[i·S + y, j·S + x] · K[y, x] (+ b)
- **겹치는 칸끼리 곱해 모두 더함** → 출력 한 칸
- **cross-correlation** → 수학의 합성곱은 필터를 뒤집지만, 신경망은 안 뒤집음 (필터가 학습되니 차이 없음)
- **Sobel 가로 에지** → [[−1, −2, −1], [0, 0, 0], [1, 2, 1]] — 위아래 밝기 차이에 반응 (세로 에지는 전치)
- **CNN** → 이런 필터를 사람이 설계하지 않고 **학습** (6교시)
##### 자세히
```text
 input 4×4          filter 3×3        output 2×2  (P = 0, S = 1)
 ┌─────────┐        ┌───────┐
 │ ▓ ▓ ▓ . │        │ 1 0 1 │         out[0,0] = Σ (▓ window × filter)
 │ ▓ ▓ ▓ . │   ×    │ 0 1 0 │   ──►   slide right → out[0,1]
 │ ▓ ▓ ▓ . │        │ 1 0 1 │         slide down  → out[1,0], out[1,1]
 │ . . . . │        └───────┘
 └─────────┘
```

#### 출력 크기 — 패딩과 스트라이드
> OH = (H + 2P − FH) / S + 1 (나누어떨어지지 않으면 내림 `//`)
| H | F | S | P | 출력 | 의미 |
|---|---|---|---|---|---|
| 28 | 5 | 1 | 0 | **24** | 패딩 없으면 줄어듦 |
| 28 | 3 | 1 | 1 | **28** | 3×3 + P1 = 크기 유지 |
| 28 | 3 | 2 | 1 | **14** | 스트라이드 2 = 절반 |
| 224 | 7 | 2 | 3 | **112** | |
| 7 | 3 | 2 | 0 | **3** | |
- **패딩** → 가장자리 픽셀도 필터 가운데에 올 수 있게 → 정보 보존 · 크기 유지
- **스트라이드** → 건너뛰며 계산 → 출력이 작아짐
- ⚠ 출력 크기에 `/` 를 쓰면 실수 → 정수 나눗셈 **`//`**

#### shape 규칙 — 다채널 · 다중 필터 · 배치
> 입력 (N, C, H, W) · 필터 (FN, C, FH, FW) · 출력 (N, FN, OH, OW)
- **필터 하나** → (C, FH, FW) 블록이 모든 채널을 덮음 → 채널마다 곱한 뒤 **더해서** 출력 한 장
- **필터 FN 개** → 출력 FN 장 (출력 채널 = 필터 수) · 필터마다 **편향 하나**
- **배치** → N 장을 한꺼번에
- ⚠ 다채널 결과를 채널마다 따로 내지 않는다 — 더해서 **한 장**
- ⚠ `np.pad` 는 높이 · 너비만: `[(0,0), (0,0), (P,P), (P,P)]`
##### 자세히
```text
 input (C=3)       filter f (3, FH, FW)        output channel f
 R ─┐               R-slice ─┐
 G ─┼── × ──────►   G-slice ─┼── sum all + b_f ──► 1 map
 B ─┘               B-slice ─┘
 FN filters → FN maps → output (FN, OH, OW)
```

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

#### 합성곱 = 행렬곱 한 번
> 영역 하나 = 행 하나, 필터 하나 = 열 하나 → col @ col_W + b
| 단계 | shape |
|---|---|
| 입력 x | (N, C, H, W) |
| `im2col(x)` → col | (N·OH·OW, C·FH·FW) |
| `W.reshape(FN, -1).T` → col_W | (C·FH·FW, FN) |
| `col @ col_W + b` | (N·OH·OW, FN) — Affine 과 같은 식 |
| `.reshape(N, OH, OW, FN).transpose(0, 3, 1, 2)` | (N, FN, OH, OW) |
- **예 MNIST 1장, 5×5 필터 30개** → col (576, 25) @ col_W (25, 30) = (576, 30) → (1, 30, 24, 24)
- **transpose(0, 3, 1, 2)** → 채널 축 FN 을 앞으로 (N, OH, OW, FN) → (N, FN, OH, OW)
- ⚠ `W.reshape(FN, -1)` 에서 `.T` 빠뜨림 → col_W 는 (C·FH·FW, **FN**)
##### 자세히
```text
 x (1, 1, 28, 28)                 col (576, 25)            col_W (25, 30)
 window at (0,0) ──flatten──►    row 0   [ 25 values ]    column f = filter f
 window at (0,1) ──flatten──►    row 1   [ 25 values ]
 …                                …
 window at (23,23) ─────────►    row 575 [ 25 values ]
                       col @ col_W  =  (576, 30)  →  reshape → (1, 30, 24, 24)
```

#### 구현 요령 — 루프는 필터 크기만큼
> 영역을 하나씩 도는 대신 필터 안의 위치 (y, x) 를 돌며, 모든 출력 위치의 값을 한꺼번에 슬라이싱
- **루프 횟수** → FH·FW 번 = 5×5 면 **25**번뿐 (출력 위치 수와 무관)
- **슬라이싱** → `img[:, :, y : y + S·OH : S, x : x + S·OW : S]` 로 (N, C, OH, OW) 를 한 번에
- **벡터화** → Day 1 "파이썬 루프 대신 NumPy 한 번"

#### 속도 ↔ 메모리 교환
> 겹치는 영역을 중복 복사 → 메모리 약 FH·FW / S² 배, 대신 행렬곱 한 번 → 백 배 이상 빠름
- **MNIST 5×5, S = 1** → 25 × 24² / 28² ≈ **18.4배** 메모리
- **실제 라이브러리** → 비슷한 방법 사용 (행렬곱은 CPU · GPU · NPU 가 가장 잘하는 연산)
- **온디바이스** → 메모리가 빠듯하면 이 교환 비율이 설계 기준

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

#### Convolution 역전파 = Affine 역전파 + col2im
> forward 가 col @ col_W + b 이므로 backward 도 Affine 식 그대로 (X 자리에 col, W 자리에 col_W)
| 기울기 | Affine (Day 3) | Convolution |
|---|---|---|
| dW | `X.T @ dout` | `(col.T @ dout).T.reshape(FN, C, FH, FW)` |
| db | `dout.sum(axis=0)` | `dout.sum(axis=0)` (dout 을 (N·OH·OW, FN) 으로 펼친 뒤) |
| dX | `dout @ W.T` | `dcol = dout @ col_W.T` → **col2im** → (N, C, H, W) |
- **col2im 은 더한다** → 한 입력 픽셀이 여러 영역(여러 col 행)에 복사되어 쓰였으므로 기울기도 그 자리들에서 온 것을 **모두 더함** (Day 3 "복사 노드의 역전파는 더하기")
- **dout 펼치기** → (N, FN, OH, OW) → `transpose(0, 2, 3, 1)` → (N, OH, OW, FN) → `reshape(-1, FN)`
- ⚠ backward 에서 `reshape(-1, FN)` 만 하고 transpose 를 빠뜨리면 채널과 위치가 섞임
- ⚠ col2im 에서 겹치는 자리를 `=` 로 덮어씀 → **`+=`**
##### 자세히
```text
 forward    x ──im2col──► col ──@ col_W + b──► (N·OH·OW, FN) ──reshape/transpose──► out
 backward   dx ◄──col2im── dcol ◄──@ col_W.T── dout (N·OH·OW, FN) ◄──transpose/reshape── dout
                                dW = col.T @ dout,  db = sum(dout)
```

#### Max Pooling — 최댓값만, 기울기도 그 자리로만
> 2×2 영역 최댓값만 남겨 크기 절반 (stride 2) — 학습할 파라미터 **없음**
- **forward** → 영역마다 최댓값 + `arg_max` 저장 · 채널 수 그대로
- **backward** → arg_max 자리로만 dout, 나머지 0 (ReLU mask 처럼)
- **(N, 30, 24, 24)** → **(N, 30, 12, 12)**
- **효과** → 크기 절반 · 영역 안에서 조금 움직여도 최댓값은 그대로 → 작은 이동에 둔감
- ⚠ Pooling 에 params 를 기대하지 말 것 — 없다 (`Sequential.params()` 에도 안 나옴)
##### 자세히
```text
 in (4×4)              max pool 2×2, S 2     out (2×2)
 ┌─────┬─────┐
 │ 1 3 │ 2 0 │                               ┌───┬───┐
 │ 4 2 │ 1 5 │         ──►                   │ 4 │ 5 │
 ├─────┼─────┤                               ├───┼───┤
 │ 0 1 │ 3 3 │                               │ 6 │ 3 │
 │ 6 2 │ 1 2 │                               └───┴───┘
 └─────┴─────┘   backward: dout goes only to 4, 5, 6, 3 positions
```

#### gradient check — 새 계층은 반드시
> L = Σ (forward(x) ⊙ dout) 의 수치 미분과 backward 비교 (Day 3 4교시 방법)
- **`layer_check(layer, x, rng)`** → dx · dW · db 의 상대 오차 — Convolution · Pooling 모두 통과
- **평가 과제** → 입력 (2, 2, 9, 9), 채널 2 · 필터 3개 · 4×4 · S2 · P1 Convolution (출력 4×4) + 3×3 Pooling S3 (출력 1×1)

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

#### 구조와 단계별 shape (배치 100)
> Conv(1→30, 5×5) → ReLU → Pool(2×2) → Affine(4320→100) → ReLU → Affine(100→10) → Softmax
| 단계 | shape | 계산 |
|---|---|---|
| input | (100, 1, 28, 28) | `reshape(-1, 1, 28, 28)` |
| conv → ReLU | (100, **30**, 24, 24) | (28 − 5)/1 + 1 = 24 |
| pool | (100, 30, **12**, 12) | 24 / 2 |
| fc1 → ReLU | (100, 100) | 30·12·12 = **4,320** 입력 |
| fc2 | (100, 10) | 점수 → SoftmaxWithLoss |
- **Affine 이 (N, C, H, W) 도 받음** → 내부에서 펼침 (Day 4 에서 만들어 둠)
- **파라미터** → 780 + 432,100 + 1,010 = **433,890**
- ⚠ 입력을 (N, 784) 로 넣으면 Convolution 이 shape 오류 → **(N, 1, 28, 28)** (채널 축 1)
##### 자세히
```text
 (1, 28, 28) ─Conv 5×5 ×30─► (30, 24, 24) ─ReLU─► ─Pool 2×2─► (30, 12, 12)
      ─flatten─► 4320 ─Affine─► 100 ─ReLU─► ─Affine─► 10 ─Softmax─► p(0…9)
```

#### 계층 규약의 힘 — 프레임워크를 고치지 않는다
> Sequential · Trainer 그대로, 계층 리스트만 바꿈
- **Day 4 MLP 와 차이** → 리스트 앞에 `Convolution, Relu, Pooling` 이 들어간 것뿐
- **gradient()** → 모든 계층의 backward 를 반대 순서로 — Convolution 이든 Affine 이든 같은 방식
- **PyTorch · Keras** → 같은 원리 (nn.Sequential, model.fit)

#### 3일의 모델 비교
> 공간 구조를 쓰는 모델이 더 적은 학습으로 더 잘 맞힌다
| 모델 | 학습 | test |
|---|---|---|
| Day 3 784-100-10 MLP | SGD 10 epoch | 0.9694 |
| Day 4 5층 MLP + BN + He | Adam 5 epoch | 0.9757 |
| Day 5 SimpleConvNet | Adam **1** epoch | **0.9810** |

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
> 1 epoch (Adam 0.001, 600 step) 후 필터별 상관 **0.86 ~ 0.999**, 바뀐 양은 처음 크기의 약 **28%**
- **무작위 필터도 쓸모** → 5×5 무작위 필터도 이웃 픽셀 밝기 차이(국소 패턴)에 반응
- **학습의 대부분** → 뒤쪽 Affine 이 그 반응들을 조합하는 법을 배움
- **변화량의 무늬** → 빨강 · 파랑이 한쪽으로 몰림 = 데이터가 필터를 경계 · 획 방향으로 밀기 시작한 흔적
- **더 오래 · 더 깊게** → 2교시 Sobel 같은 뚜렷한 에지 검출기가 나타남 (Vision 과정)
- **습관** → "그럴 것이다" 가 아니라 **그려 보고 숫자로** 확인

#### 특징 맵 — 부품 검출 결과
> 필터 30개 → 특징 맵 30장, 맵마다 밝게 반응하는 부분이 다르다
- **어떤 맵** → 가로획 · 오른쪽 경계 · 배경 등에 반응
- **뒤쪽 Affine** → 30장의 "부품 검출 결과"를 조합해 숫자 판단
- **층이 깊어지면** → 부품 → 모양 → 물체 로 점점 추상적인 특징 (Vision 과정의 깊은 CNN)
##### 자세히
```text
 digit "7" ──Conv(30 filters)──► 30 maps ──► Affine combines
                                 one map: top stroke lights up
                                 another: right edge lights up
                                 another: background lights up
```

#### 위치가 조금 바뀌면 — CNN vs MLP
> 가중치 공유 + 풀링 → 작은 이동에 덜 민감
| 모델 | 파라미터 | 학습 | test | 2 px 이동 |
|---|---|---|---|---|
| Day 4 MLP | 110,610 | 5 epoch | 0.9757 | 0.7975 |
| CNN | 433,890 | 1 epoch | 0.9810 | **0.9119** |
- **3 px 이동** → CNN 도 **0.748** — 풀링이 한 번(2×2)뿐이라 이동 둔감성이 제한적
- **실제 모델** → 풀링 여러 번 + **데이터 증강**으로 보완
- **파라미터는 CNN 이 더 많다** → 어디에 몰려 있는지 7교시에서 계산

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

#### 비용 공식 — 저장 · 시간 · 실행 메모리
> 파라미터 → 저장 공간 (× 4 바이트), MAC → 시간 · 전력, 활성화 → 실행 중 메모리
| 지표 | Convolution | Affine | 의미 |
|---|---|---|---|
| 파라미터 | FN·C·FH·FW + FN | n_in·n_out + n_out | 저장 공간 (float32 × 4 바이트) |
| MAC | FN·OH·OW·C·FH·FW | n_in·n_out | 시간 · 전력 (1장 기준) |
| 활성화 | FN·OH·OW | n_out | 실행 중 메모리 |
- **합성곱** → 가중치를 공유해 파라미터는 **적지만**, 같은 가중치를 모든 출력 위치에서 다시 쓰니 MAC 은 **많다**
- **Affine** → 파라미터 하나당 MAC 하나 (1장 기준)
- **Pooling · ReLU** → 파라미터 0, MAC 0 (비교 연산은 무시)

#### SimpleConvNet 의 비용
> "파라미터는 Affine 에, 연산은 Conv 에"
| 층 | 파라미터 | MAC | 출력 |
|---|---|---|---|
| Conv 1→30, 5×5 | 780 | 432,000 | (30, 24, 24) |
| Pool 2×2 | 0 | 0 | (30, 12, 12) |
| Affine 4320→100 | **432,100** | 432,000 | (100,) |
| Affine 100→10 | 1,010 | 1,000 | (10,) |
| 합계 | **433,890** (1.74 MB) | **865,000** | |
- **Affine1** → 파라미터의 **99.6%** — Pooling 출력 4,320 개를 100 개에 모두 연결
- **Conv** → 파라미터 780 개로 MAC 의 **절반**
##### 자세히
```text
 params   Conv ▏780        Affine1 ████████████████████ 432,100
 MACs     Conv ██████████ 432,000   Affine1 ██████████ 432,000
```

#### 경량 설계 — 줄일 곳이 보인다
> Affine 에 들어가는 특징을 줄이면 파라미터가 급감 — 큰 이미지에서는 MAC · 활성화가 병목
| 모델 | 파라미터 | MAC (1장) | float32 |
|---|---|---|---|
| SimpleConvNet | 433,890 | 865,000 | 1.74 MB |
| 경량 CNN (3×3 Conv 16 · Pool · 3×3 Conv 32 · Pool · Affine 10) | **20,490** (21배 ↓) | 1,031,744 | 0.08 MB |
| 224×224×3, 3×3×64 Conv 한 층 | 1,792 | **86,704,128** | 활성화 12.8 MB |
- **경량 CNN** → 파라미터는 21배 줄었지만 MAC 은 오히려 늘었다 → 저장과 속도는 **다른 문제**
- **큰 이미지** → 파라미터보다 MAC 과 활성화 메모리가 병목 → MobileNet 의 **depthwise separable conv**
- **int8 양자화** → 메모리 1/4 (float32 대비, Day 3) · **BN folding** → 층 수 감소 (Day 4)
- **평가 과제 모델** → Conv 8 (3×3, P1) · Pool · Conv 16 · Pool · Affine 10 → 파라미터 **9,098**, MAC **290,080**
- **본격적으로** → TensorRT · Quantization · Pruning 은 Vision 과정 이후

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
> 미니 프로젝트 50 · 필답 30 · CNN 실습 20
| 평가 | 비중 | 방법 |
|---|---|---|
| 미니 프로젝트 (Day 1~4) | 50% | 각 노트북 채점 셀 |
| 필답 | 30% | 30문항 — 객관식 · shape 계산 · 역전파 유도 |
| CNN 실습 | 20% | gradient check · 저장/불러오기 test ≥ 0.98 · 비용표 · 발표 |
- **실습 과제 배점** → gradient check 30 · 저장/불러오기 30 · 비용표 30 · 발표 정리 10
- **발표 정리 (3~5줄)** → 내 CNN 과 Day 3 · 4 MLP 비교와 이유, 필터 · 특징 맵 관찰 하나, 스마트폰 / Jetson 에 올린다면 무엇을 줄일지(비용표 근거)

#### 5일 정리 — 학습의 고리를 직접 만들었다
> 모델 → 손실 → 기울기 → 갱신, 그리고 그 고리를 빠르고 안정적으로
| Day | 만든 것 | 고리에서 |
|---|---|---|
| 1 | NumPy · 미분 · 경사하강 | 도구 · 갱신 |
| 2 | 뉴런 · 순전파 · 손실 | 모델 · 손실 |
| 3 | 계산 그래프 · 역전파 → MNIST | 기울기 (빠르게) |
| 4 | 옵티마이저 · 초기화 · BN · 규제 → mynn.py | 갱신 · 안정적 학습 |
| 5 | 합성곱 · 풀링 → CNN, 모델 비용 | 더 좋은 모델 · 온디바이스 |
- **다음 과정 (Vision)** → 깊은 CNN · 데이터 증강 · 양자화 · TensorRT

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

**연관 문제**
<<QA 8>>
