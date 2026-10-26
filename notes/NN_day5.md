## Day 5 — CNN 원리

### 1교시 · 복습 퀴즈 · 완전연결층의 한계

**키워드**: `완전연결층`, `위치 관계`, `픽셀 뒤섞기`, `파라미터 폭발`, `지역성`, `가중치 공유`

**개요**: Affine(완전연결)은 이미지를 784 벡터로 펼쳐 "어느 픽셀이 어느 옆인지"를 쓰지 않는다 — 픽셀을 모든 이미지에서 똑같이 뒤섞어도 정확도가 같다(0.9436 vs 0.9455). 큰 이미지에서는 파라미터가 폭발. 합성곱 = 지역성 + 가중치 공유.

**핵심 용어**
- **완전연결층(fully connected, Affine)**: 모든 입력과 모든 출력이 각자 가중치로 연결된 층
- **지역성(locality)**: 출력 하나가 입력의 작은 영역(예 3×3)만 보는 것 — 이웃 관계를 구조적으로 사용
- **가중치 공유(weight sharing)**: 같은 필터를 이미지의 모든 위치에 쓰는 것 — 파라미터 수가 이미지 크기와 무관

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `fc_weights()` | `n_in * n_out` | 완전연결층 가중치 수 |
| `conv_params()` | `FN * C * F * F + FN` | 합성곱 층 파라미터 수 (필터 + 편향) |

**핵심 내용**

#### 펼치면 잃는 것
> 위아래로 붙은 두 픽셀이 펼치면 28칸 떨어진다 — Affine 은 그 관계를 모른다
- **고정 순열로 뒤섞은 MNIST** → MLP 정확도 그대로 — 위치 정보를 안 쓴다는 증거

#### 파라미터 폭발
| 층 | 파라미터 |
|---|---|
| FC 28×28 → 100 | 78,400 |
| FC 224×224×3 → 1000 | 150,528,000 (float32 약 600 MB) |
| Conv 3×3, RGB → 64 | 1,792 |
- **온디바이스 불가능** → 첫 층만 600 MB

### 2교시 · 합성곱 연산

**키워드**: `합성곱`, `필터(커널)`, `패딩 P`, `스트라이드 S`, `출력 크기 공식`, `(N, C, H, W)`, `(FN, C, FH, FW)`

**개요**: 합성곱은 작은 필터를 이미지 위에서 미끄러뜨리며 겹치는 칸끼리 곱해 더한 값을 출력에 적는 연산(신경망에서는 뒤집지 않는 cross-correlation). 필터 하나가 입력의 모든 채널을 덮어 출력 한 장, 필터 FN 개면 FN 장.

**핵심 용어**
- **합성곱(convolution)**: 필터를 슬라이딩하며 원소별 곱의 합을 계산하는 연산
- **필터(filter, 커널)**: 학습되는 작은 가중치 블록 (C, FH, FW)
- **패딩(padding, P)**: 입력 둘레에 0 을 P 줄 덧대기 — 가장자리 정보를 살리고 출력 크기 유지
- **스트라이드(stride, S)**: 필터를 한 번에 S 칸씩 이동 — 출력이 작아진다
- **특징 맵(feature map)**: 필터 하나의 출력 — 그 필터가 반응한 위치

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `conv_output_size()` | `(H + 2 * P - F) // S + 1` | 출력 크기 |
| `conv_naive()` | `conv_naive(x, W, b, stride=1, pad=0)` | 4중 for 루프 합성곱 (배치 · 다채널 · 다중 필터) |

**핵심 내용**

#### 출력 크기
> OH = (H + 2P − FH) / S + 1
| H | F | S | P | 출력 |
|---|---|---|---|---|
| 28 | 5 | 1 | 0 | 24 |
| 28 | 3 | 1 | 1 | 28 (크기 유지) |
| 28 | 3 | 2 | 1 | 14 |
| 224 | 7 | 2 | 3 | 112 |
| 7 | 3 | 2 | 0 | 3 |

#### shape 규칙
> 입력 (N, C, H, W) · 필터 (FN, C, FH, FW) · 출력 (N, FN, OH, OW)
- **필터 하나** → 모든 채널을 곱한 뒤 **더해서** 출력 한 장 · 필터마다 편향 하나
- **필터는 사람이 설계하지 않고 학습**
- ⚠ 출력 크기는 정수 나눗셈 `//`
- ⚠ 다채널 결과를 채널마다 따로 내지 않는다 — 더해서 한 장
- ⚠ `np.pad` 는 높이 · 너비만: `[(0,0), (0,0), (P,P), (P,P)]`

**기본 사용형태**
```python
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

### 3교시 · im2col: 합성곱을 행렬곱으로

**키워드**: `im2col`, `영역 하나 = 행 하나`, `필터 하나 = 열 하나`, `col @ col_W + b`, `reshape · transpose`

**개요**: 4중 for 루프는 느리다. im2col 은 필터가 닿는 영역 하나하나를 행 하나로 펼친 큰 행렬을 만든다 — 그러면 합성곱이 Affine 과 같은 X @ W + b 행렬곱 한 번. 속도 ↑↑, 메모리 ↑(겹친 영역만큼 중복).

**핵심 용어**
- **im2col(image to column)**: 합성곱 영역을 행으로 펼쳐 합성곱을 행렬곱으로 바꾸는 변환

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `im2col()` | `im2col(x, FH, FW, stride=1, pad=0)` | (N, C, H, W) → (N·OH·OW, C·FH·FW) |
| `col2im()` | `col2im(col, input_shape, FH, FW, stride, pad)` | im2col 의 반대 — 겹치는 자리는 더함 (역전파용) |

**핵심 내용**

#### 합성곱 = 행렬곱
| 단계 | shape |
|---|---|
| 입력 x | (N, C, H, W) |
| `im2col(x)` → col | (N·OH·OW, C·FH·FW) — 영역 하나 = 행 하나 |
| `W.reshape(FN, -1).T` → col_W | (C·FH·FW, FN) — 필터 하나 = 열 하나 |
| `col @ col_W + b` | (N·OH·OW, FN) |
| `.reshape(N, OH, OW, FN).transpose(0, 3, 1, 2)` | (N, FN, OH, OW) |
- **구현 요령** → 영역을 하나씩 도는 대신 필터 안의 위치 (y, x) 를 돌며 모든 출력 위치의 값을 한꺼번에 슬라이싱

**연결**
- [NN D1·4교시 벡터화](#/nn/1/4) — 루프를 행렬곱 한 번으로

### 4교시 · Convolution · Pooling 계층 → mynn.py

**키워드**: `Convolution 계층`, `Affine 그대로`, `col2im`, `Max Pooling`, `arg_max`, `파라미터 없음`

**개요**: Convolution forward 는 col @ col_W + b — Affine 의 X @ W + b 와 같으니 backward 도 Affine 그대로, 마지막에 col2im 으로 되돌린다(겹친 자리는 더함). Max Pooling 은 2×2 영역 최댓값만 남겨 크기를 절반으로 — 파라미터가 없고, 역전파는 최댓값이었던 자리로만.

**핵심 용어**
- **Max Pooling**: 영역(예 2×2, stride 2)마다 최댓값만 남기는 다운샘플링. 채널 수는 그대로
- **col2im**: im2col 의 반대 — 여러 영역에 복사되어 쓰인 픽셀의 기울기를 모두 더한다

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Convolution` | `Convolution(in_channels, out_channels, filter_size, stride=1, pad=0, weight_init="he", seed)` | 합성곱 계층 (mynn 규약) |
| `Pooling` | `Pooling(pool_h=2, pool_w=2, stride=2)` | Max Pooling 계층 |

**핵심 내용**

#### Convolution 역전파 = Affine 역전파
| | Affine (Day 3) | Convolution |
|---|---|---|
| dW | `X.T @ dout` | `(col.T @ dout).T.reshape(FN, C, FH, FW)` |
| db | `dout.sum(axis=0)` | `dout.sum(axis=0)` (dout 을 (N·OH·OW, FN) 으로 펼친 뒤) |
| dX | `dout @ W.T` | `dcol = dout @ col_W.T` → **col2im** → (N, C, H, W) |
- **col2im 은 더한다** → 한 픽셀이 여러 영역에 복사되어 쓰였으므로 (Day 3 "복사 노드의 역전파는 더하기")

#### Max Pooling
- **forward** → 최댓값 + `arg_max` 저장 · **backward** → arg_max 자리로만 dout, 나머지 0 (ReLU mask 처럼)
- **효과** → 크기 절반, 작은 이동에 둔감
- ⚠ Pooling 에 params 를 기대하지 말 것 — 없다 (Sequential.params() 에도 안 나옴)

#### 함정
- ⚠ `W.reshape(FN, -1)` 에서 `.T` 빠뜨림 — col_W 는 (C·FH·FW, FN)
- ⚠ backward 에서 dout 을 `reshape(-1, FN)` 만 하면 채널과 위치가 섞임 → 먼저 `transpose(0, 2, 3, 1)`
- ⚠ col2im 에서 겹치는 자리를 `=` 로 덮어씀 → `+=`

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
- [NN D4·8교시 mynn.py](#/nn/4/8) — 같은 계층 규약에 추가

### 5교시 · SimpleConvNet 으로 MNIST 학습

**키워드**: `SimpleConvNet`, `Conv → ReLU → Pool → Affine`, `단계별 shape`, `433,890`, `1 epoch 0.981`

**개요**: Day 4 의 Sequential · Trainer 를 고치지 않고 계층 리스트에 Convolution · Pooling 만 넣으면 CNN — 계층 규약의 힘. 1 epoch 만에 test 0.981 — 공간 구조를 쓰는 모델이 더 적은 학습으로 더 잘 맞힌다.

**핵심 용어**
- **CNN(합성곱 신경망)**: 합성곱 · 풀링 계층으로 공간 구조를 쓰는 신경망

**핵심 내용**

#### 구조와 shape (배치 100)
> Conv(1→30, 5×5) → ReLU → Pool(2×2) → Affine(4320→100) → ReLU → Affine(100→10) → Softmax
| 단계 | shape |
|---|---|
| input | (100, 1, 28, 28) |
| conv | (100, 30, 24, 24) — (28 − 5)/1 + 1 = 24 |
| pool | (100, 30, 12, 12) |
| fc1 | (100, 100) — 30·12·12 = 4,320 입력 |
| fc2 | (100, 10) |
- **Affine 이 (N, C, H, W) 도 받음** → 내부에서 펼침 (Day 4 에서 만들어 둠)

#### 3일의 모델 비교
| 모델 | 학습 | test |
|---|---|---|
| Day 3 784-100-10 MLP | SGD 10 epoch | 0.9694 |
| Day 4 5층 MLP + BN + He | Adam 5 epoch | 0.9757 |
| Day 5 SimpleConvNet | Adam 1 epoch | 0.9810 |

**기본 사용형태**
```python
cnn = mynn.Sequential([
    mynn.Convolution(1, 30, 5, seed=1), mynn.Relu(), mynn.Pooling(2, 2, 2),
    mynn.Affine(30 * 12 * 12, 100, seed=2), mynn.Relu(), mynn.Affine(100, 10, seed=3)])
mynn.Trainer(cnn, mynn.Adam(lr=0.001), batch_size=100).fit(X4_train, t_train, epochs=1)
```

### 6교시 · 학습 결과 해석

**키워드**: `1층 필터`, `특징 맵`, `상관계수 0.86~0.999`, `이동 둔감성`, `데이터 증강`

**개요**: 1 epoch 후 1층 필터는 초기값에서 조금만 바뀐다(상관 0.86~0.999, 변화량 약 28%) — 무작위 5×5 필터도 국소 패턴에 반응하고, 학습의 대부분은 뒤쪽 Affine 이 그 반응을 조합하는 데 쓰였다. CNN 은 작은 이동에 MLP 보다 강하지만 풀링이 한 번뿐이라 한계가 있다.

**핵심 용어**
- **데이터 증강(augmentation)**: 학습 때 이미지를 일부러 옮기고 돌려 이동 · 회전에 강하게 만드는 기법

**핵심 내용**

#### 필터와 특징 맵
- **변화량** → 경계 방향 무늬(빨강 · 파랑이 한쪽으로)
- **특징 맵** → 필터마다 반응한 위치 — Conv → ReLU 출력 30장

#### 2 px 이동에 대한 강건성
| 모델 | 파라미터 | 학습 | test | 2 px 이동 |
|---|---|---|---|---|
| Day 4 MLP | 110,610 | 5 epoch | 0.9757 | 0.7975 |
| CNN | 433,890 | 1 epoch | 0.9810 | 0.9119 |
- **3 px 이동** → CNN 도 0.748 — 풀링 한 번(2×2)의 한계 → 실제 모델은 풀링 여러 번 + 데이터 증강
- **파라미터는 CNN 이 더 많다** → 어디에 몰려 있는지 7교시

### 7교시 · 모델 비용 계산

**키워드**: `파라미터`, `MAC`, `활성화 메모리`, `float32 × 4 바이트`, `경량 CNN`, `depthwise separable`, `양자화`

**개요**: 온디바이스에서는 정확도만이 아니라 메모리(파라미터 · 활성화)와 연산량(MAC — 시간 · 전력)을 계산해 모델을 고른다. SimpleConvNet 의 파라미터는 대부분 Affine1, MAC 은 절반이 Conv. 파라미터를 줄이면 저장 공간, MAC 을 줄이면 속도 · 전력 — 다른 문제다.

**핵심 용어**
- **MAC(multiply-accumulate)**: 곱하고 더하기 한 번. 연산량(시간 · 전력)의 단위
- **활성화 메모리**: 실행 중 각 층 출력을 담는 메모리
- **양자화(quantization)**: float32 를 int8 등으로 줄여 메모리를 1/4 로

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `conv_cost()` | `conv_cost((C, H, W), FN, F, S=1, P=0)` | (params, MACs, 출력 shape) |
| `pool_cost()` | `pool_cost((C, H, W), P=2)` | (0, 0, (C, H//P, W//P)) |
| `affine_cost()` | `affine_cost(in_shape, n_out)` | (n_in·n_out + n_out, n_in·n_out, (n_out,)) |

**핵심 내용**

#### 비용 공식
| 지표 | Convolution | Affine | 의미 |
|---|---|---|---|
| 파라미터 | FN·C·FH·FW + FN | n_in·n_out + n_out | 저장 공간 (×4 바이트) |
| MAC | FN·OH·OW·C·FH·FW | n_in·n_out | 시간 · 전력 (1장 기준) |
| 활성화 | FN·OH·OW | n_out | 실행 중 메모리 |
- **합성곱** → 가중치를 공유해 파라미터는 적지만, 모든 출력 위치마다 계산하니 MAC 은 많다

#### SimpleConvNet 의 비용
| 층 | 파라미터 | MAC |
|---|---|---|
| Conv 1→30, 5×5 | 780 | 432,000 |
| Affine 4320→100 | 432,100 | 432,000 |
| Affine 100→10 | 1,010 | 1,000 |
| 합계 | **433,890** (1.74 MB) | **865,000** |

#### 경량 설계
| 모델 | 파라미터 | MAC (1장) | float32 |
|---|---|---|---|
| SimpleConvNet | 433,890 | 865,000 | 1.74 MB |
| 경량 CNN (3×3 Conv 16 · Pool · 3×3 Conv 32 · Pool · Affine) | 20,490 (21배 ↓) | 1,031,744 | 0.08 MB |
| 224×224×3, 3×3×64 Conv 한 층 | 1,792 | 86,704,128 | 활성화 12.8 MB |
- **Affine 입력 줄이기** → 풀링 · 작은 필터 → 파라미터 급감
- **큰 이미지** → 파라미터보다 MAC 과 활성화 메모리가 병목 → MobileNet 의 depthwise separable conv
- **양자화** → 메모리 1/4 · **BN folding** → 층 수 감소 (TensorRT · Quantization · Pruning 은 Vision 과정 이후)

**기본 사용형태**
```python
def conv_cost(in_shape, FN, F, S=1, P=0):
    C, H, W = in_shape
    OH, OW = conv_output_size(H, F, S, P), conv_output_size(W, F, S, P)
    params = FN * C * F * F + FN
    macs   = FN * OH * OW * C * F * F
    return params, macs, (FN, OH, OW)

def affine_cost(in_shape, n_out):
    n_in = int(np.prod(in_shape))
    return n_in * n_out + n_out, n_in * n_out, (n_out,)
```

**연결**
- [NN D2·5교시 행렬로 보는 순전파](#/nn/2/5) — 파라미터 수 = 모델 크기

### 8교시 · 평가 · 과정 정리

**키워드**: `평가 비중`, `CNN 실습 평가`, `gradient check`, `저장 · 불러오기`, `5일 정리`

**개요**: 일차별 미니 프로젝트 50% · 필답 30%(객관식 · shape 계산 · 역전파 유도) · CNN 실습 20%. 실습은 새 설정의 Convolution · Pooling gradient check, 학습한 CNN 저장 → 새 모델로 불러와 test ≥ 0.98, 경량 CNN 비용표.

**핵심 내용**

#### 평가 구성
| 평가 | 비중 | 방법 |
|---|---|---|
| 미니 프로젝트 (Day 1~4) | 50% | 각 노트북 채점 셀 |
| 필답 | 30% | 30문항 — 객관식 · shape 계산 · 역전파 유도 |
| CNN 실습 | 20% | gradient check · 저장/불러오기 test ≥ 0.98 · 비용표 |

#### 5일 정리 — 학습의 고리를 직접 만들었다
- **Day 1** → 도구: NumPy · 미분 · 경사하강
- **Day 2** → 뉴런 · 순전파 · 손실
- **Day 3** → 역전파 → MNIST 학습
- **Day 4** → 옵티마이저 · 초기화 · BN · 규제 → mynn.py
- **Day 5** → 합성곱 · 풀링 → CNN, 모델 비용

**연결**
- [NN D1·1교시 과정 안내](#/nn/1/1) — 첫날의 "학습 = argmin L(θ)"
