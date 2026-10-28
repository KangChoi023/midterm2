## Day 2 — CNN 대표 구조: LeNet · VGG · ResNet

### 1교시 · Day 1 복습 퀴즈 · CIFAR-10 과 MLP 의 한계

**키워드**: `CIFAR-10`, `32×32×3`, `정규화 상수(train 만)`, `MLP 1,707,274`, `0.4994`, `이웃 관계 상실`

**개요**: 오늘의 데이터는 MNIST(28×28 흑백) 보다 실제 사진에 가까운 CIFAR-10(32×32 컬러, 10 클래스). MNIST 에서 97% 를 낸 MLP 가 여기서는 50% 에 머문다 — 이미지를 3072 개 숫자의 줄로 펼치는 순간 "이웃 픽셀" 정보가 사라지기 때문. 합성곱(지역성 · 가중치 공유)이 이 문제를 푼다 (NN Day 5).

**핵심 용어**
- **CIFAR-10**: 32×32 RGB 컬러, 10 클래스(airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck), train 50,000 · test 10,000
- **정규화 상수(MEAN · STD)**: train 데이터의 채널별 평균 · 표준편차 — test 에도 **같은 값**을 쓴다
- **이웃 관계(공간 구조)**: 가까운 픽셀끼리 의미가 이어진다는 성질 — Flatten 하면 옆 픽셀이 32칸 · 1024칸 떨어진 위치로 가 버림

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `datasets.CIFAR10()` | `datasets.CIFAR10("data", train=True, download=True, transform=tf)` | CIFAR-10 불러오기 (첫 실행에 약 170 MB) |
| `transforms.Compose()` | `transforms.Compose([t1, t2])` | 변환을 순서대로 묶기 |
| `transforms.Normalize()` | `transforms.Normalize(MEAN, STD)` | 채널별 `(x − 평균) / 표준편차` |

**핵심 내용**

#### CIFAR-10 vs MNIST
| 항목 | MNIST | CIFAR-10 |
|---|---|---|
| 크기 | 28×28×1 = 784 | 32×32×3 = **3072** (4배) |
| 내용 | 가운데 놓인 숫자 | 배경 · 크기 · 각도 제각각 |
| MLP 정확도 | 약 97% | 약 **50%** (3 epoch) |
- 원본 `train_set.data` 는 `(50000, 32, 32, 3) uint8` — **HWC**. `ToTensor` 가 `(3, 32, 32)` CHW · 0 ~ 1 float 로 바꾼다
- 비슷한 클래스(고양이 ↔ 개)가 많아 "변화를 견디는 특징"을 배워야 한다

#### MLP 3072-512-256-10 의 파라미터
| 층 | 계산 | 파라미터 |
|---|---|---|
| Linear(3072, 512) | 3072·512 + 512 | 1,573,376 |
| Linear(512, 256) | 512·256 + 256 | 131,328 |
| Linear(256, 10) | 256·10 + 10 | 2,570 |
| **합계** | | **1,707,274** (첫 층이 92%) |
- 3 epoch test 0.4994 (강사 실행 예) — 파라미터 170만 개로도 절반만 맞힘
- ⚠ 정규화 상수를 test 로 계산하면 안 된다 (시험 문제를 미리 본 것) → `train_set.data / 255.0` 에 `mean(axis=(0, 1, 2))`
- ⚠ GPU 를 안 켬 — 오늘은 학습량이 Day 1 의 수십 배, `device : cpu` 면 T4 GPU 로

##### 자세히
- 오늘의 모델 3종 비교: MLP 1,707,274 (0.4994) · LeNet-5 62,006 (3교시) · ResNet-20 272,474 (8교시, 85% 목표).

**기본 사용형태**
```python
plain_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
train_set = datasets.CIFAR10("data", train=True, download=True, transform=plain_tf)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
data01 = train_set.data / 255.0
ch_mean = data01.mean(axis=(0, 1, 2))      # 채널 3개만 남는다
```

**연결**
- [NN D5·1교시 완전연결층의 한계](#/nn/5/1) — 같은 이유(이웃 관계 상실)로 합성곱이 필요했다
- [VB D1·6교시 디지털 이미지](#/vb/1/6) — 정규화 · HWC → CHW

### 2교시 · Conv 다시 보기: 출력 크기 · 파라미터 · 수용 영역

**키워드**: `stride`, `padding`, `groups`, `(N+2P−K)//S+1`, `Cout·Cin/G·K²+Cout`, `수용 영역`, `depthwise`

**개요**: `nn.Conv2d` 의 인자 하나하나가 **출력 크기 · 파라미터 수 · 연산량**을 정한다. 오늘 만드는 모든 모델은 이 세 가지 계산으로 설계된다. 층을 쌓고 크기를 줄일수록 출력 한 칸이 보는 입력 영역(수용 영역)이 넓어진다.

**핵심 용어**
- **same 패딩**: K=3, P=1, S=1 이면 출력 크기가 입력과 같다
- **depthwise 합성곱**: `groups = C_in` — 채널마다 따로 합성곱 (Day 3 MobileNet)
- **수용 영역(receptive field)**: 출력 한 칸이 영향을 받는 입력 영역의 크기

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `nn.Conv2d()` | `nn.Conv2d(cin, cout, k, stride=1, padding=0, groups=1, bias=True)` | 합성곱 (Day 1 에서 기본형, 오늘은 stride · padding · groups) |

**핵심 내용**

#### 두 공식
> 출력 크기 `O = (N + 2P − K) // S + 1` · 파라미터 `Cout · (Cin / G) · K² + Cout`
| 인자 | 하는 일 | 공식에서 |
|---|---|---|
| K (kernel_size) | 한 번에 보는 영역 | 파라미터 ∝ K² |
| S (stride) | 건너뛰는 간격 — 2 면 절반 | 출력 ÷ S |
| P (padding) | 가장자리에 0 을 덧댐 | 출력 + 2P |
| G (groups) | 입력 채널을 묶음으로 나눔 | 파라미터 ÷ G |
- `//` 는 내림 — 나누어떨어지지 않으면 오른쪽 · 아래 가장자리가 버려진다

#### 출력 크기 · 파라미터 계산 (직접 계산 확인)
| N, K, S, P | 출력 | | 설정 | 파라미터 |
|---|---|---|---|---|
| 32, 5, 1, 0 | 28 | | Conv(3→6, 5×5) | 3·6·25 + 6 = 456 |
| 32, 3, 1, 1 | 32 (same) | | Conv(64→128, 3×3) | 73,856 |
| 32, 3, 2, 1 | 16 | | Conv(64→64, 3×3, **groups=64**) | **640** |
| 28, 2, 2, 0 | 14 | | Conv(64→128, 1×1, bias 없음) | 8,192 |
| 224, 7, 2, 3 | 112 (ResNet 첫 층) | | Conv(16→32, 3×3, bias 없음) | 4,608 |
| 7, 3, 2, 0 | 3 (가장자리 버림) | | | |
- 64→64 3×3 : 일반 36,928 → depthwise 640 — 약 **1/58** (Day 3 MobileNet 의 핵심)
- `bias=False` 는 바로 뒤에 BatchNorm 이 올 때 (BN 의 β 가 bias 역할, 5교시)

#### 수용 영역
> 층마다 `rf ← rf + (K−1)·jump`, `jump ← jump·S` (시작 rf=1, jump=1)
| 구성 | 수용 영역 |
|---|---|
| 3×3 두 층 | 5×5 |
| 3×3 세 층 | 7×7 |
| LeNet 의 5-풀2-5-풀2 | 16 |
| VGG-mini 마지막 합성곱 | 32 → 32×32 이미지 전체를 본다 |
- Conv(64→128, 3×3, S=2, P=1) 에 (2, 64, 32, 32) 를 넣으면 (2, 128, 16, 16) — 크기 1/2 면 위치 수 1/4, 채널 2배여도 연산량은 **절반** → 층마다 연산량을 비슷하게 맞추는 설계
- ⚠ 풀링 · stride 뒤에서는 jump 가 커져서 같은 3×3 이 입력에서 더 넓게 퍼진다

**기본 사용형태**
```python
def conv_out(n, k, s=1, p=0):
    return (n + 2 * p - k) // s + 1

def conv_params(cin, cout, k, groups=1, bias=True):
    return cout * (cin // groups) * k * k + (cout if bias else 0)

down = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
print(tuple(down(torch.zeros(2, 64, 32, 32)).shape))   # (2, 128, 16, 16)
```

**연결**
- [NN D5·2교시 합성곱 연산](#/nn/5/2) — 출력 크기 공식의 출처
- [NN D5·7교시 모델 비용 계산](#/nn/5/7) — MAC 공식 (7교시에서 코드로)
- [VB D2·4교시 VGG](#/vb/2/4) — 수용 영역 · 3×3 쌓기

### 3교시 · LeNet-5 · AlexNet

**키워드**: `LeNet-5`, `62,006`, `귀납적 편향`, `AlexNet`, `ILSVRC 2012`, `61.8%`

**개요**: LeNet-5(1998, LeCun)는 Conv-Pool-Conv-Pool-FC-FC-FC 의 뼈대를 만든 최초의 실용 CNN. AlexNet(2012)은 이를 키워 ImageNet 대회에서 2위를 10%p 이상 앞섰고, ReLU · Dropout · 데이터 증강 · GPU 학습을 정착시켰다. 이후 우승 모델은 모두 CNN 이고 해마다 깊어졌다.

**핵심 용어**
- **귀납적 편향(inductive bias)**: 구조가 미리 가진 가정(합성곱 = 이웃 픽셀 · 위치 불변)이 데이터 없이도 학습을 돕는 것
- **ILSVRC**: ImageNet 대회 — 1,000 클래스 · 학습 이미지 약 128만 장 · 224×224

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torchvision.models.alexnet()` | `torchvision.models.alexnet(weights=None)` | AlexNet 구조 (가중치 없이) |
| `Module.named_parameters()` | `model.named_parameters()` | (이름, 파라미터) 쌍 반복 |

**핵심 내용**

#### LeNet-5 (CIFAR-10 용)
> Conv(3→6, 5) → ReLU → Pool → Conv(6→16, 5) → ReLU → Pool → Flatten(400) → FC 120 → FC 84 → FC 10
| 층 | 출력 shape | 파라미터 |
|---|---|---|
| 입력 | (N, 3, 32, 32) | — |
| conv1 3→6, 5×5 | (N, 6, 28, 28) → 풀링 (N, 6, 14, 14) | 3·6·25 + 6 = 456 |
| conv2 6→16, 5×5 | (N, 16, 10, 10) → 풀링 (N, 16, 5, 5) | 6·16·25 + 16 = 2,416 |
| fc1 400→120 | (N, 120) | 48,120 |
| fc2 120→84 | (N, 84) | 10,164 |
| fc3 84→10 | (N, 10) | 850 |
| **합계** | | **62,006** (MLP 의 1/27) |
- 3 epoch test: MLP 0.4994 (170만) vs LeNet-5 **0.5489** (6만) → 파라미터 1/27 로 더 높다 = 귀납적 편향
- 원래 LeNet 은 32×32 흑백 · tanh · 평균 풀링 — 오늘은 컬러 · ReLU · Max 풀링

#### AlexNet (2012)
| 기법 | 이유 | 이 과정에서 |
|---|---|---|
| ReLU | 기울기 소실이 적고 빠름 | NN Day 2 |
| Dropout | 거대한 FC 층의 과적합 방지 | NN Day 4 |
| 데이터 증강 | 자르기 · 뒤집기로 학습 데이터를 부풀림 | 7교시 |
| GPU 학습 | GTX 580 두 장으로 5 ~ 6일 | Colab T4 |
- ILSVRC 2012 top-5 오류 15.3% (2위 26.2%). 이후 AlexNet 8층 → VGG 19층 → GoogLeNet 22층 → ResNet 152층
- 파라미터는 약 6,110만: 가장 큰 텐서 `classifier.1` = Linear(9216→4096) 가중치 **37,748,736** 개 = 전체의 **61.8%**. 합성곱 5층을 다 합쳐도 2,469,696 개(4.0%)
- **"파라미터는 FC 에, 연산은 Conv 에"** (NN Day 5 결론) → 뒤의 모델들은 거대 FC 를 줄이는 방향 (ResNet = GAP + FC 하나)
- ⚠ 계산 확인: 9216 × 4096 = 37,748,736 (bias 4,096 은 별도)

**기본 사용형태**
```python
class LeNet5(nn.Module):
    def __init__(self, in_ch=3, n_cls=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 6, 5)      # 32 → 28
        self.conv2 = nn.Conv2d(6, 16, 5)         # 14 → 10
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_cls)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # (N, 6, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))     # (N, 16, 5, 5)
        x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.fc3(x)
```

**연결**
- [NN D5·5교시 SimpleConvNet](#/nn/5/5) — LeNet 계열의 NumPy 판
- [VB D1·4교시 nn.Module](#/vb/1/4) — 클래스로 모델을 짜는 방법
- [VB D2·6교시 ResNet](#/vb/2/6) — 거대 FC 대신 GAP + FC

### 4교시 · VGG: 3×3 을 쌓는 이유

**키워드**: `VGG`, `3×3 두 층 = 5×5`, `18C² vs 25C²`, `VGG 블록`, `VGG-mini 307,946`, `0.7526`

**개요**: VGG(2014)는 모든 합성곱을 3×3 으로 통일하고 같은 블록을 반복한다. 3×3 두 층은 5×5 한 층과 **같은 수용 영역**을 보면서 파라미터는 적고 ReLU 는 두 번 들어간다. 지금은 Conv 뒤에 BatchNorm 을 넣는 것이 표준.

**핵심 용어**
- **VGG 블록**: `[Conv3×3 → BN → ReLU] × n → MaxPool 2×2`
- **VGG-mini**: 블록 3개(채널 32 → 64 → 128, 블록마다 합성곱 2개) → (N, 128, 4, 4) → Flatten → Linear(2048 → 10)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `nn.BatchNorm2d()` | `nn.BatchNorm2d(C)` | 채널마다 정규화 (자세히 5교시) |

**핵심 내용**

#### 3×3 을 쌓는 이유 (채널 C, bias 없음)
| 비교 | 파라미터 | C=64 일 때 | 감소 |
|---|---|---|---|
| 5×5 한 층 | 25C² | 102,400 | |
| 3×3 두 층 | 18C² | 73,728 | **28%** |
| 7×7 한 층 | 49C² | 200,704 | |
| 3×3 세 층 | 27C² | 110,592 | **45%** |
- 같은 수용 영역(5×5, 7×7) · 파라미터 적음 · **비선형(ReLU)이 더 많이** 들어가 표현력이 큼

#### VGG-mini 구조 · 파라미터
| 단계 | 출력 shape | 파라미터 |
|---|---|---|
| 블록1 (3→32, 32→32) | (N, 32, 16, 16) | 864 + 9,216 = 10,080 |
| 블록2 (32→64, 64→64) | (N, 64, 8, 8) | 18,432 + 36,864 = 55,296 |
| 블록3 (64→128, 128→128) | (N, 128, 4, 4) | 73,728 + 147,456 = 221,184 |
| Linear(2048, 10) | (N, 10) | 20,490 |
| BN 6개 (γ, β) | | 896 |
| **합계** | | **307,946** (LeNet 의 5배, MLP 의 1/5.5) |
- 합성곱 6층(bias 없음) 286,560 + BN 896 + Linear 20,490 = 307,946
- 크기가 1/2 이 될 때마다 채널은 2배
- 3 epoch test **0.7526** — LeNet-5 0.5489 보다 크게 높다 (깊이 · 3×3 · BN)
- MAC (직접 계산) : 약 38,653,952 — LeNet-5 의 약 59배 (7교시 `count_macs` 로 확인)
- ⚠ 원래 VGG 에는 BN 이 없었다(BN 은 2015년) — 오늘 것은 현대판
- ⚠ 깊게 쌓으면 좋아진다 → 그럼 끝없이? → 5교시

**기본 사용형태**
```python
def vgg_block(cin, cout, n_convs):
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False),
                   nn.BatchNorm2d(cout),
                   nn.ReLU(inplace=True)]
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

vgg = nn.Sequential(vgg_block(3, 32, 2), vgg_block(32, 64, 2), vgg_block(64, 128, 2),
                    nn.Flatten(), nn.Linear(128 * 4 * 4, 10))
```

**연결**
- [VB D2·2교시 Conv 다시 보기](#/vb/2/2) — 수용 영역 · 파라미터 공식
- [VB D2·5교시 깊이의 문제 · BatchNorm](#/vb/2/5) — 더 깊게 쌓으면?

### 5교시 · 깊이의 문제 · BatchNorm

**키워드**: `기울기 소실`, `plain-N`, `BatchNorm2d`, `γ · β`, `running_mean`, `model.eval()`, `성능 저하(degradation)`

**개요**: 그냥 층만 깊게 쌓으면 입력 쪽 기울기가 사라져 학습이 안 된다. BatchNorm 이 층마다 출력 분포를 다시 맞춰 기울기가 고르게 흐르게 한다. 그런데 BN 이 있어도 20층이 8층보다 **train 손실이 더 높다** — 과적합이 아니라 최적화 자체가 어려운 성능 저하(degradation), 6교시 ResNet 의 출발점.

**핵심 용어**
- **기울기 소실(vanishing gradient)**: 역전파가 층마다 1 보다 작은 값을 곱해 입력 쪽 기울기가 0 에 가까워지는 현상
- **성능 저하(degradation)**: 깊은 망이 얕은 망보다 **train** 에서도 못하는 현상 (과적합이 아님)
- **GAP(전역 평균 풀링)**: 채널마다 전체 평균 하나 → (N, C, 1, 1)
- **buffer**: 학습 대상은 아니지만 모델과 함께 저장되는 값 (BN 의 running_mean · running_var)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `nn.BatchNorm2d()` | `nn.BatchNorm2d(C)` | Conv 뒤 채널별 정규화 + γ · β |
| `nn.AdaptiveAvgPool2d()` | `nn.AdaptiveAvgPool2d(1)` | 크기에 상관없이 지정한 크기로 평균 풀링 (1 = GAP) |
| `torch.utils.data.Subset()` | `Subset(train_set, range(10000))` | 데이터셋의 일부만 쓰기 |

**핵심 내용**

#### plain-N 실험
> 3×3 Conv (N−1)개(채널 16, same) → GAP → Linear(16 → 10)
- 파라미터: plain-8 = 3·16·9 + 16 + 6·(16·16·9 + 16) + 170 = **14,538**, plain-20 = **42,378** (직접 계산)
- 기울기 크기 (강사 실행 예, 첫 층): plain-8 3.2e-04 · plain-20 **1.1e-09** (마지막 층 2.2e-02 의 약 2천만분의 1) · plain-20 + BN **3.4e-01**
- 10,000 장 · SGD(lr 0.05, momentum 0.9) · 3 epoch (마지막 epoch train 손실 / test):
| | BN 없음 | BN 있음 |
|---|---|---|
| plain-8 | 2.304 / 0.100 | 1.699 / 0.304 |
| plain-20 | 2.303 / 0.100 | 1.785 / 0.270 |
- **BN 없으면** 8층조차 손실 2.30(= ln 10, 찍기)에서 안 움직인다 (기본 초기화로 쌓으면 층마다 신호 · 기울기가 작아짐)
- **BN 있으면** 둘 다 배우지만 20층의 train 손실이 8층보다 높다 → **성능 저하**

#### BatchNorm — NN Day 4 의 그 BN
| | NN Day 4 (Affine 뒤) | `nn.BatchNorm2d(C)` (Conv 뒤) |
|---|---|---|
| 평균 · 분산을 내는 축 | 배치 (axis=0) | 배치 · 세로 · 가로 (**채널마다 하나**) |
| 학습 파라미터 | γ, β (특징 수) | γ, β 각 C 개 |
| 추론 | running 평균 · 분산 | 같음 — `model.eval()` 이 전환 |
> `x̂ = (x − μ_batch) / sqrt(σ²_batch + ε)`, `y = γ·x̂ + β`
- 층마다 분포를 0 근처 · 분산 1 근처로 맞춰 기울기가 고르게 흐르고 **큰 학습률**도 견딘다
- `BatchNorm2d(16)` : 학습 파라미터 16 + 16 = **32** (weight = γ, bias = β), `state_dict` 에는 `running_mean` · `running_var` · `num_batches_tracked` 도 들어 있다 → 저장 · 불러오기에 함께 포함됨
- ⚠ 추론 때 `model.eval()` 을 빼먹으면 test 배치의 평균을 써서 결과가 배치마다 달라진다
- ⚠ 배치 크기 1 로 BN 학습 → 분산을 구할 수 없어 오류 · 불안정 (배치는 충분히 크게)
- ⚠ BN 앞 Conv 에 `bias=True` → 오류는 없지만 β 와 겹쳐 낭비

**기본 사용형태**
```python
def plain_net(depth, bn=False, width=16):
    layers, cin = [], 3
    for _ in range(depth - 1):
        layers.append(nn.Conv2d(cin, width, 3, padding=1, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm2d(width))
        layers.append(nn.ReLU(inplace=True))
        cin = width
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, 10)]
    return nn.Sequential(*layers)

grads = [m.weight.grad.norm().item() for m in net.modules() if isinstance(m, nn.Conv2d)]
```

**연결**
- [NN D4·4교시 Batch Normalization](#/nn/4/4) — γ · β · running 통계의 원리
- [NN D3·4교시 활성화 계층](#/nn/3/4) — 층마다 국소 미분이 곱해지는 이유
- [VB D2·6교시 ResNet](#/vb/2/6) — 성능 저하의 해법

### 6교시 · ResNet: Residual block

**키워드**: `skip connection`, `y = ReLU(F(x) + x)`, `기울기 +1`, `BasicBlock`, `1×1 Conv 지름길`, `ResNet-20 = 6n+2`, `272,474`

**개요**: 깊은 plain 망이 얕은 망보다 못한 건, 남는 층이 "아무것도 안 하기(항등 함수)"를 배우기 어렵기 때문. 입력을 출력에 그대로 더하는 **지름길**을 두면 층은 "얼마나 바꿀지(잔차)"만 배우고, 바꿀 게 없으면 0 을 배우면 된다. 덧셈 노드 덕분에 기울기에 항상 +1 이 남는다.

**핵심 용어**
- **잔차 블록(residual block)**: 출력 = `ReLU(F(x) + x)`, F = Conv-BN-ReLU-Conv-BN
- **지름길(skip connection)**: 입력 x 를 F 를 건너뛰어 출력에 더하는 경로
- **BasicBlock**: 3×3 Conv 두 번 + 지름길 (ResNet-20 의 단위)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `nn.Identity()` | `nn.Identity()` | 입력을 그대로 내보내는 층 (모양이 같을 때의 지름길) |
| `nn.AdaptiveAvgPool2d()` | `nn.AdaptiveAvgPool2d(1)` | GAP (5교시에 정리) |

**핵심 내용**

#### 왜 기울기가 흐르나
> 덧셈 노드는 기울기를 그대로 두 갈래로 보낸다 (NN Day 3 계산 그래프) → `∂y/∂x = F′(x) + 1`
- F′ 가 아무리 작아져도 **1 이 남아** 입력 쪽까지 기울기가 줄지 않는다. 블록이 쌓이면 지름길끼리 이어져 출력 → 입력의 "고속도로"

#### BasicBlock
| 상황 | 지름길 |
|---|---|
| 크기 · 채널 그대로 (예: 16→16, stride 1) | `nn.Identity()` — x 를 그대로 |
| 크기 1/2 · 채널 2배 (예: 16→32, stride 2) | `Conv 1×1 (stride 2) + BN` 으로 x 의 모양을 맞춤 |
- 16→32 stride 2 블록의 파라미터: conv1 16·32·9 = 4,608 · bn1 64 · conv2 32·32·9 = 9,216 · bn2 64 · 지름길 1×1 16·32 = 512 + BN 64 → **14,528**

#### ResNet-20 (CIFAR-10 용, 깊이 = 6n + 2, n = 3)
| 단계 | 출력 크기 | 채널 | 구성 |
|---|---|---|---|
| stem | 32×32 | 16 | Conv3×3 + BN + ReLU |
| layer1 | 32×32 | 16 | BasicBlock × 3 |
| layer2 | 16×16 | 32 | BasicBlock × 3 (첫 블록 stride 2) |
| layer3 | 8×8 | 64 | BasicBlock × 3 (첫 블록 stride 2) |
| head | 1×1 | 64 → 10 | GAP + Linear |
- 가중치 층 수: stem 1 + 블록 9 × 2 + Linear 1 = **20**
- 파라미터 **272,474** (VGG-mini 307,946 보다 적고 MLP 의 1/6) — 거대한 FC 대신 GAP + Linear(64 → 10) = **650** 개뿐
- 3 epoch 비교 (강사 실행 예): 학습 전 첫 층 기울기 plain-20 9.84e-01 vs ResNet-20 8.32e-01 (둘 다 BN 이 있어 시작은 문제없음), 3 epoch 뒤 train 손실 1.641 vs **1.438**, test 0.380 vs **0.443** → 성능 저하가 사라짐
- ⚠ `residual=False` 로 두면 지름길 Conv 는 만들어지되 forward 에서 안 쓰여 파라미터 수가 같은 plain 이 된다 (공정 비교용)
- ⚠ 모양이 바뀌는 블록에 Identity 를 쓰면 더할 수 없어 오류 → 1×1 Conv 지름길

**기본 사용형태**
```python
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, residual=True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)
        return F.relu(out)
```

**연결**
- [NN D3·2교시 계산 그래프](#/nn/3/2) — 덧셈 노드는 기울기를 그대로 분배
- [VB D2·5교시 깊이의 문제](#/vb/2/5) — 풀려는 성능 저하 문제
- [VB D2·8교시 미니 프로젝트](#/vb/2/8) — ResNet-20 을 20 epoch 학습

### 7교시 · 학습 기법: 데이터 증강 · 학습률 스케줄 · 혼동 행렬 · 연산량

**키워드**: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `OneCycleLR`, `혼동 행렬`, `MAC`, `forward hook`

**개요**: 같은 모델도 학습 방법에 따라 몇 %p 가 갈린다. 데이터 증강(train 에만) · OneCycle 학습률 · 혼동 행렬(무엇을 무엇으로 틀리나) · MAC(연산량 — 파라미터만으로는 속도를 알 수 없다) 네 도구를 쓴다.

**핵심 용어**
- **데이터 증강(augmentation)**: 학습할 때마다 이미지를 조금씩 바꿔 보여 줌 — 정답이 바뀌지 않는 변형만
- **OneCycleLR**: 학습률을 작게 시작 → 최댓값(warm-up) → 거의 0 까지
- **혼동 행렬(confusion matrix)**: `M[i, j]` = 정답이 i 인데 j 로 예측한 개수
- **MAC**: 곱셈-덧셈 횟수 — 온디바이스에서 중요한 연산량
- **forward hook**: 계층이 실행될 때마다 불리는 함수

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `transforms.RandomCrop()` | `transforms.RandomCrop(32, padding=4)` | 사방 0 패딩 후 무작위 위치에서 자르기 (±4 픽셀 이동) |
| `transforms.RandomHorizontalFlip()` | `transforms.RandomHorizontalFlip()` | 50% 확률 좌우 반전 |
| `OneCycleLR()` | `torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, epochs, steps_per_epoch)` | 학습률 스케줄러 |
| `np.bincount()` | `np.bincount(idx, minlength=K*K)` | 값별 개수 세기 (혼동 행렬) |
| `Module.register_forward_hook()` | `m.register_forward_hook(hook)` | 순전파 때 호출되는 함수 등록 |

**핵심 내용**

#### 데이터 증강
> train 에만 쓴다. test 는 언제나 원본 그대로.
- `train_tf = Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(), Normalize(MEAN, STD)])` — **Normalize 는 맨 뒤**
- `aug_set[7]` 을 꺼낼 때마다 새로 자르고 뒤집으므로 매번 다른 그림 (그릴 때는 `× std + mean` 으로 복원)
- ⚠ 정답이 바뀌는 변형은 금지 (숫자 6 을 뒤집으면 9 → MNIST 에 좌우 뒤집기 X)

#### OneCycleLR (max_lr 0.1)
| 지점 | 학습률 |
|---|---|
| 시작 | max_lr / 25 = **0.004** |
| 전체의 30% 지점 | 최댓값 **0.1** |
| 끝 | 시작의 10⁻⁴ 배 = 4e-07 |
- **배치마다** `scheduler.step()` (epoch 마다 아님)
- ⚠ `steps_per_epoch` 는 `len(train_loader)` 와 같아야 한다 — 틀리면 학습 끝에 `Tried to step ... times` 오류

#### 혼동 행렬
> `np.bincount(true * K + pred, minlength=K*K).reshape(K, K)` — 행 = 정답, 열 = 예측
- 대각선 합 / 전체 = 정확도. 대각선을 0 으로 지우고 큰 칸을 찾으면 "헷갈리는 쌍" (실행 예: cat → dog 180장)
- 칸 번호 k 에서 정답 = `k // 10`, 예측 = `k % 10`

#### MAC 공식 (직접 계산 확인)
> Conv MAC = Cout · OH · OW · (Cin / G) · K² · Linear MAC = in · out
| 모델 | 파라미터 | MAC |
|---|---|---|
| MLP | 1,707,274 | 1,706,496 |
| LeNet-5 | 62,006 | 651,720 (conv1 352,800 + conv2 240,000 + FC 58,920) |
| VGG-mini | 307,946 | 38,653,952 |
| ResNet-20 | 272,474 | **40,813,184** (MLP 의 약 24배) |
- 파라미터는 MLP 의 1/6 인데 MAC 은 24배 — 합성곱은 **같은 가중치를 모든 위치에서 다시 쓰기** 때문. 온디바이스에서는 두 숫자를 함께 줄인다 (Quantization 등)
- ⚠ hook 은 다 쓰면 `h.remove()` 로 떼야 다음 실행에 중복 등록되지 않는다
- ⚠ 평가용 예측은 `model.eval()` + `torch.no_grad()`

**기본 사용형태**
```python
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, epochs=15, steps_per_epoch=391)

def confusion_matrix(pred, true, K=10):
    return np.bincount(true * K + pred, minlength=K * K).reshape(K, K)

def hook(m, inp, out):                       # Conv MAC 세기
    cout, oh, ow = out.shape[1:]
    total[0] += cout * oh * ow * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
```

**연결**
- [NN D5·7교시 모델 비용 계산](#/nn/5/7) — MAC 공식의 출처
- [NN D4·6교시 규제](#/nn/4/6) — weight decay (8교시에서 사용)
- [VB D1·5교시 학습 루프](#/vb/1/5) — `train_one_epoch` 안에서 `scheduler.step()`

### 8교시 · 미니 프로젝트: CIFAR-10 ResNet-20 85% 도전

**키워드**: `ResNet-20 + 증강`, `SGD nesterov`, `weight_decay 5e-4`, `OneCycleLR 20 epoch`, `391 step`, `0.8587`, `혼동 행렬`

**개요**: 오늘 만든 것을 합쳐 ResNet-20 + 데이터 증강 + SGD(momentum · weight decay) + OneCycleLR 로 20 epoch 학습해 test 85% 이상을 목표로 한다. 이어서 혼동 행렬로 "가장 헷갈린 쌍"을 찾고, 네 모델(MLP · LeNet-5 · VGG-mini · ResNet-20)을 파라미터 · MAC · 정확도로 비교한다.

**핵심 용어**
- **weight decay**: 가중치 크기에 벌점을 주는 L2 규제 (NN Day 4)
- **nesterov**: momentum 의 앞서 보기 변형 (`nesterov=True`)

**핵심 내용**

#### 과제 구성 (80점 자동 채점 + 해석 20점)
| 과제 | 내용 | 배점 |
|---|---|---|
| 1 | 증강 train 로더 · 학습 설정 (SGD · OneCycle) | 20 |
| 2 | ResNet-20 20 epoch 학습 — test ≥ 0.85 | 35 |
| 3 | 오답 분석 — 혼동 행렬 · 가장 헷갈린 쌍 3개 · 오답 그림 | 15 |
| 4 | 모델 비교표 (파라미터 · MAC · test) | 10 |
| 5 | 결과 해석 (마크다운) | 10 + 10 (강사) |

#### 설정 값
| 항목 | 값 |
|---|---|
| train 로더 | 7교시 `aug_set` (증강), 배치 128, 섞기 |
| 옵티마이저 | `SGD(lr 0.1, momentum 0.9, nesterov=True, weight_decay=5e-4)` |
| 스케줄러 | `OneCycleLR(max_lr=0.1, epochs=20, steps_per_epoch=len(aug_loader))` |
| step / epoch | **391** (50,000 / 128 = 390.6 → 올림) |
| 파라미터 | 272,474 |
- 시간: T4 로 epoch 당 약 20초, 20 epoch 약 7분. 강사 실행 예 20 epoch test **0.8587** (15 epoch 는 0.8479 로 목표 미달)
- GPU 가 없으면 과제 2 는 강사가 학습한 가중치를 내려받는다 (과제 2 점수 제외)
- `fit(model, aug_loader, test_loader, EPOCHS, opt, sched, label=...)` — **다섯 번째 인자가 scheduler**
- 오답 분석: `off = cm.copy(); np.fill_diagonal(off, 0)` → `np.argsort(off.ravel())[::-1][:3]` → 쌍 = `(CLASSES[k // 10], CLASSES[k % 10], 개수)`
- 비교표 조건이 다르다 (MLP · LeNet · VGG 는 3 epoch, ResNet 은 20 epoch + 증강) → 해석에 쓴다
- ⚠ 채점: LeNet-5 의 MAC 은 `651_720`, ResNet-20 의 파라미터는 `272_474`
- ⚠ 구조 vs 학습 방법의 기여를 가르려면 같은 조건에서 다시 학습해야 공정하다
- ⚠ 온디바이스(Jetson · 라즈베리파이)라면 파라미터 · MAC 을 먼저 본다

##### 자세히
- 85% 다음: 같은 ResNet-20 을 약 160 epoch 학습하면 91.25% (원 논문). 더 깊게(ResNet-56 · 110) 또는 넓게 하면 93 ~ 95%. 반대로 온디바이스는 정확도를 조금 내주고 연산을 크게 줄이는 방향 → **Day 3 의 MobileNet (depthwise separable conv, 2교시의 groups) 과 전이학습**.

**기본 사용형태**
```python
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, epochs=20, steps_per_epoch=len(aug_loader))
hist_res = fit(model, aug_loader, test_loader, 20, opt, sched, label="ResNet-20")

pred, true = predict_all(model, test_loader)
cm = confusion_matrix(pred, true)
```

**연결**
- [VB D2·6교시 ResNet](#/vb/2/6) — 모델 정의
- [VB D2·7교시 학습 기법](#/vb/2/7) — 증강 · OneCycle · 혼동 행렬 · MAC
- [NN D4·6교시 규제](#/nn/4/6) — weight decay (L2)
