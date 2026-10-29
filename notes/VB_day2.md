## Day 2 — CNN 대표 구조: LeNet · VGG · ResNet

### 1교시 · Day 1 복습 퀴즈 · CIFAR-10 과 MLP 의 한계

**키워드**: `CIFAR-10`, `32×32×3 = 3072`, `10 클래스`, `train 50,000 · test 10,000`, `정규화 상수는 train 으로만`, `MLP 1,707,274`, `0.4994`, `이웃 관계 상실`

**개요**: 오늘의 데이터는 MNIST(28×28 흑백)보다 실제 사진에 가까운 **CIFAR-10**(32×32 컬러, 10 클래스). ① 배경 · 크기 · 각도가 제각각이고 고양이 ↔ 개처럼 비슷한 클래스가 많아 "변화를 견디는 특징"을 배워야 한다. ② MNIST 에서 97% 를 낸 MLP 가 여기서는 파라미터 170만 개로도 **약 50%** — 이미지를 3072 개 숫자의 줄로 펼치는 순간 이웃 픽셀 정보가 사라지기 때문. ③ 합성곱(지역성 · 가중치 공유)이 이 문제를 푼다 → 오늘 LeNet · VGG · ResNet 으로 85% 까지.

**핵심 용어**
- **CIFAR-10**: 32×32 RGB 컬러, 10 클래스(airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck), train **50,000** · test **10,000**
- **정규화 상수(MEAN · STD)**: train 데이터의 채널별 평균 · 표준편차 — test 에도 **같은 값**을 쓴다
- **이웃 관계(공간 구조)**: 가까운 픽셀끼리 의미가 이어진다는 성질 — Flatten 하면 바로 아래 픽셀이 32칸, 다음 채널은 1024칸 떨어진 위치로 가 버림
- **transforms.Compose**: 여러 변환(ToTensor · Normalize 등)을 순서대로 묶은 하나의 변환

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `datasets.CIFAR10()` | `datasets.CIFAR10("data", train=True, download=True, transform=tf)` | CIFAR-10 불러오기 (첫 실행에 약 170 MB) |
| `transforms.Compose()` | `transforms.Compose([t1, t2])` | 변환을 순서대로 묶기 |
| `transforms.Normalize()` | `transforms.Normalize(MEAN, STD)` | 채널별 `(x − 평균) / 표준편차` |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 오늘의 데이터 — CIFAR-10 | 10 클래스 예시 그림 · train 데이터로 채널별 MEAN · STD 계산 · MLP 3072-512-256-10 을 Adam 3 epoch 학습 → test 약 **0.50** |

**핵심 내용**

#### CIFAR-10 vs MNIST
> 크기는 4배, 내용은 훨씬 제각각 — MLP 로는 절반만 맞힌다
| 항목 | MNIST | CIFAR-10 |
|---|---|---|
| 크기 | 28×28×1 = 784 | 32×32×3 = **3072** (4배) |
| 내용 | 가운데 놓인 숫자 | 배경 · 크기 · 각도 제각각 |
| MLP 정확도 | 약 97% | 약 **50%** (3 epoch) |
- **원본 `train_set.data`** → (50000, 32, 32, 3) uint8 — **HWC** · `ToTensor` 가 (3, 32, 32) CHW · 0~1 float 로
- **비슷한 클래스** → 고양이 ↔ 개, 자동차 ↔ 트럭 → 위치 · 크기 변화를 견디는 특징이 필요
- ⚠ GPU 를 안 켬 — 오늘은 학습량이 Day 1 의 수십 배, `device : cpu` 면 **T4 GPU** 로
- ⚠ CIFAR-10 첫 다운로드 약 170 MB — 1~2분 기다린다

#### 정규화 상수는 train 으로만
> MEAN · STD = train 데이터의 채널별 평균 · 표준편차 → test 에도 같은 값
- **계산** → `data01 = train_set.data / 255.0` → `data01.mean(axis=(0, 1, 2))` — 채널 3개만 남음
- **`axis=(0, 1, 2)`** → (N, H, W, C) 에서 N · H · W 축을 모두 평균 → 채널마다 하나
- ⚠ 정규화 상수를 test 로 계산하면 안 된다 — 시험 문제를 미리 본 것 (NN Day 4 validation 원칙)

#### MLP 3072-512-256-10 — 파라미터 170만으로 50%
> 펼치는 순간 이웃 관계가 사라져, 파라미터를 늘려도 구조를 못 본다
| 층 | 계산 | 파라미터 |
|---|---|---|
| Linear(3072, 512) | 3072·512 + 512 | 1,573,376 |
| Linear(512, 256) | 512·256 + 256 | 131,328 |
| Linear(256, 10) | 256·10 + 10 | 2,570 |
| 합계 | | **1,707,274** (첫 층이 92%) |
- **3 epoch test 0.4994** (강사 실행 예) — 파라미터 170만 개로도 절반만 맞힘
- **오늘의 모델 비교** → MLP 1,707,274 (0.4994) · LeNet-5 62,006 (3교시) · VGG-mini 307,946 (4교시) · ResNet-20 272,474 (8교시, 85% 목표)
##### 자세히
```text
 image (3, 32, 32)                  flattened 3072
 R[0,0] R[0,1] …                    [ R row0 … | R row1 … | … | G … | B … ]
 R[1,0]  ← directly below R[0,0]       index 0      index 32            index 1024 = G[0,0]
 neighbors in the picture end up far apart in the vector → MLP cannot see shape
```

**기본 사용형태**
```python
data01 = train_set.data / 255.0                 # (50000, 32, 32, 3)
MEAN = data01.mean(axis=(0, 1, 2)); STD = data01.std(axis=(0, 1, 2))   # 채널 3개
plain_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
train_set = datasets.CIFAR10("data", train=True, download=True, transform=plain_tf)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
```

**연결**
- [NN D5·1교시 복습 퀴즈 · 완전연결층의 한계](#/nn/5/1) — 같은 이유(이웃 관계 상실)로 합성곱이 필요했다
- [VB D1·6교시 디지털 이미지: 픽셀 · 채널 · 정규화](#/vb/1/6) — 정규화 · HWC → CHW
- [NN D4·7교시 하이퍼파라미터 탐색](#/nn/4/7) — test 는 마지막에 한 번만 본다

**연관 문제**
<<QA 1>>

### 2교시 · Conv 다시 보기: 출력 크기 · 파라미터 · 수용 영역

**키워드**: `kernel K`, `stride S`, `padding P`, `groups G`, `O = (N + 2P − K) // S + 1`, `Cout · (Cin / G) · K² + Cout`, `same 패딩`, `depthwise 1/58`, `수용 영역`, `bias=False + BN`

**개요**: `nn.Conv2d` 의 인자 하나하나가 **출력 크기 · 파라미터 수 · 연산량**을 정한다 — 오늘 만드는 모든 모델은 이 계산으로 설계된다. ① 출력 크기 `(N + 2P − K) // S + 1`, 파라미터 `Cout · (Cin / G) · K² + Cout` 두 공식. ② `groups = Cin` 이면 채널마다 따로 합성곱(depthwise) — 파라미터가 약 1/58 (Day 3 MobileNet). ③ 층을 쌓고 크기를 줄일수록 출력 한 칸이 보는 입력 영역(**수용 영역**)이 넓어진다 — 3×3 두 층 = 5×5.

**핵심 용어**
- **same 패딩**: K = 3, P = 1, S = 1 이면 출력 크기가 입력과 같다
- **groups(G)**: 입력 채널을 G 묶음으로 나눠 묶음끼리만 합성곱 — 파라미터 ÷ G
- **depthwise 합성곱**: `groups = Cin` — 채널마다 따로 합성곱 (Day 3 MobileNet)
- **수용 영역(receptive field)**: 출력 한 칸이 영향을 받는 입력 영역의 크기
- **jump**: 현재 층의 한 칸이 입력에서 몇 픽셀 간격인지 — stride · 풀링마다 곱해짐

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `nn.Conv2d()` | `nn.Conv2d(cin, cout, k, stride=1, padding=0, groups=1, bias=True)` | 합성곱 (Day 1 에서 기본형, 오늘은 stride · padding · groups) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | Conv2d 인자 · 공식 그림 → `conv_out` · `conv_params` 로 표의 출력 크기 · 파라미터를 계산하고 `nn.Conv2d` 실제 결과와 비교 |
| 수용 영역 (receptive field) | 3×3 두 층 = 5×5 그림 → 층 목록으로 수용 영역을 계산하는 함수 (LeNet 16, VGG-mini 32) |

**핵심 내용**

#### 두 공식 — 출력 크기와 파라미터
> 출력 크기 `O = (N + 2P − K) // S + 1` · 파라미터 `Cout · (Cin / G) · K² + Cout`
| 인자 | 하는 일 | 공식에서 |
|---|---|---|
| K (kernel_size) | 한 번에 보는 영역 | 파라미터 ∝ K² |
| S (stride) | 건너뛰는 간격 — 2 면 절반 | 출력 ÷ S |
| P (padding) | 가장자리에 0 을 덧댐 | 출력 + 2P |
| G (groups) | 입력 채널을 묶음으로 나눔 | 파라미터 ÷ G |
- **`//` 는 내림** → 나누어떨어지지 않으면 오른쪽 · 아래 가장자리가 버려진다
- **`bias=False`** → 바로 뒤에 BatchNorm 이 올 때 (BN 의 β 가 bias 역할, 5교시)

#### 계산 확인 — 출력 크기 · 파라미터
> 표의 숫자는 모두 공식으로 직접 계산해 확인한 값
| N, K, S, P | 출력 |
|---|---|
| 32, 5, 1, 0 | **28** |
| 32, 3, 1, 1 | **32** (same) |
| 32, 3, 2, 1 | **16** |
| 28, 2, 2, 0 | 14 |
| 224, 7, 2, 3 | 112 (ResNet 첫 층) |
| 7, 3, 2, 0 | 3 (가장자리 버림) |

| 설정 | 파라미터 |
|---|---|
| Conv(3→6, 5×5) | 3·6·25 + 6 = **456** |
| Conv(64→128, 3×3) | 73,856 |
| Conv(64→64, 3×3, groups=64) | **640** |
| Conv(64→128, 1×1, bias 없음) | 8,192 |
| Conv(16→32, 3×3, bias 없음) | 4,608 |
- **64→64 3×3** → 일반 36,928 → depthwise 640 — 약 **1/58** (Day 3 MobileNet 의 핵심)
- **Conv(64→128, 3×3, S2, P1) 에 (2, 64, 32, 32)** → (2, 128, 16, 16) — 크기 1/2 면 위치 수 1/4, 채널 2배여도 연산량은 **절반** → 층마다 연산량을 비슷하게 맞추는 설계

#### 수용 영역 — 쌓을수록 넓게 본다
> 층마다 `rf ← rf + (K − 1)·jump`, `jump ← jump·S` (시작 rf = 1, jump = 1)
| 구성 | 수용 영역 |
|---|---|
| 3×3 두 층 | **5×5** |
| 3×3 세 층 | **7×7** |
| LeNet 의 5-풀2-5-풀2 | 16 |
| VGG-mini 마지막 합성곱 | 32 → 32×32 이미지 **전체**를 본다 |
- **풀링 · stride 뒤** → jump 가 커져 같은 3×3 이 입력에서 더 넓게 퍼진다
- ⚠ 수용 영역을 셀 때 풀링도 층으로 넣어야 한다 (K = 2, S = 2)
##### 자세히
```text
 input   ■ ■ ■ ■ ■        layer 2 (3×3) sees layer 1's 3×3,
         ■ ■ ■ ■ ■        each of which saw 3×3 of the input
         ■ ■ ● ■ ■   ──►  → one output cell = 5×5 of the input
         ■ ■ ■ ■ ■        rf: 1 → 1+2·1 = 3 → 3+2·1 = 5
         ■ ■ ■ ■ ■
```

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
- [NN D5·7교시 모델 비용 계산](#/nn/5/7) — 파라미터 · MAC 공식 (7교시에서 코드로)
- [VB D2·4교시 VGG: 3×3 을 쌓는 이유](#/vb/2/4) — 수용 영역 · 3×3 쌓기

**연관 문제**
<<QA 2>>

### 3교시 · LeNet-5 · AlexNet

**키워드**: `LeNet-5 (1998)`, `Conv-Pool-Conv-Pool-FC-FC-FC`, `62,006`, `0.5489`, `귀납적 편향`, `AlexNet (2012)`, `ILSVRC top-5 15.3%`, `ReLU · Dropout · 증강 · GPU`, `FC 한 층 61.8%`

**개요**: CNN 의 역사를 두 모델로 본다. ① **LeNet-5**(1998, LeCun)는 Conv-Pool-Conv-Pool-FC-FC-FC 뼈대를 만든 최초의 실용 CNN — 파라미터 **6만**으로 170만 MLP 보다 높다(귀납적 편향). ② **AlexNet**(2012)은 이를 키워 ImageNet 대회에서 2위를 10%p 이상 앞섰고 ReLU · Dropout · 데이터 증강 · GPU 학습을 정착시켰다 — 이후 우승 모델은 모두 CNN, 해마다 깊어졌다. ③ AlexNet 파라미터의 62% 가 FC 한 층 → "파라미터는 FC 에, 연산은 Conv 에" → 뒤 모델들은 거대 FC 를 줄인다.

**핵심 용어**
- **귀납적 편향(inductive bias)**: 구조가 미리 가진 가정(합성곱 = 이웃 픽셀 · 위치 불변)이 데이터 없이도 학습을 돕는 것
- **ILSVRC**: ImageNet 대회 — **1,000** 클래스 · 학습 이미지 약 **128만** 장 · 224×224
- **top-5 오류**: 모델이 고른 상위 5개 안에 정답이 없는 비율
- **named_parameters**: (이름, 파라미터) 쌍 — 어느 층에 파라미터가 몰렸는지 세기

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torchvision.models.alexnet()` | `torchvision.models.alexnet(weights=None)` | AlexNet 구조 (가중치 없이) |
| `Module.named_parameters()` | `model.named_parameters()` | (이름, 파라미터) 쌍 반복 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| LeNet-5 (1998) — 합성곱 신경망의 시작 | 구조 그림 → `LeNet5` 클래스 · 층별 shape · 파라미터 **62,006** 확인 → Adam 3 epoch 학습 → test 약 **0.55** (MLP 0.50) |
| AlexNet (2012) — 딥러닝의 귀환 | ImageNet top-5 오류 변화 그림 → `alexnet(weights=None)` 의 `named_parameters` 로 가장 큰 텐서와 합성곱 · FC 비율 계산 |

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
| 합계 | | **62,006** (MLP 의 1/27) |
- **3 epoch test** → MLP 0.4994 (170만) vs LeNet-5 **0.5489** (6만) → 파라미터 1/27 로 더 높다 = **귀납적 편향**
- **원래 LeNet** → 32×32 흑백 · tanh · 평균 풀링 — 오늘은 컬러 · ReLU · Max 풀링
##### 자세히
```text
 (3,32,32) ─conv5─► (6,28,28) ─pool─► (6,14,14) ─conv5─► (16,10,10) ─pool─► (16,5,5)
   ─flatten─► 400 ─fc─► 120 ─fc─► 84 ─fc─► 10
```

#### AlexNet (2012) — 딥러닝의 귀환
> ILSVRC 2012 top-5 오류 **15.3%** (2위 26.2%) — 이후 우승 모델은 모두 CNN
| 기법 | 이유 | 이 과정에서 |
|---|---|---|
| ReLU | 기울기 소실이 적고 빠름 | NN Day 2 |
| Dropout | 거대한 FC 층의 과적합 방지 | NN Day 4 |
| 데이터 증강 | 자르기 · 뒤집기로 학습 데이터를 부풀림 | 7교시 |
| GPU 학습 | GTX 580 두 장으로 5 ~ 6일 | Colab T4 |
- **해마다 깊어짐** → AlexNet 8층 → VGG 19층 → GoogLeNet 22층 → ResNet 152층

#### 파라미터는 FC 에, 연산은 Conv 에
> AlexNet 약 6,110만 파라미터 중 Linear(9216→4096) 한 층이 **61.8%**
- **가장 큰 텐서 `classifier.1`** → 9216 × 4096 = **37,748,736** 개 (bias 4,096 별도)
- **합성곱 5층 합계** → 2,469,696 개 (**4.0%**)
- **NN Day 5 결론과 같다** → 뒤의 모델들은 거대 FC 를 줄이는 방향 (ResNet = GAP + FC 하나)

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
- [NN D5·5교시 SimpleConvNet 으로 MNIST 학습](#/nn/5/5) — LeNet 계열의 NumPy 판
- [NN D5·7교시 모델 비용 계산](#/nn/5/7) — "파라미터는 Affine 에, 연산은 Conv 에"
- [VB D1·4교시 nn.Module · Dataset · DataLoader](#/vb/1/4) — 클래스로 모델을 짜는 방법

**연관 문제**
<<QA 3>>

### 4교시 · VGG: 3×3 을 쌓는 이유

**키워드**: `VGG (2014)`, `3×3 두 층 = 5×5 수용 영역`, `18C² vs 25C² (28% ↓)`, `27C² vs 49C² (45% ↓)`, `비선형 2번`, `VGG 블록`, `VGG-mini 307,946`, `0.7526`, `크기 1/2 · 채널 2배`

**개요**: VGG(2014, Oxford)는 모든 합성곱을 **3×3 으로 통일**하고 같은 블록을 반복한다. ① 3×3 두 층은 5×5 한 층과 **같은 수용 영역**을 보면서 파라미터는 28% 적고 ReLU 는 두 번 들어간다. ② 블록 = [Conv3×3 → BN → ReLU] × n → MaxPool, 크기가 1/2 이 될 때마다 채널은 2배. ③ VGG-mini 는 3 epoch 에 **0.7526** — LeNet 보다 크게 높다. 그럼 끝없이 깊게 쌓으면? → 5교시.

**핵심 용어**
- **VGG 블록**: `[Conv3×3 → BN → ReLU] × n → MaxPool 2×2`
- **VGG-mini**: 블록 3개(채널 32 → 64 → 128, 블록마다 합성곱 2개) → (N, 128, 4, 4) → Flatten → Linear(2048 → 10)
- **BatchNorm2d**: Conv 출력을 채널마다 정규화하는 층 (자세히 5교시)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `nn.BatchNorm2d()` | `nn.BatchNorm2d(C)` | 채널마다 정규화 (자세히 5교시) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | 5×5 한 층 vs 3×3 두 층 그림 → 채널 64 에서 두 구성의 파라미터를 계산해 비교 |
| VGG 블록 | `vgg_block` 으로 VGG-mini 를 조립해 shape · 파라미터 **307,946** 확인 → 3 epoch 학습, MLP · LeNet-5 · VGG-mini test 곡선 비교 |

**핵심 내용**

#### 3×3 을 쌓는 이유 (채널 C, bias 없음)
> 같은 수용 영역 · 적은 파라미터 · 비선형(ReLU)이 더 많이
| 비교 | 파라미터 | C = 64 일 때 | 감소 |
|---|---|---|---|
| 5×5 한 층 | 25C² | 102,400 | |
| 3×3 두 층 | 18C² | 73,728 | **28%** |
| 7×7 한 층 | 49C² | 200,704 | |
| 3×3 세 층 | 27C² | 110,592 | **45%** |
- **ReLU 가 두 번** → 같은 영역을 보면서 더 복잡한 함수를 표현
- **현대 CNN** → 대부분 3×3 (ResNet 도)
##### 자세히
```text
 5×5 one layer:   input 5×5 ──[25C²]──► 1 cell                  1 ReLU
 3×3 two layers:  input 5×5 ──[9C²]──► 3×3 ──[9C²]──► 1 cell     2 ReLU, 18C²
```

#### VGG-mini 구조 · 파라미터
> 블록 3개(32 → 64 → 128) — 크기가 1/2 이 될 때마다 채널은 2배
| 단계 | 출력 shape | 파라미터 |
|---|---|---|
| 블록1 (3→32, 32→32) | (N, 32, 16, 16) | 864 + 9,216 = 10,080 |
| 블록2 (32→64, 64→64) | (N, 64, 8, 8) | 18,432 + 36,864 = 55,296 |
| 블록3 (64→128, 128→128) | (N, 128, 4, 4) | 73,728 + 147,456 = 221,184 |
| Linear(2048, 10) | (N, 10) | 20,490 |
| BN 6개 (γ, β) | | 896 |
| 합계 | | **307,946** (LeNet 의 5배, MLP 의 1/5.5) |
- **3 epoch test 0.7526** → LeNet-5 0.5489 보다 크게 높다 (깊이 · 3×3 · BN)
- **MAC** → 약 **38,653,952** — LeNet-5 의 약 59배 (7교시 `count_macs` 로 확인)
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
- [VB D2·2교시 Conv 다시 보기: 출력 크기 · 파라미터 · 수용 영역](#/vb/2/2) — 수용 영역 · 파라미터 공식
- [VB D2·5교시 깊이의 문제 · BatchNorm](#/vb/2/5) — 더 깊게 쌓으면?
- [NN D2·4교시 활성화 함수](#/nn/2/4) — 비선형이 있어야 쌓는 의미가 있다

**연관 문제**
<<QA 4>>

### 5교시 · 깊이의 문제 · BatchNorm

**키워드**: `기울기 소실`, `plain-8 · plain-20`, `첫 층 기울기 1.1e-09`, `BatchNorm2d`, `채널마다 하나`, `γ · β`, `running_mean · running_var`, `model.eval()`, `성능 저하(degradation)`

**개요**: VGG 가 깊이로 성공했으니 층을 더 쌓아 본다. ① 그냥 깊게 쌓으면 입력 쪽 기울기가 사라져(**기울기 소실**) BN 없는 plain 망은 8층조차 3 epoch 동안 못 배운다. ② **BatchNorm** 이 층마다 출력 분포를 다시 맞춰 기울기가 고르게 흐르고 큰 학습률도 견딘다. ③ 그런데 BN 이 있어도 20층이 8층보다 **train 손실이 더 높다** — 과적합이 아니라 최적화 자체가 어려운 **성능 저하(degradation)**, 6교시 ResNet 의 출발점.

**핵심 용어**
- **기울기 소실(vanishing gradient)**: 역전파가 층마다 1 보다 작은 값을 곱해 입력 쪽 기울기가 0 에 가까워지는 현상
- **성능 저하(degradation)**: 깊은 망이 얕은 망보다 **train** 에서도 못하는 현상 (과적합이 아님)
- **GAP(전역 평균 풀링)**: 채널마다 전체 평균 하나 → (N, C, 1, 1)
- **buffer**: 학습 대상은 아니지만 모델과 함께 저장되는 값 (BN 의 running_mean · running_var)
- **plain-N**: 3×3 Conv (N − 1)개 + GAP + Linear 로 만든, 지름길 없는 N 층 망

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `nn.BatchNorm2d()` | `nn.BatchNorm2d(C)` | Conv 뒤 채널별 정규화 + γ · β |
| `nn.AdaptiveAvgPool2d()` | `nn.AdaptiveAvgPool2d(1)` | 크기에 상관없이 지정한 크기로 평균 풀링 (1 = GAP) |
| `torch.utils.data.Subset()` | `Subset(train_set, range(10000))` | 데이터셋의 일부만 쓰기 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | `plain_net(depth, bn)` 으로 plain-8 · plain-20 (BN 유무) 을 만들어 층별 기울기 크기 측정 → 10,000장 · SGD(0.05, momentum 0.9) · 3 epoch 학습 비교 · BatchNorm2d 의 파라미터 · state_dict 확인 |

**핵심 내용**

#### plain-N 실험 — 깊이 쌓기만 하면
> 3×3 Conv (N − 1)개(채널 16, same) → GAP → Linear(16 → 10)
- **파라미터** → plain-8 = 3·16·9 + 16 + 6·(16·16·9 + 16) + 170 = **14,538**, plain-20 = **42,378**
- **첫 층 기울기 (강사 실행 예)** → plain-8 3.2e-04 · plain-20 **1.1e-09** (마지막 층 2.2e-02 의 약 2천만분의 1) · plain-20 + BN **3.4e-01**
- **BN 없으면** → 8층조차 손실 2.30(= ln 10, 찍기)에서 안 움직임 (기본 초기화로 쌓으면 층마다 신호 · 기울기가 작아짐)
- **BN 있으면** → 둘 다 배우지만 20층의 train 손실이 8층보다 높다 → **성능 저하**
| 3 epoch (train 손실 / test) | BN 없음 | BN 있음 |
|---|---|---|
| plain-8 | 2.304 / 0.100 | 1.699 / 0.304 |
| plain-20 | 2.303 / 0.100 | 1.785 / 0.270 |
##### 자세히
```text
 layer 1 ◄──×small── layer 2 ◄──×small── … ◄── layer 19 ◄── loss
 plain-20 first layer grad ≈ 1e-09  (vanishing)
 with BN every layer re-centers → grad ≈ 3e-01 reaches layer 1
 but train loss: 20 layers (1.785) > 8 layers (1.699)  → degradation, not overfitting
```

#### BatchNorm2d — NN Day 4 의 그 BN, Conv 판
> x̂ = (x − μ_batch) / √(σ²_batch + ε), y = γ·x̂ + β — 평균 · 분산은 **채널마다 하나**
| 항목 | NN Day 4 (Affine 뒤) | `nn.BatchNorm2d(C)` (Conv 뒤) |
|---|---|---|
| 평균 · 분산을 내는 축 | 배치 (axis=0) | 배치 · 세로 · 가로 (**채널마다 하나**) |
| 학습 파라미터 | γ, β (특징 수) | γ, β 각 C 개 |
| 추론 | running 평균 · 분산 | 같음 — `model.eval()` 이 전환 |
- **효과** → 층마다 분포를 0 근처 · 분산 1 근처로 → 기울기가 고르게 흐르고 **큰 학습률**도 견딤
- **`BatchNorm2d(16)`** → 학습 파라미터 16 + 16 = **32** (weight = γ, bias = β)
- **`state_dict`** → `running_mean` · `running_var` · `num_batches_tracked` 도 들어 있음 → 저장 · 불러오기에 함께 포함
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
- [NN D2·4교시 활성화 함수](#/nn/2/4) — 기울기 소실 (0.25¹⁰)
- [VB D2·6교시 ResNet: Residual block](#/vb/2/6) — 성능 저하의 해법

**연관 문제**
<<QA 5>>

### 6교시 · ResNet: Residual block

**키워드**: `skip connection`, `y = ReLU(F(x) + x)`, `잔차`, `∂y/∂x = F′(x) + 1`, `BasicBlock`, `1×1 Conv 지름길`, `ResNet-20 = 6n + 2`, `272,474`, `GAP + FC 650`

**개요**: 2015년 Kaiming He 등이 성능 저하를 풀었다. ① 깊은 plain 망이 못한 이유는 남는 층이 "아무것도 안 하기(항등 함수)"를 배우기 어려워서 — 입력을 출력에 그대로 더하는 **지름길**을 두면 층은 "얼마나 바꿀지(**잔차**)"만 배우고, 바꿀 게 없으면 0 을 배우면 된다. ② 덧셈 노드 덕분에 기울기에 항상 **+1** 이 남아 입력까지 흐른다. ③ CIFAR 용 **ResNet-20**(6·3 + 2 층)은 거대 FC 대신 GAP + Linear 하나로 파라미터 272,474 개.

**핵심 용어**
- **잔차 블록(residual block)**: 출력 = `ReLU(F(x) + x)`, F = Conv-BN-ReLU-Conv-BN
- **지름길(skip connection)**: 입력 x 를 F 를 건너뛰어 출력에 더하는 경로
- **잔차(residual)**: 목표 출력 − 입력 = F 가 배우는 "바꿀 양"
- **BasicBlock**: 3×3 Conv 두 번 + 지름길 (ResNet-20 의 단위)
- **projection 지름길**: 크기 · 채널이 바뀔 때 1×1 Conv(stride) + BN 으로 x 의 모양을 맞춘 지름길

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `nn.Identity()` | `nn.Identity()` | 입력을 그대로 내보내는 층 (모양이 같을 때의 지름길) |
| `nn.AdaptiveAvgPool2d()` | `nn.AdaptiveAvgPool2d(1)` | GAP (5교시에 정리) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 들어가며 | BasicBlock 그림 → `BasicBlock` 클래스 완성, 16→32 stride 2 블록의 shape · 파라미터 **14,528** 확인 |
| ResNet-20 for CIFAR-10 | 구조 그림 → `resnet20()` 조립, 파라미터 **272,474** · 가중치 층 20 확인 → plain-20 과 같은 조건 3 epoch 비교 |

**핵심 내용**

#### 지름길 — 바꿀 양만 배운다
> y = ReLU(F(x) + x) — 바꿀 게 없으면 F = 0 을 배우면 항등 함수
- **plain 의 문제** → 남는 층이 정확히 y = x 를 배우기 어려움 → 깊을수록 오히려 나빠짐
- **잔차 학습** → F(x) = 목표 − x 만 배움 → 0 근처에서 시작하기 쉬움
- **덧셈 노드** → 기울기를 그대로 두 갈래로 보냄 (NN Day 3) → **∂y/∂x = F′(x) + 1**
- **F′ 가 아무리 작아져도** → **1 이 남아** 입력 쪽까지 기울기가 줄지 않음 — 블록이 쌓이면 지름길끼리 이어져 출력 → 입력의 "고속도로"
##### 자세히
```text
 x ──┬──► Conv-BN-ReLU ──► Conv-BN ──► (+) ──► ReLU ──► y
     └──────────── shortcut (identity or 1×1 conv) ──┘
 backward:  dL/dx = dL/dy · (F'(x) + 1)   → the "+1" path never vanishes
```

#### BasicBlock — 지름길 두 종류
> 모양이 같으면 Identity, 크기 1/2 · 채널 2배면 1×1 Conv(stride 2) + BN
| 상황 | 지름길 |
|---|---|
| 크기 · 채널 그대로 (예 16→16, stride 1) | `nn.Identity()` — x 를 그대로 |
| 크기 1/2 · 채널 2배 (예 16→32, stride 2) | `Conv 1×1 (stride 2) + BN` 으로 x 의 모양을 맞춤 |
- **16→32 stride 2 블록 파라미터** → conv1 16·32·9 = 4,608 · bn1 64 · conv2 32·32·9 = 9,216 · bn2 64 · 지름길 1×1 16·32 = 512 + BN 64 → **14,528**
- ⚠ 모양이 바뀌는 블록에 Identity 를 쓰면 더할 수 없어 오류 → **1×1 Conv 지름길**
- ⚠ `residual=False` 로 두면 지름길 Conv 는 만들어지되 forward 에서 안 쓰여 파라미터 수가 같은 plain 이 된다 (공정 비교용)

#### ResNet-20 (CIFAR-10 용, 깊이 = 6n + 2, n = 3)
> stem → 블록 3개씩 3단계(16 → 32 → 64, 크기 1/2) → GAP + Linear
| 단계 | 출력 크기 | 채널 | 구성 |
|---|---|---|---|
| stem | 32×32 | 16 | Conv3×3 + BN + ReLU |
| layer1 | 32×32 | 16 | BasicBlock × 3 |
| layer2 | 16×16 | 32 | BasicBlock × 3 (첫 블록 stride 2) |
| layer3 | 8×8 | 64 | BasicBlock × 3 (첫 블록 stride 2) |
| head | 1×1 | 64 → 10 | GAP + Linear |
- **가중치 층 수** → stem 1 + 블록 9 × 2 + Linear 1 = **20**
- **파라미터 272,474** → VGG-mini 307,946 보다 적고 MLP 의 1/6 — 거대 FC 대신 GAP + Linear(64 → 10) = **650** 개뿐
- **3 epoch 비교 (강사 실행 예)** → train 손실 plain-20 1.641 vs ResNet-20 **1.438**, test 0.380 vs **0.443** → 성능 저하가 사라짐
- **학습 전 첫 층 기울기** → plain-20 9.84e-01 vs ResNet-20 8.32e-01 (둘 다 BN 이 있어 시작은 문제없음, 차이는 학습이 진행되며 벌어짐)

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
- [NN D3·3교시 연쇄 법칙](#/nn/3/3) — 덧셈 노드 = 그대로 복사
- [VB D2·5교시 깊이의 문제 · BatchNorm](#/vb/2/5) — 풀려는 성능 저하 문제

**연관 문제**
<<QA 6>>

### 7교시 · 학습 기법: 데이터 증강 · 학습률 스케줄 · 혼동 행렬 · 연산량

**키워드**: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `train 에만`, `OneCycleLR 0.004 → 0.1 → 4e-07`, `배치마다 step()`, `혼동 행렬`, `cat → dog`, `MAC`, `forward hook`, `40,813,184`

**개요**: 같은 모델도 **학습 방법**에 따라 몇 %p 가 갈린다. 8교시 85% 를 위한 네 도구. ① **데이터 증강** — train 에만, 정답이 바뀌지 않는 변형(±4 px 이동 · 좌우 반전)으로 외우기를 막는다. ② **OneCycleLR** — 학습률을 작게 시작 → 최댓값 → 거의 0, 배치마다 step. ③ **혼동 행렬**로 "무엇을 무엇으로" 틀리는지 보고, ④ **MAC** 으로 연산량을 잰다 — 파라미터만으로는 속도를 알 수 없다.

**핵심 용어**
- **데이터 증강(augmentation)**: 학습할 때마다 이미지를 조금씩 바꿔 보여 줌 — 정답이 바뀌지 않는 변형만
- **OneCycleLR**: 학습률을 작게 시작 → 최댓값(warm-up) → 거의 0 까지 내리는 스케줄
- **혼동 행렬(confusion matrix)**: `M[i, j]` = 정답이 i 인데 j 로 예측한 개수
- **MAC**: 곱셈-덧셈 횟수 — 온디바이스에서 중요한 연산량
- **forward hook**: 계층이 실행될 때마다 불리는 함수 — 출력 shape 로 MAC 을 셀 때

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `transforms.RandomCrop()` | `transforms.RandomCrop(32, padding=4)` | 사방 0 패딩 후 무작위 위치에서 자르기 (±4 픽셀 이동) |
| `transforms.RandomHorizontalFlip()` | `transforms.RandomHorizontalFlip()` | 50% 확률 좌우 반전 |
| `OneCycleLR()` | `torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, epochs, steps_per_epoch)` | 학습률 스케줄러 |
| `np.bincount()` | `np.bincount(idx, minlength=K*K)` | 값별 개수 세기 (혼동 행렬) |
| `Module.register_forward_hook()` | `m.register_forward_hook(hook)` | 순전파 때 호출되는 함수 등록 |

**핵심 내용**

#### 데이터 증강 — train 에만
> 정답이 바뀌지 않는 변형으로 매번 다른 그림 → 외우기를 막고 위치 · 좌우 변화에 강해진다
- **`train_tf`** → `Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(), Normalize(MEAN, STD)])` — **Normalize 는 맨 뒤**
- **`RandomCrop(32, padding=4)`** → 사방 4 px 0 패딩 (40×40) 후 32×32 를 무작위로 → ±4 px 이동
- **`aug_set[7]`** → 꺼낼 때마다 새로 자르고 뒤집음 → 매번 다른 그림 (그릴 때는 `× std + mean` 으로 복원)
- **test** → 언제나 원본 그대로 (`plain_tf`)
- ⚠ 정답이 바뀌는 변형은 금지 (숫자 6 을 뒤집으면 9 → MNIST 에 상하 뒤집기 X)

#### OneCycleLR — 작게 → 크게 → 거의 0
> max_lr 0.1 이면 시작 0.004 → 30% 지점 0.1 → 끝 4e-07
| 지점 | 학습률 |
|---|---|
| 시작 | max_lr / 25 = **0.004** |
| 전체의 30% 지점 | 최댓값 **0.1** |
| 끝 | 시작의 10⁻⁴ 배 = **4e-07** |
- **처음 작게(warm-up)** → 초기의 불안정한 큰 기울기를 견딤 · **끝에 작게** → 바닥에 정착
- **`scheduler.step()`** → **배치마다** (epoch 마다 아님)
- ⚠ `steps_per_epoch` 는 `len(train_loader)` 와 같아야 한다 — 틀리면 학습 끝에 `Tried to step ... times` 오류
##### 자세히
```text
 lr
 0.1   ┤        ╱‾‾‾╲
       │      ╱       ╲
       │    ╱           ╲
 0.004 ┤──╱               ╲___
       └──┴───────┴──────────── step
          0      30%        100%
```

#### 혼동 행렬 — 무엇을 무엇으로 틀리나
> `np.bincount(true * K + pred, minlength=K*K).reshape(K, K)` — 행 = 정답, 열 = 예측
- **대각선 합 / 전체** → 정확도
- **헷갈리는 쌍** → 대각선을 0 으로 지우고 큰 칸 찾기 (실행 예: **cat → dog 180장**)
- **칸 번호 k** → 정답 = `k // 10`, 예측 = `k % 10`
- ⚠ 평가용 예측은 `model.eval()` + `torch.no_grad()`

#### MAC — 파라미터와 연산량은 따로
> Conv MAC = Cout · OH · OW · (Cin / G) · K² · Linear MAC = in · out
| 모델 | 파라미터 | MAC |
|---|---|---|
| MLP | 1,707,274 | 1,706,496 |
| LeNet-5 | 62,006 | 651,720 (conv1 352,800 + conv2 240,000 + FC 58,920) |
| VGG-mini | 307,946 | 38,653,952 |
| ResNet-20 | 272,474 | **40,813,184** (MLP 의 약 24배) |
- **ResNet-20** → 파라미터는 MLP 의 1/6 인데 MAC 은 **24배** — 합성곱은 같은 가중치를 모든 위치에서 다시 쓰기 때문
- **온디바이스** → 두 숫자를 함께 줄인다 (Quantization · 경량 구조)
- ⚠ hook 은 다 쓰면 `h.remove()` 로 떼야 다음 실행에 중복 등록되지 않는다

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
- [NN D4·5교시 과적합](#/nn/4/5) — 데이터를 늘리는 것이 가장 확실 (증강)
- [VB D1·5교시 학습 루프 · 평가 · 저장](#/vb/1/5) — 학습 루프 안에서 `scheduler.step()`

**연관 문제**
<<QA 7>>

### 8교시 · 미니 프로젝트: CIFAR-10 ResNet-20 85% 도전

**키워드**: `ResNet-20 + 증강`, `SGD momentum 0.9 · nesterov`, `weight_decay 5e-4`, `OneCycleLR 20 epoch`, `391 step`, `0.8587`, `가장 헷갈린 쌍`, `모델 비교표`

**개요**: 오늘 만든 것을 모두 합친다. ① ResNet-20 + 데이터 증강 + SGD(momentum · nesterov · weight decay) + OneCycleLR 로 **20 epoch** 학습해 test **85% 이상**. ② 혼동 행렬로 "가장 헷갈린 쌍" 3개와 오답 그림을 보며 **왜** 틀리는지 본다. ③ 네 모델(MLP · LeNet-5 · VGG-mini · ResNet-20)을 파라미터 · MAC · 정확도로 비교하되, 학습 조건이 다르다는 점을 해석에 쓴다.

**핵심 용어**
- **weight decay**: 가중치 크기에 벌점을 주는 L2 규제 (NN Day 4) — `weight_decay=5e-4`
- **nesterov**: momentum 의 앞서 보기 변형 (`nesterov=True`)
- **steps_per_epoch**: epoch 당 배치 수 = `len(loader)` — 50,000 / 128 → **391**

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 과제 1 — 증강 로더 · 학습 설정 | 7교시 aug_set 로 배치 128 로더, SGD(0.1, momentum 0.9, nesterov, wd 5e-4) · OneCycleLR(max_lr 0.1, 20 epoch, steps_per_epoch = len(loader)) |
| 과제 2 — 학습 | ResNet-20 20 epoch → test **0.85** 이상 (T4 로 epoch 당 약 20초, 약 7분) |
| 과제 3 — 오답 분석 | 혼동 행렬 → 대각선을 지운 뒤 가장 큰 3칸 = 가장 헷갈린 쌍 · 오답 그림 |
| 과제 4 — 모델 비교표 | MLP · LeNet-5 · VGG-mini · ResNet-20 의 파라미터 · MAC · test 를 표로 |

**핵심 내용**

#### 과제와 설정
> ResNet-20 + 증강 + SGD(nesterov, wd 5e-4) + OneCycle 20 epoch → 85% 이상
| 과제 | 내용 | 배점 |
|---|---|---|
| 1 | 증강 train 로더 · 학습 설정 (SGD · OneCycle) | 20 |
| 2 | ResNet-20 20 epoch 학습 — test ≥ 0.85 | 35 |
| 3 | 오답 분석 — 혼동 행렬 · 가장 헷갈린 쌍 3개 · 오답 그림 | 15 |
| 4 | 모델 비교표 (파라미터 · MAC · test) | 10 |
| 5 | 결과 해석 (마크다운) | 10 + 10 (강사) |

| 항목 | 값 |
|---|---|
| train 로더 | 7교시 `aug_set` (증강), 배치 128, 섞기 |
| 옵티마이저 | `SGD(lr 0.1, momentum 0.9, nesterov=True, weight_decay=5e-4)` |
| 스케줄러 | `OneCycleLR(max_lr=0.1, epochs=20, steps_per_epoch=len(aug_loader))` |
| step / epoch | **391** (50,000 / 128 = 390.6 → 올림) |
| 파라미터 | 272,474 |
- **강사 실행 예** → 20 epoch test **0.8587** (15 epoch 는 0.8479 로 목표 미달)
- **GPU 가 없으면** → 과제 2 는 강사가 학습한 가중치를 내려받음 (과제 2 점수 제외)
- **`fit(model, aug_loader, test_loader, EPOCHS, opt, sched, label=...)`** → **다섯 번째 인자가 scheduler**
- ⚠ 채점: LeNet-5 의 MAC 은 `651_720`, ResNet-20 의 파라미터는 `272_474`

#### 오답 분석과 모델 비교
> 정확도 하나가 아니라 "무엇을 왜 틀리는지" · "얼마나 비싼지"를 함께 본다
- **헷갈린 쌍** → `off = cm.copy(); np.fill_diagonal(off, 0)` → `np.argsort(off.ravel())[::-1][:3]` → 쌍 = `(CLASSES[k // 10], CLASSES[k % 10], 개수)`
- **비교표 조건이 다름** → MLP · LeNet · VGG 는 3 epoch, ResNet 은 20 epoch + 증강 → 해석에 반드시 언급
- ⚠ 구조 vs 학습 방법의 기여를 가르려면 **같은 조건**에서 다시 학습해야 공정
- ⚠ 온디바이스(Jetson · 라즈베리파이)라면 파라미터 · MAC 을 먼저 본다
##### 자세히
| 모델 | 파라미터 | MAC | test (학습 조건) |
|---|---|---|---|
| MLP | 1,707,274 | 1,706,496 | 0.4994 (3 epoch) |
| LeNet-5 | 62,006 | 651,720 | 0.5489 (3 epoch) |
| VGG-mini | 307,946 | 38,653,952 | 0.7526 (3 epoch) |
| ResNet-20 | 272,474 | 40,813,184 | 0.8587 (20 epoch + 증강) |
- 85% 다음: 같은 ResNet-20 을 약 160 epoch 학습하면 **91.25%** (원 논문). 더 깊게(ResNet-56 · 110) 또는 넓게 하면 93 ~ 95%
- 온디바이스는 반대로 정확도를 조금 내주고 연산을 크게 줄이는 방향 → **Day 3 의 MobileNet (depthwise separable conv, 2교시의 groups) 과 전이학습**

**기본 사용형태**
```python
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, epochs=20, steps_per_epoch=len(aug_loader))
hist_res = fit(model, aug_loader, test_loader, 20, opt, sched, label="ResNet-20")

pred, true = predict_all(model, test_loader)
cm = confusion_matrix(pred, true)
off = cm.copy(); np.fill_diagonal(off, 0)
top3 = np.argsort(off.ravel())[::-1][:3]            # 가장 헷갈린 쌍
```

**연결**
- [VB D2·6교시 ResNet: Residual block](#/vb/2/6) — 모델 정의
- [VB D2·7교시 학습 기법: 데이터 증강 · 학습률 스케줄 · 혼동 행렬 · 연산량](#/vb/2/7) — 증강 · OneCycle · 혼동 행렬 · MAC
- [NN D4·6교시 규제: weight decay · Dropout](#/nn/4/6) — weight decay (L2)
- [NN D4·2교시 옵티마이저 구현](#/nn/4/2) — Momentum

**연관 문제**
<<QA 8>>
