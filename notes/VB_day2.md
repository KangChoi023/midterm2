## Day 2 — CNN 대표 구조: LeNet · VGG · ResNet

### 1교시 · Day 1 복습 퀴즈 · CIFAR-10 과 MLP 의 한계

**키워드**: `CIFAR-10`, `32×32×3 = 3072`, `10 클래스`, `train 50,000 · test 10,000`, `정규화 상수는 train 으로만`, `MLP 1,707,274`, `0.4994`, `이웃 관계 상실`

**개요**: 오늘의 데이터는 MNIST(28×28 흑백)보다 실제 사진에 가까운 **CIFAR-10**(32×32 컬러, 10 클래스). ① 배경 · 크기 · 각도가 제각각이고 고양이 ↔ 개처럼 비슷한 클래스가 많아 "변화를 견디는 특징"을 배워야 한다. ② MNIST 에서 97% 를 낸 MLP 가 여기서는 파라미터 170만 개로도 **약 50%** — 이미지를 3072 개 숫자의 줄로 펼치는 순간 이웃 픽셀 정보가 사라지기 때문. ③ 합성곱(지역성 · 가중치 공유)이 이 문제를 푼다 → 오늘 LeNet · VGG · ResNet 으로 85% 까지.

**한눈에 보기**
CIFAR-10 과 MLP 의 한계 — 컬러 사진에는 공간 구조가 필요
> 32 × 32 RGB 사진 10 클래스(train 50,000 · test 10,000) — 배경 · 크기 · 각도가 제각각이라, 픽셀을 펼쳐 버리는 MLP 로는 절반밖에 못 맞힌다
- **입력이 MNIST 의 4배 — 32·32·3 = 3072**: 고양이 ↔ 개, 자동차 ↔ 트럭처럼 비슷한 클래스도 있다
- **정규화 상수는 train 으로만 계산**: 채널별 평균 · 표준편차 `mean(axis=(0, 1, 2))` → test 에도 같은 값으로 `Normalize`
- **MLP 3072-512-256-10 → 3 epoch test 0.4994**: 파라미터 1,707,274 (첫 층이 92%) — MNIST 에선 97% 였던 구조
- **오늘의 모델**: LeNet-5 62,006 · VGG-mini 307,946 · ResNet-20 272,474 (85% 목표)
- **⚠ test 로 정규화 상수를 구하면 시험 문제를 미리 본 것**: 학습량이 Day 1 의 수십 배라 T4 GPU 로

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

#### CIFAR-10
> 32 × 32 RGB 컬러 사진 10 클래스, train **50,000** · test **10,000** — 배경 · 크기 · 각도가 제각각이라 "변화를 견디는 특징" 이 필요
- **`datasets.CIFAR10()`** → 원본 `train_set.data` 는 (50000, 32, 32, 3) uint8 HWC — 첫 다운로드 약 170 MB
- **MNIST 와 비교** → 입력 28·28·1 = 784 vs 32·32·3 = **3072** (4배) · 비슷한 클래스(고양이 ↔ 개, 자동차 ↔ 트럭)
- ⚠ 학습량이 Day 1 의 수십 배 — `device : cpu` 면 T4 GPU 로

#### 정규화 상수는 train 으로만
> 정규화 상수(MEAN · STD) = **train** 데이터의 채널별 평균 · 표준편차 → test 에도 같은 값
- **계산** → `(train_set.data / 255).mean(axis=(0, 1, 2))` — N · H · W 축을 평균해 채널 3개만 남김
- **`transforms.Normalize()`** → 채널마다 (x − MEAN) / STD · **`transforms.Compose()`** 로 `[ToTensor(), Normalize(MEAN, STD)]` 를 묶음
- ⚠ test 로 계산하면 시험 문제를 미리 본 것 (NN Day 4 validation 원칙)

#### MLP 의 한계 — 이웃 관계 상실
> Flatten 하는 순간 바로 아래 픽셀은 32칸, 다음 채널은 1024칸 떨어진다 — 파라미터를 늘려도 **이웃 관계(공간 구조)** 를 못 본다
- **MLP 3072-512-256-10** → 파라미터 **1,707,274** (첫 층이 92%)
- **예** → 3 epoch test **0.4994** (강사 실행 예) — MNIST 에선 97% 였던 구조가 절반
- **오늘의 모델** → LeNet-5 62,006 · VGG-mini 307,946 · ResNet-20 272,474 (85% 목표)

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

**상세 학습**

#### 이 교시에서 답하는 질문
Day 1 은 MNIST(가운데 놓인 흑백 숫자)였다. 오늘은 실제 사진에 가까운 **CIFAR-10** 으로 간다. 이 교시는 새 데이터가 무엇이 어려운지, 정규화 상수를 어떻게 구하는지 보고, MNIST 에서 97% 를 낸 MLP 가 여기서는 왜 절반밖에 못 맞히는지 확인한다 — 오늘 배울 CNN 구조들의 출발점이다.

#### 개념 1 · CIFAR-10 — 변화가 많은 사진
| 항목 | MNIST | CIFAR-10 |
|---|---|---|
| 크기 | 28 × 28 × 1 = 784 | 32 × 32 × 3 = 3072 (4배) |
| 내용 | 가운데 놓인 숫자 | 배경 · 크기 · 각도 제각각 |
| 클래스 | 숫자 0 ~ 9 | airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck |
| 개수 | train 60,000 · test 10,000 | train 50,000 · test 10,000 |

고양이 ↔ 개, 자동차 ↔ 트럭처럼 비슷한 클래스가 많아, 위치 · 크기가 바뀌어도 견디는 특징을 배워야 한다.

##### 🔎 왜 이렇게 되나
- **비유**: MNIST 가 증명사진이라면 CIFAR-10 은 길거리 스냅 사진 — 배경 · 크기 · 각도가 제각각
- **① 어려운 이유**: 같은 클래스도 모양이 크게 다르고(앉은 고양이 · 누운 고양이), 다른 클래스끼리 닮았다(고양이 ↔ 개, 자동차 ↔ 트럭)
- **② 입력이 4배**: 32 × 32 × 3 = 3072 — 채널이 3개라 색 정보까지 본다
- **③ 그래서 필요한 것**: 위치 · 크기가 바뀌어도 같은 특징을 찾는 구조(합성곱)와 더 많은 학습
- ⚠ 학습량이 Day 1 의 수십 배 — CPU 면 GPU 런타임으로 바꾼다

<<CODE 1-c1 | CIFAR-10 불러오기 — ToTensor + Normalize · DataLoader>>

#### 개념 2 · 정규화 상수는 train 으로만
Day 1 의 정규화((x − mean) / std)를 CIFAR-10 에 쓰려면 이 데이터의 채널별 평균 · 표준편차가 필요하다. 반드시 **train** 데이터로 계산하고 test 에도 같은 값을 쓴다. test 로 계산하면 시험 문제를 미리 본 셈이다 (NN Day 4 의 validation 원칙).

원본 `train_set.data` 는 (50000, 32, 32, 3) — N · H · W 세 축을 평균하면(`axis=(0, 1, 2)`) 채널 3개만 남는다.

##### 🔎 왜 이렇게 되나
- **비유**: 시험 채점 기준(평균 · 표준편차)은 시험 보기 전에 정해야 한다 — 시험지를 보고 정하면 부정행위
- **① 정규화가 필요한 이유**: 채널마다 밝기 범위가 달라도 평균 0 · 표준편차 1 근처로 맞춰 학습을 고르게 (VB D1 6교시)
- **② train 으로만 계산하는 이유**: test 는 "처음 보는 데이터" 여야 성능을 믿을 수 있다 (NN Day 4 validation 원칙)
- **③ axis=(0, 1, 2)**: 사진 · 세로 · 가로 세 축을 모아서 없애면 채널 3개만 남는다
- ⚠ 사전학습 모델(Day 3)엔 이 값이 아니라 ImageNet 평균 · 표준편차를 쓴다

<<CODE 1-1 | 정규화 상수 MEAN · STD 를 train 데이터로 계산>>

#### 개념 3 · MLP 는 절반만 맞힌다
| 층 | 계산 | 파라미터 |
|---|---|---|
| Linear(3072, 512) | 3072·512 + 512 | 1,573,376 |
| Linear(512, 256) | 512·256 + 256 | 131,328 |
| Linear(256, 10) | 256·10 + 10 | 2,570 |
| 합계 | | **1,707,274** |

3 epoch test 는 약 **0.50**(강사 실행 예 0.4994). 이미지를 3072개 숫자의 줄로 펼치면 바로 아래 픽셀은 32칸, 다음 채널의 같은 위치는 1024칸 떨어진다 — NN Day 5 에서 본 그 문제다. 파라미터를 170만 개로 늘려도 공간 구조를 보지 못한다.

![이미지를 펼치면 이웃 픽셀이 멀리 떨어진다 (NN Day 5 그림 — CIFAR 는 한 줄이 32칸)](assets/fig/nn/d5-flatten.svg)

##### 🔎 왜 이렇게 되나
- **비유**: 사진을 한 줄로 길게 잘라 이어 붙이면, 위아래로 붙어 있던 픽셀이 멀리 떨어진다
- **① 이웃을 잃는 이유**: Flatten 하면 바로 아래 픽셀은 32칸, 다음 채널의 같은 자리는 1024칸 떨어진다 — MLP 는 그 관계를 처음부터 배워야 한다
- **② 파라미터만 많다**: 1,707,274개 중 첫 층이 92% — 늘려도 "이웃" 이라는 구조를 모른다
- **③ 결과**: 3 epoch test 약 0.50 — MNIST 에선 97% 였던 구조가 절반
- ⚠ 파라미터 수가 많다고 잘하는 게 아니다 — 데이터에 맞는 구조가 중요하다 (3교시 귀납적 편향)

<<CODE 1-2 | CIFAR-10 용 MLP 3072 → 512 → 256 → 10>>
<<CODE 1-3 | MLP 를 Adam 으로 3 epoch 학습>>

#### 한 번에 정리
- CIFAR-10: 32 × 32 × 3 컬러, 10 클래스, train 5만 · test 1만 — 변화가 많다.
- 정규화 상수는 train 으로만 계산해 test 에도 같은 값.
- MLP 는 170만 파라미터로 약 50% — 펼치면 이웃 관계가 사라진다.
- ⚠ 오늘은 학습량이 많다 — GPU(T4) 로.

**연관 문제**
<<QA 1>>

### 2교시 · Conv 다시 보기: 출력 크기 · 파라미터 · 수용 영역

**키워드**: `kernel K`, `stride S`, `padding P`, `groups G`, `O = (N + 2P − K) // S + 1`, `Cout · (Cin / G) · K² + Cout`, `same 패딩`, `depthwise 1/58`, `수용 영역`, `bias=False + BN`

**개요**: `nn.Conv2d` 의 인자 하나하나가 **출력 크기 · 파라미터 수 · 연산량**을 정한다 — 오늘 만드는 모든 모델은 이 계산으로 설계된다. ① 출력 크기 `(N + 2P − K) // S + 1`, 파라미터 `Cout · (Cin / G) · K² + Cout` 두 공식. ② `groups = Cin` 이면 채널마다 따로 합성곱(depthwise) — 파라미터가 약 1/58 (Day 3 MobileNet). ③ 층을 쌓고 크기를 줄일수록 출력 한 칸이 보는 입력 영역(**수용 영역**)이 넓어진다 — 3×3 두 층 = 5×5.

**한눈에 보기**
Conv 다시 보기 — 출력 크기 · 파라미터 · 수용 영역
> nn.Conv2d 의 인자(cin, cout, k, stride, padding, groups)가 출력 크기 O = (N + 2P − K) // S + 1 과 파라미터 Cout · (Cin / G) · K² + Cout 을 정한다
- **K 3 · P 1 · S 1 이면 크기 그대로**: 32 → 32 · S 2 면 32 → 16
- **파라미터 예**: Conv(3→6, 5×5) = 456 · Conv(64→128, 3×3) = 73,856 — 뒤에 BN 이 오면 `bias=False`
- **groups = Cin 이면 depthwise**: 채널마다 따로 합성곱 → 64→64 3×3 이 36,928 → 640 (Day 3 MobileNet)
- **수용 영역 — 출력 한 칸이 보는 입력 영역**: rf ← rf + (K − 1)·jump, jump ← jump·S — 3×3 두 층 = 5×5, 세 층 = 7×7
- **크기 1/2 · 채널 2배 설계**: 위치 수 1/4 × 채널 2배 → 층마다 연산량을 비슷하게
- **⚠ `//` 는 내림이라 가장자리가 버려질 수 있다**: 수용 영역을 셀 때 풀링도 층으로 넣는다 (K 2, S 2)

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

#### Conv 의 두 공식
> 출력 크기 O = (N + 2P − K) // S + 1 · 파라미터 = Cout · (Cin / G) · K² + Cout
- **`nn.Conv2d()`** → `nn.Conv2d(cin, cout, k, stride, padding, groups, bias)` — 인자 하나하나가 크기 · 파라미터 · 연산량을 정한다
- **same 패딩** → K 3 · P 1 · S 1 이면 크기 그대로 (32 → 32) · S 2 면 32 → 16
- **예** → Conv(3→6, 5×5) = 3·6·25 + 6 = **456** · Conv(64→128, 3×3) = 73,856
- **`bias=False`** → 바로 뒤에 BN 이 올 때 (BN 의 β 가 bias 역할)
- ⚠ `//` 는 내림 — 나누어떨어지지 않으면 오른쪽 · 아래 가장자리가 버려진다 (7, 3, 2, 0 → 3)

#### groups — depthwise 합성곱
> **groups(G)** = 입력 채널을 G 묶음으로 나눠 묶음끼리만 합성곱 → 파라미터 ÷ G
- **depthwise 합성곱** → `groups = Cin` — 채널마다 따로
- **예** → 64→64 3×3: 일반 36,928 → depthwise **640** (약 1/58) — Day 3 MobileNet 의 핵심
- **크기 1/2 · 채널 2배** → 위치 수 1/4 × 채널 2배 → 연산량은 절반 — 층마다 연산량을 비슷하게 맞추는 설계

#### 수용 영역(receptive field)
> 출력 한 칸이 영향을 받는 입력 영역 — 층마다 rf ← rf + (K − 1)·**jump**, jump ← jump·S (시작 1, 1)
- **예** → 3×3 두 층 = **5×5**, 세 층 = **7×7**
- **jump** → 현재 층 한 칸이 입력에서 몇 픽셀 간격인지 — stride · 풀링마다 곱해져 같은 3×3 이 더 넓게
- **LeNet** 16 · **VGG-mini** 마지막 합성곱 32 → 32 × 32 이미지 **전체**를 본다
- ⚠ 풀링도 층으로 넣어 센다 (K 2, S 2)

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

**상세 학습**

#### 이 교시에서 답하는 질문
오늘 만드는 모든 모델(LeNet · VGG · ResNet)은 `nn.Conv2d` 를 쌓아 만든다. 인자를 어떻게 정하느냐에 따라 출력 크기 · 파라미터 수 · 연산량이 정해진다. 이 교시는 그 두 공식을 다시 정리하고, 층을 쌓을수록 출력 한 칸이 "얼마나 넓게 보는지"(수용 영역)를 계산한다.

#### 개념 1 · 두 공식
> 출력 크기 O = (N + 2P − K) // S + 1
> 파라미터 = Cout · (Cin / G) · K² + Cout

| 인자 | 하는 일 | 공식에서 |
|---|---|---|
| K (kernel_size) | 한 번에 보는 영역 | 파라미터 ∝ K² |
| S (stride) | 건너뛰는 간격 | 출력 ÷ S |
| P (padding) | 가장자리에 0 덧대기 | 출력 + 2P |
| G (groups) | 입력 채널을 G 묶음으로 나눔 | 파라미터 ÷ G |

| 설정 | 출력 · 파라미터 |
|---|---|
| N 32, K 3, S 1, P 1 | 32 (same 패딩) |
| N 32, K 3, S 2, P 1 | 16 |
| Conv(3 → 6, 5×5) | 3·6·25 + 6 = 456 |
| Conv(64 → 64, 3×3) 일반 / groups = 64 | 36,928 / **640** (약 1/58) |

`groups = Cin` 이면 채널마다 따로 합성곱하는 **depthwise** — Day 3 MobileNet 의 핵심이다.

##### 🔎 왜 이렇게 되나
- **비유**: 도장(필터)을 종이 위에 일정 간격으로 찍는 것 — 도장 크기 K, 간격 S, 종이 테두리 여백 P
- **① 출력 크기 식이 그 모양인 이유**: 여백을 붙인 길이 N + 2P 에서 도장이 들어갈 수 있는 시작 위치의 수가 (N + 2P − K) // S + 1
- **② 파라미터 식**: 필터 하나 = (Cin / G) · K² 개 가중치, 필터가 Cout 개 + 편향 Cout 개
- **③ groups**: 입력 채널을 G 묶음으로 나눠 묶음끼리만 계산 → 파라미터 ÷ G (Day 3 depthwise 의 바탕)
- ⚠ `//` 는 내림 — 나누어떨어지지 않으면 오른쪽 · 아래 가장자리가 버려진다

<<CODE 2-1 | conv_out — 출력 크기 공식을 nn.Conv2d 로 확인>>
<<CODE 2-2 | conv_params — 파라미터 수 공식 (groups 포함)>>

#### 개념 2 · 크기 1/2 · 채널 2배
stride 2 로 크기를 절반으로 줄이면 위치 수는 1/4. 채널을 2배로 늘려도 연산량은 절반이다. 그래서 CNN 은 보통 "크기 1/2 마다 채널 2배" 로 설계해 층마다 연산량을 비슷하게 맞춘다. 예: Conv(64 → 128, 3×3, S2, P1) 에 (2, 64, 32, 32) → (2, 128, 16, 16).

##### 🔎 왜 이렇게 되나
- **비유**: 지도를 축소할 때마다(크기 1/2) 표시하는 정보 종류(채널)를 2배로 늘리는 것
- **① 연산량이 절반인 이유**: 크기 1/2 → 위치 수 1/4, 채널 2배 → 위치당 계산 2배(출력 채널) → 1/4 × 2 = 1/2 (입력 채널도 2배면 층마다 비슷)
- **② 왜 이렇게 설계하나**: 층마다 연산량이 고르게 되고, 뒤로 갈수록 "무엇이 있나" 같은 추상 정보를 많이 담는다
- **③ stride 2 · padding 1**: 3×3 에서 크기를 정확히 절반으로 (32 → 16)
- ⚠ 크기만 줄이고 채널을 안 늘리면 정보가 너무 빨리 줄어든다

<<CODE 2-4 | 크기 1/2 · 채널 2배 Conv 설계 — stride 2 · padding 1>>

#### 개념 3 · 수용 영역 — 쌓을수록 넓게 본다
출력 한 칸이 영향을 받는 입력 영역의 크기를 **수용 영역**이라 한다.

![3 × 3 두 층이 보는 입력은 5 × 5 — 수용 영역 계산](assets/fig/vb/d2-receptive.svg)

> 층마다 rf ← rf + (K − 1) · jump, jump ← jump · S (시작 rf = 1, jump = 1)

| 구성 | 수용 영역 |
|---|---|
| 3×3 두 층 | 5 |
| 3×3 세 층 | 7 |
| LeNet (5 · 풀2 · 5 · 풀2) | 16 |
| VGG-mini 마지막 합성곱 | 32 — 32 × 32 이미지 전체 |

jump 는 "현재 층 한 칸이 입력에서 몇 픽셀 간격인지" 다. 풀링 · stride 를 지날 때마다 곱해져, 뒤쪽의 같은 3×3 이 입력에서는 훨씬 넓게 퍼진다.

##### 🔎 왜 이렇게 되나
- **비유**: 작은 창을 여러 겹 겹치면 맨 위 한 칸이 맨 아래에서 보는 범위가 넓어진다
- **① rf ← rf + (K − 1) · jump 인 이유**: 새 층의 K 칸은 아래 층 칸 사이 간격(jump)만큼 떨어져 있으니, 넓어지는 폭이 (K − 1) · jump
- **② jump 가 곱해지는 이유**: stride · 풀링을 지날 때마다 한 칸이 입력에서 차지하는 간격이 S 배
- **③ 3×3 두 층 = 5, 세 층 = 7**: 작은 필터를 쌓아도 큰 필터와 같은 범위를 본다 (4교시 VGG 의 바탕)
- ⚠ 수용 영역을 셀 때 풀링도 층으로 넣는다 (K 2, S 2)

<<CODE 2-3 | receptive_field — 쌓을수록 넓어지는 수용 영역>>

#### 한 번에 정리
- O = (N + 2P − K) // S + 1, 파라미터 = Cout · (Cin / G) · K² + Cout.
- groups = Cin 이면 depthwise — 파라미터 약 1/58.
- 크기 1/2 마다 채널 2배 → 층마다 연산량이 비슷.
- 수용 영역: 3×3 두 층 = 5×5, 풀링 뒤엔 더 넓게.
- ⚠ `//` 는 내림 — 가장자리가 버려질 수 있다.

**연관 문제**
<<QA 2>>

### 3교시 · LeNet-5 · AlexNet

**키워드**: `LeNet-5 (1998)`, `Conv-Pool-Conv-Pool-FC-FC-FC`, `62,006`, `0.5489`, `귀납적 편향`, `AlexNet (2012)`, `ILSVRC top-5 15.3%`, `ReLU · Dropout · 증강 · GPU`, `FC 한 층 61.8%`

**개요**: CNN 의 역사를 두 모델로 본다. ① **LeNet-5**(1998, LeCun)는 Conv-Pool-Conv-Pool-FC-FC-FC 뼈대를 만든 최초의 실용 CNN — 파라미터 **6만**으로 170만 MLP 보다 높다(귀납적 편향). ② **AlexNet**(2012)은 이를 키워 ImageNet 대회에서 2위를 10%p 이상 앞섰고 ReLU · Dropout · 데이터 증강 · GPU 학습을 정착시켰다 — 이후 우승 모델은 모두 CNN, 해마다 깊어졌다. ③ AlexNet 파라미터의 62% 가 FC 한 층 → "파라미터는 FC 에, 연산은 Conv 에" → 뒤 모델들은 거대 FC 를 줄인다.

**한눈에 보기**
LeNet-5 · AlexNet — CNN 의 뼈대와 도약
> LeNet-5(1998)가 Conv-Pool-Conv-Pool-FC 뼈대를 만들었고, AlexNet(2012)이 ImageNet 대회에서 압승하며 딥러닝 시대를 열었다
- **LeNet-5 — 파라미터 1/27 로 MLP 보다 높다**: 62,006개, 3 epoch test 0.5489 (MLP 170만 개 0.4994)
- **귀납적 편향 — 구조가 미리 가진 가정**: 이웃 픽셀 · 위치 불변이 데이터 없이도 학습을 돕는다
- **AlexNet — ILSVRC 2012 top-5 오류 15.3% (2위 26.2%)**: ReLU · Dropout · 데이터 증강 · GPU 학습을 정착시켰다
- **파라미터는 FC 에, 연산은 Conv 에**: 약 6,110만 개 중 Linear(9216 → 4096) 한 층이 61.8%, 합성곱 5층은 4.0%
- **⚠ 그래서 뒤의 모델들은 거대 FC 를 줄인다**: ResNet 은 GAP + FC 하나

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

#### LeNet-5 (1998)
> Conv-Pool-Conv-Pool-FC-FC-FC 뼈대를 만든 최초의 실용 CNN — CIFAR 판 파라미터 **62,006**
- **shape** → (3, 32, 32) → conv5 (6, 28, 28) → pool (6, 14, 14) → conv5 (16, 10, 10) → pool (16, 5, 5) → 400 → 120 → 84 → 10
- **예** → 3 epoch test MLP 0.4994 (170만) vs LeNet-5 **0.5489** (6만, 1/27)
- **귀납적 편향(inductive bias)** → 구조가 미리 가진 가정(이웃 픽셀 · 위치 불변)이 데이터 없이도 학습을 돕는다
- **원래 LeNet** → 32 × 32 흑백 · tanh · 평균 풀링 (오늘은 컬러 · ReLU · Max)

#### AlexNet (2012)
> **ILSVRC** 2012 top-5 오류 **15.3%** (2위 26.2%) — 이후 우승 모델은 모두 CNN
- **ILSVRC** → ImageNet 대회 1,000 클래스 · 약 128만 장 · 224 × 224 · **top-5 오류** = 상위 5개 안에 정답이 없는 비율
- **정착시킨 기법** → ReLU · Dropout · 데이터 증강 · GPU 학습 (GTX 580 두 장, 5 ~ 6일)
- **해마다 깊어짐** → AlexNet 8층 → VGG 19 → GoogLeNet 22 → ResNet 152

#### 파라미터는 FC 에, 연산은 Conv 에
> AlexNet 약 6,110만 파라미터 중 Linear(9216 → 4096) 한 층이 **61.8%**
- **`torchvision.models.alexnet()`** → `weights=None` 으로 구조만 · **`Module.named_parameters()`** 로 (이름, 파라미터) 쌍을 세기
- **예** → `classifier.1` = 9216 × 4096 = **37,748,736** · 합성곱 5층 합계 2,469,696 (**4.0%**)
- **그래서** → 뒤의 모델들은 거대 FC 를 줄인다 (ResNet = GAP + FC 하나)

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

**상세 학습**

#### 이 교시에서 답하는 질문
CNN 이 어떻게 지금의 모습이 되었는지 두 모델로 본다. 최초의 실용 CNN 인 LeNet-5 는 MLP 와 무엇이 달랐고, 딥러닝 붐을 연 AlexNet 은 무엇을 더했을까? 그리고 두 모델에서 "파라미터는 어디에 몰려 있나" 를 센다.

#### 개념 1 · LeNet-5 — 뼈대를 만들다
1998년 LeCun 의 LeNet-5 는 Conv-Pool-Conv-Pool-FC-FC-FC 구조로 우편번호 숫자를 읽었다. 오늘은 CIFAR 용(컬러 · ReLU · Max 풀링)으로 만든다.

![LeNet-5 의 구조와 단계별 shape](assets/fig/vb/d2-lenet.svg)

| 층 | 출력 | 파라미터 |
|---|---|---|
| conv1 3→6, 5×5 → pool | (6, 28, 28) → (6, 14, 14) | 456 |
| conv2 6→16, 5×5 → pool | (16, 10, 10) → (16, 5, 5) | 2,416 |
| fc 400→120 · 120→84 · 84→10 | 10 | 48,120 + 10,164 + 850 |
| 합계 | | **62,006** |

MLP 의 1/27 파라미터로 test 0.5489 > 0.4994. 합성곱이라는 구조가 "이웃 픽셀이 중요하다, 위치가 달라도 같은 특징이다" 라는 가정을 미리 갖고 있어서다 — 이것을 **귀납적 편향**이라 한다.

##### 🔎 왜 이렇게 되나
- **비유**: 오늘날 CNN 의 설계도 원본 — "특징 뽑기(Conv-Pool) 두 번 → 판단(FC)"
- **① Conv-Pool 을 반복하는 이유**: Conv 가 특징을 찾고 Pool 이 크기를 줄여, 다음 Conv 가 더 넓은 범위를 본다
- **② 6만 개로 MLP 170만 개보다 잘하는 이유**: "이웃 픽셀이 중요" · "같은 특징은 어디서나 같다" 는 가정을 구조에 넣었기 때문 (귀납적 편향)
- **③ 오늘 판과 원래 판**: 원래는 흑백 · tanh · 평균 풀링 — 오늘은 컬러 · ReLU · Max
- ⚠ 3 epoch 0.5489 는 MLP(0.4994)보다 낫지만 아직 낮다 — 더 깊은 구조가 필요 (4 · 6교시)

<<CODE 3-1 | LeNet-5 를 nn.Module 로 — 파라미터 62,006>>
<<CODE 3-2 | LeNet-5 를 MLP 와 같은 조건으로 학습>>

#### 개념 2 · AlexNet — 딥러닝의 귀환
2012년 ImageNet 대회(ILSVRC: 1,000 클래스 · 약 128만 장)에서 AlexNet 은 top-5 오류 **15.3%** 로 2위(26.2%)를 10%p 이상 앞섰다. 이후 우승 모델은 모두 CNN 이고 해마다 깊어졌다 (8층 → VGG 19 → GoogLeNet 22 → ResNet 152).

| 기법 | 이유 | 이 과정에서 |
|---|---|---|
| ReLU | 기울기 소실이 적고 빠름 | NN Day 2 |
| Dropout | 거대한 FC 의 과적합 방지 | NN Day 4 |
| 데이터 증강 | 학습 데이터를 부풀림 | 7교시 |
| GPU 학습 | GTX 580 두 장으로 5 ~ 6일 | Colab T4 |

##### 🔎 왜 이렇게 되나
- **비유**: 오래 잠자던 기술(신경망)이 GPU · 큰 데이터라는 연료를 만나 다시 달리기 시작한 순간
- **① 압도적인 이유**: top-5 오류 15.3% vs 2위 26.2% — 손으로 만든 특징 대신 데이터에서 특징을 배웠다
- **② ReLU · Dropout · 증강 · GPU**: 기울기 소실 완화 · 거대 FC 의 과적합 방지 · 데이터 늘리기 · 학습 시간 단축 — 모두 이 과정에서 배운 것
- **③ 그 뒤**: 해마다 더 깊어졌다 (8층 → 19 → 22 → 152)
- ⚠ AlexNet 의 성공은 구조 하나가 아니라 기법 묶음 + 계산 자원의 결과

#### 개념 3 · 파라미터는 FC 에, 연산은 Conv 에
AlexNet 약 6,110만 파라미터 중 Linear(9216 → 4096) 한 층이 **61.8%**(37,748,736), 합성곱 5층은 모두 합쳐 4.0%. NN Day 5 SimpleConvNet 에서 본 것과 같은 결론이다. 그래서 뒤의 모델들은 거대 FC 를 줄여 간다 — ResNet 은 GAP + FC 하나.

##### 🔎 왜 이렇게 되나
- **① FC 에 파라미터가 몰리는 이유**: 입력 9216 개 × 출력 4096 개를 모두 연결 — 한 층이 37,748,736개 (61.8%)
- **② Conv 는 적은 이유**: 작은 필터를 모든 위치에서 다시 쓰므로(가중치 공유) 이미지 크기와 무관
- **③ 연산은 반대**: Conv 는 적은 가중치를 수많은 위치에서 곱하니 연산량이 크다 (7교시 MAC)
- **④ 그래서**: 뒤 모델들은 거대 FC 를 없앤다 — ResNet 은 GAP + FC 하나 (650개)
- ⚠ 파라미터 수(저장 크기)와 연산량(속도)은 다른 숫자다

<<CODE 3-3 | AlexNet 구조 — 전체 파라미터와 가장 큰 텐서>>

#### 한 번에 정리
- LeNet-5: Conv-Pool-Conv-Pool-FC×3, 62,006 파라미터로 MLP(170만)보다 높다 — 귀납적 편향.
- AlexNet(2012): top-5 15.3%, ReLU · Dropout · 증강 · GPU 를 정착.
- 파라미터는 FC 에 몰린다 (AlexNet 61.8%) → 뒤의 모델은 FC 를 줄인다.
- ⚠ 원래 LeNet 은 흑백 · tanh · 평균 풀링 — 오늘 것은 현대판.

**연관 문제**
<<QA 3>>

### 4교시 · VGG: 3×3 을 쌓는 이유

**키워드**: `VGG (2014)`, `3×3 두 층 = 5×5 수용 영역`, `18C² vs 25C² (28% ↓)`, `27C² vs 49C² (45% ↓)`, `비선형 2번`, `VGG 블록`, `VGG-mini 307,946`, `0.7526`, `크기 1/2 · 채널 2배`

**개요**: VGG(2014, Oxford)는 모든 합성곱을 **3×3 으로 통일**하고 같은 블록을 반복한다. ① 3×3 두 층은 5×5 한 층과 **같은 수용 영역**을 보면서 파라미터는 28% 적고 ReLU 는 두 번 들어간다. ② 블록 = [Conv3×3 → BN → ReLU] × n → MaxPool, 크기가 1/2 이 될 때마다 채널은 2배. ③ VGG-mini 는 3 epoch 에 **0.7526** — LeNet 보다 크게 높다. 그럼 끝없이 깊게 쌓으면? → 5교시.

**한눈에 보기**
VGG — 3×3 을 쌓는 설계
> 3×3 두 층은 5×5 한 층과 같은 수용 영역을 더 적은 파라미터로 보고 ReLU 를 두 번 거친다 — VGG 는 이 3×3 블록을 쌓아 만든다
- **5×5 한 층 25C² vs 3×3 두 층 18C² (28% ↓)**: 7×7 49C² vs 3×3 세 층 27C² (45% ↓) — C = 64 면 102,400 vs 73,728
- **VGG 블록 = [Conv3×3 → BN → ReLU] × n → MaxPool**: 크기가 1/2 이 될 때마다 채널 2배
- **VGG-mini — 3 epoch test 0.7526**: 블록 3개(32 → 64 → 128) + Linear(2048 → 10), 파라미터 307,946 (LeNet 0.5489)
- **정확도는 오르지만 연산도 는다**: MAC 38,653,952 — LeNet 의 약 59배
- **⚠ 원래 VGG(2014)에는 BN 이 없었다**: 오늘 것은 현대판, BN 앞 Conv 는 `bias=False`

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

#### 3 × 3 을 쌓는 이유
> 3 × 3 두 층 = 5 × 5 한 층과 **같은 수용 영역** · 파라미터는 적고 · ReLU 는 두 번
- **파라미터 (채널 C, bias 없음)** → 5×5 한 층 25C² vs 3×3 두 층 18C² (**28%** ↓) · 7×7 49C² vs 3×3 세 층 27C² (**45%** ↓)
- **예** → C = 64: 102,400 vs 73,728
- **비선형 2번** → 같은 영역을 보면서 더 복잡한 함수 — 현대 CNN 은 대부분 3 × 3

#### VGG 블록 · VGG-mini
> **VGG 블록** = [Conv3×3 → BN → ReLU] × n → MaxPool 2×2 — 크기가 1/2 이 될 때마다 채널 2배
- **VGG-mini** → 블록 3개(32 → 64 → 128) → (N, 128, 4, 4) → Flatten → Linear(2048 → 10) — 파라미터 **307,946**
- **`nn.BatchNorm2d()`** → Conv 출력을 채널마다 정규화 (BatchNorm2d 자세히 5교시) — 앞 Conv 는 `bias=False`
- **예** → 3 epoch test **0.7526** (LeNet 0.5489) · MAC 38,653,952 (LeNet 의 약 59배)
- ⚠ 원래 VGG(2014)에는 BN 이 없었다 — 오늘 것은 현대판

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

**상세 학습**

#### 이 교시에서 답하는 질문
AlexNet 은 11×11, 5×5 같은 큰 필터를 썼다. 2014년 VGG 는 **모든 필터를 3×3 으로** 통일하고 같은 블록을 반복했다. 왜 작은 필터를 쌓는 것이 나을까? 그리고 그렇게 만든 VGG-mini 는 LeNet 보다 얼마나 좋을까?

#### 개념 1 · 3×3 두 층 vs 5×5 한 층
2교시에서 3×3 두 층의 수용 영역이 5×5 라는 것을 봤다. 같은 영역을 보는 두 구성을 비교하면 (채널 C, bias 없음):

| 구성 | 파라미터 | C = 64 | ReLU |
|---|---|---|---|
| 5×5 한 층 | 25C² | 102,400 | 1번 |
| 3×3 두 층 | 18C² | 73,728 (**28%** ↓) | 2번 |
| 7×7 한 층 | 49C² | 200,704 | 1번 |
| 3×3 세 층 | 27C² | 110,592 (**45%** ↓) | 3번 |

같은 영역을 보면서 파라미터는 적고, 비선형(ReLU)은 더 많이 들어가 더 복잡한 함수를 표현한다. 현대 CNN 은 대부분 3×3 이다.

##### 🔎 왜 이렇게 되나
- **비유**: 큰 그물 한 번보다 작은 그물 두 번 — 같은 범위를 덮으면서 재료(파라미터)는 적게
- **① 같은 범위인 이유**: 3×3 두 층의 수용 영역 = 5 (2교시 공식)
- **② 파라미터가 적은 이유**: 9C² × 2 = 18C² < 25C² (28% ↓) · 세 층이면 27C² < 49C² (45% ↓)
- **③ 비선형이 늘어나는 이유**: 층마다 ReLU 가 끼어 같은 범위를 더 복잡한 함수로 본다
- ⚠ 층이 깊어지는 만큼 기울기 소실 위험도 커진다 → 5 · 6교시 BN · ResNet

<<CODE 4-1 | 3×3 두 층 vs 5×5 한 층 — 파라미터 비교표>>

#### 개념 2 · VGG 블록과 VGG-mini
> VGG 블록 = [Conv 3×3 → BN → ReLU] × n → MaxPool 2×2

![VGG-mini — 블록 3개로 크기 1/2 마다 채널 2배](assets/fig/vb/d2-vgg.svg)

| 단계 | 출력 | 파라미터 |
|---|---|---|
| 블록1 (3→32, 32→32) | (32, 16, 16) | 864 + 9,216 = 10,080 |
| 블록2 (32→64, 64→64) | (64, 8, 8) | 18,432 + 36,864 = 55,296 |
| 블록3 (64→128, 128→128) | (128, 4, 4) | 73,728 + 147,456 = 221,184 |
| Linear(2048, 10) | 10 | 20,490 |
| BN 6개 (γ, β) | | 896 |
| 합계 | | **307,946** |

BN 은 원래 VGG 에 없었다(2015년 등장) — 오늘 것은 학습을 쉽게 하려고 넣은 현대판이다. BN 바로 앞 Conv 는 `bias=False` (β 가 bias 역할).

##### 🔎 왜 이렇게 되나
- **비유**: 같은 모양 블록을 레고처럼 쌓아 올리는 설계 — 블록마다 크기 1/2 · 채널 2배
- **① 블록 구성 [Conv → BN → ReLU] × n → Pool 인 이유**: 특징을 여러 번 다듬고(Conv · ReLU), 분포를 맞추고(BN), 크기를 줄인다(Pool)
- **② BN 앞 Conv 는 bias=False**: BN 의 β 가 편향 역할을 대신한다 (NN Day 4)
- **③ 결과**: 307,946개로 3 epoch 0.7526 — LeNet 0.5489 보다 크게 오른다
- ⚠ 원래 VGG(2014)엔 BN 이 없었다 — 오늘 것은 현대판, 연산량은 LeNet 의 약 59배

<<CODE 4-2 | vgg_block 과 VGG-mini 만들기>>
<<CODE 4-3 | VGG-mini 를 같은 조건으로 학습>>
<<CODE 4-c1 | MLP · LeNet-5 · VGG-mini test 정확도 곡선>>

#### 한 번에 정리
- 3×3 두 층 = 5×5 수용 영역, 파라미터 28% 적고 ReLU 2번.
- VGG 블록 = [Conv3×3 → BN → ReLU] × n → MaxPool, 크기 1/2 마다 채널 2배.
- VGG-mini 307,946 파라미터, 3 epoch 0.7526.
- ⚠ 깊게 쌓으면 좋아진다 — 그럼 끝없이 쌓으면? → 5교시.

**연관 문제**
<<QA 4>>

### 5교시 · 깊이의 문제 · BatchNorm

**키워드**: `기울기 소실`, `plain-8 · plain-20`, `첫 층 기울기 1.1e-09`, `BatchNorm2d`, `채널마다 하나`, `γ · β`, `running_mean · running_var`, `model.eval()`, `성능 저하(degradation)`

**개요**: VGG 가 깊이로 성공했으니 층을 더 쌓아 본다. ① 그냥 깊게 쌓으면 입력 쪽 기울기가 사라져(**기울기 소실**) BN 없는 plain 망은 8층조차 3 epoch 동안 못 배운다. ② **BatchNorm** 이 층마다 출력 분포를 다시 맞춰 기울기가 고르게 흐르고 큰 학습률도 견딘다. ③ 그런데 BN 이 있어도 20층이 8층보다 **train 손실이 더 높다** — 과적합이 아니라 최적화 자체가 어려운 **성능 저하(degradation)**, 6교시 ResNet 의 출발점.

**한눈에 보기**
깊이의 문제 — 기울기 소실과 성능 저하
> 그냥 깊게 쌓으면 입력 쪽 기울기가 사라지고(기울기 소실), BatchNorm 으로 고쳐도 깊은 망이 train 에서조차 얕은 망보다 못하다(성능 저하)
- **첫 층 기울기 — plain-8 3.2e-04 · plain-20 1.1e-09**: plain-20 + BN 은 3.4e-01 로 살아난다
- **성능 저하는 과적합이 아니다**: BN 을 넣어도 20층 train 손실 1.785 > 8층 1.699 — 최적화 자체가 어렵다 → ResNet
- **BatchNorm2d — 채널마다 평균 · 분산 하나**: `BatchNorm2d(16)` 학습 파라미터 γ · β 32개, running 통계는 buffer 로 state_dict 에
- **GAP — 채널마다 평균 하나**: `nn.AdaptiveAvgPool2d(1)` → (N, C, 1, 1)
- **⚠ 추론 때 `model.eval()` 을 빼먹으면 결과가 배치마다 달라진다**: 배치 크기 1 로 BN 학습도 불안정

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

#### 깊이의 문제 — 기울기 소실과 성능 저하
> 그냥 깊게 쌓으면 입력 쪽 기울기가 사라지고(**기울기 소실**), BN 으로 고쳐도 깊은 망이 **train** 에서조차 못하다(**성능 저하**)
- **plain-N** → 3×3 Conv (N − 1)개(채널 16) + GAP + Linear, 지름길 없음 — plain-8 14,538 · plain-20 42,378 파라미터
- **예** → 첫 층 기울기 plain-8 3.2e-04 · plain-20 **1.1e-09** · plain-20 + BN 3.4e-01 (강사 실행 예)
- **BN 없음** → 8층조차 손실 2.30(= ln 10, 찍기)에서 안 움직임 · **BN 있음** → 20층 train 손실 1.785 > 8층 1.699
- **성능 저하(degradation)** → 과적합이 아니라 최적화 자체가 어렵다 → 6교시 ResNet

#### BatchNorm2d
> NN Day 4 의 BN 을 Conv 뒤에 — 평균 · 분산을 배치 · 세로 · 가로에 걸쳐 **채널마다 하나**, y = γx̂ + β
- **`nn.BatchNorm2d()`** → `BatchNorm2d(16)` 학습 파라미터 16 + 16 = **32** (weight = γ, bias = β)
- **buffer** → running_mean · running_var · num_batches_tracked — 학습 대상은 아니지만 state_dict 에 함께 저장
- **GAP** → `nn.AdaptiveAvgPool2d()` (1) — 채널마다 평균 하나 (N, C, 1, 1) · 실험 데이터는 `torch.utils.data.Subset()` 으로 10,000장
- ⚠ 추론 때 `model.eval()` 을 빼먹으면 test 배치의 평균을 써서 결과가 배치마다 달라진다
- ⚠ 배치 크기 1 로 BN 학습 → 분산을 구할 수 없어 불안정

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

**상세 학습**

#### 이 교시에서 답하는 질문
VGG 는 깊이로 성공했다. 그렇다면 층을 계속 쌓으면 계속 좋아질까? 이 교시는 지름길 없는 단순한 깊은 망(plain)으로 실험해 두 가지 문제 — 기울기 소실과 성능 저하 — 를 확인하고, BatchNorm 이 앞의 것을 어떻게 고치는지 본다.

#### 개념 1 · 기울기 소실
역전파는 층마다 국소 미분을 곱해 내려간다 (NN Day 3). 곱하는 값이 1 보다 작으면 층이 깊을수록 입력 쪽 기울기가 0 에 가까워진다. **plain-N** (3×3 Conv (N − 1)개 · 채널 16 · GAP · Linear) 의 첫 층 기울기를 재 보면 (강사 실행 예):

![첫 층 기울기 크기 — plain-20 은 1e-9 까지 사라지고, BN 을 넣으면 0.34](assets/fig/vb/d2-vanish.svg)

BN 이 없으면 8층조차 3 epoch 동안 손실이 2.30(= ln 10, 찍기)에서 움직이지 않는다.

##### 🔎 왜 이렇게 되나
- **비유**: 귓속말 전달 게임 — 사람(층)이 많을수록 맨 앞 사람에게 닿는 말이 희미해진다
- **① 왜 사라지나**: 역전파는 층마다 국소 미분을 곱한다 — 1 보다 작은 값을 계속 곱하면 0 에 가까워진다
- **② 숫자로**: 첫 층 기울기 plain-8 3.2e-04 · plain-20 1.1e-09 (강사 실행 예)
- **③ 결과**: BN 없으면 8층조차 손실 2.30(= ln 10, 찍기 수준)에서 안 움직인다
- ⚠ 층을 쌓는다고 저절로 좋아지지 않는다

<<CODE 5-c1 | plain_net — 3×3 Conv 를 depth 개 쌓은 망>>
<<CODE 5-1 | 층마다 기울기 크기 — plain-8 · plain-20 · +BN>>

#### 개념 2 · BatchNorm2d — 층마다 분포를 다시 맞춘다
NN Day 4 의 BN 을 Conv 뒤에 쓰는 판이다.

| 항목 | NN Day 4 (Affine 뒤) | `nn.BatchNorm2d(C)` (Conv 뒤) |
|---|---|---|
| 평균 · 분산을 내는 축 | 배치 | 배치 · 세로 · 가로 → **채널마다 하나** |
| 학습 파라미터 | γ, β (특징 수) | γ, β 각 C 개 |
| 추론 | running 평균 · 분산 | 같음 — `model.eval()` 이 전환 |

running 평균 · 분산은 학습 대상이 아니지만 모델과 함께 저장되는 값(**buffer**)이다.

##### 🔎 왜 이렇게 되나
- **비유**: 층마다 중간 점검 — 분포가 틀어지면 매번 평균 0 · 분산 1 로 다시 맞춘다
- **① 채널마다 하나인 이유**: 한 채널(필터 하나)의 출력은 모든 위치에서 같은 특징 — 배치 · 세로 · 가로를 묶어 평균 · 분산 하나
- **② buffer 가 있는 이유**: running 평균 · 분산은 학습 대상(γ · β)은 아니지만 추론에 필요해 state_dict 에 함께 저장
- **③ 기울기가 살아나는 이유**: 분포가 맞으니 국소 미분이 지나치게 작아지지 않는다 — plain-20 + BN 첫 층 3.4e-01
- ⚠ 추론 때 `model.eval()` 을 빼면 test 배치의 평균을 써서 결과가 배치마다 달라진다

<<CODE 5-3 | BatchNorm2d 의 학습 파라미터 · state_dict 항목>>

#### 개념 3 · 성능 저하 — BN 으로도 남는 문제
| 3 epoch (train 손실 / test) | BN 없음 | BN 있음 |
|---|---|---|
| plain-8 | 2.304 / 0.100 | 1.699 / 0.304 |
| plain-20 | 2.303 / 0.100 | 1.785 / 0.270 |

BN 을 넣으면 둘 다 배운다. 그런데 20층의 **train** 손실(1.785)이 8층(1.699)보다 높다. 과적합(train 은 좋고 test 만 나쁨)이 아니라 학습 데이터조차 더 못 맞추는 것 — 깊은 망의 최적화 자체가 어려운 **성능 저하(degradation)** 다. 6교시 ResNet 이 이것을 푼다.

##### 🔎 왜 이렇게 되나
- **비유**: 계단을 더 많이 쌓았더니 오히려 꼭대기에 덜 올라가는 것 — 과식(과적합)이 아니라 체력(최적화) 문제
- **① 과적합이 아닌 이유**: 과적합이면 train 은 좋고 test 만 나쁘다 — 여기선 train 손실도 20층(1.785)이 8층(1.699)보다 나쁘다
- **② 왜 생기나**: 깊은 망은 "아무것도 안 하는 층(y = x)" 조차 배우기 어렵다
- **③ 그래서**: 층이 y = x 를 쉽게 할 수 있게 지름길을 붙인다 → 6교시 ResNet
- ⚠ BN 만으로는 이 문제가 남는다

<<CODE 5-2 | plain-8 · plain-20 을 BN 없이 · 있이 학습해 비교>>

#### 한 번에 정리
- 그냥 깊게 쌓으면 기울기 소실 — plain-20 첫 층 1e-9.
- BatchNorm2d: 채널마다 정규화 → 기울기가 고르게, 큰 학습률도 견딤.
- BN 으로도 20층이 8층보다 train 손실이 높다 = 성능 저하 (과적합 아님).
- ⚠ 추론 땐 `model.eval()` — running 통계를 쓰도록.

**연관 문제**
<<QA 5>>

### 6교시 · ResNet: Residual block

**키워드**: `skip connection`, `y = ReLU(F(x) + x)`, `잔차`, `∂y/∂x = F′(x) + 1`, `BasicBlock`, `1×1 Conv 지름길`, `ResNet-20 = 6n + 2`, `272,474`, `GAP + FC 650`

**개요**: 2015년 Kaiming He 등이 성능 저하를 풀었다. ① 깊은 plain 망이 못한 이유는 남는 층이 "아무것도 안 하기(항등 함수)"를 배우기 어려워서 — 입력을 출력에 그대로 더하는 **지름길**을 두면 층은 "얼마나 바꿀지(**잔차**)"만 배우고, 바꿀 게 없으면 0 을 배우면 된다. ② 덧셈 노드 덕분에 기울기에 항상 **+1** 이 남아 입력까지 흐른다. ③ CIFAR 용 **ResNet-20**(6·3 + 2 층)은 거대 FC 대신 GAP + Linear 하나로 파라미터 272,474 개.

**한눈에 보기**
ResNet — 지름길로 잔차만 배운다
> y = ReLU(F(x) + x) — 입력을 지름길(skip connection)로 더하면 층은 목표와 입력의 차이(잔차)만 배우고, 기울기가 입력까지 흐른다
- **기울기에 1 이 남는다**: ∂y/∂x = F′(x) + 1 — 덧셈 노드가 기울기를 그대로 두 갈래로 (2015, 152층까지)
- **남는 층은 F = 0 만 배우면 된다**: plain 은 y = x 를 정확히 배우기 어렵다
- **BasicBlock — 지름길 두 종류**: 모양이 같으면 `nn.Identity()`, 크기 1/2 · 채널 2배면 1×1 Conv stride 2 + BN (projection) — 16→32 블록 14,528개
- **ResNet-20 = stem + BasicBlock 3 × 3단계 + GAP + Linear**: 가중치 층 20, 파라미터 272,474 (FC 는 650개뿐)
- **3 epoch — plain-20 test 0.380 → ResNet-20 0.443**: train 손실 1.641 → 1.438, 성능 저하가 사라짐
- **⚠ 모양이 바뀌는 블록에 Identity 를 쓰면 더할 수 없어 오류**: 공정 비교는 `residual=False` 로 파라미터 수를 같게

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

#### 잔차 블록(residual block)
> y = ReLU(F(x) + x) — **지름길(skip connection)** 로 입력을 더하면 층은 **잔차(residual)** F = 목표 − x 만 배운다
- **왜** → plain 은 남는 층이 정확히 y = x 를 배우기 어렵다 — 잔차면 F = 0 만 배우면 된다
- **기울기** → 덧셈 노드는 기울기를 그대로 두 갈래로 → ∂y/∂x = F′(x) + **1** — 1 이 남아 입력까지 흐른다
- **2015** → Kaiming He 등, 152층까지

#### BasicBlock — 지름길 두 종류
> **BasicBlock** = 3×3 Conv-BN-ReLU-Conv-BN + 지름길 — 모양이 같으면 `nn.Identity()`, 크기 1/2 · 채널 2배면 **projection 지름길** (1×1 Conv stride 2 + BN)
- **예** → 16→32 stride 2 블록 = 4,608 + 64 + 9,216 + 64 + 512 + 64 = **14,528**
- **`residual=False`** → 지름길을 만들되 안 더함 → 파라미터 수가 같은 plain (공정 비교용)
- ⚠ 모양이 바뀌는 블록에 Identity → 더할 수 없어 오류

#### ResNet-20 (CIFAR-10)
> stem → BasicBlock 3개씩 3단계(16 → 32 → 64, 크기 1/2) → GAP + Linear — 깊이 6n + 2 (n = 3)
- **가중치 층** → stem 1 + 블록 9 × 2 + Linear 1 = **20**
- **파라미터 272,474** → 거대 FC 대신 `nn.AdaptiveAvgPool2d(1)` + Linear(64 → 10) = 650 개뿐
- **예** → 3 epoch: plain-20 train 손실 1.641 · test 0.380 → ResNet-20 **1.438** · **0.443** (강사 실행 예) — 성능 저하가 사라짐

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

**상세 학습**

#### 이 교시에서 답하는 질문
5교시의 성능 저하: 20층이 8층보다 학습 데이터조차 못 맞혔다. 이상하다 — 20층은 남는 12층을 "입력을 그대로 내보내기"(항등 함수)로 두기만 해도 8층만큼은 해야 한다. 그런데 그 "아무것도 안 하기" 를 배우는 게 어렵다. 2015년 ResNet 은 이것을 **지름길** 하나로 풀었다.

#### 개념 1 · 잔차 학습 — 바꿀 양만 배운다
![Residual block — 입력 x 를 지름길로 출력에 더한다](assets/fig/vb/d2-residual.svg)

> y = ReLU(F(x) + x)

| 기호 | 뜻 |
|---|---|
| x | 블록 입력 |
| F(x) | Conv-BN-ReLU-Conv-BN 이 배우는 **잔차** = 목표 − x |
| + x | 지름길(skip connection) |

바꿀 게 없으면 F = 0 을 배우면 된다. 가중치를 0 근처로 두는 것은 쉽다. 그래서 층을 더 쌓아도 최소한 얕은 망만큼은 할 수 있다.

##### 🔎 왜 이렇게 되나
- **비유**: 원본 사진에 "수정할 부분만" 덧칠하기 — 바꿀 게 없으면 덧칠을 안 하면 된다
- **① 잔차를 배우는 이유**: 목표를 통째로 배우는 것(y = H(x))보다 "x 에서 얼마나 바꿀지"(F = H − x)를 배우는 게 쉽다
- **② 바꿀 게 없을 때**: F = 0 — 가중치를 0 근처로 두면 되니 쉽다 (5교시의 y = x 문제 해결)
- **③ 그래서**: 층을 더 쌓아도 최소한 얕은 망만큼은 한다
- ⚠ 더하려면 F(x) 와 x 의 모양이 같아야 한다 (개념 3 의 projection)

<<CODE 6-1 | BasicBlock — 두 Conv + 지름길 더하기 (1×1 shortcut)>>

#### 개념 2 · 기울기에 항상 +1
덧셈 노드는 기울기를 그대로 두 갈래로 보낸다 (NN Day 3). 그래서

> ∂y/∂x = F′(x) + 1

F′ 이 아무리 작아져도 1 이 남는다. 블록이 쌓이면 지름길끼리 이어져 출력에서 입력까지 기울기가 줄지 않고 흐르는 "고속도로" 가 생긴다.

##### 🔎 왜 이렇게 되나
- **비유**: 계단 옆에 엘리베이터(지름길) — 계단이 막혀도 기울기는 엘리베이터로 내려간다
- **① +1 이 생기는 이유**: y = F(x) + x 를 x 로 미분하면 F′(x) + 1 — 덧셈 노드는 기울기를 그대로 두 갈래로 (NN Day 3)
- **② 쌓이면**: 블록마다 1 이 남아 출력에서 입력까지 줄지 않는 "고속도로"
- **③ 결과**: 3 epoch plain-20 test 0.380 → ResNet-20 0.443 — 지름길만 바꾼 비교
- ⚠ 공정한 비교는 지름길만 빼고 나머지(파라미터 수)를 같게 (`residual=False`)

<<CODE 6-3 | plain-20 vs ResNet-20 — 지름길만 빼고 비교>>

#### 개념 3 · BasicBlock 과 ResNet-20
| 상황 | 지름길 |
|---|---|
| 크기 · 채널 그대로 (16 → 16, stride 1) | `nn.Identity()` — x 를 그대로 |
| 크기 1/2 · 채널 2배 (16 → 32, stride 2) | 1×1 Conv(stride 2) + BN 으로 x 의 모양을 맞춤 (projection) |

![ResNet-20 — stem, BasicBlock 3개씩 3단계, GAP + Linear](assets/fig/vb/d2-resnet20.svg)

깊이 6n + 2 (n = 3) = 20층, 파라미터 272,474. 거대 FC 대신 GAP + Linear(64 → 10) 650 개뿐이다.

| 3 epoch (같은 조건) | train 손실 | test |
|---|---|---|
| plain-20 (BN) | 1.641 | 0.380 |
| ResNet-20 | **1.438** | **0.443** |

(강사 실행 예) 지름길만 더했는데 train 손실이 낮아졌다 — 성능 저하가 사라졌다.

##### 🔎 왜 이렇게 되나
- **① 지름길 두 종류인 이유**: 모양이 같으면 x 를 그대로(Identity), 크기 1/2 · 채널 2배면 x 도 1×1 Conv(stride 2)로 모양을 맞춰야 더할 수 있다
- **② 6n + 2 = 20 인 이유**: stem 1 + 블록 9개 × Conv 2 + Linear 1
- **③ 파라미터가 적은 이유**: 거대 FC 대신 GAP + Linear(64 → 10) = 650개 — 전체 272,474
- **④ make_layer 의 첫 블록만 stride 2**: 단계가 바뀔 때 한 번만 크기를 줄인다
- ⚠ 모양이 바뀌는 블록에 Identity 를 쓰면 더할 수 없어 오류

<<CODE 6-2 | ResNet 클래스 · resnet20 — 3단 × 블록 3개>>

#### 한 번에 정리
- y = ReLU(F(x) + x): 층은 잔차(바꿀 양)만 배운다 — 바꿀 게 없으면 F = 0.
- ∂y/∂x = F′ + 1 — 기울기가 입력까지 흐른다.
- 모양이 같으면 Identity, 다르면 1×1 Conv(stride 2) + BN 지름길.
- ResNet-20 = 6·3 + 2 층, 272,474 파라미터, 성능 저하 해결.
- ⚠ 모양이 바뀌는 블록에 Identity 를 쓰면 더할 수 없다.

**연관 문제**
<<QA 6>>

### 7교시 · 학습 기법: 데이터 증강 · 학습률 스케줄 · 혼동 행렬 · 연산량

**키워드**: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `train 에만`, `OneCycleLR 0.004 → 0.1 → 4e-07`, `배치마다 step()`, `혼동 행렬`, `cat → dog`, `MAC`, `forward hook`, `40,813,184`

**개요**: 같은 모델도 **학습 방법**에 따라 몇 %p 가 갈린다. 8교시 85% 를 위한 네 도구. ① **데이터 증강** — train 에만, 정답이 바뀌지 않는 변형(±4 px 이동 · 좌우 반전)으로 외우기를 막는다. ② **OneCycleLR** — 학습률을 작게 시작 → 최댓값 → 거의 0, 배치마다 step. ③ **혼동 행렬**로 "무엇을 무엇으로" 틀리는지 보고, ④ **MAC** 으로 연산량을 잰다 — 파라미터만으로는 속도를 알 수 없다.

**한눈에 보기**
학습 기법 — 증강 · 학습률 스케줄 · 혼동 행렬 · 연산량
> 같은 모델도 학습 방법으로 성능이 달라진다 — 데이터 증강 · OneCycleLR 로 더 잘 배우고, 혼동 행렬 · MAC 으로 무엇을 틀리고 얼마나 비싼지 본다
- **데이터 증강 — train 에만, 정답이 안 바뀌는 변형**: `RandomCrop(32, padding=4)` ±4 px 이동 · `RandomHorizontalFlip()` 좌우 반전 · Normalize 는 맨 뒤
- **OneCycleLR — 작게 → 크게 → 거의 0**: max_lr 0.1 이면 0.004 → 0.1(30% 지점) → 4e-07, `scheduler.step()` 은 배치마다
- **혼동 행렬 M[i, j] = 정답 i 를 j 로 예측한 수**: `np.bincount(true * K + pred)` — 대각선 밖 큰 칸 = 헷갈리는 쌍 (cat → dog 180장)
- **MAC 은 forward hook 으로 센다**: ResNet-20 파라미터 272,474 (MLP 의 1/6) · MAC 40,813,184 (MLP 의 약 24배)
- **⚠ 정답이 바뀌는 변형 금지 (6 을 뒤집으면 9)**: `steps_per_epoch` 가 로더 길이와 다르면 오류 · 다 쓴 hook 은 `h.remove()`

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

#### 데이터 증강(augmentation)
> train 에만, 정답이 바뀌지 않는 변형으로 매번 다른 그림 → 외우기를 막고 위치 · 좌우 변화에 강해진다
- **`transforms.RandomCrop()`** → `RandomCrop(32, padding=4)` — 사방 4 px 0 패딩(40 × 40) 후 무작위 32 × 32 → ±4 px 이동
- **`transforms.RandomHorizontalFlip()`** → 50% 확률 좌우 반전 · Normalize 는 **맨 뒤**
- **test** → 언제나 원본 그대로
- ⚠ 정답이 바뀌는 변형 금지 (6 을 뒤집으면 9)

#### OneCycleLR — 작게 → 크게 → 거의 0
> 학습률을 max_lr / 25 에서 시작 → 30% 지점 최댓값 → 끝은 시작의 10⁻⁴ 배
- **예** → max_lr 0.1: **0.004** → **0.1** → **4e-07**
- **왜** → 처음 작게(warm-up) 불안정한 큰 기울기를 견디고, 끝에 작게 바닥에 정착
- **`OneCycleLR()`** → `scheduler.step()` 은 **배치마다** (epoch 마다 아님)
- ⚠ `steps_per_epoch` 가 `len(train_loader)` 와 다르면 끝에 `Tried to step ... times` 오류

#### 혼동 행렬(confusion matrix)
> M[i, j] = 정답이 i 인데 j 로 예측한 개수 — 행 = 정답, 열 = 예측
- **`np.bincount()`** → `np.bincount(true * K + pred, minlength=K*K).reshape(K, K)` — 칸 번호 k 의 정답 = k // 10, 예측 = k % 10
- **대각선 합 / 전체** → 정확도 · 대각선을 지우고 큰 칸 = 헷갈리는 쌍 (실행 예 **cat → dog 180장**)

#### MAC — 파라미터와 연산량은 따로
> Conv MAC = Cout · OH · OW · (Cin / G) · K² · Linear MAC = in · out
- **`Module.register_forward_hook()`** → **forward hook** 이 계층 실행 때마다 출력 shape 로 MAC 을 센다
- **예** → ResNet-20 파라미터 272,474 (MLP 의 1/6) · MAC **40,813,184** (MLP 의 약 24배) — 같은 가중치를 모든 위치에서 다시 쓰므로
- ⚠ 다 쓴 hook 은 `h.remove()` — 안 하면 다음 실행에 중복 등록

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

**상세 학습**

#### 이 교시에서 답하는 질문
같은 모델도 **어떻게 학습시키느냐**에 따라 몇 %p 가 갈린다. 8교시에서 ResNet-20 으로 85% 를 넘기려면 무엇이 필요할까? 이 교시는 네 가지 도구 — 데이터 증강, 학습률 스케줄, 혼동 행렬(무엇을 틀리나), MAC(얼마나 비싼가) — 를 만든다.

#### 개념 1 · 데이터 증강 — train 에만
학습할 때마다 이미지를 조금씩 바꿔 보여 주면 모델이 그림을 외우지 못하고, 위치 · 좌우 변화에 강해진다. 단, **정답이 바뀌지 않는 변형**만 쓴다.

![데이터 증강 — 패딩 후 무작위 자르기(±4 px 이동)와 좌우 반전](assets/fig/vb/d2-augment.svg)

| 변환 | 하는 일 |
|---|---|
| `RandomCrop(32, padding=4)` | 사방 4 px 0 패딩(40 × 40) 후 무작위 32 × 32 → ±4 px 이동 |
| `RandomHorizontalFlip()` | 50% 확률 좌우 반전 |
| `ToTensor()` → `Normalize(MEAN, STD)` | 맨 뒤 |

test 는 언제나 원본 그대로다. 숫자 6 을 위아래로 뒤집으면 9 가 되듯, 정답이 바뀌는 변형은 금지.

##### 🔎 왜 이렇게 되나
- **비유**: 같은 문제를 숫자만 바꿔 여러 번 풀게 하는 것 — 답을 외우지 못하고 푸는 법을 익힌다
- **① 왜 train 에만**: test 는 실제 성능을 재는 기준이라 원본 그대로
- **② RandomCrop(32, padding=4)**: 사방 4 px 덧대고 무작위로 잘라 ±4 px 이동 — 위치가 조금 달라도 같은 물체
- **③ 좌우 반전**: 비행기 · 고양이는 뒤집어도 같은 클래스
- ⚠ 정답이 바뀌는 변형은 금지 (숫자 6 을 뒤집으면 9) · Normalize 는 맨 뒤

<<CODE 7-1 | 증강 transform — RandomCrop · HorizontalFlip>>

#### 개념 2 · OneCycleLR — 작게 → 크게 → 거의 0
![OneCycleLR — max_lr 0.1 이면 0.004 에서 시작해 30% 지점 0.1, 끝은 4e-07](assets/fig/vb/d2-onecycle.svg)

처음엔 작은 학습률로 시작해(warm-up) 초기의 불안정한 큰 기울기를 견디고, 최댓값으로 빠르게 내려간 뒤, 끝에는 거의 0 으로 바닥에 정착한다. `scheduler.step()` 은 **배치마다** 부른다.

##### 🔎 왜 이렇게 되나
- **비유**: 운전 — 출발은 천천히(warm-up), 고속도로에선 빠르게, 주차할 땐 아주 천천히
- **① 처음에 작게**: 학습 초기의 기울기는 크고 불안정 — 큰 학습률이면 튀어 나간다
- **② 중간에 크게**: 손실 지형을 빠르게 내려가고, 날카로운 골짜기를 넘어 넓은 바닥을 찾는다
- **③ 끝에 거의 0**: 바닥 근처에서 작게 움직여 정착한다 — max_lr 0.1 이면 0.004 → 0.1 → 4e-07
- ⚠ `scheduler.step()` 은 배치마다 — steps_per_epoch 가 로더 길이와 다르면 끝에서 오류

<<CODE 7-2 | OneCycleLR 학습률 곡선 그려 보기>>

#### 개념 3 · 혼동 행렬 — 무엇을 무엇으로 틀리나
M[i, j] = 정답이 i 인데 j 로 예측한 개수. 대각선이 맞힌 것, 대각선 밖이 틀린 것이다. 대각선을 지우고 가장 큰 칸을 찾으면 가장 헷갈리는 쌍이 나온다 (VGG-mini 실행 예: cat → dog 180장).

##### 🔎 왜 이렇게 되나
- **비유**: 오답 노트 — "몇 개 틀렸나" 가 아니라 "무엇을 무엇으로 착각했나"
- **① 행 = 정답, 열 = 예측**: 대각선이 맞힌 것, 대각선 밖이 틀린 것
- **② bincount 한 줄로 되는 이유**: 정답 × K + 예측 은 K×K 칸 번호 하나 — 그 번호의 개수를 세면 행렬
- **③ 쓰임**: 대각선을 지우고 가장 큰 칸 = 가장 헷갈리는 쌍 (cat → dog 180장) → 그 클래스 데이터를 늘리는 등 개선 방향
- ⚠ 정확도 하나로는 어떤 클래스가 약한지 안 보인다

<<CODE 7-3 | predict_all · confusion_matrix — 혼동 행렬 만들기>>

#### 개념 4 · MAC — 파라미터와 연산량은 따로
> Conv MAC = Cout · OH · OW · (Cin / G) · K², Linear MAC = in · out

![네 모델의 파라미터와 MAC — ResNet-20 은 파라미터가 MLP 의 1/6 인데 MAC 은 24배](assets/fig/vb/d2-compare.svg)

합성곱은 같은 가중치를 모든 위치에서 다시 쓰므로 파라미터는 적어도 연산량이 많다. 온디바이스에서는 두 숫자를 함께 봐야 한다.

##### 🔎 왜 이렇게 되나
- **비유**: 파라미터 = 공구 개수, MAC = 그 공구를 휘두르는 횟수
- **① Conv MAC 이 큰 이유**: 같은 필터를 출력의 모든 위치(OH × OW)에서 다시 쓴다
- **② 식**: Conv = Cout · OH · OW · (Cin / G) · K², Linear = in · out (파라미터 하나당 한 번)
- **③ hook 으로 세는 이유**: 출력 크기(OH · OW)는 실제로 한 번 돌려 봐야 정확히 안다
- ⚠ 온디바이스는 둘 다 본다 — ResNet-20 은 파라미터는 MLP 의 1/6 인데 MAC 은 약 24배

<<CODE 7-4 | count_macs — hook 으로 MAC 세고 네 모델 비교>>

#### 한 번에 정리
- 증강: train 에만, 정답이 바뀌지 않는 변형(±4 px 이동 · 좌우 반전), Normalize 는 맨 뒤.
- OneCycleLR: 0.004 → 0.1 → 4e-07, 배치마다 step.
- 혼동 행렬: 행 = 정답, 열 = 예측 → 헷갈리는 쌍.
- MAC 은 파라미터와 따로 — ResNet-20 은 파라미터 1/6, MAC 24배 (MLP 대비).
- ⚠ hook 은 다 쓰면 `remove()`.

**연관 문제**
<<QA 7>>

### 8교시 · 미니 프로젝트: CIFAR-10 ResNet-20 85% 도전

**키워드**: `ResNet-20 + 증강`, `SGD momentum 0.9 · nesterov`, `weight_decay 5e-4`, `OneCycleLR 20 epoch`, `391 step`, `0.8587`, `가장 헷갈린 쌍`, `모델 비교표`

**개요**: 오늘 만든 것을 모두 합친다. ① ResNet-20 + 데이터 증강 + SGD(momentum · nesterov · weight decay) + OneCycleLR 로 **20 epoch** 학습해 test **85% 이상**. ② 혼동 행렬로 "가장 헷갈린 쌍" 3개와 오답 그림을 보며 **왜** 틀리는지 본다. ③ 네 모델(MLP · LeNet-5 · VGG-mini · ResNet-20)을 파라미터 · MAC · 정확도로 비교하되, 학습 조건이 다르다는 점을 해석에 쓴다.

**한눈에 보기**
CIFAR-10 85% 도전 — ResNet-20 + 오늘의 기법
> ResNet-20 + 증강 + SGD(momentum 0.9 · nesterov · weight decay 5e-4) + OneCycleLR(max_lr 0.1) 20 epoch 로 test 0.85 를 넘긴다
- **20 epoch test 0.8587**: 15 epoch 는 0.8479 로 미달 — steps_per_epoch = 50,000 / 128 → 391
- **오답 분석 — 대각선을 지우고 큰 칸 3개**: `np.argsort(off.ravel())[::-1][:3]` → (정답, 예측) 클래스 쌍
- **모델 비교**: MLP 0.4994 · LeNet-5 0.5489 · VGG-mini 0.7526 (3 epoch) · ResNet-20 0.8587 (20 epoch + 증강)
- **다음 — 연산을 줄이는 방향**: 160 epoch 면 91.25% (원 논문), 온디바이스는 Day 3 MobileNet
- **⚠ 학습 조건이 다르면 구조의 기여를 가를 수 없다**: 같은 조건에서 다시 학습해야 공정 · `fit` 의 다섯 번째 인자가 scheduler

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

#### 85% 도전 — 설정
> ResNet-20 + 증강 + SGD(momentum 0.9 · nesterov · weight decay 5e-4) + OneCycleLR(max_lr 0.1) 20 epoch → test **0.85** 이상
- **weight decay** → L2 규제 (NN Day 4) `weight_decay=5e-4` · **nesterov** → momentum 의 앞서 보기 변형
- **steps_per_epoch** → `len(aug_loader)` = 50,000 / 128 → **391**
- **예** → 20 epoch test **0.8587** (15 epoch 는 0.8479 로 미달, 강사 실행 예)
- ⚠ `fit(model, aug_loader, test_loader, EPOCHS, opt, sched, …)` — 다섯 번째 인자가 scheduler

#### 오답 분석 · 모델 비교
> 정확도 하나가 아니라 "무엇을 왜 틀리나" 와 "얼마나 비싼가" 를 함께
- **헷갈린 쌍** → 대각선을 0 으로 → `np.argsort(off.ravel())[::-1][:3]` → (CLASSES[k // 10], CLASSES[k % 10])
- **비교** → MLP 0.4994 · LeNet-5 0.5489 · VGG-mini 0.7526 (3 epoch) · ResNet-20 0.8587 (20 epoch + 증강)
- **다음** → 같은 ResNet-20 을 160 epoch 이면 91.25% (원 논문) · 온디바이스는 반대로 연산을 줄이는 방향 → Day 3 MobileNet
- ⚠ 학습 조건이 다르면 구조의 기여를 가를 수 없다 — 같은 조건에서 다시 학습해야 공정

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

**상세 학습**

#### 이 교시에서 답하는 질문
오늘 만든 것을 모두 합쳐 CIFAR-10 에서 **85%** 를 넘긴다. 그리고 정확도 하나로 끝내지 않고, 무엇을 왜 틀리는지와 네 모델의 비용을 함께 비교한다.

#### 개념 1 · 설정
| 항목 | 값 | 출처 |
|---|---|---|
| 모델 | ResNet-20 (272,474) | 6교시 |
| train 로더 | 증강 aug_set, 배치 128 | 7교시 |
| 옵티마이저 | SGD(lr 0.1, momentum 0.9, nesterov, weight_decay 5e-4) | NN Day 4 |
| 스케줄러 | OneCycleLR(max_lr 0.1, 20 epoch, steps_per_epoch = len(loader) = 391) | 7교시 |

**weight decay** 는 가중치 크기에 벌점을 주는 L2 규제, **nesterov** 는 momentum 이 갈 곳을 앞서 보고 기울기를 재는 변형이다. 50,000 / 128 = 390.6 → 마지막 작은 배치까지 **391** step.

강사 실행 예: 20 epoch test **0.8587** (15 epoch 는 0.8479 로 미달).

##### 🔎 왜 이렇게 되나
- **비유**: 오늘 배운 기법 전부를 한 레시피에 넣기
- **① weight decay**: 가중치가 커지지 않게 벌점 — 과적합 완화 (NN Day 4 L2)
- **② nesterov**: momentum 이 갈 곳을 미리 보고 기울기를 재는 변형 — 지나치게 튀는 것을 줄인다
- **③ OneCycleLR + 증강**: 학습률 일정과 데이터 늘리기로 같은 모델을 더 잘 학습
- ⚠ steps_per_epoch = len(loader) = 50,000 / 128 → 391 · fit 의 다섯 번째 인자가 scheduler

<<CODE 8-1 | 과제 1 — ResNet-20 · SGD(nesterov) · OneCycleLR 20 epoch 준비>>
<<CODE 8-2 | 과제 2 — 학습 (GPU 없으면 강사 가중치) · 저장>>
<<CODE 8-c1 | 손실 · 정확도 곡선>>

#### 개념 2 · 오답 분석과 모델 비교
| 모델 | 파라미터 | MAC | test (학습 조건) |
|---|---|---|---|
| MLP | 1,707,274 | 1,706,496 | 0.4994 (3 epoch) |
| LeNet-5 | 62,006 | 651,720 | 0.5489 (3 epoch) |
| VGG-mini | 307,946 | 38,653,952 | 0.7526 (3 epoch) |
| ResNet-20 | 272,474 | 40,813,184 | 0.8587 (20 epoch + 증강) |

이 표는 학습 조건이 다르다. 구조 덕인지 학습 방법 덕인지 가르려면 같은 조건에서 다시 학습해야 공정하다. 같은 ResNet-20 을 약 160 epoch 학습하면 91.25%(원 논문)까지 간다. 온디바이스는 반대로 정확도를 조금 내주고 연산을 크게 줄이는 방향 — Day 3 MobileNet 과 전이학습이다.

##### 🔎 왜 이렇게 되나
- **① 학습 조건이 다르면 비교가 안 되는 이유**: ResNet-20 만 20 epoch + 증강 — 구조 덕인지 학습 덕인지 가를 수 없다
- **② 표를 읽는 법**: 파라미터(저장) · MAC(속도) · 정확도를 함께 — 온디바이스는 셋의 균형
- **③ 오답 분석**: 헷갈린 쌍 · 클래스별 정확도로 약점이 어디인지 본다
- **④ 다음**: 같은 ResNet-20 을 160 epoch 이면 91.25% (원 논문) — 반대로 연산을 줄이는 방향이 Day 3 MobileNet
- ⚠ 공정한 비교는 같은 조건에서 다시 학습해야 한다

<<CODE 8-3 | 과제 3 — 가장 헷갈린 쌍 · 클래스별 정확도>>
<<CODE 8-c2 | 틀린 사진 16장 보기>>
<<CODE 8-4 | 과제 4 — MLP · LeNet · VGG · ResNet 비교표>>

#### 한 번에 정리
- ResNet-20 + 증강 + SGD(nesterov, wd 5e-4) + OneCycle 20 epoch → 0.8587.
- steps_per_epoch = len(loader) = 391, 스케줄러는 배치마다.
- 오답은 혼동 행렬로 "무엇을 무엇으로" 를 본다.
- 비교표는 파라미터 · MAC · 정확도를 함께, 학습 조건 차이를 해석에 쓴다.
- ⚠ 조건이 다른 비교로 구조의 우열을 단정하지 않는다.

**연관 문제**
<<QA 8>>
