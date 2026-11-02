## Day 3 — 경량 CNN · 전이학습 · Grad-CAM

### 1교시 · Day 2 복습 퀴즈 · 정확도 vs 속도

**키워드**: `온디바이스`, `정확도 + 지연시간 + 파라미터 + MAC`, `latency_ms`, `CPU · 스레드 1 · 배치 1 · warm-up`, `ResNet-50 4.1G MAC`, `MobileNetV3-Small 56.5M MAC`, `72배 차이`

**개요**: 어제 ResNet-20 은 CIFAR-10 에서 85% 를 넘겼다. 그런데 라즈베리파이 · Jetson 같은 작은 장치에서는 정확도만큼 **얼마나 빠른가(지연시간)**, **얼마나 무거운가(파라미터 · 메모리 · MAC)** 가 중요하다. ① 모델 선택 기준은 정확도 · 지연시간 · 파라미터 · MAC 네 가지. ② 지연시간은 장치를 흉내 내려고 **CPU · 스레드 1 · 배치 1장 · warm-up 버리고 평균**으로 잰다. ③ 경량 모델은 연산량을 수십 분의 1로 줄이면서 정확도는 몇 %p 만 내준다.

**핵심 용어**
- **온디바이스(on-device)**: 서버가 아니라 장치(라즈베리파이 · Jetson)에서 직접 추론하는 것
- **지연시간(latency)**: 이미지 한 장을 넣고 결과가 나올 때까지 걸리는 시간 (ms)
- **warm-up**: 처음 몇 번의 실행은 메모리 할당 · 캐시 때문에 느려서 재지 않고 버리는 것
- **MAC**: 곱셈-덧셈 횟수 (이미지 한 장) — 계산 시간 · 전력의 척도

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.set_num_threads()` | `torch.set_num_threads(1)` | CPU 스레드 수 지정 (지연시간 측정용) |
| `time.perf_counter()` | `time.perf_counter()` | 정밀한 시간 측정 (초) |
| `torchvision.models.resnet50()` | `torchvision.models.resnet50(weights=None)` | ResNet-50 구조 (가중치 없이) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 1-1 | Day 2 의 `resnet20_cifar10.pt` 를 내려받아 `load_state_dict` → test 정확도 확인 (> 0.8) |
| 1-2 | `latency_ms(model, input_size)` — CPU 복사 · 스레드 1 · warm-up 3회 · 30회 평균 |
| 1-3 | ResNet-20 · ResNet-18 · ResNet-50 · MobileNetV3-S 의 파라미터 · MAC · 지연시간을 한 표로 |

**핵심 내용**

#### 온디바이스 모델 선택 기준
> 정확도만 보지 않는다 — 정확도 · 지연시간 · 파라미터(메모리) · MAC 을 함께
| 지표 | 장치에서 의미 |
|---|---|
| 정확도 | 얼마나 잘 맞히나 |
| 지연시간 | 실시간(30 FPS 면 한 장 **33 ms** 안) 가능한가 |
| 파라미터 | 모델 파일 크기 · 메모리 |
| MAC | 계산 시간 · 전력 |
- **ImageNet top-1 (torchvision)** → ResNet-18 69.8% · ResNet-50 76.1% · MobileNetV2 71.9% · MobileNetV3-Large 74.0% · MobileNetV3-Small 67.7%
- **ResNet-50 vs MobileNetV3-Small** → MAC 4,089,184,256 vs 56,510,400 → 약 **72배** 차이, 정확도는 8.4%p 차이
- **실행 예 (CPU 1 스레드, 배치 1)** → ResNet-20 2.87 ms · ResNet-18 70 ms · ResNet-50 134 ms · MobileNetV3-S 8.8 ms
- ⚠ ResNet-50 은 장치에서 33 ms 안에 못 끝난다 — 실시간이 안 됨

#### 지연시간 재는 법 (온디바이스 흉내)
> CPU · 스레드 1 · 배치 1 · warm-up 후 평균
| 조건 | 이유 |
|---|---|
| CPU | 작은 장치에는 GPU 가 없거나 약함 |
| 스레드 1 | 작은 장치의 코어 하나를 흉내 |
| 배치 1 | 카메라 프레임은 한 장씩 들어온다 |
| warm-up 3회 버림 | 첫 실행은 메모리 할당 · 캐시로 느림 |
- **`copy.deepcopy(model).cpu().eval()`** → 원래 GPU 에 있는 모델을 건드리지 않고 복사해서 잰다
- **스레드 수는 재고 나서 원래대로** → `torch.set_num_threads(old)`
##### 자세히
```text
 latency_ms(model, size, reps=30, threads=1)
   m = deepcopy(model).cpu().eval()      # 원본 보존
   set_num_threads(threads)
   m(x) × 3      ← warm-up (시간 안 잼)
   t0 → m(x) × reps → dt = (now - t0) / reps
   return dt × 1000                       # ms
```

**기본 사용형태**
```python
def latency_ms(model, input_size, reps=30, threads=1):
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(input_size)
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    with torch.no_grad():
        for _ in range(3):                     # warm-up
            m(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            m(x)
        dt = (time.perf_counter() - t0) / reps
    torch.set_num_threads(old)
    return dt * 1000
```

**연결**
- [VB D2·7교시 학습 기법: 연산량](#/vb/2/7) — `count_macs` 를 어제 만들었다
- [VB D2·8교시 ResNet-20 85% 도전](#/vb/2/8) — 오늘 비교 대상이 되는 모델
- [NN D5·7교시 모델 비용 계산](#/nn/5/7) — 파라미터 · MAC 계산의 출처

**연관 문제**
<<QA 1>>

### 2교시 · 연산량 계산: 파라미터 · MAC · 메모리

**키워드**: `파라미터 × 4 bytes`, `활성화 메모리`, `model_cost`, `register_forward_hook`, `ResNet-50 102 MB`, `1×1 Conv`, `bottleneck`, `1,179,648 → 69,632`, `16.9배`

**개요**: 장치에서 모델이 차지하는 메모리는 **파라미터 × 4 B(float32) + 활성화**, 계산량은 **MAC**. ① 어제의 `count_macs` 를 넓혀 활성화 메모리(각 층 출력 크기)까지 한 번에 재는 `model_cost` 를 만든다. ② ResNet-50 은 ResNet-18 보다 층이 3배 가까이 많은데 MAC 은 2.3배 — **bottleneck 블록**(1×1 로 줄이고 → 3×3 → 1×1 로 늘리기) 덕분. ③ 256 채널에서 basic 블록 대비 가중치가 **1/17**.

**핵심 용어**
- **활성화(activation) 메모리**: 각 층 출력의 크기 합 — 실행 중 쓰는 메모리, 해상도가 클수록 큼
- **1×1 Conv**: 위치마다 채널끼리만 섞는 합성곱 — 채널 수를 줄이거나 늘리는 데 쓴다
- **bottleneck 블록**: 1×1(줄이기) → 3×3 → 1×1(늘리기) 3단 구조 (ResNet-50 이상)
- **forward hook**: 계층이 실행될 때마다 불리는 함수 — 출력 shape 을 엿볼 때 쓴다

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Module.register_forward_hook()` | `h = m.register_forward_hook(hook)` | 계층 출력 엿보기 (`h.remove()` 로 해제) |
| `Tensor.numel()` | `out.numel()` | 원소 개수 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 2-1 | `model_cost` — hook 으로 Conv · Linear 출력 원소 수를 더해 `param_MB` · `act_MB` · MAC 반환 |
| 2-2 | 256 채널 basic vs bottleneck 가중치 수 계산 |

**핵심 내용**

#### 메모리와 계산 — 세 가지 지표
> 파일 크기 = 파라미터 × 4 B · 실행 메모리 = + 활성화 · 계산 = MAC
| 지표 | 의미 | 장치에서 |
|---|---|---|
| 파라미터 | 저장할 가중치 개수 | 모델 파일 크기 · 메모리 (× 4 B) |
| MAC | 곱셈-덧셈 횟수 | 계산 시간 · 전력 |
| 활성화 | 각 층 출력 크기 | 실행 중 메모리 (해상도 ↑ → 커짐) |
- **ResNet-50** → 파라미터 25,557,032 × 4 B ≈ **102.2 MB** (가중치만) + 활성화 수십 MB
- **ResNet-20** → 272,474 × 4 B ≈ 1.09 MB · **MobileNetV3-S** (CIFAR 10 클래스) 1,528,106 × 4 B ≈ 6.11 MB
- ⚠ 파라미터가 적다고 활성화도 적은 건 아니다 — 큰 해상도를 오래 유지하면 활성화가 커진다

#### bottleneck — 1×1 로 줄였다 늘리기
> 채널이 많은(256) 곳에서 바로 3×3 을 하지 않고 64 로 줄여서 한다
| 블록 | 계산 | 가중치 |
|---|---|---|
| basic (3×3 두 번) | 2 · (256·256·9) | **1,179,648** |
| bottleneck | 256·64 + 64·64·9 + 64·256 | **69,632** |
- **비율** → 1,179,648 / 69,632 = **16.9배 적음** (약 1/17)
- **1×1 Conv** → 위치마다 채널만 섞는 '채널 압축기' — 비싼 3×3 을 채널이 적은(64) 곳에서 하게 해 준다
##### 자세히
```text
 basic           256 ─3×3→ 256 ─3×3→ 256          2 × 589,824 = 1,179,648
 bottleneck      256 ─1×1→  64 ─3×3→  64 ─1×1→ 256
                       16,384    36,864    16,384  =    69,632
 ※ 3×3 가중치 = Cout · Cin · 9 → Cin · Cout 이 작을수록 급감
```

**기본 사용형태**
```python
def model_cost(model, input_size):
    acts = [0]
    def hook(m, inp, out):
        acts[0] += out.numel()
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_size, device=next(model.parameters()).device))
    for h in hs: h.remove()
    p = count_params(model)
    return {"params": p, "macs": count_macs(model, input_size),
            "param_MB": p * 4 / 1e6, "act_MB": acts[0] * 4 / 1e6}
```

**연결**
- [VB D2·2교시 Conv 다시 보기](#/vb/2/2) — 가중치 공식 `Cout · Cin · K²`
- [VB D2·6교시 ResNet: Residual block](#/vb/2/6) — basic 블록의 출처
- [NN D5·7교시 모델 비용 계산](#/nn/5/7) — MAC 세기

**연관 문제**
<<QA 2>>

### 3교시 · Depthwise Separable Convolution

**키워드**: `Depthwise 3×3`, `Pointwise 1×1`, `groups = Cin`, `1/Cout + 1/K²`, `73,728 → 8,768`, `75,497,472 → 8,978,432`, `0.1189`, `실제 속도 4.1배 < MAC 8.4배`

**개요**: 일반 3×3 합성곱은 공간(3×3 이웃)과 채널(모든 입력 채널)을 한꺼번에 섞는다. **MobileNet 은 이걸 둘로 쪼갠다.** ① Depthwise 3×3 — 채널마다 따로 공간만 섞기(`groups = Cin`). ② Pointwise 1×1 — 위치마다 채널만 섞기. ③ 연산량이 `1/Cout + 1/K²` ≈ **1/9** 로 줄어든다. ④ 하지만 실제 속도 향상은 MAC 비율보다 작다 — depthwise 는 메모리 병목.

**핵심 용어**
- **Depthwise Conv**: `groups = Cin` — 채널마다 자기 필터 하나로 공간만 섞는 합성곱
- **Pointwise Conv**: 1×1 합성곱 — 위치마다 채널끼리만 섞음
- **DS Conv(Depthwise Separable)**: Depthwise → Pointwise 순서의 2단계 합성곱
- **메모리 병목**: 곱셈은 적어도 데이터 옮기는 시간이 커서 속도가 MAC 만큼 안 줄어드는 현상

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `nn.Conv2d()` | `nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False)` | `groups = cin` 이면 depthwise (오늘은 groups 로 쓰는 법) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 3-1 | `DSConv` 클래스 — dw(3×3, groups=cin) + BN + ReLU → pw(1×1) + BN + ReLU |
| 3-2 | 64→128, 32×32 에서 일반 vs DS 의 가중치 · MAC 비교, 공식 `1/128 + 1/9` 와 대조 |
| 3-3 | 64×56×56 입력 CPU 1 스레드 지연시간 비교 (일반 5.14 ms vs DS 1.26 ms) |

**핵심 내용**

#### 연산량 공식
> DS / 일반 = (Cin·K²·HW + Cin·Cout·HW) / (Cin·Cout·K²·HW) = **1/Cout + 1/K²**
| 방식 | 가중치 (64→128, 3×3) | MAC (32×32) |
|---|---|---|
| 일반 3×3 | 128·64·9 = **73,728** | 73,728 × 1024 = **75,497,472** |
| Depthwise 3×3 | 64·9 = 576 | |
| Pointwise 1×1 | 64·128 = 8,192 | |
| DS 합계 | **8,768** (11.9%) | 8,768 × 1024 = **8,978,432** |
- **MAC 비율** → 8,978,432 / 75,497,472 = **0.1189** = 1/128 + 1/9 (공식과 일치)
- **K = 3** 이면 약 1/9 ~ 1/8 — 큰 Cout 일수록 1/9 에 가까워짐
- **연산의 대부분(93%)은 pointwise 1×1** → 8,192 / 8,768

#### 실제 속도는 덜 빨라진다
> MAC 8.4배 감소 → 실측 4.1배 (64×56×56, CPU 1 스레드)
- **원인** → depthwise 는 곱셈은 적지만 채널마다 따로 메모리를 읽어 계산 대비 메모리 이동이 많다
- ⚠ **MAC 이 적다고 반드시 빠르지 않다** → 실제 장치에서 재야 한다 (Day 5 ONNX · TensorRT)
##### 자세히
```text
 일반 3×3 (64→128)          DS 3×3
 ┌───────────────┐          dw: 채널 64개 각각 3×3 (공간만)
 │ 128 필터 ×    │            ▼
 │ (64 채널 × 9) │          pw: 1×1 × 128 (채널만)
 └───────────────┘
 73,728                      576 + 8,192 = 8,768
```

**기본 사용형태**
```python
class DSConv(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Sequential(nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
                                nn.BatchNorm2d(cin), nn.ReLU(inplace=True))
        self.pw = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False),
                                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.pw(self.dw(x))
```

**연결**
- [VB D2·2교시 Conv 다시 보기](#/vb/2/2) — `groups` 와 depthwise 1/58
- [VB D3·2교시 연산량 계산](#/vb/3/2) — 1×1 Conv 의 역할
- [VB D3·4교시 MobileNet V1 · V2 · V3](#/vb/3/4) — DS Conv 를 쌓아 만든 모델

**연관 문제**
<<QA 3>>

### 4교시 · MobileNet V1 · V2 · V3

**키워드**: `MobileNet V1 (2017)`, `V2 (2018) Inverted residual`, `linear bottleneck`, `V3 (2019) NAS + SE + h-swish`, `×6 늘리기`, `ReLU6`, `InvertedResidual 8,832`, `블록 11개 · SE 9개`, `2,542,856`

**개요**: MobileNet 은 3교시의 DS Conv 를 뼈대로 세 번 진화했다. ① **V1** — 일반 Conv 를 DS Conv 로 바꿔 쌓기. ② **V2** — **inverted residual**: 1×1 로 채널을 ×6 늘리고 → depthwise → 1×1 로 줄이기, 마지막 1×1 뒤엔 ReLU 없음(linear bottleneck), 입출력 크기가 같으면 지름길. ③ **V3** — 자동 구조 탐색(NAS) + **SE**(채널 중요도 재조정) + **h-swish** 활성화.

**핵심 용어**
- **Inverted residual**: ResNet bottleneck(256→64→256)과 반대로 '늘렸다 줄이기'(24→144→24) — depthwise 가 싸서 넓은 곳에서 해도 부담이 적다
- **Linear bottleneck**: 블록 끝 1×1(줄이기) 뒤에 활성화를 안 넣는 것 — 채널이 적은 곳에서 ReLU 가 음수 정보를 잘라 손실이 커서
- **ReLU6**: `min(max(0, x), 6)` — 양자화 · 저정밀 계산에 유리
- **NAS**: 구조를 자동으로 탐색 (V3 의 블록 구성)
- **SE(Squeeze-and-Excitation)**: 채널마다 평균(squeeze) → 작은 FC 두 개 → 0~1 가중치(excitation) → 채널마다 곱하기
- **h-swish**: `x · ReLU6(x + 3) / 6` — swish(`x · σ(x)`)를 sigmoid 없이 근사, 장치에서 계산이 싸다

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `nn.ReLU6()` | `nn.ReLU6(inplace=True)` | 0 ~ 6 으로 자르는 활성화 |
| `F.relu6()` | `F.relu6(x + 3)` | 함수형 ReLU6 (h-swish 에서) |
| `nn.Hardswish()` | `nn.Hardswish()(x)` | h-swish 내장 (V3 의 활성화) |
| `torchvision.models.mobilenet_v3_small()` | `mobilenet_v3_small(weights=None)` | MobileNetV3-Small 구조 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 4-1 | `InvertedResidual` — 1×1 늘리기 → dw 3×3 → 1×1 줄이기(활성화 없음), `stride == 1 and cin == cout` 이면 지름길 |
| 4-2 | `h_swish(x) = x · relu6(x + 3) / 6` 를 직접 만들어 `nn.Hardswish` 와 `allclose` 비교, ReLU · swish 와 함께 그림 |
| 4-3 | `mobilenet_v3_small` 의 InvertedResidual 11개 · SE 9개 · 마지막 특징맵 (576, 7, 7) 확인 |

**핵심 내용**

#### 세 버전 한눈에
> V1 = DS Conv · V2 = inverted residual · V3 = NAS + SE + h-swish
| 버전 | 해 | 핵심 아이디어 |
|---|---|---|
| V1 | 2017 | 일반 Conv → **DS Conv** 로 바꿔 쌓기 |
| V2 | 2018 | **Inverted residual** (×6 늘리고 → dw → 줄이기) + linear bottleneck + 지름길 |
| V3 | 2019 | **NAS** + **SE** + **h-swish** |
- **ResNet bottleneck vs V2** → ResNet: 256→64→256 (줄였다 늘림) · V2: 24→144→24 (늘렸다 줄임) → 그래서 *inverted*
- **왜 늘린 곳에서 depthwise?** → depthwise 는 싸다 + 블록 사이를 지나는 텐서가 얇아(24채널) 메모리가 적다

#### InvertedResidual 블록 (V2)
> 1×1 늘리기 → ReLU6 → dw 3×3 → ReLU6 → 1×1 줄이기 (활성화 없음) → (지름길)
| 단계 | 예 (cin = cout = 24, t = 6) | 파라미터 |
|---|---|---|
| 1×1 늘리기 (24→144) | 24·144 | 3,456 (+ BN 288) |
| depthwise 3×3 (144) | 144·9 | 1,296 (+ BN 288) |
| 1×1 줄이기 (144→24) | 144·24 | 3,456 (+ BN 48) |
| 합계 | | **8,832** |
- **`InvertedResidual(16, 24, stride=2)`** → 지름길 없음 (stride 2 · 채널 다름) → 5,136
- **`use_res`** → `stride == 1 and cin == cout` 일 때만 `x + out`
- ⚠ 마지막 1×1 뒤에 ReLU 를 넣으면 안 된다 — 채널이 적은 곳에서 음수 정보가 잘려 정확도가 떨어진다 (V2 논문)

#### V3 의 두 부품 — h-swish 와 SE
> h-swish = `x · ReLU6(x + 3) / 6` · SE = 채널마다 '지금 중요한가' 를 이미지별로 다시 정함
| 입력 x | h-swish |
|---|---|
| x ≤ −3 | 0 |
| −3 ~ 3 | 2차 곡선 |
| x ≥ 3 | x |
- **sigmoid(exp) 대신** ReLU6 · 곱 · 나눗셈만 쓴다 → 계산이 싸고 int8 양자화에도 유리
- **SE 순서** → 채널 평균(squeeze) → FC → ReLU → FC → 0~1 → 채널마다 곱(excitation)

#### MobileNetV3-Small 구조
| 항목 | 값 |
|---|---|
| InvertedResidual 블록 | **11개** (SE 가 있는 블록 9개) |
| 마지막 특징맵 (224 입력) | (576, 7, 7) |
| classifier | Linear(576→1024) · Hardswish · Dropout · Linear(1024→1000) |
| 파라미터 | **2,542,856** |
| MAC (224) | **56,510,400** (ResNet-50 의 1/72) |
##### 자세히
```text
 ResNet bottleneck             MobileNetV2 inverted residual
 256 ─1×1→ 64 (줄임)           24 ─1×1→ 144 (늘림, ×6)
     ─3×3→ 64                       ─dw 3×3→ 144
     ─1×1→ 256 (늘림)               ─1×1→ 24 (줄임, ReLU 없음)
 블록 사이 텐서: 굵다(256)      블록 사이 텐서: 얇다(24) → 메모리 절약
```

**기본 사용형태**
```python
class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride=1, t=6):
        super().__init__()
        hid = cin * t
        self.use_res = stride == 1 and cin == cout
        self.block = nn.Sequential(
            nn.Conv2d(cin, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(inplace=True),
            nn.Conv2d(hid, hid, 3, stride, 1, groups=hid, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(inplace=True),
            nn.Conv2d(hid, cout, 1, bias=False), nn.BatchNorm2d(cout),     # 줄이기 (linear)
        )
    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out

def h_swish(x):
    return x * F.relu6(x + 3) / 6
```

**연결**
- [VB D3·3교시 Depthwise Separable Convolution](#/vb/3/3) — V1 의 재료
- [VB D3·2교시 연산량 계산](#/vb/3/2) — ResNet bottleneck (반대 방향)
- [VB D2·6교시 ResNet: Residual block](#/vb/2/6) — 지름길의 출처
- [VB D2·5교시 깊이의 문제 · BatchNorm](#/vb/2/5) — 블록마다 BN

**연관 문제**
<<QA 4>>

### 5교시 · 전이학습 (transfer learning)

**키워드**: `전이학습`, `feature extraction`, `fine-tuning`, `requires_grad = False`, `classifier[3] = nn.Linear(1024, 10)`, `1,528,106`, `601,098`, `ImageNet 평균 · 표준편차`, `32 → 96`, `0.6607 · 0.8273 · 0.5470`

**개요**: ImageNet(1,000 클래스 · 128만 장)으로 학습한 모델의 앞쪽 층은 에지 · 질감 · 모양 같은 **일반 특징**을 이미 안다. 그 특징 추출기(`features`)는 가져오고, **마지막 분류기만 우리 클래스 수(10)로 바꿔** 학습하는 것이 전이학습. ① **feature extraction** — features 를 동결하고 분류기만 학습(빠르고 과적합 적음). ② **fine-tuning** — 전체를 작은 학습률로 학습(정확도 높음). ③ 사전학습 모델은 학습 때와 **같은 전처리**(ImageNet 평균 · 표준편차)를 기대하고, 입력은 32 → **96** 으로 키워 넣는다.

**핵심 용어**
- **전이학습(transfer learning)**: 사전학습 특징 추출기 + 새 분류기
- **feature extraction**: `features` 동결(`requires_grad = False`), 분류기만 학습
- **fine-tuning**: 동결 없이 전체를 작은 학습률로 학습
- **동결(freeze)**: 파라미터의 `requires_grad` 를 꺼서 기울기를 안 구하게 하는 것
- **사전학습 가중치(weights)**: `MobileNet_V3_Small_Weights.IMAGENET1K_V1` — ImageNet 으로 학습된 값

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Weights.transforms()` | `W_MNV3S.transforms()` | 이 가중치가 기대하는 전처리 확인 |
| `Parameter.requires_grad` | `p.requires_grad = False` | 파라미터 동결 |
| `copy.deepcopy()` | `copy.deepcopy(model)` | 모델을 통째로 복사 (실험끼리 섞이지 않게) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 5-1 | 사전학습 모델 불러오고 `classifier[3]` 를 `Linear(1024, 10)` 으로 교체 → 파라미터 1,528,106 |
| 5-2 | features 동결 → 학습 파라미터 601,098 개만 Adam 에 → 1 epoch (test ≈ 0.6607) |
| 5-3 | 동결 없이 fine-tuning (≈ 0.8273) vs 처음부터 학습 (≈ 0.5470), 모두 1 epoch |

**핵심 내용**

#### 두 가지 방법
> 사전학습 features + 새 classifier — 동결하면 feature extraction, 전체 학습이면 fine-tuning
| 방법 | 학습하는 부분 | 장점 | 언제 |
|---|---|---|---|
| feature extraction | 분류기만 | 빠르고 과적합 적음 | 데이터 아주 적음 · 원래 데이터와 비슷 |
| fine-tuning | 전체 (작은 lr) | 정확도 높음 | 데이터가 어느 정도 있음 |

#### 파라미터 숫자 — 분류기를 바꾸면
> 1000 클래스용 Linear(1024→1000) → 10 클래스용 Linear(1024→10)
| 항목 | 계산 | 파라미터 |
|---|---|---|
| 원래 마지막 Linear | 1024·1000 + 1000 | 1,025,000 |
| 새 마지막 Linear | 1024·10 + 10 | 10,250 |
| 전체 | 2,542,856 − 1,025,000 + 10,250 | **1,528,106** |
| classifier 전체 | 576·1024 + 1024 + 10,250 | **601,098** (39%) |
| features | 1,528,106 − 601,098 | 927,008 |
- **feature extraction** → 학습되는 것은 classifier 601,098 개뿐
- **같은 1 epoch (실행 예)** → feature extraction **0.6607** · fine-tuning **0.8273** · 처음부터 **0.5470**
- **fine-tuning 이 더 높은 이유** → CIFAR-10 사진(32×32, 흐림)이 ImageNet 과 꽤 달라 앞쪽 특징도 조금 고쳐야 한다

#### 전처리는 사전학습 때와 같게
> RGB · 0~1 → **ImageNet 평균 · 표준편차**로 정규화, 입력은 32 → 96
- **`IMNET_MEAN, IMNET_STD`** = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) — `W_MNV3S.transforms()` 출력과 같은지 확인
- **입력 키우기** → 32×32 그대로면 특징이 사라진다 → `Resize(96)`
- ⚠ 사전학습 모델에 **CIFAR 정규화 상수**를 쓰면 안 된다
- ⚠ 동결한 뒤에도 `model.parameters()` 전체를 옵티마이저에 넘기면 의도가 흐려진다 → `[p for p in … if p.requires_grad]`
- ⚠ 동결한 층의 BN 은 `model.train()` 에서 running 통계가 계속 바뀐다 → 엄격히 고정하려면 그 층만 `eval()`
##### 자세히
```text
 mobilenet_v3_small(weights=IMAGENET)
 ┌──────────── features ─────────────┐ ┌──────── classifier ────────┐
 │ 에지·질감·모양 (927,008)          │ │ 576→1024→Hardswish→Dropout │
 │ ← feature extraction: 동결(❄)     │ │ →[3] Linear(1024, 1000)    │
 │ ← fine-tuning: 같이 학습          │ │      ↓ 교체                │
 └───────────────────────────────────┘ │ [3] Linear(1024, 10)       │
                                       └────────────────────────────┘
```

**기본 사용형태**
```python
torch.manual_seed(0)
mnet = torchvision.models.mobilenet_v3_small(weights=W_MNV3S)
mnet.classifier[3] = nn.Linear(1024, 10)       # 마지막 층만 10 클래스로
mnet = mnet.to(device)

fe = copy.deepcopy(mnet)                        # feature extraction
for p in fe.features.parameters():
    p.requires_grad = False
opt = torch.optim.Adam([p for p in fe.parameters() if p.requires_grad], lr=1e-3)
```

**연결**
- [VB D2·1교시 CIFAR-10 과 MLP 의 한계](#/vb/2/1) — CIFAR 정규화 상수는 train 으로 (여기선 ImageNet 상수)
- [VB D1·6교시 디지털 이미지: 정규화](#/vb/1/6) — 정규화 개념
- [VB D3·4교시 MobileNet V1 · V2 · V3](#/vb/3/4) — 가져오는 모델의 구조
- [VB D1·5교시 학습 루프 · 평가 · 저장](#/vb/1/5) — `state_dict` / 학습 루프

**연관 문제**
<<QA 5>>

### 6교시 · 적은 데이터와 전이학습

**키워드**: `클래스당 100장`, `Subset`, `np.bincount`, `처음부터 0.3560`, `사전학습 0.6561`, `BN momentum 0.01`, `set_bn_momentum(m, 0.1)`, `ImageFolder`, `클래스별 정확도`

**개요**: 현장에서는 클래스당 수천 장을 모으기 어렵다. 전이학습의 진짜 힘은 **데이터가 적을 때** 나온다. ① CIFAR-10 에서 클래스당 **100장(총 1,000장)** 만 써서 처음부터 vs 사전학습을 비교 — 0.3560 vs 0.6561. ② torchvision MobileNetV3 의 BN `momentum=0.01` 때문에 적은 step 에서 test 가 **0.10(찍기)** 으로 나올 수 있다 → momentum 을 0.1 로 맞춘다. ③ 내 데이터는 **ImageFolder** 폴더 구조로 넣으면 나머지 코드는 그대로.

**핵심 용어**
- **Subset**: 데이터셋에서 지정한 인덱스만 골라 쓰는 래퍼
- **균형 잡힌 부분집합**: 클래스마다 같은 개수 (여기선 100장씩)
- **BN momentum**: running 평균 · 분산이 step 마다 새 배치 쪽으로 움직이는 비율 (torchvision MobileNetV3 는 0.01)
- **ImageFolder**: `폴더이름 = 클래스` 구조에서 데이터셋을 만드는 클래스

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.utils.data.Subset()` | `Subset(train_mn, idx_small)` | 지정 인덱스만 골라 쓰기 |
| `np.bincount()` | `np.bincount(targets[idx_small])` | 클래스별 개수 세기 |
| `np.where()` | `np.where(targets == c)[0][:100]` | 조건을 만족하는 인덱스 |
| `datasets.ImageFolder()` | `datasets.ImageFolder("safety/train", transform=tf)` | 폴더 구조로 데이터셋 만들기 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 6-1 | 클래스마다 앞 100장씩 → `idx_small` 1,000개, `np.bincount` 로 전부 100 확인 |
| 6-2 | 처음부터 vs 사전학습 MobileNetV3-S 를 1,000장으로 10 epoch (BN momentum 0.1 로 맞춤) |
| 6-3 | `predict_all` 로 클래스별 정확도 → 사전학습이 가장 많이 올린 클래스 순 출력 |

**핵심 내용**

#### 적은 데이터 실험 (클래스당 100장)
> 데이터가 50배 적어도 사전학습 모델은 쓸 만하다
| 모델 | test 정확도 (실행 예) |
|---|---|
| 처음부터 학습 | **0.3560** |
| 사전학습 + 전체 학습 | **0.6561** |
- **1,000 / 50,000 = 1/50** — 데이터가 50배 적다
- **정확도가 가장 크게 오른 클래스** → frog (+0.59) · bird (+0.43) · dog (+0.38) — ImageNet 에 비슷한 대상이 많을수록 사전학습 특징이 바로 쓰인다
- ⚠ 데이터를 직접 모을 때도 `np.bincount` 로 **클래스별 개수** 부터 확인 — 한쪽이 많으면 모델이 그쪽으로 기운다 (안전 데이터에서 '미착용' 이 적은 경우)
- **클래스별 정확도** 도 같이 본다 — 전체 평균만 보면 약한 클래스가 숨는다

#### BN momentum 함정
> torchvision MobileNetV3 의 BN 은 `momentum=0.01` → step 이 적으면 running 통계가 초기값에서 못 벗어난다
- **step 수** → 1,000장 · 배치 64 = epoch 당 **16 step** → 10 epoch = 160 step
- **증상** → `model.train()` 에서는 학습되는데(train 정확도 ↑) `model.eval()` 에서는 엉뚱한 통계로 정규화돼 test **0.10(찍기)**
- **해결** → 두 모델 모두 `set_bn_momentum(m, 0.1)` 로 맞춘다 (환경 셀의 함수)
- 사전학습 모델은 ImageNet 의 running 통계를 이미 갖고 있어서 문제가 작다
- ⚠ train 정확도는 오르는데 test 만 찍기 수준이면 **BN · `eval()` 을 먼저 의심**
##### 자세히
```text
 running_mean ← (1 − momentum) · running_mean + momentum · batch_mean
 momentum 0.01 : 160 step 후에도 초기값(평균 0 · 분산 1) 영향이 남음
                 (1 − 0.01)^160 ≈ 0.20 → 초기값이 아직 20% 남아 있다
 momentum 0.1  : (1 − 0.1)^160 ≈ 0 → 충분히 따라온다
```

#### 내 데이터로 바꾸려면 — ImageFolder
> 클래스 이름으로 폴더를 나눠 두면 나머지 코드(전처리 · DataLoader · 학습)는 그대로
```text
safety/
  train/ helmet/ 0001.jpg … · no_helmet/ … · no_person/ …
  val/   helmet/ … · no_helmet/ … · no_person/ …
```
- **`train_set.classes`** → `['helmet', 'no_helmet', 'no_person']` — **폴더 이름 순서가 레이블 번호**
- **분류기 크기** → `nn.Linear(1024, len(train_set.classes))`

**기본 사용형태**
```python
targets = np.array(train_mn.targets)
idx_small = np.concatenate([np.where(targets == c)[0][:100] for c in range(10)])   # 클래스당 100장
small_loader = DataLoader(Subset(train_mn, idx_small), batch_size=64, shuffle=True, num_workers=NW)
print(len(idx_small), np.bincount(targets[idx_small]))      # 1000 [100 100 … 100]

train_set = datasets.ImageFolder("safety/train", transform=mn_train_tf)
mnet.classifier[3] = nn.Linear(1024, len(train_set.classes))
```

**연결**
- [VB D3·5교시 전이학습](#/vb/3/5) — 같은 사전학습 모델 · 같은 교체
- [VB D2·5교시 깊이의 문제 · BatchNorm](#/vb/2/5) — running_mean · running_var, `eval()` 의 의미
- [NN D4·5교시 과적합](#/nn/4/5) — 데이터가 적을 때 일어나는 일

**연관 문제**
<<QA 6>>

### 7교시 · Grad-CAM: 모델은 어디를 보는가

**키워드**: `Grad-CAM`, `마지막 합성곱 특징맵 A`, `기울기의 공간 평균 α`, `ReLU(Σ αA)`, `0~1 정규화`, `register_forward_hook`, `ResNet-20 8×8`, `MobileNetV3 3×3`, `오답 CAM 비교`

**개요**: 정확도가 높아도 모델이 엉뚱한 곳(배경 · 워터마크)을 보고 맞힐 수 있다. 현장 안전 모델이 안전모가 아니라 '공사장 배경' 을 본다면 위험하다. **Grad-CAM** 은 마지막 합성곱 특징맵에서 판단 근거가 된 위치를 히트맵으로 보여 준다. ① 마지막 합성곱 출력 A(채널 K개)와 클래스 점수 y^c 의 A 에 대한 기울기를 구한다. ② 채널 중요도 α_k = 기울기의 공간 평균. ③ CAM = ReLU(Σ α_k A^k) → 0~1 정규화 → 입력 크기로 키워 겹침. ④ CAM 해상도는 마지막 특징맵 크기(ResNet-20 **8×8**, MobileNetV3 **3×3**).

**핵심 용어**
- **Grad-CAM**: 분류 모델이 이미지의 어느 위치를 보고 판단했는지 히트맵으로 보여 주는 방법
- **채널 중요도 α_k**: 클래스 점수에 대한 기울기를 공간(H, W)으로 평균낸 값 — 채널 k 가 그 클래스에 얼마나 중요한가
- **CAM(Class Activation Map)**: 중요도로 가중합한 특징맵 (위치별 근거)
- **정규화(0~1)**: `cam / cam.max()` — 가장 뜨거운 곳을 1 로

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `Tensor.register_hook()` | `out.register_hook(fn)` | 역전파 때 이 텐서의 기울기를 가로채기 |
| `F.interpolate()` | `F.interpolate(cams[:, None], size=32, mode="bilinear", align_corners=False)` | CAM 을 입력 크기로 키우기 |
| `Tensor.amax()` | `cam.amax(dim=(1, 2), keepdim=True)` | 여러 축의 최댓값 |
| `Tensor.mean()` | `grads.mean(dim=(2, 3), keepdim=True)` | 공간 평균 (α) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 7-1 | `GradCAM` 클래스 — hook 으로 `layer` 출력과 기울기를 저장, `α`, `ReLU(Σ αA)`, 정규화. ResNet-20 `layer3` → CAM (16, 8, 8) |
| 7-2 | ResNet-20 오답 8장에서 **예측 클래스** CAM 과 **정답 클래스** CAM(`cls=` 지정)을 나란히 |
| 7-3 | fine-tuning 된 MobileNet 의 `features[-1]` 에 Grad-CAM → 3×3 |
| `show_cams` | CAM 을 `F.interpolate` 로 키워 원본 위에 `jet` 히트맵 (alpha 0.45) 으로 겹쳐 그림 |

**핵심 내용**

#### Grad-CAM 계산 4단계
> A = 마지막 합성곱 출력 · y^c = 클래스 점수 → α_k = mean(∂y^c/∂A^k) → CAM = ReLU(Σ α_k A^k)
| 단계 | 하는 일 | 코드 |
|---|---|---|
| ① | forward 때 층 출력 A 저장, 클래스 점수 y^c 만 backward | `logits[range, cls].sum().backward()` |
| ② | 채널 중요도 α: 기울기의 공간 평균 | `grads.mean(dim=(2, 3), keepdim=True)` → (N, K, 1, 1) |
| ③ | 채널 가중합 후 ReLU | `F.relu((alpha * acts).sum(dim=1))` → (N, H, W) |
| ④ | 0~1 정규화 후 입력 크기로 키워 겹침 | `cam / (cam.amax(...) + 1e-8)` |
- **ReLU 를 거는 이유** → 그 클래스 점수를 **올리는** 위치만 보고 싶어서
- **`cls=None`** → 예측 클래스(argmax) · `cls=정답` 으로 주면 '정답이라면 어디를 봐야 했나'
- **`if out.requires_grad`** → `torch.no_grad()` 안에서는 hook 등록을 건너뜀
- ⚠ 쓰고 나면 `cam.remove()` — hook 이 남아 중복 등록되지 않게

#### CAM 해상도 = 마지막 특징맵 크기
| 모델 | 마지막 합성곱 | CAM 크기 |
|---|---|---|
| ResNet-20 (32×32 입력) | `layer3` 출력 (64, 8, 8) | **8×8** |
| MobileNetV3-S (96×96 입력) | `features[-1]` — stride 32 → 96/32 | **3×3** |
- **경량 모델은 빨리 줄여서 연산을 아끼므로 CAM 이 거칠다**
- ⚠ 위치를 정밀하게 알아야 하면 분류가 아니라 **객체 탐지**(Day 4)

#### 오답에서 무엇에 속았나
> 예측 클래스 CAM 이 물체가 아닌 배경 · 가장자리에 몰려 있으면 맥락에 속은 것
- **예측 CAM ≠ 정답 CAM** → 두 위치를 비교하면 헷갈린 이유가 보인다
- **오답이 배경에 쏠려 있으면** → 데이터에 그 상황(다양한 배경)을 더 넣어야 한다
##### 자세히
```text
 이미지 ─► [마지막 conv 출력 A (K, h, w)] ─► GAP ─► FC ─► 점수 y^c
                        ▲                               │
                        └──── ∂y^c/∂A (backward) ◄──────┘
 α_k = mean_{h,w}(∂y^c/∂A^k)            채널 k 의 중요도
 CAM = ReLU(Σ_k α_k · A^k)  (h, w)  →  /max  →  bilinear 로 H×W 로 키움
```

**기본 사용형태**
```python
class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        self.handle = layer.register_forward_hook(self._save)
    def _save(self, module, inp, out):
        self.acts = out
        if out.requires_grad:                       # no_grad 안에서는 건너뜀
            out.register_hook(lambda g: setattr(self, "grads", g))
    def __call__(self, x, cls=None):
        self.model.eval()
        logits = self.model(x)
        if cls is None: cls = logits.argmax(1)
        self.model.zero_grad()
        logits[torch.arange(len(x)), cls].sum().backward()
        alpha = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * self.acts).sum(dim=1))
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach(), logits.argmax(1).detach()
    def remove(self):
        self.handle.remove()

cam_r20 = GradCAM(r20, r20.layer3)
```

**연결**
- [NN D3·3교시 연쇄 법칙](#/nn/3/3) — 기울기를 구하는 원리 (역전파)
- [VB D1·3교시 autograd](#/vb/1/3) — `backward()` 로 기울기를 얻는 방법
- [VB D2·6교시 ResNet: Residual block](#/vb/2/6) — `layer3` 출력 (64, 8, 8)
- [VB D3·2교시 연산량 계산](#/vb/3/2) — `register_forward_hook` 사용법

**연관 문제**
<<QA 7>>

### 8교시 · 미니 프로젝트: MobileNetV3 전이학습 vs ResNet-20

**키워드**: `MobileNetV3-S fine-tuning 5 epoch`, `AdamW + OneCycleLR`, `test 0.8842`, `ResNet-20 0.8587`, `MAC 11,057,600 vs 40,813,184`, `1/3.7`, `지연시간 6.18 ms vs 2.87 ms`, `비교표`, `오답 8장 Grad-CAM`

**개요**: 사전학습 MobileNetV3-Small 을 CIFAR-10 전체로 fine-tuning 해서 Day 2 의 ResNet-20(처음부터 20 epoch, test ≈ 0.86)과 비교한다. 정확도만이 아니라 **연산량 · 지연시간**까지 보고 장치에 올릴 모델을 고르는 게 목표. 결과(실행 예): MobileNetV3-S **0.8842** > ResNet-20 **0.8587**, MAC 은 약 **1/3.7**, 그런데 CPU 지연시간은 오히려 MobileNet 이 **더 길다**(6.18 vs 2.87 ms).

**핵심 용어**
- **AdamW**: Adam + weight decay 를 분리해서 적용한 옵티마이저
- **OneCycleLR**: 학습률을 올렸다 내리는 스케줄러 (`steps_per_epoch = len(loader)`)
- **fine-tuning**: 사전학습 모델 전체를 조금씩 고쳐 학습 (5교시)
- **비교표**: 모델마다 (파라미터, MAC, 지연시간, test 정확도)를 같은 기준으로 적은 표

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.optim.AdamW()` | `torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)` | weight decay 분리형 Adam |
| `OneCycleLR()` | `OneCycleLR(opt, max_lr=1e-3, epochs=5, steps_per_epoch=len(loader))` | 학습률 올렸다 내리기 |
| `torch.save()` | `torch.save(model.state_dict(), "mnv3s_cifar10.pt")` | 가중치 저장 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 과제 1 (20점) | `train_mn`(증강 + Resize 96 + ImageNet 정규화) 배치 128 · 사전학습 모델 · `classifier[3]=Linear(1024,10)` · AdamW + OneCycleLR → step / epoch = **391** |
| 과제 2 (30점) | `fit` 으로 5 epoch fine-tuning → test ≥ 0.85, 저장 |
| 과제 3 (20점) | `compare` 딕셔너리 — (파라미터, MAC, 지연시간, 정확도) 두 모델을 각자의 입력(32 · 96)으로 |
| 과제 4 (10점) | 오답 8장 Grad-CAM (`wrong_mn`, `cams_wrong` (8, 3, 3)) |
| 과제 5 (10 + 10점) | 결과 해석 — 라즈베리파이에 올릴 모델 고르고 숫자로 근거 |

**핵심 내용**

#### 비교표 (실행 예)
> 정확도는 MobileNet 이 높고 MAC 은 1/3.7 이지만, CPU 지연시간은 ResNet-20 이 더 짧다
| 모델 | 파라미터 | MAC | 지연시간 | test |
|---|---|---|---|---|
| ResNet-20 (scratch, 32) | 272,474 | 40,813,184 | **2.87 ms** | 0.8587 |
| MobileNetV3-S (transfer, 96) | 1,528,106 | 11,057,600 | 6.18 ms | **0.8842** |
- **파라미터** → 1,528,106 / 272,474 = **5.6배** (MobileNet 이 더 많다)
- **MAC** → 40,813,184 / 11,057,600 = **3.7배** (MobileNet 이 더 적다)
- **step / epoch** → 50,000 / 128 → 올림 **391**

#### 과제 5 해석 포인트
> 파라미터 5.6배인데 MAC 은 1/3.7 — 파라미터와 연산량은 별개
- **파라미터 ↑ 인데 MAC ↓** → 파라미터는 '위치마다 한 번' 쓰이는 1×1 · FC 에 많이 있고, 96 입력은 stride 2 를 일찍 써서 큰 특징맵에서의 연산이 적다. ResNet-20 은 32×32 를 유지한 채 3×3 을 여러 번 해서 파라미터는 적어도 MAC 이 크다
- **MAC 적다고 빠른 건 아니다** → MobileNet 입력이 96×96(9배), depthwise · SE · 층 수가 많아 메모리 이동과 호출 오버헤드가 크다 (3교시와 같은 이야기) → **실제 장치에서 재서 결정**
- **224×224 로 넣으면** → MAC 은 해상도 제곱에 비례 `(224/96)² ≈ 5.4배` (12.1M → 56.5M, 1000 클래스 기준). 32×32 원본을 크게 키운 것이라 정확도 이득은 제한적
- **현장 데이터(클래스당 수백 장)** → 6교시 (1,000장에서 처음부터 0.3560 vs 사전학습 0.6561) → 사전학습 경량 모델 fine-tuning + 증강 + Grad-CAM 으로 안전모를 보는지 확인
- ⚠ 모델 선택은 **정확도 · 파라미터 · MAC · 실제 지연시간을 함께**

#### 다음 단계 (더 알아보기)
- 오늘의 MobileNetV3 는 float32 · PyTorch 그대로 → 장치에서는 **ONNX 내보내기 → TensorRT 엔진**(11/4~), **int8 양자화 · 가지치기**(11/16~)가 더 필요
- '어디에 있는가' 까지 알아야 하는 현장 안전은 다음 시간의 **객체 탐지**(Day 4)
##### 자세히
```text
 자동 채점 80점 (과제 5 해석 20점은 강사 채점)
   과제 1 +20  AdamW · OneCycleLR · out_features == 10 · len(loader) == 391
   과제 2 +30  history 길이 == EPOCHS, mn_acc ≥ 0.85
   과제 3 +20  MAC: MobileNetV3-S 11,057,600 · ResNet-20 40,813,184
   과제 4 +10  오답 8개 · cams_wrong.shape == (8, 3, 3)
```

**기본 사용형태**
```python
EPOCHS = 5
mn_loader = DataLoader(train_mn, batch_size=128, shuffle=True, num_workers=NW, pin_memory=True)
model = torchvision.models.mobilenet_v3_small(weights=W_MNV3S)
model.classifier[3] = nn.Linear(1024, 10)
model = model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(mn_loader))
hist_mn = fit(model, mn_loader, test_mn_loader, EPOCHS, opt, sched, label="MobileNetV3-S")
```

**연결**
- [VB D2·8교시 ResNet-20 85% 도전](#/vb/2/8) — 비교 대상
- [VB D2·7교시 학습 기법: 증강 · 학습률 스케줄](#/vb/2/7) — OneCycleLR · 증강
- [VB D3·5교시 전이학습](#/vb/3/5) — fine-tuning
- [VB D3·7교시 Grad-CAM](#/vb/3/7) — 오답 분석
- [VB D3·1교시 정확도 vs 속도](#/vb/3/1) — `latency_ms` 로 지연시간 측정

**연관 문제**
<<QA 8>>
