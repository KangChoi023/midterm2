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
> 장치(라즈베리파이 · Jetson)에서 직접 추론할 때는 정확도 · **지연시간(latency)** · 파라미터(메모리) · **MAC** 을 함께 본다
- **지연시간** → 한 장 넣고 결과까지 ms — 30 FPS 실시간이면 **33 ms** 안
- **예** → ResNet-50 vs MobileNetV3-Small: MAC 4,089,184,256 vs 56,510,400 (약 **72배**), ImageNet top-1 76.1% vs 67.7%
- **실행 예 (CPU 1 스레드)** → ResNet-20 2.87 ms · MobileNetV3-S 8.8 ms · ResNet-18 70 ms · ResNet-50 134 ms
- **`torchvision.models.resnet50()`** → `weights=None` 으로 구조만 만들어 비용을 잰다
- ⚠ ResNet-50 은 장치에서 33 ms 안에 못 끝난다

#### 지연시간 재는 법 — 온디바이스 흉내
> CPU · 스레드 1 · 배치 1 · **warm-up** 3회 버리고 30회 평균
- **`torch.set_num_threads()`** → 1 로 작은 장치의 코어 하나를 흉내 — 재고 나서 원래 값으로
- **warm-up** → 처음 몇 번은 메모리 할당 · 캐시 때문에 느려서 버린다
- **`time.perf_counter()`** → 반복 전후 시간 차 / 횟수 × 1000 = ms
- **`copy.deepcopy(model).cpu().eval()`** → GPU 의 원본을 건드리지 않고 복사해서 잰다

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

**상세 학습**

#### 이 교시에서 답하는 질문
어제 ResNet-20 은 CIFAR-10 에서 85% 를 넘겼다. 그런데 이 모델을 라즈베리파이나 Jetson 같은 작은 장치에 올린다면? 정확도만큼 중요한 것이 **얼마나 빠른가**, **얼마나 무거운가** 다. 이 교시는 모델을 고르는 네 가지 기준을 정하고, 장치 환경을 흉내 내 지연시간을 재는 법을 만든다.

#### 개념 1 · 온디바이스 모델 선택 기준
서버가 아니라 장치에서 직접 추론하는 것을 **온디바이스**라 한다.

| 지표 | 장치에서 의미 |
|---|---|
| 정확도 | 얼마나 잘 맞히나 |
| 지연시간 (latency) | 한 장 넣고 결과까지 ms — 30 FPS 실시간이면 33 ms 안 |
| 파라미터 | 모델 파일 크기 · 메모리 |
| MAC | 계산 시간 · 전력 |

![CPU 1 스레드 지연시간 (실행 예) — ResNet-50 은 33 ms 를 크게 넘는다](assets/fig/vb/d3-latency.svg)

ResNet-50 과 MobileNetV3-Small 의 MAC 은 약 72배 차이인데 ImageNet 정확도 차이는 8.4%p 뿐이다. 경량 모델은 연산을 수십 분의 1로 줄이면서 정확도는 몇 %p 만 내준다.

#### 개념 2 · 지연시간 재는 법 — 장치 흉내
| 조건 | 이유 |
|---|---|
| CPU | 작은 장치는 GPU 가 없거나 약하다 |
| 스레드 1 | 작은 장치의 코어 하나를 흉내 |
| 배치 1 | 카메라 프레임은 한 장씩 들어온다 |
| warm-up 3회 버림 | 첫 실행은 메모리 할당 · 캐시로 느리다 |
| 30회 평균 | 한 번 재면 잡음이 크다 |

#### 코드로 확인
```python
def latency_ms(model, input_size, reps=30, threads=1):
    m = copy.deepcopy(model).cpu().eval()          # 원본 보존
    x = torch.randn(input_size)
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    with torch.no_grad():
        for _ in range(3):                          # warm-up
            m(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            m(x)
        dt = (time.perf_counter() - t0) / reps
    torch.set_num_threads(old)
    return dt * 1000
```
- **무엇을 하나**: 모델을 CPU 로 복사해 스레드 1개 · 배치 1장으로 warm-up 후 평균 시간을 ms 로 잰다.
- **어떤 개념**: 개념 2.
- **출력**: ResNet-20 (32 입력) **2.87 ms** (실행 예 — 컴퓨터마다 다름)

```python
models_224 = {"ResNet-18": torchvision.models.resnet18(weights=None),
              "ResNet-50": torchvision.models.resnet50(weights=None),
              "MobileNetV3-S": torchvision.models.mobilenet_v3_small(weights=None)}
for name, m in models_224.items():
    print(name, count_params(m), count_macs(m, (1, 3, 224, 224)), latency_ms(m, (1, 3, 224, 224), reps=5))
```
- **무엇을 하나**: 가중치 없이 구조만 만들어 파라미터 · MAC · 지연시간을 한 표로 비교한다.
- **어떤 개념**: 개념 1.
- **출력**: ResNet-50 MAC 4,089,184,256 · 134 ms / MobileNetV3-S MAC **56,510,400** · 8.8 ms (실행 예)

#### 한 번에 정리
- 온디바이스 모델은 정확도 · 지연시간 · 파라미터 · MAC 을 함께 본다.
- 지연시간은 CPU · 스레드 1 · 배치 1 · warm-up 후 평균.
- 경량 모델: MAC 은 수십 분의 1, 정확도는 몇 %p 차이.
- ⚠ 30 FPS 실시간 = 한 장 33 ms — ResNet-50 은 장치에서 안 된다.

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

#### 메모리와 계산 — 세 지표
> 파일 크기 = 파라미터 × 4 B (float32) · 실행 메모리 = + **활성화 메모리** · 계산 = MAC
- **활성화(activation)** → 각 층 출력 크기의 합 — 해상도가 클수록 커진다 (`Tensor.numel()` 로 원소 수)
- **forward hook** → `Module.register_forward_hook()` 으로 Conv · Linear 출력을 엿봐 활성화를 더하고 `h.remove()`
- **예** → ResNet-50 25,557,032 × 4 B ≈ **102 MB** (가중치만) · ResNet-20 1.09 MB
- ⚠ 파라미터가 적어도 큰 해상도를 오래 유지하면 활성화가 크다

#### bottleneck 블록 — 1×1 로 줄였다 늘리기
> 채널이 많은(256) 곳에서 바로 3×3 을 하지 않고 1×1 로 64 로 줄여서 → 3×3 → 1×1 로 256 (ResNet-50 이상)
- **1×1 Conv** → 위치마다 채널끼리만 섞는 "채널 압축기"
- **예** → basic 2 · 256·256·9 = **1,179,648** vs bottleneck 256·64 + 64·64·9 + 64·256 = **69,632** (16.9배 ↓)
- **그래서** → ResNet-50 은 ResNet-18 보다 층이 3배 가까이 많은데 MAC 은 2.3배

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

**상세 학습**

#### 이 교시에서 답하는 질문
"무겁다" 를 숫자로 나누면 무엇일까? 이 교시는 모델이 장치에서 차지하는 **저장 공간 · 실행 메모리 · 계산량**을 한 번에 재는 함수를 만들고, ResNet-50 이 층이 많은데도 연산을 아끼는 비결인 **bottleneck** 을 계산으로 본다.

#### 개념 1 · 세 가지 지표
| 지표 | 계산 | 장치에서 |
|---|---|---|
| 파라미터 | 가중치 개수 × 4 B (float32) | 모델 파일 크기 |
| 활성화 | 각 층 출력 원소 수의 합 × 4 B | 실행 중 메모리 — 해상도 ↑ 이면 커짐 |
| MAC | 곱셈-덧셈 횟수 | 계산 시간 · 전력 |

예: ResNet-50 은 25,557,032 × 4 B ≈ 102 MB (가중치만), ResNet-20 은 272,474 × 4 B ≈ 1.09 MB. 파라미터가 적어도 큰 해상도를 오래 유지하면 활성화 메모리는 크다.

#### 개념 2 · bottleneck — 1×1 로 줄였다 늘리기
3×3 합성곱의 가중치는 Cout · Cin · 9 — 채널이 많으면 급격히 커진다. 그래서 채널이 256 인 곳에서는 1×1 Conv 로 64 로 줄이고, 비싼 3×3 을 거기서 한 뒤, 1×1 로 다시 256 으로 늘린다.

![basic 블록과 bottleneck 블록 — 1×1 로 채널을 줄인 곳에서 3×3 을 한다](assets/fig/vb/d3-bottleneck.svg)

| 블록 | 계산 | 가중치 |
|---|---|---|
| basic (3×3 두 번) | 2 · 256·256·9 | 1,179,648 |
| bottleneck | 256·64 + 64·64·9 + 64·256 | **69,632** (16.9배 ↓) |

1×1 Conv 는 위치마다 채널끼리만 섞는 "채널 압축기" 다. 덕분에 ResNet-50 은 ResNet-18 보다 층이 3배 가까이 많은데 MAC 은 2.3배다.

#### 코드로 확인
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
    return {"params": p, "macs": count_macs(model, input_size), "param_MB": p * 4 / 1e6, "act_MB": acts[0] * 4 / 1e6}
```
- **무엇을 하나**: 가짜 입력을 한 번 통과시키며 hook 으로 Conv · Linear 출력 원소 수를 더해 활성화 메모리를 구하고, 파라미터 · MAC 과 함께 돌려준다.
- **어떤 개념**: 개념 1 — Day 2 의 count_macs 를 넓힌 것.
- **출력**: ResNet-50 (224) 파라미터 약 **102.2 MB** · ResNet-20 1.09 MB

```python
def w(cin, cout, k):
    return cout * cin * k * k

basic = 2 * w(256, 256, 3)
bottleneck = w(256, 64, 1) + w(64, 64, 3) + w(64, 256, 1)
```
- **무엇을 하나**: 256 채널에서 두 블록의 가중치 수를 계산한다.
- **어떤 개념**: 개념 2.
- **출력**: basic **1,179,648** · bottleneck **69,632** · **16.9배** 적음

#### 한 번에 정리
- 파일 크기 = 파라미터 × 4 B, 실행 메모리 = + 활성화, 계산 = MAC.
- bottleneck: 1×1 로 줄이고 → 3×3 → 1×1 로 늘리기 — 가중치 1/17.
- 1×1 Conv = 채널 압축기.
- ⚠ hook 은 다 쓰면 remove().

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

#### Depthwise Separable Convolution
> 일반 3×3 이 한꺼번에 하던 공간 섞기 · 채널 섞기를 **Depthwise Conv**(채널마다 3×3) → **Pointwise Conv**(1×1) 둘로 쪼갠 **DS Conv**
- **`nn.Conv2d()`** → depthwise 는 `nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False)`
- **식** → DS / 일반 = 1/Cout + 1/K² — K = 3 이면 약 1/9
- **예** → 64→128, 3×3: 73,728 → 576 + 8,192 = **8,768** (11.9%) · MAC 75,497,472 → 8,978,432 (0.1189)
- **연산의 93%** → pointwise 1×1

#### 실제 속도는 덜 빨라진다
> MAC 8.4배 ↓ → 실측 **4.1배** (64 × 56 × 56, CPU 1 스레드: 5.14 → 1.26 ms)
- **메모리 병목** → depthwise 는 곱셈은 적지만 채널마다 따로 메모리를 읽어, 계산 대비 데이터 이동이 많다
- ⚠ MAC 이 적다고 반드시 빠르지 않다 → 실제 장치에서 잰다 (Day 5 ONNX · TensorRT)

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

**상세 학습**

#### 이 교시에서 답하는 질문
bottleneck 은 채널 수를 줄여서 3×3 을 싸게 했다. 3×3 합성곱 자체를 더 싸게 만들 수는 없을까? 일반 합성곱은 "이웃 9칸(공간)" 과 "모든 입력 채널" 을 한꺼번에 섞는다. MobileNet 은 이 둘을 **쪼갠다** — Depthwise Separable Convolution.

#### 개념 1 · 공간 섞기와 채널 섞기를 둘로
![일반 3 × 3 합성곱과 Depthwise Separable 합성곱 — 64 → 128 에서 가중치 73,728 → 8,768](assets/fig/vb/d3-dsconv.svg)

| 단계 | 하는 일 | 가중치 (64 → 128) |
|---|---|---|
| 일반 3×3 | 필터 하나가 64채널 × 3×3 을 한꺼번에 | 128·64·9 = 73,728 |
| ① Depthwise 3×3 (`groups = 64`) | 채널마다 자기 필터 하나로 공간만 | 64·9 = 576 |
| ② Pointwise 1×1 | 위치마다 채널만 | 64·128 = 8,192 |
| DS 합계 | | **8,768** (11.9%) |

#### 개념 2 · 연산량 공식
> DS / 일반 = (Cin·K²·HW + Cin·Cout·HW) / (Cin·Cout·K²·HW) = 1/Cout + 1/K²

K = 3 이면 약 1/9 ~ 1/8, Cout 이 클수록 1/9 에 가깝다. 64 → 128, 32 × 32 에서 MAC 75,497,472 → 8,978,432, 비율 0.1189 = 1/128 + 1/9 로 공식과 같다. 남은 연산의 93% 는 pointwise 다.

#### 개념 3 · 실제 속도는 덜 빨라진다
64 × 56 × 56 입력, CPU 1 스레드에서 일반 5.14 ms → DS 1.26 ms — 4.1배 (MAC 은 8.4배 감소). depthwise 는 곱셈은 적지만 채널마다 따로 메모리를 읽어, 계산에 비해 데이터 이동이 많다(**메모리 병목**). MAC 이 적다고 반드시 빠르지는 않다 → 실제 장치에서 재야 한다.

#### 코드로 확인
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
- **무엇을 하나**: depthwise(groups = cin) → pointwise(1×1) 두 단계 합성곱 블록.
- **어떤 개념**: 개념 1.
- **출력**: `DSConv(64, 128)` 과 일반 Conv 모두 (1, 64, 32, 32) → **(1, 128, 32, 32)** — 출력 모양은 같다

```python
conv_w = lambda m: sum(c.weight.numel() for c in m.modules() if isinstance(c, nn.Conv2d))
w_std, w_ds = conv_w(std), conv_w(ds)
m_std, m_ds = count_macs(std, (1, 64, 32, 32)), count_macs(ds, (1, 64, 32, 32))
formula = 1 / 128 + 1 / 9
```
- **무엇을 하나**: 일반 vs DS 의 Conv 가중치와 MAC 을 세고 공식과 비교한다.
- **어떤 개념**: 개념 2.
- **출력**: 가중치 73,728 → **8,768** (0.1189) · MAC 75,497,472 → 8,978,432 (**0.1189**) · 공식 0.1189

#### 한 번에 정리
- DS Conv = Depthwise 3×3 (채널마다 공간) + Pointwise 1×1 (채널 섞기).
- 연산량 비율 = 1/Cout + 1/K² ≈ 1/9.
- 64 → 128: 73,728 → 8,768.
- ⚠ 실제 속도 향상(4.1배)은 MAC 감소(8.4배)보다 작다 — 메모리 병목.

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

#### MobileNet V1 · V2 · V3
> V1 (2017) = DS Conv 쌓기 · V2 (2018) = **Inverted residual** · V3 (2019) = **NAS** + **SE** + **h-swish**
- **V1** → 일반 Conv 를 DS Conv 로 바꿔 쌓기
- **V2** → 1×1 로 ×6 늘리고 → dw 3×3 → 1×1 로 줄이기 (ResNet bottleneck 과 반대라서 inverted)
- **V3** → 구조를 자동 탐색(NAS)하고 SE · h-swish 를 더함
- **`torchvision.models.mobilenet_v3_small()`** → InvertedResidual 11개 (SE 9개) · 마지막 특징맵 (576, 7, 7) · 파라미터 **2,542,856** · MAC 56,510,400

#### Inverted residual 블록 (V2)
> 1×1 늘리기 → ReLU6 → dw 3×3 → ReLU6 → 1×1 줄이기 (활성화 없음) → stride 1 · cin = cout 이면 지름길
- **왜 넓은 곳에서 depthwise** → depthwise 는 싸고, 블록 사이 텐서가 얇아(24) 메모리가 적다
- **Linear bottleneck** → 마지막 1×1 뒤에 활성화를 안 넣는다 — 채널이 적은 곳에서 ReLU 가 음수 정보를 잘라 손실이 커서
- **ReLU6** → `nn.ReLU6()` = min(max(0, x), 6) — 양자화 · 저정밀 계산에 유리
- **예** → 24 → 144 → 24 블록 = 3,456 + 1,296 + 3,456 + BN 624 = **8,832** · 16 → 24 stride 2 (지름길 없음) 5,136

#### V3 의 부품 — h-swish 와 SE
> **h-swish** = x · ReLU6(x + 3) / 6 — swish(x · σ(x)) 를 sigmoid 없이 근사, 장치에서 싸다
- **`F.relu6()` · `nn.Hardswish()`** → 직접 만든 h_swish 와 내장 결과가 같다 (allclose) — x ≤ −3 → 0, x ≥ 3 → x
- **SE(Squeeze-and-Excitation)** → 채널 평균(squeeze) → FC → ReLU → FC → 0 ~ 1 → 채널마다 곱(excitation) — 이미지마다 채널 중요도를 다시 정함

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

**상세 학습**

#### 이 교시에서 답하는 질문
3교시의 DS Conv 를 뼈대로 MobileNet 은 세 번 진화했다. 각 버전이 무엇을 더했는지, 특히 V2 의 **inverted residual** 이 왜 ResNet 의 bottleneck 과 "반대" 인지, V3 의 새 부품(h-swish · SE)이 무엇인지 본다.

#### 개념 1 · 세 버전
| 버전 | 해 | 핵심 아이디어 |
|---|---|---|
| V1 | 2017 | 일반 Conv → DS Conv 로 바꿔 쌓기 |
| V2 | 2018 | Inverted residual + linear bottleneck + 지름길 |
| V3 | 2019 | NAS(구조 자동 탐색) + SE + h-swish |

#### 개념 2 · Inverted residual — 늘렸다 줄이기
![ResNet bottleneck 은 줄였다 늘리고, MobileNetV2 는 늘렸다 줄인다](assets/fig/vb/d3-inverted.svg)

ResNet bottleneck 은 256 → 64 → 256 (줄였다 늘림), V2 는 24 → 144 → 24 (×6 늘렸다 줄임). depthwise 는 싸서 넓은(144) 곳에서 해도 부담이 적고, 블록 사이를 지나는 텐서가 얇아(24) 메모리를 아낀다.
- **linear bottleneck**: 마지막 1×1(줄이기) 뒤에는 활성화를 넣지 않는다. 채널이 적은 곳에서 ReLU 가 음수 정보를 잘라 손실이 커지기 때문이다.
- **ReLU6** = min(max(0, x), 6) — 값의 범위가 정해져 저정밀 계산 · 양자화에 유리.
- 지름길은 stride 1 이고 입출력 채널이 같을 때만.

| 단계 (24 → 24, t = 6) | 파라미터 |
|---|---|
| 1×1 늘리기 24 → 144 | 3,456 + BN 288 |
| depthwise 3×3 (144) | 1,296 + BN 288 |
| 1×1 줄이기 144 → 24 | 3,456 + BN 48 |
| 합계 | **8,832** |

#### 개념 3 · V3 의 부품 — h-swish 와 SE
![h-swish 는 swish 를 sigmoid 없이 근사한다](assets/fig/vb/d3-hswish.svg)

**h-swish** = x · ReLU6(x + 3) / 6 — swish(x · σ(x))와 거의 같은 모양인데 exp 가 없어 장치에서 싸고 int8 양자화에도 유리하다.

![SE — 채널 평균 → 작은 FC 두 개 → 0 ~ 1 가중치 → 채널마다 곱하기](assets/fig/vb/d3-se.svg)

**SE** 는 이미지마다 "지금 중요한 채널" 을 다시 정한다. MobileNetV3-Small 은 InvertedResidual 11개 중 9개에 SE 가 있고, 파라미터 2,542,856 · MAC 56,510,400 (224 입력).

#### 코드로 확인
```python
class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride=1, t=6):
        super().__init__()
        hid = cin * t
        self.use_res = stride == 1 and cin == cout
        self.block = nn.Sequential(
            nn.Conv2d(cin, hid, 1, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(inplace=True),                        # 늘리기
            nn.Conv2d(hid, hid, 3, stride, 1, groups=hid, bias=False), nn.BatchNorm2d(hid), nn.ReLU6(inplace=True),  # depthwise
            nn.Conv2d(hid, cout, 1, bias=False), nn.BatchNorm2d(cout))                                               # 줄이기 (linear)
    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out
```
- **무엇을 하나**: V2 블록 — 늘리기 → depthwise → 줄이기(활성화 없음), 조건이 맞으면 지름길.
- **어떤 개념**: 개념 2.
- **출력**: `InvertedResidual(24, 24)` **8,832** · (1, 24, 16, 16) 그대로 / `InvertedResidual(16, 24, stride=2)` **5,136** · (1, 16, 32, 32) → (1, 24, 16, 16)

```python
def h_swish(x):
    return x * F.relu6(x + 3) / 6

x = torch.linspace(-6, 6, 241)
torch.allclose(h_swish(x), nn.Hardswish()(x))
```
- **무엇을 하나**: h-swish 를 직접 만들어 내장 Hardswish 와 비교한다.
- **어떤 개념**: 개념 3.
- **출력**: `True` — x = −3 에서 0, x = 3 에서 3, x = 1 에서 1 · 4/6 ≈ 0.667

```python
mnv3 = torchvision.models.mobilenet_v3_small(weights=None)
blocks = [b for b in mnv3.features if b.__class__.__name__ == "InvertedResidual"]
n_se = sum(any(m.__class__.__name__ == "SqueezeExcitation" for m in b.modules()) for b in blocks)
feat = mnv3.features(torch.zeros(1, 3, 224, 224))
```
- **무엇을 하나**: torchvision MobileNetV3-Small 의 블록 · SE 개수와 마지막 특징맵을 확인한다.
- **어떤 개념**: 개념 1 · 3.
- **출력**: InvertedResidual **11** · SE **9** · 마지막 특징맵 **(1, 576, 7, 7)**

#### 한 번에 정리
- V1 = DS Conv, V2 = inverted residual(늘렸다 줄이기) + linear bottleneck, V3 = NAS + SE + h-swish.
- inverted residual 은 넓은 곳에서 depthwise, 블록 사이는 얇게.
- h-swish = x · ReLU6(x + 3) / 6, SE = 채널 중요도 재조정.
- ⚠ 마지막 1×1 뒤에 ReLU 를 넣지 않는다.

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

#### 전이학습(transfer learning)
> ImageNet 으로 학습한 특징 추출기(features)는 가져오고, 분류기 마지막 층만 우리 클래스 수로 바꿔 학습
- **feature extraction** → features **동결(freeze)** — `Parameter.requires_grad` = False, 분류기만 학습 (빠르고 과적합 적음)
- **fine-tuning** → 동결 없이 전체를 작은 학습률로 (정확도 높음)
- **예** → 같은 1 epoch: feature extraction **0.6607** · fine-tuning **0.8273** · 처음부터 0.5470 (실행 예)
- **`copy.deepcopy()`** → 실험마다 모델을 통째로 복사해 서로 섞이지 않게

#### 분류기 교체 — 파라미터 숫자
> `classifier[3]` 의 Linear(1024 → 1000) 을 Linear(1024 → 10) 으로
- **전체** → 2,542,856 − 1,025,000 + 10,250 = **1,528,106**
- **classifier** → 576·1024 + 1024 + 10,250 = **601,098** (39%) — feature extraction 은 이것만 학습 · features 927,008
- ⚠ 동결 후 옵티마이저엔 `[p for p in model.parameters() if p.requires_grad]`
- ⚠ 동결한 층의 BN 도 `model.train()` 에선 running 통계가 바뀐다 — 엄격히 고정하려면 그 층만 eval()

#### 전처리는 사전학습 때와 같게
> **사전학습 가중치(weights)** `MobileNet_V3_Small_Weights.IMAGENET1K_V1` 은 ImageNet 정규화를 기대한다
- **`Weights.transforms()`** → 이 가중치가 기대하는 전처리 확인 — mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)
- **입력 키우기** → 32 × 32 그대로면 특징이 사라진다 → `Resize(96)`
- ⚠ 사전학습 모델에 CIFAR 정규화 상수를 쓰면 안 된다

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

**상세 학습**

#### 이 교시에서 답하는 질문
MobileNet 을 처음부터 학습하려면 많은 데이터와 시간이 필요하다. 그런데 ImageNet 128만 장으로 이미 학습된 가중치가 공개되어 있다. 이 모델의 앞쪽 층은 에지 · 질감 · 모양 같은 **어떤 사진에나 쓰이는 특징**을 이미 안다. 이것을 가져와 우리 문제에 쓰는 것이 **전이학습**이다.

#### 개념 1 · 특징 추출기는 가져오고, 분류기만 교체
![전이학습 — features 는 가져오고, classifier 의 마지막 Linear 만 10 클래스로 바꾼다](assets/fig/vb/d3-transfer.svg)

| 항목 | 계산 | 파라미터 |
|---|---|---|
| 원래 마지막 Linear(1024 → 1000) | 1024·1000 + 1000 | 1,025,000 |
| 새 마지막 Linear(1024 → 10) | 1024·10 + 10 | 10,250 |
| 전체 | 2,542,856 − 1,025,000 + 10,250 | **1,528,106** |
| classifier 전체 | 576·1024 + 1024 + 10,250 | 601,098 |
| features | | 927,008 |

#### 개념 2 · 두 가지 방법
| 방법 | 학습하는 부분 | 장점 | 언제 | 1 epoch (실행 예) |
|---|---|---|---|---|
| feature extraction | 분류기만 (features 동결) | 빠르고 과적합 적음 | 데이터가 아주 적음 · 원래와 비슷 | 0.6607 |
| fine-tuning | 전체 (작은 lr) | 정확도 높음 | 데이터가 어느 정도 있음 | **0.8273** |
| (비교) 처음부터 | 전체 · 무작위 시작 | | | 0.5470 |

**동결**은 `requires_grad = False` 로 기울기를 안 구하게 하는 것이다. CIFAR-10(32 × 32, 흐림)은 ImageNet 과 꽤 달라 앞쪽 특징도 조금 고쳐야 해서 fine-tuning 이 더 높다.

#### 개념 3 · 전처리는 사전학습 때와 같게
사전학습 모델은 학습 때 받던 입력을 기대한다 — RGB 0 ~ 1 을 ImageNet 평균 (0.485, 0.456, 0.406) · 표준편차 (0.229, 0.224, 0.225) 로 정규화. CIFAR 정규화 상수를 쓰면 안 된다. 또 32 × 32 그대로면 몇 번의 stride 2 를 거치며 특징이 사라지므로 96 × 96 으로 키워 넣는다.

#### 코드로 확인
```python
mnet = torchvision.models.mobilenet_v3_small(weights=W_MNV3S)     # ImageNet 사전학습
print(W_MNV3S.transforms())                                          # 기대하는 전처리
mnet.classifier[3] = nn.Linear(1024, 10)                             # 마지막 층만 교체
```
- **무엇을 하나**: 사전학습 가중치로 모델을 만들고 마지막 분류 층을 10 클래스로 바꾼다.
- **어떤 개념**: 개념 1 · 3.
- **출력**: 파라미터 **1,528,106** · features 927,008 · classifier 601,098 · 전처리의 mean · std = ImageNet 값

```python
fe = copy.deepcopy(mnet)
for p in fe.features.parameters():
    p.requires_grad = False                                          # 동결
opt = torch.optim.Adam([p for p in fe.parameters() if p.requires_grad], lr=1e-3)
hist_fe = fit(fe, plain_loader, test_mn_loader, 1, opt, label="feature extraction")
```
- **무엇을 하나**: 복사본의 features 를 동결하고 학습되는 파라미터만 옵티마이저에 넣어 1 epoch 학습한다.
- **어떤 개념**: 개념 2.
- **출력**: 학습되는 파라미터 **601,098** / 1,528,106 · test **0.6607** (실행 예) — 같은 방식의 fine-tuning 은 0.8273

#### 한 번에 정리
- 전이학습 = 사전학습 특징 추출기 + 새 분류기.
- feature extraction(동결, 분류기만) vs fine-tuning(전체, 작은 lr).
- 1 epoch: 0.6607 · 0.8273 · 처음부터 0.5470.
- ⚠ 전처리는 사전학습 때와 같게 — ImageNet 상수, 입력 96.

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

#### 적은 데이터에서 전이학습
> 클래스당 100장(총 1,000장, 1/50) — 처음부터 **0.3560** vs 사전학습 **0.6561** (10 epoch, 실행 예)
- **균형 잡힌 부분집합** → `np.where(targets == c)[0][:100]` 로 클래스마다 같은 개수 → `torch.utils.data.Subset()` 으로 골라 쓰기
- **`np.bincount()`** → 클래스별 개수 확인 — 모두 100 (한쪽이 많으면 모델이 그쪽으로 기운다)
- **가장 크게 오른 클래스** → frog +0.59 · bird +0.43 · dog +0.38 — ImageNet 에 비슷한 대상이 많을수록
- ⚠ 전체 평균만 보면 약한 클래스가 숨는다 — 클래스별 정확도도 본다

#### BN momentum 함정
> torchvision MobileNetV3 의 **BN momentum** 은 0.01 → step 이 적으면 running 통계가 초기값에서 못 벗어난다
- **예** → 1,000장 · 배치 64 = epoch 당 16 step → 160 step 후에도 (1 − 0.01)¹⁶⁰ ≈ **0.20** 이 초기값
- **증상** → train 정확도는 오르는데 eval() 에서 test **0.10**(찍기)
- **해결** → `set_bn_momentum(m, 0.1)` — (1 − 0.1)¹⁶⁰ ≈ 0
- ⚠ train 은 오르는데 test 만 찍기 수준이면 BN · eval() 을 먼저 의심

#### 내 데이터 — ImageFolder
> **ImageFolder** 는 "폴더 이름 = 클래스" 구조에서 데이터셋을 만든다 — 나머지 코드는 그대로
- **`datasets.ImageFolder()`** → `ImageFolder("safety/train", transform=tf)` — train/helmet · train/no_helmet · train/no_person
- **레이블** → `train_set.classes` 의 순서 (폴더 이름 순)
- **분류기** → `nn.Linear(1024, len(train_set.classes))`

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

**상세 학습**

#### 이 교시에서 답하는 질문
현장에서는 클래스당 수천 장을 모으기 어렵다 (예: 안전모 착용 · 미착용 사진). 전이학습의 진짜 힘은 **데이터가 적을 때** 나온다. 이 교시는 CIFAR-10 의 1/50 만 써서 확인하고, 그 과정에서 만나는 BN 함정과 내 데이터를 넣는 방법을 본다.

#### 개념 1 · 클래스당 100장 실험
클래스마다 앞 100장씩, 총 1,000장(전체의 1/50)으로 10 epoch 학습한다 (실행 예).

| 모델 | test |
|---|---|
| 처음부터 | 0.3560 |
| 사전학습 + 전체 학습 | **0.6561** |

가장 크게 오른 클래스는 frog (+0.59) · bird (+0.43) · dog (+0.38) — ImageNet 에 비슷한 대상이 많을수록 사전학습 특징이 바로 쓰인다. 데이터를 모을 때는 클래스별 개수(균형)부터, 평가할 때는 클래스별 정확도를 같이 본다.

#### 개념 2 · BN momentum 함정
torchvision MobileNetV3 의 BN 은 momentum 0.01 — running 평균 · 분산이 매 step 새 배치 쪽으로 1% 만 움직인다.

> 초기값이 남아 있는 비율 = (1 − momentum)ᵗ

![BN running 통계가 초기값을 벗어나는 속도 — momentum 0.01 은 160 step 후에도 20% 가 초기값](assets/fig/vb/d3-bnmomentum.svg)

1,000장 · 배치 64 = epoch 당 16 step → 10 epoch 160 step. momentum 0.01 이면 (0.99)¹⁶⁰ ≈ 0.20 이 아직 초기값이다. 그러면 `model.train()` 에서는 배치 통계로 잘 학습되는데, `model.eval()` 에서는 엉뚱한 running 통계로 정규화되어 test 가 0.10(찍기)으로 나온다. 해결: momentum 을 0.1 로 (`set_bn_momentum`).

#### 개념 3 · 내 데이터 — ImageFolder
클래스 이름으로 폴더를 나눠 두면 `datasets.ImageFolder` 가 데이터셋을 만든다. 나머지(전처리 · DataLoader · 학습)는 그대로다.

| 경로 | 레이블 |
|---|---|
| safety/train/helmet/*.jpg | 0 |
| safety/train/no_helmet/*.jpg | 1 |
| safety/train/no_person/*.jpg | 2 |

레이블 번호는 `train_set.classes` 순서(폴더 이름 순)이고, 분류기는 `nn.Linear(1024, len(train_set.classes))`.

#### 코드로 확인
```python
targets = np.array(train_mn.targets)
idx_small = np.concatenate([np.where(targets == c)[0][:100] for c in range(10)])
small_loader = DataLoader(Subset(train_mn, idx_small), batch_size=64, shuffle=True, num_workers=NW)
print(len(idx_small), np.bincount(targets[idx_small]))
```
- **무엇을 하나**: 클래스마다 앞 100장의 인덱스를 모아 그 부분만 쓰는 로더를 만든다.
- **어떤 개념**: 개념 1 — 균형 잡힌 부분집합.
- **출력**: `1000 [100 100 100 100 100 100 100 100 100 100]` · `len(small_loader)` = **16**

```python
for name, w in [("scratch", None), ("pretrained", W_MNV3S)]:
    torch.manual_seed(0)
    m = torchvision.models.mobilenet_v3_small(weights=w)
    m.classifier[3] = nn.Linear(1024, 10)
    m = set_bn_momentum(m).to(device)            # 두 모델 모두 BN momentum 0.1
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    small_res[name] = (m, fit(m, small_loader, test_mn_loader, 10, opt, label=name))
```
- **무엇을 하나**: 같은 조건(1,000장 · 10 epoch · BN momentum 0.1)으로 처음부터 vs 사전학습을 비교한다.
- **어떤 개념**: 개념 1 · 2.
- **출력**: 처음부터 **0.3560** · 사전학습 **0.6561** (실행 예)

#### 한 번에 정리
- 데이터가 1/50 이어도 사전학습은 0.66, 처음부터는 0.36.
- 클래스별 개수(bincount)와 클래스별 정확도를 함께 본다.
- BN momentum 0.01 은 적은 step 에서 running 통계가 못 따라온다 → 0.1.
- ImageFolder: 폴더 이름 = 클래스.
- ⚠ train 은 오르는데 test 만 찍기 수준이면 BN · eval() 부터 의심.

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

#### Grad-CAM
> 클래스 점수 y^c 를 마지막 합성곱 특징맵 A 로 미분해, 모델이 이미지의 **어느 위치**를 보고 판단했는지 히트맵으로 — CAM = ReLU(Σ α_k A^k)
- **채널 중요도 α_k** → ∂y^c/∂A^k 의 공간 평균 — `Tensor.mean()` (dim=(2, 3))
- **기울기 얻기** → forward hook 으로 A 저장 + `Tensor.register_hook()` 으로 역전파 때 A 의 기울기를 가로챔
- **ReLU** → 그 클래스 점수를 **올리는** 위치만
- **정규화(0~1)** → `cam / (Tensor.amax(dim=(1, 2)) + 1e-8)` → `F.interpolate()` (bilinear) 로 입력 크기로 키워 겹침
- ⚠ 쓰고 나면 `cam.remove()` — hook 이 중복 등록되지 않게

#### CAM 해상도 = 마지막 특징맵 크기
> **CAM(Class Activation Map)** 은 마지막 특징맵 위의 지도 — 그보다 정밀할 수 없다
- **예** → ResNet-20 (32 입력) `layer3` (64, 8, 8) → **8 × 8** · MobileNetV3-S (96 입력) `features[-1]` → **3 × 3**
- **경량 모델** → 빨리 줄여서 연산을 아끼므로 CAM 이 거칠다
- ⚠ 정밀한 위치가 필요하면 분류가 아니라 객체 탐지 (Day 4)

#### 오답에서 무엇에 속았나
> 예측 클래스 CAM 과 정답 클래스 CAM(`cls=` 지정)을 나란히 비교
- **배경 · 가장자리에 몰림** → 물체가 아니라 맥락에 속은 것 → 다양한 배경의 데이터를 더한다
- **현장 안전** → 안전모가 아니라 공사장 배경을 보고 맞힌다면 위험 — 정확도와 함께 CAM 으로 확인

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

**상세 학습**

#### 이 교시에서 답하는 질문
정확도가 높아도 모델이 엉뚱한 곳을 보고 맞힐 수 있다 — 예를 들어 안전모가 아니라 "공사장 배경" 을 보고 "착용" 이라고 한다면 현장에서 위험하다. 이 교시는 분류 모델이 이미지의 **어디를 보고** 판단했는지 히트맵으로 그리는 **Grad-CAM** 을 만든다.

#### 개념 1 · Grad-CAM 계산
![Grad-CAM — 마지막 합성곱 특징맵 A 와 클래스 점수의 기울기로 위치별 근거를 구한다](assets/fig/vb/d3-gradcam.svg)

> α_k = mean_{h, w}(∂y^c / ∂A^k),  CAM = ReLU(Σ_k α_k · A^k)

| 기호 | 뜻 |
|---|---|
| A^k | 마지막 합성곱 출력의 k 번째 채널 (h × w 지도) |
| y^c | 클래스 c 의 점수 (softmax 전) |
| α_k | 채널 k 가 클래스 c 에 얼마나 중요한가 — 기울기의 공간 평균 |
| ReLU | 점수를 **올리는** 위치만 남김 |

마지막으로 최댓값으로 나눠 0 ~ 1 로 맞추고, bilinear 보간으로 입력 크기까지 키워 원본 위에 겹친다.

#### 개념 2 · CAM 해상도와 오답 분석
CAM 은 마지막 특징맵 위의 지도라 그보다 정밀할 수 없다.

| 모델 | 마지막 합성곱 | CAM |
|---|---|---|
| ResNet-20 (32 입력) | layer3 (64, 8, 8) | 8 × 8 |
| MobileNetV3-S (96 입력) | features[-1], stride 32 | 3 × 3 |

경량 모델은 빨리 줄여서 연산을 아끼므로 CAM 이 거칠다. 오답에서는 **예측 클래스** CAM 과 **정답 클래스** CAM(`cls=` 지정)을 나란히 본다. 예측 CAM 이 물체가 아니라 배경 · 가장자리에 몰려 있으면 맥락에 속은 것이다 → 다양한 배경의 데이터를 더한다.

#### 코드로 확인
```python
class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        self.handle = layer.register_forward_hook(self._save)
    def _save(self, module, inp, out):
        self.acts = out
        if out.requires_grad:                       # no_grad 안에서는 건너뜀
            out.register_hook(self._save_grad)
    def _save_grad(self, g):
        self.grads = g
```
- **무엇을 하나**: 지정한 층이 실행될 때 출력 A 를 저장하고, 역전파 때 A 의 기울기를 가로채 저장하도록 hook 두 개를 건다.
- **어떤 개념**: 개념 1 — forward hook 은 값, tensor hook 은 기울기.
- **출력**: 없음 (준비 단계)

```python
def __call__(self, x, cls=None):
    self.model.eval()
    logits = self.model(x)
    if cls is None:
        cls = logits.argmax(1)
    self.model.zero_grad()
    logits[torch.arange(len(x)), cls].sum().backward()          # 클래스 c 점수만
    alpha = self.grads.mean(dim=(2, 3), keepdim=True)           # (N, K, 1, 1)
    cam = F.relu((alpha * self.acts).sum(dim=1))                 # (N, H, W)
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.detach(), logits.argmax(1).detach()
```
- **무엇을 하나**: 클래스 점수만 역전파해 α 를 구하고, 가중합 → ReLU → 0 ~ 1 정규화.
- **어떤 개념**: 개념 1.
- **출력**: `GradCAM(r20, r20.layer3)` 에 16장 → CAM **(16, 8, 8)** · `GradCAM(ft, ft.features[-1])` 에 8장 → **(8, 3, 3)**

#### 한 번에 정리
- Grad-CAM: α_k = 기울기의 공간 평균, CAM = ReLU(Σ α_k A^k) → 0 ~ 1 → 키워서 겹침.
- 해상도는 마지막 특징맵 크기 — ResNet-20 8 × 8, MobileNetV3 3 × 3.
- 오답은 예측 CAM vs 정답 CAM 비교.
- ⚠ 쓰고 나면 `remove()` — hook 이 남지 않게.

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

#### 미니 프로젝트 — MobileNetV3 전이학습 vs ResNet-20
> 사전학습 MobileNetV3-S 를 CIFAR-10 전체로 5 epoch fine-tuning → test **0.8842** (ResNet-20 처음부터 20 epoch 0.8587)
- **`torch.optim.AdamW()`** → **AdamW** = Adam 에 weight decay 를 분리해 적용 (lr 1e-3, wd 1e-4)
- **`OneCycleLR()`** → max_lr 1e-3 · `steps_per_epoch = len(loader)` = **391**
- **입력** → 증강 + Resize 96 + ImageNet 정규화 · 학습 후 `torch.save()` 로 state_dict 저장
- ⚠ 과제 4 의 오답 8장 Grad-CAM 은 (8, 3, 3)

#### 비교표 — 정확도 · 파라미터 · MAC · 지연시간
> **비교표** 를 같은 기준으로: 파라미터 5.6배인데 MAC 은 1/3.7, 그런데 CPU 지연시간은 MobileNet 이 더 길다
- **ResNet-20** → 272,474 · 40,813,184 · **2.87 ms** · 0.8587
- **MobileNetV3-S** → 1,528,106 · 11,057,600 · 6.18 ms · **0.8842** (실행 예)
- **왜 더 느린가** → 입력 96 × 96 (9배) · depthwise · SE · 층 수 → 메모리 이동과 호출 오버헤드
- **다음** → ONNX → TensorRT · int8 양자화 · 가지치기, 위치까지 필요하면 객체 탐지 (Day 4)
- ⚠ 모델 선택은 정확도 · 파라미터 · MAC · **실제 지연시간**을 함께

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

**상세 학습**

#### 이 교시에서 답하는 질문
오늘 배운 경량 모델 + 전이학습으로 Day 2 의 ResNet-20 (처음부터 20 epoch, 0.8587) 을 이길 수 있을까? 그리고 정확도가 높다면 그 모델을 장치에 올려야 할까? 정확도 · 파라미터 · MAC · 지연시간을 한 표에 놓고 고른다.

#### 개념 1 · 설정
| 항목 | 값 |
|---|---|
| 모델 | 사전학습 MobileNetV3-S, `classifier[3] = Linear(1024, 10)` |
| 입력 | 증강 + Resize 96 + ImageNet 정규화, 배치 128 |
| 옵티마이저 | AdamW(lr 1e-3, weight_decay 1e-4) — Adam 에 weight decay 를 분리해 적용 |
| 스케줄러 | OneCycleLR(max_lr 1e-3, 5 epoch, steps_per_epoch = 391) |
| 학습 | 전체 fine-tuning 5 epoch → 목표 test ≥ 0.85 |

#### 개념 2 · 비교표 읽기
![ResNet-20 vs MobileNetV3-S — 정확도는 MobileNet, CPU 지연시간은 ResNet-20](assets/fig/vb/d3-compare.svg)

| 모델 | 파라미터 | MAC | 지연시간 | test |
|---|---|---|---|---|
| ResNet-20 (처음부터, 32) | 272,474 | 40,813,184 | **2.87 ms** | 0.8587 |
| MobileNetV3-S (전이, 96) | 1,528,106 | 11,057,600 | 6.18 ms | **0.8842** |

- **파라미터 5.6배인데 MAC 은 1/3.7** — MobileNet 의 파라미터는 위치마다 한 번 쓰이는 1×1 · FC 에 많고 stride 2 를 일찍 써서 큰 특징맵 연산이 적다. ResNet-20 은 32 × 32 를 유지한 채 3×3 을 반복해 파라미터는 적어도 MAC 이 크다.
- **MAC 이 적은데 더 느리다** — 입력이 96 × 96(9배), depthwise · SE · 층 수가 많아 메모리 이동과 호출 오버헤드가 크다 (3교시와 같은 이야기).
- 결론: 모델은 정확도 · 파라미터 · MAC · **실제 지연시간**을 함께 보고, 실제 장치에서 재서 고른다. 장치에서는 ONNX → TensorRT · int8 양자화가 더 필요하다.

#### 코드로 확인
```python
model = torchvision.models.mobilenet_v3_small(weights=W_MNV3S)
model.classifier[3] = nn.Linear(1024, 10)
model = model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(mn_loader))
hist_mn = fit(model, mn_loader, test_mn_loader, EPOCHS, opt, sched, label="MobileNetV3-S")
torch.save(model.state_dict(), "mnv3s_cifar10.pt")
```
- **무엇을 하나**: 사전학습 모델을 CIFAR-10 전체로 5 epoch fine-tuning 하고 저장한다.
- **어떤 개념**: 개념 1 · 5교시 fine-tuning.
- **출력**: `len(mn_loader)` = **391** · test **0.8842** (실행 예)

```python
compare = {
    "ResNet-20 (scratch, 32)": (count_params(r20), count_macs(r20, (1, 3, 32, 32)), latency_ms(r20, (1, 3, 32, 32)), acc_r20),
    "MobileNetV3-S (transfer, 96)": (count_params(model), count_macs(model, (1, 3, IMG, IMG)),
                                     latency_ms(model, (1, 3, IMG, IMG)), mn_acc),
}
```
- **무엇을 하나**: 두 모델을 각자의 입력 크기(32 · 96)로 같은 기준의 표에 담는다.
- **어떤 개념**: 개념 2.
- **출력**: MAC — ResNet-20 **40,813,184** · MobileNetV3-S **11,057,600** (채점 기준값) · 지연시간 2.87 vs 6.18 ms (실행 예)

#### 한 번에 정리
- 사전학습 MobileNetV3-S 5 epoch fine-tuning (AdamW + OneCycle) → 0.8842 > ResNet-20 0.8587.
- 파라미터 5.6배 · MAC 1/3.7 · 그런데 CPU 지연시간은 2배 이상.
- 파라미터 · MAC · 지연시간은 서로 다른 숫자 — 함께 본다.
- ⚠ 장치에 올릴 모델은 그 장치에서 직접 재서 고른다.

**연관 문제**
<<QA 8>>
