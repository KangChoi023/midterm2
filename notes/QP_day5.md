## Day 5 — Pruning: 가지치기와 실제 속도

### 1교시 · Pruning 개념: 무엇을, 어떤 단위로 자르나

**키워드**: `pruning(가지치기)`, `criterion · magnitude`, `granularity · 단위`, `unstructured · semi-structured · structured`, `one-shot vs iterative`, `fine-tuning`, `mask`, `filter L1-norm`, `weight 분포`

**개요**: Day 1 ~ 4 는 숫자를 **작게**(32 → 8 · 3비트) 했고, 오늘은 숫자를 **적게** — 필요 없는 weight · 채널을 잘라 낸다(**Pruning**). 가지치기는 네 가지 질문의 조합: **무엇을**(기준 — magnitude |w| · gradient w × ∂L/∂w) × **어떤 단위로**(weight 1개 · N:M · filter · layer) × **어떻게**(one-shot · iterative) × **자른 뒤**(그대로 · fine-tuning). 근거는 weight 분포 — ResNet-20 conv weight 270,256개 중 |w| < 0.01 이 **21%**, < 0.05 가 **76%** 로 0 근처에 몰려 있다. 오늘의 결론을 먼저: **실무에서 속도를 얻는 것은 structured pruning** — 단위가 클수록 정확도 손해는 크지만 텐서가 진짜로 작아진다.

**핵심 용어**
- **Pruning(가지치기)**: 중요하지 않은 weight · 채널 · layer 를 0 으로 만들거나 빼서 모델을 줄이는 기법
- **criterion(기준)**: 무엇을 지울지 고르는 점수 — magnitude(|w| 가 작은 것) · gradient(w × ∂L/∂w, 지우면 손실이 얼마나 변하나)
- **granularity(단위)**: 한 번에 지우는 묶음의 크기 — weight 1개 · N:M · filter(출력 채널) · 입력 채널 · layer
- **unstructured pruning**: weight 를 한 개씩 0 으로 — 모양은 그대로, 0 만 생긴다
- **structured pruning**: filter · channel 을 통째로 빼서 텐서 모양 자체를 줄인다
- **mask**: weight 와 같은 모양의 0/1 텐서 — 남길 곳 1, 지울 곳 0
- **filter L1-norm**: 출력 채널 하나의 weight 절댓값 합 Σ|w| — structured 의 magnitude 기준

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `magnitude_mask()` | `magnitude_mask(w, sparsity)` | \|w\| 가 작은 sparsity 비율을 0 으로 하는 mask (이 교시에서 만듦) |
| `filter_l1()` | `filter_l1(conv)` | Conv 의 filter(출력 채널)별 L1-norm (이 교시에서 만듦) |
| `kthvalue()` | `w.abs().flatten().kthvalue(k).values` | k 번째로 작은 값 — 임계값 |
| `convs()` | `convs(model)` | Conv2d · Linear 의 (이름, 모듈) 목록 — 가지치기 대상 (제공) |

**핵심 내용**

#### Pruning 의 네 가지 질문
> 가지치기 = 기준(무엇을) × 단위(얼마나 크게) × 방식(한 번 · 여러 번) × 회복(fine-tuning) — 오늘은 magnitude(L1) 기준
- **기준** → magnitude(|w| 작은 것) · gradient(w × ∂L/∂w) · 학습된 중요도 — 오늘은 magnitude
- **방식** → one-shot(한 번에) · iterative(조금씩 자르고 회복 반복) — 6교시에서 비교
- **회복** → 자른 뒤 짧게 다시 학습(fine-tuning) — structured 는 필수
- ⚠ magnitude 는 "값이 작다" 만 볼 뿐 "중요하지 않다" 를 보장하지 않는다 — 그래서 gradient 기준도 쓰인다

#### 단위(granularity) — 정확도 vs 실제 속도
> 단위가 작을수록 정확도는 지키지만 속도는 안 나고, 클수록 정확도 손해는 크지만 어디서나 빨라진다
- **unstructured**(weight 1개) → 정확도 영향 가장 작음 · 일반 커널은 **0 도 곱한다** → 속도 그대로
- **semi-structured**(2:4) → 중간 · Ampere 이후 Sparse Tensor Core 에서만 최대 약 2배
- **structured**(filter · channel) · depth(layer · block) → 정확도 손해 큼 → fine-tuning 필요 · 텐서가 작아져 어디서나 빨라짐
- **예** → layer2[0].conv1 (32 × 16 × 3 × 3) 에서 하나 지우면 사라지는 weight: weight **1** · filter **144** · 입력 채널 **288** · layer **4,608**

#### weight 분포 — magnitude 기준의 근거
> 학습된 weight 는 0 근처에 몰린 종 모양 — 작은 것부터 지우면 출력이 거의 안 변한다
- **분포** → conv weight 270,256개 중 \|w\| < 0.01 **21%** · < 0.02 **40%** · < 0.05 **76%**
- **magnitude_mask** → k = round(sparsity × 원소 수) 번째로 작은 \|w\| 가 임계값, 그 이하를 0
- **filter_l1** → layer1[0].conv1 의 16개 filter L1 = 최소 **7.45** ~ 최대 **11.37**
- ⚠ k = 0 이면 `kthvalue(0)` 이 오류 → 먼저 `torch.ones_like(w)` 로 돌려준다

**기본 사용형태**
```python
def magnitude_mask(w, sparsity):
    k = int(round(sparsity * w.numel()))
    if k == 0:
        return torch.ones_like(w)
    thr = w.detach().abs().flatten().kthvalue(k).values   # k 번째로 작은 |w|
    return (w.detach().abs() > thr).float()               # 임계값 이하는 0

def filter_l1(conv):
    return conv.weight.detach().abs().sum(dim=(1, 2, 3))  # (out,) — 출력 채널마다 Σ|w|
```

**연결**
- [QP D1·1교시 왜 경량화인가](#/qp/1/1) — 크기 = 파라미터 수 × byte, 양자화는 byte 를 · 가지치기는 파라미터 수를 줄인다
- [NN D4·6교시 규제: weight decay · Dropout](#/nn/4/6) — weight decay 가 weight 를 0 쪽으로 당겨 분포가 0 근처에 몰린다
- [VB D3·2교시 연산량 계산](#/vb/3/2) — 파라미터 · MAC 를 세는 법, 단위별로 줄어드는 양

**상세 학습**

#### 이 교시에서 답하는 질문
양자화(Day 1 ~ 4)는 숫자 하나의 byte 를 줄였다. 그런데 모델에는 "있으나 마나 한" weight 도 많지 않을까? 이 교시는 학습된 ResNet-20 의 weight 분포를 재서 그 근거를 확인하고, 가지치기를 고르는 네 가지 질문(기준 · 단위 · 방식 · 회복)과 단위별 장단점을 정리한다.

#### 개념 1 · weight 분포 — 왜 작은 것부터 지우나
- 비유: 회의에서 거의 발언하지 않는 사람을 빼도 결론이 크게 안 바뀐다 — |w| 가 작으면 그 입력이 출력에 주는 영향 w·x 도 작다.
- 실제로 재 보면 0 근처에 몰려 있다 (로그 스케일 히스토그램).

![ResNet-20 conv weight |w| 히스토그램 — 0.01 · 0.02 · 0.05 아래가 21% · 40% · 76%](assets/fig/qp/d5-hist.svg)

| 임계값 t | \|w\| < t 비율 |
|---|---|
| 0.01 | 0.2112 |
| 0.02 | 0.4002 |
| 0.05 | 0.7619 |

#### 개념 2 · 단위 — 하나 지울 때 사라지는 양
![가지치기 단위 — weight 1개 · filter 144개 · 입력 채널 288개 · layer 4,608개, 단위가 클수록 실제 속도 이득](assets/fig/qp/d5-units.svg)

| 단위 | 이름 | layer2[0].conv1 에서 하나 지우면 | 정확도 영향 | 실제 속도 |
|---|---|---|---|---|
| weight 1개 | unstructured | 1 | 가장 작음 | 그대로 (0 도 곱함) |
| N:M (2:4) | semi-structured | 4개 중 2개 | 중간 | Ampere+ 전용 하드웨어만 |
| filter (출력 채널) | structured | 16 × 3 × 3 = 144 | 큼 | 어디서나 빨라짐 |
| 입력 채널 | structured | 32 × 3 × 3 = 288 | 큼 | 어디서나 빨라짐 |
| layer | depth | 4,608 | 매우 큼 | 빨라짐 |

- filter L1-norm 은 "filter 하나 = weight 144개" 의 \|w\| 합 — 4교시 structured pruning 에서 지울 filter 를 고르는 점수.

#### 코드로 확인
```python
w_all = torch.cat([m.weight.detach().abs().flatten()
                   for n, m in convs(rn20) if isinstance(m, nn.Conv2d)])
small = {t: float((w_all < t).float().mean()) for t in [0.01, 0.02, 0.05]}
plt.hist(w_all.numpy(), bins=100, log=True)
```
- **무엇을 하나**: 모든 Conv2d weight 의 \|w\| 를 한 줄로 이어 붙여 작은 값 비율을 센다 (예제 1-A).
- **어떤 개념**: 개념 1.
- **출력**: `conv weight 270,256개 | {0.01: 0.211, 0.02: 0.4, 0.05: 0.762}` (직접 실행)

```python
_m = magnitude_mask(torch.tensor([0.1, -0.5, 0.3, 0.05, -0.2, 0.4]), 0.5)
w2 = rn20.layer2[0].conv1.weight                         # (32, 16, 3, 3)
units_df = pd.DataFrame({"unit": ["weight", "filter", "in_channel", "layer"],
                         "removed": [1, w2[0].numel(), w2[:, 0].numel(), w2.numel()]})
```
- **무엇을 하나**: 6개 중 작은 절반을 지우는 mask 를 확인하고, 단위별로 사라지는 weight 수를 센다 (실습 1-1 · 1-2).
- **어떤 개념**: 개념 2.
- **출력**: `[0, 1, 1, 0, 0, 1]` · removed `[1, 144, 288, 4608]` (직접 실행)

#### 한 번에 정리
- 가지치기 = 기준 × 단위 × 방식 × 회복 — 오늘은 magnitude(L1) 기준.
- 학습된 weight 는 0 근처에 몰려 있다(\|w\| < 0.05 가 76%) → magnitude 기준의 근거.
- 단위가 클수록 정확도 손해 ↑, 실제 속도 이득 ↑ — 속도는 structured 에서.
- ⚠ 작은 \|w\| = 중요하지 않음은 아니다 — 입력이 큰 곳의 작은 weight 는 중요할 수 있다.

---

### 2교시 · Unstructured pruning: mask · layer 별 vs global

**키워드**: `torch.nn.utils.prune`, `l1_unstructured`, `global_unstructured`, `weight_orig × weight_mask`, `forward pre-hook`, `prune.remove`, `sparsity`, `layer 별 vs global`

**개요**: `torch.nn.utils.prune` 은 weight 를 직접 지우지 않고 **mask 를 곱하는** 방식이다 — 적용하면 모듈 안의 `weight` 가 `weight_orig`(Parameter, 원래 값) + `weight_mask`(buffer, 0/1)로 바뀌고, forward 직전마다 hook 이 `weight = weight_orig × weight_mask` 를 계산한다. `prune.remove` 로 영구 반영하면 다시 `weight` 하나만 남는다(0 은 그대로). **layer 별**(층마다 똑같이 50%)은 weight 가 적고 중요한 층(stem · fc)도 50% 를 잃는다. **global**(전체에서 작은 50%)은 층마다 비율이 달라져 fc **5.8%** · stem **12.5%** 만 지우고 마지막 블록 conv2 는 **80%** 를 지운다. fine-tuning 없이: 30% 0.8415 / 0.838 · 50% **0.8125 / 0.789** · 70% 0.482 / 0.2945 · 90% 0.10 (global / layer 별) — 작은 ResNet-20 은 70% 부터 무너진다.

**핵심 용어**
- **sparsity(희소도)**: 전체 weight 중 0 의 비율 — 0.9 = 90% 가 0
- **weight_orig · weight_mask**: prune 적용 뒤 모듈이 갖는 원래 weight(Parameter)와 0/1 mask(buffer)
- **layer 별(layer-wise) pruning**: 층마다 같은 비율을 지움
- **global pruning**: 모든 층의 weight 를 한데 모아 작은 것부터 지움 — 층마다 비율이 달라짐
- **prune.remove**: mask 를 weight 에 곱해 영구 반영하고 weight_orig · weight_mask · hook 을 없앰

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `prune.l1_unstructured()` | `prune.l1_unstructured(module, "weight", amount=0.5)` | 그 층 안에서 \|w\| 작은 50% 를 mask 0 |
| `prune.global_unstructured()` | `prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=0.5)` | 모든 층을 모아 작은 50% |
| `prune.remove()` | `prune.remove(module, "weight")` | mask 를 weight 에 영구 반영 |
| `named_buffers()` | `sorted(n for n, _ in conv.named_buffers())` | 학습하지 않는 텐서(buffer) 이름 — `weight_mask` 가 여기 |
| `sparsity()` | `sparsity(model)` | convs() 전체 weight 중 0 의 비율 (이 교시에서 만듦) |
| `prune_unstructured()` | `prune_unstructured(model, amount, scope="global")` | 복사본에 global · layer 별 pruning + remove (이 교시에서 만듦) |

**핵심 내용**

#### prune 의 내부 — mask 곱하기
> 적용 = weight → weight_orig × weight_mask (forward 전에 hook 이 매번 곱함) / remove = 결과를 weight 로 굳힘
- **적용 직후** → parameters `['weight_orig']` · buffers `['weight_mask']` · 0 비율 **0.500**
- **prune.remove 뒤** → parameters `['weight']` · 0 비율 0.500 그대로 (모양도 그대로)
- **왜 mask 로** → 학습을 계속해도 지운 자리는 0 으로 유지 · 되돌리기 쉬움
- ⚠ remove 전에 `torch.save` / ONNX 로 내보내면 weight_orig · weight_mask 가 따로 저장된다 → 배포 전에 꼭 remove

#### layer 별 vs global
> layer 별 = 층마다 같은 비율 / global = 전체에서 작은 것 — 같은 50% 라도 global 이 덜 망가진다
- **global 50% 의 층별 sparsity** → fc **0.058** · stem **0.125** · shortcut 0.135 … layer3.2.conv2 **0.801**
- **정확도** (fine-tuning 없음) → 50%: global **0.8125** vs layer 별 **0.789**
- **이유** → weight 가 적고 중요한 층(stem · fc)은 \|w\| 가 커서 global 에서는 덜 지워진다
- ⚠ global 은 한 층을 거의 다 지울 수도 있다 — 층이 통째로 사라지면 모델이 끊긴다 (대상 · 비율을 확인)

#### sparsity 와 정확도
> fine-tuning 없는 unstructured — ResNet-20 은 30 ~ 50% 까지 버티고 70% 부터 무너진다
- **30%** → 0.8415 / 0.838 (거의 공짜)
- **70%** → 0.482 / 0.2945 · **90%** → 0.10 / 0.10 (무작위 = 0.1)
- **큰 모델** → ResNet-50 · BERT 는 fine-tuning 과 함께 80 ~ 90% 까지 버티기도 한다 (교안 설명)
- ⚠ 작은 모델(27만 params)은 여유가 적다 — 같은 비율이라도 모델 크기에 따라 결과가 다르다

**기본 사용형태**
```python
def sparsity(model):
    z = t = 0
    for n, m in convs(model):
        z += (m.weight == 0).sum().item(); t += m.weight.numel()
    return z / t

def prune_unstructured(model, amount, scope="global"):
    m = copy.deepcopy(model)                                  # 원본은 그대로
    targets = [(mod, "weight") for n, mod in convs(m)]
    if scope == "global":
        prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)
    else:
        for mod, name in targets:
            prune.l1_unstructured(mod, name, amount=amount)
    for mod, name in targets:
        prune.remove(mod, name)                               # mask 영구 반영
    return m
```

**연결**
- [QP D3·3교시 Mixed precision](#/qp/3/3) — 민감한 층(stem · fc)은 덜 건드린다는 같은 생각 — global 이 자동으로 한다
- [NN D3·5교시 Affine · Softmax-with-Loss 계층](#/nn/3/5) — y = xW + b 에서 w = 0 이면 그 입력은 출력에 기여하지 않는다
- [VB D2·6교시 ResNet: Residual block](#/vb/2/6) — 오늘 자르는 ResNet-20 의 구조

**상세 학습**

#### 이 교시에서 답하는 질문
1교시에서 "작은 weight 를 지운다" 는 생각을 세웠다. 이 교시는 PyTorch 의 `torch.nn.utils.prune` 으로 실제로 지워 보고, (1) 지운다는 것이 모듈 안에서 무엇으로 바뀌는지, (2) 층마다 같은 비율로 지우는 것과 전체에서 지우는 것이 어떻게 다른지, (3) 몇 % 까지 지워도 되는지를 숫자로 확인한다.

#### 개념 1 · mask 방식
| 상태 | parameters | buffers | weight 는 |
|---|---|---|---|
| 원래 | weight | — | Parameter |
| `l1_unstructured` 직후 | weight_orig | weight_mask | 속성 — forward 전 hook 이 weight_orig × weight_mask 계산 |
| `prune.remove` 뒤 | weight | — | Parameter (0 이 50% 들어 있음) |

- buffer = 모델에 저장되지만 학습(gradient)은 안 하는 텐서 — BatchNorm 의 running_mean 과 같은 종류.

#### 개념 2 · layer 별 vs global
![같은 50% — global 은 fc 5.8% · stem 12.5% 만 지우고 마지막 블록은 80% 를 지운다 · sparsity 별 정확도](assets/fig/qp/d5-us.svg)

| sparsity | global | layer 별 |
|---|---|---|
| 0.0 | 0.846 | 0.846 |
| 0.3 | 0.8415 | 0.838 |
| 0.5 | 0.8125 | 0.789 |
| 0.7 | 0.482 | 0.2945 |
| 0.9 | 0.100 | 0.100 |

(fine-tuning 없음 · 평가 2,000장 · 직접 실행 — 교안 강사 실측과 같다.)

#### 코드로 확인
```python
m = copy.deepcopy(rn20); conv = m.layer1[0].conv1
prune.l1_unstructured(conv, "weight", amount=0.5)
params_before = sorted(n for n, _ in conv.named_parameters())
buffers = sorted(n for n, _ in conv.named_buffers())
sp_masked = float((conv.weight == 0).float().mean())
prune.remove(conv, "weight")
params_after = sorted(n for n, _ in conv.named_parameters())
```
- **무엇을 하나**: 한 층에 50% pruning 을 걸고 이름이 어떻게 바뀌는지 본 뒤 remove 한다 (실습 2-1).
- **어떤 개념**: 개념 1.
- **출력**: `적용 직후 parameters: ['weight_orig'] | buffers: ['weight_mask'] | 0 비율 0.500` · `prune.remove 뒤 parameters: ['weight'] | 0 비율 0.500` (직접 실행)

```python
g50 = prune_unstructured(rn20, 0.5, "global")
per_layer = pd.DataFrame([{"layer": n, "global_sp": float((m.weight == 0).float().mean())}
                          for n, m in convs(g50)])
per_layer.nsmallest(3, "global_sp")
```
- **무엇을 하나**: global 50% 에서 층마다 실제로 몇 % 가 지워졌는지 센다 (실습 2-2).
- **어떤 개념**: 개념 2.
- **출력**: 가장 적게 — `fc 0.058 · stem.0 0.125 · layer2.0.shortcut.0 0.135` · 가장 많이 `layer3.2.conv2 0.801` (직접 실행)

#### 한 번에 정리
- `prune.*` = weight_orig × weight_mask (hook) → `prune.remove` 로 굳힘 — 모양은 그대로, 0 만 생긴다.
- global 은 층마다 비율이 달라 중요한 층을 지킨다 → 같은 50% 에서 0.8125 vs 0.789.
- ResNet-20 은 fine-tuning 없이 30 ~ 50% 까지, 70% 부터 무너진다.
- ⚠ 배포 전에 `prune.remove` — 안 하면 weight_orig · weight_mask 두 벌이 저장된다.

---

### 3교시 · 왜 빨라지지 않나: 크기 · 지연 · 2:4 sparsity

**키워드**: `dense 텐서`, `저장 크기 vs 압축 크기`, `gzip`, `0 도 곱한다`, `sparse 커널 (CSR)`, `2:4 semi-structured`, `Sparse Tensor Core`, `Ampere+`, `SPARSE_WEIGHTS`

**개요**: 90% 를 지운 모델은 10배 작고 10배 빠를까? 재 보면 **둘 다 아니다**. (1) **저장 크기**(state_dict · .onnx)는 그대로 — 0 도 FP32 4 byte 로 저장되는 dense 텐서다(1,109.7 KB = 1,109.7 KB). (2) **압축 크기**(gzip)는 크게 준다 — 0 이 반복돼 1,003 → **192 KB(5.2배)**, 배포 · 다운로드 크기 이득. (3) **지연**은 그대로 — cuDNN · MKL 의 Conv 는 0 도 곱한다(강사 실측 dense 1.80 · 50% 1.81 · 90% 1.78 ms). 0 을 건너뛰는 sparse 커널은 sparsity 가 아주 높거나 행렬이 클 때만 이득이다. 예외 하나: **2:4 semi-structured** — 연속한 weight 4개마다 정확히 2개가 0 이면 Ampere 이후 GPU(A100 · RTX 30 · Jetson Orin)의 Sparse Tensor Core 가 최대 약 2배(T4 는 미지원). 대가: 규칙 때문에 같은 50% 라도 정확도 **0.6485**(unstructured global 0.8125).

**핵심 용어**
- **dense 텐서**: 0 을 포함한 모든 원소를 그대로 저장 · 계산하는 보통 텐서
- **sparse 커널(CSR SpMM)**: 0 이 아닌 값과 그 위치(인덱스)만 저장해 곱하는 계산 — 인덱스를 읽는 비용이 있다
- **2:4 semi-structured sparsity**: 연속한 4개마다 정확히 2개가 0 인 규칙적인 희소 패턴
- **Sparse Tensor Core**: NVIDIA Ampere(A100 · RTX 30 · Orin) 이후 GPU 의 2:4 전용 연산기 — 절반만 계산

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `state_kb()` | `state_kb(model)` | state_dict 를 메모리에 저장한 (원래 KB, gzip KB) (이 교시에서 만듦) |
| `io.BytesIO()` | `buf = io.BytesIO(); torch.save(sd, buf)` | 파일 대신 메모리에 저장 |
| `gzip.compress()` | `len(gzip.compress(raw)) / 1024` | byte 열 압축 — 압축 크기 |
| `latency_torch()` | `latency_torch(model)` | PyTorch CPU 배치 1 평균 지연 ms (제공) |
| `mask_2to4()` | `mask_2to4(w)` | 입력 채널 4개마다 큰 2개만 남기는 mask (이 교시에서 만듦) |
| `topk()` | `g.topk(2, dim=1).indices` | 행마다 큰 k 개의 위치 |
| `scatter_()` | `torch.zeros_like(g).scatter_(1, idx, 1.0)` | 지정 위치에 값 채우기 |

**핵심 내용**

#### 0 은 저장 크기 · 지연을 바꾸지 않는다
> unstructured 0 은 dense 텐서 안의 숫자일 뿐 — 모양이 그대로라 크기 · 계산량도 그대로
- **저장 크기** → dense **1,109.7 KB** = 90% sparse **1,109.7 KB** (0 도 4 byte)
- **압축 크기** → gzip **1,003 → 192 KB (5.2배)** — 반복되는 0 이 잘 압축됨 → 배포 · 다운로드 이득
- **지연** → 강사 실측 CPU dense 1.80 · 50% 1.81 · 90% 1.78 ms — 측정 잡음 수준으로 같다 (TensorRT 도 같음, 7교시)
- ⚠ sparse 커널은 sparsity 95% 이상이거나 큰 행렬에서만 이득 — 작은 Conv 에서는 인덱스 읽는 비용이 더 크다

#### 2:4 semi-structured sparsity
> 연속한 4개마다 큰 2개만 남김 — Ampere+ Sparse Tensor Core 가 절반만 계산해 최대 약 2배
- **예** → [0.3 −0.1 0.05 0.6 \| −0.4 0.02 0.2 −0.01] → mask [1 0 0 1 \| 1 0 1 0]
- **ResNet-20** → 입력 채널이 4의 배수인 Conv 전부(stem 제외) → sparsity **0.498** · 정확도 **0.6485** (fine-tuning 없음)
- **하드웨어** → A100 · RTX 30 · Jetson Orin + TensorRT `SPARSE_WEIGHTS` 플래그 · **T4(Turing) 미지원**
- ⚠ 같은 50% 라도 규칙이 있어 unstructured(0.8125)보다 정확도 손해가 크다 → 보통 fine-tuning 과 함께

**기본 사용형태**
```python
def state_kb(model):
    buf = io.BytesIO(); torch.save(model.state_dict(), buf); raw = buf.getvalue()
    return len(raw) / 1024, len(gzip.compress(raw)) / 1024        # (원래, 압축) KB

def mask_2to4(w):
    wp = w.detach().permute(0, 2, 3, 1)                 # 입력 채널을 마지막 축으로
    g = wp.abs().reshape(-1, 4)                         # 한 행 = 입력 채널 4개
    m = torch.zeros_like(g).scatter_(1, g.topk(2, dim=1).indices, 1.0)
    return m.reshape(wp.shape).permute(0, 3, 1, 2).contiguous()

with torch.no_grad():
    for n, c in m24.named_modules():
        if isinstance(c, nn.Conv2d) and c.in_channels % 4 == 0:   # stem(입력 3채널) 제외
            c.weight.mul_(mask_2to4(c.weight))
```

**연결**
- [QP D1·1교시 왜 경량화인가](#/qp/1/1) — 크기 · 메모리 · 속도는 따로 잰다 — 오늘 "크기 이득은 압축에서만" 이 그 예
- [QP D2·7교시 성능 측정](#/qp/2/7) — 같은 자(배치 1 평균 지연)로 재는 방법
- [OT D4·7교시 왜 빨라졌나: 엔진 레이어 들여다보기](#/ot/4/7) — 속도는 커널이 결정한다 — 0 을 건너뛰는 커널이 없으면 그대로

**상세 학습**

#### 이 교시에서 답하는 질문
2교시에서 90% 를 지워도 모델 모양은 그대로였다. 그럼 실제로 얻는 것이 있을까? 이 교시는 저장 크기 · 압축 크기 · 지연 세 가지를 재서 "unstructured 의 이득은 압축 크기뿐" 임을 확인하고, 하드웨어가 0 을 건너뛸 수 있는 유일한 예외인 2:4 sparsity 를 만들어 본다.

#### 개념 1 · 세 가지 크기 · 속도
![0 이 90% 여도 저장 크기 · 지연은 그대로, 압축 크기만 1/5 — 2:4 mask 는 4개마다 큰 2개](assets/fig/qp/d5-sparse.svg)

| 무엇 | dense | 90% sparse | 이유 |
|---|---|---|---|
| 저장 크기 (state_dict) | 1,109.7 KB | 1,109.7 KB | 0 도 FP32 4 byte |
| gzip 압축 크기 | 1,003.3 KB | 191.7 KB | 0 이 반복 → 압축 잘 됨 (5.2배) |
| 지연 (강사 실측 CPU) | 1.80 ms | 1.78 ms | 일반 Conv 는 0 도 곱함 |
| 지연 (이 실행 · CPU 1스레드) | 2.02 ms | 2.17 ms | 50% 는 1.95 ms — 잡음 수준 |

- dense 는 gzip 해도 1,109.7 → 1,003.3 KB 로 조금만 준다(실수의 bit 는 무작위에 가까움). 90% sparse 는 191.7 KB.

#### 개념 2 · 2:4 규칙
| 기호 | 뜻 |
|---|---|
| (out, in, kh, kw) | Conv weight 모양 — 입력 채널(in) 축으로 4개씩 묶는다 |
| permute(0, 2, 3, 1) | in 을 마지막 축으로 → reshape(−1, 4) 하면 한 행 = 연속한 입력 채널 4개 |
| topk(2) | 행마다 \|w\| 큰 2개 위치 → mask 1 |

- 하드웨어는 남은 2개 값 + 위치 정보(2bit)만 읽어 절반만 곱한다 → 이론상 2배, 실제로는 그보다 작다.

#### 코드로 확인
```python
m90 = prune_unstructured(rn20, 0.9, "global")
size_df = pd.DataFrame([{"model": n, "raw_kb": r, "gzip_kb": z}
                        for n, (r, z) in [("FP32", state_kb(rn20)), ("global 90%", state_kb(m90))]])
```
- **무엇을 하나**: 원본과 90% sparse 모델의 저장 · 압축 크기를 비교한다 (실습 3-1).
- **어떤 개념**: 개념 1.
- **출력**: `원래 크기는 같고 (1110 KB), 압축하면 1003 → 192 KB (5.2배)` (직접 실행)

```python
m24 = copy.deepcopy(rn20)
with torch.no_grad():
    for n, c in m24.named_modules():
        if isinstance(c, nn.Conv2d) and c.in_channels % 4 == 0:
            c.weight.mul_(mask_2to4(c.weight))
acc_24 = acc_torch(m24)
```
- **무엇을 하나**: stem 을 뺀 모든 Conv 에 2:4 mask 를 곱하고 정확도를 잰다 (실습 3-2).
- **어떤 개념**: 개념 2.
- **출력**: `2:4 적용 — sparsity 0.498 · 정확도 0.6485 (global 50% unstructured 0.8125)` (직접 실행)

#### 한 번에 정리
- unstructured 0 → 저장 크기 · 지연 그대로, 압축 크기만 준다 (192 KB, 5.2배).
- 2:4 semi-structured = Ampere+ Sparse Tensor Core 에서만 최대 2배 · T4 미지원 · 정확도 손해 더 큼.
- 어떤 하드웨어에서나 빨라지려면 텐서 **모양** 자체를 줄여야 한다 → structured (4교시).
- ⚠ "0 이 90%" 를 "10배 빠름" 으로 읽지 말 것 — 지연은 반드시 직접 잰다.

---

### 4교시 · Structured pruning 직접 하기: L1-norm filter 제거

**키워드**: `filter(출력 채널) 제거`, `conv → BN → conv 사슬`, `keep 인덱스`, `BN 4가지 값`, `블록 안쪽 vs 바깥쪽`, `residual 덧셈 제약`, `prune_block()`, `manual_prune()`

**개요**: filter 를 지우면 Conv 의 출력 채널이 C → C′ 로 줄고, 그 출력을 받는 **모든 층**도 함께 고쳐야 한다: conv1 `weight[keep]` → bn1 의 γ · β · running_mean · running_var `[keep]` → conv2 `weight[:, keep]`(입력 축). ResNet 의 BasicBlock 에서 **블록 안쪽**(conv1 출력 = conv2 입력)은 블록 밖으로 안 나가니 자유롭게 자를 수 있지만, **바깥쪽**(conv2 출력)은 `out + shortcut(x)` 덧셈 때문에 함께 잘라야 한다 — conv2 를 8채널로 바꾸면 `The size of tensor a (8) must match the size of tensor b (16)` 오류. 9개 블록 안쪽을 25% · 50% 자르면 params 272,474 → 205,490 → **138,506**, 정확도(fine-tuning 전) 0.846 → 0.5025 → **0.168**. 대신 텐서가 진짜로 작아져 지연이 준다(강사 실측 1.72 → 1.36 ms, 1.26배).

**핵심 용어**
- **keep 인덱스**: 남길 채널 번호(오름차순) — 모든 관련 층을 같은 번호로 자른다
- **블록 안쪽 채널**: BasicBlock 의 conv1 출력 = conv2 입력 — 블록 밖에서 안 보인다
- **블록 바깥쪽 채널**: conv2 출력 — residual 덧셈으로 shortcut · 이전 · 다음 블록 · stem 과 묶여 있다

**이 교시에서 만드는 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `prune_block()` | `keep = prune_block(block, ratio)` | 블록 안쪽 채널을 L1 큰 순서로 (1 − ratio) 만 남김 (이 교시에서 만듦) |
| `manual_prune()` | `manual_prune(model, ratio)` | 9개 블록 모두에 prune_block (이 교시에서 만듦) |
| `argsort()` | `filter_l1(c1).argsort(descending=True)[:n_keep].sort().values` | 큰 순서 위치 → 앞 n_keep 개 → 오름차순 |
| `n_params()` | `n_params(model)` | 파라미터 수 합 (제공) |

**핵심 내용**

#### filter 제거 = conv → BN → conv 사슬
> filter 하나를 빼면 그 채널을 만드는 곳(conv1 출력)과 쓰는 곳(bn1 · conv2 입력)을 같은 keep 으로 함께 자른다
- **conv1** → `weight[keep]` (출력 축, dim 0) · 새 Conv 의 out_channels = n_keep
- **bn1** → γ(weight) · β(bias) · running_mean · running_var 4가지를 `[keep]`
- **conv2** → `weight[:, keep]` (입력 축, dim 1) · out_channels 는 그대로 → 블록의 입출력 모양 불변
- ⚠ 새 `nn.BatchNorm2d` 는 train 모드로 만들어진다 → `.train(b1.training)` 으로 원래 모드를 맞춘다

#### residual 덧셈 — 함께 잘라야 하는 채널 묶음
> conv2 출력은 out + shortcut(x) 덧셈의 한쪽 — 두 쪽의 채널 수가 같아야 하므로 혼자 못 자른다
- **블록 안쪽** → 자유롭게 (오늘 손으로 하는 것)
- **블록 바깥쪽** → shortcut · 이전 블록 conv2 · 다음 블록 conv1 입력 · stem 까지 같은 채널을 함께
- **예** → layer1[0].conv2 를 8채널로 → `The size of tensor a (8) must match the size of tensor b (16) at non-singleton dimension 1`
- ⚠ 사람이 따라가면 틀리기 쉬움 → 5교시 DependencyGraph 가 자동으로

#### structured 의 득실
> 같은 비율에서 정확도 손해는 크지만, 텐서가 진짜로 작아져 바로 빨라진다
- **params** → 272,474 → 25% **205,490** → 50% **138,506** (블록 안쪽만이라 절반까지는 안 준다)
- **정확도** (fine-tuning 전) → 0.846 → 0.5025 → **0.168** (unstructured 50% 는 0.8125)
- **지연** → 강사 실측 CPU 1.72 → 1.36 ms (**1.26배**) — unstructured 90% 는 1.0배
- ⚠ fine-tuning 없이 쓰면 안 된다 — 6교시에서 회복

**기본 사용형태**
```python
def prune_block(block, ratio):
    c1, b1, c2 = block.conv1, block.bn1, block.conv2
    n_keep = int(round(c1.out_channels * (1 - ratio)))
    keep = filter_l1(c1).argsort(descending=True)[:n_keep].sort().values
    nc1 = nn.Conv2d(c1.in_channels, n_keep, 3, stride=c1.stride, padding=1, bias=False)
    nc1.weight.data = c1.weight.data[keep].clone()                    # 출력 축
    nb1 = nn.BatchNorm2d(n_keep).train(b1.training)
    for a in ["weight", "bias", "running_mean", "running_var"]:
        getattr(nb1, a).data = getattr(b1, a).data[keep].clone()
    nc2 = nn.Conv2d(n_keep, c2.out_channels, 3, padding=1, bias=False)
    nc2.weight.data = c2.weight.data[:, keep].clone()                 # 입력 축!
    block.conv1, block.bn1, block.conv2 = nc1, nb1, nc2
    return keep
```

**연결**
- [VB D2·6교시 ResNet: Residual block](#/vb/2/6) — out + shortcut(x) 덧셈, 채널이 다르면 1×1 Conv shortcut
- [VB D2·5교시 깊이의 문제 · BatchNorm](#/vb/2/5) — BN 이 채널마다 갖는 γ · β · running_mean · running_var
- [QP D2·3교시 양자화 전처리](#/qp/2/3) — BN folding 처럼 Conv 와 BN 을 한 묶음으로 고치는 작업

**상세 학습**

#### 이 교시에서 답하는 질문
3교시 결론은 "모양을 줄여야 빨라진다" 였다. 그럼 filter 하나를 빼면 무엇을 고쳐야 할까? 이 교시는 BasicBlock 하나의 안쪽 채널을 손으로 줄이면서 conv → BN → conv 사슬을 모두 고치고, 바깥쪽 채널은 왜 손으로 자르기 어려운지(residual) 오류로 확인한다.

#### 개념 1 · 사슬 고치기
![structured pruning — conv1 filter · bn1 4가지 값 · conv2 입력 채널을 같은 keep 으로 줄인다, 바깥쪽은 residual 로 묶여 있다](assets/fig/qp/d5-struct.svg)

| 층 | 원래 모양 (layer2[1]) | ratio 0.5 뒤 | 자르는 축 |
|---|---|---|---|
| conv1.weight | (32, 32, 3, 3) | (16, 32, 3, 3) | dim 0 (출력) |
| bn1 γ · β · μ · σ² | (32,) × 4 | (16,) × 4 | 같은 keep |
| conv2.weight | (32, 32, 3, 3) | (32, 16, 3, 3) | dim 1 (입력) |
| 블록 출력 | (2, 32, 16, 16) | (2, 32, 16, 16) | 그대로 |

- keep (layer2[1], 50%) = `[1, 2, 3, 4, 6, 7, 15, 17, 18, 21, 23, 25, 27, 28, 30, 31]` — L1 이 큰 16개를 오름차순으로.
- ratio 0 이면 모든 채널을 남기므로 출력이 원래와 같아야 한다 (확인 셀의 atol 1e-5 검사).

#### 개념 2 · 9개 블록 모두
| ratio | params | 정확도 (FT 전) | 지연 (강사 실측 CPU) | 지연 (이 실행 · CPU 1스레드) |
|---|---|---|---|---|
| 0 (원본) | 272,474 | 0.846 | 1.72 ms | 2.00 ms |
| 0.25 | 205,490 | 0.5025 | — | 1.75 ms |
| 0.5 | 138,506 | 0.168 | 1.36 ms (1.26배) | 1.51 ms (1.32배) |

- 블록 안쪽 50% → 블록마다 conv1 · conv2 weight 가 각각 절반. stem · shortcut · fc 는 그대로라 전체는 51% (272,474 − 138,506 = 133,968 줄어듦).

#### 코드로 확인
```python
blk = copy.deepcopy(rn20.layer2[1]).eval()
x_ = torch.randn(2, 32, 16, 16); y0 = blk(x_)
keep = prune_block(blk, 0.5)
tuple(blk.conv1.weight.shape), tuple(blk.conv2.weight.shape), tuple(blk(x_).shape)
```
- **무엇을 하나**: 블록 하나의 안쪽 채널을 32 → 16 으로 줄이고 입출력 모양을 확인한다 (실습 4-1).
- **어떤 개념**: 개념 1.
- **출력**: `conv1 (16, 32, 3, 3) | conv2 (32, 16, 3, 3) | 출력 (2, 32, 16, 16)` (직접 실행)

```python
b_ = copy.deepcopy(rn20.layer1[0])
b_.conv2 = nn.Conv2d(16, 8, 3, padding=1, bias=False); b_.bn2 = nn.BatchNorm2d(8)
try:
    b_.eval()(torch.randn(1, 16, 32, 32)); err = None
except RuntimeError as e:
    err = str(e)
```
- **무엇을 하나**: 블록 바깥쪽(conv2 출력)을 자르면 residual 덧셈에서 무슨 일이 나는지 본다 (예제 4-A).
- **어떤 개념**: 개념 1 · 바깥쪽 제약.
- **출력**: `The size of tensor a (8) must match the size of tensor b (16) at non-singleton dimension 1` (직접 실행)

#### 한 번에 정리
- filter 제거 = conv(출력 축) → BN(4가지 값) → 다음 conv(입력 축)를 같은 keep 으로.
- 블록 안쪽은 자유, 바깥쪽은 residual 덧셈 때문에 묶음으로.
- 50% 안쪽 → params 51% · 정확도 0.168(FT 전) · 지연 1.26배(강사 실측).
- ⚠ conv2 는 `weight[:, keep]` — 출력 축(`[keep]`)으로 자르면 모양 오류.

---

### 5교시 · torch-pruning: DependencyGraph · MetaPruner · MACs

**키워드**: `torch-pruning`, `DependencyGraph`, `build_dependency`, `pruning group`, `group.prune()`, `MetaPruner`, `MagnitudeImportance(p=1)`, `ignored_layers`, `MACs`, `(1 − r)²`

**개요**: torch-pruning 은 예시 입력 하나를 흘려 연산 그래프를 따라가며 "이 채널을 자르면 함께 잘라야 하는 모든 곳"(**group**)을 찾아 준다. layer1[0].conv2 의 출력 채널 [0, 1] 을 자르는 group 은 20항목 — stem · 세 블록 conv2(출력) · 세 블록 conv1(입력) · layer2[0] conv1 · shortcut(입력)까지 **Conv 9개 + BN 4개**. layer1 의 세 블록이 모두 같은 residual "통로"(stem 출력 16채널)에 덧셈으로 묶여 있기 때문이다. 통로 채널 2개만 잘라도 0.846 → **0.634**. **MetaPruner** 는 모든 group 의 중요도(L1 합)를 매겨 비율만큼 자동으로 자른다 — `ignored_layers=[model.fc]` 로 클래스 10개 출력을 지킨다. 채널 50% → 16/32/64 → 8/16/32, **MACs 41.24M → 10.53M(25.5%)** · params 272,474 → 68,786(25.2%). 채널 r 을 자르면 Conv 의 입력 · 출력이 함께 줄어 MACs ≈ (1 − r)². fine-tuning 전 정확도는 20% 만 잘라도 **0.2415** 로 무너진다.

**핵심 용어**
- **DependencyGraph(의존성 그래프)**: 층 사이에 "채널을 같이 잘라야 하는" 관계를 기록한 그래프
- **pruning group**: 한 채널을 자를 때 함께 바뀌는 (층, 축, 인덱스) 묶음
- **MetaPruner**: group 마다 중요도를 매겨 정해진 비율만큼 자동으로 자르는 torch-pruning 의 기본 pruner
- **MACs(Multiply-Accumulate operations)**: 곱셈-누산 횟수 — 연산량, Conv 는 out × in × kh × kw × H_out × W_out
- **ignored_layers**: 자르지 않을 층 — 마지막 분류기(fc)의 출력

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `tp.DependencyGraph()` | `DG = tp.DependencyGraph().build_dependency(model, example_inputs=x)` | 그래프 추적 → 층 사이 의존성 기록 |
| `get_pruning_group()` | `DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=[0, 1])` | 그 채널을 자를 때 함께 바뀌는 묶음 |
| `group.prune()` | `group.prune()` | 묶음 전체를 한 번에 자름 |
| `tp.pruner.MetaPruner()` | `tp.pruner.MetaPruner(model, x, importance=…, pruning_ratio=0.5, ignored_layers=[model.fc])` | 모든 group 을 중요도 순으로 비율만큼 |
| `tp.importance.MagnitudeImportance()` | `tp.importance.MagnitudeImportance(p=1)` | group 안 weight 의 L1 합 = 중요도 |
| `tp.utils.count_ops_and_params()` | `macs, params = tp.utils.count_ops_and_params(model, x)` | MACs · params 세기 |
| `make_pruner()` | `make_pruner(model, ratio, steps=1)` | MetaPruner 만들기 (이 교시에서 만듦) |
| `prune_channels()` | `prune_channels(model, ratio)` | 복사본을 step() 한 번으로 자름 (이 교시에서 만듦) |

**핵심 내용**

#### DependencyGraph — 함께 잘라야 하는 곳을 찾는다
> 예시 입력을 흘려 연산 그래프를 따라가며, 한 채널을 자를 때 함께 바뀌는 모든 층을 group 으로 묶는다
- **layer1[0].conv2 출력 [0, 1]** → group 20항목 · Conv **9개**(stem · layer1.0~2 conv1 · conv2 · layer2.0 conv1 · shortcut) + BN 4개
- **왜 9개** → layer1 세 블록이 모두 stem 출력 16채널 통로에 덧셈으로 연결됨
- **결과** → stem 출력 14 · layer1[2].conv2 출력 14 · layer2[0].conv1 입력 14 · 정확도 0.846 → **0.634**
- ⚠ 통로 채널은 모든 블록이 쓰는 중요한 채널 — 2개만 잘라도 크게 떨어진다

#### MetaPruner · MACs
> MetaPruner = group 마다 L1 중요도를 매겨 pruning_ratio 만큼 자동으로 — 채널 r 자르면 MACs ≈ (1 − r)²
- **50%** → 채널 16/32/64 → 8/16/32 · MACs **41.24M → 10.53M (25.5%)** · params **272,474 → 68,786 (25.2%)**
- **ignored_layers=[model.fc]** → fc 출력 10(클래스)은 그대로 · 입력은 64 → 32 로 같이 줄어듦
- **(1 − r)²** → Conv MACs = out × in × k² × H × W 에서 out · in 이 둘 다 (1 − r) 배 → 0.5² = 0.25
- ⚠ `pruning_ratio` 는 **채널 수** 기준 — params · MACs 가 그만큼 주는 것이 아니다

#### 비율 다이얼 (fine-tuning 전)
> MACs · params 는 비율과 함께 매끄럽게 줄지만, fine-tuning 전 정확도는 20% 부터 무너진다
- **MACs** → 0% 41.24 · 20% 24.89 · 40% 14.19 · 60% **6.15** M
- **정확도** → 0.846 · **0.2415** · 0.101 · 0.100
- **채널 수** → 20% 는 stem 12 · layer2 25 · layer3 51 — 어중간한 수 (7 · 8교시 함정)
- ⚠ structured 는 통로 채널까지 자르므로 unstructured 보다 훨씬 예민 — fine-tuning 이 필수

**기본 사용형태**
```python
import torch_pruning as tp
EX_IN = torch.randn(1, 3, 32, 32)

def make_pruner(model, ratio, steps=1):
    return tp.pruner.MetaPruner(model, EX_IN, importance=tp.importance.MagnitudeImportance(p=1),
                                pruning_ratio=ratio, iterative_steps=steps, ignored_layers=[model.fc])

def prune_channels(model, ratio):
    m = copy.deepcopy(model).cpu().eval()
    make_pruner(m, ratio).step()                      # step() 한 번 = 비율만큼 자름
    return m

macs, params = tp.utils.count_ops_and_params(prune_channels(rn20, 0.5), EX_IN)
```

**연결**
- [VB D3·2교시 연산량 계산: 파라미터 · MAC · 메모리](#/vb/3/2) — Conv MAC = out × in × k² × H × W, 오늘 (1 − r)² 의 근거
- [VB D2·6교시 ResNet: Residual block](#/vb/2/6) — 블록 출력이 덧셈으로 이어진 통로 구조
- [QP D3·2교시 Layer 별 오차 분석](#/qp/3/2) — 단독 민감도처럼, 어느 채널 · 층이 중요한지 재는 같은 생각

**상세 학습**

#### 이 교시에서 답하는 질문
4교시에서 블록 바깥쪽 채널은 residual 덧셈으로 묶여 있어 손으로 자르기 어려웠다. 이 교시는 torch-pruning 의 DependencyGraph 로 "함께 잘라야 하는 곳" 을 자동으로 찾고, MetaPruner 로 모델 전체를 비율만큼 자른 뒤 MACs · params 가 얼마나 줄었는지 센다.

#### 개념 1 · group — residual 통로
![DependencyGraph — layer1 의 통로 채널 하나를 자르면 stem 부터 layer2.0 입력까지 Conv 9개 + BN 4개가 함께 잘린다](assets/fig/qp/d5-depgraph.svg)

| group 의 Conv | 자르는 축 | 이유 |
|---|---|---|
| stem.0 | 출력 | 통로의 시작 |
| layer1.0 · 1.1 · 1.2 conv2 | 출력 | 덧셈의 한쪽 |
| layer1.0 · 1.1 · 1.2 conv1 | 입력 | 통로를 읽음 |
| layer2.0.conv1 · layer2.0.shortcut.0 | 입력 | 다음 stage 가 통로를 읽음 |

- BN 4개 = stem.1 · layer1.0~2 bn2 (통로 채널을 정규화하는 BN).

#### 개념 2 · MACs 계산
| 기호 | 뜻 | 예 (layer3.1.conv1, 원본 → 50%) |
|---|---|---|
| out · in | 출력 · 입력 채널 | 64 · 64 → 32 · 32 |
| k² | 커널 3 × 3 | 9 |
| H × W | 출력 크기 | 8 × 8 |
| MACs | out × in × k² × H × W | 2,359,296 → 589,824 (25%) |

- 전체: 41,235,082 → 10,525,002 MACs (25.5%) — stem(입력 3) · fc(출력 10)는 한쪽만 줄어 25% 보다 조금 크다.

| ratio | MACs (M) | params | 정확도 (FT 전) | 채널 stem · layer2 · layer3 |
|---|---|---|---|---|
| 0.0 | 41.24 | 272,474 | 0.846 | 16 · 32 · 64 |
| 0.2 | 24.89 | 170,772 | 0.2415 | 12 · 25 · 51 |
| 0.4 | 14.19 | 96,086 | 0.101 | 9 · 19 · 38 |
| 0.6 | 6.15 | 41,293 | 0.100 | 6 · 12 · 25 |

(직접 실행 · 지연은 7 · 8교시 표에.)

#### 코드로 확인
```python
m_dg = copy.deepcopy(rn20).eval()
DG = tp.DependencyGraph().build_dependency(m_dg, example_inputs=EX_IN)
group = DG.get_pruning_group(m_dg.layer1[0].conv2, tp.prune_conv_out_channels, idxs=[0, 1])
group_convs = sorted({it.dep.target.name.split(" ")[0] for it in group
                      if isinstance(it.dep.target.module, nn.Conv2d)})
group.prune(); acc_dg = acc_torch(m_dg)
```
- **무엇을 하나**: 통로 채널 2개를 자르는 group 을 구해 함께 바뀌는 Conv 를 보고 자른다 (예제 5-A).
- **어떤 개념**: 개념 1.
- **출력**: `group 항목 20개 · 그중 Conv2d 9개` · `자른 뒤 — stem 출력 14 · layer1[2].conv2 출력 14 · layer2[0].conv1 입력 14 · 정확도 0.6340` (직접 실행)

```python
p50 = prune_channels(rn20, 0.5)
rows = []
for name, m in [("FP32", rn20), ("structured 50%", p50)]:
    macs, params = tp.utils.count_ops_and_params(m, EX_IN)
    rows.append({"model": name, "macs_m": macs / 1e6, "params": int(params)})
count_df = pd.DataFrame(rows)
```
- **무엇을 하나**: MetaPruner 로 채널 50% 를 자르고 MACs · params 를 원본과 비교한다 (실습 5-1).
- **어떤 개념**: 개념 2.
- **출력**: `채널 50% → MACs 41.2M → 10.5M (26%) · params 25%` (직접 실행)

#### 한 번에 정리
- DependencyGraph = residual 로 묶인 채널을 찾아 함께 자른다 (통로 채널 = Conv 9개 + BN 4개).
- MetaPruner = group 마다 L1 중요도 → 비율만큼 자동 · `ignored_layers=[fc]` 로 출력 지킴.
- 채널 r → MACs ≈ (1 − r)² (50% → 25.5%).
- ⚠ fine-tuning 전 정확도는 20% 에서 이미 0.24 — 자른 모델은 반드시 다시 학습.

---

### 6교시 · Fine-tuning · iterative pruning

**키워드**: `fine-tuning(회복 학습)`, `lr 1e-2`, `SGD momentum · cosine`, `회복 곡선`, `one-shot vs iterative`, `iterative_steps`, `같은 학습 예산`, `rn20_p50_ft.pth`, `Day 6 KD 학생`

**개요**: 잘라 낸 모델을 학습 데이터로 짧게 다시 학습하면 남은 weight 들이 빠진 채널의 역할을 나눠 맡으며 정확도를 되찾는다 — QP Day 4 의 `train_qat` 와 같은 루프(학습용 10,000장 · 증강 · SGD momentum 0.9 · cosine · 3 epoch)에 lr 만 **1e-2**(QAT 는 거의 다 된 모델을 살짝 1e-3, 오늘은 크게 망가진 모델을 다시 세움). 5교시의 structured 50% 모델(MACs 25%): **0.10 → 0.632 → 0.712 → 0.7225**(3 epoch, FP32 와 12.3%p 차). **iterative**(make_pruner 의 `iterative_steps=3` — 1/3 씩 자르고 1 epoch 씩 회복)로 같은 50% · 같은 총 3 epoch 를 만들면 0.8135 → 0.7465 → **0.68** — 이 짧은 예산에서는 one-shot 이 나았다. 비교는 반드시 **같은 학습 예산**으로. 50% + fine-tuning 모델은 `rn20_p50_ft.pth` 로 저장 → 7교시 배포 · Day 6 지식 증류(KD)의 학생 모델.

**핵심 용어**
- **fine-tuning(회복 학습)**: 가지치기 · 양자화로 손상된 모델을 작은 학습률로 짧게 다시 학습해 정확도를 되찾는 것
- **one-shot pruning**: 목표 비율까지 한 번에 자르고 → fine-tuning
- **iterative pruning**: 조금 자르고 → fine-tuning → 또 자르고 … 반복 — 한 번의 충격이 작다
- **학습 예산**: 비교에 쓴 총 epoch · 데이터 양 — 같아야 공정한 비교

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `finetune()` | `finetune(model, epochs=EPOCHS, lr=1e-2, on_epoch=None)` | 자른 모델 회복 학습 → 걸린 초 (이 교시에서 만듦, Day 4 train_qat 구조) |
| `CosineAnnealingLR()` | `CosineAnnealingLR(opt, epochs * (len(Xtr_u8) // 128 + 1))` | 학습률을 cosine 곡선으로 0 까지 — step 마다 `sched.step()` |
| `pruner.step()` | `pruner = make_pruner(m, 0.5, steps=3); pruner.step()` | 부를 때마다 0.5 의 1/3 씩 더 자름 |
| `torch.save()` | `torch.save(p50_ft.cpu(), "rn20_p50_ft.pth")` | 모양이 바뀐 모델은 **모델 통째로** 저장 |
| `torch.load()` | `torch.load("rn20_p50_ft.pth", weights_only=False)` | 통째로 저장한 모델 읽기 |

**핵심 내용**

#### Fine-tuning — 자른 모델 회복
> 자른 모델을 짧게 다시 학습 — 남은 채널이 빠진 채널의 역할을 나눠 맡는다 (Day 4 train_qat 와 같은 루프, lr 1e-2)
- **설정** → 학습용 10,000장 · 증강(좌우 뒤집기 · 4픽셀 이동) · SGD momentum 0.9 · weight_decay 5e-4 · cosine · 3 epoch
- **회복 곡선** (structured 50%) → 0.10 → **0.632** → 0.712 → **0.7225** (FP32 0.846 대비 −12.3%p · MACs 25%)
- **lr 1e-2 vs 1e-3** → QAT 는 거의 다 된 모델을 살짝 · 오늘은 0.10 까지 망가진 모델을 다시 세움
- ⚠ 증강 · 평가 데이터를 바꾸면 비교가 깨진다 — 비율 0(자르지 않은 모델)도 같은 fine-tuning 으로 (8교시)

#### one-shot vs iterative
> iterative = 1/3 씩 자르고 1 epoch 회복 × 3 — 충격은 작지만, 같은 총 학습량에서 꼭 낫지는 않다
- **iterative 단계** → MACs 27.67 → 17.34 → **10.53** M · 정확도 0.8135 → 0.7465 → **0.68**
- **비교** (같은 MACs 10.53M · 같은 총 3 epoch) → one-shot **0.7225** vs iterative **0.68**
- **왜** → iterative 는 마지막 단계 회복이 1 epoch 뿐 · 총 학습량을 늘려야 이득이 분명해진다
- ⚠ "iterative 가 낫다" 는 일반론 — 결론은 학습 예산과 함께 적는다

**기본 사용형태**
```python
def finetune(model, epochs=EPOCHS, lr=1e-2, on_epoch=None):
    t0 = time.time(); model.to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (len(Xtr_u8) // 128 + 1))
    for ep in range(epochs):
        model.train()
        for x, y in batches(seed=ep):
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step(); sched.step()
        model.eval()
        if on_epoch: on_epoch(ep, model)
    return time.time() - t0

pruner = make_pruner(m_it, 0.5, steps=3)          # iterative: step 마다 1/3 씩
for st in range(3):
    m_it.cpu(); pruner.step(); finetune(m_it, epochs=1)
```

**연결**
- [NN D3·7교시 미니배치 SGD](#/nn/3/7) — 미니배치로 나눠 기울기를 구하고 갱신하는 학습 루프
- [VB D2·7교시 학습 기법: 데이터 증강 · 학습률 스케줄](#/vb/2/7) — 좌우 뒤집기 · 이동 증강, cosine 학습률
- [VB D3·5교시 전이학습](#/vb/3/5) — 이미 학습된 weight 에서 출발해 짧게 다시 학습하는 같은 생각

**상세 학습**

#### 이 교시에서 답하는 질문
5교시의 structured 50% 모델은 MACs 가 1/4 로 줄었지만 정확도가 0.10(무작위)으로 무너졌다. 이 모델은 버려야 할까? 이 교시는 짧은 다시 학습(fine-tuning)으로 얼마나 회복되는지 epoch 별로 재고, 한 번에 자르기(one-shot)와 나눠 자르기(iterative)를 **같은 학습 예산**에서 비교한다.

#### 개념 1 · 회복 곡선
![fine-tuning 회복 곡선 — one-shot 0.10 → 0.7225, iterative 단계마다 자르고 회복해 0.68 (같은 총 3 epoch)](assets/fig/qp/d5-ft.svg)

| epoch | 0 (FT 전) | 1 | 2 | 3 |
|---|---|---|---|---|
| 정확도 (structured 50%) | 0.100 | 0.632 | 0.712 | 0.7225 |

- 이 실행: CPU 에서 3 epoch 95초(다른 작업과 CPU 공유) · 교안 기준 GPU 십여 초 · CPU 40 ~ 60초.
- 저장 `rn20_p50_ft.pth` 323.6 KB — 모양이 바뀐 모델이라 state_dict 가 아니라 모델 통째로 저장(읽을 때 `weights_only=False`).

#### 개념 2 · 같은 예산 비교
| 방식 | 자르기 | 학습 | 최종 MACs | 최종 정확도 |
|---|---|---|---|---|
| one-shot | 50% 한 번 | 3 epoch | 10.53 M | 0.7225 |
| iterative | 1/3 씩 3번 | 1 epoch × 3 | 10.53 M | 0.68 |

| iterative step | MACs (M) | 정확도 (1 epoch 뒤) |
|---|---|---|
| 1 | 27.67 | 0.8135 |
| 2 | 17.34 | 0.7465 |
| 3 | 10.53 | 0.68 |

- 마지막 단계에서 크게 자른 직후 회복이 1 epoch 뿐 → one-shot 의 3 epoch 회복보다 짧다.

#### 코드로 확인
```python
p50_ft = copy.deepcopy(p50)
hist = [{"epoch": 0, "acc": acc_torch(p50_ft)}]
t_ft = finetune(p50_ft, on_epoch=lambda ep, m: hist.append({"epoch": ep + 1, "acc": acc_torch(m)}))
hist_df = pd.DataFrame(hist)
torch.save(p50_ft.cpu(), "rn20_p50_ft.pth")
```
- **무엇을 하나**: structured 50% 모델을 3 epoch fine-tuning 하며 epoch 별 정확도를 기록하고 저장한다 (실습 6-1).
- **어떤 개념**: 개념 1.
- **출력**: `50% 모델 0.10 → 0.723 — FP32 와의 차이 12.3%p (MACs 25%)` (직접 실행 · CPU)

```python
m_it = copy.deepcopy(rn20).eval(); pruner = make_pruner(m_it, 0.5, steps=3); rows = []
for st in range(3):
    m_it.cpu(); pruner.step()
    macs, _ = tp.utils.count_ops_and_params(m_it, EX_IN)
    finetune(m_it, epochs=1)
    rows.append({"step": st + 1, "macs_m": macs / 1e6, "acc": acc_torch(m_it)})
```
- **무엇을 하나**: 1/3 씩 자르고 1 epoch 회복을 3번 반복한다 (실습 6-2).
- **어떤 개념**: 개념 2.
- **출력**: `같은 MACs (10.5M) · 같은 총 3 epoch — one-shot 0.723 vs iterative 0.680` (직접 실행 · CPU)

#### 한 번에 정리
- structured 뒤 fine-tuning(lr 1e-2 · 3 epoch) → 50% 모델 0.10 → 0.72.
- one-shot vs iterative 는 같은 학습 예산으로 비교 — 짧은 예산에서는 one-shot 이 나았다.
- 모양이 바뀐 모델은 통째로 저장(`torch.save(model)`) · 읽을 때 `weights_only=False`.
- ⚠ 학습 결과는 GPU · 세션 · 난수마다 조금씩 다르다 — 교안 숫자는 예시.

---

### 7교시 · 배포: ONNX → TensorRT FP16 에서 정말 빨라지나

**키워드**: `export_b1()`, `onnx_conv_stats()`, `deploy_lat()`, `TensorRT FP16 엔진`, `dense 커널`, `unstructured = dense 속도`, `structured = MACs 만큼은 아니게`, `채널 수 배수 (8 · 16)`, `round_to`, `qp_baseline.csv`

**개요**: pruning 의 최종 판정은 **배포 엔진에서의 지연**이다. 세 모델 — dense(원본) · unstructured 50%(global, FT 없음) · structured 50% + FT(`rn20_p50_ft.pth`) — 를 같은 방법(배치 1 ONNX)으로 내보내 같은 자(`deploy_lat`: GPU 면 TensorRT FP16 엔진, 아니면 ORT CPU)로 잰다. **unstructured**: ONNX 크기 1,070.8 KB · Conv weight 270,256개로 dense 와 똑같고 0 만 50.1% → 엔진도 dense 커널을 써서 속도 같음(이 실행 ORT CPU 1.00배). **structured**: Conv weight 67,672개(25%) · ONNX **276.6 KB** · 정확도 0.7225 → 이 실행 **2.06배** 빠름(TensorRT 에서는 커널 실행 오버헤드 때문에 MACs 비율 4배만큼은 안 빨라짐). ⚠ 채널 수도 속도를 좌우한다 — 커널은 채널을 4 · 8 · 16 개씩 묶어 계산하므로 13 · 26 · 51 같은 어중간한 수는 MACs 가 줄어도 느려질 수 있다(50% 모델은 8 · 16 · 32 로 딱 맞음, torch-pruning 의 `round_to=8`). 결과 3행은 `qp_baseline.csv` 에 Day 2 ~ 4 와 함께 모은다.

**핵심 용어**
- **dense 커널**: 0 이 있든 없든 모든 원소를 곱하는 보통 Conv 구현 — TensorRT · cuDNN · MKL 의 기본
- **커널 실행 오버헤드**: 연산마다 GPU 커널을 띄우는 고정 비용 — 작은 모델에서는 MACs 보다 이것이 지연을 좌우
- **채널 정렬(alignment)**: 채널 수를 4 · 8 · 16 의 배수로 맞추는 것 — Tensor Core 가 묶어서 계산하는 단위
- **round_to**: torch-pruning 옵션 — 자른 뒤 채널 수를 지정한 배수로 맞춘다

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `export_b1()` | `export_b1(model, "rn20_p50.onnx")` | 배치 1 고정 ONNX (opset 17 · 입력 input · 출력 logits, 제공) |
| `onnx_conv_stats()` | `onnx_conv_stats(path)` | ONNX 안 Conv weight 원소 수 · 0 비율 (제공) |
| `onnx_acc()` | `onnx_acc(path)` | ORT 로 평가 2,000장 정확도 (제공) |
| `deploy_lat()` | `deploy_lat(onnx_path, n=300)` | GPU 면 TensorRT FP16 엔진, 아니면 ORT CPU 배치 1 평균 ms (제공) |
| `drop_duplicates()` | `pd.concat([old, new]).drop_duplicates(subset=["day", "model"], keep="last")` | 같은 (day, model) 은 마지막 것만 — qp_baseline 갱신 |

**핵심 내용**

#### 배포 비교 — 같은 엔진 · 같은 자
> unstructured 는 ONNX 크기 · Conv 모양 · 엔진 속도 모두 dense 와 같고, structured 만 Conv 가 작아져 빨라진다
- **dense** → 1,070.8 KB · conv 270,256 · 0 비율 0 · 정확도 0.846
- **unstructured 50%** → 1,070.8 KB · conv 270,256 · **0 비율 0.501** · 정확도 0.8125 · 속도 ≈ 1배
- **structured 50% + FT** → **276.6 KB** · conv **67,672** · 정확도 0.7225 · 이 실행 ORT CPU **2.06배**
- ⚠ TensorRT 도 FP16 엔진을 만들 때 dense 커널을 고른다 → unstructured 는 GPU 에서도 그대로 (교안: 0.75 ~ 1.3배 범위 확인)

#### 채널 수가 속도를 좌우한다
> 커널은 채널을 4 · 8 · 16 개씩 묶어 계산 — 어중간한 채널 수는 MACs 가 줄어도 느려질 수 있다
- **50% 모델** → 채널 8 · 16 · 32 — 딱 맞는 배수
- **20% 모델** → 채널 12 · 25 · 51 — 묶음을 다 못 채워 낭비 (8교시에서 직접 마주침)
- **고치는 법** → torch-pruning `round_to=8` 로 채널 수를 8의 배수로 맞춰 자름
- ⚠ MACs 는 이론 연산량일 뿐 — 지연은 배포 장치에서 직접 잰다

#### qp_baseline.csv — 한 표에 모으기
> Day 2 ~ 4 의 양자화 결과와 오늘의 가지치기 결과를 같은 열(day · model · quant · size_kb · acc · mean_ms)로
- **Day5 3행** → rn20_dense · rn20_us50 · rn20_p50, quant = "FP32 (<model 이름>)"
- **p95_ms · fps_b32** → 오늘은 NaN
- **갱신** → `drop_duplicates(subset=["day", "model"], keep="last")` — 다시 실행해도 중복 없음
- ⚠ 파일이 없으면 새로 만든다 (`os.path.exists` 로 확인)

**기본 사용형태**
```python
models = {"dense": rn20,
          "unstructured 50%": prune_unstructured(rn20, 0.5),
          "structured 50% + FT": torch.load("rn20_p50_ft.pth", weights_only=False)}
paths = {"dense": "rn20_dense.onnx", "unstructured 50%": "rn20_us50.onnx", "structured 50% + FT": "rn20_p50.onnx"}
rows = []
for name, m in models.items():
    p = export_b1(m, paths[name])
    rows.append({"model": name, "size_kb": size_kb(p), **onnx_conv_stats(p), "acc": onnx_acc(p)})
onnx_df = pd.DataFrame(rows)
deploy_df = onnx_df[["model", "acc"]].copy()
deploy_df["latency_ms"] = [deploy_lat(paths[m]) for m in deploy_df.model]
deploy_df["speedup"] = deploy_df.latency_ms[0] / deploy_df.latency_ms
```

**연결**
- [OT D4·6교시 FP16 으로 더 빠르게](#/ot/4/6) — FP16 엔진 빌드, 오늘 deploy_lat 의 GPU 쪽
- [OT D2·2교시 ONNX 파일 = 그래프](#/ot/2/2) — initializer 에서 Conv weight 를 꺼내는 onnx_conv_stats 의 바탕
- [QP D2·7교시 성능 측정](#/qp/2/7) — qp_baseline.csv 를 처음 만든 곳

**상세 학습**

#### 이 교시에서 답하는 질문
3교시는 PyTorch CPU 에서 unstructured 가 안 빨라짐을, 4 · 5교시는 structured 가 텐서를 줄임을 봤다. 실제 배포 엔진(TensorRT)에서도 그럴까? 이 교시는 세 모델을 같은 형식(배치 1 ONNX) · 같은 엔진으로 내보내 크기 · 정확도 · 지연을 한 표로 비교하고 결과를 qp_baseline.csv 에 모은다.

#### 개념 1 · 세 모델 한 표
![배포 비교 — unstructured 는 dense 와 크기 · 속도가 같고, structured 50% + FT 만 크기 1/4 · 2.06배 빠름](assets/fig/qp/d5-deploy.svg)

| 모델 | ONNX KB | Conv weight | 0 비율 | 정확도 (ORT) | 지연 ms (ORT CPU) | 속도비 |
|---|---|---|---|---|---|---|
| dense | 1,070.8 | 270,256 | 0.000 | 0.846 | 0.703 | 1.00 |
| unstructured 50% | 1,070.8 | 270,256 | 0.501 | 0.8125 | 0.703 | 1.00 |
| structured 50% + FT | 276.6 | 67,672 | 0.000 | 0.7225 | 0.341 | 2.06 |

(이 실행 · GPU 없음 → ORT CPU · 스레드 1 · 배치 1 · 5번 잰 중앙값.) 교안 확인 셀의 기대: unstructured 0.75 ~ 1.3배 · structured 1.15배 이상(TensorRT FP16 · T4).

- ONNX 크기 비율 276.6 / 1,070.8 = 25.8% ≈ Conv weight 비율 67,672 / 270,256 = 25.0%.

#### 개념 2 · 왜 MACs 비율(4배)만큼은 안 빨라지나
- MACs 41.24 → 10.53M 은 이론상 3.9배. 실제 지연에는 연산마다 붙는 고정 비용(커널 실행 · 메모리 이동)이 있어 작은 모델일수록 비율이 줄어든다.
- 채널 수가 묶음 단위(8 · 16)와 맞지 않으면 더 손해 — 20% 모델(12 · 25 · 51)은 MACs 가 줄어도 dense 보다 느릴 수 있다.

#### 코드로 확인
```python
def onnx_conv_stats(path):
    g = onnx.load(path).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    ws = [init[n.input[1]] for n in g.node if n.op_type == "Conv" and n.input[1] in init]
    total = sum(w.size for w in ws)
    return {"conv_params": int(total), "zero_frac": float(sum((w == 0).sum() for w in ws) / total)}
```
- **무엇을 하나**: ONNX 그래프의 Conv 노드마다 두 번째 입력(weight initializer)을 꺼내 원소 수와 0 비율을 센다 (제공 도구).
- **어떤 개념**: 개념 1.
- **출력**: dense `{'conv_params': 270256, 'zero_frac': 0.0}` · unstructured `{'conv_params': 270256, 'zero_frac': 0.501}` · structured `{'conv_params': 67672, 'zero_frac': 0.0}` (직접 실행)

```python
tags = {"dense": "rn20_dense", "unstructured 50%": "rn20_us50", "structured 50% + FT": "rn20_p50"}
new = pd.DataFrame([{"day": "Day5", "model": tags[r.model], "quant": f"FP32 ({tags[r.model]})",
                     "size_kb": d_.size_kb[r.model], "acc": r.acc, "mean_ms": r.latency_ms,
                     "p95_ms": np.nan, "fps_b32": np.nan} for r in deploy_df.itertuples()])
old = pd.read_csv("qp_baseline.csv") if os.path.exists("qp_baseline.csv") else None
pd.concat([old, new]).drop_duplicates(subset=["day", "model"], keep="last").to_csv("qp_baseline.csv", index=False)
```
- **무엇을 하나**: 세 모델 결과를 qp_baseline.csv 에 Day5 로 추가한다 (실습 7-2).
- **어떤 개념**: 한 표에 모으기.
- **출력**: `ORT CPU — unstructured 1.00배 · structured 2.06배 — 속도는 structured 에서만` (직접 실행)

#### 한 번에 정리
- unstructured 50% → ONNX 크기 · Conv 모양 · 엔진 속도 모두 dense 와 같다 (0 만 50%).
- structured 50% + FT → Conv 25% · ONNX 1/4 · 실제로 빨라짐 — 단 MACs 비율만큼은 아니다.
- 실무에서 속도를 얻는 것은 structured — 결과는 qp_baseline.csv 에 모은다.
- ⚠ 채널 수를 8 · 16 배수로 (`round_to=8`) — 어중간한 채널은 MACs 가 줄어도 느릴 수 있다.

---

### 8교시 · 미니 프로젝트: 가지치기 비율 곡선

**키워드**: `비율 곡선`, `MACs – 정확도`, `지연 – 정확도`, `무릎(knee)`, `FP32 − 3%p 기준`, `best_ratio`, `공정한 비교 (비율 0 도 같은 FT)`, `pruning_sweep.csv`, `rn20_pruned_best.onnx`, `채널 수 함정`

**개요**: "ResNet-20 을 얼마나 잘라도 될까?" 를 하나의 비율이 아니라 **곡선**으로 답한다. structured 비율 0 / 20 / 40 / 60% 각각 → `prune_channels` → **같은 fine-tuning**(lr 1e-2 · 3 epoch, 비율 0 도) → MACs · params · 정확도 · 배포 지연. 선택 기준: 정확도가 FP32(0.846) 대비 **3%p 이내**(≥ 0.816)인 가장 큰 비율 — 3%p 로 둔 이유는 오늘의 짧은 학습 예산(10,000장 · 3 epoch)에서는 자르지 않은 모델도 0.84 안팎이기 때문(실무는 50,000장 · 수십 epoch 뒤 1%p). 이 실행(CPU): 정확도 0.842 · **0.8195** · 0.7595 · 0.603 → **best_ratio = 0.2**(MACs 1.66배 ↓ · params 1.60배 ↓ · 하락 2.65%p). 그런데 ORT CPU 지연은 0.704 → **0.839 ms 로 오히려 느려졌다** — 20% 모델의 채널 수가 12 · 25 · 51 로 어중간하기 때문(7교시 ⚠). 40% · 60% 는 빨라지지만 기준 밖. 과제 6(결론)의 핵심 질문이 바로 이것.

**핵심 용어**
- **비율 곡선(sweep)**: 비율을 바꿔 가며 같은 절차로 잰 (MACs · 지연, 정확도) 점들 — 무릎을 찾는다
- **무릎(knee)**: 곡선에서 정확도가 갑자기 꺾이기 시작하는 지점 — 쓸 만한 최대 비율
- **선택 기준**: 미리 정한 규칙 (FP32 대비 몇 %p 이내 중 가장 많이 줄인 것)
- **공정한 비교**: 비율 0 도 같은 fine-tuning · 같은 평가 데이터 · 같은 엔진

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `ax.annotate()` | `ax.annotate(f"{int(r * 100)}%", (x, y), textcoords="offset points", xytext=(5, 5))` | 점마다 비율 글자 |
| `ax.axhline()` | `ax.axhline(FP32_ACC - 0.03, ls="--")` | FP32 − 3%p 기준선 |

**핵심 내용**

#### 비율 곡선 — 무릎 찾기
> 비율 하나가 아니라 곡선(MACs – 정확도 – 지연)을 그려 정확도가 꺾이는 '무릎' 을 찾는다
- **절차** → 비율마다 prune_channels → MACs 측정(FT 전) → acc_before → finetune 3 epoch → acc → ONNX → deploy_lat
- **정확도 (FT 뒤)** → 0% **0.842** · 20% **0.8195** · 40% 0.7595 · 60% 0.603 (FT 전 0.846 · 0.2415 · 0.101 · 0.100)
- **MACs** → 41.24 · 24.89 · 14.19 · 6.15 M
- ⚠ 비율 0 도 같은 fine-tuning — 안 하면 "자른 효과" 와 "학습 효과" 가 섞인다 (0.846 → 0.842 로 오히려 조금 떨어짐)

#### 기준과 선택
> FP32 − 3%p(0.816) 이상인 가장 큰 비율 → 이 실행 best_ratio = 0.2
- **summary** (비율 0 대비) → MACs **1.66배** ↓ · params **1.60배** ↓ · 지연 속도비 **0.84** · 정확도 하락 **2.65%p**
- **1%p 기준이면** → 0.836 이상인 것은 0% 뿐 → 자를 수 없음 — 학습 예산(전체 50,000장 · 더 많은 epoch) · KD(Day 6)로 끌어올린다
- **결과 저장** → pruning_sweep.csv · rn20_pruned_best.onnx · qp_baseline.csv (Day5 · rn20_pruned_best)
- ⚠ 기준(몇 %p)과 학습 예산이 결론을 바꾼다 — 둘 다 결과와 함께 적는다

#### MACs ↓ 인데 지연 ↑ — 채널 수 함정
> 20% 모델은 MACs 가 40% 줄었는데 ORT CPU 에서 0.704 → 0.839 ms — 채널 12 · 25 · 51 이 계산 묶음(4 · 8 · 16)과 안 맞는다
- **확인** → 층마다 채널 수 출력 · 층별 profiling (OT D3·7교시)
- **고치기** → torch-pruning `round_to=8` 로 채널 수를 8 의 배수로 · 또는 비율을 25% 처럼 딱 나눠지게
- **40% · 60%** → 0.629 · 0.342 ms 로 빨라짐 (60% 는 2.06배) — 그러나 정확도 기준 밖
- ⚠ "MACs 가 줄었으니 빠르다" 는 가정 금지 — 배포 엔진에서 직접 잰다 (T4 TensorRT 에서는 또 다를 수 있음)

**기본 사용형태**
```python
RATIOS = [0.0, 0.2, 0.4, 0.6]
rows, sweep_models = [], {}
for r in RATIOS:
    m = copy.deepcopy(rn20) if r == 0 else prune_channels(rn20, r)
    macs, params = tp.utils.count_ops_and_params(m, EX_IN)
    acc_before = acc_torch(m); t = finetune(m); acc = acc_torch(m)
    p = export_b1(m, f"rn20_s{int(r * 100)}.onnx")
    rows.append({"ratio": r, "macs_m": macs / 1e6, "params": int(params), "acc_before": acc_before,
                 "acc": acc, "latency_ms": deploy_lat(p), "ft_s": t})
    sweep_models[r] = m
sweep_df = pd.DataFrame(rows)
ok_ = sweep_df[sweep_df.acc >= FP32_ACC - 0.03]
best_ratio = float(ok_.ratio.max()) if len(ok_) else 0.0
best_row = sweep_df.set_index("ratio").loc[best_ratio]
```

**연결**
- [QP D2·8교시 미니 프로젝트: 최적 PTQ 설정 찾기](#/qp/2/8) — 설정 여러 개를 같은 자로 재고 규칙으로 고르는 같은 형식
- [QP D3·8교시 미니 프로젝트](#/qp/3/8) — 배포 심사표 · "규칙 → 선택" · 장치에서 다시 재기
- [OT D3·7교시 profiling: 어디가 느린가](#/ot/3/7) — MACs 가 줄었는데 느릴 때 층별로 확인하는 도구

**상세 학습**

#### 이 교시에서 답하는 질문
1 ~ 7교시에서 가지치기의 방법(unstructured · structured · fine-tuning · 배포)을 모두 배웠다. 남은 질문은 "그래서 얼마나 자를까?" — 이 교시는 비율 네 개를 같은 절차로 재서 곡선을 그리고, 미리 정한 규칙(FP32 대비 3%p)으로 비율을 고른 뒤 그 선택의 득실을 숫자로 요약한다.

#### 개념 1 · 과제 구성 (100점)
| 과제 | 내용 | 배점 |
|---|---|---|
| 1 | 비율 4개 × (prune → fine-tuning → 측정) 표 `sweep_df` | 20 |
| 2 | MACs – 정확도 · 지연 – 정확도 그림 `prune_curve.png` (점마다 비율 · 기준선) | 20 |
| 3 | 기준(FP32 대비 3%p 이내)으로 `best_ratio` · `best_row` | 20 |
| 4 | `pruning_sweep.csv` · `rn20_pruned_best.onnx` · `qp_baseline.csv` 저장 | 20 |
| 5 | 고른 모델의 이득 요약 `summary` | 10 |
| 6 | 결론 쓰기 (마크다운, 강사 채점) | 10 |

정답 보기 셀 없음 · 예상 시간 GPU 1 ~ 2분, CPU 약 4분 (fine-tuning 4번).

#### 개념 2 · 이 실행의 곡선
![가지치기 비율 곡선 — FT 뒤 정확도 0.842 · 0.8195 · 0.7595 · 0.603, 3%p 기준선 안의 가장 큰 비율 20% · 지연은 20% 에서 오히려 늘어남](assets/fig/qp/d5-sweep.svg)

| ratio | MACs (M) | params | acc_before | acc (FT 뒤) | 지연 ms (ORT CPU) | ONNX KB |
|---|---|---|---|---|---|---|
| 0.0 | 41.24 | 272,474 | 0.846 | 0.842 | 0.704 | 1,070.8 |
| 0.2 | 24.89 | 170,772 | 0.2415 | **0.8195** ✓ | 0.839 | 674.1 |
| 0.4 | 14.19 | 96,086 | 0.101 | 0.7595 | 0.629 | 383.0 |
| 0.6 | 6.15 | 41,293 | 0.100 | 0.603 | 0.342 | 169.6 |

(직접 실행 · CPU fine-tuning 비율마다 28 ~ 60초 · 지연은 ORT CPU 스레드 1 · 5번 잰 중앙값.) 기준 0.846 − 0.03 = 0.816 → 통과: 0.0 · 0.2 → best_ratio 0.2.

| summary 항목 | 계산 | 값 |
|---|---|---|
| macs_x | 41.24 / 24.89 | 1.66 |
| params_x | 272,474 / 170,772 | 1.60 |
| speedup | 0.704 / 0.839 | 0.84 (느려짐) |
| acc_drop_pp | (0.846 − 0.8195) × 100 | 2.65 |

#### 개념 3 · 과제 6 의 생각할 거리
- **1%p 기준이면**: 0.836 이상은 비율 0 뿐 → 자를 수 없다. 전체 50,000장 · 수십 epoch fine-tuning, Day 6 지식 증류(KD)로 회복 폭을 키우면 1%p 안에 들 수 있다.
- **T4 배포: unstructured 70% vs structured 20%**: unstructured 는 T4 에서 속도 이득 0(2:4 도 T4 미지원) · 70% 는 FT 없이 0.482 → structured 20% + FT 가 낫다 (단 채널 수를 8 배수로).
- **MACs ↓ 지연 ↑ (이 실행의 20%)**: 층별 채널 수(12 · 25 · 51) 확인 → `round_to=8` 로 다시 자르기 · profiling 으로 느린 층 찾기.
- **INT8 과 합치는 순서**: prune → fine-tuning → (PTQ 또는 QAT) — 모양을 먼저 정하고, 그 모양에서 scale 을 잡는다. 반대로 하면 자른 뒤 scale 을 다시 보정해야 한다.

#### 코드로 확인
```python
fig_, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, col, xl in [(axs[0], "macs_m", "MACs (M)"), (axs[1], "latency_ms", "latency (ms)")]:
    ax.plot(sweep_df[col], sweep_df.acc, "o-")
    for r in sweep_df.itertuples():
        ax.annotate(f"{int(r.ratio * 100)}%", (getattr(r, col), r.acc), textcoords="offset points", xytext=(5, 5))
    ax.axhline(FP32_ACC - 0.03, ls="--", c="tab:red"); ax.set_xlabel(xl); ax.set_ylabel("accuracy")
fig_.savefig("prune_curve.png", dpi=120)
```
- **무엇을 하나**: 두 칸짜리 곡선에 점마다 비율 · 3%p 기준선을 그려 저장한다 (과제 2).
- **어떤 개념**: 개념 2.
- **출력**: `prune_curve.png` — 왼쪽 MACs 가 줄수록 정확도 하락 · 오른쪽 20% 점이 0% 보다 오른쪽(느림)

```python
b0 = sweep_df.set_index("ratio").loc[0.0]
summary = {"best_ratio": best_ratio,
           "macs_x": b0.macs_m / best_row.macs_m, "params_x": b0.params / best_row.params,
           "speedup": b0.latency_ms / best_row.latency_ms,
           "acc_drop_pp": (FP32_ACC - best_row.acc) * 100}
```
- **무엇을 하나**: 고른 모델의 이득을 비율 0(fine-tuning 한 dense) 대비 배수로 요약한다 (과제 5).
- **어떤 개념**: 개념 2.
- **출력**: `{'best_ratio': 0.2, 'macs_x': 1.657, 'params_x': 1.596, 'speedup': 0.839, 'acc_drop_pp': 2.65}` (직접 실행 · ORT CPU)

#### 한 번에 정리
- 비율 하나가 아니라 곡선(MACs – 정확도 – 지연)에서 무릎을 찾는다.
- 공정성: 비율 0 도 같은 fine-tuning · 같은 평가 데이터 · 같은 엔진.
- 이 실행: 3%p 기준 → 20% (MACs 1.66배 ↓ · 하락 2.65%p) — 하지만 채널 수 때문에 CPU 지연은 늘었다.
- ⚠ 기준(몇 %p)과 학습 예산이 결론을 바꾼다 — 둘 다 결과와 함께 적고, 지연은 배포 장치에서 다시 잰다.
