## Day 5 — 세그멘테이션 · 포즈 · ONNX · 성능 측정 → 위험구역 침입 감지

### 1교시 · 복습 퀴즈 · 박스로는 부족한 상황

**키워드**: `분류 · 탐지 · 세그멘테이션 · 포즈`, `Penn-Fudan 마스크 PNG`, `0 = 배경, k = k번째 사람`, `박스 안 사람 픽셀 비율`, `h/w 종횡비`, `90° 회전`

**개요**: Day 4 의 박스는 "무엇이 어디에 몇 개" 까지만 답한다. 현장 안전에는 박스로 답하기 어려운 질문이 있다. ① 몸이 위험구역 경계를 넘었나 — 박스는 몸 주변 **배경**까지 포함. ② 바닥 기름의 **넓이** — 모양이 박스가 아님. ③ 쓰러졌나 — 박스의 가로세로 비율은 자세 · 촬영 방향에 약함. 그래서 오늘은 출력이 다른 두 태스크, **세그멘테이션**(픽셀) 과 **포즈**(관절) 를 배운다.

**핵심 용어**
- **세그멘테이션(segmentation)**: 픽셀마다 어떤 물체인지 정하는 문제 — 모양 · 넓이 · 경계에 답함
- **포즈 추정(pose estimation)**: 사람마다 관절 17개의 (x, y, conf) 를 찾는 문제 — 자세에 답함
- **인스턴스 마스크(instance mask)**: 물체 하나하나의 픽셀 지도 (Penn-Fudan PNG: 0 = 배경, k = k번째 사람)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `np.unique()` | `np.unique(m0)` | 마스크에 들어 있는 값(0, 1, 2…) 목록 |
| `mask_box()` | `mask_box(m)` | 마스크의 픽셀 → 둘러싸는 박스 `xyxy` |
| `np.where()` | `np.where(m)` | True 인 픽셀의 (행, 열) 위치 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 1-1 | 마스크 PNG 값 `[0, 1, 2]` → 사람마다 `m0 == k` 마스크, 마스크에서 박스 복원 |
| 1-2 | 정답 박스 안에서 사람 픽셀이 차지하는 비율 (423명 평균) |
| 1-3 | 서 있는 사람의 h/w 와 90° 돌렸을 때의 h/w |

**핵심 내용**

#### 박스 · 마스크 · 관절 — 출력이 답하는 질문
> 같은 사람에 대해 분류 = 라벨 1개, 탐지 = (클래스, 박스, 점수), 세그멘테이션 = 픽셀 지도, 포즈 = 관절 17개 — 출력이 곧 답할 수 있는 질문
| 태스크 | 출력 | 답하는 질문 |
|---|---|---|
| 분류 | 라벨 1개 | 무엇인가 |
| 탐지 | 물체마다 (클래스, 박스, 점수) | 무엇이 어디에, 몇 개 |
| 세그멘테이션 | 픽셀마다 클래스 / 물체마다 마스크 | 어떤 모양 · 넓이 · 경계 |
| 포즈 | 사람마다 관절 17개 (x, y, conf) | 어떤 자세 |
- ⚠ 박스 → 마스크는 자동으로 안 된다. 거꾸로 마스크 → 박스(`mask_box`)는 최솟값 · 최댓값으로 바로 나온다

#### 박스의 한계 두 가지
> 박스 안에는 배경이 절반쯤 섞여 있고, 박스의 종횡비는 시점에 따라 바뀐다
- **박스 안 배경** → 사람 픽셀 / 박스 넓이 = 423명 평균 0.50 (0.27 ~ 0.78) — 경계 · 넓이 문제는 세그멘테이션
- **종횡비** → 서 있는 사람 h/w 평균 2.77, 90° 돌리면 역수(1 / ar) 평균 0.38 로 — 박스 비율만으로 쓰러짐을 판정하면 틀린다 → 포즈
- ⚠ Penn-Fudan 어노테이션은 1부터 센 좌표 — 배열 인덱스로 쓰려면 `b - 1`

**기본 사용형태**
```python
m0 = np.array(Image.open(mask_path(img_paths[0])))
ids = np.unique(m0)                       # [0, 1, 2]  (0 = 배경)
inst = [m0 == k for k in ids[1:]]         # 사람마다 True/False 마스크
ys, xs = np.where(inst[0])
box = [xs.min(), ys.min(), xs.max(), ys.max()]
```

**연결**
- [VB D4·1교시 분류기로 사람 수](#/vb/4/1) — 박스로 푸는 문제와 그 한계의 출발점
- [VB D4·3교시 IoU · NMS](#/vb/4/3) — 오늘 IoU 를 박스에서 픽셀 집합으로 확장
- [VB D1·6교시 디지털 이미지](#/vb/1/6) — 마스크도 H×W 배열

**상세 학습**

#### 이 교시에서 답하는 질문
Day 4 까지 "사람이 몇 명, 어디에" 를 박스로 풀었다. 그런데 현장 안전의 질문 중에는 박스로 믿을 만한 답이 안 나오는 것이 있다 — **① 몸이 위험구역 경계를 넘었나 ② 바닥 기름이 몇 ㎡ 인가 ③ 사람이 쓰러졌나**. 이 교시는 Penn-Fudan 의 **마스크 정답**으로 박스가 왜 부족한지 숫자로 확인하고, 오늘 배울 두 출력(마스크 · 관절)이 어떤 질문을 대신 맞히는지 정리한다.

#### 개념 1 · 출력이 곧 답할 수 있는 질문
![같은 사람에 대한 박스 · 마스크 · 관절 출력과 박스의 한계 두 가지](assets/fig/vb/d5-outputs.svg)

| 태스크 | 출력 | 답하는 질문 | 오늘 쓰는 모델 |
|---|---|---|---|
| 분류 | 라벨 1개 | 무엇인가 | MobileNetV3 (Day 3) |
| 탐지 | 물체마다 (클래스, 박스, 점수) | 어디에 · 몇 개 | YOLOv8n (Day 4) |
| 세그멘테이션 | 픽셀마다 클래스 / 물체마다 마스크 | 모양 · 넓이 · 경계 | YOLOv8n-seg · DeepLabV3 |
| 포즈 | 사람마다 관절 17개 (x, y, conf) | 자세 | YOLOv8n-pose |

#### 개념 2 · 마스크 PNG — 값 하나가 사람 하나
Penn-Fudan 의 마스크는 사진과 같은 크기의 PNG 다. 픽셀값 **0 = 배경, 1 = 첫째 사람, 2 = 둘째 사람** … 이렇게 사람마다 번호가 다르므로 "물체마다 따로" 인 **인스턴스 마스크**다.
- `np.unique(m0)` → 들어 있는 값 `[0 1 2]` → 사람 2명
- `m0 == k` → k 번째 사람만 True 인 마스크
- 마스크 → 박스는 쉽다: True 픽셀의 x · y 최솟값 · 최댓값 (`mask_box`). 첫 사진에서 마스크 박스 `(159, 181, 301, 430)` vs 어노테이션 `(160, 182, 302, 431)` — 어노테이션은 **1부터 센 좌표**라 정확히 1씩 크다
- 거꾸로 박스 → 마스크는 안 된다 (박스 안 어디가 사람인지 모름)

#### 개념 3 · 박스의 한계 — 숫자로
| 확인 | 식 | 실제 결과 (423명) | 뜻 |
|---|---|---|---|
| 박스 안 사람 비율 | 사람 픽셀 수 / 박스 넓이 | 평균 **0.5036** (최소 0.27, 최대 0.78) | 박스의 절반은 배경 → 경계 · 넓이 판정이 부정확 |
| 서 있는 사람 h/w | 박스 높이 / 너비 | 평균 **2.77** (최소 1.59) | 서 있으면 세로로 길다 |
| 90° 회전 h/w | 1 / (h/w) | 평균 **0.38** | 같은 자세라도 카메라가 돌면 비율이 뒤집힌다 |

"h/w < 1 이면 쓰러짐" 같은 박스 규칙은 카메라 각도 · 자세(쪼그려 앉음)에 쉽게 속는다 → 관절로 몸의 방향을 직접 재야 한다 (3교시).

#### 코드로 확인
```python
m0 = np.array(Image.open(mask_path(img_paths[0])))
ids = np.unique(m0)
inst = [m0 == k for k in ids[1:]]                 # 0(배경) 빼고 사람마다
print("픽셀값:", ids, "| 사람", len(inst), "명 | 픽셀 수", [int(m.sum()) for m in inst])
print("마스크 박스", [mask_box(m) for m in inst])
```
- **무엇을 하나**: 첫 사진 마스크의 값 · 사람 수 · 사람별 픽셀 수, 마스크에서 복원한 박스.
- **어떤 개념**: 개념 2.
- **출력**: `픽셀값: [0 1 2] | 사람 2 명 | 픽셀 수 [11241, 17721]`, 마스크 박스 `(159, 181, 301, 430)` · `(419, 170, 534, 485)` (직접 실행)

```python
fills = []
for p, g in zip(img_paths, gts):
    m = np.array(Image.open(mask_path(p)))
    for k, b in enumerate(g, start=1):
        x1, y1, x2, y2 = (b - 1).astype(int)              # 1부터 센 좌표 → 0부터
        box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        fills.append((m[y1:y2 + 1, x1:x2 + 1] == k).sum() / box_area)
all_boxes = np.concatenate(gts)
ar = (all_boxes[:, 3] - all_boxes[:, 1]) / (all_boxes[:, 2] - all_boxes[:, 0])   # h / w
ar_rot = 1 / ar                                           # 90° 돌리면 가로 · 세로가 바뀜
```
- **무엇을 하나**: 정답 박스마다 사람 픽셀 비율과 종횡비, 90° 돌렸을 때(역수)의 종횡비.
- **어떤 개념**: 개념 3.
- **출력**: `423명 | 박스 중 사람 픽셀 비율 평균 0.5036 (최소 0.27, 최대 0.78)`, `서 있음 h/w 평균 2.77 (최소 1.59) | 90° 회전 h/w 평균 0.38`, h/w < 1 인 서 있는 사람 0명 (직접 실행)

#### 한 번에 정리
- 박스 = 어디에 · 몇 개, 마스크 = 모양 · 넓이 · 경계, 관절 = 자세.
- 마스크 PNG: 0 = 배경, k = k번째 사람 → `m0 == k`. 마스크 → 박스는 min/max 로 바로.
- 박스 안은 평균 50% 가 배경, h/w 는 회전하면 2.77 → 0.38 로 뒤집힌다.
- ⚠ "더 복잡한 태스크가 더 좋다" 가 아니다 — 4교시에서 필요한 출력부터 고른다.

**연관 문제**
<<QA 1>>

### 2교시 · 세그멘테이션: Encoder-Decoder · mIoU

**키워드**: `semantic vs instance`, `Encoder-Decoder`, `skip connection`, `U-Net`, `Transposed Conv`, `(n−1)s − 2p + k`, `bilinear upsample`, `IoU · mIoU`, `0.5833`, `YOLOv8n-seg`, `prototype 32장`, `fillPoly`, `DeepLabV3`

**개요**: ① **semantic** = 픽셀마다 클래스 (같은 클래스의 두 사람 구분 안 함), **instance** = 물체마다 마스크. ② 구조는 **Encoder-Decoder** — 인코더(Day 2~3 의 CNN)가 해상도를 줄이며 "무엇", 디코더가 해상도를 키우며 "어디" 를 복원. U-Net 은 같은 해상도 특징을 디코더로 직접 넘기는 skip connection 으로 경계를 살린다. ③ 해상도 키우기: **bilinear upsample** (파라미터 없음) 또는 **Transposed Conv** (학습). ④ 평가는 **mIoU** — IoU 를 픽셀 집합으로. ⑤ YOLOv8-seg 는 탐지 head 에 마스크 계수 32개를 더한다.

**핵심 용어**
- **semantic / instance 세그멘테이션**: 픽셀마다 클래스(사람 구분 X) / 물체마다 마스크 + 클래스 + 점수
- **인코더-디코더(Encoder-Decoder)**: 줄이며 특징 추출 → 키우며 픽셀 단위 복원
- **skip connection**: 인코더 특징을 같은 해상도의 디코더로 바로 전달 — 경계 보존
- **Transposed Conv**: 학습하는 업샘플링 (`nn.ConvTranspose2d`)
- **mIoU**: 클래스마다 픽셀 IoU 를 구해 평균 (배경도 한 클래스)
- **prototype 마스크**: YOLOv8-seg 가 이미지마다 만드는 160×160 마스크 32장 — 계수로 섞어 물체 마스크를 만듦

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `tconv_out()` | `tconv_out(n, k, s, p)` | Transposed Conv 출력 크기 공식 |
| `nn.ConvTranspose2d()` | `nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)` | 학습하는 업샘플링 |
| `nn.Upsample()` | `nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)` | 학습 없는 업샘플링 |
| `miou()` | `miou(pred, gt, n_cls=2)` | 클래스별 IoU 와 평균 |
| `seg_person_mask()` | `seg_person_mask(model, img_bgr, conf=0.25)` | YOLOv8-seg 결과 → 사람 마스크(0/1) |
| `cv2.fillPoly()` | `cv2.fillPoly(pm, [poly.astype(np.int32)], 1)` | 외곽선 다각형 안을 값으로 채우기 |
| `YOLO()` | `YOLO("yolov8n-seg.pt")` | ultralytics 모델 불러오기 |
| `model.predict()` | `model.predict(img, conf=0.25, classes=[0], verbose=False)` | 추론 — `r.masks.xy` 가 외곽선 목록 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 2-1 | Transposed Conv 공식 vs 실제 출력 (k, s, p 4가지) · bilinear ×2 |
| 2-2 | `miou` — 4×4 손 계산 예제 |
| 2-3 | YOLOv8n-seg 로 첫 사진 사람 마스크 → 정답 마스크와 IoU |
| 2-4 | 170장 전체 mIoU — 교집합 · 합집합을 누적 (+ ms/장) |
| (선택) | DeepLabV3-MobileNetV3 — 21 클래스, 15 = person (채점 무관) |

**핵심 내용**

#### Encoder-Decoder 와 업샘플링
> 줄이면서 "무엇" 을 뽑고(인코더), 키우면서 "어디" 를 픽셀 단위로 복원(디코더) — 출력 = 입력과 같은 크기의 라벨 지도
- **skip connection** → 인코더의 같은 해상도 특징을 디코더에 이어 붙여 경계를 살림 (U-Net)
- **업샘플 두 가지** → bilinear (학습 파라미터 없음, 뒤에 Conv) · Transposed Conv (학습)
- **식** → Transposed Conv 출력 = (n − 1)·s − 2p + k
- **예** → n = 8: (k4, s2, p1) → 16 · (k2, s2, p0) → 16 · (k3, s2, p1) → 15 · (k3, s1, p1) → 8
- ⚠ (k3, s2, p1) 는 15 — 깔끔한 2배가 아니다. 2배로 키우려면 (4, 2, 1) 이나 (2, 2, 0)

#### mIoU — 픽셀 집합의 IoU
> IoU = |A ∩ B| / |A ∪ B| (픽셀 수) 를 클래스마다 구해 평균 — 배경도 한 클래스
- **예** → 4×4: 정답 사람 = 왼쪽 2열(8픽셀), 예측 = 왼쪽 3열(12픽셀) → 사람 IoU 8/12 = 0.6667, 배경 4/8 = 0.5 → mIoU **0.5833**
- **데이터 전체** → 이미지마다 IoU 를 평균하지 말고 교집합 · 합집합 픽셀을 누적한 뒤 나눈다
- ⚠ 배경이 넓어 배경 IoU 는 높게 나오기 쉽다 (YOLOv8n-seg 170장: 배경 0.94 · 사람 0.77 → mIoU 0.8553) — 사람 IoU 를 따로 본다

#### YOLOv8-seg — 마스크 계수 32개
> 탐지 head 가 칸마다 박스 · 클래스 + 계수 32개를 내고, 이미지 전체의 prototype 32장을 계수로 섞어(선형결합 → sigmoid) 물체별 마스크를 만든다
- **흐름** → prototype(160×160 × 32장) × 계수 → sigmoid → 박스 밖 지움
- **쓰기** → `r.masks.xy` = 원본 좌표 외곽선 목록 → `cv2.fillPoly` 로 원본 크기 마스크
- ⚠ `r.masks is None` (사람 0명) 을 먼저 확인

**기본 사용형태**
```python
def miou(pred, gt, n_cls=2):
    ious = []
    for c in range(n_cls):
        p, g = (pred == c), (gt == c)
        union = (p | g).sum()
        ious.append((p & g).sum() / union if union > 0 else np.nan)
    return np.array(ious), float(np.nanmean(ious))
```

**연결**
- [VB D4·3교시 IoU](#/vb/4/3) — 박스 IoU → 픽셀 IoU
- [VB D4·6교시 YOLO](#/vb/4/6) — seg 는 같은 탐지 head 에 마스크 계수를 더한 것
- [VB D2·1교시 CNN 구조](#/vb/2/1) — 인코더 = 해상도를 줄이는 CNN
- [VB D3·1교시 정확도 vs 속도](#/vb/3/1) — DeepLabV3-MobileNetV3 의 경량 backbone

**상세 학습**

#### 이 교시에서 답하는 질문
1교시에서 박스가 경계 · 넓이에 약하다는 걸 봤다. 그러면 **픽셀 단위 출력은 어떻게 만들고**, 잘 맞았는지는 **어떻게 점수 매기나**? 분류 CNN 은 해상도를 줄이기만 하므로 다시 키우는 디코더가 필요하고, 박스 IoU 대신 **픽셀 집합의 IoU (mIoU)** 가 필요하다. 마지막으로 실제 모델(YOLOv8n-seg)로 Penn-Fudan 170장의 mIoU 를 잰다.

#### 개념 1 · semantic vs instance
| 종류 | 출력 | 두 사람이 붙어 있으면 | 대표 모델 |
|---|---|---|---|
| semantic | 픽셀마다 클래스 | 하나의 "사람" 영역 | FCN · U-Net · DeepLabV3 |
| instance | 물체마다 마스크 + 클래스 + 점수 | 사람 1 · 사람 2 따로 | Mask R-CNN · YOLOv8-seg |

#### 개념 2 · Encoder-Decoder 와 업샘플링
![Encoder-Decoder(U-Net) 구조 — 인코더가 해상도를 줄이고 디코더가 키우며 skip connection 으로 경계를 살린다. Transposed Conv 출력 크기 공식](assets/fig/vb/d5-encdec.svg)

- **인코더** = Day 2~3 의 CNN. 해상도를 줄이며 "사람이다" 라는 뜻을 얻지만 위치는 흐려진다.
- **디코더** = 해상도를 다시 입력 크기까지 키운다.
- **skip connection** (U-Net) = 인코더의 같은 해상도 특징을 디코더에 이어 붙인다 — 흐려진 경계를 복원 (ResNet 의 skip 은 더하기, U-Net 은 이어 붙이기).

| 업샘플 | 학습? | 특징 |
|---|---|---|
| bilinear (`nn.Upsample`) | X | 주변 4픽셀의 가중 평균 — 보통 뒤에 Conv 를 붙인다 |
| Transposed Conv (`nn.ConvTranspose2d`) | O | 출력 = **(n − 1)·s − 2p + k** — 일반 Conv 크기 공식을 거꾸로 |

n = 8 에서: (k4, s2, p1) → 7×2 − 2 + 4 = **16**, (k2, s2, p0) → 16, (k3, s2, p1) → **15** (2배가 아님), (k3, s1, p1) → 8.

#### 개념 3 · mIoU — 픽셀 집합의 IoU
![4×4 예제의 mIoU 손 계산 — 사람 IoU 8/12, 배경 IoU 4/8, 평균 0.5833](assets/fig/vb/d5-miou.svg)

| 기호 | 뜻 |
|---|---|
| IoU_c | 클래스 c 의 \|정답 ∩ 예측\| / \|정답 ∪ 예측\| (픽셀 수) |
| mIoU | 모든 클래스(배경 포함) IoU_c 의 평균 |
| 합집합 0 | 그 클래스가 정답 · 예측 어디에도 없음 → `nan` 으로 두고 `nanmean` |

**데이터 전체**의 mIoU 는 사진마다 IoU 를 구해 평균하지 않는다. 클래스별 교집합 · 합집합 **픽셀 수를 누적**한 뒤 마지막에 나눈다 — 큰 사람과 작은 사람이 픽셀 수만큼 반영된다.

#### 개념 4 · YOLOv8-seg — 계수 32개로 마스크 만들기
- 탐지 head 가 칸마다 박스 4 + 클래스 80 에 **마스크 계수 32개**를 더 낸다 (4교시: 116 채널).
- 이미지마다 **prototype 마스크 32장 (160×160)** 을 따로 만든다.
- 물체 마스크 = 32장을 그 물체의 계수로 섞기(선형결합) → sigmoid → 박스 밖은 지움.
- ultralytics 결과의 `r.masks.xy` 는 **원본 좌표 외곽선(다각형) 목록** → `cv2.fillPoly` 로 원본 크기 0/1 마스크.

| 실험 (Penn-Fudan) | 배경 IoU | 사람 IoU | mIoU |
|---|---|---|---|
| 첫 사진 (인스턴스 2개) | 0.9869 | 0.8896 | 0.9383 |
| 170장 누적 | 0.9425 | 0.7682 | **0.8553** (CPU 274 ms/장) |

선택 실습 DeepLabV3-MobileNetV3 는 semantic 모델로, 픽셀마다 **21 클래스**(VOC) 점수를 내고 `argmax == 15` 가 사람이다.

#### 코드로 확인
```python
def tconv_out(n, k, s, p):
    return (n - 1) * s - 2 * p + k
x8 = torch.randn(1, 64, 8, 8)
for k, s, p in [(4, 2, 1), (2, 2, 0), (3, 2, 1), (3, 1, 1)]:
    y = nn.ConvTranspose2d(64, 32, kernel_size=k, stride=s, padding=p)(x8)
    print(f"k={k} s={s} p={p}: 공식 {tconv_out(8, k, s, p)} | 실제 {tuple(y.shape[2:])}")
```
- **무엇을 하나**: 공식과 실제 `ConvTranspose2d` 출력 크기 비교.
- **어떤 개념**: 개념 2.
- **출력**: `16 | (16, 16)`, `16 | (16, 16)`, `15 | (15, 15)`, `8 | (8, 8)` (직접 실행)

```python
def seg_person_mask(model, img_bgr, conf=0.25):
    H, W = img_bgr.shape[:2]
    r = model.predict(img_bgr, conf=conf, classes=[0], verbose=False)[0]
    pm = np.zeros((H, W), np.uint8)
    if r.masks is not None:
        for poly in r.masks.xy:                     # 사람마다 외곽선 (원본 좌표)
            if len(poly):
                cv2.fillPoly(pm, [poly.astype(np.int32)], 1)
    return pm, (0 if r.masks is None else len(r.masks))
```
- **무엇을 하나**: YOLOv8n-seg 결과를 원본 크기 사람 마스크(0/1)로.
- **어떤 개념**: 개념 4.
- **출력**: 첫 사진 `인스턴스 2개 | 배경 IoU 0.9869 · 사람 IoU 0.8896 · mIoU 0.9383` (직접 실행)

```python
inter_sum, union_sum = np.zeros(2), np.zeros(2)
for p in img_paths:
    pm, _ = seg_person_mask(seg, cv2.imread(p))
    g = (np.array(Image.open(mask_path(p))) > 0).astype(int)
    for c in range(2):
        inter_sum[c] += ((pm == c) & (g == c)).sum()
        union_sum[c] += ((pm == c) | (g == c)).sum()
seg_ious = inter_sum / union_sum
```
- **무엇을 하나**: 170장의 교집합 · 합집합을 누적해 데이터 전체 IoU.
- **어떤 개념**: 개념 3.
- **출력**: `배경 IoU 0.9425 · 사람 IoU 0.7682 → mIoU 0.8553 | 274 ms/장 (cpu)` (직접 실행)

#### 한 번에 정리
- semantic = 픽셀마다 클래스, instance = 물체마다 마스크.
- Encoder-Decoder + skip, 업샘플 = bilinear 또는 Transposed Conv (n−1)s−2p+k.
- mIoU = 클래스별 픽셀 IoU 평균 (4×4 예제 0.5833), 데이터 전체는 교집합 · 합집합 누적.
- YOLOv8-seg = prototype 32장 × 계수 32개, `r.masks.xy` → `fillPoly`.
- ⚠ 배경은 넓어서 IoU 가 높게 나온다 (0.94 vs 사람 0.77) — 사람 IoU 를 따로 보고, 마스크는 박스 안에서만 만들어지므로 박스를 놓치면 마스크도 없다.

**연관 문제**
<<QA 2>>

### 3교시 · 포즈 추정: 관절 · 쓰러짐 규칙

**키워드**: `COCO 17 관절`, `top-down · bottom-up · 1-stage`, `heatmap vs 직접 회귀`, `YOLOv8n-pose`, `r.keypoints.xy · conf`, `관절 각도 arccos`, `body_tilt`, `is_fallen`, `임계 60°`, `conf 0.5`

**개요**: **COCO 17 관절**: 0 코 · 1–2 눈 · 3–4 귀 · 5–6 어깨 · 7–8 팔꿈치 · 9–10 손목 · 11–12 엉덩이 · 13–14 무릎 · 15–16 발목 (홀수 = 왼쪽, 짝수 = 오른쪽). 방식은 ① top-down (박스 먼저 → 박스마다 관절, 정확하지만 사람 수만큼 느림) ② bottom-up (관절 먼저 → 사람별로 묶기, OpenPose) ③ 1-stage (YOLOv8-pose, 한 번에). 위치를 찾는 법은 heatmap(확률 지도의 최댓값) 또는 직접 회귀(좌표 숫자). 관절 좌표로 각도를 구하면 **규칙 기반 쓰러짐 감지**를 만들 수 있다.

**핵심 용어**
- **키포인트(keypoint)**: 몸의 관절 위치 (x, y) + 신뢰도 conf
- **top-down / bottom-up / 1-stage**: 박스→관절 / 관절→묶기 / 한 번에
- **heatmap vs 회귀**: 관절마다 확률 지도에서 최댓값 / 좌표를 숫자로 직접 예측
- **몸통 기울기(body tilt)**: 엉덩이 중점에서 위쪽 수직과 어깨 중점 사이의 각

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `run_pose()` | `run_pose(model, img_bgr, conf=0.25)` | 관절 좌표 · 신뢰도 · 박스를 numpy 로 |
| `angle()` | `angle(a, b, c)` | 점 b 에서 a-b-c 가 이루는 각도(°) |
| `body_tilt()` | `body_tilt(kpts)` | 몸통과 수직선 사이 각도 |
| `is_fallen()` | `is_fallen(kpts, conf, thr=60, min_conf=0.5)` | 쓰러짐 판정 (신뢰 낮으면 `None`) |
| `biggest()` | `biggest(boxes)` | 가장 큰 박스 번호 (주인공) |
| `draw_pose()` | `draw_pose(img, kpts, confs, min_conf=0.5)` | 뼈대 · 관절 그리기 |
| `np.arccos()` | `np.degrees(np.arccos(cos))` | 코사인 → 각도 |
| `cv2.rotate()` | `cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` | 이미지 90° 회전 (쓰러진 사람 흉내) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 3-1 | YOLOv8n-pose 로 사진 · 영상 프레임 200 의 관절 `(사람, 17, 2)` 와 conf |
| 3-2 | `angle` — 직각 90° · 일직선 180°, 첫 사람 무릎 각도 |
| 3-3 | `body_tilt` · `is_fallen` — 원본(서 있음) vs 90° 회전(쓰러짐) |

**핵심 내용**

#### COCO 17 관절과 출력 모양
> 관절 17개 × (x, y) 좌표와 conf — 번호만 외우면 어떤 규칙이든 만들 수 있다
- **번호** → 어깨 5 · 6, 엉덩이 11 · 12, 무릎 13 · 14, 발목 15 · 16
- **출력** → `r.keypoints.xy` `(사람, 17, 2)` · `r.keypoints.conf` `(사람, 17)`
- **세 방식** → top-down(정확 · 사람 수만큼 느림) · bottom-up(사람 수 무관) · 1-stage YOLOv8-pose(빠름)
- ⚠ 홀수 = 왼쪽은 **사진 속 사람 기준** — 화면 왼쪽이 아니다

#### 관절 각도
> 각도 = arccos( BA·BC / (|BA||BC|) ) — 점 b 가 꼭짓점, a-b-c 가 이루는 각
- **예** → `angle((1,0),(0,0),(0,1))` = 90°, 일직선 `(0,0),(0,1),(0,2)` = 180°
- **쓰는 곳** → 무릎 각도(11-13-15) 로 앉음 · 섬 판별 등
- ⚠ `np.clip(cos, -1, 1)` 필수 — 부동소수 오차로 1.0000001 이 되면 `arccos` 가 nan

#### 몸통 기울기로 쓰러짐 판정
> 기울기 = 엉덩이 중점에서 (위쪽 수직선) 과 (어깨 중점) 이 이루는 각 — 60° 초과면 쓰러짐
- **식** → `shoulder = (kpts[5]+kpts[6])/2`, `hip = (kpts[11]+kpts[12])/2`, `angle(hip + [0,-1], hip, shoulder)`
- **예** → 똑바로 0° · 누우면 90° · 어깨 (160, 70)·엉덩이 (100, 150) 면 36.9° (직접 계산) — 실제 사진 3.3° → 90° 회전 84.2°
- **신뢰도 확인** → 어깨 · 엉덩이 conf 가 0.5 미만이면 `None` (판정 보류)
- ⚠ 이미지 y 축은 아래가 +, 그래서 위쪽 수직 = `[0, -1]` 이다

**기본 사용형태**
```python
def body_tilt(kpts):
    shoulder = (kpts[5] + kpts[6]) / 2
    hip = (kpts[11] + kpts[12]) / 2
    return angle(hip + np.array([0, -1]), hip, shoulder)

def is_fallen(kpts, conf, thr=60, min_conf=0.5):
    if conf[[5, 6, 11, 12]].min() < min_conf:
        return None
    return body_tilt(kpts) > thr
```

**연결**
- [VB D5·1교시 박스의 한계](#/vb/5/1) — 박스 종횡비로는 안 되던 쓰러짐
- [VB D4·6교시 YOLO](#/vb/4/6) — pose 도 같은 YOLO head 의 변형
- [NN D1·3교시 NumPy ndarray](#/nn/1/3) — 점 연산 (벡터 · 내적 · norm)

**상세 학습**

#### 이 교시에서 답하는 질문
마스크는 모양을 주지만 "서 있나 누웠나" 는 바로 알려 주지 않고, 1교시에서 본 것처럼 박스 비율은 카메라 방향에 속는다. **관절 좌표**가 있으면 몸의 방향을 기하(각도)로 직접 계산할 수 있다. 이 교시는 관절 번호 · 출력 모양 → 각도 공식 → 쓰러짐 규칙 순으로 쌓고, 사진을 90° 돌려 규칙을 시험한다.

#### 개념 1 · COCO 17 관절과 YOLOv8-pose 출력
![COCO 17 관절 번호와 몸통 기울기 규칙 — 엉덩이 중점에서 위쪽 수직과 어깨 중점 사이 각이 60° 를 넘으면 쓰러짐](assets/fig/vb/d5-pose.svg)

| 번호 | 관절 | 번호 | 관절 |
|---|---|---|---|
| 0 | 코 | 9 · 10 | 손목 |
| 1 · 2 / 3 · 4 | 눈 / 귀 | 11 · 12 | **엉덩이** |
| 5 · 6 | **어깨** | 13 · 14 | 무릎 |
| 7 · 8 | 팔꿈치 | 15 · 16 | 발목 |

홀수 = 왼쪽, 짝수 = 오른쪽 — **사진 속 사람의** 왼쪽이다. 출력은 `r.keypoints.xy` `(사람, 17, 2)` 와 `r.keypoints.conf` `(사람, 17)`. 사진 한 장에서 `(2, 17, 2)`, 영상 프레임 200 에서는 사람이 없어 `(0, 17, 2)`.

#### 개념 2 · 포즈 추정 방식
| 방식 | 순서 | 장단점 |
|---|---|---|
| top-down | 사람 박스 → 박스마다 관절 | 정확, 사람 수만큼 느려짐 |
| bottom-up (OpenPose) | 모든 관절 → 사람별로 묶기 | 사람 수와 무관한 속도 |
| 1-stage (YOLOv8-pose) | 박스와 관절을 한 번에 | 빠름 — 오늘 사용 |

위치를 찾는 법도 두 가지: **heatmap** (관절마다 확률 지도를 그리고 최댓값 위치) vs **직접 회귀** (좌표 숫자를 바로 예측, YOLOv8-pose).

#### 개념 3 · 관절 각도 — 내적으로
점 b 를 꼭짓점으로 a-b-c 가 이루는 각:

| 단계 | 식 |
|---|---|
| 벡터 | BA = a − b, BC = c − b |
| 코사인 | cos θ = BA·BC / (\|BA\| \|BC\|) — 분모에 `+ 1e-9` (길이 0 방어) |
| 각도 | θ = degrees(arccos(clip(cos, −1, 1))) |

시험: `angle((1,0),(0,0),(0,1))` = **90°**, 일직선 `(0,0),(0,1),(0,2)` = **180°** (실행값 179.997 — 1e-9 때문). 첫 사람 무릎 각도 (11-13-15) 는 왼쪽 179.7° · 오른쪽 177.1° → 다리를 편 채 서 있다.

#### 개념 4 · 몸통 기울기 → 쓰러짐 규칙
- 어깨 중점 = (kpts[5] + kpts[6]) / 2, 엉덩이 중점 = (kpts[11] + kpts[12]) / 2
- 기울기 = 엉덩이 중점에서 **위쪽 수직**(`[0, −1]`, 이미지 y 축은 아래가 +) 과 어깨 중점이 이루는 각
- **기울기 > 60° → 쓰러짐**. 단, 어깨 · 엉덩이 4개 관절 중 하나라도 conf < 0.5 면 `None` (판정 보류)

| 상황 | 기울기 | 판정 |
|---|---|---|
| 원본 사진 (서 있음) | **3.3°** | False |
| 같은 사진 90° 회전 (누운 것처럼) | **84.2°** | True |
| 손 계산: 어깨 (160, 70) · 엉덩이 (100, 150) | 36.9° | False (숙임) |

임계값 60° 는 선택이다 — 낮추면 허리 숙인 사람도 경보, 높이면 비스듬히 넘어진 사람을 놓친다.

#### 코드로 확인
```python
def angle(a, b, c):
    ba, bc = np.asarray(a, float) - b, np.asarray(c, float) - b
    cos = ba @ bc / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
print(angle((1, 0), (0, 0), (0, 1)), angle((0, 0), (0, 1), (0, 2)))
p0 = k0[0]
print(f"왼쪽 무릎 {angle(p0[11], p0[13], p0[15]):.1f}° · 오른쪽 {angle(p0[12], p0[14], p0[16]):.1f}°")
```
- **무엇을 하나**: 각도 함수를 직각 · 일직선으로 시험하고, 첫 사람의 무릎 각도.
- **어떤 개념**: 개념 3.
- **출력**: `90.0 179.997…`, `왼쪽 무릎 179.7° · 오른쪽 177.1°` (직접 실행)

```python
img_rot = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
kr, cr, br = run_pose(pose, img_rot)
for name, (k, c, b) in [("원본", (k0, c0, b0)), ("90° 회전", (kr, cr, br))]:
    i = biggest(b)                                  # 가장 큰 사람 = 주인공
    print(f"{name}: 기울기 {body_tilt(k[i]):.1f}° → 쓰러짐 {is_fallen(k[i], c[i])}")
```
- **무엇을 하나**: 같은 사진을 90° 돌려 "누운 사람" 을 흉내 내고 쓰러짐 판정.
- **어떤 개념**: 개념 4.
- **출력**: `원본: 기울기 3.3° → 쓰러짐 False`, `90° 회전: 기울기 84.2° → 쓰러짐 True` (직접 실행)

#### 한 번에 정리
- COCO 17: 어깨 5·6, 엉덩이 11·12, 무릎 13·14, 발목 15·16 (홀수 = 왼쪽).
- 방식 3가지 (top-down · bottom-up · 1-stage), 위치 찾기 2가지 (heatmap · 회귀).
- 각도 = arccos(내적 / 길이 곱), 몸통 기울기 > 60° → 쓰러짐 (3.3° vs 84.2°).
- ⚠ 관절 conf 가 낮으면 `None` — 가려진 사람을 쓰러짐으로 오판하지 않게. `np.clip` 을 빼면 arccos 가 nan.

**연관 문제**
<<QA 3>>

### 4교시 · 비전 태스크 정리: 출력 텐서 · 시나리오 매칭

**키워드**: `같은 backbone · neck, 다른 head`, `8,400 칸`, `84 · 116 · 56`, `4 + 80 + 32`, `4 + 1 + 17×3`, `3,157,200 · 3,409,968 · 3,295,470`, `시나리오 → 태스크`

**개요**: YOLOv8n 의 세 모델(탐지 · seg · pose) 은 backbone · neck 이 같고 head 만 다르다. 칸 수는 모두 8,400 (80² + 40² + 20²) 이고 **칸마다 내는 숫자**만 다르다. 태스크는 "필요한 출력" 으로 고르고, 가장 단순한 것부터 쓴다.

**핵심 용어**
- **head**: 특징 지도를 최종 출력(박스 · 마스크 계수 · 관절)으로 바꾸는 마지막 부분
- **출력 채널(channel)**: 칸 하나가 내는 숫자 개수 — 탐지 84, seg 116, pose 56

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `n_channels()` | `n_channels(task, nc=80, nm=32, nk=17)` | 태스크별 칸당 출력 숫자 개수 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 4-1 | 세 모델에 640×640 영0 입력 → 원시 출력 모양, 공식과 비교 |
| 4-2 | 현장 시나리오 6개 → cls · det · seg · pose · det+roi 매칭 |

**핵심 내용**

#### 같은 몸통, 다른 head
> 칸 8,400 개는 같고, 칸마다 내는 숫자 = 박스 4 + (태스크별 나머지)
| 모델 | 파라미터 | 칸당 숫자 |
|---|---|---|
| yolov8n | 3,157,200 | 4 + 클래스 80 = **84** |
| yolov8n-seg | 3,409,968 | 4 + 80 + 마스크 계수 32 = **116** (+ 이미지마다 prototype 32×160×160) |
| yolov8n-pose | 3,295,470 | 4 + 사람 점수 1 + 17×(x, y, conf) = **56** |
- **8,400** → 80×80 + 40×40 + 20×20 = 6,400 + 1,600 + 400 (직접 계산)
- ⚠ pose 는 클래스 80개 대신 "사람 점수 1개" — 사람만 찾는다

#### 시나리오 → 태스크 고르기
> 필요한 출력이 무엇인가 — 있나 / 어디 · 몇 개 / 어떤 모양 / 어떤 자세 — 에서 가장 단순한 것
| 시나리오 | 태스크 |
|---|---|
| 제품 사진의 양품 / 불량 | cls (분류) |
| 작업장 출입 인원 세기 · 지게차-사람 거리 경고 | det (탐지) |
| 바닥 기름 영역 넓이 | seg |
| 작업자 쓰러짐 | pose |
| 크레인 아래 위험구역에 발을 들였나 | det + roi (7교시) |
- ⚠ 위험구역 침입은 세그 · 포즈가 필요 없다 — 박스 아래 중앙(발 위치) + 다각형이면 충분

**기본 사용형태**
```python
def n_channels(task, nc=80, nm=32, nk=17):
    if task == "detect":  return 4 + nc
    if task == "segment": return 4 + nc + nm
    if task == "pose":    return 4 + 1 + nk * 3
```

**연결**
- [VB D4·6교시 YOLO](#/vb/4/6) — 탐지 head 의 84 채널 (`decode_yolo`)
- [VB D5·2교시 YOLOv8-seg](#/vb/5/2) — +32 마스크 계수
- [VB D5·3교시 포즈](#/vb/5/3) — 17 × 3 관절
- [VB D5·7교시 침입 감지](#/vb/5/7) — det + roi

**상세 학습**

#### 이 교시에서 답하는 질문
2 · 3교시에서 세 가지 출력(박스 · 마스크 · 관절)을 배웠다. 세 모델은 얼마나 다르고, 현장 문제에는 어떤 걸 골라야 하나? 세 모델의 원시 출력을 나란히 놓으면 차이는 **"칸마다 숫자를 몇 개 더 내나"** 뿐이다. 그러니 고르는 기준은 성능이 아니라 **필요한 출력**이다.

#### 개념 1 · 같은 몸통, 다른 head
![YOLOv8n 탐지 · seg · pose 의 칸당 출력 숫자 84 · 116 · 56 과 파라미터 수](assets/fig/vb/d5-heads.svg)

| 모델 | 칸당 숫자 | 원시 출력 | 파라미터 |
|---|---|---|---|
| yolov8n | 4 + 80 = **84** | (1, 84, 8400) | 3,157,200 |
| yolov8n-seg | 4 + 80 + 32 = **116** | (1, 116, 8400) + prototype (1, 32, 160, 160) | 3,409,968 |
| yolov8n-pose | 4 + 1 + 17×3 = **56** | (1, 56, 8400) | 3,295,470 |

- **8,400 칸** = 80² + 40² + 20² (stride 8 · 16 · 32 의 세 특징 지도) — 세 모델 모두 같다.
- backbone · neck 이 같으므로 파라미터 차이도 head 만큼뿐이다 (seg +25만, pose +14만).
- pose 는 클래스 80개 대신 **사람 점수 1개** — 사람만 찾는 모델이다.

#### 개념 2 · 시나리오 → 태스크 고르기
질문 순서: **있나?** (cls) → **어디 · 몇 개?** (det) → **어떤 모양 · 넓이?** (seg) → **어떤 자세?** (pose). 답할 수 있는 것 중 **가장 단순한** 것을 고른다 — 단순할수록 빠르고, 라벨 만들기 · 검증도 쉽다.

| 현장 시나리오 | 태스크 | 이유 |
|---|---|---|
| 제품 사진 한 장의 양품 / 불량 | cls | 사진 전체에 답 하나 |
| 작업장 출입 인원 세기 | det | 몇 명 |
| 지게차와 사람 사이 거리 경고 | det | 박스 위치로 거리 |
| 바닥 기름 유출 영역의 넓이 | seg | 모양이 박스가 아님 |
| 작업자 쓰러짐 감지 | pose | 자세 |
| 크레인 아래 위험구역에 발을 들였나 | det + roi | 발 위치(박스 아래 중앙) + 다각형 — 7교시 |

#### 코드로 확인
```python
def n_channels(task, nc=80, nm=32, nk=17):
    if task == "detect":  return 4 + nc
    if task == "segment": return 4 + nc + nm
    if task == "pose":    return 4 + 1 + nk * 3
x640 = torch.zeros(1, 3, 640, 640)
for task, w in [("detect", "yolov8n.pt"), ("segment", "yolov8n-seg.pt"), ("pose", "yolov8n-pose.pt")]:
    with torch.no_grad():
        o = YOLO(w).model.eval()(x640)[0]
    o = o[0] if isinstance(o, (list, tuple)) else o       # seg 는 (출력, prototype)
    print(f"{task:8s} 원시 출력 {tuple(o.shape)} | 공식 {n_channels(task)}")
```
- **무엇을 하나**: 세 모델에 같은 빈 입력을 넣어 원시 출력 모양을 공식과 비교.
- **어떤 개념**: 개념 1.
- **출력**: `detect (1, 84, 8400) | 84`, `segment (1, 116, 8400) | 116`, `pose (1, 56, 8400) | 56` (직접 실행)

#### 한 번에 정리
- 같은 backbone · neck, 다른 head: 84 · 116(+ prototype) · 56, 칸은 모두 8,400.
- 태스크는 필요한 출력으로, 가장 단순한 것부터.
- 위험구역 침입은 세그 · 포즈가 필요 없다 — det + 다각형.
- ⚠ seg · pose 는 파라미터가 더 많고 후처리도 무겁다 — "되니까 쓴다" 가 아니라 "필요하니까 쓴다".

**연관 문제**
<<QA 4>>

### 5교시 · ONNX export · ONNX Runtime

**키워드**: `ONNX`, `계산 그래프 + 가중치`, `torch.onnx.export`, `tracing`, `input_names · output_names`, `dynamic_axes`, `opset 17`, `ONNX Runtime`, `InferenceSession`, `최대 차이 < 1e-4`, `yolo.export(format="onnx")`, `(1, 84, 8400)`

**개요**: **ONNX** 는 프레임워크에 묶이지 않은 모델 파일 형식(계산 그래프 + 가중치). PyTorch 에서 내보내면 ONNX Runtime · TensorRT(Jetson, 내일부터) · OpenVINO 가 읽는다. `torch.onnx.export` 는 예시 입력으로 모델을 한 번 돌리며 연산을 기록(**tracing**)한다. 변환 후에는 **반드시** 같은 입력으로 PyTorch 출력과 ONNX Runtime 출력을 비교한다. YOLO 의 ONNX 는 원시 출력 `(1, 84, 8400)` 만 주므로 letterbox · decode · NMS 는 직접 한다.

**핵심 용어**
- **ONNX(Open Neural Network Exchange)**: 프레임워크 독립 모델 형식
- **tracing**: 예시 입력으로 한 번 실행하며 연산을 기록해 그래프를 만드는 방식
- **dynamic_axes**: 바뀔 수 있는 축(배치 크기)을 지정
- **opset**: ONNX 연산자 버전 — 받는 쪽이 지원해야 함
- **ONNX Runtime**: ONNX 모델을 실행하는 추론 엔진 (CPU · 여러 장치)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `torch.onnx.export()` | `torch.onnx.export(model, x, path, input_names=["input"], output_names=["logits"], dynamic_axes={...}, opset_version=17)` | PyTorch → ONNX 파일 |
| `export_onnx()` | `export_onnx(model, path, input_size=(1, 3, 224, 224))` | 내보내기 + `check_model` |
| `onnx.load()` | `onnx.load(path)` | ONNX 파일 읽기 (`graph.node` · `graph.input`) |
| `onnx.checker.check_model()` | `onnx.checker.check_model(m)` | 그래프가 올바른지 검사 |
| `ort.InferenceSession()` | `ort.InferenceSession("m.onnx", providers=["CPUExecutionProvider"])` | ONNX Runtime 세션 만들기 |
| `sess.run()` | `sess.run(None, {"input": x})` | 추론 — `None` = 출력 전부, 입력은 이름으로 |
| `yolo.export()` | `yolo.export(format="onnx", imgsz=640, opset=17)` | ultralytics 의 ONNX 변환 |
| `copy.deepcopy()` | `copy.deepcopy(model).cpu().eval()` | 원본 모델을 건드리지 않고 복사 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 5-1 | MobileNetV3-Small → `mnv3s.onnx`, 노드 159개, 연산 종류, 파일 크기 ≈ 파라미터 × 4 B |
| 5-2 | 배치 4 로 PyTorch vs ONNX Runtime 출력의 최대 차이 (< 1e-4) |
| 5-3 | YOLOv8n → ONNX, 원시 출력 `(1, 84, 8400)` → letterbox · `decode_yolo` 로 PyTorch 결과와 일치 확인 (짝 IoU > 0.99) |

**핵심 내용**

#### ONNX export 의 인자
> 예시 입력으로 모델을 한 번 돌려 계산 그래프를 기록(tracing)해 파일로 저장
| 인자 | 뜻 |
|---|---|
| 예시 입력 | 입력 모양을 정해 주는 텐서 `(1, 3, 224, 224)` |
| `input_names` · `output_names` | 그래프 입출력 이름 — ONNX Runtime 에서 이 이름으로 넣고 뺌 |
| `dynamic_axes` | 바뀔 축 (배치) — 안 주면 배치 1 로 굳음 |
| `opset_version` | 연산자 버전 (17) |
- **모델 준비** → `.eval()` 로 (BatchNorm · Dropout 추론 모드) 그리고 `.cpu()`
- **예** → MobileNetV3-S: 노드 159개, `HardSwish` 연산 포함, 파일 크기 ≈ 파라미터 수 × 4 B (float32, 오차 0.3 MB 이내)
- ⚠ tracing 은 예시 입력 경로만 기록 — 입력에 따라 갈라지는 if 는 고정될 수 있다

#### 변환 후 검증 — 최대 차이
> 같은 입력을 PyTorch 와 ONNX Runtime 에 넣어 출력의 최대 절대 차이 (`np.abs(a - b).max()`) 를 본다
- **기준** → 분류 로짓 최대 차이 < 1e-4 (MobileNetV3-S 배치 4 실행: 1.48e-05)
- **YOLO** → 원시 출력은 `(1, 84, 8400)`. 후처리(letterbox · decode_yolo · NMS)는 직접 해서 PyTorch 와 짝 IoU 확인
- ⚠ 변환이 "에러 없이 끝났다" 가 "같은 결과" 는 아니다 — 꼭 비교

**기본 사용형태**
```python
def export_onnx(model, path, input_size=(1, 3, 224, 224)):
    model = copy.deepcopy(model).cpu().eval()
    torch.onnx.export(model, torch.randn(input_size), path,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17)
    m = onnx.load(path); onnx.checker.check_model(m)
    return m

sess = ort.InferenceSession("mnv3s.onnx", providers=["CPUExecutionProvider"])
out = sess.run(None, {"input": x.numpy()})[0]
```

**연결**
- [VB D3·4교시 MobileNetV3](#/vb/3/4) — 오늘 내보내는 모델
- [VB D4·6교시 YOLO](#/vb/4/6) — `letterbox` · `decode_yolo` · 원시 출력 `(1, 84, 8400)`
- [VB D5·6교시 성능 측정](#/vb/5/6) — PyTorch vs ONNX Runtime 속도 비교
- [NN D2·3교시 순전파 · 역전파](#/nn/2/3) — 계산 그래프라는 개념

**상세 학습**

#### 이 교시에서 답하는 질문
PyTorch 모델은 파이썬과 PyTorch 가 있어야 돈다. Jetson 같은 작은 장치나 다른 언어 프로그램에서 쓰려면 **프레임워크와 분리된 모델 파일**이 필요하다. 이 교시는 모델을 **ONNX** 로 내보내고, 다른 엔진(**ONNX Runtime**)에서 **같은 결과**가 나오는지 숫자로 검증한다. 다음 과목(ONNX · TensorRT)의 출발점이다.

#### 개념 1 · ONNX = 계산 그래프 + 가중치
![PyTorch 모델을 torch.onnx.export 로 ONNX 파일로 만들고 ONNX Runtime · TensorRT · OpenVINO 에서 실행하는 흐름과 검증 결과](assets/fig/vb/d5-onnx.svg)

- `torch.onnx.export` 는 **예시 입력으로 모델을 한 번 돌리며** 지나간 연산을 기록한다 (**tracing**).
- 결과 파일에는 연산 노드(Conv · HardSwish · Gemm …)의 그래프와 가중치가 들어 있다. MobileNetV3-S: 노드 **159개**.
- 파일 크기 ≈ 파라미터 수 × 4 B (float32): 2,542,856 × 4 = 10.17 MB → 실제 **10.18 MB**.
- 받는 쪽(ONNX Runtime · TensorRT · OpenVINO)은 PyTorch 없이 그래프를 읽어 실행한다.

#### 개념 2 · export 인자
| 인자 | 뜻 | 빼먹으면 |
|---|---|---|
| 예시 입력 `torch.randn(1, 3, 224, 224)` | 입력 모양을 정해 줌 | — |
| `input_names` · `output_names` | 그래프 입출력 이름 → `sess.run(None, {"input": x})` 의 키 | 자동 이름 (`input.1` 등) |
| `dynamic_axes={"input": {0: "batch"}, …}` | 0번 축을 `batch` 라는 변수로 | 배치 1 로 굳어서 배치 4 를 넣으면 에러 |
| `opset_version=17` | ONNX 연산자 버전 | 받는 엔진이 지원하는 버전이어야 함 |
| `dynamo=False` | 예전 tracing 방식 사용 (강의 환경) | — |
| `model.eval()` · `.cpu()` | BatchNorm · Dropout 을 추론 모드로 | 학습 모드 그래프가 기록됨 |

⚠ tracing 은 **예시 입력이 지나간 길만** 기록한다 — 입력 값에 따라 갈라지는 `if` 는 한쪽으로 굳는다.

#### 개념 3 · 변환 후 검증
"에러 없이 끝났다" ≠ "같은 결과". 같은 입력을 PyTorch 와 ONNX Runtime 에 넣고 **최대 절대 차이** `np.abs(a − b).max()` 를 본다.

| 모델 | 비교 | 결과 (실행) |
|---|---|---|
| MobileNetV3-S | 배치 4 로짓 (4, 1000) | 최대 차이 **1.48e-05** < 1e-4 ✔ (입력 `['batch', 3, 224, 224]`) |
| YOLOv8n | `yolo.export(format="onnx", imgsz=640, opset=17)` | 입력 `images` (1, 3, 640, 640), 출력 `output0` (1, 84, 8400) |
| YOLOv8n | 사진 한 장 → letterbox → ONNX → `decode_yolo` | ONNX 7명 · PyTorch 7명, 짝 IoU 최소 **1.0000** |

YOLO 의 ONNX 는 **원시 출력만** 준다. letterbox · decode(전치 → conf 필터 → xywh→xyxy → NMS) · 원본 좌표 되돌리기는 Day 4 의 함수를 그대로 쓴다.

#### 코드로 확인
```python
def export_onnx(model, path, input_size=(1, 3, 224, 224)):
    model = copy.deepcopy(model).cpu().eval()          # 원본은 그대로 두고 복사본으로
    torch.onnx.export(model, torch.randn(input_size), path,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    m = onnx.load(path); onnx.checker.check_model(m)
    return m
```
- **무엇을 하나**: MobileNetV3-S 를 ONNX 로 내보내고 그래프 검사.
- **어떤 개념**: 개념 1 · 2.
- **출력**: `노드 159개 | 'HardSwish' 포함 | 파일 10.2 MB ≈ 파라미터 × 4 B`, 입력 0번 축 `dim_param = 'batch'` (교안 검증 셀)

```python
sess = ort.InferenceSession("mnv3s.onnx", providers=["CPUExecutionProvider"])
xb = torch.randn(4, 3, 224, 224)
with torch.no_grad():
    ref = mnv3(xb).numpy()
onx = sess.run(None, {"input": xb.numpy()})[0]     # None = 출력 전부, 입력은 이름으로
print([(i.name, i.shape) for i in sess.get_inputs()], onx.shape, float(np.abs(onx - ref).max()))
```
- **무엇을 하나**: 배치 4 로 PyTorch · ONNX Runtime 출력 비교 (dynamic_axes 덕분에 배치 4 가 들어간다).
- **어떤 개념**: 개념 3.
- **출력**: `[('input', ['batch', 3, 224, 224])] (4, 1000) 1.48e-05` (직접 실행)

#### 한 번에 정리
- ONNX = 프레임워크 독립 모델 파일 (그래프 + 가중치), 크기 ≈ 파라미터 × 4 B.
- `torch.onnx.export` = tracing — 예시 입력 · 이름 · dynamic_axes · opset 17.
- 변환 후 반드시 같은 입력으로 최대 차이 (1e-4 미만), YOLO ONNX 는 후처리를 직접.
- ⚠ `dynamic_axes` 를 빼면 배치 크기가 1 로 굳는다. `.eval()` 을 빼면 BatchNorm 이 학습 모드로 기록된다.

**연관 문제**
<<QA 5>>

### 6교시 · 성능 측정법: baseline 성능표

**키워드**: `warm-up 5회 버림`, `평균 · p50 · p95`, `FPS = 1000 / 평균 ms`, `스레드 1 · 2 · 전체`, `같은 조건`, `파일 크기 MB`, `baseline.csv`

**개요**: Day 3 의 `latency_ms` 를 보고서용으로 키운다. ① **warm-up** 을 버린다 (첫 실행은 메모리 할당 · 캐시 때문에 느림). ② 지연시간은 가끔 크게 튀므로 평균 · **p50** · **p95** 를 함께 본다 — 실시간 시스템은 p95 로 판단. ③ **FPS = 1000 / 평균 ms**. ④ 스레드 1 · 2 · 전체로 작은 장치의 코어 수를 흉내. ⑤ **같은 조건**(CPU · 배치 1 · 전/후처리 제외)에서 PyTorch 와 ONNX Runtime 비교. 결과는 `baseline.csv` 로 저장해 다음 과목(ONNX·TensorRT 등)의 비교 기준으로 쓴다.

**핵심 용어**
- **warm-up**: 측정에서 버리는 처음 몇 번의 실행
- **p50 · p95**: 중앙값 / 100번 중 95번은 이보다 빠른 시간 (꼬리 지연)
- **FPS**: 초당 처리 프레임 = 1000 / 평균 ms
- **baseline**: 이후 최적화(양자화 · TensorRT)의 비교 기준 성능표

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `benchmark()` | `benchmark(fn, warmup=5, reps=50)` | warm-up 후 반복 측정 → mean · p50 · p95 · fps |
| `time.perf_counter()` | `time.perf_counter()` | 정밀 시계 (초) |
| `np.percentile()` | `np.percentile(ts, 95)` | 백분위수 |
| `torch.set_num_threads()` | `torch.set_num_threads(n)` | PyTorch CPU 스레드 수 |
| `ort.SessionOptions()` | `so.intra_op_num_threads = n` | ONNX Runtime 스레드 수 |
| `os.cpu_count()` | `os.cpu_count()` | CPU 코어 수 |
| `csv.DictWriter()` | `csv.DictWriter(f, fieldnames=cols)` | 딕셔너리 → CSV 행 |
| `os.path.getsize()` | `os.path.getsize(path)` | 파일 크기(바이트) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 6-1 | `benchmark` — `time.sleep(0.01)` 로 시험 (평균 5 ~ 50 ms) |
| 6-2 | MobileNetV3-S · YOLOv8n × PyTorch · ONNX Runtime × 스레드 수 측정 |
| 6-3 | 파일 크기 열 추가 → `baseline.csv` 저장 |

**핵심 내용**

#### 올바른 측정 — warm-up · 평균 · p50 · p95
> 처음 몇 번을 버리고, 반복 시간의 평균 · p50 · p95 와 FPS 를 함께 낸다
- **warm-up** → 첫 실행은 메모리 할당 · 캐시 · 최적화로 느리다
- **p95** → 100번 중 95번은 이보다 빠름 — 실시간은 평균이 아닌 p95 로 판단
- **FPS** → 1000 / 평균 ms (평균 100 ms → 10 FPS)
- **예** → 100번 중 80번 10 ms, 20번 40 ms: 평균 16, p50 10, p95 40 ms, FPS 62.5 (직접 계산) — 평균만 보면 꼬리를 놓친다
- ⚠ `time.time()` 대신 `perf_counter()`, 그리고 측정 대상은 모델 실행만 (전 · 후처리 제외)

#### 공정한 비교 — 같은 조건
> CPU · 배치 1 · 스레드 수 · 입력 크기를 맞춰야 PyTorch vs ONNX Runtime 비교가 의미 있다
- **스레드** → 1 · 2 · 전체 코어 — 늘려도 그만큼 빨라지지 않는다
- **예** → CPU · 스레드 1: MobileNetV3-S PyTorch 5.5 ms vs ONNX Runtime 2.6 ms, YOLOv8n 117.5 vs 90.8 ms
- **파일 크기** → 장치 저장 공간 · 로딩 시간 — YOLOv8n `.pt` 6.55 MB(float16) vs `.onnx` 12.85 MB(float32)
- **baseline.csv** → `model, framework, input, threads, mean, p50, p95, fps, file_MB`
- ⚠ 같은 모델이라도 입력(224 vs 640) · 스레드가 다르면 다른 실험이다

**기본 사용형태**
```python
def benchmark(fn, warmup=5, reps=50):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"mean": float(ts.mean()), "p50": float(np.percentile(ts, 50)),
            "p95": float(np.percentile(ts, 95)), "fps": float(1000 / ts.mean())}
```

**연결**
- [VB D3·1교시 정확도 vs 속도](#/vb/3/1) — `latency_ms` 의 출발점
- [VB D5·5교시 ONNX](#/vb/5/5) — 비교 대상 `.onnx`
- [VB D5·7교시 침입 감지](#/vb/5/7) — 전체 파이프라인의 한 줄을 baseline 에 추가

**상세 학습**

#### 이 교시에서 답하는 질문
"ONNX 가 빠르다" 는 말은 조건 없이는 의미가 없다. Day 3 의 한 장 시간(latency) 을 **보고서에 쓸 수 있는 측정**으로 키운다 — 무엇을 버리고(warm-up), 어떤 통계를 보며(평균 · p50 · p95), 어떤 조건을 맞추는지(CPU · 배치 1 · 스레드). 결과는 `baseline.csv` 로 남겨 다음 과목에서 TensorRT · 양자화 결과와 같은 표로 비교한다.

#### 개념 1 · 평균 · p50 · p95
![100번 중 80번 10 ms, 20번 40 ms 인 지연시간 분포에서 p50 · 평균 · p95 의 위치](assets/fig/vb/d5-latency.svg)

| 통계 | 뜻 | 위 예 |
|---|---|---|
| 평균 | 모두 더해 나눔 — 튄 값에 끌려감 | 16 ms |
| p50 (중앙값) | 보통의 한 번 | 10 ms |
| p95 | 100번 중 95번은 이보다 빠름 — "거의 최악" | 40 ms |
| FPS | 1000 / 평균 ms | 62.5 |

카메라가 10 FPS 면 프레임마다 100 ms 안에 끝나야 한다. 평균이 100 ms 아래라도 **p95 가 넘으면 프레임을 놓친다** → 실시간 판단은 p95.

#### 개념 2 · 올바른 측정 순서
1. **warm-up 5번 버림** — 첫 실행은 메모리 할당 · 캐시 · 내부 최적화 때문에 느리다.
2. `time.perf_counter()` 로 반복(50번, YOLO 는 30번) 측정 — `time.time()` 보다 정밀.
3. 모델 실행만 잰다 — 전 · 후처리는 따로 (7교시 파이프라인에서 전체를 따로 잰다).
4. **같은 조건**: CPU · 배치 1 · 같은 입력 크기 · 같은 스레드 수.
   - 스레드: `torch.set_num_threads(n)` / `ort.SessionOptions().intra_op_num_threads = n`
   - 1 · 2 · 전체 코어로 바꿔 작은 장치(코어가 적은 Jetson) 를 흉내.

#### 개념 3 · 실제 측정 (이 사이트의 실행, CPU · 스레드 1)
| 모델 | 프레임워크 | 입력 | 평균 ms | p95 ms | FPS | 파일 MB |
|---|---|---|---|---|---|---|
| MobileNetV3-S | PyTorch | 224 | 5.49 | 5.69 | 182 | 10.29 (.pt) |
| MobileNetV3-S | ONNX Runtime | 224 | 2.61 | 2.85 | 384 | 10.18 |
| YOLOv8n | PyTorch | 640 | 117.5 | 168.2 | 8.5 | 6.55 (.pt) |
| YOLOv8n | ONNX Runtime | 640 | 90.8 | 128.4 | 11.0 | 12.85 |

- 같은 조건에서 ONNX Runtime 이 MobileNetV3-S 는 약 2.1배, YOLOv8n 은 약 1.3배 빠르다.
- YOLOv8n 의 `.pt` 가 `.onnx` 의 절반인 이유: ultralytics `.pt` 는 가중치를 **float16** 으로 저장 (3,157,200 × 2 B ≈ 6.3 MB), ONNX 는 float32 (× 4 B ≈ 12.6 MB).
- 스레드를 늘려도 그만큼 빨라지지 않는다 — 작은 모델일수록 스레드 나누는 비용이 커지고, 다른 작업과 CPU 를 나누면 p95 가 크게 튄다.

#### 코드로 확인
```python
def benchmark(fn, warmup=5, reps=50):
    for _ in range(warmup):
        fn()                                    # 버리는 실행
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"mean": float(ts.mean()), "p50": float(np.percentile(ts, 50)),
            "p95": float(np.percentile(ts, 95)), "fps": float(1000 / ts.mean())}
print(benchmark(lambda: time.sleep(0.01), warmup=1, reps=20))
```
- **무엇을 하나**: 10 ms 짜리 가짜 작업으로 측정 함수를 시험.
- **어떤 개념**: 개념 1 · 2.
- **출력**: `mean 10.22, p50 10.10, p95 10.33, fps 97.8` (직접 실행 — sleep 자체의 오차로 10 ms 보다 조금 길다)

```python
def ort_session(path, n):
    so = ort.SessionOptions(); so.intra_op_num_threads = n
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
for n in sorted({1, 2, os.cpu_count()}):
    torch.set_num_threads(n)
    s_m = ort_session("mnv3s.onnx", n)
    b = benchmark(lambda: s_m.run(None, {"input": x224.numpy()}))
    rows.append({"model": "MobileNetV3-S", "framework": "ONNX Runtime", "input": 224, "threads": n, **b})
```
- **무엇을 하나**: 스레드 수를 바꿔 가며 측정해 한 줄씩 `rows` 에 쌓는다 (강의는 PyTorch · YOLO 까지 4가지).
- **어떤 개념**: 개념 2 · 3.
- **출력**: 위 표 (스레드 1). `pd.DataFrame(rows)` 에 파일 크기 열을 더해 `baseline.csv` 로 저장

#### 한 번에 정리
- warm-up 버리고 → 평균 · p50 · p95 · FPS. 실시간 판단은 p95.
- 같은 조건(CPU · 배치 1 · 입력 · 스레드)에서 PyTorch vs ONNX Runtime — MobileNetV3-S 2.6 vs 5.5 ms.
- `baseline.csv` 열: `model, framework, input, threads, mean, p50, p95, fps, file_MB` — 조건을 열로 남긴다.
- ⚠ 측정 중 다른 프로그램이 CPU 를 쓰면 숫자가 크게 흔들린다 — 같은 상태에서 여러 번 재고 p95 를 함께 본다.

**연관 문제**
<<QA 6>>

### 7교시 · 종합 미니 프로젝트: 위험구역 침입 감지

**키워드**: `영상 → ONNX YOLOv8n → 사람 박스 → 발 위치 → 다각형 → 3프레임 연속 → 경보`, `GRASS 다각형`, `pointPolygonTest`, `발 위치 (x1+x2)/2, y2`, `debounce`, `streak >= n_consec`, `find_events`, `CSV 로그`

**개요**: 5일 동안 만든 것을 모은다 — 영상 파이프라인(Day 1) · CNN 탐지기(Day 2~4) · letterbox · decode · NMS(Day 4) · ONNX Runtime · 성능 측정(Day 5). `people.mp4` → ONNX Runtime YOLOv8n → 사람 박스 → **발 위치**(박스 아래 중앙) → 위험구역 다각형 안? → **3프레임 연속**이면 경보 → 결과 영상 + CSV 로그 (`frame, time_s, n_people, n_inside, alarm`). 위험구역은 영상 왼쪽 아래 잔디 보호구역 다각형 `GRASS` (768×576 좌표). 배점: 구역 판정 20 · ONNX 탐지 20 · 영상 처리 20 · 이벤트 요약 + 성능 20 · 발표 20(강사).

**핵심 용어**
- **ROI(관심 영역)**: 판정할 구역 — 여기서는 다각형 `GRASS`
- **발 위치(foot point)**: 박스 아래 중앙 `((x1+x2)/2, y2)` — 바닥에 닿은 점
- **debounce**: 한두 프레임만 들어갔다 나오는 오탐 · 깜빡임을 막으려고 연속 N프레임일 때만 경보
- **이벤트(event)**: 경보가 1 로 이어진 구간 (시작 초, 끝 초)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `cv2.pointPolygonTest()` | `cv2.pointPolygonTest(poly, (x, y), False)` | 점이 다각형 안(+1) · 선 위(0) · 밖(−1) |
| `in_zone()` | `in_zone(pt, poly=GRASS)` | 선 위 포함 "안" 이면 True |
| `foot_points()` | `foot_points(boxes)` | 박스 → 발 위치 `(N, 2)` |
| `detect_onnx()` | `detect_onnx(frame, sess=sess_yolo, conf=0.25, iou_thr=0.7)` | 프레임 → 원본 좌표 사람 박스 · 점수 |
| `run_intrusion()` | `run_intrusion(VIDEO, "intrusion.mp4")` | 영상 전체 처리 · 경보 · 결과 영상 · CSV |
| `find_events()` | `find_events(alarm, fps)` | 경보 배열 → (시작 초, 끝 초) 목록 |
| `cv2.fillPoly()` | `cv2.fillPoly(img, [poly], color)` | 구역 칠하기 |
| `cv2.polylines()` | `cv2.polylines(img, [poly], True, color, 2)` | 구역 테두리 |
| `cv2.addWeighted()` | `cv2.addWeighted(ov, 0.3, vis, 0.7, 0)` | 반투명 겹치기 |
| `to_h264()` | `to_h264(src, dst)` | ffmpeg 로 브라우저에서 재생되는 H.264 로 변환 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| (그림) | `GRASS` 다각형을 프레임 200 에 반투명으로 그림 |
| 7-1 | `in_zone` · `foot_points` (과제 1) |
| 7-2 | `detect_onnx` — letterbox → ONNX → decode → 원본 좌표 (과제 2) |
| 7-3 | `run_intrusion` — streak · 경보 · 그리기 · CSV (과제 3) |
| 7-4 | `find_events` 와 파이프라인 전체 ms · FPS 를 baseline 에 추가 (과제 4) |
| 과제 5 | 오경보 · 놓침 · n_consec · 실시간 여부 · Jetson 이전 발표 (8교시) |

**핵심 내용**

#### 발 위치와 다각형 판정
> 사람이 구역 안에 있나 = 발 위치 `((x1+x2)/2, y2)` 가 다각형 안인가 (`pointPolygonTest ≥ 0`)
- **왜 발** → 박스 전체나 중심은 배경 · 몸통을 포함 (1교시) — 바닥 판정에는 바닥에 닿은 점
- **판정** → `+1` 안 · `0` 선 위 · `−1` 밖, 선 위도 안으로 (`>= 0`)
- **예** → 박스 `[100, 50, 200, 300]` → 발 위치 `(150, 300)` (직접 계산)
- ⚠ `pointPolygonTest` 의 점은 `float` 튜플, 다각형은 `int32` 배열

#### ONNX 탐지 → 원본 좌표
> letterbox → `(1,3,640,640)` float32 → ONNX Runtime → `decode_yolo` → 패딩 빼고 ÷ r → 화면 안으로 자르기
- **전처리** → BGR→RGB, HWC→CHW (`transpose(2,0,1)`), `/255`, 배치 축 추가
- **되돌리기** → `(boxes − [pl, pt, pl, pt]) / r`, 그리고 `clip(0, W)` · `clip(0, H)`
- ⚠ letterbox 의 패딩 · 비율을 빼먹으면 박스가 어긋난다 (Day 4)

#### debounce — 3프레임 연속이면 경보
> `streak = streak + 1 if inside.any() else 0`, `alarm = int(streak >= n_consec)`
- **예** → 프레임별 구역 안 사람 유무 `0 1 1 0 1 1 1 1 0 0 1`, n_consec = 3 → 경보 `0 0 0 0 0 0 1 1 0 0 0` (직접 계산) — 앞의 짧은 침입 · 마지막 한 프레임은 무시
- **이벤트** → 경보가 이어진 구간 하나 = 이벤트 1. `[0,0,1,1,1,0,0,1,1,0,1]`, fps = 10 → `[(0.2, 0.4), (0.7, 0.8), (1.0, 1.0)]` (직접 계산)
- **교안 예** → 강사 사전 실행: 795 프레임 중 경보 224 프레임, 이벤트 5번 (경보 비율 28.2%)
- ⚠ n_consec 가 크면 오탐은 줄지만 경보가 늦다 — 현장 값은 오경보 · 지연의 균형 (과제 5)

**기본 사용형태**
```python
def in_zone(pt, poly=GRASS):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

def foot_points(boxes):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1)

def find_events(alarm, fps):
    events, start = [], None
    for i, a in enumerate(alarm):
        if a and start is None:
            start = i
        elif not a and start is not None:
            events.append((start / fps, (i - 1) / fps)); start = None
    if start is not None:
        events.append((start / fps, (len(alarm) - 1) / fps))
    return events
```

**연결**
- [VB D1·7교시 영상 처리](#/vb/1/7) — `VideoCapture` · `VideoWriter` 파이프라인
- [VB D4·6교시 YOLO](#/vb/4/6) — `letterbox` · `decode_yolo`
- [VB D5·5교시 ONNX](#/vb/5/5) — `sess_yolo`
- [VB D5·6교시 성능 측정](#/vb/5/6) — 파이프라인 ms · FPS 를 baseline 에 추가
- [VB D5·4교시 태스크 고르기](#/vb/5/4) — det + roi 시나리오

**상세 학습**

#### 이 교시에서 답하는 질문
이제 부품이 다 있다 — 영상 읽기 · 쓰기(Day 1), 탐지기 · letterbox · decode · NMS(Day 4), ONNX Runtime · 측정(Day 5). 이 교시는 **"사람이 위험구역에 들어오면 경보"** 라는 한 문장짜리 요구를 ① 구역 판정 ② ONNX 탐지 ③ 영상 처리 · 경보 · 로그 ④ 이벤트 요약 · 성능 으로 쪼개 하나의 파이프라인으로 완성한다.

#### 개념 1 · 발 위치와 다각형 판정
![위험구역 GRASS 다각형, 발 위치 판정, 3프레임 연속 debounce 경보](assets/fig/vb/d5-intrusion.svg)

- 위험구역 = 영상(768×576) 왼쪽 아래 잔디 다각형 `GRASS = [(0,270), (180,300), (380,350), (520,470), (560,575), (0,575)]`
- **발 위치** = 박스 아래 중앙 `((x1 + x2) / 2, y2)` — 바닥에 닿은 점. 박스 전체 · 중심은 몸통 · 배경을 포함하므로 바닥 구역 판정에 맞지 않다 (1교시: 박스의 절반은 배경).
- `cv2.pointPolygonTest(poly, (x, y), False)` → +1 안 · 0 선 위 · −1 밖 → **`>= 0`** 이면 안 (선 위도 안).

| 시험 점 | 결과 | 이유 |
|---|---|---|
| (100, 400) | 안 | 왼쪽 아래 잔디 한가운데 |
| (600, 200) | 밖 | 오른쪽 위 |
| (300, 340) | 안 | 경계 (180,300)–(380,350) 아래 |
| (300, 320) | 밖 | 같은 x 에서 경계(약 330) 보다 위 |

#### 개념 2 · ONNX 탐지 → 원본 좌표 (`detect_onnx`)
| 단계 | 코드 |
|---|---|
| letterbox | `lb, r, (pl, pt) = letterbox(frame)` — 640×640, 비율 r, 패딩 |
| 텐서로 | `lb[..., ::-1]` (BGR→RGB) → `.transpose(2, 0, 1)` (HWC→CHW) → `[None]` (배치) → `/ 255.0` → float32 |
| 추론 | `sess.run(None, {"images": x})[0]` → (1, 84, 8400) |
| decode | `decode_yolo(out, conf, iou_thr)` — 사람만, NMS |
| 되돌리기 | `(boxes − [pl, pt, pl, pt]) / r` → `clip(0, W)` · `clip(0, H)` |

프레임 200 에서 7명, 발 위치 예 `(633, 368)` · `(502, 221)` · `(235, 267)`.

#### 개념 3 · debounce 와 이벤트
- 한두 프레임만 구역에 들어갔다 나오는 **깜빡임 · 오탐**에 경보가 울리면 현장에서 무시당한다.
- `streak = streak + 1 if inside.any() else 0` — 구역 안 사람이 있는 프레임이 연속 몇 번째인가
- `alarm = int(streak >= n_consec)` — n_consec = 3 프레임 연속이어야 경보
- 로그 한 줄: `frame, time_s, n_people, n_inside, alarm` → `intrusion_log.csv`
- **이벤트** = alarm 이 1 로 이어진 구간 하나 → `find_events(alarm, fps)` 가 (시작 초, 끝 초) 목록

| 예 | 값 |
|---|---|
| 구역 안 유무 | `0 1 1 0 1 1 1 1 0 0 1` |
| 경보 (n = 3) | `0 0 0 0 0 0 1 1 0 0 0` — 2프레임 침입 · 마지막 1프레임은 무시 |
| 강사 사전 실행 | 795 프레임 중 경보 224 (28.2%), 이벤트 5번 |

n_consec 를 키우면 오경보는 줄지만 경보가 늦어진다 (10 FPS 에서 3프레임 = 0.3초 지연) — 과제 5 에서 이 균형을 발표한다.

#### 개념 4 · 채점 구성
| 과제 | 내용 | 배점 |
|---|---|---|
| 1 | `in_zone` · `foot_points` | 20 |
| 2 | `detect_onnx` | 20 |
| 3 | `run_intrusion` — 영상 · 경보 · 로그 · `intrusion.mp4` | 20 |
| 4 | `find_events` + 파이프라인 전체 ms · FPS 를 `baseline.csv` 에 추가 | 20 |
| 5 | 발표: 오경보 · 놓침 사례, n_consec, 실시간 여부, Jetson 이전 계획 | 20 (강사) |

#### 코드로 확인
```python
def in_zone(pt, poly=GRASS):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0
def foot_points(boxes):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1)
for p in [(100, 400), (600, 200), (300, 340), (300, 320)]:
    print(p, "안" if in_zone(p) else "밖")
```
- **무엇을 하나**: 점 4개의 구역 판정 (과제 1).
- **어떤 개념**: 개념 1.
- **출력**: `(100, 400) 안`, `(600, 200) 밖`, `(300, 340) 안`, `(300, 320) 밖` (직접 실행)

```python
def detect_onnx(frame, sess=sess_yolo, conf=0.25, iou_thr=0.7):
    lb, r, (pl, pt) = letterbox(frame)
    x = (lb[..., ::-1].transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    out = sess.run(None, {"images": x})[0]
    boxes, scores = decode_yolo(out, conf, iou_thr)
    boxes = (boxes - [pl, pt, pl, pt]) / r
    H, W = frame.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W); boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    return boxes, scores
```
- **무엇을 하나**: 원본 프레임 → 원본 좌표 사람 박스 (과제 2).
- **어떤 개념**: 개념 2.
- **출력**: 프레임 200 에서 `7 명 | 발 위치 [[633 368] [502 221] [235 267]] ...` (직접 실행)

```python
streak = 0
for i in range(n_frames):
    ok, frame = cap.read()
    boxes, _ = detect_onnx(frame)
    inside = np.array([in_zone(p) for p in foot_points(boxes)], bool)
    streak = streak + 1 if inside.any() else 0
    alarm = int(streak >= n_consec)
    log.append([i, round(i / fps, 2), len(boxes), int(inside.sum()), alarm])
```
- **무엇을 하나**: `run_intrusion` 의 핵심 — 프레임마다 탐지 → 발 위치 → 구역 판정 → streak → 경보 → 로그 (과제 3, 그리기 · 저장 생략).
- **어떤 개념**: 개념 3.
- **출력**: `intrusion.mp4` + `intrusion_log.csv` (강사 실행: 795 프레임 중 경보 224, 이벤트 5번)

#### 한 번에 정리
- 발 위치 = `((x1+x2)/2, y2)`, 판정 = `pointPolygonTest ≥ 0`.
- `detect_onnx` = letterbox → CHW float32 → ONNX → decode → (− 패딩) ÷ r → clip.
- debounce: `streak ≥ 3` 이어야 경보, 이벤트 = 경보가 이어진 구간.
- 파이프라인 전체(읽기 ~ 저장) ms 를 baseline 에 한 줄 — 모델만 잰 6교시 숫자와 구분.
- ⚠ 박스 아래가 항상 발은 아니다 (가림 · 화면 밖 잘림) — 오경보 · 놓침의 주 원인.

**연관 문제**
<<QA 7>>

### 8교시 · 발표 · 최종 필기 퀴즈 · baseline 제출

**키워드**: `팀별 3분 발표`, `최종 퀴즈 30문항`, `submission.zip`, `baseline.csv`, `5일 지도`, `다음 과목 연결`

**개요**: 17:00 팀 발표(결과 영상 · baseline 표 · 과제 5 답), 17:25 최종 필기 퀴즈 30문항(Day 1~5), 17:45 제출. 제출물은 셀을 실행하면 만들어지는 `submission.zip` (`baseline.csv`, `intrusion_log.csv`, `intrusion_h264.mp4`, `mnv3s.onnx`, `yolov8n.onnx`). 오늘까지의 산출물은 다음 과목이 이어받는다.

**핵심 용어**
- **submission.zip**: 제출용 압축 파일 (산출물 5개)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `zipfile.ZipFile()` | `zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED)` | zip 만들기 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 제출 셀 | 산출물 5개를 `submission.zip` 으로 묶고 크기 출력 |

**핵심 내용**

#### Vision 기본 모델 5일 지도
> 각 Day 의 핵심과 손에 남은 산출물
| Day | 핵심 | 산출물 |
|---|---|---|
| 1 | Tensor · autograd · nn.Module · 이미지 · 영상 파이프라인 | MNIST CNN, 손글씨 사진 인식 |
| 2 | LeNet · VGG · ResNet, BatchNorm · skip connection | CIFAR-10 ResNet-20 (≈ 0.86) |
| 3 | 연산량 · Depthwise Separable · MobileNet · 전이학습 · Grad-CAM | MobileNetV3 전이학습, 정확도-속도 비교표 |
| 4 | bbox · IoU · NMS · AP/mAP · Faster R-CNN · SSD · YOLO | `iou_nms_map.py`, 탐지기 비교표, 인원 카운트 영상 |
| 5 | 세그멘테이션 · 포즈 · ONNX · 성능 측정 | `.onnx` 2종, `baseline.csv`, 침입 감지 코드 · 영상 |

#### 다음 과목으로 이어지는 것
> 오늘 만든 산출물이 곧 다음 과목의 출발점 — 같은 표 · 같은 코드에 새 조건만 바꿔 비교한다
| 산출물 | 이어받는 과목 | 쓰임 |
|---|---|---|
| `.onnx` · `baseline.csv` | ONNX·TensorRT (11/4~) | Jetson TensorRT 엔진과 같은 표로 속도 비교 |
| MobileNetV3 · 비교표 | Quantization·Pruning (11/16~) | int8 · 가지치기 전후 정확도 비교 (5교시처럼 출력 차이 검증) |
| 침입 감지 코드 | 현장 안전 모니터링 프로젝트 (11/25~) | `source` 만 카메라로 바꿔 출발 |
| `iou_nms_map.py` | YOLOv8 Edge 모델 (12/11~) | 커스텀 학습 모델 평가 |
- ⚠ 제출은 `submission.zip` 하나 — 파일이 없으면 조용히 빠진다 (`os.path.exists` 로 건너뜀)

**기본 사용형태**
```python
files = ["baseline.csv", "intrusion_log.csv", "intrusion_h264.mp4", "mnv3s.onnx", "yolov8n.onnx"]
with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if os.path.exists(f): z.write(f)
```

**연결**
- [VB D5·6교시 성능 측정](#/vb/5/6) — `baseline.csv`
- [VB D5·7교시 침입 감지](#/vb/5/7) — 발표할 결과 영상 · 과제 5
- [VB D4·4교시 평가 지표](#/vb/4/4) — 오늘 `iou_nms_map.py` 가 이어짐

**상세 학습**

#### 이 교시에서 답하는 질문
5일 동안 무엇을 배웠고 손에 무엇이 남았나? 이 교시는 발표 · 최종 퀴즈 · 제출로 마무리하면서, 오늘 산출물(`.onnx` 2개, `baseline.csv`, 침입 감지 코드)이 다음 과목에서 **어떻게 그대로 이어지는지** 정리한다.

#### 개념 1 · 일정과 제출물
| 시각 | 내용 |
|---|---|
| 17:00 | 팀별 3분 발표 — 결과 영상 · baseline 표 · 과제 5 답 |
| 17:25 | 최종 필기 퀴즈 30문항 (Day 1~5) |
| 17:45 | 제출 — 셀을 실행하면 `submission.zip` 이 만들어진다 |

`submission.zip` = `baseline.csv` · `intrusion_log.csv` · `intrusion_h264.mp4` · `mnv3s.onnx` · `yolov8n.onnx`.

#### 개념 2 · 5일을 한 줄로
| Day | 질문 | 답 |
|---|---|---|
| 1 | 이미지를 어떻게 숫자로? | Tensor (C, H, W) · autograd · nn.Module · 영상 파이프라인 |
| 2 | 깊게 쌓으면? | LeNet → VGG → ResNet (BatchNorm · skip) |
| 3 | 가볍게 만들면? | Depthwise Separable · MobileNet · 전이학습 · Grad-CAM |
| 4 | 여러 물체의 위치는? | bbox · IoU · NMS · AP · YOLO |
| 5 | 모양 · 자세 · 배포는? | 세그 · 포즈 · ONNX · 측정 · 침입 감지 |

#### 개념 3 · 다음 과목으로
| 다음 과목 | 이어받는 것 |
|---|---|
| ONNX · TensorRT (11/4~) | 오늘 만든 `.onnx` → TensorRT 엔진, `baseline.csv` 에 FP16 · INT8 줄 추가 |
| Quantization · Pruning (11/16~) | 같은 모델을 더 작게 — 정확도 · 속도를 baseline 과 비교 |
| 프로젝트 (11/25~) | 침입 감지 같은 파이프라인을 팀 주제로 |
| YOLOv8 Edge (12/11~) | Jetson 에서 실시간 — p95 · FPS 로 판단 |

#### 한 번에 정리
- 제출: `submission.zip` (CSV 2 · 영상 1 · ONNX 2).
- 5일 = 표현(Day 1) → 깊이(2) → 가벼움(3) → 위치(4) → 모양 · 자세 · 배포(5).
- baseline 표는 "같은 조건 · 같은 열" 로 다음 과목 결과를 이어 붙이는 기준.
- ⚠ 최적화 결과는 항상 baseline 과 같은 조건(입력 · 스레드 · 배치)에서 비교해야 의미가 있다.

**연관 문제**
<<QA 8>>
