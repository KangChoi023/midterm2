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
- **박스 안 배경** → 사람 픽셀 / 박스 넓이 평균이 0.3 ~ 0.9 사이 (교안 검증 범위) — 경계 · 넓이 문제는 세그멘테이션
- **종횡비** → 서 있는 사람은 h/w > 1.5 (평균), 90° 돌리면 h/w 가 역수(1 / ar) 로 — 박스 비율만으로 쓰러짐을 판정하면 틀린다 → 포즈
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
Day 4 까지 "사람이 몇 명, 어디에" 를 박스로 풀었다. 그런데 "발이 경계선을 넘었나", "기름이 몇 ㎡ 인가", "쓰러졌나" 는 박스만으로 믿을 만한 답이 안 나온다. 이 교시는 숫자로 왜 부족한지 확인하고, 오늘 배울 두 출력(마스크 · 관절) 이 어떤 질문을 대신 맞히는지 정리한다.

#### 개념 1 · 마스크 PNG 읽기 — 값 하나가 물체 하나
Penn-Fudan 의 마스크는 이미지와 같은 크기의 PNG 로, 픽셀값이 **0 = 배경, 1 = 첫 사람, 2 = 둘째 사람** 이다. `np.unique` 로 값을 보고 `m0 == k` 로 사람마다 떼어 낸다. 이렇게 "물체마다 따로" 인 마스크가 **인스턴스** 마스크다.

#### 개념 2 · 박스가 부족한 이유 — 숫자로 확인
| 확인 | 식 | 결과 (교안 검증) |
|---|---|---|
| 박스 안 사람 비율 | 사람 픽셀 수 / 박스 넓이 | 423명 평균 0.3 ~ 0.9 |
| 종횡비 | h / w | 서 있음 평균 > 1.5, 90° 회전 → 1 / ar |

#### 코드로 확인
```python
ids = np.unique(m0)
inst = [m0 == k for k in ids[1:]]
print(ids, len(inst), [int(m.sum()) for m in inst])
```
- **무엇을 하나**: 첫 사진의 마스크 값 · 사람 수 · 사람별 픽셀 수.
- **어떤 개념**: 개념 1.
- **출력**: ids = `[0 1 2]`, 사람 2명 (교안 검증 셀 기대값; 픽셀 수는 교안 실행값 참고)

#### 한 번에 정리
- 박스 = 어디에 · 몇 개, 마스크 = 모양 · 넓이 · 경계, 관절 = 자세.
- 박스 안은 약 절반이 배경, 종횡비는 시점에 따라 바뀐다.
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
- ⚠ 배경이 넓어 배경 IoU 는 높게 나오기 쉽다 (교안 검증: 배경 IoU > 사람 IoU) — 사람 IoU 를 따로 본다

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
1교시에서 박스가 경계 · 넓이에 약하다는 걸 봤다. 그러면 픽셀 단위 출력은 어떻게 만들고, 잘 맞았는지는 어떻게 점수 매기나? 분류 CNN 은 해상도를 줄이기만 하므로 다시 키우는 디코더가 필요하고, 박스 IoU 대신 픽셀 집합 IoU 가 필요하다.

#### 개념 1 · Encoder-Decoder — 줄였다가 키운다
인코더가 해상도를 줄이며 "사람이다" 를 알아내면 위치 정밀도는 흐려진다. 디코더가 해상도를 다시 키우되, 인코더의 같은 해상도 특징을 **skip connection** 으로 받아 경계를 복원한다 (Day 2 ResNet 의 skip 과 같은 발상, 여기서는 이어 붙임).

#### 개념 2 · 업샘플링 — Transposed Conv 크기 공식
| 기호 | 뜻 |
|---|---|
| n | 입력 한 변 길이 |
| k · s · p | 커널 · stride · padding |
| 출력 | (n − 1)·s − 2p + k |

n = 8 에서 (4, 2, 1) → 7×2 − 2 + 4 = 16. 일반 Conv 의 크기 공식을 거꾸로 푼 것이다.

#### 개념 3 · mIoU 손 계산
정답 사람 8픽셀, 예측 사람 12픽셀, 교집합 8, 합집합 12 → 사람 IoU 0.6667. 배경은 교집합 4, 합집합 8 → 0.5. 평균 0.5833. 클래스가 한 번도 안 나오면 합집합 0 → `nan` 으로 두고 `nanmean`.

#### 코드로 확인
```python
def tconv_out(n, k, s, p):
    return (n - 1) * s - 2 * p + k
x8 = torch.randn(1, 64, 8, 8)
y = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)(x8)
print(tconv_out(8, 4, 2, 1), tuple(y.shape[2:]))
```
- **무엇을 하나**: 공식과 실제 `ConvTranspose2d` 출력 크기 비교.
- **어떤 개념**: 개념 2.
- **출력**: 공식 16 (손 계산 확인), 실제 `(16, 16)` (교안 검증 셀이 일치를 assert)

```python
gt4 = np.zeros((4, 4), int); gt4[:, :2] = 1
pr4 = np.zeros((4, 4), int); pr4[:, :3] = 1
print(miou(pr4, gt4))
```
- **무엇을 하나**: 4×4 손 계산 예제의 mIoU.
- **어떤 개념**: 개념 3.
- **출력**: IoU `[0.5, 0.6667]`, mIoU `0.5833` (직접 실행해 확인)

#### 한 번에 정리
- semantic = 픽셀마다 클래스, instance = 물체마다 마스크.
- Encoder-Decoder + skip, 업샘플 = bilinear 또는 Transposed Conv (n−1)s−2p+k.
- mIoU = 클래스별 픽셀 IoU 평균, 데이터 전체는 교집합 · 합집합 누적.
- ⚠ YOLOv8-seg 도 마스크는 박스 안에서만 만든다 — 박스를 놓치면 마스크도 없다.

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
- **예** → 어깨 (100, 50)·엉덩이 (100, 150) 똑바로 서면 0°, 어깨 (200, 150)·엉덩이 (100, 150) 누우면 90°, 어깨 (160, 70) 면 36.9° (직접 계산)
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
마스크는 모양을 주지만 "서 있나 누웠나" 는 바로 알려 주지 않는다. 관절 좌표가 있으면 몸의 방향을 기하로 직접 계산할 수 있다. 이 교시는 관절 번호 · 출력 모양 → 각도 공식 → 쓰러짐 규칙 순으로 쌓는다.

#### 개념 1 · 각도 공식 — 내적으로
a, c 를 b 기준 벡터 BA, BC 로 바꾸고 `cos θ = BA·BC / (|BA||BC|)`. 분모 `+ 1e-9` 는 길이 0 방어.

#### 개념 2 · 몸통 기울기 규칙과 한계
| 상황 | 기울기 | 판정 |
|---|---|---|
| 똑바로 섬 | ≈ 0° | 정상 |
| 앞으로 숙임 (36.9°) | 36.9° | 정상 (임계 60° 미만) |
| 누움 | 90° | 쓰러짐 |
임계값 60° 는 선택이다 — 낮추면 허리 숙인 사람도 경보, 높이면 비스듬히 넘어진 사람을 놓친다.

#### 코드로 확인
```python
p0 = k0[0]
print(f"무릎 왼쪽 {angle(p0[11], p0[13], p0[15]):.1f}°")
img_rot = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
kr, cr, br = run_pose(pose, img_rot)
print(body_tilt(k0[biggest(b0)]), body_tilt(kr[biggest(br)]))
```
- **무엇을 하나**: 한 사람의 무릎 각도, 원본 vs 90° 회전의 몸통 기울기.
- **어떤 개념**: 개념 1 · 2.
- **출력**: 원본 < 30°, 회전 > 60°, 회전은 `is_fallen is True` (교안 검증 셀 기대값; 실제 숫자는 모델 실행값)

#### 한 번에 정리
- COCO 17: 어깨 5·6, 엉덩이 11·12. 방식 3가지, 위치 찾기 2가지.
- 각도 = arccos(내적 / 길이 곱), 몸통 기울기 > 60° → 쓰러짐.
- ⚠ 관절 conf 가 낮으면 판정하지 말고 `None` — 가려진 사람을 쓰러짐으로 오판하지 않게.

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
2 · 3교시에서 세 가지 출력을 배웠다. 현장에서는 어떤 걸 쓸까? 세 모델의 출력 모양을 나란히 놓고 보면 차이는 "칸마다 몇 개 숫자를 더 내나" 뿐임을 알 수 있다.

#### 개념 1 · 출력 채널 공식
`detect = 4 + nc`, `segment = 4 + nc + nm`, `pose = 4 + 1 + nk×3`. 기본값 nc = 80, nm = 32, nk = 17 에서 84 · 116 · 56.

#### 개념 2 · 태스크는 가장 단순한 것부터
분류로 되면 분류, 위치가 필요하면 탐지, 모양이 필요하면 세그, 자세가 필요하면 포즈. 단순할수록 빠르고 학습 · 검증도 쉽다.

#### 코드로 확인
```python
for task in ["detect", "segment", "pose"]:
    print(task, n_channels(task))
```
- **무엇을 하나**: 세 태스크의 칸당 숫자 개수.
- **어떤 개념**: 개념 1.
- **출력**: `detect 84`, `segment 116`, `pose 56` (직접 실행해 확인)

#### 한 번에 정리
- 같은 backbone · neck, 다른 head: 84 · 116(+prototype) · 56.
- 태스크는 필요한 출력으로, 단순한 것부터.
- ⚠ 모델이 커질수록 파라미터 · 느려짐이 늘어난다 (seg 340만, pose 330만 vs 탐지 316만).

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
- **기준** → 분류 로짓 최대 차이 < 1e-4 (교안 검증)
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
PyTorch 모델은 파이썬 · PyTorch 가 있어야 돈다. Jetson 이나 작은 장치로 옮기려면 프레임워크와 분리된 형식이 필요하다. 이 교시는 모델을 ONNX 로 내보내고, 다른 엔진(ONNX Runtime)에서 같은 결과가 나오는지 검증하는 방법을 배운다.

#### 개념 1 · ONNX 는 "그래프 + 가중치" 파일
PyTorch → `.onnx` → ONNX Runtime / TensorRT / OpenVINO. 받는 쪽은 PyTorch 없이 그래프를 읽어 실행한다.

#### 개념 2 · 이름으로 넣고 이름으로 빼기
`input_names=["input"]` 로 정한 이름이 ONNX Runtime 의 `sess.run(None, {"input": x})` 의 키가 된다. YOLO ONNX 는 입력 이름이 `images`.

#### 코드로 확인
```python
sess = ort.InferenceSession("mnv3s.onnx", providers=["CPUExecutionProvider"])
xb = torch.randn(4, 3, 224, 224)
ref = mnv3(xb).detach().numpy()
onx = sess.run(None, {"input": xb.numpy()})[0]
print(onx.shape, float(np.abs(onx - ref).max()))
```
- **무엇을 하나**: 배치 4 로 PyTorch · ONNX Runtime 출력 비교.
- **어떤 개념**: 개념 2 + 검증.
- **출력**: `(4, 1000)`, 최대 차이 < 1e-4 (교안 검증 셀 기대값)

#### 한 번에 정리
- `torch.onnx.export` = tracing, 이름 · dynamic_axes · opset 지정.
- 변환 후 반드시 최대 차이로 검증, YOLO ONNX 는 후처리를 직접.
- ⚠ `dynamic_axes` 를 빼면 배치 크기가 고정되어 다른 배치에서 에러.

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
- **파일 크기** → 장치 저장 공간 · 로딩 시간 (`.pt` vs `.onnx`)
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
"이 모델 빠르다" 는 말은 조건 없이는 의미가 없다. Day 3 의 한 장 시간(latency)을 보고서에 쓸 수 있는 수준으로 — 무엇을 버리고, 어떤 통계를 보며, 어떤 조건을 맞추는지 — 정리한다.

#### 개념 1 · 평균만 보면 안 된다
지연시간은 대부분 빠르다가 가끔 크게 튄다. 평균은 튄 값에 끌려가고, p50 은 보통의 값, p95 는 "거의 최악" 이다. 카메라 10 FPS 라면 한 프레임 100 ms 안에 처리해야 하는데 p95 가 넘으면 프레임을 놓친다.

#### 개념 2 · 표로 남기기
측정마다 한 행으로 CSV 에 쌓으면 다음 과목에서 같은 표에 TensorRT · int8 결과를 이어 붙여 비교할 수 있다.

#### 코드로 확인
```python
print(benchmark(lambda: time.sleep(0.01), warmup=1, reps=20))
```
- **무엇을 하나**: 10 ms 짜리 가짜 작업으로 측정 함수를 시험.
- **어떤 개념**: 개념 1.
- **출력**: mean 은 5 ~ 50 ms, p95 ≥ p50, fps ≈ 1000 / mean (교안 검증 셀 기대값)

#### 한 번에 정리
- warm-up 버리고 → 평균 · p50 · p95 · FPS. 실시간 판단은 p95.
- 같은 조건에서 PyTorch vs ONNX Runtime, 스레드 효과는 줄어든다.
- ⚠ baseline.csv 는 다음 과목의 비교 기준 — 조건(스레드 · 입력)을 열로 남겨 둔다.

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
이제 부품이 다 있다 — 영상 읽기, 탐지, ONNX, 측정. 이 교시는 "사람이 위험구역에 들어오면 경보" 라는 한 문장짜리 요구를, 발 위치 · 다각형 · debounce · 로그 · 이벤트 요약으로 쪼개 하나의 파이프라인으로 완성한다.

#### 개념 1 · 파이프라인 한 줄 보기
`people.mp4` → `detect_onnx` → `foot_points` → `in_zone` → `n_inside` → `streak` → `alarm` → 그리기 + CSV 한 줄. 한 단계씩 따로 시험할 수 있게 함수로 나눈다.

#### 개념 2 · 결과 해석
CSV 로그 `frame, time_s, n_people, n_inside, alarm` 에서 `find_events` 로 사건을 세고, 파이프라인 전체의 ms 는 `baseline.csv` 에 한 줄 추가한다. 영상이 10 FPS 이면 프레임당 100 ms 안에 끝나야 실시간이다.

#### 코드로 확인
```python
alarm = [0,0,1,1,1,0,0,1,1,0,1]
print(find_events(alarm, 10))
```
- **무엇을 하나**: 경보 배열에서 이벤트 구간 추출.
- **어떤 개념**: 개념 2.
- **출력**: `[(0.2, 0.4), (0.7, 0.8), (1.0, 1.0)]` (직접 실행해 확인)

#### 한 번에 정리
- 발 위치 = `((x1+x2)/2, y2)`, 판정 = `pointPolygonTest ≥ 0`.
- debounce: 연속 N프레임이어야 경보, 이벤트 = 경보 연속 구간.
- ⚠ 박스 아래가 항상 발은 아니다 (가림 · 잘림) — 오경보 · 놓침의 주 원인.

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
5일 동안 무엇을 만들었고, 그것이 다음 과목에서 어떻게 다시 쓰이나? 오늘은 정리하고 발표하는 날이라, 과제 5 의 질문에 숫자로 답하는 틀을 정리한다.

#### 개념 1 · 과제 5 답하는 틀
| 질문 | 근거 |
|---|---|
| 오경보 · 놓침 장면, 원인 | 결과 영상 (가림 · 박스 아래가 발 아님 · 구역 경계 근처) |
| n_consec 를 1 · 3 · 10 으로 | 이벤트 수 · 경보 지연 변화 |
| 실시간(10 FPS) 인가 | `baseline.csv` 의 프레임당 ms (100 ms 이하?) |
| Jetson 으로 옮길 때 바꿀 줄 | `source` 한 곳, ONNX → TensorRT |

#### 코드로 확인
```python
with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if os.path.exists(f): z.write(f)
```
- **무엇을 하나**: 있는 산출물만 zip 에 담는다.
- **어떤 개념**: 제출.
- **출력**: `{파일: 크기 MB} → submission.zip 크기` (실행 환경에 따라 다름)

#### 한 번에 정리
- Day 5 의 한 줄: 박스 → 마스크 · 관절 → ONNX 로 내보내기 → baseline 으로 재기 → 침입 감지로 묶기.
- ⚠ 숫자 없는 해석은 점수를 못 받는다 — baseline.csv · 이벤트 수로 답한다.

**연관 문제**
<<QA 8>>
