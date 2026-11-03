## Day 4 — 객체 탐지 기본: bbox · IoU · NMS · mAP · YOLO

### 1교시 · 복습 퀴즈 · 분류기로 "사람이 몇 명?"

**키워드**: `분류 vs 탐지`, `(클래스, 박스, 점수)`, `Penn-Fudan 170장 · 423명`, `xyxy 픽셀 좌표`, `parse_annotation`, `re.findall`, `sliding window`, `창 806개 × 6 ms = 4.8초`

**개요**: Day 3 의 MobileNetV3 분류기는 사진 한 장에 **라벨 하나**를 낸다. 현장 안전에서는 "사람이 **몇 명**? **어디에** 있나?" 가 필요하다. ① 탐지의 출력은 물체마다 (클래스, 박스, 점수) 이고 **개수가 이미지마다 다르다**. ② 오늘 쓰는 Penn-Fudan 보행자 데이터는 170장 · 423명, 정답은 이미지마다 텍스트 파일 하나에 `(Xmin, Ymin) - (Xmax, Ymax)` 로 들어 있다. ③ 분류기를 창으로 훑는 sliding window 는 창이 수백~수천 개라 너무 느리다 → 탐지기는 한 번에 예측한다.

**핵심 용어**
- **객체 탐지(object detection)**: 이미지에서 물체마다 (클래스, 박스, 점수) 를 찾는 문제
- **바운딩 박스(bounding box)**: 물체를 둘러싼 직사각형 — 왼쪽 위 · 오른쪽 아래 두 점으로 나타냄
- **sliding window**: 작은 창을 일정 간격(stride)으로 옮기며 창마다 분류기를 돌리는 방식
- **어노테이션(annotation)**: 사람이 달아 둔 정답(박스 · 클래스)

**이 교시에서 만드는 함수 · 새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `parse_annotation()` | `parse_annotation(path)` | 정답 텍스트 파일 → `(N, 4)` 박스 배열 |
| `re.findall()` | `re.findall(r"\((\d+), (\d+)\) - \((\d+), (\d+)\)", text)` | 정규식으로 좌표 숫자 4개씩 뽑기 |
| `np.bincount()` | `np.bincount(counts)` | 값(사람 수)별로 몇 번 나왔는지 세기 |
| `n_windows()` | `n_windows(W, H, win=(64, 128), stride=16)` | sliding window 의 창 개수 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 1-1 | 정답 텍스트에서 박스 좌표를 읽는 `parse_annotation` → 170장의 `gts` 목록 |
| 1-2 | 사람 수 분포 · 박스 폭 · 높이 산점도 |
| 1-3 | 창 개수 계산 → 창마다 6 ms 면 몇 초인지 |

**핵심 내용**

#### 분류와 탐지
> 분류 = 이미지 → 클래스 1개, 탐지 = 이미지 → 물체마다 (클래스, 박스, 점수) 여러 개
- **분류 (Day 1~3)** → 입력 이미지, 출력 클래스 1개(+ 확률)
- **탐지 (오늘)** → 출력 개수가 이미지마다 다르다 (0명일 수도, 10명일 수도)
- **세그멘테이션 · 포즈 (Day 5)** → 픽셀마다 클래스 · 관절 좌표
- **예** → Penn-Fudan: 170장에 423명, 정답 박스는 `xyxy` 픽셀 좌표 `(160, 182, 302, 431)`
- ⚠ 탐지 데이터는 "한 장 = 라벨 하나" 가 아니다 — 정답도 길이가 제각각인 배열의 목록

#### sliding window 의 한계
> 창을 옮기며 분류기를 창마다 돌리는 방식 — 창 수 × 분류기 시간이 곧 탐지 시간
- **식** → 창 수 = `((W − 창폭) // stride + 1) × ((H − 창높이) // stride + 1)`
- **예** → 559×536 이미지, 창 64×128, stride 16 → 31 × 26 = **806개** × 6 ms = **4.8초 / 장**
- ⚠ 이것도 창 크기 **1가지**만 — 크기 · 비율마다 더 돌려야 한다 → 탐지기는 한 번에 예측

**기본 사용형태**
```python
def parse_annotation(path):
    text = open(path, encoding="latin-1").read()
    pairs = re.findall(r"\((\d+), (\d+)\) - \((\d+), (\d+)\)", text)   # [('160','182','302','431'), ...]
    return np.array(pairs, dtype=np.float32).reshape(-1, 4)
```

**연결**
- [VB D3·1교시 정확도 vs 속도](#/vb/3/1) — 분류기의 한 장 시간 (6 ms 가정의 출처)
- [VB D1·6교시 디지털 이미지](#/vb/1/6) — 이미지 좌표 · HWC 배열
- [NN D1·3교시 NumPy ndarray](#/nn/1/3) — `reshape(-1, 4)` 로 박스를 모으는 배열 다루기

**상세 학습**

#### 이 교시에서 답하는 질문
Day 1~3 에서 만든 분류기는 "이 사진은 고양이" 처럼 **라벨 하나**를 낸다. 그런데 공사 현장 카메라에 필요한 것은 "지금 사람이 몇 명이고 어디에 있나" 다. 이 교시는 출력 모양이 어떻게 달라지는지 정리하고, 분류기를 그대로 돌려서 해결하려는 가장 단순한 방법(sliding window)이 왜 안 되는지 숫자로 확인한다.

#### 개념 1 · 분류와 탐지의 출력 차이
| 과제 | 입력 | 출력 |
|---|---|---|
| 분류 (Day 1~3) | 이미지 | 클래스 1개 (+ 확률) |
| 탐지 (오늘) | 이미지 | 물체마다 (클래스, 박스, 점수) — **개수가 이미지마다 다르다** |
| 세그멘테이션 · 포즈 (Day 5) | 이미지 | 픽셀마다 클래스 · 관절 좌표 |

Penn-Fudan 의 정답 파일은 `Bounding box for object 1 "PASpersonWalking" (Xmin, Ymin) - (Xmax, Ymax) : (160, 182) - (302, 431)` 처럼 한 줄에 박스 하나다. 정규식으로 숫자 4개를 뽑아 `(N, 4)` 배열로 만든다.

#### 개념 2 · sliding window 가 느린 이유
창을 가로 · 세로로 옮기며 모두 분류기에 넣으면 창 수가 곧 호출 횟수다.

| 기호 | 뜻 |
|---|---|
| W, H | 이미지 폭 · 높이 |
| 창 (64, 128) | 사람처럼 세로로 긴 창 |
| stride 16 | 창을 16 px 씩 이동 |

559×536 이미지: `nx = (559 − 64) // 16 + 1 = 31`, `ny = (536 − 128) // 16 + 1 = 26` → 31 × 26 = **806개**. 창마다 6 ms 면 806 × 6 / 1000 = **4.8초 / 장** (크기 1가지만).

#### 코드로 확인
```python
def parse_annotation(path):
    text = open(path, encoding="latin-1").read()
    pairs = re.findall(r"\((\d+), (\d+)\) - \((\d+), (\d+)\)", text)
    return np.array(pairs, dtype=np.float32).reshape(-1, 4)

gts = [parse_annotation(ann_path(p)) for p in img_paths]
```
- **무엇을 하나**: 이미지마다 정답 파일을 읽어 박스 `(N, 4)` 배열을 만들고 목록 `gts` 에 모은다.
- **어떤 개념**: 개념 1.
- **출력**: 170장 · 첫 장 `[[160. 182. 302. 431.] ...]` (강의 예시 값)

```python
def n_windows(W, H, win=(64, 128), stride=16):
    nx = (W - win[0]) // stride + 1
    ny = (H - win[1]) // stride + 1
    return nx * ny
```
- **무엇을 하나**: 창 개수를 센다. 559×536 에서 806 → 창당 6 ms 면 4.8초.
- **어떤 개념**: 개념 2.
- **출력**: `806` (직접 계산)

#### 한 번에 정리
- 탐지 = 물체마다 (클래스, 박스, 점수), 개수는 이미지마다 다르다.
- Penn-Fudan: 170장 · 423명, 정답은 `xyxy` 픽셀 좌표.
- sliding window = 창 수 × 분류기 시간 → 806개 × 6 ms = 4.8초.
- ⚠ 창 크기를 늘리면 더 느려진다 — 그래서 탐지기는 한 번에 예측한다.

**연관 문제**
<<QA 1>>

### 2교시 · 탐지 문제 정의: bbox 표현 · COCO 포맷

**키워드**: `xyxy`, `xywh`, `cxcywh 정규화`, `Pascal VOC · COCO · YOLO`, `COCO JSON`, `images · annotations · categories`, `분류 손실 + 박스 회귀 손실`, `Smooth L1`

**개요**: 이미지 좌표는 **왼쪽 위가 원점**, x 는 오른쪽 · y 는 아래쪽으로 커진다. 같은 박스를 세 가지로 쓴다. ① `xyxy` (x1, y1, x2, y2) — VOC · torchvision · NMS 계산. ② `xywh` (x, y, 폭, 높이) — COCO. ③ `cxcywh` **정규화** (cx/W, cy/H, w/W, h/H, 0~1) — YOLO. 데이터셋은 대부분 **COCO JSON** 으로 주고받는다. 탐지 손실은 분류 손실에 **박스 회귀 손실**이 더해진 것이다.

**핵심 용어**
- **xyxy**: 왼쪽 위 · 오른쪽 아래 두 점 — Pascal VOC, torchvision, NMS
- **xywh**: 왼쪽 위 + 폭 · 높이 — COCO 어노테이션
- **cxcywh 정규화**: 중심 + 폭 · 높이를 이미지 크기로 나눈 0~1 값 — YOLO 라벨
- **COCO JSON**: `images` · `annotations` · `categories` 세 목록이 id 로 연결된 탐지 데이터 형식
- **박스 회귀(box regression)**: 박스 좌표 숫자 4개를 맞히는 회귀 — Smooth L1 또는 IoU 계열 손실

**이 교시에서 만드는 함수 · 새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `xyxy_to_xywh()` | `xyxy_to_xywh(b)` | 두 점 → 왼쪽 위 + 폭 · 높이 |
| `xywh_to_xyxy()` | `xywh_to_xyxy(b)` | 반대 방향 변환 |
| `xyxy_to_cxcywh_norm()` | `xyxy_to_cxcywh_norm(b, W, H)` | 픽셀 → YOLO 형식(0~1) |
| `cxcywh_norm_to_xyxy()` | `cxcywh_norm_to_xyxy(b, W, H)` | YOLO 형식 → 픽셀 |
| `np.stack()` | `np.stack([a, b, c, d], -1)` | 열 4개를 마지막 축으로 쌓아 `(N, 4)` 로 |
| `json.dump()` | `json.dump(coco, open("x.json", "w"))` | 파이썬 dict → JSON 파일 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 2-1 | 박스 표현 변환 4개 함수 + 첫 박스를 세 형식으로 출력 |
| 2-2 | 사람 많은 순 4장에 정답 박스를 그려 보기 (새 데이터는 반드시 그려서 확인) |
| 2-3 | 170장 정답을 COCO JSON `pennfudan_coco.json` 으로 저장 |

**핵심 내용**

#### 박스 표현 세 가지
> 같은 박스를 `xyxy` (두 점) · `xywh` (점 + 크기) · `cxcywh` 정규화 (중심 + 크기 ÷ 이미지) 로 쓴다
- **식** → `w = x2 − x1`, `h = y2 − y1`, `cx = x1 + w/2`, `cy = y1 + h/2`
- **쓰는 곳** → xyxy: VOC · torchvision · NMS / xywh: COCO / cxcywh 정규화: YOLO 라벨 · 출력
- **왜 정규화** → 이미지 크기를 바꿔(resize) 도 값이 그대로라 학습에 편하다
- **예** → 박스 `(160, 182, 302, 431)`, 이미지 559×536 → xywh `(160, 182, 142, 249)` · cxcywh `(0.4132, 0.5718, 0.2540, 0.4646)`
- ⚠ 형식을 착각하면 박스가 엉뚱한 곳에 그려진다 — 변환 전 "이 값이 어느 형식인가" 부터 확인

#### COCO 어노테이션 포맷
> 이미지 · 박스 · 클래스를 id 로 연결한 JSON — `pycocotools` · torchvision · ultralytics 변환기가 읽는다
- **images** → `{id, file_name, width, height}`
- **annotations** → `{id, image_id, category_id, bbox, area, iscrowd}` — `bbox` 는 **xywh**
- **categories** → `{id, name}` (여기선 `1 = person`)
- **예** → 첫 박스: `bbox [160, 182, 142, 249]`, `area 35358` (= 142 × 249)

#### 탐지기가 학습하는 것
> 탐지 손실 = 분류 손실 (이 위치에 무엇이 있나) + 박스 회귀 손실 (박스를 얼마나 옮기고 늘리나)
- **분류 손실** → cross-entropy
- **박스 회귀 손실** → Smooth L1, 또는 IoU 를 직접 키우는 IoU 계열 손실
- **Day 2 · 3 에서** → 분류 학습에 "숫자 4개를 맞히는 회귀" 가 더해진 것
- ⚠ 오늘은 학습 없이 **사전학습 탐지기**를 쓰고 그 결과를 후처리 · 평가하는 코드를 직접 만든다

**기본 사용형태**
```python
def xyxy_to_cxcywh_norm(b, W, H):
    b = np.asarray(b, float)
    w, h = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    cx, cy = b[..., 0] + w / 2, b[..., 1] + h / 2
    return np.stack([cx / W, cy / H, w / W, h / H], -1)
```

**연결**
- [VB D4·1교시 분류와 탐지](#/vb/4/1) — 정답 박스는 `xyxy` 로 읽었다
- [NN D2·6교시 출력층과 손실 함수](#/nn/2/6) — cross-entropy · 회귀 손실이 탐지 손실의 두 부분
- [VB D4·3교시 IoU](#/vb/4/3) — 박스 계산은 `xyxy` 로 한다

**상세 학습**

#### 이 교시에서 답하는 질문
1교시에서 정답을 `xyxy` 로 읽었다. 그런데 데이터셋마다 박스를 다른 방식으로 적는다(VOC · COCO · YOLO). 형식을 헷갈리면 학습도 평가도 틀어진다. 이 교시는 세 형식과 변환식, 그리고 데이터를 주고받는 COCO JSON 구조를 정리한다.

#### 개념 1 · 박스 표현 세 가지
| 표현 | 값 | 쓰는 곳 |
|---|---|---|
| `xyxy` | (x1, y1, x2, y2) | VOC · torchvision · NMS 계산 |
| `xywh` | (x, y, w, h) | COCO |
| `cxcywh` 정규화 | (cx/W, cy/H, w/W, h/H) | YOLO 라벨 · 출력 (출력은 픽셀 단위) |

예제 박스 `(160, 182, 302, 431)`, 이미지 W=559 · H=536:

| 변환 | 계산 | 결과 |
|---|---|---|
| xywh | w = 302−160 = 142, h = 431−182 = 249 | `(160, 182, 142, 249)` |
| cx, cy | 160+142/2 = 231, 182+249/2 = 306.5 | |
| 정규화 | 231/559, 306.5/536, 142/559, 249/536 | `(0.4132, 0.5718, 0.2540, 0.4646)` |

#### 개념 2 · COCO JSON — id 로 연결
`images` 에서 `id` 를 정하고, `annotations` 의 `image_id` 가 그 id 를, `category_id` 가 `categories` 의 id 를 가리킨다. `bbox` 는 xywh, `area` 는 w × h (여기선 142 × 249 = 35,358).

#### 코드로 확인
```python
def xyxy_to_xywh(b):
    b = np.asarray(b, float)
    return np.stack([b[..., 0], b[..., 1], b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], -1)

box = gts[0][0]
print("xyxy", box, "| xywh", xyxy_to_xywh(box))
```
- **무엇을 하나**: 두 점 표현을 왼쪽 위 + 폭 · 높이로 바꾼다. `...` 인덱스라 박스 1개 · N개 모두 된다.
- **어떤 개념**: 개념 1.
- **출력**: xywh `[160. 182. 142. 249.]` (직접 계산)

```python
coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}
...
coco["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": 1,
                            "bbox": [float(v) for v in b], "area": float(b[2] * b[3]), "iscrowd": 0})
json.dump(coco, open("pennfudan_coco.json", "w"))
```
- **무엇을 하나**: 170장 · 423개 박스를 COCO 형식으로 저장한다 (박스는 xywh 로 변환).
- **어떤 개념**: 개념 2.
- **출력**: `170 images | 423 annotations` (강의 값)

#### 한 번에 정리
- xyxy (VOC · torchvision) · xywh (COCO) · cxcywh 정규화 (YOLO).
- COCO JSON = images · annotations (bbox 는 xywh) · categories, id 로 연결.
- 탐지 손실 = 분류 + 박스 회귀.
- ⚠ 새 데이터는 반드시 그려서 확인 — 형식 착각은 박스 위치로 바로 드러난다.

**연관 문제**
<<QA 2>>

### 3교시 · IoU · NMS

**키워드**: `IoU = 교집합 / 합집합`, `np.clip`, `iou_matrix`, `torchvision.ops.box_iou`, `NMS`, `keep`, `IoU 0.80 · 0.78 · 0.75`, `conf threshold vs IoU threshold`

**개요**: **IoU** 는 두 박스가 얼마나 겹치는지 0~1 로 나타낸다. NMS(중복 제거) · 평가(정답과 맞았나) · 학습(anchor 매칭) 거의 모든 곳에 쓰인다. ① 교집합 = (max x1, max y1) ~ (min x2, min y2), 안 겹치면 폭 · 높이가 음수 → 0 으로 자른다. ② 탐지기는 한 사람 주변에 비슷한 박스를 여러 개 내므로 **NMS** 가 점수 높은 순으로 하나만 남긴다. ③ 손잡이가 둘: **conf**(오탐 ↔ 놓침) 와 **IoU 기준**(중복 ↔ 붙은 물체 삭제).

**핵심 용어**
- **IoU(Intersection over Union)**: 교집합 넓이 ÷ 합집합 넓이 — 0(안 겹침) ~ 1(완전히 같음)
- **NMS(Non-Maximum Suppression)**: 점수 높은 박스를 남기고 그와 많이 겹친 박스를 지우는 후처리
- **conf threshold**: 점수가 이 값 미만인 박스를 NMS 전에 버리는 기준
- **iou_thr(NMS)**: 이 값보다 많이 겹치면 같은 물체로 보고 지운다

**이 교시에서 만드는 함수 · 새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `iou()` | `iou(box, boxes)` | 박스 1개 vs 여러 개의 IoU |
| `iou_matrix()` | `iou_matrix(A, B)` | A 의 N개 × B 의 M개 → `(N, M)` IoU 행렬 |
| `nms()` | `nms(boxes, scores, iou_thr=0.5)` | 직접 만든 NMS — 남길 인덱스 반환 |
| `torchvision.ops.box_iou()` | `torchvision.ops.box_iou(A, B)` | 라이브러리 IoU 행렬 (검증용) |
| `torchvision.ops.nms()` | `torchvision.ops.nms(boxes, scores, 0.5)` | 라이브러리 NMS (검증용) |
| `np.clip()` | `np.clip(x2 - x1, 0, None)` | 음수를 0 으로 잘라 "안 겹침" 처리 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 3-1 | `iou` 구현 → `(0,0,10,10)` vs 3개 박스, 10 px 밀린 박스 |
| 3-2 | `iou_matrix` 를 브로드캐스팅으로 구현 → `torchvision.ops.box_iou` 와 비교 |
| 3-3 | `nms` 구현 → 사람 2명 · 박스 5개 예제 + 무작위 300개를 torchvision 과 비교 |
| 3-4 | 가짜 탐지기 결과에 conf × IoU 기준 조합별로 남는 박스 수 표 |

**핵심 내용**

#### IoU
> IoU = 교집합 넓이 / 합집합 넓이 = `|A∩B| / (|A| + |B| − |A∩B|)` — 두 박스의 겹침 정도를 0~1 로
- **교집합 박스** → 왼쪽 위 `(max x1, max y1)`, 오른쪽 아래 `(min x2, min y2)`
- **안 겹칠 때** → 폭 · 높이가 음수 → `np.clip(…, 0, None)` 으로 0
- **예** → `(0,0,10,10)` vs `(5,5,15,15)` = 25 / 175 = **0.1429**, 같은 박스 = 1, 멀리 떨어진 박스 = 0
- **엄격함** → 100×100 박스가 (10, 10) 만 밀려도 IoU = **0.6807**
- ⚠ 합집합 = A + B − 교집합 (교집합을 한 번 빼야 한다)

#### NMS
> 점수 순으로 하나 남기고, 그 박스와 IoU 가 기준보다 큰 박스를 지우는 것을 반복 — 한 물체에 박스 하나만
- **순서** → ① 점수 높은 순 정렬 ② 가장 높은 박스를 `keep` ③ 그와 IoU > `iou_thr` 인 박스 삭제 ④ 남은 박스로 ②부터 반복
- **예** → 사람 2명에 박스 5개, 점수 `0.95 · 0.90 · 0.85 · 0.60 · 0.40` → IoU(0,1) = 0.8005, IoU(0,3) = 0.7808, IoU(2,4) = 0.75 → `keep = [0, 2]`
- **검증** → 직접 만든 `nms` = `torchvision.ops.nms` (무작위 300개 결과 일치)
- ⚠ 판정은 `<= iou_thr` 를 남기고 `>` 를 지운다 — 경계 조건에 주의

#### conf 와 IoU, 두 개의 손잡이
> conf = 점수 낮은 박스를 NMS 전에 버림, IoU 기준 = 이만큼 겹치면 같은 물체로 봄
- **conf 너무 낮으면** → 오탐(FP) 많음 / **너무 높으면** → 사람을 놓침(FN)
- **IoU 기준 너무 낮으면** → 붙어 선 두 사람 중 하나가 지워짐 / **너무 높으면** → 한 사람에 박스가 여러 개 남음
- ⚠ 두 값은 서로 다른 문제를 조절한다 — 하나만 만지면 다른 문제가 남는다

##### 자세히
- 이 교시의 `iou(box, boxes)` 는 박스 1개 대 여러 개(NMS 안에서 반복 호출), `iou_matrix` 는 모든 쌍을 한 번에 `(N, M)` 으로 (`A[:, None]` · `B[None, :]` 브로드캐스팅).
- 같은 계산을 `torchvision.ops.box_iou` 가 하므로 최대 차이가 거의 0 이어야 정상이다.

**기본 사용형태**
```python
def nms(boxes, scores, iou_thr=0.5):
    order = np.argsort(-scores)                 # 점수 높은 순
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        rest = order[1:]
        order = rest[iou(boxes[i], boxes[rest]) <= iou_thr]    # 많이 겹치는 박스는 지운다
    return np.array(keep, dtype=int)
```

**연결**
- [VB D4·2교시 박스 표현](#/vb/4/2) — 계산은 `xyxy` 로
- [VB D4·4교시 평가 지표](#/vb/4/4) — TP 판정이 IoU ≥ 0.5 로 정해진다
- [VB D4·6교시 YOLO 구조](#/vb/4/6) — YOLO 원시 출력도 NMS 로 마무리한다

**상세 학습**

#### 이 교시에서 답하는 질문
탐지기는 사람 한 명 주변에 비슷한 박스를 여러 개 낸다. 이걸 그대로 쓰면 "사람 5명" 이 된다. 이 교시는 박스끼리 **얼마나 겹치는지(IoU)** 를 숫자로 정의하고, 그것으로 중복을 지우는 **NMS** 를 직접 만든다. IoU 는 다음 교시의 평가에도 그대로 쓰인다.

#### 개념 1 · IoU — 겹침 정도를 숫자로
교집합 ÷ 합집합. 두 박스 `(x1, y1, x2, y2)` 에서:

| 기호 | 뜻 |
|---|---|
| `max(x1)`, `max(y1)` | 교집합의 왼쪽 위 |
| `min(x2)`, `min(y2)` | 교집합의 오른쪽 아래 |
| `clip(…, 0)` | 안 겹치면 음수 → 0 |

![IoU — 겹침 정도 숫자로](assets/fig/vb/d4-iou.svg)

`iou([50,50,150,150], [[60,60,160,160]])` — 100×100 박스가 (10, 10) 밀리면 교집합 90×90 = 8,100, 합집합 20,000 − 8,100 = 11,900 → **0.6807**.

#### 개념 2 · NMS — 점수 순으로 하나만
![NMS 전 · 후](assets/fig/vb/d4-nms.svg)

점수 0.95 박스를 남기고, 그와 IoU 0.80 · 0.78 인 0.90 · 0.60 박스를 지운다. 남은 0.85 · 0.40 중 0.85 를 남기고 IoU 0.75 인 0.40 을 지운다 → `[0, 2]`.

#### 개념 3 · conf 와 IoU 기준
| 손잡이 | 낮추면 | 높이면 |
|---|---|---|
| conf | 오탐 많음 | 사람 놓침 |
| NMS IoU 기준 | 붙은 두 사람 중 하나 삭제 | 한 사람에 박스 여러 개 |

#### 코드로 확인
```python
def iou(box, boxes):
    box = np.asarray(box, float); boxes = np.asarray(boxes, float).reshape(-1, 4)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter)
```
- **무엇을 하나**: 박스 1개와 여러 개의 IoU 를 한 번에 계산한다 (벡터화).
- **어떤 개념**: 개념 1.
- **출력**: `[0.1429 1. 0.]`, `[0.6807]` (직접 실행해 확인)

```python
print("IoU(0, 1..4):", iou(toy_boxes[0], toy_boxes[1:]).round(4), "| IoU(2, 4):", iou(toy_boxes[2], toy_boxes[4:]).round(4))
print("toy:", nms(toy_boxes, toy_scores, 0.5))
```
- **무엇을 하나**: 사람 2명 · 박스 5개 예제의 겹침과 NMS 결과.
- **어떤 개념**: 개념 2.
- **출력**: `IoU(0, 1..4): [0.8005 0. 0.7808 0.]` · `IoU(2, 4): [0.75]` · `toy: [0 2]` (직접 실행해 확인)

#### 한 번에 정리
- IoU = 교집합 / 합집합, 10 px 밀려도 크게 떨어지는 엄격한 지표.
- NMS = 점수 순으로 하나 남기고 많이 겹친 것 지우기 반복 → 직접 만든 코드 = `torchvision.ops.nms`.
- conf (오탐 ↔ 놓침) 와 IoU 기준 (중복 ↔ 붙은 물체 삭제) 은 다른 손잡이.
- ⚠ 같은 NMS 라도 IoU 기준이 낮으면 붙어 선 사람이 지워진다.

**연관 문제**
<<QA 3>>

### 4교시 · 평가 지표: Precision · Recall · AP · mAP

**키워드**: `TP · FP · FN`, `Precision = TP/(TP+FP)`, `Recall = TP/정답 수`, `PR 곡선`, `envelope`, `AP`, `손 계산 0.6875`, `AP50`, `mAP50-95`, `iou_nms_map.py`

**개요**: 탐지 하나하나를 정답과 비교해 판정한다. **TP**: 같은 클래스 정답과 IoU ≥ 0.5 (정답 하나에는 **처음** 매칭된 탐지만), **FP**: 맞는 정답이 없거나 이미 차지된 정답에 또 매칭된 **중복**, **FN**: 어떤 탐지도 매칭 안 된 정답. conf 기준을 바꿀 때마다 P · R 이 바뀌므로, 모든 기준에서의 성능을 한 숫자로 요약한 것이 **AP**(PR 곡선 아래 넓이)다. 그 평균이 mAP.

**핵심 용어**
- **TP / FP / FN**: 맞음 / 오탐(중복 포함) / 놓침
- **Precision(정밀도)**: TP / (TP + FP) — 찾은 것 중 맞은 비율
- **Recall(재현율)**: TP / 정답 수 — 정답 중 찾은 비율
- **AP(Average Precision)**: PR 곡선(envelope) 아래 넓이
- **mAP**: 클래스 평균 AP — 여기서는 클래스 1개(person)라 IoU 기준 0.50~0.95 10개의 평균(`mAP50-95`)

**이 교시에서 만드는 함수 · 새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `precision_recall()` | `precision_recall(tp, n_gt)` | TP/FP 열 → 누적 P · R |
| `ap_from_pr()` | `ap_from_pr(rec, prec)` | P · R → envelope 아래 넓이 AP |
| `match_detections()` | `match_detections(dets, gts, iou_thr=0.5)` | 모든 이미지의 탐지를 점수 순으로 TP/FP 판정 |
| `evaluate_ap()` | `evaluate_ap(dets, gts, iou_thr=0.5)` | IoU 기준 하나의 AP |
| `map_50_95()` | `map_50_95(dets, gts)` | IoU 기준 0.50~0.95 (10개) 평균 AP |
| `np.cumsum()` | `np.cumsum(tp)` | 누적합 — 순위별 누적 TP · FP |
| `inspect.getsource()` | `inspect.getsource(f)` | 함수 소스를 문자열로 — 모듈 파일로 저장할 때 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 4-1 | `precision_recall` — 손 계산 예제(TP,TP,FP,TP,FP) 의 P · R |
| 4-2 | `ap_from_pr` — envelope + 넓이 → AP 0.6875, PR 곡선 그림 |
| 4-3 | `match_detections` — 이미지 여러 장의 탐지를 점수 순으로 판정 |
| 4-4 | `evaluate_ap` · `map_50_95` — 완벽 / 10% 밀린 탐지기 비교 |
| 저장 | 오늘 함수들을 `iou_nms_map.py` 로 저장 (Day 5 · YOLOv8 Edge 과목에서 import) |

**핵심 내용**

#### TP · FP · FN 과 Precision · Recall
> Precision = TP / (TP + FP) 찾은 것 중 맞은 비율, Recall = TP / 정답 수 정답 중 찾은 비율
- **TP** → 같은 클래스 정답과 IoU ≥ 0.5, 정답 하나에 **처음** 매칭된 탐지 하나
- **FP** → 맞는 정답이 없거나, 이미 차지된 정답에 또 매칭된 중복
- **FN** → 어떤 탐지도 매칭 안 된 정답
- **예** → 정답 4개, 탐지 5개 (TP,TP,FP,TP,FP): 3번째까지 P = 0.67 · R = 0.50
- ⚠ 한 사람을 박스 2개가 맞혀도 TP 는 1개 — 나머지는 FP

#### AP — PR 곡선 아래 넓이
> 점수 높은 순으로 탐지를 훑으며 (Recall, Precision) 을 찍고, envelope(오른쪽 최댓값) 아래 넓이를 구한 것
- **순서** → 점수순 정렬 → TP/FP 판정 → 누적 TP · FP → P · R → envelope → 넓이
- **envelope** → 오른쪽에서 왼쪽으로 가며 precision 을 "오른쪽 최댓값" 으로 채움 (톱니 제거)
- **예** → recall 0~0.5 구간 P = 1, 0.5~0.75 구간 0.75, 0.75~1 구간 0 → AP = 0.25×1 + 0.25×1 + 0.25×0.75 = **0.6875**
- ⚠ 마지막 recall 이상은 못 찾았으므로 넓이 0

#### AP50 과 mAP50-95
> AP50 = IoU ≥ 0.5 로 찾았나 (느슨), mAP50-95 = IoU 기준 0.50~0.95 10개의 평균 (위치가 정밀한가)
- **AP50** → 대충 겹쳐도 TP
- **mAP50-95** → 기준이 높아질수록 엄격 — 위치 정확도까지 반영
- **예** → 박스를 폭의 10% 만큼 밀면 AP50 **1.0**, mAP50-95 **0.7** (강의 실행 값)
- ⚠ AP50 이 같아도 mAP50-95 는 다를 수 있다 — 두 숫자를 같이 본다

**기본 사용형태**
```python
def ap_from_pr(rec, prec):
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):       # envelope: 오른쪽 최댓값으로 채우기
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]      # recall 이 바뀌는 지점
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
```

**연결**
- [VB D4·3교시 IoU](#/vb/4/3) — TP 판정의 기준
- [VB D4·8교시 미니 프로젝트](#/vb/4/8) — 직접 만든 `evaluate_ap` 로 탐지기 2종 비교
- [NN D2·6교시 출력층과 손실 함수](#/nn/2/6) — 분류 정확도와 달리 탐지는 P · R · AP 로 잰다

**상세 학습**

#### 이 교시에서 답하는 질문
분류는 "맞았나 틀렸나" 한 숫자(정확도)로 충분했다. 탐지는 박스가 **몇 개 · 어디에** 있는지, 점수 기준(conf)에 따라 결과가 달라진다. 이 교시는 탐지 하나하나를 TP/FP 로 판정하고, conf 에 상관없는 한 숫자(AP)로 요약하는 법을 직접 구현한다. 오늘 만든 평가 코드는 어떤 탐지기에도 그대로 쓴다.

#### 개념 1 · TP · FP · FN 판정
탐지를 **점수 높은 순**으로 하나씩 본다. 가장 IoU 가 큰 정답이 있고(IoU ≥ 0.5) 아직 아무도 안 차지했으면 TP, 아니면 FP. 끝까지 차지되지 않은 정답이 FN.

| 구분 | 조건 |
|---|---|
| TP | 같은 클래스 정답과 IoU ≥ 0.5, 그 정답에 처음 |
| FP | 맞는 정답 없음, 또는 중복 |
| FN | 매칭 안 된 정답 |

#### 개념 2 · PR 곡선과 AP — 손 계산
정답 4개, 탐지 5개 (TP, TP, FP, TP, FP):

![손 계산 AP](assets/fig/vb/d4-ap.svg)

| 순위 | 판정 | 누적 TP | 누적 FP | P | R |
|---|---|---|---|---|---|
| 1 | TP | 1 | 0 | 1.00 | 0.25 |
| 2 | TP | 2 | 0 | 1.00 | 0.50 |
| 3 | FP | 2 | 1 | 0.67 | 0.50 |
| 4 | TP | 3 | 1 | 0.75 | 0.75 |
| 5 | FP | 3 | 2 | 0.60 | 0.75 |

AP = 0.25 × 1 + 0.25 × 1 + 0.25 × 0.75 = **0.6875**.

#### 개념 3 · AP50 vs mAP50-95
같은 탐지 결과를 IoU 기준 0.50, 0.55, …, 0.95 (10개) 로 각각 평가해 평균 내면 mAP50-95. 박스가 정답보다 폭의 10% 밀린 탐지기는 IoU 가 대략 0.8 이라 기준 0.5~0.75 는 통과, 0.8 이상은 탈락 → AP50 은 1.0 인데 mAP50-95 는 0.7 로 내려간다 (강의 실행 값).

#### 코드로 확인
```python
def precision_recall(tp, n_gt):
    tp = np.asarray(tp, float)
    ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    recall = ctp / max(n_gt, 1)
    return precision, recall

P_toy, R_toy = precision_recall(np.array([1, 1, 0, 1, 0]), 4)
print("P", P_toy.round(4), "\nR", R_toy, "AP", ap_from_pr(R_toy, P_toy))
```
- **무엇을 하나**: 누적 TP · FP 로 순위별 P · R 을 구하고 AP 를 계산한다.
- **어떤 개념**: 개념 2.
- **출력**: `P [1. 1. 0.6667 0.75 0.6]` · `R [0.25 0.5 0.5 0.75 0.75]` · `AP 0.6875` (직접 실행해 확인)

```python
mini_gts = [np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float), np.array([[0, 0, 10, 10]], float)]
mini_dets = [np.array([[0, 0, 10, 10, 0.9], [1, 1, 10, 10, 0.8], [50, 50, 60, 60, 0.7]]), np.array([[0, 0, 9, 10, 0.6]])]
print(match_detections(mini_dets, mini_gts))
```
- **무엇을 하나**: 두 이미지의 탐지를 점수 순으로 판정한다. 0.8 박스는 정답을 이미 0.9 가 차지해서 **중복 FP**.
- **어떤 개념**: 개념 1.
- **출력**: 점수 `[0.9 0.8 0.7 0.6]` · tp `[1 0 0 1]` · 정답 수 `3` → AP 0.5 (직접 실행해 확인)

#### 한 번에 정리
- TP = IoU ≥ 기준 + 정답에 처음 매칭, 중복은 FP, 못 찾은 정답은 FN.
- AP = 점수 순 누적 P · R 곡선의 envelope 아래 넓이 (손 계산 0.6875).
- AP50 = 찾았나, mAP50-95 = 위치까지 정확한가.
- ⚠ 한 정답을 박스 여러 개가 맞히면 첫 번째만 TP.

**연관 문제**
<<QA 4>>

### 5교시 · 2-stage vs 1-stage: Faster R-CNN · SSD

**키워드**: `anchor`, `9 anchors = 3 크기 × 3 비율`, `RPN`, `RoI Align`, `Faster R-CNN (2-stage)`, `SSD · SSDLite (1-stage)`, `MobileNetV3-Large backbone`, `COCO box mAP 32.8 vs 21.3`, `label 1 = person`

**개요**: 박스 좌표를 맨손으로 맞히기는 어렵다. 그래서 특징맵의 모든 위치에 크기 · 모양이 다른 **anchor**(기준 박스)를 깔아 두고, 모델은 anchor 마다 ① 물체가 있나(분류 점수) ② 얼마나 옮기고 늘릴까(dx, dy, dw, dh 회귀)를 예측한다. **2-stage**(Faster R-CNN): RPN 이 후보를 만들고 RoI Align 으로 잘라 다시 분류 — 정확 · 느림. **1-stage**(SSD): 한 번에 클래스 + 박스 — 빠름 · 작은 물체에 약함. torchvision 사전학습 탐지기를 같은 backbone 으로 비교한다.

**핵심 용어**
- **anchor**: 특징맵 위치마다 미리 깔아 둔 크기 · 비율이 다른 기준 박스
- **RPN(Region Proposal Network)**: anchor 마다 "물체 있음?" + 박스 보정으로 후보(proposal) 수백~수천 개를 만드는 1단계
- **RoI Align**: 후보 영역을 특징맵에서 7×7 로 잘라 내는 연산 — 2단계 입력
- **2-stage / 1-stage**: 후보 생성 후 재분류 / 한 번에 예측
- **SSDLite**: SSD 에 깊이별 합성곱(경량 연산)을 쓴 모바일용

**이 교시에서 만드는 함수 · 새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `make_anchors()` | `make_anchors(fh, fw, stride, sizes, ratios)` | 특징맵 모든 위치에 anchor 9개씩 → `(N, 4)` |
| `np.meshgrid()` | `np.meshgrid(cx, cy)` | 가로 · 세로 좌표 격자 만들기 |
| `detection.fasterrcnn_mobilenet_v3_large_fpn()` | `detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")` | 2-stage 탐지기 (COCO 사전학습) |
| `detection.ssdlite320_mobilenet_v3_large()` | `detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")` | 1-stage 경량 탐지기 |
| `run_tv()` | `run_tv(model, img, score_thr=0.5)` | torchvision 탐지기로 사람 박스 · 점수 얻기 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 5-1 | anchor 를 깔고 정답과 IoU 가 가장 큰 anchor 확인 |
| 5-2 | Faster R-CNN · SSDLite 를 한 장에 돌려 박스 · 시간 비교 |
| 5-3 | Penn-Fudan 에서 두 모델의 AP50 · mAP50-95 · ms/장 평가 (CPU 면 40장) |

**핵심 내용**

#### Anchor
> anchor = 위치마다 미리 깔아 둔 기준 박스 — 모델은 anchor 별 "물체 있나" 점수와 보정값(dx, dy, dw, dh)을 낸다
- **구성** → 위치마다 크기 3(64 · 128 · 256) × 비율 3(0.5 · 1 · 2) = **9개**, 넓이는 크기² 유지
- **개수** → 특징맵 위치 수 × 9 — 559×536 이미지, stride 16 → 35×34 = 1,190 위치 → **10,710개**
- **학습** → 정답과 IoU 가 큰 anchor 가 그 정답을 맡는다
- **비율 식** → `w = s/√ratio`, `h = s·√ratio` (ratio = h/w)
- ⚠ anchor 크기 · 비율이 데이터 물체와 안 맞으면 잘 못 맞힌다

#### 2-stage vs 1-stage
> 2-stage = 후보를 먼저 뽑고(RPN) 다시 분류, 1-stage = anchor 마다 클래스 + 박스를 한 번에
- **Faster R-CNN (2015)** → RPN 후보 수백~수천 → RoI Align 7×7 → 재분류 + 박스 보정: 정확 · 느림
- **SSD · SSDLite (2016~)** → 여러 해상도 특징맵에서 한 번에 → NMS: 빠름 · 작은 물체에 약함
- **비교 (COCO 사전학습)** → R-CNN R50-FPN 41,755,286 파라미터 · 134.38 GFLOPS · mAP 37.0 / R-CNN MNv3-L FPN 19,386,354 · 4.49 · 32.8 / SSDLite320 3,440,060 · 0.58 · 21.3
- ⚠ 같은 backbone 이어도 방식이 달라 정확도 · 속도 차이가 크다

#### torchvision 탐지기 사용법
> 입력 = 0~1 float 텐서의 **리스트**, 출력 = 이미지마다 `{boxes, labels, scores}` dict
- **boxes** → `(N, 4)` `xyxy` 픽셀 / **labels** → COCO 번호, **1 = person** (0 은 background)
- **전처리** → 정규화 · 크기 조정은 모델 안에서 한다
- ⚠ 필터는 `labels == 1` 과 `scores ≥ thr` 를 함께 — 사람만 남긴다

**기본 사용형태**
```python
from torchvision.models import detection
frcnn = detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT").eval().to(device)

def run_tv(model, img, score_thr=0.5):
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(device)
    with torch.no_grad():
        out = model([x])[0]                                # 이미지 리스트 → 결과 dict 리스트
    m = (out["labels"] == 1) & (out["scores"] >= score_thr)
    return out["boxes"][m].cpu().numpy(), out["scores"][m].cpu().numpy()
```

**연결**
- [VB D3·4교시 MobileNet V1 · V2 · V3](#/vb/3/4) — 두 탐지기의 같은 backbone
- [VB D4·3교시 IoU · NMS](#/vb/4/3) — anchor 매칭과 후처리에 쓰인다
- [VB D3·5교시 전이학습](#/vb/3/5) — `weights="DEFAULT"` 사전학습 모델을 가져다 쓰는 방식

**상세 학습**

#### 이 교시에서 답하는 질문
3 · 4교시에서 박스를 비교하고 평가하는 법은 만들었다. 그런데 **박스는 누가 만드나?** 좌표 4개를 처음부터 맞히기는 어렵다. 이 교시는 기준 박스(anchor)로 문제를 쉽게 만든 뒤, 그 위에 두 가지 설계(2-stage · 1-stage)가 어떻게 다른지, torchvision 사전학습 탐지기로 비교한다.

#### 개념 1 · Anchor — 미리 깔아 둔 기준 박스
특징맵의 각 위치에 크기 · 비율이 다른 anchor 를 9개 깔고, 모델은 anchor 마다 "물체가 있나(분류)" 와 "얼마나 옮기고 늘리나(회귀)" 를 예측한다.

| 기호 | 뜻 |
|---|---|
| fh × fw | 특징맵 높이 × 폭 (stride 16 → 이미지 ÷ 16 올림) |
| sizes (64, 128, 256) | anchor 크기(넓이 s²) |
| ratios (0.5, 1, 2) | 높이 ÷ 폭 |

예: 559×536 → fh = 34, fw = 35 → 1,190 × 9 = **10,710 anchor**. 크기 64 · 비율 0.5 → 폭 90.5 × 높이 45.3.

#### 개념 2 · 2-stage vs 1-stage
| | Faster R-CNN (2-stage) | SSD / SSDLite (1-stage) |
|---|---|---|
| 1단계 | RPN: anchor 별 물체 유무 + 보정 → 후보 | 여러 해상도 특징맵에서 클래스 + 박스 한 번에 |
| 2단계 | RoI Align 7×7 → 재분류 + 보정 | 없음 → NMS |
| 장단점 | 정확, 느림 | 빠름, 작은 물체에 약함 |

#### 코드로 확인
```python
def make_anchors(fh, fw, stride, sizes=(64, 128, 256), ratios=(0.5, 1.0, 2.0)):
    base = []
    for s in sizes:
        for ratio in ratios:                              # ratio = h / w, 넓이 s² 유지
            w = s / np.sqrt(ratio); h = s * np.sqrt(ratio)
            base.append([-w / 2, -h / 2, w / 2, h / 2])
    base = np.array(base)                                 # (9, 4) — 중심 (0, 0) 기준
    cx = (np.arange(fw) + 0.5) * stride; cy = (np.arange(fh) + 0.5) * stride
    cxx, cyy = np.meshgrid(cx, cy)
    centers = np.stack([cxx, cyy, cxx, cyy], -1).reshape(-1, 1, 4)
    return (centers + base[None]).reshape(-1, 4)
```
- **무엇을 하나**: 기준 박스 9개를 모든 칸의 중심에 옮겨 `(칸 수 × 9, 4)` 로 만든다.
- **어떤 개념**: 개념 1.
- **출력**: 559×536 · stride 16 → 34 × 35 × 9 = `10,710` (직접 계산)

```python
frcnn = detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT").eval().to(device)
ssd = detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT").eval().to(device)
bx, sc = run_tv(frcnn, img_b)          # person(label 1) 만 · score ≥ 0.5
```
- **무엇을 하나**: 같은 backbone 의 두 탐지기를 불러와 사람 박스를 얻는다.
- **어떤 개념**: 개념 2.
- **출력**: 파라미터 `19,386,354` vs `3,440,060` (강의 표 값)

#### 한 번에 정리
- anchor = 위치마다 깔아 둔 기준 박스, 모델은 점수 + 오프셋을 예측.
- 2-stage (RPN → RoI Align → 재분류): 정확 · 느림, 1-stage (한 번에): 빠름 · 작은 물체 약함.
- torchvision 탐지기: 입력 0~1 텐서 리스트, 출력 dict (boxes xyxy · labels · scores), person = 1.
- ⚠ label 0 은 background 다 — person 은 1.

**연관 문제**
<<QA 5>>

### 6교시 · YOLO 구조: grid · 출력 텐서

**키워드**: `YOLO`, `Backbone · Neck · Head`, `FPN + PAN`, `stride 8 · 16 · 32`, `anchor-free`, `8,400 칸`, `(1, 84, 8400)`, `letterbox 114`, `decode_yolo`, `YOLOv8n · YOLOv8s`

**개요**: **YOLO**(You Only Look Once)는 이미지를 grid 로 나눠 칸마다 박스와 클래스를 한 번에 예측하는 1-stage 탐지기다. YOLOv8 은 Backbone(특징 추출, CSP) → Neck(의미 정보 + 위치 정보 섞기, FPN + PAN) → Head(stride 8 · 16 · 32 에서 예측, 박스 / 클래스 분리 · anchor-free) 구조. 640 입력이면 **8,400 칸**이고 원시 출력은 `(1, 84, 8400)` = 박스 4 + 클래스 80. 라이브러리의 predict 가 해 주는 전처리 · 후처리를 **직접 열어 보며** 만든다 (Day 5 의 ONNX 모델도 원시 출력만 나온다).

**핵심 용어**
- **grid / stride**: 입력을 stride 픽셀 간격의 칸으로 나눔 (stride 8 → 640 ÷ 8 = 80×80)
- **anchor-free**: 미리 깐 anchor 없이 칸 중심에서 박스 네 변까지 거리를 직접 예측
- **FPN + PAN (Neck)**: 깊은 층(의미) → 얕은 층(위치), 얕은 → 깊은 양방향으로 정보 섞기
- **letterbox**: 비율을 유지하며 긴 변을 640 에 맞추고 남는 곳을 회색(114)으로 채우는 전처리
- **디코드(decode)**: 원시 출력 → conf 필터 → xyxy → NMS → 원본 좌표로 되돌리는 후처리

**이 교시에서 만드는 함수 · 새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `n_preds()` | `n_preds(imgsz, strides=(8, 16, 32))` | 입력 크기별 예측 칸 수 |
| `letterbox()` | `letterbox(img, size=640, color=114)` | 비율 유지 + 회색 패딩 → 640×640 |
| `decode_yolo()` | `decode_yolo(out, conf_thr=0.25, iou_thr=0.7, cls=0)` | 원시 출력 `(1, 84, 8400)` → 박스 · 점수 |
| `YOLO()` | `YOLO("yolov8n.pt")` | ultralytics 모델 불러오기 |
| `model.predict()` | `model.predict(img, imgsz=640, conf=0.25, classes=[0])` | 전처리 · 추론 · NMS 까지 한 번에 |
| `cv2.resize()` | `cv2.resize(img, (nw, nh))` | 크기 변경 — 인자는 (폭, 높이) |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 6-1 | `n_preds` — 640 → 8,400 · 320 → 2,100 |
| 6-2 | letterbox → 텐서 `(1, 3, 640, 640)` → 원시 출력 shape 확인 |
| 6-3 | `decode_yolo` 로 직접 디코드 → ultralytics 결과와 같은지 비교 |
| 6-4 | YOLOv8n · YOLOv8s 의 파라미터 · 단계별 시간(preprocess · inference · postprocess) |

**핵심 내용**

#### YOLOv8 구조와 출력 텐서
> Backbone → Neck → Head, 입력 640 → 칸 8,400개 × (박스 4 + 클래스 80) = `(1, 84, 8400)`
- **칸 수** → `(640/8)² + (640/16)² + (640/32)² = 6,400 + 1,600 + 400 = 8,400`
- **84** → 앞 4 = 박스 `cx cy w h` (픽셀), 뒤 80 = COCO 클래스 점수(sigmoid)
- **해상도 효과** → 320 이면 칸 2,100 (예측 · 연산 약 1/4)
- **작은 물체** → stride 8 (80×80) 칸이 맡는다
- ⚠ objectness 점수가 따로 없다 — 클래스 점수 최댓값이 곧 conf

#### letterbox 전처리
> 비율을 유지해 긴 변을 640 에 맞추고, 남는 곳을 회색(114)으로 채워 640×640 으로 만든다
- **식** → `r = min(640/h, 640/w)`, 새 크기 `(round(w·r), round(h·r))`, 패딩 `((640 − nh)//2, (640 − nw)//2)`
- **예** → 559×536 → r = 1.1449, 640×614, 위 · 아래 패딩 13 → `pad = (0, 13)`
- **이어서** → BGR→RGB, 0~1, `(1, 3, 640, 640)`
- ⚠ 결과 박스는 letterbox 좌표 — 원본으로 되돌리려면 **패딩을 빼고 r 로 나눈다**

#### 직접 디코드
> 원시 출력 → 전치 → 클래스 최댓값 = conf → conf 필터 → cxcywh → xyxy → NMS → 패딩 빼고 ÷ r
- **전치** → `out[0].T` 로 `(84, 8400)` → `(8400, 84)`
- **필터** → `conf > conf_thr` 와 `best == cls` (person = 0, 라벨 번호가 torchvision 과 다름)
- **검증** → `ultralytics predict` 결과와 짝 IoU 최소값이 거의 1
- ⚠ YOLO 의 person 은 **0**, torchvision COCO 의 person 은 **1**

**기본 사용형태**
```python
from ultralytics import YOLO
yolo = YOLO("yolov8n.pt")
r = yolo.predict(img, imgsz=640, conf=0.25, classes=[0], verbose=False)[0]
boxes, scores = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
print(r.speed)                  # {'preprocess': ..., 'inference': ..., 'postprocess': ...} ms
```

**연결**
- [VB D4·5교시 2-stage vs 1-stage](#/vb/4/5) — YOLO 는 1-stage, anchor-free
- [VB D4·3교시 IoU · NMS](#/vb/4/3) — 디코드의 마지막 단계가 NMS
- [VB D3·2교시 연산량 계산](#/vb/3/2) — MAC 이 해상도²에 비례 → 320 이면 약 1/4
- [VB D1·6교시 디지털 이미지](#/vb/1/6) — BGR ↔ RGB, HWC ↔ CHW 변환

**상세 학습**

#### 이 교시에서 답하는 질문
5교시의 anchor 방식은 anchor 설계와 후보 처리가 복잡하다. YOLO 는 한 번에 칸마다 박스와 클래스를 예측한다. 라이브러리 `predict()` 는 편하지만 안이 보이지 않는다. 이 교시는 YOLOv8 의 구조를 보고, 원시 출력 `(1, 84, 8400)` 을 직접 열어 후처리를 만들어 본다.

#### 개념 1 · YOLOv8 구조 — Backbone · Neck · Head
![YOLOv8 구조](assets/fig/vb/d4-yolo.svg)

| 부분 | 하는 일 | 구조 |
|---|---|---|
| Backbone | 특징 추출 (Day 2 · 3 의 CNN 역할) | CSP Conv 블록 |
| Neck | 깊은 층(의미)과 얕은 층(위치) 섞기 | FPN(위→아래) + PAN(아래→위) |
| Head | stride 8 · 16 · 32 에서 박스 / 클래스 분리 예측 | decoupled · anchor-free |

각 칸은 "칸 중심에서 박스 네 변까지 거리" 와 "80 클래스 점수" 를 낸다.

#### 개념 2 · letterbox 와 디코드
입력은 정사각형(640×640)이어야 하므로 비율을 유지한 채 줄이고 회색으로 채운다. 출력 `(1, 84, 8400)` 에서 `[:4]` 는 박스, `[4:]` 는 클래스 점수. 디코드 후에는 원본 좌표 = `(letterbox 좌표 − (padl, padt)) / r`.

| 단계 | 하는 일 |
|---|---|
| 1 | letterbox → RGB · 0~1 → `(1, 3, 640, 640)` |
| 2 | 모델 통과 → `(1, 84, 8400)` |
| 3 | 전치 `(8400, 84)`, 클래스 최댓값 = conf, `conf > 0.25` · person |
| 4 | cxcywh → xyxy |
| 5 | NMS (iou 0.7) |
| 6 | 패딩 빼고 ÷ r → 원본 좌표 |

#### 코드로 확인
```python
def n_preds(imgsz, strides=(8, 16, 32)):
    return sum((imgsz // s) ** 2 for s in strides)

print("640 →", n_preds(640), "| 320 →", n_preds(320))
```
- **무엇을 하나**: 입력 크기별 예측 칸 수를 센다.
- **어떤 개념**: 개념 1.
- **출력**: `640 → 8400 | 320 → 2100` (직접 실행해 확인)

```python
def decode_yolo(out, conf_thr=0.25, iou_thr=0.7, cls=0):
    p = out[0].T.cpu().numpy()                             # (8400, 84)
    boxes_c, cls_scores = p[:, :4], p[:, 4:]
    best, conf = cls_scores.argmax(1), cls_scores.max(1)
    m = (conf > conf_thr) & (best == cls)
    cx, cy, w, h = boxes_c[m].T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    keep = nms(boxes, conf[m], iou_thr)
    return boxes[keep], conf[m][keep]
```
- **무엇을 하나**: 원시 출력에서 사람 박스만 골라 NMS 까지 마친다 (직접 만든 `nms` 재사용).
- **어떤 개념**: 개념 2.
- **출력**: ultralytics 결과와 같은 사람 수 · 짝 IoU 최소값 거의 1 (강의 실행 결과 — 직접 실행 필요)

#### 한 번에 정리
- YOLOv8 = Backbone (CSP) → Neck (FPN+PAN) → Head (stride 8 · 16 · 32, anchor-free).
- 640 입력 → 8,400 칸, 출력 `(1, 84, 8400)` = 박스 4 + 클래스 80.
- 디코드: letterbox → 추론 → 전치 · 최댓값 · conf 필터 → xyxy → NMS → 패딩 빼고 ÷ r.
- ⚠ 원시 출력의 박스는 letterbox 좌표 — 원본으로 되돌리는 것을 잊지 않는다.

**연관 문제**
<<QA 6>>

### 7교시 · 영상 탐지 파이프라인

**키워드**: `VideoCapture(source)`, `전처리 → 추론 → 후처리 → 오버레이 → VideoWriter`, `실시간 예산 1000 / FPS ms`, `r.speed`, `병목`, `H.264 · ffmpeg`, `people_det.mp4`

**개요**: Day 1 의 영상 파이프라인에 탐지를 끼운다. `VideoCapture(source)` → 프레임 → 전처리(letterbox) → 추론 → 후처리(NMS) → 오버레이(박스 · 인원 수) → `VideoWriter`. **규칙**: 영상은 `cv2.VideoCapture(source)` **한 곳에서만** 열고 `source` 를 인자로 받는다. 나중에 파일 경로를 카메라 주소(GStreamer 문자열)로 바꾸기만 하면 Jetson · 라즈베리파이에서 그대로 돈다. **실시간 예산**: 10 FPS 영상 → 프레임당 100 ms, 30 FPS → 33 ms 안에 모든 단계가 끝나야 한다.

**핵심 용어**
- **실시간 예산(frame budget)**: 1000 ÷ FPS ms — 프레임 처리가 끝나야 하는 시간
- **병목(bottleneck)**: 파이프라인에서 가장 오래 걸리는 단계 (보통 추론)
- **오버레이(overlay)**: 프레임 위에 박스 · 인원 수를 그리는 것
- **H.264 변환**: 브라우저에서 재생하려고 `mp4v` 영상을 ffmpeg 으로 다시 인코딩

**이 교시에서 만드는 함수 · 새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `detect_people()` | `detect_people(frame, model, imgsz=640, conf=0.25)` | 프레임 → 사람 박스 · 점수 · 단계별 시간 |
| `process_video()` | `process_video(source, out_path, model, ...)` | 영상 전체 탐지 → 결과 영상 · 프레임별 인원 수 · 단계별 평균 시간 |
| `realtime_report()` | `realtime_report(times, label, video_fps)` | 단계별 ms 합 → FPS · 실시간 여부 |
| `cv2.VideoCapture()` | `cv2.VideoCapture(source)` | 영상 파일 · 카메라 열기 |
| `cv2.VideoWriter()` | `cv2.VideoWriter(path, fourcc, fps, (W, H))` | 프레임을 영상 파일로 저장 |
| `cv2.rectangle()` | `cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)` | 박스 그리기 |
| `cv2.putText()` | `cv2.putText(img, "people: 3", (12, 36), font, 1.1, color, 3)` | 글자 그리기 |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 7-1 | `detect_people` — 영상 200번째 프레임에서 사람 찾기 · 단계별 시간 |
| 7-2 | `process_video` — 전체 영상 처리 → `people_det.mp4` · 프레임별 인원 수 · 단계별 평균 시간 |
| 7-3 | `realtime_report` — 합계 ms → FPS, 실시간 여부 (GPU 런타임이면 CPU 도 비교) |

**핵심 내용**

#### 영상 탐지 파이프라인
> 프레임 하나마다 읽기 → 전처리 → 추론 → 후처리 → 그리기 · 저장을 반복하는 흐름
- **순서** → `VideoCapture(source)` → 전처리(letterbox) → 추론 → 후처리(NMS) → 오버레이 → `VideoWriter`
- **규칙** → 영상을 여는 곳은 한 군데, `source` 는 인자로 → 파일 경로 ↔ 카메라 주소 교체만으로 장치에서 동작
- **단계별 시간** → ultralytics 가 `r.speed` 에 `preprocess · inference · postprocess` (ms) 를 준다. 읽기 · 그리기 + 쓰기는 직접 잰다
- ⚠ `cap.read()` 가 `False` 면 끝 — 루프를 빠져나오고 `release()` 를 꼭 부른다

#### 실시간 예산
> 영상이 N FPS 면 프레임당 `1000 / N` ms 안에 모든 단계가 끝나야 실시간
- **예** → 10 FPS = 100 ms, 30 FPS = 33.3 ms
- **판정** → 단계별 ms 합 ≤ 1000 / FPS → 실시간, 아니면 프레임을 놓친다
- **병목** → 단계별 시간을 재서 가장 큰 것(보통 추론)부터 줄인다 — 작은 모델 · 작은 입력 해상도
- ⚠ 추론만 빨라도 읽기 · 그리기 · 저장이 느리면 합계가 예산을 넘는다

**기본 사용형태**
```python
def detect_people(frame, model, imgsz=640, conf=0.25):
    r = model.predict(frame, imgsz=imgsz, conf=conf, classes=[0], verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.speed

cap = cv2.VideoCapture(source)          # source: 파일 경로 또는 카메라 주소 — 여는 곳은 여기 한 곳
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
```

**연결**
- [VB D1·7교시 영상 파일 파이프라인](#/vb/1/7) — `VideoCapture` · `VideoWriter` 의 첫 등장
- [VB D3·1교시 정확도 vs 속도](#/vb/3/1) — 지연시간 · 실시간 33 ms 기준
- [VB D4·6교시 YOLO 구조](#/vb/4/6) — `predict` 안의 전처리 · 추론 · 후처리

**상세 학습**

#### 이 교시에서 답하는 질문
6교시까지는 사진 한 장이었다. 현장에서는 영상이 들어온다. 이 교시는 Day 1 의 영상 파이프라인에 탐지를 끼워 프레임별 인원 수를 세고, 단계별 시간을 재서 **실시간이 되는지** 확인한다.

#### 개념 1 · 영상 파이프라인 — 한 곳에서 열고 한 프레임씩
```
VideoCapture(source) → 프레임 → letterbox → 추론 → NMS → 박스 · 인원 수 그리기 → VideoWriter
```
`source` 는 함수 인자로 받는다 — 파일 `people.mp4` 든 카메라 주소든 같은 코드가 돈다.

#### 개념 2 · 실시간 예산과 병목
| FPS | 프레임당 예산 |
|---|---|
| 10 | 100 ms |
| 30 | 33.3 ms |

단계별 시간(읽기 · 전처리 · 추론 · 후처리 · 그리기 + 쓰기)의 **합** 이 예산 이하여야 한다. 가장 큰 단계가 병목.

#### 코드로 확인
```python
def realtime_report(times, label, video_fps=V_FPS):
    total = sum(times.values())
    print(f"[{label}] " + " · ".join(f"{k} {v:.1f}" for k, v in times.items()) + f"  = {total:.1f} ms → {1000 / total:.1f} FPS"
          + ("  ✔ 실시간" if total <= 1000 / video_fps else "  ✘ 실시간 아님") + f" (영상 {video_fps:.0f} FPS 기준)")
    return 1000 / total
```
- **무엇을 하나**: 단계별 평균 ms 를 합쳐 FPS 를 계산하고 영상 FPS 예산과 비교한다.
- **어떤 개념**: 개념 2.
- **출력**: 합계 100 ms → `10.0 FPS`, 33.3 ms 예산이면 ✘ (식으로 확인, 실제 ms 는 장치마다 다름)

```python
cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
```
- **무엇을 하나**: 영상을 열어 FPS · 크기를 읽고 같은 크기 · FPS 로 결과 영상을 저장할 준비를 한다.
- **어떤 개념**: 개념 1.
- **출력**: `(W, H)` 는 `(폭, 높이)` 순서 — `cv2.VideoWriter` 도 `cv2.resize` 처럼 (폭, 높이)

#### 한 번에 정리
- 파이프라인: `VideoCapture(source)` → 전처리 → 추론 → 후처리 → 오버레이 → `VideoWriter`, `source` 는 인자 하나로.
- 실시간 예산 = 1000 / FPS ms, 단계별 시간 합이 예산 이하여야 한다.
- 병목(보통 추론)을 찾아 작은 모델 · 해상도로 줄인다.
- ⚠ 브라우저 재생은 H.264 — `mp4v` 결과는 ffmpeg 으로 변환해야 보인다.

**연관 문제**
<<QA 7>>

### 8교시 · 미니 프로젝트: 직접 만든 mAP 로 탐지기 2종 평가 + 영상 인원 카운트

**키워드**: `predict_dataset`, `AP50 · mAP50-95`, `YOLOv8n vs YOLOv8s`, `320 · 640`, `cpu_ms`, `Precision · Recall · F1`, `best_conf`, `people_count.csv`, `YOLOv8n 640 AP50 0.9692 · mAP50-95 0.8357`

**개요**: 현장에 넣을 "사람 세기" 탐지기를 고른다. 정확도(AP)는 4교시에 **직접 만든 평가 코드**로, 속도는 CPU 에서 재고, **운영 threshold** 를 정한 뒤 영상에 적용한다. 과제: ① `evaluate_ap` · `map_50_95` 로 YOLOv8n · YOLOv8s 평가(Penn-Fudan 170장, 640, conf 0.001 — 낮게 해야 PR 곡선 전체) ② 해상도 320 / 640 × 모델 2종 → AP50 · mAP50-95 · CPU ms 표 ③ conf 0.10~0.90 의 P · R · F1 → 운영 threshold ④ 영상 인원 카운트 + 프레임별 CSV ⑤ 결과 해석. 강사 사전 실행 예(CPU): YOLOv8n 640 AP50 0.9692 · mAP50-95 0.8357.

**핵심 용어**
- **F1**: `2PR / (P + R)` — P 와 R 의 조화평균, 운영 threshold 를 고르는 한 숫자
- **운영 threshold**: AP 는 모든 threshold 의 요약이고, 현장은 conf **하나**로 돈다 — 그 값
- **cpu_ms**: CPU 에서 추론 한 장 ms (모델은 CPU 용으로 따로 불러온다)
- **평가용 conf(0.001) vs 운영 conf**: AP 를 잴 땐 아주 낮게, 현장에서는 F1 · 목적으로 정한 값

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `predict_dataset()` | `predict_dataset(model, imgsz)` | 170장 예측 → 이미지마다 `(K, 5)` `[x1, y1, x2, y2, score]` |
| `cpu_ms()` | `cpu_ms("yolov8n.pt", imgsz)` | CPU 에서 추론 ms / 장 |
| `np.argmax()` | `np.argmax(F1_c)` | F1 이 가장 큰 위치 → `best_conf` |

**예제 코드가 하는 일**
| 예제 | 하는 일 |
|---|---|
| 8-1 | 두 모델을 640 으로 평가 → `res[(이름, 640)]` = {dets, AP50, mAP} |
| 8-2 | 320 도 평가 + 네 조합 CPU 지연시간 `ms` |
| 8-3 | conf 0.10~0.90 (17개) 마다 P · R · F1 → `best_conf` |
| 8-4 | `best_conf` 로 영상 전체 처리 → `count_result.mp4` · `people_count.csv` |
| 8-5 | 해석(강사 채점) — 라즈베리파이급 장치에 올릴 조합 · 근거 |

**핵심 내용**

#### 같은 기준으로 탐지기 비교
> 평가 코드를 직접 가지고 있으면 torchvision · YOLO · ONNX · TensorRT 어떤 탐지기든 같은 기준으로 비교한다
- **절차** → 탐지 결과를 이미지마다 `(K, 5)` 로 모은다 → `evaluate_ap(dets, gts, 0.5)` · `map_50_95(dets, gts)`
- **conf** → AP 를 잴 때는 **0.001** 처럼 아주 낮게 (PR 곡선 전체를 얻으려고)
- **표** → 모델 2종 × 해상도 2종 = 4조합의 AP50 · mAP50-95 · CPU ms
- **예** → YOLOv8n 640: AP50 **0.9692**, mAP50-95 **0.8357** (강사 사전 실행, CPU)
- ⚠ 모델 · 속도를 함께 본다 — AP50 이 같아도 mAP50-95 · 속도가 다르다

#### 운영 threshold 고르기 — F1
> 현장은 conf 하나로 돈다 — conf 마다 P · R · F1 을 구해 F1 이 가장 큰 값을 고르고, 목적에 맞게 조정
- **식** → `F1 = 2PR / (P + R)`, 예: P 0.9 · R 0.8 → 0.8471
- **구하기** → conf 이상만 남기고 `match_detections(kept, gts, 0.5)` → `P = TP / 탐지 수`, `R = TP / 정답 수`
- **목적별** → 출입 인원 세기: F1 근처 / 위험구역 침입 감지: **놓치면 안 되니 conf 낮춤** (오경보 감수)
- ⚠ conf 가 높아질수록 Recall 은 줄어든다 (`R_c` 가 단조 감소)

#### 해석할 때 볼 것
> 숫자(AP50 · mAP50-95 · CPU ms)로 근거를 대고, 데이터의 특징(박스 크기 · 정답 방식)과 연결
- **해상도** → 데이터의 물체 크기에 맞는 해상도를 고른다 (큰 사람이면 320 으로도 충분할 수 있음)
- **AP50 차이 vs mAP50-95 차이** → 위치 정밀도는 mAP50-95 에서 더 크게 드러난다
- **P 가 1 이 아닌 이유** → 정답에 안 달린 사람(작은 · 가려진)을 모델이 찾으면 FP 로 센다
- ⚠ 장치에 올릴 모델은 그 장치에서 직접 재서 고른다

**기본 사용형태**
```python
for c in CONFS:                                          # 0.10, 0.15, ..., 0.90 (17개)
    kept = [d[d[:, 4] >= c] for d in dets_n]
    _, tp, n_gt = match_detections(kept, gts, 0.5)
    P = tp.sum() / max(len(tp), 1); Rc = tp.sum() / n_gt
    P_c.append(P); R_c.append(Rc); F1_c.append(2 * P * Rc / max(P + Rc, 1e-12))
best_conf = float(CONFS[np.argmax(F1_c)])
```

**연결**
- [VB D4·4교시 평가 지표](#/vb/4/4) — `evaluate_ap` · `map_50_95` · `match_detections`
- [VB D4·7교시 영상 탐지 파이프라인](#/vb/4/7) — `process_video` 로 영상 인원 카운트
- [VB D3·8교시 MobileNetV3 vs ResNet-20](#/vb/3/8) — 정확도와 속도를 한 표로 비교하는 같은 사고방식

**상세 학습**

#### 이 교시에서 답하는 질문
오늘 배운 것을 하나로 묶는다. "정확도가 높은 탐지기" 와 "현장에 넣을 탐지기" 는 다르다. 이 교시는 직접 만든 평가 코드로 탐지기 2종 · 해상도 2종을 같은 기준으로 비교하고, 현장용 conf 하나를 고르고, 영상에 적용해 결과를 CSV 로 남긴다.

#### 개념 1 · 평가 — 모든 conf 에서의 성능 = AP
`predict_dataset` 가 conf 0.001 로 낮게 모든 후보를 남기면, `evaluate_ap` 가 PR 곡선 전체를 쓴다.

| 조합 | 보는 지표 |
|---|---|
| YOLOv8n · 640 / 320 | AP50 · mAP50-95 · CPU ms |
| YOLOv8s · 640 / 320 | 같음 |

#### 개념 2 · 운영 conf — F1 로 고르기
AP 는 요약이고, 현장은 conf 하나다. conf 가 높아지면 FP 가 줄어 Precision ↑, 놓치는 사람이 늘어 Recall ↓. F1 이 두 값의 균형점.

| 목적 | 방향 |
|---|---|
| 출입 인원 세기 | F1 최대 근처 |
| 위험구역 침입 감지 | 놓치면 안 됨 → conf 낮춤 (오경보 감수) |

#### 코드로 확인
```python
for name, m in yolos.items():
    dets = predict_dataset(m, 640)
    res[(name, 640)] = {"dets": dets, "AP50": evaluate_ap(dets, gts, 0.5), "mAP": map_50_95(dets, gts)}
```
- **무엇을 하나**: 모델마다 170장을 예측하고 직접 만든 함수로 AP50 · mAP50-95 를 구한다.
- **어떤 개념**: 개념 1.
- **출력**: YOLOv8n 640 AP50 `0.9692` · mAP50-95 `0.8357` (강사 사전 실행 예)

```python
counts_p, times_p = process_video(VIDEO, "count_result.mp4", yolo_n, imgsz=640, conf=best_conf)
with open("people_count.csv", "w") as f:
    f.write("frame,time_s,count\n")
    for i, c in enumerate(counts_p):
        f.write(f"{i},{i / V_FPS:.1f},{c}\n")
```
- **무엇을 하나**: 고른 `best_conf` 로 영상 전체의 인원을 세고 프레임별 CSV 를 쓴다 (열: `frame, time_s, count`).
- **어떤 개념**: 개념 2.
- **출력**: CSV 줄 수 = 프레임 수 + 1 (머리글)

#### 한 번에 정리
- 평가 코드를 직접 가지면 어떤 탐지기든 같은 기준으로 비교.
- AP50 이 같아도 mAP50-95 · 속도가 다르다. 데이터의 물체 크기에 맞는 해상도를 고른다.
- 운영 threshold 는 F1 과 현장 목적(놓침 vs 오경보)으로 정한다.
- ⚠ 평가용 conf(0.001)와 운영용 conf 를 섞지 않는다.

**연관 문제**
<<QA 8>>
