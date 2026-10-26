# 노트 정리 — 작업 지침

현대 AI 인사이트 캠퍼스 강의를 정리하는 정적 사이트. 배포: GitHub Pages (`.github/workflows/site.yml` 이 `main` 에 Merge 될 때 · 매일 07:20 KST 빌드).

## 사용자
- 한국어로 소통. 경영학 전공 · 코딩 초보 — 작업마다 **왜 필요한지, 무엇을 하는지** 쉬운 말로 설명한다.
- 오픈북 시험장에서 이 사이트를 본다 → 노트는 "10초 안에 찾는 레퍼런스". 길게 늘리지 말고 큰 개념 + 개조식 + 표.
- 채팅에서 문제를 내지 않는다 (문제는 사이트에서 푼다).
- 반영은 **확인 후** — 노트는 PR 로 올리고 사용자가 Merge 한다. Claude 는 Merge 하지 않는다.

## 구조
| 경로 | 무엇 |
|---|---|
| `notes/<SUBJ>_dayN.md` | **노트 원본** (직접 작성). SUBJ = ARCH · OS · PY · C · LINUX · MCU · EIO · NN |
| `notes/api_detail.<sid><N>.json` | 함수 상세 `{"이름": {role, args:[[인자, 뜻]], flow, ret, hdr}}` — **Day 별 파일** (PR 끼리 충돌 방지) |
| `notes/code_guide*.json` · `review_quiz*.json` | 코드 읽기 가이드 · 복습 퀴즈 |
| `tools/build_site.py` | 빌드. `SUBJECTS` 에 과목 · 교안 경로 (배운 순서). 교안은 `STUDY_ROOT/AI_CAMPUS` |
| `tools/lecture_parse.py` · `nbparse.py` · `wsparse.py` | html · ipynb/pptx 교안, 워크시트 파서 |
| `tools/dump_lecture.py` | 교안을 교시별 글로 뽑기 — 노트 쓰기 전에 읽는다 |
| `tools/check_note.py` | 노트 검증 (빌드 경고 · 링크 · 함수 상세 · 이름 겹침) |
| `tools/sync_campus.py` | 사용자 PC 에서 교안 → 구글 드라이브 과목 폴더 정리 (클라우드에서 실행하지 않음) |
| `data/` · `anim/` · `index.html` 의 `?v=` | **빌드 산출물** — 노트 PR 에 넣지 않는다 (Actions 가 만든다) |

강사님 교안: https://github.com/imguru-mooc/AI_CAMPUS (공개) — `git clone --depth 1 … $STUDY_ROOT/AI_CAMPUS`.

## 과목별 비중
- **컴퓨터 구조 · 운영체제 · Python**: 시험이 끝난 기록용 — 교시마다 키워드 · 개요 · 핵심 용어 · 핵심 내용만 짧게.
- **C · LINUX · MCU · EIO**: 이미 정리 완료.
- **Neural Network 와 그 뒤에 새로 올라오는 과목**: 중요 — C · LINUX 노트 수준으로 자세하게 (함수 표 + `api_detail`, 기본 사용형태 코드, 실험 숫자 표, 함정 ⚠, 과목 간 연결).

## 노트 형식
```markdown
## Day N — 제목

### M교시 · 제목

**키워드**: `용어1`, `용어2`
**개요**: 한 단락
**핵심 용어**
- **용어(영문)**: 정의                      ← 📖 사전에 자동 수집
**새로 나온 함수**  (또는 **이 교시에서 만드는 함수**, **새로 나온 명령**)
| 함수 | 기본형 | 용도 |                    ← 첫 칸 = 함수 · 명령 사전 이름 (영문자 · 숫자가 있어야 수집)
**핵심 내용**
#### 큰 개념
> 한 줄 정의
- **키워드** → 결과
- ⚠ 함정
##### 자세히                                 ← 접힘 (이유 · 원리)
**기본 사용형태**
```python … ```
**연결**
- [NN D3·5교시 Affine 계층](#/nn/3/5) — 왜 이어지는지
```
- 표 칸 안의 `|` 는 `\|`.
- **닫히지 않은 대괄호 금지** (예: 구간 `[0, ∞)`) — 연결 지도가 다음 링크까지 하나로 묶는다. `0 ~ ∞` 처럼 쓴다.
- 함수 이름이 다른 과목 함수와 같으면 그 과목 상세가 붙는다 (예: Python `open()` ← Linux `open()`) → `with open()` 처럼 구분.
- 노트의 숫자(파라미터 수 · 손실값 · 출력 크기 등)는 **직접 계산해서 확인**한 뒤 쓴다.
- 개인정보(이름 · 이메일 · 전화 · 개인 폴더 경로)를 넣지 않는다 — 저장소는 공개.

## 노트 작업 순서 (Day 하나 = 브랜치 하나 = PR 하나)
1. `git fetch origin main && git checkout -b notes/<sid>-day<N> origin/main`
2. `python3 -I tools/dump_lecture.py $STUDY_ROOT/AI_CAMPUS/<교안 파일> full` 로 교안 · 코드 · 빈칸 정답을 읽는다
3. `notes/<SUBJ>_dayN.md` (+ 필요하면 `notes/api_detail.<sid><N>.json`) 작성
4. `STUDY_ROOT=… python3 -I tools/check_note.py <sid> <SUBJ>_dayN.md` → 문제 0 · 상세 없음 0 (기록용 과목은 상세 생략 가능)
5. **`notes/` 만** 커밋 → push → `main` 으로 PR. 본문: 교시별 요약 표 · 함수 수 · 연결 수 · 점검 결과

## 새 과목이 올라왔을 때
- 강사님 저장소에 처음 보는 폴더(예: `Vision-day1/`)가 생기면: `tools/build_site.py` 의 `SUBJECTS` 에 과목 추가(배운 순서 끝, id · 이름 · 교안 · 워크시트 · 애니메이션 경로), 새 교안 · 워크시트 형식이면 파서 보강, `assets/style.css` 에 `--<id>` 색 · `.sc-<id>`, `tools/sync_campus.py` 의 `SUBJECT_DIRS` 에 폴더 대응 추가 → **과목 추가 PR 을 먼저** 따로 올린 뒤 노트 PR.
- 검증: `python3 -I tools/build_site.py` 경고 0, 브라우저 점검 (`tools/browsertest/`, 로컬 Chromium 으로 `--remote-debugging-port=9333`).
