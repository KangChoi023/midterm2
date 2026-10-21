# 임베디드 학습 위키

C → LINUX → MCU → Embedded IO 시험 대비 정리 위키.
교시마다 **정리 노트 · 예제 코드(+빌드·실행) · 강의 개념 목록 · 연관 문제**를 한 페이지에 모으고,
모르는 개념은 검색해서 **정의와 처음 나온 교시**로 바로 이동한다.

- 사이트: GitHub Pages (Settings → Pages → Branch `main` / root)
- 로컬 확인: `index.html` 을 브라우저로 열면 된다 (서버 불필요)

## 교시 페이지 구성

| 영역 | 출처 | 내용 |
|---|---|---|
| 정리 노트 | `summary/.src/<과목>_dayN.md` (직접 작성) | 키워드 → 개요 → 핵심 용어 → 새로 나온 함수 → 예제 코드가 하는 일 → 핵심 내용 → 기본 사용형태 → 연결 |
| 부록 A · 예제 코드 | `lecture.html` (자동 추출) | 예제별 핵심 코드 + 빌드·실행 명령 + 출력 |
| 부록 B · 강의 개념 | `lecture.html` (자동 추출) | 강의 원문을 주제(h3)별로, 읽음 체크 |
| 연관 문제 | `worksheet.html` (자동 추출) | 보기 클릭 채점, 퀴즈 페이지에서 오답 노트 |

## 노트 작성 규칙 (`summary/.src/*.md`)

```markdown
## Day N — 제목

### M교시 · 제목

**키워드**: `용어1`, `용어2`, …          ← 검색 칩
**개요**: 한 단락
**핵심 용어**
- **용어(영문)**: 정의               ← 용어 사전으로 자동 수집
**새로 나온 함수**                    ← (함수/명령/것 모두 인식) 표의 첫 칸 = 함수·명령 사전
| 함수 | 기본형 | 용도 |
**예제 코드가 하는 일**
**핵심 내용**
**기본 사용형태**
**연결**
- [C D3·5교시 volatile](#/c/3/5) — 왜 이어지는지   ← 과목 연결 지도로 수집
**연관 문제**
<<QA M>>                              ← worksheet M교시 문항
```

## 빌드

```bash
python3 -I tools/build_site.py          # 전 과목 → data/*.js
python3 -I tools/build_site.py c linux  # 일부만
git add -A && git commit -m "…" && git push
```

원본 경로 기본값은 `~/c_study` (`AI_CAMPUS/` 강의·워크시트, `summary/.src/` 노트). 바꾸려면 `STUDY_ROOT=… python3 -I tools/build_site.py`.

| 파일 | 역할 |
|---|---|
| `tools/build_site.py` | 노트·강의·워크시트 → `data/index.js`(목차·검색 색인·사전) + `data/<과목>.js`(교시 본문) |
| `tools/lecture_parse.py` | `lecture.html` 의 `secN` = N교시, `h3` = 토픽, `pre` = 코드(코드/명령/출력 분류). 빈칸 문제(`h4.fill` 이후)는 제외 |
| `tools/wsparse.py` | `worksheet.html` 의 `const Q=[…]` 문항 파서 |
| `assets/app.js` | 해시 라우팅 SPA (교시·검색·용어/함수 사전·연결·퀴즈) |
