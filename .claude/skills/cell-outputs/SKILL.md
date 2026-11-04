---
name: cell-outputs
description: 위키 강의 코드의 "▶ 실행하면" 상자에 들어갈 실행 결과(notes/cell_outputs.*.json)를 사용자 PC 에서 다시 만든다. "실행 결과 다시 만들어줘", "교안 기대값을 실제 값으로", "cell_outputs 돌려줘", "새 Day 실행 결과 만들어줘" 같은 요청에 쓴다. 사용자 PC(Windows)에서 도는 Claude 용 — 클라우드 세션은 CIFAR-10 · 사전학습 가중치 다운로드가 막혀 일부를 못 돌린다.
---

# 실행 결과 다시 만들기 (PC)

사용자는 경영학 전공 · 코딩 초보다. 한국어로, 무엇을 왜 하는지 쉬운 말로 짧게 설명한다.
무엇을 하는지: `tools/cell_outputs.py` 가 강사님 노트북을 처음부터 끝까지 실제로 실행해 빈칸 정답 코드의 출력을 `notes/cell_outputs.<nn|vb><Day>.json` 에 모은다.
위키는 이 값을 "▶ 실행하면 · 직접 실행" 상자로 보여 준다. 실행하지 못한 문제는 "교안 기대값"으로 남는다.
이미 있는 값은 실패해도 지워지지 않고, `"fixed": true` 항목은 덮어쓰지 않는다.

## 순서
1. **이 저장소 최신화**
   - `git -C <이 저장소> pull --ff-only`
   - 실패하면 멈추고 상황을 설명한다.
   - 그다음 새 브랜치: `git checkout -b notes/cell-outputs-YYMMDD origin/main`
2. **교안 준비** — 이 저장소 폴더와 **같은 곳**에 `AI_CAMPUS` 가 있으면 도구가 알아서 찾는다.
   - 없으면 `git clone --depth 1 https://github.com/imguru-mooc/AI_CAMPUS.git` 를 저장소 옆 폴더에서 실행한다.
   - 있으면 `git -C ..\AI_CAMPUS pull`.
   - 다른 곳에 두었으면 아래 명령마다 `--root "<AI_CAMPUS 가 들어 있는 폴더>"` 를 붙인다.
3. **Python · 패키지**
   - Python 찾기: `py -V` → 없으면 `python -V`. 아래 `py -3` 은 찾은 것으로 바꾼다.
   - 패키지 설치: `py -3 -m pip install numpy matplotlib ipython torch torchvision opencv-python-headless ultralytics`
     - 처음엔 몇 GB · 10분 이상 걸릴 수 있다고 미리 말한다.
   - NVIDIA GPU 가 있으면(`nvidia-smi` 가 됨) https://pytorch.org 의 CUDA 판 torch 설치 명령을 쓰면 VB 학습 셀이 훨씬 빨리 끝난다고 한 줄 알려 준다. 없으면 CPU 로 그대로 진행한다.
4. **실행** — 무엇을 돌릴지 사용자에게 묻는다.
   - **"교안 기대값"만 채우기 (기본)**: `py -3 -I tools\cell_outputs.py all --only assert --timeout 1800`
     - 지금은 NN Day 4 · VB Day 2 · 3 · 4 가 대상이다.
   - **새로 올라온 Day 하나**: `py -3 -I tools\cell_outputs.py vb 5 --timeout 1800`
   - **전부**: `py -3 -I tools\cell_outputs.py all --timeout 1800`
   - 알려 줄 것:
     - 셀마다 `[셀 12] 3.1s ok` 처럼 진행이 찍힌다.
     - `--timeout` 은 셀 하나에 주는 최대 시간(초)이다. 넘으면 그 셀은 건너뛴다.
     - GPU 가 없으면 VB 학습 셀(ResNet-20 20 epoch 등)은 시간 제한에 걸려 계속 "교안 기대값"으로 남을 수 있다. 정상이다.
     - 전체는 몇 시간 걸릴 수 있다. 실행 중에는 PC 를 끄지 않는다.
5. **결과 확인**
   - `git diff --stat notes/` 로 바뀐 파일을 본다.
   - Day 마다 마지막 줄 `→ … N개` 를 표로 정리한다.
   - 시간(ms · 초) 숫자만 바뀐 파일은 이 PC 기준 값으로 바뀐 것이라고 설명한다. 되돌릴지 사용자에게 묻는다.
6. **올리기** — `notes\cell_outputs.*.json` **만** 커밋한다.
   - `git add notes\cell_outputs.*.json`
   - `git commit -m "실행 결과: …"`
   - `git push -u origin <브랜치>`
   - PR 만들기
     - `gh` 가 있으면 `gh pr create --fill` 로 만든다.
     - 없으면 GitHub 화면의 "Compare & pull request" 버튼을 누르라고 알려 준다.
   - Merge 는 사용자가 한다.

## 하지 않는 것
- `notes/cell_outputs.*.json` 이외 파일의 수정 · 커밋.
  - 도구가 안 돌면 오류 메시지를 보여 주고, 고치는 일은 클라우드 세션에 PR 로 요청하자고 안내한다.
- `data/` · `anim/` · `index.html` 커밋 — 빌드 산출물이다.
- `main` 에 직접 push.
