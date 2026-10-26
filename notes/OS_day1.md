## Day 1 — 관리인을 만난다: File System 과 두 개의 창구

### 1교시 · 오프닝

**키워드**: `3일의 지도`, `File System`, `Process`, `Script`

**개요**: 3일의 도착점 — Terminal 명령 여러 줄을 Script 하나로 묶어 폴더 생성부터 정리 · 기록까지 한 번에. Day 1 File System · GUI vs CLI → Day 2 Process · 자원 → Day 3 Script 자동화.

**핵심 내용**

#### 3일의 지도
- **Day 1** → File System(directory · path) · GUI vs CLI
- **Day 2** → Process · CPU 스케줄링 · Memory
- **Day 3** → 파일 I/O 의 커널 내부 · Shell Script · 자동 실행

**연결**
- [구조 D1·5교시 부품이 주고받는 것](#/arch/1/5) — 컴퓨터 구조의 4대 부품 위에서 OS 가 일한다

### 2교시 · Operating System

**키워드**: `OS`, `시스템 소프트웨어`, `4가지 관리 업무`, `GUI`, `CLI`

**개요**: OS 는 하드웨어와 프로그램 사이에서 자원을 관리 · 분배하는 관리인. 프로그램은 하드웨어를 직접 만지지 않고 OS 에 요청한다. OS 를 부르는 창구는 GUI 와 CLI 두 가지.

**핵심 용어**
- **운영체제(OS, Operating System)**: 하드웨어 자원을 관리하고 프로그램에 분배하는 시스템 소프트웨어
- **GUI(Graphical User Interface)**: 마우스로 화면을 조작하는 창구 — File Explorer
- **CLI(Command Line Interface)**: 키보드 명령으로 지시하는 창구 — Terminal

**핵심 내용**

#### 3층 그림
> 프로그램 → OS → 하드웨어(CPU · Memory · Storage · I/O)
- **OS 가 없다면** → 두 프로그램이 같은 Memory 칸을 덮어써 충돌
- **OS 가 있으면** → 빈 구획을 확인해 서로 다른 구획 배정

#### OS 의 4가지 관리 업무
- **File System 관리** → 파일을 tree 로 정리, path 로 찾기 (Day 1)
- **Process 관리** → 실행 중인 프로그램에 CPU 배분 (Day 2)
- **Memory 관리** → 구획 배정과 회수 (Day 2)
- **Device(I/O) 관리** → 키보드 입력 → 화면 출력 중계

#### GUI vs CLI
> 같은 OS 를 부르는 두 창구 — 결과는 같다
- **GUI** → 보이는 대로, 배우기 쉬움 · 반복에 손이 많이 감
- **CLI** → 글자로 정확히 · 반복 · 자동화 · 원격에 압도적

**연결**
- [LINUX D1·4교시 시스템 콜과 커널 경계](#/linux/1/4) — "프로그램은 OS 에 요청한다"의 실제 모습

### 3교시 · File System

**키워드**: `Drive`, `Root`, `Directory Tree`, `절대 경로`, `상대 경로`, `..`, `separator`

**개요**: 모든 파일에는 주소(path)가 있다. Root 에서 출발하는 절대 경로, 지금 위치(Working Directory)가 기준인 상대 경로.

**핵심 용어**
- **Root**: 모든 경로가 출발하는 맨 위 — Windows `C:\`, Linux `/`
- **Directory**: 폴더. 파일과 다른 directory 를 담는다
- **절대 경로(absolute path)**: Root 부터 전체 주소 — 어디서 읽어도 같은 파일
- **상대 경로(relative path)**: Working Directory 기준 주소 — 기준이 바뀌면 주소도 바뀐다
- **Working Directory(현재 위치)**: 지금 서 있는 directory

**핵심 내용**

#### 두 가지 주소
> 절대 = 도로명 주소, 상대 = "여기서 두 블록 위"
- **절대** → `C:\Users\student\Documents\report.txt`
- **상대** → `Downloads` 에서 `..\Documents\report.txt`
- **`..`** → 상위 directory 로 한 칸 · **`.`** → 지금 위치

#### Separator 와 표기
| OS | 예 | separator |
|---|---|---|
| Windows | `C:\Users\student` | `\` · Drive 문자 |
| macOS · Linux | `/home/student` | `/` · `/` 에서 시작 |
- ⚠ 이름에 못 쓰는 문자 `\ / : * ? " |` · 공백 든 이름은 따옴표로

### 4교시 · 확장자와 숨김 파일

**키워드**: `확장자`, `해석 방법`, `이중 확장자`, `파일 확장명 표시`

**개요**: 확장자는 내용이 아니라 "어떤 프로그램으로 열지(해석 방법)"를 정한다. 확장자를 바꿔도 바이트는 그대로. 확장자 숨김 상태의 이중 확장자는 위장 파일의 함정.

**핵심 용어**
- **확장자(extension)**: 파일 이름 끝의 `.txt` `.jpg` `.exe` 등. OS 가 연결할 프로그램을 정한다

**핵심 내용**

#### 확장자 = 해석 방법
> 내용(바이트)은 고정, 해석만 바뀐다
- **`.txt` → `.jpg`** → 같은 바이트를 그림으로 해석 → 실패
- **이중 확장자** → `이력서.pdf.exe` 가 숨김 상태에선 `이력서.pdf` 로 보임 — 진짜 확장자는 **맨 끝**
- ⚠ 대응 → File Explorer 에서 "파일 확장명" 표시 켜기

**연결**
- [구조 D2·8교시 종합](#/arch/2/8) — 같은 8칸도 읽는 규칙이 정한다 — 확장자가 그 규칙

### 5교시 · Terminal 첫 대면

**키워드**: `Terminal`, `Shell`, `Prompt`, `cd`, `dir`, `Tab 자동완성`

**개요**: Terminal 은 창, Shell 은 통역, OS 는 실행. Prompt 가 지금 위치를 보여 준다. 탐색 3종 — 어디인가 · 뭐가 있나 · 이동.

**핵심 용어**
- **Terminal**: 글자를 받고 보여 주는 화면 — 입출력 창구
- **Shell**: 입력된 글자를 명령으로 해석하는 통역 — CMD · PowerShell · bash
- **Prompt**: 입력을 기다리는 표시. `C:\Users\student>` 처럼 현재 위치를 보여 준다

**새로 나온 명령**
| 명령 | 기본형 | 용도 |
|---|---|---|
| `cd` | `cd [경로]` · `cd ..` | 인자 없으면 현재 위치 출력, 있으면 이동 (Linux `pwd` · `cd`) |
| `dir` | `dir [경로]` | 목록 보기 (Linux `ls`) |

**핵심 내용**

#### 명령 한 줄의 여행
> 입력 → Terminal → Shell(해석) → OS(실행) → 결과가 역순으로 화면에
- **명령 구조** → `명령어 [옵션] [인자]`
- **습관** → `Tab` 자동완성 · `↑` 이전 명령
- ⚠ 에러 점검 순서 → 오타 · 대소문자 → separator → 공백 · 따옴표

**연결**
- [LINUX D1·2교시 Shell 필수](#/linux/1/2) — 같은 탐색을 bash 로 (`pwd` · `ls` · `cd`)

### 6교시 · CLI로 만들고 다루기

**키워드**: `mkdir`, `copy`, `move`, `ren`, `del`, `따옴표`, `토큰`

**개요**: 조작 5종 — 만들고 · 복사하고 · 옮기고 · 바꾸고 · 지운다. Shell 은 공백으로 인자를 자르므로 공백 든 이름은 따옴표로. `del` 은 휴지통이 없다.

**새로 나온 명령**
| 명령 | 기본형 | 용도 |
|---|---|---|
| `mkdir` | `mkdir backup` | directory 생성 |
| `copy` | `copy report.txt backup\` | 복사 — 원본 유지 (Linux `cp`) |
| `move` | `move photo.jpg backup\` | 이동 — 원본 사라짐 (Linux `mv`) |
| `ren` | `ren report.txt final.txt` | 이름 변경 — 자리 이동 없는 move |
| `del` | `del final.txt` | 삭제 — 휴지통 없이 즉시 (Linux `rm`) |

**핵심 내용**

#### 공백과 따옴표
> Shell 은 공백을 인자 구분자로 읽는다
- **`mkdir My Folder`** → 토큰 [mkdir][My][Folder] → directory **2개**
- **`mkdir "My Folder"`** → 토큰 [mkdir][My Folder] → 1개

#### del 은 되돌릴 수 없다
- **GUI 삭제** → 휴지통 → 복원 가능
- ⚠ **CLI `del`** → 즉시 소멸 — 지우기 전에 `dir` 로 대상 확인

### 7교시 · GUI vs CLI

**키워드**: `같은 결과, 다른 과정`, `반복`, `100배 스케일`

**개요**: 같은 directory 구조를 GUI 로 한 번, CLI 로 한 번 만들어 시간과 과정을 비교. GUI 는 클릭 17회 · 8단계, CLI 는 명령 4줄.

**핵심 내용**

#### 같은 결과물, 다른 과정
- **GUI** → 개수만큼 마우스 반복
- **CLI** → `mkdir` ×3 + `copy` ×1 — 100개가 돼도 명령은 짧다 (→ Day 3 Script)

### 8교시 · 종합

**키워드**: `GUI ↔ CLI 대응표`

**개요**: 같은 일, 두 이름 — GUI 동작과 CLI 명령의 대응표로 하루 정리.

**핵심 내용**

#### 대응표
| GUI | CLI (CMD) | Linux |
|---|---|---|
| 새 폴더 | `mkdir` | `mkdir` |
| 복사 → 붙여넣기 | `copy` | `cp` |
| 잘라내기 → 붙여넣기 | `move` | `mv` |
| 이름 바꾸기 | `ren` | `mv` |
| 삭제 | `del` (휴지통 없음) | `rm` |

**연결**
- [OS D2·1교시 프로세스의 일생](#/os/2/1) — 다음 날: 실행 중인 프로그램(Process)
