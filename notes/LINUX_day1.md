## Day 1 — Shell 필수 · 시스템 콜 · 프로세스

### 1교시 · 오리엔테이션과 환경 점검

**키워드**: `uname -r`, `uname -m`, `nproc`, `lsb_release`, `PuTTY 127.0.0.1:2222`, `FileZilla SFTP`, `apt update/install`, `build-essential`, `$(uname -r)`, `linux-headers`, `strace`, `valgrind`, `perf`, `stress-ng`, `check_env.sh`, `CFS vs EEVDF`, `vCPU`

**개요**: 운영체제 과목에서는 스케줄러·메모리·파일시스템이 어떻게 동작하는지 **애니메이션으로 보았다.** 이 과목(Linux & Shell, 5일)은 같은 동작을 **C 프로그램으로 직접 일으키고 숫자로 측정**한다. 1교시는 그 출발점으로 실습 VM 에 접속하고, 예제를 올리고, 환경이 갖춰졌는지 `check_env.sh` 로 점검한다.

| Day | 주제 | 대표 결과물 |
|---|---|---|
| 1 | 시스템 콜 · 프로세스 | `myshell.c` |
| 2 | 스케줄링 | `jitter.c` |
| 3 | 메모리 | `memlab.c` |
| 4 | 파일시스템 · sysfs | `sysmon.c`, `mysysfs.ko` |
| 5 | 쓰레드 · 동기화 | `edge_pipeline.c` |

**핵심 용어**
- **VM(가상 머신)**: PC 안에서 돌아가는 가짜 컴퓨터. 실습은 VirtualBox 위의 **Ubuntu Server 22.04.5**(GUI 없음, vCPU 2 · 메모리 4GB · 디스크 30GB).
- **vCPU**: VM 에 할당된 가상 CPU 코어 수. 1개면 Day 2(affinity)·Day 5(race) 실습 결과가 나오지 않는다.
- **커널(kernel)**: 운영체제의 핵심. 22.04 의 커널은 **5.15** → 일반 프로세스 스케줄러는 **CFS** (커널 6.6 이후는 EEVDF).
- **PuTTY / SSH**: 터미널로 VM 에 접속하는 도구 (글자만 주고받음).
- **FileZilla / SFTP**: 파일을 VM 에 올리고 내리는 도구 (같은 `.ppk` 키 사용).
- **NAT · 포트 포워딩**: VM 의 실제 주소(10.0.2.15)는 Windows 에서 직접 닿지 않으므로, VirtualBox 가 Windows 의 `127.0.0.1:2222` 를 VM 의 22번 포트로 넘겨 준다.
- **패키지 / 메타 패키지**: 설치 단위 / 여러 패키지를 묶어 설치하는 패키지 (`build-essential` = gcc, g++, make, libc 헤더).
- **명령 치환 `$( )`**: 명령의 출력 결과를 그 자리에 끼워 넣는 셸 문법.

**새로 나온 명령**

| 명령 | 기본형 | 용도 |
|---|---|---|
| `uname` | `uname -r` / `-m` / `-n` / `-a` | 커널 릴리스(`-r`) / 아키텍처(`-m`, x86_64) / 호스트 이름(`-n`) / 전부(`-a`) |
| `lsb_release` | `lsb_release -a` | Ubuntu 버전 확인 |
| `nproc` | `nproc` | CPU(vCPU) 개수를 숫자 하나로 출력 |
| `mkdir` | `mkdir -p 경로` | 중간 디렉터리까지 한 번에 생성 |
| `apt` | `sudo apt update` / `sudo apt install -y 패키지` | 패키지 **목록 갱신**(update) / 패키지 **설치**(install, `-y` = 묻지 않고 예) |
| `bash` | `bash check_env.sh` | 환경 점검 스크립트 실행 (`[ OK ]` / `[WARN]` 출력) |

**설치하는 도구**: `build-essential`(C 빌드 도구 묶음), `gdb`(디버거), `strace`(**시스템 콜 추적**), `valgrind`(메모리 오류 검사), `perf`(성능 분석, `linux-tools-$(uname -r)`), `stress-ng`(부하 생성), `linux-headers-$(uname -r)`(Day 4 커널 모듈 빌드용).

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `check_env.sh` | 환경 점검 스크립트: ① OS·커널(22.04 / 5.15 인지) ② vCPU·메모리·디스크 여유 ③ 가상화 종류와 Secure Boot(켜져 있으면 Day 4 커널 모듈 로드 거부) ④ sudo 권한 ⑤ 필수 도구와 `linux-headers` 설치 여부를 `[ OK ]` / `[WARN]` 으로 보고 |

**핵심 내용**

#### 환경 확인
> 커널 · 배포판 · 코어 수 · CPU 종류를 먼저 확인한다
- **`uname -r`** → 커널 (`5.15.0-125-generic`)
- **`lsb_release -a`** → 배포판 (Ubuntu 22.04.5)
- **`nproc`** → 코어 수 (2)
- **`uname -m`** → `x86_64` = Intel/AMD 64비트 PC / `aarch64` = Jetson · 라즈베리파이 같은 64비트 ARM
- ⚠ **vCPU 1개 · sudo 권한 없음** → Day 2 이전에 반드시 해결
- ⚠ **커널 6.x** → HWE 커널, 스케줄러가 **EEVDF** → Day 2 해석이 달라짐
- ⚠ **Secure Boot 켜짐** → Day 4 커널 모듈 로드가 거부됨
##### 자세히
- 이 노트의 실습 환경은 WSL2(Ubuntu 26.04, 커널 6.18, 코어 8)라 수업 VM 과 다르다
- `check_env.sh` 는 WSL 이면 `[WARN] Day 4 커널 모듈 실습은 대체 과제` 를 출력한다

#### 접속과 파일 전송
> VM 에는 포트 포워딩된 주소로 들어간다
- **PuTTY 접속값** → `127.0.0.1` 포트 **`2222`**
- ⚠ `10.0.2.15:22` 로는 직접 접속 안 됨
- **파일 올리기** → **FileZilla(SFTP)** (PuTTY 로는 끌어다 놓을 수 없음)

#### 패키지 설치
> `apt update` 는 목록만, 설치는 `apt install`
- **`sudo apt update`** → **목록 갱신만** (설치된 것을 올리지 않음)
- **`sudo apt install -y 패키지`** → 설치
- **패키지 이름의 `$(uname -r)`** → 실행 중인 커널 버전에 맞는 패키지를 고르는 명령 치환

#### 도구 구분
> 무엇을 볼 때 어떤 도구인가
- **`strace`** → 시스템 콜 추적
- **`valgrind`** → 메모리 오류
- **`gdb`** → 디버거
- **`perf`** → 성능 분석

**기본 사용형태**
```bash
uname -r          # 5.15.0-125-generic
nproc             # 2
lsb_release -a
mkdir -p ~/linux-course/day1
sudo apt update
sudo apt install -y build-essential gdb strace valgrind stress-ng \
    linux-tools-$(uname -r) linux-headers-$(uname -r)
cd ~/linux-course/day1/src && bash check_env.sh
```

**연관 문제**
<<QA 1>>

### 2교시 · Shell 필수 (압축)

**키워드**: `pwd`, `cd`, `ls -l`, `cat`, `less`, `절대/상대 경로`, `~ . ..`, `mkdir -p`, `touch`, `cp`, `mv`, `rm -r`, `rwx`, `chmod +x`, `>`, `>>`, `2>`, `2>&1`, `순서 함정`, `|`, `grep`, `wc -l`, `nano`, `&`, `jobs`, `$!`, `kill %1`

**개요**: 이 과목에서 Shell 은 **C 프로그램을 빌드·실행·측정하는 도구**로만 쓴다. 2교시는 그 최소 명령만 압축해서 익힌다 — ① 경로·파일·권한 다루기 ② **리다이렉션과 파이프**로 출력의 방향 바꾸기 ③ `nano` 로 파일 고치기 ④ 백그라운드 실행. 핵심은 프로그램의 **출력이 두 갈래(stdout=fd 1, stderr=fd 2)** 라는 점과 셸이 그 갈래를 어디로 보낼지 정하는 문법이다.

**핵심 용어**
- **절대 경로 / 상대 경로**: `/` 로 시작하는 전체 주소 / 현재 위치 기준 주소 (`~`=홈, `.`=현재, `..`=상위).
- **표준 출력(stdout, fd 1) / 표준 에러(stderr, fd 2)**: 프로그램이 정상 결과와 오류 메시지를 내보내는 두 통로. 둘 다 기본은 화면.
- **파일 디스크립터(fd)**: 열린 통로에 붙은 번호 (0=입력, 1=출력, 2=에러).
- **리다이렉션(redirection)**: 통로의 방향을 파일로 바꾸는 것.
- **파이프(`|`)**: 앞 명령의 stdout 을 뒤 명령의 stdin 으로 잇는 연결.
- **권한 `rwx`**: 읽기·쓰기·실행. 세 묶음이 차례로 **소유자 · 그룹 · 그 외**.
- **PID / 백그라운드**: 프로세스 번호 / 끝을 기다리지 않고 실행.

**새로 나온 명령**

| 명령 | 기본형 | 용도 |
|---|---|---|
| `pwd` / `cd` | `pwd` · `cd 경로` · `cd ..` · `cd -` | 현재 위치 출력 / 이동 (`-` 는 직전 위치) |
| `ls -l` | `ls -l [경로]` | 자세히 보기 (맨 앞 `-rwxr-xr-x` = 종류+권한) |
| `cat` / `less` | `cat 파일` · `less 파일` | 내용 보기 (`less` 는 `q` 로 종료) |
| `mkdir -p` | `mkdir -p test/a/b` | 중간 디렉터리까지 한 번에 |
| `touch` / `cp` / `mv` | `touch 파일` · `cp 원본 대상` · `mv 원본 대상` | 빈 파일 / 복사 / 이름 변경·이동 |
| `rm -r` | `rm -r 디렉터리` | 디렉터리째 삭제 — **휴지통 없이 즉시, 되돌릴 수 없음** |
| `chmod` | `chmod +x run.sh` · `chmod u+x run.sh` | 실행 권한 추가 (`u+x` 는 소유자만, `600` 은 rw-------) |
| `>` `>>` `2>` `2>&1` | `cmd > f` · `cmd >> f` · `cmd 2> f` · `cmd > f 2>&1` | stdout 덮어쓰기 / 덧붙이기 / stderr 만 / 둘 다 |
| `\|` | `cmd1 \| cmd2` | 파이프 |
| `grep` / `wc -l` | `grep 패턴 파일` · `wc -l` | 패턴 줄만 고르기 / 줄 수 세기 |
| `nano` | `nano 파일` | 편집 (`Ctrl+O`,Enter 저장 / `Ctrl+X` 종료 / `Ctrl+W` 찾기) |
| `&` · `jobs` · `$!` · `kill` | `sleep 30 &` · `jobs` · `echo $!` · `kill %1` | 백그라운드 실행 / 작업 목록 / 마지막 백그라운드 PID / 종료 |

**예제 코드가 하는 일** ([02_shell/](Linux_day1/02_shell/))

| 예제 | 하는 일 |
|---|---|
| `out_err.c` | stdout 에 한 줄, stderr 에 한 줄을 내보내는 **리다이렉션 실험용** 프로그램 |
| `demo.sh` | `out_err` 를 `>`, `2>`, `> f 2>&1`, `2>&1 > f`, `>>`, `\|` 로 실행해 어디로 가는지 보여 주고, 권한(`chmod`), 백그라운드(`$!`), 파일 조작(`mkdir -p`/`cp`/`mv`/`rm -r`)을 시연 |

**핵심 내용**

#### 경로와 권한
> `..` 은 상위 디렉터리, 권한은 소유자 · 그룹 · 기타 세 칸
- **`cd ../day2`** (현재 `/home/ubuntu/linux-course/day1`) → `/home/ubuntu/linux-course/day2`
- **`-rwxr-x---`** → 소유자 `rwx` · 그룹 `r-x` (읽기+실행만) · 기타 `---`
- ⚠ 실행 권한이 없으면 `./run.sh` → **`Permission denied`** → `chmod +x` (`-rw-r--r--` → `-rwxr-xr-x`)
- **`mkdir -p test/a/b`** → 중간 디렉터리까지 **한 번에**
- ⚠ `rm -r` → **되돌릴 수 없다**

#### 리다이렉션
> stdout(1) 과 stderr(2) 는 따로 보낼 수 있다

| 명령 | stdout | stderr |
|---|---|---|
| `./a.out` | 화면 | 화면 |
| `./a.out > out.txt` | **파일** | 화면 |
| `./a.out > out.txt 2> err.txt` | 파일 | **err.txt** (`2>` 는 stderr 만) |
| `./a.out > both.log 2>&1` | 파일 | **같은 파일** |
| `./a.out 2>&1 > both.log` | 파일 | **화면** ← 순서 함정 |
| `./a.out >> app.log` | 덧붙임 | 화면 |
| `./a.out \| wc -l` | 파이프로 | **화면** (파이프는 stdout 만) |

- **`2>&1`** → "fd 2 를 fd 1 이 *지금* 가리키는 곳으로"
- ⚠ 셸은 왼쪽부터 적용 → **반드시 `> 파일` 뒤에** (`make > build.log 2>&1`), 앞에 쓰면 stderr 는 화면
- **stdout 은 파일·파이프일 때 버퍼링** → stderr 보다 늦게 나온다 (`both.log` 에서 에러 줄이 먼저인 이유, 4교시)

#### 파이프
> 앞 명령의 stdout 을 뒤 명령의 stdin 으로
- **`grep processor /proc/cpuinfo | wc -l`** → processor 줄만 골라 줄 수 = **논리 코어 수** (`nproc` 과 같은 8)

#### 백그라운드와 셸 변수
> `&` 를 붙이면 셸이 기다리지 않는다
- **`sleep 30 &`** → 프롬프트가 바로 돌아옴
- **`$!`** → 마지막 백그라운드 PID
- **`$?`** → 직전 종료 코드 / **`$$`** → 셸 자신의 PID / **`$#`** → 인자 개수
- **`kill` 로 끄면** → 종료 코드 **143** (= 128 + SIGTERM 15)
- ⚠ PuTTY 의 `Ctrl+C` → 복사가 아니라 **실행 중인 프로그램을 멈춤** (SIGINT) / 복사는 드래그, 붙여넣기는 오른쪽 클릭

**기본 사용형태**
```bash
cd ../day2                          # 상위로 갔다가 day2 로
mkdir -p test/a/b && touch test/hello.c && cp test/hello.c test/a/ && mv test/hello.c test/main.c
rm -r test                          # 되돌릴 수 없음
chmod +x run.sh && ./run.sh         # 실행 권한 주기
make > build.log 2>&1               # 출력과 에러를 모두 파일로 (순서 중요)
./a.out > out.txt 2> err.txt        # 따로 담기
date >> log.txt                     # 덧붙이기
grep processor /proc/cpuinfo | wc -l
sleep 30 &   echo $!   kill %1      # 백그라운드 → PID → 종료
```

**연관 문제**
<<QA 2>>

### 3교시 · Makefile · 측정 스크립트

**키워드**: `Makefile`, `패턴 규칙 %: %.c`, `$@ $<`, `$(wildcard *.c)`, `$(SRCS:.c=)`, `수정 시각(mtime)`, `touch`, `Nothing to be done`, `.PHONY`, `missing separator`, `탭`, `run.sh`, `shebang #!/bin/bash`, `$1 $2 $#`, `${3:-기본값}`, `$(seq 1 N)`, `date +%s%N`, `$(( ))`, `CSV`

**개요**: 예제가 10개라면 `gcc -o ...` 를 10번 쳐야 한다. **Makefile** 은 "무엇을 무엇으로 어떻게 만드는가" 를 적어 두면 **바뀐 파일만 다시 빌드**해 준다. 3교시는 ① 패턴 규칙으로 여러 예제를 한 번에 빌드하고 ② make 가 다시 빌드할지 판단하는 기준(**수정 시각**)을 이해하고 ③ **`run.sh`** 로 같은 프로그램을 N회 실행해 걸린 시간을 **CSV** 로 남기는 것을 다룬다. 측정은 한 번으로 믿지 않는다.

**핵심 용어**
- **타깃(target)**: 만들 대상 (실행 파일 이름). **의존 파일(재료)**: 그것을 만드는 데 쓰는 파일.
- **패턴 규칙**: `% : %.c` — 이름이 같은 `.c` 로 같은 이름의 실행 파일을 만드는 일반 규칙.
- **자동 변수**: `$@` = 만들 대상, `$<` = 첫 번째 재료.
- **mtime(수정 시각)**: 파일이 마지막으로 수정된 시각. make 의 재빌드 판단 기준.
- **가짜 타깃(`.PHONY`)**: 파일이 아니라 명령 이름인 타깃 (`all`, `clean`).
- **셔뱅(shebang) `#!/bin/bash`**: 커널에게 "이 파일을 bash 로 실행하라" 고 알려 주는 첫 줄.
- **명령 치환 `$( )`**, **산술 `$(( ))`**.

**새로 나온 것**

| 항목 | 기본형 | 용도 |
|---|---|---|
| 변수 | `CC = gcc` · `$(CC)` | 값을 한 곳에 두고 꺼내 쓰기 |
| 파일 목록 | `SRCS = $(wildcard *.c)` | 현재 폴더의 모든 `.c` 목록 |
| 확장자 치환 | `TARGETS = $(SRCS:.c=)` | 목록에서 `.c` 를 지움 (`a.c b.c` → `a b`) |
| 패턴 규칙 | `%: %.c` ⏎ `<탭>$(CC) $(CFLAGS) -o $@ $<` | 모든 `.c` 를 같은 명령으로 빌드 |
| 실행 | `make` / `make 03_errno` / `make clean` | 전부 / 하나만 / 삭제 |
| 스크립트 인자 | `$1 $2 $3` · `$#` · `$0` | 첫째·둘째·셋째 인자 / 인자 개수 / 스크립트 이름 |
| 기본값 | `${3:-result.csv}` | 셋째 인자가 없거나 비었으면 기본값 |
| 반복 | `for i in $(seq 1 "$N"); do … done` | 1 부터 N 까지 반복 |
| 시각 측정 | `date +%s%N` | 나노초 단위 현재 시각 |
| 조건 | `if [ $# -lt 2 ]; then … fi` | 인자가 2개 미만이면 |

**예제 코드가 하는 일** ([03_make/](Linux_day1/03_make/) — 강의 예제 `src` 를 복사해 실행)

| 예제 | 하는 일 |
|---|---|
| `Makefile` | 폴더의 모든 `.c`(01~09, `myshell`)를 패턴 규칙 하나로 빌드. `make clean` 은 실행 파일만 삭제 |
| `run.sh` | `./run.sh <실행파일> <반복횟수> [출력CSV]` — 프로그램을 N회 실행해 매번의 시간(ms)을 `run,elapsed_ms` CSV 로 저장 |

**핵심 내용**

#### make 재빌드 기준
> 타깃이 재료보다 새로우면 건너뛴다 — 기준은 수정 시각
- **처음 `make`** → 10개 전부 빌드
- **아무것도 안 고치고 다시** → `make: Nothing to be done for 'all'.` (gcc 를 안 부름)
- **`touch 03_errno.c && make`** → `03_errno` **하나만** (`touch` 는 mtime 만 갱신)

#### Makefile 문법
> 패턴 규칙 · 자동 변수 · 치환으로 이름 반복을 없앤다
- **패턴 규칙에서 `03_errno` 빌드** → `$@` = `03_errno`, **`$<` = `03_errno.c`**
- **`TARGETS = $(SRCS:.c=)`** → `a.c b.c` → **`a b`**
- **`.PHONY: all clean`** → `clean` 이라는 파일이 있어도 항상 실행
- ⚠ 명령 줄은 **탭**으로 시작 → 스페이스면 `Makefile:3: *** missing separator. Stop.`

#### 셸 스크립트 인자
> `$1 $2 …` 로 받고, 없으면 기본값
- **`./run.sh ./02_syscall_direct 5 r.csv`** → `$1=./02_syscall_direct`, **`$2=5`**, `$3=r.csv`, `$#=3`
- **`OUT=${3:-result.csv}`** → 세 번째 인자가 없으면 `result.csv`
- **인자 부족** → 사용법 출력 후 **`exit 1`**
- **실행** → `chmod +x` 필요, 첫 줄 **셔뱅** 덕분에 커널이 `/bin/bash run.sh` 로 실행

#### 시간 측정
> 나노초 시각의 차이로 밀리초를 잰다 — 한 번 잰 값은 믿지 않는다
- **`date +%s%N`** → 나노초 단위 현재 시각
- **`(end - start) / 1000000`** → **밀리초**
- **여러 번 재면** → 값이 흔들림 (9, 6, 4 ms)
- **출력 버리기** → `> /dev/null` 로 시간만 잰다

**기본 사용형태**
```make
CC      = gcc
CFLAGS  = -Wall -Wextra -g
SRCS    = $(wildcard *.c)
TARGETS = $(SRCS:.c=)
all: $(TARGETS)
%: %.c
	$(CC) $(CFLAGS) -o $@ $<
clean:
	rm -f $(TARGETS)
.PHONY: all clean
```
```bash
make                          # 전부 빌드 (다시 하면 Nothing to be done)
touch 03_errno.c && make      # 03 만 다시 빌드
chmod +x run.sh && ./run.sh ./02_syscall_direct 5 r.csv
```
```bash
#!/bin/bash
if [ $# -lt 2 ]; then echo "사용법: $0 <실행파일> <반복횟수> [출력CSV]"; exit 1; fi
PROG=$1; N=$2; OUT=${3:-result.csv}
for i in $(seq 1 "$N"); do
    start=$(date +%s%N); "$PROG" > /dev/null; end=$(date +%s%N)
    echo "$i,$(( (end - start) / 1000000 ))" >> "$OUT"
done
```

**Makefile 구조 뜯어보기** (직접 실행한 [03_make/anatomy/](Linux_day1/03_make/anatomy/) 기준: `a.c`, `b.c` 두 파일)

- **규칙(rule)의 모양**: `타깃: 재료들` 한 줄 + 그 아래 **탭으로 시작하는 명령(레시피)**. "재료가 타깃보다 새로우면(또는 타깃이 없으면) 레시피를 실행한다."
- **make 의 두 단계**: ① Makefile 을 위에서 아래로 읽으며 변수 정의와 규칙 등록 (`$(info …)` 가 이때 출력됨) → ② **목표(goal)** 를 정해 의존 관계를 따라 필요한 레시피만 실행. `make` 만 치면 목표는 **파일의 첫 번째 규칙(`all`)** — 그래서 `all` 을 맨 위에 둔다.
- **줄별 해설**

| 줄 | 뜻 |
|---|---|
| `CC = gcc`, `CFLAGS = …` | 변수. `$(CC)` 로 꺼내 씀. 명령줄에서 `make CFLAGS="-O2"` 로 덮어쓸 수 있음 |
| `SRCS = $(wildcard *.c)` | 폴더의 `.c` 파일 목록 → `a.c b.c` |
| `TARGETS = $(SRCS:.c=)` | 각 단어 끝의 `.c` 삭제 → `a b` |
| `all: $(TARGETS)` | `all` 의 재료는 `a b`. 레시피가 없으므로 "재료를 다 만들면 끝"인 **가짜 목표** |
| `%: %.c` | **패턴 규칙**. 타깃 이름이 `a` 이면 `%`=`a` → 재료는 `a.c`. 어떤 이름이든 같은 규칙 하나로 처리 |
| `$(CC) $(CFLAGS) -o $@ $<` | 레시피. `$@`=타깃(`a`), `$<`=첫 재료(`a.c`) → `gcc … -o a a.c` |
| `clean:` | 재료도 없고 파일도 만들지 않는 **명령용 타깃** |
| `.PHONY: all clean` | `all`, `clean` 은 파일 이름이 아님을 선언 → 같은 이름 파일이 있어도 항상 실행 |

- **의존 그래프**: `all` → (`a` ← `a.c`), (`b` ← `b.c`). `touch a.c` 후 `make --debug=b` 는 `Must remake target 'a'` → `gcc -o a a.c` → `Must remake target 'all'` 순으로 **a 쪽 가지만** 다시 만든다. `make b` 로 목표를 `b` 로 정하면 `b` 가지만 본다.
- **레시피의 각 줄은 별개의 셸**에서 실행된다: `cd /tmp` 한 줄 뒤의 `pwd` 는 `/tmp` 가 아니라 원래 위치를 출력한다 (같은 셸이 필요하면 한 줄에 `cd /tmp && pwd`). 줄 앞 **`@`** 는 명령 자체의 에코를 끈다.
- **디버깅 도구**: `make -n`(실행하지 않고 **명령만 출력**) / `make --debug=b`(make 의 판단 과정) / `$(info 문구)`(변수 값 확인).
- **자주 보는 메시지**: `missing separator`(탭 대신 스페이스) / `No rule to make target 'X'`(만드는 방법을 모르는 타깃) / `Nothing to be done for 'all'`(재료가 안 바뀜) / `'a' is up to date`(그 타깃이 최신).

**연관 문제**
<<QA 3>>

### 4교시 · 시스템 콜과 커널 경계

**키워드**: `사용자 모드`, `커널 모드`, `시스템 콜`, `glibc 래퍼`, `write()`, `printf()`, `stdio 버퍼`, `줄 단위/블록 단위 버퍼링`, `strace`, `strace -c`, `syscall()`, `SYS_getpid(39)`, `SYS_write(1)`, `rax`, `man 2/3`

**개요**: 프로그램은 평소 **사용자 모드**에서 돌고, 파일·화면·네트워크·프로세스 생성처럼 하드웨어나 다른 프로세스에 영향을 주는 일은 **시스템 콜**로 **커널**에 부탁해야 한다. 4교시는 그 경계를 눈으로 확인한다 — `printf`(라이브러리 함수)와 `write`(시스템 콜)의 차이, `printf` 의 **stdio 버퍼링**이 출력 순서와 호출 횟수에 미치는 영향, 그리고 `syscall()` 로 래퍼 없이 번호로 직접 호출하는 법.

**핵심 용어**
- **사용자 모드 / 커널 모드**: CPU 의 두 권한 단계. 사용자 모드에서는 장치 접근이 금지되어 있고, 시스템 콜로 커널 모드에 들어가 커널 코드를 실행한 뒤 돌아온다.
- **시스템 콜(system call)**: 사용자 프로그램이 커널에 서비스를 요청하는 공식 창구 (`write`, `getpid`, `fork` …).
- **glibc 래퍼**: 시스템 콜 번호를 레지스터에 넣고 `syscall` 명령을 실행해 주는 **얇은 포장 함수** (`getpid()` 가 그 예).
- **stdio 버퍼**: `printf` 같은 C 라이브러리가 출력을 모아 두는 메모리. 모았다가 `write()` 로 내보낸다.
- **줄 단위 / 블록 단위 버퍼링**: 출력이 **터미널**이면 줄바꿈마다, **파이프·파일**이면 버퍼가 찰 때(또는 종료 때) 한꺼번에 내보냄.
- **strace**: 프로그램이 부른 시스템 콜을 한 줄씩 보여 주는 추적 도구.

**새로 나온 것**

| 항목 | 기본형 | 용도 |
|---|---|---|
| `write` | `ssize_t write(int fd, const void *buf, size_t n);` | fd 에 n 바이트를 **시스템 콜로 직접** 쓴다. **실제로 쓴 바이트 수**를 반환 (fd 1 = stdout) |
| `syscall` | `long syscall(long number, ...);` (`#define _GNU_SOURCE`, `<sys/syscall.h>`) | 시스템 콜을 **번호로 직접** 호출. `SYS_getpid`(39), `SYS_write`(1) @x86_64 |
| `getpid` | `pid_t getpid(void);` | 자기 PID. 내부는 `syscall(SYS_getpid)` 와 같음 |
| `strace` | `strace -e trace=write ./prog` · `strace -c ./prog` | 시스템 콜 추적 / **시스템 콜별 호출 횟수·시간 요약** |
| `man` | `man 2 write` · `man 3 printf` | 섹션 **1=명령, 2=시스템 콜, 3=라이브러리 함수** |

**예제 코드가 하는 일** ([04_syscall/](Linux_day1/04_syscall/))

| 예제 | 하는 일 |
|---|---|
| `01_write_vs_printf.c` | `write()` 로 한 줄, `printf()` 로 다섯 줄을 출력. 터미널에서는 줄마다 `write`, 파이프·파일에서는 `printf` 5줄이 **write 한 번**(70바이트)으로 합쳐짐 (`strace` 로 관찰) |
| `02_syscall_direct.c` | `getpid()` 와 `syscall(SYS_getpid)` 가 같은 값임을 확인하고 `SYS_getpid`(39)·`SYS_write`(1) 번호를 출력, `syscall(SYS_write, …)` 로 직접 출력 |
| `03_order.c` (추가) | **`printf` 를 먼저, `write` 를 나중에** 호출 — 터미널과 파이프/파일에서 출력 순서가 달라지는 것을 확인 |

**핵심 내용**

#### 시스템 콜 흐름
> 사용자 모드는 장치에 직접 쓸 수 없다 — 반드시 시스템 콜
- **흐름** → `printf()` → (glibc 버퍼) → `write()` 래퍼 → `syscall` 명령(번호=1) → **[커널 모드]** `sys_write()` → 터미널 드라이버
- **x86_64 규약** → **`rax` = 번호**, `rdi`·`rsi`·`rdx`… = 인자, **반환값도 `rax`**
- **`errno`** → 실패했을 때 **glibc 래퍼가** 설정 (커널이 쓰는 것이 아님)
- **`getpid()` = `syscall(SYS_getpid)`** → 항상 같은 값 (번호 39 / `SYS_write` 는 1)

#### stdio 버퍼링과 호출 횟수
> 모드 전환 비용 때문에 모아서 적게 부른다
- **버퍼링 이유** → 사용자↔커널 모드 전환 비용
- **`write(1,"A\n",2); printf("B\n"); printf("C\n");` 터미널** → write **3번** (직접 1 + 줄 단위 printf 2)
- **같은 코드 `| cat`** → B·C 는 종료 때 **한 번** → 총 **2번**, 순서는 A → B → C
- ⚠ `printf` 먼저 · `write` 나중 → 파이프·파일에서는 **`write` 문장이 먼저** 나온다 (터미널은 호출 순서 그대로)
- **같은 이유로** → stdout 과 stderr 순서가 뒤집히기도 함 (2교시)

#### strace 읽기
> 시스템 콜 · 인자 · 반환값을 한 줄씩 보여 준다
- **`write(1, "printf line 0\n…"..., 70) = 70`** → `=` 뒤 **70** = 반환값 = 실제로 쓴 바이트 수
- **`strace -c`** → 시스템 콜별 호출 횟수 · 소요 시간 표
- **도구 구분** → `strace` 시스템 콜 / `valgrind` 메모리 / `gdb` 디버거 / `perf` 성능
- (※ `strace` 는 수업 VM 에서 실행 — 로컬 WSL 에는 미설치)

**기본 사용형태**
```c
#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>
write(1, msg, strlen(msg));                      // 시스템 콜 직접 (fd 1 = stdout)
pid_t a = getpid();                              // glibc 래퍼
long  b = syscall(SYS_getpid);                   // 번호로 직접 → a 와 항상 같다
syscall(SYS_write, 1, "직접\n", 5);
```
```bash
make 01_write_vs_printf && ./01_write_vs_printf | cat     # 파이프에서의 순서·버퍼링
strace -e trace=write ./01_write_vs_printf                 # write 만 추적
strace -c ./02_syscall_direct > /dev/null                  # 시스템 콜 통계
man 2 write      # 섹션 2 = 시스템 콜
man 3 printf     # 섹션 3 = 라이브러리 함수
```

**연관 문제**
<<QA 4>>

### 5교시 · 에러 처리와 /proc

**키워드**: `-1 반환 + errno`, `ENOENT(2)`, `EACCES(13)`, `EBADF(9)`, `EISDIR(21)`, `strerror`, `perror`, `errno 규칙`, `/proc`, `procfs`, `/proc/self`, `/proc/self/status`, `/proc/self/fd`, `/proc/loadavg`, `VmRSS`, `malloc vs memset`, `demand paging 예고`

**개요**: 시스템 콜은 **실패한다.** 대부분 실패하면 **`-1` 을 반환**하고 원인 번호를 전역 변수 **`errno`** 에 남긴다 — 반환값을 검사하지 않는 코드는 실패해도 모른 채 다음 줄로 간다. 5교시는 ① 실패를 감지하고 원인을 출력하는 법(`errno`, `strerror`, `perror`) ② 커널이 보여 주는 **가상 파일시스템 `/proc`** 으로 내 프로세스 상태를 읽는 법, 그리고 ③ `malloc` 만 했을 때와 실제로 썼을 때 **실제 메모리 사용량(`VmRSS`)** 이 다르다는 것을 관찰한다.

**핵심 용어**
- **`errno`**: 마지막 시스템 콜 실패의 원인 번호를 담는 전역 변수.
- **`ENOENT`(2) / `EACCES`(13) / `EBADF`(9) / `EISDIR`(21)**: 파일·경로 없음 / 권한 없음 / 열려 있지 않은 fd / 디렉터리를 쓰기로 열려 함.
- **`/proc` (procfs)**: 디스크에 없는 **가상 파일시스템**. 파일을 읽는 순간 **커널이 그때의 값을 만들어 준다** (그래서 크기가 0 으로 보이는 파일이 많다).
- **`/proc/self`**: "지금 이 경로를 읽고 있는 **프로세스 자신**" 을 가리키는 바로가기 (`cat` 이 읽으면 cat 의, 내 프로그램이 읽으면 내 프로그램의 정보).
- **`VmRSS`(Resident Set Size)**: 프로세스가 **실제로 물리 메모리(RAM)에 올려 둔 크기**. 예약한 가상 주소 크기는 `VmSize`.

**새로 나온 것**

| 항목 | 기본형 | 용도 |
|---|---|---|
| `open` | `int open(const char *path, int flags);` | 파일을 열어 **fd** 를 반환, **실패하면 -1** (`O_RDONLY`, `O_WRONLY`) |
| `close` | `int close(int fd);` | fd 를 닫음. 열려 있지 않은 fd 면 -1 + `EBADF` |
| `strerror` | `char *strerror(int errnum);` | errno 번호 → 메시지 |
| `perror` | `void perror(const char *prefix);` | **`접두어: 메시지`** 를 stderr 로 출력 (`close(100): Bad file descriptor`) |
| `/proc/self/status` | `cat /proc/self/status` | 내 프로세스의 Name, State, Pid, PPid, Threads, VmRSS … |
| `/proc/self/fd` | `ls -l /proc/self/fd` | 내가 열어 둔 fd 목록 (0·1·2 + 열린 파일) |
| `/proc/loadavg` | `cat /proc/loadavg` | 시스템 부하 평균 |

**예제 코드가 하는 일** ([05_errno_proc/](Linux_day1/05_errno_proc/))

| 예제 | 하는 일 |
|---|---|
| `03_errno.c` | 네 경로를 `open` 해 성공(`fd=3`) / 실패 원인(`errno=2,13,21`)을 출력하고, `close(100)` 의 실패를 `perror` 로 출력 |
| `04_proc_self.c` | `/proc/self/status` 를 읽어 Name·State·Pid·PPid·Threads·VmRSS 를 출력하고, 50MB 를 `malloc`+`memset` 한 뒤 `VmRSS` 의 변화를 출력 (1760 kB → 52964 kB) |
| `05_no_memset.c` (추가) | `malloc` 만 했을 때와 `memset` 으로 썼을 때의 `VmRSS` 를 비교 (1956 kB → 53156 kB) |
| `06_errno_trap.c` (추가) | `errno` 를 **실패 직후 저장**해야 하는 이유와, 성공한 호출 뒤에도 이전 값이 남아 있는 것을 확인 |

**핵심 내용**

#### 실패 = -1 + errno
> 시스템 콜이 실패하면 `-1` 을 돌려주고 원인은 `errno`
- **`open` 실패** → `fd` 는 **`-1`** (NULL 아님)
- **내부** → 커널이 준 `-2(-ENOENT)` 를 glibc 가 `-1` 과 `errno=2` 로 바꿈

| 호출 | 결과 |
|---|---|
| `open("/etc/hostname")` | 성공, `fd=3` (0·1·2 는 이미 표준 입출력) |
| `open("/no/such/file")` | `errno=2 (No such file or directory)` — **ENOENT** |
| `open("/etc/shadow")` | `errno=13 (Permission denied)` — **EACCES** (`sudo` 면 성공) |
| `open("/etc", O_WRONLY)` | `errno=21 (Is a directory)` — **EISDIR** |
| `close(100)` | `close(100): Bad file descriptor` — **EBADF** |

#### errno 읽는 규칙
> 실패를 반환했을 때만 읽고, 바로 복사한다
- **`perror("접두어")`** → `접두어: strerror(errno)` 를 stderr 로
- ⚠ 성공한 호출은 `errno` 를 바꾸지 않을 수 있음 → **이전 실패 값이 남아 있을 뿐** (성공한 `open` 직후에도 `errno=2`)
- ⚠ 중간의 `printf` 도 `errno` 를 바꿀 수 있음 → **실패 직후 `int saved = errno;`**

#### /proc 가상 파일
> 디스크가 아니라 읽는 순간 커널이 만든다
- **`/proc/self/fd`** → 내가 열어 둔 fd (`0 -> 터미널`, `1 -> 파이프/터미널`)
- **`/proc/self/status` 의 `VmRSS`** → 실제로 물리 메모리를 차지한 크기 (`VmSize` = 예약한 가상 크기)
- ⚠ `fopen("/proc/self/status")` 도 반환값 검사 필수 — `NULL` 인 채 `fgets` 하면 세그폴트

#### malloc 과 RSS
> `malloc` 은 주소 공간만 예약 — 실제로 써야 물리 페이지가 생긴다 (Day 3 demand paging)
- **시작** → 1628 kB
- **`malloc` 50MB 후** → 1956 kB (거의 그대로)
- **`memset` 후** → **53156 kB**

**기본 사용형태**
```c
int fd = open("/no/such/file", O_RDONLY);
if (fd == -1) {
    int saved = errno;                                   // 실패 직후 저장
    printf("errno=%d (%s)\n", saved, strerror(saved));   // errno=2 (No such file or directory)
    perror("open");                                      // open: No such file or directory
    return 1;
}
close(fd);

FILE *fp = fopen("/proc/self/status", "r");
if (fp == NULL) { perror("fopen"); return -1; }
```
```bash
cat /proc/self/status | head -10
ls -l /proc/self/fd
cat /proc/loadavg
grep VmRSS /proc/self/status
```

**연관 문제**
<<QA 5>>

### 6교시 · 프로세스 생성 — fork · exec · wait

**키워드**: `fork`, `반환값 -1/0/양수`, `자식 PID`, `메모리 분리(Copy-on-Write)`, `fflush 후 fork`, `exec`, `execvp`, `PID 유지`, `성공하면 돌아오지 않음`, `_exit(127)`, `wait`, `waitpid`, `WIFEXITED`, `WEXITSTATUS`, `WIFSIGNALED`, `WTERMSIG`, `fork 두 번 = 4개`

**개요**: 셸이 명령을 실행하는 원리인 **프로세스 생성**을 다룬다. **`fork()`** 는 호출한 프로세스를 **복제**해 하나가 둘이 되고(한 번 호출, 두 번 반환), **`exec`** 는 그 프로세스의 코드·데이터를 **새 프로그램으로 통째로 교체**하며, **`wait`/`waitpid`** 는 부모가 자식의 종료를 기다리고 **종료 상태**를 받는다. 이 세 가지 `fork → exec → wait` 가 8교시 `myshell` 의 뼈대다.

**핵심 용어**
- **fork**: 호출한 프로세스를 복제해 자식 프로세스를 만든다. 서류를 복사기에 넣어 사본을 만드는 것.
- **exec**: 현재 프로세스의 코드·데이터를 새 프로그램으로 교체 (**PID 는 그대로**). 같은 사람이 직업만 바꾸는 것.
- **wait / waitpid**: 부모가 자식의 종료를 기다리고 종료 상태(`status`)를 받는다.
- **종료 코드 / 종료 상태(status)**: 자식이 `exit(n)` 으로 남긴 값(0~255) 또는 시그널로 죽은 정보. 매크로로 해석한다.
- **Copy-on-Write(CoW)**: fork 직후에는 메모리를 공유하다가 **쓰는 순간에 복사**하는 방식 (Day 3 에서 측정).

**새로 나온 함수**

| 함수 | 기본형 | 용도 |
|---|---|---|
| `fork` | `pid_t fork(void);` | 프로세스 복제. 반환값: **자식 0 / 부모 = 자식 PID(>0) / 실패 -1(부모)** |
| `execvp` | `int execvp(const char *file, char *const argv[]);` | 새 프로그램으로 교체 (`PATH` 검색). `argv` 는 **`NULL` 로 끝남**. 성공하면 **돌아오지 않음** |
| `waitpid` | `pid_t waitpid(pid_t pid, int *status, int options);` | 그 자식이 끝나길 기다리고 `status` 를 받음 (`wait(&s)` 는 아무 자식) |
| `_exit` | `void _exit(int code);` | 버퍼를 비우지 않고 즉시 종료 (exec 실패한 자식에 사용, 관례 **127**) |
| 상태 매크로 | `WIFEXITED(s)` `WEXITSTATUS(s)` `WIFSIGNALED(s)` `WTERMSIG(s)` | 정상 종료 여부 / 종료 코드 / 시그널로 죽었나 / 그 시그널 번호 |

**예제 코드가 하는 일** ([06_fork/](Linux_day1/06_fork/))

| 예제 | 하는 일 |
|---|---|
| `05_fork_basic.c` | `x=100` 에서 `fork`. 자식은 `x+=1`(101), 부모는 `wait` 후 `x-=1`(99)을 출력 → 메모리가 분리됨을 확인 |
| `06_fork_exec_wait.c` | 자식이 `execvp` 로 명령(기본 `ls -l /tmp`)을 실행하고 부모가 `waitpid` 로 종료 코드를 출력 (`ls /nope` → 2, `nosuchcmd` → 127) |
| `07_fork_noflush.c` (추가) | `fflush` 없이 `fork` — 파이프에서 `[fork 전]` 줄이 **두 번** 나오는 것을 확인 |
| `08_fork_twice.c` (추가) | `fork()` 를 두 번 호출하면 프로세스가 4개가 되는 것을 확인 |
| `09_signal_death.c` (추가) | 자식이 `SIGTERM` 으로 죽으면 부모가 `WIFSIGNALED`/`WTERMSIG`=15 로 알아내는 것을 확인 |

**핵심 내용**

#### fork
> 한 번 호출해 두 번 반환 — 자식은 0, 부모는 자식 PID
- **반환** → 자식 **0** / 부모 **자식의 PID** / 실패 **-1** (부모에게)
- **구분** → `if (pid == 0)` 이면 자식 코드
- **메모리는 분리** → `x=100` → 자식 **101**, 부모 **99** (실제로는 CoW 로 쓰는 순간 복사)
- **`fork` 두 번** → 1 → 2 → **4개**, `printf` 가 **4번**
- ⚠ `fflush(stdout)` 없이 fork → 버퍼까지 복제 → `./a.out | cat` 에서 **`[fork 전]` 이 두 번** (터미널은 줄 단위라 한 번)

#### exec
> 현재 프로세스의 내용물을 새 프로그램으로 교체
- **성공하면** → 다음 줄로 **돌아오지 않는다**
- **유지** → **PID, 부모 관계, 열려 있던 fd** (close-on-exec 제외)
- **교체** → 코드 · 전역/static · 스택 · 힙
- **실패하면** → 다음 줄(`perror`) 실행, 자식은 `_exit(127)` (셸의 `command not found` 와 같은 값)
- **확인** → `nosuchcmd` → **127**, `ls /nope` → **2** (`ls` 가 접근 실패 시 반환)

#### waitpid 와 종료 상태
> 부모는 자식 PID 를 기다려 어떻게 끝났는지 받는다 — 셸이 `$?` 를 얻는 원리
- **정상 종료** → `WIFEXITED(s)` 가 참 → `WEXITSTATUS(s)`
- **시그널로 죽음** → `WIFSIGNALED(s)` 가 참 → `WTERMSIG(s)` (SIGTERM = **15**)
- **`kill <자식PID>`** → 시그널 15
- **자식이 exec 로 바뀌어도** → 부모는 같은 자식 PID 를 `waitpid`

**기본 사용형태**
```c
pid_t pid = fork();
if (pid == -1) { perror("fork"); exit(1); }
if (pid == 0) {                                   // 자식
    char *argv[] = { "ls", "/nope", NULL };       // NULL 로 끝남
    execvp(argv[0], argv);                        // 성공하면 여기로 안 옴
    perror("execvp");  _exit(127);                // 실패했을 때만 실행
}
int status;
waitpid(pid, &status, 0);                         // 부모
if (WIFEXITED(status))        printf("종료 코드=%d\n", WEXITSTATUS(status));
else if (WIFSIGNALED(status)) printf("시그널=%d\n", WTERMSIG(status));
```
```bash
./06_fork_exec_wait ls /nope       # 종료 코드=2
./06_fork_exec_wait nosuchcmd      # execvp 실패 → 127
./05_fork_basic | cat              # fflush 가 없으면 [fork 전] 이 두 번
```

**연관 문제**
<<QA 6>>

### 7교시 · 좀비 · 고아 · 시그널

**키워드:** 좀비 Z defunct, 고아 orphan, init/subreaper 입양, SIGCHLD, SIGINT, SIGKILL, SIGTERM, sigaction, async-signal-safe, volatile sig_atomic_t, WNOHANG, pause, Ctrl+C 130

**개요**
- 6교시에서 만든 자식이 **끝난 뒤 / 부모가 먼저 끝났을 때** 생기는 두 가지 상태(좀비·고아)와, 프로세스 사이의 **비동기 알림인 시그널**을 다룬다.
- 좀비는 `wait` 로 해결하고, 시그널은 `sigaction` 으로 핸들러를 등록해 받는다. SIGCHLD 핸들러로 끝난 자식을 자동 수거하는 것이 마무리 패턴이다.

**핵심 용어**
- **좀비(zombie)**: 끝났지만 부모가 종료 상태를 아직 안 가져간 자식. 메모리는 이미 회수되고 **프로세스 테이블 한 칸(종료 코드)만** 남는다. `ps` 에 `Z`, `<defunct>`. 퇴사했는데 사직서 처리가 안 된 직원.
- **고아(orphan)**: 부모가 먼저 끝난 **살아 있는** 자식. **init(PID 1) 또는 subreaper** 가 입양해 PPID 가 바뀐다.
- **시그널(signal)**: 커널이 프로세스에 보내는 **비동기 알림**. 전화벨처럼 하던 일 중간에 끼어든다.
- **핸들러(handler)**: 시그널이 왔을 때 대신 실행되는 함수. `sigaction` 으로 등록.
- **async-signal-safe**: 핸들러 안에서 안전하게 부를 수 있는 함수(`write` 등). `printf` 는 내부 락 때문에 교착 위험.
- **`volatile sig_atomic_t`**: 핸들러와 main 이 **공유하는 변수** 타입. 매번 메모리에서 읽고(volatile), 원자적으로 읽고 쓰는(sig_atomic_t) 정수.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `sigaction` | `int sigaction(int sig, const struct sigaction *act, struct sigaction *old);` | 시그널 핸들러 등록. `sa.sa_handler = 함수` |
| `sigemptyset` | `int sigemptyset(sigset_t *set);` | 핸들러 실행 중 막을 시그널 집합을 비움 (`sa.sa_mask`) |
| `pause` | `int pause(void);` | 시그널이 올 때까지 잠듦. 핸들러가 끝나면 -1 로 반환 |
| `getppid` | `pid_t getppid(void);` | 부모 PID 조회 (고아가 되면 값이 바뀜) |
| `waitpid(-1, NULL, WNOHANG)` | — | `-1`=아무 자식, `WNOHANG`=끝난 자식 없으면 기다리지 않고 0 반환 |
| `ps -o pid,ppid,stat,cmd --ppid $!` | — | 자식 상태 확인 (STAT 의 `Z` 가 좀비) |

**예제 코드가 하는 일** ([07_signal/](Linux_day1/07_signal/))

| 예제 | 하는 일 |
|---|---|
| `07_zombie.c` | 자식은 바로 `exit(7)`, 부모는 20초 `sleep` 후 `waitpid` → 그 20초 동안 자식이 **좀비(Z)** |
| `08_orphan.c` | 부모가 1초 뒤 종료, 자식은 3초 뒤 `getppid()` 출력 → **PPID 가 바뀜**(고아) |
| `09_sigaction.c` | `SIGINT` 핸들러(카운트+`write`), `SIGCHLD` 핸들러(`waitpid` 반복 수거), Ctrl+C 3번이면 종료 |
| `10_sigkill_catch.c` (추가) | `SIGKILL` 에 핸들러 등록 시도 → **`Invalid argument`**, `SIGTERM` 은 성공 |

**핵심 내용**

#### 좀비와 고아
> 좀비 = 자식이 먼저 끝남(부모가 아직 wait 안 함), 고아 = 부모가 먼저 끝남
- **좀비** → 부모가 `wait` 하기 전까지 `Z` (`54295 54293 Z [07_zombie] <defunct>`), `waitpid` 후 `종료 코드=7` 을 받고 사라짐
- ⚠ 좀비는 `kill -9` 로 못 죽인다 (이미 죽어 있음)
- **없애는 법** → 부모가 `wait` / 부모를 종료 (init 이 입양해 수거)
- **고아** → 자식의 PPID 가 바뀐다 (54324 → 446)
- **새 부모** → 강의 VM(SSH) 은 `1`(systemd), WSL/데스크톱은 subreaper PID
- **`SIGCHLD`(17)** → 자식이 끝나면 커널이 부모에게, 기본 동작은 **무시** — 수거는 `wait` 가 해야 함

#### 주요 시그널
> 번호와 기본 동작, 잡을 수 없는 시그널
- **`SIGINT`(2)** → Ctrl+C / **`SIGKILL`(9)** → 강제 / **`SIGTERM`(15)** → `kill` 기본값 / **`SIGCHLD`(17)**
- ⚠ **`SIGKILL` · `SIGSTOP`** → 잡거나 무시할 수 없다 (`sigaction(SIGKILL)` → `Invalid argument`)
- **Ctrl+C** → 셸이 아니라 **tty 드라이버(커널)** 가 포그라운드 프로세스 그룹에 `SIGINT`
- **핸들러가 없으면** → 죽고 `$?` = **130** (128 + 2)

#### 핸들러 작성 규칙
> 핸들러는 언제든 끼어들 수 있어 안전한 일만 한다
- **① 출력** → `printf` 대신 **`write`** (async-signal-safe)
- **② 공유 변수** → **`volatile sig_atomic_t`**
- **③ SIGCHLD 핸들러** → `while (waitpid(-1, NULL, WNOHANG) > 0);` 로 반복 (겹친 시그널은 한 번만 전달될 수 있음)
- **`09_sigaction` 확인** → SIGINT 3번 → 핸들러 3줄 → `SIGINT 3회 → 종료` (종료 코드 0), 2초 뒤 자식은 SIGCHLD 핸들러가 수거해 좀비 없음

**기본 사용형태**
```c
static volatile sig_atomic_t g_cnt = 0;
static void on_sigint(int sig) { (void)sig; g_cnt++; write(1, "^C\n", 3); }
static void on_sigchld(int sig) { (void)sig; while (waitpid(-1, NULL, WNOHANG) > 0) ; }

struct sigaction sa;
memset(&sa, 0, sizeof sa);
sigemptyset(&sa.sa_mask);
sa.sa_handler = on_sigint;   sigaction(SIGINT,  &sa, NULL);
sa.sa_handler = on_sigchld;  sa.sa_flags = SA_RESTART;  sigaction(SIGCHLD, &sa, NULL);
while (g_cnt < 3) pause();
```
```bash
./07_zombie & ps -o pid,ppid,stat,cmd --ppid $!    # STAT 이 Z
./09_sigaction                                     # Ctrl+C 3번
```

**연관 문제**
<<QA 7>>

### 8교시 · 종합 실습 — myshell 만들기

**키워드:** myshell, 셸 루프, fork-exec-wait, 내장 명령 cd/exit, chdir, strtok, parse, argv NULL 종료, SIG_IGN/SIG_DFL, 128+시그널, 127, Ctrl+D EOF, strace -f, clone/execve/wait4

**개요**
- 1~7교시 내용을 합쳐 **미니 셸 `myshell`(86줄)** 을 완성한다. 셸의 정체는 `읽기 → 자르기 → fork → exec → wait` 를 반복하는 루프다.
- 핵심 질문 두 가지: **왜 `cd` 는 내장 명령인가**, **Ctrl+C 에 셸은 살고 자식은 죽게 하려면 어떻게 하나**.

**핵심 용어**
- **내장 명령(built-in)**: 자식이 아니라 **셸 프로세스가 직접** 처리하는 명령(`cd`, `exit`). 현재 디렉터리는 프로세스별 속성이라 자식이 바꿔 봐야 소용없기 때문.
- **파싱(parse)**: 입력 한 줄을 공백·탭·줄바꿈으로 잘라 `argv[]` 로 만드는 일. 마지막은 **`NULL`**.
- **`SIG_IGN` / `SIG_DFL`**: 시그널을 **무시** / **기본 동작으로 복원**. `signal(SIGINT, …)` 로 지정.
- **종료 코드 관례**: 정상 종료는 그 값, 시그널로 죽으면 **128 + 시그널 번호**(Ctrl+C = 130), 명령을 못 찾으면 **127**.
- **EOF**: 입력 끝. 빈 줄에서 Ctrl+D 를 누르면 `fgets` 가 `NULL` 을 반환.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `chdir` | `int chdir(const char *path);` | **호출한 프로세스**의 현재 디렉터리 변경. 실패 -1 |
| `strtok` | `char *strtok(char *s, const char *delim);` | 문자열을 구분자로 자름. 첫 호출은 `s`, 이어서는 **`NULL`** |
| `signal` | `sighandler_t signal(int sig, sighandler_t h);` | 시그널 처리 지정 (`SIG_IGN`, `SIG_DFL`, 함수). 단순 등록용 |
| `getenv` | `char *getenv(const char *name);` | 환경 변수 조회 (`HOME`) |
| `strace -f` | `strace -f -e trace=clone,execve,wait4 ./myshell` | fork 로 생긴 **자식까지** 시스템 콜 추적 (fork 는 `clone` 으로 보임) |

**예제 코드가 하는 일** ([08_myshell/](Linux_day1/08_myshell/))

| 예제 | 하는 일 |
|---|---|
| `myshell.c` | 프롬프트 `myshell[직전종료코드]$` 를 띄우고, 한 줄을 읽어 잘라서 `cd`·`exit` 는 직접 처리, 나머지는 `fork → execvp → waitpid` 로 실행하고 종료 코드를 저장 |

`myshell.c` 흐름: ① `signal(SIGINT, SIG_IGN)` 셸은 Ctrl+C 무시 → ② `printf`+`fflush` 프롬프트 → ③ `fgets` 가 `NULL` 이면 종료 → ④ `parse` → ⑤ `exit`/`cd` 내장 → ⑥ `fork` → ⑦ 자식: `SIG_DFL` 복원 후 `execvp`, 실패 시 `perror`+`_exit(127)` → ⑧ 부모: `waitpid` 후 `WIFEXITED`/`WIFSIGNALED` 로 `last` 갱신.

**핵심 내용**

#### 셸의 구조
> 입력 읽기 → 자르기 → (내장이면 직접) → fork → exec → wait → 종료 코드 저장
- **bash 도 같은 구조**
- **`argv`** → **`NULL` 로 끝나야** 한다 (`execvp` 가 끝을 `NULL` 로 판단)
- **`strtok` 구분자 `" \t\n"`** → `fgets` 가 남긴 `\n` 도 자동 제거
- **빈 줄** (`argc == 0`) → `continue`
- **Ctrl+D** → 빈 줄에서 EOF → `fgets` 가 `NULL` → `break` (SIGTERM 이 아님)

#### cd 는 내장 명령
> 자식이 `chdir` 하면 자식의 디렉터리만 바뀌고 함께 사라진다
- **확인** → `sh -c 'cd /tmp; pwd'` 는 `/tmp`, 바깥 `pwd` 는 `/home` 그대로
- **myshell** → `cd /tmp` 후 `pwd` = `/tmp`
- **없는 곳으로 `cd`** → `cd: No such file or directory` + 프롬프트 `[1]`

#### Ctrl+C 처리 두 단계
> 셸은 무시, 자식은 exec 전에 기본값으로 되돌린다
- **① 셸** → `SIG_IGN` 으로 안 죽음
- **② 자식** → **exec 전에 `SIG_DFL`** 로 복원
- ⚠ 이유 → **무시 설정은 exec 후에도 상속** (`trap '' INT` 상태의 `sleep` 은 SIGINT 를 받아도 살아 있음)
- **핸들러 함수는** → exec 로 사라져 기본값이 됨 (`SIG_IGN` 만 유지)

#### 프롬프트의 종료 코드
> 프롬프트 숫자 = 직전 종료 코드
- **`echo hi`** → `[0]` / **`ls /nope`** → `[2]` / **`nosuchcmd`** → `perror` 후 `[127]` / **`cd /nonexist`** → `[1]`
- **`sleep 100` 중 SIGINT** → `myshell[130]$` (128+2), 셸은 살아 있음
- **셸 자신에게 SIGINT** → 무시 → 다음 명령(`echo alive`) 정상 실행

#### strace 와 도전 과제
> `-f` 로 자식까지 따라가면 `clone → execve → wait4` 가 보인다
- **`strace -f`** → fork 로 생긴 자식(`clone`, 자식의 `execve`)까지 추적 (강의 VM 에서 확인)
- **도전** → 프롬프트에 `getcwd`, `&` 백그라운드 (+SIGCHLD 수거), `>` 리다이렉션 (`open`+`dup2(fd,1)` 후 exec — Day 4 예고)

**Day 1 한 줄 정리**
| 개념 | 핵심 한 줄 |
|---|---|
| 시스템 콜 | 사용자 프로그램이 커널에 일을 부탁하는 유일한 문 |
| errno | 실패(-1) 했을 때만 읽는 원인 번호 |
| /proc | 읽는 순간 커널이 만들어 주는 가상 파일 |
| fork | 한 번 호출, 두 번 반환 (자식 0, 부모 PID) |
| exec | 같은 PID 에 새 프로그램을 덮어씀 |
| wait | 자식의 종료 상태 수거 — 안 하면 좀비 |
| 시그널 | 커널이 보내는 비동기 알림, sigaction 으로 처리 |

**기본 사용형태**
```c
signal(SIGINT, SIG_IGN);                                  // 셸은 Ctrl+C 무시
for (;;) {
    printf("myshell[%d]$ ", last); fflush(stdout);
    if (fgets(line, sizeof line, stdin) == NULL) break;   // Ctrl+D
    int argc = parse(line, argv);                         // argv 끝은 NULL
    if (argc == 0) continue;
    if (strcmp(argv[0], "exit") == 0) break;
    if (strcmp(argv[0], "cd") == 0) { last = chdir(argc > 1 ? argv[1] : getenv("HOME")) == -1; continue; }
    pid_t pid = fork();
    if (pid == 0) { signal(SIGINT, SIG_DFL); execvp(argv[0], argv); perror(argv[0]); _exit(127); }
    int st; waitpid(pid, &st, 0);
    if (WIFEXITED(st)) last = WEXITSTATUS(st);
    else if (WIFSIGNALED(st)) last = 128 + WTERMSIG(st);
}
```
```bash
make myshell && ./myshell
strace -f -e trace=clone,execve,wait4 ./myshell
```

**연관 문제**
<<QA 8>>
