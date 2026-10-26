## Day 3 — 커널이 파일을 다루는 방식 → 스크립트 자동화

### 1교시 · open()·close() — fd 와 inode

**키워드**: `fd`, `fd 테이블`, `struct file`, `dentry`, `inode`, `참조 수`, `/proc/PID/fd`

**개요**: 오늘 하루 재사용할 다섯 상자 — 프로세스의 fd 테이블 → struct file → dentry → inode → 디스크 블록. fd 는 그냥 배열 인덱스이고, 파일은 이름이 아니라 inode 다.

**핵심 용어**
- **fd(file descriptor)**: 프로세스 fd 테이블의 배열 인덱스. 0 = stdin, 1 = stdout, 2 = stderr, 새로 열면 3 부터
- **struct file**: 열린 파일 하나. 읽기 위치 `f_pos` 와 inode 포인터, 참조 수를 가진다
- **dentry**: 경로의 이름표 한 칸 (`/` → `home` → `a.txt`)
- **inode**: 파일 본체 정보 — 권한 · 크기 · 블록 위치 · 링크 수. 이름은 없다

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `open()` | `int fd = open(path, flags);` | 경로를 따라 inode 를 찾고 struct file 을 만들어 빈 fd 슬롯 번호 반환 |
| `close()` | `close(fd);` | fd 슬롯 비우기 — 참조 수가 0 이면 struct file 해제 (지우기가 아니라 손 떼기) |

**핵심 내용**

#### open() 아홉 단계
> 경로는 dentry 에서 끝나고, 그 뒤는 전부 inode
- **경로 탐색** → dentry 캐시를 `/` → `home` → `user` → `a.txt` 순서로 → inode #8812
- **권한 검사** → inode 의 권한 비트 (셸이 아니라 커널이)
- **struct file 생성** → f_pos = 0, 참조 수 1 → **빈 fd 슬롯(3)** 에 연결, 3 반환

#### rm 했는데 df 가 안 줄어드는 이유
> 이름표(dentry) · 열린 파일(struct file) · 본체(inode · 블록)는 따로 사라진다
- **열린 채로 rm** → 이름표만 제거, struct file 이 inode 를 잡고 있어 블록 해제 안 됨 → `/proc/PID/fd` 에 `(deleted)`
- **마지막 close** → 비로소 블록 해제

**연결**
- [LINUX D4·1교시 파일 디스크립터](#/linux/4/1) · [LINUX D4·2교시 inode 와 디렉터리](#/linux/4/2)

### 2교시 · read()·write() — 디스크는 언제 실제로 움직이나

**키워드**: `페이지 캐시`, `캐시 미스 · 히트`, `dirty`, `writeback`, `fsync`

**개요**: read 는 두 번째부터 빠르고(페이지 캐시 히트), write 는 아직 디스크에 없다(dirty 페이지). 디스크에 닿는 시점은 writeback 스레드 또는 fsync 가 정한다.

**핵심 용어**
- **페이지 캐시(page cache)**: 파일 내용을 담아 두는 RAM. inode 단위로 공유
- **dirty 페이지**: 캐시에서 고쳐졌지만 아직 디스크에 안 쓴 페이지
- **writeback**: dirty 페이지를 디스크로 내보내는 커널 동작 (약 30초 주기)

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `read()` | `read(fd, buf, 4096);` | 페이지 캐시에서 buf 로 복사, 없으면(미스) 디스크에서 먼저 채움 |
| `write()` | `write(fd, buf, n);` | 페이지 캐시에만 쓰고 dirty 표시 후 즉시 반환 |
| `fsync()` | `fsync(fd);` | dirty 페이지를 디스크에 쓸 때까지 잠들었다 반환 — 디스크 보장 |

**핵심 내용**

#### read 는 두 번째부터 빠르다
- **미스** → 디스크(ms) → 캐시 → `copy_to_user`
- **히트** → 캐시에서 바로(µs) — 같은 모델 파일 두 번째 로드가 빠른 이유
- **f_pos 는 struct file 마다**, 캐시는 inode 단위 공유

#### write 는 아직 디스크에 없다
> write 반환 = 캐시에 썼다, 디스크에 썼다가 아니다
- **방치** → dirty → 약 30초 뒤 writeback
- **fsync** → 디스크 기록 완료까지 대기
- ⚠ write 성공 직후 전원 차단 → dirty 페이지 소멸 — 보드 전원을 끊기 전엔 `sync`

**기본 사용형태**
```bash
dd if=/dev/urandom of=big bs=1M count=200     # 200MB 파일
time cat big > /dev/null                      # 두 번 실행: 느림 → 빠름
free -h                                       # buff/cache 증가
echo 3 | sudo tee /proc/sys/vm/drop_caches    # 캐시 비우기 → 다시 느려짐
```

**연결**
- [LINUX D4·3교시 버퍼링과 동기화 비용](#/linux/4/3) · [LINUX D4·4교시 Page cache](#/linux/4/4)

### 3교시 · fork()·dup2() — 프로세스가 복제될 때 fd는 어떻게 되나

**키워드**: `fd 테이블 복사`, `struct file 공유`, `dup2`, `리다이렉션 >`

**개요**: fork 는 fd 테이블(화살표)을 복사하고 struct file 은 공유한다 — 그래서 f_pos 가 같이 움직이고, 그래서 `>` 리다이렉션이 된다.

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `dup2()` | `dup2(oldfd, newfd);` | newfd 슬롯의 화살표를 oldfd 가 가리키는 struct file 로 바꿈 |

**핵심 내용**

#### 복사되는 것 = 화살표, 공유되는 것 = struct file
- **부모가 100B 읽으면** → 자식은 100 부터 (f_pos 공유)

#### `ls > out.txt` 일곱 단계
> fd 1 의 화살표만 파일로 바꾸고 execve — ls 는 모른다
- bash fork → 자식 `open(out.txt)` = fd 3 → `dup2(3, 1)` → `close(3)` → `execve(ls)`(fd 테이블 유지) → ls 는 fd 1 에 씀 → 파일로

**연결**
- [OS D2·2교시 fork()](#/os/2/2) — Memory 는 복제, fd 의 struct file 은 공유

### 4교시 · ./collect.sh — 스크립트가 실행되고 프로세스가 된다

**키워드**: `#!`, `x 비트`, `chmod +x`, `$?`, `빌트인 cd`, `vi`

**개요**: 텍스트 파일이 실행되는 건 커널이 x 비트를 검사하고 첫 두 바이트 `#!` 를 읽어 인터프리터를 대신 띄우기 때문. 스크립트는 한 줄마다 fork · exec · wait 를 반복하는 공장.

**핵심 용어**
- **shebang(`#!`)**: 스크립트 첫 줄. 커널이 읽고 뒤의 인터프리터(`/bin/bash`)를 실행
- **종료 코드(`$?`)**: 직전 명령의 종료값. 0 = 성공, 0 이 아니면 실패
- **빌트인(builtin)**: fork 없이 bash 자신이 처리하는 명령 (`cd` 등)

**새로 나온 명령**
| 명령 | 기본형 | 용도 |
|---|---|---|
| `chmod` | `chmod +x collect.sh` | 실행 권한(x 비트) 부여 |
| `vi` | `vi collect.sh` → `i` 입력 → `Esc` → `:wq` | 터미널 편집기 (세 모드) |

**핵심 내용**

#### 커널이 `#!` 를 읽는 순간
- **① x 비트 검사** → 없으면 여기서 Permission denied(EACCES), `#!` 는 읽지도 않음
- **② 첫 2바이트 `#!`** → 스크립트, 뒤 경로가 인터프리터
- **③ `/bin/bash` 를 대신 exec** → argv = [/bin/bash, ./run.sh]
- **`bash run.sh`** → 실행 대상은 /bin/bash → run.sh 는 r 만 있으면 됨

#### 한 줄 = 한 프로세스
> fork → exec → wait → `$?` → 다음 줄
- `ls -l` → 0, `ls /nonexist` → 2
- ⚠ 스크립트 안의 `cd` → 자식 bash 의 위치만 바뀜, 터미널은 제자리

**기본 사용형태**
```bash
#!/bin/bash
# collect.sh — v1
echo start
```

### 5교시 · 변수의 수명과 스크립트의 판단

**키워드**: `./ vs source`, `export`, `[ ]`, `if`, `&&`, `||`, `for`

**개요**: 변수 상자는 프로세스마다 하나 — 부모 → 자식 한 방향으로만 흐른다. `[` 는 명령이고 if · && · || 는 `$?` 가 0 인지 본다.

**새로 나온 명령**
| 명령 | 기본형 | 용도 |
|---|---|---|
| `source` | `source collect.sh` | fork 없이 현재 셸이 직접 실행 — 변수가 남음 |
| `export` | `export LOG_DIR=~/logs` | execve 때 자식 환경(envp)에 복사될 변수로 지정 |
| `[` | `[ -d "$DIR" ]` | 조건 검사 명령. 참이면 exit 0 |

**핵심 내용**

#### 변수는 어디까지 사나
> 부모 → 자식 한 방향, 되돌아오지 않는다
- **`./collect.sh`** → 자식 bash 의 상자에 LOG_DIR → 자식 종료로 소멸 → 터미널에선 빈 값
- **`source collect.sh`** → 터미널 bash 가 직접 → LOG_DIR 남음 (`.bashrc` 를 source 로 읽는 이유)
- **`export`** → 자식에 복사될 뿐, 자식이 바꿔도 부모는 그대로

#### 조건 = 종료 코드 읽기
- **`[ -d ~/logs ] || mkdir -p ~/logs`** → 있으면 0 → 오른쪽 건너뜀
- **`cp a b && echo 복사됨`** → cp 실패면 echo 는 fork 조차 안 됨
- **0 이 참인 이유** → 성공은 하나, 실패 이유는 여럿

**기본 사용형태**
```bash
LOG_DIR=~/logs
[ -d "$LOG_DIR" ] || mkdir -p "$LOG_DIR"
for i in 1 2 3; do date >> "$LOG_DIR/test.log"; done
```

### 6교시 · 파이프로 로그 만들기

**키워드**: `파이프`, `커널 버퍼 64KB`, `동시 실행`, `흐름 제어`, `SIGPIPE`, `2>&1`

**개요**: 파이프는 3교시 dup2 를 두 프로세스에 동시에 적용한 것 — 커널 버퍼 하나에 양쪽이 꽂혀 동시에 돈다. collect.sh 본체 완성.

**핵심 용어**
- **파이프(pipe)**: 커널 버퍼(64KB) 하나를 사이에 두고 왼쪽 프로세스의 stdout 을 오른쪽 stdin 으로 잇는 통로
- **SIGPIPE**: 읽는 쪽이 먼저 끝났는데 쓰려고 할 때 쓰는 쪽에 오는 시그널

**핵심 내용**

#### 파이프 = pipe() + fork 두 번 + 각자 dup2 + 각자 execve
> 양쪽은 순서대로가 아니라 동시에 돈다
- **버퍼가 가득** → write 블록, 쓰는 쪽을 재움 — 데이터는 안 버려짐 (흐름 제어)
- **오른쪽이 먼저 종료** → 왼쪽에 SIGPIPE
- **`2>&1`** → stderr 도 stdout 과 같은 곳으로

**기본 사용형태**
```bash
{ cat /sys/class/thermal/thermal_zone0/temp \
  | awk -v t="$(date '+%F %T')" '{printf "%s temp=%.1fC\n", t, $1/1000}'
} >> "$LOG_DIR/sensor.log" 2>&1
```

### 7교시 · 사람 없이 돌게 하기 — 백그라운드와 cron

**키워드**: `&`, `SIGHUP`, `nohup`, `setsid`, `cron`, `crontab`, `PATH`

**개요**: 터미널을 닫으면 커널이 세션에 SIGHUP 을 보내 백그라운드 작업도 죽는다 — 무시(nohup)하거나 세션을 떠나면(setsid) 산다. cron 은 매 분 다섯 칸을 대조해 실행하지만 내 터미널 환경(PATH)을 모른다.

**핵심 용어**
- **SIGHUP**: 제어 터미널이 사라질 때 세션 전체에 오는 시그널. 기본 처리 = 종료
- **cron**: 매 분 깨어나 crontab 의 시간 칸과 일치하는 줄을 fork/exec 하는 데몬

**새로 나온 명령**
| 명령 | 기본형 | 용도 |
|---|---|---|
| `nohup` | `nohup ./collect.sh &` | SIGHUP 을 무시하게 한 뒤 실행, 출력은 nohup.out |
| `setsid` | `setsid ./collect.sh &` | 새 세션의 리더로 실행 — 제어 터미널 없음 |
| `crontab` | `crontab -e` | 주기 실행 표 편집 (`분 시 일 월 요일 명령`) |

**핵심 내용**

#### SSH 를 끊으면 왜 죽나
- **`./collect.sh &`** → 같은 세션 → 끊기면 SIGHUP → 종료
- **`nohup … &`** → SIGHUP 무시 → 생존
- **`setsid … &`** → 새 세션 → 옛 세션의 SIGHUP 이 안 옴

#### cron 다섯 칸
> `분 시 일 월 요일` — `* * * * *` = 매 분, `0 9 * * 1` = 월요일 09:00
- **자식이 받는 환경** → `PATH=/usr/bin:/bin`, 터미널 변수 없음
- ⚠ "터미널에선 되는데 cron 에선 안 됨" 1순위 → PATH · 상대경로 → 스크립트에 PATH 명시 + 절대경로

### 8교시 · systemd — 켜기만 하면 뜨고, 죽으면 다시 뜬다

**키워드**: `systemd`, `PID 1`, `unit 파일`, `Restart=on-failure`, `systemctl`, `journalctl`

**개요**: cron 은 주기 실행, systemd 는 상시 서비스. PID 1 이 unit 파일을 읽어 의존성 순서로 띄우고, 죽으면 wait 로 알아채 다시 fork 한다. collect.sh 를 unit 10줄로 서비스에 올리는 것이 3일의 도착점.

**핵심 용어**
- **systemd**: PID 1 로 동작하는 서비스 관리자. unit 파일대로 부팅 시 실행 · 재시작
- **unit 파일**: 서비스의 실행 명령 · 재시작 규칙 · 의존성을 적은 설정 파일

**새로 나온 명령**
| 명령 | 기본형 | 용도 |
|---|---|---|
| `systemctl` | `systemctl enable --now collect` | 서비스 등록 + 즉시 시작 (`status` · `stop` · `restart`) |
| `journalctl` | `journalctl -u collect` | 서비스 로그 보기 |

**핵심 내용**

#### Restart = wait + fork 다시
- **크래시(exit 1 / kill -9)** → 커널이 PID 1 에 알림 → failed → `Restart=on-failure`, `RestartSec=5` → 5초 뒤 새 PID 로 다시
- **세션 없음** → SIGHUP 무관

#### collect.sh 한 줄의 여행 — 여덟 세션 재등장
- systemd fork/exec → `#!` · x 비트 → 줄마다 fork · exec · wait → 변수 상자 → `|` 와 `>>` 는 dup2 → open: dentry → inode → fd 3 → write: 페이지 캐시 → writeback → 죽으면 Restart

**기본 사용형태**
```ini
# /etc/systemd/system/collect.service
[Unit]
After=network.target
[Service]
ExecStart=/home/user/collect_loop.sh
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
```

**연결**
- [EIO D5·7교시 서비스화와 장애 대응](#/eio/5/7) — 같은 unit 파일로 센서 프로그램을 서비스로
