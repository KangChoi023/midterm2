## Day 4 — 파일시스템 · sysfs

### 1교시 · 파일 디스크립터

**키워드:** 파일 디스크립터 fd, fd 테이블, 열린 파일 테이블, offset, inode, open, read, write, close, dup, dup2, 리다이렉션, /proc/self/fd, readlink, mycp, EOF, O_CREAT O_TRUNC

**개요**
- 프로그램은 파일을 **숫자 하나(fd)** 로 다루고, 커널은 그 뒤에서 열린 파일 · inode · page cache · 장치를 관리한다. 1교시는 **fd → 열린 파일 → inode** 3단 구조와, `open/read/write/close` 4개로 만든 `cp`, 셸 `>` 리다이렉션의 정체인 **`dup2`**.

**핵심 용어**
- **fd(파일 디스크립터)**: 프로세스가 연 파일을 가리키는 **번호**. 0 stdin / 1 stdout / 2 stderr. 사물함 번호표.
- **fd 테이블**: 숫자 → 열린 파일. **프로세스마다 따로**(fork 하면 복사).
- **열린 파일(open file)**: 현재 위치(**offset**)와 열기 모드. **dup·fork 한 fd 끼리 공유**.
- **inode**: 파일 그 자체 — 크기·권한·디스크 블록 위치. 같은 파일을 연 모두가 공유.
- **dup2(b, 1)**: "fd 1 이 b 와 같은 열린 파일을 가리키게". 셸의 `> out.txt` 가 이것.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `open` | `int open(const char *path, int flags, mode_t mode);` | 파일 열기. **비어 있는 가장 작은 번호** 반환, 실패 -1 |
| `read` | `ssize_t read(int fd, void *buf, size_t n);` | 읽은 바이트 수, **0 = EOF**, -1 = 오류 |
| `write` | `ssize_t write(int fd, const void *buf, size_t n);` | 쓴 바이트 수 (**요청보다 적을 수 있음**) |
| `close` | `int close(int fd);` | fd 반납 |
| `dup` / `dup2` | `int dup(int fd);` / `int dup2(int old, int new);` | fd 복제 / `new` 가 `old` 와 같은 열린 파일을 가리키게 |
| `readlink` | `ssize_t readlink(const char *path, char *buf, size_t n);` | 링크가 가리키는 경로 (널 종료 안 함) |
| `ls -l /proc/self/fd` | — | 내가 연 fd 목록 |

**예제 코드가 하는 일** ([01_fd/](Linux_day4/01_fd/))

| 예제 | 하는 일 |
|---|---|
| `01_fd.c` | 파일 두 개를 열어 번호(3, 4)를 보고 `/proc/self/fd` 를 출력 → 하나 닫고 다시 열면 3 재사용 → `dup(1)` 로 화면 보관, `dup2(b, 1)` 로 stdout 을 `out.txt` 로 돌렸다가 원래대로 |
| `mycp.c` | `open/read/write/close` 만으로 만든 `cp`. 4096 바이트씩 읽어 쓰고, `write` 반환값을 확인 |

**핵심 내용**

#### fd 번호 규칙
> `open` 은 비어 있는 가장 작은 번호를 준다
- **실행** → `a=3, b=4` → `close(a)` 후 다시 열면 **`c=3`**
- ⚠ "가장 큰 번호 다음" 이 아니다
- **`/proc/self/fd`** → `fd 3 → /etc/hostname`, `fd 4 → …/out.txt`, `fd 5 → /proc/81636/fd` (opendir 이 쓰는 fd)

#### dup2 리다이렉션
> 다른 fd 를 1번 자리에 복제하면 stdout 이 그 파일로 간다
- **`dup2(b, 1)` 후 `printf`** → `out.txt` 로
- **셸의 `./prog > out.txt`** → 자식에서 out.txt 를 열고 `dup2(fd, 1)` 후 exec
- **`dup` 으로 만든 두 fd** → 같은 열린 파일 → **offset 공유**
- ⚠ 출력이 파이프/파일이면 전체 버퍼링 → `dup2` 전에 찍은 줄까지 `fflush` 순간 **out.txt 로 같이 들어감** (터미널은 화면에)
- **안전하게** → `dup2` 전에도 `fflush(stdout)` (Day1 4교시 버퍼링)

#### read/write 루프 (mycp)
> 0 이면 EOF, -1 이면 오류, 쓴 양은 반환값으로 확인
- **`read` 반환 0** → EOF / **-1** → 오류
- **`write`** → 요청보다 적게 쓸 수 있음 → **반환값을 `n` 과 비교**
- **실행** → `./mycp /etc/os-release copy.txt && diff …` → `같음` / 없는 원본 → `/nope: No such file or directory`

**기본 사용형태**
```c
int in  = open(src, O_RDONLY);
int out = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
char buf[4096]; ssize_t n;
while ((n = read(in, buf, sizeof buf)) > 0)      // 0 = EOF
    if (write(out, buf, n) != n) { perror("write"); return 1; }
close(in); close(out);

int saved = dup(1);  dup2(fd, 1);   // stdout → 파일
printf("...\n"); fflush(stdout);
dup2(saved, 1);                     // 화면으로 복귀
```

**연관 문제**
<<QA 1>>

### 2교시 · inode 와 디렉터리

**키워드:** inode, 디렉터리 항목, 하드 링크 ln, 심볼릭 링크 ln -s, 링크 수 st_nlink, stat, lstat, readlink, opendir, readdir, closedir, struct stat, st_ino, st_mode, S_ISDIR, S_ISLNK, ls -li, myls, rm 과 데이터 삭제

**개요**
- 파일의 실체는 **inode**(번호·크기·권한·시각·데이터 위치)이고, **이름은 디렉터리 안의 "이름 → inode 번호" 한 줄**일 뿐이다. 그래서 한 파일에 이름이 여럿일 수 있다(하드 링크). 심볼릭 링크와의 차이, `lstat` + `readdir` 로 만든 `ls -li` 를 다룬다.

**핵심 용어**
- **inode**: 파일 그 자체(메타데이터 + 데이터 위치). 이름은 없다.
- **디렉터리**: **이름과 inode 번호의 목록**을 담은 특별한 파일. 전화번호부.
- **하드 링크**(`ln a b`): **같은 inode** 에 이름을 하나 더. 링크 수가 2 가 된다. 원본 이름을 지워도 남은 이름으로 그대로 읽힌다.
- **심볼릭 링크**(`ln -s a b`): **경로를 담은 작은 파일**(자기만의 inode). 원본이 지워지면 **깨진 링크**. 바로가기 아이콘.
- **`stat` vs `lstat`**: `stat` 은 링크를 **따라간 대상**의 정보, `lstat` 은 **링크 자체**의 정보.
- **rm**: 이름(디렉터리 항목)을 지우고 링크 수를 줄일 뿐. 링크 수가 0 이고 연 프로세스가 없을 때 데이터가 해제된다 → "즉시 데이터 삭제"가 아니다.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `opendir` / `readdir` / `closedir` | `DIR *opendir(const char *);` / `struct dirent *readdir(DIR *);` | 디렉터리 항목을 하나씩 (`e->d_name`) |
| `stat` / `lstat` | `int lstat(const char *path, struct stat *st);` | `st_ino`(inode) `st_mode`(종류·권한) `st_nlink`(링크 수) `st_size` `st_mtime` |
| `S_ISDIR(m)` / `S_ISLNK(m)` | — | 디렉터리인가 / 심볼릭 링크인가 |
| `strftime` | `size_t strftime(char *s, size_t n, const char *fmt, const struct tm *t);` | 시각 → 문자열 (`"%m-%d %H:%M"`) |
| `ln a.txt hard.txt` / `ln -s a.txt soft.txt` | — | 하드 / 심볼릭 링크 |
| `ls -li` | — | inode 번호와 함께 목록 |

**예제 코드가 하는 일** ([02_inode/](Linux_day4/02_inode/))

| 예제 | 하는 일 |
|---|---|
| `02_myls.c` | `opendir`/`readdir` 로 항목을 꺼내 `lstat` 으로 inode·권한(`perm_str`)·링크 수·크기·수정 시각을 출력하고, 심볼릭 링크면 `readlink` 로 `-> 대상` 표시 — 직접 만든 `ls -li` |

**핵심 내용**

#### 디렉터리 읽기 순서
> opendir → readdir → 경로 만들기 → lstat → 출력 → 반복 → closedir
- **순서** → `opendir` → `readdir` 로 하나 → `dir/이름` 경로 → `lstat` → 출력 → (링크면 `readlink`) → 반복 → `closedir`
- **링크 자체의 종류·크기** → **`lstat`** (`stat` 은 따라가 버린다)

#### 하드 링크 vs 심볼릭 링크
> 하드 링크는 같은 inode 의 다른 이름, 심볼릭 링크는 경로를 담은 별도 파일

```text
   41405 -rw-r--r--  2        3 … a.txt
   41405 -rw-r--r--  2        3 … hard.txt
   41406 lrwxrwxrwx  1        5 … soft.txt -> a.txt
   41407 drwxr-xr-x  2     4096 … sub
```

- **a.txt 와 hard.txt** → **inode 번호가 같고 링크 수 2**
- **soft.txt** → **자기 inode**, 크기 **5** = 경로 `"a.txt"` 다섯 글자
- **`rm a.txt` 후 `cat hard.txt`** → **`hi`** (그대로), 링크 수 1 로
- ⚠ **`rm a.txt` 후 `cat soft.txt`** → `No such file or directory` (깨짐)

**기본 사용형태**
```c
DIR *d = opendir(dir); struct dirent *e;
while ((e = readdir(d)) != NULL) {
    char path[1024]; snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
    struct stat st;
    if (lstat(path, &st) == -1) { perror(path); continue; }
    printf("%lu %lu %lld %s\n", (unsigned long)st.st_ino, (unsigned long)st.st_nlink, (long long)st.st_size, e->d_name);
}
closedir(d);
```

**연관 문제**
<<QA 2>>

### 3교시 · 버퍼링과 동기화 비용

**키워드:** 버퍼 크기, 시스템 콜 횟수, copybench, fsync, O_SYNC, sync, page cache, dirty, strace -c, stdio 버퍼, MB/s, dd if=/dev/urandom

**개요**
- 같은 8MB 를 복사해도 **한 번에 몇 바이트씩** 읽고 쓰느냐에 따라 시스템 콜 횟수가 수천 배 달라진다(Day 1 의 사용자↔커널 전환 비용). 또 **`write` 는 디스크에 쓰지 않는다** — page cache 에 넣고 바로 돌아오며, 디스크까지 기다리게 하는 `fsync` 는 매우 비싸다.

**핵심 용어**
- **page cache**: 커널이 파일 데이터를 담아 두는 RAM. `write` 는 여기에 복사하고 끝.
- **dirty 페이지**: page cache 에서 고쳤지만 **아직 디스크에 안 쓴** 페이지. 커널이 나중에(수 초 안에) 기록.
- **fsync(fd)**: 그 파일의 내용이 **디스크에 기록될 때까지 기다림**. 정전 대비.
- **O_SYNC**: 매 `write` 마다 디스크까지. **sync**(명령): 시스템 전체 dirty 페이지 기록.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `fsync` | `int fsync(int fd);` | 그 파일을 디스크까지 기록하고 돌아옴 |
| `open(…, O_SYNC)` | — | 매 write 를 디스크까지 |
| `dd if=/dev/urandom of=src.bin bs=1M count=8` | — | 8MB 무작위 파일 만들기 |
| `strace -c ./prog` | — | 시스템 콜 **횟수** 요약 |

**예제 코드가 하는 일** ([03_copybench/](Linux_day4/03_copybench/))

| 예제 | 하는 일 |
|---|---|
| `03_copybench.c` | 원본을 지정한 버퍼 크기로 `read`/`write` 복사하며 시간·시스템 콜 수·MB/s 출력. 세 번째 인자 `sync` 면 write 마다 `fsync` |

**핵심 내용**

#### 버퍼 크기와 시스템 콜 횟수
> 파일 I/O 비용은 시스템 콜 횟수가 정한다 — 단, 끝없이 빨라지지는 않는다 (8MB 복사)

| 버퍼 | 시간 | 시스템 콜 | 속도 |
|---|---|---|---|
| 1 B | **4941.7 ms** | **16,777,216** | 1.6 MB/s |
| 4096 B | 4.2 ms | 4096 | 1915 MB/s |
| 65536 B | 3.6 ms | 256 | 2225 MB/s |
| 4096 B + fsync | **1833.6 ms** | 6144 | 4.4 MB/s |

- **1바이트 버퍼** → read·write 각 838만 번 = **약 1677만 번**
- **4KB 로** → **1000배 이상** 빨라짐
- **64KB 이상** → 거의 그대로
- **stdio (`fwrite`)** → 사용자 공간 버퍼에 모았다가 write → 버퍼를 키운 것과 같은 효과

#### write 와 fsync
> write 성공 = page cache 에 들어감 — 디스크까지는 fsync
- **`write` 성공 직후** → 데이터는 **page cache** 에
- ⚠ fsync 를 켜면 → write 마다 디스크 기록을 기다려 **수백 배 느려짐** (강의 VM 약 50배)
- **규칙** → 로그·설정은 **꼭 보존해야 하는 시점에만** fsync

| 방법 | 돌아오는 시점 | 용도 |
|---|---|---|
| `write` | page cache 에 복사 직후 | 대부분 |
| `write` + `fsync` | 디스크 기록이 끝난 뒤 | 로그·설정 저장의 마지막 |
| `open(…, O_SYNC)` | 매 write 마다 디스크까지 | DB 저널 등 |
| `sync`(명령) | 시스템 전체 dirty 기록 | 전원 끄기 전 |

**기본 사용형태**
```c
while ((n = read(in, buf, bs)) > 0) {
    write(out, buf, n);
    if (sync) fsync(out);          // 디스크까지 기다림 — 매우 느림
}
```
```bash
dd if=/dev/urandom of=src.bin bs=1M count=8
./03_copybench src.bin 4096 ; ./03_copybench src.bin 4096 sync
strace -c ./03_copybench src.bin 4096
```

**연관 문제**
<<QA 3>>

### 4교시 · Page cache

**키워드:** page cache, buff/cache, free -h, /proc/meminfo, Cached, Dirty, clean, posix_fadvise, POSIX_FADV_DONTNEED, drop_caches, sync, 두 번째 읽기가 빠른 이유

**개요**
- 커널은 읽은 파일 페이지를 RAM 에 남겨 둔다(**page cache**). 같은 파일을 다시 읽으면 디스크 대신 RAM 에서 가져와 훨씬 빠르다. 첫 읽기/두 번째 읽기를 재고, 내 파일의 cache 만 `posix_fadvise` 로 버리는 법, dirty 페이지는 왜 안 버려지는지를 다룬다.

**핵심 용어**
- **buff/cache**(`free -h`): page cache 크기. 메모리가 필요하면 커널이 **언제든 버린다** → "free 가 적다"고 걱정할 필요 없음.
- **clean 페이지**: 디스크와 내용이 같음 → 버릴 수 있음. **dirty 페이지**: 수정됐지만 아직 기록 전 → **sync 전에는 못 버림**.
- **`Dirty`**(`/proc/meminfo`): sync 전의 수정된 page cache 크기.
- **posix_fadvise(DONTNEED)**: **이 파일의** cache 만 버려 달라는 힌트. root 불필요.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `posix_fadvise` | `int posix_fadvise(int fd, off_t off, off_t len, int advice);` | `POSIX_FADV_DONTNEED` = 이 파일 cache 버리기 (len 0 = 끝까지) |
| `free -h` | — | `buff/cache` 확인 |
| `grep -E "^(Cached\|Dirty)" /proc/meminfo` | — | cache · dirty 크기 |
| `echo 3 \| sudo tee /proc/sys/vm/drop_caches` | — | 시스템 **전체** cache 비우기 (강사 시연만) |

**예제 코드가 하는 일** ([04_pagecache/](Linux_day4/04_pagecache/))

| 예제 | 하는 일 |
|---|---|
| `04_pagecache.c` | 파일을 4MB 버퍼로 끝까지 읽어 시간·MB/s 와 `/proc/meminfo` 의 `Cached` 변화를 출력. 인자 `drop` 이면 읽은 뒤 `posix_fadvise(DONTNEED)` 로 그 파일 cache 를 버림 |

**핵심 내용**

#### page cache 측정 순서
> sync 로 dirty 를 clean 으로 만든 뒤 cache 를 비워야 정확히 잰다
- **순서** → `dd` 로 big.bin → **`sync`** (dirty → clean) → `./04_pagecache big.bin drop` (cache 비움) → 첫 읽기(디스크) → 두 번째 읽기(RAM)
- ⚠ sync 를 빼먹으면 → 막 쓴 페이지는 **dirty** 라 DONTNEED 로도 안 버려짐 (버릴 수 있는 건 clean 뿐)
- ⚠ `drop_caches` → **다른 사람 프로그램까지** 디스크에서 다시 읽게 해 느려짐 → 강사 시연만

#### 첫 읽기 vs 두 번째 읽기
> 한 번 읽은 파일은 page cache 에 남아 두 번째부터 RAM 에서 읽는다
- **`drop`** → `Cached 831512 → 626712 kB` (약 200MB ↓)
- **첫 읽기** → **48.9 ms (4091 MB/s)**, Cached **+204800 kB**
- **두 번째** → **15.1 ms (13234 MB/s)**, Cached +0 → RAM 에서

**기본 사용형태**
```c
while ((n = read(fd, buf, bs)) > 0) total += n;
posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // 이 파일 cache 만 버림
```
```bash
dd if=/dev/urandom of=big.bin bs=1M count=500 && sync
./04_pagecache big.bin drop ; ./04_pagecache big.bin ; ./04_pagecache big.bin
free -h
```

**연관 문제**
<<QA 4>>

### 5교시 · 파일 변화 감시

**키워드:** inotify, inotify_init1, inotify_add_watch, struct inotify_event, IN_CREATE, IN_MODIFY, IN_CLOSE_WRITE, IN_DELETE, IN_MOVED_FROM, IN_MOVED_TO, 폴링 vs 이벤트, 업로드 완료 판단

**개요**
- 설정 파일이 바뀌었는지 1초마다 `stat` 으로 확인(**폴링**)하면 CPU 를 쓰고 최대 1초 늦는다. **inotify** 는 변화가 생긴 순간 커널이 이벤트를 넣어 주고, 프로그램은 `read` 에서 잠들어 있다가 깨어난다.

**핵심 용어**
- **폴링(polling)**: 주기적으로 직접 확인. 1분마다 우편함 열어 보기.
- **이벤트(event) 방식**: 변화가 생기면 알림을 받음. 초인종.
- **inotify**: 파일·디렉터리 변화를 이벤트로 알려 주는 커널 기능. 이벤트가 없으면 `read` 에서 **잠들어 CPU 를 안 쓴다**.
- **IN_CLOSE_WRITE**: 쓰기로 연 파일을 닫음 → **"저장 끝" 신호로 가장 알맞다**. (MODIFY 는 write 마다 여러 번)

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `inotify_init1` | `int inotify_init1(int flags);` | inotify fd 만들기 |
| `inotify_add_watch` | `int inotify_add_watch(int fd, const char *path, uint32_t mask);` | 감시 대상·이벤트 등록 |
| `read(fd, buf, …)` | — | `struct inotify_event`(`mask`, `len`, `name`)를 꺼냄. 한 번에 여러 개 |

| 이벤트 | 언제 |
|---|---|
| `IN_CREATE` | 파일·디렉터리가 생김 |
| `IN_MODIFY` | 내용이 바뀜 (write 마다) |
| `IN_CLOSE_WRITE` | 쓰기로 연 파일을 닫음 |
| `IN_DELETE` | 지워짐 |
| `IN_MOVED_FROM` / `IN_MOVED_TO` | 이름 바꾸기·이동 |

**예제 코드가 하는 일** ([05_inotify/](Linux_day4/05_inotify/))

| 예제 | 하는 일 |
|---|---|
| `05_inotify.c` | 디렉터리에 6가지 이벤트 감시를 걸고 `read` 에서 기다리다가, 버퍼 안의 이벤트를 `p += sizeof(struct inotify_event) + ev->len` 으로 하나씩 넘기며 종류와 파일 이름을 출력 |

**핵심 내용**

#### inotify 흐름
> 파일 시스템 변화를 이벤트로 받는다 — 읽기만 해서는 이벤트가 없다
- **흐름** → `inotify_init1` → `inotify_add_watch` → `read` (잠듦) → 이벤트 해석·출력 → 반복
- **`cat` 으로 읽기만** → 이 6가지 이벤트는 오지 않음 (감시 목록에 없음)

#### 이벤트 순서
> 명령 하나가 여러 이벤트를 만든다

```text
CREATE      a.txt
MODIFY      a.txt
CLOSE_WRITE a.txt
MOVED_FROM  a.txt
MOVED_TO    b.txt
DELETE      b.txt
```

- **`echo hi > a.txt`** → CREATE → MODIFY → CLOSE_WRITE
- **`mv`** → MOVED_FROM + MOVED_TO
- **큰 파일 (FileZilla 업로드)** → MODIFY 가 여러 번 → **"다 올라왔다" 는 CLOSE_WRITE** 로 판단
- **쓰임** → 업로드 폴더 처리, 로그 줄 추가 알림, 설정 파일 변경 시 다시 읽기

**기본 사용형태**
```c
int fd = inotify_init1(0);
inotify_add_watch(fd, dir, IN_CREATE | IN_MODIFY | IN_CLOSE_WRITE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO);
char buf[4096] __attribute__((aligned(__alignof__(struct inotify_event))));
ssize_t len = read(fd, buf, sizeof buf);                 // 이벤트까지 잠듦
for (char *p = buf; p < buf + len; ) {
    struct inotify_event *ev = (struct inotify_event *)p;
    if (ev->mask & IN_CLOSE_WRITE) printf("저장 끝 %s\n", ev->name);
    p += sizeof(struct inotify_event) + ev->len;
}
```

**연관 문제**
<<QA 5>>

### 6교시 · procfs 와 sysfs

**키워드:** procfs, sysfs, /proc, /sys, kobject, attribute, 파일 하나 = 값 하나, /sys/devices, /sys/class, /sys/bus, /sys/kernel, /sys/class/net/…/statistics/rx_bytes, /sys/block/sda/stat, thermal_zone0/temp

**개요**
- "리눅스에서는 장치와 커널 정보도 파일이다." 두 가상 파일시스템 **procfs**(프로세스 중심)와 **sysfs**(장치·드라이버 속성)의 차이, sysfs 의 디렉터리 구조, C 로 값을 읽는 법. 같은 코드가 실제 보드에서 온도도 읽는다.

**핵심 용어**
- **procfs(`/proc`)**: 프로세스 정보 + (역사적으로) 이것저것. 파일마다 형식이 제각각(`/proc/self/status`, `/proc/meminfo`).
- **sysfs(`/sys`)**: 커널 객체(장치·드라이버·버스)의 속성. **파일 하나 = 값 하나(텍스트 한 줄)**. 디스크에 저장된 게 아니라 읽는 순간 커널이 만든다.
- **kobject**: 커널 안 객체. sysfs 의 **디렉터리 하나 = kobject 하나**, **파일 하나 = 속성(attribute) 하나**.
- 같은 장치를 여러 관점으로: **`/sys/devices`**(실제 연결 트리) / **`/sys/class`**(종류별, devices 로 가는 **링크**) / **`/sys/bus`**(pci·usb·i2c) / **`/sys/kernel`**(커널 설정, 7교시에 직접 만듦).

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `cat /sys/class/net/enp0s3/address` / `operstate` | MAC 주소 / 연결 상태 |
| `cat /sys/class/net/enp0s3/statistics/rx_bytes` | 받은 바이트 수 |
| `cat /sys/block/sda/stat` | 디스크 읽기·쓰기 횟수 등 |
| `cat /sys/devices/system/cpu/online` | 온라인 CPU (`0-1`) |
| `cat /sys/class/thermal/thermal_zone0/temp` | 온도 (1/1000 ℃) — 실제 보드 |

**예제 코드가 하는 일** ([06_sysfs/](Linux_day4/06_sysfs/))

| 예제 | 하는 일 |
|---|---|
| `06_sysread.c` | `read_line()`(첫 줄만 읽고 개행 제거)으로 온라인 CPU·loadavg 를 읽고, `/sys/class/net` 을 `readdir` 해 장치마다 rx/tx/상태, `/sys/block/{sda,vda,nvme0n1}/stat` 의 읽기·쓰기 횟수, 온도(있으면)를 출력 |

**핵심 내용**

#### procfs · sysfs 로 시스템 정보 읽기
> 시스템 상태를 파일 읽기로 얻는다

```text
온라인 CPU            : 0-7
loadavg (procfs)      : 0.31 0.22 0.13 1/409 81725
net lo       unknown rx 95032575 B …
net eth0     up      rx 34052373 B  tx 46196350 B
disk sda      읽기 1105 회, 쓰기 0 회
온도                  : thermal_zone 없음
```

- **강의 VM** → `0-1`, `enp0s3`
- **VM 에는 온도 센서가 없어** → `/sys/class/thermal` 이 비어 있음
- **`ls -l /sys/class/net/` 의 `->`** → `/sys/devices` 의 실제 위치를 가리키는 링크
- **rx_bytes 읽기** → `opendir("/sys/class/net")` → 장치 이름 → `…/이름/statistics/rx_bytes` → 첫 줄 읽기

#### sysfs 의 힘
> 장치가 달라도 읽는 방법은 같다
- **라즈베리파이 · Jetson** → 같은 코드로 `thermal_zone0/temp` 온도
- **Embedded I/O 의 GPIO · I2C 도** 같은 방식

**기본 사용형태**
```c
static int read_line(const char *path, char *buf, size_t size) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    if (!fgets(buf, size, fp)) { fclose(fp); return -1; }
    buf[strcspn(buf, "\n")] = '\0';
    fclose(fp); return 0;
}
read_line("/sys/class/net/eth0/statistics/rx_bytes", v, sizeof v);
```

**연관 문제**
<<QA 6>>

### 7교시 · 커널 모듈로 sysfs 속성 만들기

**키워드:** 커널 모듈 .ko, insmod, rmmod, lsmod, modinfo, dmesg, pr_info, module_init, module_exit, MODULE_LICENSE, Kbuild obj-m, linux-headers, kobject_create_and_add, kernel_kobj, __ATTR, show, store, sysfs_emit, kstrtoint, -EINVAL, sysfs_create_group, kobject_put

**개요**
- **커널 모듈**로 `/sys/kernel/mysysfs/` 아래에 속성 파일(`value`, `writes`)을 직접 만든다. `cat` 하면 커널의 **show 콜백**이, `echo … >` 하면 **store 콜백**이 불린다. 사용자 프로그램은 그냥 **파일 읽기·쓰기**만 한다 — 이것이 드라이버가 sysfs 로 값을 보여 주는 원리.

**핵심 용어**
- **커널 모듈(.ko)**: 실행 중인 커널에 넣었다 뺐다 하는 코드. 드라이버 대부분이 모듈. **커널 모드**에서 돌아 C 라이브러리를 못 쓴다 → `printf` 대신 **`pr_info`**(→ dmesg), `malloc` 대신 `kmalloc`. **버그는 VM 전체를 멈출 수 있다**(실습 전 스냅샷).
- **show 콜백**: `cat` 할 때 — `buf` 에 텍스트를 쓰고 **길이 반환**(`sysfs_emit`).
- **store 콜백**: `echo … >` 할 때 — `buf` 를 해석(`kstrtoint`), 잘못된 값이면 **음수 오류(-EINVAL)** 반환 → 사용자는 `Invalid argument`. 성공하면 `count` 반환.
- **`__ATTR(이름, 권한, show, store)`**: 속성 파일 정의. `0664` = 파일 권한, `0444` = 읽기 전용(store 는 NULL).

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `make` (Kbuild: `obj-m := mysysfs.o`, `make -C /lib/modules/$(uname -r)/build M=$(PWD) modules`) | **실행 중인 커널 버전의 linux-headers** 로 .ko 빌드 |
| `sudo insmod mysysfs.ko` | 적재 → `module_init` 실행 |
| `sudo rmmod mysysfs` | 제거 → `module_exit` 실행 |
| `lsmod \| grep mysysfs` / `modinfo mysysfs.ko` | 적재 확인 / 라이선스·설명·버전 |
| `sudo dmesg \| tail` | `pr_info` 출력 확인 |
| `kobject_create_and_add("mysysfs", kernel_kobj)` | `/sys/kernel/mysysfs` 디렉터리 생성 |
| `sysfs_create_group(kobj, &group)` | 속성 목록을 한 번에 생성 |
| `kobject_put(kobj)` | 제거 시 디렉터리와 속성을 함께 정리 |

**예제 코드가 하는 일** ([07_module/](Linux_day4/07_module/))

| 예제 | 하는 일 |
|---|---|
| `module/mysysfs.c` | 적재 시 `/sys/kernel/mysysfs/` 에 `value`(0664, 정수, 쓰면 `writes++` 하고 dmesg 출력)와 `writes`(0444, 쓴 횟수)를 만들고, 제거 시 `kobject_put` 으로 정리 |
| `07_sysfs_user.c` | `fopen/fscanf` 로 `value` 를 읽고, 인자가 있으면 `fprintf` 로 쓰기(store 호출) → 다시 읽기 → `writes` 출력. `fclose` 실패면 store 거부 |

**핵심 내용**

#### 커널 모듈로 sysfs 파일 만들기
> 커널 쪽이 show/store 를 구현하면 사용자 쪽은 파일 읽기·쓰기만 한다
- **순서** → `cd module && make` → `sudo insmod module/mysysfs.ko` → `cat /sys/kernel/mysysfs/value` (0)
- **쓰기** → `echo 42 | sudo tee …/value` → `cat value writes` (**42, 1**)
- ⚠ **`echo abc | sudo tee …/value`** → `Invalid argument` (store 가 -EINVAL)
- **제거** → `sudo rmmod mysysfs` → 디렉터리 사라짐 (`kobject_put`)
- **`sudo ./07_sysfs_user 7`** → `value = 42`, `value ← 7, 다시 읽기: 7`, `writes = 2`
- **사용자 프로그램** → `fopen · fprintf` 만 사용
- (WSL 은 커널 헤더가 불완전해 `make` 가 `No rule to make target 'modules'` 로 실패, sudo 도 없어 강의 결과로 정리)

**기본 사용형태**
```c
static ssize_t value_show(struct kobject *k, struct kobj_attribute *a, char *buf)
{ return sysfs_emit(buf, "%d\n", value); }
static ssize_t value_store(struct kobject *k, struct kobj_attribute *a, const char *buf, size_t count)
{ int ret = kstrtoint(buf, 10, &value); if (ret < 0) return ret; writes++; return count; }
static struct kobj_attribute value_attr = __ATTR(value, 0664, value_show, value_store);
// init: my_kobj = kobject_create_and_add("mysysfs", kernel_kobj); sysfs_create_group(my_kobj, &my_group);
// exit: kobject_put(my_kobj);
```
```bash
cd module && make && cd ..
sudo insmod module/mysysfs.ko && sudo dmesg | tail -1
echo 42 | sudo tee /sys/kernel/mysysfs/value
cat /sys/kernel/mysysfs/value /sys/kernel/mysysfs/writes
sudo rmmod mysysfs
```

**연관 문제**
<<QA 7>>

### 8교시 · 종합 실습 sysmon

**키워드:** sysmon, /proc/stat, CPU 사용률, 누적값 차이, /proc/meminfo, MemTotal, MemAvailable, /sys/block/sda/stat 섹터 512 바이트, rx_bytes, CSV, getopt, SIGINT, clock_nanosleep TIMER_ABSTIME, oflag=direct

**개요**
- Day 4 에서 배운 procfs·sysfs 읽기에 Day 1(getopt·시그널)과 Day 2(절대 시각 주기 실행)를 합쳐, 1초마다 **CPU · 메모리 · 디스크 · 네트워크**를 CSV 로 기록하는 `sysmon` 을 완성한다.

**핵심 용어**
- **누적값**: `/proc/stat` CPU 시간, 디스크 섹터, rx_bytes 는 모두 **부팅 후 누적** → **두 번 읽은 차이**로 속도·사용률을 구한다. 자동차 주행거리계로 속력 구하기.
- **MemAvailable**: page cache 처럼 **바로 회수할 수 있는 메모리**까지 포함한 "쓸 수 있는 양" → 사용량 = MemTotal − **MemAvailable** (MemFree 가 아님).
- **섹터**: `/sys/block/sda/stat` 의 단위, **512 바이트**(4096 아님).

**새로 나온 함수/명령**

| 값 | 읽는 곳 | 계산 |
|---|---|---|
| CPU 사용률 | `/proc/stat` 첫 줄 | 직전과의 차이: (전체 − idle) / 전체 |
| 메모리 | `/proc/meminfo` | MemTotal − MemAvailable |
| 디스크 쓰기 | `/sys/block/sda/stat` 7번째 값(쓴 섹터) | 차이 × 512 바이트 |
| 네트워크 | `/sys/class/net/<장치>/statistics/rx_bytes` | 직전과의 차이 |

| 명령 | 용도 |
|---|---|
| `./sysmon -i 1 -n 10 -o sys.csv [-d sda] [-N enp0s3]` | 주기·횟수·출력·디스크·네트워크 지정 |
| `dd if=/dev/zero of=z.bin bs=1M count=100 oflag=direct` | **page cache 를 건너뛰어** 그 순간 디스크에 씀(그래프에 보이게) |

**예제 코드가 하는 일** ([08_sysmon/](Linux_day4/08_sysmon/))

| 예제 | 하는 일 |
|---|---|
| `sysmon.c` | `getopt` 로 옵션, 디스크·네트워크 장치 자동 탐색 → 시작값 저장 → 루프: `next += interval` → `clock_nanosleep(TIMER_ABSTIME)` → 새 값 읽기 → 차이로 CPU%·디스크 KB/s·수신 KB/s 계산 + 메모리 → 화면·CSV 출력 → 이전 값 갱신. Ctrl+C(SIGINT 핸들러)로 끝내도 CSV 저장 |

**핵심 내용**

#### sysmon 결과 읽기
> 카운터의 1초 차이로 CPU · 메모리 · 디스크 · 네트워크 사용량을 낸다

```text
디스크=sda 네트워크=eth0 주기=1초 → sys.csv
 초    CPU% 메모리MB 디스크쓰기KB/s 수신KB/s
   1     0.5     2247          0.0        0.2
   2    10.2     2247          0.0        0.2     ← 1코어 바쁨 ≈ 1/8 = 12.5 %
   3     3.1     2246          0.0        0.2
```

- **CSV** → `time,cpu_pct,mem_used_mb,disk_write_kbps,net_rx_kbps`
- **강의 VM** → stress-ng → 80%, dd 100MB direct → 102400 KB/s
- **디스크 계산** → 1초에 섹터 **204800** ↑ → 204800 × 512 / 1024 = **102,400 KB/s**

#### 앞 날짜 기법 조립
> 절대 시각 주기 · 시그널 정상 종료 · getopt
- **`clock_nanosleep(TIMER_ABSTIME)`** → 측정 시간만큼 주기가 밀리지 않게 (D2 5교시)
- **SIGINT 핸들러로 정상 종료** → `volatile sig_atomic_t g_stop` (D1)
- **`getopt`** → 옵션 처리 (D2 jitter)

#### Day 4 정리
> 파일 I/O 비용 = 시스템 콜 횟수 + 디스크까지 가느냐(fsync)
- **읽기** → page cache 덕분에 두 번째부터 빠름
- **쓰기** → page cache 에 넣고 바로 돌아옴

**기본 사용형태**
```c
cpu_times(&pt, &pi);                                   // 시작값
for (int k = 1; k <= count && !g_stop; k++) {
    next.tv_sec += interval;
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
    cpu_times(&t, &i);
    double cpu = 100.0 * ((t - pt) - (i - pi)) / (t - pt);   // 차이로 사용률
    double dk  = (s - ps) * 512.0 / 1024 / interval;         // 섹터 → KB/s
    pt = t; pi = i; ps = s;
}
```

**연관 문제**
<<QA 8>>
