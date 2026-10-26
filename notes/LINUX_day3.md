## Day 3 — 메모리

### 1교시 · 가상 주소 공간

**키워드:** 가상 주소 공간, text, rodata, data, bss, heap, mmap 영역, stack, /proc/self/maps, 권한 r-xp rw-p r--p, ASLR, setarch -R, 페이지 테이블

**개요**
- 프로세스마다 0 부터 수십 TB 까지의 **가상 주소 공간**을 혼자 쓰는 것처럼 보이지만, 실제 RAM 은 VM 전체에 4GB 뿐이다. Day 3 은 그 사이를 잇는 **페이지 테이블 · page fault** 가 언제 일어나는지를 fault 수로 증명한다.
- 1교시는 주소 공간의 **지도**: 변수 종류마다 어느 영역에 놓이는지, `/proc/self/maps` 로 확인하고 ASLR 을 켜고 끈다.

**핵심 용어**
- **가상 주소(virtual address)**: 프로그램이 보는 주소. **페이지 테이블**이 실제 RAM(물리 주소)으로 바꿔 준다. 아파트 호수(가상)와 실제 지번(물리).
- **영역 7가지**: **text**(기계어, `r-xp`) / **rodata**(문자열 상수, `r--p`) / **data**(초기값 있는 전역·static, `rw-p`) / **bss**(초기값 없는 전역·static, 0 으로 시작) / **heap**(작은 malloc, `brk` 로 위로) / **mmap 영역**(큰 malloc·공유 라이브러리·파일 매핑) / **stack**(지역 변수, 아래로 자람).
- **`/proc/self/maps`**: 내 주소 공간의 영역 목록 (`시작-끝 권한 오프셋 장치 inode 경로`).
- **ASLR(Address Space Layout Randomization)**: 실행할 때마다 영역 시작 주소를 **무작위로** 옮겨 공격자가 주소를 예측 못 하게 함. `setarch -R` 로 끄면 매번 `0x5555555…`.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `cat /proc/self/maps` | — | 주소 공간 영역 + 권한 |
| `setarch -R ./prog` | — | ASLR 끄고 실행 |
| `%p` | `printf("%p", (void *)&x)` | 주소 출력 |

**예제 코드가 하는 일** ([01_layout/](Linux_day3/01_layout/))

| 예제 | 하는 일 |
|---|---|
| `01_layout.c` | `main`·문자열 상수·전역(초기값 有/無)·static·`malloc(64)`·`malloc(1MB)`·지역 변수의 주소를 출력하고 `/proc/self/maps` 의 해당 줄(실행 파일, `[heap]`, `[stack]`)과 대조 |

**핵심 내용**

#### 메모리 배치 순서
> 낮은 주소 → 높은 주소: text → data·bss → heap → mmap 영역 → stack
- **`text main()`** `0x574da092e36a` → maps 의 `r-xp …/01_layout` 줄 안
- **`data g_init` / `static s_local` / `bss g_zero`** → `rw-p …/01_layout` 줄 (`0x…931010`, `…931014`, `…931024`)
- **`heap malloc(64)`** `0x574db9e7b010` → `[heap] rw-p`
- **`mmap malloc(1MB)`** `0x746fdfc7f010` → **`0x7…` 쪽, heap 과 전혀 다른 곳** (큰 요청은 mmap 영역 — 2교시)
- **`stack local`** `0x7ffd978918f4` → `[stack] rw-p`

#### 영역별 권한
> `/proc/self/maps` 의 권한 칸
- **text** → `r-xp` (읽기+실행)
- **rodata** → `r--p` (읽기만) → 리터럴 수정 시 segfault (C Day2)
- **data · bss · heap · stack** → `rw-p`

#### ASLR
> 실행할 때마다 배치 주소를 무작위로 바꾼다
- **두 번 실행** → `main` 이 `0x574d…` → `0x5c35…` 로 바뀜
- **`setarch -R` 두 번** → 둘 다 **`0x55555555536a`** (같은 주소)

**기본 사용형태**
```c
int g_init = 42;            // data
int g_zero;                 // bss
const char *g_msg = "hello";   // "hello" 는 rodata
int main(void) {
    static int s_local = 7;    // data
    int local = 1;             // stack
    int *small = malloc(64);   // heap
    char *big = malloc(1 << 20);  // mmap 영역
    printf("%p %p\n", (void *)&local, (void *)small);
}
```
```bash
./01_layout ; setarch -R ./01_layout | head -2
grep -E "heap|stack" /proc/self/maps
```

**연관 문제**
<<QA 1>>

### 2교시 · malloc 내부

**키워드:** malloc, glibc, brk, sbrk, mmap, munmap, M_MMAP_THRESHOLD, 128KB, mallopt, 132KB 초기 heap, strace -e trace=brk,mmap,munmap, 4KB 올림

**개요**
- `malloc` 은 **시스템 콜이 아니라 glibc 함수**다. 커널에서 메모리를 받아 오는 길은 두 가지 — 작은 요청은 **heap 늘리기(`brk`)**, 큰 요청은 **독립 매핑(`mmap`)**. 그 기준(128KB)과 `free` 동작의 차이를 확인한다.

**핵심 용어**
- **brk**: heap 의 끝을 옮겨 heap 을 늘리는 시스템 콜. `sbrk(0)` 은 현재 heap 끝 주소.
- **mmap(익명)**: 주소 공간에 독립된 영역을 새로 붙이는 시스템 콜. 큰 malloc 이 쓴다.
- **M_MMAP_THRESHOLD**: mmap 을 쓰기 시작하는 기준 크기, **기본 128KB**. `mallopt` 로 변경.
- **초기 heap**: glibc 는 첫 malloc 때 heap 을 **약 132KB** 한 번에 늘려 두고 작은 요청은 그 안에서 잘라 준다(시스템 콜 없음).

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `sbrk` | `void *sbrk(intptr_t inc);` | heap 끝 이동, `sbrk(0)` = 현재 끝 |
| `mallopt` | `int mallopt(int param, int value);` | malloc 설정 (`M_MMAP_THRESHOLD, 512*1024`) |
| `munmap` | `int munmap(void *addr, size_t len);` | 매핑 해제 (큰 블록 free 시 호출됨) |
| `strace -e trace=brk,mmap,munmap ./prog` | — | malloc 이 부르는 시스템 콜만 보기 |

**예제 코드가 하는 일** ([02_malloc/](Linux_day3/02_malloc/))

| 예제 | 하는 일 |
|---|---|
| `02_malloc_path.c` | `free(malloc(1))` 로 초기화를 끝낸 뒤 `kb` KB 를 malloc 해 **heap 끝(`sbrk(0)`)이 움직였는지**와 주소가 heap 안인지로 brk/mmap 판정. 두 번째 인자로 `mallopt` 기준 변경 |

**핵심 내용**

#### malloc 의 두 경로
> 작은 요청은 brk 로 heap 을 늘리고, 큰 요청은 mmap 으로 따로 받는다

| 방법 | 시스템 콜 | 언제 | free 하면 |
|---|---|---|---|
| heap 늘리기 | `brk` | 작은 요청 (128KB 미만) | glibc 가 보관해 다시 씀 — 커널에 바로 안 돌려줌 |
| 독립 매핑 | `mmap` | 큰 요청 (128KB 이상) | **`munmap` 으로 즉시 커널에 반환** |

- **`./02_malloc_path 64`** → `heap 끝 … (+0 KB)`, **heap** — 미리 늘려 둔 132KB 안에서 잘라 줌
- **`./02_malloc_path 256`** → 주소 `0x76b7…` (heap `0x55…` 과 딴 곳), **mmap**
- **`./02_malloc_path 256 512`** (기준 512KB) → `+256 KB`, **heap** — 기준을 올리니 brk 로

#### strace 로 본 mmap
> 요청보다 조금 크게, 4KB 단위로 매핑한다
- **강의 strace** → `mmap(NULL, 266240, …MAP_PRIVATE|MAP_ANONYMOUS…)` → `munmap(…, 266240)`
- **256KB 요청 → 260KB 매핑** → 관리용 머리 + **4KB 단위 올림**

**기본 사용형태**
```c
void *before = sbrk(0);
char *p = malloc(256 * 1024);          // 128KB 이상 → mmap
void *after  = sbrk(0);                // heap 끝 그대로
mallopt(M_MMAP_THRESHOLD, 512 * 1024); // 기준을 512KB 로
```
```bash
strace -e trace=brk,mmap,munmap ./02_malloc_path 256
```

**연관 문제**
<<QA 2>>

### 3교시 · Demand paging

**키워드:** demand paging, page fault, MMU, 페이지 테이블, 4KB 페이지, VmSize, VmRSS, minor fault, major fault, minflt, getrusage, _SC_PAGESIZE, memutil.h

**개요**
- `malloc(100MB)` 는 **주소만 예약**한다. 물리 페이지는 그 주소를 **처음 읽거나 쓸 때** 받는다(demand paging). VmSize · VmRSS · minor fault 수로 이를 숫자로 증명한다. (Linux Day1 5교시 "malloc 만 하면 RSS 가 안 는다"의 원리)

**핵심 용어**
- **demand paging**: 필요할 때(처음 만질 때) 페이지를 할당. 식당 예약석은 잡아 두고 손님이 오면 그때 상을 차림.
- **page fault**: CPU 가 접근한 주소의 매핑이 없어 **커널로 진입**하는 사건. 오류가 아니라 정상 동작.
- **MMU**: 가상 → 물리 주소 변환 하드웨어. 페이지 테이블을 본다.
- **페이지**: 메모리 관리 단위, **4KB**(4096 바이트).
- **VmSize**: 예약한 가상 주소 **전체 크기**. **VmRSS**: 지금 **실제 RAM 에 올라온** 크기.
- **minor fault**: 디스크 없이 처리(새 0 페이지, page cache 에 이미 있는 페이지). **major fault**: 디스크에서 읽어야 함 — 훨씬 느림.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `getrusage` | `int getrusage(int who, struct rusage *ru);` | `ru.ru_minflt` / `ru.ru_majflt` = minor / major fault 수 |
| `sysconf(_SC_PAGESIZE)` | — | 페이지 크기 (4096) |
| `memutil.h` | `status_kb("VmRSS")`, `minflt()`, `now_ms()` | Day 3 공용 측정 함수 |

**예제 코드가 하는 일** ([03_demand/](Linux_day3/03_demand/))

| 예제 | 하는 일 |
|---|---|
| `03_demand.c` | 100MB malloc → 앞 50MB 를 **4KB 마다 1바이트씩** 쓰기 → 전체 `memset` 순서로, 단계마다 VmSize·VmRSS·minflt 를 출력 |

**핵심 내용**

#### page fault 처리 순서
> 매핑이 없는 페이지에 처음 닿는 순간 커널이 물리 페이지를 연결한다
- **①** CPU 가 `p[i] = 1` 실행
- **②** MMU 가 페이지 테이블에 매핑 없음을 앎 → **page fault** (커널 진입)
- **③** 커널이 빈 물리 페이지(4KB)를 **0 으로 채워 연결**
- **④** 그 명령을 **다시 실행**

#### demand paging 실측
> malloc 은 주소만 예약 — 실제로 쓴 페이지만 RSS 가 된다

| 단계 | VmSize | VmRSS | minflt |
|---|---|---|---|
| 시작 | 2772 kB | 1628 kB | +0 |
| `malloc(100MB)` 직후 | **105176 kB** | **1956 kB** (그대로) | +10 |
| 앞 50MB, 페이지마다 1바이트 | 105176 kB | **53152 kB** | **+12809** |
| `memset` 100MB | 105176 kB | **104356 kB** | +25610 |

- **malloc 직후** → **VmSize 만 100MB ↑**, VmRSS 그대로
- **1바이트만 써도** → 그 페이지(4KB) 전체 할당
- **50MB / 4KB** → **12800 페이지** = fault 약 12800번 ≈ RSS 50MB ↑

**기본 사용형태**
```c
long f0 = minflt();
char *p = malloc(100 * MB);                  // VmSize ↑, VmRSS 그대로
for (long i = 0; i < 50 * MB; i += 4096) p[i] = 1;   // fault ≈ 12800
printf("%ld\n", minflt() - f0);
```
```bash
grep -E "VmSize|VmRSS" /proc/self/status
```

**연관 문제**
<<QA 3>>

### 4교시 · mmap

**키워드:** mmap, munmap, msync, MS_SYNC, MAP_SHARED, MAP_PRIVATE, MAP_ANONYMOUS, MAP_FAILED, PROT_READ, PROT_WRITE, page cache, fstat, 파일 매핑, toupper

**개요**
- `mmap` 으로 **파일을 주소 공간에 연결**해, `read`/`write` 복사 없이 **포인터로 바로** 읽고 쓴다. 실제 데이터는 커널의 page cache 에 있고 프로세스는 그 페이지를 자기 페이지 테이블로 가리킨다. `MAP_SHARED` 와 `MAP_PRIVATE` 의 차이가 핵심.

**핵심 용어**
- **page cache**: 커널이 파일 내용을 RAM 에 보관하는 캐시 (Day 4 에서 자세히).
- **MAP_SHARED**: page cache 를 **직접 고침** → 파일과 다른 프로세스에 보인다 (파일 수정, 프로세스 간 공유).
- **MAP_PRIVATE**: 쓰는 순간 **내 복사본** 생성(COW) → 파일은 그대로 (실행 파일 로딩, 읽기 위주).
- **MAP_ANONYMOUS**: 파일 없이 0 으로 시작 (큰 malloc, 공유 메모리 — Day 2 `02_nice_race` 의 공유 카운터).
- **MAP_FAILED**: mmap 실패 반환값. **`NULL` 이 아니다**(`(void *)-1`).

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `mmap` | `void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off);` | 파일/익명 메모리를 주소 공간에 매핑. 실패 시 **`MAP_FAILED`** |
| `msync` | `int msync(void *addr, size_t len, int flags);` | 고친 내용을 디스크에 반영 (`MS_SYNC` = 즉시) |
| `munmap` | `int munmap(void *addr, size_t len);` | 매핑 해제 |
| `fstat` | `int fstat(int fd, struct stat *st);` | 파일 정보 (`st.st_size` = 크기) |

**예제 코드가 하는 일** ([04_mmap/](Linux_day3/04_mmap/))

| 예제 | 하는 일 |
|---|---|
| `04_mmap_upper.c` | 파일을 `open(O_RDWR)` → `fstat` 으로 크기 → `mmap` → 바로 `close` → 배열처럼 `toupper` 로 대문자 변환 → (SHARED 면) `msync` → `munmap`. 인자 `private` 면 `MAP_PRIVATE` |

**핵심 내용**

#### mmap 으로 파일 고치기
> 파일을 메모리처럼 읽고 쓴다
- **순서** → `open(O_RDWR)` → `fstat` (크기) → `mmap(…, MAP_SHARED, fd, 0)` → `close(fd)` → `p[i] = toupper(p[i])` → `msync` → `munmap`
- ⚠ 빈 파일은 매핑 불가 (`fstat` 으로 크기 확인)
- **매핑은 `close(fd)` 해도 유지** → `munmap` 으로 끝낸다

#### MAP_PRIVATE vs MAP_SHARED
> PRIVATE 은 내 사본만, SHARED 는 파일과 다른 프로세스에도 반영
- **`./04_mmap_upper test.txt private`** → 메모리 안 `HELLO MMAP WORLD`, **`cat` 은 `hello mmap world` (그대로)**
- **`./04_mmap_upper test.txt`** (SHARED) → `cat` 이 **`HELLO MMAP WORLD`** (파일에 반영)
- **SHARED 로 A 가 쓰면** → page cache 페이지가 바뀌므로 같은 파일을 SHARED 로 매핑한 **B 도 본다**

**기본 사용형태**
```c
int fd = open(path, O_RDWR);
struct stat st; fstat(fd, &st);
char *p = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
if (p == MAP_FAILED) { perror("mmap"); return 1; }     // NULL 아님!
close(fd);                                             // 매핑은 유지
for (off_t i = 0; i < st.st_size; i++) p[i] = toupper((unsigned char)p[i]);
msync(p, st.st_size, MS_SYNC);
munmap(p, st.st_size);
```

**연관 문제**
<<QA 4>>

### 5교시 · Copy-on-Write

**키워드:** Copy-on-Write, COW, fork, 페이지 테이블 복사, 읽기 전용 공유, page fault 복사, minflt, fork 가 빠른 이유, fork 후 exec

**개요**
- Day 1 에서 fork 후 자식의 `x` 와 부모의 `x` 가 달랐다. 그렇다고 fork 가 메모리를 그 자리에서 다 복사하지는 않는다. **쓰는 순간 그 페이지만 복사**하는 **Copy-on-Write** 를 fault 수로 증명한다.

**핵심 용어**
- **Copy-on-Write(COW)**: fork 직후 부모·자식이 **같은 물리 페이지를 읽기 전용으로 공유**하다가, 누군가 **처음 쓸 때** page fault → 커널이 **그 페이지만 복사**해 쓰는 쪽에 준다. 같은 교과서를 같이 보다 필기할 사람만 복사본을 받는 것.
- **fork 가 빠른 이유**: 물리 페이지 대신 **페이지 테이블만 복사**하기 때문.

**새로 나온 함수/명령** — 새 함수 없음 (`fork`, `minflt()`, `sysconf(_SC_PAGESIZE)` 재사용)

**예제 코드가 하는 일** ([05_cow/](Linux_day3/05_cow/))

| 예제 | 하는 일 |
|---|---|
| `05_cow.c` | 부모가 50MB 버퍼를 모두 채운 뒤 fork, 자식이 버퍼의 `pct`% 를 페이지마다 1바이트씩 쓰고 그동안의 minflt 를 출력. 부모는 `buf[0]` 이 그대로인지 확인 |

**핵심 내용**

#### copy-on-write 순서
> fork 직후에는 같은 물리 페이지를 공유하고, 쓰는 순간 그 페이지만 복사
- **①** 부모가 `fork()` → 자식 페이지 테이블이 같은 물리 페이지를 가리킴 (읽기 전용)
- **②** 자식이 쓰기 → page fault
- **③** 커널이 **그 페이지만** 복사 → 자식 테이블을 새 페이지로

#### COW 실측
> fault 수 = 쓴 페이지 수 (50MB = 12800 페이지)

| 자식이 쓴 비율 | 페이지 | minflt |
|---|---|---|
| 0 % | 0 | +0 |
| 10 % | 1280 | **+1280** |
| 50 % | 6400 | +6400 |
| 100 % | 12800 | +12800 |

- **부모 `buf[0] = 1`** → 자식이 써도 부모 값은 그대로
- **쓰지 않은 페이지** → 끝까지 공유
- **부모가 먼저 써도** → 복사가 일어난다
- **fork 직후 exec 하는 셸** → 거의 복사하지 않는다

**기본 사용형태**
```c
memset(buf, 1, SIZE);                 // 부모가 채움
if (fork() == 0) {
    long f0 = minflt();
    for (long i = 0; i < limit; i += 4096) buf[i] = 2;   // 쓰는 페이지만 복사
    printf("%ld\n", minflt() - f0);   // = 쓴 페이지 수
    exit(0);
}
wait(NULL);                           // 부모 buf[0] 은 여전히 1
```

**연관 문제**
<<QA 5>>

### 6교시 · 메모리 잠금 · 오버커밋 · OOM

**키워드:** mlockall, MCL_CURRENT, MCL_FUTURE, VmLck, ulimit -l, overcommit, overcommit_memory 0/1/2, OOM killer, oom_score, SIGKILL, MemoryMax, systemd-run, dmesg

**개요**
- 실시간 루프 안에서 **처음 만지는 메모리**의 page fault 도 지연이다 → `mlockall` 로 미리 확보. 또 리눅스는 실제 RAM 보다 많은 malloc 을 허락(overcommit)하므로, 진짜 부족해지면 `malloc` 이 NULL 이 아니라 **OOM killer 의 SIGKILL** 로 실패한다는 것을 cgroup 안에서 안전하게 관찰한다.

**핵심 용어**
- **mlockall(MCL_CURRENT | MCL_FUTURE)**: 지금과 **앞으로의** 모든 페이지를 RAM 에 확보하고 swap 으로 내보내지 않음. `MCL_FUTURE` 면 이후 malloc 하는 **순간 실제 페이지 확보**.
- **VmLck**: 잠긴 메모리 크기(`/proc/self/status`).
- **overcommit**: 실제 RAM 보다 많은 할당을 허락. 대부분 프로그램이 받은 걸 다 안 쓰기 때문(항공사 초과 예약).
- **OOM killer**: RAM 이 정말 바닥나면 점수(`/proc/PID/oom_score`)가 높은 프로세스를 골라 **SIGKILL**.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `mlockall` | `int mlockall(int flags);` | 모든 페이지 잠금 (한도 초과 시 sudo 필요) |
| `ulimit -l` | — | 일반 사용자가 잠글 수 있는 한도(kB) |
| `/proc/sys/vm/overcommit_memory` | `0` / `1` / `2` | 0 터무니없는 요청만 거절(기본) / 1 항상 허용 / 2 엄격(한도 넘으면 malloc NULL) |
| `sudo systemd-run --scope -p MemoryMax=100M -p MemorySwapMax=0 ./prog` | — | 메모리 100MB 한도 cgroup 에서 실행 |
| `sudo dmesg \| tail -5` | — | oom-kill 기록 확인 |

**예제 코드가 하는 일** ([06_mlock_oom/](Linux_day3/06_mlock_oom/))

| 예제 | 하는 일 |
|---|---|
| `06_mlock.c` | (인자 `lock` 이면 `mlockall`) 32MB malloc → "실시간 루프"에서 4KB 마다 처음 만짐. malloc 중 / 루프 중 minflt, 루프 시간, VmLck 출력 |
| `07_oom.c` | overcommit 설정을 출력하고 10MB 씩 malloc + `memset` 하며 VmRSS 를 출력 — cgroup 한도를 넘으면 OOM killer 에게 죽음 |

**핵심 내용**

#### mlockall
> 페이지를 미리 확보하고 잠가 루프 안의 fault 를 없앤다
- **잠금 없음** → malloc 중 minflt +1 / **루프 중 +8193** / 루프 **7.99 ms** / VmLck 0
- **mlockall** → **malloc 중 +8193** / 루프 중 **+1** / 루프 **0.16 ms** / VmLck 35512 kB
- **효과** → fault 가 malloc 시점으로 옮겨 감, 루프 시간 약 **50~100배** ↓ (강의: 13.04 → 0.12 ms)
- **`ulimit -l`** → 강의 VM 8192 kB 라 32MB 잠금에 sudo 필요 (WSL 은 65536 kB 라 sudo 없이 됨)

#### overcommit 과 OOM
> 기본 설정에서 malloc 은 거의 실패하지 않고, 실제로 쓸 때 죽는다
- **기본 overcommit (0)** → malloc 이 NULL 을 거의 안 돌려준다
- **강의 실행** → 100MB 한도에서 `90 MB 사용` 후 **`Killed`**, dmesg `Memory cgroup out of memory: Killed process … (07_oom)`
- ⚠ 실패는 **`memset` (실제 사용) 순간에 SIGKILL** → `if (p == NULL)` 만으로는 OOM 을 못 막는다
- ⚠ VM 전체를 OOM 으로 몰면 sshd 까지 죽을 수 있다 → **반드시 cgroup 한도 안에서**
- (WSL 은 sudo·systemd 가 없어 강의 결과로 정리, `overcommit_memory` = 0 은 확인)

**기본 사용형태**
```c
if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) perror("mlockall");
char *buf = malloc(SIZE);       // 이 순간 페이지 확보 → 루프 안 fault ≈ 0
```
```bash
ulimit -l
cat /proc/sys/vm/overcommit_memory
sudo systemd-run --scope -p MemoryMax=100M -p MemorySwapMax=0 ./07_oom
sudo dmesg | tail -5
```

**연관 문제**
<<QA 6>>

### 7교시 · 메모리 오류 도구

**키워드:** 메모리 누수 leak, use-after-free, 버퍼 오버플로, double free, valgrind, --leak-check=full, definitely lost, Invalid read, AddressSanitizer, -fsanitize=address, heap-buffer-overflow, -Wuse-after-free, -Wstringop-overflow

**개요**
- 대표 메모리 오류 4가지는 **바로 죽지 않는 경우가 많다** — 그래서 도구로 찾는다. 컴파일러 경고, **valgrind**(재빌드 불필요), **AddressSanitizer**(줄 번호까지) 세 가지를 비교한다. (C Day2 7교시 valgrind 의 확장)

**핵심 용어**
- **누수(leak)**: malloc 후 free 를 잊음 → 오래 도는 프로그램의 메모리가 계속 는다.
- **use-after-free**: free 한 포인터로 읽기·쓰기 → 다른 데이터를 망가뜨리거나 **우연히 동작**.
- **버퍼 오버플로**: 8바이트에 10바이트 복사 → 이웃 블록·malloc 관리 정보 파괴.
- **double free**: 같은 포인터를 두 번 free → **glibc 가 abort** 하거나 보안 취약점.
- **레드존(redzone)**: ASan 이 블록 앞뒤에 둔 감시 구역. 건드리면 즉시 오류.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `valgrind --leak-check=full ./prog` | — | 재빌드 없이 검사, **10~50배 느림** |
| `gcc -g -fsanitize=address` | — | ASan 빌드, **약 2배 느림**, 오류 순간 **즉시 중단 + 줄 번호** |
| `-Wuse-after-free`, `-Wstringop-overflow` | — | gcc 가 빌드 단계에서 일부 오류를 경고 |

**예제 코드가 하는 일** ([07_bugs/](Linux_day3/07_bugs/))

| 예제 | 하는 일 |
|---|---|
| `08_bugs.c` | 인자로 `leak`(free 안 함) / `uaf`(free 후 읽기) / `overflow`(8바이트에 `"123456789"`) / `double`(두 번 free) 중 하나를 일부러 실행. `08_bugs_asan` 은 ASan 빌드 |

**핵심 내용**

#### 컴파일러 경고가 첫 번째 도구
> 빌드할 때 이미 일부 메모리 버그를 알려 준다
- **`pointer 'p' used after 'free' [-Wuse-after-free]`**
- **`writing 10 bytes into a region of size 8 overflows the destination [-Wstringop-overflow=]`**

#### 도구 없이 실행하면
> 메모리 버그는 대부분 조용히 지나간다
- **`leak`** → 정상 출력, 종료 0
- **`uaf`** → `해제 후 읽기: 657823656` 같은 **쓰레기 값**, 종료 0 (항상 segfault 가 아니다)
- **`overflow`** → 정상 출력될 수 있다 (넘친 바이트가 우연히 빈 공간에)
- **`double`** → `free(): double free detected in tcache 2`, **Aborted(134)** — glibc 가 스스로 멈추는 유일한 경우

#### valgrind 와 ASan
> 두 도구 모두 4가지(누수 · 해제 후 사용 · 넘침 · 이중 해제)를 잡는다
- **valgrind** → `100 bytes in 1 blocks are definitely lost` … `by 0x…: leak (08_bugs.c:13)` / `uaf` → `Invalid read of size 4`
- **ASan** → `heap-buffer-overflow … #1 … in overflow 08_bugs.c:29` / `attempting double-free` / `heap-use-after-free`
- **ASan 순서** → `-g -fsanitize=address` 로 빌드 → 실행 → 오류 순간 멈추고 종류 · 스택(줄 번호) 출력

**기본 사용형태**
```bash
gcc -Wall -Wextra -g 08_bugs.c -o 08_bugs
valgrind --leak-check=full ./08_bugs leak        # definitely lost
gcc -g -fsanitize=address 08_bugs.c -o 08_bugs_asan
./08_bugs_asan overflow                          # heap-buffer-overflow … 08_bugs.c:29
```

**연관 문제**
<<QA 7>>

### 8교시 · 종합 실습 memlab

**키워드:** memlab, demand, cow, mmap, lock, CSV, minflt, majflt, rss_kb, fault-around, mkstemp, ftruncate, memlab_run.sh, 온디바이스 AI 메모리 기법

**개요**
- Day 3 의 실험 네 가지(demand · cow · mmap · lock)를 `memlab` 하나로 묶어 **CSV 한 줄**(`exp,size_mb,minflt,majflt,rss_kb,ms`)씩 출력하고, 반복 측정해 비교·해석한다.

**핵심 용어**
- **fault-around**: 파일 매핑에서 fault 가 나면 page cache 의 **이웃 페이지를 한 번에** 연결해 fault 수를 줄이는 커널 기능.
- **rss_kb**: 실험 직후의 VmRSS.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `mkstemp` | `int mkstemp(char *template);` | `"/tmp/memlab_XXXXXX"` 로 임시 파일 생성 |
| `ftruncate` | `int ftruncate(int fd, off_t len);` | 파일 크기를 `len` 으로 |
| `./memlab demand\|cow\|mmap\|lock [MB]` | — | 실험 1회 → CSV 한 줄 |
| `./memlab_run.sh 5` | — | 실험마다 5회 반복해 `memlab.csv` 저장 후 `column -t` 로 표 출력 |

**예제 코드가 하는 일** ([08_memlab/](Linux_day3/08_memlab/))

| 예제 | 하는 일 |
|---|---|
| `memlab.c` | `demand`: malloc 후 처음 만짐 / `cow`: 채운 뒤 fork, 자식이 전부 씀 / `mmap`: 임시 파일을 MAP_SHARED 로 매핑해 만짐 / `lock`: `mlockall` 후 malloc·만짐 — 각각 fault·RSS·시간을 CSV 로 |
| `memlab_run.sh` | 실험별 N회 반복 실행해 CSV 로 모으고(lock 은 sudo 가능할 때만) 표로 출력 |

**핵심 내용**

#### 네 실험 비교 (64MB)
> 같은 양을 써도 방식에 따라 fault 수와 시간이 다르다

| 실험 | minflt | ms | 해석 |
|---|---|---|---|
| demand | 16393 | 25.03 | 64MB = **16384 페이지** → fault ≈ 16384 |
| cow | 16403 | 52.81 | fault 수는 비슷, **복사 비용 때문에 더 느림** |
| mmap | 556 | 11.31 | **fault-around** 로 이웃 페이지를 한 번에 → 훨씬 적음 (환경마다 다름) |
| lock | 1 | 0.50 | `mlockall` 로 **malloc 때 이미 확보** → 루프 fault ≈ 0, 가장 빠름 |

- **WSL 실행** → demand 16391 / cow 16399 / **mmap 16391** — `/tmp` 가 **tmpfs** 라 fault-around 효과가 안 나타남
- ⚠ `lock 64` → 한도(64MB) 초과로 malloc 실패 → segfault / **`lock 32`** → minflt **34**, **0.18 ms** (demand 32 는 8197, 8.65 ms)

#### page fault 를 보는 관점
> 나쁜 것이 아니라 RAM 을 아끼는 장치 — 실시간 루프 안에서만 문제
- **평소** → 필요한 순간까지 RAM 을 아낀다
- ⚠ **실시간 루프 안** → 지연
- **온디바이스 AI** → 모델 가중치는 **mmap** 으로 필요한 부분만 / 실시간 태스크는 **mlockall** / 장시간 프로그램은 **valgrind·ASan** 으로 누수 점검

**기본 사용형태**
```bash
make memlab
./memlab demand ; ./memlab cow ; ./memlab mmap ; sudo ./memlab lock
sudo -v && ./memlab_run.sh 5          # 반복 측정 → memlab.csv
```

**연관 문제**
<<QA 8>>
