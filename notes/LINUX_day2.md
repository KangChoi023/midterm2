## Day 2 — 스케줄링

### 1교시 · 컨텍스트 스위치 관찰

**키워드:** 컨텍스트 스위치, 프로세스 상태 R S D Z, 자발적/비자발적 전환, 선점(preemption), voluntary_ctxt_switches, nonvoluntary_ctxt_switches, /proc/self/status, load average, /proc/loadavg, run queue, usleep, stallwatch, Hyper-V 거북이

**개요**
- Day 1 에서 프로세스를 만들고(fork)·바꾸고(exec)·거뒀다면(wait), Day 2 는 그 프로세스들이 **CPU 를 언제, 얼마나 받는가**를 숫자로 잰다. 1교시는 그 출발점인 **컨텍스트 스위치**와 **프로세스 상태**.
- VM 의 vCPU 는 2개인데 프로세스는 수백 개 → 커널이 짧은 시간마다 실행할 태스크를 바꾼다. 이 교체를 **자발적**(스스로 잠듦) 과 **비자발적**(빼앗김) 으로 나누어 `/proc/self/status` 에서 읽는다.
- 실습 전 점검(1-0): Windows 에서 Hyper-V 가 켜진 채 VirtualBox 를 쓰면 **잠든 vCPU 가 제때 못 깨어나** 지연 측정이 수 초씩 틀어진다. `stallwatch` 로 확인.

**핵심 용어**
- **컨텍스트 스위치(context switch)**: CPU 를 한 태스크에서 다른 태스크로 넘기는 일. 지금 태스크의 **레지스터·스택 포인터를 저장**하고 다음 태스크의 것을 **복원**한다. 책 읽다 책갈피 꽂고 다른 책을 펴는 것.
- **자발적 전환(voluntary)**: 프로세스가 `sleep`·I/O 대기로 **스스로** CPU 를 내려놓음.
- **비자발적 전환(nonvoluntary)**: 할 일이 남았는데 스케줄러가 **빼앗음**. 이 빼앗기를 **선점(preemption)** 이라 한다.
- **run queue**: CPU 를 기다리는 실행 가능 태스크의 줄.
- **load average**: `/proc/loadavg` 의 1·5·15분 **평균 실행 대기 태스크 수**(R + D 상태의 이동 평균). vCPU 2개에서 2.0 이면 꽉 참.
- **프로세스 상태(STAT)**: `R` 실행·대기 / `S` 잠듦 / `D` 깨울 수 없는 잠(주로 디스크 I/O) / `Z` 좀비.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `usleep` | `int usleep(useconds_t us);` | µs 동안 잠듦 (`usleep(1000)` = 1ms). 부를 때마다 자발적 전환 |
| `clock_gettime` | `int clock_gettime(clockid_t id, struct timespec *t);` | 시각 읽기. `CLOCK_MONOTONIC` = 되돌아가지 않는 시계 |
| `/proc/self/status` | `grep ctxt /proc/self/status` | 자기 프로세스의 상태·메모리·**전환 횟수** (커널이 읽는 순간 만들어 줌) |
| `/proc/loadavg` | `cat /proc/loadavg` | `0.85 0.42 0.20 1/215 2410` = 1/5/15분 load, 실행수/전체수, 마지막 PID |
| `ps -e --no-headers \| wc -l` | — | 지금 프로세스 수 |
| `stress-ng --cpu N --timeout 20 &` | — | CPU 를 N 개 바쁘게 만드는 경쟁자 |

**예제 코드가 하는 일** ([01_ctxsw/](Linux_day2/01_ctxsw/))

| 예제 | 하는 일 |
|---|---|
| `01_ctxsw.c` | 실행 전후로 `/proc/self/status` 의 두 전환 카운터를 읽어 차이를 출력. `sleep` 모드는 `usleep(1000)` 을 1000번, `busy` 모드는 2초 동안 계속 계산 |
| `stallwatch.c` | 쉬지 않고 `clock_gettime` 을 연속으로 읽다가 **두 번 읽은 사이가 5ms 이상** 벌어지면 "그동안 실행 못 됨"으로 기록 → VM 이 호스트에게 멈춰지는지 판단 |

**핵심 내용**

#### 프로세스 상태
> `R` `S` `D` `Z` — 실행 · 잠듦 · 깨울 수 없는 잠 · 좀비
- **`R`** → 실행/대기 / **`S`** → 잠듦 / **`Z`** → 좀비
- **`D`** → 깨울 수 없는 잠 (I/O 완료 전엔 시그널로도 안 깸)
- ⚠ `D` 상태는 `kill -9` 도 바로 안 들을 수 있다

#### 자발적 vs 비자발적 문맥 전환
> 스스로 잠들면 자발적, 빼앗기면 비자발적 — 경쟁자가 있어야 선점이 늘어난다
- **읽는 곳** → `/proc/self/status` (커널이 프로세스별 통계를 가상 파일로)
- **`./01_ctxsw sleep`** → 자발적 **1000**, 비자발적 0 (`usleep(1000)` 을 1000번)
- **`./01_ctxsw busy` 경쟁자 없음** → 자발적 0, 비자발적 **6** (빼앗아 갈 태스크가 없음)
- **경쟁자(busy × CPU 수) + `busy`** → 비자발적 **약 80** (강의: vCPU 2 + `stress-ng --cpu 2` 에서 약 270)

#### load average
> CPU 를 쓰거나 기다리는 태스크 수의 평균 — 사용률(%)이 아니다
- **vCPU 2개에서 계속 4.0** → 평균 2개 정도가 **CPU 를 기다리는 중**

#### VM 지연 점검
> 잠드는 예제만 이상하면 시계가 아니라 늦게 깨어난 것
- **`stallwatch` (WSL)** → 3초에 21회, 최대 16.4ms → "가벼움" / 0회 정상 / 최대 **20ms 이상** 심각
- **강의 VM `jitter -n 20000`** → 최대 수 ms 이하 정상 / **100ms 이상** → Hyper-V 문제
- **해결** → 거북이 아이콘 → Hyper-V/VBS 끄기, 못 끄면 `idle=poll`
- ⚠ 계산 예제(02·03·04)는 정상인데 **잠드는 예제(06, jitter)만** 이상 → vCPU idle 후 깨우기 지연

**기본 사용형태**
```c
long status_value(const char *key) {          // /proc/self/status 에서 숫자 읽기
    FILE *fp = fopen("/proc/self/status", "r");
    char line[256]; long v = -1; size_t n = strlen(key);
    while (fgets(line, sizeof line, fp))
        if (strncmp(line, key, n) == 0) { v = atol(line + n + 1); break; }
    fclose(fp); return v;
}
long v0 = status_value("voluntary_ctxt_switches");
for (int i = 0; i < 1000; i++) usleep(1000);   // 자발적 +1000
long v1 = status_value("voluntary_ctxt_switches");
```
```bash
./01_ctxsw sleep                        # 자발적 1000 / 비자발적 0
stress-ng --cpu 2 --timeout 20 & ./01_ctxsw busy   # 비자발적 증가
cat /proc/loadavg; grep ctxt /proc/self/status
```

**연관 문제**
<<QA 1>>

### 2교시 · CFS 실측

**키워드:** CFS, Completely Fair Scheduler, vruntime, weight, nice, renice, setpriority, sched_prio_to_weight, 1024, 레드블랙 트리, /proc/self/sched, EEVDF, 02_nice_race, 03_vruntime

**개요**
- 일반 프로세스(SCHED_OTHER)를 나누는 리눅스 스케줄러 **CFS** 의 원리를 숫자로 확인한다. 핵심은 **vruntime 이 가장 작은 태스크를 고른다** 와 **nice → weight → CPU 몫**.
- 같은 CPU 에 고정한 두 프로세스의 계산량 비율(02)과, `/proc/self/sched` 의 vruntime 증가 속도(03)로 공식을 검증한다.

**핵심 용어**
- **CFS(Completely Fair Scheduler)**: 커널 5.15(수업 VM)의 기본 스케줄러. 모두에게 "공평하게" CPU 를 나눈다. 6.6 부터는 **EEVDF** 로 바뀌었다.
- **vruntime(가상 실행 시간)**: 실제 실행 시간을 weight 로 보정한 값. **가장 작은 태스크가 다음에 실행**된다(레드블랙 트리의 맨 왼쪽). 덜 먹은 사람에게 먼저 주는 급식 줄.
- **nice**: -20(우대) ~ 19(양보). 1 오를 때마다 weight 가 약 **1.25배** 줄어든다. 일반 사용자는 **올리기만**(양보만) 가능.
- **weight**: nice 로 정해지는 몫. **nice 0 = 1024**, nice 5 = 335, nice 10 = 110, nice 19 = 15.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `setpriority` | `int setpriority(int which, id_t who, int prio);` | nice 값 설정 (`PRIO_PROCESS, 0, 10` = 나 자신을 nice 10) |
| `getpriority` | `int getpriority(int which, id_t who);` | 현재 nice 값 |
| `nice -n 10 ./prog` | — | nice 10 으로 실행 |
| `renice -n 5 -p PID` | — | 실행 중인 프로세스의 nice 변경 |
| `/proc/self/sched` | `grep vruntime /proc/self/sched` | `se.vruntime`, `se.sum_exec_runtime`(ms) 등 스케줄러 통계 |

**예제 코드가 하는 일** ([02_cfs/](Linux_day2/02_cfs/))

| 예제 | 하는 일 |
|---|---|
| `02_nice_race.c` | 자식 2개를 **CPU 0 에 고정**하고 B 만 nice 를 올려 3초 동안 계산 횟수를 세고, 공유 메모리(`mmap`)로 모아 **실측 비율 vs weight 예상 비율**을 출력 |
| `03_vruntime.c` | 0.5초씩 4번 계산하며 `/proc/self/sched` 의 실행시간·vruntime 증가량과 **그 비율**을 출력 |

**핵심 내용**

#### vruntime 과 weight
> 스케줄러는 항상 vruntime 이 가장 작은 태스크를 고른다
- **vruntime 증가량** → 실제 실행 시간 × **1024 / weight**
- **nice 가 높으면** → vruntime 이 빨리 늘어 트리 오른쪽으로 → **덜 뽑힘**
- **`03_vruntime`** → nice 0 비율 **1.00**, `nice -n 10` 비율 **9.31** (=1024/110)
- **nice 낮추기 (-5)** → root 만 가능

#### CPU 몫 = weight 비율
> 내 weight / 전체 weight 합
- **`./02_nice_race 10`** → A **90.3%** / B **9.7%** (1024:110 과 일치)
- **`./02_nice_race 5`** → A 75.4% / B 24.6% (예상 75.3 / 24.7)
- **세 태스크 (nice 0, 5, 10)** → nice 0 이 가장 많이
- ⚠ 두 자식을 **같은 CPU 에 고정**해야 경쟁이 생긴다 (아니면 각자 다른 CPU 에서 둘 다 100%)
- **커널 6.18 (EEVDF) 에서도** → 비율은 그대로 (weight 표는 같음), 수업 VM(5.15) 은 CFS 로 해석

**기본 사용형태**
```c
setpriority(PRIO_PROCESS, 0, 10);          // 나를 nice 10 으로
// 몫 = w / (wa + wb):  1024 / (1024 + 110) = 90.3 %
```
```bash
./02_nice_race 10                # 90.3 % : 9.7 %
nice -n 10 ./03_vruntime         # 비율 9.31
renice -n 5 -p 2410              # 실행 중 프로세스 nice 변경
```

**연관 문제**
<<QA 2>>

### 3교시 · CPU affinity

**키워드:** CPU affinity, 친화도, taskset, sched_setaffinity, sched_getaffinity, sched_getcpu, cpu_set_t, CPU_ZERO, CPU_SET, CPU_ISSET, CPU_COUNT, _GNU_SOURCE, EINVAL, 고정의 대가

**개요**
- 프로세스가 **어느 CPU 에서 돌 수 있는지**(affinity)를 정하고 확인한다. 고정하면 캐시가 데워진 채 유지되고 실행 위치가 예측 가능하지만, 그 CPU 가 바쁘면 다른 CPU 가 놀아도 기다려야 한다.

**핵심 용어**
- **affinity(친화도)**: 프로세스가 실행될 수 있는 **CPU 목록**. 기본값은 모든 CPU. 지정석 같은 것.
- **`cpu_set_t`**: CPU 목록을 담는 비트 집합. `CPU_ZERO`(비우기) → `CPU_SET(n)`(n 추가) → `CPU_ISSET`(확인) / `CPU_COUNT`(개수).
- **`_GNU_SOURCE`**: `sched_setaffinity` 등 GNU 확장을 쓰려면 소스 **맨 앞**에 정의.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `sched_setaffinity` | `int sched_setaffinity(pid_t pid, size_t sz, const cpu_set_t *set);` | CPU 고정 (`pid` 0 = 나). 없는 CPU 면 **EINVAL** |
| `sched_getaffinity` | `int sched_getaffinity(pid_t pid, size_t sz, cpu_set_t *set);` | 허용 CPU 목록 읽기 |
| `sched_getcpu` | `int sched_getcpu(void);` | **지금** 실행 중인 CPU 번호 |
| `sysconf(_SC_NPROCESSORS_ONLN)` | — | 온라인 CPU 수 |
| `taskset -c 0 ./prog` | — | CPU 0 에서만 실행 |
| `taskset -cp 1 PID` | — | 실행 중인 PID 를 CPU 1 로 |

**예제 코드가 하는 일** ([03_affinity/](Linux_day2/03_affinity/))

| 예제 | 하는 일 |
|---|---|
| `04_affinity.c` | 허용 CPU 목록을 출력하고, 인자가 있으면 그 CPU 에 고정한 뒤 0.2초마다 `sched_getcpu()` 를 15번 찍는다 |
| `07_busy.c` | 1초마다 계산 횟수를 출력하는 CPU 부하 프로그램 (고정의 대가 · cgroup 실습용) |

**핵심 내용**

#### CPU affinity 사용
> 프로세스가 돌 수 있는 CPU 를 제한한다
- **`./04_affinity 1`** → `[고정] 허용 CPU: 1 (총 1개)`, 출력이 `1 1 1 … 1`
- **`taskset -c 0 ./04_affinity`** → `[시작] 허용 CPU: 0`
- **`./04_affinity 99`** (없는 CPU) → `sched_setaffinity: Invalid argument`
- **`CPU_ZERO` 후 `CPU_SET(1)`** → 집합에는 **CPU 1 하나**

#### 고정의 장단점
> 캐시 재사용 · 예측 가능 vs 다른 CPU 가 비어도 기다림
- **장점** → 캐시 재사용, 실행 위치 예측 가능
- ⚠ **단점** → 고정한 CPU 가 바쁘면 **다른 CPU 가 비어 있어도 기다림**
- **확인** → CPU 0 에 busy 루프 + `taskset -c 0 ./07_busy` → 1초 **2313만 회** / 빈 CPU 1 에서 **4672만 회** → **약 50%**
- **고정하지 않으면** → 스케줄러가 빈 CPU 로 옮겨 100% 가까이

**기본 사용형태**
```c
#define _GNU_SOURCE            // 맨 앞
#include <sched.h>
cpu_set_t set;
CPU_ZERO(&set);
CPU_SET(1, &set);
if (sched_setaffinity(0, sizeof set, &set) == -1) perror("sched_setaffinity");
printf("%d\n", sched_getcpu());   // 1
```
```bash
taskset -c 0 ./prog       # 셸에서 고정
taskset -cp 1 3001        # 실행 중인 PID 를 CPU 1 로
```

**연관 문제**
<<QA 3>>

### 4교시 · 실시간 정책

**키워드:** SCHED_OTHER, SCHED_FIFO, SCHED_RR, 실시간 우선순위 1~99, sched_setscheduler, sched_getscheduler, sched_param, chrt, EPERM, RT throttling, sched_rt_runtime_us, sched_rt_period_us, 950000

**개요**
- CFS(공평)와 달리 **실시간(RT) 정책** 은 우선순위가 높은 태스크가 **무조건 먼저** 실행된다. FIFO·RR 의 규칙, root 권한이 필요한 이유, 무한 루프를 막는 **RT throttling** 안전장치를 다룬다.

**핵심 용어**
- **SCHED_OTHER**: 일반 정책(CFS). 우선순위는 nice -20~19.
- **SCHED_FIFO**: RT 1~99. **끝나거나, 잠들거나, 더 높은 RT 가 올 때까지** 계속 실행. 새치기 없는 줄.
- **SCHED_RR**: FIFO + **같은 우선순위끼리 시간 조각(기본 100ms)** 씩 돌아가며. 
- **RT throttling**: 1초(`sched_rt_period_us`=1000000) 중 **0.95초**(`sched_rt_runtime_us`=950000)까지만 RT 를 실행하고 0.05초는 일반 태스크에 남기는 안전장치.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `sched_setscheduler` | `int sched_setscheduler(pid_t pid, int policy, const struct sched_param *p);` | 정책 변경. 일반 사용자는 **EPERM** |
| `sched_getscheduler` | `int sched_getscheduler(pid_t pid);` | 현재 정책 |
| `sched_get_priority_min/max` | `int sched_get_priority_max(int policy);` | 우선순위 범위 (FIFO 1~99) |
| `chrt -p PID` | — | 정책·우선순위 확인 |
| `sudo chrt -f 50 ./prog` | — | FIFO 50 으로 실행 (`-r` 은 RR) |

**예제 코드가 하는 일** ([04_rt/](Linux_day2/04_rt/))

| 예제 | 하는 일 |
|---|---|
| `05_rt_policy.c` | 현재 정책·FIFO 범위·RT throttling 값을 출력하고 `SCHED_FIFO`(기본 50)로 바꾼 뒤 10초 대기 (그동안 `chrt -p` 로 확인) |

**핵심 내용**

#### 실시간 정책 권한
> RT 는 항상 일반 태스크보다 먼저 — 그래서 root(sudo) 필요
- **일반 사용자 실행** → `현재 정책=SCHED_OTHER`, `FIFO 우선순위 범위: 1 ~ 99`, `RT throttling: 950000 us / 1000000 us`
- ⚠ 일반 사용자 `sched_setscheduler` → `Operation not permitted`
- **`sudo ./05_rt_policy 50`** → `변경 후 정책=SCHED_FIFO, 우선순위=50`
- **`chrt -p PID`** → `current scheduling policy: SCHED_OTHER`, `priority: 0` (일반 정책)

#### RT throttling
> FIFO 무한 루프여도 1초 중 0.05초는 다른 태스크가 돈다
- **효과** → PuTTY 가 **매우 느리지만 가끔 반응**
- ⚠ 값을 `-1` 로 → 안전장치가 꺼짐 (수업에서 절대 금지)
- **올바른 RT 태스크 모양** → 주기마다 짧게 일하고 **`clock_nanosleep` 으로 잠든다** (5교시)

**기본 사용형태**
```c
struct sched_param sp = { .sched_priority = 50 };
if (sched_setscheduler(0, SCHED_FIFO, &sp) == -1) perror("sched_setscheduler");   // 일반 사용자 → EPERM
```
```bash
sudo chrt -f 50 ./prog                      # FIFO 50
chrt -p 2410                                # 정책 확인
cat /proc/sys/kernel/sched_rt_runtime_us    # 950000
```

**연관 문제**
<<QA 4>>

### 5교시 · 시간 측정과 주기 실행

**키워드:** CLOCK_REALTIME, CLOCK_MONOTONIC, clock_gettime, timespec, tv_sec, tv_nsec, nanosleep, clock_nanosleep, TIMER_ABSTIME, 상대 sleep, 절대 sleep, 드리프트(누적 오차), 주기 루프

**개요**
- 주기 태스크(1ms 마다 센서 읽기 등)를 만들 때 **어떤 시계로 재고, 어떻게 잠들어야 하는지**. "지금부터 1ms" 를 반복하면 오차가 쌓이고, "시작 + i×1ms 시각까지" 잠들면 쌓이지 않는다.

**핵심 용어**
- **CLOCK_REALTIME**: 벽시계. NTP·사용자가 바꿀 수 있음 → **로그의 날짜·시각**용.
- **CLOCK_MONOTONIC**: 부팅 후 계속 증가, **거꾸로 가지 않음** → **경과 시간 측정·주기 실행**용.
- **상대 sleep**: "지금부터 N 동안". 일한 시간 + 깨어나는 지연이 **매 주기 더해진다**(드리프트).
- **절대 sleep**: "이 시각까지". 늦은 만큼이 **다음 주기에서 흡수**된다.
- **단위**: 1 s = 1,000 ms = 1,000,000 us = 1,000,000,000 ns.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `clock_gettime` | `int clock_gettime(clockid_t id, struct timespec *t);` | 시각 읽기 (`tv_sec` 초 + `tv_nsec` 나노초) |
| `nanosleep` | `int nanosleep(const struct timespec *req, struct timespec *rem);` | **상대** sleep |
| `clock_nanosleep` | `int clock_nanosleep(clockid_t id, int flags, const struct timespec *t, struct timespec *rem);` | `TIMER_ABSTIME` 이면 **절대 시각까지** sleep |

**예제 코드가 하는 일** ([05_time/](Linux_day2/05_time/))

| 예제 | 하는 일 |
|---|---|
| `06_sleep_drift.c` | 매 주기 0.2ms 일하는 1ms 주기 루프를 1000번, ① 상대 `nanosleep` ② 절대 `clock_nanosleep(TIMER_ABSTIME)` 로 돌려 총 걸린 시간과 오차를 비교 |

**핵심 내용**

#### 경과 시간 재기
> 경과 시간은 MONOTONIC — 시계를 조정해도 거꾸로 가지 않는다
- **µs 계산** → `(b.tv_sec - a.tv_sec) * 1000000LL + (b.tv_nsec - a.tv_nsec) / 1000`

#### 상대 vs 절대 sleep
> 주기 태스크는 반드시 절대 시각으로 — 상대 sleep 은 오차가 쌓인다
- **상대 nanosleep** → **1292.3 ms** (오차 +292.3, 강의 +303)
- **절대 clock_nanosleep** → **1000.1 ms** (오차 +0.1)
- ⚠ 주기마다 0.3ms 밀리면 → 1시간(360만 번)에 약 **18분** 늦어짐
- **다음 시각** → `next.tv_nsec += 1000000` 뒤 **10억 이상이면 `tv_sec` 로 올림** (`add_ns`)
- **`work()` 가 주기보다 길면** → 다음 시각이 이미 지나 **바로 깨어남** (쉬지 않고 따라잡기)
- ⚠ 상대 오차 +1000ms 초과 · 절대 오차 수십 ms → 측정이 아니라 **VM 문제** (1교시 점검)

**기본 사용형태**
```c
struct timespec next;
clock_gettime(CLOCK_MONOTONIC, &next);
for (;;) {
    work();
    next.tv_nsec += 1000000;                                   // +1ms
    while (next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; next.tv_sec++; }
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);   // 절대 시각까지
}
```

**연관 문제**
<<QA 5>>

### 6교시 · cgroup v2 로 CPU 제한

**키워드:** cgroup, control group, cgroup v2, /sys/fs/cgroup, cpu.max, quota, period, cpu.weight, cpu.stat, nr_throttled, throttled_usec, cgroup.procs, systemd-run, CPUQuota, Docker

**개요**
- 프로세스 **하나가 아니라 묶음**에 자원 한도를 거는 커널 기능 **cgroup** 으로 CPU 사용률을 제한한다. Docker·systemd 가 모두 이것을 쓴다. `systemd-run` 으로 간단히, `/sys/fs/cgroup` 에 직접 디렉터리를 만들어서도 해 본다.

**핵심 용어**
- **cgroup(control group)**: 프로세스 묶음에 CPU·메모리 등 **자원 한도**를 거는 기능. 가족 단위 데이터 요금제.
- **cgroup v2**: Ubuntu 22.04 는 **하나의 트리**를 `/sys/fs/cgroup` 에 보여 준다. 디렉터리 = cgroup, 파일 = 설정.
- **`cpu.max`**: `quota period` (us). `20000 100000` = 100ms 중 20ms = **20%**. `max 100000` = **제한 없음**.
- **`cpu.weight`**: 경쟁할 때 cgroup 사이의 **상대 몫**(기본 100, 200 이면 두 배).
- **`cpu.stat`**: 실제 사용량과 **제한당한 횟수(`nr_throttled`)·시간(`throttled_usec`)**.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `cat /proc/self/cgroup` | — | 내가 속한 cgroup (`0::/user.slice/…`) |
| `sudo systemd-run --scope -p CPUQuota=20% ./prog` | — | CPU 20% 제한으로 실행 |
| `echo "30000 100000" \| sudo tee …/cpu.max` | — | 직접 만든 cgroup 에 30% 한도 |
| `echo PID \| sudo tee …/cgroup.procs` | — | 프로세스를 그 cgroup 에 넣기 |

**예제 코드가 하는 일** ([06_cgroup/](Linux_day2/06_cgroup/))

| 예제 | 하는 일 |
|---|---|
| `07_busy.c` | 1초마다 계산 횟수와 "첫 1초 대비 %" 를 출력. cgroup 안팎에서 실행해 계산량 비교 |

**핵심 내용**

#### cgroup v2 확인
> 프로세스 묶음에 자원 한도를 거는 커널 기능
- **`mount`** → `cgroup2 on /sys/fs/cgroup`
- **`cgroup.controllers`** → `cpuset cpu io memory hugetlb pids rdma`
- **WSL** → systemd 가 아니라 `0::/non-systemd`, sudo 가 없어 제한 실습은 강의 VM 기준

#### cpu.max — quota / period
> period 마다 quota 만큼만 실행 — 다 쓰면 다음 period 까지 멈춘다
- **quota 를 다 쓰면** → 다음 period 시작까지 멈춤 → `nr_throttled` 증가
- **한도는 cgroup 전체 합계** → quota 20ms / period 100ms 에 busy 2개(각자 다른 CPU) → **각 10%**
- ⚠ `systemd-run … CPUQuota=20%` 의 07_busy 가 "100%" 로 보임 → 기준인 **첫 1초부터 이미 20%** (제한 없는 실행과 비교하면 약 1/5)
- **직접 만들기 (강사 시연)** → `mkdir /sys/fs/cgroup/lab` → `cpu.max` 에 `30000 100000` → `cgroup.procs` 에 PID → `cpu.stat` 확인 → `rmdir`

**기본 사용형태**
```bash
sudo systemd-run --scope -p CPUQuota=20% ./07_busy
sudo mkdir /sys/fs/cgroup/lab
echo "30000 100000" | sudo tee /sys/fs/cgroup/lab/cpu.max     # 30 %
./07_busy & echo $! | sudo tee /sys/fs/cgroup/lab/cgroup.procs
cat /sys/fs/cgroup/lab/cpu.stat                                # nr_throttled
kill %1 && sudo rmdir /sys/fs/cgroup/lab
```

**연관 문제**
<<QA 6>>

### 7교시 · 측정 실습 jitter.c

**키워드:** jitter, 지연(latency), 깨어남 지연, 최소/평균/99%/최대, 꼬리 지연, getopt, qsort, CSV, SCHED_FIFO + CPU 고정, stress-ng 부하, 로그 눈금 히스토그램, VM 지연

**개요**
- 1ms 마다 깨어나야 하는 태스크가 **실제로 얼마나 늦게 깨어나는지**를 5000번 재서 통계를 낸다. 정책(OTHER/FIFO) · 부하 · CPU 고정 조합별로 비교하고, **평균이 아니라 최대·99%** 로 판단하는 법을 익힌다.

**핵심 용어**
- **지연(latency)**: **깨어나야 할 시각보다 늦게 깨어난 시간** (`now - next`).
- **jitter**: 지연이 매번 들쭉날쭉한 정도.
- **99% 값**: 정렬했을 때 하위 99% 경계. 5000개면 인덱스 **4950**. 
- **꼬리(tail)**: 드물게 나오는 큰 지연. 실시간 시스템은 **가장 나쁜 경우**에 실패하므로 꼬리가 중요.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `getopt` | `int getopt(int argc, char *argv[], const char *optstring);` | `-p 1000 -n 5000` 같은 옵션 파싱 (`"p:n:f:c:o:"`, 값은 `optarg`) |
| `./jitter [-p us] [-n 횟수] [-f prio] [-c cpu] [-o csv]` | — | 주기·횟수·FIFO 우선순위·CPU 고정·CSV 출력 |
| `tail -n +2 a.csv \| cut -d, -f2 \| sort -n \| tail -1` | — | CSV 에서 최대값 |
| `awk -F, '{s+=$2} END {print s/NR}'` | — | 평균 |

**예제 코드가 하는 일** ([07_jitter/](Linux_day2/07_jitter/))

| 예제 | 하는 일 |
|---|---|
| `jitter.c` | 옵션으로 CPU 고정·FIFO 를 설정하고, 절대 시각 `clock_nanosleep` 루프에서 매번 `now - next` 를 기록 → `qsort` 로 정렬해 최소·평균·99%·최대를 출력, `-o` 면 회차별 CSV 저장 |

**핵심 내용**

#### jitter 측정 결과
> OTHER 는 부하가 걸리면 최대 지연이 수 ms 로 튄다 (WSL, 3000회)
- **OTHER · 부하 없음** → 최소 51 / 평균 91 / 99% 197 / **최대 383 us**
- **OTHER · 부하 있음** (CPU 수만큼 busy) → 평균 539 / 99% 6102 / **최대 9596 us**
- **`-f 80` 일반 사용자** → `sched_setscheduler (sudo 필요): Operation not permitted`
- **안정 순서** → OTHER·부하 ≫ OTHER·무부하 > FIFO 80·부하 > **FIFO 80 + CPU 고정·부하 (가장 안정)**

#### 왜 튀는가
> OTHER 는 다른 태스크의 실행 조각을 기다리고, VM 은 vCPU 자체가 멈춘다
- **OTHER + 부하** → 다른 태스크의 **최소 실행 조각**이 끝나길 기다림
- **FIFO** → 즉시 선점
- **VM** → 호스트가 vCPU 를 멈출 수 있어 실제 보드보다 크게 튐
- ⚠ 부하 없이 최대 **100ms 이상** → Hyper-V 문제

#### 측정 원칙
> 부하를 건 상태의 최대값으로 판단한다
- **반드시 부하를 건 상태에서도** (실제 운용 중엔 다른 일이 함께 돈다)
- **히스토그램 y 축** → **로그 눈금** (드문 꼬리도 보이게)

**기본 사용형태**
```c
clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
clock_gettime(CLOCK_MONOTONIC, &now);
lat[i] = ns(now) - ns(next);                 // 늦게 깨어난 만큼
qsort(lat, loops, sizeof(long long), cmp_ll);
printf("99%% %.1f 최대 %.1f\n", lat[loops * 99 / 100] / 1000.0, lat[loops - 1] / 1000.0);
```
```bash
./jitter -o other_idle.csv
stress-ng --cpu 2 &  ./jitter -o other_load.csv
sudo ./jitter -f 80 -c 1 -o fifo_cpu1_load.csv
```

**연관 문제**
<<QA 7>>

### 8교시 · 정리 · 임베디드에서 정책 고르기

**키워드:** 정책 선택 기준, RT 5원칙, 우선순위 99 회피, 잠드는 지점, mlockall, nice/renice/taskset/chrt/systemd-run 비교, 센서 FIFO + AI OTHER, Day 2 정리

**개요**
- Day 2 측정 결과를 한 표로 정리하고, 임베디드 장치(센서·모터·AI 추론이 함께 도는)에서 **어떤 작업에 어떤 정책·CPU 를 줄지** 고르는 기준을 세운다.

**핵심 용어**
- **정책 조합**: 정책(OTHER/FIFO/RR) + 우선순위 + CPU 고정 + cgroup 한도. 작업 성격에 맞게 조합한다.
- **잠드는 지점**: RT 태스크가 스스로 CPU 를 내려놓는 곳(`clock_nanosleep`, 이벤트 대기). 없으면 다른 태스크가 굶는다.

**새로 나온 함수/명령** — 도구 비교

| 도구 | 하는 일 |
|---|---|
| `nice` / `renice` | CFS 몫(weight) 조정 — 실행할 때 / 실행 중 |
| `taskset` | **CPU 고정** (affinity) |
| `chrt` | **정책·RT 우선순위** 설정/확인 |
| `systemd-run -p CPUQuota=` | **cgroup** 으로 묶음 CPU 한도 |

**예제 코드가 하는 일** — Day 2 실험 정리

| 실험 | 결과 | 의미 |
|---|---|---|
| `01_ctxsw` | sleep → 자발적, busy → 비자발적 | 잠들면 CPU 를 양보한다 |
| `02_nice_race` | nice 10 → 9.7 % | CFS 는 weight 비율로 나눈다 |
| `03_vruntime` | nice 10 → 9.31배 | vruntime = 실행 시간 × 1024/weight |
| `04_affinity` | 고정하면 그 CPU 에만 | 예측 가능 · 대신 놀고 있는 CPU 를 못 씀 |
| `05_rt_policy` | sudo 필요, 0.95초 제한 | RT 는 강력하고 위험하다 |
| `06_sleep_drift` | 상대 +292ms, 절대 +0.1ms | 주기 태스크는 절대 시각으로 |
| `07_busy` + cgroup | 20% 로 제한 | 묶음 단위 자원 한도 |
| `jitter` | FIFO + 고정이 가장 안정 | 최대 · 99% 를 보라 |

**핵심 내용**

#### RT 정책을 쓰는 기준 5가지
> 늦으면 안 되는 짧은 주기 일만 RT, 나머지는 OTHER
- **① 대상** → 주기가 짧고 늦으면 안 되는 일만 (모터 제어, 센서 샘플링, 오디오 버퍼)
- **② 잠드는 지점** → RT 태스크는 반드시 있어야
- **③ 우선순위** → 필요한 만큼만, **99 는 커널 스레드용**
- **④ 무거운 계산** (AI 추론, 로그 저장) → **OTHER** + **다른 CPU**
- **⑤ 판단** → 부하를 건 상태의 **최대값** (무부하 최대 50us 는 아직 판단 불가)

#### 배치 예
> 센서는 FIFO + CPU 1, AI 는 OTHER + CPU 0
- **센서 샘플링(10ms) + AI 추론** → 센서 FIFO + CPU 1 / AI OTHER + CPU 0
- **CPU 하나 비우기** → 다른 작업을 `taskset` 으로 다른 CPU 에
- **Day 3 연결** → RT 태스크의 **page fault 지연은 `mlockall`** 로 없앤다

**기본 사용형태**
```bash
sudo chrt -f 80 taskset -c 1 ./sensor_loop     # 센서: FIFO + CPU 1
taskset -c 0 nice -n 5 ./ai_infer              # AI: OTHER + CPU 0
chrt -p 1234; taskset -cp 1234                 # 확인
```

**연관 문제**
<<QA 8>>
