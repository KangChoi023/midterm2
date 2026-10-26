## Day 5 — 쓰레드 · 동기화

### 1교시 · 쓰레드 기초

**키워드:** 쓰레드, 프로세스 vs 쓰레드, pthread_create, pthread_join, pthread_t, TID, gettid, LWP, ps -L, /proc/PID/task, 주소 공간 공유, -pthread

**개요**
- Day 1~4 는 프로세스 · 스케줄링 · 메모리 · 파일을 측정했다. Day 5 는 **하나의 프로세스 안에서 여러 일을 동시에** 하는 **쓰레드**. 메모리를 공유해 빠르고 편하지만, 같은 변수를 동시에 고치면 값이 사라진다(2교시 race). 1교시는 쓰레드를 만들고 회수하며 **fork 와의 차이**를 확인한다.

**핵심 용어**
- **쓰레드(thread)**: 한 프로세스 안의 실행 흐름. 리눅스 커널에게는 **주소 공간을 공유하는 태스크**. 한 집(프로세스)에 사는 가족(쓰레드)이 냉장고(전역·힙)를 같이 쓰는 것.
- **TID**: 쓰레드마다의 ID. 같은 프로세스의 쓰레드는 **PID 가 같고 TID 만 다르다**. `ps -L` 의 **LWP** 열.
- **pthread_join**: 그 쓰레드가 **끝나길 기다려 반환값을 받는다** (프로세스의 `wait` 에 해당).

| | 프로세스 (fork) | 쓰레드 (pthread_create) |
|---|---|---|
| 주소 공간 | 따로 (COW) | **공유** — 전역·힙·열린 파일 |
| 스택·레지스터 | 따로 | 쓰레드마다 따로 |
| 식별 | PID | 같은 PID, 쓰레드마다 TID |
| 만드는 비용 | 페이지 테이블 복사 | 스택만 새로 |
| 통신 | 파이프·공유 메모리 | 그냥 변수 (대신 **동기화 필요**) |

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `pthread_create` | `int pthread_create(pthread_t *th, const pthread_attr_t *attr, void *(*fn)(void *), void *arg);` | 쓰레드 생성 — `fn(arg)` 실행 |
| `pthread_join` | `int pthread_join(pthread_t th, void **ret);` | 끝날 때까지 기다리고 반환값 회수 |
| `gettid` | `pid_t gettid(void);` | 내 TID (`_GNU_SOURCE`) |
| `gcc -pthread` | — | pthread 프로그램 빌드 옵션 |
| `ps -L -p PID` / `ls /proc/PID/task` | — | 쓰레드 목록 (쓰레드마다 디렉터리 하나) |

**예제 코드가 하는 일** ([01_threads/](Linux_day5/01_threads/))

| 예제 | 하는 일 |
|---|---|
| `01_threads.c` | 쓰레드 3개가 전역 `g_value += 10` 하고 PID·TID·`&g_value` 를 출력, `malloc` 한 반환값(id×100)을 `join` 으로 회수 → 그 뒤 `fork` 자식이 `g_value += 1000` 해도 부모는 그대로임을 비교 |

**핵심 내용**

#### 쓰레드는 주소 공간을 공유한다
> 같은 프로세스 안의 쓰레드는 PID 가 같고 TID 만 다르며, 같은 변수를 본다

```text
main   : PID=82201 TID=82201
쓰레드 1: PID=82201 TID=82203  &g_value=0x5a8c56b8c01c  g_value=10
쓰레드 2: PID=82201 TID=82204  &g_value=0x5a8c56b8c01c  g_value=20
쓰레드 3: PID=82201 TID=82205  &g_value=0x5a8c56b8c01c  g_value=30
join   : 쓰레드 1 반환값 100 …
쓰레드가 모두 끝난 뒤 g_value = 30 (공유)
fork 자식: g_value = 1030
fork 부모: g_value = 30 (자식이 바꿔도 그대로)
```

- **PID 같음 · TID 다름 · `g_value` 주소 같음** → 하나의 주소 공간
- **fork 자식** → 주소 공간의 **복사본** → 부모 값은 그대로 (Day 3 COW)
- **`ps -L`** → LWP `82201, 82203, 82204, 82205`, `/proc/82201/task` 에도 같은 4개
- ⚠ fork 직전 `fflush(stdout)` 를 빼고 `| cat` → 앞부분이 두 번 찍힘 (Day 1 6교시)

#### 쓰레드도 스케줄 단위
> 커널이 쓰레드를 하나씩 따로 스케줄한다
- **Day 2 의 nice · affinity · FIFO** → 쓰레드마다 줄 수 있다 (6교시)

**기본 사용형태**
```c
static void *worker(void *arg) {
    int id = *(int *)arg;
    int *ret = malloc(sizeof *ret); *ret = id * 100;
    return ret;
}
pthread_t th; int id = 1; void *r;
pthread_create(&th, NULL, worker, &id);
pthread_join(th, &r);                 // 끝날 때까지 기다리고 반환값
printf("%d\n", *(int *)r); free(r);
```
```bash
gcc -Wall -pthread 01_threads.c -o 01_threads
ps -L -p $(pgrep 01_threads)
```

**연관 문제**
<<QA 1>>

### 2교시 · race condition

**키워드:** race condition, 경쟁 상태, 임계 구역 critical section, counter++, 읽기-더하기-쓰기, mov add mov, objdump -d, volatile, 선점, lock 접두어, 02_race

**개요**
- `counter++` 는 한 번에 일어나지 않는다 — 기계어로 **읽기 → 더하기 → 쓰기** 세 단계다. 두 쓰레드가 그 사이에 끼어들면 한쪽의 더하기가 사라진다. 이 **race condition** 을 재현하고, CPU 가 1개여도 생기는 이유를 본다.

**핵심 용어**
- **race condition**: 결과가 **실행 순서(타이밍)에 따라 달라지는** 상황. 두 사람이 같은 통장 잔고를 동시에 읽고 각자 입금해 쓰면 한 번이 사라진다.
- **임계 구역(critical section)**: 공유 데이터를 고치는 코드 구간. 한 번에 한 쓰레드만 들어가야 한다.
- **원자적(atomic)**: 중간에 끼어들 수 없이 한 번에 일어남. `counter++` 는 원자적이지 **않다**.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `objdump -d --no-show-raw-insn ./02_race \| grep -A4 "<add_none>:"` | `counter++` 의 기계어 보기 |
| `./02_race 쓰레드수 반복수 none\|mutex\|spin\|atomic` | 보호 방식별 결과·시간 |

**예제 코드가 하는 일** ([02_race/](Linux_day5/02_race/))

| 예제 | 하는 일 |
|---|---|
| `02_race.c` | 쓰레드 n 개가 `volatile long counter` 를 각각 iters 번 증가. 모드 `none`(보호 없음) / `mutex` / `atomic`(`__atomic_fetch_add`) / `spin`(spinlock) 중 하나로 실행해 **기대값 vs 실제값, 잃어버린 수, 시간** 출력 |

**핵심 내용**

#### race condition
> 읽기와 쓰기 사이에 다른 쓰레드가 끼어들면 증가분을 잃는다
- **순서** (counter = 5) → A 읽기(5) → B 읽기(5) → A 더하기·쓰기(6) → B 더하기·쓰기(6) → **두 번 더했는데 6**
- **`./02_race 2 1000000 none` 세 번** → **1093524 / 1097716 / 1070300** (기대 2000000) — 실행할 때마다 다름

```text
<add_none>:
  mov    0x2dc9(%rip),%rax     # 읽기
  add    $0x1,%rax             # 더하기
  mov    %rax,0x2dbe(%rip)     # 쓰기
```

#### CPU 1개여도 race
> 선점은 읽기와 쓰기 사이에서도 일어난다
- **원인** → A 가 읽은 직후 선점 → B 가 한 타임 슬라이스 동안 더한 수백만 번을 A 의 쓰기 한 번이 덮어씀
- **확인** → `taskset -c 0 ./02_race 2 20000000 none` → 기대 40000000, 실제 **26420002**

#### volatile 은 해결책이 아니다
> 매번 메모리를 쓰게 할 뿐, 원자성을 주지 않는다
- **이 예제의 volatile** → race 가 **잘 드러나게** 붙인 것
- ⚠ 최적화로 `addq $1, mem` 한 명령이 돼도 → `lock` 접두어가 없으면 여러 CPU 사이에서 원자적이지 않음
- **막는 방법** → **mutex**, **atomic**, **spinlock** (3·5교시)

**기본 사용형태**
```c
static volatile long counter = 0;
static void *w_none(void *a) { for (long i = 0; i < iters; i++) counter++; return NULL; }  // race!
```
```bash
./02_race 2 1000000 none        # 매번 다른 결과
```

**연관 문제**
<<QA 2>>

### 3교시 · mutex 와 교착

**키워드:** mutex, 상호 배제, pthread_mutex_lock, pthread_mutex_unlock, PTHREAD_MUTEX_INITIALIZER, 잠금 비용, 교착 deadlock, 잠금 순서, trylock, alarm, SIGALRM, gdb -p thread apply all bt

**개요**
- **mutex** 로 임계 구역을 감싸 race 를 고치고, 그 대가(속도)를 잰다. 그리고 mutex 두 개를 **반대 순서**로 잡으면 생기는 **교착(deadlock)** 을 재현하고 **잠금 순서 통일**로 해결한다.

**핵심 용어**
- **mutex(상호 배제)**: 잠금. 먼저 `lock` 한 쓰레드만 임계 구역에 들어가고, 나머지는 `unlock` 될 때까지 **잠든다**. 화장실 열쇠 하나.
- **교착(deadlock)**: 두 쓰레드가 **서로 상대가 가진 잠금을 기다리며 영원히 멈춤**. 좁은 길에서 마주 본 두 차.
- **잠금 순서**: 모든 쓰레드가 여러 mutex 를 **같은 순서로** 잡으면 교착이 생기지 않는다.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `pthread_mutex_lock` | `int pthread_mutex_lock(pthread_mutex_t *m);` | 잠금 — 다른 쓰레드가 가졌으면 기다림 |
| `pthread_mutex_unlock` | `int pthread_mutex_unlock(pthread_mutex_t *m);` | 해제 — 기다리던 쓰레드 하나 진행 |
| `pthread_mutex_trylock` | `int pthread_mutex_trylock(pthread_mutex_t *m);` | 못 잡으면 기다리지 않고 바로 실패 |
| `PTHREAD_MUTEX_INITIALIZER` | `static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;` | 정적 초기화 |
| `alarm` | `unsigned alarm(unsigned sec);` | sec 초 뒤 SIGALRM (감시자) |
| `sudo gdb -p PID -batch -ex "thread apply all bt"` | — | 멈춘 쓰레드들이 **어디서** 기다리는지 |

**예제 코드가 하는 일** ([03_deadlock/](Linux_day5/03_deadlock/))

| 예제 | 하는 일 |
|---|---|
| `02_race.c mutex` | 2교시 예제를 mutex 모드로 — 정확도와 시간 비교 |
| `03_deadlock.c` | t1 은 A→B, t2 는 B→A 순서로 잠금(사이에 0.1초 쉬어 엇갈리게). `alarm(3)` 감시자가 3초 안에 안 끝나면 교착 메시지 후 `_exit(1)`. 인자 `fixed` 면 t2 도 A→B |

**핵심 내용**

#### mutex 의 비용
> 정확해지지만 느려진다 — 임계 구역은 꼭 필요한 줄만
- **`./02_race 2 1000000 mutex`** → 잃어버림 **0**, **113.9 ms** (none 5.5 ms)
- **비용** → lock·unlock + 경쟁하면 잠들었다 깨는 비용 (Day 2 컨텍스트 스위치)
- **강의** → 5.3 → 39 ms (**8배**)
- **규칙** → 오래 걸리는 계산 · 파일 I/O 는 **잠금 밖**으로

#### 교착 (deadlock)
> 두 쓰레드가 서로가 가진 잠금을 영원히 기다린다
- **순서** → main 이 `alarm(3)` 후 t1·t2 생성 → t1 은 A, t2 는 B 를 잡음 → t1 은 B, t2 는 A 를 기다림 → 3초 뒤 SIGALRM → 감시자 메시지
- **`./03_deadlock`** → `감시자: 3초 동안 진행 없음 — 교착(deadlock)…`, 종료 코드 1
- **`./03_deadlock fixed`** → `t1: A, B 모두 잡음` … `정상 종료 (잠금 순서를 A→B 로 통일)`
- ⚠ 파이프로 받으면 `t1: A 잡음` 같은 **stdout 줄이 안 보임** → 버퍼 내용이 핸들러의 `_exit` 로 버려짐 (터미널은 줄 버퍼링이라 보임)

#### mutex 잘 쓰는 법
> 순서 통일 · 짧게 · 잡은 쓰레드가 푼다
- **여러 mutex** → 항상 **같은 순서**로
- **임계 구역** → 짧게
- **unlock** → lock 한 쓰레드가

**기본 사용형태**
```c
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_lock(&mtx);
counter++;                  // 임계 구역 — 짧게!
pthread_mutex_unlock(&mtx);
// 교착 방지: 모든 쓰레드가 lock(&A) → lock(&B) 순서
```

**연관 문제**
<<QA 3>>

### 4교시 · 조건 변수

**키워드:** 조건 변수 condition variable, pthread_cond_wait, pthread_cond_signal, pthread_cond_broadcast, 생산자 소비자, 링 버퍼, not_full, not_empty, while 재확인, spurious wakeup, 바쁜 대기 busy waiting

**개요**
- 센서 쓰레드가 값을 만들고(생산) 처리 쓰레드가 꺼내 쓰는(소비) 구조에서, 버퍼가 비었으면 소비자가, 가득 찼으면 생산자가 **기다려야** 한다. 반복 확인(바쁜 대기)은 CPU 를 낭비하므로 **조건 변수로 잠든다**. 링 버퍼는 C Day4 의 그것.

**핵심 용어**
- **생산자·소비자(producer-consumer)**: 만드는 쪽과 쓰는 쪽 사이에 **버퍼(큐)** 를 두는 구조. 빵집 진열대.
- **조건 변수(condition variable)**: "~가 될 때까지" **잠들어 기다리는** 도구. 항상 **mutex 와 짝**.
- **바쁜 대기(busy waiting)**: 조건을 계속 반복 확인하며 CPU 를 씀. 피해야 할 것.
- **spurious wakeup**: 이유 없이 깨는 경우(표준이 허용) → **`while` 로 다시 확인**.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `pthread_cond_wait` | `int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m);` | **m 을 풀고 잠든다** → 깨면 **m 을 다시 잡고** 돌아온다 |
| `pthread_cond_signal` | `int pthread_cond_signal(pthread_cond_t *c);` | 기다리는 쓰레드 **하나** 깨움 |
| `pthread_cond_broadcast` | `int pthread_cond_broadcast(pthread_cond_t *c);` | **모두** 깨움 (종료 알림 등) |
| `PTHREAD_COND_INITIALIZER` | — | 정적 초기화 |

**예제 코드가 하는 일** ([04_prodcons/](Linux_day5/04_prodcons/))

| 예제 | 하는 일 |
|---|---|
| `04_prodcons.c` | 8칸 링 버퍼에 생산자가 1~40 을 넣고 소비자가 꺼내 합계(820)를 낸다. 가득 차면 생산자가 `not_full` 에서, 비면 소비자가 `not_empty` 에서 `while` + `cond_wait` 로 잠들고 기다린 횟수를 센다. 기본은 소비자가 느림, `fastcons` 면 생산자가 느림 |

**핵심 내용**

#### 조건 변수로 기다리기
> 조건이 될 때까지 mutex 를 풀고 잠든다
- **소비자 순서** → `lock(&m)` → `while (count == 0) cond_wait(&not_empty, &m)` → 값 꺼내기 · `count--` → `signal(&not_full)` → `unlock(&m)`
- **`cond_wait`** → mutex 를 **잡은 채로 잠들지 않는다** (풀고 잠든다)
- **조건을 바꾼 쪽이** → `signal` / `broadcast`
- ⚠ **`if` 가 아니라 `while`** → 깨어난 사이 다른 쓰레드가 먼저 가져갔을 수 있고, spurious wakeup 도 있다

#### 누가 기다리나
> 느린 쪽이 아니라 빠른 쪽이 기다린다 — 기다리는 동안 CPU 0%
- **기본 (소비자 느림)** → 합계 **820**, 생산자 '가득 참' 대기 **32**, 소비자 '비어 있음' 대기 1
- **`fastcons` (생산자 느림)** → 합계 820, 생산자 대기 0, 소비자 대기 **40**

**기본 사용형태**
```c
pthread_mutex_lock(&m);
while (count == CAP)                       // if 아님!
    pthread_cond_wait(&not_full, &m);      // m 풀고 잠듦 → 깨면 다시 잡음
buf[tail] = v; tail = (tail + 1) % CAP; count++;
pthread_cond_signal(&not_empty);           // 소비자 하나 깨움
pthread_mutex_unlock(&m);
```

**연관 문제**
<<QA 4>>

### 5교시 · 세마포어와 atomic

**키워드:** 세마포어 semaphore, sem_init, sem_wait, sem_post, sem_destroy, 동시 실행 수 제한, atomic, __atomic_fetch_add, stdatomic.h, lock add, spinlock, pthread_spin_lock, 동기화 도구 비교

**개요**
- **세마포어**로 "동시에 N 개까지"를 표현하고(가속기 2개를 쓰레드 5개가 나눠 쓰기), 정수 하나를 더하는 정도면 잠금 없이 **atomic** 명령으로 충분함을 본다. mutex · spinlock · atomic 의 비용을 비교하고 상황별 도구를 정리한다.

**핵심 용어**
- **세마포어(semaphore)**: **정수 카운터**(남은 자리 수). `sem_wait` 는 1 빼되 **0 이면 기다림**, `sem_post` 는 1 더하며 기다리던 쓰레드를 깨움. 주차장 남은 칸 전광판. 초기값 1 이면 mutex 처럼 쓸 수 있다.
- **atomic 연산**: CPU 의 원자적 명령(`lock add`)으로 **잠금 없이** 변수 하나를 안전하게 고침. **변수 하나의 단순 연산만** — 여러 변수를 함께 고치는 구간은 못 지킨다.
- **spinlock**: 경쟁하면 잠들지 않고 **계속 돌며** 기다림. **아주 짧고 잠들 수 없는 구간**(커널 안)용.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `sem_init` | `int sem_init(sem_t *s, int pshared, unsigned value);` | `pshared` 0 = 쓰레드끼리, `value` = 초기 자리 수 |
| `sem_wait` / `sem_post` | `int sem_wait(sem_t *s);` / `int sem_post(sem_t *s);` | 자리 얻기(0 이면 대기) / 자리 반납 |
| `sem_destroy` | `int sem_destroy(sem_t *s);` | 정리 |
| `__atomic_fetch_add` | `__atomic_fetch_add(&x, 1, __ATOMIC_RELAXED)` | 원자적 덧셈 (C11 `<stdatomic.h>` 의 `atomic_fetch_add` 도) |
| `pthread_spin_lock` / `unlock` | `int pthread_spin_lock(pthread_spinlock_t *s);` | spinlock (`pthread_spin_init` 으로 초기화) |

**예제 코드가 하는 일** ([05_sem/](Linux_day5/05_sem/))

| 예제 | 하는 일 |
|---|---|
| `05_sem.c` | 초기값 2 세마포어로 쓰레드 5개가 "가속기"를 0.2초씩 쓰게 하고, mutex 로 보호한 `active` 로 동시 실행 수와 최대값을 출력 |
| `02_race.c` (재사용) | `none/mutex/spin/atomic` 시간 비교 |

**핵심 내용**

#### 세마포어
> 남은 자리 수 — 0 이면 `sem_wait` 에서 잠든다
- **`05_sem`** → 작업 1·2 시작(동시 2) → 1·2 끝 → 3·4 시작 → … → **최대 동시 실행 = 2**
- **작업 3** → 자리가 0 이라 `sem_wait` 에서 잠들었다가 누군가 `sem_post` 하면 진행

#### 동기화 방식 비교
> 모두 정확하지만 비용과 쓰임이 다르다 (2 × 100만)

| 방식 | 결과 | 시간 | 특징 |
|---|---|---|---|
| none | 틀림 | 5.5 ms | 보호 없음 |
| mutex | 정확 | 113.9 ms | 경쟁하면 잠듦 — 임계 구역이 길 때 |
| spinlock | 정확 | 93.3 ms | 계속 돌며 기다림 — 아주 짧은 구간, 커널 안 |
| atomic | 정확 | **37.8 ms** | 변수 하나의 단순 연산만 |

- **강의 VM** → 5 / 39 / 36 / 13 ms
- **atomic 이 빠른 이유** → 잠금·해제 없이 **명령 하나로**

#### 도구 고르기
> 무엇을 보호·대기하느냐로 고른다

| 도구 | 언제 |
|---|---|
| mutex | 여러 변수를 함께 고치는 구간 |
| 조건 변수 | "~가 될 때까지" 기다림 (mutex 와 짝) |
| 세마포어 | 개수 제한, 신호 횟수 세기 |
| atomic | 카운터·플래그 하나 |

**기본 사용형태**
```c
static sem_t slots;
sem_init(&slots, 0, 2);          // 자리 2개
sem_wait(&slots);                // 0 이면 여기서 기다림
use_accelerator();
sem_post(&slots);                // 반납
__atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);   // 잠금 없는 덧셈
```

**연관 문제**
<<QA 5>>

### 6교시 · 쓰레드와 스케줄링 · 캐시

**키워드:** 쓰레드별 정책, pthread_setaffinity_np, pthread_setschedparam, ps -L -o tid,cls,rtprio,psr, 캐시 줄 cache line 64 바이트, false sharing, 패딩, __attribute__((aligned(64)))

**개요**
- 커널은 쓰레드를 따로 스케줄하므로 **한 프로세스 안에서도 쓰레드마다** 정책·CPU 를 달리할 수 있다(Day 2 8교시 원칙). 또 서로 다른 변수인데도 **같은 64바이트 캐시 줄**에 있으면 두 CPU 가 서로 방해하는 **false sharing** 을 측정하고 패딩으로 해결한다.

**핵심 용어**
- **캐시 줄(cache line)**: CPU 캐시가 메모리를 주고받는 단위, x86 은 **64 바이트**.
- **false sharing**: 두 쓰레드가 **서로 다른 변수**를 고쳐도 같은 캐시 줄에 있으면, 그 줄이 두 CPU 사이를 계속 오가며(무효화) 느려지는 현상. 한 공책의 다른 페이지를 쓰는데 공책을 계속 주고받는 것.
- **패딩 / 정렬**: 변수 사이에 빈 바이트를 넣거나 `aligned(64)` 로 **캐시 줄을 나눔**. (C Day3 구조체 패딩과 같은 원리를 성능에 씀)

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `pthread_setaffinity_np` | `int pthread_setaffinity_np(pthread_t th, size_t sz, const cpu_set_t *set);` | **그 쓰레드의** CPU 고정 |
| `pthread_setschedparam` | `int pthread_setschedparam(pthread_t th, int policy, const struct sched_param *sp);` | **그 쓰레드의** 정책·우선순위 (FIFO 는 sudo) |
| `pthread_self` | `pthread_t pthread_self(void);` | 나 자신 |
| `ps -L -o tid,cls,rtprio,psr,comm -p PID` | — | 쓰레드별 TID · 정책(CLS: TS/FF/RR) · RT 우선순위 · **실행 CPU(PSR)** |
| `__attribute__((aligned(64)))` | — | 64바이트 경계에 놓기 |

**예제 코드가 하는 일** ([06_false_sharing/](Linux_day5/06_false_sharing/))

| 예제 | 하는 일 |
|---|---|
| `06_false_sharing.c` | 두 쓰레드를 CPU 0·1 에 고정해 각자 2억 번 증가. ① 같은 구조체 안 붙은 두 변수(거리 8B) ② 패딩으로 64B 떨어뜨린 두 변수 — 시간 비교 |

**핵심 내용**

#### false sharing
> 서로 다른 변수라도 같은 캐시 라인(64B)에 있으면 CPU 사이를 오가며 느려진다

```text
&T.a=…90d0 &T.b=…90d8 (거리 8 B)
&P.a=…9080 &P.b=…90c0 (거리 64 B)
같은 캐시 라인 :    446.9 ms
64B 떨어뜨림   :     50.8 ms  (8.8 배 빠름)
```

- **강의 VM** → 905 → 252 ms (3.6배), 2~6배가 흔함
- **순서** → CPU 0 이 a 를 쓰려고 줄을 가져옴 → CPU 1 이 b 를 쓰려 하자 CPU 0 의 줄이 무효화되고 CPU 1 로 → CPU 0 이 다시 a 를 쓰면 되돌아옴 → 반복
- **CPU 가 1개면** → 번갈아 실행돼 차이가 거의 없다
- **피하는 법** → 쓰레드별 카운터를 따로 두고 **마지막에 합치기** / 멤버 사이 **64B 패딩** / **`__attribute__((aligned(64)))`**

#### 쓰레드별 정책
> 센서 쓰레드만 실시간으로
- **센서 쓰레드** → `pthread_setschedparam(…, SCHED_FIFO, …)` + `pthread_setaffinity_np(CPU 1)`
- **나머지** → OTHER (7교시 edge_pipeline)

**기본 사용형태**
```c
struct apart { volatile long a; char pad[64 - sizeof(long)]; volatile long b; };   // 다른 캐시 줄
cpu_set_t set; CPU_ZERO(&set); CPU_SET(1, &set);
pthread_setaffinity_np(pthread_self(), sizeof set, &set);
struct sched_param sp = { .sched_priority = 80 };
pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);     // sudo
```
```bash
ps -L -o tid,cls,rtprio,psr,comm -p $(pgrep edge_pipeline)
```

**연관 문제**
<<QA 6>>

### 7교시 · 종합 실습 edge_pipeline

**키워드:** edge_pipeline, 센서 처리 기록 3단 파이프라인, 링 버퍼 64칸, 10ms 절대 주기, 이동 평균, 임계치 알람, 1초 통계, fsync, mlockall, SCHED_FIFO, CPU 고정, max_lat_us, dropped, -r 옵션

**개요**
- 5일의 기법을 **쓰레드 3개 파이프라인**으로 합친다. 센서(10ms 주기) → 링 버퍼 → 처리(이동 평균·알람) → 1초 통계 → 기록(CSV + fsync). 부하·정책 조합별 지연을 측정한다.

**핵심 용어**
- **파이프라인**: 단계마다 쓰레드를 나누고 **큐로 이음** → 단계마다 정책·CPU 를 달리할 수 있다. 공장 컨베이어 벨트.
- **이동 평균**: 최근 10개 값의 평균 — 잡음 제거.
- **dropped**: 링 버퍼가 가득 차 **가장 오래된 값을 버린** 수 (센서는 기다리지 않는다). 0 이어야 정상.
- **max_lat_us**: 센서가 늦게 깨어난 최대값 (Day 2 jitter).

**새로 나온 함수/명령**

| 구성 | 쓰인 기법 | 날 |
|---|---|---|
| 센서 쓰레드: 10ms 절대 주기 · FIFO · CPU 고정 | `clock_nanosleep(TIMER_ABSTIME)`, `pthread_setschedparam`, `pthread_setaffinity_np` | Day 2 · 5 |
| 링 버퍼 64칸 (mutex + 조건 변수) | `cond_wait` / `cond_signal` | Day 5 |
| 처리 쓰레드: 계산은 **잠금 밖** | — | Day 5 |
| 기록 쓰레드: 1초마다 CSV + `fsync` | `fflush` + `fsync(fileno(fp))` | Day 4 |
| main: `mlockall` · SIGINT 정상 종료 · `getopt` | — | Day 3 · 1 |

| 명령 | 용도 |
|---|---|
| `./edge_pipeline -d 10 [-o out.csv] [-t 0.8]` | 일반 정책, 10초 |
| `sudo ./edge_pipeline -d 10 -r` | 센서 FIFO 80 + CPU 1 + mlockall |

**예제 코드가 하는 일** ([07_edge/](Linux_day5/07_edge/))

| 예제 | 하는 일 |
|---|---|
| `edge_pipeline.c` | 센서가 10ms 마다 사인파+잡음 값을 링 버퍼에 넣고(가득 차면 오래된 값 버림·dropped++) signal → 처리가 깨어나 꺼내고 **잠금을 푼 뒤** 이동 평균·알람 계산 → 기록이 1초마다 통계를 CSV 한 줄로 쓰고 fsync → Ctrl+C 또는 `-d` 초 뒤 정상 종료 |

**핵심 내용**

#### 파이프라인 흐름
> 센서 → 링 버퍼 → 처리 → 기록, 단계마다 쓰레드
- **값 하나** → 센서가 10ms 눈금에 깨어나 값 생성 → 링 버퍼에 넣고 signal → 처리 쓰레드가 깨어나 꺼냄 → 이동 평균·통계 → 기록 쓰레드가 1초마다 CSV
- **처리 쓰레드** → **잠금을 풀고 나서** 계산 → 그동안 센서가 버퍼에 넣을 수 있음
- **기록** → **1초에 한 번만 fsync** (fsync 는 비쌈, Day 4 3교시)

#### 실행 결과 읽기
> samples ≈ 100, dropped 0 이면 정상

```text
 초  samples      avg  max_lat_us  alarms  dropped
   1      100    0.630       333.1      38        0
   2       99   -0.627       241.1       0        0
   3      100    0.644       208.7      41        0
```

- **samples ≈ 100** → 센서 주기 10ms
- **alarms** → 1초씩 번갈아 (사인파 2초 주기)
- **dropped 0** → 정상
- **부하 (CPU 수만큼)** → max_lat_us **3600~5400 us**
- **`-r` 일반 사용자** → `센서: SCHED_FIFO 실패 (sudo 필요) — 일반 정책으로 계속`
- **부하 상태 `sudo … -r`** → max_lat_us 가 작게 유지 (센서가 FIFO · CPU 고정 · mlockall 로 먼저 돈다)

**기본 사용형태**
```c
// 처리 쓰레드
pthread_mutex_lock(&rmx);
while (count == 0 && !g_stop) pthread_cond_wait(&not_empty, &rmx);
double v = ring[head]; head = (head + 1) % CAP; count--;
pthread_mutex_unlock(&rmx);            // 계산은 잠금 밖
// 기록 쓰레드
fprintf(fp, "%d,%ld,%.3f,%.1f,%ld,%ld\n", sec, n, avg, lat / 1000.0, a, d);
fflush(fp); fsync(fileno(fp));         // 1초에 한 번만
```
```bash
stress-ng --cpu 2 &
./edge_pipeline -d 10 -o other_load.csv
sudo ./edge_pipeline -d 10 -r -o rt_load.csv
```

**연관 문제**
<<QA 7>>

### 8교시 · 결과 발표 · 5일 정리

**키워드:** 정상 종료, g_stop, volatile sig_atomic_t, broadcast 로 깨우기, pthread_join, 5일 결과물, myshell jitter memlab sysmon mysysfs edge_pipeline, 온디바이스 AI 구조, Embedded I/O 예고

**개요**
- 조별로 edge_pipeline 측정(부하 있음/없음 × OTHER/-r) 결과를 해석해 발표하고, 5일 동안 만든 프로그램을 **온디바이스 AI 장치 구조**에 연결한다. 시그널과 join 으로 하는 **정상 종료**를 정리한다.

**핵심 용어**
- **정상 종료(graceful shutdown)**: 데이터 손실 없이 모든 쓰레드가 정리하고 끝남. 플래그 + 깨우기 + join.
- **`static volatile sig_atomic_t g_stop`**: 시그널 핸들러와 여러 쓰레드가 함께 보는 종료 플래그 (Day 1 7교시).

**새로 나온 함수/명령** — 새 함수 없음 (`signal`, `pthread_cond_broadcast`, `pthread_join` 재사용)

**예제 코드가 하는 일** — 5일 결과물

| 날 | 주제 | 결과물 | 온디바이스 AI 에서 |
|---|---|---|---|
| Day 1 | 시스템 콜 · 프로세스 | `myshell.c` | 추론 서버를 fork · exec 로 띄우고 감시 |
| Day 2 | 스케줄링 | `jitter.c` | 센서·제어는 FIFO + 고정, 추론은 OTHER |
| Day 3 | 메모리 | `memlab.c` | 모델은 mmap, 실시간 쓰레드는 mlockall |
| Day 4 | 파일 · sysfs | `sysmon.c`, `mysysfs.ko` | 장치 값은 sysfs, 로그는 필요한 곳만 fsync |
| Day 5 | 쓰레드 · 동기화 | `edge_pipeline.c` | 수집·추론·기록을 쓰레드로 나누고 큐로 잇기 |

**핵심 내용**

#### 정상 종료 순서
> Ctrl+C → 플래그 → broadcast 로 깨우기 → join 으로 기다리기
- **순서** → SIGINT 핸들러가 `g_stop = 1` → 센서 쓰레드가 루프를 빠져나와 **`broadcast`** → `cond_wait` 에서 자던 처리 쓰레드가 깨어 종료 → 기록 쓰레드가 마지막 줄을 쓰고 닫음 → main 이 **`pthread_join` 세 번** → 종료
- ⚠ main 이 join 없이 return → 프로세스 종료로 쓰레드도 함께 끝남 → 끝까지 실행 안 됨
- ⚠ broadcast 로 안 깨우면 → 처리 쓰레드가 영원히 잠듦

#### 구조의 장점과 다음 과목
> 큐로 잇는 쓰레드 구조는 단계마다 정책 · CPU 를 달리할 수 있다
- **발표 질문** → 부하 시 OTHER 와 `-r` 의 max_lat_us 차이? dropped 가 생긴 조합과 이유? 실제 장치에서 센서 대신 무엇을 읽을까 (sysfs 온도 · ADC, Day 4)?
- **Embedded I/O** → 같은 Ubuntu 지식 위에서 GPIO · I2C · SPI 장치를 sysfs · ioctl 로
- **Jetson Orin Nano** → edge_pipeline 을 실제 센서 · 추론 쓰레드로

**기본 사용형태**
```c
static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int s) { (void)s; g_stop = 1; }
signal(SIGINT, on_sigint);
/* 센서 루프 종료 시 */ pthread_cond_broadcast(&not_empty);
pthread_join(ts, NULL); pthread_join(tp, NULL); pthread_join(tl, NULL);
```

**연관 문제**
<<QA 8>>
