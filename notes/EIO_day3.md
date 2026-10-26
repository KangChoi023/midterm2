## Day 3 — PWM · 모터 · 타이밍

### 1교시 · 소프트웨어 PWM 과 jitter

**키워드:** PWM, 주기 period, 주파수, 듀티 duty cycle, 소프트웨어 PWM, pthread, clock_nanosleep TIMER_ABSTIME, 절대 시각, 누적 오차, jitter, 깨어난 지연 히스토그램, 부하 yes, 스케줄링 지연, 페이지 폴트, 01_swpwm

**개요**
- GPIO 는 0V 아니면 3.3V 뿐이지만, 아주 빠르게 켰다 껐다 하면 **켜진 비율(듀티)** 만큼 LED 가 밝아 보이고 모터가 느려진다. 1~2교시는 이 파형을 **쓰레드(소프트웨어)** 로 만들고, CPU 부하가 걸리면 깨어나는 시각이 흔들리는 **jitter** 를 측정한다. (Linux Day2 5·7교시 절대 sleep·jitter 의 GPIO 판)

**핵심 용어**
- **주기(period)**: 한 번 켜졌다 꺼지기까지. **주파수** = 1 / 주기 (Hz). **듀티**: 주기 중 켜진 시간(또는 %). 주기 5ms 듀티 20% → **200Hz, 켜짐 1ms**.
- **절대 시각 sleep**: `next = 이전 목표 + 주기` 로 잡고 `clock_nanosleep(TIMER_ABSTIME)` → 하나가 늦어도 다음 목표는 그대로, **누적 오차 없음**. (`usleep(on); usleep(off);` 상대 sleep 은 깨어나는 지연이 매 주기 더해져 전체가 늦어짐)
- **jitter**: 목표 시각보다 **얼마나 늦게 깨어났나**의 흔들림. PWM 품질은 평균보다 **최대값** — 한 번의 큰 지연이 눈에 보이는 깜빡임이 된다.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL)` | 절대 시각까지 잠 (Linux Day2) |
| `pthread_create` / `pthread_mutex_lock` | PWM 쓰레드 / 히스토그램 보호 (Linux Day5) |
| `yes > /dev/null &` ×4 / `killall yes` | 4코어 부하 / 정리 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `01_swpwm.c` | PWM 쓰레드가 5ms 주기로 LED 를 켜고(→ 끌 시각 = 시작 + on_ns 까지 절대 sleep → 끄기 → 다음 주기까지 절대 sleep) 깨어날 때마다 늦은 정도를 히스토그램(<10µs … ≥1ms)에 기록, 메인은 2초마다 출력. 인자: 듀티 % / `-b` 숨쉬기 / `-f` SCHED_FIFO |

**핵심 내용**

#### 소프트웨어 PWM 한 주기
> 켜고 → 끌 시각까지 절대 sleep → 끄고 → 다음 시작까지 절대 sleep
- **순서** → LED 켜기 → 끌 시각 = `next + on_ns` → 그 시각까지 절대 sleep → LED 끄기 → `next += PERIOD` → 다음 시작까지 절대 sleep
- **`TIMER_ABSTIME`** → 누적 오차는 없애지만 **깨어나는 지연을 0 으로 만들지는 못한다**

#### 부하와 jitter
> 일반 쓰레드는 부하가 걸리면 늦게 깨어나 LED 가 깜빡인다
- **부하 없음** → 평균 **62µs** · 최대 **148µs**, 대부분 <100µs
- **yes ×4 부하** → 평균 **1840µs** · 최대 **11962µs**, **≥1ms 가 426회** → 눈에 띄게 깜빡임

| jitter 원인 | 설명 |
|---|---|
| 스케줄링 지연 | SCHED_OTHER 쓰레드는 다른 프로세스와 CPU 를 나눠 씀 |
| 타이머 해상도·인터럽트 | 깨우는 신호 자체가 수십 µs 늦을 수 있음 |
| 페이지 폴트 | 처음 쓰는 메모리에서 커널이 페이지를 준비하느라 멈춤 (Linux Day3) |

**기본 사용형태**
```c
struct timespec next; clock_gettime(CLOCK_MONOTONIC, &next);
while (!g_stop) {
    long on_ns = PERIOD_NS / 100 * g_duty;
    struct timespec t_off = next; ts_add(&t_off, on_ns);
    set(1); clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t_off, NULL);
    set(0); ts_add(&next, PERIOD_NS);                 // 이전 목표 + 주기
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
}
```

**연관 문제**
<<QA 1>>

### 2교시 · RGB LED 와 실시간 우선순위

**키워드:** RGB LED 공통 음극, 3채널 소프트웨어 PWM, 쓰레드 하나, 끄는 순서 정렬, set_values, hue_to_rgb, SCHED_FIFO, pthread_attr_setschedpolicy, setschedparam, PTHREAD_EXPLICIT_SCHED, mlockall, setcap cap_sys_nice cap_ipc_lock, EPERM, 02_rgb

**개요**
- 쓰레드 **하나**로 RGB 세 채널 PWM 을 만들어 색을 섞고, 부하 중 깜빡임을 줄이는 **SCHED_FIFO + mlockall** 의 효과와 위험을 본다. sudo 대신 `setcap` 으로 **실행 파일 하나에 필요한 권한만** 준다. (Linux Day2 4교시 실시간 정책, Day3 6교시 mlockall)

**핵심 용어**
- **RGB LED(공통 음극)**: 가장 긴 다리 = 공통(−) → GND, R·G·B 에 각각 220Ω → GPIO17·27·22.
- **한 쓰레드로 세 채널**: 주기가 같으니 주기 시작에 켤 채널을 모두 켜고(`set_values` 한 번), **꺼질 시각이 빠른 채널부터** 그 시각까지 잠들었다 끈다 → 주기당 깨어남 **최대 4번**. R=80, G=30, B=10 이면 끄는 순서 **B → G → R**.
- **SCHED_FIFO 1~99**: 실시간 — 깨어나면 **즉시**, 스스로 양보할 때까지 계속. SCHED_OTHER 보다 **먼저** CPU 를 받는다.
- **mlockall(MCL_CURRENT | MCL_FUTURE)**: 메모리를 RAM 에 고정 → **페이지 폴트로 멈추지 않음**.
- **setcap**: 실행 파일에 **특정 권한(capability)만** 부여. `cap_sys_nice`(실시간 우선순위) + `cap_ipc_lock`(mlockall). **다시 빌드하면 사라진다**(파일이 바뀌므로).

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `pthread_attr_setschedpolicy(&at, SCHED_FIFO)` | 쓰레드 정책 FIFO |
| `pthread_attr_setschedparam(&at, &sp)` | `sp.sched_priority = 50` |
| `pthread_attr_setinheritsched(&at, PTHREAD_EXPLICIT_SCHED)` | **이걸 빼면 위 설정이 무시되고 부모 정책을 물려받는다** |
| `mlockall(MCL_CURRENT \| MCL_FUTURE)` | 메모리 고정 |
| `sudo setcap cap_sys_nice,cap_ipc_lock+ep ./02_rgb` | 권한만 주기 (일반 사용자 `-f` 는 `Operation not permitted`) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `02_rgb.c` | 10ms 주기 RGB PWM 쓰레드(켤 채널 한꺼번에 켜기 → 꺼질 순서로 정렬해 차례로 끄기), 메인은 `hue_to_rgb` 로 무지개 페이드 또는 인자 `R G B`(0~255) 고정, `-f` 면 FIFO 50 + mlockall |
| 2-4 비교 실험 | setcap → yes ×4 → `./01_swpwm 20` vs `./01_swpwm 20 -f` 기록 비교 |

**핵심 내용**

#### SCHED_FIFO 효과
> 실시간 우선순위면 부하 중에도 부하 없을 때와 비슷하다
- **SCHED_OTHER (부하 중)** → 평균 **1840µs** · 최대 **11962µs** (깜빡임)
- **SCHED_FIFO (부하 중)** → 평균 **58µs** · 최대 **171µs**
- ⚠ FIFO 쓰레드가 잠들지 않고 돌면 → 같은 코어의 일반 프로세스가 거의 못 돈다
- **우리 코드가 안전한 이유** → 매 주기 `clock_nanosleep` 으로 양보 (Linux Day2 8교시 "잠드는 지점")
- **권한 필요** · 우선순위는 필요한 만큼만

**기본 사용형태**
```c
pthread_attr_t at; pthread_attr_init(&at);
if (fifo) {
    mlockall(MCL_CURRENT | MCL_FUTURE);
    struct sched_param sp = { .sched_priority = 50 };
    pthread_attr_setinheritsched(&at, PTHREAD_EXPLICIT_SCHED);   // 필수
    pthread_attr_setschedpolicy(&at, SCHED_FIFO);
    pthread_attr_setschedparam(&at, &sp);
}
int e = pthread_create(&th, &at, rgb_thread, NULL);              // EPERM → setcap
```
```bash
sudo setcap cap_sys_nice,cap_ipc_lock+ep ./02_rgb && ./02_rgb -f
```

**연관 문제**
<<QA 2>>

### 3교시 · 하드웨어 PWM

**키워드:** 하드웨어 PWM, BCM2711 PWM 회로, pwm-2chan, PWM0 GPIO18, PWM1 GPIO19, /sys/class/pwm/pwmchip0, export, unexport, npwm, period ns, duty_cycle, enable, polarity, duty ≤ period 순서 함정, Invalid argument, udev 권한 대기, pwm_sysfs.h, 03_hwpwm

**개요**
- BCM2711 안의 **PWM 회로**는 주기와 듀티를 한 번 써 두면 CPU 가 바빠도 정확한 파형을 계속 낸다. Day 1 에 켠 `pwm-2chan` 으로 생긴 `/sys/class/pwm` 을 셸로 먼저 다루고, **duty ≤ period 순서 함정**을 처리하는 C 모듈 `pwm_sysfs.h` 를 만든다. (Linux Day4 sysfs "파일 하나 = 값 하나")

**핵심 용어**
- **채널**: PWM0 = **GPIO18(물리 12)** → `pwmchip0/pwm0`, PWM1 = **GPIO19(물리 35)** → `pwm1`. 2개뿐.
- **sysfs PWM 파일**: `export`/`unexport`(채널 번호) / **`period`**(주기 **ns**) / **`duty_cycle`**(켜짐 ns, **항상 ≤ period**) / `enable`(1/0, period 를 쓰기 전에 켜면 실패) / `polarity`(normal/inversed, **enable 이 0 일 때만**).
- **순서 함정**: duty 가 900000 인데 period 를 500000 으로 줄이면 "duty > period" → **Invalid argument**. → **주기를 늘릴 때는 period 먼저, 줄일 때는 duty 먼저**. export 직후 period 를 안 쓰고 duty 부터 써도(period 0) Invalid argument.
- **udev 대기**: export 직후 권한이 gpio 그룹으로 바뀌는 데 시간이 걸려 `pwm_open()` 이 최대 1초(10ms × 100) 기다린다.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `echo 0 > /sys/class/pwm/pwmchip0/export` | pwm0 디렉터리 생성 |
| `echo 1000000 > pwm0/period` / `echo 250000 > pwm0/duty_cycle` / `echo 1 > pwm0/enable` | 1kHz · 25% · 켜기 |
| `pwm_open(&p, 0, ch)` / `pwm_set(&p, period_ns, duty_ns)` / `pwm_set_duty` / `pwm_enable` / `pwm_close` | `pwm_sysfs.h` — 순서를 알아서 처리 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 3-3 셸 조작 | export → period 1ms → duty 0.2ms → enable → duty 0.9ms(밝게) → disable → unexport |
| `pwm_sysfs.h` | sysfs 파일에 숫자 문자열을 쓰는 `pwm_write`, export + 권한 대기 `pwm_open`, 주기를 늘리면 period 먼저·줄이면 duty 먼저 쓰는 `pwm_set` |
| `03_hwpwm.c` | GPIO18 하드웨어 PWM 1kHz 로 숨쉬기(듀티 0→100→0) 또는 인자 듀티 고정 |

**핵심 내용**

#### 하드웨어 PWM
> 파형을 SoC 안의 PWM 회로가 만든다 — 부하가 걸려도 흔들리지 않는다
- **같은 부하(yes ×4)** → 전혀 흔들리지 않음
- **프로그램이 늦게 깨어나면** → 밝기가 바뀌는 시점만 늦을 뿐

| | 소프트웨어 PWM (01) | 하드웨어 PWM (03) |
|---|---|---|
| 파형 만드는 곳 | 내 쓰레드 (매 주기 깨어남) | PWM 회로 |
| 부하 영향 | 깜빡임 | 없음 |
| CPU | 주기마다 깨어남 | 값을 바꿀 때만 |
| 채널 수 | **아무 GPIO 핀이나** | 2개뿐 (GPIO18 · 19) |

**기본 사용형태**
```bash
cd /sys/class/pwm/pwmchip0
echo 0 > export
echo 1000000 > pwm0/period          # 1 ms = 1 kHz (ns)
echo 250000  > pwm0/duty_cycle      # 25 %
echo 1 > pwm0/enable
```
```c
pwm_t p;
pwm_open(&p, 0, 0);                  // GPIO18
pwm_set(&p, 1000000, 0); pwm_enable(&p, 1);
pwm_set_duty(&p, 1000000 / 100 * 30);
pwm_close(&p);
```

**연관 문제**
<<QA 3>>

### 4교시 · 서보와 패시브 부저

**키워드:** 서보 SG90, 50Hz 20ms, 펄스 폭 0.5ms 1.5ms 2.5ms, duty_cycle 이 정보, 끝점 보정, angle_to_ns, 서보 전원 5V 신호 3.3V, 패시브 부저, 주파수 = 1/period, 듀티 50%, 음계 C4 262Hz A4 440Hz, 작은 별, 04_servo, 05_melody

**개요**
- 하드웨어 PWM 두 채널로 서로 다른 정보를 싣는다: **서보는 주기(20ms)를 고정하고 펄스 폭(duty)** 으로 각도를, **패시브 부저는 듀티(50%)를 고정하고 주기(period)** 로 음정을. (MCU Day2 서보, Day1 ledcWriteTone 의 Pi 판)

**핵심 용어**
- **서보(SG90)**: 20ms(50Hz) 마다 오는 펄스의 **폭**을 각도로 — 0.5ms = 0°, **1.5ms = 90°**, 2.5ms = 180°. 제품마다 끝점이 달라 **보정값**을 받는다(끝점을 넘으면 끝에서 "징—" 떨림 → 최소 펄스 폭을 조금 올림). 전원은 **5V**, 신호선은 3.3V 로도 동작.
- **패시브 부저**: 안에 발진 회로가 없어 **우리가 준 주파수**대로 소리 → period = 1/주파수. **듀티 50%** 가 가장 크게 들림. (액티브 부저는 전원만 주면 정해진 소리)

| 음 | 주파수 | period (ns) |
|---|---|---|
| 도 C4 | 262 Hz | 3 816 794 |
| 미 E4 | 330 Hz | 3 030 303 |
| 솔 G4 | 392 Hz | 2 551 020 |
| **라 A4** | **440 Hz** | **2 272 727** |
| 도 C5 | 523 Hz | 1 912 045 (C4 의 약 **절반**) |

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `angle_to_ns(deg)` | `(min_us + (max_us − min_us) × deg / 180) × 1000` 직선 보간 |
| `tone(hz, ms)` | `period = 1e9 / hz` → `pwm_set(period, period/2)` → enable → ms → disable → 30ms(같은 음 끊어 들리게) |
| `./04_servo 90 500 2400` | 90° + 끝점 보정 (0° = 500µs, 180° = 2400µs) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `04_servo.c` | PWM0(GPIO18) 50Hz 로 인자 각도 유지 또는 0° ↔ 180° 를 20ms 마다 2° 씩 왕복 (`0° 500 µs`, `90° 1500 µs`) |
| `05_melody.c` | PWM1(GPIO19) 로 도레미파솔라시도 → "작은 별"(전래 동요), 인자면 그 주파수를 1초 |

**핵심 내용**

#### 서보 vs 부저
> 서보는 주기 고정 · 펄스 폭을, 부저는 듀티 고정 · 주기를 바꾼다

| | 서보 | 패시브 부저 |
|---|---|---|
| 고정 | 주기 20 ms | 듀티 50 % |
| 바꾸는 것 | **duty_cycle (펄스 폭)** | **period (주파수)** |
| MCU 에서 | `servo.write()` | `ledcWriteTone()` |

- **90°** → duty_cycle **1500000** (ns)
- **한 주기마다 2° 씩만** → 너무 빠르면 서보가 따라오지 못함

#### 음 바꾸기와 duty ≤ period
> 주기를 줄일 때는 duty 를 먼저 쓴다
- **높은 음으로** (주기 ↓) → `pwm_set()` 이 **duty 를 먼저** 써서 duty ≤ period 유지 (3교시 함정)
- **한 음 순서** → `pwm_set(period, period/2)` → enable 1 → ms 기다림 → enable 0 → 30ms 쉼

**기본 사용형태**
```c
static long angle_to_ns(int deg) { if (deg < 0) deg = 0; if (deg > 180) deg = 180;
    return (g_min_us + (g_max_us - g_min_us) * deg / 180) * 1000; }
pwm_set(&servo, 20000000L, angle_to_ns(90)); pwm_enable(&servo, 1);
static void tone(int hz, int ms) {
    if (hz > 0) { long period = 1000000000L / hz; pwm_set(&buz, period, period / 2); pwm_enable(&buz, 1); }
    sleep_ms(ms); pwm_enable(&buz, 0); sleep_ms(30);
}
```

**연관 문제**
<<QA 4>>

### 5교시 · DC 모터와 릴레이

**키워드:** DC 모터, GPIO 직접 금지, 역기전력, L293D, 진리표, EN1 PWM 속도, IN1 IN2 방향, 정회전 역회전 자유 정지 브레이크, 방향 전환 순서, 20kHz, throttled 전원 부족, 릴레이, 전기적 분리, COM NO NC, 220V 금지, 06_dcmotor, 07_relay

**개요**
- 모터는 GPIO 의 전압·전류로 감당할 수 없다 → **L293D** 가 5V 에서 큰 전류를 끌어와 모터에 주고, GPIO 는 방향(IN1·IN2)·속도(EN 에 하드웨어 PWM) 신호만 준다. **릴레이**로 전기적으로 분리된 다른 회로를 켜고 끈다. (MCU Day3 7교시의 Pi 판)

**핵심 용어**
- **GPIO vs 모터**: 3.3V·핀당 16mA vs **3~6V·수백 mA**(시동 순간 더 큼) + 멈출 때 **역기전력**(반대 방향 전압 스파이크).
- **L293D 진리표** (EN1 = 1 또는 PWM):

| IN1 | IN2 | 모터 |
|---|---|---|
| 1 | 0 | 정회전 |
| 0 | 1 | 역회전 |
| 0 | 0 | 자유 정지 |
| 1 | 1 | **브레이크** (두 단자를 같은 전압으로 — 빠르게 멈춤) |

  EN1 = 0 이면 전원 끊김(자유 정지). EN1 에 PWM 을 주면 **듀티만큼 속도**.
- **방향 전환 순서**: **EN 듀티 0 → 브레이크(11) → 잠깐 대기 → 반대 방향 → 원래 속도**. 회전 중 바로 바꾸면 역기전력과 전원이 더해져 **큰 전류**가 흐름.
- **릴레이**: 전자석이 기계 접점을 당기는 스위치 — GPIO 쪽과 접점 쪽이 **전기적으로 분리**. **NO**(평소 열림 → ON 이면 COM 과 연결) / **NC**(평소 닫힘 → ON 이면 끊김).

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `set_dir(in1, in2)` | IN1·IN2 를 `set_values` 로 한 번에 |
| `pwm_set(&en, 50000, 0)` | **50µs = 20kHz** — 모터 "삐—" 소리를 들리지 않는 대역으로 |
| `vcgencmd get_throttled` | 0x0 이 아니면 전원 부족 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `06_dcmotor.c` | EN1 = PWM0(GPIO18), IN1·IN2 = GPIO20·21, 버튼 A(5)는 속도 단계 0 → 40 → 70 → 100%, 버튼 B(6)는 방향 반전(속도 0 → 브레이크 300ms → 반대 방향 → 원래 속도). 두 버튼을 한 요청·line offset 으로 구분 |
| `07_relay.c` | 버튼 A 를 누를 때마다 릴레이(GPIO25) 토글 — 딸깍, COM↔NO 사이에 끼운 3V3 LED 회로가 켜졌다 꺼짐 |

**핵심 내용**

#### DC 모터 제어
> 버튼 A 는 속도 단계, 버튼 B 는 방향
- **예시** → `정회전 40 %` → `정회전 70 %` → `역회전 70 %` → `역회전 100 %`
- **L293D 핀 번호** → 홈(반달)을 왼쪽에 두면 왼쪽 아래가 1번, 반시계 방향
- ⚠ 모터가 돌기 시작할 때 **Pi 가 재부팅** 또는 `get_throttled` ≠ 0x0 → **전원 부족** → 바로 정지, 5V 3A 어댑터 확인

#### 릴레이 안전
> 접점에는 저전압 회로만
- ⚠ 오늘은 3V3 LED 회로만 — **220V 상용 전원 절대 금지**

**기본 사용형태**
```c
static void set_dir(int in1, int in2) {
    enum gpiod_line_value v[2] = { in1, in2 };
    gpiod_line_request_set_values(g_dir, v);
}
pwm_set(&en, 50000L, 0); pwm_enable(&en, 1); set_dir(1, 0);
/* 방향 바꾸기 */
pwm_set_duty(&en, 0); set_dir(1, 1); sleep_ms(300);       // 브레이크
set_dir(0, 1); pwm_set_duty(&en, 50000L / 100 * 70);       // 역회전 70 %
```

**연관 문제**
<<QA 5>>

### 6교시 · 스테퍼 모터

**키워드:** 스테퍼 모터, 28BYJ-48, ULN2003, 4상, 감속기 1:64, 하프 스텝 8단계 순서표, 4096 스텝, 각도 → 스텝, (phase + dir + 8) % 8, 탈조, 사다리꼴 가감속, FAST_US SLOW_US RAMP, 절대 시각 스텝 간격, 코일 끄기, 08_stepper

**개요**
- 스테퍼 모터는 코일을 순서대로 켤 때마다 **정해진 각도씩만** 움직인다. 하프 스텝 순서표로 28BYJ-48 을 원하는 각도만큼 돌리고, 너무 빨리 돌릴 때 생기는 **탈조**를 관찰한 뒤 **사다리꼴 가감속**으로 막는다. (MCU Day3 8교시의 Pi 판)

**핵심 용어**
- **28BYJ-48**: 5V, 4상(IN1~IN4), 감속기 약 1:64, **하프 스텝 약 4096 스텝 = 한 바퀴**(실측 약 4076). 드라이버 **ULN2003**(트랜지스터 4개 — GPIO 신호로 코일 전류를 켬).
- **하프 스텝**: 코일 **1개 → 2개 → 1개**를 번갈아 켜 8단계 (A, AB, B, BC, C, CD, D, DA). 위에서 아래로 = 한 방향, 거꾸로 = 반대.
- **각도 → 스텝**: `steps = |deg| × 4096 / 360` → **90° = 1024**.
- **탈조**: 스텝 간격을 너무 짧게(예 1.0ms) 하면 회전자가 따라오지 못하고 **떨기만** — 프로그램은 **알아챌 수 없다**(위치 센서가 없으니).
- **사다리꼴 가감속**: **천천히 출발(4ms) → 빨라졌다가(1.2ms) → 천천히 멈춤**. 가속·감속에 각 200 스텝.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `phase = (phase + dir + 8) % 8` | dir = −1 일 때 **음수가 되지 않게** |
| `gpiod_line_request_set_values(req, v)` | 코일 4개 값을 **한 번에** |
| `./08_stepper 90` / `-360` / `360 0` | 시계 90° / 반시계 한 바퀴 / 가감속 없이(탈조 비교) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `08_stepper.c` | COILS {12, 16, 20, 21} 을 하프 스텝 표로 구동, 남은 스텝·지난 스텝 중 가까운 쪽으로 간격을 정해(사다리꼴) **절대 시각**으로 기다리고, 끝나면 코일 모두 끄기 → `90° = 하프 스텝 1024 개, 가감속 / 완료 1.66 초` |

**핵심 내용**

#### 스테퍼 하프 스텝
> 코일 하나 → 둘 → 하나 순서로 한 칸씩
- **정방향 0 → 4 단계** → **IN1 → IN1·IN2 → IN2 → IN2·IN3 → IN3**
- **ULN2003 보드 LED** → 코일이 켜지는 순서가 보임 (아주 느리게 돌려 확인)
- **배선** → 6교시는 L293D 를 빼고 같은 GPIO20·21 을 ULN2003 IN3·IN4 에

#### 간격과 정지
> 간격은 절대 시각으로 일정하게, 다 돌면 코일을 끈다
- **절대 시각으로 간격 유지** → 간격이 흔들리면 소리가 거칠고 탈조가 쉬움
- **다 돈 뒤 코일 끄기** → 켜 둔 채면 뜨거워지고 전류를 계속 씀 (대신 버티는 힘은 사라짐)

**기본 사용형태**
```c
static const int HALF[8][4] = {{1,0,0,0},{1,1,0,0},{0,1,0,0},{0,1,1,0},{0,0,1,0},{0,0,1,1},{0,0,0,1},{1,0,0,1}};
long steps = labs((long)deg) * 4096 / 360;  int dir = deg >= 0 ? 1 : -1;
for (long i = 0; i < steps && !g_stop; i++) {
    phase = (phase + dir + 8) % 8;
    enum gpiod_line_value v[4]; for (int c = 0; c < 4; c++) v[c] = HALF[phase][c];
    gpiod_line_request_set_values(req, v);
    long edge = i < steps - 1 - i ? i : steps - 1 - i;
    long us = edge < 200 ? 4000 - (4000 - 1200) * edge / 200 : 1200;    // 사다리꼴
    ts_add(&next, us * 1000); clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
}
```

**연관 문제**
<<QA 6>>

### 7교시 · 초음파 거리 측정

**키워드:** HC-SR04, Trig 10µs, 40kHz 8번, Echo 폭 = 왕복 시간, 거리 = t ÷ 58, Echo 5V 분압 1kΩ 2kΩ, 3.33V, 커널 타임스탬프 rising falling 차이, 1cm = 58µs, 측정 전 이벤트 버리기, 60ms 마감, 측정 간격 60ms, 5회 중앙값, qsort, bias DISABLED, 09_sonar

**개요**
- HC-SR04 의 Trig·Echo 동작과 거리 공식을 이해하고, **5V Echo 를 분압**해 안전하게 연결한다. Echo 폭을 사용자 공간 시각이 아니라 **커널 엣지 타임스탬프**로 재고(Day 2 반응속도와 같은 원리), **중앙값 필터**로 튀는 값을 거른다. (MCU Day2 6교시의 Pi 판)

**핵심 용어**
- **동작 순서**: Trig 에 **10µs 이상 HIGH** → 센서가 40kHz 초음파를 8번 쏘고 Echo 를 HIGH → 반사파가 돌아오면 Echo LOW → **Echo HIGH 시간 = 왕복 시간**.
- **거리 = t × 343 m/s ÷ 2** ≈ **t(µs) ÷ 58** cm. ÷2 는 **갔다 돌아오므로**. 580µs ≈ 10cm, **2900µs ≈ 50cm**, 11600µs ≈ 2m, 약 38ms = 감지 실패(센서 시간 초과 — Echo 가 영원히 HIGH 로 남지는 않음).
- **Echo 분압**: `Echo ─[1kΩ]─┬─ GPIO24`, `┴─[2kΩ]─ GND` → **5V × 2k / 3k ≈ 3.33V**. 같은 비율 10kΩ : 20kΩ 도 가능.
- **왜 커널 타임스탬프**: 1cm 오차 = 약 58µs. `get_value` 로 계속 읽으며 `now_ns()` → 읽기 간격 + 스케줄링 지연 → **수 cm ~ 수십 cm** 오차. rising·falling **이벤트 타임스탬프 차이** → 인터럽트 순간 → **약 1mm 수준**.
- **중앙값**: 정렬해 가운데 값 — 튀는 값 하나(옷·비스듬한 면)에 흔들리지 않음. **평균은 튀는 값에 끌려간다**.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `measure_cm()` | 찌꺼기 이벤트 버리기 → Trig 10µs → 60ms 마감으로 rising·falling 타임스탬프 → `(fall − rise)/1000` µs → 30000µs 초과면 −1 |
| `median_cm()` | 5번 측정(사이 60ms) → `qsort` → 가운데 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `09_sonar.c` | Trig = GPIO23, Echo = GPIO24(분압, BOTH 엣지, bias DISABLED) 로 5회 중앙값 거리와 막대를 출력 → `23.4 cm \|====`, 실패면 "측정 실패 (너무 멀거나 각도가 비스듬함)" |

**핵심 내용**

#### 초음파 측정 순서
> Trig 10µs → 발사 → 반사 → Echo 폭 ÷ 58
- **한 번 측정** → Trig 10µs HIGH → 40kHz 8번 발사 후 Echo HIGH → 물체에 반사 → Echo LOW → 타임스탬프 차이 ÷ 58

#### 측정을 안정시키는 장치
> 찌꺼기 · 무한 대기 · 메아리 · 튀는 값을 막는다

| 장치 | 이유 |
|---|---|
| 측정 전 이벤트 버리기 | **지난 측정의 늦은 메아리** 제거 |
| 60ms 마감 | 물체가 없으면 Echo 가 오래 HIGH — 무한 대기 방지 |
| 측정 사이 60ms | **이전 메아리가 사라질 시간** |
| 5회 중앙값 | 튀는 값에 흔들리지 않음 |

- **5번 중 하나가 크게 튀어도** (옷 · 비스듬한 면) → 정렬 후 **가운데 값**이라 흔들리지 않음

**기본 사용형태**
```c
gpiod_line_settings_set_edge_detection(s, GPIOD_LINE_EDGE_BOTH);
gpiod_line_settings_set_bias(s, GPIOD_LINE_BIAS_DISABLED);    // 센서가 직접 0/1
/* measure_cm */
set(TRIG, 1); nanosleep(&(struct timespec){0, 10000}, NULL); set(TRIG, 0);
if (type == GPIOD_EDGE_EVENT_RISING_EDGE) rise = ts; else if (rise) fall = ts;
double us = (fall - rise) / 1000.0;
return us > 30000 ? -1 : us / 58.0;
```

**연관 문제**
<<QA 7>>

### 8교시 · 미니 프로젝트 — 거리 경보기

**키워드:** 거리 경보기, alarm.c, 측정 쓰레드 sensor_thread, 공유 g_cm mutex, 메인 루프 20ms, LED 초록 노랑 빨강, 부저 간격 = 거리 × 10ms, 연속음, 서보 차단기 0° 90°, 3회 중앙값, 하드웨어 PWM 부하 무관, 히스테리시스 도전

**개요**
- Day 3 부품을 총동원한 **초음파 거리 경보기**. 측정은 최대 60ms 를 기다리므로 **측정 쓰레드**를 분리하고 **mutex** 로 거리 값을 공유한다. 메인 루프는 20ms 마다 거리에 따라 LED·부저 박자·서보 차단기를 구동한다.

**핵심 용어** — 경보 규칙

| 거리 | LED | 부저 | 서보 차단기 |
|---|---|---|---|
| 50 cm 이상 · 측정 없음 | 초록 | 조용 | 올림 (90°) |
| 20 ~ 50 cm | 노랑 | 삑 — **간격 = 거리 × 10 ms** | 올림 |
| 10 ~ 20 cm | 빨강 | 삑 (더 빠르게) | 올림 |
| 10 cm 미만 | 빨강 | **연속음** | **내림 (0°)** |

- **구조**: `sensor_thread`(3회 측정 → 중앙값 → `g_cm`, 약 0.2초) + 메인(20ms 마다 `g_cm` 읽어 출력 갱신). 같은 루프에서 측정하면 **최대 60ms 기다려 부저 박자가 끊긴다**.

**새로 나온 함수/명령** — 새 함수 없음 (Day 2~3 의 libgpiod · pwm_sysfs · pthread 조합)

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `alarm.c` | 측정 쓰레드가 mutex 안에서 `g_cm` 갱신 / 메인이 mutex 로 복사 → LED 구간(`set_values`) → 부저 켜기·끄기 결정(바뀔 때만 `pwm_enable`, 2kHz 고정) → 서보(바뀔 때만 `pwm_set_duty`) → 0.5초마다 상태 출력 |

**핵심 내용**

#### 거리 경보기 동작
> 거리 구간에 따라 LED · 부저 · 서보
- **예시** → `72.4 cm 초록` / `38.1 cm 노랑` / `15.6 cm 빨강` / `7.9 cm 빨강 ▼ 차단기 내림`
- **30cm** → **노랑, 300ms 마다 삑**
- **8cm** → 서보 **0° (내림)**
- **메인 루프 한 바퀴** → mutex 로 `g_cm` 복사 → LED 구간 → 부저 결정 → 서보 → 출력 → 20ms

#### 하드웨어 PWM 덕분에
> 부하에도 음정 · 서보 위치는 흔들리지 않는다 — 반응만 조금 늦을 수 있다
- **부저** → 박자마다 period 를 새로 쓰지 않음 — **2kHz 고정**, `enable` 만 켜고 끔
- **부하(yes ×4)** → 부저 음정 · 서보 위치는 그대로 (파형은 하드웨어 PWM)
- ⚠ 측정 쓰레드 · 메인은 일반 쓰레드 → **반응은 조금 늦어질 수 있다**
- **10cm 경계 떨림** → 3회 중앙값이 걸러 줌 → 더 안정하려면 **히스테리시스** (10cm 미만 내림, 15cm 이상 올림 — 도전 과제, MCU Day2 5교시)

**기본 사용형태**
```c
static double g_cm = -1; static pthread_mutex_t g_mx = PTHREAD_MUTEX_INITIALIZER;
/* sensor_thread */ pthread_mutex_lock(&g_mx); g_cm = m; pthread_mutex_unlock(&g_mx);
/* main */
pthread_mutex_lock(&g_mx); double d = g_cm; pthread_mutex_unlock(&g_mx);
int zone = (d < 0 || d >= 50) ? 2 : (d >= 20 ? 1 : 0);             // 2 초록 · 1 노랑 · 0 빨강
long gap = (long)(d * 10) * 1000000LL;                               // 간격 = 거리 × 10 ms
int down = d > 0 && d < 10;
if (down != gate_down) { pwm_set_duty(&servo, servo_ns(down ? 0 : 90)); gate_down = down; }
```

**연관 문제**
<<QA 8>>
