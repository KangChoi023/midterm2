## Day 2 — GPIO 출력 · 입력 · 이벤트

### 1교시 · libgpiod v2 구조와 첫 출력

**키워드:** libgpiod v2, gpiod_chip_open, gpiod_line_settings, gpiod_line_config, gpiod_request_config, consumer, gpiod_chip_request_lines, gpiod_line_request, set_value, release, 객체 흐름, SIGINT 정리, SA_RESTART 없음, EINTR, gpio_common.h, 01_blink

**개요**
- Day 1 의 `gpioset` 명령이 안에서 부르던 **libgpiod C 함수**로 같은 일을 한다. v2 는 "무엇을 · 어떻게 · 누가" 를 각각 객체로 만든 뒤 **한 번에 요청**한다. Ctrl+C 를 잡아 LED 를 끄고 라인을 반납하는 정리 패턴과, Day 2~5 공통 헤더 `gpio_common.h` 를 만든다.

**핵심 용어**
- **객체 흐름(v2)**: ① **chip** 열기 → ② **line_settings**(방향·초기값·bias·edge·active-low = "어떻게") → ③ **line_config**(offset ↔ settings = "어떤 라인에") → ④ **request_config**(consumer = "누가") → ⑤ **chip_request_lines** → **line_request**(커널에 요청 → 소유) → ⑥ 사용 후 **release**.
- **재료 vs 결과**: settings·line_config·request_config 는 요청을 만드는 **재료**라 요청 후 바로 free 해도 된다. line_request 만 있으면 **chip 은 닫아도 된다**.
- **consumer**: `gpioinfo` 에 보이는 사용자 이름(`"blink"`). 실행 중 다른 창에서 `gpioset 17=1` → **busy**.

| Day 1 명령 | Day 2 C 함수 |
|---|---|
| `gpiodetect` · `gpioinfo` | `gpiod_chip_open()` · `gpiod_chip_get_line_info()` |
| `gpioset 17=1` | `gpiod_chip_request_lines()` → `gpiod_line_request_set_value()` |
| `gpioget -b pull-up 5` | 입력·bias 설정 후 `gpiod_line_request_get_value()` |
| `gpiomon -e falling 5` | edge 설정 후 `poll()` + `gpiod_line_request_read_edge_events()` |
| Ctrl+C → 라인 해제 | `gpiod_line_request_release()` |

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `gpiod_chip_open("/dev/gpiochip0")` / `gpiod_chip_close` | 칩 열기/닫기 |
| `gpiod_line_settings_new` / `_set_direction(s, GPIOD_LINE_DIRECTION_OUTPUT)` / `_set_output_value(s, GPIOD_LINE_VALUE_INACTIVE)` | 설정 만들기 |
| `gpiod_line_config_new` / `gpiod_line_config_add_line_settings(cfg, &offset, 1, s)` | offset 에 설정 묶기 |
| `gpiod_request_config_new` / `_set_consumer(rc, "blink")` | consumer 이름 |
| `gpiod_chip_request_lines(chip, rc, cfg)` | **커널에 실제로 요청** → `gpiod_line_request *` (실패 NULL) |
| `gpiod_line_request_set_value(req, off, GPIOD_LINE_VALUE_ACTIVE)` | 값 쓰기 |
| `gpiod_line_request_release(req)` / `*_free(...)` | 반납 / 재료 해제 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `01_blink.c` | 객체를 하나씩 직접 만들어 GPIO17 을 요청하고 0.5초마다 깜빡임, Ctrl+C → `g_stop` → LED 끄고 `release` → "라인 반납 완료" |
| `gpio_common.h` | `gpio_install_sigint()`(SA_RESTART 없이 SIGINT·SIGTERM), `now_ns()`(MONOTONIC ns), `sleep_ms()`, `gpio_request()` / `gpio_request_output()` / `gpio_request_input(…, edge, debounce_us)` 한 줄 요청 |

**핵심 내용**

#### Ctrl+C 와 라인 반납
> 핸들러는 플래그만 — 루프가 끝나야 LED 를 끄고 반납할 수 있다
- **SA_RESTART 를 주지 않는 이유** → `poll()` · `nanosleep()` 같은 잠든 호출이 **EINTR 로 깨어나** 루프가 끝나게 (Linux Day1 7교시)
- **공유 플래그** → `static volatile sig_atomic_t g_stop` (Linux Day1)
- ⚠ 핸들러 없이 Ctrl+C 로 죽으면 → fd 가 닫혀 라인은 해제되지만 LED 를 끄는 코드는 실행 안 됨 → **켜진 채** 남을 수 있음
- **그래서** → 직접 반납 ("끄고 나가기")

**기본 사용형태**
```c
unsigned int off = 17;
struct gpiod_chip *chip = gpiod_chip_open("/dev/gpiochip0");
struct gpiod_line_settings *s = gpiod_line_settings_new();
gpiod_line_settings_set_direction(s, GPIOD_LINE_DIRECTION_OUTPUT);
struct gpiod_line_config *lc = gpiod_line_config_new();
gpiod_line_config_add_line_settings(lc, &off, 1, s);
struct gpiod_request_config *rc = gpiod_request_config_new();
gpiod_request_config_set_consumer(rc, "blink");
struct gpiod_line_request *req = gpiod_chip_request_lines(chip, rc, lc);
gpiod_request_config_free(rc); gpiod_line_config_free(lc); gpiod_line_settings_free(s); gpiod_chip_close(chip);
while (!g_stop) { gpiod_line_request_set_value(req, off, on ^= 1); sleep_ms(500); }
gpiod_line_request_set_value(req, off, GPIOD_LINE_VALUE_INACTIVE);
gpiod_line_request_release(req);
```

**연관 문제**
<<QA 1>>

### 2교시 · 여러 라인 동시 출력

**키워드:** 여러 라인 한 요청, offsets 배열, gpiod_line_request_set_values, set_values_subset, values 순서, 신호등 상태표, active-low, set_active_low, drive, PUSH_PULL, OPEN_DRAIN, OPEN_SOURCE, 02_traffic

**개요**
- 라인 세 개(빨강 17 · 노랑 27 · 초록 22)를 **한 요청으로 묶어** `set_values()` 한 번으로 동시에 바꾼다. 신호등 순서를 **상태표(배열)** 로 표현하고, 논리값과 전압을 뒤집는 **active-low** 와 출력 회로 종류(**drive**)를 이해한다.

**핵심 용어**
- **values 순서**: `set_values(req, v[])` 의 `v[]` 순서 = 요청할 때 넘긴 **offsets 순서**. `{17, 27, 22}` 에 `{0, 1, 0}` → **노랑(27)만**.
- **active-low**: 논리 1(ACTIVE) = **0V**. LED 를 `3V3 → 220Ω → LED → GPIO` 로 연결했을 때 코드에서 "1 = 켜짐" 으로 쓰려면 active-low 로 요청. 전압 반전은 **커널**이 해 주고 `gpioinfo` 에 active-low 로 표시.
- **drive**: **PUSH_PULL**(기본, 1 = 3.3V 밀어냄, 0 = 0V) / **OPEN_DRAIN**(1 = 손을 놓음·떠 있음 → 외부 풀업이 끌어올림, 0 = 0V — 여러 장치가 한 선 공유, I2C 원리) / OPEN_SOURCE(드묾).

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `gpiod_line_request_set_value(req, off, v)` | 라인 하나 |
| `gpiod_line_request_set_values(req, v[])` | 요청한 **모든 라인** — 시스템 콜 한 번 |
| `gpiod_line_request_set_values_subset(req, n, offsets, v)` | 일부만 |
| `gpiod_line_settings_set_active_low(s, true)` | 논리 1 = 0V |
| `gpiod_line_settings_set_drive(s, GPIOD_LINE_DRIVE_OPEN_DRAIN)` | 출력 회로 종류 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `02_traffic.c` | `STEPS[] = {R, Y, G, ms, name}` 상태표(초록 3초 → 노랑 1초 → 빨강 3초 → 빨강+노랑 1초)를 반복하며 `set_values` 로 세 LED 를 한 번에, `-l` 이면 active-low 로 요청(켜질 LED 가 꺼짐) |

**핵심 내용**

#### 여러 라인을 한 요청으로
> 세 라인을 시스템 콜 한 번으로 동시에 바꾼다
- **장점** → **시스템 콜 한 번으로 동시에**
- **순서 바꾸기** → **상태표만** 고치면 루프 코드는 그대로 (MCU Day3 비트맵 표와 같은 원리)
- **LED 세 개 동시** → 핀 하나당 전류는 그대로 (약 5.9mA), 합계 약 18mA → Pi 전체 GPIO 합계도 수십 mA 수준이라 220Ω 이상 유지

#### active-low
> 논리 1 = 전압 0V 로 뒤집는다
- **`./02_traffic -l`** → 표는 그대로인데 켜져야 할 LED 가 꺼지고 꺼져야 할 LED 가 켜짐 (배선이 active-high 라서)

**기본 사용형태**
```c
static const unsigned int LINES[3] = {17, 27, 22};          // R, Y, G
struct gpiod_line_settings *s = gpiod_line_settings_new();
gpiod_line_settings_set_direction(s, GPIOD_LINE_DIRECTION_OUTPUT);
gpiod_line_settings_set_active_low(s, active_low);
struct gpiod_line_request *req = gpio_request("traffic", LINES, 3, s);
gpiod_line_settings_free(s);
enum gpiod_line_value v[3] = { GPIOD_LINE_VALUE_INACTIVE, GPIOD_LINE_VALUE_ACTIVE, GPIOD_LINE_VALUE_INACTIVE };
gpiod_line_request_set_values(req, v);                       // 노랑만
```

**연관 문제**
<<QA 2>>

### 3교시 · 레지스터 직접 접근 비교

**키워드:** BCM2711 GPIO 레지스터, GPFSEL, GPSET0, GPCLR0, GPLEV0, GPIO_PUP_PDN_CNTRL, 비트 계산 pin/10 (pin%10)*3, /dev/gpiomem, /dev/mem, mmap, volatile uint32_t, 읽기-수정-쓰기, ioctl 시스템 콜, 64배, 소유권 없음, 이식성 없음 Pi 5 RP1

**개요**
- MCU 과목의 `digitalWrite()` 아래에 레지스터 쓰기가 있었듯, Pi 의 GPIO 도 SoC 안의 레지스터다. `/dev/gpiomem` 을 **mmap** 해 레지스터만으로 버튼 → LED 를 구현하고, libgpiod 와 **속도**를 비교한 뒤 **평소에는 왜 mmap 을 쓰지 않는지** 정리한다. (MCU Day1 W1TS/W1TC, Linux Day3 mmap 의 결합)

**핵심 용어**
- **GPFSELn**: 핀당 **3비트** 기능(000 입력 · 001 출력 · 그 외 ALT). 번호 = `pin / 10`, 시작 비트 = `(pin % 10) × 3` → **GPIO17 = GPFSEL1 비트 21~23**. 읽기-수정-쓰기.
- **GPSET0(0x1C) / GPCLR0(0x28)**: 1 을 쓴 비트만 High / Low (0 은 무시) → **읽기 없이 쓰기 한 번으로 한 핀만** (MCU W1TS/W1TC 와 같은 이유).
- **GPLEV0(0x34)**: 핀 0~31 현재 레벨 읽기.
- **GPIO_PUP_PDN_CNTRL0~3(0xE4~)**: 핀당 2비트 pull(00 없음 · 01 up · 10 down). 번호 = `pin / 16`, 시작 = `(pin % 16) × 2`.
- **/dev/gpiomem**: GPIO 레지스터 **4KB 만** 보여 주는 장치 파일(gpio 그룹이면 열림) — 물리 메모리 전체인 `/dev/mem` 보다 **안전**.

| 오프셋 | 이름 | 의미 |
|---|---|---|
| 0x00 ~ 0x14 | GPFSEL0~5 | 핀당 3비트 기능 |
| 0x1C | GPSET0 | 1 쓴 비트 High |
| 0x28 | GPCLR0 | 1 쓴 비트 Low |
| 0x34 | GPLEV0 | 현재 레벨 |
| 0xE4 ~ 0xF0 | PUP_PDN_CNTRL0~3 | 핀당 2비트 pull |

**새로 나온 함수/명령**

| 식 | 뜻 |
|---|---|
| `gpio[1] = (gpio[1] & ~(7u << 21)) \| (1u << 21)` | GPIO17 = 출력 |
| `gpio[0x1C / 4] = 1u << 22` | GPIO22 High (워드 인덱스 = 바이트 오프셋 / 4) |
| `!(gpio[0x34 / 4] & (1u << 5))` | GPIO5 눌림(pull-up → 누르면 0) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `03_gpiomem.c` | `open("/dev/gpiomem")` → `mmap` 4096 → GPFSEL 로 17 출력·5 입력 → PUP_PDN 으로 5 pull-up → 1ms 폴링으로 GPLEV0 를 읽어 누르는 동안 GPSET/GPCLR → 끝나면 입력으로 되돌리고 munmap |
| `04_speed.c` | 같은 핀을 libgpiod(`set_value`)와 mmap(GPSET/GPCLR)으로 100만 번 토글해 ns/회 비교 |

**핵심 내용**

#### mmap 으로 레지스터 직접 쓰기
> 메모리 쓰기 한 번 — 하지만 커널은 모른다
- **`03_gpiomem` 실행 중 `gpioinfo 17`** → **consumer 가 없다** — 커널은 모름
- **`04_speed`** → libgpiod **950.3 ns/회 (1.05MHz)** vs mmap **14.8 ns/회 (67.57MHz)** → 약 **64배**
- **libgpiod** → 토글마다 **ioctl 시스템 콜** (Linux Day1 4교시) / **mmap** → 메모리 쓰기 한 번

#### 평소 mmap 을 쓰지 않는 이유
> 속도 대신 안전 · 이식성 · 인터럽트를 잃는다
- ⚠ **① 소유권 없음** → busy 보호가 없어 다른 프로그램 · 드라이버와 충돌
- ⚠ **② 이식성 없음** → SoC 마다 주소가 다름 (Pi 5 는 RP1 칩이라 완전히 다름)
- ⚠ **③ 인터럽트 없음** → 엣지를 기다리려면 폴링
- ⚠ **④ 정리 책임** → 죽으면 핀이 출력으로 남음
- **결론** → 1MHz 이상 비트뱅잉이 꼭 필요할 때만, 나머지는 libgpiod / 정확한 파형은 하드웨어 PWM · SPI (Day 3·4)

**기본 사용형태**
```c
int fd = open("/dev/gpiomem", O_RDWR | O_SYNC);
volatile uint32_t *gpio = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
close(fd);                                          // 매핑은 남는다 (Linux Day3)
if (gpio == MAP_FAILED) { perror("mmap"); return 1; }
gpio[1] = (gpio[1] & ~(7u << 21)) | (1u << 21);     // GPFSEL1: GPIO17 출력
gpio[0x1C / 4] = 1u << 17;                           // GPSET0: High
gpio[0x28 / 4] = 1u << 17;                           // GPCLR0: Low
munmap((void *)gpio, 4096);
```

**연관 문제**
<<QA 3>>

### 4교시 · 디지털 입력과 bias

**키워드:** bias 네 가지, PULL_UP, PULL_DOWN, DISABLED, AS_IS, gpiod_line_settings_set_bias, get_value, 폴링, 소프트웨어 디바운스, STABLE_CNT 20, 바운싱, 폴링 간격 대가, CPU 100%, 커널 디바운스 set_debounce_period_us, Ball Tilt 센서, 05_button_poll

**개요**
- bias 네 가지를 구분해 버튼에 맞는 bias 를 고르고, **폴링 + 소프트웨어 디바운스**로 버튼을 정확히 센다. 폴링 간격의 대가와, 코드 카운터가 필요 없는 **커널 디바운스**의 장점을 비교한다. (MCU Day1 5교시 디바운스의 Linux 판)

**핵심 용어**

| `set_bias()` | 뜻 | MCU 에서 |
|---|---|---|
| `GPIOD_LINE_BIAS_PULL_UP` | 내부 저항으로 3.3V 쪽 — **GND 쪽 버튼의 짝** | `INPUT_PULLUP` |
| `GPIOD_LINE_BIAS_PULL_DOWN` | 내부 저항으로 GND 쪽 | `INPUT_PULLDOWN` |
| `GPIOD_LINE_BIAS_DISABLED` | 저항 끄기 — **센서가 스스로 0/1 을 낼 때(PIR)** | `INPUT` |
| `GPIOD_LINE_BIAS_AS_IS` | 지금 상태 그대로 (기본값) | — |

- **소프트웨어 디바운스**: "값이 바뀌면 바로 믿지 말고, **20ms 동안 같은 값이 이어질 때** 믿는다." 1ms 폴링 × 20번 = 확정까지 최소 **약 20ms**.
- **커널 디바운스**: `set_debounce_period_us()` — 커널이 그 시간 동안 안정된 값만 보고 → **카운터 불필요**, 이벤트와 함께 사용, 설정값 하나.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `gpiod_line_settings_set_bias(s, GPIOD_LINE_BIAS_PULL_UP)` | bias |
| `gpiod_line_request_get_value(req, off)` | 값 읽기 (1 안 누름 · 0 누름 · <0 오류) |
| `gpiod_line_settings_set_debounce_period_us(s, us)` | 커널 디바운스 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `05_button_poll.c` | GPIO5 를 pull-up 입력으로 1ms 마다 읽어 `same` 카운터가 20 이상이고 값이 바뀌면 "눌림 #n" — `-n` 이면 디바운스 없이(바뀌는 즉시), 숫자 인자면 그 라인(6 = 기울기 센서) |

**핵심 내용**

#### 바운싱
> 기계 접점은 한 번 누를 때 여러 번 붙었다 떨어진다
- **`-n` (디바운스 없음)** → 한 번 눌렀는데 `눌림 #1, #2` → 바운싱을 그대로 믿음
- **Ball Tilt 센서** (버튼 B 자리, GPIO6 ↔ GND) → 금속 공이 굴러 붙었다 떼는 구조 → **버튼보다 심함** → `-n 6` 이면 수십 번

#### 폴링의 대가
> 간격이 짧을수록 정확하지만 CPU 를 쓴다

| 간격 | 1초당 시스템 콜 | 놓칠 수 있는 것 | CPU |
|---|---|---|---|
| 1 ms | 약 1000번 | 1ms 보다 짧은 펄스 | 낮지만 0 은 아님 |
| 10 ms (50ms 도) | 약 100번 | 그보다 짧은 펄스 | 더 낮음 |
| 0 (sleep 없음) | 수십만 번 | 거의 없음 | **코어 하나 100%** |

**기본 사용형태**
```c
struct gpiod_line_request *req = gpio_request_input("button_poll", &line, 1, GPIOD_LINE_EDGE_NONE, 0);
int stable = gpiod_line_request_get_value(req, line), last = stable, same = 0;
while (!g_stop) {
    int v = gpiod_line_request_get_value(req, line);
    same = (v == last) ? same + 1 : 0; last = v;
    if (same >= 20 && v != stable) { stable = v; if (v == 0) printf("눌림 #%d\n", ++presses); }
    sleep_ms(1);
}
```

**연관 문제**
<<QA 4>>

### 5교시 · 엣지 이벤트와 poll()

**키워드:** 엣지 이벤트, GPIOD_LINE_EDGE_BOTH FALLING, gpiod_line_request_get_fd, poll, POLLIN, gpiod_edge_event_buffer, read_edge_events, event_type RISING FALLING, timestamp_ns, line_offset, line_seqno, CLOCK_MONOTONIC, 짧게 길게 누름, CPU 0%, ESP32 ISR 비교

**개요**
- 폴링 대신 **"알려 줘"**: 커널이 GPIO 인터럽트를 받아 이벤트 큐에 넣고, 내 프로그램은 `poll()` 안에서 잠들어 있다가 깨어난다. 이벤트의 종류·**커널 타임스탬프**로 짧게/길게 누름을 구분하고, 폴링과 CPU 사용률, MCU 인터럽트와의 차이를 비교한다. (Linux Day4 inotify 와 같은 이벤트 방식)

**핵심 용어**
- **엣지 이벤트**: 라인 값이 바뀐 순간(rising 0→1 / falling 1→0)을 커널이 기록한 것. **큐에 쌓여 짧은 펄스도 놓치지 않음**. `poll` 이 깨어난 뒤 `read_edge_events` 한 번에 **여러 개**가 읽힐 수 있다.
- **커널 타임스탬프**: 커널이 **인터럽트를 받은 순간**의 시각(ns, 기본 **CLOCK_MONOTONIC**) → `clock_gettime(CLOCK_MONOTONIC)` 과 바로 뺄 수 있다.
- **받는 순서**: **edge 설정 요청 → `get_fd()` → `poll()` 로 잠들기 → `read_edge_events()`**.

| | 폴링 (05) | 엣지 이벤트 (06) |
|---|---|---|
| 누가 확인 | 내 프로그램이 1ms 마다 | 커널이 GPIO 인터럽트를 받아서 |
| 기다리는 동안 | 깨어났다 잠들기 반복 | `poll()` 안에서 잠듦 — **CPU 0%** |
| 시각 정보 | 내가 읽은 시각 (최대 1ms 오차) | **커널 타임스탬프 (ns)** |
| 짧은 펄스 | 놓칠 수 있음 | 큐에 쌓여 놓치지 않음 |

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `gpiod_line_settings_set_edge_detection(s, GPIOD_LINE_EDGE_BOTH)` | 엣지 감지 (FALLING / RISING / BOTH / NONE) |
| `gpiod_line_request_get_fd(req)` | poll 할 fd |
| `poll(&pfd, 1, -1)` | `{.fd, .events = POLLIN}` — 이벤트까지 잠듦 (-1 = 무한) |
| `gpiod_edge_event_buffer_new(16)` / `gpiod_line_request_read_edge_events(req, buf, 16)` | 버퍼에 이벤트 읽기 (개수 반환) |
| `gpiod_edge_event_buffer_get_event(buf, i)` | i 번째 이벤트 |
| `gpiod_edge_event_get_event_type` / `_get_timestamp_ns` / `_get_line_offset` / `_get_line_seqno` | 종류 / 시각 / 어느 라인 / 일련번호(빠진 이벤트 확인) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `06_edge_poll.c` | GPIO5 를 BOTH 엣지 + 커널 디바운스 10ms 로 요청해 poll 로 대기, falling 시각을 저장했다가 rising 이 오면 차이로 **0.5초 미만은 LED 토글, 이상은 "길게"** |

**핵심 내용**

#### 엣지 이벤트로 짧게 · 길게
> 누름~뗌의 커널 타임스탬프 차이로 판정 — 기다리는 동안 CPU 0%
- **예시** → `4211.081532917 falling` → `4211.204118220 rising` → **짧게 122 ms, LED ON**
- **예시** → `4213.550902114 falling` → `4214.390557036 rising` → **길게 839 ms**
- **판정 순서** → poll 에서 깨어남 → falling → `t_down` 저장 → rising → 차이 계산 → 500ms 미만이면 토글
- **`top`** → `05_button_poll` **1.3%** (1ms 마다 깨어남) vs `06_edge_poll` **0.0%**

#### ESP32 ISR vs Linux 엣지 이벤트
> 지연은 늘었지만 시각은 커널이 찍어 두므로 "언제 눌렸나" 는 정확

| | ESP32 `attachInterrupt` | Linux 엣지 이벤트 |
|---|---|---|
| 코드가 도는 곳 | ISR (인터럽트 문맥) | 일반 프로세스 |
| 제약 | 짧게 · volatile · IRAM_ATTR · printf 금지 | **제약 없음** — printf·파일·네트워크 |
| 지연 | 수 µs | 깨어나기까지 수십 µs ~ ms |
| 시각 | ISR 에서 micros() | 커널이 인터럽트 순간에 찍음 |

**기본 사용형태**
```c
struct gpiod_line_request *in = gpio_request_input("edge", &btn, 1, GPIOD_LINE_EDGE_BOTH, 10000);
struct gpiod_edge_event_buffer *buf = gpiod_edge_event_buffer_new(16);
struct pollfd pfd = { .fd = gpiod_line_request_get_fd(in), .events = POLLIN };
while (!g_stop) {
    if (poll(&pfd, 1, -1) < 0) { if (errno == EINTR) continue; break; }
    int n = gpiod_line_request_read_edge_events(in, buf, 16);
    for (int i = 0; i < n; i++) {
        struct gpiod_edge_event *ev = gpiod_edge_event_buffer_get_event(buf, i);
        long long ts = gpiod_edge_event_get_timestamp_ns(ev);
        int falling = gpiod_edge_event_get_event_type(ev) == GPIOD_EDGE_EVENT_FALLING_EDGE;
    }
}
```

**연관 문제**
<<QA 5>>

### 6교시 · 여러 입력 · 타이머 한 루프로

**키워드:** 이벤트 루프, epoll, epoll_create1, epoll_ctl EPOLL_CTL_ADD, epoll_wait, data.u32 태그, timerfd, timerfd_create, timerfd_settime, itimerspec, 만료 횟수 uint64, stdin fd, PIR 3V3 bias DISABLED, 모든 것은 fd, select poll epoll 비교

**개요**
- 버튼도 기다리고, PIR 도 기다리고, 0.5초마다 LED 도 깜빡이고, 키보드 명령도 받고 싶다. Linux 는 버튼 · 타이머 · 키보드 · 네트워크가 **모두 fd** 라서 **epoll 하나**로 함께 기다릴 수 있다. 쓰레드 없는 이벤트 루프를 만든다.

**핵심 용어**
- **이벤트 루프**: 여러 fd 를 등록해 두고, **준비된 fd 만** 처리하는 루프 하나. 쓰레드가 없어 **경쟁이 없고 mutex 가 필요 없다** (Linux Day5 와 대비).
- **epoll**: fd 를 **처음 한 번 등록**(`epoll_ctl`)하고 `epoll_wait` 가 **준비된 것만** 돌려줌. `data.u32` 같은 태그로 누구의 이벤트인지 구분.
- **timerfd**: 정해진 시간이 되면 **읽을 수 있는 상태**가 되는 타이머 fd. read 하면 그동안 **만료된 횟수(uint64)**. **읽어야** 다시 대기 상태 — 안 읽으면 `epoll_wait` 가 **계속 즉시 반환**.
- **PIR 배선**: VCC 는 **3V3**(5V 면 OUT 이 3.3V 를 넘을 수 있음), OUT → GPIO6, 움직임 1 · 조용 0, **bias DISABLED**. 전원 후 **30~60초 안정화**(바로 정확하지 않음).

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `epoll_create1(0)` | epoll 인스턴스 |
| `epoll_ctl(ep, EPOLL_CTL_ADD, fd, &e)` | `e = {.events = EPOLLIN, .data.u32 = tag}` 로 감시 목록 등록 |
| `epoll_wait(ep, evs, max, -1)` | 준비된 fd 수 반환, `evs[i].data.u32` 로 구분 |
| `timerfd_create(CLOCK_MONOTONIC, 0)` | 타이머 fd |
| `timerfd_settime(tfd, 0, &its, NULL)` | `it_value` = 첫 만료, `it_interval` = 주기 |
| `read(tfd, &exp, 8)` | 만료 횟수 (uint64_t) |

| | select | poll | epoll |
|---|---|---|---|
| fd 개수 | 1024 제한 | 제한 없음 | 제한 없음 |
| 매번 넘기는 것 | fd 집합 전체 | 배열 전체 | **처음 한 번 등록** |
| 돌려주는 것 | 전체 다시 검사 | 전체 다시 검사 | **준비된 것만** |
| 이 과목 | — | 06 · react | 07 · Day 5 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `07_evloop.c` | ① 버튼(GPIO5, falling, 10ms) 누른 횟수 ② PIR(GPIO6, both, bias 끔) 움직임 시작/끝 로그 ③ timerfd 0.5초 → LED 토글 ④ stdin `s` = 통계, `q` = 종료 — 네 fd 를 epoll 하나로 |

**핵심 내용**

#### epoll 이벤트 루프
> 여러 fd 를 한 번 등록하고, 준비된 것만 받는다
- **예시 출력** → `[PIR ] 움직임 감지` / `[버튼] 1 번째` / `[버튼] 2 번째` / `[PIR ] 조용해짐` / `s` → `[통계] 버튼 2 · 움직임 1 · 타이머 23` / `q` → 종료
- **그동안 LED** → 0.5초마다 계속 깜빡임 (버튼을 기다리느라 멈추지 않음)
- **버튼과 PIR 이 동시에** → `epoll_wait` 반환값 = **2**
- **준비 순서** → 라인 요청 (버튼·PIR·LED) → timerfd 만들고 주기 설정 → `epoll_create1` 후 fd 4개 등록 → `epoll_wait` 루프 → 정리
- **poll vs epoll** → fd 가 몇 개뿐이면 poll 로 충분, 등록과 대기를 나누거나 fd 가 늘어날 때 epoll

**기본 사용형태**
```c
int tfd = timerfd_create(CLOCK_MONOTONIC, 0);
struct itimerspec its = { .it_interval = {0, 500000000L}, .it_value = {0, 500000000L} };
timerfd_settime(tfd, 0, &its, NULL);
int ep = epoll_create1(0);
struct epoll_event e = { .events = EPOLLIN, .data.u32 = TAG_TIMER };
epoll_ctl(ep, EPOLL_CTL_ADD, tfd, &e);
struct epoll_event evs[4];
int n = epoll_wait(ep, evs, 4, -1);
for (int i = 0; i < n; i++)
    if (evs[i].data.u32 == TAG_TIMER) { uint64_t exp; read(tfd, &exp, sizeof exp); /* LED 토글 */ }
```

**연관 문제**
<<QA 6>>

### 7교시 · 미니 프로젝트 설계 — 반응속도 게임

**키워드:** 반응속도 게임, 상태 머신 READY FOUL GO RESULT TIMEOUT, 무작위 지연 2000 + rand() % 3000, 부정 출발, 커널 타임스탬프 vs 깨어난 뒤 시각, wait_press, drain, beep, 액티브 부저 GPIO13, 남은 시간 재계산

**개요**
- MCU Day1 과 같은 규칙의 반응속도 게임을 Pi 에서 설계한다. 규칙을 **상태 머신**으로 표현하고, 반응 시간을 **커널 타임스탬프**로 재는 이유를 이해한 뒤, 핵심 함수 `wait_press()` · `drain()` 의 골격을 짠다.

**핵심 용어**
- **게임 규칙**: "준비" 후 **2~5초 무작위** 대기 → LED 가 켜지면 버튼 A → 반응 시간(ms) + 부저 짧게. LED 전에 누르면 **부정 출발**(부저 길게, 무효). 3초 안에 안 누르면 시간 초과. 5라운드 뒤 평균·최고, `react_log.csv` 에 한 줄.
- **액티브 부저**: GPIO 를 1 로 만들면 **스스로** 소리를 냄 (GPIO13).
- **상태 머신**:

| 상태 | 하는 일 | 다음 |
|---|---|---|
| READY | "준비", 쌓인 이벤트 버림, 무작위 지연 | 지연 중 누름 → **FOUL** · 지연 끝 → GO |
| FOUL | 부저 0.8초 | 다음 라운드 |
| GO | LED 켜고 `t_on` 기록, 최대 3초 대기 | 누름 → RESULT · 3초 → TIMEOUT |
| RESULT | `ms = (ts − t_on) / 1e6`, 부저 0.1초 | 다음 라운드 |
| TIMEOUT | "시간 초과" | 다음 라운드 |

**새로 나온 함수/명령** — 설계할 함수

| 함수 | 왜 필요한가 |
|---|---|
| `drain()` | 이전 라운드에서 뗄 때 생긴 이벤트·바운싱이 다음 라운드의 **부정 출발**로 잡히지 않게 |
| `wait_press(timeout_ms, &ts, &who)` | timeout 동안 falling 대기 — **1 = 눌림, 0 = 시간 초과, -1 = Ctrl+C**. **남은 시간을 매번 다시 계산**해 poll 에 넘김 (rising 으로 깨어나도 전체 대기가 늘지 않게) |
| `beep(ms)` | 부저 켜기 → sleep → 끄기 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 7-5 골격 | `drain()` → `wait_press(2000 + rand() % 3000)` 이 1 이면 부정 출발 → LED 켜고 `t_on = now_ns()` → `wait_press(3000)` 이 1 이면 `(ts - t_on) / 1000000` ms |

**핵심 내용**

#### wait_press 하나로 두 상황
> "지연 중 누름" 과 "LED 후 누름" 을 같은 함수로 — 차이는 타임아웃과 반환값의 의미
- **정상 라운드** → `drain()` → 무작위 지연 동안 `wait_press` — 시간 초과 → LED 켜고 `t_on` → `wait_press(3000)` — 눌림 → ms 계산 · 출력 → 부저 0.1초

#### 반응 시간 재는 법
> 깨어난 시각이 아니라 커널이 찍은 엣지 시각으로
- ⚠ **깨어난 뒤 `now_ns()` − `t_on`** → 스케줄러가 깨워 주기까지의 지연이 더해짐 (수십 µs ~ ms, 부하 시 더 큼)
- **`gpiod_edge_event_get_timestamp_ns()` − `t_on`** → 인터럽트 순간에 찍힘, **부하와 무관**
- **바로 뺄 수 있는 이유** → 둘 다 CLOCK_MONOTONIC

**기본 사용형태**
```c
drain();
int delay = 2000 + rand() % 3000;                    // 2~5초
if (wait_press(delay, &ts, &who) == 1) { beep(800); /* FOUL */ }
else {
    set_line(LED, 1); long long t_on = now_ns();
    if (wait_press(3000, &ts, &who) == 1) printf("%lld ms\n", (ts - t_on) / 1000000);
    else printf("시간 초과\n");
    set_line(LED, 0);
}
```

**연관 문제**
<<QA 7>>

### 8교시 · 미니 프로젝트 구현

**키워드:** react.c, 출력 요청 {LED, BUZ}, 입력 요청 {BTN_A, BTN_B}, 커널 디바운스 5ms, 평균 최고 부정 출발, react_log.csv append, strftime, 2인 대전 -2, line offset 으로 승자, 한 큐 시간 순서, CPU 부하 yes, killall yes, 도전 과제

**개요**
- `react.c` 를 완성해 반응 시간·부정 출발·기록 저장을 구현하고, **2인 대전**에서 이벤트의 **line offset** 으로 승자를 가린다. CPU 부하 중에도 측정이 정확한지(커널 타임스탬프 덕분) 확인한다.

**핵심 용어**
- **2인 대전**: 두 버튼을 **한 요청에 offsets {5, 6}** 으로 묶어 이벤트가 **한 큐에 시간 순서대로** 들어오게 → 먼저 들어온 falling 의 **line offset** 이 승자. (서로 다른 프로그램에서 요청하는 것이 아님)
- **append 로그**: `fopen("react_log.csv", "a")` → 실행할 때마다 **기존 내용 뒤에 한 줄 추가**. 열: 시각, 모드, 성공 횟수, 평균 ms, **최고 ms**, 부정 출발.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `./react` / `./react 10` / `./react -2` | 1인 5라운드 / 10라운드 / 2인 대전 |
| `column -t -s, react_log.csv` | CSV 를 표로 |
| `yes > /dev/null &` ×4 / `killall yes` | CPU 부하 걸기 / **반드시 정리** |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `react.c` | `srand(now_ns())` → 출력 {LED 17, BUZ 13}·입력 {5(, 6)} 요청(falling, 5ms 디바운스) → 라운드마다 drain → 무작위 지연 wait_press(부정 출발 처리) → LED·`t_on` → wait_press(3000) → 커널 타임스탬프로 ms, 최고·평균 갱신 → 결과 출력 → `react_log.csv` 에 시각·모드·통계 append → 반납 |

**핵심 내용**

#### 결과와 기록
> 성공 · 평균 · 최고 · 반칙을 출력하고 CSV 에 추가
- **예시** → `243 ms` / `부정 출발! (A)` / `211 ms` / `198 ms` / `226 ms` → **성공 4회 · 평균 219 ms · 최고 198 ms · 부정 출발 1회**
- **로그** → `2026-10-14 17:32:05 solo 4 219 198 1` 에서 **198 = 최고 기록**
- **2인** → `B 승 · 187 ms`, A 가 부정 출발하면 B 가 1점 → `A 2 : 3 B`
- **실행 순서** → `srand` → 출력·입력 요청 → 라운드 반복 → 결과 출력 → CSV 추가 → 반납

#### 부하 실험
> 커널 타임스탬프는 부하가 걸려도 정확하다
- **`yes` 4개로 CPU 를 채워도** → 기록이 거의 변하지 않음
- ⚠ `*ts = now_ns()` 로 바꾸면 → 기록이 늦어지고 들쭉날쭉
- **도전 과제** → 결과별 LED 색 (200ms 미만 초록, 300ms 미만 노랑, 그 이상 빨강), 최고 기록 `best.txt`, epoll 로 키보드 q 중단, now_ns 버전과 부하 중 평균·표준편차 비교

**기본 사용형태**
```c
unsigned int outs[2] = {LED, BUZ}, ins[2] = {BTN_A, BTN_B};
g_out = gpio_request_output("react", outs, 2);
g_in  = gpio_request_input("react", ins, duo ? 2 : 1, GPIOD_LINE_EDGE_FALLING, 5000);
/* wait_press 안에서 */
*ts  = gpiod_edge_event_get_timestamp_ns(ev);
*who = gpiod_edge_event_get_line_offset(ev);        // 2인: 누가 먼저
FILE *fp = fopen("react_log.csv", "a");             // 이어 쓰기
```

**연관 문제**
<<QA 8>>
