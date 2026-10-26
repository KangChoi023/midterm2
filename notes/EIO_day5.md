## Day 5 — 커널 드라이버와 종합 프로젝트

### 1교시 · 커널 드라이버로 DHT11 읽기

**키워드:** DHT11 µs 비트, High 27µs = 0 70µs = 1, 시작 신호 Low 18ms, 40비트 5바이트 체크섬, 사용자 공간 폴링 실패, 엣지 이벤트 버퍼 256, reconfigure_lines, dtoverlay=dht11,gpiopin=4, IIO Industrial I/O, /sys/bus/iio/devices, in_temp_input 밀리 단위, in_humidityrelative_input, EIO ETIMEDOUT 재시도, glob 장치 찾기, dht.h dht.c, 01_dht_poll, 02_dht_iio

**개요**
- DHT11 은 0 과 1 의 차이가 **약 40µs** 다. 사용자 공간에서 직접 읽으면 부하가 걸릴 때 비트를 놓쳐 실패하는 것을 실험으로 확인하고, **커널 드라이버(`dtoverlay=dht11`)** 가 만든 **IIO 파일**로 온습도를 읽는다. "µs 가 필요한 일은 커널에"가 Day 5 의 첫 결론.

**핵심 용어**
- **DHT11 프로토콜**: Pi 가 **Low 18ms 이상**(시작 신호) → Pi 가 입력으로 놓음(풀업 High, 20~40µs) → DHT11 응답 Low·High 각 80µs → **비트 × 40**(Low 50µs 뒤 High — **26~28µs = 0, 70µs = 1**) → 5바이트(습도 정수 · 소수 · 온도 정수 · 소수 · **체크섬**). 한 번 읽기 약 4ms.
- **체크섬**: `b[0] + b[1] + b[2] + b[3]` 의 하위 8비트 == `b[4]`. 한 비트라도 틀리면 걸린다.
- **사용자 공간이 실패하는 이유**: `get_value` 한 번이 시스템 콜이라 그 사이 에지를 늦게 봄 / **4ms 중 한 번이라도 CPU 를 빼앗기면 비트 여러 개를 통째로 놓침**(Day 3 jitter) / 출력 → 입력 방향 전환이 늦으면 응답 펄스를 놓침 / 엣지 80개가 4ms 에 몰려 커널 이벤트 버퍼 기본값(라인당 16개)이면 넘침.
- **MCU 는 왜 됐나**: ESP32 라이브러리는 읽는 동안 **인터럽트를 끄고** 다른 일을 안 한다 — Linux 사용자 프로그램은 그럴 수 없다.
- **IIO(Industrial I/O)**: 센서 값을 `/sys/bus/iio` 아래 파일로 내놓는 커널의 공통 틀(가속도·조도·ADC 도 같은 모양). 값은 **밀리 단위** — `in_temp_input` 24000 = 24.0°C, `in_humidityrelative_input` **41000 = 41%**.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `gpiod_line_request_reconfigure_lines(req, cfg)` | 요청한 라인을 출력 ↔ 입력으로 재설정 (시작 신호용) |
| `gpiod_request_config_set_event_buffer_size(rc, 256)` | 커널 이벤트 버퍼 늘리기 |
| `dtoverlay=dht11,gpiopin=4` (config.txt) + 재부팅 | 커널 DHT11 드라이버 |
| `dmesg \| grep -i dht` / `lsmod \| grep dht11` / `ls /sys/bus/iio/devices/` | 적용 확인 |
| `cat /sys/bus/iio/devices/iio:device0/in_temp_input` | 24000 |
| `glob("/sys/bus/iio/devices/iio:device*/in_humidityrelative_input", …)` | 장치 번호가 부팅마다 바뀔 수 있어 파일로 찾기 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `01_dht_poll.c` | (실패를 보려고 만든 예제) GPIO4 를 출력 Low 20ms → 입력으로 재설정 → 폴링(`get_value` + `now_ns`) 또는 `-e` 엣지 타임스탬프로 High 폭을 재서 40비트 해석, 체크섬 확인 — 2초 간격 N 회 성공률 출력 |
| `dht.h` / `dht.c` | `dht_open`(glob 으로 IIO 장치 찾기), `dht_read`(두 파일을 읽어 ÷1000, 실패하면 0.3초 뒤 **3번까지 재시도**) |
| `02_dht_iio.c` | 2초마다 `dht_read` → `24.0 °C 41 % (23 ms) 성공 1 / 1` |

**핵심 내용**

#### 사용자 공간에서 DHT11 직접 읽기
> µs 를 다투는 장치는 사용자 공간에서 자주 실패한다 — 커널 드라이버로
- **부하 없음** → 성공 **8 / 10** (대부분 되지만 가끔 틀림)
- **yes ×4 부하** → 성공 **1 / 10** (`비트 부족 (27 / 40)`, `체크섬 오류`)
- **`-e`** → 시각이 커널에서 찍혀 낫지만 시작 신호 · 방향 전환은 여전히 사용자 공간
- **한 번 읽기** → Pi 가 Low 18ms 이상 → Pi 가 놓음 (풀업 High) → DHT11 응답 Low·High 80µs → 40비트 → 체크섬
- **배선** → 데이터선이 Pi GPIO 로 바로 오므로 + 를 **3V3** (DHT11 은 3.3~5.5V)

#### 커널 드라이버 (IIO)
> overlay 로 켜면 파일 읽기만으로 온습도
- **`02_dht_iio` 두 번째 읽기 `(0 ms)`** → 2초 안에 다시 읽으면 드라이버가 저장해 둔 값
- **드라이버도 실패할 수 있다** (`Connection timed out` · `Input/output error`) → **0.3초 뒤 다시** (dht_read 재시도)
- **overlay 를 켠 뒤 `01_dht_poll`** → **busy** — 정상, GPIO4 는 커널 드라이버의 것
- ⚠ `iio:device0` 이 없으면 → overlay 줄 오타 · 재부팅 안 함 · 핀 번호

**기본 사용형태**
```c
/* dht.c */
for (int i = 0; i < 3; i++) {
    long t, h;
    if (read_long(d->dir, "in_temp_input", &t) == 0 && read_long(d->dir, "in_humidityrelative_input", &h) == 0) {
        *temp_c = t / 1000.0; *humi = h / 1000.0; d->fails = 0; return 0;
    }
    nanosleep(&(struct timespec){0, 300000000L}, NULL);    // 0.3 초 뒤 다시
}
d->fails++; return -1;
```
```bash
# config.txt [all] 아래: dtoverlay=dht11,gpiopin=4  → sudo reboot
cat /sys/bus/iio/devices/iio:device0/in_temp_input              # 24000
```

**연관 문제**
<<QA 1>>

### 2교시 · Device Tree overlay 직접 작성

**키워드:** Device Tree, compatible, overlay, .dts .dtbo, dtc -@, /plugin/, fragment, target <&gpio>, target-path "/", brcm,pins brcm,function brcm,pull, gpio-leds, gpio-keys, gpios = <&gpio 17 0>, active-low 1, linux,default-trigger heartbeat, linux,code 28 KEY_ENTER, debounce-interval, /sys/class/leds trigger brightness, /dev/input/event, struct input_event, EVIOCGNAME, overlay 내리기, 03_led_key

**개요**
- Linux 커널은 보드마다 다른 하드웨어를 **Device Tree** 라는 설명서로 받는다. Day 1 의 `pwm-2chan`·`disable-bt`, 1교시의 `dht11` 이 모두 overlay 조각이었다 — 오늘은 **직접 써서** GPIO17 LED 와 GPIO5 버튼을 **커널 장치**(LED 클래스 · input)로 만든다.

**핵심 용어**
- **compatible**: 노드에 맞는 **드라이버를 고르는 속성**. `compatible = "gpio-leds"` → gpio-leds 드라이버가 붙어 `/sys/class/leds/…` 생성.
- **overlay(.dtbo)**: 기본 `.dtb` 에 덧붙이는 조각. 부팅 때 펌웨어가 config.txt 의 `dtoverlay=` 를 적용 → 커널이 드라이버를 붙임 → 장치 파일 생성. **재부팅해야** 생긴다.
- **fragment**: `target = <&gpio>`(GPIO 컨트롤러 노드에 — 핀 5 를 입력·pull-up) / `target-path = "/"`(루트에 새 장치 노드 추가).
- **`gpios = <&gpio 17 0>`**: GPIO 컨트롤러 · 17번 · **0 = active-high**. `<&gpio 5 1>` 의 **1 = active-low**(누르면 0 = 눌림).
- **LED 트리거**: `linux,default-trigger = "heartbeat"` → 부팅하자마자 심장 박동처럼 깜빡임. `trigger` 파일에 `none`/`default-on`/`timer`/`heartbeat` 를 쓰면 **프로그램 없이** 커널이 동작.
- **gpio-keys**: 버튼 → **input 장치(키보드와 같은 종류)**. `linux,code = <28>` = **KEY_ENTER**, `debounce-interval = <20>` = 커널이 20ms 디바운스.
- **struct input_event**: `/dev/input/event*` 에서 읽는 구조체 (시각 · type · code · value). **value 1 = 누름, 0 = 뗌, 2 = 길게 눌러 반복**.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `dtc -@ -I dts -O dtb -o eio.dtbo eio-overlay.dts` (`make eio.dtbo`) | overlay 컴파일 |
| `sudo cp eio.dtbo /boot/firmware/overlays/` + config.txt `dtoverlay=eio` + reboot | 적용 |
| `ls /sys/class/leds/` / `cat …/eio:green/trigger` | `ACT PWR eio:green` / `[heartbeat]` |
| `echo timer \| sudo tee /sys/class/leds/eio:green/trigger` | 1초 깜빡임 — 커널이 알아서 |
| `echo 1 \| sudo tee …/brightness` | 켜기 (trigger 를 none 으로 한 뒤) |
| `cat /proc/bus/input/devices \| grep -A4 eio-keys` | `Handlers=kbd event0` |
| `ioctl(fd, EVIOCGNAME(sizeof name), name)` | input 장치 이름 묻기 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `eio-overlay.dts` | fragment@0: 핀 5 입력·pull-up / fragment@1: `eio-leds`(gpio-leds, GPIO17, `eio:green`, heartbeat) + `eio-keys`(gpio-keys, GPIO5 active-low, KEY_ENTER, 20ms) |
| `03_led_key.c` | `/dev/input` 에서 이름이 `eio-keys` 인 장치를 찾아 poll + `read(struct input_event)` — 버튼 A(KEY_ENTER, value 1)를 누를 때마다 LED `trigger` 를 none → default-on → timer → heartbeat 로 바꿈 (GPIO 를 직접 요청하지 않음, LED sysfs 쓰기라 sudo) |

**핵심 내용**

#### overlay 적용과 문제 해결
> dtc → 복사 → config.txt → 재부팅 → 장치 확인
- **순서** → `dtc` 로 .dtbo → overlays 폴더로 복사 → config.txt 에 `dtoverlay=eio` → `sudo reboot` → `/sys/class/leds` · `/dev/input` 확인
- ⚠ `eio:green` 이 안 생기고 오류도 없음 → **compatible 오타** · 복사 안 함 · config.txt 오타 · 재부팅 안 함 → `dmesg | grep -i overlay`
- ⚠ `dtc: … syntax error` → 세미콜론 · 중괄호 (줄 번호가 나옴)
- **`gpioset 17=1` 이 busy** → 정상 (커널 LED 드라이버 소유)

#### libgpiod vs 커널 드라이버
> 핀을 프로그램이 가지느냐, 커널이 부팅부터 가지느냐

| | Day 2 (libgpiod) | 오늘 (overlay + 커널 드라이버) |
|---|---|---|
| 핀 소유 | 내 프로그램 (요청·반납) | **커널 — 부팅부터 계속** |
| 디바운스 | 내 코드 / 요청 설정 | DT 의 `debounce-interval` |
| 깜빡임 | 내 루프가 켜고 끔 | **trigger 한 줄 — 프로그램이 없어도** |
| 다른 프로그램 | 버튼 이벤트를 각자 요청 | `/dev/input` 을 여러 프로그램이 같이 읽음 |

- ⚠ **4교시 전에 반드시 내리기** → 종합 프로젝트는 GPIO5·17 을 libgpiod 로 직접 씀 → `dtoverlay=eio` 앞에 `#` → 재부팅 → `gpioinfo` consumer 비어 있는지

**기본 사용형태**
```dts
/dts-v1/;
/plugin/;
/ {
    compatible = "brcm,bcm2711";
    fragment@1 {
        target-path = "/";
        __overlay__ {
            eio-leds { compatible = "gpio-leds";
                led-green { label = "eio:green"; gpios = <&gpio 17 0>; linux,default-trigger = "heartbeat"; }; };
            eio-keys { compatible = "gpio-keys";
                button-a { label = "Button A"; gpios = <&gpio 5 1>; linux,code = <28>; debounce-interval = <20>; }; };
        };
    };
};
```
```c
struct input_event ev;
if (read(kfd, &ev, sizeof ev) == sizeof ev && ev.type == EV_KEY && ev.code == KEY_ENTER && ev.value == 1)
    write_str("trigger", "timer");
```

**연관 문제**
<<QA 2>>

### 3교시 · 1602A LCD 4비트 드라이버

**키워드:** 1602A LCD, HD44780, RS E D4~D7, RW GND, V0 대비, 니블 두 번, E 펄스 High → Low, 명령 0x01 0x06 0x0C 0x28, 0x80 | 주소, 2행 0x40 → 0xC0, CGRAM 0x40 | slot<<3, 초기화 0x3 세 번 0x2, 최소 시간만 있는 장치, 사용자 공간 드라이버, 불투명 포인터 typedef struct lcd lcd_t, lcd_printf 16칸 공백 채움, 사용자 문자 1~7, lcd.h lcd.c, 04_lcd_demo, hd44780-lcd overlay

**개요**
- MCU Day3 5교시와 같은 결선의 1602A LCD 를 이번에는 **라이브러리 없이** libgpiod 로 직접 구동하는 `lcd.c` 를 만든다. 니블 두 번 · E 펄스 · 초기화 순서를 구현하고, **왜 LCD 는 사용자 공간에서 되고 DHT11 은 안 되는지** 설명한다.

**핵심 용어**
- **연결**: RS → GPIO12, **RW → GND**, E → GPIO16, D4~D7 → GPIO20·21·26·22, VDD 5V, V0 가변저항 가운데, A → 220Ω → 5V. LCD 는 2.2V 이상을 High 로 읽어 Pi 3.3V 로 동작, **RW 를 GND** 로 두면 LCD 가 Pi 쪽으로 **5V 를 내보내지 않는다**.
- **4비트 전송**: 한 바이트 = **위 니블 → E 내림 → 아래 니블 → E 내림** → 50µs 쉼(명령 실행 ≥ 37µs). E 가 **High → Low 로 떨어지는 순간** LCD 가 읽는다.
```text
'A' = 0x41 = 0100 0001, RS = 1
① D7..D4 = 0100 → E ‾\_   ② D7..D4 = 0001 → E ‾\_   ③ 50 µs
```

| 명령 | 뜻 |
|---|---|
| `0x01` | 지우기 — **1.52ms 이상** 기다림 |
| `0x06` | 쓰면 커서 오른쪽으로 |
| `0x0C` | 화면 켬 · 커서 숨김 |
| `0x28` | 4비트 · 2줄 · 5×8 |
| `0x80 \| 주소` | 1행 0x00~0x0F · 2행 0x40~0x4F → **2행 첫 칸 = 0xC0** |
| `0x40 \| (번호 << 3)` | CGRAM — 사용자 문자 5×8 비트맵 8바이트 |

- **초기화**: 전원 후 **40ms 이상** → 니블 **0x3 세 번**(4.5ms · 150µs 간격, 8비트 모드라고 세 번 말해 상태를 맞춤) → **0x2**(4비트 모드로) → `0x28 · 0x08 · 0x01 · 0x06 · 0x0C`.
- **최소 시간만 있는 장치**:

| | DHT11 | HD44780 LCD |
|---|---|---|
| 타이밍을 정하는 쪽 | **센서** — 정해진 순간에 비트를 보냄 | **Pi** — E 를 내릴 때 읽음 |
| 시간 조건 | 26µs 와 70µs 구분 (위아래 모두) | **모두 최소 시간** (≥ 450ns, ≥ 37µs, ≥ 1.52ms) |
| 늦어지면 | 비트를 놓침 → 실패 | **조금 느려질 뿐 → 성공** |
| 그래서 | 커널 드라이버 | libgpiod 사용자 공간 드라이버로 충분 |

- **불투명 포인터**: `lcd.h` 에는 `typedef struct lcd lcd_t;` 만, 내용은 `lcd.c` 안에만 → 쓰는 쪽은 핀 번호를 몰라도 된다(C Day4 모듈화).

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `lcd_open()` / `lcd_close(l)` / `lcd_release(l)` | 열기+초기화 / 지우고 반납 / 화면 그대로 반납 |
| `lcd_cmd` / `lcd_putc` / `lcd_goto(l, col, row)` / `lcd_puts` | RS 0 / RS 1 / 커서 / 문자열 |
| `lcd_printf(l, row, fmt, …)` | 그 행을 **16칸 공백으로 채워 덮어쓰기** (vsnprintf) |
| `lcd_custom(l, slot, bitmap8)` | 사용자 문자 **1~7** → 문자열 속 `"\x01"` (0번은 문자열 끝 `'\0'` 이라 안 씀) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `lcd.h` / `lcd.c` | 6개 라인 출력 요청 → 초기화, `nibble`(D4~D7 + E 펄스), `send`(RS + 위·아래 니블 + 50µs), `lcd_printf` 16칸 채움, `lcd_custom` CGRAM |
| `04_lcd_demo.c` | `°`(1번)·하트(2번) 사용자 문자를 만들고 윗줄 `Hello, Pi 4!` + 오른쪽 끝 하트, 아랫줄 `%4d` 카운터와 `25.0 °C` |

**핵심 내용**

#### LCD 초기화와 빌드
> 정해진 순서로 4비트 모드에 들어간다
- **`lcd_open` 순서** → 6개 라인 출력 요청 → 40ms 이상 → 니블 0x3 세 번 → 니블 0x2 → 0x28 → 0x08 → clear → 0x06 → 0x0C
- **빌드** → `make 04_lcd_demo` → `04_lcd_demo.c` 와 `lcd.o` 를 함께 링크

#### LCD 문제 해결
> 증상 → 원인

| 증상 | 원인 |
|---|---|
| 백라이트만 켜지고 글자 없음 | **대비(V0)** — 가변저항을 천천히 (가장 흔함) |
| 윗줄에 검은 네모 16개 | 초기화 실패 — RS·E·D4~D7 배선, 4비트 순서 |
| 이상한 글자 | D4~D7 순서가 바뀜 / **니블 순서(위 → 아래)** |
| 1000 다음 999 가 9990 | 덮어쓴 칸만 바뀜 → `lcd_printf` 는 16칸을 공백으로 채운다 |

#### 무엇을 커널에 맡기나
> 타이밍 요구로 정한다
- **(시연) `dtoverlay=hd44780-lcd,pin_rs=12,…`** → echo 한 줄이면 되지만, 사용자 문자 · 배치는 우리 모듈이 더 자유롭다

**기본 사용형태**
```c
static void nibble(lcd_t *l, uint8_t n) {
    for (int i = 0; i < 4; i++) l->v[D4 + i] = (n >> i) & 1;
    l->v[E] = 1; apply(l); us_sleep(1);
    l->v[E] = 0; apply(l); us_sleep(1);          // 떨어지는 순간 LCD 가 읽음
}
static void send(lcd_t *l, uint8_t b, int rs) { l->v[RS] = rs; nibble(l, b >> 4); nibble(l, b & 0x0F); us_sleep(50); }
void lcd_goto(lcd_t *l, int col, int row) { lcd_cmd(l, 0x80 | ((row ? 0x40 : 0x00) + col)); }
lcd_custom(l, 1, DEG);
lcd_printf(l, 1, "%4d 25.0\x01" "C", n);
```

**연관 문제**
<<QA 3>>

### 4교시 · 종합 프로젝트 설계

**키워드:** envmon, 스마트 환경 모니터, sensor proximity control display main 쓰레드, 공유 상태 env_t, mutex + 조건 변수, 잠그고 복사 후 풀고 쓴다, 센서 읽는 동안 잠그지 않음, 실패는 값으로 -1 --, pthread_cond_signal 즉시 다시 그리기, 팬 히스테리시스 28/27, 접근 30cm, 버튼 짧게 화면 길게 음소거, 상태 LED, 모듈 인터페이스, 핀 점검, eio overlay 내리기, sonar.h

**개요**
- Day 1~5 와 Linux 과목의 기법을 모은 **envmon(스마트 환경 모니터)** 을 설계한다. 쓰레드 다섯 개의 역할 · 동작 규칙 · 화면을 정하고, **공유 상태 구조체 하나 + mutex + 조건 변수** 규칙과 모듈 인터페이스·핀을 점검한다.

**핵심 용어** — 쓰레드와 규칙

| 쓰레드 | 하는 일 | 주기 | 쓰는 것 |
|---|---|---|---|
| **sensor** | 온습도 · 조도 측정 | **2초** | IIO(DHT11) · `/dev/i2c-1`(PCF8591 IN0) |
| proximity | 거리 — 30cm 안이면 접근 경고 | 약 0.3초 | HC-SR04 엣지 타임스탬프 |
| control | 버튼 이벤트 · 판단 · 릴레이 · 부저 · LED | 50ms | libgpiod 이벤트 · 출력 |
| display | LCD 화면 · 1분마다 CSV | 0.5초 (또는 즉시) | lcd.c · 파일 I/O |
| main | 시작 · Ctrl+C/SIGTERM 기다림 · 정리 | | |

| 조건 | 동작 |
|---|---|
| 온도 ≥ 28°C | 릴레이(팬) ON — **≤ 27°C** 가 되어야 OFF (히스테리시스) |
| 거리 < 30cm | LCD `!! APPROACH !!` · 경보 LED(27) · 부저 삑(0.5초마다) |
| 버튼 A 짧게 | 화면 전환: 온습도 → 조도 → 거리 → 상태 |
| 버튼 A 길게 (1초) | 부저 음소거 켜기·끄기 |
| 상태 LED(17) | 정상 1초에 한 번 · 센서 오류가 있으면 빠르게 |

- **공유 상태 규칙** (Linux Day5):
  - **잠그고 복사한 뒤 풀고 쓴다** (`env_t c = S.d;`) — LCD 출력(수 ms) 동안 다른 쓰레드를 막지 않게.
  - **센서를 읽는 동안에는 잠그지 않는다** — DHT11 수십 ms, 초음파 최대 180ms.
  - **실패는 값으로 표시**(-1, `th_ok = 0`) — 쓰레드가 죽지 않고 화면에 `--`.
  - 화면 전환·접근은 **`pthread_cond_signal`** — display 가 0.5초를 다 기다리지 않고 **바로** 그림.

**새로 나온 함수/명령** — 모듈 인터페이스

| 헤더 | 주요 함수 | 만든 날 |
|---|---|---|
| `gpio_common.h` | `gpio_request_output/input` · `g_stop` · `now_ns` · `sleep_ms` | Day 2 |
| `dht.h` | `dht_open` · `dht_read(&t, &h)` | Day 5 1교시 |
| `pcf8591.h` | `pcf_open` · `pcf_read(fd, 0)` | Day 4 |
| `sonar.h` | `sonar_open(23, 24)` · `sonar_median(s, 3)` (−1 = 실패) | Day 3 → 모듈로 |
| `lcd.h` | `lcd_open` · `lcd_printf(l, row, …)` · `lcd_custom` | Day 5 3교시 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 4-2 공유 상태 | `static struct { pthread_mutex_t mx; pthread_cond_t cv; env_t d; } S;` — 모든 쓰레드가 이 구조체 하나만 주고받음 |
| 4-4 핀 점검 | I2C 2·3 / DHT11 4(커널) / 버튼 5 / LCD 12·16·20·21·26·22 / 부저 13 / LED 17·27 / HC-SR04 23·24 / 릴레이 25 — 겹치지 않는지, `grep -E "^dtoverlay" config.txt`(**dht11 은 있고 eio 는 없어야**), `gpioinfo \| grep -v unused` |

**핵심 내용**

#### 배선 전원 구분
> 5V 장치와 3V3 장치를 나눈다
- **5V** → 릴레이 · HC-SR04 · LCD
- **3V3** → PCF8591 · DHT11 · 조도 분압
- ⚠ HC-SR04 Echo **분압** (Day 3) 필수
- **배선 변경** → `sudo poweroff` 후

#### 팀 역할과 display 루프
> 센서 쪽 · 사용자 쪽으로 나누고, 화면은 복사 후 그린다
- **센서 쪽** → sensor · proximity, dht · sonar 모듈 → 6교시 오류 주입 시험
- **사용자 쪽** → LCD 배선 · 모듈 · 화면 → control · display · CSV
- **display 한 바퀴** → mutex 잠그고 `S.d` 복사 → 풀기 → LCD 두 줄 → 1분 지났으면 CSV → `cond_timedwait` 0.5초

**기본 사용형태**
```c
static struct { pthread_mutex_t mx; pthread_cond_t cv; env_t d; } S =
    { PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, { .light = -1, .cm = -1 } };
/* display */
pthread_mutex_lock(&S.mx); env_t c = S.d; pthread_mutex_unlock(&S.mx);   // 잠근 채 복사 → 풀고 그린다
/* control: 화면 전환 */
pthread_mutex_lock(&S.mx); S.d.screen = (S.d.screen + 1) % 4; pthread_cond_signal(&S.cv); pthread_mutex_unlock(&S.mx);
```

**연관 문제**
<<QA 4>>

### 5교시 · 종합 프로젝트 구현 (1)

**키워드:** 센서 하나씩 먼저, 단계별 확인, sonar.c 모듈 struct sonar, envmon.c, sensor_thread, proximity_thread, 상태가 바뀔 때만 로그, sleep_ms(100) × 20, 실패 표시 T --.- Light -- Dist ---.-, 여러 .c 컴파일·링크, Makefile envmon: envmon.o lcd.o dht.o sonar.o, %.o: %.c, 바뀐 것만 다시

**개요**
- envmon 을 바로 돌리지 않고 **센서를 하나씩 확인한 뒤** 통합한다. Day 3 의 초음파 코드를 `sonar.c` 모듈로 옮기고, sensor · proximity 쓰레드를 구현한 뒤 **여러 .c 파일을 따로 컴파일해 한 번에 링크**한다.

**핵심 용어**
- **단계별 확인**: `./02_dht_iio`(온습도) → `01_pcf_read`(조도) → `09_sonar`(거리) → `./04_lcd_demo`(LCD) → `make envmon && ./envmon`. 하나씩 안 하면 화면이 안 나올 때 **원인이 다섯 군데** — 한 군데로 좁히려고.
- **모듈화(sonar.c)**: Day 3 의 `measure_cm`·`median_cm` 을 옮기고 **전역 변수를 `struct sonar` 안으로**.
- **상태가 바뀔 때만 로그**: DHT11 선을 뽑아도 2초마다 쌓이지 않고 "읽기 실패" **한 줄**, 꽂으면 "복구" 한 줄.
- **나눠 자기**: sensor 쓰레드가 2초를 `sleep_ms(100) × 20` 으로 → Ctrl+C·SIGTERM 에 **0.1초 안에 반응**.
- **실패 표시**: DHT11 → `T --.-` / PCF8591 → `Light --` / HC-SR04 → `Dist ---.-`. HC-SR04 요청에 실패해도 **종료하지 않고 계속**.

**새로 나온 함수/명령** — 빌드

```make
CFLAGS = -Wall -Wextra -O2 -g $(shell pkg-config --cflags libgpiod 2>/dev/null)
LDLIBS = $(shell pkg-config --libs libgpiod 2>/dev/null) -lpthread -lm
envmon: envmon.o lcd.o dht.o sonar.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
%.o: %.c *.h
	$(CC) $(CFLAGS) -c -o $@ $<
eio.dtbo: eio-overlay.dts
	dtc -@ -I dts -O dtb -o $@ $<
```

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `sonar.c` | `sonar_open(trig, echo)` 로 Trig 출력·Echo(BOTH 엣지, bias 끔) 요청, 찌꺼기 이벤트 버림 → 측정 → 중앙값(qsort) |
| `envmon.c` (sensor · proximity) | sensor: 2초마다 DHT11(IIO)·PCF8591 IN0 조도를 읽어 mutex 안에서 `S.d` 갱신, 상태가 바뀔 때만 로그 / proximity: 0.3초마다 중앙값 3회 거리 → `near = cm > 0 && cm < 30`, 접근 시작 시 `cond_signal` |

**핵심 내용**

#### 여러 파일 빌드
> 모듈마다 `-c` 로 .o, 마지막에 한 번 링크 — 바뀐 것만 다시
- **`make envmon`** → `envmon.c → envmon.o`, `lcd.c → lcd.o`, `dht.c → dht.o`, `sonar.c → sonar.o` 각각 `-c` → `-o envmon … -lgpiod -lpthread -lm` 한 번 링크
- **lcd.c 만 고치면** → lcd.o 와 envmon 링크만 다시 (MCU Day2 헤더 분리, Linux Day1 mtime 비교)

#### 단위 시험
> 기능마다 눈으로 확인
- **LCD 첫 화면** → `T 24.0°C H 41% / Fan OFF Mute N`
- **DHT11 을 손으로 30초 감싸면** → 천천히 오름 (1°C 단위)
- **손을 20cm 앞에** → `!! APPROACH !!` · 경보 LED · 부저, 로그 `[10:21:19] 접근 감지`
- **버튼 짧게 3번** → 조도 → 거리 → 상태 화면

**기본 사용형태**
```c
static void *sensor_thread(void *arg) {
    dht_t d; int have_dht = dht_open(&d) == 0;
    while (!g_stop) {
        double t, h; int ok = have_dht && dht_read(&d, &t, &h) == 0;     // 잠그지 않고 읽기
        pthread_mutex_lock(&S.mx); S.d.t = ok ? t : -1; S.d.th_ok = ok; pthread_mutex_unlock(&S.mx);
        if (ok != was_dht) log_msg("DHT11 %s", ok ? "복구" : "읽기 실패 — 선 확인");   // 바뀔 때만
        was_dht = ok;
        for (int i = 0; i < 20 && !g_stop; i++) sleep_ms(100);           // 2초를 나눠 자기
    }
    return NULL;
}
```

**연관 문제**
<<QA 5>>

### 6교시 · 종합 프로젝트 구현 (2)

**키워드:** control_thread, 버튼 짧게 길게 1초, falling rising 타임스탬프 차이, 커널 디바운스 20ms, 팬 히스테리시스 28/27, 경보 LED near, 부저 near && !mute && (ms % 500) < 100, 상태 LED 1초/빠르게, 출력 네 개 한 요청 set_values, display 화면 네 장, !! APPROACH !! 우선, 영어 약어, 사용자 문자 °, CSV 1분 fopen("a") fclose, pthread_cond_timedwait

**개요**
- control(버튼 · 판단 · 출력)과 display(화면 네 장 · CSV) 쓰레드를 완성한다. 버튼 짧게/길게(Day 2 5교시), 히스테리시스 팬, 거리 경보, LCD 화면과 1분마다 CSV 기록.

**핵심 용어**
- **짧게/길게**: 버튼을 엣지 이벤트 + 커널 디바운스 **20ms** 로 요청, falling(누름) 타임스탬프를 기억했다가 rising(뗌)에서 빼면 누른 시간 → **1초 이상이면 음소거 전환**(1.2초 → 음소거), 아니면 화면 전환 + `cond_signal`.
- **판단과 출력 (50ms 마다)**:

| 출력 | 규칙 |
|---|---|
| 릴레이 (GPIO25) | t ≥ 28 에서 1, **t ≤ 27 에서 0, 그 사이는 그대로** (27.5°C 에서 켜져 있으면 계속 켜짐) |
| 경보 LED (GPIO27) | `near` |
| 부저 (GPIO13) | `near && !mute && (ms % 500) < 100` → 0.5초마다 0.1초 삑 |
| 상태 LED (GPIO17) | 정상 `ms % 1000 < 100`, 오류 `ms % 250 < 125` |

  출력 네 개는 **한 요청(OUTS[4])** 으로 `set_values` 한 번에 — 시스템 콜 한 번에 동시에 (Day 2 2교시).
- **기준 하나(28°C)** 로만 판단하면 잡음에 **릴레이가 떤다** → 히스테리시스.
- **화면 네 장**:

| screen | 1행 | 2행 |
|---|---|---|
| 0 | `T 24.0°C H 41%` | `Fan OFF Mute N` |
| 1 | `Light  64 %` | 막대 (0xFF 칸) |
| 2 | `Dist   42.0 cm` | near < 30 cm |
| 3 | `up 1234 s` | `E D0 P0 S3` (오류 횟수) |
| 접근 시 | `!! APPROACH !!` | `dist 21.4 cm` — **화면을 넘겨도 경고가 우선** |

  LCD 에는 **한글이 없다** → 영어 약어로 16칸. `°` 는 사용자 문자 1번(`\x01`).
- **CSV**: 1분마다 `2026-10-19 10:31:00,24.0,41,64,42.0,0,0`(시각, 온도, 습도, 조도 %, 거리 cm, 팬, 접근). `fopen("a") → 한 줄 → fclose` — **매번 닫으면 전원이 갑자기 꺼져도 앞 줄들은 남는다**. 거리 실패는 −1.0.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `pthread_cond_timedwait(&S.cv, &S.mx, &dl)` | 0.5초 기다리되 signal 이 오면 바로 깨어남 (dl = 절대 시각) |
| `tail -f envmon.csv` | 다른 창에서 기록 확인 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `envmon.c` (control) | 버튼 이벤트로 짧게/길게 처리 → 50ms 마다 복사한 상태로 팬 히스테리시스·경보·부저·상태 LED 를 계산해 `set_values` |
| `envmon.c` (display) | 복사 → 화면 번호(또는 접근)에 따라 두 줄 → `lcd_printf` → 1분마다 CSV → `cond_timedwait` 0.5초 |

**핵심 내용**

#### 버튼 → 화면 전환 경로
> 엣지 이벤트 → 짧은 누름 판정 → cond_signal → 즉시 다시 그리기
- **순서** → 커널이 엣지 이벤트 기록 → falling 타임스탬프 기억 → rising 에서 차이 < 1초 → `screen++` + `cond_signal` → display 가 즉시 깨어나 그림
- **거리 20cm + 음소거 켜짐** → **경보 LED 켜짐, 부저 조용**

#### 통합 시험
> 팬 · 음소거 · 경고 화면 우선
- **DHT11 을 감싸 28°C 넘기기** (또는 `FAN_ON` 을 잠시 25 로) → 릴레이 딸깍 · `Fan ON`
- **버튼 1초 이상** → `Mute Y`
- **손 접근 중 버튼 짧게** → 경고 화면 유지

**기본 사용형태**
```c
/* control: 판단과 출력 */
if (!fan && c.t >= 28.0) fan = 1; else if (fan && c.t >= 0 && c.t <= 27.0) fan = 0;
enum gpiod_line_value v[4] = {
    (ms % (err ? 250 : 1000)) < (err ? 125 : 100),     // 상태 LED
    c.near,                                            // 경보 LED
    fan,                                               // 릴레이
    c.near && !c.mute && (ms % 500) < 100,             // 부저
};
gpiod_line_request_set_values(g_out, v);
/* display: 0.5초 또는 신호 */
pthread_mutex_lock(&S.mx); pthread_cond_timedwait(&S.cv, &S.mx, &dl); pthread_mutex_unlock(&S.mx);
```

**연관 문제**
<<QA 6>>

### 7교시 · 서비스화와 장애 대응

**키워드:** systemd, unit 파일 envmon.service, [Unit] [Service] [Install], Type=simple, User=pi, WorkingDirectory, ExecStart, KillSignal=SIGTERM, TimeoutStopSec, Restart=on-failure, RestartSec=3, WantedBy=multi-user.target, daemon-reload, enable --now, systemctl status stop start restart, journalctl -u -f, kill -9 재시작, setvbuf _IOLBF, 장애 주입, disable --now

**개요**
- envmon 을 **systemd 서비스**로 만들어 부팅하면 저절로 실행되게 한다. `systemctl stop` 의 SIGTERM 으로 정리되는지, `kill -9` 같은 비정상 종료에서 **자동 재시작**되는지, 센서 선을 뽑아도 죽지 않고 복구되는지 시험한다.

**핵심 용어**
- **systemd unit**: 서비스를 설명하는 파일. `/etc/systemd/system/` 에 두고 **`daemon-reload`** 로 읽힌다.

| 설정 | 뜻 |
|---|---|
| `User=pi` | **root 가 아닌 내 계정** — gpio·i2c 그룹 권한으로 충분 |
| `WorkingDirectory=` | CSV 가 생기는 곳 |
| `ExecStart=/home/pi/eio/day5/src/envmon -c …/envmon.csv` | 실행할 명령 |
| `KillSignal=SIGTERM` · `TimeoutStopSec=5` | stop 이 보내는 신호 → envmon 핸들러가 출력 끄고 반납 |
| `Restart=on-failure` · `RestartSec=3` | **비정상 종료**(코드 ≠ 0, 신호로 죽음)일 때만 3초 뒤 재시작 |
| `WantedBy=multi-user.target` | enable 하면 부팅 순서에 들어감 |

- **ExecStop 을 안 쓰는 이유**: 기본 종료 방법이 SIGTERM 이고 envmon 이 SIGTERM 을 받아 **스스로 정리**한다(`gpio_install_sigint`).
- **로그 버퍼링**: systemd 아래 stdout 은 **파이프**라 printf 가 버퍼에 쌓여 늦게 보임 → `setvbuf(stdout, NULL, _IOLBF, 0)` 으로 줄 단위 (Linux Day1 4교시).
- **enable vs start**: `enable` 은 부팅 때 시작하도록 **등록만**(지금 시작하지 않음), `--now` 를 붙이면 지금도 시작.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `sudo cp envmon.service /etc/systemd/system/` → `sudo systemctl daemon-reload` | 등록 → **먼저 daemon-reload** |
| `sudo systemctl enable --now envmon` | 부팅 자동 + 지금 시작 |
| `systemctl status envmon` | `active (running)` |
| `journalctl -u envmon -f` / `-n 20` | 로그 따라 보기 / 최근 20줄 |
| `sudo systemctl stop` / `start` / `restart envmon` | 정지 / 시작 / 재시작 |
| `sudo kill -9 $(pidof envmon)` | 비정상 종료 시험 |
| `sudo systemctl disable --now envmon` | 과정 끝 — 해제 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `envmon.service` | 위 설정으로 envmon 을 pi 계정·작업 폴더에서 실행, SIGTERM 으로 정지, 비정상 종료 시 3초 뒤 재시작, multi-user 부팅에 포함 |
| 7-2 장애 시험 | stop/start, kill -9, 센서 선 뽑기, LCD 충돌, reboot |

**핵심 내용**

#### systemd 서비스 등록
> 고치면 daemon-reload, 등록은 enable --now
- **순서** → service 파일의 사용자 · 경로 수정 (`whoami`) → `cp` → **`daemon-reload`** → `enable --now` → `status` / `journalctl`
- ⚠ 실행 파일을 다시 빌드해도 → 자동으로 새 파일로 재시작하지 않음 → `make envmon && sudo systemctl restart envmon`
- **성공 확인** → `sudo reboot` 후 다시 접속하기 전에 LCD 가 이미 켜져 있으면 성공

#### stop 과 kill -9
> SIGTERM 은 정리하고, SIGKILL 은 정리 없이 죽는다 — systemd 가 다시 띄운다
- **`systemctl stop`** → **SIGTERM** → 릴레이·부저 꺼짐, LCD `envmon stopped`, GPIO 반납 후 종료
- ⚠ **`kill -9`** → SIGKILL 은 핸들러가 못 받음 (Linux Day1) → 출력이 켜진 채 남을 수 있음
- **그 뒤** → `Main process exited, code=killed, status=9/KILL` → `Scheduled restart job, restart counter is at 1` → **3초 뒤 재시작**이 다시 정리

#### 장애 주입 시험
> 센서가 빠져도 계속 돌고, 필수 장치가 없으면 재시작한다

| 장애 | 기대 결과 |
|---|---|
| DHT11 선 뽑기 | `T --.-` · 상태 LED 빠르게 · 로그 1줄 → 다시 꽂으면 "복구" |
| PCF8591 Sda 뽑기 | `Light --` · 로그 → 복구 |
| HC-SR04 Echo 뽑기 | `Dist ---.-` · 접근 경고 없음 |
| `04_lcd_demo` 실행 중 서비스 시작 | **LCD 요청 실패 → 재시작 반복**, journalctl 에 "요청 실패" |

**기본 사용형태**
```ini
[Unit]
Description=EIO smart environment monitor (Day 5)
[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/eio/day5/src
ExecStart=/home/pi/eio/day5/src/envmon -c /home/pi/eio/day5/src/envmon.csv
KillSignal=SIGTERM
TimeoutStopSec=5
Restart=on-failure
RestartSec=3
[Install]
WantedBy=multi-user.target
```
```bash
sudo cp envmon.service /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl enable --now envmon
journalctl -u envmon -f
```

**연관 문제**
<<QA 7>>

### 8교시 · 시연 · 평가 · 과정 정리

**키워드:** 시연 체크, 코드 리뷰 체크리스트, 쓰레드 구조 오류 처리 자원 해제 타이밍 재사용, 5일 I/O 경로, 같은 LED 다른 경로, gpioset → libgpiod → PWM → 버스 → 커널 드라이버 · DT · 서비스, 누가 소유하나 gpioinfo, µs 는 하드웨어·커널, 죽으면 커널이 라인 반납, systemd 재시작, Neural Network · Vision 예고

**개요**
- 팀별로 envmon 을 시연하고, 쓰레드 구조 · 오류 처리 · 자원 해제를 리뷰한다. 5일 동안 **같은 GPIO 가 어떤 경로로 다뤄졌는지** 한 장으로 정리하고, 세 가지 질문(누가 소유하나 · 타이밍은 누가 책임지나 · 죽으면 어떻게 되나)으로 마무리한다.

**핵심 용어** — 코드 리뷰 체크리스트

| 항목 | 질문 |
|---|---|
| 쓰레드 구조 | 공유 변수를 모두 **mutex 안에서** 쓰는가? 잠근 채 오래 걸리는 일(센서·LCD)을 하지 않는가? |
| 오류 처리 | 모든 open·request·read 의 **반환값을 검사**하는가? 실패가 화면·로그로 보이는가? |
| 자원 해제 | Ctrl+C · `systemctl stop` 뒤 **릴레이·부저가 꺼지는가**? `gpioinfo` 에 envmon 이 남지 않는가? |
| 타이밍 | **µs 가 필요한 일은 커널에**(DHT11·PWM), 나머지는 사용자 공간에 두었는가? |
| 재사용 | 모듈 헤더만 보고 다른 사람이 쓸 수 있는가? |

**새로 나온 함수/명령** — 시연에서 보여 줄 것
- `sudo reboot` 후 **아무것도 실행하지 않았는데 LCD 가 켜진다**(서비스)
- 화면 네 장 전환 · 음소거 / 손 접근 → 경고 · DHT11 감싸기 → 팬
- 센서 선 하나 뽑기 → **죽지 않고 `--`** · 다시 꽂으면 복구
- `journalctl -u envmon -n 10` · `tail envmon.csv`

**예제 코드가 하는 일** — 5일 정리: 같은 LED, 다른 경로

| Day | 경로 | 장치 파일 · 도구 |
|---|---|---|
| 1 | 명령 | `gpioset` · config.txt · /dev · /sys |
| 2 | libgpiod (사용자 공간) | `/dev/gpiochip0` · 엣지 이벤트 · epoll · (mmap 비교) |
| 3 | 사용자 공간 타이밍 vs 하드웨어 | SW PWM · SCHED_FIFO · `/sys/class/pwm` |
| 4 | 버스 드라이버 | `/dev/i2c-1` · `/dev/spidev0.0` · `/dev/serial0` |
| 5 | 커널 드라이버 · DT · 서비스 | IIO · `/sys/class/leds` · `/dev/input` · overlay · systemd |

**핵심 내용**

#### 세 가지 질문
> 소유 · 타이밍 · 죽었을 때
- **이 I/O 는 누가 소유하나?** → 요청한 프로세스 또는 커널 드라이버 — **`gpioinfo`** 로 확인
- **타이밍은 누가 책임지나?** → **µs → 하드웨어 블록 · 커널** (하드웨어 PWM 은 PWM 블록, DHT11 은 커널 드라이버) / ms 이상 → 사용자 공간 (+ SCHED_FIFO)
- **죽으면 어떻게 되나?** → **커널이 라인 반납**, 출력 상태는 **프로그램 책임**, systemd 가 재시작
- **`/dev/gpiomem` mmap 을 일반 코드에 안 쓰는 이유** → 커널을 건너뛰어 **소유 · 이식성이 없다** (Day 2)

#### 과정 정리
> GPIO 를 다루는 경로가 점점 커널 쪽으로
- **경로** → `gpioset` 명령 → libgpiod C 프로그램 → 하드웨어 PWM · SCHED_FIFO → 버스 (I2C · SPI · UART) → 커널 드라이버 · overlay · systemd
- **과정이 끝나면** → `sudo systemctl disable --now envmon` (다음 과목에서 같은 Pi)
- **다음 과목 (Neural Network · Vision)** → 같은 Pi 4 에 카메라 모듈, envmon 구조 (센서 쓰레드 + 판단 + 출력 + 서비스) 그대로, 센서가 카메라와 추론으로

**기본 사용형태**
```bash
gpioinfo -c gpiochip0 | grep -v unused     # 지금 누가 무엇을 잡고 있나
journalctl -u envmon -n 10
tail envmon.csv
sudo systemctl disable --now envmon        # 과정 끝
```

**연관 문제**
<<QA 8>>
