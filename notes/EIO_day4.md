## Day 4 — 버스 통신: I2C · SPI · UART

### 1교시 · I2C 와 PCF8591

**키워드:** I2C, SDA SCL, 7비트 주소 0x48, 풀업, open-drain, ACK NACK, START STOP, PCF8591, 첫 바이트는 이전 값, i2cdetect -y 1, i2cget, i2cset, /dev/i2c-1, open, ioctl I2C_SLAVE, write read, pcf8591.h, Vcc 3V3, 01_pcf_read

**개요**
- Pi 4 에는 ADC 가 없다 → I2C 로 외부 ADC 칩 **PCF8591** 을 붙인다. I2C 의 두 선·주소·풀업·ACK 를 이해하고, 셸(`i2c-tools`)로 먼저 확인한 뒤 **`open · ioctl(I2C_SLAVE) · write · read`** 로 채널을 읽는다. (MCU Day3 6교시 `Wire` 를 Linux 로)

**핵심 용어**
- **I2C**: SDA(데이터)·SCL(클럭) **두 선**에 여러 장치, 장치마다 **7비트 주소**(PCF8591 = **0x48**). 두 선은 평소 High — 장치는 **Low 로 끌어내리기만**(Day 2 open-drain).
- **ACK / NACK**: 바이트를 받을 때마다 받는 쪽이 SDA 를 Low 로 당겨 "받았다". 마스터가 **마지막 바이트를 받은 뒤엔 NACK**.
- **START / STOP**: SCL 이 High 일 때 SDA 가 내려가면 시작, 올라가면 끝.
- **주소 바이트**: 7비트 주소 + R/W 비트 → 0x48 쓰기 = `1001000 0` = **0x90**(0x48 이 아님).
```text
쓰기: START | 1001000 0 (0x48+W) | ACK | 0100 0000 (제어 0x40) | ACK | STOP
읽기: START | 1001000 1 (0x48+R) | ACK | 이전 값 | ACK | 지금 값 | NACK | STOP
```
- **첫 바이트는 이전 값**: PCF8591 은 읽기 요청을 받는 순간 다음 변환을 시작 → **2바이트를 읽고 첫 바이트를 버린다**.
- **Vcc 는 반드시 3V3**: 기판의 I2C 풀업(4.7kΩ)이 Vcc 에 연결 → 5V 면 SDA·SCL 이 5V 로.

| 하는 일 | MCU (Arduino) | Linux |
|---|---|---|
| 버스 시작 | `Wire.begin(21, 22)` | `fd = open("/dev/i2c-1", O_RDWR)` |
| 상대 고르기 | `Wire.beginTransmission(0x48)` | **`ioctl(fd, I2C_SLAVE, 0x48)`** |
| 쓰기 | `Wire.write(0x40); endTransmission()` | `write(fd, &ctrl, 1)` |
| 읽기 | `Wire.requestFrom(0x48, 2); Wire.read()` | `read(fd, buf, 2)` |
| 장치 찾기 | 스캐너 스케치 | `i2cdetect -y 1` |

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `i2cdetect -y 1` | 버스 1 에서 응답하는 주소 표 (`48` 표시) |
| `i2cget -y 1 0x48 0x40` | 제어 0x40 쓰고 1바이트 읽기 (첫 번째는 이전 값) |
| `i2cset -y 1 0x48 0x40 200` | DAC = 200 → Aout LED |
| `pcf_open(bus, addr)` / `pcf_read(fd, ch)` / `pcf_mv(raw)` | `pcf8591.h` — 열기+주소 / 채널 읽기(0~255, 실패 −1) / `raw × 3300 / 255` |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `pcf8591.h` | `pcf_open`(open + I2C_SLAVE), `pcf_read`(제어 `0x40 \| ch` 쓰기 → 2바이트 읽기 → 두 번째 반환), `pcf_read_all`(I2C_RDWR), `pcf_write_dac`, `pcf_mv` |
| `01_pcf_read.c` | AIN0(가변저항)을 0.2초마다 raw·mV·막대로 출력 (`raw 97 1254 mV`), `-n` 이면 1바이트만 읽어 **한 박자 늦은 값** 확인 |

**핵심 내용**

#### I2C 로 PCF8591 읽기
> open → 주소 지정 → 제어 바이트 쓰기 → 2바이트 읽기 → 두 번째 값
- **순서** → `open("/dev/i2c-1")` → `ioctl(I2C_SLAVE, 0x48)` → `write(fd, 0x40 | ch)` → `read(fd, buf, 2)` → `buf[1]`
- **`i2cget` 값이 한 박자 늦게 바뀜** → 정상 (첫 바이트는 지난번 변환 값)

#### I2C 문제 해결
> 증상 → 원인

| 증상 | 원인 |
|---|---|
| `i2cdetect` 가 전부 `--` | **Scl·Sda 바뀜**, Vcc 빠짐, I2C 꺼짐 |
| `Could not open file /dev/i2c-1` | I2C 꺼짐 → Day 1 `do_i2c 0` |
| `Permission denied` | i2c 그룹 아님 → `usermod -aG i2c` 후 재로그인 |

**기본 사용형태**
```c
int fd = open("/dev/i2c-1", O_RDWR | O_CLOEXEC);
ioctl(fd, I2C_SLAVE, 0x48);                    // 이후 read/write 는 이 주소로
uint8_t ctrl = 0x40 | (ch & 3), buf[2];
write(fd, &ctrl, 1);
read(fd, buf, 2);
int v = buf[1];                                // buf[0] = 이전 값 → 버림
int mv = v * 3300 / 255;
```
```bash
i2cdetect -y 1
i2cget -y 1 0x48 0x40 ; i2cget -y 1 0x48 0x40
```

**연관 문제**
<<QA 1>>

### 2교시 · DAC 와 다채널 읽기

**키워드:** 제어 바이트 비트, bit6 DAC, bit5·4 입력 방식, bit2 자동 증가, bit1·0 채널, 0x40 0x42 0x44, DAC AOUT = 값/255 × 3.3V, Aout LED, I2C_RDWR, struct i2c_msg, I2C_M_RD, i2c_rdwr_ioctl_data, 묶음 트랜잭션 STOP 없음, 5바이트, 조이스틱 3V3, 데드존, 02_pcf_dac, 03_joy_servo

**개요**
- 데이터시트의 **제어 바이트를 비트로** 만들어 DAC 로 진짜 전압을 내고, **자동 증가 + `I2C_RDWR` 묶음 트랜잭션**으로 네 채널을 한 번에 읽어 조이스틱으로 서보를 움직인다. (MCU Day2 3교시 조이스틱 → 서보의 Linux 판)

**핵심 용어**

| 비트 | 7 | 6 | 5 · 4 | 3 | 2 | 1 · 0 |
|---|---|---|---|---|---|---|
| 의미 | 0 | **DAC 출력 켜기** | 입력 방식 (00 = 단일 4채널) | 0 | **자동 증가** | **채널 번호** |

| 하고 싶은 일 | 제어 바이트 |
|---|---|
| AIN0 읽기 + DAC 켜기 | `0100 0000` = **0x40** |
| AIN2 읽기 + DAC 켜기 | `0100 0010` = **0x42** |
| AIN0 부터 자동 증가로 네 채널 | `0100 0100` = **0x44** |
| DAC 에 값 쓰기 | 0x40 다음에 값 1바이트 |

- **DAC**: PWM(빠른 켜기/끄기)이 아니라 **실제로 중간 전압** — AOUT = 값 / 255 × 3.3V → 128 이면 **약 1.66V**. LED 는 약 1.8V 아래에선 안 켜져 값 140 근처까지 어두움(전압과 밝기가 비례하지 않음).
- **자동 증가로 네 채널**: 5바이트 읽기 = [이전 값, AIN0, AIN1, AIN2, AIN3].
- **I2C_RDWR 묶음**: `write()`·`read()` 를 따로 부르면 그 사이에 **STOP** 이 들어가 다른 프로세스가 버스를 쓸 틈이 생김 → `ioctl(I2C_RDWR)` 은 **[쓰기] + 다시 START + [읽기]** 를 **중간 STOP 없이** 한 덩어리로.
```text
write + read : START 0x48+W 0x44 STOP START 0x48+R [5바이트] STOP
I2C_RDWR     : START 0x48+W 0x44 ─ START ─ 0x48+R [5바이트] STOP
```
- **조이스틱**: `+5V` 라고 인쇄되어 있어도 **3V3** (출력이 3.3V 를 넘지 않게). 서보는 반대로 5V.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `struct i2c_msg {addr, flags, len, buf}` | flags 0 = 쓰기, **`I2C_M_RD`** = 읽기 |
| `struct i2c_rdwr_ioctl_data {msgs, nmsgs}` | 메시지 배열 |
| `ioctl(fd, I2C_RDWR, &data)` | 묶음 트랜잭션 |
| `pcf_read_all(fd, 0x48, out[4])` / `pcf_write_dac(fd, v)` | 네 채널 / DAC |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `02_pcf_dac.c` | DAC 0 → 255 → 0 숨쉬기(Aout LED), 인자 값 고정, `-k` 면 가변저항(AIN0) 값을 그대로 DAC 로 |
| `03_joy_servo.c` | `pcf_read_all` 로 네 채널(AIN0 가변저항, IN1 X, IN2 Y)을 한 번에 읽어 X 와 가운데 값 차이로 서보 각도 증감(데드존 ±8), 조이스틱 SW(GPIO6) 누르면 90°, `-c` 면 시작 시 가운데 보정 |

**핵심 내용**

#### 조이스틱으로 서보
> 네 채널을 한 번에 읽고, 데드존 밖에서만 각도를 바꾼다
- **예시** → `가운데 X = 131` / `AIN0 97 X 255 Y 129 AIN3 12 → 142°`
- **한 바퀴** → `pcf_read_all` → X − 가운데 → 데드존 밖이면 각도 변경 (많이 밀수록 빠르게) → SW 확인 → `pwm_set_duty`
- **채널마다 쓰기+읽기** → 시스템 콜 **8번** / **자동 증가 + I2C_RDWR** → **1번**

#### MCU 와 Linux 비교
> 같은 일을 다른 하드웨어 경로로

| MCU (ESP32) | Linux |
|---|---|
| `analogRead(36)` 12비트 0~4095 | PCF8591 AIN1 **8비트 0~255** |
| `ledcWrite()` 서보 50Hz | `/sys/class/pwm` (Day 3 `pwm_sysfs.h`) |
| 핀마다 ADC | I2C 두 선에 4채널 — 핀이 모자랄 때 외부 ADC |

**기본 사용형태**
```c
uint8_t ctrl = 0x40 | 0x04 | 0;                // 0x44: DAC + 자동 증가 + 채널 0
uint8_t buf[5];
struct i2c_msg msgs[2] = {
    { .addr = 0x48, .flags = 0,        .len = 1, .buf = &ctrl },
    { .addr = 0x48, .flags = I2C_M_RD, .len = 5, .buf = buf },
};
struct i2c_rdwr_ioctl_data data = { .msgs = msgs, .nmsgs = 2 };
ioctl(fd, I2C_RDWR, &data);
for (int i = 0; i < 4; i++) out[i] = buf[i + 1];   // buf[0] = 이전 값
uint8_t dac[2] = { 0x40, 128 }; write(fd, dac, 2);  // AOUT ≈ 1.66 V
```

**연관 문제**
<<QA 2>>

### 3교시 · 아날로그 센서 확장

**키워드:** 입력 최대 전압 계산, 포토레지스터 분압, LM35 +Vs 5V 출력 1.5V 이하, 불꽃 2핀 3V3, 수위 센서 GPIO 전원, MQ-2 제외 5V, raw → mV 12.9mV, °C = mV/10, 분해능 1.3°C, 조도 %, 이동 평균 8개, 히스테리시스 120/90, 04_sensors

**개요**
- PCF8591 입력은 Vcc(3.3V)를 넘으면 안 된다 → 센서마다 **꽂기 전에 최대 전압부터 계산**한다. 8비트 값을 전압 · 물리량(조도 %, °C)으로 바꾸고, **이동 평균**과 **히스테리시스**로 값을 안정시킨다. (MCU Day2 2·4·5·7교시의 Linux 판)

**핵심 용어**

| 센서 | 연결 | 최대 전압 | 판단 |
|---|---|---|---|
| 포토레지스터 | 3V3 – 포토레지스터 – **IN0** – 10kΩ – GND | 3.3V 미만 (분압) | ✅ |
| LM35 | +Vs → **5V**(4V 이상 필요), Vout → **IN1** | 10mV/°C → 150°C 에서 **1.5V** | ✅ 분압 불필요 |
| 불꽃 (2핀) | 짧은 다리(−) → **3V3**, 긴 다리(+) → **IN2** · 10kΩ → GND | 3.3V 미만 | ✅ (설명서의 5V 연결은 쓰지 않음) |
| 수위 | + → **GPIO25**(3.3V), S → **IN3** | 3.3V | ✅ 잴 때만 전원 |
| MQ-2 가스 | AO 가 **5V 까지** | 5V | ❌ 오늘 안 씀 |

- **변환**: mV = raw × 3300 / 255 (**한 칸 ≈ 12.9mV**) → raw 23 = 297mV / LM35 °C = mV / 10 → **297mV = 29.7°C** / 조도 % = raw × 100 / 255 → raw 180 = 70%.
- **분해능**: LM35 는 1°C = 10mV 인데 한 칸이 12.9mV → **약 1.3°C**. ESP32 12비트(한 칸 0.8mV)보다 거칠다 → 여러 번 평균.
- **수위 센서 전원을 GPIO 로**: 전극에 계속 전류가 흐르면 **전기 분해로 부식** → 잴 때만 10ms 켬 (MCU Day2 7교시).

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `avg_push(&a, v)` | 링 버퍼 이동 평균 8개 (MCU Day2 `ring_push` 와 같음) |
| 히스테리시스 `FLAME_ON 120` / `FLAME_OFF 90` | 120 위면 ON, 90 아래로 내려가야 OFF |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `04_sensors.c` | 수위 전원 ON → 10ms → `pcf_read_all` 로 네 채널 → 전원 OFF → 조도%·온도°C·불꽃(이동 평균) 계산, 불꽃은 120/90 히스테리시스로 🔥 경보, 수위 % — 250ms 마다 출력 |

**핵심 내용**

#### 센서 넷 읽기
> raw → 이동 평균 → mV → 물리량
- **평소** → `68% 24.5 31 0%`
- **포토레지스터를 가림** → `12% …`
- **LM35 를 쥠** → `66% 27.1 …` (천천히 오름)
- **라이터 · 리모컨** → `67% 24.6 158 🔥`
- **수위 센서를 물에** → `… 58%`
- **온도 순서** → `pcf_read_all` → 이동 평균 → mV = raw × 3300 / 255 → °C = mV / 10
- ⚠ LM35 방향 → 평평한 면을 나를 향하게 하면 왼쪽부터 +Vs · Vout · GND, 거꾸로 꽂으면 뜨거워짐

#### 값을 안정시키는 장치
> 8비트 · 잡음으로 ±1~2칸 흔들린다
- **이동 평균 8개** → 흔들림 ↓
- **히스테리시스 120/90** → 불꽃 경보가 경계에서 떨지 않게

**기본 사용형태**
```c
typedef struct { int buf[8], head, count, sum; } avg_t;
static int avg_push(avg_t *a, int v) {
    if (a->count == 8) a->sum -= a->buf[a->head]; else a->count++;
    a->buf[a->head] = v; a->sum += v; a->head = (a->head + 1) % 8;
    return a->sum / a->count;
}
double temp = pcf_mv(avg_push(&at, v[1])) / 10.0;        // 10 mV = 1 °C
if (!alarm && flame > 120) alarm = 1; else if (alarm && flame < 90) alarm = 0;
```

**연관 문제**
<<QA 3>>

### 4교시 · SPI 와 74HC595

**키워드:** SPI, MOSI GPIO10, MISO GPIO9, SCLK GPIO11, CE0 GPIO8, 칩 선택, SPI 모드 CPOL CPHA, MODE_0, ACK 없음, 빠름, 74HC595 DS SH_CP ST_CP, CE0 상승 에지 = 래치, MSB 먼저 bit7 → Q7, OE GND MR 3V3, VCC 3V3, spidev, SPI_IOC_WR_MODE, SPI_IOC_MESSAGE, spi_ioc_transfer, spi595.h, 1kΩ, 05_spi595

**개요**
- SPI 는 클럭에 맞춰 한 비트씩 보내는 빠른 버스다. **MOSI · SCLK · CE0** 를 74HC595 의 **DS · SH_CP · ST_CP** 에 이으면, SPI 전송 한 번 = MCU Day3 의 `shiftOut + 래치` 한 번이 된다. `spidev` 로 1바이트를 보내 7-Segment 에 0~F 를 표시한다.

**핵심 용어**
- **SPI 4선**: **MOSI**(GPIO10, 물리 19 — Pi 가 내보냄) / MISO(GPIO9, 21 — 오늘 안 씀) / **SCLK**(GPIO11, 23 — Pi 가 만든 클럭) / **CE0**(GPIO8, 24 — **칩 선택, 전송 중 Low**).
- **모드(CPOL·CPHA)**: 평소 클럭 레벨 · 읽는 에지. **MODE_0** = 평소 Low, 상승 에지에서 샘플(595·대부분 센서).
- **I2C vs SPI**: I2C 는 주소로 상대를 고르고 ACK 로 확인, SPI 는 **CE 선으로 고르고 ACK 가 없다** — 대신 **빠르다**(수 MHz ~ 수십 MHz).
- **CE0 = 래치**: CE0 는 전송하는 동안 Low 였다가 끝나면 High → **그 상승 에지가 ST_CP 래치**. 그래서 따로 래치 핀을 토글할 필요가 없다.
- **MSB 먼저**: 8번 밀면 처음 보낸 **bit7 은 Q7**, bit0 은 Q0.
- **595 전원 3V3**: Pi 와 같은 3.3V 논리(5V 아님). OE → **GND**(출력 켜기), MR → **3V3**(지우기 안 함).

| 595 핀 | Pi | 역할 |
|---|---|---|
| DS (14) | MOSI | 넣을 비트 |
| SH_CP (11) | SCLK | 상승 에지마다 한 칸 밀기 |
| ST_CP (12) | **CE0** | 상승 에지에 8비트를 출력으로 (래치) |
| OE (13) · MR (10) | GND · 3V3 | 출력 켜기 · 지우기 안 함 |
| VCC (16) · GND (8) | 3V3 · GND | |
| Q0~Q7 | 1kΩ → a ~ dp | |

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `ls -l /dev/spidev0.*` | spidev0.0(CE0) · spidev0.1(CE1) |
| `ioctl(fd, SPI_IOC_WR_MODE, &mode)` / `SPI_IOC_WR_BITS_PER_WORD` / `SPI_IOC_WR_MAX_SPEED_HZ` | 모드 0 · 8비트 · 1MHz |
| `struct spi_ioc_transfer {tx_buf, len, speed_hz, bits_per_word}` + `ioctl(fd, SPI_IOC_MESSAGE(1), &tr)` | 전송 — `(1)` = **전송 구조체 개수** |
| `spi_open(dev, hz)` / `spi_send(fd, b)` / `rev8(x)` / `SEG_HEX[16]` / `seg_char(c)` | `spi595.h` — 열기 / 1바이트 / 비트 뒤집기 / 0~F 글꼴 / 글자 → 패턴 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `spi595.h` | spidev 를 모드 0·8비트·속도로 설정하는 `spi_open`, `SPI_IOC_MESSAGE` 로 n 바이트 `spi_send_buf`, `rev8`, 7-Segment 글꼴(a = bit0 … dp = bit7) |
| `05_spi595.c` | 1MHz 로 0 ~ F 를 0.5초마다(홀수는 점 → `1 → 0x86`, `3 → 0xCF`), `-s` 면 a → dp 하나씩(배선 확인), 인자 바이트면 그대로(`0x76` = "H") |

**핵심 내용**

#### SPI 로 595 에 1바이트
> CE0 Low 동안 8비트를 밀어 넣고, CE0 High 가 래치
- **순서** → CE0 Low → MOSI 에 bit7 부터 → SCLK 상승마다 한 칸 → 8비트 → **CE0 High (래치)** → Q0~Q7 이 동시에 바뀜
- **"0"** → **0x3F** (MCU Day3 과 같은 글꼴)
- **저항** → MCU Day3 은 COM 에 220Ω 하나, 오늘은 **세그먼트마다 1kΩ**, COM 은 GND (5교시 전류 계산)

#### 595 문제 해결
> 증상 → 원인

| 증상 | 원인 |
|---|---|
| 아무것도 안 켜짐 | **OE 가 떠 있음** / VCC 빠짐 / SPI 꺼짐 |
| 모양이 이상함 | Q 순서와 a~dp 순서가 다름 → `-s` 로 확인 |
| 가끔 전부 꺼짐 | **MR 이 떠 있음** → 3V3 고정 |
| 전부 켜지고 원하는 곳만 꺼짐 | 공통 양극(CA) → COM 을 3V3, 패턴을 `~` |

**기본 사용형태**
```c
int fd = open("/dev/spidev0.0", O_RDWR);
uint8_t mode = SPI_MODE_0, bits = 8; uint32_t hz = 1000000;
ioctl(fd, SPI_IOC_WR_MODE, &mode);
ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits);
ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &hz);
uint8_t b = 0x3F;                                        // "0"
struct spi_ioc_transfer tr = { .tx_buf = (unsigned long)&b, .len = 1, .speed_hz = hz, .bits_per_word = 8 };
ioctl(fd, SPI_IOC_MESSAGE(1), &tr);                      // 끝나면 CE0 ↑ = 래치
```

**연관 문제**
<<QA 4>>

### 5교시 · 4자리 7-Segment 멀티플렉싱

**키워드:** 4자리 7-Segment, 세그먼트 공유, 자리 COM D1~D4, 멀티플렉싱, 자리 핀 sink 전류, 1kΩ 이유 47mA → 10mA, 16mA 한계, 끄기 → 모양 → 켜기, 고스팅, 자리당 2ms 125Hz, active-low 자리 선택, 표시 쓰레드, seg4.h, 화면 버퍼 mutex, 오른쪽 정렬 dp, SCHED_FIFO, 06_seg4

**개요**
- 4자리 모듈은 a~dp 8개를 **네 자리가 같이 쓰고** 자리마다 공통 단자(D1~D4)가 따로 있다. 한 순간에 한 자리만 켜고 빠르게 돌리는 멀티플렉싱을 **표시 쓰레드**로 만들고, 자리 핀 전류를 계산해 세그먼트 저항을 고르며, 부하 중 깜빡임을 SCHED_FIFO 로 줄인다. (MCU Day3 3교시의 Pi 판)

**핵심 용어**
- **자리 핀 전류(sink)**: 자리 선택 핀(D1~D4)이 그 자리의 켜진 세그먼트 전류를 **모두 받아** GND 로 보낸다.

| | 220Ω (MCU 과정) | 1kΩ (오늘) |
|---|---|---|
| 세그먼트 하나 (3.3 − 2.0V) | 1.3V / 220Ω ≈ 5.9mA | 1.3V / 1kΩ ≈ **1.3mA** |
| "8." 일 때 자리 핀 하나로 | 8 × 5.9 ≈ **47mA** | 8 × 1.3 ≈ **10mA** |
| Pi GPIO 한 핀 권장 한계 | **16mA → 초과** | 이내 |

- **자리 전환 순서**: ① **모두 끄기** → ② 모양 바꾸기(spi_send) → ③ 새 자리 켜기. 이전 자리를 끄기 전에 모양을 바꾸면 옆 자리에 숫자가 희미하게 겹침(**고스팅**).
- **자리 선택 active-low**: 공통 음극이라 자리 핀 LOW = 켜짐 → **active-low 로 요청**해 코드의 1(ACTIVE) = 자리 켜짐 (Day 2 2교시).
- **자리당 2ms × 4 = 8ms → 약 125Hz**.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `seg4_start(&d, spi_fd, fifo)` | 자리 라인 요청(active-low) + 표시 쓰레드 시작(FIFO 실패 시 일반 쓰레드) |
| `seg4_text(&d, "23.4")` | **오른쪽 정렬**, `'.'` 은 앞 글자의 dp → `[ ][2][3.][4]` |
| `seg4_stop(&d)` | 쓰레드 종료 · 끄기 · 반납 |
| D1~D4 | GPIO **17 · 27 · 22 · 5** |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `seg4.h` | 표시 쓰레드가 2ms 마다 절대 시각으로 한 자리씩(① 모두 끄기 → ② 595 로 모양 → ③ 그 자리 켜기) 돌고 최대 지연을 기록, 화면 버퍼는 mutex 로 보호 |
| `06_seg4.c` | 0000 → 9999 카운터 / `-l` PCF8591 조도 % / `-t 23.4` 글자 고정 / `-f` FIFO, 1초마다 최대 지연 출력 |

**핵심 내용**

#### 4자리 멀티플렉싱 쓰레드
> 한 순간에 한 자리만 — 세그먼트 선은 네 자리가 공유
- **4자리 모듈** → 자리마다 세그먼트 선이 따로 있지 **않다** (공유)
- **최대 지연 부하 없음** → **84µs**
- **yes ×4** → **9317µs** — 한 자리가 9ms 켜진 채 → 깜빡임
- **`-f` SCHED_FIFO** → **61µs**

#### 해결은 우선순위
> Day 3 소프트웨어 PWM 과 같은 문제, 같은 해결
- **595 에는** → 하드웨어 PWM 같은 "알아서 도는" 회로가 없다 → **우선순위**로 해결
- **권한** → `setcap cap_sys_nice,cap_ipc_lock+ep`

**기본 사용형태**
```c
for (int k = 0; d->run; k = (k + 1) % 4) {
    for (int i = 0; i < 4; i++) v[i] = GPIOD_LINE_VALUE_INACTIVE;
    gpiod_line_request_set_values(d->dig, v);            // ① 모두 끄기
    spi_send(d->spi, d->buf[k]);                         // ② 모양
    v[k] = GPIOD_LINE_VALUE_ACTIVE;
    gpiod_line_request_set_values(d->dig, v);            // ③ 이 자리만
    next.tv_nsec += 2000000L; /* 올림 */ clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
}
```

**연관 문제**
<<QA 5>>

### 6교시 · 8×8 도트 매트릭스

**키워드:** 8×8 도트 매트릭스, 1088AS 핀 표, 행 GPIO 열 595, 행 스캔 1.5ms 83Hz, 열 데이터 반전 ~, rev8 비트 뒤집기, bit7 = 왼쪽 C1 = Q0, 이중 버퍼, 찢어짐 tearing, 프레임 경계 교체, atomic_int, 글자 스크롤, 행 전류 10mA, 07_matrix

**개요**
- 4자리의 "자리"가 매트릭스에서는 **행**이다. 한 행씩 켜고 그 행의 8칸 모양을 595 로 열에 보낸다. SPI 의 MSB 먼저 때문에 **`rev8`** 로 비트를 뒤집고, 그리는 중인 그림이 보이지 않게 **이중 버퍼**로 찢어짐을 막아 글자를 스크롤한다. (MCU Day3 4교시의 Pi 판)

**핵심 용어**
- **배선**: 행 R1~R8 → GPIO **5 · 6 · 12 · 13 · 16 · 20 · 21 · 26**, 열 C1~C8 ← 1kΩ ← 595 Q0~Q7. 1088AS 핀은 **행·열 순서로 나열되어 있지 않다**(핀 1~8 ≠ R1~R8) → 핀 표.
- **열 반전**: 행이 High 일 때 켜지는 제품이면 **켜질 칸의 열을 Low 로** → 열 데이터를 `~` 로 보냄. 그림이 반전되어 보이면 `-r`.
- **rev8**: 그림은 **bit7 = 왼쪽 열(C1)** 로 그렸는데 SPI 는 MSB 먼저라 **bit7 → Q7**. C1 은 Q0 에 연결 → 비트 순서를 뒤집어야 함. `img[r] = 1100 0001` → `rev8 = 1000 0011` → Q0 = C1, Q1 = C2, Q7 = C8.
- **이중 버퍼**: 버퍼 1장이면 스캔 도중 메인이 그림을 바꿔 **위 몇 행은 옛 그림, 아래는 새 그림**(찢어짐). → 메인은 **뒤 버퍼**에 다 그린 뒤 교체 요청, 스캔 쓰레드는 **행 0 을 그리기 직전(프레임 경계)** 에만 앞·뒤를 바꿈.
- **행 전류**: 한 행에 8칸이 켜지면 1kΩ 기준 그 행 GPIO 로 약 **10mA**.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `atomic_int g_front, g_swap` (`<stdatomic.h>`) | 보이는 버퍼 번호 / 교체 요청 (Linux Day5 atomic) |
| `show(img)` | 뒤 버퍼 = `1 − front` 에 복사 → `g_swap = 1` → 스캔 쓰레드가 바꿀 때까지 기다림 |
| `img[r] = 0x80` | r 행의 **가장 왼쪽 칸** |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `07_matrix.c` | 스캔 쓰레드가 1.5ms 마다 행을 돌며(프레임 경계에서 버퍼 교체 → 모든 행 끄기 → `rev8(~row)` 를 595 로 → 이 행 켜기), 메인은 하트 → 웃는 얼굴 → "HELLO PI" 스크롤. `-r` 행 극성 반대, `-1` 버퍼 1장(찢어짐 비교), `-f` FIFO |

**핵심 내용**

#### 도트 매트릭스 스캔
> 한 행씩 1.5ms — 8행이면 약 83Hz
- **1.5ms × 8행** → 12ms → **약 83Hz**
- **스크롤** → 글자 5열 + 간격 1열의 긴 띠를 만들어 오른쪽 밖에서 한 열씩 왼쪽으로 밀며 `show()`

#### 이중 버퍼
> 뒤 버퍼에 그리고 프레임 경계에서 교체 — 화면이 찢어지지 않는다
- **순서** → 뒤 버퍼 번호 구하기 → 뒤 버퍼에 복사 → `g_swap = 1` → 스캔 쓰레드가 행 0 직전에 front 교체 → `g_swap = 0`

#### 매트릭스 문제 해결
> 증상 → 원인

| 증상 | 원인 |
|---|---|
| 점이 흩어져 엉뚱한 모양 | 핀 1~8 을 행으로 착각 → 1088AS 표 |
| 좌우가 뒤집힘 | **rev8 빠뜨림** / 열 배선이 C8 → Q0 |
| 위아래가 뒤집힘 | `ROWS[]` 순서 반대 |
| 모든 점이 켜지고 그림만 꺼짐 | 극성 반대 → `-r` |

**기본 사용형태**
```c
if (r == 0 && atomic_load(&g_swap)) {                      // 프레임 경계에서만
    atomic_store(&g_front, 1 - atomic_load(&g_front));
    atomic_store(&g_swap, 0);
}
uint8_t row = fb[atomic_load(&g_front)][r];
uint8_t col = g_row_high ? (uint8_t)~row : row;
/* 모든 행 끄기 */ spi_send(g_spi, rev8(col)); /* 이 행 켜기 */
```

**연관 문제**
<<QA 6>>

### 7교시 · UART 와 termios

**키워드:** UART, 클럭 선 없음, baud 115200, 8N1, 시작 비트 정지 비트 LSB 먼저, 비트 8.7µs, 초당 11520 바이트, PL011 ttyAMA0, disable-bt, 시리얼 콘솔 끄기, /dev/serial0, termios, cfmakeraw, CS8 CLOCAL CREAD ~CRTSCTS, cfsetispeed, tcsetattr TCSANOW, tcflush, VMIN VTIME, 줄 단위 버퍼 lb_feed, 루프백, TX RX 교차, GND 공통, uart.h, 08_uart_term

**개요**
- UART 는 클럭 선 없이 양쪽이 **미리 약속한 속도(baud)** 로 각자 시간을 잰다. `/dev/serial0` 은 원래 "사람이 타자 치는 터미널"이라 **termios 로 raw 바이트 통로**로 바꾸고, `read()` 가 언제 돌아오는지(VMIN·VTIME)를 정한다. 루프백 시험 후 ESP32 와 **TX ↔ RX 를 엇갈려** 연결한다.

**핵심 용어**
- **8N1 프레임**: 쉼(High) → **시작 비트 0** → b0(**LSB 먼저**) … b7 → **정지 비트 1**. 데이터 8 · 패리티 없음 · 정지 1 → **글자 하나 = 10비트**. `'A'` = 0x41.
- **115200 baud**: 비트 하나 약 **8.7µs**, 처리량 115200 / 10 = **초당 약 11 520 바이트**.
- **PL011 vs mini UART**: PL011(`ttyAMA0`)은 독립 클럭이라 정확 → `dtoverlay=disable-bt` 로 GPIO14·15 에 (Day 1 5교시). `ls -l /dev/serial0` → **ttyAMA0**, cmdline 에 `console=serial0` 이 **없어야** 한다(시리얼 콘솔이 켜져 있으면 커널 로그·로그인 프롬프트가 섞임).
- **termios raw**: `cfmakeraw` = **에코 · 줄 편집 · 특수 문자 · CR/LF 변환 끔**. `CS8` 8비트, `~PARENB` 패리티 없음, `~CSTOPB` 정지 1, `CLOCAL | CREAD` 모뎀 선 무시·수신 켬, `~CRTSCTS` 흐름 제어 끔.
- **VMIN · VTIME** (VTIME 단위 0.1초):

| VMIN | VTIME | `read(fd, buf, 64)` |
|---|---|---|
| 0 | 0 | 즉시 — 있는 만큼 (0 일 수도) |
| **0** | **1** | **1바이트라도 오면 즉시, 없으면 0.1초 뒤 0** (오늘 설정) |
| 1 | 0 | 1바이트 올 때까지 무한 대기 |
| 10 | 2 | 10바이트가 모이거나, 바이트 사이가 0.2초 비면 |

- **read() 한 번 ≠ 한 줄**: `"T=23"` 과 `".0,H=45\n"` 처럼 쪼개져 올 수 있다 → **줄 단위 버퍼(lb_feed)** 에 모아 `\n` 에서 한 줄.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `cfmakeraw(&t)` | raw 모드 |
| `cfsetispeed(&t, B115200)` / `cfsetospeed` | 속도 |
| `tcsetattr(fd, TCSANOW, &t)` / `tcflush(fd, TCIOFLUSH)` | 바로 적용 / 쌓인 바이트 버리기 |
| `uart_open(dev, baud)` / `lb_feed(&lb, c)` / `cs_xor` / `uart_send_line` | `uart.h` |
| `./08_uart_term -t` / `-b 9600` | 루프백 ping / 속도 바꾸기(상대와 다르면 깨짐) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `uart.h` | termios 로 raw 115200 8N1 + VMIN 0 · VTIME 1 설정하는 `uart_open`, `\n` 에서 한 줄을 돌려주는 `lb_feed`, XOR 체크섬 `cs_xor`·`cs_check`, `본문*HH\n` 보내는 `uart_send_line` |
| `08_uart_term.c` | epoll 한 루프로 키보드 한 줄 → UART, UART 한 줄 → 화면. `-t` 면 1초 timerfd 로 `ping N` 을 보내고 돌아오는 시간 측정 |

**핵심 내용**

#### 루프백 시험
> TXD 와 RXD 를 선 하나로 이어 Pi 혼자 먼저 확인
- **배선** → GPIO14 TXD (물리 8) ↔ GPIO15 RXD (물리 10)
- **결과** → `← ping 1 0.71 ms` …
- **선을 뽑으면** → `(응답 없음 — TX·RX 연결 확인)` → `보냄 3 · 받음 2 · 응답 없음 1`
- **계산** → `"ping 1\n"` 7바이트 × 10비트 / 115200 ≈ **0.61ms** (측정값과 거의 같음)

#### ESP32 와 연결
> TX ↔ RX 엇갈려, GND 는 반드시 공통
- **Pi TXD (GPIO14)** → ESP32 RX2 (16)
- **Pi RXD (GPIO15)** ← ESP32 TX2 (17)
- **GND ↔ GND** → 반드시 공통
- ⚠ 3V3 · 5V 는 잇지 않음 (ESP32 는 PC USB 전원)
- **레벨 변환** → 둘 다 3.3V 논리라 불필요 (5V 아두이노 우노라면 필요)
- ⚠ **글자가 깨지는 원인** → 양쪽 속도가 다름 · GND 를 잇지 않음 · 시리얼 콘솔이 켜져 있음 · mini UART 사용

**기본 사용형태**
```c
int fd = open("/dev/serial0", O_RDWR | O_NOCTTY | O_CLOEXEC);
struct termios t; tcgetattr(fd, &t);
cfmakeraw(&t);
t.c_cflag &= ~(CSIZE | PARENB | CSTOPB | CRTSCTS);
t.c_cflag |= CS8 | CLOCAL | CREAD;
t.c_cc[VMIN] = 0; t.c_cc[VTIME] = 1;          // 오면 바로, 없으면 0.1초
cfsetispeed(&t, B115200); cfsetospeed(&t, B115200);
tcsetattr(fd, TCSANOW, &t); tcflush(fd, TCIOFLUSH);
ssize_t n = read(fd, tmp, sizeof tmp);
for (ssize_t i = 0; i < n; i++) if (lb_feed(&lb, tmp[i])) printf("%s\n", lb.line);
```

**연관 문제**
<<QA 7>>

### 8교시 · 미니 프로젝트 — 센서 게이트웨이

**키워드:** 센서 게이트웨이, gateway.c, esp32_node.ino, 한 줄 프로토콜, T=23.0,H=45*2E, LED=1*41, XOR 체크섬 NMEA, 체크섬 오류 버림, 형식·범위 검사, epoll UART + timerfd, seg4 표시 쓰레드, 조도 히스테리시스 30/40, 3초 타임아웃 ----, -s -k -f 옵션, I2C SPI UART 함께

**개요**
- I2C(PCF8591 조도) · SPI(4자리 7-Segment) · UART(ESP32) 세 버스를 함께 쓰는 **게이트웨이**. ESP32 가 1초마다 DHT11 온·습도를 **한 줄 프로토콜 + XOR 체크섬**으로 보내면 Pi 가 검사·표시하고, 어두우면 ESP32 LED 를 켜라는 명령을 되돌린다. Day 2 6교시 epoll 루프에 오늘 만든 모듈을 꽂은 모양.

**핵심 용어** — 프로토콜

| 방향 | 예 | 뜻 |
|---|---|---|
| ESP32 → Pi | `T=23.0,H=45*2E` | 1초마다 DHT11 온도·습도 |
| ESP32 → Pi | `ERR=DHT*..` | DHT11 읽기 실패 |
| Pi → ESP32 | `LED=1*41` / `LED=0*40` | 어두우면 ESP32 LED 켜기 |

- **XOR 체크섬**: `*` 앞의 **모든 글자를 XOR** 한 값을 16진 두 자리로 붙임(GPS NMEA 문장과 같은 방식). 선이 흔들려 한 비트만 바뀌어도 값이 달라짐 → **버리고 다음 줄을 기다린다**.
```text
"LED=1" → 'L' 0x4C ^ 'E' 0x45 ^ 'D' 0x44 ^ '=' 0x3D ^ '1' 0x31 = 0x41 → "LED=1*41"
```
- **구조**: epoll — UART fd(받은 바이트 → 줄 버퍼 → 체크섬·형식 검사 → 값 저장) + timerfd 1초(조도 → LED 명령 히스테리시스 30/40%, 화면 결정, 3초 타임아웃 검사) + seg4 표시 쓰레드.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `./gateway` | ESP32 연결 |
| `./gateway -s` | ESP32 없이 TXD↔RXD 루프백 선 + Pi 가 **가짜 프레임**을 보냄 |
| `./gateway -k` | ESP32 없이 **키보드로** 프레임 입력 (체크섬 없이도 받음) |
| `./gateway -f` | 7-Segment 표시 쓰레드 FIFO |
| `Serial2.begin(115200, SERIAL_8N1, 16, 17)` (ESP32) | RX 16, TX 17 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `esp32_node.ino` | 1초마다 DHT11 을 읽어 `T=…,H=…*CS` (실패면 `ERR=DHT`)를 Serial2 로 보내고, `LED=1*CS` 를 받으면 체크섬 확인 후 LED(GPIO25) 제어 |
| `gateway.c` | 받은 줄을 `handle_line`(0 정상 · −1 체크섬 · −2 형식/범위)로 처리해 `reading_t` 갱신, 1초마다 조도로 LED 명령(바뀔 때 + 10초마다), 7-Segment 에 온도 `23.0` ↔ 습도 `H 45` 를 2초씩, 3초 동안 정상 프레임이 없으면 `----` |

**핵심 내용**

#### 게이트웨이 동작
> 한 줄씩 받아 검사하고, 조도로 명령을 보내고, 표시한다
- **예시** → `조도 64% → LED=0` / `수신 재개` / `T 23.0 °C H 45 %` / `조도 21% → LED=1` (포토레지스터 가림 → ESP32 LED)
- **선이 흔들림** → `버림(체크섬): T=2#.0,H=45*2E`
- **ESP32 USB 뽑음** → `3초 동안 정상 프레임 없음` → `----`
- **한 줄 처리** → read 로 바이트 → 줄 버퍼에 모아 `\n` 에서 한 줄 → 체크섬 검사 → 형식·범위 검사 → 값 저장 · 시각 기록

#### 확인 항목
> 시험해 볼 것과 기대 결과

| 확인 | 기대 |
|---|---|
| 포토레지스터 가렸다 뗌 | **30% 아래에서 LED=1, 40% 위에서 LED=0** — 그 사이는 그대로(경계에서 명령이 계속 바뀌지 않게) |
| `-k` 로 `T=99.0,H=45` | 버림(형식) — 범위 검사 |
| `-k` 로 `T=23.0,H=45*00` | 버림(체크섬) |
| ESP32 GND 선 뺌 | 글자가 깨져 체크섬 오류 증가 |
| ESP32 끔 | 3초 뒤 `----`, 다시 켜면 "수신 재개" |

#### 세 버스를 한 프로그램에
> I2C · SPI · UART

| 구성 | 버스 |
|---|---|
| PCF8591 조도 | I2C |
| 4자리 7-Segment | SPI + GPIO |
| ESP32 프레임 · 명령 | UART |

- **도전 과제** → 1분마다 `gateway.csv` 기록 (Day 5 에서 씀), 매트릭스 온도 막대, `BUZ=1` 명령, ESP32 ADC 로 조도도 보내 비교

**기본 사용형태**
```c
static inline uint8_t cs_xor(const char *s) { uint8_t x = 0; while (*s && *s != '*') x ^= (uint8_t)*s++; return x; }
/* 보내기: "LED=1" → "LED=1*41\n" */
char buf[48]; snprintf(buf, sizeof buf, "%s*%02X\n", body, cs_xor(body)); write(fd, buf, strlen(buf));
/* 받기 */
if (cs_check(lb.line) != 1) { printf("버림(체크섬): %s\n", lb.line); continue; }
if (sscanf(lb.line, "T=%lf,H=%d", &t, &h) != 2 || t < -20 || t > 80 || h < 0 || h > 100) { /* 버림(형식) */ }
```

**연관 문제**
<<QA 8>>
