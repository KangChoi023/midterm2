## Day 2 — 아날로그 입력 · 센서 · 코드 모듈화

### 1교시 · ADC 기초 — 가변저항

**키워드:** ADC, analogRead, 12비트 0~4095, analogReadMilliVolts, eFuse 보정, 감쇠 11dB, 비선형, ADC1 32~39, ADC2 와 Wi-Fi, 가변저항 와이퍼, 포인터 인자 평균 최소 최대, ADC 잡음

**개요**
- 센서 대부분은 **전압**으로 값을 낸다. ADC 가 전압을 0~4095 정수로 바꾸는 원리(12비트·감쇠·비선형), `analogRead` 와 `analogReadMilliVolts` 의 차이, 여러 번 읽어 평균·최소·최대를 **포인터 인자**로 돌려주는 함수를 다룬다.

**핵심 용어**
- **ADC(Analog-to-Digital Converter)**: 전압 → 정수. ESP32 는 **12비트** → **0~4095** (2¹² 단계). Arduino Uno 는 10비트(0~1023).
- **감쇠(attenuation)**: 잴 수 있는 전압 범위. 기본 **11dB** = 약 0.15V~3.1V. 그 밖은 **0 또는 4095 로 붙는다** (3.25V → 4095).
- **비선형**: 양 끝(0V 근처, 3.1V 이상)에서 값이 비례하지 않음.
- **analogReadMilliVolts**: 칩에 저장된 **보정값(eFuse)** 으로 계산한 mV — 전압이 필요하면 이것.
- **가변저항**: 양 끝 다리 = 3V3·GND, **가운데 다리(와이퍼)** 를 GPIO 에 → 돌리면 0~3.3V.
- **ADC1 / ADC2**: ADC2 핀(25 등)은 **Wi-Fi 를 켜면 못 쓴다** → 아날로그는 처음부터 **ADC1(32~39)**.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `analogRead` | `int analogRead(pin)` | raw 0~4095. **`pinMode` 불필요** |
| `analogReadMilliVolts` | `int analogReadMilliVolts(pin)` | 보정된 mV |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 1-1 `adc_raw.ino` | GPIO 34 의 raw 를 200ms 마다 출력 (왼쪽 끝 0~수십, 오른쪽 끝 4095) |
| 1-2 `adc_mv.ino` | raw · 보정 mV · 단순 비례(`raw*3300/4095`)를 나란히 출력 → `raw=2048 mv=1680 naive=1650.6` |
| 1-3 `adc_avg.ino` | `read_avg(pin, 32, &mn, &mx)` 로 32번 읽어 평균은 반환, 최소·최대는 포인터로 → `noise=42` |
| 1-4 ⚠ 함정 | ADC2 핀과 Wi-Fi, 5V 연결, 끝 다리 연결 |

**핵심 내용**

#### 여러 결과 돌려주기
> 반환값은 하나 — 여러 결과는 주소를 받아 거기에 써 준다
- **`read_avg(pin, n, &mn, &mx)`** → 평균은 반환, 최소·최대는 포인터 인자 (C Day2)

#### ADC 잡음과 함정
> 손을 대지 않아도 값이 수십씩 흔들리는 것은 정상 — 평균으로 줄인다
- **잡음** → 정상적인 ADC 잡음 → 평균
- ⚠ **ADC2 (GPIO 25)** → Day 4 Wi-Fi 에서 값이 안 나옴
- ⚠ **한쪽 끝을 5V 에** → 3.1V 이상 4095 고정 + **3.3V 초과로 핀 손상**
- ⚠ **끝 다리를 GPIO 에** → 돌려도 0 이나 4095

**기본 사용형태**
```c
int read_avg(int pin, int n, int *min_out, int *max_out) {
    long sum = 0; int mn = 4095, mx = 0;
    for (int i = 0; i < n; i++) { int v = analogRead(pin); sum += v; if (v < mn) mn = v; if (v > mx) mx = v; }
    *min_out = mn; *max_out = mx;
    return sum / n;
}
int mn, mx, avg = read_avg(34, 32, &mn, &mx);
int mv = analogReadMilliVolts(34);
```

**연관 문제**
<<QA 1>>

### 2교시 · 조도 센서와 디지털 필터

**키워드:** 포토레지스터 CdS, 전압 분배 분압, Vout 공식, 10kΩ, 시리얼 플로터 이름:값, 링 버퍼, 이동평균, (head+1)%RN, sum/count, 자동 조명, 되먹임 진동

**개요**
- 저항이 바뀌는 센서는 ADC 가 직접 못 잰다 → **전압 분배(분압)** 로 저항 변화를 전압 변화로 바꾼다. 흔들리는 값을 **링 버퍼 이동평균**으로 다듬고(C Day4 링 버퍼), 어두울수록 LED 를 밝히는 자동 조명을 만든다.

**핵심 용어**
- **포토레지스터(CdS)**: **밝을수록 저항이 작아짐**.
- **전압 분배**: 저항 두 개를 직렬로, 가운데 전압을 읽음. **Vout = 3.3V × R_10k ÷ (R_photo + R_10k)**.
- **이동평균**: 최근 N 개 값의 평균 → 잡음 감소.
- **링 버퍼**: 끝에 닿으면 `% RN` 으로 **처음으로 돌아가** 가장 오래된 값을 덮어씀.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| 도구 → 시리얼 플로터 | `Serial.printf("raw:%d,avg:%d\n", …)` — **`이름:값`** 형식이면 범례와 함께 그래프 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 2-1 `light.ino` | 분압 회로의 GPIO 35 를 `light:값` 으로 50ms 마다 출력 (가리면 작아짐) |
| 2-2 `ring_filter.ino` | `ring_t {buf[8], head, count, sum}` 와 `ring_push` 로 이동평균, raw·avg 를 함께 플로터에 |
| 2-3 `auto_light.ino` | 평균 밝기로 `duty = 255 - avg*255/4095`, 20 미만이면 0 → 어두울수록 LED 밝게 |
| 2-4 ⚠ 함정 | 10kΩ 누락 / 220Ω 사용 / LED 를 센서 옆에 |

**핵심 내용**

#### 전압 분배로 저항을 전압으로
> 조도 센서(CdS) + 10kΩ — 밝을수록 저항이 작아져 전압이 커진다
- **밝음 (1kΩ)** → 3.0V · 약 3700
- **실내 (10kΩ)** → **1.65V** · 약 2000
- **가림 (100kΩ)** → 0.3V · 약 300
- ⚠ 10kΩ 없이 → 밝기와 상관없이 **4095 근처**
- ⚠ 10kΩ 대신 220Ω → 항상 작은 값
- ⚠ LED 를 센서 옆에 → **빛 되먹임으로 깜빡이며 진동**

#### 링 버퍼 이동 평균
> 합계를 들고 다니며 가장 오래된 값을 빼고 새 값을 더한다
- **`ring_push`** → 가득이면 **`sum -= buf[head]`** → 새 값 쓰고 `sum += v` → **`head = (head + 1) % RN`** → `return sum / count`
- **합계를 들고 다니는 이유** → 매번 다시 더하지 않고 한 번씩 더하고 빼려고
- ⚠ `sum / RN` 이 아니라 **`sum / count`** → 덜 찼을 때 RN 으로 나누면 처음 평균이 작게 나옴

**기본 사용형태**
```c
#define RN 8
typedef struct { int buf[RN]; int head; int count; long sum; } ring_t;
int ring_push(ring_t *r, int v) {
    if (r->count == RN) r->sum -= r->buf[r->head]; else r->count++;
    r->buf[r->head] = v; r->sum += v;
    r->head = (r->head + 1) % RN;
    return r->sum / r->count;
}
ring_t light = {0};
int avg = ring_push(&light, analogRead(35));
Serial.printf("raw:%d,avg:%d\n", raw, avg);
```

**연관 문제**
<<QA 2>>

### 3교시 · 조이스틱과 서보

**키워드:** 조이스틱 X Y SW, 3V3 전원, 서보 SG90, 50Hz 20ms, 펄스 폭 0.5ms 1.5ms 2.5ms, LEDC 14비트, map 직접 구현, 곱하기 먼저, 데드존, brown-out 리셋, GND 공통

**개요**
- 2축 조이스틱(X·Y 는 ADC, 버튼은 디지털)을 읽고, **50Hz PWM 의 펄스 폭**으로 서보 각도를 제어한다. 입력 범위를 출력 범위로 옮기는 `map` 을 직접 만들고, 가운데 잡음을 **데드존**으로 처리한다.

**핵심 용어**
- **서보(SG90)**: **20ms(50Hz) 마다** 오는 펄스의 **HIGH 길이**로 각도 결정 — 약 **0.5ms = 0°, 1.5ms = 90°, 2.5ms = 180°**.
- **map(비례 변환)**: "입력 범위의 몇 % 위치인가"를 출력 범위로. **곱하기 먼저, 나누기 나중**(정수 나눗셈 0 방지, C Day1).
- **데드존**: 가운데 근처의 작은 흔들림은 무시(90° 고정).
- **brown-out**: 전압이 순간 떨어져 칩이 리셋되는 것. 서보를 3V3 에 물리면 생길 수 있음.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `ledcAttach(pin, 50, 14)` | 서보용 50Hz, 14비트(0~16383, 한 단계 약 1.2µs) |
| `my_map(x, in_min, in_max, out_min, out_max)` | `(x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min` |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 3-1 `joystick.ino` | X(36)·Y(39) 를 ADC 로, SW(4)를 `INPUT_PULLUP` 으로 읽어 플로터 출력 (가운데 약 1800~2000) |
| 3-2 `servo.ino` | `servo_write_us`(µs → 듀티 `us * MAX_DUTY / 20000`)와 `servo_write_deg`(0°→500µs, 180°→2500µs)로 0°/90°/180° 반복 |
| 3-3 `joy_servo.ino` | `my_map(x, 0, 4095, 0, 180)` 로 조이스틱 X 를 서보 각도로 |
| 3-4 ⚠ 함정 | 서보 전원·GND·주파수·데드존 |

**핵심 내용**

#### 조이스틱
> 아날로그 두 축 + 버튼 — 전원은 3V3
- ⚠ 모듈에 `+5V` 라고 써 있어도 **3V3 레일**에 (5V 면 3.3V 를 넘어 4095 고정·손상)
- **SW** → `INPUT_PULLUP`, **누르면 LOW**
- **가운데 값** → 제품마다 다름 → **처음에 한 번 재서 저장**
- ⚠ 가운데에서 움찔 → **데드존**

#### 서보 PWM
> 50Hz 주기에서 펄스 폭이 각도 — 해상도는 14비트
- **1.5ms 펄스** → 20ms 의 **7.5%**
- **14비트** → 한 단계 약 **1.2µs** (부드러움) / 8비트 → 78µs (거침)
- **`my_map(2048, 0, 4095, 0, 180)`** → **약 90**
- ⚠ `ledcAttach(18, 5000, 8)` → 주기 0.2ms 라 서보가 해석 못함 → **50Hz**

#### 서보 전원
> 서보는 5V, GND 는 반드시 공통
- ⚠ 빨강 선을 3V3 → 힘이 약해 떨리거나 **ESP32 리셋** → 5V 에
- ⚠ 5V 에 꽂고 **GND 공통 안 함** → 기준이 달라 무작위로 떨림

**기본 사용형태**
```c
const int BITS = 14, MAX_DUTY = (1 << BITS) - 1;
void servo_write_us(int us) { ledcWrite(18, (long)us * MAX_DUTY / 20000); }
void servo_write_deg(int deg) { if (deg < 0) deg = 0; if (deg > 180) deg = 180; servo_write_us(500 + deg * 2000 / 180); }
long my_map(long x, long a, long b, long c, long d) { return (x - a) * (d - c) / (b - a) + c; }
ledcAttach(18, 50, BITS);
int deg = my_map(analogRead(36), 0, 4095, 0, 180);
if (abs(deg - 90) < 5) deg = 90;          // 데드존
servo_write_deg(deg);
```

**연관 문제**
<<QA 3>>

### 4교시 · 온도와 습도 — LM35 · DHT11

**키워드:** LM35, 10mV/°C, 5V 전원, 방향 평평한 면, analogReadMilliVolts 평균, DHT11, 1-wire 40비트, 체크섬, NaN, isnan, DHT 라이브러리, Adafruit Unified Sensor, reading_t 구조체 반환, 2초 간격

**개요**
- **아날로그** 온도 센서 LM35(mV → °C)와 **디지털** 센서 DHT11(라이브러리, 실패 시 NaN)을 읽는다. 측정값·시각·유효 여부를 **구조체**로 묶어 두 센서를 비교한다.

**핵심 용어**
- **LM35**: **10mV = 1°C** 를 내는 아날로그 온도 센서. **4V 이상**에서 동작 → **5V 레일**. 평평한(글자) 면을 나를 향하게 두면 왼쪽부터 `+Vs · Vout · GND`. 거꾸로 꽂으면 **뜨거워짐 → 즉시 USB 분리**.
- **DHT11**: 선 하나로 **40비트(습도 16 + 온도 16 + 체크섬 8)** 를 정해진 타이밍으로 보내는 디지털 센서. 정확도 ±2°C, 분해능 1°C. **1~2초 간격** 이상으로 읽어야 함.
- **NaN(Not a Number)**: 체크섬이 틀리거나 응답이 없을 때 라이브러리가 돌려주는 값. **자기 자신과도 같지 않아** `t == NAN` 은 항상 거짓 → **`isnan(t)`**.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `#include <DHT.h>` / `DHT dht(27, DHT11);` / `dht.begin()` | DHT 라이브러리 (DHT sensor library + **Adafruit Unified Sensor** 함께 설치) |
| `dht.readTemperature()` / `dht.readHumidity()` | °C / % (실패 시 NaN) |
| `isnan(x)` | NaN 검사 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 4-1 `lm35.ino` | GPIO 33 을 `analogReadMilliVolts` 로 16번 평균 → `mv / 10` 으로 °C → `mv:245,temp:24.5` |
| 4-2 `dht11.ino` | 2초마다 온·습도를 읽고 `isnan` 이면 "DHT 읽기 실패" |
| 4-3 `temp_compare.ino` | `reading_t {temp_c, humi, t_ms, ok}` 를 반환하는 `read_lm35()`·`read_dht()` 로 두 센서 차이 출력 |
| 4-4 ⚠ 함정 | 방향·전원·간격·NaN 비교 |

**핵심 내용**

#### LM35 아날로그 온도
> 10mV = 1°C — 낮은 전압은 mV 함수 + 평균으로
- **245mV** → **24.5°C**
- ⚠ 250mV 근처는 11dB 감쇠의 **아래쪽 비선형 구간** → `analogRead` 비례 계산은 오차 큼 → **`analogReadMilliVolts` + 평균**
- ⚠ LM35 를 3V3 에 → 동작 전압 부족

#### DHT11 디지털 온습도
> 라이브러리로 읽고, 1초에 한 번 이하, 실패는 NaN
- **두 센서 차이 1~2°C** → 정상 (DHT11 ±2°C)
- ⚠ `delay(100)` 마다 읽기 → **절반 이상 NaN**
- ⚠ `if (t == NAN)` → 항상 거짓 → `isnan(t)`

#### 측정값 구조체
> "값 + 언제 + 유효한가" 를 한 번에 넘긴다
- **`return r;`** → 구조체 전체가 **값으로 복사**되어 반환

**기본 사용형태**
```c
#include <DHT.h>
DHT dht(27, DHT11);
typedef struct { float temp_c, humi; uint32_t t_ms; bool ok; } reading_t;
reading_t read_dht(void) {
    reading_t r;
    r.temp_c = dht.readTemperature(); r.humi = dht.readHumidity(); r.t_ms = millis();
    r.ok = !isnan(r.temp_c) && !isnan(r.humi);
    return r;                                 // 구조체 값 복사
}
float lm35_c(void) { long s = 0; for (int i = 0; i < 16; i++) s += analogReadMilliVolts(33); return s / 16.0f / 10.0f; }
```

**연관 문제**
<<QA 4>>

### 5교시 · 환경 감지 — 가스 · 불꽃 · 소리

**키워드:** MQ-2 가스 센서, 히터 예열, 5V 출력 분압 10kΩ+10kΩ, 불꽃 센서, 소리 센서 최대값, 히스테리시스, ON/OFF 두 임계값, 경보 떨림, any |= 경보 합치기

**개요**
- **5V 출력** 센서(MQ-2)를 분압해 안전하게 읽고, 불꽃·소리 센서를 같은 모양으로 읽는다. 임계값 근처에서 경보가 떨리지 않게 **히스테리시스**(켜는 선과 끄는 선을 다르게)를 적용하고, 여러 경보를 하나의 부저로 모은다.

**핵심 용어**
- **MQ-2**: 가스가 많을수록 AO 가 올라 **5V 까지**. 안의 **히터 예열** — 전원 후 **최소 1~2분** 기다린 뒤 깨끗한 공기 값을 기준으로.
- **분압 10kΩ + 10kΩ**: `V_gpio = V_AO ÷ 2` → 최대 2.5V 로 GPIO 보호.
- **히스테리시스(hysteresis)**: 켜는 임계값(ON)과 끄는 임계값(OFF)을 **다르게** 둬서 그 사이에서는 상태를 **유지**. 에어컨이 26°C 에 켜지고 24°C 에 꺼지는 것.

**새로 나온 함수/명령** — 새 함수 없음 (`analogReadMilliVolts`, `analogRead`, `digitalWrite`)

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 5-1 `mq2.ino` | 분압된 GPIO 32 mV × 2 = 실제 AO 전압 출력 (깨끗한 공기 수백 mV, 가스 2000mV 이상) |
| 5-2 `flame_sound.ino` | 불꽃(34)은 한 번, 소리(35)는 **200번 읽어 최대값** 출력 |
| 5-3 `gas_alarm.ino` | `ON_MV 1500` / `OFF_MV 1200` 히스테리시스로 부저 경보 |
| 5-4 ⚠ 함정 | 분압 누락·3V3 전원·예열·소리 한 번 읽기 |

**핵심 내용**

#### MQ-2 가스 센서
> 5V 센서 — AO 는 분압해서 읽고 ×2
- **분압된 GPIO 전압 900mV** → AO 실제 **1800mV**
- ⚠ AO 를 **분압 없이** → 5V 가 핀에
- ⚠ MQ-2 를 3V3 에 → 히터 부족
- ⚠ 켜자마자 임계값 결정 → 예열 중이라 틀림

#### 소리 · 불꽃 센서
> 같은 아날로그 모양, 읽는 방법은 신호에 맞게
- **소리** → **파동**이라 순간값이 제멋대로 → 짧은 시간 **최대값**을 본다
- **불꽃** → 제품마다 가까울수록 커지거나 작아짐 → **먼저 방향 확인**
- **2핀 센서 3V3–센서–GPIO34–10kΩ–GND** → 불꽃이 가까울수록 **커짐**

#### 히스테리시스
> 켜는 기준과 끄는 기준을 다르게 — 경계에서 떨지 않게
- ⚠ 임계값 하나(1500) → 그 근처에서 경보가 **켜졌다 꺼졌다 반복**
- **ON 1500 / OFF 1200** → 켜진 상태에서 1350 → **켜진 상태 유지**
- **간격** → **센서 잡음보다 넓게**
- **여러 경보 합치기** → `any |= update_alarm(&a[i], v)` → 하나라도 true 면 true

**기본 사용형태**
```c
typedef struct { int on_th, off_th; bool on; } alarm_t;
bool update_alarm(alarm_t *a, int v) {
    if (!a->on && v > a->on_th)       a->on = true;
    else if (a->on && v < a->off_th)  a->on = false;
    return a->on;
}
alarm_t gas = {1500, 1200, false};
bool any = false;
any |= update_alarm(&gas, analogReadMilliVolts(32) * 2);
digitalWrite(19, any);
```

**연관 문제**
<<QA 5>>

### 6교시 · 거리와 움직임 — 초음파 · PIR

**키워드:** HC-SR04, Trig 10µs, Echo 펄스, pulseIn, 타임아웃 30000, 58µs/cm, 왕복, Echo 5V 분압 10k/20k, PIR HC-SR501, 체온 적외선 변화, 예열 30초, AND 침입 감지, 5V 레일

**개요**
- 초음파 센서 HC-SR04 의 **Trig · Echo 타이밍**으로 거리를 계산하고 타임아웃을 처리한다. 5V Echo 를 분압해 GPIO 를 보호하고, PIR 로 움직임을 감지해 두 센서를 AND 로 묶은 침입 감지를 만든다.

**핵심 용어**
- **HC-SR04**: Trig 에 **10µs HIGH 펄스**를 주면 초음파를 쏘고, Echo 가 **왕복 시간만큼 HIGH**. 5V 부품(VCC 는 **5V 레일**).
- **58µs = 1cm**: 소리는 1cm 를 약 29µs 에 가고 **왕복이라 두 배**. 1160µs → **20cm**.
- **Echo 분압**: 10kΩ 직렬 + 20kΩ(10kΩ 두 개) GND → `5 × 20 / 30 ≈ 3.33V`.
- **PIR(HC-SR501)**: **체온(적외선)의 변화**를 감지 — 가만히 있으면 못 잡는다. 전원 후 **30초~1분 예열**. VCC 는 5V.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `delayMicroseconds` | `delayMicroseconds(us)` | µs 대기 (Trig 펄스) |
| `pulseIn` | `unsigned long pulseIn(pin, HIGH, timeout_us)` | HIGH 펄스 길이(µs). **타임아웃이면 0** |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 6-1 `ultrasonic.ino` | Trig 10µs → `pulseIn(ECHO, HIGH, 30000)` → 0 이면 "범위 밖", 아니면 `us / 58.0` cm |
| 6-2 `pir.ino` | 30초 예열 후 PIR 값을 LED 와 시리얼에 |
| 6-3 `intruder.ino` | **움직임 AND 30cm 이내**일 때만 LED 경보 |
| 6-4 ⚠ 함정 | Echo 분압 누락·3V3 전원·타임아웃 누락·센서 두 개 동시 |

**핵심 내용**

#### 초음파 거리
> Trig 10µs → Echo 펄스 폭 — 실패는 0
- **`pulseIn` 이 0** → 측정 실패 (타임아웃) → 거리로 쓰지 않는다
- **연속 실패 세기** → 성공 시 `fail_count = 0`
- ⚠ 타임아웃 없이 `pulseIn(PIN_ECHO, HIGH)` → 물체가 없으면 **loop 가 약 1초씩 멈춤**
- ⚠ Echo 를 분압 없이 → 5V 가 핀에 (당장 동작해서 더 위험)
- ⚠ VCC 3V3 → 짧아지거나 항상 타임아웃
- ⚠ 초음파 두 개 동시 → 서로의 소리를 들음 → **번갈아 측정**

#### PIR + 초음파
> 넓게 "움직임" + 좁게 "얼마나 가까운가" → AND
- **PIR** → 넓게 "움직임이 있다"
- **초음파** → 좁게 "얼마나 가까운가"
- **AND** 로 묶으면 → 오작동이 준다

**기본 사용형태**
```c
float read_cm(void) {
    digitalWrite(TRIG, LOW);  delayMicroseconds(2);
    digitalWrite(TRIG, HIGH); delayMicroseconds(10);
    digitalWrite(TRIG, LOW);
    unsigned long us = pulseIn(ECHO, HIGH, 30000);    // 최대 30ms
    if (us == 0) return -1;                           // 실패
    return us / 58.0f;
}
bool alert = digitalRead(PIR) && (cm > 0 && cm < 30);
```

**연관 문제**
<<QA 6>>

### 7교시 · 수분 센서와 코드 모듈화

**키워드:** 토양 습도 센서, 수위 센서, 전기 분해 부식, GPIO 로 센서 전원, delay(10) 안정, 헤더 분리, sensor.h, sensor.cpp, include guard #ifndef, extern, multiple definition, undefined reference, 링커 ld collect2, .c vs .cpp, 새 탭

**개요**
- 토양·수위 센서는 계속 전기가 흐르면 전극이 부식된다 → **잴 때만 GPIO 로 전원**을 켠다. 핀만 다른 같은 함수가 늘어나면 **`.h`(선언) / `.cpp`(정의)** 로 나누고, `extern`·include guard·링크 오류를 C 과정 Day1 과 연결해 정리한다.

**핵심 용어**
- **전기 분해 부식**: 전극에 전류가 계속 흐르면 녹슨다 → V 핀을 **GPIO(14, 23)** 에 연결해 잴 때만 HIGH (1초에 10ms → 1%).
- **선언(.h) vs 정의(.cpp)**: 헤더에는 **타입·함수 원형·`extern` 선언**, 몸체와 전역 변수 **정의는 .cpp 한 곳**에.
- **`extern int x;`**: "다른 파일에 정의된 전역 변수가 있다"는 선언.
- **include guard**: `#ifndef SENSOR_H / #define SENSOR_H / … / #endif` — 두 번 포함돼도 한 번만.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| 스케치 탭 **⋯ → 새 탭** → `sensor.h`, `sensor.cpp` | 파일 추가 |
| `#include "sensor.h"` | 우리 헤더 포함 (따옴표) |

| 파일 | 담는 것 | 컴파일 결과 |
|---|---|---|
| `sensor.h` | 타입, 함수 원형, `extern` | 따로 컴파일 안 됨 — include 된 곳에 복사 |
| `sensor.cpp` | 함수 몸체, 전역 변수 정의 | `sensor.cpp.o` |
| `water_soil.ino` | `setup` · `loop` | `water_soil.ino.cpp.o` |
| 링크 | 두 `.o` 의 이름 연결 | `.elf` → `.bin` |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 7-1 `soil.ino` | 전원 GPIO 14 HIGH → 10ms → GPIO 34 읽기 → LOW, 1초마다 |
| 7-2 `water.ino` | 같은 모양으로 수위(35, 전원 23) — 핀만 다름 |
| 7-3 `sensor.h` / `sensor.cpp` / `water_soil.ino` | `powered_sensor_t {pin, pwr, name}` 와 `sensor_init`·`sensor_read`(+ `extern int sensor_read_count`)로 분리해 두 센서를 한 함수로 |
| 7-4 ⚠ 함정 | 링크 오류 4가지 |

**핵심 내용**

#### 잴 때만 전원
> 전원 GPIO 를 켜고 안정 시간 후 읽고 바로 끈다
- **`delay(10)`** → 출력이 **안정될 시간**
- **`powered_sensor_t` + `sensor_read(&soil)`** → **핀만 다른 센서를 한 함수로**

#### 파일 분리와 링크 오류
> 링크 오류는 컴파일러가 아니라 링커(`ld`, `collect2`)가 낸다

| 한 일 | 오류 | 원인 |
|---|---|---|
| `sensor.h` 에 `int sensor_read_count = 0;` | `multiple definition` | 헤더를 포함한 **파일마다** 정의 → 헤더엔 `extern` |
| `.cpp` 의 정의를 지움 | `undefined reference` | 선언만 있고 **정의 없음** |
| include guard 없이 두 번 포함 | `redefinition of 'struct …'` | 같은 typedef 두 번 |
| 탭 이름을 `sensor.c` | `undefined reference` | `.c` 는 **C**, `.ino` 는 **C++** 로 컴파일 → 이름 규칙이 달라서 → `.cpp` 로 |

**기본 사용형태**
```c
/* sensor.h */
#ifndef SENSOR_H
#define SENSOR_H
typedef struct { int pin; int pwr; const char *name; } powered_sensor_t;
void sensor_init(const powered_sensor_t *s);
int  sensor_read(const powered_sensor_t *s);
extern int sensor_read_count;
#endif
/* sensor.cpp */
#include <Arduino.h>
#include "sensor.h"
int sensor_read_count = 0;                 // 정의는 한 곳에만
int sensor_read(const powered_sensor_t *s) {
    digitalWrite(s->pwr, HIGH); delay(10);
    int v = analogRead(s->pin);
    digitalWrite(s->pwr, LOW); sensor_read_count++;
    return v;
}
```

**연관 문제**
<<QA 7>>

### 8교시 · 미니 프로젝트 — 다중 센서 모니터 · 정리

**키워드:** 다중 센서 모니터, sensor_t 구조체 배열, float (*read)(void), 함수 포인터 배열, period_ms last_ms 센서별 주기, 시리얼 플로터 형식, CSV 형식 머리줄, Serial.available Serial.read 모드 전환, NaN 대신 -99

**개요**
- 센서마다 **읽는 함수의 주소**를 구조체에 넣은 **표**를 만들어, `loop` 는 표를 돌기만 한다 (Day 1 8교시 `table[current]()` 와 같은 원리). 센서마다 다른 주기로 읽고, 시리얼로 `m` 을 보내면 **플로터 형식 ↔ CSV 형식**을 바꾼다.

**핵심 용어**
- **`float (*read)(void)`**: **float 를 돌려주는 인자 없는 함수의 포인터** 멤버. `sensors[i].read()` = i 번째 센서의 읽기 함수 호출.
- **센서별 주기**: `period_ms`(읽는 주기), `last_ms`(마지막 읽은 시각) → `now - last_ms >= period_ms` 면 읽기.
- **CSV**: 머리줄 `ms,pot,light,lm35,dht_t` 다음에 값만 쉼표로.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `Serial.available()` | 받은 바이트가 있는지 |
| `Serial.read()` | 한 바이트 읽기 → `== 'm'` 이면 모드 전환 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 8-1 함수 포인터 배열 | `sensor_t {name, read, value}` 배열을 돌며 `sensors[i].read()` 결과를 `pot:2041.0,light:1733.0` 형식으로 |
| 8-A `multi_monitor.ino` | 가변저항·조도·LM35·DHT11 네 센서를 주기(100/100/500/2000ms)별로 읽고, `m` 을 받으면 CSV 머리줄 출력 후 CSV 모드로 |

**핵심 내용**

#### 센서 표와 주기 읽기
> 센서를 데이터(표)로 — 추가는 표 한 줄
- **센서 수** → `NS = sizeof(sensors) / sizeof(sensors[0])`
- **TODO 3 조건** → `now - sensors[i].last_ms >= sensors[i].period_ms` → 읽고 `last_ms = now`
- **새 센서 추가** → 읽기 함수를 만들고 표에 한 줄 — loop 는 그대로

#### 출력 형식
> 플로터는 `이름:값`, CSV 는 머리줄 + 값
- **플로터** → `pot:2040,light:1731,lm35:24.6`
- **CSV 머리줄** → `ms,pot,light,lm35,dht_t` → `15230,2040.0,1731.0,24.6,25.0`
- **DHT 실패 시 -99** → NaN 이 CSV 를 깨지 않게 약속된 값으로

**기본 사용형태**
```c
typedef struct { const char *name; float (*read)(void); float value; uint32_t period_ms, last_ms; } sensor_t;
float read_pot(void) { return analogRead(34); }
float read_dht_t(void) { float t = dht.readTemperature(); return isnan(t) ? -99 : t; }
sensor_t sensors[] = { {"pot", read_pot, 0, 100, 0}, {"dht_t", read_dht_t, 0, 2000, 0} };
const int NS = sizeof(sensors) / sizeof(sensors[0]);
void loop() {
    uint32_t now = millis();
    for (int i = 0; i < NS; i++)
        if (now - sensors[i].last_ms >= sensors[i].period_ms) { sensors[i].value = sensors[i].read(); sensors[i].last_ms = now; }
    if (Serial.available() && Serial.read() == 'm') { csv = !csv; if (csv) Serial.println("ms,pot,dht_t"); }
}
```

**연관 문제**
<<QA 8>>
