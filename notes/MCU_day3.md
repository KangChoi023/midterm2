## Day 3 — 디스플레이 · I2C · 모터

### 1교시 · 7-Segment — 비트맵으로 숫자 그리기

**키워드:** 7-Segment, 세그먼트 a~g dp, 공통 음극 CC, 공통 양극 CA, 비트맵 배열 DIGITS, 0x3F 0x06, (pattern >> i) & 1, ~pattern 반전, |= 0x80 점, & 0x0F 범위

**개요**
- 숫자 모양을 **1바이트 비트맵**으로 표현하고(a = bit0 … dp = bit7), `(pattern >> i) & 1` 로 비트를 하나씩 꺼내 핀에 출력한다. 모양이 **코드가 아니라 데이터(표)** 라는 것이 Day 3 디스플레이의 공통 원리.

**핵심 용어**
- **7-Segment**: 막대 7개(a~g) + 점(dp)으로 숫자를 그리는 표시 장치.
```text
   ─a─
  f   b         비트:  7  6  5  4  3  2  1  0
   ─g─          이름: dp  g  f  e  d  c  b  a
  e   c
   ─d─  .dp     "0" = a b c d e f 켜짐 = 0011 1111 = 0x3F
```
- **공통 음극(CC)**: COM 이 GND(저항 거쳐), 세그먼트에 **HIGH** 면 켜짐. **공통 양극(CA)**: COM 이 3V3, 세그먼트에 **LOW** 면 켜짐 → 같은 표를 쓰려면 **`~pattern`** 으로 뒤집기.
- **비트맵**: 비트 하나 = 점(세그먼트) 하나. 모양을 바꾸려면 **표 한 줄만** 고친다.

**새로 나온 함수/명령** — 비트 연산 (C Day3)

| 식 | 뜻 |
|---|---|
| `(p >> i) & 1` | p 의 **i 번째 비트만 꺼냄** |
| `p \|= 0x80` | 다른 비트는 두고 **dp(bit7)만 켬** |
| `p &= ~(1 << i)` | i 번째 비트만 끔 |
| `~p` | 전체 반전 (CA 용) |
| `HEX_FONT[v & 0x0F]` | 0~15 로 만들어 배열 범위를 지킴 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 1-1 `seg_test.ino` | 세그먼트 8개를 하나씩만 HIGH → CC 면 하나씩 켜지고, CA 면 **하나만 꺼지고 나머지가 켜짐** |
| 1-2 `seg_digits.ino` | `DIGITS[10] = {0x3F, 0x06, 0x5B, …}` 와 `seg_show()`(비트를 꺼내 8핀에 출력, CA 면 `~`)로 0~9 와 점 표시 |
| 1-3 ⚠ 함정 | CA 를 GND 에 / 비트 순서 착각 / COM 한 핀만 / 저항 없이 |

**핵심 내용**

#### 7-Segment 비트맵
> 숫자 모양 = 1바이트, a = bit0 … dp = bit7
- **1** (b, c) → **0x06**
- **0** → **0x3F** (꺼진 것은 **g 와 dp**)
- **표의 장점** → 모양이 데이터라 **표 한 줄로 수정**, if 문 열 개가 필요 없음
- ⚠ a 를 bit7 로 착각 → 엉뚱한 모양 (오늘 규칙 **a = bit0, dp = bit7**)

#### CC · CA 와 저항
> 공통 단자의 극성과 전류 나눔
- **COM 에 220Ω 하나만** → 켜진 세그먼트가 많을수록 전류를 나눠 써서 **"8" 이 "1" 보다 어둡다**
- ⚠ CA 부품인데 COM 을 GND → 아무것도 안 켜짐 → COM 을 3V3, `COMMON_ANODE = true`
- ⚠ 저항 없이 COM 직결 → 과전류

**기본 사용형태**
```c
const int SEG_PINS[8] = {13, 14, 27, 26, 25, 33, 32, 23};       // a b c d e f g dp
const uint8_t DIGITS[10] = {0x3F,0x06,0x5B,0x4F,0x66,0x6D,0x7D,0x07,0x7F,0x6F};
void seg_show(uint8_t p) {
    if (COMMON_ANODE) p = ~p;
    for (int i = 0; i < 8; i++) digitalWrite(SEG_PINS[i], (p >> i) & 1);
}
seg_show(DIGITS[3] | 0x80);                                     // "3."
```

**연관 문제**
<<QA 1>>

### 2교시 · 74HC595 — 선 3개로 출력 8개

**키워드:** 74HC595, 시프트 레지스터, 직렬 → 병렬, DS, SH_CP 클럭, ST_CP 래치, OE GND, MR 3V3, Q0~Q7, Q7' 연결, my_shift_out, shiftOut, MSBFIRST, LSBFIRST

**개요**
- GPIO 가 모자랄 때 **직렬(한 줄로 한 비트씩) → 병렬(8비트 동시)** 로 바꾸는 74HC595 를 쓴다. 데이터(DS)·클럭(SH_CP)·래치(ST_CP) 세 선의 역할을 구분하고 `my_shift_out` 을 직접 구현한다. 1교시의 GPIO 8개가 **3개**로 준다.

**핵심 용어**
- **시프트 레지스터**: 클럭이 올라갈 때마다 비트를 한 칸씩 밀어 넣는 칸들. 줄 서기.
- **DS(SER)**: 넣을 비트. **SH_CP(SRCLK)**: **↑ 에지마다 한 칸 밀기**(클럭). **ST_CP(RCLK)**: **↑ 에지에 8비트를 출력 핀으로 복사(래치)**.
- **래치**: 올리기 전까지는 출력이 **바뀌지 않음** → 밀어 넣는 동안 깜빡임 없음.
- **OE** → GND(LOW = 출력 켜짐), **MR** → 3V3(LOW = 전체 지우기라 평소 HIGH). 떠 있으면 출력이 깜빡이거나 가끔 지워짐.
- **Q7'**: 다음 595 의 DS 로 → 클럭·래치는 공유해 **같은 3선으로 16, 24 … 개**.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `shiftOut` | `shiftOut(dataPin, clockPin, MSBFIRST, value)` | Arduino 내장 — `my_shift_out` 반복문과 같음 |
| `MSBFIRST` / `LSBFIRST` | — | bit7 먼저 / bit0 먼저 (LSB 면 `for (i = 0; i < 8; i++)`) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 2-1 `shift595.ino` | `my_shift_out`(bit7 부터 DS 에 올리고 SH_CP 펄스) + `write595`(래치 LOW → 밀기 → 래치 HIGH)로 `1 << i` 를 보내 a~dp 를 차례로 켬 |
| 2-2 `seg595.ino` | `shiftOut(…, MSBFIRST, DIGITS[d])` 로 1교시와 같은 숫자를 **핀 3개**로 |
| 2-3 ⚠ 함정 | 래치 안 올림 / MR·OE 떠 있음 / 거꾸로 꽂음 / LSBFIRST |

**핵심 내용**

#### 74HC595 동작
> 비트를 DS 에 올리고 SH_CP 로 밀기 ×8, ST_CP ↑ 에서 8개 출력이 동시에
- **①** 비트를 DS 에 올리고
- **②** SH_CP ↑ 로 한 칸 밀기 × 8
- **③** ST_CP ↑ → **8개 출력이 동시에** 바뀜
- **MSBFIRST 로 0x3F** → 처음 넣은 bit7 이 Q7 까지 밀려가고 **Q0 에는 bit0**
- **595 두 개** → 앞 **Q7' → 뒤 DS**, 클럭 · 래치 공유

#### 595 함정
> 래치 · 방향 · 비트 순서
- ⚠ 래치(ST_CP)를 안 올림 → 아무리 보내도 출력 그대로
- ⚠ **홈이 오른쪽으로 거꾸로** → VCC·GND 뒤바뀌어 **뜨거워짐, 즉시 USB 분리**
- ⚠ `LSBFIRST` → 비트 순서가 뒤집혀 거울 모양

**기본 사용형태**
```c
void my_shift_out(uint8_t data) {                 // MSB 먼저
    for (int i = 7; i >= 0; i--) {
        digitalWrite(DS, (data >> i) & 1);
        digitalWrite(SH, HIGH); digitalWrite(SH, LOW);   // 한 칸 밀기
    }
}
void write595(uint8_t d) { digitalWrite(ST, LOW); my_shift_out(d); digitalWrite(ST, HIGH); }   // 래치
```

**연관 문제**
<<QA 2>>

### 3교시 · 4자리 7-Segment — 멀티플렉싱과 타이머

**키워드:** 4자리 7-Segment, 세그먼트 공유, 자리 공통 D1~D4, 멀티플렉싱, 잔상 50Hz, 83Hz, 고스팅, 끄기→바꾸기→켜기, 하드웨어 타이머 인터럽트, hw_timer_t, timerBegin, timerAttachInterrupt, timerAlarm, % 10 / 10 자릿수

**개요**
- 4자리 모듈은 **세그먼트 a~dp 8선을 네 자리가 공유**한다. 한 순간에 **한 자리만** 켜고 빠르게 돌려 잔상으로 네 자리가 동시에 보이게 한다(멀티플렉싱). loop 가 바빠도 화면이 돌도록 **하드웨어 타이머 인터럽트**로 갱신한다.

**핵심 용어**
- **멀티플렉싱**: 공유 선을 시간으로 나눠 씀. 한 자리씩 빠르게 돌림. 사람 눈은 약 **50Hz 이상** 깜빡임을 못 느낀다.
- **고스팅**: 자리를 바꿀 때 이전 자리를 안 끄면 옆 자리에 숫자가 **희미하게 겹쳐** 보임 → **끄기 → 모양 바꾸기 → 켜기** 순서.
- **타이머 인터럽트**: Day 1 의 핀 변화 대신 **시간**이 ISR 을 부름. 규칙은 같다(`IRAM_ATTR`, 짧게, Serial·delay 금지, 공유 변수 volatile).

**새로 나온 함수/명령** (core 3.x)

| 이름 | 기본형 | 용도 |
|---|---|---|
| `timerBegin` | `hw_timer_t *timerBegin(uint32_t freq_hz)` | `timerBegin(1000000)` → 1틱 = 1µs |
| `timerAttachInterrupt` | `timerAttachInterrupt(t, &onTimer)` | ISR 등록 |
| `timerAlarm` | `timerAlarm(t, ticks, autoreload, count)` | `timerAlarm(t, 2000, true, 0)` = 2ms 마다 반복 |
| (2.x) `timerBegin(0, 80, true)` | — | 3.x 에서 `too many arguments` 에러 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 3-1 `mux_loop.ino` | loop 에서 `show_digit(k)`(전부 끄기 → 595 로 모양 → 그 자리만 LOW) 를 3ms 씩 돌려 "1234" 표시 — `delay(100)` 을 켜면 깜빡임 |
| 3-2 `mux_timer.ino` | 2ms 타이머 ISR 이 한 자리씩 돌고, loop 는 `show_number(count)` 로 `disp[]` 만 갱신 (loop 가 쉬어도 화면 유지) |
| 3-3 ⚠ 함정 | 고스팅 / 20ms / ISR 에서 Serial / 2.x 타이머 API / CA 모듈 |

**핵심 내용**

#### 멀티플렉싱
> 한 자리씩 빠르게 번갈아 켜면 동시에 켜진 것처럼 보인다
- **3ms × 4자리** → 12ms → **약 83Hz**
- **20ms 씩** → 80ms → 12Hz → 눈에 띄게 깜빡임
- ⚠ 자리 공통 단자(D)에 최대 8개 세그먼트 전류 → **한 자리만 오래 켜 두지 않기**

#### 타이머 ISR 로 화면 갱신
> loop 에 `delay` 가 있어도 화면은 타이머가 그린다
- ⚠ loop 멀티플렉싱에 `delay(100)` → **한 자리씩 오래 켜져 깜빡임** → 타이머 ISR 로 분리
- **`disp[]`** → ISR 과 loop 가 함께 씀 → **`volatile`**
- **`show_number(1234)`** → 오른쪽 자리부터 `disp[k] = DIGITS[n % 10]; n /= 10;` → k = 3 에 **4**
- **앞자리 0 끄기** → `k < 3` 조건으로 맨 오른쪽 "0" 은 남긴다

**기본 사용형태**
```c
volatile uint8_t disp[4]; volatile int cur = 0; hw_timer_t *timer;
void IRAM_ATTR onTimer() {
    digitalWrite(DIG[cur], HIGH);          // 지금 자리 끄기
    cur = (cur + 1) % 4;
    write595(disp[cur]);                   // 모양 바꾸기
    digitalWrite(DIG[cur], LOW);           // 다음 자리 켜기
}
void show_number(int n) { for (int k = 3; k >= 0; k--) { disp[k] = DIGITS[n % 10]; n /= 10; } }
timer = timerBegin(1000000);
timerAttachInterrupt(timer, &onTimer);
timerAlarm(timer, 2000, true, 0);          // 2ms 마다
```

**연관 문제**
<<QA 3>>

### 4교시 · 8×8 도트 매트릭스 — 2차원 비트맵

**키워드:** 8×8 도트 매트릭스, 1088AS, img[8] 바이트=행 비트=열, 행 스캔 멀티플렉싱, 극성 ROW_ON, ~img[row], 점 켜기 img[y] |= 0x80 >> x, 점 끄기, 16비트 띠 스크롤, 핀 표

**개요**
- 1교시의 1차원 비트맵을 **2차원**으로: 8바이트 배열 `img[8]` 에서 **바이트 = 행, 비트 = 열**. 행을 하나씩 빠르게 켜는 **행 스캔**을 타이머로 하고, 비트 시프트로 글자를 옆으로 스크롤한다.

**핵심 용어**
- **행 스캔**: 행·열 선을 공유하므로 한 번에 **한 행만** 켜야 한다(두 행을 켜면 다른 점까지 켜짐). 8행 × 1.5ms = 12ms → **약 83Hz**.
- **극성**: 점이 켜지려면 **행은 한쪽, 열은 반대 극성**이어야 LED 에 전류가 흐른다 → 행을 HIGH 로 켜는 제품은 열 데이터를 **`~img[row]`** 로 뒤집어 보냄.
- **1088AS 핀 순서**: 16핀이 행·열 순서로 나열되어 있지 **않다** → **핀 표**를 본다 (1:R5, 2:R7, 3:C2, …).

**새로 나온 함수/명령** — 점 다루기 (x = 0 이 bit7 = 왼쪽)

| 식 | 뜻 |
|---|---|
| `img[y] \|= 0x80 >> x` | 점 (x, y) 켜기 |
| `img[y] &= ~(0x80 >> x)` | 점 (x, y) 끄기 |
| `(FONT_H[r] << 8) \| FONT_I[r]` | 두 글자를 **16비트 띠**로 잇기 |
| `(line << shift) >> 8` | 띠를 왼쪽으로 밀며 **위 8비트 창**만 보기 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 4-1 `matrix_heart.ino` | 하트 `img[8]` 을 1.5ms 타이머 ISR 이 행마다(이전 행 끄기 → 595 로 열 데이터 → 다음 행 켜기) 스캔 |
| 4-2 `matrix_scroll.ino` | H·I 폰트를 16비트로 이어 `shift` 0~8 로 밀며 `img` 갱신 → "HI" 가 옆으로 스크롤 |
| 4-3 ⚠ 함정 | 핀 착각 / 상하좌우 뒤집힘 / 극성 반대 / 열 저항 없음 |

**핵심 내용**

#### 8×8 도트 매트릭스
> 행마다 1바이트, bit7 = 왼쪽 열 — 한 행씩 빠르게
- **하트 두 번째 행 `0b01100110`** → 켜지는 열(0 = 왼쪽) **1, 2, 5, 6**
- ⚠ 핀 1~8 을 행으로 착각 → 점이 흩어짐
- ⚠ 위아래 · 좌우 뒤집힘 → `ROWS[]` 순서 · `bit7 = 왼쪽` 확인
- ⚠ **모든 점이 켜지고 그림만 꺼짐** → 극성 반대 → **`ROW_ON` 변경**
- ⚠ 열에 저항 없이 → 열마다 220Ω

**기본 사용형태**
```c
volatile uint8_t img[8] = {0, 0b01100110, 0b11111111, 0b11111111, 0b01111110, 0b00111100, 0b00011000, 0};
void IRAM_ATTR onTimer() {
    digitalWrite(ROWS[row], !ROW_ON);
    row = (row + 1) % 8;
    write595(ROW_ON == HIGH ? ~img[row] : img[row]);   // 열은 반대 극성
    digitalWrite(ROWS[row], ROW_ON);
}
img[y] |= 0x80 >> x;                                    // 점 켜기
```

**연관 문제**
<<QA 4>>

### 5교시 · 1602A LCD — 병렬 4비트 연결

**키워드:** 1602A LCD, 16핀 병렬, 4비트 모드, RS 명령/데이터, E Enable, RW GND, D4~D7, V0 대비 가변저항, VDD 5V, 백라이트 A 220Ω, LiquidCrystal, lcd.begin, setCursor, 고정 폭 %4d, DDRAM 0x40, createChar 5×8

**개요**
- 키트의 LCD 는 I2C 백팩이 **없는 16핀 병렬형**이다. 데이터선 8개 중 **D4~D7 네 개만** 쓰는 4비트 모드(GPIO 6개)로 연결하고, 센서 값을 **고정 폭**으로 출력하며 사용자 문자(막대 그래프)를 만든다.

**핵심 용어**
- **4비트 모드**: 글자 한 개(8비트)를 **위 4비트 → 아래 4비트 두 번**에 나눠 보냄. 필요한 GPIO = **RS · E · D4~D7 = 6개**.
- **RS**: 0 = **명령**(지우기·커서 이동), 1 = **화면에 표시할 글자**.
- **E(Enable)**: HIGH → LOW 순간에 LCD 가 데이터를 읽어 감.
- **RW → GND**: 쓰기 전용 — LCD 가 **ESP32 쪽으로 5V 를 내보내지 않게**. (LCD 는 2.2V 이상을 HIGH 로 읽어 3.3V 신호로 동작)
- **V0(대비)**: 가변저항 가운데 → 글자 진하기. **"LCD 가 안 된다"의 대부분이 대비 문제**.
- **DDRAM 주소**: 1행 0x00~, **2행 0x40~** → `setCursor(4, 1)` = **0x44**.

| 핀 | 이름 | 연결 |
|---|---|---|
| 1 / 2 | VSS / VDD | GND / **5V** |
| 3 | V0 | 가변저항 가운데 |
| 4 / 5 / 6 | RS / RW / E | GPIO13 / **GND** / GPIO14 |
| 11~14 | D4~D7 | GPIO 23 · 32 · 33 · 4 |
| 15 / 16 | A / K | 220Ω → 5V / GND |

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `LiquidCrystal lcd(13, 14, 23, 32, 33, 4);` | RS, E, D4, D5, D6, **D7(=4)** |
| `lcd.begin(16, 2)` | 16칸 2줄 초기화 (누락 시 검은 네모만) |
| `lcd.setCursor(col, row)` / `lcd.print` / `lcd.printf` | 커서 (열, 행 — 0부터) / 출력 |
| `lcd.createChar(n, bytes)` / `lcd.write((uint8_t)n)` | 사용자 문자 0~7 정의(**5×8 점, 한 줄 5비트씩 8바이트**) / 출력 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 5-1 `lcd_hello.ino` | 윗줄 `Hello ESP32`, 아랫줄 `Day 3 LCD 1602A` |
| 5-2 `lcd_counter.ino` | 1200 → 0 으로 줄어드는 값을 `%4d` 고정 폭으로 같은 자리에, 아랫줄에 초 |
| 5-3 `lcd_sensor.ino` | 조도 raw·% 를 윗줄에, 꽉 찬 사용자 문자로 아랫줄에 0~16칸 막대 그래프 |
| 5-4 ⚠ 함정 | 대비·초기화·핀 순서·RW·VDD·백라이트·clear·버퍼 크기 |

**핵심 내용**

#### LCD 출력 규칙
> 덮어쓴 칸만 바뀐다 — 폭을 고정한다
- ⚠ 1000 다음 999 → "9990" → **`%4d`** 처럼 폭 고정
- ⚠ loop 마다 `lcd.clear()` → 깜빡이고 느림
- ⚠ `char line[16]` 에 16글자 → `'\0'` 자리 없음 → **17칸**

#### LCD 증상별 점검
> 화면 상태로 원인을 좁힌다
- **백라이트만 켜지고 글자 없음** → **V0 가변저항을 천천히 돌린다**
- **윗줄에 검은 네모 16개만** → 전원·대비 정상, **초기화 실패** (RS·E·D4~D7 핀 번호, `lcd.begin` 확인)
- **글자가 깨짐** → D4~D7 순서 · 느슨한 선
- ⚠ RW 를 GPIO/비워 둠 → LCD 가 5V 를 내보낼 수 있음 → **GND**
- ⚠ VDD 를 3V3 → 매우 흐림 → 5V
- ⚠ A 를 저항 없이 5V → 과전류

**기본 사용형태**
```c
#include <LiquidCrystal.h>
LiquidCrystal lcd(13, 14, 23, 32, 33, 4);           // RS E D4 D5 D6 D7
byte BAR[8] = {0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,0x1F};
void setup() { lcd.begin(16, 2); lcd.createChar(0, BAR); }
void loop() {
    char buf[17];                                   // 16 + '\0'
    snprintf(buf, sizeof buf, "light:%4d %3d%%", raw, pct);
    lcd.setCursor(0, 0); lcd.print(buf);
    lcd.setCursor(0, 1); for (int i = 0; i < 16; i++) i < bars ? lcd.write((uint8_t)0) : lcd.print(' ');
}
```

**연관 문제**
<<QA 5>>

### 6교시 · I2C 와 PCF8591 — 라이브러리 없이 직접

**키워드:** I2C, SDA SCL, 7비트 주소, ACK NACK, 풀업, Wire.begin(21, 22), I2C 스캐너, endTransmission 0, PCF8591, 0x48, 8비트 ADC 4ch, DAC, 제어 바이트 0x40 0x42, 첫 바이트 버림, requestFrom, 드라이버 구조체 오류 -1, Vcc 3V3

**개요**
- 5교시 LCD 는 선 6개가 필요했다. **I2C** 는 **SDA · SCL 두 선**에 여러 장치를 달고 **7비트 주소**로 구분한다. 스캐너로 장치를 찾고, 데이터시트의 **제어 바이트를 비트 연산으로** 만들어 PCF8591(ADC + DAC)을 라이브러리 없이 `Wire` 로 직접 제어한 뒤, 드라이버를 **구조체 + 함수 + 오류 반환**으로 설계한다.

**핵심 용어**
- **I2C**: SDA(데이터) · SCL(클럭) 두 선, 장치마다 **7비트 주소**가 달라 여러 장치를 한 버스에. ESP32 기본 **SDA 21 · SCL 22**.
- **ACK / NACK**: 주소를 부르면 그 장치가 응답(ACK). `endTransmission()` **0 = 응답 있음**, 2 = 주소 NACK, 그 밖 = 버스 오류.
- **PCF8591**: **8비트 ADC 4채널(AIN0~3) + 8비트 DAC 1채널(AOUT)**, 주소 **0x48**. 기판 풀업이 Vcc 에 연결 → **Vcc 는 반드시 3V3** (5V 면 SDA·SCL 이 5V).
- **제어 바이트**: bit6 = **DAC 켜기**, bit5·4 = 입력 방식(00 단일), bit2 = 자동 증가, bit1·0 = **채널**. AIN0 + DAC = **0x40**, AIN2 + DAC = **0x42**.
- **첫 바이트는 이전 값**: PCF8591 은 요청을 받는 순간 변환을 시작 → **2바이트를 요청하고 첫 바이트를 버린다**.
- **DAC**: PWM(빠른 켜기/끄기)이 아니라 **진짜 중간 전압**을 낸다.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `Wire.begin(21, 22)` | SDA, SCL |
| `Wire.beginTransmission(addr)` / `Wire.write(b)` / `Wire.endTransmission()` | 쓰기 시작 / 바이트 / 끝 (0 = 성공) |
| `Wire.requestFrom(addr, n)` / `Wire.read()` | n 바이트 요청 (받은 수 반환) / 한 바이트 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 6-1 `i2c_scan.ino` | 주소 1~126 에 `beginTransmission/endTransmission` 해서 0 이면 `발견: 0x48` |
| 6-2 `pcf_adc.ino` | `0x40 \| ch` 를 쓰고 2바이트 요청, 첫 바이트 버리고 0~255 반환 (실패 -1) |
| 6-3 `pcf_dac.ino` | `0x40` + 값(0~255)을 써서 Aout LED 가 부드럽게 밝아졌다 어두워짐 |
| 6-4 `pcf_lcd.ino` | `pcf8591_t {addr, last[4], errors}` 드라이버로 AIN0 와 오류 횟수를 LCD 에 |
| 6-5 ⚠ 함정 | SDA/SCL 반대 / Vcc 5V / 1바이트만 / 0x40 누락 / 반환값 무시 / LCD 스캔 |

**핵심 내용**

#### I2C 스캔과 제어 바이트
> 주소로 장치를 찾고, 제어 바이트의 비트로 동작을 정한다
- **LCD + PCF8591 연결 후 스캔** → **0x48 하나만** 나와도 정상 (병렬 LCD 는 I2C 장치가 아니다)
- **DAC 비트(bit6) 0** → **AOUT 꺼져 0V**
- ⚠ `requestFrom(…, 1)` → 항상 한 박자 늦은 값
- ⚠ 모듈은 **Scl 이 왼쪽** → 바꿔 꽂으면 스캐너가 못 찾음

#### 드라이버 = 구조체 + 함수 + 오류 반환
> 장치 상태를 구조체에, 실패는 정상 범위 밖의 값으로
- **구조체에 addr** → 칩이 둘(0x48, 0x49)이어도 **변수만 추가**
- **통신 실패 시 -1** → 정상 값(0~255)과 구분
- ⚠ 반환값을 안 보면 → 선이 빠져도 255 나 이전 값이 나와 고장을 모름

**기본 사용형태**
```c
#include <Wire.h>
typedef struct { uint8_t addr; uint8_t last[4]; int errors; } pcf8591_t;
int pcf_read(pcf8591_t *p, uint8_t ch) {
    Wire.beginTransmission(p->addr);
    Wire.write(0x40 | (ch & 0x03));                    // DAC 켜기 + 채널
    if (Wire.endTransmission() != 0) { p->errors++; return -1; }
    if (Wire.requestFrom(p->addr, (uint8_t)2) != 2) { p->errors++; return -1; }
    Wire.read();                                       // 이전 값 버림
    return p->last[ch & 0x03] = Wire.read();
}
pcf8591_t pcf = {0x48, {0}, 0};
Wire.begin(21, 22);
```

**연관 문제**
<<QA 6>>

### 7교시 · DC 모터 — L293D 와 릴레이

**키워드:** DC 모터, GPIO 직접 금지, 역전압, H-브리지, L293D, EN PWM 속도, IN1 IN2 방향, 브레이크, 자유 정지, motor_set 부호 있는 속도, ramp_to 가속, 20kHz, Brownout, 릴레이, 코일, COM NO NC, ACTIVE_LOW, 최소 전환 간격, 비상 정지, 아크

**개요**
- 모터는 **절대 GPIO 에 직접 연결하지 않는다**(수백 mA + 멈출 때 역전압). **H-브리지 L293D** 로 IN1·IN2 는 방향, EN 의 PWM 은 속도를 제어하고, **릴레이**로 모터 주전원을 켜고 끄는 단계를 거쳐 둘의 역할 차이를 정리한다.

**핵심 용어**
- **H-브리지**: 스위치 4개로 모터 전류 방향을 바꾸는 회로. L293D 는 ESP32 의 작은 신호로 큰 전류를 대신 흘려 준다.
- **방향 표**:

| IN1 | IN2 | 모터 |
|---|---|---|
| HIGH | LOW | 정방향 |
| LOW | HIGH | 역방향 |
| LOW | LOW | 자유 정지 (관성) |
| HIGH | HIGH | **브레이크** (빨리 멈춤) |

- **Brownout**: 모터 시작 순간 USB 5V 가 떨어져 ESP32 리셋(`Brownout detector was triggered`) → **천천히 가속**, PC 본체 포트.
- **릴레이**: 전자석으로 금속 접점을 움직이는 스위치 — **작은 신호로 분리된 큰 스위치**. 코일 쪽(S·+·−)만 ESP32, 접점(COM·NO·NC)은 별도 회로.
- **NO(Normally Open)**: 평소 열림, 코일 ON 이면 COM 과 연결. **NC(Normally Closed)**: 평소 닫힘, 코일 OFF 일 때 COM 과 연결.
- **아크**: 전류가 흐르는 채로 접점이 붙거나 떨어질 때 생기는 불꽃 → 접점 수명 단축.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `motor_set(int speed)` | -255~+255, **부호 = 방향** (IN1 = s > 0, IN2 = s < 0, `ledcWrite(EN, abs(s))`) |
| `ramp_to(int *cur, int target)` | 현재 속도를 **포인터로** 받아 5씩 20ms 간격으로 가속/감속 |
| `ledcAttach(EN, 20000, 8)` | 20kHz PWM — 귀에 거슬리는 소리가 안 들림 |
| `relay_set(bool on)` | `ACTIVE_LOW ? !on : on` (모듈에 따라 LOW 에서 켜짐) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 7-1 `motor_dir.ino` | 정방향 2초 → 정지 → 역방향 → 브레이크 반복 |
| 7-2 `motor_speed.ino` | `ramp_to` 로 0 → 200 → 0 → -200 → 0 부드럽게 |
| 7-3 ① `relay_basic.ino` | 시리얼 `1`/`0` 으로 릴레이 켜고 끄기 — 업로드 직후 이미 켜지면 ACTIVE_LOW |
| 7-3 ② `relay_contacts.ino` | COM 3V3, NO 빨강, NC 초록 → 코일 OFF 면 **초록(NC)**, ON 이면 빨강 |
| 7-3 ③ `relay_motor.ino` | 버튼으로 모터 켜고 끄기, **1초 안에 다시 바꾸면 무시**(접점 보호) |
| 7-3 ④ `relay_estop.ino` | 릴레이 = 주전원(비상 정지), L293D = 속도·방향. 시작: `motor_set(0)` → 릴레이 ON → 천천히 가속 / 정지: 릴레이부터 끊기 |

**핵심 내용**

#### DC 모터 방향과 속도
> 방향은 H-브리지, 속도는 PWM — 반대로 돌기 전에 정지
- **정방향 → 역방향** → **정지(감속) 후 반대로** (바로 바꾸면 큰 전류로 리셋·과열)
- **`motor_set(-150)`** → **역방향, duty 150**
- ⚠ L293D GND 핀 일부만 → 뜨거워짐 (4개 모두)
- ⚠ 모터 GND 와 ESP32 GND 분리 → 제멋대로

#### 릴레이
> 켜기·끄기만, 전기적으로 분리된 스위치
- **속도 조절 불가** → 켜기·끄기만, 항상 최고 속도
- ⚠ 릴레이를 PWM 으로 빠르게 → 접점이 떨리며 수명 단축
- **시작 순서** → **PWM 0 인 채로 릴레이를 켜면** 접점이 전류 없이 붙어 **불꽃이 없다**
- ⚠ 릴레이 코일 + 를 3V3 → 약함 (5V)
- ⚠ **릴레이 접점에 콘센트 전원 절대 금지** (5V 이하만)

#### L293D vs 릴레이
> 용도로 고른다

| | L293D (H-브리지) | 릴레이 |
|---|---|---|
| 할 수 있는 것 | 속도(PWM)·방향·브레이크 | 켜기·끄기만 |
| 전환 속도 | 초당 수만 번 | 초당 몇 번 — PWM 금지 |
| 수명 | 반도체 — 사실상 무제한 | 기계 접점 — 닳음 |
| 전기적 분리 | 없음 (GND 공통) | 있음 |
| 어울리는 곳 | 팬 속도, 바퀴 | 펌프·히터·램프 전원, 비상 정지 |

**기본 사용형태**
```c
void motor_set(int s) {
    if (s > 255) s = 255; if (s < -255) s = -255;
    digitalWrite(IN1, s > 0); digitalWrite(IN2, s < 0);
    ledcWrite(EN, abs(s));
}
void ramp_to(int *cur, int target) {
    while (*cur != target) { *cur += (target > *cur) ? 5 : -5; if (abs(target - *cur) < 5) *cur = target; motor_set(*cur); delay(20); }
}
ledcAttach(EN, 20000, 8);
motor_set(0); relay_set(true); delay(100);   // 시작: 0 에서 주전원 → 가속
```

**연관 문제**
<<QA 7>>

### 8교시 · 스테퍼 모터 · 미니 프로젝트 · 정리

**키워드:** 스테퍼 모터, 28BYJ-48, ULN2003, 코일 4개, 반스텝 8단계 SEQ, 4096 스텝 한 바퀴, (phase + dir + 8) % 8, 역방향, 스텝 간격, 코일 끄기 발열, 스마트 환기, 히스테리시스 27/26, 상태 바뀔 때만 동작, fan_speed 비례, %-5s

**개요**
- 스테퍼 모터는 **코일 4개를 정해진 순서로** 켜서 한 칸씩 돈다 — 순서를 **비트맵 표**로 두는 1교시와 같은 원리(비트 하나 = 코일 하나). 마지막으로 DHT11 온도에 따라 팬(L293D)과 환기구(스테퍼)를 제어하고 LCD 에 표시하는 **스마트 환기 시스템**을 만든다.

**핵심 용어**
- **스테퍼 모터(28BYJ-48)**: 코일 순서대로 한 칸씩 도는 모터. 기어 때문에 **반스텝 약 4096 스텝이 한 바퀴**(1024 = 90°). 드라이버는 ULN2003.
- **반스텝 시퀀스**: `SEQ[8] = {0001, 0011, 0010, 0110, 0100, 1100, 1000, 1001}` (bit0 = IN1). **거꾸로 읽으면 반대 방향**.
- **`(phase + dir + 8) % 8`**: dir = -1 로 0 에서 **-1 이 되면 7 로** 돌리려고 +8.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `step_once(dir)` | phase 를 한 칸 옮기고 `(SEQ[phase] >> i) & 1` 로 코일 4개 출력 |
| `step_motor(steps, us)` | 스텝 반복 후 **코일 모두 LOW** |
| `snprintf(buf, 17, "VENT:%-5s FAN:%3d", …)` | `%-5s` = 왼쪽 정렬 5칸 ("OPEN " vs "CLOSE" 길이 맞춤) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 8-1 `stepper.ino` | 1000µs 간격으로 4096 스텝(한 바퀴) 정방향, -1024 스텝(90°) 역방향 |
| 8-A `smart_vent.ino` | DHT11 온도로 `ST_CLOSED`(< 26°C 로 돌아옴) / `ST_OPEN`(≥ 27°C) 전이 — 상태가 바뀔 때만 스테퍼 ±90°, 열린 동안 팬 속도는 27°C → 120, 32°C → 255 비례, LCD 에 `T:24.0C` / `VENT:OPEN FAN:180` |

**핵심 내용**

#### 스테퍼 구동
> 코일 순서 표를 한 칸씩 — 너무 빠르면 떨기만 한다
- ⚠ 스텝 간격 200µs 처럼 너무 짧게 → **따라오지 못해 떨기만**
- **멈춘 뒤 코일 모두 LOW** → 멈춘 채 전류가 흐르면 **발열**

#### 스마트 환기
> 히스테리시스 + 상태가 바뀔 때만 동작
- **열림 27°C · 닫힘 26°C** → **히스테리시스** (Day 2 5교시) → 27 근처에서 여닫기 반복 방지
- **ST_OPEN 에서 계속 28°C** → 스테퍼가 계속 돌지 않게 **상태가 바뀔 때만** 호출
- **`fan_speed(29.5)`** → `120 + (29.5 − 27) × 135 / 5` = **187**

#### Day 3 여섯 문장
> 표시 · 시프트 레지스터 · 멀티플렉싱 · LCD/I2C · 드라이버 · 모터
- **①** 표시 모양은 비트맵 표, `(p >> i) & 1` 로 꺼내고 `|=`·`&= ~` 로 켜고 끔
- **②** 595 는 DS·SH_CP·ST_CP 3선으로 8개
- **③** 여러 자리는 멀티플렉싱, 갱신은 타이머, 끄기 → 바꾸기 → 켜기
- **④** 병렬 LCD 6선 (RW 는 GND), I2C 는 두 선에 주소
- **⑤** 제어 바이트는 비트 연산, 드라이버는 구조체 + 함수 + 오류 반환
- **⑥** 모터는 드라이버를 거쳐 — 방향은 H-브리지, 속도는 PWM, 스테퍼는 코일 순서 표

**기본 사용형태**
```c
const uint8_t SEQ[8] = {0b0001,0b0011,0b0010,0b0110,0b0100,0b1100,0b1000,0b1001};
void step_motor(int steps) {
    int dir = steps > 0 ? 1 : -1;
    for (int s = 0; s < abs(steps); s++) {
        phase = (phase + dir + 8) % 8;
        for (int i = 0; i < 4; i++) digitalWrite(COIL[i], (SEQ[phase] >> i) & 1);
        delayMicroseconds(1000);
    }
    for (int i = 0; i < 4; i++) digitalWrite(COIL[i], LOW);   // 발열 방지
}
if (state == ST_CLOSED && t >= 27)    { state = ST_OPEN;   step_motor(1024); }
else if (state == ST_OPEN && t < 26)  { state = ST_CLOSED; step_motor(-1024); }
```

**연관 문제**
<<QA 8>>
