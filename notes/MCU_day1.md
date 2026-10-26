## Day 1 — ESP32 빌드 과정과 디지털 I/O

> MCU 과목은 ESP32 보드 실습이라 이 정리의 출력값은 **강의 자료 기준**이다 (PC 에서 실행 불가).

### 1교시 · 오리엔테이션 · ESP32 구조

**키워드:** MCU, ESP32, ESP-32U 모듈, DevKit 보드, Xtensa LX6 듀얼코어 240MHz, SRAM 520KB, 4MB Flash, CP2102 USB-UART, EN 버튼, Boot 버튼, 핀 분류, 입력 전용 34~39, Strapping 핀, Flash 핀 6~11, 3.3V 로직

**개요**
- PC 의 C 프로그램은 OS 위에서 돌지만, MCU 프로그램은 **OS 없이 칩 하나에서** 돈다. 1교시는 ESP32 의 **칩 → 모듈 → 보드** 구성과, 38개 핀을 **쓸 수 있는 핀 / 쓰면 안 되는 핀**으로 나누는 법. `printf` 가 화면 대신 **UART → CP2102 → USB → PC** 로 나간다는 점이 PC 와 가장 큰 차이.

**핵심 용어**
- **MCU(Microcontroller Unit)**: CPU · 메모리 · 입출력이 **한 칩에** 들어 있고 OS 없이 바로 도는 컴퓨터. 원룸(모든 게 한 방) vs PC 는 여러 방이 있는 집.
- **ESP32 칩**: Xtensa LX6 **듀얼코어 최대 240MHz**, SRAM 520KB, Wi-Fi · Bluetooth. `setup()`/`loop()` 는 **Core 1 (loopTask)** 에서 돈다.
- **모듈(ESP-32U)**: 칩 + 4MB Flash + 크리스털 + 안테나 커넥터를 금속 캔으로 덮은 단위. **U = 외부 안테나형** → Wi-Fi 쓸 때 안테나를 꽂는다.
- **CP2102**: **USB ↔ UART 변환** 칩. PC 의 `COMx` 포트가 바로 이 칩.
- **EN 버튼** = 리셋, **Boot 버튼** = 누른 채 리셋하면 다운로드 모드(GPIO 0 을 LOW).
- **Strapping 핀**: 부팅 순간의 전압으로 동작 모드가 정해지는 핀(0, 2, 5, 12, 15).

**새로 나온 함수/명령** — 핀 분류표 (매일 보는 표)

| 분류 | 핀 | 규칙 |
|---|---|---|
| 전원 | 3V3, 5V, GND | 3V3 은 출력용 전원, 5V 는 USB 전원 그대로 |
| 일반 GPIO (출력 가능) | 4, 13, 14, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33 | 오늘 쓰는 핀은 여기서 고름 |
| 입력 전용 | 34, 35, 36(VP), 39(VN) | **출력 불가, 내부 풀업 없음**. 아날로그 센서용 (Day 2) |
| Strapping | 0, 2, 5, 12, 15 | 부팅 모드 결정. **12 는 피한다** |
| Flash 연결 | 6~11 (보드 표기 CLK, D0~D3, CMD) | **절대 사용 금지** — 내부 SPI Flash 선, 부팅 불가 |
| UART0 | 1(TX), 3(RX) | 업로드·시리얼 모니터가 사용 중 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 1-1 보드 구성 | ESP-32U 모듈 · CP2102 · 레귤레이터(5V→3.3V) · EN · Boot · Micro USB 의 역할 확인 |
| 1-2 핀 분류표 | 38개 핀을 전원/일반/입력 전용/Strapping/Flash/UART0 으로 분류 |
| 1-3 ⚠ 함정 | "핀이 있으니 쓸 수 있겠지" — 잘못 쓴 핀별 결과 |

**핵심 내용**

#### 핀 선택 함정
> 아무 핀이나 쓰면 안 된다 — 입력 전용 · Flash · Strapping 핀
- ⚠ **GPIO 34 에 LED** → `digitalWrite(34, HIGH)` 해도 **아무 일 없음** (입력 전용)
- ⚠ **GPIO 9 (보드 표기 D2)** → **부팅 실패 · 무한 리셋** (내부 Flash 선)
- ⚠ **GPIO 12 에 풀업 부품** → 부팅 시 Flash 전압이 잘못 잡혀 업로드·부팅 실패
- **`is_safe_output(15)`** → 15 는 Strapping 이라 "안전한 출력 핀" 목록이 아니면 **0**

#### 3.3V 로직
> ESP32 GPIO 는 3.3V — 5V 신호를 직접 넣지 않는다
- ⚠ **5V 센서 출력을 직접** → 핀 손상 위험 → Day 2 에서 **분압**
- **핀 전류 한계** → 지킨다 (LED 는 저항 필수)

#### 브레드보드와 키트
> 보드 폭과 전용 부품을 확인한다
- ⚠ **38핀 보드를 브레드보드 가운데** → 한쪽 열만 남음 → 한쪽 끝에 걸치거나 **M-F 점퍼선**
- **GPIO Extension Board · 40핀 케이블** → **Raspberry Pi 전용** (Embedded IO 과목)

**기본 사용형태**
```c
/* 안전한 출력 핀만 허용 */
int is_safe_output(int pin) {
    static const int ok[] = {4, 13, 14, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33};
    for (unsigned i = 0; i < sizeof ok / sizeof ok[0]; i++) if (ok[i] == pin) return 1;
    return 0;
}
```

**연관 문제**
<<QA 1>>

### 2교시 · 개발 환경과 빌드 과정

**키워드:** Arduino IDE 2.x, 보드 매니저 URL, esp32 by Espressif 3.x, ESP32 Dev Module, CP210x 드라이버, COM 포트, 스케치 .ino, 컴파일 Ctrl+R, 업로드 Ctrl+U, 시리얼 모니터 115200, Boot 버튼 Connecting, 빌드 단계 .ino→.cpp→.o→.elf→.bin, 0x10000, app_main, loopTask, arduino-cli

**개요**
- Arduino IDE 와 ESP32 코어를 설치하고 **새 스케치 → 저장 → 컴파일 → 업로드 → 시리얼 모니터** 순서를 익힌다. `.ino` 가 C 과정의 gcc 빌드와 같은 단계를 거쳐 `.bin` 이 되고 **Flash 의 정해진 주소**에 기록되는 과정, `setup()`/`loop()` 를 누가 부르는지를 본다.

**핵심 용어**
- **스케치(.ino)**: Arduino 프로그램 파일. **폴더 이름 = 파일 이름**이어야 열린다 (`hello_esp32\hello_esp32.ino`).
- **보드 매니저 URL**: `https://espressif.github.io/arduino-esp32/package_esp32_index.json` — ESP32 보드 패키지를 찾게 함. 설치할 것은 **esp32 by Espressif Systems 3.x** ("Arduino ESP32 Boards" 아님).
- **baud**: 시리얼 통신 속도. 코드의 `Serial.begin(115200)` 과 모니터 속도가 **같아야** 한다.
- **app_main / loopTask**: ESP-IDF 의 진짜 시작점이 `loopTask` 를 Core 1 에 만들고, 그 안에서 `setup()` 한 번 → `loop()` 무한 반복.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `Serial.begin(115200)` / `Serial.println` / `Serial.printf` | 시리얼 출력 (형식 지정자는 C 와 같음) |
| `ESP.getChipModel()` `getChipCores()` `getCpuFreqMHz()` `getFlashChipSize()` `getFreeHeap()` | 칩 정보 |
| `millis()` / `delay(ms)` | 부팅 후 ms / 기다리기 |
| `pcTaskGetName(NULL)` / `xPortGetCoreID()` | 지금 태스크 이름 / 코어 번호 |
| `Ctrl+R` 컴파일 · `Ctrl+U` 업로드 · `Ctrl+Shift+M` 시리얼 모니터 | IDE 단축키 |
| `arduino-cli compile --fqbn esp32:esp32:esp32 …` / `upload -p COM5` / `monitor` | 명령줄 빌드 (심화) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 2-1 `hello_esp32.ino` | 칩 모델·코어 수(2)·CPU(240MHz)·Flash(4194304)·남은 heap 을 출력하고 1초마다 uptime 출력 |
| 2-2 빌드 로그 읽기 | "자세한 출력"으로 전처리 → 컴파일 → 링크 → elf2image → 크기 보고 → 업로드 단계를 찾기 |
| 2-3 setup/loop 호출자 | `setup` 안에서 태스크 이름·코어 출력 → `task=loopTask core=1` |
| 2-4 ⚠ 함정 | 시리얼 모니터를 9600 으로 → 글자 깨짐 |

**핵심 내용**

#### 빌드 단계
> .ino → .cpp → .o → .elf → .bin, 앱 이미지는 0x10000 에

| 단계 | 로그에서 | C 대응 |
|---|---|---|
| 1 스케치 전처리 | `.ino` → `.ino.cpp` (함수 원형 자동 추가) | 전처리 |
| 2 컴파일 | `xtensa-esp32-elf-g++ -c … -o .cpp.o` | `gcc -c` |
| 3 링크 | `-o .ino.elf` (core 라이브러리와 결합) | 링크 |
| 4 이미지 변환 | `esptool elf2image` → `.ino.bin` | (PC 에는 없음) |
| 5 크기 보고 | `Sketch uses … bytes` | `size` |
| 6 업로드 | `write_flash 0x1000 bootloader 0x8000 partitions 0xe000 boot_app0 0x10000 app.bin` | (PC 에는 없음) |

- **MCU 에는 OS 가 없다** → Flash 의 정해진 주소에 직접 써 두고, 전원이 켜지면 ROM → bootloader → 앱 순서로 실행
- **`.elf` 가 `.bin` 보다 큰 이유** → 디버그 정보
- **컴파일 오류** → `hello_esp32.ino:4:5: error: 'Serail' was not declared` = **4번째 줄 5번째 칸** 오타 (gcc 와 같은 형식)

#### setup · loop 의 정체
> `setup()`/`loop()` 는 FreeRTOS 태스크(loopTask) 안의 평범한 함수
- **C 와 달리** → `main()` 이 없다
- **Day 4** → 그 옆에 태스크를 더 만든다

#### 업로드 · 연결 문제 해결
> 증상별 조치

| 증상 | 조치 |
|---|---|
| 업로드가 `Connecting.....` 에서 멈추고 `Failed to connect` | Connecting 중에 **Boot 버튼을 누르고 있다가** Writing 이 나오면 뗀다 |
| 전원 LED 는 켜지는데 포트가 안 생김 | **충전 전용 케이블** · CP210x 드라이버 미설치 |
| Micro USB 가 안 들어감 | 힘주지 말고 **뒤집어 넓은 쪽끼리** 맞춘다 |
| 한글 계정명 경로 빌드 오류 | 스케치북 위치를 `C:\esp32` 같은 **영문 경로**로 |
| 글자가 깨짐 | 모니터 속도를 **115200** 으로 (코드 문제 아님) |
| 업로드 도중 끊김 | Upload Speed 115200 |

**기본 사용형태**
```c
void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.printf("Chip %s, %d cores, %lu MHz\n", ESP.getChipModel(), ESP.getChipCores(), ESP.getCpuFreqMHz());
    Serial.printf("setup: task=%s core=%d\n", pcTaskGetName(NULL), xPortGetCoreID());   // loopTask 1
}
void loop() {
    Serial.printf("uptime %lu ms\n", millis());
    delay(1000);
}
```

**연관 문제**
<<QA 2>>

### 3교시 · Digital Output — LED 와 신호등

**키워드:** 브레드보드, 가운데 홈, LED 극성 긴 다리 +, 전류 제한 저항 220Ω, 옴의 법칙, 순방향 전압 Vf, pinMode OUTPUT, digitalWrite HIGH LOW, 배열 + sizeof, enum, switch, static 지역 변수, 신호등 상태

**개요**
- LED 를 **안전하게** 켜는 회로(전류 제한 저항)를 옴의 법칙으로 계산하고, `pinMode`·`digitalWrite` 로 여러 LED 를 배열로 제어한 뒤, `enum` + `switch` 로 **상태 기반 신호등**을 만든다. C 과정의 배열 길이 계산·static·enum 이 그대로 쓰인다.

**핵심 용어**
- **브레드보드**: 한 열의 **a~e 세로 다섯 칸**이 서로 이어져 있고(f~j 도 따로), **가운데 홈 위아래는 끊겨** 있다. 그래서 저항을 `e4–f4` 처럼 홈을 건너 꽂아 이어 준다.
- **LED 극성**: **긴 다리 = +(애노드)** 를 저항/전원 쪽, 짧은 다리를 GND 쪽.
- **순방향 전압 Vf**: LED 에 걸리는 거의 일정한 전압 (빨강 약 2.0V).
- **전류 제한 저항**: 나머지 전압을 받아 전류를 정한다. 없으면 전류가 핀 한계를 넘어 LED·GPIO 손상.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `pinMode` | `pinMode(pin, OUTPUT)` | 핀을 출력으로 설정 (빠뜨리면 거의 안 켜짐) |
| `digitalWrite` | `digitalWrite(pin, HIGH/LOW)` | 3.3V / 0V 출력 |
| 옴의 법칙 | `I = (3.3V − Vf) ÷ R` | LED 전류 계산 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 3-1 저항 계산 | 색별 Vf 로 220Ω·100Ω·10kΩ 일 때 전류 계산 |
| 3-2 `leds.ino` | `LED_PINS[] = {25, 26, 27}` 배열과 `N_LED = sizeof/sizeof` 로 3개 LED 를 차례로 0.3초씩 켬 |
| 3-3 `traffic.ino` | `enum {ST_RED, ST_GREEN, ST_YELLOW}` + `switch` 로 빨강 3초 → 초록 3초 → 노랑 1초 반복, 시리얼에 상태 출력 |
| 3-4 ⚠ 함정 | LED 가 안 켜질 때 점검 순서 |

**핵심 내용**

#### LED 전류와 저항
> 220Ω 이 기준
- **빨강 220Ω** → `(3.3 − 2.0) / 220 ≈ 5.9mA`
- **100Ω** → 13mA / **10kΩ** → 0.13mA (거의 안 보임)
- **220Ω 색띠** → 빨강 빨강 갈색

#### 배열과 static 상태
> C 문법이 그대로 쓰인다
- **`N_LED = sizeof(LED_PINS) / sizeof(LED_PINS[0])`** → **3** (C Day2 배열 길이)
- **`static light_t state = ST_RED;`** → loop 사이 상태 유지
- ⚠ static 을 빼면 → `loop()` 마다 ST_RED 로 다시 초기화 → **빨강만 계속** (C Day1 static 지역 변수)
- **`enum` 값** → 0, 1, 2 … → `ST_YELLOW` = **2**

#### LED 가 안 켜질 때
> 점검 순서
- **①** LED 방향 (긴 다리)
- **②** 브레드보드 열 (같은 5칸 줄, 홈 건너기)
- **③** GND 연결
- **④** 핀 번호 (입력 전용 34~39 아닌지)
- **⑤** `pinMode(OUTPUT)`
- **보드 폭** → DevKit 은 넓어 한쪽 핀에 꽂을 곳이 없다 → 옆에 두고 **M-F 점퍼선**

**기본 사용형태**
```c
const int LED_PINS[] = {25, 26, 27};
const int N_LED = sizeof(LED_PINS) / sizeof(LED_PINS[0]);
typedef enum { ST_RED, ST_GREEN, ST_YELLOW } light_t;
void setup() { for (int i = 0; i < N_LED; i++) pinMode(LED_PINS[i], OUTPUT); }
void loop() {
    static light_t state = ST_RED;              // static 필수
    switch (state) {
        case ST_RED:    setLights(1, 0, 0); delay(3000); state = ST_GREEN;  break;
        case ST_GREEN:  setLights(0, 0, 1); delay(3000); state = ST_YELLOW; break;
        case ST_YELLOW: setLights(0, 1, 0); delay(1000); state = ST_RED;    break;
    }
}
```

**연관 문제**
<<QA 3>>

### 4교시 · 레지스터 직접 제어

**키워드:** 레지스터, 메모리 맵 I/O, GPIO_OUT_REG, GPIO_OUT_W1TS_REG, GPIO_OUT_W1TC_REG, GPIO_OUT1_W1TS_REG, REG_WRITE, 비트마스크, 1UL, 읽기-수정-쓰기, micros 속도 비교

**개요**
- `digitalWrite` 의 뒤에는 **메모리 주소에 있는 32비트 레지스터**가 있다. C 과정 Day3 의 비트 연산으로 레지스터에 직접 써서 **여러 핀을 같은 순간에** 바꾸고, `digitalWrite` 와 속도를 비교한다. 전체를 덮어쓰는 레지스터와 **W1TS/W1TC** 의 차이가 핵심.

**핵심 용어**
- **레지스터**: 주변장치를 제어하는 특수 메모리 주소. 거기에 쓰면 핀이 바뀐다 (C Day3 volatile·MMIO).
- **`GPIO_OUT_REG`**: 0~31번 핀 출력값 32비트 전체를 **덮어씀**.
- **W1TS(Write 1 To Set)**: **1 인 비트만 켜기**, 0 인 비트는 영향 없음.
- **W1TC(Write 1 To Clear)**: **1 인 비트만 끄기**.
- **읽기-수정-쓰기(RMW)**: 읽고 → 비트 바꾸고 → 다시 쓰기. 중간에 인터럽트가 끼어들면 다른 핀 변경이 사라질 수 있다. W1TS/W1TC 는 **다른 핀을 읽지 않아** 안전.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `#include "soc/soc.h"`, `"soc/gpio_reg.h"` | 레지스터 이름·`REG_WRITE` |
| `REG_WRITE(GPIO_OUT_W1TS_REG, mask)` | mask 의 1 비트 핀들 켜기 |
| `REG_WRITE(GPIO_OUT_W1TC_REG, mask)` | mask 의 1 비트 핀들 끄기 |
| `GPIO_OUT1_W1TS_REG` / `W1TC` | **32~39번 핀** — `1UL << (pin - 32)` |
| `micros()` | 부팅 후 µs (속도 측정) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 4-1 `bits.ino` | `reg \|= 1UL<<25; reg \|= 1UL<<27;` → `print_bits` 로 32비트 출력, `&= ~` 로 끄기, `>> & 1` 로 검사 |
| 4-2 `reg_leds.ino` | `LED_MASK = (1UL<<25)\|(1UL<<26)\|(1UL<<27)` 를 W1TS/W1TC 에 써서 LED 3개를 **동시에** 켜고 끔 |
| 4-3 `reg_speed.ino` | `digitalWrite` 와 `REG_WRITE` 로 HIGH/LOW 를 10만 번씩 반복해 µs 와 배율 비교 |
| 4-4 ⚠ 함정 | `GPIO_OUT_REG` 에 한 비트만 쓰면 남의 핀이 전부 꺼짐 |

**핵심 내용**

#### 비트 연산으로 레지스터 값 만들기
> 켜기는 OR, 끄기는 AND NOT — 마스크는 항상 `1UL`
- **`reg = 0; reg |= 1UL<<25; reg |= 1UL<<27;`** → **`0x0A000000`** (`00001010_00000000_…`)
- **이어서 `reg &= ~(1UL<<25)`** → **`0x08000000`**
- ⚠ `1 << 31` → `int` 부호 비트 → 부호 있는 시프트 범위 문제 → **`1UL`**
- **0~7 을 2진수로** (핀 25·26·27 = 비트 0·1·2) → n = 5 (`101`) → **25, 27** 켜짐

#### W1TS · W1TC 레지스터
> 1 을 쓴 비트의 핀만 켜거나 끈다 — 나머지는 그대로
- **`REG_WRITE(GPIO_OUT_W1TS_REG, 1UL << 26)`** → **26번 핀만 켬**
- **W1TC 에 `0x0C000000`** → **26, 27 끔** (비트 26·27)
- **13번 켜진 상태에서 W1TC 에 `1UL<<13`** → 13번 꺼짐
- **GPIO 33** → `GPIO_OUT1_W1TS_REG` 에 **`1UL << 1`** (32 를 뺀다)
- ⚠ `REG_WRITE(GPIO_OUT_REG, 1UL<<25)` → 32비트 통째로 덮어써 **다른 핀 전부 LOW**
- ⚠ `pinMode` 없이 레지스터만 → 출력이 활성화 안 됨

#### 레지스터를 쓸 때
> 여러 핀을 같은 순간에, 또는 아주 빠르게
- **알맞은 경우** → **여러 핀을 같은 순간에** 바꿀 때, 아주 빠른 토글
- **`digitalWrite` 가 느린 이유** → 핀 번호 검사 · 테이블 조회 (배율은 보드·core 마다 다름)

**기본 사용형태**
```c
#include "soc/soc.h"
#include "soc/gpio_reg.h"
#define LED_MASK ((1UL << 25) | (1UL << 26) | (1UL << 27))
pinMode(25, OUTPUT); pinMode(26, OUTPUT); pinMode(27, OUTPUT);   // 방향 설정은 필요
REG_WRITE(GPIO_OUT_W1TS_REG, LED_MASK);     // 3개 동시에 ON
REG_WRITE(GPIO_OUT_W1TC_REG, LED_MASK);     // 3개 동시에 OFF
REG_WRITE(GPIO_OUT1_W1TS_REG, 1UL << (33 - 32));   // GPIO 33
```

**연관 문제**
<<QA 4>>

### 5교시 · Digital Input — 풀업과 디바운스

**키워드:** digitalRead, 플로팅, INPUT, INPUT_PULLUP, INPUT_PULLDOWN, 풀업 저항, 채터링 bounce, 하강 엣지, 디바운스, millis 논블로킹, unsigned 뺄셈 오버플로, button_t 구조체 포인터, 기울기 센서

**개요**
- 버튼 입력의 두 문제 — **플로팅**(안 눌러도 값이 요동)과 **채터링**(한 번 눌러도 여러 번) — 을 풀업과 **millis() 기반 논블로킹 디바운스**로 해결하고, 버튼 상태를 **구조체 + 포인터**로 관리해 같은 함수를 여러 입력에 재사용한다.

**핵심 용어**
- **플로팅(floating)**: 핀이 **어디에도 연결되지 않은** 상태 → 주변 잡음을 읽어 0/1 이 요동. 손을 가까이 대면 더 심함.
- **풀업(pull-up)**: 내부 저항으로 핀을 3.3V 쪽으로 당겨 둠 → 안 누르면 HIGH, **누르면(GND 연결) LOW**.
- **채터링(bounce)**: 기계식 접점이 붙는 순간 수 ms 동안 붙었다 떨어졌다 반복 → 한 번 눌러도 `press #1, #2, #3`.
- **디바운스(debounce)**: 값이 일정 시간(예 30ms) **그대로 유지될 때만** 확정.
- **논블로킹**: `delay` 로 멈추지 않고 **시각을 기록·비교** → loop 가 다른 일도 함.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `digitalRead(pin)` | HIGH/LOW 읽기 |
| `pinMode(pin, INPUT_PULLUP)` | 내부 풀업 (핀 ↔ 버튼 ↔ GND) |
| `INPUT_PULLDOWN` | 내부 풀다운 (핀 ↔ 버튼 ↔ 3V3, 누르면 HIGH) |

| 모드 | 안 누름 | 누름 | 배선 |
|---|---|---|---|
| `INPUT` | 불안정 | 불안정 | 외부 저항 필요 |
| `INPUT_PULLUP` | 1 (HIGH) | **0 (LOW)** | 핀 ↔ 버튼 ↔ GND |
| `INPUT_PULLDOWN` | 0 | 1 | 핀 ↔ 버튼 ↔ 3V3 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 5-1 `floating.ino` | `INPUT` 으로 버튼 핀을 100ms 마다 출력 → 안 눌러도 `0 1 1 0 1 …` |
| 5-2 `chatter.ino` | `INPUT_PULLUP` 에서 HIGH→LOW 엣지를 셈 → 한 번 눌러도 press 가 여러 번 |
| 5-3 `debounce.ino` | `button_t {pin, stable, last_raw, changed_at}` 와 `button_pressed(&btn)` 로 30ms 안정 시 눌림 확정, LED 토글 |
| 5-4 기울기 센서 | `button_t tilt = {17, …}` 하나 더 만들어 **같은 함수** 재사용 |
| 5-5 ⚠ 함정 | GPIO 34~39 에 버튼 + INPUT_PULLUP → 요동 (내부 풀업 없음) |

**핵심 내용**

#### 풀업 버튼
> 평소 HIGH, 누르면 LOW — 눌림 = 하강 엣지
- **누르면** → **LOW**
- **눌림** → **HIGH → LOW (하강 엣지)**
- **배선** → **대각선 두 다리** (방향과 상관없이 누를 때만 이어짐)

#### 논블로킹 디바운스
> `delay` 대신 `millis()` 비교 — 흔들림이 끝날 때까지 기다렸다가 확정
- **강의 측정** → 흔들림 약 **8ms** → 디바운스 **20ms** 면 정확히 1번 (강의 코드는 30ms)
- **`delay(30)` 대신 `millis()`** → loop 를 멈추지 않아 다른 버튼 · LED 도 동시에
- **`millis() - b->changed_at > 30`** → **뺄셈으로** 비교 → 약 **49.7일** 뒤 millis 가 0 으로 돌아와도 정상
- ⚠ `millis() > changed_at + 30` → 그 순간 틀린다

#### 구조체 포인터로 재사용
> 입력마다 상태를 따로 두고 함수는 하나
- **`button_pressed(&btn)`, `button_pressed(&tilt)`** → 같은 함수 재사용

**기본 사용형태**
```c
typedef struct { int pin; int stable; int last_raw; uint32_t changed_at; } button_t;
int button_pressed(button_t *b) {
    int raw = digitalRead(b->pin);
    if (raw != b->last_raw) { b->last_raw = raw; b->changed_at = millis(); }
    if (millis() - b->changed_at > 30 && raw != b->stable) {   // 뺄셈 비교
        b->stable = raw;
        return raw == LOW;                 // 눌림 확정
    }
    return 0;
}
button_t btn = {4, HIGH, HIGH, 0};
void setup() { pinMode(btn.pin, INPUT_PULLUP); }
void loop() { if (button_pressed(&btn)) Serial.println("pressed"); }
```

**연관 문제**
<<QA 5>>

### 6교시 · Interrupt — ISR 과 공유 변수

**키워드:** 폴링, 인터럽트, ISR, attachInterrupt, digitalPinToInterrupt, FALLING RISING CHANGE, IRAM_ATTR, volatile, portMUX_TYPE, portENTER_CRITICAL, portENTER_CRITICAL_ISR, 임계 구역, ISR 규칙, 함수 포인터 콜백

**개요**
- loop 가 바쁘면(`delay(2000)`) 폴링으로는 빠른 버튼을 놓친다. **하드웨어 인터럽트**로 눌리는 순간 ISR 이 실행되게 하고, ISR 과 loop 가 **공유하는 변수**를 `volatile` 과 **임계 구역**으로 지킨다. Linux Day1 의 시그널 핸들러 규칙과 같은 사고방식.

**핵심 용어**
- **폴링(polling)**: loop 가 **그 순간** 직접 읽을 때만 안다. 바쁘면 놓친다.
- **인터럽트(interrupt)**: 하드웨어가 사건이 생긴 순간 CPU 를 불러 **ISR** 을 실행. 초인종.
- **ISR(Interrupt Service Routine)**: 인터럽트 때 실행되는 함수. **표시만 하고 나온다**.
- **`IRAM_ATTR`**: ISR 코드를 Flash 가 아닌 **내부 RAM** 에 둔다 → Flash 접근 중에도 즉시 실행.
- **`volatile`**: "코드 밖에서 바뀐다, **매번 메모리에서 읽어라**" (C Day3).
- **임계 구역(critical section)**: `portENTER_CRITICAL` ~ `EXIT` 사이에는 인터럽트가 끼어들지 못함. 그 안에서 눌리면 **구간이 끝난 뒤** ISR 실행.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `attachInterrupt` | `attachInterrupt(digitalPinToInterrupt(pin), isr, FALLING)` | 인터럽트 등록. `isr` 는 **함수 포인터** |
| 모드 | `FALLING` / `RISING` / `CHANGE` | HIGH→LOW / LOW→HIGH / 둘 다. 풀업 버튼 눌림 = **FALLING** |
| `portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED` | — | 임계 구역용 잠금 |
| `portENTER_CRITICAL(&mux)` / `EXIT` | — | loop 쪽 임계 구역 |
| `portENTER_CRITICAL_ISR(&mux)` / `EXIT_ISR` | — | ISR 쪽 임계 구역 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 6-1 `polling_miss.ino` | `delay(2000)` loop 에서 `digitalRead` 로 세기 → 2초에 5번 눌러도 0~1 만 증가 |
| 6-2 `isr_counter.ino` | FALLING 인터럽트 ISR 이 50ms 디바운스 후 `press_count++`(임계 구역), loop 는 임계 구역 안에서 "읽고 0 으로 비우기"를 한 덩어리로 → `pressed 5 time(s)` |
| 6-3 ⚠ 함정 | ISR 안의 `Serial.println`, `delay`, volatile 없는 변수 |

**핵심 내용**

#### 폴링 vs 인터럽트
> 하드웨어가 알려 주면 loop 가 바빠도 놓치지 않는다
- **폴링** → 2초 사이 5번 눌러도 **대부분 놓침**
- **인터럽트** → loop 가 여전히 `delay(2000)` 이어도 **5번 모두** 센다
- **`attachInterrupt(…, onButton, …)` 의 `onButton`** → **함수 포인터** (C Day3 `qsort` 비교 함수와 같은 모양)

#### ISR 규칙
> 카운트 · 플래그만 바꾸고 출력 · 계산은 loop 에서
- ⚠ **`Serial.println` 금지** → 내부 잠금을 기다리다 멈추거나 크래시
- ⚠ **`delay` 금지** → ISR 안에서 시간이 흐르지 않음
- **50ms 재진입 무시** → 채터링으로 여러 번 오는 인터럽트 거르기

#### 임계 구역
> 읽고 비우기를 한 덩어리로 — 빼면 눌림이 사라진다
- ⚠ 빼면 → loop 가 `n = press_count;` 직후 · `press_count = 0;` 직전에 ISR 이 1 을 더하면 그 1 이 곧바로 지워짐 → **눌림 하나가 사라짐**

**기본 사용형태**
```c
volatile uint32_t press_count = 0;
portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
void IRAM_ATTR onButton() {
    portENTER_CRITICAL_ISR(&mux);
    press_count++;                     // 표시만 하고 나온다
    portEXIT_CRITICAL_ISR(&mux);
}
void setup() {
    pinMode(4, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(4), onButton, FALLING);
}
void loop() {
    portENTER_CRITICAL(&mux);
    uint32_t n = press_count; press_count = 0;   // 읽고 비우기를 한 덩어리로
    portEXIT_CRITICAL(&mux);
    if (n) Serial.printf("pressed %lu\n", n);
}
```

**연관 문제**
<<QA 6>>

### 7교시 · PWM (LEDC) — 밝기 · 색 · 소리

**키워드:** PWM, 듀티비, 주파수, 분해능, LEDC, ledcAttach, ledcWrite, ledcWriteTone, core 3.x 핀 중심 API, ledcSetup 오류, RGB LED 공통 캐소드/애노드, Active/Passive 부저, 멜로디 구조체 배열

**개요**
- 디지털 핀으로 **밝기**를 조절하는 방법 PWM — 아주 빠르게 켰다 껐다 하며 **켜진 시간의 비율(듀티비)** 을 바꾼다. ESP32 의 LEDC 로 LED 페이드, RGB 색 혼합, 부저 멜로디를 만든다. **core 3.x 에서 API 가 바뀐 점**이 함정.

**핵심 용어**
- **PWM(Pulse Width Modulation)**: 빠른 켜기/끄기의 비율로 평균 전력을 조절. 5000Hz 면 눈은 평균 밝기만 본다. 수도꼭지를 빠르게 열고 닫기.
- **듀티비**: 한 주기 중 HIGH 시간 비율 → **밝기를 결정**.
- **주파수**: 1초에 켜고 끄는 횟수. 1Hz 로 낮추면 **눈에 보이게 깜빡인다**.
- **분해능**: 듀티를 몇 단계로 나눌지(비트 수). 8비트 → 0~255, **10비트 → 0~1023**.
- **공통 캐소드 / 애노드 RGB LED**: 공통 다리가 GND / 3V3. **애노드형은 LOW 가 켜짐** → 듀티를 `255 − 값` 으로 뒤집는다.
- **Active 부저**: 내장 발진기 — `HIGH` 만 주면 정해진 음. **Passive 부저**: 주파수를 우리가 만든다(`ledcWriteTone`).

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `ledcAttach` | `ledcAttach(pin, freq, bits)` | 핀에 PWM 설정 (core 3.x) |
| `ledcWrite` | `ledcWrite(pin, duty)` | 듀티 설정 (8비트면 0~255) |
| `ledcWriteTone` | `ledcWriteTone(pin, hz)` | 그 주파수의 소리, **0 = 소리 끄기** |
| (구) `ledcSetup` / `ledcAttachPin` | core 2.x 채널 중심 | 3.x 에서는 **`'ledcSetup' was not declared`** |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 7-1 `fade.ino` | `ledcAttach(25, 5000, 8)` 후 듀티 0→255→0 을 5ms 간격으로 → LED 가 서서히 밝아졌다 어두워짐 |
| 7-2 `rgb.ino` | `color_t {r, g, b, name}` 구조체 배열의 색을 `set_rgb(&COLORS[i])` 로 차례로 표시, `COMMON_ANODE` 면 `255 − 값` |
| 7-3 `buzzers.ino` | Active 부저(19)는 `digitalWrite`, Passive 부저(23)는 `ledcWriteTone(23, 440)` |
| 7-4 `melody.ino` | `note_t {freq, ms}` 배열(도레미파솔라시도)을 루프로 연주, 끝에 `ledcWriteTone(…, 0)` |
| 7-5 ⚠ 함정 | 인터넷의 core 2.x 예제 → 컴파일 오류 |

**핵심 내용**

#### PWM 듀티
> 켜져 있는 비율이 밝기
- **`ledcAttach(25, 5000, 8)` 후 `ledcWrite(25, 128)`** → 듀티 **약 50%** (128/255)
- **core 3.x 변경** → LEDC API 가 **채널 중심 → 핀 중심**
- **2.x 예제** → `ledcAttach(pin, freq, bits)` + `ledcWrite(pin, duty)` 로 바꾼다

#### RGB LED
> 공통 캐소드는 HIGH 가, 공통 애노드는 LOW 가 켜짐
- **공통 애노드에서 빨강** → R, G, B 듀티 **0, 255, 255** (켤 색만 LOW)
- **구별** → 가장 긴 다리를 GND 에 꽂아 빨강이 나오면 캐소드형, 3V3 에 꽂아야 나오면 애노드형
- **배선** → 빈 40~43열에 꽂고 공통 다리는 레일, **GPIO 선 3개만 옮긴다**

#### 멜로디 = 데이터 + 코드
> 악보는 배열, 연주는 루프 — 곡을 바꿔도 코드는 그대로
- **원소 개수** → `sizeof(song) / sizeof(song[0])`
- **쉼표** → `{0, 300}`

**기본 사용형태**
```c
ledcAttach(25, 5000, 8);                  // 5kHz, 8비트
ledcWrite(25, 128);                       // 50 %
typedef struct { uint16_t freq; uint16_t ms; } note_t;
const note_t melody[] = { {262, 300}, {294, 300}, {330, 300}, {0, 300} };
const int N = sizeof(melody) / sizeof(melody[0]);
ledcAttach(23, 1000, 10);
for (int i = 0; i < N; i++) { ledcWriteTone(23, melody[i].freq); delay(melody[i].ms); }
ledcWriteTone(23, 0);                     // 소리 끄기
```

**연관 문제**
<<QA 7>>

### 8교시 · 미니 프로젝트 — 반응속도 게임 · 정리

**키워드:** 반응속도 게임, 상태 머신, 함수 포인터 표, typedef state_t (*state_fn)(void), table[current](), ST_COUNT, IDLE WAIT GO RESULT FOUL, random, millis 비교, UINT32_MAX 최고 기록

**개요**
- Day 1 의 입력(디바운스) · 출력(LED) · PWM(부저)을 **하나의 게임**으로 통합한다. 상태를 `switch` 하나에 넣는 대신 **함수 포인터 표**로 구성해, `loop()` 가 `current = table[current]();` **한 줄**이 된다 (C Day3 함수 포인터·상태 머신).

**핵심 용어**
- **상태 머신**: 지금 상태에 따라 할 일과 다음 상태가 정해지는 구조. 자판기.
- **함수 포인터 표**: 상태 번호(enum) → 그 상태의 함수. `typedef state_t (*state_fn)(void);` = **state_t 를 돌려주는 인자 없는 함수의 포인터 타입**.
- **`ST_COUNT`**: enum 끝에 둬서 **상태 개수를 자동으로 셈** (표 크기).

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `random(a, b)` | a 이상 **b 미만** 난수 → `random(1000, 4001)` = 1000~4000 |
| `UINT32_MAX` | 최고 기록 초기값 — **첫 기록이 무조건 신기록** |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 8-1 함수 포인터 표 | `enum {ST_IDLE, ST_WAIT, ST_GO, ST_RESULT, ST_FOUL, ST_COUNT}` 와 `table[ST_COUNT] = {on_idle, …}`, `loop(){ current = table[current](); }` |
| 반응속도 게임 | IDLE(누르면 1~4초 랜덤 대기 시작) → WAIT(LED 꺼짐, 누르면 FOUL) → GO(LED 켜짐, 누를 때까지 ms 측정, 3초 초과) → RESULT(기록·최고 기록·멜로디) / FOUL(반칙, 낮은 멜로디, LED 3번) |

**핵심 내용**

#### 반응속도 게임 상태 흐름
> IDLE → WAIT → GO → RESULT → IDLE, WAIT 중 누르면 FOUL
- **흐름** → IDLE →(누름) WAIT →(시간 경과) GO →(누름) RESULT → IDLE
- **WAIT 중 누름** → **FOUL** → IDLE
- **GO 에서 3초 초과** → IDLE
- **WAIT 끝 조건** → `millis() - t_mark >= wait_ms`
- **반응 시간** → `reaction_ms = millis() - t_mark;`

#### 함수 포인터 표 상태 머신
> enum 값이 그대로 배열 인덱스 — 상태가 늘어도 loop 는 한 줄
- **`current` 가 `ST_GO(2)`** → `table[current]()` → **`on_go`** 호출
- **장점** → 상태 코드가 **함수로 나뉘어** 추가가 쉽다
- **확장 (2인 대결)** → **`button_t` 배열**로 같은 디바운스 함수 재사용

**기본 사용형태**
```c
typedef enum { ST_IDLE, ST_WAIT, ST_GO, ST_RESULT, ST_FOUL, ST_COUNT } state_t;
typedef state_t (*state_fn)(void);
state_t on_idle(void), on_wait(void), on_go(void), on_result(void), on_foul(void);
const state_fn table[ST_COUNT] = { on_idle, on_wait, on_go, on_result, on_foul };
state_t current = ST_IDLE;
void loop() { current = table[current](); }
state_t on_wait(void) {
    if (button_pressed(&btn)) return ST_FOUL;
    if (millis() - t_mark >= wait_ms) { digitalWrite(LED, HIGH); t_mark = millis(); return ST_GO; }
    return ST_WAIT;
}
```

**연관 문제**
<<QA 8>>
