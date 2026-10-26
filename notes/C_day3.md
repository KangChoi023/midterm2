## Day 3 — 구조체 · 비트 연산 · 매크로 · 함수 포인터

### 1교시 · 구조체 기초

**키워드**: `struct`, `typedef struct`, `Reading`, `순서 초기화`, `지정 초기화 .t =`, `구조체 대입 = 통째 복사`, `구조체 안 배열 복사`, `. vs ->`, `(*p).h`, `a[i].t == (a+i)->t`, `값 전달 vs 포인터 전달`, `const Reading *`, `== 비교 불가`, `필드별 비교`, `memcmp 패딩 함정`, `sizeof(Reading) 16`

**개요**: Day 1·2 에서 "여러 값을 한 묶음으로" 정도로만 쓰던 `Reading` 을 제대로 본다. 구조체는 **서로 다른 타입의 값을 한 덩어리로 묶는 사용자 정의 타입**이다. 1교시는 ① 정의와 두 가지 초기화(순서대로 / 지정 초기화), ② 구조체 대입이 **통째 복사**라는 것(Day 2 정렬의 `Reading t = a[j]` 가 동작한 이유), ③ 변수는 `.`, 포인터는 `->` 로 접근, ④ 함수에 넘길 때는 **`const Reading *`** 가 기본인 이유, ⑤ 구조체는 `==` 로 비교할 수 없다는 것을 다룬다.

**핵심 용어**
- **구조체(struct)**: 여러 타입의 필드(멤버)를 한 묶음으로 만든 타입. `typedef` 로 이름을 붙이면 `struct` 없이 `Reading r;` 로 쓴다.
- **지정 초기화(designated initializer)**: `{.d = 150, .t = 36.9}` 처럼 필드 이름으로 초기화 (C99). 빠진 필드는 0.
- **구조체 대입**: `Reading c = r;` — 모든 필드(안의 배열 포함)를 그대로 복사한 **독립 사본** (memcpy 와 같음).
- **멤버 접근 연산자**: 변수는 `r.t`, 포인터는 `p->t` (= `(*p).t`). `*p.t` 는 `*(p.t)` 로 읽혀 에러.
- **const 포인터 인자**: `const Reading *a` = 복사 비용 없이 넘기되 "읽기만 한다" 는 약속.

**새로 나온 함수·명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `typedef struct` | `typedef struct { double t; int h; int d; } Reading;` | 구조체 타입 정의. |
| `->` | `p->h` | 포인터로 필드 접근 (`(*p).h` 줄임). |
| `memcmp` | `int memcmp(const void *a, const void *b, size_t n);` | n 바이트 비교 (구조체에는 패딩 때문에 믿으면 안 됨). |
| `same` | `int same(const Reading *a, const Reading *b);` | 필드별 비교 함수 (직접 구현). |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 1-1 `1_reading.c` | 순서/지정 초기화(`36.9 0 150`), 복사본 `c.t = 99` 해도 원본 25.3, `p->h` == `(*p).h` |
| 1-2 `2_pass.c` | `bump_by_value` 는 25.0 그대로, `bump_by_ptr` 은 26.0, `avg_t(log, 3)` = 26.0 |
| 1-3 `3_compare.c` | `a == b` 는 컴파일 에러, `same()` 필드 비교 = 1, `memcmp` 는 "보통 1" (패딩 주의) |

**핵심 내용**

#### 구조체 정의와 초기화
> `typedef struct {…} Reading;` — 여러 타입을 한 묶음 타입으로
- **순서대로 초기화** → `{25.3, 60, 120}`
- **지정 초기화** → `{.d = 150, .t = 36.9}` (빠진 필드는 **0**)
- **`sizeof(Reading)`** → **16** (8+4+4, 패딩 없음)

#### 대입과 접근
> 구조체는 `=` 로 통째 복사된다
- **`Reading c = r;`** → 독립 사본
- **구조체 안의 배열** → 함께 복사됨 (배열 단독은 `=` 불가)
- **변수** → `.` / **포인터** → `->` (`p->h` ≡ `(*p).h`)
- ⚠ `*p.h` → `*(p.h)` 로 읽혀 에러
- **배열 요소** → `a[i].t` ≡ `(a + i)->t`

#### 함수 인자로 넘기기
> 기본은 `const Reading *` — 복사 비용을 줄이고 읽기 전용을 약속
- **값 전달** → 복사본 (원본 불변, 16바이트 복사)
- **`const Reading *`** → 복사 비용 ↓ + 읽기 전용
- **수정해야 할 때만** → `const` 제거 (Day 2 "원본을 바꾸려면 주소" 그대로)

#### 구조체 비교
> 구조체는 `==` 로 비교할 수 없다
- ⚠ `==` → 컴파일 에러 → **필드별 비교**
- ⚠ `memcmp` → **패딩** 때문에 믿으면 안 됨

**기본 사용형태**
```c
typedef struct { double t; int h; int d; } Reading;
Reading r = {25.3, 60, 120};  Reading c = r;           // 통째 복사
void bump(Reading *r) { r->t += 1.0; }                  // 원본 수정
double avg_t(const Reading *a, int n);                  // 읽기 전용 + 길이
const Node *hottest(const Node *a, int n);              // 포인터 반환
```

**연결**
- [C D2·8교시 센서 로그 분석기](#/c/2/8) — 구조체 swap 이 동작한 이유
- [C D4·3교시 바이너리 I/O](#/c/4/3) — 구조체 배열을 `fwrite` 로 통째 저장
- [LINUX D5·1교시 쓰레드 기초](#/linux/5/1) — 쓰레드 함수에 인자 묶음을 구조체 포인터로 전달

**연관 문제**
<<QA 1>>

### 2교시 · 패딩 · 정렬 · 공용체 · 열거형

**키워드**: `정렬(alignment)`, `패딩(padding)`, `자기 크기의 배수 주소`, `구조체 크기 = 가장 큰 필드의 배수`, `offsetof`, `큰 것부터 배치`, `#pragma pack(push, 1)`, `정렬 안 된 접근`, `union`, `같은 메모리 공유`, `IEEE 754 0x3F800000`, `enum`, `ST_COUNT`, `enum 은 int`, `default 방어`

**개요**: `Reading` 은 8+4+4 = 16 으로 딱 맞았지만, `{char; int; char;}` 는 6 이 아니라 **12** 다. 2교시는 컴파일러가 필드 사이에 빈칸(**패딩**)을 넣는 **정렬 규칙 두 가지**를 그림으로 이해하고, 필드 순서를 바꿔 크기를 줄이는 방법과 `#pragma pack` 으로 패딩을 없애는 대가를 본다. 이어서 같은 메모리를 여러 타입으로 읽는 **`union`**, 상태에 이름과 개수를 붙이는 **`enum`** 을 다룬다(7교시 상태 머신의 준비).

**핵심 용어**
- **정렬(alignment)**: 각 필드를 자기 크기의 배수 주소에 놓는 규칙 (int 는 4의 배수, double 은 8의 배수). CPU 가 한 번에 읽기 위해서.
- **패딩(padding)**: 정렬을 맞추려고 컴파일러가 필드 사이·끝에 넣는 빈 바이트. 값은 쓰레기일 수 있다.
- **`offsetof(타입, 필드)`**: 구조체 시작에서 필드까지의 바이트 거리 (`<stddef.h>`).
- **`#pragma pack(1)`**: 패딩 금지 지시. 통신 프레임과 1:1 이지만 정렬 안 된 접근이 생긴다.
- **공용체(union)**: 모든 멤버가 **같은 주소에서 시작**해 메모리를 공유. 크기 = 가장 큰 멤버. 마지막에 쓴 멤버만 의미.
- **열거형(enum)**: 이름 붙은 정수 상수 묶음. 0 부터 1씩 증가. 실제 타입은 `int`.

**새로 나온 함수·명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `offsetof` | `size_t offsetof(type, member);` | 필드 오프셋 확인 (`<stddef.h>`). |
| `#pragma pack` | `#pragma pack(push, 1)` … `#pragma pack(pop)` | 이 구간 구조체의 패딩 제거. |
| `union` | `typedef union { uint32_t w; uint8_t b[4]; float f; } Word;` | 같은 4바이트를 세 가지로 읽기. |
| `enum` | `typedef enum { ST_IDLE, …, ST_COUNT } State;` | 상태 이름 + 개수. |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 2-1 `1_padding.c` | `Bad`=12 (id@0 v@4 f@8), `Good`=8, `Reading`=16 을 `offsetof` 와 함께 출력 |
| 2-2 `2_pack.c` | `#pragma pack(1)` 의 `Packed`=6, `Frame{uint16_t; int8_t; uint8_t}`=4 |
| 2-3 `3_union.c` | `x.w=0x11223344` → `b[]` 는 `44 33 22 11`, `x.f=1.0f` → `0x3F800000`, `sizeof`=4 |
| 2-4 `4_enum.c` | `ST_COOL` = 2, `NAME[st]` = `COOL`, `ST_COUNT` 로 전체 상태 이름 순회 |

**핵심 내용**

#### 정렬과 패딩
> 필드는 자기 크기의 배수 주소에서 시작하고, 구조체 크기는 가장 큰 필드의 배수
- **규칙 ①** → 각 필드는 **자기 크기의 배수 주소**에서 시작
- **규칙 ②** → 전체 크기는 **가장 큰 필드 크기의 배수** (배열로 늘어놓아도 ①이 유지되도록)
- **`{char id; int v; char f;}`** → **12** (id@0, v@4, f@8, 패딩 3+3)
- **큰 것부터 `{int v; char id; char f;}`** → **8**
- **확인** → `offsetof(타입, 필드)` (`<stddef.h>`)

#### 패딩 없애기
> 큰 필드부터 놓거나 pack — 통신 프레임은 바이트 조립이 안전
- **필드를 큰 것부터** → 이식성 100%, 손실 없음
- **`#pragma pack(push, 1)`** → 통신 프레임과 1:1
- ⚠ pack → **정렬 안 된 접근** → x86 은 느려지고 일부 MCU 는 죽음
- **`(Frame *)buf` 캐스팅 대신** → **바이트 조립**

#### union
> 멤버가 같은 메모리를 공유한다 — 마지막에 쓴 멤버만 의미
- **크기** → 가장 큰 멤버
- **`x.w = 0x11223344`** → `x.b[0]` = **0x44** (리틀 엔디안)
- **`x.f = 1.0f`** → `x.w` = **0x3F800000** (IEEE 754)
- **전형적 용도** → 센서가 float 을 4바이트로 보내면 `b[]` 에 받아 `f` 로 읽기

#### enum
> 0 부터 증가하는 이름 붙은 정수
- **마지막에 `ST_COUNT`** → 배열 크기 · 루프 상한 자동
- ⚠ enum 은 그냥 `int` → `State st = 99;` 도 컴파일됨 → `default` 로 방어

**기본 사용형태**
```c
typedef struct { int v; char id; char f; } Good;               // sizeof 8
#pragma pack(push, 1)
typedef struct { char id; int v; char f; } Packed;             // sizeof 6
#pragma pack(pop)
typedef union { uint32_t w; uint8_t b[4]; float f; } Word;
typedef enum { ST_IDLE, ST_HEAT, ST_COOL, ST_ALERT, ST_COUNT } State;
static const char *NAME[ST_COUNT] = {"IDLE", "HEAT", "COOL", "ALERT"};
```

**연결**
- [C D2·2교시 포인터 산술과 const](#/c/2/2) — 정렬이 안 맞는 캐스팅이 위험한 이유
- [C D4·3교시 바이너리 I/O](#/c/4/3) — 패딩·엔디안 때문에 다른 기계에서 못 읽는 파일
- [MCU D1·3교시 Digital Output — 신호등](#/mcu/1/3) — `enum` 상태 신호등

**연관 문제**
<<QA 2>>

### 3교시 · 비트 연산 기초

**키워드**: `& | ^ ~ << >>`, `마스크`, `0b 접두`, `켜기 |= (1u << n)`, `끄기 &= ~(1u << n)`, `토글 ^=`, `검사 (reg >> n) & 1`, `1u`, `필드 쓰기 = 지우고 넣기`, `정수 승격 a << 1 = 356`, `(uint8_t) 캐스팅`, `산술 시프트 vs 논리 시프트`, `-3 >> 1 = -2`, `unsigned 로만`, `니블 교환`

**개요**: 임베디드에서 하드웨어를 다루는 것은 결국 **레지스터의 특정 비트를 켜고 끄는 일**이다. 3교시는 여섯 비트 연산자의 결과를 2진수로 예측해 보고, 어떤 상황에서도 쓰는 **네 관용구**(켜기·끄기·토글·검사)를 손에 익힌다. 이어서 여러 비트짜리 필드를 쓸 때는 **먼저 지우고 넣어야** 한다는 것, 그리고 `uint8_t` 정수 승격과 signed 시프트가 만드는 함정을 본다. 결론은 **비트 연산은 unsigned 로만**.

**핵심 용어**
- **비트 마스크(bit mask)**: 원하는 비트만 1(또는 0)인 값. `&` 로 남기고, `|` 로 켜고, `^` 로 뒤집는 데 쓴다.
- **시프트(shift)**: `<<` 왼쪽(×2ⁿ), `>>` 오른쪽(÷2ⁿ). `1u << n` = n번 비트만 1인 마스크.
- **산술 시프트 vs 논리 시프트**: 음수를 `>>` 하면 부호 비트를 채우는지(산술) 0 을 채우는지(논리)가 **구현 정의** → unsigned 만 쓴다.
- **필드(field)**: 레지스터 안에서 여러 비트가 모여 하나의 값을 이루는 구간 (예: 비트 8~15).
- **니블(nibble)**: 4비트 = 16진수 한 자리.

**새로 나온 함수·명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `print_bin8` / `print_bin32` | `void print_bin8(const char *label, uint8_t v);` | 값을 2진수로 출력 (예제용 도우미, `bits.h`). |
| `0b` 리터럴 | `uint8_t a = 0b10110010;` | 2진수 상수 (gcc 확장, C23 표준). |
| `1u << n` | `reg \|= (1u << 3);` | unsigned 1 을 n칸 → n번 비트 마스크. |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 3-1 `1_operators.c` | `a=0b10110010` 에 `& 0x0F`, `\| 0x01`, `^ 0xFF`, `~`, `>> 4` 를 2진수로, `a << 1` = 356 |
| 3-2 `2_idioms.c` | 32비트 `reg` 에 비트 3·5 켜기 → 3 끄기 → 5 토글 → 검사 (`bit5 = 1, bit3 = 0`) |
| 3-3 `3_field.c` | `0x12345678` 의 비트 8~15 를 0xAB 로 → `0x1234AB78`, 다시 꺼내면 `AB` |
| 3-4 `4_neg_shift.c` | `-8 >> 1` = −4(산술), `(uint8_t)-8 >> 1` = 124(논리), `-3 >> 1` = −2 |

**핵심 내용**

#### 비트 연산자
> AND 는 남기기, OR 는 켜기, XOR 는 뒤집기
- **`&`** → AND (남기기) / **`|`** → OR (켜기) / **`^`** → XOR (반전, 두 번 하면 원상) / **`~`** → NOT / **`<<` `>>`** → 시프트
- **`a & 0x0F`** → 하위 4비트만
- **`a | 0x01`** → 최하위 비트 켬
- **`a ^ 0xFF`** → 전체 반전
- **니블 교환** → `(uint8_t)((v << 4) | (v >> 4))`

#### 비트 하나 다루기 — 네 관용구
> 다른 비트는 그대로 두고 n 번 비트만
- **켜기** → `reg |= (1u << n)` (OR 은 1 을 이기지 못함)
- **끄기** → `reg &= ~(1u << n)` (그 자리만 0 인 마스크와 AND)
- **토글** → `reg ^= (1u << n)`
- **검사** → `(reg >> n) & 1` (그 비트를 0번으로 내려서 본다)

#### 여러 비트 필드 쓰기
> 먼저 지우고(AND NOT) 그다음 넣는다(OR)
- **쓰기** → `reg = (reg & ~(0xFFu << 8)) | (0xABu << 8);`
- ⚠ OR 만 하면 → 기존 값과 섞임
- **읽기** → `(reg >> 8) & 0xFF`
- **실제 예** → ESP32 `GPIO_MODE` 레지스터 설정이 정확히 이 모양

#### 비트 연산 함정
> 비트 연산은 unsigned 로만
- ⚠ `uint8_t` 는 연산 시 **int 로 승격** → `a << 1` = **356** (8비트로 자르려면 `(uint8_t)(a << 1)`)
- ⚠ `~a` 도 캐스팅 필요 (`0xFFFFFF4D`)
- ⚠ `1 << 31` → signed 오버플로 → **`1u`**
- ⚠ 음수 `>>` → 구현 정의 (`-3 >> 1` = −2, 내림) / signed `<<` → UB 가능
- ⚠ 한 식에서 같은 변수를 두 번 바꾸기 (`reg = (reg &= …) | …`) → 정의되지 않은 동작

**기본 사용형태**
```c
reg |= (1u << 3);   reg &= ~(1u << 3);   reg ^= (1u << 5);   int b = (reg >> 5) & 1;
reg = (reg & ~(0xFFu << 8)) | (0xABu << 8);      // 비트 8~15 에 0xAB
```

**연결**
- [C D1·5교시 형변환](#/c/1/5) — 정수 승격이 `a << 1` = 356 을 만든 원인
- [MCU D1·4교시 레지스터 직접 제어](#/mcu/1/4) — 같은 관용구로 실제 ESP32 레지스터 조작
- [EIO D2·3교시 레지스터 직접 접근 비교](#/eio/2/3) — 라즈베리파이 GPIO 레지스터

**연관 문제**
<<QA 3>>

### 4교시 · 비트 연산 실전

**키워드**: `가상 GPIO 레지스터`, `GPIO_OUT`, `extern`, `GPIO_SET CLR TOGGLE READ`, `매크로 인자 (pin)`, `실제 주소 0x3FF44004`, `비트 플래그 FLAG_ALARM (1u << 5)`, `LEVEL_MASK 0x0F`, `!!`, `바이트 조립/분해`, `XOR 체크섬`, `$본문*HH`, `NMEA`, `strtol(…, 16)`, `a ^ b ^ b == a`

**개요**: 3교시의 관용구를 실제 임베디드 모양으로 쓴다. ① 32비트 변수 하나를 **가상 GPIO 레지스터**로 보고 비트 하나 = 핀 하나로 LED·버튼을 매크로로 제어한다(다음 주 MCU 에서는 이 변수만 실제 주소로 바뀐다). ② 센서 프레임 `D=120` 을 8비트 **상태 플래그**로 해석한다. ③ 프레임이 깨졌는지 확인하는 **XOR 체크섬**(GPS NMEA 방식)을 계산·검증한다.

**핵심 용어**
- **`extern` 선언**: "이 변수는 다른 `.c` 에 정의돼 있다" 는 선언. 헤더에 `extern`, 정의는 `.c` 한 곳.
- **메모리 맵 레지스터**: 하드웨어 레지스터가 특정 메모리 주소에 붙어 있어, 그 주소를 읽고 쓰면 핀이 바뀐다.
- **비트 플래그(bit flag)**: 한 바이트의 각 비트에 on/off 의미를 붙인 것 (전원·온라인·알람…).
- **`!!x`**: 0 이 아니면 1, 0 이면 0 으로 만드는 관용구. `d & FLAG_ALARM` 은 1 이 아니라 32.
- **체크섬(checksum)**: 데이터로 계산한 작은 값을 함께 보내 수신 측에서 다시 계산·비교해 손상을 감지.

**새로 나온 함수·명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `GPIO_SET/CLR/TOGGLE` | `#define GPIO_SET(pin) (GPIO_OUT \|= (1u << (pin)))` | 핀 켜기/끄기/토글 매크로. |
| `GPIO_READ` | `#define GPIO_READ(pin) ((GPIO_IN >> (pin)) & 1u)` | 입력 핀 읽기 (0/1). |
| `checksum` | `uint8_t checksum(const char *s);` | `'*'` 전까지 모든 글자를 XOR (직접 구현). |
| `strtol(…, 16)` | `strtol(star + 1, NULL, 16)` | `"2C"` 같은 16진수 문자열 → 숫자. |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 4-1 `04_register/gpio.h`, `1_gpio_demo.c` | LED_R(2)·LED_G(4) 켜고 끄기 → `0x04 → 0x14 → 0x10`, 버튼 입력이 있으면 LED_G 토글 → `0x00` |
| 4-2 `2_flags.c` | `D=120 (0b0111_1000)` → `power=0 online=1 alarm=1 lowbat=1 level=8` |
| 4-3 `3_checksum.c` | 프레임 본문의 XOR 체크섬 계산, 한 글자 바뀐 프레임은 다른 값 → 폐기 |

**핵심 내용**

#### 가상 GPIO 레지스터
> 32비트 변수의 비트 하나 = 핀 하나
- **선언** → `gpio.h` 에 `extern uint32_t GPIO_OUT;` / **정의** → `.c` 한 곳
- **`GPIO_SET(pin)`** → `(GPIO_OUT |= (1u << (pin)))`
- **`GPIO_CLR`** → `&= ~` / **`GPIO_TOGGLE`** → `^=` / **`GPIO_READ`** → `(GPIO_IN >> pin) & 1u`
- **매크로 인자** → `(pin)` 처럼 괄호로 (5교시)
- **실제 MCU** → `#define GPIO_OUT (*(volatile uint32_t *)0x3FF44004)` — **이 한 줄만 바꾸면** 나머지 코드는 그대로

#### 비트 플래그
> 비트마다 의미를 붙여 한 바이트에 여러 상태
- **정의** → `#define FLAG_ALARM (1u << 5)`
- **`d & FLAG_ALARM`** → **0 또는 32** (마스크 값)
- **0/1 이 필요하면** → `!!` (`if` 조건에는 그대로 써도 됨)
- **하위 필드** → `d & 0x0Fu`

#### 바이트 조립 · 분해
> 시프트와 OR 로 합치고, 시프트와 AND 로 꺼낸다
- **조립** → `buf[0] | (buf[1] << 8)`
- **분해** → `v & 0xFF`, `(v >> 8) & 0xFF`
- **합치기** → `((uint32_t)dist << 16) | hum`

#### XOR 체크섬
> `$본문*HH` (GPS NMEA 형식) — 본문 글자를 전부 XOR
- **한 글자 변화** → 감지
- **`a ^ b ^ b == a`** → 같은 값 2번 XOR = 원상 (토글 · 간단한 암호화에도 쓰임)
- ⚠ 순서 바뀜은 못 잡음 (교환법칙)
- **실제 값** → `T=25.3,H=60,D=120` → **`4A`** (강의 자료 예시 `2C` 는 오기)

**기본 사용형태**
```c
GPIO_SET(PIN_LED_R);  GPIO_CLR(PIN_LED_R);  if (GPIO_READ(PIN_BUTTON)) GPIO_TOGGLE(PIN_LED_G);
uint8_t cs = 0; for (; *s && *s != '*'; s++) cs ^= (uint8_t)*s;
long rx = strtol(star + 1, NULL, 16);        // "2C" → 16진수 파싱
int alarm = !!(d & FLAG_ALARM);
```

**연결**
- [MCU D1·4교시 레지스터 직접 제어](#/mcu/1/4) — `GPIO_OUT` 이 실제 주소가 되는 곳
- [EIO D4·7교시 UART 와 termios](#/eio/4/7) — 시리얼 프레임 무결성 검사
- [C D4·4교시 링 버퍼](#/c/4/4) — 같은 `gpio.h` 를 종합 프로젝트에서 재사용

**연관 문제**
<<QA 4>>

### 5교시 · 매크로 · volatile · 메모리 맵

**키워드**: `#define`, `텍스트 치환`, `괄호 셋 규칙`, `SQ(1+2) = 5`, `부수효과 인자 MAX(i++)`, `gcc -E`, `static inline`, `#ifdef DEBUG`, `-DDEBUG`, `LOG 매크로`, `((void)0)`, `__FILE__ __LINE__ __VA_ARGS__`, `#if defined(ESP32)`, `volatile`, `-O2 무한 루프`, `메모리 맵 I/O`, `(*(volatile uint32_t *)ADDR)`, `ARRAY_LEN 감쇠 함정`

**개요**: 4교시 `GPIO_SET(pin)` 같은 **매크로**는 함수처럼 보이지만 컴파일 전에 **글자를 그대로 바꿔 끼우는 것**이라 괄호를 빠뜨리면 계산이 틀어진다. 5교시는 ① 매크로의 세 가지 규칙(인자 괄호·전체 괄호·부수효과 금지)을 함정으로 확인하고, ② **조건부 컴파일**로 디버그 출력을 빌드 옵션 하나로 켜고 끄며, ③ "밖에서 바뀔 수 있는" 변수를 컴파일러가 최적화로 없애지 못하게 하는 **`volatile`**, ④ 주소를 레지스터처럼 다루는 **메모리 맵 I/O** 의 형태를 본다.

**핵심 용어**
- **매크로(macro)**: `#define` 으로 정의한 텍스트 치환 규칙. 전처리 단계(Day 1 2교시)에서 펼쳐지며 타입 검사가 없다.
- **함수형 매크로**: 인자를 받는 매크로 `#define SQ(x) ((x) * (x))`. 인자가 **그대로 복사**되므로 두 번 쓰이면 두 번 평가된다.
- **`static inline` 함수**: 매크로 대신 쓰는 작은 함수. 타입 검사가 되고 인자는 한 번만 평가.
- **조건부 컴파일**: `#ifdef / #if / #else / #endif` 로 특정 조건일 때만 코드를 포함. 빠진 코드는 실행 파일에 아예 없다.
- **`volatile`**: "이 변수는 프로그램 밖(하드웨어·인터럽트·다른 스레드)에서 바뀔 수 있으니 **매번 진짜 메모리에서 읽고 접근을 생략하지 마라**" 는 지시.
- **메모리 맵 I/O(memory-mapped I/O)**: 하드웨어 레지스터를 특정 주소의 메모리처럼 읽고 쓰는 방식.

**새로 나온 함수·명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `gcc -E` | `gcc -E file.c` | 매크로가 어떻게 펼쳐졌는지 확인. |
| `-D이름` | `gcc -DDEBUG …` | 소스 맨 위에 `#define 이름` 을 쓴 것과 같다. |
| `LOG` | `#define LOG(fmt, ...) fprintf(stderr, "[%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)` | 파일·줄 번호가 붙는 디버그 출력 (릴리스에선 `((void)0)`). |
| `volatile` | `volatile uint32_t hw_flag;` | 최적화로 읽기·쓰기를 없애지 못하게. |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 5-1 `1_macro.c` | 괄호 없는 `SQ_BAD(1+2)` = 5 vs `SQ(1+2)` = 9, `100 / SQ(5)` = 4 (전체 괄호 덕분), `MAX(i++, 2)` → `4 5` (i++ 두 번 평가) |
| 5-2 `2_dbg.c` | `-DDEBUG` 유무로 `LOG` 출력이 나오거나 코드째 사라짐 |
| 5-3 `3_volatile.c` | 일반 플래그와 `volatile` 플래그 비교 — 대기 루프는 주석(일반 쪽은 `-O2` 에서 무한 루프로 최적화될 수 있음), 크기는 둘 다 4 (**차이는 의미뿐**) |
| 5-4 `4_mmio.c` | PC 에선 진짜 주소에 쓰면 세그폴트라 `fake_mem[1]` 의 주소를 `(*(volatile uint32_t *)…)` 로 감싸 레지스터처럼 `\|= (1u << 2)` → `0x00000004` |

**핵심 내용**

#### 매크로 = 텍스트 치환
> 인자마다 괄호, 전체에 괄호, 부수효과 있는 인자 금지
- ⚠ `SQ(x) x*x` 에 `SQ(1+2)` → **5** (`1 + 2*1 + 2`)
- ⚠ 전체 괄호 없음 → `100 / SQ(5)` 가 **100**
- ⚠ `MAX(i++, 2)` → `i++` 가 **두 번 평가** (i=3 → 5)
- **전개 결과 확인** → `gcc -E`
- **타입 검사가 필요하면** → `static inline` 함수 (`bits.h`)
- ⚠ `ARRAY_LEN(a)` 를 포인터에 쓰면 → `8/4 = 2` (감쇠 함정)

#### 조건부 컴파일
> 빌드 옵션으로 코드 자체를 넣고 뺀다
- **`#ifdef DEBUG … #else #define LOG(...) ((void)0) #endif`**
- **`-DDEBUG`** = 소스 첫 줄 `#define DEBUG`
- **릴리스** → 코드 자체가 사라져 플래시 · 속도 비용 **0**
- **플랫폼 분기** → `#if defined(ESP32)`

#### volatile
> "밖에서 바뀔 수 있다" — 매번 메모리에서 읽고 접근을 생략하지 마라
- ⚠ 없으면 → `-O2` 가 `while (flag == 0) {}` 를 무한 루프로 만듦
- **필요한 곳 세 가지** → 하드웨어 레지스터 / 인터럽트 핸들러가 바꾸는 변수 / 다른 스레드가 바꾸는 플래그
- **일반 카운터에 붙이면** → 최적화만 막힘
- ⚠ 스레드 동기화는 volatile 로 부족

#### 메모리 맵 I/O
> 주소를 `volatile` 포인터로 캐스팅해 역참조하면 레지스터가 변수처럼 된다
- **`(*(volatile uint32_t *)0x3FF44004)`** → 캐스팅 → 역참조 → `|=` 사용
- **4교시 연결** → `extern uint32_t GPIO_OUT` 한 줄을 이걸로 바꾸면 실제 하드웨어 코드

**기본 사용형태**
```c
#define SQ(x) ((x) * (x))
#define BIT(n) (1u << (n))
#ifdef DEBUG
#  define LOG(fmt, ...) fprintf(stderr, "[%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#  define LOG(fmt, ...) ((void)0)
#endif
volatile uint32_t hw_flag;
#define GPIO_OUT_REG (*(volatile uint32_t *)0x3FF44004)
```
```bash
gcc -Wall -Wextra -DDEBUG dbg.c -o dbg
```

**연결**
- [C D1·2교시 빌드 과정 해부](#/c/1/2) — 매크로가 펼쳐지는 전처리 단계
- [MCU D1·6교시 Interrupt — ISR 과 공유 변수](#/mcu/1/6) — ISR 이 바꾸는 변수에 `volatile`
- [LINUX D1·7교시 시그널](#/linux/1/7) — 시그널 핸들러 플래그 `volatile sig_atomic_t`
- [LINUX D5·2교시 race condition](#/linux/5/2) — volatile 로는 부족한 스레드 동기화

**연관 문제**
<<QA 5>>

### 6교시 · 함수 포인터

**키워드**: `함수 이름 = 주소`, `int (*op)(int, int)`, `괄호가 *op 를 묶는다`, `int *op(int,int) 와 구분`, `typedef 함수 포인터`, `op(2,3) == (*op)(2,3)`, `NULL 검사`, `함수 포인터 배열`, `OPS[i](7,3)`, `함수 포인터 매개변수`, `apply`, `qsort`, `비교 함수 const void *`, `음수/0/양수`, `(x > y) - (x < y)`, `뺄셈 비교 함정`, `bsearch`

**개요**: 배열 이름이 첫 요소의 주소이듯, **함수 이름은 함수 코드의 주소**다. 그 주소를 변수에 담으면 "어떤 함수를 부를지" 를 실행 중에 고를 수 있다. 6교시는 ① 함수 포인터 선언을 읽고 쓰는 법(괄호의 의미, `typedef`), ② 함수 포인터 **배열**로 `if-else`/`switch` 사다리를 **표 한 줄**로 바꾸기, ③ 함수를 인자로 넘기는 패턴, ④ Day 2 의 손수 짠 버블 정렬을 라이브러리 **`qsort` + 비교 함수**로 바꾸는 것, ⑤ 비교 함수를 뺄셈으로 짜면 생기는 함정을 다룬다.

**핵심 용어**
- **함수 포인터(function pointer)**: 함수의 주소를 담는 변수. 타입에 인자·반환 타입이 들어간다.
- **선언 읽는 법**: `int (*op)(int, int)` — 괄호 안 `*op` 먼저: "op 는 포인터" → 바깥: "(int,int)를 받아 int 를 돌려주는 함수를". 괄호가 없으면 `int *op(int,int)` = "int 포인터를 돌려주는 **함수 선언**".
- **디스패치 테이블(dispatch table)**: 함수 포인터 배열. 인덱스로 함수를 골라 호출 (`OPS[i](a, b)`).
- **비교 함수(comparator)**: 두 요소를 받아 순서를 음수/0/양수로 알려 주는 함수. `qsort` 는 요소 타입을 모르므로 `const void *` 로 받고, 비교 함수가 원래 타입으로 되돌린다.
- **`void *`**: 아무 타입이나 가리킬 수 있는 범용 포인터. 역참조하려면 원래 타입으로 바꿔야 한다.

**새로 나온 함수·명령**

| 함수 | 기본형 | 용도 |
|---|---|---|
| 함수 포인터 typedef | `typedef int (*BinOp)(int, int);` | 함수 포인터 타입에 이름 붙이기. |
| `apply` | `void apply(double *a, int n, double (*f)(double));` | 배열 각 요소에 함수 f 적용 (직접 구현). |
| `qsort` | `void qsort(void *base, size_t n, size_t size, int (*cmp)(const void *, const void *));` | 범용 정렬. `<stdlib.h>` |
| `bsearch` | `void *bsearch(const void *key, const void *base, size_t n, size_t size, int (*cmp)(const void *, const void *));` | 범용 이진 검색 (정렬 전제), 없으면 `NULL`. |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 6-1 `1_funcptr.c` | 함수 포인터 `op` 에 `add` 를 담아 `op(2,3)` = 5, 다른 함수로 바꿔 6, `add` 와 `op` 의 주소가 같음 |
| 6-2 `2_table.c` | `OPS[] = {add, sub, mul}` 로 계산기 메뉴를 표 한 줄로 (`OPS[i](7,3)`) |
| 6-3 `3_qsort.c` | `Reading` 배열을 `qsort` 로 온도 오름차순(`24.8 26.0 37.2`), 비교 함수만 바꿔 습도 내림차순(`70 61 59`) |
| 6-4 `4_cmp_trap.c` | `return *(int*)a - *(int*)b` 비교 함수는 `INT_MAX - (-1)` 에서 오버플로 → 잘못된 순서. 안전형 `(x > y) - (x < y)` |

**핵심 내용**

#### 함수 포인터 선언
> 함수 이름 = 함수의 주소 — 괄호가 `*op` 를 묶는다
- **`int (*op)(int, int);`** → "(int,int) 를 받아 int 를 돌려주는 함수의 포인터 `op`"
- ⚠ 괄호 없으면 `int *op(int,int)` → int 포인터를 돌려주는 **함수 선언**
- **이름 붙이기** → `typedef int (*BinOp)(int, int);`
- **호출** → `op(2,3)` ≡ `(*op)(2,3)`
- **크기** → 8
- ⚠ `NULL` 이면 → 호출 전 `if (op)` 검사 (안 하면 세그폴트)

#### 함수 포인터 배열과 매개변수
> `switch` 를 표로 — 연산 추가 = 배열에 한 항목
- **배열** → `static const BinOp OPS[] = {add, sub, mul};` → `OPS[i](7,3)`
- **7교시 상태 머신의 원리**
- ⚠ 인덱스 범위 검사 필수
- **매개변수** → `void apply(double *a, int n, double (*f)(double))`
- **넘길 때** → `apply(temps, 3, c_to_f)` (괄호 없이 이름만)

#### qsort 와 비교 함수
> 무엇을 비교할지를 함수로 넘기는 범용 정렬
- **`qsort(배열, 개수, 요소크기, 비교함수)`**
- **비교 함수** → `int cmp(const void *a, const void *b)`, 안에서 원래 타입으로 되돌린다
- **반환** → 음수 = a 가 앞 / 0 / 양수 = a 가 뒤
- **오름차순** → `(x > y) - (x < y)`
- **내림차순** → `a, b` 를 바꾸거나 `(x < y) - (x > y)`
- ⚠ 뺄셈 비교 → int 는 `INT_MAX - (-1)` 오버플로, double 은 소수부가 잘려 0 → 정렬 안 됨
- **`bsearch`** → 라이브러리 이진 검색 (정렬 전제)

**기본 사용형태**
```c
typedef int (*BinOp)(int, int);
static const BinOp OPS[] = {add, sub, mul};
int cmp_temp(const void *a, const void *b) {
    const Reading *x = a, *y = b;
    return (x->t > y->t) - (x->t < y->t);
}
qsort(log, n, sizeof log[0], cmp_temp);
```

**연결**
- [C D2·4교시 정렬과 검색](#/c/2/4) — 직접 짠 버블 정렬·이진 검색
- [LINUX D1·7교시 시그널](#/linux/1/7) — `signal`/`sigaction` 에 핸들러 함수 포인터 등록
- [LINUX D5·1교시 쓰레드 기초](#/linux/5/1) — `pthread_create` 에 쓰레드 함수 포인터 전달

**연관 문제**
<<QA 6>>

### 7교시 · 콜백과 상태 머신

**키워드**: `콜백(callback)`, `register_alert`, `on_alert`, `NULL 검사 후 호출`, `attachInterrupt`, `콜백 배열 + 개수`, `상태 머신(FSM)`, `상태 전이표`, `enum + 함수 포인터 테이블`, `TABLE[st](t)`, `핸들러 반환값 = 다음 상태`, `IDLE HEAT COOL ALERT`, `지정 초기화 [ST_HEAT] = on_heat`, `테이블 순서 어긋남`, `visits[ST_COUNT]`

**개요**: 함수 포인터의 두 가지 대표 쓰임새를 만든다. **콜백**은 "이 일이 생기면 이 함수를 불러라" 를 **미리 등록**해 두는 패턴으로, 라이브러리 코드가 내 함수를 모르는 채로 부를 수 있게 한다(다음 주 `attachInterrupt`, 타이머, 버튼 이벤트가 전부 이 모양). **상태 머신**은 `enum` 상태를 인덱스로 하는 **함수 포인터 테이블**로, `switch` 없이 `TABLE[st](t)` 한 줄로 "현재 상태의 일을 하고 다음 상태를 정한다". 테이블과 enum 의 순서가 어긋나는 함정과 **지정 초기화**로 막는 법까지 본다.

**핵심 용어**
- **콜백(callback)**: 다른 코드에 넘겨 두었다가 그 코드가 적절한 때 불러 주는 함수.
- **이벤트(event)**: 콜백을 부를 계기 (온도 초과, 버튼 눌림, 타이머 만료, 인터럽트).
- **유한 상태 머신(FSM)**: 시스템을 몇 개의 **상태**와 조건에 따른 **전이**로 표현한 모델. 지금 상태 + 입력 → 출력 + 다음 상태.
- **상태 전이표**: 현재 상태 / 조건 / 다음 상태 / 출력을 표로 정리한 설계도. 코드로는 핸들러 테이블이 된다.
- **핸들러(handler)**: 한 상태에서 할 일을 맡은 함수. 반환값이 다음 상태.

**새로 나온 함수·명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `register_alert` | `void register_alert(Callback cb);` | 콜백 등록 (직접 구현). |
| `feed` | `void feed(double t);` | 값을 넣으면 조건에 맞을 때 등록된 콜백 호출. |
| 핸들러 타입 | `typedef State (*Handler)(double);` | 온도를 받아 다음 상태를 돌려주는 함수 포인터. |
| 지정 초기화 배열 | `[ST_HEAT] = on_heat` | enum 값으로 배열 칸을 지정 → 순서 실수 방지. |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 7-1 `1_callback.c` | 등록 전 `feed(36.0)` 은 무반응 → `register_alert(beep)` 후 `BEEP! 36.0` → `log_it` 으로 다시 등록하면 교체되어 `log: alert at 37.5`, 20.0 은 무반응 |
| 7-2 `2_fsm.c` | 온도열을 넣어 IDLE/HEAT/COOL/ALERT 를 테이블로 전이시키며 상태·핀 출력 |
| 7-3 `3_table_trap.c` | 테이블의 HEAT/COOL 순서를 바꿔 히터 자리에서 팬이 도는 버그 재현 → 지정 초기화로 수정 |

**핵심 내용**

#### 콜백
> "이 이벤트가 생기면 이 함수를 불러라" 를 미리 등록
- **등록** → `register_alert(beep)`
- **라이브러리 쪽 (`feed`)** → `if (t > 35 && on_alert) on_alert(t);`
- ⚠ 등록 전 → `NULL` 검사 필수
- **`feed` 를 짠 사람은** → `beep` 이 존재하는지 모른다
- **다음 주** → `attachInterrupt(pin, handler, mode)` 가 이 패턴 (ESP32 코드의 절반이 콜백 등록)
- **여러 개 등록** → 함수 포인터 배열 + 개수 (가득 차면 0 반환)

#### 상태 전이표 (온도 조절기)
> 현재 상태 + 조건 → 다음 상태 + 출력

| 현재 | 조건 | 다음 | 출력 |
|---|---|---|---|
| IDLE | t < 18 / t > 28 | HEAT / COOL | — |
| HEAT | t ≥ 20 | IDLE | HEATER |
| COOL | t > 35 / t ≤ 26 | ALERT / IDLE | FAN |
| ALERT | t ≤ 30 | COOL | ALARM + FAN |

#### 함수 포인터 테이블 상태 머신
> `enum` + 함수 포인터 테이블 — 상태가 곧 인덱스, 핸들러의 반환값 = 다음 상태
- **한 줄** → `State next = TABLE[st](temp);`
- **핸들러** → 먼저 자기 상태의 **핀 출력**, 조건을 보고 다음 상태 반환 (COOL, 36도 → 팬 켬(0x2) 후 ALERT 반환)
- **상태 추가** → enum 한 줄(COUNT 앞) + `NAME[]` + 핸들러 + `TABLE` 항목
- **체류 횟수** → `int visits[ST_COUNT]; visits[st]++`
- ⚠ **테이블 순서가 enum 과 어긋나면** (HEAT/COOL 바뀜) → 컴파일·실행은 되는데 히터 자리에서 팬이 돈다
- **막는 법** → **지정 초기화** `[ST_HEAT] = on_heat`
- ⚠ 빠뜨린 항목은 `NULL` → 호출 전 `NULL`/범위 검사

**기본 사용형태**
```c
typedef void (*Callback)(double);
static Callback on_alert = NULL;
void register_alert(Callback cb) { on_alert = cb; }
typedef State (*Handler)(double);
static const Handler TABLE[ST_COUNT] = {
    [ST_IDLE] = on_idle, [ST_HEAT] = on_heat, [ST_COOL] = on_cool, [ST_ALERT] = on_alert };
State next = TABLE[st](t);
```

**연결**
- [C D1·6교시 연산자와 제어문](#/c/1/6) — `switch` 로 짜던 분기를 테이블로
- [MCU D1·6교시 Interrupt — ISR](#/mcu/1/6) — `attachInterrupt` 콜백 등록
- [MCU D1·3교시 Digital Output — 신호등](#/mcu/1/3) — 신호등 상태 머신
- [EIO D2·7교시 반응속도 게임 설계](#/eio/2/7) — 게임 흐름을 상태로 설계

**연관 문제**
<<QA 7>>

### 8교시 · 종합 — 온도 조절기 컨트롤러

**키워드**: `controller`, `parse_frame + 체크섬`, `fsm_step`, `TABLE[cur](t)`, `가상 GPIO`, `HEATER FAN ALARM 핀`, `LOG -DDEBUG`, `visits[ST_COUNT]`, `qsort 내림차순 상위 3`, `extern const char *STATE_NAME[]`, `헤더엔 선언 .c 에 정의`, `하드웨어 의존 gpio.h 격리`, `make CFLAGS=…`

**개요**: Day 2 센서 로그 분석기에 오늘 배운 것을 모두 얹는다. 표준 입력의 프레임을 파싱·체크섬 검증하고, **상태 머신**으로 온도 조절기를 돌리며, **가상 GPIO 레지스터**로 히터·팬·알람 핀을 제어하고, 디버그 빌드에서만 전이 로그를 남긴 뒤, 상태별 체류 횟수와 온도 상위 3개를 출력한다. 핵심 설계 원칙은 **하드웨어 의존 부분을 `gpio.h` 한 곳에 가두는 것** — 다음 주 실제 보드로 옮길 때 그 파일만 고치면 된다.

**핵심 용어**
- **파이프라인**: 입력 → 파싱 → 판단 → 출력 → 기록처럼 단계를 순서대로 잇는 구조.
- **하드웨어 추상화**: 하드웨어에 직접 닿는 코드를 한 파일(`gpio.h`)에 모아, 나머지 코드는 하드웨어를 몰라도 되게 하는 설계.
- **선언/정의 분리 (전역 배열)**: 헤더에는 `extern` 선언, 정의는 `.c` 한 곳. 헤더에 정의를 두면 include 한 파일마다 정의가 생겨 **중복 정의 링크 에러**.

**새로 나온 함수·명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `fsm_step` | `State fsm_step(State cur, double t);` | `return TABLE[cur](t);` — 한 단계 전이. |
| `cmp_temp_desc` | `int cmp_temp_desc(const void *a, const void *b);` | 온도 내림차순 비교 (상위 3개). |
| `make CFLAGS=` | `make CFLAGS="-Wall -Wextra -g -DDEBUG"` | Makefile 변수를 명령줄에서 덮어써 디버그 빌드. |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `controller/fsm.h, fsm.c` | `State` enum, `STATE_NAME` (extern 선언/정의), 핸들러 4개 + 지정 초기화 `TABLE` + `fsm_step` |
| `controller/gpio.h` | 히터·팬·알람 핀 번호와 `GPIO_SET/CLR` 매크로 (하드웨어 의존 격리) |
| `controller/main.c` | 프레임 읽기 → 파싱(+체크섬) → `fsm_step` → 로그 → 마지막에 체류 횟수·상위 3 온도 출력 |
| `temps.txt` | 상태 전이를 모두 거치는 테스트 온도열 |

**핵심 내용**

#### 파이프라인
> 프레임 → 파싱 → 상태 전이 → 가상 GPIO → 로그 → 통계
- **파싱** → `parse_frame` + `strtol(star+1, NULL, 16)` 체크섬
- **전이** → `fsm_step` = `TABLE[cur](t)`
- **출력** → 가상 GPIO (`GPIO_SET/CLR` 매크로로만)
- **로그** → `LOG` 조건부 컴파일
- **마지막** → 상태별 체류 횟수 + 온도 상위 3개 (`qsort` 내림차순)

#### 파일 분리와 extern
> 헤더에는 선언, 정의는 `.c` 한 곳
- **`fsm.h`** → `extern const char *STATE_NAME[ST_COUNT];` (**선언**) / **`fsm.c`** → **정의**
- ⚠ 헤더에 정의를 넣으면 → 중복 정의 링크 에러
- **`gpio.h` 의 `extern uint32_t GPIO_OUT`** → `main.c` 에 **정의** (없으면 undefined reference)
- **하드웨어 의존 부분** → `gpio.h` **한 곳에 가둔다** → 실제 하드웨어로 옮길 때 그 파일만 수정

#### 디버그 빌드와 결과 읽기
> `-DDEBUG` 를 주면 전이 로그가 stderr 에
- **빌드** → `make CFLAGS="-Wall -Wextra -g -DDEBUG"`
- **ALERT(29도)** → 핸들러가 먼저 핀을 켜서 GPIO=**0x6**, 다음 상태는 COOL
- **`visits[st]++`** → 전이 후 상태를 센다
- **제출 체크** → 경고 0 / `TABLE` 지정 초기화 / 핀 제어는 매크로로만 (직접 `|=` 금지) / `visits[ST_COUNT]`

**기본 사용형태**
```c
State fsm_step(State cur, double t) { return TABLE[cur](t); }
qsort(log, n, sizeof log[0], cmp_temp_desc);
uint32_t GPIO_OUT = 0, GPIO_IN = 0;        // main.c 에 정의
```

**연결**
- [C D4·7교시 종합 프로젝트 ①](#/c/4/7) — 파일 I/O·링 버퍼를 더한 `logger`
- [C D4·6교시 코드 품질 — 모듈 구조](#/c/4/6) — 오늘의 파일 분리를 모듈 규칙으로
- [MCU D4·8교시 IoT 환경 모니터](#/mcu/4/8) — 같은 구조를 실제 보드에서

**연관 문제**
<<QA 8>>
