## Day 2 — 포인터 · 배열 · 문자열 · 동적 메모리

### 1교시 · 포인터 기초

**키워드**: `&x 주소`, `*p 역참조`, `int *p`, `%p`, `(void *)`, `swap`, `주소 전달`, `scanf &n`, `초기화 안 된 포인터`, `wild pointer`, `NULL`, `Segmentation fault`, `-Wuninitialized`, `gdb bt`, `*p++ vs (*p)++`, `후위 ++ 우선순위`

**개요**: Day 1 에서 세 번 미뤄 둔 것(`(unsigned char *)&v`, `const char *level()`, `judge(…, &s, &b)`)을 갚는 시간이다. 변수의 **주소(`&`)** 를 담는 변수가 **포인터**이고, 포인터 앞에 `*` 를 붙이면 그 주소에 있는 **값**을 읽고 쓸 수 있다. 1교시는 ① 주소는 그냥 숫자라는 것, ② 함수에 **주소를 넘기면** 호출한 쪽 변수를 바꿀 수 있다는 것(Day 1 `inc` 가 실패한 자리에서 `swap` 성공), ③ 아무 데도 가리키지 않는 포인터를 쓰면 죽는다는 것, ④ `*p++` 와 `(*p)++` 의 차이를 본다.

**핵심 용어**
- **주소(address)**: 메모리의 위치를 나타내는 숫자. `&x` = x 가 놓인 주소. 64비트 Linux 에서 8바이트.
- **포인터(pointer)**: 주소를 값으로 담는 변수. `int *p` 는 "p 를 역참조(`*p`)하면 int 가 나온다" 로 읽는다.
- **역참조(dereference)**: `*p` — p 가 가리키는 곳의 값에 접근(읽기·쓰기).
- **`NULL`**: "아무 데도 가리키지 않음" 을 뜻하는 특별한 포인터 값. 역참조하면 죽지만 `if (p == NULL)` 로 **검사할 수 있다**.
- **와일드 포인터(wild pointer)**: 초기화하지 않아 쓰레기 주소를 담은 포인터. 역참조하면 대부분 **Segmentation fault**.
- **세그멘테이션 폴트(segfault)**: 접근 권한이 없는 메모리를 건드려 운영체제가 프로그램을 강제 종료하는 것.

**새로 나온 함수·명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `%p` | `printf("%p", (void *)p)` | 주소 출력. 인자는 `void *` 로 캐스팅하는 것이 규칙. |
| `swap` | `void swap(int *a, int *b);` | 두 변수 값 교환 (직접 구현, 4교시 정렬에서 재사용). |
| `gdb -batch` | `gdb -batch -ex run -ex bt ./prog` | 프로그램을 돌려 죽은 곳의 호출 스택(`bt`)과 **줄 번호** 출력 (`-g` 필요). |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 1-1 `1_addr.c` | `x`, `&x`, `p`, `*p` 를 출력하고 `*p = 42` 로 x 를 바꿈 |
| 1-2 `2_swap.c` | `inc_wrong(x)` 는 5 그대로, `inc_right(&x)` 는 6, `swap(&x, &y)` 로 `9 6` |
| 1-3 `3_wild.c` | 초기화 안 된 `int *p` 에 `*p = 10` → `-Wuninitialized` 경고 + 세그폴트, gdb 로 죽은 줄 확인 |
| 1-4 `4_postinc.c` | `*p++` 는 3 을 출력하고 p 이동, `(*p)++` 는 `arr[1]` 을 1→2 |

**핵심 내용**

#### 주소와 역참조
> `&x` 는 x 의 주소, `*p` 는 p 가 가리키는 곳의 값
- **`&x`** → x 의 **주소**
- **`*p`** → p 가 가리키는 곳의 **값** (읽기·쓰기 모두)
- **`int *p = &x;`** → `p` 는 **8바이트** 주소, `*p` 는 4바이트 int (Day 1 `sizeof(int *)` = 8 이었던 이유)

#### 주소를 넘겨 원본 바꾸기
> 값 전달은 그대로 — 전달되는 값이 "주소" 일 뿐이다
- **원본을 바꾸려면** → 주소를 넘긴다
- **예** → `swap(&x, &y)`, `scanf("%d", &n)`, 숫자 야구 `judge(…, &s, &b)`
- **함수 모양** → 반환값은 성공 여부, 결과는 포인터로 (`scanf`, `strtol`)

#### 초기화와 NULL
> 포인터는 반드시 가리킬 곳을 정해 두고 쓴다
- ⚠ 초기화 안 된 포인터 역참조 → 쓰레기 주소 → 대부분 **Segmentation fault**
- **초기화** → `&변수` / 배열 / `malloc` / **`NULL`**
- **`NULL` 이면** → `if (p == NULL)` 로 검사 가능
- **죽은 줄 찾기** → `-g` + `gdb … -ex bt`

#### `*p++` vs `(*p)++`
> 후위 `++` 가 `*` 보다 먼저 묶인다
- **`*p++`** = `*(p++)` → 값을 쓰고 **포인터**를 다음 칸으로
- **`(*p)++`** → 가리키는 **값**을 +1
- **숫자 야구** → `(*strike)++` 의 괄호가 그래서 필요

**기본 사용형태**
```c
int x = 5; int *p = &x;
*p = 42;                                  // x 가 42
printf("%p\n", (void *)p);                // 주소 출력
void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }
swap(&x, &y);
void inc(int *p) { (*p)++; }
```

**연결**
- [C D1·7교시 함수와 스코프](#/c/1/7) — 값 전달의 한계 (`inc` 실패)
- [C D3·6교시 함수 포인터](#/c/3/6) — 함수의 주소를 담는 포인터
- [C D4·5교시 디버깅 — gdb](#/c/4/5) — 오늘 맛본 `bt` 의 본편

**연관 문제**
<<QA 1>>

### 2교시 · 포인터 산술과 const

**키워드**: `p + n`, `n × sizeof(*p)`, `arr[i] == *(arr + i)`, `포인터 뺄셈 = 요소 개수`, `(char *) 캐스팅`, `uint8_t buf → uint16_t`, `바이트 조립 buf[0] | buf[1] << 8`, `정렬(alignment)`, `리틀 엔디안`, `const int *`, `int *const`, `const int *const`, `오른쪽에서 왼쪽으로 읽기`, `read-only`

**개요**: 포인터에 1을 더하면 주소가 1 늘어날까? 아니다 — **가리키는 타입 크기만큼** 이동한다. 2교시는 ① 포인터 덧셈·뺄셈의 단위가 "바이트" 가 아니라 "요소" 라는 것, ② 센서가 보낸 바이트 배열을 `uint16_t` 로 읽는 두 방법(포인터 캐스팅 vs 바이트 조립)과 왜 임베디드는 **바이트 조립**을 쓰는지(Day 1 엔디안 회수), ③ `const` 가 포인터 어디에 붙느냐에 따라 "값을 못 바꿈" 과 "포인터를 못 바꿈" 으로 갈린다는 것을 다룬다.

**핵심 용어**
- **포인터 산술(pointer arithmetic)**: `p + n` 은 `n × sizeof(*p)` 바이트 뒤의 주소. 그래서 `arr[i]` 는 `*(arr + i)` 의 다른 표기.
- **포인터 뺄셈**: 같은 배열 안의 두 포인터 차이 = **요소 개수**. 바이트 거리는 `(char *)` 로 바꿔서 뺀다.
- **정렬(alignment)**: CPU 가 `uint16_t`/`uint32_t` 를 2·4의 배수 주소에서 읽기를 원하는 규칙. 맞지 않는 주소를 캐스팅해 읽으면 일부 MCU 는 죽는다.
- **바이트 조립**: `buf[0] | (buf[1] << 8)` 처럼 시프트와 OR 로 여러 바이트를 한 값으로 만드는 방법. 정렬·엔디안과 무관하게 동작.
- **`const` 포인터 읽는 법**: 오른쪽에서 왼쪽으로. `const int *p` = "const int 를 가리키는 포인터"(값 고정), `int *const p` = "int 를 가리키는 const 포인터"(포인터 고정).

**새로 나온 함수·명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `const int *p` | `int sum(const int *a, int n);` | 가리키는 값을 못 바꿈 → "읽기만 한다" 는 약속 + 컴파일러 검사. |
| `int *const p` | `int *const p = &a;` | 포인터 자체를 못 바꿈 (레지스터 주소 고정 등). |
| `%ld` + `(long)` | `printf("%ld", (long)(q - p))` | 포인터 뺄셈 결과(`ptrdiff_t`) 출력. |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 2-1 `1_ptr_plus.c` | `int*`/`double*`/`char*` 에 +1 했을 때 4/8/1 바이트 이동, `*(p+2)` = 30, `&arr[4]-&arr[0]` = 4 |
| 2-2 `2_bytes_to_u16.c` | `{0x2C,0x01,0x3C,0x00}` 을 캐스팅으로 `300 60`, 바이트 조립으로 `300` |
| 2-3 `3_const.c` | `p1`(값 고정)·`p2`(포인터 고정)·`p3`(둘 다) 에 허용/금지되는 대입 확인 → `9 2 9` |

**핵심 내용**

#### 포인터 산술
> `p + n` 은 n 칸 이동 — 바이트 수는 가리키는 타입 크기 × n
- **`p + n`** → `n × sizeof(*p)` 바이트 이동
- **예** → `int*` +1 = **4**, `double*` +2 = **16**, `char*` +1 = **1**
- **`arr[i]`** ≡ `*(arr + i)`
- **같은 주소라도** → `int *` 냐 `char *` 냐에 따라 +1 의 의미가 다름
- **포인터 뺄셈** (같은 타입) → **요소 개수** (`&arr[4] - &arr[0]` = 4)
- **바이트 거리** → `(char *)` 로 캐스팅해서 뺀다

#### 바이트 조립
> 바이트 배열을 큰 정수로 바꿀 때는 캐스팅 대신 비트 조립
- ⚠ 포인터 캐스팅 (`*(uint16_t *)buf`) → **정렬 · 엔디안** 문제로 위험
- **`buf[0] | (buf[1] << 8)`** → **이식성 100%** (임베디드 권장, Day 3 비트 연산의 출발점)

#### const 위치
> `const` 가 `*` 왼쪽이면 값, 오른쪽이면 포인터를 고정 — 오른쪽에서 왼쪽으로 읽는다
- **`const int *p`** → **값**을 못 바꿈 (포인터는 이동 가능)
- **`int *const p`** → **포인터**를 못 바꿈 (값은 변경 가능)
- **`const int *const p`** → 둘 다 못 바꿈
- **함수 인자 `const int *a`** → "읽기만 한다" 는 약속 + 컴파일러 검사 (Day 1 `is_valid(const int g[3])`)

**기본 사용형태**
```c
int *p = arr;  *(p + 2);                           // arr[2]
uint16_t dist = buf[0] | (buf[1] << 8);            // 0x2C,0x01 → 300
const int *p1 = &a;   // *p1 = 9; 불가, p1 = &b; 가능
int *const p2 = &a;   // *p2 = 9; 가능, p2 = &b; 불가
int sum(const int *a, int n);
```

**연결**
- [C D1·4교시 자료형과 메모리 표현](#/c/1/4) — 리틀 엔디안의 출발점
- [C D3·3교시 비트 연산 기초](#/c/3/3) — 시프트·OR 로 값 조립
- [EIO D4·1교시 I2C 와 PCF8591](#/eio/4/1) — 버스로 받은 바이트를 값으로 조립하는 실제 상황

**연관 문제**
<<QA 2>>

### 3교시 · 배열과 포인터

**키워드**: `배열 감쇠(decay)`, `arr == &arr[0]`, `2[arr]`, `sizeof(arr) 20 vs 8`, `int arr[] 매개변수 = int *`, `요소 개수 sizeof(arr)/sizeof(arr[0])`, `(배열, 길이)`, `2차원 배열`, `행 우선`, `int m[][3]`, `평균 n ? s/n : 0`, `max_index`

**개요**: 배열과 포인터는 비슷해 보이지만 같지 않다. 3교시는 ① 배열 이름이 식에서 **첫 요소의 주소로 바뀌는(감쇠)** 현상, ② 그 때문에 함수에 배열을 넘기면 **길이 정보가 사라져** `sizeof` 가 8 이 되는 함정, ③ 그래서 C 함수는 항상 **(배열, 길이)** 를 함께 받는다는 관례, ④ 2차원 배열이 메모리에 **행 우선으로 한 덩어리**로 놓인다는 것을 본다. 마지막으로 센서 온도 배열의 평균·최댓값을 함수로 구해 연습한다.

**핵심 용어**
- **배열 감쇠(array decay)**: 배열 이름이 식에서 첫 요소의 주소(`&arr[0]`)로 바뀌는 것. 예외는 `sizeof arr` 와 `&arr`.
- **배열 매개변수**: `void f(int arr[])` 는 사실 `void f(int *arr)`. 함수 안의 `sizeof(arr)` 는 포인터 크기(8).
- **요소 개수 관용구**: `sizeof(arr) / sizeof(arr[0])` — **배열이 선언된 곳에서만** 맞다.
- **행 우선(row-major)**: 2차원 배열 `m[R][C]` 를 행 단위로 이어 붙여 저장. `m[i][j]` 주소 = 시작 + `(i × C + j) × sizeof(요소)`.

**새로 나온 함수·명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `avg` | `double avg(const double *a, int n);` | 평균 (`n ? s / n : 0` 로 0 나누기 방지). |
| `max_index` | `int max_index(const double *a, int n);` | 최댓값의 **인덱스** 반환 (값은 `a[i]` 로). |
| 2차원 인자 | `void f(const int a[][3]);` | 열 수를 알려야 주소 계산이 가능. |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 3-1 `1_arr_ptr.c` | `arr[2]`, `*(arr+2)`, `2[arr]` 가 모두 30, `arr` == `&arr[0]`, `sizeof(arr)`=20, 개수 5 |
| 3-2 `2_decay.c` | main 에서 `sizeof`=20, 함수 안에서 8 (+ 경고), `sum(arr, 5)` = 150 |
| 3-3 `3_2d.c` | `int m[2][3]` 의 `sizeof`=24, 행 간격 12바이트, 1차원처럼 순회하면 `1 2 3 4 5 6` |
| 3-4 `4_stats.c` | 온도 6개의 평균 27.37, 최댓값 37.2 와 위치 3 출력 |

**핵심 내용**

#### 배열 이름의 감쇠
> 배열 이름은 식에서 첫 요소의 주소가 된다 (`sizeof`, `&` 는 예외)
- **`arr`** ≡ `&arr[0]`
- **`arr[i]`** = `*(arr+i)` = `*(i+arr)` = `i[arr]` (이해만, 쓰지 말 것)

#### sizeof 와 배열 길이
> 배열 크기는 선언된 곳에서만 알 수 있다
- **선언된 곳 `sizeof(arr)`** → 전체 크기 (int 5개면 **20**)
- ⚠ **함수 안 `sizeof(arr)`** → 포인터 크기 **8** (`int arr[]` 매개변수 = `int *arr`, gcc 가 경고)
- **요소 개수** → `sizeof(arr) / sizeof(arr[0])` (배열이 보이는 곳에서만)
- **규칙** → `(배열, 길이)` 를 항상 함께 넘긴다
- **라이브러리도 같은 모양** → `fwrite(buf, size, n, fp)`

#### 2차원 배열
> 행 우선 한 덩어리 메모리
- **`m[i][j]` 오프셋** → `(i × 열수 + j) × 4`
- **함수 인자** → **열 수 필수** (`int a[][3]`)

#### 배열 평균
> 0 개일 때를 막는다
- **`n ? s / n : 0`** → 0 나누기 방지

**기본 사용형태**
```c
int n = sizeof(arr) / sizeof(arr[0]);
double avg(const double *a, int n) { double s = 0; for (int i = 0; i < n; i++) s += a[i]; return n ? s / n : 0; }
int m[2][3] = {{1,2,3},{4,5,6}};   // m[1][0] 은 시작에서 12바이트 뒤
void matmul(const int a[][3], const int b[][3], int c[][3]);
```

**연결**
- [C D4·3교시 바이너리 I/O](#/c/4/3) — `fwrite(buf, size, n, fp)` 의 (배열, 길이) 모양
- [MCU D3·4교시 8×8 도트 매트릭스](#/mcu/3/4) — 2차원 비트맵 배열

**연관 문제**
<<QA 3>>

### 4교시 · 정렬과 검색

**키워드**: `버블 정렬`, `n - 1 - i`, `off-by-one`, `배열 밖 접근`, `swap 재사용`, `선택 정렬`, `O(n²)`, `이진 검색`, `정렬 전제`, `mid = lo + (hi - lo) / 2`, `오버플로`, `못 찾으면 -1`, `stack smashing`, `0x524F534E`, `중앙값`

**개요**: 1교시의 `swap` 을 재사용해 **버블 정렬**을 만들고, 정렬된 배열에서만 쓸 수 있는 **이진 검색**을 구현한다. 그 과정에서 반복 범위를 하나 잘못 잡는 **off-by-one** 이 배열 밖을 건드리는 순간을 보고, 배열 밖에 쓰면 컴파일러가 막지 않아 **옆 변수가 조용히 바뀐다**는 것을 확인한다. MCU 에는 이런 침범을 잡아 줄 보호 장치가 없다는 점이 핵심이다.

**핵심 용어**
- **버블 정렬(bubble sort)**: 이웃한 두 값을 비교해 바꾸는 것을 반복. 한 바퀴 돌 때마다 가장 큰 값이 맨 뒤로 간다. O(n²).
- **이진 검색(binary search)**: 정렬된 배열의 가운데와 비교해 절반씩 버리는 검색. O(log n). **정렬이 전제**.
- **off-by-one**: 경계 값을 1 틀리는 실수 (`<` vs `<=`, `n` vs `n-1`). 배열 밖 접근의 가장 흔한 원인.
- **경계 침범(out-of-bounds write)**: 배열 크기를 넘어 쓰는 것. 정의되지 않은 동작이며 옆 메모리를 덮는다.
- **스택 보호(stack smashing detected)**: glibc 가 스택 버퍼 넘침을 감지해 종료하는 장치. MCU 에는 없다.

**새로 나온 함수·명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `bubble` | `int bubble(int *a, int n);` | 버블 정렬, 교환 횟수 반환 (직접 구현). |
| `bsearch_int` | `int bsearch_int(const int *a, int n, int key);` | 이진 검색. 찾으면 인덱스, 없으면 `-1`. |
| `strcpy` | `char *strcpy(char *dst, const char *src);` | 문자열 복사 — **길이 검사 없음** (함정 시연용, 5교시에서 본격). |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 4-1 `1_bubble.c` | `{26,25,37,25,24}` 를 정렬해 `24 25 25 26 37 (7 swaps)` |
| 4-2 `2_bsearch.c` | 정렬된 배열에서 26 은 3, 30 은 `-1` |
| 4-3 `3_overflow.c` | `struct Dev { char name[8]; int level; }` 에 13글자 복사 → `level` 이 `0x524F534E`("NSOR")로 바뀜 |

**핵심 내용**

#### 버블 정렬
> 이웃을 비교해 바꾸기를 반복 — 한 바퀴마다 큰 값이 뒤로
- **안쪽 루프 상한** → `n - 1 - i` (뒤쪽 i 개는 이미 자리 잡음)
- ⚠ `-1` 을 빼면 → `a[j+1]` 이 `a[n]` → **off-by-one, 배열 밖 접근**
- **선택 · 버블 정렬** → **O(n²)**

#### 이진 검색
> 정렬된 배열에서 가운데와 비교해 절반씩 버린다
- **전제** → **정렬된 배열**
- **가운데** → `mid = lo + (hi - lo) / 2`
- ⚠ `(lo + hi) / 2` → 오버플로 가능
- **못 찾으면** → **`-1`** (유효 인덱스는 0 이상 → 반환값 하나로 결과와 실패를 같이 표현)

#### 중앙값
> 짝수 개면 가운데 두 값의 평균
- **`(a[n/2-1] + a[n/2]) / 2.0`**
- ⚠ `/ 2` → 정수 나눗셈

#### 배열 밖 쓰기
> 컴파일러가 막지 않는다 — 옆 변수가 조용히 바뀐다
- **`name[8]` 에 13글자** → 옆 `level` 이 `0x524F534E` = "NSOR"
- **스택이면** → glibc 가 `*** stack smashing detected ***` 로 잡기도 함
- ⚠ **MCU 에는** 그런 보호가 없다

**기본 사용형태**
```c
for (int i = 0; i < n - 1; i++)
    for (int j = 0; j < n - 1 - i; j++)
        if (a[j] > a[j + 1]) swap(&a[j], &a[j + 1]);
int lo = 0, hi = n - 1;
while (lo <= hi) { int mid = lo + (hi - lo) / 2;
    if (a[mid] == key) return mid; if (a[mid] < key) lo = mid + 1; else hi = mid - 1; }
return -1;
```

**연결**
- [C D3·6교시 함수 포인터](#/c/3/6) — 직접 짠 정렬을 `qsort` + 비교 함수로
- [C D4·5교시 디버깅 — sanitizer](#/c/4/5) — 배열 밖 접근을 `-fsanitize=address` 로 잡기

**연관 문제**
<<QA 4>>

### 5교시 · 문자열

**키워드**: `'\0' 널 종료`, `"ABC" = 4바이트`, `sizeof vs strlen`, `char s[] vs char *s`, `문자열 리터럴 읽기 전용`, `const char *`, `my_strlen my_strcpy my_strcmp`, `*d++ = *s++`, `strcmp == 0`, `strstr`, `strchr`, `atoi atof 실패 모름`, `strcpy 오버플로`, `stack smashing detected`, `buffer overflow`

**개요**: C 에는 문자열 타입이 없다. 문자열은 **`'\0'`(널) 로 끝나는 `char` 배열**이라는 약속일 뿐이다. 5교시는 ① `"ABC"` 가 왜 4바이트인지(`sizeof` 4 / 포인터 8 / `strlen` 3 세 숫자 구분), ② `char s[]`(수정 가능한 복사본)와 `char *s`(읽기 전용 리터럴)의 차이, ③ `strlen/strcpy/strcmp` 를 포인터 순회로 **직접 구현**해 원리를 보고, ④ `strstr/strchr/atof` 로 센서 프레임에서 값을 찾으며, ⑤ 길이를 검사하지 않는 `strcpy` 가 만드는 **버퍼 오버플로**를 확인한다.

**핵심 용어**
- **널 종료 문자열(null-terminated string)**: 끝에 `'\0'`(값 0)이 있는 char 배열. 모든 문자열 함수는 `'\0'` 을 만날 때까지 읽는다.
- **문자열 리터럴**: 코드에 쓴 `"ABC"`. 프로그램의 **읽기 전용 영역**에 놓인다.
- **`char s[] = "ABC"`**: 리터럴을 배열(스택)에 **복사** → 수정 가능.
- **`char *s = "ABC"`**: 리터럴을 **가리키기만** → 수정하면 세그폴트. `const char *` 로 선언해 컴파일 에러로 바꾸는 것이 안전.
- **버퍼 오버플로(buffer overflow)**: 배열보다 긴 데이터를 써서 옆 메모리를 덮는 것. 입력이 사용자에게서 오면 **보안 취약점**.

**새로 나온 함수·명령**

| 함수 | 기본형 | 용도 |
|---|---|---|
| `strlen` | `size_t strlen(const char *s);` | `'\0'` 앞까지의 글자 수. |
| `strcpy` | `char *strcpy(char *dst, const char *src);` | `'\0'` 까지 복사. **크기 검사 없음**. |
| `strcmp` | `int strcmp(const char *a, const char *b);` | 같으면 **0**, a 가 앞서면 음수, 뒤면 양수. |
| `strstr` | `char *strstr(const char *s, const char *sub);` | 부분 문자열이 시작하는 위치, 없으면 `NULL`. |
| `strchr` | `char *strchr(const char *s, int c);` | 글자 c 의 첫 위치, 없으면 `NULL`. |
| `atoi` / `atof` | `int atoi(const char *s);` / `double atof(const char *s);` | 문자열 → 숫자. 숫자 아닌 곳에서 멈추고 **실패를 알 수 없다**. |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 5-1 `1_abc.c` | `sizeof(s1)`=4, `sizeof(s2)`=8, `strlen`=3, `s1[0]='X'` 는 OK(`XBC`), `s2[0]` 수정은 세그폴트 |
| 5-2 `2_mystr.c` | `my_strlen/my_strcpy/my_strcmp` 를 포인터 순회로 구현 → `6 sensor 0 -1` |
| 5-3 `3_find.c` | `"$T=25.3,H=60,D=120*"` 에서 `strstr` 로 `T=`/`H=` 위치, `atof/atoi` 로 `25.3 60`, `'*'` 인덱스 18 |
| 5-4 `4_ovf.c` | `char buf[8]` 에 `"ESP32-SENSOR"` 복사 → 경고, 출력 후 `stack smashing detected` |

**핵심 내용**

#### 문자열 = 널로 끝나는 char 배열
> `"ABC"` 는 `'\0'` 까지 4바이트
- **`sizeof(s[])`** → **4** (널 포함)
- **`sizeof(char*)`** → **8**
- **`strlen`** → **3** (널 전까지)
- **핵심** → 이 세 숫자를 구분한다

#### 배열 vs 리터럴 포인터
> 배열은 수정 가능한 복사본, 리터럴은 읽기 전용
- **`char s[] = "ABC"`** → 스택에 **복사본** (수정 가능)
- **`char *s = "ABC"`** → **읽기 전용 리터럴**을 가리킴
- ⚠ 리터럴을 수정하면 → **Segmentation fault**
- **`const char *`** 로 선언하면 → 컴파일 에러로 바뀜

#### 문자열 함수 직접 구현
> 포인터를 널까지 옮기는 루프
- **`strlen`** → `while (*p) p++; return p - s;` (포인터 뺄셈 = 글자 수)
- **`strcpy`** → `while ((*d++ = *s++) != '\0') {}` (1교시 `*p++` 응용)
- **`strcmp`** → **0 이면 같다** (`if (strcmp(a,b))` 는 "다르면")
- **뒤집기** → `b = s + strlen(s) - 1` (널 앞이 마지막 글자)

#### 찾기 · 변환과 위험
> 표준 함수는 편하지만 실패·크기를 확인하지 않는 것이 있다
- **`strstr`** → 부분 문자열 위치 / **`strchr`** → 글자 위치
- ⚠ **`atoi/atof`** → 실패를 알 수 없음 (0 반환) → 6교시 `strtol`
- ⚠ **`strcpy`** → 길이 검사 없음 → 8칸에 13바이트 = **버퍼 오버플로** (`stack smashing detected`)
- ⚠ 사용자 입력이면 → 보안 취약점

**기본 사용형태**
```c
char s1[] = "ABC";  const char *s2 = "ABC";
if (strcmp(cmd, "quit") == 0) break;
const char *t = strstr(frame, "T=");  double v = atof(t + 2);
size_t my_strlen(const char *s) { const char *p = s; while (*p) p++; return p - s; }
```

**연결**
- [C D1·6교시 연산자와 제어문](#/c/1/6) — 그때 미뤄 둔 `const char *level()` 의 정체
- [LINUX D1·8교시 myshell 만들기](#/linux/1/8) — 명령줄 문자열을 잘라 실행

**연관 문제**
<<QA 5>>

### 6교시 · 안전한 문자열 처리와 파싱

**키워드**: `fgets`, `scanf("%s") 위험`, `sizeof buf`, `strcspn 개행 제거`, `strncpy 널 미보장`, `snprintf 항상 널`, `strtok`, `원본 수정`, `strtok(NULL, ",")`, `strtol`, `strtod`, `end 포인터`, `end == s 실패`, `parse_frame`, `성공 1 / 실패 0`, `out 은 성공 시만`, `static 도우미`, `센서 프레임 $T=..,H=..,D=..*`

**개요**: 5교시의 함수들은 크기를 모르거나(`strcpy`, `scanf("%s")`) 실패를 알려 주지 않는다(`atoi`). 6교시는 그 **안전한 대체품**을 익힌다 — 크기를 아는 입력 `fgets`, 항상 널로 끝나는 복사 `snprintf`, 토큰 분리 `strtok`, 실패를 알려 주는 변환 `strtol/strtod`. 그리고 이것들을 묶어 센서 프레임 `"$T=25.3,H=60,D=120*"` 을 `Reading` 구조체로 채우는 **`parse_frame`** 을 만든다. 잘못된 프레임은 거부하고, 실패 시 결과를 반쯤 채우지 않는 것이 규칙이다.

**핵심 용어**
- **파싱(parsing)**: 문자열을 규칙에 따라 잘라 의미 있는 값(숫자, 필드)으로 바꾸는 것.
- **토큰(token)**: 구분자(`,` 등)로 잘린 조각 하나.
- **`end` 포인터**: `strtol(s, &end, 10)` 이 "숫자가 끝난 위치" 를 돌려주는 출력 인자. `end == s` 면 숫자가 하나도 없었다 = 실패.
- **출력 매개변수(out parameter)**: 결과를 담아 돌려받는 포인터 인자 (`Reading *out`). 반환값은 성공 여부.
- **`static` 도우미 함수**: 파일 안에서만 쓰는 함수 (Day 1 7교시의 내부 연결).

**새로 나온 함수·명령**

| 함수 | 기본형 | 용도 |
|---|---|---|
| `fgets` | `char *fgets(char *buf, int size, FILE *fp);` | 최대 `size-1` 글자 + `'\0'`. 개행도 담는다. EOF/오류면 `NULL`. |
| `strcspn` | `size_t strcspn(const char *s, const char *reject);` | reject 글자가 처음 나오는 위치 → `line[strcspn(line,"\n")] = '\0'`. |
| `strncpy` | `char *strncpy(char *d, const char *s, size_t n);` | 최대 n 글자 복사. 꽉 차면 **`'\0'` 을 안 붙인다** (함정). |
| `snprintf` | `int snprintf(char *d, size_t n, const char *fmt, ...);` | 크기 안에서 자르고 **항상 `'\0'`**. 안전한 복사·조립. |
| `strtok` | `char *strtok(char *s, const char *delim);` | 구분자 자리에 `'\0'` 을 써서 토큰 반환. 다음은 `strtok(NULL, …)`. |
| `strtol` / `strtod` | `long strtol(const char *s, char **end, int base);` / `double strtod(const char *s, char **end);` | 문자열 → 정수/실수, `end` 로 실패 감지. |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 6-1 `1_fgets.c` | `fgets` 로 한 줄 읽고 개행 제거해 `[내용] 길이` 출력 (31자 넘게 넣어도 잘리기만 함) |
| 6-2 `2_strncpy.c` | `strncpy(a, "ABCDEFG", 4)` → 널 없는 `ABCD`, `snprintf` 는 `ABC` |
| 6-3 `3_strtok.c` | 프레임을 `,` 로 잘라 `[T=25.3] [H=60] [D=120*]`, `strtol("120*")` → `120 '*'` |
| 6-4 `4_frame/` | `frame.h`(Reading) · `frame.c`(`parse_frame` + `static get_num`) · `test_frame.c`(정상 2 / 비정상 4 판별) |

**핵심 내용**

#### 안전한 입력과 복사
> 크기를 아는 함수만 쓴다 — `fgets`, `snprintf`
- ⚠ `scanf("%s")` → 크기를 몰라 오버플로
- **입력** → `fgets(buf, sizeof buf, stdin)`
- **끝의 개행 제거** → `line[strcspn(line, "\n")] = '\0';`
- ⚠ **`strncpy`** → 널을 보장하지 않음 (칸을 다 채우면 `'\0'` 없음 → `%s` 출력 시 옆 메모리까지)
- **자를 때** → `snprintf(dst, sizeof dst, "%s", src)` (항상 널 종료)

#### strtok 으로 자르기
> 구분자 자리에 `'\0'` 을 써 넣어 토큰을 자른다
- **첫 호출** → `strtok(line + 1, ",")`
- **다음 토큰** → `strtok(NULL, ",")`
- ⚠ 원본이 바뀐다 → **리터럴에는 못 씀** (배열이어야 함)

#### strtol 로 실패 감지
> 숫자가 끝난 위치를 알려 줘서 실패를 구분할 수 있다
- **`strtol(s, &end, 10)`** → `end` = 숫자가 끝난 위치
- **`end == s`** → **변환 실패** (숫자가 하나도 없음)
- **실수** → `strtod`

#### parse_frame 설계
> 검사 → 추출 → 성공일 때만 결과를 채운다
- **형식 검사** → `$` 로 시작, `*` 가 있어야
- **추출** → 각 키(`T=`,`H=`,`D=`) 뒤 숫자
- **반환** → 성공 1 / 실패 0
- **`out`** → **성공했을 때만** 채운다 (반쯤 채워진 값 방지)
- **도우미 함수** → `static` (이 파일 전용)

**기본 사용형태**
```c
if (fgets(line, sizeof line, stdin) == NULL) return 1;
line[strcspn(line, "\n")] = '\0';
snprintf(b, sizeof b, "%s", "ABCDEFG");        // "ABC"
char *end; long v = strtol(p, &end, 10);  if (end == p) return 0;
for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) { … }
```

**연결**
- [C D4·2교시 텍스트 쓰기와 CSV 로그](#/c/4/2) — 같은 `fgets`+`strtok`+`strtod` 로 파일 읽기
- [EIO D4·7교시 UART 와 termios](#/eio/4/7) — 시리얼로 들어온 프레임 파싱
- [LINUX D1·8교시 myshell](#/linux/1/8) — `fgets` + `strtok` 로 명령 분리

**연관 문제**
<<QA 6>>

### 7교시 · 동적 메모리

**키워드**: `스택 vs 힙`, `죽은 스택 주소 반환`, `malloc`, `NULL 검사`, `free`, `p = NULL`, `calloc 0 초기화`, `sizeof *p`, `소유권 주석`, `realloc`, `tmp 관용구`, `주소가 바뀔 수 있음`, `용량 두 배`, `valgrind --leak-check=full`, `definitely lost`, `Invalid read/write`, `use-after-free`, `Invalid free 이중 해제`, `strlen + 1`

**개요**: 지금까지의 변수는 모두 **스택**에 있었고 함수가 끝나면 사라졌다. 7교시는 ① 스택으로는 안 되는 두 경우(함수 밖으로 배열을 돌려줘야 할 때, 크기를 실행 중에 정해야 할 때)를 보고, ② **힙**에서 메모리를 빌리는 `malloc` → `NULL` 검사 → 사용 → `free` 사이클을 손에 익히며, ③ 개수를 모르는 입력을 `realloc` 으로 **두 배씩 키우며** 받고, ④ **valgrind** 로 누수·해제 후 사용·이중 해제 세 가지 실수를 잡는다. MCU 에는 valgrind 가 없으니 PC 에서 메시지를 눈에 익혀 두는 것이 목적이다.

**핵심 용어**
- **스택(stack)**: 함수 호출마다 자동으로 생기고 사라지는 메모리. 크기가 컴파일 시간에 정해져야 하고, 함수가 끝나면 무효.
- **힙(heap)**: `malloc` 으로 실행 중에 빌리고 `free` 로 돌려주는 메모리. 직접 해제할 때까지 살아 있다.
- **메모리 누수(memory leak)**: `free` 하지 않아 돌려받지 못한 힙 메모리. 오래 도는 프로그램(서버, 펌웨어)에서 결국 메모리가 바닥난다.
- **해제 후 사용(use-after-free)**: `free` 한 메모리를 다시 읽거나 쓰는 것. 그래서 `free` 후 `p = NULL`.
- **이중 해제(double free)**: 같은 블록을 두 번 `free`. 힙 관리 구조가 깨진다.
- **소유권(ownership)**: "이 메모리를 누가 `free` 하는가" 의 약속. 함수 주석에 적는다.

**새로 나온 함수·명령**

| 함수 | 기본형 | 용도 |
|---|---|---|
| `malloc` | `void *malloc(size_t size);` | size 바이트 할당 (값은 쓰레기). 실패 시 `NULL`. `<stdlib.h>` |
| `calloc` | `void *calloc(size_t n, size_t size);` | n×size 바이트 할당 + **0 으로 초기화**. |
| `realloc` | `void *realloc(void *p, size_t size);` | 크기 변경. **주소가 바뀔 수 있고**, 실패 시 `NULL`(원본은 살아 있음). |
| `free` | `void free(void *p);` | 해제. `free(NULL)` 은 아무 일도 안 함. |
| `valgrind` | `valgrind --leak-check=full ./prog` | 누수·잘못된 접근 검사 (`-g` 로 컴파일하면 줄 번호). |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 7-1 `1_make_array.c` | 개수를 입력받아(크기를 실행 중에 결정) `make_array(n)` 이 `malloc` 으로 만든 배열을 반환("호출자가 free"), 출력 후 `free` + `NULL` |
| 7-2 `2_grow.c` | 개수 모르는 입력을 `cap` 2→4→8 로 `realloc` 하며 저장 (`tmp` 관용구) |
| 7-3 `3_leak.c` (+ uaf, double free) | `free` 누락 → valgrind `definitely lost: 40 bytes`, 해제 후 쓰기·이중 해제 메시지 확인 |

**핵심 내용**

#### 스택 vs 힙 — 언제 malloc 을 쓰나
> 함수가 끝나도 살아 있어야 하거나, 크기를 실행 중에 정해야 하면 힙(`malloc`)
- **스택** → 함수가 끝나면 사라짐. 크기는 컴파일 시간에 정해져야 함
- **힙** → `free` 할 때까지 살아 있음. 크기를 실행 중에 정할 수 있음
- **힙이 필요한 두 경우** → ① 함수 밖으로 배열을 돌려줄 때 ② 개수를 실행 중에 알 때
- ⚠ `int a[n]; return a;` → 함수가 끝나 **죽은 스택 주소**를 돌려줌

#### malloc → free 사이클
> 빌리고(`malloc`) → 확인(`NULL`) → 쓰고 → 돌려주고(`free`) → 지운다(`p = NULL`)
- **순서** → `malloc` → `NULL` 검사 → 사용 → `free` → `p = NULL`
- **크기** → `n * sizeof *a` (타입이 바뀌어도 따라옴)
- **`malloc` vs `calloc`** → `malloc` 은 쓰레기 값 / `calloc(개수, 크기)` 는 **0** 으로 초기화
- **`NULL` 검사** → 생략 금지 (MCU 에서는 정말 실패함)
- **소유권** → 함수 주석에 `// 반환값은 호출자가 free` 를 적는다
- ⚠ 소유권을 안 적으면 → **누수** 또는 **이중 해제** (실제 코드에서 가장 흔한 누수 원인)
- ⚠ 문자열 복사 크기 → `strlen(s) + 1` (널 자리). 빼면 힙 블록 밖에 1바이트

#### realloc 으로 배열 키우기
> 꽉 찬 힙 배열을 더 큰 크기로 바꾼다. 결과는 `tmp` 에 먼저 받는다
- **실패하면** → `NULL` 반환, 원본 블록은 그대로 살아 있음
- **성공해도** → 주소가 바뀔 수 있음 (옛 포인터 무효)
- **규칙** → `tmp` 로 받고, 성공일 때만 `a = tmp`
- **용량** → **두 배씩** 키운다 (매번 +1 이면 복사가 너무 많음)
- ⚠ `a = realloc(a, …)` → 실패 시 원본 주소를 잃어 **누수**
##### 자세히
```text
전          a ──► [25 26 37 25]
성공 (A)    a ──► [25 26 37 25 _ _ _ _]     같은 자리에서 늘어남
성공 (B)    새 곳에 8칸 만들고 복사, 옛 블록 free  → a 의 주소가 바뀜
실패        NULL 반환, 옛 블록은 그대로 살아 있음
```
- `a = realloc(a, …)` 로 쓰면 실패 때 `a = NULL` → 옛 블록의 주소를 아무도 모름 = 누수
- 그래서 `tmp` 에 받아 두면, 실패해도 `free(a)` 로 원본을 정리할 수 있다

#### valgrind 로 메모리 실수 잡기
> PC 에서 누수 · 해제 후 사용 · 이중 해제를 찾아 주는 도구 (MCU 에는 없음)
- **실행** → `-g` 로 컴파일 후 `valgrind --leak-check=full ./prog`
- **`definitely lost`** → 누수 (`free` 안 함)
- **`Invalid read/write … free'd`** → 해제 후 사용 (use-after-free)
- **`Invalid free()`** → 이중 해제
- **위치** → `at malloc … by main (leak.c:5)` = 어디서 할당했는지
- **습관** → `free(p); p = NULL;` (`free(NULL)` 은 아무 일도 안 함)

**기본 사용형태**
```c
int *a = malloc(n * sizeof *a);
if (a == NULL) return NULL;
int *tmp = realloc(a, cap * 2 * sizeof *a);
if (!tmp) { free(a); return 1; }
a = tmp;
free(a); a = NULL;
```
```bash
gcc -Wall -Wextra -g leak.c -o leak && valgrind --leak-check=full ./leak
```

**연결**
- [LINUX D3·2교시 malloc 내부](#/linux/3/2) — `malloc` 이 커널에서 메모리를 받아 오는 방법(brk/mmap)
- [LINUX D3·7교시 메모리 오류 도구](#/linux/3/7) — valgrind·ASan 심화
- [C D4·4교시 링 버퍼](#/c/4/4) — MCU 에서 `realloc` 대신 고정 크기 배열을 쓰는 이유

**연관 문제**
<<QA 7>>

### 8교시 · 종합 — 센서 로그 분석기

**키워드**: `analyzer`, `fgets 루프`, `parse_frame 통합`, `BAD line N`, `continue`, `Reading *log`, `realloc 두 배`, `구조체 swap`, `온도 정렬`, `min max median avg`, `ALERT 35℃`, `n == 0`, `모든 경로 free`, `valgrind 0 bytes`, `100줄 테스트`, `< log.txt 리다이렉션`

**개요**: 오늘 배운 것 전부(안전 입력 → 파싱 → 동적 배열 → 정렬 → 통계 → 해제)를 하나의 프로그램 **센서 로그 분석기**로 묶는다. 표준 입력으로 프레임을 한 줄씩 받아 잘못된 줄은 줄 번호와 함께 `BAD` 로 알리고 건너뛰며, 정상 값은 `realloc` 으로 키우는 배열에 쌓아 온도순으로 정렬한 뒤 최소·최대·중앙값·평균·경보 개수를 출력한다. 제출 조건은 **경고 0개 + valgrind 누수 0**.

**핵심 용어**
- **리다이렉션 `<`**: 셸이 파일을 표준 입력으로 연결 (`./analyzer < log.txt`). 프로그램은 `stdin` 에서 읽을 뿐 파일인지 모른다.
- **구조체 대입**: `Reading t = a[j];` 처럼 구조체는 `=` 로 통째 복사된다 (정렬 swap 에 사용).
- **방어적 처리**: 나쁜 입력에도 죽지 않고 계속 처리, 데이터가 0개여도 통계 계산이 터지지 않게 하는 것.

**새로 나온 함수·명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `sort_by_temp` | `static void sort_by_temp(Reading *a, int n);` | 온도 기준 버블 정렬 (구조체 통째 swap). |
| `./prog < file` | `make && ./analyzer < log.txt` | 파일을 표준 입력으로 넣어 실행. |
| `valgrind … \| grep` | `valgrind --leak-check=full ./analyzer < log.txt 2>&1 \| grep -E "definitely\|ERROR SUMMARY"` | 누수·오류 요약만 확인. |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `08_analyzer/main.c` | 읽기 → `parse_frame` → `realloc` 누적 → 정렬 → 통계 출력 → `free` |
| `08_analyzer/frame.c, frame.h` | 6교시 `parse_frame` 그대로 |
| `08_analyzer/log.txt` | 정상 5줄 + 비정상 2줄 → `BAD line 4`, `BAD line 6`, `min=24.8 max=37.2 median=26.0 avg=29.88 alert=2` |
| `08_analyzer/Makefile` | Day 1 Makefile 에 `-g` 추가 |

**핵심 내용**

#### 분석기 흐름
> 읽기 → 파싱 → 쌓기 → 정렬 → 통계 → 해제
- **읽기** → `fgets` 루프, 빈 줄 무시
- **파싱 실패** → `BAD line N` 출력 후 **`continue`**
- **쌓기** → `Reading *log` 에 **`realloc` 두 배 증가**
- **통계** → 최소 `log[0]`, 최대 `log[n-1]`, 중앙값, 평균, ALERT 개수
- **끝** → `free(log)`
- **Day 3 연결** → 버블 정렬 → `qsort` + 비교 함수 포인터

#### 구조체 배열 다루기
> 구조체는 `=` 로 통째 복사된다
- **할당 크기** → `malloc(cap * sizeof *log)` (타입이 바뀌어도 따라옴)
- **swap** → `Reading t = a[j]; a[j] = a[j+1]; a[j+1] = t;`

#### 방어와 해제
> 데이터가 없어도, 어느 경로로 끝나도 안전하게
- **`n == 0`** → 통계를 건너뛴다 (`log[-1]`, 0 나누기 방지)
- **모든 return 경로에서** → `free`
- **valgrind 목표** → `definitely lost 0 bytes`, `ERROR SUMMARY: 0 errors`

#### 테스트와 제출 체크
> realloc 경로까지 실제로 실행되게 시험한다
- **100줄 입력** → `cap` 을 여러 번 넘겨 realloc 경로 확인
- **제출 체크** → `make clean && make` 성공 · 경고 0 / BAD 줄이 있어도 끝까지 처리 / 100줄 정상 / valgrind 0

**기본 사용형태**
```c
while (fgets(line, sizeof line, stdin)) {
    line[strcspn(line, "\n")] = '\0';
    if (!parse_frame(line, &r)) { printf("BAD line %d: %s\n", lineno, line); bad++; continue; }
    if (n == cap) { cap *= 2; Reading *tmp = realloc(log, cap * sizeof *log); if (!tmp) { free(log); return 1; } log = tmp; }
    log[n++] = r;
}
```

**연결**
- [C D3·6교시 함수 포인터](#/c/3/6) — 버블 정렬을 `qsort` 로 교체
- [C D4·7교시 종합 프로젝트 ①](#/c/4/7) — 같은 흐름을 파일 I/O·링 버퍼로 확장한 `logger`

**연관 문제**
<<QA 8>>
