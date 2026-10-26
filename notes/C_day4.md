## Day 4 — 파일 I/O · 링 버퍼 · 디버깅 · 종합 프로젝트

### 1교시 · 파일 열고 닫기 — FILE*, 모드, 에러

**키워드**: `fopen`, `fclose`, `FILE *`, `NULL 검사`, `perror`, `errno`, `strerror`, 모드 `r w a r+`, `stdin/stdout/stderr`, `argc/argv`, `Too many open files`

**개요**: 지금까지는 셸이 `< temps.txt` 로 파일을 열어 주었지만, 오늘부터는 **프로그램이 직접 파일을 연다**. 1교시는 그 출발점으로 `fopen`/`fclose` 와 `FILE *`, 열기 모드(`r w a r+`), 실패했을 때의 처리(`NULL` 검사, `errno`, `perror`), 명령줄 인자(`argc/argv`)를 다룬다. 핵심은 "열었으면 닫는다, 열기에 실패할 수 있으니 항상 검사한다" 는 `malloc`/`free` 와 같은 규칙이다.

**핵심 용어**
- **파일 스트림(`FILE *`)**: 열린 파일을 가리키는 핸들(손잡이). 읽기·쓰기 함수는 모두 이 핸들로 파일을 지정한다.
- **모드(mode)**: 파일을 어떻게 열지 정하는 문자열 (`"r"` 읽기, `"w"` 쓰기, `"a"` 덧붙이기).
- **표준 스트림**: 프로그램이 시작할 때 이미 열려 있는 `stdin`(입력) / `stdout`(출력) / `stderr`(오류 출력).
- **`errno`**: 마지막 오류의 번호를 담는 전역 변수. 다음 함수 호출이 덮어쓸 수 있다.
- **종료 코드**: `main` 이 돌려주는 값. `0` 정상, 그 외 비정상 (`echo $?`).
- **`argc` / `argv`**: 명령줄 인자의 개수 / 문자열 배열 (`argv[0]` 은 프로그램 이름).

**새로 나온 함수**

| 함수 | 기본형 | 용도 |
|---|---|---|
| `fopen` | `FILE *fopen(const char *path, const char *mode);` | 파일을 연다. **실패하면 `NULL`**. |
| `fclose` | `int fclose(FILE *fp);` | 파일을 닫고 버퍼에 남은 내용을 기록한다. `fopen` 과 짝. |
| `perror` | `void perror(const char *prefix);` | 마지막 오류를 `prefix: 오류 메시지` 로 stderr 에 출력한다. |
| `strerror` | `char *strerror(int errnum);` | 오류 번호를 메시지 문자열로 바꾼다. |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `1_open.c` | `temps.txt` 를 열어 줄 수를 세고 출력 (`10 lines`), 안 열리면 `perror` 로 원인 출력 |
| `2_errno.c` | 없는 파일을 열어 일부러 실패시키고 `errno` 번호와 `strerror`/`perror` 메시지를 출력 |
| `4_noclose.c` | `fclose` 없이 파일을 계속 열어 동시에 열 수 있는 한계(`Too many open files`)를 확인 |
| `headtail.c` | 명령줄 인자로 받은 파일의 **첫 줄과 마지막 줄**을 출력 (`argc` 검사, 없는 파일이면 종료 코드 1) |

**핵심 내용**

#### fopen · fclose
> `FILE *fp = fopen(경로, 모드);` — 실패하면 `NULL`, 다 쓰면 `fclose`
- **실패** → `NULL` → 사용 전에 반드시 검사 (`perror` 로 원인 출력)
- ⚠ 검사 없이 `fgets(…, fp)` → NULL 역참조 → 세그폴트
- **`fclose(fp)`** → `fopen` 과 짝 (`malloc`/`free` 처럼)
- ⚠ 안 닫으면 → 동시 열기 한계(보통 **1024**)에서 `Too many open files`, 버퍼에 남은 내용도 안 써짐
- ⚠ 같은 변수에 `fclose` 없이 다시 `fopen` → 핸들 누수

#### 미리 열린 스트림 세 개
> `stdin` / `stdout` / `stderr` 도 `FILE *`
- **`fgets(line, n, stdin)`** 의 `stdin` 자리에 `fp` 만 넣으면 → Day 2 코드가 그대로 파일에서 읽는다

#### 열기 모드
> 모드가 파일을 지우는지, 뒤에 붙이는지를 정한다
- **`"r"`** → 읽기 (없으면 NULL)
- **`"w"`** → 쓰기 (없으면 생성, **있으면 내용 삭제**)
- **`"a"`** → **덧붙이기** (로그용)
- **`"r+"`** → 읽기+쓰기 (내용 유지) / **`"w+"`** → 내용 삭제
- **`"rb" "wb" "ab"`** → 바이너리 (3교시)
- ⚠ 가장 흔한 사고 → 로그를 `"w"` 로 열어 실행할 때마다 이전 기록이 지워짐 → 로그는 **`"a"`**
- **`"r"` 로 연 파일에 `fprintf`** → 실패(음수 반환), 파일은 그대로

#### errno 와 오류 메시지
> 실패 원인은 전역 변수 `errno` — 실패 직후에 복사해 둔다
- ⚠ 다른 함수(`printf` 등)가 덮어쓸 수 있음 → **`int err = errno;`** 로 바로 복사
- **`strerror(err)`** → 메시지 문자열
- **`perror("접두")`** → `접두: 메시지` 를 stderr 로

#### 명령줄 인자
> `int main(int argc, char *argv[])` — 쓰기 전에 개수 검사
- **`argc < 2` 검사** → 없으면 `argv[1]` 이 NULL
- **파일을 못 열었거나 인자가 틀리면** → **종료 코드 1** (`$?`)

**기본 사용형태**
```c
FILE *fp = fopen("temps.txt", "r");
if (fp == NULL) { perror("fopen temps.txt"); return 1; }
char line[64];
while (fgets(line, sizeof line, fp)) {            // stdin 대신 fp
    line[strcspn(line, "\n")] = '\0';
}
fclose(fp);
FILE *log = fopen("events.log", "a");              // 로그는 "a"
int err = errno; printf("%d: %s\n", err, strerror(err));   // 실패 직후 저장
if (argc < 2) { fprintf(stderr, "usage: %s file\n", argv[0]); return 1; }
```

**연관 문제**
<<QA 1>>

### 2교시 · 텍스트 쓰기와 CSV 로그

**키워드**: `fprintf`, `fputs`, `stdout/stderr 구분`, `버퍼링`, `fflush`, `setvbuf`, `fclose 가 flush`, `CSV`, `헤더 건너뛰기`, `strtok 복사본`, `strtod end`

**개요**: 파일에 **글자(텍스트)** 로 기록하고 다시 읽는 방법. `fprintf`/`fputs` 로 쓰고, 오류는 `stderr` 로 분리하며, **버퍼링**(쓴 것이 바로 디스크에 가지 않는다)과 `fflush` 를 이해한다. 그 위에 센서 로그를 **CSV** 로 저장하고 Day 2 의 `strtok`/`strtod` 로 다시 읽어 평균을 내는 흐름까지 이어진다.

**핵심 용어**
- **버퍼(buffer)**: 데이터를 바로 보내지 않고 잠시 모아 두는 메모리 공간. 파일 쓰기는 보통 4KB 단위로 모았다가 한꺼번에 보낸다.
- **플러시(flush)**: 버퍼에 모인 내용을 실제 파일로 내보내는 것.
- **CSV**: 값을 쉼표로 구분한 텍스트 파일 (`1,25.3,IDLE`). 첫 줄은 보통 **헤더**(열 이름).
- **구분자(delimiter)**: 값과 값을 나누는 글자 (`,`).
- **줄 단위 버퍼링**: 줄바꿈(`\n`)이 나올 때마다 flush 하는 방식.

**새로 나온 함수**

| 함수 | 기본형 | 용도 |
|---|---|---|
| `fprintf` | `int fprintf(FILE *fp, const char *fmt, ...);` | `printf` 와 같은 서식으로 **파일에** 쓴다 (`printf` = `fprintf(stdout, …)`). |
| `fputs` | `int fputs(const char *s, FILE *fp);` | 서식 없이 문자열만 파일에 쓴다. 개행을 붙이지 않는다. |
| `fflush` | `int fflush(FILE *fp);` | 버퍼의 내용을 **지금 즉시** 파일로 내보낸다. |
| `setvbuf` | `int setvbuf(FILE *fp, char *buf, int mode, size_t size);` | 버퍼링 방식을 정한다 (`_IOLBF` = 줄 단위). |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `1_csv_write.c` | `log.csv` 에 헤더와 2행을 `fprintf`/`fputs` 로 쓴 뒤 다시 열어 화면에 출력 |
| `2_buffer.c` | `fprintf` → `fflush` → `fclose` 각 시점에서 **디스크 위의 파일 크기**를 재어 버퍼링을 눈으로 확인 (0 → 15 → 27) |
| `3_csv_read.c` | `log.csv` 를 읽어 헤더를 건너뛰고 행마다 `strtok`/`strtol`/`strtod` 로 파싱, 평균 온도를 계산 (잘못된 행은 stderr) |

**핵심 내용**

#### 파일에 쓰기
> `printf(...)` ≡ `fprintf(stdout, ...)` — 첫 인자만 `fp` 로 바꾸면 파일
- **서식 없이 문자열만** → `fputs(str, fp)` (`%` 가 있어도 안전)
- **`puts`** → stdout 전용, 개행을 붙임
- **오류 메시지는 `stderr`** → `./app > out.txt` 로 stdout 을 담아도 오류는 화면에 남는다
- **따로 담기** → `2>err.log`

#### 버퍼링과 fflush
> `fprintf` 는 메모리 버퍼에 모았다가 나중에 쓴다
- **실제로 쓰는 때** → 버퍼(보통 **4KB**)가 차거나 **`fflush` / `fclose`**
- ⚠ 쓴 직후 다른 터미널에서 `cat` → **비어 있음**
- ⚠ 프로그램이 죽으면 → 버퍼 내용은 **사라짐**
- **확인 실험** → `fprintf` 직후 디스크 크기 0 → `fflush` 후 15 → `fclose` 후 27
- **줄마다 확실히 남기려면** → `fflush(fp)` 또는 `setvbuf(fp, NULL, _IOLBF, 0)` (줄 단위, 성능은 조금 ↓)
- **매 호출마다 `fopen("a")`+`fclose`** → 느리지만 확실히 기록, 핸들 누수 없음 (자주 부르면 한 번 열어 두고 `fflush` 만)

#### CSV 쓰고 읽기
> 사람이 읽을 수 있고 Excel·Python 이 바로 연다 — 대신 크고 파싱이 필요
- **단점** → `25.3` = 4글자 + 구분자 → 크기 ↑, 파싱 필요 (3교시 바이너리의 이유)
- **헤더 줄** → `fgets` 한 번 읽고 버린다
- **`strtok`** → 원본을 고치므로 `snprintf` 로 **복사본**을 자른다
- **숫자 변환** → `strtol`/`strtod` 의 `end == 시작` 이면 실패 (`atof` 는 실패를 모름)

**기본 사용형태**
```c
FILE *out = fopen("log.csv", "w");
fprintf(out, "seq,temp,state\n");
fprintf(out, "%d,%.1f,%s\n", 1, 25.3, "IDLE");     // printf 와 같은 서식
fputs("2,17.0,HEAT\n", out);                       // 서식 없이
fflush(out);                                       // 지금 실제로 쓰기
fclose(out);                                       // fclose 도 flush
fprintf(stderr, "bad row: %s", line);              // 오류는 stderr
fgets(line, sizeof line, in);                      // 헤더 버리기
char buf[64]; snprintf(buf, sizeof buf, "%s", line);
char *a = strtok(buf, ","); double t = strtod(b, &end); if (end == b) return 0;
```

**연관 문제**
<<QA 2>>

### 3교시 · 바이너리 I/O — 구조체 통째로, fseek

**키워드**: `fwrite`, `fread`, `"wb" "rb" "r+b"`, `fseek`, `ftell`, `rewind`, `SEEK_SET/SEEK_END`, `파일 크기 재기`, `n번째 레코드`, `od`, `패딩·엔디안`, `직렬화`

**개요**: 텍스트 대신 **메모리 모양 그대로(바이트)** 저장하는 방법. `fwrite`/`fread` 로 구조체 배열을 통째로 쓰고 읽으며, `fseek`/`ftell` 로 **파일 크기를 재거나 n 번째 레코드로 바로 점프**한다. 텍스트(CSV)보다 파싱이 필요 없고 정확하지만, 다른 컴퓨터에서는 **패딩·엔디안** 때문에 안 읽힐 수 있다는 한계도 함께 배운다.

**핵심 용어**
- **바이너리 파일**: 글자가 아니라 메모리의 바이트를 그대로 담은 파일.
- **레코드(record)**: 파일 안에서 같은 크기로 반복되는 한 묶음 (여기서는 `Reading` 하나 = 16바이트).
- **파일 위치 지시자(커서)**: 다음에 읽거나 쓸 위치. `fread`/`fwrite` 가 지나간 만큼 앞으로 간다.
- **오프셋(offset)**: 파일 처음(또는 기준점)에서 몇 바이트 떨어졌는지.
- **EOF(End Of File)**: 파일 끝. 더 읽을 것이 없는 상태.
- **직렬화(serialization)**: 구조체를 정해진 바이트 순서로 풀어 저장·전송하는 것.

**새로 나온 함수**

| 함수 | 기본형 | 용도 |
|---|---|---|
| `fwrite` | `size_t fwrite(const void *p, size_t size, size_t n, FILE *fp);` | 메모리의 `size × n` 바이트를 파일에 쓴다. **성공한 개수**를 돌려준다. |
| `fread` | `size_t fread(void *p, size_t size, size_t n, FILE *fp);` | 파일에서 `size × n` 바이트를 읽어 메모리에 담는다. **읽은 개수**를 돌려준다. |
| `fseek` | `int fseek(FILE *fp, long offset, int whence);` | 커서를 옮긴다. `whence` = `SEEK_SET`(처음) / `SEEK_CUR`(현재) / `SEEK_END`(끝). 성공 `0`. |
| `ftell` | `long ftell(FILE *fp);` | 현재 커서 위치(바이트)를 돌려준다. 끝에서 부르면 **파일 크기**. |
| `rewind` | `void rewind(FILE *fp);` | 커서를 파일 처음으로 되돌린다 (`fseek(fp, 0, SEEK_SET)` 와 같음). |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `1_binio.c` | `Reading` 3개를 `log.bin` 에 `fwrite` → 파일 크기(48바이트)와 레코드 수 계산 → 두 번째 레코드로 `fseek` 후 `fread` |
| `3_load.c` | `load_bin` 으로 파일 전체를 `malloc` 한 배열로 읽고, `patch_temp` 로 **한 레코드의 온도만** 고쳐 쓴 뒤 다시 읽어 확인 |

**핵심 내용**

#### fwrite · fread
> 메모리를 변환 없이 그대로 쓰고 읽는다 — `qsort` 와 같은 모양
- **`fwrite(주소, 요소크기, 개수, fp)`** / **`fread(주소, 요소크기, 개수, fp)`**
- **반환값** → **성공한 요소 개수** → `fread(...) == 1` 처럼 **반드시 검사** (EOF·오류면 적게 돌아오고 변수는 쓰레기)
- **`Reading` 3개 `fwrite`** → **48바이트** (16×3, 패딩 포함)
- **`od -A d -t x1`** → 첫 8바이트 `cd cc cc cc cc 4c 39 40` = `double 25.3` (IEEE 754 리틀 엔디안)
- **`"b"`** → Linux 는 차이 없지만 Windows 의 `\n ↔ \r\n` 변환을 막는 이식성 습관

#### 바이너리 vs 텍스트
> 바이너리는 파싱이 없고 정확도 손실이 없다
- **텍스트 `25.3`** → 4글자 + `strtod` 파싱 필요
- **크기를 미리 안다** → **한 번의 `malloc`** 으로 읽음 (realloc 루프 불필요)
- ⚠ 구조체를 통째로 쓴 파일 → **같은 컴파일러 · 같은 CPU** 에서만 안전
- **다른 장치에서 안 읽히는 원인** → **패딩(정렬)** 과 **엔디안**
- **장치 간 교환** → 필드별로 바이트 순서를 정해 쓰고 읽는다 (직렬화, Day 3 바이트 조립/분해)

#### fseek · ftell 로 위치 다루기
> 파일 안 위치를 옮겨 필요한 레코드만 읽고 고친다
- **파일 크기** → `fseek(fp, 0, SEEK_END); long size = ftell(fp); rewind(fp);`
- **레코드 수** → `size / sizeof(레코드)`
- **n 번째 레코드** → `fseek(fp, n * sizeof(Reading), SEEK_SET)` (0부터, 두 번째는 1)
- **일부만 고쳐 쓰기** → `"r+b"` → `fseek` → `fread` → 수정 → **다시 `fseek`** → `fwrite`
- ⚠ `"w+b"` → 파일이 지워짐
- **다시 `fseek` 하는 이유** → 읽고 나면 커서가 다음 레코드에 가 있고, 읽기↔쓰기 전환 사이에는 `fseek`/`fflush` 가 표준 규칙

#### load_bin 정리 규칙
> 실패 경로마다 닫고 해제한다
- **실패 경로마다** → `fclose`/`free` 를 빠뜨리지 않는 것이 `load_bin` 의 절반
- **반환값** → 호출자가 `free`

**기본 사용형태**
```c
FILE *fp = fopen("log.bin", "wb");
fwrite(r, sizeof r[0], 3, fp);                      // (주소, 요소 크기, 개수, 파일)
fclose(fp);

fp = fopen("log.bin", "rb");
fseek(fp, 0, SEEK_END);  long size = ftell(fp);  rewind(fp);     // 파일 크기
int n = (int)(size / sizeof(Reading));
Reading *a = malloc(n * sizeof *a);
if (fread(a, sizeof *a, n, fp) != (size_t)n) { free(a); fclose(fp); return NULL; }

fseek(fp, idx * (long)sizeof r, SEEK_SET);          // idx 번째 레코드로 점프
FILE *fp2 = fopen("log.bin", "r+b");                // 읽기+쓰기, 내용 유지
```

**연관 문제**
<<QA 3>>

### 4교시 · 링 버퍼 — 고정 메모리로 흐르는 데이터 받기

**키워드**: `링 버퍼`, `circular buffer`, `head`, `tail`, `count`, `% RING_CAP`, `덮어쓰기 vs 거부`, `이동평균`, `volatile`, `고정 크기 배열`

**개요**: 센서는 초당 100번 값을 보내는데 로그는 1초에 한 번만 쓴다면, 그 사이의 값을 어디에 두는가? `realloc` 은 MCU 에서 위험하고, 배열을 매번 앞으로 당기는 것은 느리다. **고정 크기 배열을 원형으로 돌려 쓰는 링 버퍼**는 할당도 이동도 없다. 4교시는 `head`/`tail`/`count` 로 링 버퍼를 직접 구현하고, 가득 찼을 때의 정책(덮어쓰기 vs 거부)과 Day 1 이동평균을 링 버퍼로 다시 만드는 것을 다룬다.

**핵심 용어**
- **링 버퍼(ring / circular buffer)**: 고정 크기 배열의 끝과 처음을 이어 **원처럼** 돌려 쓰는 큐. 회전 초밥 레일처럼 끝에 닿으면 처음으로 돌아온다.
- **head**: 다음에 **쓸** 자리. **tail**: 다음에 **읽을** 자리. **count**: 지금 들어 있는 개수.
- **push / pop**: 넣기(head 에 쓰고 전진) / 꺼내기(tail 에서 읽고 전진). 가장 먼저 넣은 것이 먼저 나온다(FIFO).
- **감김(wrap-around)**: 인덱스가 끝을 넘으면 0 으로 돌아가는 것. `% RING_CAP` 로 만든다.
- **덮어쓰기(overwrite) 정책**: 가득 찼을 때 가장 오래된 값을 버리고 새 값을 넣는다. **거부(drop) 정책**: 가득 찼으면 새 값을 거부하고 실패를 알린다.
- **이동평균(moving average)**: 최근 N 개 값의 평균. 튀는 값에 완만하게 반응한다.

**이 교시에서 만드는 함수** (표준 함수가 아니라 직접 만드는 링 버퍼 API)

| 함수 | 기본형 | 용도 |
|---|---|---|
| `ring_init` | `void ring_init(Ring *r);` | `head`, `tail`, `count` 를 0 으로 초기화 |
| `ring_push` | `void ring_push(Ring *r, double v);` | 값을 넣는다. 가득 차면 **가장 오래된 값을 버린다**(덮어쓰기) |
| `ring_pop` | `bool ring_pop(Ring *r, double *out);` | 가장 오래된 값을 `*out` 으로 꺼낸다. 비었으면 `false` |
| `ring_avg` | `double ring_avg(const Ring *r);` | 들어 있는 값들의 평균 (`tail` 부터 `count` 개) |
| `ring_push_nodrop` | `bool ring_push_nodrop(Ring *r, double v);` | 가득 차면 **거부**하고 `false` (거부 정책) |
| `ring_max` | `double ring_max(const Ring *r);` | 들어 있는 값 중 최댓값 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `ring.h` / `ring.c` | `Ring` 구조체(`double buf[4]` + `head`/`tail`/`count`)와 위 함수들의 구현 |
| `ring_test.c` | 온도 8개(`25 25 25 25 37 25 25 25`)를 차례로 `push` 하며 `head`/`tail`/`count`/평균 변화를 출력하고, 끝에서 모두 `pop` |
| `nodrop_test.c` | 거부 정책 버전으로 6개를 넣어 5번째부터 `거부(가득)` 이 나오는 것과 `ring_max` 를 확인 |

**핵심 내용**

#### 링 버퍼 구조
> 고정 배열을 원형으로 — 할당(`realloc`)도 이동도 없어 MCU 에 맞다
- **쓰는 곳** → 센서 값처럼 계속 흘러오는 데이터를 고정 메모리로
- **`head`** → 다음에 **쓸** 자리 / **`tail`** → 다음에 **읽을** 자리 / **`count`** → 개수
- **push** → `head` 에 쓰고 전진 / **pop** → `tail` 에서 읽고 전진
- **감김** → `head = (head + 1) % RING_CAP`
- **`RING_CAP` 이 2 의 거듭제곱이면** → `& (RING_CAP - 1)` (더 빠름)

#### 가득 찼을 때와 평균
> 덮어쓰기 정책은 가장 오래된 값을 버린다
- **덮어쓰기** → `tail` 도 한 칸 전진(가장 오래된 값 버림, `count--`) 후 `head` 에 쓰기
- **실행 결과** → 5번째 push(37)에서 `tail=1`, `count` 는 4 유지
- **평균** → `tail` 부터 `count` 개를 `buf[(tail + i) % RING_CAP]` 로 순회 (끝을 넘어 감길 수 있어서)
- **`25 25 25 37` 평균** → **28.00** — 튀는 값 하나가 평균을 완만하게 올린다
- **정책은 용도가 정한다** → 센서 표시는 최신 우선(덮어쓰기), 이벤트 로그는 과거 보존(거부 + 경고)

#### 빈 것과 가득 찬 것 구분
> `head == tail` 만으로는 둘을 구분할 수 없다
- ⚠ `count` 없이 `head`/`tail` 만 → `head == tail` 이 **비었음과 가득 참 둘 다**
- **해법 ①** → `count` 필드 (오늘 구현)
- **해법 ②** → 한 칸 비워 두기 (CAP-1 개만 저장)
- **해법 ③** → `head`/`tail` 을 감지 않고 접근할 때만 `%`
- **인터럽트 push + 메인 pop** (다음 주) → `count` 를 두 쪽이 동시에 고치면 깨짐 → 한 칸 비워 두기 + `head`/`tail` 을 **`volatile`**

#### 구조체 = C 의 객체
> 상태를 구조체에 담고 포인터로 넘기면 여러 개를 만들 수 있다
- **Day 1 `moving_avg` 의 `static` 배열** → 함수당 상태 하나뿐
- **`Ring` (구조체 + 포인터 인자)** → 온도용 · 습도용 **여러 개**, 단독 테스트도 가능

**기본 사용형태**
```c
typedef struct { double buf[RING_CAP]; int head, tail, count; } Ring;
void ring_push(Ring *r, double v) {
    if (r->count == RING_CAP) { r->tail = (r->tail + 1) % RING_CAP; r->count--; }   // 가득: 오래된 것 버림
    r->buf[r->head] = v;
    r->head = (r->head + 1) % RING_CAP;
    r->count++;
}
bool ring_pop(Ring *r, double *out) {
    if (r->count == 0) return false;
    *out = r->buf[r->tail];  r->tail = (r->tail + 1) % RING_CAP;  r->count--;  return true;
}
s += r->buf[(r->tail + i) % RING_CAP];     // tail 부터 count 개 순회
```

**연관 문제**
<<QA 4>>

### 5교시 · 디버깅 — gdb, assert, sanitizer

**키워드**: `gdb`, `-g -O0`, `break`, `run`, `next`, `step`, `print`, `bt`, `finish`, `watch`, `조건부 중단점`, `x/16xb`, `set var`, `assert`, `NDEBUG`, `AddressSanitizer`, `-fsanitize=address`, `valgrind 비교`

**개요**: 지금까지의 버그는 컴파일러 경고가 알려 주거나 세그폴트로 요란하게 죽었다. 하지만 **경고도 없고 죽지도 않으면서 값만 틀리는 버그**는 코드를 눈으로 훑어서는 찾기 어렵다. 5교시는 프로그램을 **한 줄씩 실행하며 변수를 직접 들여다보는 디버거 `gdb`**, 버그를 일찍 터뜨리는 **`assert`**, 메모리 오류를 빠르게 잡는 **AddressSanitizer** 세 가지 도구를 익힌다.

**핵심 용어**
- **디버거(debugger)**: 프로그램을 멈추고 한 줄씩 실행하면서 변수·메모리를 보여 주는 도구 (`gdb`). 비디오를 일시정지하고 프레임을 하나씩 넘겨 보는 것과 같다.
- **중단점(breakpoint)**: 실행을 멈추게 표시해 둔 위치. **조건부 중단점**은 조건이 참일 때만 멈춘다.
- **감시점(watchpoint)**: 특정 변수의 **값이 바뀌는 순간** 멈추게 하는 장치 (`watch`).
- **스택 트레이스(backtrace)**: 지금 이 함수를 누가 불렀는지 호출 경로 (`bt`). 세그폴트 때 가장 먼저 본다.
- **어서션(assert)**: "여기서는 이 조건이 반드시 참" 이라고 코드에 적어 두는 계약. 위반하면 즉시 중단.
- **새니타이저(sanitizer)**: 컴파일할 때 검사 코드를 끼워 넣어 실행 중 메모리 오류를 잡는 도구 (`-fsanitize=address`).

**새로 나온 명령 · 옵션**

| 명령 / 옵션 | 기본형 | 용도 |
|---|---|---|
| 컴파일 | `gcc -g -O0 app.c -o app` | `-g` 변수 이름·줄 번호 정보, `-O0` 최적화 끔 (안 끄면 `next` 가 줄을 건너뜀) |
| `break` | `break 함수` / `break 파일:줄` / `break 줄 if 조건` | 중단점 설정 (`tbreak` = 한 번만) |
| `run` · `continue` | `run` / `continue` | 실행 시작 / 멈춘 뒤 이어서 실행 (`start` = main 에서 멈춤) |
| `next` · `step` | `next` / `step` | 한 줄 실행 — `next` 는 함수 호출을 **건너뜀**, `step` 은 함수 **안으로 들어감** |
| `print` | `print 식` · `print 배열@개수` · `print/x 값` | 값 보기 (`/x` 16진, `/t` 2진) |
| `bt` | `bt` · `bt N` | 호출 경로 (어느 함수의 몇 번째 줄인지) |
| `finish` | `finish` | 현재 함수를 끝까지 실행하고 **반환값** 확인 |
| `watch` | `watch 변수` | 변수 값이 바뀌는 순간 멈춤 (Old/New 값 표시) |
| `x` | `x/16xb &변수` · `x/s 주소` | 메모리를 바이트 단위(16진) / 문자열로 보기 |
| `set var` · `call` | `set var 변수 = 값` · `call 함수(인자)` | 실행 중 값을 바꿔 보거나 함수를 직접 호출 |
| `frame` · `up` · `down` | `frame N` · `up` · `down` | 호출 스택의 다른 층으로 이동해 그 층의 변수 보기 |
| `assert` | `#include <assert.h>` · `assert(조건);` | 조건이 거짓이면 `파일:줄:함수:조건` 을 출력하고 `abort` |
| `-DNDEBUG` | `gcc -DNDEBUG` | `assert` 를 **통째로 제거** (릴리스 빌드) |
| `-fsanitize=address` | `gcc -g -fsanitize=address` | 배열 밖 접근·해제 후 사용 등을 실행 중 잡아냄 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `avg.c` | 네 온도 `{24,26,25,27}` 의 평균(기대 25.50)을 구하는데 **19.50** 이 나오는 버그 코드 — `for (int i = 1; …)` 가 첫 값을 건너뜀. gdb 로 찾는 연습용 |
| `assert_demo.c` | 빈 배열로 `median_first` 를 부르면 `assert` 가 중단시키고, `-DNDEBUG` 로 빌드하면 `assert` 가 (부수효과가 있는 식까지) 사라지는 것을 확인 |
| `asan_demo.c` | `malloc(4 * sizeof(int))` 의 **4번째 칸 다음(`p[4]`)에 쓰는 힙 오버플로** — 일반 빌드는 조용히 지나가고 ASan 빌드는 줄 번호와 함께 잡아냄 |

**핵심 내용**

#### gdb 준비와 첫 흐름
> `-g -O0` 로 빌드하고, 예상과 다른 첫 순간을 찾는다
- **`-g`** → 변수 이름 · 줄 번호 / **`-O0`** → 코드 순서가 소스와 같음
- ⚠ `-O2` → `next` 가 여러 줄을 건너뜀
- **흐름** → `break average` → `run` → `print n`, `print a[0]@n` (입력 확정) → `next` → `print i` 에서 이상한 첫 순간 → `bt` → `finish` (반환값 19.5)
- **avg.c 원인** → `int i = 1` → `int i = 0`
- **`next`** → 함수 건너뜀 / **`step`** → 함수 안으로 (`on_heat` 내부를 볼 때)
- **세그폴트** → 가장 먼저 **`bt`** — 한 줄로 `gdb -batch -ex run -ex bt ./app` (Makefile `debug` 타깃)

#### watch 와 조건부 중단점
> 어디서 멈출지 모를 때 값이 바뀌는 순간 · 조건이 맞는 순간에 멈춘다
- **`watch 변수`** → "누가 이 변수를 바꿨지?" — 배열 밖 쓰기로 **엉뚱한 변수가 조용히 바뀌는** 버그에 강력
- **avg.c 확인** → `sum` 의 첫 변화가 0→26 (24 가 아니라 26 이 더해짐)
- **`break 34 if c > 30`** → 반복이 1000번이어도 한 번에 도착
- **조건 바꾸기** → `condition N 식` / **건너뛰기** → `ignore N 횟수`
- **논리 버그** (TABLE 순서 어긋남) → `print TABLE[cur]` 로 "HEAT 인데 `on_cool`" 을 실행 중에 확인하는 수밖에 없음

#### assert
> 전제 조건을 코드로 적는 계약 — 위반 순간 `파일:줄:함수` 와 조건을 출력하고 중단
- **장점** → 원인과 가까운 곳에서 멈춘다
- **`-DNDEBUG`** → `assert` 가 **사라진다**
- ⚠ `assert(ring_pop(&r, &v))` 같은 **부수효과 있는 식 금지** (기본 빌드 `calls = 1`, NDEBUG 빌드 `calls = 0`)
- **해법** → 결과를 변수에 받고 그 변수를 `assert`

#### AddressSanitizer
> 컴파일 때 검사 코드를 넣어 메모리 오류를 줄 번호와 함께 잡는다
- **잡는 것** → 배열 밖 접근, 해제 후 사용, 힙 오버플로
- **일반 빌드는** → 같은 코드가 아무 일 없이 `7` 출력, 종료 코드 0
- ⚠ 못 잡는 것 → 논리 버그 (테이블 순서 어긋남) → gdb 나 테스트로
- **ASan** → 재컴파일 필요, 빠름 (**2~3배**) / **valgrind** → 재컴파일 없음, 느림 (약 **20배**)
- **사용** → 개발 중에는 ASan, 릴리스에서는 뺀다 (`make asan` 타깃)

**기본 사용형태**
```bash
gcc -Wall -Wextra -g -O0 avg.c -o avg
gdb -q ./avg
(gdb) break average
(gdb) run
(gdb) print a[0]@n        # 포인터가 가리키는 배열을 n 개 보기
(gdb) next                # 한 줄 실행
(gdb) watch sum           # sum 이 바뀔 때마다 멈춤
(gdb) bt                  # 호출 경로
(gdb) finish              # 반환값 확인
gdb -batch -ex run -ex bt ./wild                    # 세그폴트 위치 한 줄로
gcc -g -fsanitize=address app.c -o app && ./app     # 메모리 오류 검출
gcc -DNDEBUG app.c -o app                           # assert 제거 (릴리스)
```
```c
#include <assert.h>
assert(n > 0 && "median of empty array");     // 전제 조건
bool ok = ring_pop(&r, &v); assert(ok);        // 부수효과 있는 식은 밖으로 빼서
```

**연관 문제**
<<QA 5>>

### 6교시 · 코드 품질 — 모듈 구조, 에러 규약, 테스트 타깃

**키워드**: `모듈`, `.h = 계약`, `static 으로 숨기기`, `한 방향 의존`, `main 은 조립`, `에러 규약 A/B/C`, `esp_err_t`, `단위 테스트`, `test_ring.c`, `Makefile 자동 변수 $@ $^ $<`, `패턴 규칙 %.o: %.c`, `타깃별 변수`, `.PHONY`, `make test/asan/debug`

**개요**: 4일 동안 만든 코드(`frame`, `ring`, `fsm`, `gpio`, `logger`)를 **한 프로젝트로 묶을 때** 어떻게 나누고 어떻게 믿을 수 있게 만드는가. 6교시는 코드를 **모듈(`.h` 계약 + `.c` 구현)** 로 나누는 기준, 함수들의 **에러 반환 규약**을 하나로 통일하는 법, 그리고 `Makefile` 에 **`test`/`asan`/`debug` 타깃**을 만들어 버튼 하나로 검증하는 법을 다룬다.

**핵심 용어**
- **모듈(module)**: 하나의 책임을 맡은 `.h`(계약) + `.c`(구현) 쌍 (`ring.h/ring.c`).
- **계약(contract)**: 헤더에 적은 함수 원형·구조체 — 다른 모듈이 알아야 할 것만.
- **의존(dependency)**: 한 모듈이 다른 모듈의 헤더를 쓰는 관계. **한 방향**이어야 한다.
- **에러 규약**: 함수가 실패를 알리는 약속(반환값 모양). A: 성공 1/실패 0, B: 성공 0/실패 음수 코드, C: 포인터 반환/실패 `NULL`.
- **단위 테스트(unit test)**: `main` 없이 모듈 하나만 따로 시험하는 작은 프로그램.
- **자동 변수**: Makefile 이 규칙마다 채워 주는 변수 — `$@`(타깃), `$^`(의존 전체), `$<`(첫 의존).
- **패턴 규칙**: `%.o: %.c` — 모든 `.c` 를 같은 명령으로 `.o` 로 만드는 규칙.
- **가짜 타깃(`.PHONY`)**: 파일 이름이 아니라 명령 이름인 타깃 (`test`, `clean`).

**새로 나온 것** (Makefile 문법과 구조)

| 항목 | 기본형 | 용도 |
|---|---|---|
| 패턴 규칙 | `%.o : %.c` ⏎ `<탭>$(CC) $(CFLAGS) -c $<` | 모든 `.c` → `.o` 를 한 규칙으로 (Day 1 규칙 세 개를 대신) |
| 자동 변수 | `$@` `$^` `$<` | 타깃 / 의존 전체 / 첫 의존 — 파일 이름 반복 제거 |
| 타깃별 변수 | `asan : CFLAGS += -fsanitize=address` | 그 타깃을 만들 때만 옵션 추가 (`make asan` = ASan 빌드, `make` = 일반 빌드) |
| 가짜 타깃 | `.PHONY : test asan debug clean` | 같은 이름의 파일이 있어도 규칙을 항상 실행 |
| 헤더 의존성 | `main.o : frame.h ring.h logger.h` | 패턴 규칙은 헤더를 모르므로 따로 적는다 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| `ring.h/c`, `frame.h/c` | 4교시 링 버퍼 / Day 2 프레임 파싱 모듈 (그대로 재사용) |
| `logger.h/c` | CSV 로거: `log_open`(`"a"` 로 열고 새 파일이면 헤더), `log_row`(한 줄 쓰고 `fflush`), `log_close`(`fp = NULL`) |
| `main.c` | **조립만**: `temps.txt` 를 읽어 `parse_frame` → `ring_push` → `log_row`, 마지막에 개수와 최근 평균 출력 (로직 없음) |
| `test_ring.c` | `main` 없이 링 버퍼만 `assert` 로 검증 (빈 pop → `false`, 가득 유지, 가장 오래된 값 버려짐) |
| `Makefile` | `make`(빌드) / `make test` / `make asan` / `make debug` / `make clean` |

**핵심 내용**

#### 모듈 나누는 기준
> 헤더에는 남이 알아야 할 것만, 의존은 한 방향, `main` 은 조립만
- **① 헤더** → 구조체 · 함수 원형 · 매크로만, 구현 세부는 `.c` 의 `static`
- **② 한 방향 의존** → `ring.c` 는 `frame.h` 를 모른다
- **③ `main` 은 조립만** → 모듈만 바꿔 끼워 재사용, 테스트도 모듈 단위
- ⚠ `ring` 이 `frame` 을 알면 → 의존이 양방향 → `frame` 수정 시 `ring` 도 재컴파일, 단독 테스트 · 재사용 불가

#### 에러 규약
> 한 프로젝트에 한 가지 — 섞으면 호출자가 매번 헤더를 봐야 한다
- **A: 성공 1 / 실패 0** → `parse_frame`, `ring_pop`, `log_open` (오늘 프로젝트)
- **B: 성공 0 / 실패 음수 코드** → MCU SDK 의 **`esp_err_t`**
- **C: 실패 `NULL`** → `malloc`, `fopen`, `load_bin`

#### 단위 테스트와 Makefile 도구
> 테스트가 통과하지 않으면 `main` 을 건드리지 않는다
- **단위 테스트** → `test_ring.c` + `make test`
- **`%.o: %.c`** → 한 줄이 규칙 세 개를 대신
- **`$@ $^ $<`** → 이름 반복 제거
- **`asan: CFLAGS += -fsanitize=address`** → 타깃별 변수
- **`asan: clean logger`** → 옵션이 바뀌었으니 처음부터 다시
- **`.PHONY`** → `clean` 이라는 파일이 우연히 있으면 `make clean` 이 "up to date" 로 아무것도 안 함

#### 로거 모듈 습관
> 닫은 핸들은 NULL, 헤더는 새 파일일 때만
- **`log_close` 에서 `fp = NULL`** → 닫힌 `FILE *` 를 다시 `fclose` 하면 정의되지 않은 동작 (Day 2 `free` 뒤 `NULL` 과 같은 습관)
- **헤더 한 번만** → `"a"` 로 열고 `fseek(END)` 후 `ftell(fp) == 0` (새 파일) 이면 헤더
- **확인** → 두 번 실행 후 21줄 = 헤더 1 + 데이터 20

**기본 사용형태**
```make
CC = gcc
CFLAGS = -Wall -Wextra -g -O0
OBJS = main.o frame.o ring.o logger.o
logger : $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^
test_ring : test_ring.o ring.o
	$(CC) $(CFLAGS) -o $@ $^
test : test_ring
	./test_ring
asan : CFLAGS += -fsanitize=address
asan : clean logger
debug : logger
	gdb -batch -ex run -ex bt --args ./logger temps.txt
%.o : %.c
	$(CC) $(CFLAGS) -c $<
main.o : frame.h ring.h logger.h
.PHONY : test asan debug clean
```
```c
assert(ring_pop(&r, &v) == false);              // test_ring.c
if (lg->fp) { fclose(lg->fp); lg->fp = NULL; }   // 두 번 닫아도 안전
```

**연관 문제**
<<QA 6>>

### 7교시 · 종합 프로젝트 ① — 설계와 구현 (센서 데이터 수집기 `logger`)

**키워드**: `수집기 파이프라인`, `이동평균 → 상태 머신`, `오경보 방지`, `events.log "a"`, `readings.csv "w"`, `readings.bin fwrite`, `log_open 모드 인자`, `조기 종료 정리`, `usage + 종료 코드 1`, `extern 정의 한 곳`

**개요**: 4일치 모듈을 하나의 프로그램 `./logger temps.txt` 로 묶는 **설계와 구현**. 파일에서 프레임을 읽어 **파싱(Day 2) → 링 버퍼 이동평균(Day 4) → 상태 머신(Day 3) → 전이 이력과 CSV/바이너리 기록(Day 4)** 으로 흘려보내고 통계를 출력한다. 7교시의 핵심은 코드 한 줄 한 줄이 아니라 **각 단계의 설계 결정**(무엇을 상태 머신에 넘기는가, 파일을 어떤 모드로 여는가, 실패하면 어떻게 정리하는가)이다.

**핵심 용어**
- **파이프라인(pipeline)**: 데이터가 단계를 차례로 통과하는 구조 (`파일 → 파싱 → 링 버퍼 → 필터 → 상태 머신 → 로그`).
- **필터(filter)**: 튀는 값을 걸러 내는 처리 (여기서는 링 버퍼 이동평균).
- **전이 이력(event log)**: 상태가 바뀔 때마다 한 줄 남기는 기록 (`seq,avg,from,to`).
- **조기 종료 경로(early return)**: 오류로 함수 중간에서 빠져나가는 경로 — 여기서도 열린 자원을 정리해야 한다.

**예제 코드가 하는 일** ([07_project/](C_day4/07_project/) — 직접 조립해 강의의 기대 출력과 일치함을 확인)

| 파일 | 하는 일 |
|---|---|
| `main.c` | 입력 파일을 읽어 파이프라인을 돌리고 통계 출력 (`rows=10 bad=0 events=1`, 상태별 횟수) |
| `frame.h/c` | `parse_frame` (Day 2) — BAD 프레임이면 0 |
| `ring.h/c` | 이동평균용 링 버퍼 (`RING_CAP = 4`) |
| `fsm.h/c` | 상태 머신 (`fsm_step` = `TABLE[cur](t)`), 핀은 `gpio.h` 매크로로만 제어 |
| `logger.h/c` | CSV 로거 — `log_open(&lg, path, mode)` 로 **모드를 인자로** 받는다 |
| `Makefile` | `make` / `make test` / `make asan` / `make clean` |

**요구 사항과 설계 결정**

| 요구 | 설계 / 이유 |
|---|---|
| 인자 없음/파일 못 염 | `usage` 를 stderr 로 출력하고 **종료 코드 1** (`./logger; echo $?` → 1) |
| BAD 프레임 | **stderr** 에 줄 번호와 함께 출력, `bad++`, **`continue`** (계속 진행). `2>/dev/null` 로 걸러짐 |
| 이동평균을 `fsm_step` 에 | 원값이 아니라 **평균**을 넣어 37.2 하나에 ALERT 로 가지 않게 (오경보 방지). 대가: 진짜 위험에 늦게 반응 |
| 전이 시 `events.log` | `seq,이동평균,from,to` (`7,29.75,IDLE,COOL`) — 상태 머신이 **본 값**을 남겨야 왜 전이했는지 검증 가능 |
| `events.log` 모드 | **`"a"`** — 실행 간 이력을 이어 간다 (두 번 실행하면 2배) |
| `readings.csv` 모드 | **`"w"`** — 이번 실행 결과만 (두 번 실행해도 그대로, 헤더 1 + 데이터 10 = 11줄) |
| `readings.bin` | 마지막에 `all` 배열을 `fwrite` → `10 × 16 = 160` 바이트. 링 버퍼는 최근 4개만 가지므로 **전체는 `realloc` 로 누적**(Day 2) |

**핵심 내용**

#### 이동평균 필터를 넣은 상태 머신
> 평균을 거치면 튀는 값 하나로는 경보가 나지 않는다
- **Day 3 컨트롤러** → 같은 입력에서 ALERT **2번**
- **이동평균 추가** → **0번** (`36` 한 번으로는 평균이 35 를 못 넘음)
- **`RING_CAP` 을 2 로** → 필터가 약해져 HEAT/ALERT 전이가 다시 생김
- **필터 길이** → 반응 속도와 오경보의 트레이드오프

#### 모듈 재사용과 정리
> 모듈을 특정 용도에 묶지 않고, 조기 return 에서도 정리한다
- **같은 `Logger` 를 두 곳에** → `log_open` 에 **모드 인자** 추가 (`"w"` / `"a"`)
- **두 번째 파일 열기 실패** → 이미 연 파일(`csv`, `in`)을 닫고 `return 1`
- ⚠ 함수 안 조기 `return` → **누수** — valgrind 누수의 대부분이 여기서 생긴다

#### 선언과 정의의 자리
> `gpio.h` 는 `extern` 선언만, 정의는 한 곳
- **`main.c` 의 `uint32_t GPIO_OUT = 0, GPIO_IN = 0;`** → 정의가 한 곳에 있어야 링크된다 (Day 1 선언/정의, Day 3)
- **다음 주** → 이 정의가 사라지고 `gpio.h` 한 줄이 실제 주소가 된다
- **제출 조건** → 요구 사항 + `make test` 통과 + `make asan` 에서 에러 0

**기본 사용형태**
```c
ring_push(&ring, r.t);
double avg = ring_avg(&ring);
State next = fsm_step(st, avg);                 // 원값이 아니라 이동평균
if (next != st) { fprintf(ev, "%d,%.2f,%s,%s\n", seq, avg, STATE_NAME[st], STATE_NAME[next]); fflush(ev); events++; }
st = next; visits[st]++;
log_open(&csv, "readings.csv", "w");            // 이번 실행 결과만
FILE *ev = fopen("events.log", "a");            // 이력은 이어 쓰기
if (!ev) { perror("events.log"); log_close(&csv); fclose(in); return 1; }     // 이미 연 것 정리
fwrite(all, sizeof *all, n, bin);               // 마지막에 구조체 배열 통째로
```
```bash
make && ./logger temps.txt        # rows=10 bad=0 events=1 / IDLE 6 HEAT 0 COOL 4 ALERT 0
cat events.log                    # 7,29.75,IDLE,COOL
ls -l readings.bin                # 160 = 10 × 16
```

**연관 문제**
<<QA 7>>

### 8교시 · 종합 프로젝트 ② 마무리 · 4일 정리

**키워드**: `4일 지도`, `마무리 점검 순서`, `제출 체크리스트`, `4일의 여섯 문장`, `임베디드로 가져갈 것`, `GPIO_OUT 한 줄`, `파트 평가`

**개요**: 프로젝트를 마무리 점검하고 4일치(Day 1~4) 개념이 어떻게 이어졌는지 한 장의 지도로 정리하는 교시. 새 문법은 없고 **연결 관계**를 묻는다 (예: 컴파일 4단계 ↔ 링크 에러, 배열 감쇠 ↔ `fwrite(p, size, n, fp)`, 패딩 ↔ 바이너리 이식성, `static` ↔ 링 버퍼, 함수 포인터 ↔ 상태 머신 ↔ gdb).

**마무리 점검 순서 (30분)**
1. `make test` → `make asan && ./logger temps.txt` → `make debug` 세 빌드가 모두 도는지
2. `temps.txt` 를 두 번 실행해 `events.log` 는 **이어지고** `readings.csv` 는 **새로 쓰였는지**
3. `readings.bin` 을 `load_bin` 으로 읽어 행 수가 `rows` 와 같은지
4. **빈 파일, BAD 만 있는 파일, 100줄 파일(realloc 경로)** 에서 죽지 않는지

**제출 체크리스트**: `make clean && make && make test` 경고 0 / `make asan` 에러 0 / 인자 없음 → usage + 종료 코드 1, 없는 파일 → `perror` + 1 / `events.log` 이어쓰기(`"a"`) · `readings.csv` 새로쓰기(`"w"`) / `readings.bin` 크기 = rows × 16.

**4일의 여섯 문장**
- **Day 1** 컴파일은 4단계, 선언은 `.h` 정의는 `.c`, 자료형은 바이트 수와 해석 규칙, `-Wall -Wextra` 는 항상
- **Day 2** `&` 는 주소 `*` 는 값, 배열은 넘기면 포인터, 문자열은 `\0` 까지, `malloc → 검사 → free`
- **Day 3** 구조체는 통째 복사, 비트는 `|=` `&=~` `^=`, 매크로는 텍스트, 함수도 주소, 상태 = 인덱스
- **Day 4** `fopen → 검사 → fclose`, 텍스트는 사람이 · 바이너리는 기계가, 링 버퍼는 고정 메모리, gdb · assert · ASan 은 버그를 일찍 터뜨린다
- **임베디드로 가져갈 것**: 고정폭 타입, 바이트 조립, `volatile` 레지스터, 콜백, **링 버퍼**, 반환값 에러 규약 (※ `realloc` 동적 확장은 MCU 에서 위험 → 가져가지 않음)
- **다음 주 ESP32 에서 바뀌는 것**: `gpio.h` 의 `GPIO_OUT` **한 줄**이 실제 주소로, `main` 이 `setup/loop` 로 — 나머지 모듈은 그대로.

**일정 (강의 자료 기준)**: 파트 평가 = **필답(4일 워크시트 범위) + 실습(종합 프로젝트의 변형)**. 준비 순서는 4일치 워크시트와 복습 퀴즈를 먼저 다시 풀기. 이후 MCU 동작 원리는 `extern uint32_t GPIO_OUT` 을 실제 주소로 바꾸는 것에서 시작.

**연관 문제**
<<QA 8>>
