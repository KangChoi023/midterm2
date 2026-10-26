## Day 4 — FreeRTOS · LittleFS · Wi-Fi · 디버깅 · 종합 프로젝트

### 1교시 · FreeRTOS 태스크 — loop 를 여러 개로

**키워드:** FreeRTOS, 태스크, loopTask, xTaskCreatePinnedToCore, vTaskDelay, pdMS_TO_TICKS, 우선순위, 코어 0/1, 스택 바이트, uxTaskGetStackHighWaterMark, TaskHandle_t, void *param, vTaskDelete, portMAX_DELAY, 굶음 starvation, 워치독

**개요**
- `setup()`/`loop()` 는 이미 **loopTask** 라는 FreeRTOS 태스크 안에서 돌고 있었다(Day 1 2교시). 이제 태스크를 **직접 여러 개** 만들어 각자 다른 주기로 동작시키고, `delay` vs `vTaskDelay`, 우선순위·코어·스택, `void *` 매개변수 재사용을 다룬다. (Linux Day5 의 pthread 와 같은 개념의 MCU 판)

**핵심 용어**
- **태스크(task)**: FreeRTOS 의 실행 단위(쓰레드). 각자 **끝나지 않는 루프**를 가진다. 직원 한 명씩 자기 일만 하는 것.
- **vTaskDelay**: "이 태스크를 N ms 재우고 **CPU 를 다른 태스크에 넘긴다**". (`delay` 도 ESP32 에서는 내부적으로 같지만 의미를 분명히)
- **우선순위**: 클수록 먼저. loopTask = 1, idle = 0.
- **코어**: 0 = Wi-Fi·Bluetooth 가 주로 씀, 1 = loopTask 코어.
- **스택(바이트 단위)**: ESP32 FreeRTOS 는 **바이트**. printf·String 쓰면 4096 이상.
- **High Water Mark**: 지금까지 스택이 **가장 많이 찼을 때 남아 있던 양** → 수백 바이트 이하면 늘린다.
- **굶음(starvation)**: 높은 우선순위 태스크가 vTaskDelay 없이 돌면 같은 코어의 낮은 태스크가 **실행 기회를 못 얻음** (워치독 경고·재부팅).

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `xTaskCreatePinnedToCore` | `(fn, "name", stack, param, prio, &handle, core)` | 태스크 생성 — 함수·이름·스택(바이트)·매개변수·우선순위·핸들·**코어** |
| `vTaskDelay` | `vTaskDelay(pdMS_TO_TICKS(ms))` | ms 재우고 양보 |
| `vTaskDelay(portMAX_DELAY)` | — | 영원히 재움 (loop 에 할 일이 없을 때) |
| `uxTaskGetStackHighWaterMark` | `UBaseType_t uxTaskGetStackHighWaterMark(TaskHandle_t h)` | 스택 최저 여유 |
| `vTaskDelete(NULL)` | — | 스스로 지우기 (태스크 함수는 **return 하면 안 됨**) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 1-1 `two_tasks.ino` | `taskRed`(300ms)와 `taskGreen`(1000ms)이 각자 LED 를 토글, loop 는 2초마다 출력 |
| 1-2 `task_info.ino` | 우선순위 2 `taskWork` 가 100ms 마다 ADC 읽기, loop 가 그 태스크의 스택 여유·코어·힙 여유 출력 (`스택 여유 2140 바이트`) |
| 1-3 ⚠ 함정 | 쉬지 않는 태스크 / return / 작은 스택 / 지역 변수 주소 넘기기 |

**핵심 내용**

#### 태스크로 나누기
> 각 동작이 자기 루프만 신경 쓴다 — Day 1 의 millis 비교를 태스크가 대신
- **같은 태스크 함수를 여러 번** → 매개변수 `void *param` → `blink_cfg_t *cfg = (blink_cfg_t *)param;` 로 원래 타입으로

#### 태스크 함정
> 양보 · 종료 · 스택 · 인자 수명
- ⚠ `for (;;) {}` 에 vTaskDelay 없음 → 같은 코어의 낮은 우선순위가 **굶음**
- ⚠ 태스크 함수가 **return** → 재부팅 (루프로 두거나 `vTaskDelete(NULL)`)
- ⚠ 스택 1024 에 `Serial.printf` → `Stack canary watchpoint triggered` 재부팅
- ⚠ `setup` 의 **지역 변수 주소**를 매개변수로 → setup 이 끝나면 사라진 메모리 → **전역 · static** 주소를 넘긴다

**기본 사용형태**
```c
typedef struct { int pin; int ms; } blink_cfg_t;
static blink_cfg_t red = {25, 300}, green = {26, 1000};     // static/전역 — 수명 보장
void taskBlink(void *param) {
    blink_cfg_t *cfg = (blink_cfg_t *)param;
    pinMode(cfg->pin, OUTPUT);
    for (;;) { digitalWrite(cfg->pin, !digitalRead(cfg->pin)); vTaskDelay(pdMS_TO_TICKS(cfg->ms)); }
}
void setup() {
    xTaskCreatePinnedToCore(taskBlink, "red",   2048, &red,   1, NULL, 1);
    xTaskCreatePinnedToCore(taskBlink, "green", 2048, &green, 1, NULL, 1);
}
void loop() { vTaskDelay(portMAX_DELAY); }
```

**연관 문제**
<<QA 1>>

### 2교시 · 큐와 뮤텍스 — 태스크끼리 주고받기

**키워드:** 큐 queue, xQueueCreate, xQueueSend, xQueueReceive, 값 복사, 생산자 소비자, 큐 가득 데이터 손실, 뮤텍스, xSemaphoreCreateMutex, xSemaphoreTake, xSemaphoreGive, Serial 출력 섞임, 교착 deadlock, critical vs 뮤텍스

**개요**
- 한 태스크가 만든 데이터를 다른 태스크에 **안전하게** 넘기는 **큐**(생산자·소비자)와, 여러 태스크가 쓰는 자원(Serial, 공유 구조체)을 한 번에 하나만 쓰게 하는 **뮤텍스**. 전역 변수 공유와의 차이를 정리한다. (Linux Day5 링 버퍼 + 조건 변수, mutex 의 FreeRTOS 판)

**핵심 용어**
- **큐**: 정해진 크기의 칸에 **값을 복사**해서 넣고 꺼내는 통로. 보내는 쪽 지역 변수가 사라져도 안전. 비어 있으면 받는 쪽이 **잠들어 기다림**(CPU 안 씀).
- **뮤텍스**: 화장실 열쇠 — **Take → 사용 → Give**. 다른 태스크는 Take 에서 기다림.
- **critical vs 뮤텍스**: Day 1 `portENTER_CRITICAL` 은 인터럽트까지 막으므로 **아주 짧게**, 뮤텍스는 태스크끼리라 **길어도** 된다.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `xQueueCreate` | `QueueHandle_t xQueueCreate(len, item_size)` | `xQueueCreate(8, sizeof(sample_t))` = sample_t 8칸 |
| `xQueueSend` | `xQueueSend(q, &item, wait_ticks)` | 내용 복사해 넣기. 0 = 가득이면 기다리지 않고 **실패(pdTRUE 아님)** |
| `xQueueReceive` | `xQueueReceive(q, &item, portMAX_DELAY)` | 꺼내기 — 올 때까지 잠 |
| `xSemaphoreCreateMutex` | `SemaphoreHandle_t xSemaphoreCreateMutex(void)` | 뮤텍스 생성 |
| `xSemaphoreTake` / `Give` | `xSemaphoreTake(m, portMAX_DELAY)` / `xSemaphoreGive(m)` | 잠금 / 해제 — **반드시 짝** |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 2-1 `queue_demo.ino` | `taskSensor`(우선순위 2, 100ms)가 `sample_t {raw, t_ms}` 를 큐에 보내고, `taskShow` 가 받아 LED 밝기 + `raw:…,age:…` 출력 |
| 2-2 `mutex_demo.ino` | 다른 코어의 `taskPot`·`taskDht` 가 `safe_print`(Take → print 여러 번 → Give)로 출력 — 뮤텍스 없으면 `[POT] pot=[DHT] te2041mp=24.0` 처럼 섞임 |
| 2-3 ⚠ 함정 | Give 누락 / 큐 생성 순서 / 포인터 넣기 / 느린 소비자 / 반대 순서 잠금 |

**핵심 내용**

#### 큐
> 값을 복사해 넣고 꺼내는 태스크 간 통로
- **생산자 100ms · 소비자 300ms · 큐 8칸** → 시간이 지나면 **가득 차 데이터가 버려짐** → 칸을 늘리거나 소비를 빠르게
- ⚠ 큐를 만들기 **전에** 태스크 생성 → NULL 큐 접근 **재부팅** → **큐 먼저, 태스크 나중**
- ⚠ 큐에 `&local` 같은 **포인터** → 받는 쪽이 사라진 메모리 → **값**을 넣는다

#### 뮤텍스
> 공유 자원을 한 번에 한 태스크만
- **다른 코어의 두 태스크가 `Serial.print` 를 나눠 부름** → **한 줄의 조각이 섞임** → 뮤텍스
- ⚠ Take 후 중간 `return` 으로 **Give 누락** → 다른 태스크가 **영원히 대기** (멈춘 것처럼 보임)
- ⚠ A 는 M1→M2, B 는 M2→M1 → **교착** → 잠그는 순서를 항상 같게 (Linux Day5 3교시)

**기본 사용형태**
```c
typedef struct { int raw; uint32_t t_ms; } sample_t;
QueueHandle_t q; SemaphoreHandle_t serialLock;
void taskSensor(void *p) { for (;;) { sample_t s = { analogRead(34), millis() };
    if (xQueueSend(q, &s, 0) != pdTRUE) Serial.println("큐 가득 — 버림");
    vTaskDelay(pdMS_TO_TICKS(100)); } }
void taskShow(void *p) { sample_t s; for (;;) { xQueueReceive(q, &s, portMAX_DELAY);
    xSemaphoreTake(serialLock, portMAX_DELAY); Serial.printf("raw:%d\n", s.raw); xSemaphoreGive(serialLock); } }
void setup() {
    q = xQueueCreate(8, sizeof(sample_t));        // 큐 · 뮤텍스 먼저
    serialLock = xSemaphoreCreateMutex();
    xTaskCreatePinnedToCore(taskSensor, "sensor", 3072, NULL, 2, NULL, 1);
    xTaskCreatePinnedToCore(taskShow,   "show",   3072, NULL, 1, NULL, 1);
}
```

**연관 문제**
<<QA 2>>

### 3교시 · ISR → 태스크 알림 · 소프트웨어 타이머

**키워드:** 지연 처리 deferred processing, vTaskNotifyGiveFromISR, ulTaskNotifyTake, portYIELD_FROM_ISR, FromISR 함수, 태스크 먼저 ISR 나중, 소프트웨어 타이머, xTimerCreate, xTimerStart, xTimerChangePeriod, xTimerStop, 하드웨어 타이머 비교

**개요**
- Day 1 6교시에서는 ISR 이 카운터만 올리고 loop 가 들여다봤다(폴링). 이제 **ISR 은 "알리기"만 하고, 실제 일은 태스크가 깨어나서** 한다(지연 처리). 또 FreeRTOS **소프트웨어 타이머**로 태스크 없이 주기 작업을 만든다.

**핵심 용어**
- **지연 처리(deferred processing)**: ISR 은 짧게 알림만, 긴 일(출력·대기·파일)은 태스크가. ISR 에서 600ms 일하면 **그동안 다른 일이 막힌다**.
- **태스크 알림(task notification)**: ISR 이 특정 태스크를 깨우는 가벼운 신호. 초인종.
- **FromISR**: ISR 안에서는 **이름에 `FromISR` 이 붙은 함수만**. `Serial.printf`, `vTaskDelay`, `delay` 는 태스크 몫.
- **소프트웨어 타이머**: ms 단위, 콜백이 **타이머 서비스 태스크**에서 불림. 콜백 안에서 기다리면 **다른 모든 소프트웨어 타이머가 밀린다**.

**새로 나온 함수/명령**

| 이름 | 기본형 | 용도 |
|---|---|---|
| `vTaskNotifyGiveFromISR` | `vTaskNotifyGiveFromISR(handle, &woken)` | ISR 에서 태스크 깨우기 |
| `portYIELD_FROM_ISR` | `portYIELD_FROM_ISR(woken)` | 깨운 태스크가 더 급하면 **바로 전환** |
| `ulTaskNotifyTake` | `uint32_t ulTaskNotifyTake(pdTRUE, portMAX_DELAY)` | 알림까지 잠 — **쌓인 알림 개수** 반환 |
| `xTimerCreate` | `xTimerCreate("name", pdMS_TO_TICKS(500), pdTRUE, NULL, cb)` | 이름·주기·**반복(pdTRUE)**·ID·콜백 |
| `xTimerStart` / `xTimerChangePeriod` / `xTimerStop` | `(t, …, 0)` | 시작 / 주기 변경 / 정지 |

| 필요 | 쓸 것 |
|---|---|
| µs 정밀도, 화면 갱신 | 하드웨어 타이머 (Day 3) — ISR, 할 수 있는 일 제한 |
| ms 단위 주기 작업, 깜빡임 | 소프트웨어 타이머 |
| 복잡한 일, 기다림이 있는 일 | 태스크 + vTaskDelay |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 3-1 `isr_notify.ino` | 버튼 ISR 은 200ms 디바운스 후 `vTaskNotifyGiveFromISR` 만, `taskWorker`(우선순위 3)가 깨어나 출력 + LED 3번 깜빡임 |
| 3-2 `soft_timer.ino` | 500ms 반복 소프트웨어 타이머로 LED 토글, 시리얼 `f`/`s`/`x` 로 주기 변경·정지 |
| 3-3 ⚠ 함정 | FromISR 아닌 함수 / ISR 먼저 등록 / 콜백에서 대기 / IRAM_ATTR 누락 |

**핵심 내용**

#### ISR → 태스크 알림 순서
> 태스크 먼저, `attachInterrupt` 나중 — ISR 안에서는 FromISR 함수만
- **순서** → **태스크 먼저 만들고, `attachInterrupt` 나중**
- ⚠ 반대면 → 버튼이 눌릴 때 핸들이 NULL → 재부팅
- ⚠ ISR 에서 `vTaskNotifyGive` (FromISR 아님) → 재부팅 · 이상 동작
- ⚠ ISR 에 `IRAM_ATTR` 없음 → Flash 작업 중 인터럽트가 오면 크래시 (Day 1)

**기본 사용형태**
```c
TaskHandle_t hWorker = NULL;
void IRAM_ATTR onButton() {
    BaseType_t woken = pdFALSE;
    vTaskNotifyGiveFromISR(hWorker, &woken);     // 알리기만
    portYIELD_FROM_ISR(woken);
}
void taskWorker(void *p) { for (;;) { ulTaskNotifyTake(pdTRUE, portMAX_DELAY); Serial.println("버튼 처리"); } }
void setup() {
    xTaskCreatePinnedToCore(taskWorker, "worker", 3072, NULL, 3, &hWorker, 1);   // 태스크 먼저
    attachInterrupt(digitalPinToInterrupt(4), onButton, FALLING);                // ISR 나중
    TimerHandle_t t = xTimerCreate("blink", pdMS_TO_TICKS(500), pdTRUE, NULL, onBlink);
    xTimerStart(t, 0);
}
```

**연관 문제**
<<QA 3>>

### 4교시 · LittleFS — Flash 에 파일로 저장

**키워드:** LittleFS, Flash 파일 시스템, LittleFS.begin(true) 포맷, FILE_READ FILE_WRITE FILE_APPEND, open close, f.available, openNextFile, usedBytes totalBytes, CSV 로거, 전원 꺼져도 유지, Partition Scheme, Flash 쓰기 수명 10만 번, "/log.csv"

**개요**
- RAM 의 변수는 전원이 꺼지면 사라진다. Flash 4MB 중 약 1.5MB 를 **파일 영역(LittleFS)** 으로 써서 **전원이 꺼져도 남는** 파일을 만들고, 센서 값을 CSV 로그로 남긴다. C Day4 의 `fopen · fprintf · fclose` 와 같은 흐름이고, Linux Day4 의 fsync 처럼 **닫아야 확실히 기록**된다.

**핵심 용어**
- **LittleFS**: Flash 위의 작은 파일 시스템. 새 스케치를 업로드해도 **파일은 남는다**(앱 영역만 다시 씀).
- **모드**: `"r"` FILE_READ 읽기 / `"w"` FILE_WRITE **새로 쓰기 — 기존 내용 삭제** / `"a"` FILE_APPEND **끝에 이어 쓰기 — 로그용**.
- **Flash 쓰기 수명**: 블록당 약 **10만 번** → 자주 조금씩 쓰면 수명·용량을 금방 소진 → 모아서 쓰거나 주기를 늘림.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `LittleFS.begin(true)` | 마운트, **true = 처음이면 포맷** (실패 시 false) |
| `File f = LittleFS.open("/log.csv", FILE_APPEND)` | 열기 (경로는 **`/` 로 시작**), 실패면 `!f` |
| `f.println` / `f.printf` / `f.read()` / `f.available()` / `f.close()` | 쓰기 / 읽기 / 남은 바이트 / **닫기(확실히 기록)** |
| `LittleFS.exists` / `remove` / `usedBytes` / `totalBytes` / `format` | 존재 확인 / 삭제 / 용량 / 포맷 |
| `root.openNextFile()` | 디렉터리 목록 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 4-1 `fs_basic.ino` | `/hello.txt` 를 FILE_WRITE 로 쓰고 다시 읽어 출력, 파일 목록과 사용량 출력 |
| 4-2 `csv_logger.ino` | 5초마다 DHT 온도·가변저항을 `/log.csv` 에 APPEND(처음이면 머리줄 `ms,temp,pot`), 시리얼 `d` 보기 / `x` 지우기 — USB 를 뽑았다 꽂아도 로그가 남음 |
| 4-3 ⚠ 함정 | No FS 파티션 / FILE_WRITE / close 누락 / 과도한 쓰기 / `/` 누락 / 업로드 후 파일 |

**핵심 내용**

#### LittleFS 로그
> 전원을 꺼도 남는다 — 열고, 쓰고, 바로 닫는다
- **재부팅 후 로그** → 이전 줄이 그대로, `ms` 가 다시 작은 값부터 (재부팅됐으니까) → 5교시 NTP 시각으로 개선
- **끝까지 읽기** → `while (f.available()) Serial.write(f.read());`
- **`open` 실패** → `if (!f)` 로 검사

#### LittleFS 함정
> 파티션 · 모드 · 닫기 · 수명 · 경로
- ⚠ Partition Scheme 이 **No FS** → 마운트 실패 → `Default 4MB with spiffs`
- ⚠ 로그에 **FILE_WRITE** → 매번 이전 로그가 지워짐 → **FILE_APPEND**
- ⚠ `close()` 없이 전원 끔 → **마지막 부분 유실**
- ⚠ 100ms 마다 한 줄씩 영원히 → 수명 · 용량 소진
- ⚠ 경로 앞 `/` 빠뜨림 → open 실패

**기본 사용형태**
```c
#include <LittleFS.h>
bool log_append(float t, int pot) {
    bool is_new = !LittleFS.exists("/log.csv");
    File f = LittleFS.open("/log.csv", FILE_APPEND);
    if (!f) return false;
    if (is_new) f.println("ms,temp,pot");
    f.printf("%lu,%.1f,%d\n", millis(), t, pot);
    f.close();                                  // 닫아야 확실히 기록
    return true;
}
void setup() { if (!LittleFS.begin(true)) Serial.println("LittleFS 마운트 실패"); }
```

**연관 문제**
<<QA 4>>

### 5교시 · Wi-Fi 연결 · 시간 · AP 모드

**키워드:** Wi-Fi, 외부 안테나 U.FL, 2.4GHz, WiFi.mode WIFI_STA, WiFi.begin, WL_CONNECTED, 타임아웃, setAutoReconnect, RSSI dBm, scanNetworks, NTP configTime, getLocalTime, strftime, UTC+9, AP 모드 softAP 192.168.4.1, 비밀번호 8자, ADC2, secrets.h

**개요**
- 외부 안테나를 달고 공유기에 접속(스테이션 모드)해 IP·신호 세기를 확인하고, **타임아웃·재연결**을 포함한 연결 함수를 만든다. NTP 로 실제 시각을 얻고, 공유기가 없을 때는 ESP32 가 직접 망을 만드는 **AP 모드**를 쓴다.

**핵심 용어**
- **ESP-32U**: 기판 안테나가 없는 U 모델 → **U.FL 외부 안테나 필수** (없으면 RSSI -85dBm 이하·연결 실패).
- **2.4GHz 만**: 5GHz 공유기 이름은 스캔에 안 보임. WPA2-Enterprise(아이디+비밀번호) 방식은 어려움.
- **스테이션(STA)**: 공유기에 붙는 쪽. **AP(Access Point)**: ESP32 가 공유기 역할 — 기본 주소 **192.168.4.1**, **인터넷은 안 됨**(NTP 불가).
- **RSSI**: 신호 세기(dBm, 0 에 가까울수록 좋음). -30~-60 좋음 / -60~-75 보통 / -80 이하 나쁨.
- **NTP**: 인터넷 시간 서버. 한 번 받으면 ESP32 가 스스로 시계를 유지.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `WiFi.mode(WIFI_STA)` / `WiFi.begin(SSID, PASS)` / `WiFi.status() == WL_CONNECTED` | 스테이션 모드 접속 / 상태 |
| `WiFi.localIP().toString().c_str()` / `WiFi.RSSI()` | IP 문자열 / 신호 세기 |
| `WiFi.scanNetworks()` / `WiFi.SSID(i)` / `WiFi.RSSI(i)` / `WiFi.channel(i)` | 주변 망 스캔 |
| `WiFi.setAutoReconnect(true)` | 끊기면 자동 재연결 |
| `configTime(9 * 3600, 0, "pool.ntp.org", …)` | NTP — **UTC+9 초 단위 오프셋**, 서머타임 0 |
| `getLocalTime(&tm, timeout_ms)` / `strftime(buf, n, "%Y-%m-%d %H:%M:%S", &tm)` | 현재 시각 / 문자열 |
| `WiFi.softAP(name, pass)` / `softAPIP()` / `softAPgetStationNum()` | AP 만들기(비밀번호 **8자 이상**) / 주소 / 접속 기기 수 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 5-1 `wifi_connect.ino` | 주변 망 스캔 → `wifi_connect(15000)`(타임아웃 있는 대기) → `연결됨 IP 192.168.0.23 RSSI -52 dBm`, loop 는 상태가 **바뀔 때만** 출력 |
| 5-2 `ntp_time.ino` | 접속 후 `configTime` → `getLocalTime` 으로 `지금: 2026-…` 출력, 1초마다 `HH:MM:SS` |
| 5-3 `wifi_ap.ino` | `ESP32-XXXX`(MAC 으로 보드마다 다른 이름) AP 를 만들고 접속 기기 수 출력 → `주소 192.168.4.1` |
| 5-4 ⚠ 함정 | 안테나 / 5GHz / 타임아웃 / ADC2 / 비밀번호 유출 / 7자 비밀번호 |

**핵심 내용**

#### Wi-Fi 연결
> 연결 대기에는 반드시 타임아웃 — 상태는 바뀔 때만 출력
- ⚠ 대기 `while` 에 **타임아웃이 없으면** → 비밀번호가 틀렸을 때 **영원히** 점만 찍음
- **`WiFi.status()`** → 상태 값, 바뀔 때만 출력 (Day 1 8교시 상태 머신처럼)

#### Wi-Fi 와 함께 쓸 때 주의
> ADC2 와 비밀번호
- ⚠ Wi-Fi 를 켠 뒤 **ADC2 핀(25·26·27…)** 으로 `analogRead` → 실패 (Day 2 1교시) → 아날로그는 **34 (ADC1)**
- ⚠ SSID·비밀번호를 코드에 넣고 GitHub 에 올리면 유출 → **별도 헤더(`secrets.h`)** 로 빼고 올리지 않음

**기본 사용형태**
```c
#include <WiFi.h>
bool wifi_connect(uint32_t timeout_ms) {
    WiFi.mode(WIFI_STA); WiFi.begin(SSID, PASS);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED) { if (millis() - t0 > timeout_ms) return false; delay(250); }
    return true;
}
if (wifi_connect(15000)) { WiFi.setAutoReconnect(true); configTime(9 * 3600, 0, "pool.ntp.org"); }
else { WiFi.mode(WIFI_AP); WiFi.softAP("ESP32-MONITOR", "12345678"); }   // 192.168.4.1
```

**연관 문제**
<<QA 5>>

### 6교시 · 웹 서버 — 스마트폰에서 보기 · 켜기

**키워드:** WebServer, server.on 경로 → 함수, 콜백, handleClient, HTML 페이지, JSON API, server.send 상태 코드 200 400 403 404, hasArg arg toInt, 쿼리 인자, 허용 목록 검사, fetch setInterval 자동 갱신, R"HTML( )HTML" 원시 문자열, PROGMEM, meta charset utf-8, 같은 망

**개요**
- ESP32 를 **웹 서버**로 만들어 스마트폰 브라우저에서 센서 값을 보고 LED 를 켠다. 주소(경로)마다 처리 함수를 연결하는 것은 **함수 포인터·콜백**(Day 2 8교시 센서 표와 같은 원리). HTML 페이지와 JSON API, 쿼리 인자, 새로고침 없는 갱신(fetch)을 다룬다.

**핵심 용어**
- **경로(path)와 핸들러**: `server.on("/api", handleApi)` = "/api 요청이 오면 이 함수를 불러라".
- **JSON API**: `{"temp":24.6,"pot":2041}` 처럼 **기계가 읽기 좋은** 응답 → 페이지의 JavaScript 가 값만 쉽게 꺼내 씀.
- **쿼리 인자**: `/led?pin=25&on=1` → `server.arg("pin")` = `"25"` (**String 문자열**, `.toInt()` 로 변환).
- **HTTP 상태 코드**: 200 성공 / **400 인자 누락** / 403 허용 안 됨 / 404 없음 / 303 다른 곳으로.
- **원시 문자열 `R"HTML( … )HTML"`**: 따옴표를 **이스케이프 없이** HTML 을 그대로.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `WebServer server(80);` / `server.begin()` | 80 포트 웹 서버 |
| `server.on(path, fn)` / `server.onNotFound(fn)` | 경로 → 함수 등록 |
| `server.handleClient()` | loop 에서 **자주** — 들어온 요청 처리 |
| `server.send(code, "text/html", body)` | 응답 (`application/json`, `text/plain`) |
| `server.hasArg("pin")` / `server.arg("pin").toInt()` | 인자 있나 / 값 |
| `fetch('/api')` + `setInterval(…, 1000)` (JavaScript) | 페이지가 1초마다 새 값을 가져옴 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 6-1 `web_sensor.ino` | `/` 는 온도·가변저항 HTML, `/api` 는 JSON, 나머지는 404 |
| 6-2 `web_control.ino` | 버튼 페이지에서 `fetch('/led?pin=25&on=1')` 로 LED 제어(핀 25·26 만 허용, 아니면 403, 인자 없으면 400), 1초마다 `/api` 로 가변저항 갱신 |
| 6-3 ⚠ 함정 | loop 의 delay / 느린 핸들러 / 다른 망 / 핀 검사 누락 / 작은 버퍼 / 한국어 깨짐 |

**핵심 내용**

#### 웹 서버 접속과 보안
> 같은 망에서, 허용한 입력만
- **페이지가 안 열리면** → **스마트폰과 같은 Wi-Fi 망인지** 먼저 (LTE · 다른 Wi-Fi 면 안 됨)
- **`pin` 을 허용 목록(25, 26)으로 검사** → URL 로 아무 핀이나 조작 못하게

#### 웹 서버 함정
> 응답을 늦추는 것 · 페이지를 깨뜨리는 것
- ⚠ loop 에 `delay(1000)` → 요청이 1초씩 밀림
- ⚠ 핸들러 안에서 오래 걸리는 일 (DHT 여러 번) → **다른 요청 대기** → 값은 태스크가 **미리 읽어 두고** 핸들러는 보내기만 (8교시)
- ⚠ `char html[128]` 에 긴 페이지 → 잘림
- ⚠ 한국어 깨짐 → `<meta charset="utf-8">`

**기본 사용형태**
```c
#include <WebServer.h>
WebServer server(80);
void handleApi() {
    char json[64];
    snprintf(json, sizeof json, "{\"pot\":%d}", analogRead(34));
    server.send(200, "application/json", json);
}
void handleLed() {
    if (!server.hasArg("pin") || !server.hasArg("on")) { server.send(400, "text/plain", "pin, on 필요"); return; }
    int pin = server.arg("pin").toInt();
    if (pin != 25 && pin != 26) { server.send(403, "text/plain", "허용되지 않은 핀"); return; }
    digitalWrite(pin, server.arg("on").toInt() ? HIGH : LOW);
    server.send(200, "text/plain", "OK");
}
void setup() { server.on("/api", handleApi); server.on("/led", handleLed); server.begin(); }
void loop() { server.handleClient(); }
```

**연관 문제**
<<QA 6>>

### 7교시 · 디버깅 — 로그 · 크래시 해독 · 재부팅 원인

**키워드:** 로그 레벨 매크로, __FILE__ __LINE__, ##__VA_ARGS__, do while(0), Guru Meditation Error, StoreProhibited, LoadProhibited, EXCVADDR, PC, Backtrace, addr2line, Stack canary watchpoint, Task watchdog, IllegalInstruction, Brownout, esp_reset_reason, ESP_RST_PANIC, getFreeHeap 누수, 감시 태스크

**개요**
- 문제가 생겼을 때 **어디서 · 언제**를 알려 주는 로그 매크로를 만들고, ESP32 의 크래시 메시지(**Guru Meditation**)에서 예외 종류·주소·호출 경로를 읽는다. `esp_reset_reason()` 으로 재부팅 원인을 기록하고 힙·스택을 감시하며, 증상 → 원인 디버깅 순서를 정리한다.

**핵심 용어**
- **로그 매크로**: `LOG(lv, tag, fmt, …)` 가 `[ms][레벨][파일:줄]` 을 붙여 출력. `LOG_LEVEL` 보다 상세한 것은 출력 안 됨.
- **`__FILE__` · `__LINE__`**: 전처리기가 채우는 **소스 파일 이름과 줄 번호**(C Day1). `##__VA_ARGS__`: 인자가 없을 때 앞의 쉼표를 지움.
- **`do { … } while (0)`**: 매크로를 **if · else 안에서도 안전하게** 한 문장처럼.
- **Guru Meditation Error**: ESP32 의 크래시(패닉) 메시지 → 재부팅.
- **Backtrace**: **호출 경로**(가장 안쪽이 먼저). `addr2line` 으로 소스 줄로 바꾼다.

| 메시지 | 흔한 원인 |
|---|---|
| `StoreProhibited` / `LoadProhibited` | 쓰면/읽으면 안 되는 주소 — **NULL**·해제된·초기화 안 된 포인터. `EXCVADDR: 0x00000000` 이면 NULL (0x4 면 NULL 구조체의 4바이트 멤버) |
| `Stack canary watchpoint triggered (A)` | 태스크 A **스택 부족** → 스택 늘리기, 큰 지역 배열을 static 으로 |
| `Task watchdog got triggered` | 태스크가 오래 쉬지 않음 — vTaskDelay 누락 |
| `IllegalInstruction` | 잘못된 함수 포인터 호출, 스택 손상 |
| `Brownout detector was triggered` | 전압 강하 — 모터·서보·케이블·허브 (Day 3) |

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `esp_reset_reason()` | 지난 재부팅 원인: `ESP_RST_POWERON` / `SW` / **`PANIC`(크래시)** / `INT_WDT` / `TASK_WDT` / `BROWNOUT` |
| `xtensa-esp32-elf-addr2line -pfiaC -e build/….elf 0x400d1a28 …` | Backtrace 주소 → `loop() at crash_demo.ino:32` |
| `ESP.getFreeHeap()` | 남은 힙 — 계속 줄면 **누수** |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 7-1 `debug_log.ino` | `LOGE/W/I/D` 매크로로 `[   512][I][debug_log.ino:14] 부팅 완료, 힙 여유 291340` 형식 출력 (LOG_LEVEL 3 이면 LOGD 는 안 나옴) |
| 7-2 `crash_demo.ino` | 부팅 시 지난 재부팅 원인 출력, 버튼을 누르면 `find_box()` 가 돌려준 NULL 에 `b->value = 42` → `StoreProhibited`, `EXCVADDR: 0x00000000` → 재부팅 후 "크래시(패닉)" |
| 7-3 `health.ino` | 감시 태스크가 5초마다 힙(최저값)과 taskA(1500B 배열, 스택 2048 — 빠듯)의 스택 여유 출력 |
| 7-4 디버깅 순서 | 증상별로 먼저 볼 것 표 |

**핵심 내용**

#### NULL 과 메모리 누수
> 못 찾으면 NULL 인 함수는 받자마자 검사, 힙은 감시
- **"못 찾으면 NULL"** → **받자마자 NULL 검사** (C Day2)
- ⚠ 힙이 계속 줄면 → **메모리 누수** (malloc 후 free 누락, 반복되는 String 연결)

#### 디버깅 순서
> 증상 → 확인할 것
- **업로드 안 됨** → 포트 · 케이블 · Boot
- **글자 깨짐** → 115200
- **부품 무반응** → 배선 · 핀 · 전원 · pinMode
- **값 이상** → 전압 · 분압 · ADC1 · 예열
- **I2C 없음** → 스캐너 · SDA/SCL · 주소
- **LCD 글자 없음** → V0 · RW · VDD
- **가끔 재부팅** → `esp_reset_reason()`
- **서서히 멈춤** → 힙 감시
- **멈춘 듯** → 뮤텍스 반환 누락 · 무한 대기 · 타임아웃 없는 while
- **가장 좋은 습관** → 한 번에 하나만 바꾸고 기록, "잘 되던 마지막 상태" 로 돌아가 조금씩 더한다

**기본 사용형태**
```c
#define LOG_LEVEL 3
#define LOG(lv, tag, fmt, ...) do { if ((lv) <= LOG_LEVEL) \
    Serial.printf("[%6lu][%s][%s:%d] " fmt "\n", millis(), tag, __FILE__, __LINE__, ##__VA_ARGS__); } while (0)
#define LOGE(fmt, ...) LOG(1, "E", fmt, ##__VA_ARGS__)
#define LOGI(fmt, ...) LOG(3, "I", fmt, ##__VA_ARGS__)
#include "esp_system.h"
if (esp_reset_reason() == ESP_RST_PANIC) LOGE("지난번 크래시로 재부팅");
box_t *b = find_box(7);
if (b == NULL) { LOGE("box 없음"); return; }      // NULL 검사
```

**연관 문제**
<<QA 7>>

### 8교시 · 종합 프로젝트 — IoT 환경 모니터 · 4일 정리

**키워드:** IoT 환경 모니터, taskSensor 큐 taskLogger, reading_t, latest 뮤텍스, LittleFS 로그, 히스테리시스 경보 30/29, ISR 알림 taskButton, WebServer / /api /log.csv /clear, streamFile, AP 모드 대체, 생성 순서, 잠금 구간 짧게

**개요**
- 4일 동안 배운 것(태스크 · 큐 · 뮤텍스 · 파일 · 웹 · ISR 알림)을 하나의 시스템으로 통합한다. 역할을 나눈 구조가 왜 고치기 쉬운지 설명한다.

**핵심 용어** — 시스템 구조
```text
[taskSensor] 2초마다 DHT11 · 가변저항 → reading_t
     │ xQueueSend
[ 큐 8칸 ]
     │ xQueueReceive
[taskLogger] → LittleFS /log.csv 이어 쓰기
     │ → latest (뮤텍스로 보호) 갱신
     │ → 30°C 이상이면 빨강 LED (히스테리시스 30 / 29)
[loop: WebServer] /  /api  /log.csv  /clear
[버튼 ISR] → 알림 → [taskButton] 초록 LED 토글 · 로그에 "BUTTON"
```

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `server.streamFile(f, "text/csv")` | 파일을 **RAM 에 다 올리지 않고** 조금씩 전송 |
| `server.sendHeader("Location", "/"); server.send(303);` | 처리 후 첫 페이지로 이동 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 8-A `iot_monitor.ino` | 센서 태스크가 2초마다 큐로 보내고, 로거 태스크가 뮤텍스 안에서 `latest` 갱신 + CSV 이어 쓰기 + 30/29 히스테리시스 경보, 버튼은 ISR 알림 → taskButton, 웹은 `/`(2초 자동 갱신)·`/api`(latest JSON)·`/log.csv`(다운로드)·`/clear`. Wi-Fi 실패 시 AP 모드 `ESP32-MONITOR` |

**핵심 내용**

#### IoT 모니터 구조
> 센서 → 큐 → 로거, 공유 상태는 뮤텍스, 버튼은 ISR → 태스크
- **센서와 로거 사이 큐** → 파일 쓰기가 느려도 **읽기 주기 유지**
- **`latest` · `/log.csv` 를 뮤텍스로** → 로거 · 버튼 · 웹이 동시에 접근
- **`handleApi`** → 뮤텍스를 잡은 채 `latest` 를 **지역 변수로 복사하고 바로 해제** → 잠금 구간을 짧게
- **버튼 처리를 taskButton 으로** → 파일 · 뮤텍스는 ISR 에서 못 쓴다
- **웹 핸들러는 `latest` 를 보냄** → DHT 를 직접 읽지 않아 센서를 과하게 안 읽고 빠르다

#### 시작 순서
> 준비물 먼저, 사용자는 나중
- **setup 순서** → **큐 · 뮤텍스 → 태스크 → `attachInterrupt`** (준비 전 NULL 접근 방지)
- **Wi-Fi 실패 시** → AP 모드 → 공유기 없이도 접속
- **4일 한 문장** → 하드웨어를 **구조체 · 함수로 다루고, 태스크로 나눈다**

**기본 사용형태**
```c
void taskLogger(void *p) {
    reading_t r; bool alarm = false;
    for (;;) {
        xQueueReceive(q, &r, portMAX_DELAY);
        if (isnan(r.temp)) continue;
        xSemaphoreTake(lock, portMAX_DELAY);
        latest = r;
        File f = LittleFS.open("/log.csv", FILE_APPEND);
        if (f) { f.printf("%lu,%.1f,%d\n", r.ms, r.temp, r.pot); f.close(); }
        xSemaphoreGive(lock);
        if (!alarm && r.temp >= 30) alarm = true; else if (alarm && r.temp < 29) alarm = false;
        digitalWrite(LED_R, alarm);
    }
}
void handleApi() {
    xSemaphoreTake(lock, portMAX_DELAY); reading_t r = latest; xSemaphoreGive(lock);   // 복사 후 바로 해제
    char json[96]; snprintf(json, sizeof json, "{\"temp\":%.1f,\"pot\":%d,\"ms\":%lu}", r.temp, r.pot, r.ms);
    server.send(200, "application/json", json);
}
```

**연관 문제**
<<QA 8>>
