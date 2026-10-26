## Day 1 — 라즈베리파이 설정 · I/O 지도

> Embedded IO 과목은 Raspberry Pi 4 실습이라 이 정리의 출력값은 **강의 자료 기준**이다 (PC 에서 실행 불가).

### 1교시 · 과정 소개 · Pi 4와 MCU 비교

**키워드:** Raspberry Pi 4, SBC, MPU, BCM2711, Cortex-A72 4코어, ESP32 비교, 부팅 흐름, EEPROM 부트로더, start4.elf, config.txt, Device Tree, 커널 드라이버 probe, 장치 파일, systemd, 하드웨어 PWM 2채널, ADC 없음

**개요**
- MCU 과목에서 ESP32 에 Arduino API 로 붙였던 **같은 부품(KS3014)** 을 이번에는 **Raspberry Pi 4 의 Linux** 위에서 다룬다. 1교시는 두 보드의 구조·자원 차이와, Pi 가 전원을 넣고 내 프로그램이 핀을 만지기까지 거치는 **부팅 흐름**.

**핵심 용어**
- **SBC(Single Board Computer)**: 보드 하나짜리 컴퓨터. Pi 4 = **BCM2711 SoC, Cortex-A72 4코어 1.5GHz**, LPDDR4 4GB, microSD 64GB, **Linux**.
- **MCU vs MPU**: MCU(ESP32)는 OS 없이 레지스터 직접, MPU 기반 Pi 는 **커널 드라이버 → 장치 파일**로 핀에 접근.
- **Device Tree(.dtb)**: ARM 보드가 커널에 넘기는 **하드웨어 설명서** (5교시).
- **start4.elf**: Pi 의 **GPU 펌웨어** — `config.txt` 를 읽고 Device Tree + overlay 를 준비.

| 항목 | ESP32 (MCU 과목) | Raspberry Pi 4 (이 과목) |
|---|---|---|
| CPU | Xtensa LX6 2코어 240MHz | Cortex-A72 4코어 1.5GHz |
| 메모리 / 저장 | SRAM 520KB / Flash 4MB | LPDDR4 4GB / microSD |
| OS | 없음(FreeRTOS 정도) | Linux |
| 프로그램 올리기 | PC 에서 빌드 → USB 업로드 | **보드 안에서 gcc 로 빌드·실행** |
| 전원 → 내 코드 | 1초 미만 | 약 20~40초 |
| 핀 접근 | 레지스터·Arduino API 직접 | **커널 드라이버 → 장치 파일** |
| 아날로그 입력 | ADC 내장 12비트 | **ADC 없음** → Day 4 **PCF8591**(I2C) |
| 하드웨어 PWM | LEDC 16채널 | **2채널 (GPIO18·19)** |
| 실시간성 | 좋음 (µs 예측 가능) | 보통 (스케줄러 지연) |

**새로 나온 함수/명령** — 5일 로드맵

| Day | 주제 | 결과물 |
|---|---|---|
| 1 | 라즈베리파이 설정과 I/O 지도 | `check_env.sh` |
| 2 | GPIO 출력·입력·이벤트 | `react.c` (반응속도 게임) |
| 3 | PWM·모터·타이밍 | `alarm.c` (거리 경보기) |
| 4 | I2C·SPI·UART | `gateway.c` (Pi ↔ ESP32) |
| 5 | 커널 드라이버와 종합 프로젝트 | `envmon` (systemd 서비스) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 1-2 비교표 | ESP32 와 Pi 4 의 CPU·메모리·OS·빌드·핀 접근·ADC·PWM·실시간성 비교 |
| 1-3 부팅 흐름 | 전원에서 내 프로그램이 장치 파일로 핀을 요청하기까지의 단계 |

**핵심 내용**

#### Pi 의 부팅 흐름
> 전원 → 부트로더 → config.txt · Device Tree → 커널 → 드라이버 → 장치 파일 → 내 프로그램

```text
[ESP32] 전원 ─▶ ROM 부트로더 ─▶ 내 펌웨어 setup()/loop() (레지스터 직접)
[Pi 4]  전원 ─▶ EEPROM 부트로더 ─▶ start4.elf (GPU 펌웨어)
        ─▶ config.txt 읽기 · Device Tree(.dtb) + overlay 적용
        ─▶ Linux 커널 ─▶ 드라이버 probe (gpio, i2c, spi, pwm, uart)
        ─▶ /dev/gpiochip0, /dev/i2c-1 … 장치 파일 생성
        ─▶ systemd ─▶ sshd ─▶ 로그인 ─▶ ./내프로그램 (장치 파일로 요청)
```

- **`config.txt`** → "어떤 하드웨어를 켤지" 를 정함 → 바꾸면 **재부팅해야 반영** (저장만 하면 `/sys/class/pwm` 은 아직 비어 있음)

#### ESP32 와 Pi 의 차이
> Pi 에서는 드라이버가 레지스터를 만지고, 프로그램은 장치 파일로 요청한다
- **Linux 사용자 프로그램** → 레지스터에 직접 쓰는 것이 기본이 아님 (Day 2 3교시에 비교)
- **Pi 가 잘하는 일** → 카메라 영상 처리 · 네트워크 서버, 파일 저장 · 여러 프로그램 동시 실행
- **ESP32** → 정확한 타이밍 · 저전력
- **실제 제품** → 둘을 함께 쓴다 (Day 4 gateway)

**기본 사용형태**
```bash
cat /proc/device-tree/model      # Raspberry Pi 4 Model B Rev 1.5
uname -m                         # aarch64
nproc                            # 4
```

**연관 문제**
<<QA 1>>

### 2교시 · 하드웨어 준비와 전기적 주의

**키워드:** 쿨러 케이스, 40핀 헤더, 물리 번호, BCM 번호, 대체 기능 ALT, GPIO2·3 I2C, GPIO14·15 UART, GPIO18·19 PWM, GPIO8·10·11 SPI, 3.3V 전용, 16mA, 220Ω, 5V 단락 금지, Extension Board, 40-Pin 케이블 빨간 줄, pinout

**개요**
- 쿨러 케이스와 KS3014 GPIO Extension Board 를 연결하고, 40핀 헤더의 **물리 번호 / BCM 번호 / 대체 기능**을 구분한다. Pi 는 GPIO 를 망가뜨리면 **보드 교체 외에 방법이 없으므로** 전기적 규칙이 MCU 과목보다 엄격하다.

**핵심 용어**
- **물리 번호**: 헤더에서의 위치 1~40. **BCM 번호**: SoC 의 GPIO 번호. **이 과목의 코드·명령은 모두 BCM 번호** (물리 11번 = **GPIO17**).
- **대체 기능(ALT)**: 한 핀이 여러 역할. GPIO18 은 일반 입출력도, **PWM0** 도 된다 — 무엇으로 쓸지는 **config.txt overlay 와 드라이버**가 정한다.

| 핀 | 이 과목에서 바꿔 쓰는 기능 | Day |
|---|---|---|
| GPIO2 · 3 (물리 3 · 5) | I2C1 SDA · SCL | 4 |
| GPIO8 · 10 · 11 | SPI0 CE0 · MOSI · SCLK | 4 |
| GPIO14 · 15 (물리 8 · 10) | UART TXD · RXD | 4 |
| GPIO18 · 19 (물리 12 · 35) | PWM0 · PWM1 | 3 |

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `pinout` | 40핀 배치를 그림으로 (없으면 `sudo apt install -y python3-gpiozero`) |
| 옴의 법칙 `I = (3.3 − Vf) / R` | LED 전류 — 220Ω 이면 약 5.9mA |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 2-1 키트 확인·조립 | 방열판 보호 필름, 팬 전원선 위치, 5V 3A 어댑터, USB-C 데이터 케이블 확인 |
| 2-2 40핀 헤더 | 물리 번호 ↔ BCM 번호 표 |
| 2-5 Extension Board 연결 | 40-Pin 케이블 빨간 줄을 **1번 핀(3V3) 쪽**, 보드를 브레드보드 **가운데 홈에 걸쳐**, GND 레일 연결 |

**핵심 내용**

#### 전기적 주의 5가지
> GPIO 는 3.3V 전용, 전류 제한, 단락 금지
- **①** GPIO 는 **3.3V 전용** — 5V 를 넣으면 SoC 가 망가짐 (5V 센서 출력을 그대로 입력에 금지)
- **②** 핀 하나 전류 **16mA 이하**, LED 에는 반드시 **220Ω** 직렬
- **③** **5V ↔ 3V3, 5V ↔ GND 를 맞닿게 하지 않는다** — 배선 변경은 `sudo poweroff` 후 전원 분리
- **④** 5V 를 내보내는 센서(HC-SR04 Echo 등)는 **분압 회로**를 거쳐서만 (Day 3)
- **⑤** 브레드보드 위에 **금속 물건** 금지

#### Extension Board 연결
> 방향과 위치를 전원 전에 확인
- ⚠ **거꾸로 꽂으면** → 이름표가 모두 틀어져 3V3 라고 적힌 곳에 5V
- ⚠ 한쪽 면에만 꽂으면 → 단락 위험 → **가운데 홈에 걸친다**
- ⚠ 쿨러 팬이 헤더 4번(5V)·6번(GND) 형식 → 40-Pin 케이블과 겹침 → 실습 중에는 방열판만
- **LED 배선 순서** → `sudo poweroff` → 전원 분리 → 40-Pin 케이블 · Extension Board → LED · 저항 → 확인 후 전원

**기본 사용형태**
```text
GPIO17 (물리 11) ─▶ 220Ω ─▶ LED 긴 다리 ─ LED 짧은 다리 ─▶ GND 레일
```

**연관 문제**
<<QA 2>>

### 3교시 · OS 설치와 USB 접속 준비

**키워드:** Raspberry Pi Imager, Raspberry Pi OS Lite 64-bit, Trixie Debian 13, 호스트 이름 rpi-XX 좌석 번호, SSH 켜기, 무선 LAN 비움, user-data, enable_usb_gadget, cloud-init, YAML 탭 금지, USB gadget mode, rpi-usb-gadget 드라이버, LED 상태 PWR ACT

**개요**
- 모니터·키보드가 없는 강의실에서 Pi 에 접속하는 두 단계 — 처음에는 **USB-C 케이블 하나로 PC 와 직접(USB gadget mode)**, 그 연결로 들어가 Wi-Fi 를 설정한 뒤 Wi-Fi 로 갈아탄다. 3교시는 SD 카드를 굽고 gadget mode 를 켜서 첫 부팅까지.

**핵심 용어**
- **USB gadget mode**: Pi 가 PC 에게 "나는 USB 랜카드입니다" 라고 소개하는 기능(휴대폰 USB 테더링과 같은 원리). **전원 + 네트워크를 한 줄로**. PC 와 Pi 둘만의 망(Pi = **10.12.194.1**).
- **OS 선택**: **Lite**(데스크톱 없음 — 빠르고 공간 절약) / **64-bit(aarch64)** / **Trixie(Debian 13)** — **libgpiod 2.x** 포함(이전 Bookworm 은 1.6 이라 C API 가 달라 예제가 컴파일 안 됨) / **2025-10-20 이후 이미지** — gadget 패키지 기본 포함.
- **user-data**: bootfs 의 설정 파일. **첫 부팅 때 cloud-init 이 한 번만** 읽는다 → Pi 에 꽂기 전에 고친다.
- **YAML**: user-data 형식. **들여쓰기는 스페이스 2칸, 탭 금지** — 틀리면 **설정 전체가 무시**.

**새로 나온 함수/명령** — Imager 설정

| 설정 | 입력값 | 이유 |
|---|---|---|
| 호스트 이름 | **`rpi-XX`** (좌석 번호) | Wi-Fi 에서 40대가 같은 망 → 겹치면 남의 보드 접속 |
| 사용자·비밀번호 | 강사 안내값 | 기본 계정이 없음 |
| 무선 LAN | **비워 둔다** | 4교시에 USB 로 들어가 `nmcli` 로 설정 |
| 로케일 | Asia/Seoul, 키보드 us | 로그·빌드 시각 |
| SSH | **켜기**, 비밀번호 인증 | 빠뜨리면 USB 로도 접속 불가 |

```yaml
rpi:
  enable_usb_gadget: true      # user-data 맨 끝 (스페이스 2칸)
```

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 3-2 SD 굽기 | Imager 로 Raspberry Pi 4 · OS Lite 64-bit · microSD 선택 후 설정 입력 → 쓰기·검증 |
| 3-4 user-data | bootfs 의 user-data 끝에 `enable_usb_gadget: true` 추가 (rpi: 키 중복 금지) |
| 3-5 PC 준비 | Windows 용 `rpi-usb-gadget-driver-setup.exe` 설치, PuTTY·FileZilla |
| 3-6 첫 부팅 | Pi USB-C ↔ **PC 본체 USB 포트**(데이터 케이블)로 전원 — 스스로 한 번 재부팅, 2~3분 |

**핵심 내용**

#### SD 굽기와 첫 부팅
> Imager 로 굽고 user-data 를 고쳐 USB gadget 으로 부팅
- **순서** → Imager 로 굽기 (좌석 번호 · SSH · Wi-Fi 비움) → SD 를 다시 꽂아 bootfs 열기 → user-data 수정 → Pi 에 꽂기 → PC USB 로 부팅
- **첫 부팅 전원** → **PC 의 USB 포트** (어댑터 안 씀), 허브·모니터 포트 말고 본체 포트
- ⚠ **충전 전용 USB-C 케이블** → 불은 켜져도 데이터 선이 없어 접속 불가

#### LED 상태 읽기
> 빨강 = 전원, 초록 = SD 접근
- **빨강 PWR 계속 켜짐** → 전원 정상 / **깜빡이거나 꺼짐** → PC 포트 전력 부족
- **초록 ACT 불규칙** → SD 읽는 중 (정상)
- **초록 4번씩 반복** → 부팅 파일 없음 → **다시 굽기**
- **전혀 안 깜빡임** → SD 덜 꽂힘

#### 빠뜨린 설정 → 증상
> 증상으로 무엇을 빠뜨렸는지 안다
- **user-data gadget 설정** → PC 에 USB 랜카드가 **아예 안 생김**
- **Windows 드라이버** → 장치 관리자에 느낌표
- **SSH** → `Connection refused`

**기본 사용형태**
```text
[1단계 · 3~4교시]  PC ══ USB-C 데이터 케이블 ══ Pi   (전원 + 네트워크, Pi = 10.12.194.1)
[2단계 · 4교시 끝~] PC ── 강의장 Wi-Fi ── Pi          (전원은 5V 3A 어댑터)
```

**연관 문제**
<<QA 3>>

### 4교시 · USB 접속과 Wi-Fi 설정

**키워드:** ncpa.cpl, Remote NDIS, PuTTY rpi-07.local, 10.12.194.1, mDNS, 호스트 이름 충돌 rpi-07-2, raspi-config nonint do_hostname, usb0, vcgencmd get_throttled measure_temp, nmcli, do_wifi_country KR, rfkill, nmcli dev wifi connect, wlan0, apt full-upgrade, rpi-usb-gadget off, 5V 3A 어댑터

**개요**
- USB gadget 연결로 PuTTY 접속해 기본 상태를 확인하고, `nmcli` 로 Wi-Fi 를 설정한 뒤 **Wi-Fi 접속 + 어댑터 전원**으로 갈아탄다. USB 망과 Wi-Fi 망에서 **호스트 이름 충돌**이 다른 이유를 이해한다.

**핵심 용어**
- **usb0**: USB 랜카드 인터페이스(`10.12.194.1/28`). **wlan0**: Wi-Fi.
- **mDNS(`.local`)**: `rpi-07.local` 처럼 이름으로 찾는 방식. 이름이 겹치면 먼저 켜진 보드가 이름을 갖고 나중 보드는 `rpi-07-2.local`.
- **망의 범위**: USB gadget 은 **케이블마다 PC 1대 ↔ Pi 1대**라 40대가 모두 10.12.194.1 이어도 충돌 없음. 강의장 Wi-Fi 는 **40대가 한 망**이라 이름이 겹치면 충돌 → **남의 보드에 접속**.
- **throttled**: 저전압·과열 기록. `0x0` 이 정상. PC USB 전원이면 `0x50005` 처럼 나올 수 있음.
- **NetworkManager / nmcli**: Trixie 의 네트워크 관리자와 명령. 한 번 접속에 성공하면 **저장되어 재부팅 후에도 자동 연결**.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `ncpa.cpl` (Windows) | 새 어댑터 `Raspberry Pi USB Remote NDIS Network Device` 확인 |
| PuTTY `rpi-07.local` / `10.12.194.1` 포트 22 | USB 접속 (이름이 안 풀리면 IP) |
| `sudo raspi-config nonint do_hostname rpi-07` | 호스트 이름 변경(`/etc/hostname`·`/etc/hosts`) |
| `ip -4 addr show usb0` / `wlan0` | IP 확인 |
| `vcgencmd get_throttled` / `measure_temp` | 전원 기록 / SoC 온도 |
| `sudo raspi-config nonint do_wifi_country KR` | **무선 국가 코드 — 가장 먼저** (없으면 rfkill 차단, 목록 비어 있음) |
| `sudo nmcli dev wifi rescan` / `nmcli dev wifi list` | 주변 AP 검색 |
| `sudo nmcli dev wifi connect "SSID" password "PW"` | 접속 (숨김 SSID 는 `hidden yes`) |
| `nmcli dev status` / `sudo nmtui` | 상태 / 메뉴 방식 |
| `sudo apt update && sudo apt full-upgrade -y` | 업데이트 |
| `sudo rpi-usb-gadget off` | gadget mode 끄기 (**Wi-Fi 접속 확인 후에만**) |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 4-2 PuTTY USB 접속 | `rpi-07.local` 로 SSH → `pi@rpi-07:~ $` |
| 4-4 기본 확인 | `trixie`, `aarch64`, `Asia/Seoul`, `usb0 10.12.194.1/28`, `throttled=0x50005`, `temp=45.2'C`, SD 사용량 |
| 4-5 Wi-Fi 설정 | 국가 코드 → 스캔 → connect → `wlan0 connected`, IP `192.168.0.37` |
| 4-7 갈아타기 | USB 꽂은 채 재부팅해 자동 연결 확인 → Wi-Fi IP 로 새 세션 → `rpi-usb-gadget off` → poweroff → 어댑터로 부팅 → `throttled = 0x0` |

**핵심 내용**

#### USB → Wi-Fi 갈아타기
> Wi-Fi 접속을 확인한 뒤에만 gadget 을 끈다
- **순서** → nmcli 로 연결 → **USB 꽂은 채 재부팅해 자동 연결 확인** → Wi-Fi IP 로 PuTTY 새 세션 → 그 세션에서 `rpi-usb-gadget off` → `poweroff` → USB 분리 → **5V 3A 어댑터**로 부팅
- ⚠ Wi-Fi 확인 전에 gadget 을 끄면 → 들어갈 길이 없어진다
- **어댑터로 바꾸는 이유** → PC USB 전원은 **Day 3 모터 · 스테퍼에 부족**, **throttled 저전압 기록** 제거
- **끄기** → `sudo poweroff` → 초록 LED 가 10번 깜빡이고 꺼질 때까지 기다린 뒤 전원 분리

#### 자주 막히는 곳
> 증상 → 원인
- **`nmcli dev wifi list` 가 비어 있음** → 무선 국가 코드 없음
- **`Secrets were required`** → Wi-Fi 비밀번호 오류
- **`Connection refused`** → SSH 안 켜고 구움
- **`Access denied`** → 사용자 이름 오타
- **`.local` 접속 후 호스트 키 경고 · 낯선 파일** → **이름 충돌** → 좌석 번호로 고치고 재부팅

**기본 사용형태**
```bash
hostname                                                   # rpi-07 인지
sudo raspi-config nonint do_wifi_country KR
sudo nmcli dev wifi connect "강의장SSID" password "비밀번호"
nmcli dev status; ip -4 addr show wlan0
vcgencmd get_throttled                                     # 어댑터 전원이면 0x0
```

**연관 문제**
<<QA 4>>

### 5교시 · 인터페이스 활성화와 Device Tree

**키워드:** raspi-config nonint, do_i2c do_spi do_serial_hw do_serial_cons, 0 = 켜기, Device Tree, .dtb, dtparam, dtoverlay, overlays README, /boot/firmware/config.txt, [all], pwm-2chan, disable-bt, 줄 끝 주석 금지, hciuart, PL011 ttyAMA0, mini UART ttyS0, /dev/serial0, 재부팅

**개요**
- I2C·SPI·UART 를 `raspi-config` 명령 모드로 켜고, ARM 보드의 **Device Tree** 와 `dtparam`·`dtoverlay` 의 관계를 이해한다. 메뉴에 없는 PWM 과 정확한 UART(PL011)는 **config.txt 를 직접** 고쳐 설정하고 재부팅 후 장치 파일로 확인한다.

**핵심 용어**
- **raspi-config nonint**: 메뉴 대신 명령 모드. 주의: **0 이 켜기, 1 이 끄기**.
- **Device Tree**: ARM 보드가 커널에 넘기는 **하드웨어 설명서**. 설명서에 없는 장치에는 드라이버가 붙지 않는다. (x86 PC 는 BIOS/ACPI)
- **dtparam**: 기본 Device Tree 에 **이미 있는 장치의 설정값**을 바꿈 (`i2c_arm=on`).
- **dtoverlay**: Device Tree 에 **조각(overlay)을 덧붙임** (`/boot/firmware/overlays/*.dtbo`, 설명은 overlays/README).
- **PL011 vs mini UART**: PL011(`ttyAMA0`)은 독립 클럭이라 **보율이 정확**, 기본은 Bluetooth 에 연결. mini UART(`ttyS0`)는 CPU 클럭에 묶여 보율이 흔들림, 기본으로 GPIO14·15 에. → **`disable-bt`** 로 PL011 을 GPIO 로 돌려받음.

```text
bcm2711-rpi-4-b.dtb     ← 보드 기본 설명서 (I2C·SPI 는 있지만 꺼짐)
 + dtparam=i2c_arm=on   ← 기본 설명서의 I2C 를 켜짐으로
 + dtoverlay=pwm-2chan  ← PWM 장치를 덧붙이고 GPIO18·19 를 PWM 기능으로
 = 최종 Device Tree ──▶ 커널 ──▶ 드라이버 probe ──▶ /dev, /sys 장치 생성
```

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `sudo raspi-config nonint do_i2c 0` / `do_spi 0` / `do_serial_hw 0` | I2C / SPI / UART 하드웨어 켜기 |
| `sudo raspi-config nonint do_serial_cons 1` | **시리얼 로그인 콘솔은 끄기** (UART 를 우리 프로그램이 씀) |
| `sudo nano /boot/firmware/config.txt` | config.txt 편집 (Trixie 위치) — 먼저 `.bak` 백업 |
| `sudo systemctl disable hciuart` | disable-bt 와 짝 — 블루투스 UART 서비스 끄기 |
| `ls /dev/i2c-* /dev/spidev* /dev/serial0` / `ls /sys/class/pwm/` | 재부팅 후 확인 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 5-1 raspi-config | I2C·SPI·UART 켜고 시리얼 콘솔 끄기 |
| 5-3 config.txt | `[all]` 아래 `dtoverlay=pwm-2chan`, `dtoverlay=disable-bt` 추가 → `hciuart` 비활성 → 재부팅 |
| 5-5 확인 | `/dev/i2c-1 /dev/spidev0.0 /dev/serial0`, `pwmchip0`, `serial0 -> ttyAMA0`, `console=tty1` |

**핵심 내용**

#### config.txt 설정
> 인터페이스를 켜고 overlay 를 추가한 뒤 재부팅

```text
[all]
dtparam=i2c_arm=on        ← raspi-config 가 넣은 줄 → /dev/i2c-1
dtparam=spi=on            ←                        → /dev/spidev0.0
enable_uart=1
dtoverlay=pwm-2chan       ← 직접 추가 → /sys/class/pwm/pwmchip0 (GPIO18=PWM0, 19=PWM1)
dtoverlay=disable-bt      ← 직접 추가 → serial0 -> ttyAMA0 (PL011)
```

- **순서** → config.txt 백업 → nano 로 열기 → `[all]` 아래 추가 → 저장 → `hciuart` 비활성 → **재부팅** → 장치 파일 확인
- ⚠ **줄 끝 주석 미지원** → `dtoverlay=pwm-2chan # PWM` 은 주석까지 이름으로 읽혀 적용 안 됨 → 주석은 별도 줄
- ⚠ 재부팅 후 `serial0 -> ttyS0` → disable-bt 가 반영 안 돼 **mini UART** 가 붙음 → config.txt 다시 확인
- ⚠ 시리얼 로그인 콘솔을 켜 두면 → Day 4 UART 통신을 콘솔이 가로챈다 → 끈다

**기본 사용형태**
```bash
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_serial_hw 0
sudo raspi-config nonint do_serial_cons 1
sudo cp /boot/firmware/config.txt /boot/firmware/config.txt.bak
sudo systemctl disable hciuart && sudo reboot
ls -l /dev/serial0                 # -> ttyAMA0
```

**연관 문제**
<<QA 5>>

### 6교시 · 개발 환경 구성

**키워드:** build-essential, gpiod, libgpiod-dev 2.x, i2c-tools, pkg-config --modversion libgpiod, libgpiod 1.x vs 2.x, gpiod_line_request_output, 장치 파일 권한, crw-rw----, 문자 장치, 주 번호 부 번호, gpio i2c spi dialout 그룹, usermod -aG, FileZilla SFTP, 공통 Makefile, pkg-config --libs, %: %.c, 01_board_info

**개요**
- libgpiod 2.x 개발 환경을 설치하고 버전을 확인한다. 장치 파일을 **sudo 없이** 쓸 수 있는 이유(그룹 권한)를 이해하고, Day 1~5 공통 Makefile 로 첫 C 프로그램 `01_board_info` 를 Pi 위에서 빌드한다.

**핵심 용어**
- **libgpiod**: GPIO 문자 장치를 다루는 라이브러리·명령. **1.x 와 2.x 의 C API 가 완전히 다르다** — 인터넷 예제 대부분은 1.x(`gpiod_line_request_output`)라 이 환경에서 **컴파일 안 됨**.
- **문자 장치(character device)**: `ls -l` 맨 앞 **`c`** — 바이트 단위로 읽고 씀. `254, 0` = **주 번호(어느 드라이버) · 부 번호(몇 번째 장치)**.
- **그룹 권한**: `crw-rw---- root gpio` → 소유자와 **gpio 그룹**만 읽고 씀 → 사용자가 gpio 그룹이면 sudo 없이 GPIO. Imager 로 만든 첫 사용자는 gpio·i2c·spi·dialout 에 이미 들어 있다.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `sudo apt install -y build-essential gpiod libgpiod-dev i2c-tools` | gcc·make / gpio 명령 / C 헤더·라이브러리 / i2cdetect |
| `pkg-config --modversion libgpiod` | **2.2.1** — 반드시 2.x |
| `pkg-config --cflags --libs libgpiod` | gcc 옵션 → `-lgpiod` |
| `ls -l /dev/gpiochip0 /dev/i2c-1 /dev/spidev0.0 /dev/ttyAMA0` / `groups` | 권한 / 내 그룹 |
| `sudo usermod -aG gpio,i2c,spi,dialout $USER` | 그룹 추가 (**다시 로그인해야** 반영) |
| `access(path, R_OK \| W_OK)` | 내 권한으로 열 수 있는지 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 공통 `Makefile` | `LDLIBS = $(shell pkg-config --libs libgpiod) -lpthread`, 패턴 규칙 `%: %.c` 로 `make 이름` |
| `01_board_info.c` | `/proc/device-tree/model`(모델), `uname()`(커널), `sysconf`(코어 수), `/sys/class/thermal/thermal_zone0/temp`(온도 ÷1000), `access("/dev/gpiochip0")`(권한)을 출력 |

**핵심 내용**

#### 보드 정보를 파일로 읽기
> Device Tree · sysfs 를 파일처럼 읽는다

```text
모델     : Raspberry Pi 4 Model B Rev 1.5
커널     : Linux 6.12.47+rpt-rpi-v8 (aarch64)
CPU 코어 : 4
SoC 온도 : 46.7 °C
GPIO 칩  : /dev/gpiochip0 읽기·쓰기 가능
```

- **`/proc/device-tree/model`** → Device Tree 를 커널이 파일로 보여 줌
- **온도 `46712`** → **46.712°C** (밀리도) — Linux Day4 sysfs 와 같은 원리
- **`access` 성공** → gpio 그룹 덕분
- ⚠ `usermod` 로 그룹 추가 → **지금 열린 PuTTY 세션에는 반영 안 됨** → 다시 로그인

#### 빌드와 실행
> FileZilla 로 올리고 make 로 빌드
- **순서** → `mkdir -p ~/eio/day1` → FileZilla (SFTP, `rpi-07.local`, 22) 로 src 업로드 → `cd ~/eio/day1/src` → `make 01_board_info` → `./01_board_info`
- ⚠ Makefile 명령 줄이 **스페이스로 시작** → `missing separator` → **탭** (Linux Day1 3교시)
- **`$@`** → 대상 / **`$<`** → 첫 재료

**기본 사용형태**
```make
CC     = gcc
CFLAGS = -Wall -Wextra -O2 -g
LDLIBS = $(shell pkg-config --libs libgpiod 2>/dev/null) -lpthread
SRCS   = $(wildcard *.c)
BINS   = $(SRCS:.c=)
all: $(BINS)
%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
```

**연관 문제**
<<QA 6>>

### 7교시 · 장치 파일로 보는 I/O 지도

**키워드:** 하드웨어는 파일, /dev/gpiochip0 pinctrl-bcm2711, /sys/class/pwm, /dev/i2c-1, /dev/spidev0.0, /dev/ttyAMA0, gpiodetect, gpioinfo, gpioset, -c gpiochip0, offset = BCM 번호, -t 500ms 토글, 라인 요청, consumer, busy, 자동 해제, sysfs GPIO 폐기, dmesg

**개요**
- "Linux 에서 하드웨어는 파일이다." 인터페이스별 **장치 파일과 커널 드라이버**를 짝짓고, libgpiod 명령(`gpiodetect`·`gpioinfo`·`gpioset`)으로 라인 상태를 보고 LED 를 켠다. 한 라인을 두 프로세스가 요청하면 생기는 **busy** 와 라인 소유권을 이해한다.

**핵심 용어**
- **장치 파일 ↔ 드라이버**: MCU 는 내 코드가 레지스터를 직접 썼지만, Linux 는 **드라이버만 레지스터를 만지고** 내 프로그램은 `open·read·write·ioctl` 로 요청.

| 하드웨어 | 장치 파일 | 커널 드라이버 | Day |
|---|---|---|---|
| GPIO | `/dev/gpiochip0` | pinctrl-bcm2711 | 2·3·5 |
| 하드웨어 PWM | `/sys/class/pwm/pwmchip0/` | pwm-bcm2835 | 3 |
| I2C | `/dev/i2c-1` | i2c-bcm2835 | 4 |
| SPI | `/dev/spidev0.0` | spi-bcm2835 + spidev | 4 |
| UART | `/dev/ttyAMA0` (`/dev/serial0`) | amba-pl011 | 4 |
| 온습도 DHT11 | `/sys/bus/iio/devices/…` | dht11 | 5 |
| LED·버튼 (overlay) | `/sys/class/leds/`, `/dev/input/event0` | leds-gpio, gpio-keys | 5 |

- **라인(line)과 offset**: gpiochip0 의 라인 번호(offset) = **BCM 번호** (물리 번호 아님).
- **라인 요청과 consumer**: 라인을 요청한 프로세스가 **consumer** 로 표시되고, 다른 프로세스가 같은 라인을 요청하면 **`Device or resource busy`**. 프로세스가 끝나면(fd 가 닫히면, `kill -9` 여도) **커널이 자동 해제**.
- **sysfs GPIO(`/sys/class/gpio`) 폐기**: 전역 번호가 커널마다 바뀜(17 → 529), 죽어도 export 가 남음, 누가 쓰는지 모름, 엣지 불편 → **문자 장치 + libgpiod**.

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `gpiodetect` | 칩 목록 — `gpiochip0 [pinctrl-bcm2711] (58 lines)` |
| `gpioinfo -c gpiochip0 [17]` | 라인 이름·방향·consumer |
| `gpioset -c gpiochip0 17=1` | 켜기 — **끝나지 않고 기다림**(라인을 쥐고 있으려고), Ctrl+C 로 해제 |
| `gpioset -c gpiochip0 -t 500ms 17=1` | 0.5초마다 토글하며 계속 실행 |
| `gpioset GPIO17=1` | 라인 이름으로 지정 |
| `dmesg \| grep -iE "gpio\|pwm\|spi\|i2c\|ttyAMA"` | 드라이버가 장치를 잡은 기록 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 7-2 칩과 라인 | `gpiodetect`, `gpioinfo` 로 GPIO7·8 이 이미 `consumer="spi0 CS1/CS0"` 인 것 확인 |
| 7-5 LED 켜기 | GPIO17(물리 11) → 220Ω → LED, `gpioset` 으로 켜기·토글 |
| 7-6 busy | 창 1 `gpioset 17=1` 중 창 2 `gpioinfo 17` → `consumer="gpioset"`, 창 2 `gpioset 17=0` → busy, 창 1 Ctrl+C 후 다시 성공 |

**핵심 내용**

#### 커널이 핀 소유권을 관리한다
> 다른 드라이버가 쓰는 라인은 요청할 수 없다
- **`gpioset -c gpiochip0 8=1`** → **busy** — SPI 를 켰기 때문에 **spidev 가 GPIO8 (CS0)** 을 쓰는 중
- **MCU** → 두 코드가 같은 핀을 써도 아무도 막지 않음 / **Linux** → 커널이 소유권 관리
- **`gpioinfo` 로 알 수 있는 것** → 라인 이름(GPIO17), 방향(input/output), 라인을 쥔 consumer

**기본 사용형태**
```bash
gpiodetect
gpioinfo -c gpiochip0 17
gpioset -c gpiochip0 17=1             # Ctrl+C 로 해제
gpioset -c gpiochip0 -t 500ms 17=1    # 0.5초 토글
```

**연관 문제**
<<QA 7>>

### 8교시 · 미니 프로젝트 — 버튼 입력과 check_env.sh

**키워드:** 내부 pull-up, bias, floating, gpioget -b pull-up, active inactive, falling 엣지, gpiomon -e falling -p 10ms -n 3, 커널 타임스탬프, 바운싱, CPU 0%, 셸 파이프 버튼→LED, check_env.sh, [ OK ] [FAIL] [WARN], [ -e ] vs [ -f ]

**개요**
- GPIO5 에 버튼을 달아 **내부 pull-up** 으로 읽고, `gpiomon` 으로 falling 엣지·커널 타임스탬프·바운싱을 관찰한다. 오늘 설정을 한 번에 점검하는 `check_env.sh` 가 모두 `[ OK ]` 여야 Day 2 C 실습을 시작할 수 있다.

**핵심 용어**
- **bias(pull-up/pull-down)**: 핀을 내부 저항으로 3.3V/GND 쪽에 당겨 둠. 버튼이 GND 쪽이면 **pull-up**: 안 누름 1(active), **누름 0(inactive)** → 누르는 순간 **1 → 0 = falling**. (pull-down 을 쓰면 안 눌러도 0, 눌러도 0 이라 1 이 안 읽힘)
- **floating**: bias 없이 핀이 떠 있으면 값이 0 과 1 을 오감.
- **바운싱(bouncing)**: 한 번 눌렀는데 접점이 떨려 몇 ms 차이로 엣지가 여러 번 → `-p 10ms`(디바운스).
- **커널 타임스탬프**: `gpiomon` 맨 앞 숫자 `1260.412335071` = 커널이 엣지를 받은 시각(초.나노초).

**새로 나온 함수/명령**

| 이름 | 용도 |
|---|---|
| `gpioget -c gpiochip0 -b pull-up 5` | pull-up 으로 읽기 → `"5"=active` / `inactive` (`--numeric` 이면 1/0) |
| `gpiomon -c gpiochip0 -b pull-up -e falling 5` | 누를 때마다 한 줄 |
| `-e both` / `-p 10ms` / `-n 3` | 누름·뗌 모두 / **10ms 디바운스** / 3번 받고 끝 |

**예제 코드가 하는 일**

| 예제 | 하는 일 |
|---|---|
| 8-2 gpioget · gpiomon | pull-up 으로 안 누름 1·누름 0, bias 없이 들쭉날쭉, gpiomon 에 바운싱으로 두 줄 |
| 8-3 셸로 버튼 → LED | `gpiomon … \| while read -r line; do timeout 0.3 gpioset … 17=1; done` — 누를 때마다 0.3초 켬 (누를 때마다 fork·exec 라 느림 → Day 2 C 로) |
| `check_env.sh` | ① 보드와 OS ② 전원과 온도 ③ 네트워크와 호스트 이름 ④ 인터페이스 장치 파일 ⑤ 사용자 권한(그룹) 순서로 점검해 `[ OK ]`/`[WARN]`/`[FAIL]` 출력 |

**핵심 내용**

#### 이벤트 대기와 셸 방식의 한계
> 커널이 엣지 때만 깨워 준다 — 셸 파이프는 매번 프로세스를 만들어 느리다
- **`gpiomon` 대기 중** → **CPU 0%** (Linux Day4 inotify 와 같은 이벤트 방식)
- ⚠ 셸 파이프 방식 → 누를 때마다 gpioset 프로세스를 새로 만들어 (fork·exec) **반응이 느리고** 정밀 측정 불가 → Day 2 C 프로그램

#### check_env.sh 판정
> 장치 파일은 `[ -e ]` 로 검사한다
- **`[ -e ]`** → 장치 파일은 일반 파일이 아니라 `[ -f ]` 가 **거짓**
- **`[FAIL]`** → libgpiod-dev 가 1.x, wlan0 미연결, gpio 그룹 아님, 장치 파일 없음
- **`[WARN]`** → throttled ≠ 0x0, 호스트 이름 비규칙, gadget 켜짐

**기본 사용형태**
```bash
gpioget -c gpiochip0 -b pull-up --numeric 5                 # 1 / 0
gpiomon -c gpiochip0 -b pull-up -e falling -p 10ms 5
ok()   { printf "  [ OK ] %s\n" "$*"; PASS=$((PASS + 1)); }
fail() { printf "  [FAIL] %s\n" "$*"; FAILN=$((FAILN + 1)); }
check_node() { if [ -e "$1" ]; then ok "$1 ($2)"; else fail "$1 없음 — $3"; fi; }
check_node /dev/gpiochip0 "GPIO" "커널 문제"
```

**연관 문제**
<<QA 8>>
