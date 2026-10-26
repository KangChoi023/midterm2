## Day 3 — 클래스·파일·시스템 연동

### 1교시 · 클래스 기초 — dict 를 졸업하기

**키워드**: `class`, `__init__`, `self`, `인스턴스`, `메서드`

**개요**: dict 레코드는 키 오타를 실행 전에 못 잡고 판정 함수가 데이터와 떨어져 산다. 클래스는 데이터와 동작을 한 상자에. `self` 는 "지금 이 인스턴스".

**핵심 용어**
- **클래스(class)**: 인스턴스를 찍어 내는 설계도 — 속성(데이터)과 메서드(동작)를 함께 정의
- **인스턴스(instance)**: 클래스로 만든 객체 하나. 인스턴스마다 속성이 독립
- **self**: 메서드를 부른 그 인스턴스. `r1.is_alert()` = `SensorReading.is_alert(r1)`

**핵심 내용**

#### 첫 클래스
> `__init__` 에서 `self.속성` 을 만들고, 메서드는 첫 인자로 `self`
- **dict 오타** → `r["tmep"]` 실행 중 KeyError · **클래스 오타** → `r.tmep` AttributeError (편집기가 미리 경고)

**연결**
- [C D3·1교시 구조체 기초](#/c/3/1) — 데이터만 묶는 struct, 동작까지 묶는 class

### 2교시 · 클래스 심화 — repr·클래스 변수·상속

**키워드**: `__repr__`, `클래스 변수`, `조회 순서`, `상속`, `오버라이드`, `super()`

**개요**: `__repr__` 으로 디버깅 출력. 속성 조회는 인스턴스 → 클래스 순서라 클래스 변수는 전체에 전파. 공통은 부모에, 다른 것만 자식에 — 자식 `__init__` 에서 `super().__init__()` 를 빠뜨리면 부모 속성이 없다.

**핵심 내용**

#### 조회 순서
> 인스턴스 → 클래스 → 부모 클래스
- **클래스 변수 변경** → 인스턴스 변수가 없는 모든 인스턴스에 전파
- ⚠ `self.count = …` 로 쓰면 그 인스턴스에만 새 변수가 생겨 클래스 변수를 가린다

#### 상속
- **오버라이드** → 자식이 같은 이름 메서드를 다시 정의
- ⚠ `super().__init__(…)` 누락 → 부모가 만드는 속성이 없어 AttributeError

### 3교시 · 파일 입출력

**키워드**: `with open`, `모드 r · w · a`, `encoding`, `strip`, `FileNotFoundError`

**개요**: `with open(…) as f:` 로 열고 블록을 나가면 자동으로 닫힌다. `w` 는 여는 순간 기존 내용을 지운다 — 이어쓰기는 `a`. 읽을 땐 줄마다 `strip()`.

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `open()` | `with open("log.txt", "a", encoding="utf-8") as f:` | 파일 열기 — r 읽기 · w 새로 쓰기 · a 이어쓰기 |

**핵심 내용**

#### 쓰기 3박자와 모드
- **열기 → 쓰기 → 닫기** → `with` 가 닫기를 보장
- ⚠ `"w"` 는 열자마자 내용을 지운다 · close 없이 끝나면 버퍼 내용 유실 가능
- **없는 파일** → `FileNotFoundError` → `try` 또는 `Path.exists()` 로 방어

**연결**
- [C D4·1교시 파일 열고 닫기](#/c/4/1) — 같은 모드 r · w · a
- [OS D3·2교시 read()·write()](#/os/3/2) — 그 아래에서 일어나는 페이지 캐시

### 4교시 · JSON 과 CSV

**키워드**: `json.dumps / loads`, `json.dump / load`, `직렬화`, `csv`, `to_dict`

**개요**: JSON 은 dict · list ↔ 문자열(파일) 왕복 — "s 가 붙으면 문자열, 없으면 파일". 인스턴스는 바로 저장할 수 없어 `to_dict()` 로 바꿔서. CSV 는 엑셀용 표.

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `json.dumps()` / `json.loads()` | `s = json.dumps(d)` | 객체 ↔ JSON 문자열 |
| `json.dump()` / `json.load()` | `json.dump(d, f, ensure_ascii=False, indent=2)` | 객체 ↔ JSON 파일 |
| `csv.writer` | `csv.writer(f).writerow([...])` | 표 한 줄 쓰기 |

**핵심 내용**

#### 직렬화
> 메모리의 객체 → 저장 가능한 글자 (그리고 다시 복원)
- ⚠ 클래스 인스턴스는 `json.dumps` 불가(TypeError) → `to_dict()` / `from_dict()` 패턴
- **한글** → `ensure_ascii=False`

### 5교시 · 시스템 연동

**키워드**: `pathlib`, `subprocess`, `시리얼 루프`, `KeyboardInterrupt`

**개요**: 경로는 `pathlib.Path`, 셸 명령은 `subprocess.run`. 시리얼 수신 루프 구조를 파일 리플레이로 연습하고, Ctrl+C(KeyboardInterrupt)로 안전 종료.

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `pathlib.Path` | `Path("logs").mkdir(exist_ok=True)` · `p / "a.txt"` | 경로 조립 · 폴더 만들기 · 존재 확인 |
| `subprocess.run()` | `subprocess.run(["ls", "-l"], capture_output=True, text=True)` | 셸 명령을 코드 안에서 실행하고 결과 받기 |

**핵심 내용**

#### 수신 루프의 뼈대
> 읽기 → strip → 파싱(try) → 저장 — 데이터 소스만 바꾸면 실제 시리얼
- **안전 종료** → `try: … except KeyboardInterrupt:` 에서 파일 닫고 요약 출력

**연결**
- [OS D3·5교시 변수의 수명과 스크립트의 판단](#/os/3/5) — subprocess 도 fork · exec · `$?`(returncode)

### 6교시 · 종합 설계 — SensorLogger

**키워드**: `SensorLogger`, `add / save / load`, `to_dict / from_dict`, `영속성`

**개요**: "측정을 모으고, 파일로 남기고, 다시 불러와 이어서 모으는 로거" — 클래스 + JSON + 파일을 하나로.

**핵심 내용**

#### SensorLogger 설계
- **add** → 측정 인스턴스 추가
- **save** → `[r.to_dict() for r in self.items]` 를 JSON 파일로
- **load** → JSON 을 읽어 `from_dict` 로 인스턴스 복원
- **stats_by_node · report** → 집계 · 출력
- **영속성** → 런타임을 재시작해도 load 로 이어서

### 7교시 · 종합 실습

**키워드**: `저장되는 센서 로거`

**개요**: Day3_worksheet.ipynb 7교시 — 측정 6개 add → save → 재시작 흉내 → load → 이어서 집계.

### 8교시 · 정리

**키워드**: `Day 3 정리`

**개요**: 클래스(self · 상속 · super) → 파일(with · 모드) → JSON/CSV(to_dict) → pathlib · subprocess → SensorLogger.

**연결**
- [Python D4·1교시](#/py/4/1) — 다음 날: NumPy
