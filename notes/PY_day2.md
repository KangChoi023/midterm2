## Day 2 — 자료구조와 함수

### 1교시 · 리스트와 튜플 — 참조와 복사

**키워드**: `b = a`, `얕은 복사`, `깊은 복사`, `sort vs sorted`, `튜플`, `언패킹`

**개요**: `b = a` 는 복사가 아니라 이름표 공유. `copy()` · `a[:]` · `list(a)` 는 겉 리스트만 새로 만드는 얕은 복사라 안쪽 리스트는 여전히 공유 — 끝까지 복사하려면 `deepcopy`. 튜플은 불변이고 언패킹으로 스왑 · 별표 받기.

**핵심 용어**
- **얕은 복사(shallow copy)**: 겉 컨테이너만 새로 만들고 안의 객체는 공유
- **깊은 복사(deep copy)**: 안쪽 객체까지 재귀적으로 새로 만든다 — `copy.deepcopy()`

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `list.sort()` | `a.sort()` | 제자리 정렬, **None 반환** |
| `sorted()` | `b = sorted(a)` | 정렬된 새 리스트 반환, 원본 유지 |
| `copy.deepcopy()` | `b = copy.deepcopy(a)` | 중첩까지 완전 복사 |

**핵심 내용**

#### 참조 공유 → 얕은 복사 → 깊은 복사
> `b = a` (같은 객체) → `a[:]` (겉만 새로) → `deepcopy` (전부 새로)
- ⚠ `a = a.sort()` → a 가 None 이 된다

#### 튜플과 언패킹
- **불변** → 요소 대입 불가
- **스왑** → `a, b = b, a` · **별표** → `first, *rest = xs`

**연결**
- [Python D1·2교시 변수와 자료형](#/py/1/2) — 변수 = 이름표의 연장
- [C D2·3교시 배열과 포인터](#/c/2/3) — C 에서 같은 "주소 공유"

### 2교시 · 딕셔너리와 집합

**키워드**: `dict`, `KeyError`, `.get()`, `중첩 구조`, `set`, `| & - ^`

**개요**: 딕셔너리는 키 → 값. 같은 키 재대입은 덮어쓰기, 없는 키는 KeyError 이므로 `.get(키, 기본값)` 으로 방어. 대괄호 하나가 중첩 한 층. 집합은 중복 제거와 합 · 교 · 차 · 대칭차.

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `dict.get()` | `d.get("temp", -1)` | 없는 키면 기본값 (KeyError 대신) |
| `dict.items()` | `for k, v in d.items():` | 키 · 값 함께 순회 |
| `set()` | `set([1, 1, 2])` | 중복 제거된 집합 |

**핵심 내용**

#### 딕셔너리
- **키 중복 불가** → 재대입 = 덮어쓰기
- **중첩** → `data["sensors"][0]["value"]` — 대괄호 하나 = 한 층

#### 집합 연산
> `a | b` 합집합 · `a & b` 교집합 · `a - b` 차집합 · `a ^ b` 대칭차

### 3교시 · 함수 기초

**키워드**: `def`, `return`, `None`, `기본값 인자`, `키워드 인자`, `*args`, `**kwargs`, `스코프`

**개요**: return 이 없으면 None. 기본값 · 키워드 인자, `*args` · `**kwargs`, 다중 반환은 튜플. 함수 안의 대입은 지역 변수. 가변 기본값(`def f(x=[])`)은 호출 사이에 공유되는 함정.

**핵심 용어**
- **스코프(scope)**: 이름이 보이는 범위. 함수 안에서 대입한 이름은 지역 변수
- **스택 프레임(stack frame)**: 함수 호출마다 생겼다 return 하면 사라지는 지역 변수 공간

**핵심 내용**

#### 인자와 반환
- **`def f(a, b=10, *args, **kwargs)`** → 위치 · 기본값 · 남는 위치 인자(튜플) · 남는 키워드 인자(dict)
- **다중 반환** → `return lo, hi` = 튜플 → `lo, hi = f()`

#### 가변 기본값 함정
> 기본값은 정의할 때 한 번만 만들어진다
- ⚠ `def add(x, acc=[])` → 호출마다 같은 리스트에 쌓임 → `acc=None` 후 `if acc is None: acc = []`

**연결**
- [C D1·7교시 함수와 스코프](#/c/1/7)

### 4교시 · 함수 심화 — 람다·컴프리헨션

**키워드**: `lambda`, `key=`, `리스트 컴프리헨션`, `dict 컴프리헨션`, `map`, `filter`

**개요**: 람다는 key 자리에 들어가는 작은 함수. 컴프리헨션은 for + append 를 한 줄로. map/filter 는 이터레이터를 돌려준다.

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `lambda` | `sorted(xs, key=lambda s: s["t"])` | 이름 없는 한 줄 함수 |
| `map()` | `list(map(float, parts))` | 각 요소에 함수 적용 (이터레이터) |
| `filter()` | `list(filter(lambda v: v > 0, xs))` | 조건 참인 요소만 (이터레이터) |

**핵심 내용**

#### 컴프리헨션
> `[식 for x in xs if 조건]` · `{k: v for k, v in pairs}`
- **for 문과 같은 일** → 새 리스트 만들기 + 필터
- ⚠ map/filter 결과는 한 번 쓰면 소진되는 이터레이터 — 다시 쓰려면 `list()` 로

### 5교시 · 모듈과 표준 라이브러리

**키워드**: `import`, `from … import`, `as`, `__name__ == "__main__"`, `모듈 캐시`, `Counter`, `datetime`

**개요**: `%%writefile` 로 .py 모듈을 만들어 import. 직접 실행하면 `__main__` 블록이 돌고 import 하면 안 돈다. 수정이 반영 안 되면 모듈 캐시 — `importlib.reload`.

**새로 나온 함수**
| 함수 | 기본형 | 용도 |
|---|---|---|
| `importlib.reload()` | `importlib.reload(mymod)` | 이미 import 한 모듈을 다시 읽기 |
| `collections.Counter` | `Counter(xs).most_common(3)` | 개수 세기 |
| `datetime.now()` | `datetime.now().strftime("%F %T")` | 현재 시각 · 서식 |

**핵심 내용**

#### `if __name__ == "__main__":`
> 직접 실행할 때만 도는 시험 코드 자리
- ⚠ 모듈을 고쳐도 같은 세션에서 다시 import 하면 캐시된 옛 모듈 → `reload`

### 6교시 · 예외 처리

**키워드**: `try`, `except`, `else`, `finally`, `raise`, `사용자 정의 예외`, `건너뛰고 계속`

**개요**: 구체적인 예외별로 잡고(`except ValueError as e`), 성공 시 else, 항상 finally. raise 로 던지고 잡을 때까지 호출 스택을 거슬러 전파. 깨진 데이터는 건너뛰고 계속하는 패턴.

**핵심 내용**

#### try 네 칸
> try → (실패) except → (성공) else → (항상) finally
- **구체적으로 잡기** → `except (ValueError, IndexError) as e:` — 맨 `except:` 는 버그까지 삼킨다
- **사용자 정의** → `class FrameError(Exception): pass` → `raise FrameError("bad")`
- **건너뛰고 계속** → for 안에서 try/except → 실패 줄은 세고 `continue`

**연결**
- [C D4·6교시 코드 품질 — 에러 규약](#/c/4/6) — C 는 반환값으로, Python 은 예외로

### 7교시 · 종합 실습

**키워드**: `센서 데이터 처리`, `자동 채점`

**개요**: Day2_worksheet.ipynb 7교시 — 리스트 · 딕셔너리 · 함수 · 예외 처리를 엮어 센서 기록을 집계하는 과제.

### 8교시 · 정리

**키워드**: `Day 2 정리`

**개요**: 참조와 복사, dict · set, 함수 인자 · 스코프 · 가변 기본값, 람다 · 컴프리헨션, 모듈, 예외.

**연결**
- [Python D3·1교시](#/py/3/1) — 다음 날: 클래스
