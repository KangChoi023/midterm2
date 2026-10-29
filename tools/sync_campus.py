#!/usr/bin/env python3
r"""강사님 교안 저장소(AI_CAMPUS)를 받아 내 과목 폴더(06_HyundaiAI, 구글 드라이브 동기화)로 정리한다.
PC 에서 Claude 에게 "강의자료 정리해줘" 라고 하면 이 프로그램을 돌린다 (.claude/skills/campus-sync).

    py tools\sync_campus.py --dry-run     # 무엇을 할지만 보기
    py tools\sync_campus.py               # 받기 + 정리
    py tools\sync_campus.py --migrate     # 처음 한 번: 예전 방식 폴더를 과목별 _예전정리/ 로 옮기고 정리
    py tools\sync_campus.py --mine        # 내가 모아 둔 자료(이름에 DayN)를 그 Day 폴더의 "7. 내 자료" 로
    … --add "vision=10_Vision=Vision"     # 처음 보는 과목: 강사님 폴더 앞부분=과목 폴더=파일 이름에 쓸 과목명
    … --dest "<06_HyundaiAI 폴더>" --repo "<AI_CAMPUS 를 받아 둘 폴더>"

정리 모양 (Day 폴더 하나)
    10_Vision/Day1 · <사이트 노트 제목>/          ← 과목 폴더를 열면 바로 Day 폴더 (특수 묶음 00 · 24시간 통합본 · 99 · 재평가 · 템플릿 도 같은 층)
        1. 강의 노트북 (Colab) - Vision D1.ipynb  /  1. 강의 노트 (웹) - … .html (바로가기)
                                                ← 노트북 = 내 실습 사본: 맨 앞에 Drive 마운트 셀(→ _원본 으로 이동)을 넣는다
        2. 애니메이션 - Vision D1.html          ← _원본 의 애니메이션을 여는 바로가기
        3. 교안 - Vision D1.pdf · .pptx
        4. 워크시트 - Vision D1.html            ← 바로가기
        5. 실습 코드/                           ← 강사님 src/ 와 맨 위 .py · .c · .zip
        6. 참고 - <원래 이름>.html               ← 그 밖의 html (바로가기)
        7. 내 자료/                             ← --mine: 01_강의자료 · 02_실습자료 에 내가 모아 둔 DayN 자료 (퀴즈 · 판서 · 풀이 노트북 …)
        _원본/                                  ← 나머지 원본 그대로 (html 끼리의 링크가 깨지지 않게)
        _이전판/                                ← 강사님이 고친 파일의 옛 판 (생길 때만)

규칙
- 새 파일만 복사, 내용이 바뀐 파일은 옛 판을 _이전판/ 으로 옮긴 뒤 덮어쓴다. 아무것도 지우지 않는다.
- 예전 위치(과목/01_강의자료/)의 Day 폴더 · 특수 폴더는 과목 바로 아래로 올린다. 같은 이름이 이미 있으면 옮기지 않고 경고.
  01_강의자료 가 비면 치우고, 내 파일이 남아 있으면 그대로 두고 알려 준다.
- 노트북(1. 강의 노트북 · 1. 노트북)은 내 실습 사본이다. 맨 앞에 Colab 마운트 셀을 넣고, 마지막으로 쓴 해시를 _원본/.sync_state.json 에 남긴다.
  내가 고친 노트북은 절대 덮어쓰지 않는다 — 강사님 원본이 바뀌면 옆에 "… (강사님 새 판 YYMMDD).ipynb" 로 따로 둔다.
- 번호를 바꾼 파일(예전 1. 교안 · 3. 강의 노트)은 옛 이름 파일이 있고 새 이름 파일이 없으면 이름만 바꾼다 (다시 복사하지 않음).
- Day 폴더 제목은 notes/<과목>_dayN.md 첫 줄. 노트가 아직 없으면 "DayN" 으로 두었다가 노트가 생기면 이름을 바꾼다.
- 과목을 모르는 폴더는 복사하지 않고 알려 준다 (→ --add).
- --mine: 이름에 DayN 이 있는 내 자료만 옮긴다 (저장한 웹페이지의 …_files 폴더도 이름이 같아 함께 간다).
  강사님 파일과 내용이 똑같은 것은 중복이므로 과목 폴더/_예전정리/ 로. Day 번호가 없는 것은 그대로.
- 내가 만든 파일 · 폴더(03_개인실습, 04_자체학습 …)는 건드리지 않는다.
"""
import argparse, datetime, filecmp, glob, hashlib, json, os, re, shutil, subprocess, sys
from urllib.parse import quote

REPO_URL = 'https://github.com/imguru-mooc/AI_CAMPUS.git'
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # midterm2 폴더

# 강사님 폴더 앞부분(소문자, "-dayN" 앞) → (노트 파일 앞부분, 과목 폴더, 파일 이름에 쓸 과목명)
SUBJECTS = {
    'day': ('ARCH', '01_컴퓨터구조', '컴퓨터구조'),            # Day3/ = 컴퓨터 구조 Day 3
    'os': ('OS', '02_운영체제활용', '운영체제'),
    'python': ('PY', '03_Python', 'Python'),
    'c': ('C', '04_C', 'C'),
    'mcu': ('MCU', '06_MCU', 'MCU'),
    'linux': ('LINUX', '07_LINUX', 'Linux'),
    'embeddedio': ('EIO', '08_Embbed IO', 'EmbeddedIO'),
    'nn': ('NN', '09_NeuralNetwork', 'NN'),
    'vb': ('VB', '10_Vision', 'Vision'),
}
DAY_DIR = re.compile(r'^(.*?)-?day(\d+)$', re.I)                  # NN-day3, Day3 (앞부분 없음 = 컴퓨터 구조)
# Day 가 아닌 묶음: 이름 → (과목 앞부분, 폴더 이름, 파일 이름 표시, 그대로 복사?)
SPECIAL_UNITS = {
    'os24': ('os', '00 · 24시간 통합본', '운영체제 24H', False),
    'pyextra': ('python', '99 · 재평가 · 템플릿', 'Python', True),
}
SKIP = {'.git', '.github', 'README.md', '.gitignore', '.gitattributes', '.nojekyll'}
LECTURES = '01_강의자료'
ORIG, OLD, MIGRATED, MINE = '_원본', '_이전판', '_예전정리', '7. 내 자료'
# 번호를 바꾼 파일: (옛 이름 앞부분, 새 이름 앞부분) — 옛 이름으로 정리된 Day 폴더는 실행할 때 이름만 바꾼다
RENUMBERED = (('1. 교안 - ', '3. 교안 - '), ('3. 강의 노트 (웹) - ', '1. 강의 노트 (웹) - '),
              ('3. 강의 노트북 (Colab) - ', '1. 강의 노트북 (Colab) - '), ('3. 노트북 - ', '1. 노트북 - '))
OLD_DIRS = re.compile(r'Day\d+|OS_24H|1st_re_evaluation|공통')       # 예전 방식이 만든 폴더 이름
DAYNUM = re.compile(r'day\s*0*(\d+)', re.I)
CODE_EXT = {'.py', '.c', '.h', '.zip'}
BAD = re.compile(r'[\\/:*?"<>|]')
STATE = '.sync_state.json'                                        # _원본/ 안: 정리 프로그램이 마지막으로 쓴 노트북 해시
NB_TOP = ('1. 강의 노트북 (Colab) - ', '1. 노트북 - ')
# Day 폴더 노트북 맨 앞에 넣는 셀 — Colab 에서 Drive 를 연결하고 그 Day 폴더의 _원본 으로 이동한다 (__SUBJECT__ · __DAY__ 는 채워 넣음)
MOUNT_CELL = """# [정리 프로그램(sync_campus.py)이 넣은 셀] Drive 연결 → 이 Day 폴더의 _원본 으로 이동 (data/… · img/… 상대 경로가 그대로 동작)
# Colab 밖(내 PC 의 Jupyter)에서는 아무것도 하지 않는다. 이 노트북은 내 실습 사본 — 고쳐도 정리 프로그램이 덮어쓰지 않는다.
import os
SUBJECT, DAY = __SUBJECT__, __DAY__

def find_hyundai(root='/content/drive', depth=5):
    \"\"\"Drive 의 Othercomputers/<컴퓨터>/ · MyDrive/ 아래 depth 단계 안에서 06_HyundaiAI 폴더를 찾는다\"\"\"
    tops = sorted(os.path.join(root, 'Othercomputers', n) for n in (os.listdir(os.path.join(root, 'Othercomputers'))
                  if os.path.isdir(os.path.join(root, 'Othercomputers')) else []))
    for top in tops + [os.path.join(root, 'MyDrive')]:
        level = [top]
        for _ in range(depth + 1):
            nxt = []
            for d in level:
                try: entries = sorted(os.scandir(d), key=lambda e: e.name)
                except OSError: continue
                for e in entries:
                    if e.name.startswith('.') or not e.is_dir(): continue
                    if e.name == '06_HyundaiAI': return e.path
                    nxt.append(e.path)
            level = nxt
    return None

def find_day_dir(root='/content/drive', subject=SUBJECT, day=DAY):
    \"\"\"06_HyundaiAI/<과목>/Day<N> 또는 Day<N> · 제목/_원본 경로 (못 찾으면 None)\"\"\"
    base = find_hyundai(root)
    sub = os.path.join(base, subject) if base else None
    if not sub or not os.path.isdir(sub): return None
    for n in sorted(os.listdir(sub)):
        if (n == f'Day{day}' or n.startswith(f'Day{day} · ')) and os.path.isdir(os.path.join(sub, n, '_원본')):
            return os.path.join(sub, n, '_원본')
    return None

try:
    from google.colab import drive
except ImportError:
    drive = None
if drive:
    drive.mount('/content/drive')
    HERE = find_day_dir()
    if HERE:
        os.chdir(HERE)
        print('작업 폴더:', HERE)
    else:
        print(f'⚠ Drive 에서 06_HyundaiAI/{SUBJECT}/Day{DAY} 폴더를 찾지 못했습니다 — 계속 실행은 됩니다.')
        print('  구글 드라이브(컴퓨터 동기화)에 06_HyundaiAI 가 올라가 있는지 확인하거나, os.chdir("…/_원본") 으로 직접 옮기세요.')"""


def units_of(repo, extra):
    """저장소 → {(과목 앞부분, 묶음 키): [(원본 절대경로, 묶음 안 상대경로)]}, 모르는 것 목록"""
    subjects = dict(SUBJECTS, **extra)
    units, unknown = {}, []
    add = lambda key, src, rel: units.setdefault(key, []).append((src, rel))
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for fn in sorted(files):
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, repo).replace(os.sep, '/')
            parts = rel.split('/')
            if fn in SKIP or parts[0] in SKIP: continue
            top, rest = parts[0], '/'.join(parts[1:])
            m = DAY_DIR.match(top) if len(parts) > 1 else None
            subj = (m.group(1).lower() or 'day') if m else None
            if subj in subjects:                                                 # NN-day3/…
                add((subj, int(m.group(2))), src, rest)
            elif top == 'anim':                                                  # 맨 위 anim/ = 컴퓨터 구조 애니메이션
                d = re.match(r'day(\d+)/(.+)', rest)
                add(('day', int(d.group(1)) if d else 1), src, 'anim/' + (d.group(2) if d else rest))
            elif top == 'OS':
                add(('os', 'os24'), src, rest)
            elif top == '1st_re_evaluation' or re.match(r'^(python|score|sensor)_template', top):
                add(('python', 'pyextra'), src, rel)
            elif len(parts) == 1 and re.match(r'^운영체제활용교안', top):
                add(('os', 'os24'), src, top)
            elif len(parts) == 1 and re.match(r'^컴퓨터구조_Day(\d+)', top):
                add(('day', int(re.match(r'^컴퓨터구조_Day(\d+)', top).group(1))), src, top)
            elif rel == 'worksheet.html':                                        # 맨 위 worksheet.html = Embedded IO Day 5 의 다른 판
                add(('embeddedio', 5), src, 'worksheet (저장소 맨 위).html')
            else:
                unknown.append(top)
    return units, sorted(set(unknown))


def classify(files, label):
    """묶음 안 파일 → ({보일 이름: 원본}, {바로가기 이름: _원본 안 상대경로})"""
    top = {rel: src for src, rel in files if '/' not in rel}
    rels = sorted(rel for _, rel in files)
    shown, links = {}, {}
    moved, linked = set(), set()                                      # 맨 위로 꺼낸 파일 · 바로가기를 만든 html

    def put(name, rel, src):
        if name in shown:                                             # 같은 자리에 두 개 → 원래 이름을 덧붙인다
            b, e = os.path.splitext(name); name = f'{b} ({os.path.basename(rel)}){e}'
        shown[name] = src; moved.add(rel)

    for ext in ('.pdf', '.pptx'):
        for i, r in enumerate(sorted(r for r in top if r.lower().endswith(ext))):
            put(f'3. 교안 - {label}{"" if i == 0 else " (" + os.path.splitext(r)[0] + ")"}{ext}', r, top[r])
    for slot, name, cands in (
            ('2', '애니메이션', (r'animation\.html', r'anim/index\.html', r'index\.html', r'animations\.html')),
            ('1', '강의 노트 (웹)',(r'lecture\.html', r'[^/]*_lecture\.html')),
            ('4', '워크시트', (r'worksheet\.html', r'[^/]*worksheet[^/]*\.html', r'anim/worksheet\.html'))):
        r = next((r for c in cands for r in rels if re.fullmatch(c, r, re.I) and r not in linked), None)
        if r: links[f'{slot}. {name} - {label}.html'] = r; linked.add(r)
    nbs = sorted((r for r in top if r.lower().endswith('.ipynb')), key=lambda r: ('lecture' not in r.lower(), r))
    for i, r in enumerate(nbs):
        put(f'1. 강의 노트북 (Colab) - {label}.ipynb' if i == 0 else f'1. 노트북 - {os.path.splitext(r)[0]}.ipynb', r, top[r])
    for src, rel in files:
        if rel.startswith('src/'): put('5. 실습 코드/' + rel[4:], rel, src)
        elif rel in top and os.path.splitext(rel)[1].lower() in CODE_EXT: put('5. 실습 코드/' + rel, rel, src)
    for r in sorted(top):
        if r.lower().endswith('.html') and r not in linked and r.lower() != 'index.html':
            links[f'6. 참고 - {os.path.splitext(r)[0]}.html'] = r; linked.add(r)
    for src, rel in files:                                            # 나머지(바로가기 대상 html 포함)는 원본 그대로
        if rel not in moved: shown[f'{ORIG}/{rel}'] = src
    return shown, links


def renumber(udir, names, dry):
    """옛 번호로 정리된 파일 → 새 이름으로 바꾼다 (새 이름 파일이 이미 있으면 그대로). 반환 {새 이름: 옛 이름}"""
    out = {}
    for name in names:
        for old, new in RENUMBERED:
            if not name.startswith(new): continue
            was = old + name[len(new):]
            if os.path.isfile(os.path.join(udir, was)) and not os.path.exists(os.path.join(udir, name)):
                out[name] = was
                if not dry: os.rename(os.path.join(udir, was), os.path.join(udir, name))
            break
    return out


def with_mount(src, folder, day):
    """강사님 노트북 + 맨 앞 마운트 셀 (bytes). 이미 drive.mount( 가 있거나 nbformat 4 가 아니면 원본 그대로"""
    with open(src, 'rb') as f: raw = f.read()
    try: nb = json.loads(raw.decode('utf-8'))
    except ValueError: return raw
    cells = nb.get('cells')
    if nb.get('nbformat') != 4 or not isinstance(cells, list): return raw
    if any('drive.mount(' in ''.join(c.get('source', '')) for c in cells): return raw
    code = MOUNT_CELL.replace('__SUBJECT__', repr(folder)).replace('__DAY__', str(day)).split('\n')
    cell = {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': [l + '\n' for l in code[:-1]] + [code[-1]]}
    if nb.get('nbformat_minor', 0) >= 5: cell['id'] = 'sync-campus-mount'
    nb['cells'] = [cell] + cells
    return (json.dumps(nb, ensure_ascii=False, indent=1) + '\n').encode('utf-8')


def sha(data): return hashlib.sha256(data).hexdigest()


def write_nb(path, src, want, dry, stamp, udir, state, name):
    """Day 폴더 노트북 = 내 실습 사본. 내가 고친 것(해시가 기록과 다름)은 절대 덮어쓰지 않는다.
    반환 (종류, 보일 이름): 'new' | 'changed' | 'cell'(기록 없는 옛 노트북에 셀 넣음) | 'side'(강사님 새 판 따로 둠) | 'kept' | None"""
    with open(src, 'rb') as f: raw = f.read()
    rec, s_src = state.get(name), sha(raw)
    def put(p, kind, show=name, out=True):
        if not dry:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'wb') as f: f.write(want)
        state[name] = {'out': sha(want) if out else (rec or {}).get('out'), 'src': s_src}
        return kind, show
    if not os.path.exists(path): return put(path, 'new')
    with open(path, 'rb') as f: cur = f.read()
    if rec is None:                                                   # 기록 전에 정리된 노트북
        if cur == want: state[name] = {'out': sha(want), 'src': s_src}; return None, name
        if cur == raw: return put(path, 'cell')                       # 강사님 원본 그대로 → 셀 넣은 판으로 (_이전판 없이)
        state[name] = {'out': None, 'src': s_src}; return 'kept', name   # 내가 고친 것 → 그대로
    if sha(cur) == rec.get('out'):                                    # 내가 손대지 않음
        if cur == want: return None, name
        if not dry:                                                   # 강사님이 고침 → 옛 판을 _이전판 에
            b, e = os.path.splitext(os.path.basename(path))
            os.makedirs(os.path.join(udir, OLD), exist_ok=True)
            shutil.move(path, os.path.join(udir, OLD, f'{b}.{stamp}{e}'))
        return put(path, 'changed')
    if rec.get('src') == s_src: return None, name                     # 내가 고쳤고 강사님 원본은 그대로
    b, e = os.path.splitext(path)                                     # 내가 고쳤는데 강사님 원본도 바뀜 → 새 판을 옆에
    side = f'{b} (강사님 새 판 {stamp[:6]}){e}'
    return put(side, 'side', os.path.basename(side), out=False)


def shortcut(target, title):
    url = quote(f'{ORIG}/{target}')
    return (f'<!doctype html><meta charset="utf-8"><title>{title}</title>\n'
            f'<meta http-equiv="refresh" content="0; url={url}">\n'
            f'<p><a href="{url}">{title} 열기</a></p>\n')


def note_title(prefix, day):
    try:
        with open(os.path.join(HERE, 'notes', f'{prefix}_day{day}.md'), encoding='utf-8') as f:
            m = re.match(r'##\s*Day\s*\d+\s*[—-]\s*(.+)', f.readline())
    except OSError:
        return None
    if not m: return None
    t = re.sub(r'\s+', ' ', BAD.sub(' ', m.group(1))).strip().rstrip('.')
    return t[:50].rstrip()


def day_dir(base, day, title, dry, renames):
    """과목/01_강의자료 안의 이 Day 폴더 경로 (노트 제목이 새로 생겼으면 이름을 바꾼다)"""
    want = f'Day{day} · {title}' if title else f'Day{day}'
    have = [d for d in (os.listdir(base) if os.path.isdir(base) else [])
            if (d == f'Day{day}' or d.startswith(f'Day{day} · ')) and os.path.isdir(os.path.join(base, d, ORIG))]
    if have and have[0] != want and title:
        renames.append((have[0], want))
        if not dry: os.rename(os.path.join(base, have[0]), os.path.join(base, want))
        return os.path.join(base, want)
    return os.path.join(base, have[0] if have else want)


def lift_plan(dest, subjects):
    """예전 위치(과목/01_강의자료/)의 Day 폴더 · 특수 폴더 → 과목 바로 아래. 반환 [(옛 경로, 새 경로)]"""
    specials = {v[1] for v in SPECIAL_UNITS.values()}
    out = []
    for _, folder, _ in subjects.values():
        old = os.path.join(dest, folder, LECTURES)
        if not os.path.isdir(old): continue
        for d in sorted(os.listdir(old)):
            p = os.path.join(old, d)
            if d in specials and os.path.isdir(p) or (re.fullmatch(r'Day\d+( · .+)?', d) and os.path.isdir(os.path.join(p, ORIG))):
                out.append((p, os.path.join(dest, folder, d)))
    return out


def old_layout(dest):
    """예전 정리 방식이 만든 폴더들 (과목/01_강의자료/DayN 처럼 _원본 이 없는 것)"""
    out = []
    for _, folder, _ in SUBJECTS.values():
        for sub in (LECTURES, '02_실습자료'):
            base = os.path.join(dest, folder, sub)
            if not os.path.isdir(base): continue
            for d in sorted(os.listdir(base)):
                p = os.path.join(base, d)
                if (OLD_DIRS.fullmatch(d) and os.path.isdir(p)
                        and not os.path.isdir(os.path.join(p, ORIG))):
                    out.append((p, os.path.join(dest, folder, MIGRATED, sub, d)))
    for name in ('_새로온자료', '_동기화기록.txt'):
        p = os.path.join(dest, '00_강의 자료', name)
        if os.path.exists(p): out.append((p, os.path.join(dest, '00_강의 자료', MIGRATED, name)))
    return out


def mine_moves(dest, subjects, units):
    """내가 모아 둔 DayN 자료 → [(지금 위치, 옮길 위치, 강사님 파일과 같은가)]"""
    moves = []
    for subj, (prefix, folder, _) in subjects.items():
        for sub in (LECTURES, '02_실습자료'):
            base = os.path.join(dest, folder, sub)
            if not os.path.isdir(base): continue
            for name in sorted(os.listdir(base)):
                p = os.path.join(base, name)
                if name.startswith('_') or (os.path.isdir(p) and (os.path.isdir(os.path.join(p, ORIG)) or OLD_DIRS.fullmatch(name))):
                    continue                                          # 새 Day 폴더 · 예전 방식 폴더 · _예전정리
                m = DAYNUM.search(name)
                if not m or (subj, int(m.group(1))) not in units: continue
                day = int(m.group(1))
                dup = os.path.isfile(p) and any(os.path.getsize(s) == os.path.getsize(p) and filecmp.cmp(s, p, shallow=False)
                                                for s, _ in units[(subj, day)])
                if dup:
                    dst = os.path.join(dest, folder, MIGRATED, sub, name)
                else:
                    udir = day_dir(os.path.join(dest, folder), day, note_title(prefix, day), True, [])
                    dst = os.path.join(udir, MINE, name)
                b, e = os.path.splitext(dst) if os.path.isfile(p) else (dst, '')
                n = 2
                while os.path.exists(dst): dst, n = f'{b} ({n}){e}', n + 1
                moves.append((p, dst, dup))
    return moves


def find_dest():
    """06_HyundaiAI 폴더를 바탕화면 아래에서 찾는다 — Windows (OneDrive 바탕화면 포함) · WSL(/mnt/c)."""
    home = os.path.expanduser('~')
    desktops = [os.path.join(home, 'Desktop'), os.path.join(home, 'OneDrive', 'Desktop'),
                os.path.join(home, 'OneDrive', '바탕 화면'), os.path.join(home, '바탕 화면'),
                '/mnt/c/Users/*/Desktop', '/mnt/c/Users/*/OneDrive/Desktop']
    for depth in range(4):
        for d in desktops:
            hits = [p for p in glob.glob(os.path.join(d, *(['*'] * depth), '06_HyundaiAI')) if os.path.isdir(p)]
            if hits: return hits[0]
    return None


def update_repo(repo):
    if os.path.isdir(os.path.join(repo, '.git')):
        r = subprocess.run(['git', '-C', repo, 'pull', '--ff-only', '-q'], capture_output=True, text=True)
        if r.returncode: print('! 강사님 저장소 받기 실패 — 받아 둔 내용으로 계속:', r.stderr.strip()[:200])
    elif not os.path.exists(repo):
        subprocess.run(['git', 'clone', '-q', REPO_URL, repo], check=True)
    else:
        sys.exit(f'! {repo} 가 git 저장소가 아닙니다. --repo 로 다른 위치를 지정하세요.')
    r = subprocess.run(['git', '-C', repo, 'log', '-1', '--format=%h %ad %s', '--date=format:%Y-%m-%d %H:%M'],
                       capture_output=True, text=True)
    return r.stdout.strip()


def write(path, data, dry, stamp, udir):
    """data = 원본 경로(str) 또는 바로가기 내용(bytes). 반환 'new' | 'changed' | None"""
    if os.path.exists(path):
        with open(path, 'rb') as f: same = filecmp.cmp(data, path, shallow=False) if isinstance(data, str) else f.read() == data
        if same: return None
        if not dry and isinstance(data, str):                         # 강사님이 고친 파일 → 옛 판을 Day 폴더/_이전판 에
            b, e = os.path.splitext(os.path.basename(path))
            os.makedirs(os.path.join(udir, OLD), exist_ok=True)
            shutil.move(path, os.path.join(udir, OLD, f'{b}.{stamp}{e}'))
        kind = 'changed'
    else:
        kind = 'new'
    if not dry:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(data, str): shutil.copy2(data, path)
        else:
            with open(path, 'wb') as f: f.write(data)
    return kind


def main():
    if hasattr(sys.stdout, 'reconfigure'):                            # Windows 명령창(cp949)에서 기호 · 한글 깨짐 방지
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='강사님 교안 → 내 과목 폴더 정리')
    ap.add_argument('--repo', default=os.path.join(os.path.dirname(HERE), 'AI_CAMPUS'), help='강사님 저장소를 받아 둘 위치 (기본: midterm2 옆)')
    ap.add_argument('--dest', help='06_HyundaiAI 폴더 (생략하면 바탕화면 아래에서 찾음)')
    ap.add_argument('--dry-run', action='store_true', help='복사하지 않고 무엇을 할지만 출력')
    ap.add_argument('--no-pull', action='store_true', help='강사님 저장소를 새로 받지 않음')
    ap.add_argument('--migrate', action='store_true', help='예전 방식 폴더를 과목별 _예전정리/ 로 옮긴 뒤 정리')
    ap.add_argument('--mine', action='store_true', help='내가 모아 둔 DayN 자료를 그 Day 폴더의 "7. 내 자료" 로 옮김')
    ap.add_argument('--add', action='append', default=[], metavar='앞부분=과목폴더=과목명',
                    help='처음 보는 과목 (예: vision=10_Vision=Vision)')
    a = ap.parse_args()

    extra = {}
    for s in a.add:
        p = [x.strip() for x in s.split('=')]
        if len(p) != 3: sys.exit(f'! --add 형식: 앞부분=과목폴더=과목명 (받은 값: {s})')
        extra[p[0].lower()] = (p[0].upper(), p[1], p[2])
    dest = a.dest or find_dest()
    if not dest or not os.path.isdir(dest): sys.exit('! 06_HyundaiAI 폴더를 찾지 못했습니다. --dest 로 지정하세요.')
    head = '(받지 않음)' if a.no_pull else update_repo(a.repo)
    stamp = datetime.datetime.now().strftime('%y%m%d-%H%M')
    print(f'강사님 저장소: {head}')
    print(f'내 폴더: {dest}' + ('   (dry-run — 바꾸지 않음)' if a.dry_run else ''))

    olds = old_layout(dest)
    if olds and not a.migrate and not a.dry_run:
        print(f'\n! 예전 방식으로 정리된 폴더가 {len(olds)}개 있습니다 (예: {os.path.relpath(olds[0][0], dest)}).')
        print('  --migrate 를 붙이면 과목 폴더 안 _예전정리/ 로 옮긴 뒤 새 방식으로 정리합니다 (지우지 않음).')
        return
    for src, dst in olds:
        print(f'  옮김: {os.path.relpath(src, dest)} → {os.path.relpath(dst, dest)}')
        if not a.dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            try: os.rmdir(os.path.dirname(src))                       # 비어 버린 02_실습자료 는 치운다
            except OSError: pass
    if olds and not a.migrate:
        print('  (위 폴더들은 --migrate 를 붙여야 옮깁니다)')

    units, unknown = units_of(a.repo, extra)
    subjects = dict(SUBJECTS, **extra)
    lifts, pending = lift_plan(dest, subjects), set()                 # Day 폴더를 01_강의자료/ 에서 과목 바로 아래로
    if lifts: print('\n■ Day 폴더를 과목 바로 아래로 올림')
    for src, dst in lifts:
        if os.path.exists(dst):
            print(f'  ⚠ 옮기지 않음 (같은 이름이 이미 있음): {os.path.relpath(src, dest)}'); continue
        print(f'  폴더 올림: {os.path.relpath(src, dest)} → {os.path.relpath(dst, dest)}')
        if a.dry_run: pending.add(src)
        else: shutil.move(src, dst)
    for old in sorted({os.path.dirname(s) for s, _ in lifts}):
        left = sorted(n for n in os.listdir(old) if os.path.join(old, n) not in pending)
        if not left:
            print(f'  빈 폴더 치움: {os.path.relpath(old, dest)}')
            if not a.dry_run: os.rmdir(old)
        else:
            print(f'  {os.path.relpath(old, dest)} 에 남은 것 {len(left)}개 (그대로 둠): ' + ', '.join(left))

    total = {'new': 0, 'changed': 0, 'cell': 0, 'side': 0, 'kept': 0, 'renamed': 0}
    order = list(subjects)
    for (subj, key) in sorted(units, key=lambda k: (order.index(k[0]), str(k[1]).zfill(3))):
        prefix, folder, label = subjects[subj]
        base = os.path.join(dest, folder)
        renames = []
        if key in SPECIAL_UNITS:
            _, dirname, ulabel, plain = SPECIAL_UNITS[key]
            udir = os.path.join(base, dirname)
            here = os.path.join(base, LECTURES, dirname)
        else:
            ulabel, plain = f'{label} D{key}', False
            udir = day_dir(base, key, note_title(prefix, key), a.dry_run, renames)
            here = '' if not a.dry_run or os.path.isdir(udir) else day_dir(os.path.join(base, LECTURES), key, note_title(prefix, key), True, renames)
        here = here if here in pending else udir                      # dry-run: 아직 올리지 않은 폴더를 보고 비교한다
        if plain:
            shown, links = {rel: src for src, rel in units[(subj, key)]}, {}
        else:
            shown, links = classify(units[(subj, key)], ulabel)
            if not a.dry_run: os.makedirs(os.path.join(udir, ORIG), exist_ok=True)
        moved = renumber(here, list(shown) + list(links), a.dry_run) if os.path.isdir(here) else {}
        at = lambda name: os.path.join(here, *(moved.get(name, name) if a.dry_run else name).split('/'))
        done = {'new': [], 'changed': [], 'cell': [], 'side': [], 'kept': []}
        nbs = [n for n in shown if isinstance(key, int) and not plain and n.startswith(NB_TOP) and n.endswith('.ipynb')]
        try:
            with open(os.path.join(here, ORIG, STATE), encoding='utf-8') as f: state = json.load(f)
        except (OSError, ValueError):
            state = {}
        before = json.dumps(state, sort_keys=True)
        for name in nbs:
            r, show = write_nb(at(name), shown[name], with_mount(shown[name], folder, key), a.dry_run, stamp, udir, state, name)
            if r: done[r].append(show)
        if not a.dry_run and json.dumps(state, sort_keys=True) != before:
            with open(os.path.join(udir, ORIG, STATE), 'w', encoding='utf-8') as f: json.dump(state, f, ensure_ascii=False, indent=1)
        for name, src in shown.items():
            if name in nbs: continue
            r = write(at(name), src, a.dry_run, stamp, udir)
            if r: done[r].append(name)
        for name, target in links.items():
            r = write(at(name), shortcut(target, os.path.splitext(name)[0]).encode('utf-8'), a.dry_run, stamp, udir)
            if r and name not in moved: done[r].append(name)          # 이름 바꾼 바로가기는 제목만 새로 쓴 것
        if not (any(done.values()) or renames or moved): continue
        print(f'\n■ {os.path.relpath(udir, dest)}   새 {len(done["new"])} · 바뀜 {len(done["changed"])}'
              + (f' · 이름 바꿈 {len(moved)}' if moved else ''))
        for o, n in renames: print(f'  폴더 이름: {o} → {n}')
        for n, o in sorted(moved.items()): print(f'  이름 바꿈: {o} → {n}')
        for k, mark in (('changed', '바뀜(옛 판은 _이전판/)'), ('new', '새로'), ('cell', '마운트 셀 넣음'),
                        ('side', '새 판 따로 둠 (내가 고친 노트북은 그대로)'), ('kept', '내가 고친 노트북 — 그대로 둠')):
            names = [n for n in done[k] if not n.startswith(ORIG + '/')]
            for n in sorted(names): print(f'  {mark}: {n}')
            hidden = len(done[k]) - len(names)
            if hidden: print(f'  {mark}: {ORIG}/ 안 {hidden}개')
        for k in done: total[k] += len(done[k])
        total['renamed'] += len(moved)
    print(f'\n합계: 새 파일 {total["new"]}개 · 바뀐 파일 {total["changed"]}개'
          + ''.join(f' · {t} {total[k]}개' for k, t in (('renamed', '이름 바꾼 파일'), ('cell', '마운트 셀 넣은 노트북'),
                                                         ('side', '강사님 새 판 따로 둔 노트북')) if total[k]))

    moves = mine_moves(dest, subjects, units)
    if moves:
        print(f'\n■ 내가 모아 둔 자료 {len(moves)}개' + ('' if a.mine else ' — --mine 을 붙이면 옮깁니다'))
        for src, dst, dup in moves:
            print(f'  {os.path.relpath(src, dest)} → {os.path.relpath(dst, dest)}' + ('   (강사님 파일과 같음 = 중복)' if dup else ''))
            if a.mine and not a.dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                try: os.rmdir(os.path.dirname(src))                   # 비어 버린 02_실습자료 는 치운다
                except OSError: pass
    if unknown:
        print('\n⚠ 과목을 모르는 자료 (복사하지 않음): ' + ', '.join(unknown))
        print('  → 과목 폴더 이름을 정해 --add "앞부분=과목폴더=과목명" 으로 다시 실행하세요 (예: vision=11_Vision=Vision)')


if __name__ == '__main__':
    main()
