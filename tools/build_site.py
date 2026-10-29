#!/usr/bin/env python3
"""정리 노트(.src/*.md) + 강의(lecture.html) + 워크시트(worksheet.html) -> 위키 데이터(data/*.js).

    python3 -I tools/build_site.py            # 전 과목
    python3 -I tools/build_site.py c linux    # 일부 과목

경로 기본값은 ~/c_study 기준. 바꾸려면 환경변수 STUDY_ROOT.
노트 원본은 저장소의 notes/ 가 있으면 그것을, 없으면 ~/c_study/summary/.src 를 쓴다.
결과:
    data/index.js   과목·Day·교시 목차, 검색 색인, 용어/함수/연결 사전
    data/<id>.js    과목별 교시 본문(노트 섹션, 강의 토픽, 예제 코드, 문항)
"""
import os, re, sys, json, datetime, html as htmlmod
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wsparse, lecture_parse, nbparse

REPO = os.path.dirname(HERE)
ROOT = os.environ.get('STUDY_ROOT', os.path.expanduser('~/c_study'))
CAMPUS = os.path.join(ROOT, 'AI_CAMPUS')
NOTES = os.path.join(REPO, 'notes') if os.path.isdir(os.path.join(REPO, 'notes')) else os.path.join(ROOT, 'summary', '.src')

# 배운 순서. 경로는 AI_CAMPUS 기준이고 목록이면 앞에서부터 있는 파일을 쓴다.
# lecture: .html(lecture_parse) · .ipynb/.pptx(nbparse).  anim_dir=True 면 애니메이션 파일이 든 폴더를 통째로 복사.
# ws_practice=True: 워크시트가 문항형이 아니라 칸 채우기 실습지(예측표·기록지) → 퀴즈로 옮기지 않는다.
SUBJECTS = [
    dict(id='arch', name='컴퓨터 구조', short='구조', days=3, flow='컴퓨터 부품과 숫자 표현',
         lecture='컴퓨터구조_Day{d}.pptx', ws=['Day{d}/day{d}_worksheet.html'], ws_practice=True, note='ARCH_day{d}.md',
         anim=['Day{d}/index.html', 'anim/day{d}/index.html', 'anim/index.html'], anim_dir=True),   # 맨 위 anim/ = Day 1
    dict(id='os', name='운영체제 활용', short='OS', days=3, flow='파일 시스템 · 프로세스 · 셸',
         lecture='os-day{d}/운영체제_Day{d}.pptx', ws=['os-day{d}/worksheet.html', 'os-day{d}/anim/worksheet.html'], ws_practice=True, note='OS_day{d}.md',
         anim='os-day{d}/anim/index.html', anim_dir=True),
    dict(id='py', name='Python', short='PY', days=4, flow='Colab과 파이썬 기초',
         lecture='python-day{d}/Day{d}_lecture.ipynb', ws='python-day{d}/anim/worksheet.html', note='PY_day{d}.md',
         anim='python-day{d}/anim/index.html', anim_dir=True),
    dict(id='c', name='C 언어', short='C', days=4, flow='C 언어 문법과 메모리',
         lecture='C-day{d}/C_Day{d}_lecture.html', ws='C-day{d}/anim/worksheet.html', note='C_day{d}.md', anim='C-day{d}/anim/index.html'),
    dict(id='linux', name='LINUX', short='LINUX', days=5, flow='리눅스 시스템 프로그래밍',
         lecture='Linux-day{d}/lecture.html', ws='Linux-day{d}/worksheet.html', note='LINUX_day{d}.md', anim='Linux-day{d}/animation.html'),
    dict(id='mcu', name='MCU', short='MCU', days=4, flow='ESP32 마이크로컨트롤러',
         lecture='MCU-day{d}/MCU_Day{d}_lecture.html', ws='MCU-day{d}/worksheet.html', note='MCU_day{d}.md', anim='MCU-day{d}/index.html'),
    dict(id='eio', name='Embedded IO', short='EIO', days=5, flow='라즈베리파이 I/O 제어',
         lecture='EmbeddedIO-day{d}/lecture.html', ws='EmbeddedIO-day{d}/worksheet.html', note='EIO_day{d}.md', anim='EmbeddedIO-day{d}/animation.html'),
    dict(id='nn', name='Neural Network', short='NN', days=5, flow='NumPy로 만드는 신경망',
         lecture='NN-day{d}/NN_Day{d}_lecture.ipynb', ws='NN-day{d}/worksheet.html', note='NN_day{d}.md', anim='NN-day{d}/animation.html'),
    dict(id='vb', name='Vision 기본 모델', short='VB', days=5, flow='PyTorch로 이미지 · 영상 인식',
         lecture='VB-day{d}/VB_Day{d}_lecture.ipynb', ws='VB-day{d}/worksheet.html', note='VB_day{d}.md', anim='VB-day{d}/animation.html'),
]

def campus_path(pat, d):
    """경로 패턴(또는 후보 목록) → 실제로 있는 첫 파일. 없으면 첫 후보 경로."""
    pats = pat if isinstance(pat, list) else [pat]
    paths = [os.path.join(CAMPUS, x.format(d=d)) for x in pats]
    return next((x for x in paths if os.path.exists(x)), paths[0])

# 노트 소제목 -> 정해진 키 (앞부분 일치)
SECTION_KEYS = [('키워드', 'kw'), ('개요', 'overview'), ('핵심 용어', 'terms'), ('새로 나온', 'funcs'),
                ('이 교시에서 만드는', 'funcs'), ('예제 코드가 하는 일', 'examples'), ('핵심 내용', 'core'),
                ('기본 사용형태', 'usage'), ('연결', 'links'), ('연관 문제', 'quiz')]
HEAD_RE = re.compile(r'^\*\*([^*]+?):?\*\*:?(.*)$')

def strip_md(t):
    t = re.sub(r'```[\w+-]*\n?', '', t)
    t = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', t)
    t = re.sub(r'[*`|]', ' ', t)
    t = re.sub(r'^[-\s]*-{3,}.*$', '', t, flags=re.M)
    return re.sub(r'\s+', ' ', t).strip()

def fix_links(md):
    """로컬 폴더 링크(예: [02_shell/](Linux_day1/02_shell/))는 사이트에서 깨지므로 코드 표기로.
    코드 블록(```)과 인라인 코드(`…`) 안의 `a[i](x)` 같은 코드는 건드리지 않는다."""
    out = []
    for i, part in enumerate(re.split(r'(```[\s\S]*?```)', md)):
        if i % 2: out.append(part); continue
        segs = re.split(r'(`[^`\n]*`)', part)
        out.append(''.join(seg if j % 2 else re.sub(r'\[([^\]]+)\]\((?!#|https?:)([^)\s]+)\)', r'`\1`', seg)
                           for j, seg in enumerate(segs)))
    return ''.join(out)

def parse_notes(path):
    """Day 노트 하나 -> {title, lessons:{n:{title, sections:[{key,label,inline,md}]}}}"""
    txt = open(path, encoding='utf8').read()
    m = re.search(r'^## Day \d+\s*[—-]\s*(.+)$', txt, re.M)
    day = {'title': m.group(1).strip() if m else '', 'lessons': {}}
    for blk in re.split(r'^### ', txt, flags=re.M)[1:]:
        head, _, body = blk.partition('\n')
        hm = re.match(r'(\d+)교시\s*[·:-]?\s*(.*)', head.strip())
        if not hm: continue
        n = int(hm.group(1)); secs = []; cur = None
        in_code = False
        for line in body.split('\n'):
            if line.startswith('```'): in_code = not in_code
            mm = None if in_code else HEAD_RE.match(line.strip())
            key = None
            if mm:
                label = mm.group(1).strip()
                for pre, k in SECTION_KEYS:
                    if label.startswith(pre): key = k; break
                if key is None and not line.startswith('**Q') and (mm.group(2).strip() == '' or mm.group(2).lstrip().startswith(('(', ':'))):
                    key = 'extra'
            if key:
                rest = mm.group(2).strip()
                inline = ''
                if rest.startswith(':'): inline = rest[1:].strip(); rest = ''
                elif key in ('kw', 'overview') and rest: inline = rest.lstrip(': '); rest = ''
                cur = {'key': key, 'label': (label + (' ' + rest if rest else '')).strip(), 'md': inline}
                secs.append(cur)
            elif cur is not None:
                cur['md'] += '\n' + line
            elif line.strip():
                cur = {'key': 'extra', 'label': '', 'md': line}; secs.append(cur)
        secs = [dict(s, md=fix_links(s['md'].strip()), label=fix_links(s['label'])) for s in secs if s['key'] != 'quiz']
        day['lessons'][n] = {'title': hm.group(2).strip(), 'sections': [s for s in secs if s['md'] or s['key'] == 'extra']}
    return day

TERM_RE = re.compile(r'^\s*-\s*\*\*(.+?)\*\*\s*(\([^)]*\))?\s*[:：]\s*(.+)$')

def parse_terms(md):
    out = []
    for line in md.split('\n'):
        m = TERM_RE.match(line)
        if m:
            name = m.group(1).replace('`', '').strip()
            if m.group(2): name += ' ' + m.group(2)
            out.append({'name': name, 'def': m.group(3).strip()})
    return out

# 일정·설정·핀 분류·비교 표 — 노트에는 남기고 함수 색인에서만 뺀다
NON_FUNC_TABLES = {'', 'Day', '설정', '분류', '모드', '구성', '값', '필요'}

def parse_funcs(md):
    out = []
    lines = md.split('\n')
    sep = re.compile(r'^\|[\s:|-]+\|$')
    skip = False
    for k, line in enumerate(lines):
        if not line.startswith('|') or sep.match(line): continue
        cells = [c.strip().replace('\x00', '|') for c in line.replace('\\|', '\x00').strip().strip('|').split('|')]
        if k + 1 < len(lines) and sep.match(lines[k + 1].strip()):              # 표 머리글 행
            skip = cells[0] in NON_FUNC_TABLES; continue
        if skip or len(cells) < 2: continue
        if not re.search(r'[A-Za-z0-9가-힣_]', cells[0]): continue       # "→" 같은 기호만 있는 행은 함수가 아님
        out.append({'name': cells[0].replace('`', '').strip(), 'sig': cells[1] if len(cells) > 2 else '',
                    'use': cells[-1]})
    return out

def parse_kw(md):
    # 백틱 안의 쉼표는 구분자가 아니다
    parts, buf, inq = [], '', False
    for ch in md:
        if ch == '`': inq = not inq; continue
        if ch == ',' and not inq: parts.append(buf); buf = ''; continue
        buf += ch
    parts.append(buf)
    return [p.strip() for p in parts if p.strip()]

def parse_links(md):
    return [{'text': t, 'href': h} for t, h in re.findall(r'\[([^\]]+)\]\((#/[^)]+)\)', md)]

CODE_FILE = re.compile(r'[\w./-]+\.(?:c|h|py|sh|ino|cpp)\b')

def example_rows(sections):
    """노트 '예제 코드가 하는 일' 표 → [(첫 칸, 하는 일)]"""
    out = []
    for s in sections:
        if s['key'] != 'examples': continue
        for line in s['md'].split('\n'):
            if not line.startswith('|') or re.match(r'^\|[\s:|-]+\|$', line.strip()): continue
            cells = [c.strip() for c in line.replace('\\|', '\x00').strip().strip('|').split('|')]
            if len(cells) < 2 or re.match(r'^(예제|파일|프로그램|이름)$', strip_md(cells[0])): continue
            out.append((cells[0].replace('\x00', '|'), cells[-1].replace('\x00', '|')))
    return out

def row_topic(cell, n, topics):
    """표의 첫 칸("1-1 `hello.c`", "`02_nice_race.c`") → 강의 토픽 id  (assets/concepts.js rowTopic 과 같은 규칙)"""
    t = strip_md(cell)
    def by_title(pat):
        for tp in topics:
            if re.search(pat, tp['title']): return tp['id']
        return None
    m = re.search(r'(\d+)-(\d+)', t)
    if m:
        r = by_title(r'예제\s*' + m.group(1) + '-' + m.group(2) + r'(?!\d)')
        if r is not None: return r
    for f in CODE_FILE.findall(t):
        base = f.split('/')[-1]
        for tp in topics:
            if base in tp.get('files', []): return tp['id']
        k = re.match(r'^0?(\d+)_', base)
        if k:
            r = by_title(r'예제\s*(' + str(n) + '-' + k.group(1) + '|0?' + k.group(1) + r')(?!\d)')
            if r is not None: return r
    return None

def attach_figs(topics, n, intro):
    """예제마다 붙일 배선·보드 그림: ① 자기 그림 ② 같은 교시에서 가장 가까운 앞 토픽의 그림 ③ 이 교시가 포함된 Day 공통 그림
    결과는 topic['figRefs'] = [{'t': 'self'} | {'t': 'topic', 'id': n} | {'t': 'intro', 'k': n}]"""
    for i, tp in enumerate(topics):
        if not any(b['kind'] in ('code', 'cmd') for b in tp['blocks']): continue
        if '<figure>' in tp['html']: tp['figRefs'] = [{'t': 'self'}]; continue
        prev = next((p for p in reversed(topics[:i]) if '<figure>' in p['html']), None)
        if prev: tp['figRefs'] = [{'t': 'topic', 'id': prev['id']}]; continue
        ks = [k for k, f in enumerate(intro) if n in f['lessons']]
        if ks: tp['figRefs'] = [{'t': 'intro', 'k': k} for k in ks]

def code_index(sid, d, n, topics, sections):
    """오른쪽 💻 코드 목차용: 코드·명령이 있는 토픽마다 [과목, day, 교시, 토픽 id, 파일명, 하는 일, 강의 번호]"""
    rows = {}
    for cell, what in example_rows(sections):
        tid = row_topic(cell, n, topics)
        if tid is not None and tid not in rows: rows[tid] = (cell, what)
    out = []
    for tp in topics:
        if not any(b['kind'] in ('code', 'cmd') for b in tp['blocks']): continue
        row = rows.get(tp['id'])
        files = tp.get('files') or ([CODE_FILE.findall(strip_md(row[0]))[0].split('/')[-1]] if row and CODE_FILE.findall(strip_md(row[0])) else [])
        num = (re.match(r'^(\d+-\d+)\.', tp['title']) or re.search(r'예제\s*(\d+-\d+)', tp['title']))
        bare = re.sub(r'^예제\s*[\d-]+\s*[·—-]\s*', '', re.sub(r'^\d+-\d+\.\s*', '', tp['title'])).strip()
        what = strip_md(row[1]) if row else (tp.get('desc') or bare)
        out.append([sid, d, n, tp['id'], ' · '.join(files[:1]) + (f' 외 {len(files) - 1}' if len(files) > 1 else ''), what[:120], num.group(1) if num else '', bare, 1 if tp.get('figRefs') else 0, len(tp.get('guide', []))])
    return out

def review_quiz(sid, d, n):
    """summary/.src/review_quiz.json (tools/reviewquiz.py 가 만듦) 에서 이 교시 문항 → quiz 형식"""
    out = []
    for q in REVIEW.get(sid, {}).get(str(d), []):
        if q.get('n') != n or q.get('a') is None: continue
        w = re.sub(r'^\s*(정답입니다\.|정답:\s*[①②③④⑤]?[^—]*—)\s*', '', q.get('w', ''))
        w = re.sub(r'\s*\[복습:[^\]]*\]\s*$', '', w).strip()
        item = {'ty': 'mc', 'p': htmlmod.escape(q['p']), 'o': q['o'], 'a': q['a'], 'w': htmlmod.escape(w), 'src': 'review-wiki' if q.get('by') == 'wiki' else 'review'}
        if q.get('c'): item['c'] = q['c']
        out.append(item)
    return out


def load_json(name):
    """name.json 과 name.*.json (Day 별로 나눠 쓴 파일) 을 모두 합친다. 키가 겹치면 뒤 파일이 이김."""
    import glob
    base = os.path.join(NOTES, name)
    out = {}
    for p in [base] + sorted(glob.glob(base[:-5] + '.*.json')):
        if not os.path.exists(p): continue
        try: out.update({k: v for k, v in json.load(open(p, encoding='utf8')).items() if k != '_설명'})   # _exit 같은 이름은 살린다
        except Exception as e: print('  !', os.path.basename(p), e)
    return out

API = {}      # api_detail.json  (main 에서 읽음)
API_BY = {}   # 과목 id → 그 과목 파일(api_detail.<id>*.json)만 합친 것 — 이름이 같은 다른 과목 상세보다 먼저 본다
REVIEW = {}   # review_quiz.json (복습 퀴즈, main 에서 읽음)
GUIDE = {}    # code_guide.json
UNSUPPORTED_WS = []   # 아직 읽는 법이 없는 워크시트 형식

def api_detail(name, sid=None):
    """함수 표 이름 → 상세. 이름 전체 → 첫 낱말(sudo 건너뜀) → 괄호 앞 식별자 순서로 찾는다.
    자기 과목 파일을 먼저 보고, 없으면 다른 과목 것 (C free ≠ Linux free -h)."""
    n = name.replace('`', '').strip()
    words = [w for w in n.split() if w != 'sudo']
    m = re.match(r'^([A-Za-z_][\w.]*)\s*\(', n)
    keys = [n] + words[:1] + ([m.group(1)] if m else [])
    for D in (API_BY.get(sid, {}), API):
        for k in keys:
            if k in D: return D[k]
    return None

EX_SKIP = re.compile(r'^\s*(#(?:\s|!|$)|//|/\*|\*|>>>?\s*$|"""|\'\'\')')
EX_DEF = re.compile(r'^\s*(def|class|void|int|char|long|float|double|static|unsigned|bool|size_t|ssize_t|pid_t|esp_err_t)\s+\**\s*[A-Za-z_][\w.]*\s*\([^)]*\)\s*[{:;]?\s*$|^\s*class\s+\w+\s*(\([^)]*\))?\s*:\s*$')
EX_NOISE = re.compile(r'\b(perror|fprintf|printf|print|puts|Serial\.print\w*)\s*\(\s*f?["\']')

def ex_code(t):
    """줄 끝 주석을 뺀 코드 부분 — 주석 속 낱말로 고르지 않게"""
    return re.split(r'\s+(?:#|//)\s', t, maxsplit=1)[0]

def ex_keys(name):
    """함수 표 이름 → 코드에서 찾을 낱말 후보 (긴 것 먼저). 'np.dot()' → ['np.dot', 'dot'], 'sudo apt install …' → ['apt']"""
    n = name.replace('`', '').strip()
    m = re.match(r'^(?:sudo\s+)?([A-Za-z_][\w.-]*)', n)
    if not m: return []
    w = m.group(1).rstrip('.-')
    keys = [w] + ([w.split('.')[-1]] if '.' in w else [])
    return [k for k in keys if len(k) >= 2]

def attach_examples(sid, S, rows):
    """과목 교안의 코드 줄에서 함수마다 '실제로 쓰인 한 줄'을 골라 f['ex'] 와 색인 행에 붙인다.
    같은 Day · 가까운 교시 → 호출 모양 → 짧은 줄 순. api_detail 의 "ex" 로 직접 지정("" 이면 끔)."""
    lines = []          # (day, 교시, 토픽 id, 토픽 제목, 줄, 블록 종류)
    for D in S['days']:
        for L in D['lessons']:
            for tp in L['topics']:
                for b in tp['blocks']:
                    if b['kind'] not in ('code', 'cmd'): continue
                    for ln in b['text'].split('\n'):
                        t = ln.strip()
                        if 4 <= len(t) <= 110 and not EX_SKIP.match(t) and '___' not in t:     # 빈칸 문제 줄은 뺀다
                            lines.append((D['n'], L['n'], tp['id'], tp['title'], t, b['kind']))
    hit = 0
    for d, n, f, row in rows:
        det = f.get('detail') or {}
        want = det.get('ex')
        if want == '': continue
        best, mk = None, ''
        if want:
            c = [x for x in lines if want in x[4]]
            best = min(c, key=lambda x: (abs(x[0] - d) * 100 + abs(x[1] - n), len(x[4])), default=None)
            if not best: print(f'  ! 예제 줄 못 찾음: {sid} {f["name"]} — ex={want!r}')
        for k in ([] if want else ex_keys(f['name'])):
            word = re.compile(r'(?<![\w-])' + re.escape(k) + r'(?![\w-])')
            call = re.compile(r'(?<![\w-])' + re.escape(k) + r'\s*\(')
            c = [x for x in lines if word.search(ex_code(x[4]))]
            if not c: continue
            def rank(x):
                t = ex_code(x[4])
                used = (call.search(t) or (x[5] == 'cmd' and re.match(r'^(\$\s*)?(sudo\s+)?' + re.escape(k) + r'\b', t))) and not EX_DEF.match(t)
                noise = any(m.group(1) != k for m in EX_NOISE.finditer(t))    # 다른 출력 함수의 "문자열" 안에 이름만 나온 줄
                return (0 if used else 1, 1 if noise else 0, abs(x[0] - d) * 100 + abs(x[1] - n), len(t))
            best, mk = min(c, key=rank), k; break
        if not best: continue
        # 강조할 낱말 — 코드 패널(codeHl)처럼 앞뒤가 영문·숫자·_ 가 아닌 자리에 있어야 강조된다 (my_strcmp 속 strcmp 는 안 됨)
        alone = lambda w: re.search(r'(?<![A-Za-z0-9_])' + re.escape(w) + r'(?![A-Za-z0-9_])', best[4])
        mk = mk or next((w for w in reversed(ex_keys(f['name'])) if alone(w)), '')   # np.sum → sum (코드 색칠이 np · . · sum 으로 나눈다)
        if re.fullmatch(r'[A-Za-z_][\w.]*\.\w+', mk): mk = mk.split('.')[-1]
        if not mk:      # '%02X' · '조건' 처럼 이름으로 못 찾으면 그 줄의 가장 긴 낱말로 (코드 패널이 이 줄로 이동하는 데 쓴다)
            code = ex_code(best[4])
            ws = re.findall(r'(?<![A-Za-z0-9_])[A-Za-z_]\w+', code) or code.split()
            mk = max(ws, key=len) if ws else ''
        ex = [best[0], best[1], best[2], best[4], best[3], mk]     # day, 교시, 토픽 id, 줄, 토픽 제목, 강조할 낱말
        f['ex'] = ex; row.append(ex); hit += 1
    return hit, len(rows)

EX_STAT = []

GUIDE_MISS = {}   # 키 → 못 찾은 단계 (어느 토픽에서도 못 찾았을 때만 경고)
GUIDE_HIT = set()

def attach_guides(sid, d, topics, n=0):
    """code_guide.json 의 단계(at/to 글자)를 실제 코드 블록의 줄 범위로 바꿔 topic['guide'] 에.
    같은 파일명이 본문에만 언급된 토픽도 있으므로, 그 파일의 코드 블록이 실제로 있는 토픽에만 붙인다."""
    for tp in topics:
        for f in tp.get('files', []):
            key = f'{sid}/{d}/{n}/{f}' if f'{sid}/{d}/{n}/{f}' in GUIDE else f'{sid}/{d}/{f}'   # 같은 Day 에 같은 파일명이 있으면 교시 번호로 구분
            steps = GUIDE.get(key)
            if not steps: continue
            if not any(b['kind'] == 'code' and b.get('file') == f for b in tp['blocks']): continue
            miss = []
            out, pos = [], {}
            for st in steps:
                hit = None
                order = [i for i, b in enumerate(tp['blocks']) if b['kind'] == 'code' and b.get('file') == f] + \
                        [i for i, b in enumerate(tp['blocks']) if b['kind'] == 'code' and b.get('file') != f]
                for bi in order:
                    lines = tp['blocks'][bi]['text'].split('\n')
                    start = pos.get(bi, 0)
                    a = next((i for i in range(start, len(lines)) if st['at'] in lines[i]), None)
                    if a is None: continue
                    z = a
                    if st.get('to'):
                        z = next((i for i in range(a, len(lines)) if st['to'] in lines[i]), None)
                        if z is None: continue
                    hit = (bi, a, z); break
                if not hit:
                    miss.append(f'{st["t"]} (at={st["at"]!r})'); continue
                bi, a, z = hit; pos[bi] = z + 1
                out.append({'b': bi, 'from': a, 'to': z, 't': st['t'], 'd': st.get('d', '')})
            if out: tp['guide'] = out; GUIDE_HIT.add(key)
            if miss: GUIDE_MISS.setdefault(key, miss)

def build_subject(cfg, index):
    sid = cfg['id']; S = {'id': sid, 'name': cfg['name'], 'days': []}
    meta = {'id': sid, 'name': cfg['name'], 'short': cfg['short'], 'flow': cfg.get('flow', ''), 'days': []}
    fn_rows = []        # (day, 교시, 함수, 색인 행) — 과목을 다 읽은 뒤 교안 예제 줄을 붙인다
    for d in range(1, cfg['days'] + 1):
        lp = campus_path(cfg['lecture'], d)
        wp = campus_path(cfg['ws'], d)
        np_ = os.path.join(NOTES, cfg['note'].format(d=d))
        html_lec = lp.endswith('.html')
        lp_ok = os.path.exists(lp)
        lec = (lecture_parse.parse(lp) if html_lec else nbparse.parse(lp)) if lp_ok else [None] * 8
        ws = []
        if os.path.exists(wp) and not cfg.get('ws_practice'):
            if wsparse.supported(wp):
                try: ws = wsparse.load(wp)
                except Exception as e: print('  ! worksheet', wp, e)
            else: UNSUPPORTED_WS.append(os.path.relpath(wp, CAMPUS))
        notes = parse_notes(np_) if os.path.exists(np_) else {'title': '', 'lessons': {}}
        dtitle = notes['title'] or ((lecture_parse.day_title(lp) if html_lec else nbparse.day_title(lp, d)) if lp_ok else '')
        figs = lecture_parse.intro_figs(lp) if lp_ok and html_lec else []
        D = {'n': d, 'title': dtitle, 'lessons': [], 'figs': figs}
        pending_code = []
        anim = copy_anim(cfg, d)
        dm = {'n': d, 'title': dtitle, 'lessons': []}
        for n in range(1, 9):
            L = lec[n - 1] if n - 1 < len(lec) else None
            N = notes['lessons'].get(n)
            Q = ws[n - 1]['qs'] if n - 1 < len(ws) else []
            if not L and not N and not Q: continue
            title = (N and N['title']) or (L and L['title']) or f'{n}교시'
            sections = N['sections'] if N else []
            terms, funcs, kws, links = [], [], [], []
            for s in sections:
                if s['key'] == 'terms':
                    got = parse_terms(s['md']); s['t0'] = len(terms); s['tn'] = len(got); terms += got
                    s['rest'] = '\n'.join(l for l in s['md'].split('\n') if not TERM_RE.match(l)).strip() if got else s['md']
                if s['key'] == 'funcs':
                    got = parse_funcs(s['md'])
                    for f in got:
                        det = api_detail(f['name'], sid)
                        if det: f['detail'] = det
                    s['f0'] = len(funcs); s['fn'] = len(got); funcs += got
                    s['rest'] = '\n'.join(l for l in s['md'].split('\n') if not l.startswith('|')).strip() if got else s['md']
                if s['key'] == 'kw': kws += parse_kw(s['md'])
                if s['key'] == 'links': links += parse_links(s['md'])
            status = 'done' if N else 'todo'
            quiz = [{k: q.get(k) for k in ('ty', 'p', 'c', 'o', 'a', 'w', 'left', 'right', 'steps', 'svg', 'vb', 'sw', 'regions', 'src', 'ro') if q.get(k) is not None} for q in Q]
            quiz += review_quiz(sid, d, n)                         # 복습 퀴즈는 worksheet 문항 뒤에 (오답 기록 번호 유지)
            topics = L['topics'] if L else []
            attach_figs(topics, n, figs)
            attach_guides(sid, d, topics, n)
            S_l = {'n': n, 'title': title, 'status': status, 'goals': L['goals'] if L else [],
                   'lectureTitle': L['title'] if L else '', 'sections': sections, 'terms': terms, 'funcs': funcs,
                   'keywords': kws, 'topics': topics, 'quiz': quiz,
                   'fills': [{'t': x['title'], 'k': x['kind'], 'q': x['q'], 'w': x['w']} for x in (L['fills'] if L else [])],
                   'anim': f'{anim}#l{n}' if anim else ''}
            D['lessons'].append(S_l)
            dm['lessons'].append({'n': n, 'title': title, 'status': status, 'nq': len(quiz),
                                  'nex': sum(1 for t in topics if any(b['kind'] in ('code', 'cmd') for b in t['blocks']))})
            loc = [sid, d, n]
            # ---- 색인 ----
            for i, t in enumerate(terms):
                index['terms'].append(loc + [t['name'], t['def'], i])
                index['search'].append(loc + ['term', f't{i}', t['name'], strip_md(t['def'])])
            for i, f in enumerate(funcs):
                index['funcs'].append(loc + [f['name'], f['sig'], f['use'], i, f.get('detail')])
                fn_rows.append((d, n, f, index['funcs'][-1]))
                index['search'].append(loc + ['func', f'f{i}', f['name'], strip_md(f['sig'] + ' ' + f['use'])])
            if kws: index['search'].append(loc + ['kw', 's-kw', '키워드', ' · '.join(kws)])
            for si, s in enumerate(sections):
                if s['key'] in ('kw', 'terms', 'funcs'): continue
                index['search'].append(loc + ['note', f's{si}', s['label'] or '노트', strip_md(s['md'])])
            for l in links: index['links'].append(loc + [l['text'], l['href']])
            pending_code.append((n, topics, sections))
            for t in topics:
                index['search'].append(loc + ['topic', f'L{t["id"]}', t['title'], t['text']])
                code = '\n'.join(b['text'] for b in t['blocks'] if b['kind'] in ('code', 'cmd'))
                if code: index['search'].append(loc + ['code', f'X{t["id"]}', t['title'], code])
            for qi, q in enumerate(quiz):
                body = ' '.join([q.get('p', ''), q.get('c', '') or '', ' '.join(q.get('o') or []), q.get('w', '')])
                index['search'].append(loc + ['quiz', f'q{qi}', f'Q{qi + 1}', re.sub(r'<[^>]+>', '', body)])
            for fi, x in enumerate(L['fills'] if L else []):
                index['search'].append(loc + ['fill', f'F{fi}', x['title'], x['text']])
        # ④ 그래도 그림이 없는 '코드가 있는' 예제: 같은 Day 앞 교시의 가장 가까운 그림 (보드를 이어서 쓰는 경우)
        last = None
        for ln, tps, _ in pending_code:
            for tp in tps:
                if '<figure>' in tp['html']: last = (ln, tp['id'])
                elif last and not tp.get('figRefs') and any(b['kind'] == 'code' for b in tp['blocks']) and last[0] != ln:
                    tp['figRefs'] = [{'t': 'other', 'n': last[0], 'id': last[1]}]
        for ln, tps, secs in pending_code:
            index['code'] += code_index(sid, d, ln, tps, secs)
        S['days'].append(D); meta['days'].append(dm)
    EX_STAT.append((sid,) + attach_examples(sid, S, fn_rows))
    return S, meta

ANIM_EXT = ('.html', '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.json')

def copy_anim_dir(cfg, d, src):
    """애니메이션 목차(index.html)가 옆 파일들을 상대 경로로 부르는 과목: 그 폴더의 웹 파일을 anim/<과목>/dayN/ 으로 통째로 복사."""
    import shutil
    base = os.path.dirname(src)
    dst_root = os.path.join(REPO, 'anim', cfg['id'], f'day{d}')
    if os.path.isdir(dst_root): shutil.rmtree(dst_root)
    for dp, dns, fns in os.walk(base):
        dns[:] = [x for x in dns if not x.startswith('.') and x != 'src']
        for fn in fns:
            if not fn.lower().endswith(ANIM_EXT): continue
            rel = os.path.relpath(os.path.join(dp, fn), base)
            os.makedirs(os.path.dirname(os.path.join(dst_root, rel)), exist_ok=True)
            shutil.copyfile(os.path.join(dp, fn), os.path.join(dst_root, rel))
    return f'anim/{cfg["id"]}/day{d}/{os.path.basename(src)}'

def copy_anim(cfg, d):
    """강의 애니메이션(교시별 #l1~#l8)을 위키 저장소 anim/<과목>/dayN.html 로 복사. 없으면 None."""
    src = campus_path(cfg['anim'], d)
    if not os.path.exists(src): return None
    if cfg.get('anim_dir'): return copy_anim_dir(cfg, d, src)
    html = open(src, encoding='utf8').read()
    sid = cfg['id']
    # 애니메이션 안의 강의 자료 · 워크시트 상대 링크는 위키에서 깨지므로 위키 교시 · 퀴즈로 바꾼다
    html = html.replace('href="lecture.html"', f'href="../../index.html#/{sid}/{d}/1"')
    html = html.replace('href="worksheet.html"', f'href="../../index.html#/quiz?s={sid}&d={d}&m=all"')
    rel = f'anim/{sid}/day{d}.html'
    dst = os.path.join(REPO, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'w', encoding='utf8') as f: f.write(html)
    return rel

def main():
    global API, GUIDE
    API, GUIDE = load_json('api_detail.json'), load_json('code_guide.json')
    import glob
    for p in sorted(glob.glob(os.path.join(NOTES, 'api_detail.*.json'))):
        sid = re.match(r'api_detail\.([a-z]+)', os.path.basename(p)).group(1)
        API_BY.setdefault(sid, {}).update({k: v for k, v in json.load(open(p, encoding='utf8')).items() if k != '_설명'})
    global REVIEW
    rq = os.path.join(NOTES, 'review_quiz.json')
    REVIEW = json.load(open(rq, encoding='utf8')) if os.path.exists(rq) else {}
    want = [a.lower() for a in sys.argv[1:]]
    index = {'built': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), 'subjects': [],
             'search': [], 'terms': [], 'funcs': [], 'links': [], 'code': []}
    os.makedirs(os.path.join(REPO, 'data'), exist_ok=True)
    for cfg in SUBJECTS:
        S, meta = build_subject(cfg, index)
        index['subjects'].append(meta)
        if want and cfg['id'] not in want: continue
        p = os.path.join(REPO, 'data', f"{cfg['id']}.js")
        with open(p, 'w', encoding='utf8') as f:
            f.write('(window.WIKI=window.WIKI||{})[%s]=' % json.dumps(cfg['id']))
            json.dump(S, f, ensure_ascii=False, separators=(',', ':')); f.write(';\n')
        done = sum(l['status'] == 'done' for d in meta['days'] for l in d['lessons'])
        tot = sum(len(d['lessons']) for d in meta['days'])
        print(f"{p}  {os.path.getsize(p) // 1024}KB  정리 {done}/{tot}교시")
    # 캐시 무효화: index.html 의 로컬 스크립트·스타일 주소에 빌드 버전을 붙인다 (Pages 는 10분 캐시)
    ver = datetime.datetime.now().strftime('%y%m%d%H%M')
    ip = os.path.join(REPO, 'index.html')
    html = open(ip, encoding='utf8').read()
    html = re.sub(r'((?:src|href)="(?:assets|data)/[\w.-]+\.(?:js|css))(?:\?v=\w+)?"', r'\1?v=' + ver + '"', html)
    open(ip, 'w', encoding='utf8').write(html)
    print('  · 함수 예제 줄: ' + ' · '.join(f'{s} {h}/{t}' for s, h, t in EX_STAT))
    if UNSUPPORTED_WS: print(f'  · 아직 문항을 못 읽는 워크시트 형식 {len(UNSUPPORTED_WS)}개: ' + ', '.join(UNSUPPORTED_WS))
    for k in GUIDE:
        if k not in GUIDE_HIT: print(f'  ! 코드 가이드 파일을 못 찾음: {k}')
    for k, ms in GUIDE_MISS.items():
        for m in ms: print(f'  ! 코드 가이드 단계 못 찾음: {k} — {m}')
    p = os.path.join(REPO, 'data', 'index.js')
    with open(p, 'w', encoding='utf8') as f:
        f.write('window.WIKI_INDEX=')
        json.dump(index, f, ensure_ascii=False, separators=(',', ':')); f.write(';\n')
    print(f"{p}  {os.path.getsize(p) // 1024}KB  검색 {len(index['search'])}건 · 용어 {len(index['terms'])} · 함수 {len(index['funcs'])} · 연결 {len(index['links'])} · 예제 {len(index['code'])}")

if __name__ == '__main__':
    main()
