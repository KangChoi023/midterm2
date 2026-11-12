#!/usr/bin/env python3
"""주피터 노트북(.ipynb) · 파워포인트(.pptx) 강의 -> lecture_parse.parse 와 같은 교시별 구조.

    parse(path) -> [ {n, title, goals, topics:[{id,title,example,html,text,blocks,files,desc}], fills} x8 ]

.ipynb : 마크다운 머리글 "# 1교시 · 제목" / "## 1교시 — 제목" 이 교시, 그 아래 "###" 하나가 토픽.
         "#### ✏️ 빈칸 채우기 · 괄호 넣기" 는 fills 로 (바로 뒤 "🔒 정답 보기" 셀의 base64 를 풀어 해설로).
         빈칸(____) 코드 셀은 강의 원문 · 코드 패널에서 그 "🔒 정답 보기" 의 정답 코드로 바꿔 보여 준다 (빈칸 탭은 문제 그대로).
.pptx  : 슬라이드 글자만 읽는다. 바닥글 "… · S3 제목" 이나 "11:00 · S3 제목" 으로 교시를 나누고 슬라이드 하나가 토픽.

사용(확인용):  python3 -I nbparse.py <파일> [교시]
"""
import re, html, sys, json, zipfile, base64

# ---------------- 아주 작은 마크다운 → html (노트북 설명 셀용) ----------------

def _inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r'`([^`]+)`', lambda m: '<code>' + m.group(1) + '</code>', t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'<em>\1</em>', t)
    t = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', t)                       # 그림은 노트북 밖에서 깨지므로 뺀다
    t = re.sub(r'\[([^\]]+)\]\((https?://[^)\s]+)\)', r'<a href="\2" target="_blank" rel="noopener">\1</a>', t)
    return t

def md_html(md):
    """제목(h5)·문단·목록·표·인용·코드 펜스 정도만. 코드 펜스는 <pre> 로 두고 lecture 쪽 블록 처리에 맡기지 않는다."""
    out, lines, i = [], md.split('\n'), 0
    while i < len(lines):
        l = lines[i]
        if l.strip().startswith('```'):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith('```'): j += 1
            out.append('<pre><code>' + html.escape('\n'.join(lines[i + 1:j])) + '</code></pre>'); i = j + 1; continue
        if re.match(r'^\s*\|', l):
            rows = []
            while i < len(lines) and re.match(r'^\s*\|', lines[i]):
                if not re.match(r'^\s*\|[\s:|-]+\|\s*$', lines[i]):
                    rows.append([c.strip() for c in lines[i].strip().strip('|').replace('\\|', '\x00').split('|')])
                i += 1
            if rows:
                h = '<table><thead><tr>' + ''.join(f'<th>{_inline(c.replace(chr(0), "|"))}</th>' for c in rows[0]) + '</tr></thead><tbody>'
                h += ''.join('<tr>' + ''.join(f'<td>{_inline(c.replace(chr(0), "|"))}</td>' for c in r) + '</tr>' for r in rows[1:])
                out.append(h + '</tbody></table>')
            continue
        m = re.match(r'^\s*(#{1,6})\s+(.*)$', l)
        if m: out.append(f'<h5>{_inline(m.group(2))}</h5>'); i += 1; continue
        if re.match(r'^\s*>', l):
            buf = []
            while i < len(lines) and re.match(r'^\s*>', lines[i]): buf.append(re.sub(r'^\s*>\s?', '', lines[i])); i += 1
            out.append('<blockquote>' + md_html('\n'.join(buf)) + '</blockquote>'); continue
        if re.match(r'^\s*([-*+]|\d+[.)])\s+', l):
            tag = 'ol' if re.match(r'^\s*\d', l) else 'ul'; items = []
            while i < len(lines) and (re.match(r'^\s*([-*+]|\d+[.)])\s+', lines[i]) or (lines[i].startswith('  ') and lines[i].strip() and items)):
                mm = re.match(r'^\s*(?:[-*+]|\d+[.)])\s+(.*)$', lines[i])
                if mm: items.append(mm.group(1))
                else: items[-1] += ' ' + lines[i].strip()
                i += 1
            out.append(f'<{tag}>' + ''.join(f'<li>{_inline(x)}</li>' for x in items) + f'</{tag}>'); continue
        if re.match(r'^\s*(---+|\*\*\*+)\s*$', l) or not l.strip(): i += 1; continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r'^\s*(#|>|\||```|[-*+]\s|\d+[.)]\s)', lines[i]):
            buf.append(lines[i].strip()); i += 1
        if buf: out.append('<p>' + '<br>'.join(_inline(x) for x in buf) + '</p>')
        else: i += 1
    return '\n'.join(out)

def _text(h):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', h))).strip()

# ---------------- 노트북 ----------------

LESSON_RE = re.compile(r'^#{1,2}\s*(\d+)\s*교시\s*[·:—–-]?\s*(.*?)\s*(?:\(\d{1,2}:\d{2}\s*[–~-]\s*\d{1,2}:\d{2}\))?\s*$')
TOPIC_RE = re.compile(r'^###\s+(.*)$')
FILL_RE = re.compile(r'^####\s*✏️?\s*(.*)$')

def _src(c):
    s = c.get('source', '')
    return ''.join(s) if isinstance(s, list) else s

def _outputs(c):
    """코드 셀의 텍스트 출력만 (그림·html 출력은 뺀다)"""
    out = []
    for o in c.get('outputs', []):
        if o.get('output_type') == 'stream': out.append(''.join(o.get('text', '')))
        elif 'data' in o and 'text/plain' in o['data']: out.append(''.join(o['data']['text/plain']))
        elif o.get('output_type') == 'error': out.append(f"{o.get('ename', '')}: {o.get('evalue', '')}")
    return '\n'.join(x.rstrip('\n') for x in out if x.strip())

def _answer(code):
    """'🔒 정답 보기' 셀: base64 문자열 조각을 이어 붙여 푼다."""
    parts = re.findall(r'"([A-Za-z0-9+/=]+)"', code)
    try: return base64.b64decode(''.join(parts)).decode('utf8')
    except Exception: return ''

def _solution(ans):
    """정답 글 '── 정답 1-1 ──\\n코드\\n\\n💡 해설\\n…' → (번호, 코드). 코드가 없으면 ('', '')"""
    m = re.match(r'\s*──\s*정답\s*([^─\n]*?)\s*──\s*\n', ans)
    if not m: return '', ''
    code = re.split(r'\n\s*💡\s*해설', ans[m.end():], 1)[0].strip('\n')
    return m.group(1).strip(), code if code.strip() else ''

def fill_blanks(blank, sol):
    """빈칸(____) 셀의 빈칸 줄을 정답 코드의 같은 줄로 바꿔 끼운다 — 셀에 있던 다른 코드(도우미 함수 등)를 지키려고.
    빈칸 줄 하나라도 정답에서 짝을 못 찾으면 None (그때는 정답 코드만 쓴다)."""
    if '____' not in blank: return None
    code = lambda l: re.split(r'\s+#', l, 1)[0].strip()                 # 끝 주석 · 들여쓰기 무시
    sl = sol.split('\n'); out, j = [], 0
    for line in blank.split('\n'):
        if '____' not in line: out.append(line); continue
        pat = re.compile('^' + '(.+?)'.join(re.escape(part) for part in code(line).split('____')) + '$')
        for k in range(j, len(sl)):
            if pat.match(code(sl[k])):
                out.append(line[:len(line) - len(line.lstrip())] + sl[k].strip()); j = k + 1; break
        else:
            return None
    return '\n'.join(out)

def _block(code):
    lines = code.split('\n')
    shell = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    if shell and all(l.lstrip().startswith(('!', '%')) for l in shell): return {'kind': 'cmd', 'lang': 'bash', 'text': code}
    return {'kind': 'code', 'lang': 'python', 'text': code}

# ---------------- 검증 셀 → "▶ 실행하면" ----------------
# 빈칸 정답 코드 바로 뒤의 검증 셀(assert … print("✅"))은 코드로 보여 주지 않고, 정답 코드를 돌렸을 때의 출력으로 바꾼다.
#   출력: notes/cell_outputs.<sid><day>.json (tools/cell_outputs.py 가 실제로 실행해 모은 값)
#   없으면: assert 에서 기대값을 뽑는다 ("교안 기대값")

def is_check(src):
    return bool(re.search(r'^\s*assert\b', src, re.M) or re.search(r'print\(\s*["\']✅', src))

_OUTS = {}

def _cell_outputs(path):
    m = re.search(r'(NN|VB|OT|QP)-day(\d+)', path.replace('\\', '/'))
    if not m: return {}
    key = m.group(1).lower() + m.group(2)
    if key not in _OUTS:
        import os
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notes', f'cell_outputs.{key}.json')
        _OUTS[key] = json.load(open(f, encoding='utf8')) if os.path.exists(f) else {}
    return _OUTS[key]

def assert_parts(src):
    """검증 셀의 assert 들 → [(식, 기대값 또는 None, 표시 기호)]  (식 = 실행하면 값을 얻을 수 있는 코드)"""
    import ast
    parts = []
    def walk(e):
        if isinstance(e, ast.BoolOp) and isinstance(e.op, ast.And):
            for v in e.values: walk(v)
        elif isinstance(e, ast.Compare) and len(e.ops) == 1 and isinstance(e.ops[0], ast.Eq):
            parts.append((ast.unparse(e.left), ast.unparse(e.comparators[0]), '='))
        elif isinstance(e, ast.Call) and getattr(e.func, 'attr', getattr(e.func, 'id', '')) in ('allclose', 'isclose', 'array_equal') and len(e.args) >= 2:
            parts.append((ast.unparse(e.args[0]), ast.unparse(e.args[1]), '≈'))
        else:
            parts.append((ast.unparse(e), None, ''))
    for line in re.findall(r'^[ \t]*(assert\b.*)$', src, re.M):
        stmt = re.sub(r';\s*print\(.*$', '', line).strip()
        try:
            node = ast.parse(stmt).body[0]
            walk(node.test)
        except Exception:
            parts.append((stmt[6:].strip(), None, ''))
    return parts

def _asserts(src):
    """검증 셀의 assert · ✅ 메시지 → 기대값 줄 (실행하지 못했을 때)"""
    out = []
    for a, b, sym in assert_parts(src):
        line = f'{a} {sym} {b}' if b is not None else a
        if re.search(r'\bfor\b', line) or len(line) > 140: continue       # all(… for …) 같은 검사식은 값이 아니라서 뺀다
        out.append(line)
    out += [m.strip() for m in re.findall(r'print\(\s*f?["\'](✅[^"\']+)["\']', src) if m.strip() != '✅']
    return '\n'.join(out)

def _tidy(text):
    """출력에서 검증용 표시를 걷어 낸다: 줄 앞뒤 ✅, 채점 셀 줄(과제 N +20 · 자동 채점)"""
    out = []
    for l in text.split('\n'):
        if re.match(r'^\s*(✅\s*)?과제\s*\d+\s+\+\d+', l) or re.match(r'^\s*자동 채점', l): continue
        l = re.sub(r'^(\s*)✅\s?', r'\1', l)
        l = re.sub(r'\s*✅\s*$', '', l)
        if l.strip(): out.append(l.rstrip())
    return '\n'.join(out)

def _run_blocks(outs, num, check_src):
    """정답 번호의 출력 블록 — 검증 셀의 코드(print · assert · 그래프)는 보여 주지 않는다 (사이트는 그림을 못 그린다)"""
    blocks = []
    o = outs.get(num)
    text = _tidy(o['text']) if o and o.get('text') else ''
    if text:
        if re.search(r'\d\s*(ms|초)(?![가-힣])|\(\d+(\.\d+)?\s*s\)|\d\s*시간', text):
            text += '\n(시간 · 속도 숫자는 실행한 컴퓨터마다 다릅니다)'
        blocks.append({'kind': 'out', 'lang': 'text', 'text': text, 'run': 'run'})
    elif check_src:
        exp = _tidy(_asserts(check_src))
        if exp: blocks.append({'kind': 'out', 'lang': 'text', 'text': exp, 'run': 'assert'})
    return blocks

def _chunks(src):
    """마크다운 셀을 머리글(#…) 줄마다 잘라 [(머리글 줄 또는 '', 본문)]"""
    out, head, buf = [], '', []
    for line in src.split('\n'):
        if re.match(r'^#{1,4}\s', line):
            if head or any(x.strip() for x in buf): out.append((head, '\n'.join(buf)))
            head, buf = line.strip(), []
        else: buf.append(line)
    if head or any(x.strip() for x in buf): out.append((head, '\n'.join(buf)))
    return out

PROBLEM_RE = re.compile(r'^\s*\*\*문제\s*([\d·\s-]+)\*\*')

def parse_ipynb(path):
    nb = json.load(open(path, encoding='utf8'))
    lessons = {}
    st = {'cur': None, 'topic': None, 'fill': None, 'blank': None,   # blank: 마지막 빈칸(____) 코드가 들어간 곳
          'checks': [], 'solved': None}    # checks: 빈칸 뒤 · 정답 앞에 온 검증 셀 / solved: 방금 넣은 정답 (번호, 토픽, 위치)
    outs = _cell_outputs(path)

    def insert_after(t, k, blocks):
        """토픽 t 의 블록 k 바로 뒤에 blocks 를 넣고 md 의 [[BLOCK]] 순서도 맞춘다"""
        if not blocks: return
        n0 = len(t['blocks']); t['blocks'].extend(blocks)
        tag = f'[[BLOCK {k}]]'
        new = ''.join(f'\n[[BLOCK {n0 + j}]]' for j in range(len(blocks)))
        for j, x in enumerate(t['md']):
            if tag in x: t['md'][j] = x.replace(tag, tag + new, 1); return
        t['md'].extend(f'[[BLOCK {n0 + j}]]' for j in range(len(blocks)))

    def flush(check_src=''):
        """정답 → 검증 순서(VB)에서 미뤄 둔 출력 상자를 넣는다"""
        if st['solved'] is None: return
        num, t, pos = st['solved']; st['solved'] = None
        insert_after(t, pos, _run_blocks(outs, num, check_src))

    def new_topic(title, md=''):
        st['fill'] = None
        st['topic'] = {'title': title, 'md': [md] if md.strip() else [], 'blocks': []}
        st['cur']['topics'].append(st['topic'])

    for c in nb.get('cells', []):
        src = _src(c).strip('\n')
        if not src.strip(): continue
        if c['cell_type'] == 'markdown':
            flush()
            for head, body in _chunks(src):
                m = LESSON_RE.match(head) if head else None
                if m:
                    n = int(m.group(1))
                    st['cur'] = lessons.setdefault(n, {'n': n, 'title': m.group(2).strip() or f'{n}교시', 'goals': [], 'topics': [], 'fills': []})
                    st['topic'] = st['fill'] = None
                    gm = re.search(r'\*\*학습 목표\*\*\s*\n((?:\s*[-*]\s+.*\n?)+)', body)
                    if gm:
                        st['cur']['goals'] = [_text(_inline(g)) for g in re.findall(r'^\s*[-*]\s+(.*)$', gm.group(1), re.M)]
                        body = body[:gm.start()] + body[gm.end():]
                    if re.sub(r'(?m)^\s*(---+)?\s*$', '', body).strip(): new_topic('들어가며', body)
                    continue
                if st['cur'] is None: continue                  # 1교시 앞(사용법·자료 목록)은 건너뛴다
                if head and re.match(r'^#{1,2}\s', head): st['cur'] = st['topic'] = st['fill'] = None; continue   # 교시가 아닌 큰 제목
                fm = FILL_RE.match(head) if head else None
                if fm and re.search(r'빈칸|괄호|과제', fm.group(1)):
                    st['fill'] = {'title': fm.group(1).strip(), 'q': [body], 'code': [], 'ans': ''}
                    st['cur']['fills'].append(st['fill']); st['topic'] = None; continue
                if head:
                    new_topic(re.sub(r'^#+\s*', '', head).strip(), body); continue
                pm = PROBLEM_RE.match(body)
                if pm:                                          # NN 노트북: "**문제 1-1** …" 다음 코드 셀이 빈칸
                    st['fill'] = {'title': '빈칸 채우기 ' + pm.group(1).strip(), 'q': [body], 'code': [], 'ans': ''}
                    st['cur']['fills'].append(st['fill']); continue
                if st['fill'] is not None and not st['fill']['ans']: st['fill']['q'].append(body); continue
                if st['topic'] is None: new_topic('들어가며')
                st['topic']['md'].append(body)
        elif c['cell_type'] == 'code' and st['cur'] is not None:
            first = src.split('\n', 1)[0]
            if '정답 보기' in first or ('b64decode' in src and '@title' in first):
                ans = _answer(src)
                if st['fill'] is not None: st['fill']['ans'] += ('\n' if st['fill']['ans'] else '') + ans
                # 강의 원문 · 코드 패널에는 빈칸(____) 코드 대신 정답을 채운 코드를 보여 준다 (빈칸·괄호 탭은 문제 그대로)
                num, sol = _solution(ans)
                flush()
                if sol and st['blank'] is not None:
                    where, k = st['blank']
                    blank_src = k[0]['blocks'][k[1]]['text'] if where == 'topic' and k[0] is st['topic'] else \
                        (st['fill']['code'][-1] if st['fill'] is not None and st['fill']['code'] else '')
                    sol = fill_blanks(blank_src, sol) or sol                     # 빈칸 셀의 나머지 코드도 지킨다
                    sol = f'# ✏️ 문제 {num} — 빈칸을 정답으로 채운 코드\n' + sol if num else sol
                    t, pos = None, None
                    if where == 'topic' and k[0] is st['topic']:
                        k[0]['blocks'][k[1]] = dict(_block(sol), solved=num); t, pos = k
                    elif where == 'fill' and st['topic'] is not None:
                        t = st['topic']
                        t['blocks'].append(dict(_block(sol), solved=num)); t['md'].append(f'[[BLOCK {len(t["blocks"]) - 1}]]')
                        pos = len(t['blocks']) - 1
                    if t is not None and num:
                        chk = '\n'.join(st['checks'])                      # NN: 빈칸 → 검증 → 정답 순서
                        if chk: insert_after(t, pos, _run_blocks(outs, num, chk))
                        else: st['solved'] = (num, t, pos)                # VB: 정답 → 검증 순서 — 다음 셀을 보고 넣는다
                st['blank'] = None; st['checks'] = []
                continue
            if '____' in src: st['blank'] = ('fill', None) if (st['fill'] is not None and not st['fill']['ans']) else ('topic', None)
            if st['blank'] is not None and '____' not in src and is_check(src):
                st['checks'].append(src); continue                          # 정답이 나오면 출력 상자로
            if st['solved'] is not None and is_check(src):
                flush(src); continue
            if re.match(r'\s*score\s*=\s*0\b', src) and is_check(src):        # 과제 채점 셀 — 점수만 매기는 검증 코드라 보여 주지 않는다
                flush(); continue
            flush()
            if st['fill'] is not None and not st['fill']['ans']: st['fill']['code'].append(src); continue
            if st['topic'] is None: new_topic('들어가며')
            t = st['topic']
            t['blocks'].append(_block(src)); t['md'].append(f'[[BLOCK {len(t["blocks"]) - 1}]]')
            if '____' in src: st['blank'] = ('topic', (t, len(t['blocks']) - 1))
            out = _outputs(c)
            if out:
                t['blocks'].append({'kind': 'out', 'lang': 'text', 'text': out}); t['md'].append(f'[[BLOCK {len(t["blocks"]) - 1}]]')
    flush()
    res = []
    for n in range(1, 9):
        L = lessons.get(n)
        if not L: res.append(None); continue
        topics = []
        for t in L['topics']:
            h = '\n'.join(x if x.startswith('[[BLOCK') else md_html(x) for x in t['md'] if x.strip())
            title = re.sub(r'^[▶📌⚠️\s]+', '', _text(_inline(t['title'])))
            topics.append({'id': len(topics), 'title': title, 'example': '예제' in title, 'html': h,
                           'text': _text(re.sub(r'\[\[BLOCK \d+\]\]', ' ', h)), 'blocks': t['blocks'], 'files': [], 'desc': ''})
        fills = []
        for f in L['fills']:
            q = '\n'.join(md_html(x) for x in f['q'] if x.strip())
            q += ''.join('<pre class="fillpre"><code data-hl="1">' + html.escape(c) + '</code></pre>' for c in f['code'])
            w = ('<pre class="fillpre"><code>' + html.escape(f['ans']) + '</code></pre>') if f['ans'] else ''
            kind = 'blank' if '빈칸' in f['title'] else 'paren' if '괄호' in f['title'] else 'task'
            fills.append({'id': '', 'title': _text(_inline(f['title'])), 'kind': kind, 'q': q, 'w': w, 'text': _text(q + ' ' + w)})
        res.append({'n': n, 'title': _text(_inline(L['title'])), 'goals': L['goals'], 'topics': topics, 'fills': fills})
    return res

# ---------------- 파워포인트 ----------------

A_NS = '{http://schemas.openxmlformats.org/drawingml/2006/main}'

def _slides(path):
    """[(슬라이드 번호, [문단 글자…])] — 문단(<a:p>) 단위로 모은다."""
    import xml.etree.ElementTree as ET
    z = zipfile.ZipFile(path)
    names = sorted((n for n in z.namelist() if re.match(r'ppt/slides/slide\d+\.xml$', n)), key=lambda n: int(re.findall(r'\d+', n)[-1]))
    out = []
    for nm in names:
        root = ET.fromstring(z.read(nm))
        paras = []
        for p in root.iter(A_NS + 'p'):
            t = ''.join(x.text or '' for x in p.iter(A_NS + 't')).strip()
            if t: paras.append(t)
        out.append((int(re.findall(r'\d+', nm)[-1]), paras))
    return out

def _session(paras):
    """슬라이드가 속한 교시(S번호)와 제목. 믿을 순서: ① 바닥글 '컴퓨터 구조 · Day 1 · S3 CPU와 메모리'
    ② 머리 '11:00 · S3 File System' 또는 첫 문단 'S2 Operating System' ③ 섹션 표지의 단독 문단 'S3.' + 다음 문단 제목."""
    for p in paras:
        m = re.search(r'·\s*Day\s*\d+\s*·\s*S(\d)\s+(.*)$', p, re.I)
        if m: return int(m.group(1)), m.group(2).strip(' ·—-')
    for k, p in enumerate(paras[:2]):
        m = re.match(r'^(?:\d{1,2}:\d{2}\s*·\s*)S(\d)\s*·?\s+(.*)$', p) or (k == 0 and re.match(r'^S(\d)\s+(\S.*)$', p))
        if m: return int(m.group(1)), m.group(2).strip(' ·—-')
    for k, p in enumerate(paras[:3]):
        m = re.match(r'^S(\d)\.?$', p)
        if m: return int(m.group(1)), (paras[k + 1] if k + 1 < len(paras) else '')
    return None, ''

def parse_pptx(path):
    lessons, cur, intro = {}, None, []
    for no, paras in _slides(path):
        if not paras: continue
        n, stitle = _session(paras)
        if paras[0] in ('다음 시간',): continue                   # 다음 시간 예고 슬라이드는 건너뛴다
        if n is None:
            if cur is None: intro.append((no, paras)); continue
            n = cur
        if n not in lessons: lessons[n] = {'title': stitle or f'{n}교시', 'slides': []}
        cur = n
        # 바닥글(과목 · Day · S제목)과 머리 글자(S제목)는 본문에서 뺀다
        body = [p for p in paras if not re.search(r'·\s*Day\s*\d+\s*·\s*S\d', p, re.I) and not re.match(r'^(\d{1,2}:\d{2}\s*·\s*)?S\d(\s|\.?$)', p)
                and not re.match(r'^.{0,20}·\s*DAY\s*\d+$', p, re.I)]
        lessons[n]['slides'].append((no, body))
    if 1 not in lessons and intro:                         # S1 표시가 없는 앞 슬라이드(오프닝)를 1교시로
        lessons[1] = {'title': '오프닝', 'slides': intro}
    res = []
    for n in range(1, 9):
        L = lessons.get(n)
        if not L: res.append(None); continue
        topics, goals = [], []
        for no, body in L['slides']:
            if not body: continue
            title, rest = body[0], body[1:]
            if title in ('이 시간의 도착점', '이 섹션의 도착') and rest:
                g = ''
                for x in rest:                                   # 한 문장이 여러 문단으로 나뉜 경우 '…다' 까지 잇는다
                    g = (g + ' ' + x).strip()
                    if re.search(r'다[.]?$', x): break
                goals.append(g)
            h = '<ul>' + ''.join(f'<li>{html.escape(x)}</li>' for x in rest) + '</ul>' if rest else ''
            topics.append({'id': len(topics), 'title': title, 'example': bool(re.match(r'실습\s*\d', title)), 'html': h,
                           'text': _text(' '.join(rest)), 'blocks': [], 'files': [], 'desc': '', 'slide': no})
        res.append({'n': n, 'title': L['title'], 'goals': goals, 'topics': topics, 'fills': []})
    return res

def day_title(path, d=None):
    if path.endswith('.ipynb'):
        nb = json.load(open(path, encoding='utf8'))
        for c in nb.get('cells', [])[:6]:
            if c['cell_type'] != 'markdown': continue
            for line in _src(c).split('\n'):
                m = re.match(r'^#\s+(.*Day\s*\d+.*)$', line.strip())
                if m:
                    t = m.group(1)
                    return t.split('—', 1)[1].strip() if '—' in t else t
        return ''
    dn = str(d) if d else r'\d+'
    for no, paras in _slides(path):
        for k, p in enumerate(paras):
            m = re.match(r'^Day\s*' + dn + r'\s*[—–-]\s*(.+)$', p)
            if m: return m.group(1).strip()
            if k == 0 and re.search(r'·\s*Day\s*' + dn + r'$', p, re.I) and len(paras) > 1 and not re.match(r'^S\d', paras[1]): return paras[1]
        if no > 6: break
    return ''

def parse(path):
    return parse_ipynb(path) if path.endswith('.ipynb') else parse_pptx(path)

if __name__ == '__main__':
    L = parse(sys.argv[1])
    only = int(sys.argv[2]) if len(sys.argv) > 2 else None
    print('Day 제목:', day_title(sys.argv[1]))
    for les in L:
        if not les or (only and les['n'] != only): continue
        print(f"=== {les['n']}교시 · {les['title']}  (토픽 {len(les['topics'])} · 빈칸 {len(les['fills'])})")
        for g in les['goals']: print('  목표:', g)
        for t in les['topics']:
            print(f"  -- [{t['id']}] {t['title']}  ({', '.join(b['kind'] for b in t['blocks'])})")
            if only: print('     ' + t['text'][:400])
        if only:
            for f in les['fills']: print('  ✏️', f['title'], '|', _text(f['w'])[:120])
