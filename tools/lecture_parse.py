#!/usr/bin/env python3
"""lecture.html -> 교시별 구조 데이터.

각 강의 파일은 <section id="secN" class="lesson"> 하나가 N교시다.
교시 안에서 <h3> 하나가 '강의 토픽'(개념 설명 또는 예제) 한 단위이고,
<h4 class="fill"> 부터는 빈칸/괄호 문제라서 버린다.

    parse(path) -> [ {n, title, goals, topics:[{id,title,example,html,text,blocks}]} x8 ]
    blocks = [{kind: code|cmd|out, lang, text}]

사용(확인용):  python3 -I lecture_parse.py <lecture.html> [교시]
"""
import re, html, sys

KEEP = {'p', 'ul', 'ol', 'li', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
        'strong', 'b', 'em', 'i', 'code', 'br', 'blockquote', 'h5', 'h6', 'sup', 'sub', 'kbd'}

def _text(h):
    return html.unescape(re.sub(r'<[^>]+>', '', h))

CMD_HEAD = re.compile(r'^\s*(\$ |# |gcc|cc |make|\./|ls|cat|echo|nm |objdump|valgrind|gdb|cd |sudo|python|pip|pio|idf\.py|arduino-cli|'
                      r'chmod|mkdir|cp |mv |rm |grep|find|ps |kill|strace|ltrace|time |export|gpio|i2c|dtc|insmod|rmmod|modprobe|dmesg|'
                      r'raspi|ssh|scp|tar |unzip|git |od |hexdump|xxd|size |readelf|top|htop|nice|renice|taskset|ulimit|wc |head|tail|diff)')
C_HINT = re.compile(r'(#include|#define|^\s*(int|void|char|static|typedef|struct|unsigned|const|uint\d+_t|float|double|enum|union|while|for|if|return)\b|;\s*(//.*|/\*.*)?$|^\s*[{}]\s*$)', re.M)

def classify(lang, text):
    lang = (lang or '').lower()
    if lang in ('c', 'cpp', 'c++', 'arduino', 'ino', 'h'): return 'code', 'c'
    if lang in ('python', 'py'): return 'code', 'python'
    if lang in ('makefile', 'make'): return 'code', 'makefile'
    if lang in ('bash', 'sh', 'shell', 'console', 'shell-session'): return 'cmd', 'bash'
    if lang in ('ini', 'yaml', 'json', 'dts', 'toml', 'cmake'): return 'code', lang
    if lang and lang != 'text': return 'code', lang
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines: return 'out', 'text'
    if lang == '' :
        if re.search(r'^\S+:.*\n\t', text, re.M) and ('$(' in text or '.PHONY' in text): return 'code', 'makefile'
        c_hits = sum(1 for l in lines if C_HINT.search(l))
        if c_hits >= max(2, len(lines) * 0.4): return 'code', 'c'
        cmd_hits = sum(1 for l in lines if CMD_HEAD.match(l))
        if cmd_hits and cmd_hits >= len(lines) * 0.5: return 'cmd', 'bash'
    return 'out', 'text'

def _clean_html(h, blocks):
    """허용 태그만 남기고 속성 제거. 코드 블록은 [[BLOCK n]] 자리표시로."""
    def pre_sub(m):
        lang = re.search(r'language-([\w+-]+)', m.group(1) or '')
        body = m.group(2)
        txt = _text(re.sub(r'<br\s*/?>', '\n', body)).strip('\n')
        kind, lg = classify(lang.group(1) if lang else '', txt)
        blocks.append({'kind': kind, 'lang': lg, 'text': txt})
        return f'\n[[BLOCK {len(blocks) - 1}]]\n'
    h = re.sub(r'<div class="hl">\s*<pre([^>]*)>(.*?)</pre>\s*</div>', pre_sub, h, flags=re.S)
    h = re.sub(r'<pre([^>]*)>(.*?)</pre>', pre_sub, h, flags=re.S)
    figs = []
    def fig_sub(m):
        figs.append(re.sub(r'<script[\s\S]*?</script>', '', m.group(0)))
        return f'[[FIG {len(figs) - 1}]]'
    h = re.sub(r'<figure[\s\S]*?</figure>', fig_sub, h)
    h = re.sub(r'<(script|style)[\s\S]*?</\1>', '', h)
    h = re.sub(r'<p class="anim">[\s\S]*?</p>', '', h)
    def tag_sub(m):
        name = m.group(2).lower()
        if name not in KEEP: return ''
        return f'<{m.group(1)}{name}>'
    h = re.sub(r'<(/?)([a-zA-Z0-9]+)[^>]*>', tag_sub, h)
    for i, f in enumerate(figs): h = h.replace(f'[[FIG {i}]]', f'<figure>{f[f.index(">") + 1:f.rindex("<")]}</figure>')
    h = re.sub(r'<p>\s*</p>', '', h)
    return h.strip()

FILE_RE = r'[\w.-]*\w\.(?:c|h|py|sh|ino|cpp|dts|txt|mk)'
HEAD_COMMENT = re.compile(r'^\s*(?:/\*+|//+|#+)\s*(?:src/)?(' + FILE_RE + r')\b\s*(?:[—–:-]+\s*(.*?))?\s*(?:\*/)?\s*$')

def _name_blocks(h, blocks):
    """코드 블록마다 원본 파일 이름(b['file'])을 붙이고, 토픽의 (파일 목록, 설명)을 돌려준다.
    ① 코드 첫 줄 주석 "/* 02_nice_race.c — 설명" ② 블록 바로 앞 본문의 파일명 ("sensor.h — 선언")
    ③ 빌드 명령의 소스 (gcc … x.c / make x) 순서로 찾는다."""
    segs = re.split(r'\[\[BLOCK (\d+)\]\]', h)        # [텍스트, 번호, 텍스트, 번호, …]
    before = {}
    for i in range(1, len(segs), 2):
        txt = _text(re.sub(r'<figure>[\s\S]*?</figure>', ' ', segs[i - 1]))
        found = re.findall(r'(?<![\w/])(?:src/)?(' + FILE_RE + r')(?![\w])', txt)
        before[int(segs[i])] = found[-1] if found else None
    desc = ''
    for i, b in enumerate(blocks):
        if b['kind'] != 'code': continue
        m = None
        for line in b['text'].split('\n')[:2]:
            m = HEAD_COMMENT.match(line)
            if m: break
        if m:
            b['file'] = m.group(1)
            if not desc and m.group(2): desc = m.group(2).strip().rstrip('*/').strip()
        elif before.get(i):
            b['file'] = before[i]
    files = [b['file'] for b in blocks if b.get('file')]
    if not files:                                      # 코드 없이 명령만 있는 예제: 빌드 명령에서
        for b in blocks:
            if b['kind'] != 'cmd': continue
            # gcc 가 소스 하나만 컴파일할 때만 (gcc main.c sensor.c 처럼 여러 개면 대표 이름을 정할 수 없다)
            g = re.search(r'\bgcc\b[^\n]*', b['text'])
            srcs = re.findall(r'\b([\w.-]+\.c)\b', g.group(0)) if g else []
            m = re.search(r'^\s*make\s+([\w-]+)\s*(?:&&|$)', b['text'], re.M)
            if len(srcs) == 1: files.append(srcs[0]); break
            if m and m.group(1) not in ('clean', 'all', 'install'): files.append(m.group(1) + '.c'); break
        if not files and blocks:
            txt = _text(re.sub(r'<figure>[\s\S]*?</figure>', ' ', segs[0]))
            m = re.search(r'(?<![\w/])(?:src/)?(' + FILE_RE + r')(?![\w])', txt)
            if m and any(b['kind'] == 'code' for b in blocks): files.append(m.group(1))
    codes = [b for b in blocks if b['kind'] == 'code']
    if len(codes) == 1 and not codes[0].get('file') and files: codes[0]['file'] = files[0]
    seen = []
    for f in files:
        if f not in seen: seen.append(f)
    return seen, desc

def parse(path):
    s = open(path, encoding='utf8').read()
    out = []
    for n in range(1, 9):
        i = s.find(f'<section id="sec{n}"')
        if i < 0: out.append(None); continue
        j = s.find('<section', i + 10)
        sec = s[i:j if j > 0 else len(s)]
        sec = sec[:sec.index('</section>')] if '</section>' in sec else sec
        m = re.search(r'<h4[^>]*class="fill"', sec)
        if m: sec = sec[:m.start()]
        h1 = re.search(r'<h1[^>]*>(.*?)</h1>', sec, re.S)
        title = _text(h1.group(1)).strip() if h1 else f'{n}교시'
        he = sec.find('</header>')
        head = sec[:he + 9] if he > 0 else (sec[:h1.end()] if h1 else '')
        body = sec[len(head):]
        # 학습 목표
        goals = []
        gm = re.search(r'학습 목표</strong>\s*</p>\s*<ul>(.*?)</ul>', body, re.S)
        if gm: goals = [_text(x).strip() for x in re.findall(r'<li>(.*?)</li>', gm.group(1), re.S)]
        else:
            gm = re.search(r'<p><strong>학습 목표</strong>(.*?)</p>', body, re.S)
            if gm: goals = [g.strip(' -') for g in _text(gm.group(1)).split('\n') if g.strip(' -')]
        body = re.sub(r'<p class="goal-title">.*?</p>\s*<ul>.*?</ul>', '', body, count=1, flags=re.S)
        body = re.sub(r'<p><strong>학습 목표</strong>.*?</p>', '', body, count=1, flags=re.S)
        parts = re.split(r'(<h3[^>]*>.*?</h3>)', body, flags=re.S)
        topics = []
        pre = parts[0]
        if _text(re.sub(r'<(script|style)[\s\S]*?</\1>|<p class="anim">[\s\S]*?</p>', '', pre)).strip():
            parts = ['', '<h3>들어가며</h3>', pre] + parts[1:]
        for k in range(1, len(parts), 2):
            t = _text(parts[k]).strip().lstrip('▶').strip()
            blocks = []
            h = _clean_html(parts[k + 1] if k + 1 < len(parts) else '', blocks)
            # 검색용 본문: 코드 자리표시와 배선 그림(SVG 속 글자)은 뺀다
            plain = _text(re.sub(r'\[\[BLOCK \d+\]\]', ' ', re.sub(r'<figure>[\s\S]*?</figure>', ' ', h)))
            plain = re.sub(r'\s+', ' ', plain).strip()
            files, desc = _name_blocks(h, blocks)
            topics.append({'id': len(topics), 'title': t, 'example': '예제' in t,
                           'html': h, 'text': plain, 'blocks': blocks, 'files': files, 'desc': desc})
        out.append({'n': n, 'title': title, 'goals': goals, 'topics': topics})
    return out

def fig_label(fig):
    """그림 제목: svg aria-label → figcaption → svg <title> (없으면 '')"""
    for pat in (r'aria-label="([^"]+)"', r'<figcaption[^>]*>([\s\S]*?)</figcaption>', r'<title>([\s\S]*?)</title>'):
        m = re.search(pat, fig)
        if m: return _text(m.group(1)).strip()
    return ''

def intro_figs(path):
    """강의 맨 앞(1교시 앞) 공통 그림. "(3·4·5·6·7·8교시 공통)" 처럼 적힌 교시 번호를 lessons 로."""
    s = open(path, encoding='utf8').read()
    i = s.find('<section id="sec1"')
    out = []
    for f in re.findall(r'<figure[\s\S]*?</figure>', s[:i] if i > 0 else ''):
        label = fig_label(f)
        m = re.search(r'\(([\d·,\s]+)교시', label)
        lessons = [int(x) for x in re.findall(r'\d+', m.group(1))] if m else []
        inner = re.sub(r'<script[\s\S]*?</script>', '', f[f.index('>') + 1:f.rindex('<')])
        out.append({'label': label, 'html': inner, 'lessons': lessons})
    return out

def day_title(path):
    s = open(path, encoding='utf8').read()
    m = re.search(r'<title>(.*?)</title>', s, re.S)
    t = _text(m.group(1)) if m else ''
    t = t.split('—', 1)[1].strip() if '—' in t else t
    return t

if __name__ == '__main__':
    L = parse(sys.argv[1])
    only = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for les in L:
        if not les or (only and les['n'] != only): continue
        print(f"=== {les['n']}교시 · {les['title']}")
        for g in les['goals']: print('  목표:', g)
        for t in les['topics']:
            print(f"  -- [{t['id']}] {t['title']}  ({', '.join(b['kind'] + '/' + b['lang'] for b in t['blocks'])})")
            if only:
                print('     ' + t['text'][:3000])
                for b in t['blocks']: print(f"     <{b['kind']}:{b['lang']}>\n" + b['text'] + '\n     </>')
