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
            plain = _text(re.sub(r'\[\[BLOCK \d+\]\]', ' ', h))
            plain = re.sub(r'\s+', ' ', plain).strip()
            topics.append({'id': len(topics), 'title': t, 'example': '예제' in t,
                           'html': h, 'text': plain, 'blocks': blocks})
        out.append({'n': n, 'title': title, 'goals': goals, 'topics': topics})
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
