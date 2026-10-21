#!/usr/bin/env python3
"""정리 노트(.src/*.md) + 강의(lecture.html) + 워크시트(worksheet.html) -> 위키 데이터(data/*.js).

    python3 -I tools/build_site.py            # 전 과목
    python3 -I tools/build_site.py c linux    # 일부 과목

경로 기본값은 ~/c_study 기준. 바꾸려면 환경변수 STUDY_ROOT.
결과:
    data/index.js   과목·Day·교시 목차, 검색 색인, 용어/함수/연결 사전
    data/<id>.js    과목별 교시 본문(노트 섹션, 강의 토픽, 예제 코드, 문항)
"""
import os, re, sys, json, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wsparse, lecture_parse

REPO = os.path.dirname(HERE)
ROOT = os.environ.get('STUDY_ROOT', os.path.expanduser('~/c_study'))
CAMPUS = os.path.join(ROOT, 'AI_CAMPUS')
NOTES = os.path.join(ROOT, 'summary', '.src')

SUBJECTS = [
    dict(id='c', name='C 언어', short='C', days=4,
         lecture='C-day{d}/C_Day{d}_lecture.html', ws='C-day{d}/anim/worksheet.html', note='C_day{d}.md', anim='C-day{d}/anim/index.html'),
    dict(id='linux', name='LINUX', short='LINUX', days=5,
         lecture='Linux-day{d}/lecture.html', ws='Linux-day{d}/worksheet.html', note='LINUX_day{d}.md', anim='Linux-day{d}/animation.html'),
    dict(id='mcu', name='MCU', short='MCU', days=4,
         lecture='MCU-day{d}/MCU_Day{d}_lecture.html', ws='MCU-day{d}/worksheet.html', note='MCU_day{d}.md', anim='MCU-day{d}/index.html'),
    dict(id='eio', name='Embedded IO', short='EIO', days=5,
         lecture='EmbeddedIO-day{d}/lecture.html', ws='EmbeddedIO-day{d}/worksheet.html', note='EIO_day{d}.md', anim='EmbeddedIO-day{d}/animation.html'),
]

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

def parse_funcs(md):
    out = []
    for line in md.split('\n'):
        if not line.startswith('|') or re.match(r'^\|[\s:|-]+\|$', line): continue
        cells = [c.strip().replace('\x00', '|') for c in line.replace('\\|', '\x00').strip().strip('|').split('|')]
        if len(cells) < 2 or cells[0] in ('함수', '명령', '이름', '항목', '것', '문법', '함수/명령', '명령 · 옵션'): continue
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

def build_subject(cfg, index):
    sid = cfg['id']; S = {'id': sid, 'name': cfg['name'], 'days': []}
    meta = {'id': sid, 'name': cfg['name'], 'short': cfg['short'], 'days': []}
    for d in range(1, cfg['days'] + 1):
        lp = os.path.join(CAMPUS, cfg['lecture'].format(d=d))
        wp = os.path.join(CAMPUS, cfg['ws'].format(d=d))
        np_ = os.path.join(NOTES, cfg['note'].format(d=d))
        lec = lecture_parse.parse(lp) if os.path.exists(lp) else [None] * 8
        try: ws = wsparse.load(wp) if os.path.exists(wp) else []
        except Exception as e: print('  ! worksheet', wp, e); ws = []
        notes = parse_notes(np_) if os.path.exists(np_) else {'title': '', 'lessons': {}}
        dtitle = notes['title'] or (lecture_parse.day_title(lp) if os.path.exists(lp) else '')
        D = {'n': d, 'title': dtitle, 'lessons': []}
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
                    got = parse_funcs(s['md']); s['f0'] = len(funcs); s['fn'] = len(got); funcs += got
                    s['rest'] = '\n'.join(l for l in s['md'].split('\n') if not l.startswith('|')).strip() if got else s['md']
                if s['key'] == 'kw': kws += parse_kw(s['md'])
                if s['key'] == 'links': links += parse_links(s['md'])
            status = 'done' if N else 'todo'
            quiz = [{k: q.get(k) for k in ('ty', 'p', 'c', 'o', 'a', 'w', 'left', 'right', 'steps') if q.get(k) is not None} for q in Q]
            topics = L['topics'] if L else []
            S_l = {'n': n, 'title': title, 'status': status, 'goals': L['goals'] if L else [],
                   'lectureTitle': L['title'] if L else '', 'sections': sections, 'terms': terms, 'funcs': funcs,
                   'keywords': kws, 'topics': topics, 'quiz': quiz,
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
                index['funcs'].append(loc + [f['name'], f['sig'], f['use'], i])
                index['search'].append(loc + ['func', f'f{i}', f['name'], strip_md(f['sig'] + ' ' + f['use'])])
            if kws: index['search'].append(loc + ['kw', 's-kw', '키워드', ' · '.join(kws)])
            for si, s in enumerate(sections):
                if s['key'] in ('kw', 'terms', 'funcs'): continue
                index['search'].append(loc + ['note', f's{si}', s['label'] or '노트', strip_md(s['md'])])
            for l in links: index['links'].append(loc + [l['text'], l['href']])
            for t in topics:
                index['search'].append(loc + ['topic', f'L{t["id"]}', t['title'], t['text']])
                code = '\n'.join(b['text'] for b in t['blocks'] if b['kind'] in ('code', 'cmd'))
                if code: index['search'].append(loc + ['code', f'X{t["id"]}', t['title'], code])
            for qi, q in enumerate(quiz):
                body = ' '.join([q.get('p', ''), q.get('c', '') or '', ' '.join(q.get('o') or []), q.get('w', '')])
                index['search'].append(loc + ['quiz', f'q{qi}', f'Q{qi + 1}', re.sub(r'<[^>]+>', '', body)])
        S['days'].append(D); meta['days'].append(dm)
    return S, meta

def copy_anim(cfg, d):
    """강의 애니메이션(교시별 #l1~#l8)을 위키 저장소 anim/<과목>/dayN.html 로 복사. 없으면 None."""
    src = os.path.join(CAMPUS, cfg['anim'].format(d=d))
    if not os.path.exists(src): return None
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
    want = [a.lower() for a in sys.argv[1:]]
    index = {'built': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), 'subjects': [],
             'search': [], 'terms': [], 'funcs': [], 'links': []}
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
    p = os.path.join(REPO, 'data', 'index.js')
    with open(p, 'w', encoding='utf8') as f:
        f.write('window.WIKI_INDEX=')
        json.dump(index, f, ensure_ascii=False, separators=(',', ':')); f.write(';\n')
    print(f"{p}  {os.path.getsize(p) // 1024}KB  검색 {len(index['search'])}건 · 용어 {len(index['terms'])} · 함수 {len(index['funcs'])} · 연결 {len(index['links'])}")

if __name__ == '__main__':
    main()
