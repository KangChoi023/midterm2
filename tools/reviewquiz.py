"""복습 퀴즈(Google 설문지 결과 화면을 '다른 이름으로 저장'한 .html) → summary/.src/review_quiz.json

    python3 -I tools/reviewquiz.py <폴더>

문항·코드·보기는 페이지 안의 FB_PUBLIC_LOAD_DATA_ 에서, 정답·해설은 채점 결과 화면 글자에서 읽는다.
교시는 해설 끝의 "[복습: D2-3 …]" · "[복습: 4교시 4-2]" · "[복습: 빈칸 7-1]" 표시로 정한다.
채점 전 화면(정답 없음)은 summary/.src/review_quiz_answers*.json 의 풀이 [교시, 정답 번호(0부터), 해설] 를
같은 순서로 붙이고 by="wiki" 로 표시한다 (공식 정답이 아님). 풀이가 없으면 건너뛴다.
"""
import html, json, os, re, sys

SUBJ = [(r'^Linux', 'linux'), (r'^MCU', 'mcu'), (r'^Embedded', 'eio'), (r'^C\b', 'c')]
OUT = os.path.expanduser('~/c_study/summary/.src/review_quiz.json')


def fb_items(s):
    i = s.find('FB_PUBLIC_LOAD_DATA_ = ')
    j = s.find(';</script>', i)
    d = json.loads(s[i + len('FB_PUBLIC_LOAD_DATA_ = '):j])
    out = []
    for it in d[1][1]:
        if it[3] != 2 or not it[4]: continue                       # 객관식만
        out.append({'title': it[1].strip(), 'code': (it[2] or '').strip('\n'), 'opts': [o[0] for o in it[4][0][1]]})
    return out


def text_lines(s):
    t = re.sub(r'<script.*?</script>|<style.*?</style>', '', s, flags=re.S)
    t = html.unescape(re.sub(r'<[^>]+>', '\n', t))
    return [l.strip() for l in t.split('\n') if l.strip()]


def lesson_of(w):
    m = re.search(r'\[복습:([^\]]*)\]', w)
    if not m: return None
    tag = m.group(1)
    for pat in (r'D\d-(\d+)', r'(\d+)교시', r'(?:빈칸|Day \d+)\s*(\d+)-'):
        k = re.search(pat, tag)
        if k: return int(k.group(1))
    return None


def parse(path):
    s = open(path, encoding='utf-8', errors='replace').read()
    name = os.path.basename(path)
    sid = next((v for p, v in SUBJ if re.search(p, name)), None)
    day = int(re.search(r'Day\s*(\d+)', name).group(1))
    items, lines = fb_items(s), text_lines(s)
    if '맞음' not in lines and '정답' not in lines:                 # 채점 전 화면: 문항만
        return sid, day, [{'n': None, 'p': re.sub(r'^\d+\.\s*', '', q['title']), 'c': q['code'] or None,
                           'o': q['opts'], 'a': None, 'w': '', 'by': 'wiki'} for q in items]
    norm = lambda x: re.sub(r'\s+', ' ', x).strip()
    starts = []
    for q in items:
        k = next((i for i, l in enumerate(lines) if norm(l) == norm(q['title']) and all(i > a for a in starts)), None)
        starts.append(k)
    out = []
    for qi, q in enumerate(items):
        a, b = starts[qi], next((x for x in starts[qi + 1:] if x), len(lines))
        seg = lines[a:b]
        ans = None
        if '정답' in seg:                                          # 틀린 문항: "정답" 다음 줄이 정답 보기
            t = norm(seg[seg.index('정답') + 1])
            ans = next((i for i, o in enumerate(q['opts']) if norm(o) == t), None)
        elif '맞음' in seg:                                        # 맞힌 문항: "맞음" 바로 앞 보기
            t = norm(seg[seg.index('맞음') - 1])
            ans = next((i for i, o in enumerate(q['opts']) if norm(o) == t), None)
        w = ' '.join(seg[seg.index('의견 보내기') + 1:]) if '의견 보내기' in seg else ''
        out.append({'n': lesson_of(w), 'p': re.sub(r'^\d+\.\s*', '', q['title']), 'c': q['code'] or None,
                    'o': q['opts'], 'a': ans, 'w': w})
    return sid, day, out


def load_answers():
    import glob
    ans = {}
    for f in sorted(glob.glob(os.path.join(os.path.dirname(OUT), 'review_quiz_answers*.json'))):
        for sid, days in json.load(open(f)).items():
            for d, rows in days.items(): ans.setdefault(sid, {})[d] = rows
    return ans


def main(folder):
    data = json.load(open(OUT)) if os.path.exists(OUT) else {}
    ANS = load_answers()
    for f in sorted(os.listdir(folder)):
        if not f.endswith('.html'): continue
        sid, day, qs = parse(os.path.join(folder, f))
        if qs and qs[0].get('by') == 'wiki':
            rows = ANS.get(sid, {}).get(str(day))
            if not rows or len(rows) != len(qs):
                print(f'  - 정답 없음(채점 전 화면), 풀이 파일도 없음 → 건너뜀: {f}'); continue
            for q, (n, a, w) in zip(qs, rows): q.update(n=n, a=a, w=w)
        bad = [i + 1 for i, q in enumerate(qs) if q['a'] is None or q['n'] is None]
        data.setdefault(sid, {})[str(day)] = qs
        print(f'  {sid} D{day}: {len(qs)}문항' + (' (위키 풀이)' if qs[0].get('by') else '') + (f'  ! 정답/교시 못 찾음: {bad}' if bad else ''))
    json.dump(data, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('→', OUT)


if __name__ == '__main__':
    main(sys.argv[1])
