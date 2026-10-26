#!/usr/bin/env python3
"""노트 한 Day 를 검증한다 — 빌드 경고 · 연결 링크 · 함수 상세 · 깨진 링크 · 다른 과목 함수와 이름 겹침.

    STUDY_ROOT=<AI_CAMPUS 의 부모 폴더> python3 -I tools/check_note.py <과목 id> <노트 파일 이름>
    예) python3 -I tools/check_note.py nn NN_day3.md

빌드로 바뀐 data/ · index.html · anim/ 은 끝나면 git 으로 되돌린다 (노트 PR 에는 notes/ 만 넣는다).
문제가 있으면 종료 코드 1.
"""
import json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    sid, note = sys.argv[1], sys.argv[2]
    bad = 0
    r = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'build_site.py'), sid], cwd=REPO, capture_output=True, text=True)
    out = r.stdout + r.stderr
    warns = [l for l in out.splitlines() if l.strip().startswith('!')]
    print('빌드:', 'OK' if r.returncode == 0 else '실패', '· 경고', len(warns))
    for w in warns: print('  ', w)
    bad += len(warns) + (r.returncode != 0)
    s = open(os.path.join(REPO, 'data', 'index.js'), encoding='utf8').read()
    idx = json.loads(s[s.index('=') + 1:].rstrip().rstrip(';'))
    titles = {(S['id'], D['n'], L['n']): L['title'] for S in idx['subjects'] for D in S['days'] for L in D['lessons']}
    text = open(os.path.join(REPO, 'notes', note), encoding='utf8').read()
    links = re.findall(r'\[([^\]]+)\]\(#/(\w+)/(\d+)/(\d+)\)', text)
    for label, a, b, c in links:
        t = titles.get((a, int(b), int(c)))
        if '\n' in label or len(label) > 80:
            print('  ❌ 대괄호가 닫히지 않아 링크가 길게 묶임:', label[:40].replace('\n', ' '), '…'); bad += 1
        elif not t:
            print('  ❌ 없는 교시:', label, (a, b, c)); bad += 1
        else:
            print('  ✓', label, '→', t)
    m = re.search(r'_day(\d+)\.md$', note)
    day = int(m.group(1)) if m else None
    funcs = [f for f in idx['funcs'] if f[0] == sid and f[1] == day]
    nodet = [f[3] for f in funcs if not f[7]]
    print(f'함수 {len(funcs)}개 · 상세 없음 {len(nodet)}개', nodet if nodet else '')
    # 다른 과목의 상세 설명이 이름만 같아서 붙은 경우 (예: Python open() ← Linux open())
    own = {}
    for p in sorted(os.listdir(os.path.join(REPO, 'notes'))):
        if p.startswith(f'api_detail.{sid}') and p.endswith('.json'):
            own.update(json.load(open(os.path.join(REPO, 'notes', p), encoding='utf8')))
    for f in funcs:
        if f[7] and f[3] not in own and f[3].split('(')[0].strip() not in own:
            print('  ⚠ 다른 과목의 상세가 이름으로 연결됨 — 의도한 것인지 확인:', f[3], '→', f[7].get('role'))
    print('링크', len(links), '개 · 문제', bad)
    subprocess.run(['git', 'checkout', '-q', '--', 'data', 'index.html'], cwd=REPO)
    subprocess.run(['git', 'clean', '-qfd', '--', 'anim'], cwd=REPO)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
