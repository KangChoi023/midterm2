#!/usr/bin/env python3
r"""강사님 교안 저장소(AI_CAMPUS)를 받아 내 과목 폴더(06_HyundaiAI, 구글 드라이브 동기화)로 정리한다.

    py tools\sync_campus.py                     # Windows: 받기 + 정리
    py tools\sync_campus.py --dry-run           # 무엇을 복사할지만 보기
    python3 tools/sync_campus.py                 # WSL · 리눅스
    … --dest "<06_HyundaiAI 폴더>" --repo "<AI_CAMPUS 를 받아 둘 폴더>"

Windows 와 WSL 모두 동작한다. 강사님 저장소는 기본으로 이 midterm2 폴더 옆의 AI_CAMPUS 에 받는다.
작업 스케줄러에서 창 없이(pyw) 돌려도 되도록, 매 실행 결과를 00_강의 자료/_동기화기록.txt 에 한 줄 이상 남긴다.

규칙
- 강사님 폴더 → 과목 폴더/01_강의자료/DayN/…, 그 안의 src/ → 과목 폴더/02_실습자료/DayN/…
- 새 파일만 복사, 내용이 바뀐 파일은 덮어쓰기 전에 같은 폴더의 _이전판/ 으로 옮겨 둔다
- 강사님 쪽에서 지워진 파일도 내 사본은 지우지 않는다
- 과목을 모르는 폴더 · 파일은 00_강의 자료/_새로온자료/ 로 (새 과목) — 결과에 따로 표시
- 내가 만든 파일(01_강의자료 바로 아래의 날짜 붙은 노트북, 03_개인실습, 04_자체학습 …)은 건드리지 않는다
"""
import argparse, datetime, filecmp, glob, os, re, shutil, subprocess, sys

REPO_URL = 'https://github.com/imguru-mooc/AI_CAMPUS.git'

# 강사님 폴더 이름(소문자) → 과목 폴더
SUBJECT_DIRS = [
    (r'^c-day(\d+)$', '04_C'),
    (r'^linux-day(\d+)$', '07_LINUX'),
    (r'^mcu-day(\d+)$', '06_MCU'),
    (r'^embeddedio-day(\d+)$', '08_Embbed IO'),
    (r'^nn-day(\d+)$', '09_NeuralNetwork'),
    (r'^python-day(\d+)$', '03_Python'),
    (r'^os-day(\d+)$', '02_운영체제활용'),
    (r'^day(\d+)$', '01_컴퓨터구조'),            # Day3/ = 컴퓨터 구조 Day 3
]
# 정해진 위치가 있는 예외
SPECIAL = {                                         # 앞에 있는 것부터 대조 (anim/day2 가 anim 보다 먼저)
    'OS': ('02_운영체제활용', '01_강의자료/OS_24H'),
    'anim/day2': ('01_컴퓨터구조', '01_강의자료/Day2/anim'),
    'anim': ('01_컴퓨터구조', '01_강의자료/Day1/anim'),      # 맨 위 anim/ = 컴퓨터 구조 Day 1 애니메이션
    '1st_re_evaluation': ('03_Python', '01_강의자료/1st_re_evaluation'),
}
# 저장소 맨 위 파일
ROOT_FILES = [
    (r'^컴퓨터구조_Day(\d+)', '01_컴퓨터구조'),
    (r'^운영체제', '02_운영체제활용'),
    (r'^(python|score|sensor)_template', '03_Python'),
]
SKIP = {'.git', '.github', 'README.md', '.gitignore', '.gitattributes', '.nojekyll'}
NEW_STUFF = os.path.join('00_강의 자료', '_새로온자료')


def find_dest():
    """06_HyundaiAI 폴더를 바탕화면 아래에서 찾는다 — Windows (OneDrive 바탕화면 포함) · WSL(/mnt/c)."""
    home = os.path.expanduser('~')
    desktops = [os.path.join(home, 'Desktop'), os.path.join(home, 'OneDrive', 'Desktop'),
                os.path.join(home, 'OneDrive', '바탕 화면'), os.path.join(home, '바탕 화면'),
                '/mnt/c/Users/*/Desktop', '/mnt/c/Users/*/OneDrive/Desktop']
    for depth in range(4):
        for d in desktops:
            pat = os.path.join(d, *(['*'] * depth), '06_HyundaiAI')
            hits = [p for p in glob.glob(pat) if os.path.isdir(p)]
            if hits: return hits[0]
    return None


def update_repo(repo):
    if os.path.isdir(os.path.join(repo, '.git')):
        r = subprocess.run(['git', '-C', repo, 'pull', '--ff-only', '-q'], capture_output=True, text=True)
        if r.returncode: print('! git pull 실패 — 받아 둔 내용으로 계속:', r.stderr.strip()[:200])
    elif not os.path.exists(repo):
        subprocess.run(['git', 'clone', '-q', REPO_URL, repo], check=True)
    else:
        sys.exit(f'! {repo} 가 git 저장소가 아닙니다. 폴더 이름을 바꾸거나 --repo 로 다른 위치를 지정하세요.')
    r = subprocess.run(['git', '-C', repo, 'log', '-1', '--format=%h %ad %s', '--date=short'], capture_output=True, text=True)
    return r.stdout.strip()


def target_for(rel):
    """저장소 안 상대 경로 → (과목 폴더, 과목 폴더 안 상대 경로). 모르면 과목 폴더 None."""
    parts = rel.split('/')
    for key, (subj, base) in SPECIAL.items():
        if rel == key or rel.startswith(key + '/'):
            return subj, os.path.join(base, rel[len(key) + 1:]) if rel != key else base
    if len(parts) == 1:                                   # 맨 위 파일
        for pat, subj in ROOT_FILES:
            m = re.match(pat, parts[0])
            if m:
                day = f'Day{m.group(1)}' if m.groups() and m.group(1) and m.group(1).isdigit() else '공통'
                return subj, os.path.join('01_강의자료', day, parts[0])
        if parts[0] == 'worksheet.html':                  # 맨 위 worksheet.html = Embedded IO Day 5 (Day5 폴더 것과 다른 판)
            return '08_Embbed IO', os.path.join('01_강의자료', 'Day5', 'worksheet (저장소 맨 위).html')
        return None, rel
    top = parts[0]
    for pat, subj in SUBJECT_DIRS:
        m = re.match(pat, top.lower())
        if m:
            day, rest = f'Day{m.group(1)}', parts[1:]
            if rest and rest[0] == 'src':                 # 실습 코드
                return subj, os.path.join('02_실습자료', day, *rest[1:])
            return subj, os.path.join('01_강의자료', day, *rest)
    return None, rel


def copy_one(src, dst, dry, stamp):
    """새 파일이면 복사, 내용이 다르면 이전 판을 _이전판/ 에 남기고 복사. 반환: 'new' | 'changed' | None"""
    if os.path.exists(dst):
        if filecmp.cmp(src, dst, shallow=False): return None
        if not dry:
            old_dir = os.path.join(os.path.dirname(dst), '_이전판')
            os.makedirs(old_dir, exist_ok=True)
            base, ext = os.path.splitext(os.path.basename(dst))
            shutil.move(dst, os.path.join(old_dir, f'{base}.{stamp}{ext}'))
            shutil.copy2(src, dst)
        return 'changed'
    if not dry:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return 'new'


def main():
    if sys.stdout is None:                                  # pyw (창 없음) 로 실행 — 출력은 버리고 기록 파일만
        sys.stdout = sys.stderr = open(os.devnull, 'w', encoding='utf-8')
    elif hasattr(sys.stdout, 'reconfigure'):                # Windows 명령창(cp949)에서 기호 · 한글 깨짐 방지
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='강사님 교안 → 내 과목 폴더 정리')
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))        # midterm2 폴더
    ap.add_argument('--repo', default=os.path.join(os.path.dirname(here), 'AI_CAMPUS'), help='강사님 저장소를 받아 둘 위치 (기본: midterm2 옆)')
    ap.add_argument('--dest', help='06_HyundaiAI 폴더 (생략하면 바탕화면 아래에서 찾음)')
    ap.add_argument('--dry-run', action='store_true', help='복사하지 않고 무엇을 할지만 출력')
    ap.add_argument('--no-pull', action='store_true', help='저장소를 새로 받지 않음')
    a = ap.parse_args()

    dest = a.dest or find_dest()
    if not dest or not os.path.isdir(dest): sys.exit('! 06_HyundaiAI 폴더를 찾지 못했습니다. --dest 로 지정하세요.')
    head = '(받지 않음)' if a.no_pull else update_repo(a.repo)
    stamp = datetime.datetime.now().strftime('%y%m%d-%H%M')

    done = {'new': [], 'changed': []}
    unknown, used = set(), {}
    for root, dirs, files in os.walk(a.repo):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for fn in sorted(files):
            rel = os.path.relpath(os.path.join(root, fn), a.repo).replace(os.sep, '/')
            if fn in SKIP or rel.split('/')[0] in SKIP: continue
            subj, sub = target_for(rel)
            if subj is None:
                unknown.add(rel.split('/')[0])
                dst = os.path.join(dest, NEW_STUFF, sub)
            else:
                dst = os.path.join(dest, subj, sub)
            if dst in used and used[dst] != rel:              # 두 원본이 같은 자리로 오면 뒤엣것에 이름표
                b, e = os.path.splitext(dst); dst = f'{b} ({rel.split("/")[0]}){e}'
            used[dst] = rel
            r = copy_one(os.path.join(root, fn), dst, a.dry_run, stamp)
            if r: done[r].append(os.path.relpath(dst, dest))

    lines = [f'[{stamp}] 강사님 저장소: {head}',
             f'새 파일 {len(done["new"])}개 · 바뀐 파일 {len(done["changed"])}개' + (' (dry-run — 복사 안 함)' if a.dry_run else '')]
    for k, label in (('changed', '바뀜(이전 판은 _이전판/)'), ('new', '새로')):
        for p in done[k][:200]: lines.append(f'  {label}: {p}')
        if len(done[k]) > 200: lines.append(f'  … 외 {len(done[k]) - 200}개')
    if unknown:
        lines.append('⚠ 과목을 모르는 자료 (새 과목?) → ' + NEW_STUFF + ': ' + ', '.join(sorted(unknown)))
    print('\n'.join(lines))
    if not a.dry_run:                                       # 변화가 없어도 한 줄 — 예약 실행이 돌았는지 확인용
        os.makedirs(os.path.join(dest, '00_강의 자료'), exist_ok=True)
        log = os.path.join(dest, '00_강의 자료', '_동기화기록.txt')
        if os.path.exists(log):                             # 예전 기록(BOM 없음)에 BOM 을 붙여 둔다 — 한 번만
            with open(log, 'rb') as f: raw = f.read()
            if not raw.startswith(b'\xef\xbb\xbf'):
                with open(log, 'wb') as f: f.write(b'\xef\xbb\xbf' + raw)
        # utf-8-sig: 파일 맨 앞에 BOM 표시 → Windows PowerShell · 메모장이 한글을 UTF-8 로 읽는다 (없으면 cp949 로 읽어 깨짐)
        with open(log, 'a', encoding='utf-8-sig') as f:
            f.write('\n'.join(lines if (done['new'] or done['changed'] or unknown) else lines[:2]) + '\n\n')


if __name__ == '__main__':
    main()
