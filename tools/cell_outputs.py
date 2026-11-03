#!/usr/bin/env python3
"""강의 노트북을 셀 순서대로 실행해 "빈칸 정답 코드 + 바로 뒤 검증 셀" 의 출력을 모은다
→ notes/cell_outputs.<sid><day>.json

    STUDY_ROOT=… python3 -I tools/cell_outputs.py nn 3          # NN Day 3
    STUDY_ROOT=… python3 -I tools/cell_outputs.py vb 2 --timeout 900

사이트의 강의 코드 화면은 검증 셀(assert … print("✅")) 대신 이 출력을 "▶ 실행하면" 상자로 보여 준다
(tools/nbparse.py). 빌드가 매번 돌리지 않는다 — 학습이 오래 걸리고 데이터 · GPU 가 필요하기 때문.
실행하지 못한 문제는 JSON 에 넣지 않고, 빌드가 assert 에서 기대값을 뽑는다 ("교안 기대값").

JSON: {"문제번호": {"text": 출력, "check": 검증 셀이 있었나, "fixed": 손으로 고친 값이면 true}}
"""
import contextlib, glob, io, json, os, re, shutil, signal, sys, tempfile, time, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nbparse

FOLDER = {'nn': 'NN', 'vb': 'VB'}


class CellTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise CellTimeout()


def _stub_ipython():
    """IPython 이 없는 곳에서도 display · Image · HTML 이 오류 없이 지나가게"""
    try:
        import IPython.display  # noqa: F401
        return
    except Exception:
        pass
    ip = types.ModuleType('IPython'); disp = types.ModuleType('IPython.display')
    class _Any:
        def __init__(self, *a, **k): pass
    disp.display = lambda *a, **k: None
    disp.Image = disp.HTML = disp.Video = disp.Markdown = _Any
    ip.display = disp
    sys.modules['IPython'] = ip; sys.modules['IPython.display'] = disp


def _prep(src):
    """IPython 문법을 순수 파이썬으로: %%writefile, %magic, !shell 줄"""
    lines = src.split('\n')
    m = re.match(r'^%%writefile\s+(-a\s+)?(\S+)', lines[0]) if lines else None
    if m:
        body = '\n'.join(lines[1:]) + '\n'
        return f'open({m.group(2)!r}, {"a" if m.group(1) else "w"!r}, encoding="utf-8").write({body!r})'
    out = []
    for l in lines:
        s = l.lstrip()
        if s.startswith(('%', '!')):
            out.append(l[:len(l) - len(s)] + 'pass  # ' + s)
        else:
            out.append(l)
    return '\n'.join(out)


def _clean(text):
    lines = [l.rstrip() for l in text.replace('\r', '\n').split('\n')]
    lines = [l for l in lines if l.strip() and l.strip() != '✅']
    if len(lines) > 40: lines = lines[:38] + ['…', lines[-1]]
    return '\n'.join(lines)


def _round(v):
    if isinstance(v, float): return round(v, 4)
    if isinstance(v, (list, tuple)): return type(v)(_round(x) for x in v)
    if isinstance(v, dict): return {k: _round(x) for k, x in v.items()}
    return v


def _fmt(v):
    try:
        import numpy as np
        if isinstance(v, np.generic): v = v.item()
        if isinstance(v, np.ndarray):
            return np.array2string(v, precision=4, suppress_small=True, threshold=20, max_line_width=100)
        if isinstance(v, (float, np.floating)): return f'{float(v):.6g}'
    except Exception:
        pass
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return _fmt(v.detach().cpu().numpy()) if v.numel() > 1 else _fmt(v.item())
    except Exception:
        pass
    r = repr(_round(v))
    return r if len(r) <= 200 else r[:197] + '…'


def is_check(src):
    return nbparse.is_check(src)


def run(sid, day, timeout):
    root = os.environ.get('STUDY_ROOT', '')
    nbs = glob.glob(os.path.join(root, 'AI_CAMPUS', f'{FOLDER[sid]}-day{day}', '*lecture*.ipynb'))
    if not nbs: sys.exit(f'노트북 없음: {FOLDER[sid]}-day{day}')
    nb_path = nbs[0]
    work = tempfile.mkdtemp(prefix=f'cellout_{sid}{day}_')
    for f in os.listdir(os.path.dirname(nb_path)):                      # mynn_day4.py 같은 동봉 파일
        p = os.path.join(os.path.dirname(nb_path), f)
        if os.path.isfile(p): shutil.copy(p, work)
    os.chdir(work); sys.path.insert(0, work)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    _stub_ipython()
    g = {'__name__': '__main__'}
    nb = json.load(open(nb_path, encoding='utf8'))
    signal.signal(signal.SIGALRM, _alarm)

    def execute(src, label):
        buf = io.StringIO(); t0 = time.time(); err = None
        signal.alarm(timeout)
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(_prep(src), label, 'exec'), g)
        except CellTimeout:
            err = 'timeout'
        except BaseException as e:                                     # SystemExit 포함
            err = f'{type(e).__name__}: {e}'
        finally:
            signal.alarm(0)
            try:
                import matplotlib.pyplot as plt; plt.close('all')
            except Exception:
                pass
        print(f'  [{label}] {time.time() - t0:5.1f}s {err or "ok"}', file=sys.__stderr__, flush=True)
        return buf.getvalue(), err

    res, pending, last = {}, False, None       # pending: 실행을 미룬 빈칸 셀 / last: 방금 실행한 정답 번호
    deferred = []                               # 빈칸 뒤 · 정답 앞에 온 셀 (NN: 빈칸 → 검증 → 정답 순서)

    def values(src):
        """assert 의 왼쪽 식을 실제로 계산한 값 — 검증 셀이 ✅ 만 찍을 때 보여 줄 것"""
        lines = []
        for expr, _, sym in nbparse.assert_parts(src):
            if not sym: continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    v = eval(expr, g)
                lines.append(f'{expr} {"=" if sym == "=" else "→"} {_fmt(v)}')
            except Exception:
                pass
        return '\n'.join(lines)

    def after(num, i, src):
        out, err = execute(src, f'셀 {i}')
        if num and is_check(src):
            r = res[num]
            if not _clean(r['text'] + out) and not err: out = values(src)   # 정답 · 검증 셀 모두 아무것도 안 찍었을 때만
            r['text'] += ('\n' if r['text'] and out else '') + out; r['check'] = True
            if err and not r['err']: r['err'] = err
            return True
        return False

    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code': continue
        src = nbparse._src(c).strip('\n')
        if not src.strip(): continue
        first = src.split('\n', 1)[0]
        if '정답 보기' in first or ('b64decode' in src and '@title' in first):
            num, sol = nbparse._solution(nbparse._answer(src))
            if num and sol and pending:
                out, err = execute(sol, f'정답 {num}')
                res[num] = {'text': out, 'err': err, 'check': False}
                last = num
                for j, d in deferred:
                    if after(last, j, d) is False: last = None
            pending, deferred = False, []
            continue
        if src.lstrip().startswith('#@title') and 'display(' in src:   # 그림 셀
            continue
        if '____' in src:
            pending = True; last = None; continue
        if pending:
            deferred.append((i, src)); continue
        if not after(last, i, src): last = None
        else: last = None

    final = {}
    for num, r in res.items():
        text = _clean(r['text'])
        ok = text and not r['err']
        print(f'{num:6s} {"OK " if ok else "-- "} {(r["err"] or "")[:70]}', file=sys.__stderr__)
        if ok:
            final[num] = {'text': text, 'check': r['check']}
    return final


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    timeout = 600
    if '--timeout' in sys.argv: timeout = int(sys.argv[sys.argv.index('--timeout') + 1]); args = args[:2]
    sid, day = args[0], int(args[1])
    repo = os.path.dirname(HERE)
    final = run(sid, day, timeout)
    path = os.path.join(repo, 'notes', f'cell_outputs.{sid}{day}.json')
    old = json.load(open(path, encoding='utf8')) if os.path.exists(path) else {}
    for k, v in old.items():                                            # 손으로 고친 항목(fixed)은 지키고, 이번에 실패한 것은 이전 값 유지
        if v.get('fixed') or k not in final: final[k] = v
    with open(path, 'w', encoding='utf8') as f:
        json.dump(dict(sorted(final.items(), key=lambda kv: [int(x) if x.isdigit() else x for x in re.split(r'[-·]', kv[0])])),
                  f, ensure_ascii=False, indent=1)
        f.write('\n')
    print(f'→ {path}  {len(final)}개', file=sys.__stderr__)


if __name__ == '__main__':
    main()
