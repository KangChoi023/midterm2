#!/usr/bin/env python3
"""강의 교안(.html · .ipynb · .pptx)을 교시별 글로 뽑아 본다 — 노트를 쓰기 전에 읽는 용도.

    python3 -I tools/dump_lecture.py <교안 파일> [full]

full 을 주면 토픽 본문 전체와 코드 블록 · 출력까지, 아니면 토픽마다 300자.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lecture_parse, nbparse


def main():
    path, full = sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == 'full'
    html = path.endswith('.html')
    lessons = lecture_parse.parse(path) if html else nbparse.parse(path)
    print('Day 제목:', lecture_parse.day_title(path) if html else nbparse.day_title(path))
    for les in lessons:
        if not les: continue
        print(f"\n=== {les['n']}교시 · {les['title']}")
        for g in les['goals']: print('  목표:', g)
        for t in les['topics']:
            if t['title'] in ('정답 맞추기', '다음 시간') or t['title'].startswith(('확인', '정답')): continue
            print(f"- {t['title']}: {t['text'] if full else t['text'][:300]}")
            if full:
                for b in t['blocks']:
                    if b['kind'] == 'out': print('  [출력] ' + b['text'][:300].replace('\n', ' | '))
                    else: print(f"  ```{b['lang']}\n  " + b['text'][:1500].replace('\n', '\n  ') + '\n  ```')
        for f in les['fills']:
            print('  ✏️', f['title'], '|', nbparse._text(f['w'])[:300] if full else '')


if __name__ == '__main__':
    main()
