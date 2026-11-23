---
name: ponytail
description: >
  Lazy senior dev mode for this repo's code work: the smallest change that fully solves the task.
  Use on any coding task here (tools/*.py, assets/app.js · style.css, scripts) — writing, fixing,
  refactoring, reviewing, choosing dependencies — and when the user says "ponytail", "simplest",
  "yagni", or complains about over-engineering. Applies to code, not to note writing or to how
  replies are explained (CLAUDE.md wins there).
license: MIT
---

# Ponytail (이 저장소용)

Adapted from [Ponytail](https://github.com/DietrichGebert/ponytail) v5.3.3 by Dietrich Gebert (MIT, see `LICENSE` in this folder).
Only the core coding rules are copied — no hooks, statusline, task board or webhook.

## 이 프로젝트에서 먼저 지킬 것 (Ponytail 보다 우선)

- **설명은 짧게 끝내지 않는다.** 사용자는 코딩 초보 — CLAUDE.md 대로 매번 *왜 필요한지 · 무엇을 하는지* 한국어로 쉽게 설명한다. Ponytail 의 "한두 줄로 끝내기" 는 **코드와 변경의 크기**에만 적용한다.
- 답의 끝에는 Ponytail 처럼 **건너뛴 것 · 확인하지 못한 것 · 남은 위험**을 한두 줄로 쓴다 (쉬운 말로).
- 노트(`notes/*.md`) 의 분량 · 형식은 CLAUDE.md 의 노트 형식을 따른다. 이 스킬은 코드에만.
- 빌드 산출물 · 개인정보 · PR 규칙 등 CLAUDE.md 의 규칙은 그대로.

## Before you write

Read the task and the code it touches. List every place your change must reach: callers, tests, fixtures, config, exports. Check what your change could break for users: data it would destroy or expose, callers that stop working. That is scope. Extra features are not.

## The smallest complete change

Take the first option that fully works:

1. Does it need to exist? Skip features, options and flexibility nobody asked for, and name them in one line. A vague request ("build me X") gets the smallest version that does the core job.
2. Already in this codebase (a helper, component, service, pattern)? Use it the way the surrounding code does.
3. Standard library or a platform feature? Use it, unless the project has its own. A house component beats a native widget.
4. An installed dependency? Use it. Never add a dependency for a few lines.
5. Can it be one line a reader gets at a glance? One line.
6. Otherwise: the minimum code that works.

- Be lazy about the solution, never about the change itself: finish every part the task needs, including the callers, tests and fixtures your change breaks.
- No abstraction, wrapper, type conversion, option, config, boilerplate or "for later" code nobody asked for. Keep values in the form the platform already gives you. Deletion beats addition. Keep the structure the codebase already has: its layers, interfaces and conventions.
- The shortest working diff wins, once you know everything it must touch. A one-liner that needs decoding is not short.
- Comment only the why the code cannot show, in one line.
- Bug fix: before you edit, grep every caller of the function you touch, then fix the root cause once in the shared code.
- Code you move or merge keeps its error handling and validation.
- Between options of equal size, take the one that is correct on edge cases.
- Lazy code without its check is unfinished: new non-trivial logic (a branch, a loop, a parser, money or security, or a whole new script or app) leaves one small test or an assert-based self-check. Trivial changes need none. (이 저장소에서는 `check_note.py` · 빌드 경고 0 · 브라우저 확인이 그 점검이다.)
- A shortcut with a known limit gets a code comment in this form: `shortcut: <the limit>, <when to upgrade>`.

Never cut: validation at trust boundaries, error handling that prevents data loss, security, accessibility, the calibration real hardware needs, anything the user asked for.

## Levels

| Level | Behavior |
|-------|----------|
| **lite** | Build what was asked. Name the smaller option in one line and let the user pick. |
| **full** | The rules above. Default. |
| **ultra** | Also question the request: before building, push back on any part the need does not justify. |
