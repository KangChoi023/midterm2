/* 교시 안의 "같은 개념" 연결 계산 (DOM 없음 — node 로도 테스트 가능).
 *
 * 개념 = 노트의 핵심 용어(t0, t1…) + 새로 나온 함수·명령(f0, f1…).
 * 각 개념의 이름에서 검색 키를 뽑아, 아래 단위들에 그 키가 나오는지 찾는다.
 *   - 핵심 내용의 글머리 한 줄(하위 줄 포함)
 *   - 기본 사용형태 코드의 한 줄
 *   - 강의 예제(코드·명령) / 강의 토픽 본문
 *   - 연관 문제
 * 또 "예제 코드가 하는 일" 표의 각 행을 강의 예제(부록 A)와 짝짓는다.
 */
(function (root) {
  'use strict';

  const escRe = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  function stripMd(t) {
    return String(t || '').replace(/```[\w+-]*\n?/g, ' ').replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/\*\*|__|`/g, '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
  }

  // 이름 → 검색 키.  "메모리 맵 I/O(memory-mapped I/O)" → [메모리 맵 I/O, memory-mapped I/O]
  //                  "argc / argv" → [argc / argv, argc, argv]     "`static` 지역 변수" → [static 지역 변수]
  const STOP = new Set(['함수', '변수', '값', '코드', '파일', '배열', '포인터', '주소', '타입', '명령', 'main', 'int', 'char', 'void', 'make', 'gcc', 'C']);
  // 두 단어 한글 용어는 본문에서 뒷단어만 쓰는 경우가 많다(메모리 누수 → 누수). 너무 흔한 뒷단어는 제외
  const TAIL_STOP = new Set(['사용', '변수', '함수', '코드', '파일', '타입', '해제', '영역', '방식', '규칙', '연산', '전달', '선언', '정의', '값', '시간', '처리', '번호', '구조', '모드', '이름', '목록', '상태', '검사', '출력', '입력', '비교', '변환', '크기', '순서']);
  function termKeys(name) {
    const n = name.replace(/`/g, '').trim();
    const out = new Set();
    const m = n.match(/^(.*?)\s*[(（]([^)）]*)[)）]\s*(.*)$/);
    const parts = m ? [(m[1] + ' ' + m[3]).trim(), ...m[2].split(/[,·]/)] : [n];
    parts.forEach(p => {
      p = p.trim(); if (!p) return;
      out.add(p);
      if (/\s\/\s/.test(p)) p.split(/\s\/\s/).forEach(q => out.add(q.trim()));
      const w = p.split(/\s+/);
      if (w.length === 2 && /^[가-힣]{2,}$/.test(w[1]) && !TAIL_STOP.has(w[1])) out.add(w[1]);
    });
    return [...out].filter(k => k && !STOP.has(k) && (k.length >= 3 || /[가-힣]/.test(k) || /^[%$&*.]./.test(k)));
  }
  function funcKeys(name) {
    const n = name.replace(/`/g, '').trim();
    const out = new Set();
    if (n.length <= 24) out.add(n);
    const lead = n.match(/^[A-Za-z_%][\w%.-]*/);
    if (lead && lead[0].length >= 3) out.add(lead[0]);
    n.split(/\s\/\s/).forEach(q => { q = q.trim(); if (q.length >= 3 && q.length <= 24) out.add(q); });
    return [...out].filter(k => !STOP.has(k));
  }
  // 영문 식별자 키는 단어 경계로(rand 가 srand 에 걸리지 않게), 나머지는 부분 문자열로. 반환: 처음 나온 위치 또는 -1
  function matcher(key) {
    const lo = key.toLowerCase();
    if (/^[\x20-\x7e]+$/.test(key)) {
      const pre = /^[A-Za-z0-9_]/.test(key) ? '(^|[^A-Za-z0-9_])' : '()';
      const post = /[A-Za-z0-9_]$/.test(key) ? '(?![A-Za-z0-9_])' : '';
      const re = new RegExp(pre + escRe(lo) + post);
      return t => { const m = re.exec(t); return m ? m.index + m[1].length : -1; };
    }
    return t => t.indexOf(lo);
  }

  // 핵심 내용 md → 블록 [{bullet:true, md}] (하위 글머리는 위 글머리에 포함) / [{bullet:false, md}] (표·문단)
  function splitBlocks(md) {
    const blocks = []; let cur = null;
    for (const ln of String(md || '').split('\n')) {
      if (/^[-*] /.test(ln)) { cur = { bullet: true, md: ln.slice(2) }; blocks.push(cur); }
      else if (cur && cur.bullet && (/^\s+\S/.test(ln) || (ln.trim() === '' && false))) cur.md += '\n' + ln.replace(/^ {2}/, '');
      else if (ln.trim() === '' && cur && cur.bullet) { /* 글머리 사이 빈 줄 */ }
      else { if (!cur || cur.bullet) { cur = { bullet: false, md: '' }; blocks.push(cur); } cur.md += (cur.md ? '\n' : '') + ln; }
    }
    return blocks.filter(b => b.md.trim());
  }

  // "예제 코드가 하는 일" 표 → 행 [{cells:[md…]}], 나머지 md
  function splitTable(md) {
    const rows = [], rest = [];
    for (const ln of String(md || '').split('\n')) {
      if (!ln.startsWith('|')) { rest.push(ln); continue; }
      if (/^\|[\s:|-]+\|$/.test(ln.trim())) continue;
      const cells = ln.replace(/\\\|/g, '\u0000').trim().replace(/^\||\|$/g, '').split('|').map(c => c.replace(/\u0000/g, '|').trim());
      rows.push({ cells });
    }
    const head = rows.length && /^(예제|파일|프로그램|이름)/.test(stripMd(rows[0].cells[0])) ? rows.shift() : null;
    return { head, rows, rest: rest.join('\n').trim() };
  }

  // 표의 첫 칸("1-1 `hello.c`", "`1_open.c`", "`01_blink.c`", "`08_baseball/game.c`") → 강의 예제 토픽 id
  function rowTopic(cell, n, topics) {
    const t = stripMd(cell);
    const byTitle = re => { const x = topics.find(tp => re.test(tp.title)); return x ? x.id : null; };
    let m = t.match(/(\d+)-(\d+)/);
    if (m) { const id = byTitle(new RegExp('예제\\s*' + m[1] + '-' + m[2] + '(?!\\d)')); if (id !== null) return id; }
    const files = t.match(/[\w./-]+\.(?:c|h|py|sh|ino|cpp)\b/g) || [];
    for (const f of files) {
      const base = f.split('/').pop(); const stem = base.replace(/\.\w+$/, '');
      const k = stem.match(/^0?(\d+)_/);
      if (k) {
        const id = byTitle(new RegExp('예제\\s*(' + n + '-' + k[1] + '|0?' + k[1] + ')(?!\\d)'));
        if (id !== null) return id;
      }
      const hit = topics.find(tp => tp.blocks.some(b => b.text.includes(base)) || tp.text.includes(base));
      if (hit) return hit.id;
    }
    return null;
  }

  function build(L, n) {
    const concepts = [];
    L.terms.forEach((t, i) => concepts.push({ id: 't' + i, kind: 'term', name: t.name, def: t.def, keys: termKeys(t.name) }));
    const alias = {};      // 용어와 이름이 같은 함수·명령은 용어 하나로 합친다 (f3 → t5)
    L.funcs.forEach((f, i) => {
      const nm = f.name.replace(/`/g, '').trim().toLowerCase();
      const same = concepts.find(c => c.kind === 'term' && c.keys.some(k => k.toLowerCase() === nm));
      if (same) { same.sig = f.sig; same.use = f.use; alias['f' + i] = same.id; return; }
      concepts.push({ id: 'f' + i, kind: 'func', name: f.name, def: f.use, sig: f.sig, keys: funcKeys(f.name) });
    });
    concepts.forEach(c => { c.ms = c.keys.map(matcher); c.bullets = []; c.usage = []; c.examples = []; c.lecture = []; c.quiz = []; });
    const pos = (c, lo) => { let b = -1; for (const f of c.ms) { const i = f(lo); if (i >= 0 && (b < 0 || i < b)) b = i; } return b; };
    const hitsPos = text => { const lo = text.toLowerCase(); return concepts.map(c => ({ id: c.id, p: pos(c, lo) })).filter(x => x.p >= 0); };
    const hits = text => hitsPos(text).map(x => x.id);
    const byId = Object.fromEntries(concepts.map(c => [c.id, c]));

    const exTopics = L.topics.filter(t => t.blocks.some(b => b.kind !== 'out'));
    const topicCode = t => t.blocks.filter(b => b.kind !== 'out').map(b => b.text).join('\n');

    // 강의 예제 / 토픽 / 문항
    const topicConcepts = {};
    L.topics.forEach(t => {
      const inCode = exTopics.includes(t) ? hits(t.title + '\n' + topicCode(t)) : [];
      const inText = hits(t.title + ' ' + t.text);
      inCode.forEach(id => byId[id].examples.push(t.id));
      inText.filter(id => !inCode.includes(id)).forEach(id => byId[id].lecture.push(t.id));
      topicConcepts[t.id] = [...new Set([...inCode, ...inText])];
    });
    L.quiz.forEach((q, qi) => hits(stripMd([q.p, q.c, (q.o || []).join(' ')].join(' '))).forEach(id => byId[id].quiz.push(qi)));

    // 노트 섹션별 단위
    const sections = L.sections.map((s, si) => {
      if (s.key === 'core') {
        let k = 0;
        const blocks = splitBlocks(s.md).map(b => {
          if (!b.bullet) return b;
          const id = `c${si}-${k++}`; const text = stripMd(b.md);
          const hp = hitsPos(text);
          const cs = hp.map(x => x.id);
          // 대표 개념: 글머리에서 가장 먼저 나오는 개념(같으면 용어 우선)
          const primary = hp.length ? hp.slice().sort((a, b) => a.p - b.p || (a.id[0] === 't' ? -1 : 1))[0].id : null;
          cs.forEach(cid => byId[cid].bullets.push(id));
          // 이 글머리와 직접 이어지는 예제: 글머리 속 코드 조각(`…`)이 예제 코드에 그대로 있거나, 같은 개념을 공유
          const spans = (b.md.match(/`([^`]{3,})`/g) || []).map(x => x.slice(1, -1).trim()).filter(x => /[A-Za-z_]/.test(x));
          const scored = exTopics.map(t => {
            const code = topicCode(t);
            const sp = spans.filter(x => code.includes(x)).length;
            const sh = cs.filter(cid => byId[cid].examples.includes(t.id)).length;
            return { id: t.id, sc: sp * 2 + sh };
          }).filter(x => x.sc >= 2).sort((a, b) => b.sc - a.sc).slice(0, 2).map(x => x.id);
          return Object.assign({}, b, { id, text, concepts: cs, primary, examples: scored });
        });
        return { si, key: s.key, blocks };
      }
      if (s.key === 'usage') {
        const fences = [];
        String(s.md).replace(/```([\w+-]*)\n([\s\S]*?)```/g, (_, lang, body) => { fences.push({ lang: lang || 'c', body }); return ''; });
        fences.forEach((f, fi) => f.body.split('\n').forEach(line => {
          if (!line.trim()) return;
          hits(line).forEach(cid => byId[cid].usage.push({ lang: f.lang, line }));
        }));
        return { si, key: s.key };
      }
      if (s.key === 'examples') {
        const tb = splitTable(s.md);
        tb.rows.forEach(r => { r.topic = rowTopic(r.cells[0] || '', n, L.topics); r.concepts = hits(stripMd(r.cells.join(' '))); });
        return Object.assign({ si, key: s.key }, tb);
      }
      return { si, key: s.key };
    });
    // 표의 행과 짝지은 예제는 그 행의 개념과도 연결
    sections.filter(x => x.key === 'examples').forEach(x => x.rows.forEach(r => {
      if (r.topic === null) return;
      r.concepts.forEach(cid => { const c = byId[cid]; if (!c.examples.includes(r.topic)) c.examples.push(r.topic); c.lecture = c.lecture.filter(t => t !== r.topic); });
    }));
    concepts.forEach(c => { c.weight = c.bullets.length + c.examples.length + c.usage.length; delete c.ms; });
    return { concepts, byId, alias, sections, topicConcepts, exTopicIds: exTopics.map(t => t.id) };
  }

  const api = { build, termKeys, funcKeys, splitBlocks, splitTable, rowTopic, stripMd };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.WikiLink = api;
})(typeof window !== 'undefined' ? window : globalThis);
