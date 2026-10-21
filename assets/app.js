/* 학습 위키 — 해시 라우팅 단일 페이지 앱.
 * 데이터: data/index.js (목차·검색 색인·사전), data/<과목>.js (교시 본문, 필요할 때 로드)
 * 주소:   #/                 홈
 *         #/c                과목
 *         #/c/2/7?a=t3&q=…   교시 (a = 이동할 앵커, q = 강조할 검색어)
 *         #/search?q=…  #/terms  #/funcs  #/links  #/quiz?s=c&d=2
 */
(function () {
  'use strict';
  const IDX = window.WIKI_INDEX;
  const W = (window.WIKI = window.WIKI || {});
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const main = $('#main'), side = $('#side');
  const SUBJ = IDX.subjects;
  const SORDER = Object.fromEntries(SUBJ.map((s, i) => [s.id, i]));
  const sById = id => SUBJ.find(s => s.id === id);
  const KIND = { term: '용어', func: '함수·명령', kw: '키워드', note: '정리 노트', topic: '강의 개념', code: '예제 코드', quiz: '문항' };
  const KIND_W = { term: 40, func: 34, kw: 20, note: 14, topic: 12, code: 4, quiz: 0 };

  /* ---------- 유틸 ---------- */
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const store = {
    get(k, d) { try { const v = localStorage.getItem('wiki.' + k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('wiki.' + k, JSON.stringify(v)); } catch (e) { /* 저장 불가 환경 */ } },
  };
  function md(text) {
    if (!text) return '';
    if (window.marked) return marked.parse(text, { gfm: true, breaks: false });
    return '<pre>' + esc(text) + '</pre>';
  }
  function mdInline(text) {
    if (!text) return '';
    if (window.marked) return marked.parseInline(text, { gfm: true });
    return esc(text);
  }
  function highlightIn(el) {
    if (!window.hljs) return;
    $$('pre code', el).forEach(c => {
      if (c.dataset.hl) return; c.dataset.hl = 1;
      let lang = (c.className.match(/language-([\w+-]+)/) || [])[1] || c.dataset.lang;
      if (lang === 'text' || lang === 'plaintext') return;
      if (lang && !hljs.getLanguage(lang)) lang = null;
      try {
        const r = lang ? hljs.highlight(c.textContent, { language: lang, ignoreIllegals: true }) : hljs.highlightAuto(c.textContent, ['c', 'bash', 'makefile', 'python']);
        c.innerHTML = r.value;
      } catch (e) { /* 그대로 둔다 */ }
    });
  }
  const loc = (s, d, n) => `${sById(s).short} D${d}·${n}교시`;
  const href = (s, d, n, a, q) => `#/${s}/${d}/${n}` + (a || q ? '?' + new URLSearchParams(Object.assign({}, a ? { a } : {}, q ? { q } : {})).toString() : '');
  function lessonMeta(s, d, n) {
    const S = sById(s); const D = S && S.days.find(x => x.n === d);
    return D && D.lessons.find(x => x.n === n);
  }
  function flatLessons() {
    const out = [];
    SUBJ.forEach(S => S.days.forEach(D => D.lessons.forEach(L => out.push({ s: S.id, d: D.n, n: L.n, L, D, S }))));
    return out;
  }

  /* ---------- 과목 데이터 로드 ---------- */
  const loading = {};
  function loadSubject(id) {
    if (W[id]) return Promise.resolve(W[id]);
    if (loading[id]) return loading[id];
    loading[id] = new Promise((res, rej) => {
      const sc = document.createElement('script');
      sc.src = `data/${id}.js?v=${encodeURIComponent(IDX.built)}`;
      sc.onload = () => (W[id] ? res(W[id]) : rej(new Error('데이터 없음')));
      sc.onerror = () => { delete loading[id]; rej(new Error(`data/${id}.js 를 불러오지 못했습니다`)); };
      document.head.appendChild(sc);
    });
    return loading[id];
  }

  /* ---------- 사이드바 ---------- */
  function renderSide(cur) {
    const openS = store.get('sideOpen', {});
    let h = '<div class="mobile-nav"><a class="tool" href="#/terms">📖 용어 사전</a><a class="tool" href="#/funcs">🔧 함수·명령 사전</a><a class="tool" href="#/links">🔗 과목 연결</a><a class="tool" href="#/quiz">📝 퀴즈</a></div>';
    h += '<div class="stitle">과목</div>';
    SUBJ.forEach((S, si) => {
      const isCur = cur && cur.s === S.id;
      h += `<details class="subj sc-${S.id}" data-s="${S.id}" ${isCur || openS[S.id] ? 'open' : ''}><summary><span class="dot" style="background:var(--sc)"></span>${si + 1}. ${esc(S.name)}</summary>`;
      S.days.forEach(D => {
        const isCurD = isCur && cur.d === D.n;
        const done = D.lessons.filter(l => l.status === 'done').length;
        h += `<details class="day" ${isCurD || openS[S.id + D.n] ? 'open' : ''} data-k="${S.id}${D.n}"><summary>Day ${D.n} <span class="hit-loc">${esc(D.title)} · ${done}/${D.lessons.length}</span></summary>`;
        D.lessons.forEach(L => {
          const on = isCurD && cur.n === L.n;
          h += `<a class="les${on ? ' on' : ''}" href="${href(S.id, D.n, L.n)}"><span class="dot ${L.status}"></span><span>${L.n}. ${esc(L.title)}</span></a>`;
        });
        h += '</details>';
      });
      h += '</details>';
    });
    h += `<div class="stitle">빌드 ${esc(IDX.built)}</div>`;
    side.innerHTML = h;
    $$('details', side).forEach(dt => dt.addEventListener('toggle', () => {
      const o = store.get('sideOpen', {}); o[dt.dataset.s || dt.dataset.k] = dt.open; store.set('sideOpen', o);
    }));
    const on = $('.les.on', side); if (on) on.scrollIntoView({ block: 'nearest' });
  }

  /* ---------- 홈 ---------- */
  function renderHome() {
    const flow = ['C 언어 문법과 메모리', '리눅스 시스템 프로그래밍', 'ESP32 마이크로컨트롤러', '라즈베리파이 I/O 제어'];
    let h = `<div class="hero"><h1>임베디드 학습 위키</h1><p>C → LINUX → MCU → Embedded IO. 교시마다 <b>핵심 정리 · 강의 개념 · 예제 코드 · 연관 문제</b>를 한 곳에 모으고, 모르는 개념은 검색으로 처음 나온 곳까지 따라간다.</p></div><div class="cards">`;
    SUBJ.forEach((S, i) => {
      const ls = S.days.flatMap(d => d.lessons);
      const done = ls.filter(l => l.status === 'done').length;
      const nq = ls.reduce((a, l) => a + l.nq, 0), nex = ls.reduce((a, l) => a + l.nex, 0);
      h += `<a class="card sc-${S.id}" href="#/${S.id}"><div class="step">STEP ${i + 1} · ${esc(flow[i] || '')}</div><h3>${esc(S.name)}</h3>
        <div class="bar"><i style="width:${ls.length ? (done / ls.length * 100).toFixed(0) : 0}%"></i></div>
        <div class="meta">정리 ${done}/${ls.length}교시 · ${S.days.length}일<br>예제 ${nex} · 문항 ${nq}</div></a>`;
    });
    h += `</div><div class="howto"><h3>이렇게 쓰면 된다</h3><ul>
      <li><b>검색</b>(<kbd>/</kbd>): 생소한 단어를 치면 <b>정의(용어 사전)</b>와 <b>처음 나온 교시</b>가 맨 위에 뜬다. 결과를 누르면 그 자리로 이동해 강조된다.</li>
      <li><b>교시 페이지</b> = ① 정리 노트(개요 → 핵심 용어 → 새 함수 → 핵심 내용 → 사용형태) ② 부록 A 예제 코드 + 빌드·실행 ③ 부록 B 강의 개념 목록(읽은 것 체크) ④ 연관 문제.</li>
      <li>아직 정리 전인 교시(<span class="dot"></span>)도 강의 개념·예제·문항은 미리 볼 수 있고 검색에도 걸린다. 정리를 마친 교시는 <span class="dot done"></span>.</li>
      <li><a href="#/terms">용어 사전</a> · <a href="#/funcs">함수·명령 사전</a>은 교시 노트에서 자동으로 모인다. <a href="#/quiz">퀴즈</a>는 오답을 이 브라우저에 기억한다.</li></ul></div>`;
    main.innerHTML = h;
  }

  /* ---------- 과목 개요 ---------- */
  function renderSubject(s) {
    const S = sById(s); if (!S) return notFound();
    let h = `<div class="page sc-${s}"><div class="crumb"><a href="#/">홈</a></div><h1>${esc(S.name)}</h1><p class="sub">${S.days.length}일 · ${S.days.reduce((a, d) => a + d.lessons.length, 0)}교시</p>`;
    S.days.forEach(D => {
      h += `<div class="group-h">Day ${D.n} — ${esc(D.title)}</div><div class="hits">`;
      D.lessons.forEach(L => {
        h += `<a class="hit" href="${href(s, D.n, L.n)}"><span class="dot ${L.status}"></span> <span class="hit-t">${L.n}교시 · ${esc(L.title)}</span>
          <span class="hit-loc">예제 ${L.nex} · 문항 ${L.nq}</span> <span class="tag ${L.status}">${L.status === 'done' ? '정리됨' : '정리 전'}</span></a>`;
      });
      h += '</div>';
    });
    main.innerHTML = h + '</div>';
  }

  /* ---------- 코드 블록 ---------- */
  const BLK_LABEL = { code: '코드', cmd: '빌드·실행', out: '출력·결과' };
  function codeBlock(b) {
    const lang = b.kind === 'cmd' ? 'bash' : b.lang;
    return `<div class="cb ${b.kind}"><div class="cb-h"><span class="k ${b.kind}">${BLK_LABEL[b.kind]}</span><span>${esc(b.lang === 'text' ? '' : b.lang)}</span><button type="button" data-copy>복사</button></div><pre><code data-lang="${esc(lang)}">${esc(b.text)}</code></pre></div>`;
  }
  function exampleBody(t) {
    const codes = t.blocks.filter(b => b.kind === 'code');
    const run = t.blocks.filter(b => b.kind !== 'code');
    let h = codes.map(codeBlock).join('');
    if (run.length) h += `<div class="runbox"><div class="rh">${codes.length ? '부록 · ' : ''}빌드·실행 / 결과</div>${run.map(codeBlock).join('')}</div>`;
    return h;
  }
  function lectureBody(t) {
    return t.html.replace(/\[\[BLOCK (\d+)\]\]/g, (_, i) => codeBlock(t.blocks[+i]));
  }

  /* ---------- 교시 페이지: 같은 개념끼리 연결 ----------
   * CX = 지금 보고 있는 교시 { s, d, n, L, LK }.  LK = WikiLink.build(L) (assets/concepts.js)
   * 칩(data-peek-c / data-peek-x)을 누르면 옆 패널에 그 개념/예제의 정의·핵심 내용·코드·문항을 모아 보여 준다. */
  let CX = null;
  const shortName = c => c.name.replace(/`/g, '').replace(/\s*[(（][^)）]*[)）]\s*/g, ' ').trim() || c.name;
  const cid = id => (CX.LK.alias[id] || id);
  function cChip(id, cls = '') {
    const c = CX.LK.byId[cid(id)]; if (!c) return '';
    return `<button type="button" class="cchip ${c.kind} ${cls}" data-peek-c="${c.id}" title="${esc(c.name)} — 관련 내용 모아 보기">${esc(shortName(c))}</button>`;
  }
  function exLabel(t) { const m = t.title.match(/예제\s*([\d-]+)/); return m ? '예제 ' + m[1] : t.title.replace(/^\d+-\d+\.\s*/, '').slice(0, 16); }
  function exChip(tid) { const t = CX.L.topics[tid]; return t ? `<button type="button" class="xchip" data-peek-x="${tid}" title="${esc(t.title)}">▶ ${esc(exLabel(t))}</button>` : ''; }
  function countBtn(c) {
    const parts = [c.bullets.length && `핵심 ${c.bullets.length}`, c.usage.length && `사용형태 ${c.usage.length}`, c.examples.length && `예제 ${c.examples.length}`, c.quiz.length && `문항 ${c.quiz.length}`].filter(Boolean);
    return parts.length ? `<button type="button" class="cnt" data-peek-c="${c.id}">${parts.join(' · ')} ›</button>` : '';
  }
  function bulletLi(b, skip) {
    const cs = b.concepts.slice().sort((x, y) => (x === b.primary ? -1 : y === b.primary ? 1 : 0)).filter(x => x !== skip).slice(0, 4);
    const rel = cs.map(x => cChip(x)).join('') + b.examples.map(exChip).join('');
    return `<li id="${b.id}">${md(b.md)}${rel ? `<div class="rel">${rel}</div>` : ''}</li>`;
  }
  function coreHtml(sec) {
    let h = '', open = false;
    sec.blocks.forEach(b => {
      if (b.bullet) { if (!open) { h += '<ul class="core-list">'; open = true; } h += bulletLi(b); }
      else { if (open) { h += '</ul>'; open = false; } h += md(b.md); }
    });
    return h + (open ? '</ul>' : '');
  }
  function examplesTable(sec) {
    if (!sec.rows.length) return md(sec.rest);
    const head = sec.head ? sec.head.cells : ['예제', '하는 일'];
    return `<table class="ex-tab"><thead><tr>${head.map(c => `<th>${mdInline(c)}</th>`).join('')}</tr></thead><tbody>` +
      sec.rows.map(r => `<tr>${r.cells.map((c, i) => `<td>${i === 0 && r.topic !== null ? `<button type="button" class="xrow" data-peek-x="${r.topic}" title="예제 코드 옆에 펼치기">${mdInline(c)} <span>▶</span></button>` : mdInline(c)}</td>`).join('')}</tr>`).join('') +
      '</tbody></table>' + md(sec.rest);
  }
  function noteSection(s, si, L) {
    const LS = CX.LK.sections[si] || {};
    const cls = s.key === 'overview' ? 'nsec overview' : 'nsec';
    let body = '';
    if (s.key === 'kw') {
      body = '<div class="kw">' + L.keywords.map(k => `<a href="#/search?q=${encodeURIComponent(k)}">${esc(k)}</a>`).join('') + '</div>';
    } else if (s.key === 'terms' && s.tn) {
      body = '<dl class="terms">' + L.terms.slice(s.t0, s.t0 + s.tn).map((t, i) => {
        const c = CX.LK.byId['t' + (s.t0 + i)];
        return `<div id="t${s.t0 + i}"><dt>${mdInline(t.name)}</dt><dd>${mdInline(t.def)}${c && c.sig ? `<div class="sig">${mdInline(c.sig)}</div>` : ''}${c ? countBtn(c) : ''}</dd></div>`;
      }).join('') + '</dl>' + md(s.rest);
    } else if (s.key === 'funcs' && s.fn) {
      body = '<table class="ftab"><thead><tr><th>이름</th><th>기본형</th><th>용도</th></tr></thead><tbody>' +
        L.funcs.slice(s.f0, s.f0 + s.fn).map((f, i) => {
          const id = 'f' + (s.f0 + i);
          return `<tr id="${id}"><td><button type="button" class="fname" data-peek-c="${cid(id)}">${esc(f.name)}</button></td><td>${mdInline(f.sig)}</td><td>${mdInline(f.use)}</td></tr>`;
        }).join('') + '</tbody></table>' + md(s.rest);
    } else if (s.key === 'core' && LS.blocks) body = coreHtml(LS);
    else if (s.key === 'examples' && LS.rows) body = examplesTable(LS);
    else body = md(s.md);
    const id = s.key === 'kw' ? 's-kw' : 's' + si;
    return `<section class="${cls}" id="${id}">${s.label ? `<h3>${mdInline(s.label)}</h3>` : ''}${body}</section>`;
  }

  // 개념별 묶어 보기: 핵심 용어·새 함수·핵심 내용을 개념 카드로 재배치 (카드 = 정의 + 그 개념의 글머리 + 사용형태 + 예제 코드)
  function conceptCards() {
    const { L, LK } = CX;
    const bullets = LK.sections.filter(x => x.blocks).flatMap(x => x.blocks.filter(b => b.bullet));
    const own = {}; const loose = [];
    bullets.forEach((b, i) => { if (b.primary) (own[b.primary] = own[b.primary] || []).push(Object.assign({ i }, b)); else loose.push(b); });
    const first = c => (own[c.id] ? own[c.id][0].i : 1e3 + LK.concepts.indexOf(c));
    const list = LK.concepts.filter(c => c.kind === 'term' || own[c.id] || c.examples.length).sort((a, b) => first(a) - first(b));
    let h = list.map(c => {
      const bs = own[c.id] || [];
      const usage = c.usage.length ? `<pre><code data-lang="${esc(c.usage[0].lang)}">${esc(c.usage.map(u => u.line).join('\n'))}</code></pre>` : '';
      const exs = c.examples.map(tid => { const t = L.topics[tid]; return `<details class="box ex"><summary><span class="t">${esc(t.title)}</span></summary><div class="in">${exampleBody(t)}</div></details>`; }).join('');
      const more = [...c.bullets.filter(id => !bs.some(b => b.id === id)).map(id => `<button type="button" class="cnt" data-go="${id}">다른 글머리에서도 언급 ↗</button>`).slice(0, 1),
        c.quiz.length ? `<button type="button" class="cnt" data-peek-c="${c.id}">문항 ${c.quiz.length} ›</button>` : ''].join('');
      return `<section class="ccard" id="${c.id}"><h3>${esc(c.name)} <span class="tag">${c.kind === 'term' ? '용어' : '함수·명령'}</span></h3>
        <div class="cdef">${mdInline(c.def || '')}${c.sig ? `<div class="sig">${mdInline(c.sig)}</div>` : ''}</div>
        ${bs.length ? `<ul class="core-list">${bs.map(b => bulletLi(b, c.id)).join('')}</ul>` : ''}${usage}${exs}${more ? `<div class="rel">${more}</div>` : ''}</section>`;
    }).join('');
    if (loose.length) h += `<section class="ccard loose"><h3>그 밖의 핵심 내용</h3><ul class="core-list">${loose.map(b => bulletLi(b)).join('')}</ul></section>`;
    return h;
  }
  function notesHtml(view) {
    const { L } = CX;
    if (!L.sections.length) return '';
    if (view !== 'concept') return L.sections.map((x, i) => noteSection(x, i, L)).join('');
    let h = '', placed = false;
    L.sections.forEach((x, i) => {
      if (['terms', 'funcs', 'core'].includes(x.key)) { if (!placed) { h += `<div class="cards-h">개념별 정리 <small>정의 · 그 개념의 핵심 내용 · 사용형태 · 예제 코드를 한 카드에</small></div>` + conceptCards(); placed = true; } return; }
      h += noteSection(x, i, L);
    });
    return h;
  }

  /* 옆 패널 */
  const peekEl = $('#peek'); let peekStack = [];
  function openPeek(kind, id) { peekStack.push([kind, id]); renderPeek(); }
  function closePeek() { peekStack = []; peekEl.hidden = true; document.body.classList.remove('peek-on'); }
  function findBullet(id) { for (const s of CX.LK.sections) if (s.blocks) { const b = s.blocks.find(x => x.id === id); if (b) return b; } return null; }
  function rowOf(tid) { for (const s of CX.LK.sections) if (s.rows) { const r = s.rows.find(x => x.topic === tid); if (r) return r; } return null; }
  const pkSec = (t, body) => `<section class="pk-sec"><h4>${t}</h4>${body}</section>`;
  function renderPeek() {
    if (!peekStack.length || !CX) return closePeek();
    const [kind, id] = peekStack[peekStack.length - 1]; const { L, LK } = CX;
    let title = '', h = '';
    if (kind === 'c') {
      const c = LK.byId[id]; if (!c) return closePeek();
      title = `<span class="tag">${c.kind === 'term' ? '용어' : '함수·명령'}</span> ${esc(c.name)}`;
      h += `<div class="pk-def">${mdInline(c.def || '')}${c.sig ? `<div class="sig">${mdInline(c.sig)}</div>` : ''}</div>`;
      if (c.bullets.length) h += pkSec(`핵심 내용 ${c.bullets.length}`, c.bullets.map(bid => { const b = findBullet(bid); return b ? `<div class="pk-item">${md(b.md)}<button type="button" class="go" data-go="${bid}">본문 ↗</button></div>` : ''; }).join(''));
      if (c.usage.length) h += pkSec('기본 사용형태', `<pre><code data-lang="${esc(c.usage[0].lang)}">${esc(c.usage.map(u => u.line).join('\n'))}</code></pre>`);
      if (c.examples.length) h += pkSec(`예제 코드 ${c.examples.length}`, c.examples.map((tid, i) => { const t = L.topics[tid]; const r = rowOf(tid);
        return `<details class="box ex" ${i === 0 ? 'open' : ''}><summary><span class="t">${esc(t.title)}</span></summary><div class="in">${r && r.cells[1] ? `<p class="pk-what">${mdInline(r.cells[1])}</p>` : ''}${exampleBody(t)}<button type="button" class="go" data-go="X${tid}">부록 A ↗</button></div></details>`; }).join(''));
      const co = {}; c.bullets.forEach(bid => { const b = findBullet(bid); b && b.concepts.forEach(x => { if (x !== id) co[x] = (co[x] || 0) + 1; }); });
      const coIds = Object.keys(co).sort((a, b) => co[b] - co[a]).slice(0, 8);
      if (coIds.length) h += pkSec('함께 나오는 개념', `<div class="rel">${coIds.map(x => cChip(x)).join('')}</div>`);
      if (c.lecture.length) h += pkSec('강의에서 언급', `<div class="rel">${c.lecture.map(t => `<button type="button" class="cnt" data-go="L${t}">${esc(L.topics[t].title)} ↗</button>`).join('')}</div>`);
      if (c.quiz.length) h += pkSec(`연관 문제 ${c.quiz.length}`, `<div class="rel">${c.quiz.map(q => `<button type="button" class="cnt" data-go="q${q}">Q${q + 1} ↗</button>`).join('')}</div>`);
      h += `<p class="pk-foot"><a href="#/search?q=${encodeURIComponent(shortName(c))}">다른 교시에서 찾기 →</a></p>`;
    } else {
      const t = L.topics[id]; if (!t) return closePeek();
      title = esc(t.title);
      const r = rowOf(+id);
      if (r && r.cells[1]) h += `<div class="pk-def">${mdInline(r.cells[1])}</div>`;
      const cs = (LK.topicConcepts[id] || []).filter(x => LK.byId[x]);
      if (cs.length) h += pkSec('이 예제의 개념', `<div class="rel">${cs.map(x => cChip(x)).join('')}</div>`);
      const bs = LK.sections.filter(x => x.blocks).flatMap(x => x.blocks.filter(b => b.bullet && b.examples.includes(+id)));
      if (bs.length) h += pkSec('이어지는 핵심 내용', bs.map(b => `<div class="pk-item">${md(b.md)}<button type="button" class="go" data-go="${b.id}">본문 ↗</button></div>`).join(''));
      h += pkSec('코드', exampleBody(t) + `<button type="button" class="go" data-go="L${t.id}">강의 원문 ↗</button>`);
    }
    peekEl.innerHTML = `<div class="pk-h">${peekStack.length > 1 ? '<button type="button" class="icon-btn" data-pk-back aria-label="뒤로">←</button>' : ''}<div class="pk-t">${title}</div><button type="button" class="icon-btn" data-pk-close aria-label="닫기">✕</button></div><div class="pk-b">${h}</div>`;
    peekEl.hidden = false; document.body.classList.add('peek-on');
    highlightIn(peekEl); $('.pk-b', peekEl).scrollTop = 0;
  }
  function revealInMain(id) {
    const el = document.getElementById(id); if (!el || !main.contains(el)) return;
    if (matchMedia('(max-width: 1199px)').matches) closePeek();
    let p = el; while (p) { if (p.tagName === 'DETAILS') p.open = true; p = p.parentElement; }
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
  }
  document.addEventListener('click', e => {
    const b = e.target.closest('[data-peek-c],[data-peek-x],[data-go],[data-pk-back],[data-pk-close],[data-view]'); if (!b || !CX) return;
    if (b.dataset.peekC) { e.preventDefault(); openPeek('c', b.dataset.peekC); }
    else if (b.dataset.peekX !== undefined) { e.preventDefault(); openPeek('x', +b.dataset.peekX); }
    else if (b.dataset.go) revealInMain(b.dataset.go);
    else if (b.dataset.pkBack !== undefined) { peekStack.pop(); renderPeek(); }
    else if (b.dataset.pkClose !== undefined) closePeek();
    else if (b.dataset.view) { store.set('noteView', b.dataset.view); renderNotesBody(b.dataset.view); }
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && !peekEl.hidden) closePeek(); });
  function renderNotesBody(view) {
    const box = $('#notes-body', main); if (!box) return;
    box.innerHTML = notesHtml(view);
    $$('[data-view]', main).forEach(x => x.classList.toggle('on', x.dataset.view === view));
    highlightIn(box);
  }

  function quizCard(q, i, key) {
    const ty = q.ty || 'mc';
    const tag = { ox: '[O/X] ', multi: '[복수선택] ', match: '[짝짓기] ', order: '[순서] ' }[ty] || '';
    let h = `<div class="q" id="q${i}" data-key="${esc(key)}"><div class="q-p"><span class="n">Q${i + 1}</span>${tag}${q.p}</div>`;
    if (q.c) h += `<div class="q-body"><pre><code data-lang="c">${esc(q.c)}</code></pre></div>`;
    if (ty === 'mc' || ty === 'multi') h += '<div class="opts">' + q.o.map((o, k) => `<button type="button" class="opt" data-k="${k}">${'①②③④⑤⑥⑦⑧'[k]} ${esc(o)}</button>`).join('') + '</div>' + (ty === 'multi' ? '<button class="btn" data-check>채점</button>' : '');
    else if (ty === 'ox') h += '<div class="opts" style="grid-template-columns:1fr 1fr"><button class="opt" data-k="1">O (참)</button><button class="opt" data-k="0">X (거짓)</button></div>';
    else if (ty === 'match') h += `<div class="q-list">왼쪽: ${q.left.map(esc).join(' / ')}<br>오른쪽: ${q.right.map(esc).join(' / ')}</div><button class="btn" data-show>정답 보기</button>`;
    else if (ty === 'order') h += `<div class="q-list">항목: ${q.steps.map(esc).join(' / ')}</div><button class="btn" data-show>정답 보기</button>`;
    return h + '<div class="q-ans" hidden></div></div>';
  }
  function answerText(q) {
    const ty = q.ty || 'mc', C = '①②③④⑤⑥⑦⑧';
    if (ty === 'ox') return q.a ? 'O (참)' : 'X (거짓)';
    if (ty === 'multi') return q.a.map(i => `${C[i]} ${q.o[i]}`).join(', ');
    if (ty === 'match') return q.a.map((v, k) => `${q.left[k]} → ${q.right[v]}`).join(' / ');
    if (ty === 'order') return q.a.map(i => q.steps[i]).join(' → ');
    return `${C[q.a]} ${q.o[q.a]}`;
  }
  function bindQuiz(root, qs, onResult) {
    $$('.q', root).forEach(card => {
      const i = +card.id.replace(/^q/, ''); const q = qs[i]; if (!q) return;
      const ty = q.ty || 'mc'; const ans = $('.q-ans', card);
      const show = ok => {
        ans.hidden = false; ans.className = 'q-ans' + (ok === true ? ' ok' : ok === false ? ' no' : '');
        ans.innerHTML = `${ok === true ? '✅ 정답' : ok === false ? '❌ 오답' : '정답'} — <b>${esc(answerText(q))}</b><br>${mdInline(q.w || '')}`;
        if (ok !== null && onResult) onResult(card.dataset.key, ok);
      };
      card.addEventListener('click', e => {
        const b = e.target.closest('button'); if (!b) return;
        if (b.dataset.show !== undefined) return show(null);
        if (ty === 'multi') {
          if (b.dataset.check !== undefined) {
            const sel = $$('.opt.sel', card).map(x => +x.dataset.k).sort().join(',');
            $$('.opt', card).forEach(x => { if (q.a.includes(+x.dataset.k)) x.classList.add('right'); else if (x.classList.contains('sel')) x.classList.add('wrong'); });
            return show(sel === q.a.slice().sort().join(','));
          }
          if (b.classList.contains('opt')) b.classList.toggle('sel');
          return;
        }
        if (!b.classList.contains('opt') || card.dataset.done) return;
        card.dataset.done = 1;
        const k = +b.dataset.k, right = ty === 'ox' ? (q.a ? 1 : 0) : q.a;
        $$('.opt', card).forEach(x => { if (+x.dataset.k === right) x.classList.add('right'); });
        if (k !== right) b.classList.add('wrong');
        show(k === right);
      });
    });
  }

  async function renderLesson(s, d, n, params) {
    const S = sById(s); if (!S) return notFound();
    main.innerHTML = '<div class="empty">불러오는 중…</div>';
    let data;
    try { data = await loadSubject(s); } catch (e) { main.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
    const D = data.days.find(x => x.n === d); const L = D && D.lessons.find(x => x.n === n);
    if (!L) return notFound();
    const exTopics = L.topics.filter(t => t.blocks.some(b => b.kind !== 'out'));
    const readKey = `${s}|${d}|${n}`; const readSet = new Set(store.get('read', {})[readKey] || []);
    let h = `<article class="lesson sc-${s}"><div class="l-head"><div class="crumb"><a href="#/">홈</a> › <a href="#/${s}">${esc(S.name)}</a> › Day ${d} ${esc(D.title)}</div>
      <h1>${n}교시 · ${esc(L.title)}</h1><div class="tags"><span class="tag s">${esc(S.short)} Day ${d}</span>
      <span class="tag ${L.status}">${L.status === 'done' ? '정리됨' : '정리 전'}</span>
      ${L.lectureTitle && L.lectureTitle !== L.title ? `<span class="tag">강의: ${esc(L.lectureTitle)}</span>` : ''}</div>
      ${L.goals.length ? `<ul class="goals"><b>학습 목표 (강의)</b>${L.goals.map(g => `<li>${esc(g)}</li>`).join('')}</ul>` : ''}</div>
      <nav class="jump"><a href="#notes" data-jump>정리 노트</a><a href="#examples" data-jump>부록 A · 예제 코드 ${exTopics.length}</a><a href="#lecture" data-jump>부록 B · 강의 개념 ${L.topics.length}</a><a href="#quiz" data-jump>연관 문제 ${L.quiz.length}</a></nav>`;

    CX = { s, d, n, L, LK: window.WikiLink.build(L, n) }; peekStack = []; closePeek();
    const anchor = params.get('a');
    // 검색에서 노트 섹션(s3 등)으로 오면 그 섹션이 있는 노트 순서 보기로
    const view = params.get('v') || (anchor && /^s\d/.test(anchor) ? 'flow' : store.get('noteView', 'flow'));
    h += `<section class="part" id="notes"><h2>정리 노트 ${L.sections.length ? `<span class="view-tg"><button type="button" data-view="flow" class="${view === 'flow' ? 'on' : ''}">노트 순서</button><button type="button" data-view="concept" class="${view === 'concept' ? 'on' : ''}">개념별 묶어 보기</button></span>` : ''}</h2>`;
    if (L.sections.length) h += `<p class="part-note">점선 칩(개념)·▶ 칩(예제)을 누르면 관련 정의·핵심 내용·코드·문항이 옆에 모입니다.</p><div id="notes-body">${notesHtml(view)}</div>`;
    else h += `<div class="empty">아직 정리 전인 교시입니다. 공부하면서 <code>summary/.src/${{ c: 'C', linux: 'LINUX', mcu: 'MCU', eio: 'EIO' }[s]}_day${d}.md</code> 에 <b>키워드 · 개요 · 핵심 용어 · 새로 나온 함수 · 예제 코드가 하는 일 · 핵심 내용 · 기본 사용형태 · 연결</b> 순서로 채우면 여기에 나타납니다. 그 전까지는 아래 강의 개념 목록과 예제 코드로 먼저 훑어보세요.</div>`;
    h += '</section>';

    h += `<section class="part" id="examples"><h2>부록 A · 예제 코드 모음 <small>강의 예제의 핵심 코드와 빌드·실행 명령, 결과</small></h2>
      <div class="tools-row"><button class="btn" data-all="examples">모두 펼치기</button></div>`;
    h += exTopics.length ? exTopics.map(t => {
      const kinds = [...new Set(t.blocks.map(b => b.kind))].map(k => `<span class="tag">${BLK_LABEL[k]}</span>`).join('');
      const cs = (CX.LK.topicConcepts[t.id] || []).filter(x => CX.LK.byId[x]);
      const r = rowOf(t.id);
      return `<details class="box ex" id="X${t.id}"><summary><span class="t">${esc(t.title)}</span><span class="sp">${kinds}</span></summary><div class="in">${r && r.cells[1] ? `<p class="pk-what">${mdInline(r.cells[1])}</p>` : ''}${cs.length ? `<div class="rel">${cs.map(x => cChip(x)).join('')}</div>` : ''}${exampleBody(t)}</div></details>`;
    }).join('') : '<div class="empty">이 교시에는 강의 예제 코드가 없습니다.</div>';
    h += '</section>';

    h += `<section class="part" id="lecture"><h2>부록 B · 강의 개념 목록 <small>강의 원문을 주제별로. 읽고 정리했으면 체크</small></h2>
      <div class="tools-row"><button class="btn" data-all="lecture">모두 펼치기</button><span class="hit-loc" id="readCnt"></span></div>`;
    h += L.topics.map(t => `<details class="box lec${readSet.has(t.id) ? ' read' : ''}" id="L${t.id}"><summary><input type="checkbox" class="chk" data-t="${t.id}" ${readSet.has(t.id) ? 'checked' : ''} aria-label="읽음 표시"><span class="t">${esc(t.title)}</span>${t.example ? '<span class="sp"><span class="tag">예제</span></span>' : ''}</summary><div class="in lec-body">${lectureBody(t)}</div></details>`).join('') || '<div class="empty">강의 자료가 없습니다.</div>';
    h += '</section>';

    h += `<section class="part" id="quiz"><h2>연관 문제 <small>worksheet ${L.quiz.length}문항 — 보기를 누르면 채점</small></h2>
      <div class="tools-row"><button class="btn" data-reveal>정답 모두 보기</button></div>`;
    h += L.quiz.map((q, i) => quizCard(q, i, `${s}|${d}|${n}|${i}`)).join('') || '<div class="empty">문항이 없습니다.</div>';
    h += '</section>';

    const all = flatLessons(); const k = all.findIndex(x => x.s === s && x.d === d && x.n === n);
    const pv = all[k - 1], nx = all[k + 1];
    h += `<div class="pager">${pv ? `<a href="${href(pv.s, pv.d, pv.n)}"><small>← 이전 · ${loc(pv.s, pv.d, pv.n)}</small>${esc(pv.L.title)}</a>` : '<span></span>'}${nx ? `<a class="nx" href="${href(nx.s, nx.d, nx.n)}"><small>다음 · ${loc(nx.s, nx.d, nx.n)} →</small>${esc(nx.L.title)}</a>` : '<span></span>'}</div></article>`;
    main.innerHTML = h;
    highlightIn(main);

    const updRead = () => { $('#readCnt').textContent = `읽음 ${readSet.size}/${L.topics.length}`; };
    updRead();
    $$('.chk', main).forEach(c => {
      c.addEventListener('click', e => e.stopPropagation());
      c.addEventListener('change', () => {
        const id = +c.dataset.t; c.checked ? readSet.add(id) : readSet.delete(id);
        c.closest('details').classList.toggle('read', c.checked);
        const all = store.get('read', {}); all[readKey] = [...readSet]; store.set('read', all); updRead();
      });
    });
    $$('[data-all]', main).forEach(b => b.addEventListener('click', () => {
      const ds = $$(`#${b.dataset.all} details`, main); const open = !ds.every(x => x.open);
      ds.forEach(x => (x.open = open)); b.textContent = open ? '모두 접기' : '모두 펼치기';
    }));
    $('[data-reveal]', main)?.addEventListener('click', () => $$('.q', main).forEach((c, i) => {
      const a = $('.q-ans', c); a.hidden = false; a.className = 'q-ans'; a.innerHTML = `정답 — <b>${esc(answerText(L.quiz[i]))}</b><br>${mdInline(L.quiz[i].w || '')}`;
    }));
    bindQuiz($('#quiz', main), L.quiz, recordResult);
    $$('[data-jump]', main).forEach(a => a.addEventListener('click', e => { e.preventDefault(); $(a.getAttribute('href'), main).scrollIntoView({ behavior: 'smooth' }); }));
    const pk = (params.get('p') || '').match(/^([cx]):(\w+)$/);   // #/c/4/1?p=c:t3 → 그 개념 패널을 연 채로
    if (pk) openPeek(pk[1], pk[1] === 'x' ? +pk[2] : pk[2]);
    jumpTo(anchor && CX.LK.alias[anchor] && !document.getElementById(anchor) ? CX.LK.alias[anchor] : anchor, params.get('q'));
  }

  function jumpTo(anchor, q) {
    if (!anchor) { window.scrollTo(0, 0); return; }
    const el = document.getElementById(anchor);
    if (!el) { window.scrollTo(0, 0); return; }
    let p = el; while (p) { if (p.tagName === 'DETAILS') p.open = true; p = p.parentElement; }
    if (q) markText(el, q);
    requestAnimationFrame(() => {
      const target = $('mark', el) || el;
      target.scrollIntoView({ block: 'center' });
      el.classList.add('flash'); setTimeout(() => el.classList.remove('flash'), 2300);
    });
  }
  function markText(root, q) {
    const words = q.trim().toLowerCase().split(/\s+/).filter(Boolean); if (!words.length) return;
    const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT); const nodes = [];
    while (tw.nextNode()) nodes.push(tw.currentNode);
    nodes.forEach(nd => {
      const t = nd.nodeValue, lo = t.toLowerCase(); let i = 0, out = null;
      const frag = document.createDocumentFragment();
      while (true) {
        let best = -1, bw = '';
        words.forEach(w => { const j = lo.indexOf(w, i); if (j >= 0 && (best < 0 || j < best)) { best = j; bw = w; } });
        if (best < 0) break;
        out = true;
        frag.appendChild(document.createTextNode(t.slice(i, best)));
        const m = document.createElement('mark'); m.textContent = t.slice(best, best + bw.length); frag.appendChild(m);
        i = best + bw.length;
      }
      if (out) { frag.appendChild(document.createTextNode(t.slice(i))); nd.parentNode.replaceChild(frag, nd); }
    });
  }

  /* ---------- 검색 ---------- */
  // 색인 행: [과목, day, 교시, 종류, 앵커, 제목, 본문]
  const ROWS = IDX.search.map((r, i) => ({ s: r[0], d: r[1], n: r[2], k: r[3], a: r[4], t: r[5], x: r[6], lt: (r[5] || '').toLowerCase(), lx: (r[6] || '').toLowerCase(), o: SORDER[r[0]] * 1e4 + r[1] * 100 + r[2], i }));
  function search(q) {
    const words = q.trim().toLowerCase().split(/\s+/).filter(Boolean); if (!words.length) return [];
    const ql = words.join(' ');
    const hits = [];
    for (const r of ROWS) {
      let ok = true, sc = KIND_W[r.k] || 0;
      for (const w of words) {
        const inT = r.lt.includes(w), inX = r.lx.includes(w);
        if (!inT && !inX) { ok = false; break; }
        sc += inT ? 30 : 8;
      }
      if (!ok) continue;
      const base = r.lt.replace(/\s*\(.*\)\s*$/, '');
      if (r.lt === ql || base === ql || base.split(/[(（]/)[0].trim() === ql) sc += 80;
      else if (r.lt.startsWith(ql)) sc += 30;
      if (r.k === 'kw' && new RegExp('(^| · )' + ql.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '( · |$)').test(r.lx)) sc += 40;
      hits.push({ r, sc });
    }
    return hits;
  }
  function snippet(text, q, len = 150) {
    const words = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const lo = text.toLowerCase(); let p = -1;
    for (const w of words) { const j = lo.indexOf(w); if (j >= 0 && (p < 0 || j < p)) p = j; }
    let st = Math.max(0, p - 50); let s = text.slice(st, st + len);
    s = (st > 0 ? '…' : '') + s + (st + len < text.length ? '…' : '');
    let h = esc(s);
    words.forEach(w => { h = h.replace(new RegExp(esc(w).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'), m => `<mark>${m}</mark>`); });
    return h;
  }
  const hitHref = (r, q) => href(r.s, r.d, r.n, r.a, q);
  function hitHtml(r, q, cls = 'hit') {
    return `<a class="${cls}" href="${hitHref(r, q)}"><span class="hit-t">${snippet(r.t || '', q, 80)}</span><span class="hit-loc">${KIND[r.k]} · ${loc(r.s, r.d, r.n)}</span><span class="hit-x">${snippet(r.x || '', q)}</span></a>`;
  }
  function firstAppearance(hits) {
    const c = hits.filter(h => ['term', 'kw', 'note', 'topic', 'func'].includes(h.r.k));
    if (!c.length) return null;
    c.sort((a, b) => a.r.o - b.r.o || (KIND_W[b.r.k] - KIND_W[a.r.k]) || a.r.i - b.r.i);
    return c[0].r;
  }

  function renderSearch(q, kindFilter) {
    const hits = search(q);
    const defs = hits.filter(h => h.r.k === 'term' && h.sc >= 100).sort((a, b) => a.r.o - b.r.o);
    const first = firstAppearance(hits);
    let h = `<div class="page"><h1>검색: “${esc(q)}”</h1><p class="sub">${hits.length}건 · 같은 종류 안에서는 처음 나온 순서</p>`;
    if (defs.length) h += defs.slice(0, 3).map(d => `<div class="first"><div class="lbl">정의 · ${loc(d.r.s, d.r.d, d.r.n)}</div><div><b>${esc(d.r.t)}</b> — ${mdInline(d.r.x)}</div><a href="${hitHref(d.r, q)}">그 교시로 →</a></div>`).join('');
    if (first && !(defs[0] && defs[0].r === first)) h += `<div class="first"><div class="lbl">처음 나온 곳 · ${KIND[first.k]}</div>${hitHtml(first, q, 'hit')}</div>`;
    const kinds = Object.keys(KIND).filter(k => hits.some(x => x.r.k === k));
    h += '<div class="chips">' + ['all', ...kinds].map(k => `<button class="chip${(kindFilter || 'all') === k ? ' on' : ''}" data-k="${k}">${k === 'all' ? '전체' : KIND[k]} ${k === 'all' ? hits.length : hits.filter(x => x.r.k === k).length}</button>`).join('') + '</div>';
    kinds.filter(k => !kindFilter || kindFilter === 'all' || kindFilter === k).forEach(k => {
      const g = hits.filter(x => x.r.k === k).sort((a, b) => a.r.o - b.r.o || b.sc - a.sc);
      h += `<div class="group-h">${KIND[k]} ${g.length}</div><div class="hits">` + g.slice(0, 60).map(x => hitHtml(x.r, q)).join('') + (g.length > 60 ? `<div class="hit-loc">… ${g.length - 60}건 더 (검색어를 좁혀 보세요)</div>` : '') + '</div>';
    });
    if (!hits.length) h += '<div class="empty">찾지 못했습니다. 띄어쓰기를 줄이거나 함수 이름·영문 용어로 다시 검색해 보세요.</div>';
    main.innerHTML = h + '</div>';
    $$('.chip', main).forEach(c => c.addEventListener('click', () => { location.hash = `#/search?q=${encodeURIComponent(q)}${c.dataset.k === 'all' ? '' : '&k=' + c.dataset.k}`; }));
  }

  // 상단 입력창 실시간 결과
  const qIn = $('#q'), drop = $('#drop'); let dropSel = -1;
  function updateDrop() {
    const q = qIn.value.trim(); if (!q) { drop.hidden = true; return; }
    const hits = search(q).sort((a, b) => b.sc - a.sc || a.r.o - b.r.o);
    const top = hits.slice(0, 8);
    drop.innerHTML = top.map(x => hitHtml(x.r, q, '')).join('') + `<a class="more" href="#/search?q=${encodeURIComponent(q)}">전체 결과 ${hits.length}건 · 처음 나온 곳 보기 (Enter)</a>`;
    drop.hidden = false; dropSel = -1;
  }
  qIn.addEventListener('input', updateDrop);
  qIn.addEventListener('focus', updateDrop);
  qIn.addEventListener('keydown', e => {
    const items = $$('a', drop);
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault(); dropSel = (dropSel + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
      items.forEach((a, i) => a.classList.toggle('sel', i === dropSel));
    } else if (e.key === 'Enter') {
      e.preventDefault(); const a = items[dropSel];
      location.hash = a ? a.getAttribute('href') : `#/search?q=${encodeURIComponent(qIn.value.trim())}`;
      drop.hidden = true; qIn.blur();
    } else if (e.key === 'Escape') { drop.hidden = true; qIn.blur(); }
  });
  document.addEventListener('click', e => { if (!e.target.closest('.search-wrap')) drop.hidden = true; else if (e.target.closest('#drop a')) drop.hidden = true; });
  document.addEventListener('keydown', e => {
    if (e.key === '/' && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) { e.preventDefault(); qIn.focus(); qIn.select(); }
  });

  /* ---------- 용어 / 함수 / 연결 사전 ---------- */
  function dictPage(kind) {
    const isT = kind === 'terms';
    const rows = isT ? IDX.terms.map(r => ({ s: r[0], d: r[1], n: r[2], name: r[3], def: r[4], a: 't' + r[5] }))
                     : IDX.funcs.map(r => ({ s: r[0], d: r[1], n: r[2], name: r[3], def: (r[4] ? r[4] + ' — ' : '') + r[5], a: 'f' + r[6] }));
    // 같은 이름은 하나로 합치고 나온 곳을 모두 표시
    const map = new Map();
    rows.forEach(r => {
      const key = r.name.toLowerCase().replace(/\s+/g, ' ').trim();
      if (!map.has(key)) map.set(key, { name: r.name, def: r.def, src: [], o: SORDER[r.s] * 1e4 + r.d * 100 + r.n, s: r.s });
      map.get(key).src.push(r);
    });
    const list = [...map.values()].sort((a, b) => a.o - b.o);
    main.innerHTML = `<div class="page"><h1>${isT ? '📖 용어 사전' : '🔧 함수·명령 사전'}</h1>
      <p class="sub">교시 노트의 <b>${isT ? '핵심 용어' : '새로 나온 함수/명령'}</b>에서 자동으로 모은 ${list.length}개 · 처음 나온 순서. 이름을 누르면 그 교시로 이동</p>
      <input class="filter" placeholder="걸러 보기…" autofocus>
      <div class="chips">${['all', ...SUBJ.map(s => s.id)].map(k => `<button class="chip${k === 'all' ? ' on' : ''}" data-s="${k}">${k === 'all' ? '전체' : sById(k).short}</button>`).join('')}</div>
      <div class="dict">${list.map(e => `<div data-s="${e.s}" data-f="${esc((e.name + ' ' + e.def).toLowerCase())}"><span class="nm">${isT ? esc(e.name) : `<code>${esc(e.name)}</code>`}</span><span>${mdInline(e.def)}</span><span class="src">${e.src.map(r => `<a href="${href(r.s, r.d, r.n, r.a)}">${loc(r.s, r.d, r.n)}</a>`).join('')}</span></div>`).join('') || '<div class="empty">아직 없습니다.</div>'}</div></div>`;
    highlightIn(main);
    let sf = 'all';
    const apply = () => { const v = $('.filter', main).value.trim().toLowerCase(); $$('.dict > div[data-f]', main).forEach(x => { x.hidden = (sf !== 'all' && x.dataset.s !== sf) || (v && !x.dataset.f.includes(v)); }); };
    $('.filter', main).addEventListener('input', apply);
    $$('.chip', main).forEach(c => c.addEventListener('click', () => { sf = c.dataset.s; $$('.chip', main).forEach(x => x.classList.toggle('on', x === c)); apply(); }));
  }
  function linksPage() {
    const rows = IDX.links;
    let h = `<div class="page"><h1>🔗 과목 연결 지도</h1><p class="sub">교시 노트의 <b>연결</b> 칸에 적은 링크를 모은 곳. 나중 과목을 공부하다 “이건 C 에서 뭐였지?” 를 따라가는 길.</p>`;
    if (!rows.length) h += `<div class="empty">아직 연결이 없습니다. 노트의 교시 끝에 <code>**연결**</code> 칸을 만들고 <code>- [C D3·5교시 volatile](#/c/3/5) — 레지스터 접근에 그대로 쓰임</code> 처럼 적으면 여기에 모입니다.</div>`;
    const by = {};
    rows.forEach(r => { const k = r.slice(0, 3).join('|'); (by[k] = by[k] || []).push(r); });
    Object.values(by).forEach(g => {
      const [s, d, n] = g[0];
      h += `<div class="group-h"><a href="${href(s, d, n)}">${loc(s, d, n)} · ${esc((lessonMeta(s, d, n) || {}).title || '')}</a></div><div class="hits">` +
        g.map(r => `<a class="hit" href="${esc(r[4])}">→ ${esc(r[3])}</a>`).join('') + '</div>';
    });
    main.innerHTML = h + '</div>';
  }

  /* ---------- 퀴즈 ---------- */
  function recordResult(key, ok) {
    const wrong = store.get('wrong', {});
    if (ok) delete wrong[key]; else wrong[key] = Date.now();
    store.set('wrong', wrong);
  }
  async function renderQuiz(params) {
    const s = params.get('s') || 'c', dsel = params.get('d') || 'all', mode = params.get('m') || 'rand';
    main.innerHTML = '<div class="empty">문항 불러오는 중…</div>';
    let data; try { data = await loadSubject(s); } catch (e) { main.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
    const wrong = store.get('wrong', {});
    let pool = [];
    data.days.forEach(D => D.lessons.forEach(L => {
      if (dsel !== 'all' && +dsel !== D.n) return;
      L.quiz.forEach((q, i) => pool.push({ q, key: `${s}|${D.n}|${L.n}|${i}`, d: D.n, n: L.n, i }));
    }));
    if (mode === 'wrong') pool = pool.filter(p => wrong[p.key]);
    else if (mode === 'rand') { for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; } pool = pool.slice(0, 10); }
    const nWrong = Object.keys(wrong).filter(k => k.startsWith(s + '|')).length;
    const S = sById(s);
    let h = `<div class="page"><h1>📝 퀴즈</h1><p class="sub">worksheet 문항으로 연습. 틀린 문항은 오답 노트에 남고, 맞히면 빠진다 (이 브라우저에 저장).</p>
      <div class="quiz-top"><select id="qs">${SUBJ.map(x => `<option value="${x.id}" ${x.id === s ? 'selected' : ''}>${esc(x.name)}</option>`).join('')}</select>
      <select id="qd"><option value="all">전체 Day</option>${S.days.map(D => `<option value="${D.n}" ${String(D.n) === dsel ? 'selected' : ''}>Day ${D.n} · ${esc(D.title)}</option>`).join('')}</select>
      <select id="qm"><option value="rand" ${mode === 'rand' ? 'selected' : ''}>랜덤 10문제</option><option value="all" ${mode === 'all' ? 'selected' : ''}>전부 순서대로</option><option value="wrong" ${mode === 'wrong' ? 'selected' : ''}>오답 노트 (${nWrong})</option></select>
      <button class="btn pri" id="qgo">새로 출제</button><span class="score" id="score"></span></div>`;
    if (!pool.length) h += `<div class="empty">${mode === 'wrong' ? '오답 노트가 비어 있습니다 👍' : '문항이 없습니다.'}</div>`;
    h += pool.map((p, k) => quizCard(p.q, k, p.key).replace('<div class="q-p">', `<div class="crumb"><a href="${href(s, p.d, p.n, 'q' + p.i)}">${loc(s, p.d, p.n)} · Q${p.i + 1}</a></div><div class="q-p">`)).join('');
    main.innerHTML = h + '</div>';
    highlightIn(main);
    let right = 0, tried = 0;
    bindQuiz(main, pool.map(p => p.q), (key, ok) => { recordResult(key, ok); tried++; if (ok) right++; $('#score').textContent = `${right} / ${tried} 정답`; });
    const go = () => { location.hash = `#/quiz?s=${$('#qs').value}&d=${$('#qd').value}&m=${$('#qm').value}&r=${Date.now() % 1e5}`; };
    $('#qgo').addEventListener('click', go);
    $('#qs').addEventListener('change', () => { $('#qd').value = 'all'; go(); });
    $('#qd').addEventListener('change', go); $('#qm').addEventListener('change', go);
  }

  function notFound() { main.innerHTML = '<div class="empty">페이지를 찾지 못했습니다. <a href="#/">홈으로</a></div>'; }

  /* ---------- 라우터 ---------- */
  function route() {
    const raw = location.hash.replace(/^#/, '') || '/';
    const [path, qs] = raw.split('?'); const params = new URLSearchParams(qs || '');
    const seg = path.split('/').filter(Boolean);
    document.body.classList.remove('menu'); $('#scrim').hidden = true;
    closePeek(); CX = null;
    $$('.top-nav a').forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#/' + (seg[0] || '')));
    let cur = null;
    if (!seg.length) renderHome();
    else if (seg[0] === 'search') { qIn.value = params.get('q') || ''; renderSearch(params.get('q') || '', params.get('k')); }
    else if (seg[0] === 'terms' || seg[0] === 'funcs') dictPage(seg[0]);
    else if (seg[0] === 'links') linksPage();
    else if (seg[0] === 'quiz') renderQuiz(params);
    else if (sById(seg[0]) && seg.length >= 3) { cur = { s: seg[0], d: +seg[1], n: +seg[2] }; renderLesson(cur.s, cur.d, cur.n, params); }
    else if (sById(seg[0])) { cur = { s: seg[0] }; renderSubject(seg[0]); }
    else notFound();
    renderSide(cur);
    if (!(cur && cur.n)) window.scrollTo(0, 0);
    const t = cur && cur.n ? lessonMeta(cur.s, cur.d, cur.n) : null;
    document.title = t ? `${loc(cur.s, cur.d, cur.n)} ${t.title} — 학습 위키` : '임베디드 학습 위키';
  }
  window.addEventListener('hashchange', route);

  // 복사 버튼, 메뉴, 테마
  document.addEventListener('click', e => {
    const b = e.target.closest('[data-copy]'); if (!b) return;
    const code = b.closest('.cb').querySelector('code').textContent;
    (navigator.clipboard ? navigator.clipboard.writeText(code) : Promise.reject()).then(() => { b.textContent = '복사됨'; setTimeout(() => (b.textContent = '복사'), 1200); }, () => { b.textContent = '실패'; });
  });
  $('#menuBtn').addEventListener('click', () => { const o = document.body.classList.toggle('menu'); $('#scrim').hidden = !o; });
  $('#scrim').addEventListener('click', () => { document.body.classList.remove('menu'); $('#scrim').hidden = true; });
  $('#themeBtn').addEventListener('click', () => {
    const r = document.documentElement; const dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    r.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('wiki.theme', r.dataset.theme); } catch (e) { /* 무시 */ }
  });
  route();
})();
