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
  const main = $('#main'), side = $('#side'), sideToc = $('#sideToc'), sideDict = $('#sideDict');   // sideDict = 오른쪽 사전 패널 본문
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
    let h = '<div class="mobile-nav"><a class="tool" href="#/links">🔗 과목 연결</a><a class="tool" href="#/quiz">📝 퀴즈</a></div>';
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
    sideToc.innerHTML = h;
    $$('details', sideToc).forEach(dt => dt.addEventListener('toggle', () => {
      const o = store.get('sideOpen', {}); o[dt.dataset.s || dt.dataset.k] = dt.open; store.set('sideOpen', o);
    }));
    const on = $('.les.on', sideToc); if (on) side.scrollTop = Math.max(0, on.offsetTop - side.clientHeight / 2);   // 창은 움직이지 않고 목차 안에서만
    SIDE_CUR = cur; rpRefresh();
  }

  /* ---------- 왼쪽 사전 패널: 용어 + 함수·명령을 한 목록으로 ----------
   * 페이지를 옮겨도 남아 있어 교시 본문과 나란히 본다. 항목을 누르면 그 자리에서 정의·기본형·나온 교시가 펼쳐진다.
   * 검색어가 없고 교시 페이지면 "이 교시의 용어·코드" 를 먼저 보여 준다. */
  let SIDE_CUR = null, DICT = null, dictKind = 'all', dictSubj = 'all', dictRaf = 0;
  function dictBuild() {
    const map = new Map();
    const add = (kind, s, d, n, name, def, sig, a) => {
      const k = kind + '|' + name.toLowerCase().replace(/\s+/g, ' ').trim();
      if (!map.has(k)) map.set(k, { kind, name, def, sig, src: [], o: SORDER[s] * 1e4 + d * 100 + n });
      map.get(k).src.push({ s, d, n, a });
    };
    IDX.terms.forEach(r => add('term', r[0], r[1], r[2], r[3], r[4], '', 't' + r[5]));
    IDX.funcs.forEach(r => { add('code', r[0], r[1], r[2], r[3], r[5], r[4], 'f' + r[6]); if (r[7]) { const e = map.get('code|' + r[3].toLowerCase().replace(/\s+/g, ' ').trim()); if (e && !e.detail) e.detail = r[7]; } });
    DICT = [...map.values()].sort((a, b) => a.o - b.o);
    DICT.forEach((e, i) => { e.i = i; e.nameLo = e.name.toLowerCase(); e.key = (e.name + ' ' + stripTags(e.def) + ' ' + stripTags(e.sig)).toLowerCase(); e.subj = [...new Set(e.src.map(x => x.s))]; });
  }
  function dictSetup() {
    sideDict.innerHTML = `<div class="dict-top">
        <input class="dict-q" type="search" placeholder="용어·함수·명령 걸러 보기" autocomplete="off" spellcheck="false" aria-label="사전 검색">
        <div class="dict-chips" data-g="kind"><button data-v="all" class="on">전체</button><button data-v="term">용어</button><button data-v="code">코드</button></div>
        <div class="dict-chips sm" data-g="subj"><button data-v="all" class="on">전 과목</button>${SUBJ.map(x => `<button data-v="${x.id}">${esc(x.short)}</button>`).join('')}</div>
      </div><div class="dict-list"></div>
      <div class="dict-foot"><a href="#/terms">용어 전체 화면</a> · <a href="#/funcs">함수·명령 전체 화면</a></div>`;
    const q = $('.dict-q', sideDict);
    const sched = () => { if (!dictRaf) dictRaf = requestAnimationFrame(() => { dictRaf = 0; dictRender(); }); };
    q.addEventListener('input', e => { if (!e.isComposing) sched(); });
    q.addEventListener('compositionend', sched);
    q.addEventListener('keydown', e => { if (e.key === 'Escape') { q.value = ''; sched(); } });
    $$('.dict-chips', sideDict).forEach(g => g.addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      $$('button', g).forEach(x => x.classList.toggle('on', x === b));
      if (g.dataset.g === 'kind') dictKind = b.dataset.v; else dictSubj = b.dataset.v;
      sched();
    }));
  }
  function dictItem(e, words) {
    const hl = t => { let h = esc(t); words.forEach(w => { h = h.replace(new RegExp(esc(w).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'), m => `<mark>${m}</mark>`); }); return h; };
    const srcs = e.src.map(x => `<a href="${href(x.s, x.d, x.n, x.a)}">${loc(x.s, x.d, x.n)}</a>`).join('');
    return `<details class="dent" data-i="${e.i}"><summary><span class="dk ${e.kind}">${e.kind === 'term' ? '용어' : '코드'}</span><span class="dn${e.kind === 'code' ? ' mono' : ''}">${hl(e.name)}</span></summary>
      <div class="dbody">${e.detail && e.detail.role ? `<div class="drole">${esc(e.detail.role)}</div>` : ''}${e.sig ? `<code class="fsig">${esc(String(e.sig).replace(/`\s*·\s*`/g, '\n').replace(/`|\*\*/g, ''))}</code>` : ''}${e.detail ? detailHtml(e.detail, true) + `<div class="ddef note">노트: ${mdInline(e.def || '')}</div>` : `<div class="ddef">${mdInline(e.def || '')}</div>`}<div class="dsrc">${srcs}</div></div></details>`;
  }
  function dictRender() {
    if (!DICT) dictBuild();
    if (!$('.dict-q', sideDict)) dictSetup();
    const v = $('.dict-q', sideDict).value.trim().toLowerCase();
    const words = v.split(/\s+/).filter(Boolean);
    const ok = e => (dictKind === 'all' || e.kind === dictKind) && (dictSubj === 'all' || e.subj.includes(dictSubj));
    const box = $('.dict-list', sideDict);
    let h = '';
    if (!words.length && SIDE_CUR && SIDE_CUR.n) {                      // 교시 페이지: 이 교시 것 먼저
      const here = DICT.filter(e => ok(e) && e.src.some(x => x.s === SIDE_CUR.s && x.d === SIDE_CUR.d && x.n === SIDE_CUR.n));
      h += `<div class="dict-h">이 교시 (${loc(SIDE_CUR.s, SIDE_CUR.d, SIDE_CUR.n)}) · ${here.length}</div>` + (here.map(e => dictItem(e, [])).join('') || '<div class="dict-empty">이 교시에 정리된 용어·코드가 없습니다.</div>');
      h += '<div class="dict-h">검색하면 전 과목에서 찾습니다</div>';
    } else {
      const hits = DICT.filter(e => ok(e) && words.every(w => e.key.includes(w)));
      // 이름이 입력어로 시작 → 이름에 포함 → 설명에만, 같은 순위면 지금 과목 먼저, 그다음 처음 나온 순
      const rank = e => !words.length ? 0 : e.nameLo.startsWith(words[0]) ? 0 : words.every(w => e.nameLo.includes(w)) ? 1 : 2;
      const subjFirst = e => (SIDE_CUR && e.subj.includes(SIDE_CUR.s)) ? 0 : 1;
      hits.sort((a, b) => rank(a) - rank(b) || subjFirst(a) - subjFirst(b) || a.o - b.o);
      const LIM = 120;
      h += `<div class="dict-h">${words.length ? `“${esc(v)}” ` : ''}${hits.length}개</div>` + hits.slice(0, LIM).map(e => dictItem(e, words)).join('');
      if (hits.length > LIM) h += `<div class="dict-empty">… ${hits.length - LIM}개 더 — 검색어를 더 입력해 좁혀 보세요</div>`;
      if (!hits.length) h += `<div class="dict-empty">없습니다. <a href="#/search?q=${encodeURIComponent(v)}">전체 검색 →</a></div>`;
    }
    box.innerHTML = h;
    const ds = box.querySelectorAll('.dent');                                  // 하나뿐이거나 이름이 그대로 맞으면 바로 펼침
    const exact = [...ds].find(x => DICT[+x.dataset.i].nameLo === v);
    if (exact) exact.open = true; else if (words.length === 1 && ds.length === 1) ds[0].open = true;
  }
  /* ---------- 오른쪽 패널: 📖 사전 | 💻 코드 (열림·탭 상태 기억) ---------- */
  const dictPanel = $('#dictPanel'), sideCode = $('#sideCode');
  const dictIsOpen = () => !dictPanel.hidden;
  let rTab = store.get('rTab', 'dict');
  function rpSet(open, tab, focus) {
    if (tab) rTab = tab;
    dictPanel.hidden = !open;
    sideDict.hidden = rTab !== 'dict'; sideCode.hidden = rTab !== 'code';
    $$('.rtabs button', dictPanel).forEach(b => b.classList.toggle('on', b.dataset.rt === rTab));
    document.body.classList.toggle('dict-on', open);
    [['dictBtn', 'dict'], ['codeBtn', 'code']].forEach(([id, t]) => { const on = open && rTab === t; $('#' + id).classList.toggle('on', on); $('#' + id).setAttribute('aria-expanded', String(on)); });
    store.set('dictOn', open); store.set('rTab', rTab);
    if (!open) return;
    if (rTab === 'dict') { dictRender(); if (focus) setTimeout(() => $('.dict-q', sideDict).focus(), 0); }
    else codeRender();
  }
  const rpRefresh = () => { if (dictIsOpen()) (rTab === 'dict' ? dictRender() : codeRender()); };
  $('#dictClose').addEventListener('click', () => rpSet(false));
  $('.rtabs', dictPanel).addEventListener('click', e => { const b = e.target.closest('[data-rt]'); if (b) rpSet(true, b.dataset.rt, true); });

  /* 💻 코드 탭: 과목 → Day → 교시 → 예제 목차, 누르면 그 자리에서 코드 (패널이 넓어짐)
   *   목록 데이터 = data/index.js 의 code [과목, day, 교시, 토픽 id, 파일명, 하는 일, 강의 번호, 제목] */
  let CODE = { view: 'list', s: null, d: 0, n: 0, tid: 0, q: '' };
  const CODE_BY = {};
  (IDX.code || []).forEach(r => { const k = r[0] + '|' + r[1] + '|' + r[2]; (CODE_BY[k] = CODE_BY[k] || []).push(r); });
  const ctxCache = {};
  async function lessonCtx(s, d, n) {
    const k = s + '|' + d + '|' + n;
    if (CX && CX.s === s && CX.d === d && CX.n === n) return CX;
    if (ctxCache[k]) return ctxCache[k];
    const data = await loadSubject(s); const D = data.days.find(x => x.n === d); const L = D && D.lessons.find(x => x.n === n);
    if (!L) return null;
    return (ctxCache[k] = { s, d, n, L, D, LK: window.WikiLink.build(L, n) });
  }
  function openCode(s, d, n, tid) { CODE = Object.assign(CODE, { view: 'detail', s, d, n, tid: +tid }); rpSet(true, 'code'); sideCode.scrollTop = 0; }
  function codeItem(r, cur) {
    const name = (r[4] ? `📄 ${esc(r[4])}` : esc(r[7])) + (r[8] ? ' <span class="cfig" title="배선·보드 그림 있음">🔌</span>' : '') + (r[9] ? ' <span class="cfig" title="코드 읽기 가이드 있음">🧭</span>' : '');
    return `<button type="button" class="citem${cur ? ' cur' : ''}" data-code-open="${r[0]}|${r[1]}|${r[2]}|${r[3]}"><span class="cn${r[4] ? ' mono' : ''}">${name}</span>${r[6] ? `<span class="cnum">${esc(r[6])}</span>` : ''}<span class="cw">${esc(r[5])}</span></button>`;
  }
  function codeRender() {
    if (CODE.view === 'detail') return codeDetail();
    const cur = SIDE_CUR && SIDE_CUR.n ? SIDE_CUR : null;
    const q = CODE.q.trim().toLowerCase(), words = q.split(/\s+/).filter(Boolean);
    let h = `<div class="dict-top"><input class="dict-q code-q" type="search" placeholder="파일명·하는 일 걸러 보기" autocomplete="off" spellcheck="false" value="${esc(CODE.q)}"></div>`;
    if (words.length) {
      const hits = (IDX.code || []).filter(r => words.every(w => (r[4] + ' ' + r[5] + ' ' + r[7]).toLowerCase().includes(w)));
      h += `<div class="dict-h">${hits.length}개</div>` + hits.slice(0, 150).map(r => `<div class="cloc">${loc(r[0], r[1], r[2])}</div>` + codeItem(r)).join('');
    } else {
      if (cur) {
        const here = CODE_BY[cur.s + '|' + cur.d + '|' + cur.n] || [];
        h += `<div class="dict-h">이 교시 (${loc(cur.s, cur.d, cur.n)}) · ${here.length}</div>` + (here.map(r => codeItem(r)).join('') || '<div class="dict-empty">이 교시에는 예제 코드가 없습니다.</div>');
        h += '<div class="dict-h">전체 목차</div>';
      }
      const openK = store.get('codeOpen', {});
      SUBJ.forEach(S => {
        const sOpen = cur ? cur.s === S.id : openK[S.id];
        h += `<details class="ctree s" data-k="${S.id}" ${sOpen ? 'open' : ''}><summary>${esc(S.name)}</summary>`;
        S.days.forEach(D => {
          const dOpen = cur ? cur.s === S.id && cur.d === D.n : openK[S.id + D.n];
          h += `<details class="ctree d" data-k="${S.id}${D.n}" ${dOpen ? 'open' : ''}><summary>Day ${D.n} <small>${esc(D.title)}</small></summary>`;
          D.lessons.forEach(L => {
            const rows = CODE_BY[S.id + '|' + D.n + '|' + L.n] || []; if (!rows.length) return;
            const isCur = cur && cur.s === S.id && cur.d === D.n && cur.n === L.n;
            h += `<details class="ctree l" data-k="${S.id}${D.n}-${L.n}" ${isCur || openK[S.id + D.n + '-' + L.n] ? 'open' : ''}><summary>${L.n}. ${esc(L.title)} <small>${rows.length}</small></summary>${rows.map(r => codeItem(r, false)).join('')}</details>`;
          });
          h += '</details>';
        });
        h += '</details>';
      });
    }
    sideCode.innerHTML = h;
    const qi = $('.code-q', sideCode);
    let raf = 0;
    const sched = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; CODE.q = qi.value; const pos = qi.selectionStart; codeRender(); const n = $('.code-q', sideCode); n.focus(); n.setSelectionRange(pos, pos); }); };
    qi.addEventListener('input', e => { if (!e.isComposing) sched(); });
    qi.addEventListener('compositionend', sched);
    $$('details.ctree', sideCode).forEach(dt => dt.addEventListener('toggle', () => { const o = store.get('codeOpen', {}); o[dt.dataset.k] = dt.open; store.set('codeOpen', o); }));
  }
  // 예제의 배선·보드 그림 (빌드 때 정한 figRefs: 자기 그림 / 같은 교시 앞 토픽 / Day 공통)
  const figsIn = h => (String(h).match(/<figure>[\s\S]*?<\/figure>/g) || []);
  function figsHtml(t, ctx) {
    const items = [];
    (t.figRefs || []).forEach(ref => {
      if (ref.t === 'self') figsIn(t.html).forEach(f => items.push({ f, from: '' }));
      else if (ref.t === 'topic') { const p = ctx.L.topics[ref.id]; if (p) figsIn(p.html).forEach(f => items.push({ f, from: p.title })); }
      else if (ref.t === 'other') { const ol = ctx.D && ctx.D.lessons.find(x => x.n === ref.n); const p = ol && ol.topics[ref.id]; if (p) figsIn(p.html).forEach(f => items.push({ f, from: `${ref.n}교시 · ${p.title}` })); }
      else if (ref.t === 'intro') { const g = ctx.D && ctx.D.figs && ctx.D.figs[ref.k]; if (g) items.push({ f: `<figure>${g.html}</figure>`, from: `Day ${ctx.d} 공통` }); }
    });
    if (!items.length) return '';
    const label = f => ((f.match(/aria-label="([^"]+)"/) || [])[1] || '');
    return `<details class="figbox" open><summary>🔌 배선 · 보드 <small>${items.length > 1 ? items.length + '장 · ' : ''}그림을 누르면 크게</small></summary>${items.map(it =>
      `<div class="figitem">${(label(it.f) || it.from) ? `<div class="figcap">${esc(label(it.f) || it.from)}${it.from && label(it.f) ? ` <small>· ${esc(it.from)}</small>` : ''}</div>` : ''}<button type="button" class="figzoom" aria-label="그림 크게 보기">${it.f}</button></div>`).join('')}</details>`;
  }
  document.addEventListener('click', e => {                                   // 그림 크게 보기 / 닫기
    const z = e.target.closest('.figzoom');
    if (z) { const ov = document.createElement('div'); ov.className = 'figov'; ov.innerHTML = `<div class="figov-in">${z.innerHTML}</div><span class="figov-x">✕ 닫기 (Esc)</span>`; document.body.appendChild(ov); return; }
    const ov = e.target.closest('.figov'); if (ov) ov.remove();
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') { const ov = $('.figov'); if (ov) ov.remove(); } });
  /* 코드 강조: 핵심 개념 칩 → 아래 코드(.cb)에서 그 용어가 쓰인 곳에 배경색, 같은 칩을 다시 누르면 다음 위치 */
  let CODECTX = null, CODEHL = { key: '', marks: [], i: -1 };
  function codeHlClear() {
    $$('mark.chl', sideCode).forEach(m => m.replaceWith(document.createTextNode(m.textContent)));
    $$('.cb code', sideCode).forEach(c => c.normalize());
    $$('[data-hlc].on, [data-hl-word].on', sideCode).forEach(b => { b.classList.remove('on'); const n = b.querySelector('.hln'); if (n) n.remove(); });
    const x = $('.hlbar', sideCode); if (x) x.remove();
    CODEHL = { key: '', marks: [], i: -1 };
  }
  function codeHlGo(i) {
    const ms = CODEHL.marks; if (!ms.length) return;
    CODEHL.i = (i + ms.length) % ms.length;
    ms.forEach(m => m.classList.remove('cur'));
    const m = ms[CODEHL.i]; m.classList.add('cur');
    const box = sideCode.getBoundingClientRect(), r = m.getBoundingClientRect();
    sideCode.scrollTop += r.top - box.top - box.height / 2;                    // 창이 아니라 코드 패널 안에서만 스크롤
    const pos = $('.hlbar .hlpos', sideCode); if (pos) pos.textContent = `${CODEHL.i + 1}/${ms.length}`;
  }
  function codeHl(btn) {
    const key = btn.dataset.hlc || ('w:' + btn.dataset.hlWord);
    if (CODEHL.key === key && CODEHL.marks.length) return codeHlGo(CODEHL.i + 1);
    codeHlClear();
    let keys = [];
    if (btn.dataset.hlc && CODECTX) { const c = CODECTX.LK.byId[btn.dataset.hlc]; keys = c ? c.keys.slice() : []; }
    else if (btn.dataset.hlWord) keys = [btn.dataset.hlWord];
    keys = [...new Set(keys.filter(Boolean))].sort((a, b) => b.length - a.length);       // 긴 키 먼저 (sched_getaffinity > sched)
    if (!keys.length) return;
    const pats = keys.map(k => { const e = k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      return /^[\x20-\x7e]+$/.test(k) ? (/^\w/.test(k) ? '(?<![A-Za-z0-9_])' : '') + e + (/\w$/.test(k) ? '(?![A-Za-z0-9_])' : '') : e; });
    const re = new RegExp(pats.join('|'), 'gi');
    const marks = [];
    $$('.cb pre code', sideCode).forEach(code => {
      const tw = document.createTreeWalker(code, NodeFilter.SHOW_TEXT); const nodes = [];
      while (tw.nextNode()) nodes.push(tw.currentNode);
      nodes.forEach(nd => {
        const t = nd.nodeValue; re.lastIndex = 0; let m, p = 0; const frag = document.createDocumentFragment(); let hit = false;
        while ((m = re.exec(t))) {
          if (!m[0]) { re.lastIndex++; continue; }
          hit = true; frag.appendChild(document.createTextNode(t.slice(p, m.index)));
          const mk = document.createElement('mark'); mk.className = 'chl'; mk.textContent = m[0]; frag.appendChild(mk); marks.push(mk);
          p = m.index + m[0].length;
        }
        if (hit) { frag.appendChild(document.createTextNode(t.slice(p))); nd.parentNode.replaceChild(frag, nd); }
      });
    });
    CODEHL = { key, marks, i: -1 };
    btn.classList.add('on'); btn.insertAdjacentHTML('beforeend', `<span class="hln"> · ${marks.length}곳</span>`);
    const firstCb = $('.cb', sideCode);
    if (firstCb) firstCb.insertAdjacentHTML('beforebegin', `<div class="hlbar">${marks.length ? `<b>${esc(btn.firstChild.textContent)}</b> 코드에서 <span class="hlpos">0/${marks.length}</span>
      <button type="button" class="icon-btn sm" data-hlnav="-1" aria-label="이전 위치">‹</button><button type="button" class="icon-btn sm" data-hlnav="1" aria-label="다음 위치">›</button>` : `<b>${esc(btn.firstChild.textContent)}</b> 은(는) 이 코드에 직접 나오지 않습니다`}<button type="button" class="icon-btn sm" data-hlnav="0" aria-label="강조 지우기">✕</button></div>`);
    const nav = $('.code-nav', sideCode), bar = $('.hlbar', sideCode);       // 강조 막대는 고정 헤더 바로 아래
    if (nav && bar) bar.style.top = nav.offsetHeight + 'px';
    if (marks.length) codeHlGo(0);
  }
  document.addEventListener('click', e => {
    const b = e.target.closest('#sideCode [data-hlc], #sideCode [data-hl-word]'); if (b) { codeHl(b); return; }
    const n = e.target.closest('#sideCode [data-hlnav]'); if (!n) return;
    const v = +n.dataset.hlnav; if (v === 0) codeHlClear(); else codeHlGo(CODEHL.i + v);
  });

  /* 🧭 코드 읽기 가이드 (summary/.src/code_guide*.json → 빌드 때 줄 범위로)
   * 단계마다 그 줄들을 미리 색칠해 두고, 칠해진 곳을 누르면 블록 바로 아래에 작은 설명 팝업 — 코드와 설명을 한 화면에 */
  let CODEGUIDE = [], GPOP = -1;
  function guideHtml(t) {
    const g = t.guide || []; if (!g.length) return '';
    const multi = new Set(g.map(x => x.b)).size > 1;
    return `<details class="guide"><summary>🧭 코드 읽기 가이드 <small>${g.length}단계 · 코드에서 색칠된 줄을 누르면 설명</small></summary><ol>${g.map((x, i) =>
      `<li><button type="button" class="gstep" data-gi="${i}"><span class="gt">${esc(x.t)}</span><span class="gl">${multi && t.blocks[x.b].file ? esc(t.blocks[x.b].file) + ' ' : ''}${x.from + 1}${x.to > x.from ? '–' + (x.to + 1) : ''}줄</span></button></li>`).join('')}</ol></details>`;
  }
  const guideLines = (t, x) => `${t.blocks[x.b] && t.blocks[x.b].file && new Set(CODEGUIDE.map(y => y.b)).size > 1 ? esc(t.blocks[x.b].file) + ' ' : ''}${x.from + 1}${x.to > x.from ? '–' + (x.to + 1) : ''}줄`;
  // 그려진 코드 높이 ÷ 줄 수 = 줄 높이 (글꼴 계산과 화면이 어긋나지 않게)
  function preMetrics(pre) {
    const code = $('code', pre);
    const n = code.textContent.replace(/\n$/, '').split('\n').length;
    const cr = code.getBoundingClientRect(), pr = pre.getBoundingClientRect();
    return { lh: cr.height / n, top0: cr.top - pr.top + pre.scrollTop, n };
  }
  function guideDraw() {
    $$('.gband', sideCode).forEach(x => x.remove());
    CODEGUIDE.forEach((g, i) => {
      const pre = $(`.cb[data-bi="${g.b}"] pre`, sideCode); if (!pre) return;
      pre.classList.add('has-guide');
      const { lh, top0 } = preMetrics(pre);
      const band = document.createElement('div');
      band.className = `gband g${i % 2}${i === GPOP ? ' on' : ''}`; band.dataset.gi = i;
      band.style.top = (top0 + g.from * lh) + 'px';
      band.style.height = ((g.to - g.from + 1) * lh) + 'px';
      band.style.width = pre.scrollWidth + 'px';
      band.innerHTML = `<span class="gnum">${i + 1}</span>`;
      pre.appendChild(band);
    });
  }
  function guidePopClose() {
    GPOP = -1;
    $$('.gpop', sideCode).forEach(x => x.remove());
    $$('.gband.on, .gstep.on', sideCode).forEach(x => x.classList.remove('on'));
    $$('.cb.pop-on', sideCode).forEach(x => x.classList.remove('pop-on'));
  }
  function guidePop(i, scroll) {
    const g = CODEGUIDE[i]; if (!g) return guidePopClose();
    guidePopClose(); GPOP = i;
    const cb = $(`.cb[data-bi="${g.b}"]`, sideCode), pre = cb && $('pre', cb), band = pre && $(`.gband[data-gi="${i}"]`, pre);
    if (!band) return;
    band.classList.add('on');
    const st = $(`.gstep[data-gi="${i}"]`, sideCode); if (st) st.classList.add('on');
    const pop = document.createElement('div'); pop.className = 'gpop'; pop.setAttribute('role', 'dialog');
    pop.innerHTML = `<div class="gpop-h"><b>${i + 1}. ${esc(g.t)}</b><span class="gl">${guideLines(CODECTX && CODECTX.L.topics[CODE.tid] || { blocks: [] }, g)}</span><span class="sp"></span>
      <button type="button" class="icon-btn sm" data-gpop="-1" aria-label="이전 단계" ${i ? '' : 'disabled'}>‹</button><span class="gpos">${i + 1}/${CODEGUIDE.length}</span><button type="button" class="icon-btn sm" data-gpop="1" aria-label="다음 단계" ${i < CODEGUIDE.length - 1 ? '' : 'disabled'}>›</button><button type="button" class="icon-btn sm" data-gpop="0" aria-label="닫기">✕</button></div>${g.d ? `<div class="gpop-d">${mdInline(g.d)}</div>` : ''}`;
    cb.classList.add('pop-on');
    pop.style.top = (pre.offsetTop + band.offsetTop + band.offsetHeight + 6) + 'px';   // 그 단계 마지막 줄 바로 아래
    cb.appendChild(pop);
    if (scroll) {                                                            // 띠와 팝업이 패널 안에 함께 보이게
      const box = sideCode.getBoundingClientRect(), b = band.getBoundingClientRect(), p = pop.getBoundingClientRect();
      if (b.top < box.top + 60 || p.bottom > box.bottom) sideCode.scrollTop += Math.min(b.top - box.top - 70, p.bottom - box.bottom + 16 > 0 ? p.bottom - box.bottom + 16 : b.top - box.top - 70);
    }
  }
  document.addEventListener('click', e => {
    if (!sideCode.contains(e.target)) return;
    const nav = e.target.closest('[data-gpop]');
    if (nav) { const v = +nav.dataset.gpop; return v === 0 ? guidePopClose() : guidePop(GPOP + v, true); }
    const st = e.target.closest('.gstep'); if (st) return guidePop(+st.dataset.gi, true);
    if (e.target.closest('.gpop')) return;
    const pre = e.target.closest('.cb[data-bi] pre.has-guide');
    if (pre && !String(window.getSelection() || '')) {                    // 드래그로 코드를 고를 때는 열지 않음
      const bi = +pre.parentElement.dataset.bi, { lh, top0 } = preMetrics(pre);
      const line = Math.floor((e.clientY - pre.getBoundingClientRect().top + pre.scrollTop - top0) / lh);
      const i = CODEGUIDE.findIndex(g => g.b === bi && line >= g.from && line <= g.to);
      if (i >= 0) return i === GPOP ? guidePopClose() : guidePop(i, false);
    }
    if (GPOP >= 0 && !e.target.closest('[data-hlc], [data-hl-word], [data-hlnav]')) guidePopClose();
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && GPOP >= 0) guidePopClose(); });
  window.addEventListener('resize', () => { if (CODEGUIDE.length && !sideCode.hidden) { const k = GPOP; guideDraw(); if (k >= 0) guidePop(k); } });

  let codeSeq = 0;                                                             // 늦게 끝난 이전 요청이 화면을 덮어쓰지 않게
  async function codeDetail() {
    const my = ++codeSeq, want = { s: CODE.s, d: CODE.d, n: CODE.n, tid: CODE.tid };
    sideCode.innerHTML = '<div class="dict-empty">불러오는 중…</div>';
    const ctx = await lessonCtx(want.s, want.d, want.n);
    if (my !== codeSeq) return;
    CODEHL = { key: '', marks: [], i: -1 };
    const t = ctx && ctx.L.topics[CODE.tid];
    if (!t) { sideCode.innerHTML = '<div class="dict-empty">코드를 찾지 못했습니다.</div>'; return; }
    const rows = CODE_BY[CODE.s + '|' + CODE.d + '|' + CODE.n] || [];
    const k = rows.findIndex(r => r[3] === CODE.tid);
    const pv = rows[k - 1], nx = rows[k + 1];
    const x = exInfo(t, ctx);
    const same = CX && CX.s === CODE.s && CX.d === CODE.d && CX.n === CODE.n;
    // 칩: 누르면 (같은 교시면) 용어 설명 패널 + 아래 코드에서 쓰인 곳 강조
    const cs = x.concepts.length ? x.concepts.map(c => { const cc = ctx.LK.byId[c];
        return `<button type="button" class="cchip ${cc.kind}" ${same ? `data-peek-c="${c}"` : ''} data-hlc="${c}" title="코드에서 쓰인 곳 강조${same ? ' · 용어 설명' : ''}">${esc(shortName(cc))}</button>`; }).join('')
      : x.calls.map(f => `<button type="button" class="callchip" data-hl-word="${esc(f)}" title="코드에서 쓰인 곳 강조">${esc(f)}()</button>`).join('');
    CODECTX = ctx;
    sideCode.innerHTML = `<div class="code-nav"><button type="button" class="btn sm" data-code-back>← 목록</button><a href="${href(CODE.s, CODE.d, CODE.n)}">${loc(CODE.s, CODE.d, CODE.n)}</a><span class="sp"></span>
        <button type="button" class="icon-btn sm" ${pv ? `data-code-open="${pv[0]}|${pv[1]}|${pv[2]}|${pv[3]}"` : 'disabled'} aria-label="이전 예제">‹</button><span class="cpos">${k + 1}/${rows.length}</span><button type="button" class="icon-btn sm" ${nx ? `data-code-open="${nx[0]}|${nx[1]}|${nx[2]}|${nx[3]}"` : 'disabled'} aria-label="다음 예제">›</button></div>
      <div class="code-head"><span class="xname">${x.name ? '📄 ' + esc(x.name) : esc(x.bare)}${x.files.length > 1 ? `<small> 외 ${x.files.length - 1}개</small>` : ''}</span>${x.num ? `<span class="tag">강의 ${esc(x.num)}</span>` : ''}</div>
      <dl class="xinfo">
        ${x.files.length > 1 ? `<div><dt>파일</dt><dd>${x.files.map(f => `<code>${esc(f)}</code>`).join(' ')}</dd></div>` : ''}
        <div><dt>무엇을 보나</dt><dd>${mdInline(x.purpose)}</dd></div>
        ${cs ? `<div><dt>${x.concepts.length ? '핵심 개념' : '주요 호출'}</dt><dd class="rel">${cs}</dd></div>` : ''}
        <div><dt>강의</dt><dd>${esc(t.title)}</dd></div></dl>
      ${figsHtml(t, ctx)}
      ${guideHtml(t)}
      ${exampleBody(t)}`;
    CODEGUIDE = t.guide || []; GPOP = -1;
    highlightIn(sideCode);
    if (CODEGUIDE.length) requestAnimationFrame(guideDraw);              // 글꼴·강조가 그려진 뒤 줄 높이를 잰다
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
      <li><b>교시 페이지</b> = ① 정리 노트(개념별 묶어 보기 가능) ② 강의 원문(읽은 것 체크) ③ 연관 문제. 오른쪽 <b>📖 사전</b> · <b>💻 코드</b> 패널을 본문과 나란히 띄워 둘 수 있다.</li>
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
  function codeBlock(b, bi) {
    const lang = b.kind === 'cmd' ? 'bash' : b.lang;
    return `<div class="cb ${b.kind}"${bi !== undefined ? ` data-bi="${bi}"` : ''}><div class="cb-h"><span class="k ${b.kind}">${BLK_LABEL[b.kind]}</span>${b.file ? `<span class="cb-file">${esc(b.file)}</span>` : `<span>${esc(b.lang === 'text' ? '' : b.lang)}</span>`}<button type="button" data-copy>복사</button></div><pre><code data-lang="${esc(lang)}">${esc(b.text)}</code></pre></div>`;
  }
  function exampleBody(t) {
    const ix = t.blocks.map((b, i) => [b, i]);
    const codes = ix.filter(([b]) => b.kind === 'code');
    const run = ix.filter(([b]) => b.kind !== 'code');
    let h = codes.map(([b, i]) => codeBlock(b, i)).join('');
    if (run.length) h += `<div class="runbox"><div class="rh">빌드·실행 / 결과</div>${run.map(([b, i]) => codeBlock(b, i)).join('')}</div>`;
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
  /* 예제 하나의 이름표: 원본 파일명 · 무엇을 보는 코드인지 · 핵심 개념
   *   이름   강의 코드 첫 줄 주석/앞 본문/빌드 명령 (lecture_parse) → 노트 "예제 코드가 하는 일" 표의 파일명
   *   목적   노트 표의 "하는 일" → 강의 코드 첫 줄 주석의 설명 → 강의 제목
   *   개념   노트의 용어·함수 중 이 예제에 나오는 것 / 노트가 없으면 코드에서 부르는 주요 함수 */
  const CALL_SKIP = new Set(['if', 'for', 'while', 'switch', 'return', 'sizeof', 'main', 'printf', 'defined', 'setup', 'loop']);
  function codeCalls(t) {
    const code = t.blocks.filter(b => b.kind === 'code').map(b => b.text).join('\n');
    const defs = new Set([...code.matchAll(/^[\w \t*]*?\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{/gm)].map(m => m[1]));
    const cnt = {};
    for (const m of code.matchAll(/\b([A-Za-z_]\w{2,})\s*\(/g)) { const f = m[1]; if (!CALL_SKIP.has(f) && !defs.has(f)) cnt[f] = (cnt[f] || 0) + 1; }
    return Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a]).slice(0, 6);
  }
  function exInfo(t, ctx = CX) {
    const r = ctx && ctx.LK ? rowOf(t.id, ctx.LK) : null;
    const rowFile = r ? (window.WikiLink.stripMd(r.cells[0] || '').match(/[\w./-]+\.(?:c|h|py|sh|ino|cpp)\b/) || [])[0] : null;
    const files = t.files && t.files.length ? t.files : (rowFile ? [rowFile.split('/').pop()] : []);
    const num = (t.title.match(/^(\d+-\d+)\./) || t.title.match(/예제\s*(\d+-\d+)/) || [])[1] || '';
    const bare = t.title.replace(/^\d+-\d+\.\s*/, '').replace(/^예제\s*[\d-]+\s*[·—-]\s*/, '').trim();
    const purpose = r && r.cells[1] ? r.cells[1] : (t.desc || bare);
    const LKx = ctx && ctx.LK;
    let cs = LKx ? (LKx.topicConcepts[t.id] || []).filter(x => LKx.byId[x]) : [];
    cs = cs.sort((a, b) => (LKx.byId[b].examples.includes(t.id) ? 1 : 0) - (LKx.byId[a].examples.includes(t.id) ? 1 : 0)).slice(0, 6);
    return { files, name: files[0] || '', num, bare, purpose, concepts: cs, calls: cs.length ? [] : codeCalls(t) };
  }
  function exHead(t) {
    const x = exInfo(t);
    const name = x.name ? `<span class="xname">📄 ${esc(x.name)}${x.files.length > 1 ? `<small> 외 ${x.files.length - 1}개</small>` : ''}</span>` : `<span class="xname nofile">${esc(x.bare)}</span>`;
    return `${name}<span class="xmeta">${x.num ? `<span class="tag">강의 ${esc(x.num)}</span>` : ''}</span><span class="xpurpose">${mdInline(x.purpose)}</span>`;
  }
  function exInfoBox(t) {
    const x = exInfo(t);
    const cs = x.concepts.length ? x.concepts.map(c => cChip(c)).join('') : x.calls.map(f => `<a class="callchip" href="#/search?q=${encodeURIComponent(f)}">${esc(f)}()</a>`).join('');
    return `<dl class="xinfo">
      ${x.files.length > 1 ? `<div><dt>파일</dt><dd>${x.files.map(f => `<code>${esc(f)}</code>`).join(' ')}</dd></div>` : ''}
      <div><dt>무엇을 보나</dt><dd>${mdInline(x.purpose)}</dd></div>
      ${cs ? `<div><dt>${x.concepts.length ? '핵심 개념' : '주요 호출'}</dt><dd class="rel">${cs}</dd></div>` : ''}
      <div><dt>강의</dt><dd>${esc(t.title)}</dd></div></dl>`;
  }
  function exLabel(t) { const x = exInfo(t); if (x.name) return x.name; const m = t.title.match(/예제\s*([\d-]+)/); return m ? '예제 ' + m[1] : t.title.replace(/^\d+-\d+\.\s*/, '').slice(0, 16); }
  function exChip(tid) { const t = CX.L.topics[tid]; return t ? `<button type="button" class="xchip" data-peek-x="${tid}" title="${esc(t.title)}">▶ ${esc(exLabel(t))}</button>` : ''; }
  // 개념 블록 오른쪽 위의 "코드) 04_affinity.c" 버튼 → 오른쪽 💻 코드 패널에 그 예제
  function codeBtns(tids) {
    const ts = [...new Set(tids)].map(id => CX.L.topics[id]).filter(Boolean);
    return ts.length ? `<span class="cbtns">${ts.map(t => `<button type="button" class="cbtn" data-peek-x="${t.id}" title="${esc(t.title)} — 오른쪽 코드 패널에서 보기"><b>코드)</b> ${esc(exLabel(t))}</button>`).join('')}</span>` : '';
  }
  /* 함수·명령 상세 (summary/.src/api_detail.json): 역할 · 인자별 뜻 · 동작 · 반환 · 준비 */
  function detailHtml(det, noRole) {
    if (!det) return '';
    const args = (det.args || []).length ? `<table class="apid-args"><thead><tr><th>인자</th><th>뜻</th></tr></thead><tbody>${det.args.map(a => `<tr><td><code>${esc(a[0])}</code></td><td>${mdInline(a[1])}</td></tr>`).join('')}</tbody></table>` : '';
    const row = (k, v) => v ? `<div class="apid-row"><span class="k">${k}</span><span>${mdInline(v)}</span></div>` : '';
    return `<div class="apid">${noRole ? '' : row('역할', det.role ? `<b>${esc(det.role)}</b>` : '')}${args}${row('동작', det.flow)}${row('반환', det.ret)}${row('준비', det.hdr)}</div>`;
  }
  /* 함수·명령 한 줄: 이름 + 역할만 (기본형 · 인자 · 동작은 이름을 누르면 옆 패널에) */
  function fnRole(use, det) { return det && det.role ? `<span class="frole">${esc(det.role)}</span>` : `<span class="frole">${mdInline(use || '')}</span>`; }

  function countBtn(c) {
    const parts = [c.bullets.length && `핵심 ${c.bullets.length}`, c.usage.length && `사용형태 ${c.usage.length}`, c.examples.length && `예제 ${c.examples.length}`, c.quiz.length && `문항 ${c.quiz.length}`].filter(Boolean);
    return parts.length ? `<button type="button" class="cnt" data-peek-c="${c.id}">${parts.join(' · ')} ›</button>` : '';
  }
  /* 핵심 내용 글머리 한 개를 읽기 좋게: 머리말 강조 · ①② 줄바꿈 · " — " 뒤 설명은 ↳ 다음 줄
   *   "**핸들러 작성 규칙 3가지**: ① A ② B ③ C — 이유"  →  [머리말] / ① A / ② B / ③ C / ↳ 이유
   * 코드(`…`) 안의 ①·—·: 는 건드리지 않는다. 둘째 줄부터(하위 글머리 등)는 그대로 markdown. */
  const CIRC_RE = /[①②③④⑤⑥⑦⑧⑨⑩]/;
  function bulletBody(src) {
    const nl = src.indexOf('\n');
    let first = nl < 0 ? src : src.slice(0, nl); const tail = nl < 0 ? '' : src.slice(nl + 1);
    const codes = [];
    first = first.replace(/`[^`]*`/g, m => `\u0001${codes.push(m) - 1}\u0001`);
    const back = t => t.replace(/\u0001(\d+)\u0001/g, (_, i) => codes[+i]);
    let lead = '';
    // 굵은 머리말: "**제목**: …" 은 항상, "**제목** …" 은 뒤가 조사(를·이·은…)로 이어지는 문장이 아닐 때만
    const kv = first.match(/^\*\*([^*]+?)\*\*\s*→\s*([\s\S]*)$/);     // "**실패하면** → NULL …" : 키워드 → 결과 (한 줄)
    if (kv) {
      const rest = kv[2].split(/\s+—\s+/);
      return `<div class="b-body kv"><div class="b-line"><span class="b-key">${mdInline(back(kv[1]))}</span><span class="b-arr">→</span>${mdInline(back(rest[0]))}</div>` +
        rest.slice(1).map(x => `<div class="b-line sub"><span class="arr">↳</span> ${mdInline(back(x))}</div>`).join('') + '</div>' + (tail.trim() ? md(tail) : '');
    }
    let m = first.match(/^\*\*([^*]+?)\*\*\s*[:：]\s*/) || first.match(/^\*\*([^*]+?)\*\*\s+(?!(?:을|를|이|가|은|는|의|에|에서|로|으로|와|과|도|만|이다|다|라|하면|하고|해서)(?=[\s.,)]|$))/);
    if (m && !first.slice(m[0].length).trim()) m = null;              // 굵은 글씨만 있는 줄은 그대로
    if (m) { lead = m[1]; first = first.slice(m[0].length); }
    else if ((m = first.match(/^([^\s:*\u0001][^:\u0001]{0,22}?)\s*:\s+/)) && !/[.。]/.test(m[1])) { lead = m[1]; first = first.slice(m[0].length); }
    // ①② 가 두 개 이상이면 번호마다 줄 바꿈, " — " 뒤는 ↳ 줄
    let parts = [first];
    // 괄호 밖, 줄 처음이나 띄어쓰기 뒤의 ①② 만 항목 번호로 본다 ("(…①이 유지되도록)" 같은 참조는 그대로)
    const cuts = []; let depth = 0;
    for (let i = 0; i < first.length; i++) {
      const ch = first[i];
      if (ch === '(' || ch === '（') depth++; else if ((ch === ')' || ch === '）') && depth) depth--;
      else if (!depth && CIRC_RE.test(ch) && (i === 0 || /\s/.test(first[i - 1]))) cuts.push(i);
    }
    if (cuts.length >= 2) parts = [first.slice(0, cuts[0]), ...cuts.map((c, k) => first.slice(c, cuts[k + 1]))].filter(x => x.trim());
    const lines = [];
    parts.forEach(pt => {
      const seg = pt.split(/\s+—\s+/);
      lines.push({ t: seg[0], cls: CIRC_RE.test(pt.trim()[0]) ? 'num' : '' });
      seg.slice(1).forEach(x => lines.push({ t: x, cls: 'sub' }));
    });
    let h = lead ? `<div class="b-lead">${mdInline(back(lead))}</div>` : '';
    h += lines.filter(l => l.t.trim()).map(l => `<div class="b-line ${l.cls}">${l.cls === 'sub' ? '<span class="arr">↳</span> ' : ''}${mdInline(back(l.t.trim()))}</div>`).join('');
    return `<div class="b-body${lead ? ' has-lead' : ''}">${h}</div>` + (tail.trim() ? md(tail) : '');
  }
  function bulletLi(b, skip, bare, noEx) {
    if (bare) return `<li id="${b.id}">${bulletBody(b.md)}</li>`;            // 레퍼런스 카드: 칩 없이 내용만
    const cs = b.concepts.slice().sort((x, y) => (x === b.primary ? -1 : y === b.primary ? 1 : 0)).filter(x => x !== skip).slice(0, 4);
    const rel = cs.map(x => cChip(x)).join('') + (noEx ? '' : b.examples.map(exChip).join(''));   // 카드 안에선 코드 버튼이 머리에 있으니 생략
    return `<li id="${b.id}">${bulletBody(b.md)}${rel ? `<div class="rel">${rel}</div>` : ''}</li>`;
  }
  const moreBox = mdText => `<details class="cg-more"><summary>자세히 · 왜 그런지</summary><div class="in">${md(mdText)}</div></details>`;
  function coreHtml(sec) {
    let h = '', open = false;
    sec.blocks.forEach(b => {
      if (b.bullet) { if (!open) { h += '<ul class="core-list">'; open = true; } h += bulletLi(b); return; }
      if (open) { h += '</ul>'; open = false; }
      if (b.head !== undefined) h += `<h4 class="cg-h" id="g${sec.si}-${b.g}">${mdInline(b.head)}</h4>`;
      else if (b.sum !== undefined) h += `<p class="cg-sum">${mdInline(b.sum)}</p>`;
      else if (b.more !== undefined) h += moreBox(b.more);
      else h += md(b.md);
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
      body = '<table class="ftab"><thead><tr><th>이름</th><th>역할</th></tr></thead><tbody>' +
        L.funcs.slice(s.f0, s.f0 + s.fn).map((f, i) => {
          const id = 'f' + (s.f0 + i);
          return `<tr id="${id}"><td class="fcell"><button type="button" class="fname" data-peek-c="${cid(id)}">${esc(f.name)}</button></td><td>${fnRole(f.use, f.detail)}</td></tr>`;
        }).join('') + '</tbody></table>' + md(s.rest);
    } else if (s.key === 'core' && LS.blocks) body = coreHtml(LS);
    else if (s.key === 'examples' && LS.rows) body = examplesTable(LS);
    else body = md(s.md);
    const id = s.key === 'kw' ? 's-kw' : 's' + si;
    return `<section class="${cls}" id="${id}">${s.label ? `<h3>${mdInline(s.label)}</h3>` : ''}${body}</section>`;
  }

  // 개념별 묶어 보기: 핵심 용어·새 함수·핵심 내용을 개념 카드로 재배치 (카드 = 정의 + 그 개념의 글머리 + 사용형태 + 예제 코드)
  // 큰 개념 카드 (노트에 "#### 큰 개념" 이 있는 교시)
  // 모양 고정: 제목 → 한 줄 정의 → 핵심 → 관련 함수·용어 → 사용형태 → ⚠ 함정 → 예제 칩 → 자세히(접힘)
  function groupCards(groups) {
    const { L, LK } = CX;
    // 용어·함수는 처음 언급된 큰 개념 한 곳에만 붙인다 (카드마다 중복되지 않게)
    // 가장 많이 언급된 큰 개념 한 곳 (같으면 먼저 나온 쪽)
    const cnt = {};
    groups.forEach((g, gi) => g.bullets.forEach(b => b.concepts.forEach(c => { (cnt[c] = cnt[c] || {})[gi] = (cnt[c][gi] || 0) + 1; })));
    const home = {};
    Object.entries(cnt).forEach(([c, m]) => {
      const keys = (LK.byId[c] ? LK.byId[c].keys : []).map(k => k.toLowerCase());
      // 제목에 이름이 있는 카드가 우선, 그중(없으면 전체 중) 가장 많이 언급된 카드
      const titled = groups.map((g, gi) => gi).filter(gi => keys.some(k => groups[gi].title.replace(/`/g, '').toLowerCase().includes(k)));
      const pool = titled.length ? titled : Object.keys(m).map(Number);
      home[c] = pool.sort((a, b) => (m[b] || 0) - (m[a] || 0) || a - b)[0];
    });
    // 글머리에 언급이 없으면 카드의 제목 · 정의 · 표 · 자세히에서 이름을 찾아 배정
    // 노트가 줄여 쓴 이름도 찾는다: pthread_cond_wait → cond_wait, sudo systemctl enable --now x → systemctl enable
    const gtext = groups.map(g => [g.title, g.sum, ...(g.plain || []), g.more, ...g.bullets.map(b => b.md)].join('\n').replace(/`|\*\*/g, '').toLowerCase());
    const shortKeys = c => {
      const out = new Set((c.keys || []).map(k => k.toLowerCase()).filter(k => k.length >= 2));
      String(c.name || '').toLowerCase().split(/\s*\/\s*|\s*·\s*/).forEach(part => {
        const w = part.replace(/^sudo\s+/, '').trim(); if (!w) return;
        const id = (w.match(/^[a-z_][\w.]*/) || [''])[0];
        if (id.length >= 3) out.add(id);
        const seg = id.split('_'); if (seg.length >= 3) out.add(seg.slice(1).join('_'));   // 접두어(pthread_·gpiod_ …) 뺀 것
        const words = w.split(/\s+/).filter(x => !/^-/.test(x)); if (words.length >= 2) out.add(words.slice(0, 2).join(' '));
      });
      return [...out].filter(k => k.length >= 3);
    };
    LK.concepts.forEach(c => {
      if (home[c.id] !== undefined || !(c.kind === 'func' || c.kind === 'term' || c.sig)) return;
      const keys = shortKeys(c);
      const scores = gtext.map(t => keys.reduce((n, k) => n + (t.includes(k) ? k.length : 0), 0));
      const best = Math.max(...scores);
      if (best > 0) home[c.id] = scores.indexOf(best);
    });
    const fnList = fs => fs.length ? `<div class="gsec"><div class="gsec-h">함수·명령</div>${fs.map(c => `<div class="gfn"><button type="button" class="fname" data-peek-c="${c.id}">${esc(c.name)}</button><div>${fnRole(c.def, c.detail)}</div></div>`).join('')}</div>` : '';
    const termList = ts => ts.length ? `<div class="gsec"><div class="gsec-h">용어</div><dl class="gterms">${ts.map(c => `<div><dt><button type="button" class="cchip" data-peek-c="${c.id}">${esc(shortName(c))}</button></dt><dd>${mdInline(c.def)}</dd></div>`).join('')}</dl></div>` : '';
    const isFn = c => c.sig || c.kind === 'func';
    let h = groups.map((g, gi) => {
      const cs = LK.concepts.filter(c => home[c.id] === gi);
      const normal = g.bullets.filter(b => !/^⚠/.test(b.md.trim())), traps = g.bullets.filter(b => /^⚠/.test(b.md.trim()));
      const funcs = cs.filter(isFn), terms = cs.filter(c => !isFn(c) && c.kind === 'term');
      const usage = [...new Set(cs.flatMap(c => c.usage.map(u => u.line)))];
      const exs = [...new Set([...g.bullets.flatMap(b => b.examples), ...cs.flatMap(c => c.examples)])];
      return `<section class="ccard gcard" id="gc${gi}"><div class="ccard-h"><h3>${mdInline(g.title)}</h3>${codeBtns(exs)}</div>
        ${g.sum ? `<p class="cg-sum">${mdInline(g.sum)}</p>` : ''}
        ${(g.plain || []).map(m => `<div class="gplain">${md(m)}</div>`).join('')}
        ${normal.length ? `<ul class="core-list">${normal.map(b => bulletLi(b, null, true)).join('')}</ul>` : ''}
        ${fnList(funcs)}
        ${termList(terms)}
        ${usage.length ? `<div class="gsec"><div class="gsec-h">사용형태</div><pre><code data-lang="${esc((cs.find(c => c.usage.length) || { usage: [{ lang: 'c' }] }).usage[0].lang)}">${esc(usage.join('\n'))}</code></pre></div>` : ''}
        ${traps.length ? `<div class="gsec trap"><div class="gsec-h">⚠ 함정</div><ul class="core-list">${traps.map(b => bulletLi(Object.assign({}, b, { md: b.md.replace(/^\s*⚠\s*/, '') }), null, true)).join('')}</ul></div>` : ''}
        ${g.more ? moreBox(g.more) : ''}</section>`;
    }).join('');
    const rest = LK.concepts.filter(c => home[c.id] === undefined && (c.kind === 'term' || isFn(c)));
    if (rest.length) h += `<section class="ccard gcard loose"><div class="ccard-h"><h3>그 밖의 함수·명령 · 용어</h3></div>${fnList(rest.filter(isFn))}${termList(rest.filter(c => !isFn(c)))}</section>`;
    return h;
  }
  function conceptCards() {
    const { L, LK } = CX;
    const gs = LK.sections.filter(x => x.groups && x.groups.length).flatMap(x => x.groups);
    if (gs.length) return groupCards(gs);
    const bullets = LK.sections.filter(x => x.blocks).flatMap(x => x.blocks.filter(b => b.bullet));
    const own = {}; const loose = [];
    bullets.forEach((b, i) => { if (b.primary) (own[b.primary] = own[b.primary] || []).push(Object.assign({ i }, b)); else loose.push(b); });
    const first = c => (own[c.id] ? own[c.id][0].i : 1e3 + LK.concepts.indexOf(c));
    const list = LK.concepts.filter(c => c.kind === 'term' || own[c.id] || c.examples.length).sort((a, b) => first(a) - first(b));
    let h = list.map(c => {
      const bs = own[c.id] || [];
      const usage = c.usage.length ? `<pre><code data-lang="${esc(c.usage[0].lang)}">${esc(c.usage.map(u => u.line).join('\n'))}</code></pre>` : '';
      const more = [...c.bullets.filter(id => !bs.some(b => b.id === id)).map(id => `<button type="button" class="cnt" data-go="${id}">다른 글머리에서도 언급 ↗</button>`).slice(0, 1),
        c.quiz.length ? `<button type="button" class="cnt" data-peek-c="${c.id}">문항 ${c.quiz.length} ›</button>` : ''].join('');
      return `<section class="ccard" id="${c.id}"><div class="ccard-h"><h3>${esc(c.name)} <span class="tag">${c.kind === 'term' ? '용어' : '함수·명령'}</span></h3>${codeBtns(c.examples)}</div>
        <div class="cdef">${mdInline(c.def || '')}${c.sig ? `<div class="sig">${mdInline(c.sig)}</div>` : ''}</div>
        ${bs.length ? `<ul class="core-list">${bs.map(b => bulletLi(b, c.id, false, true)).join('')}</ul>` : ''}${usage}${more ? `<div class="rel">${more}</div>` : ''}</section>`;
    }).join('');
    if (loose.length) h += `<section class="ccard loose"><h3>그 밖의 핵심 내용</h3><ul class="core-list">${loose.map(b => bulletLi(b)).join('')}</ul></section>`;
    return h;
  }
  function notesHtml(view) {
    const { L } = CX;
    if (!L.sections.length) return '';
    if (view !== 'concept') return L.sections.map((x, i) => x.key === 'kw' ? '' : noteSection(x, i, L)).join('');
    let h = '', placed = false;
    L.sections.forEach((x, i) => {
      if (['terms', 'funcs', 'core'].includes(x.key)) { if (!placed) { const grouped = CX.LK.sections.some(y => y.groups && y.groups.length); h += `<div class="cards-h">개념별 정리 <small>${grouped ? '큰 개념마다 정의 · 핵심 · 함수 · 사용형태 · 함정을 한 카드에' : '정의 · 그 개념의 핵심 내용 · 사용형태 · 예제 코드를 한 카드에'}</small></div>` + (grouped ? `<nav class="gtoc">${CX.LK.sections.filter(y => y.groups).flatMap(y => y.groups).map((g, gi) => `<button type="button" data-go="gc${gi}">${mdInline(g.title)}</button>`).join('')}</nav>` : '') + conceptCards(); placed = true; } return; }
      if (x.key !== 'kw') h += noteSection(x, i, L);
    });
    return h;
  }

  /* 옆 패널 */
  const peekEl = $('#peek'); let peekStack = [];
  function openPeek(kind, id) { peekStack.push([kind, id]); renderPeek(); }
  function closePeek() { peekStack = []; peekEl.hidden = true; document.body.classList.remove('peek-on'); }
  function findBullet(id) { for (const s of CX.LK.sections) if (s.blocks) { const b = s.blocks.find(x => x.id === id); if (b) return b; } return null; }
  function rowOf(tid, LK = CX.LK) { for (const s of LK.sections) if (s.rows) { const r = s.rows.find(x => x.topic === tid); if (r) return r; } return null; }
  const pkSec = (t, body) => `<section class="pk-sec"><h4>${t}</h4>${body}</section>`;
  function renderPeek() {
    if (!peekStack.length || !CX) return closePeek();
    const [kind, id] = peekStack[peekStack.length - 1]; const { L, LK } = CX;
    let title = '', h = '';
    if (kind === 'c') {
      const c = LK.byId[id]; if (!c) return closePeek();
      title = `<span class="tag">${c.kind === 'term' ? '용어' : '함수·명령'}</span> ${esc(c.name)}`;
      h += `<div class="pk-def">${mdInline(c.def || '')}${c.sig ? `<div class="sig">${mdInline(c.sig)}</div>` : ''}</div>${detailHtml(c.detail)}`;
      if (c.bullets.length) h += pkSec(`핵심 내용 ${c.bullets.length}`, c.bullets.map(bid => { const b = findBullet(bid); return b ? `<div class="pk-item">${bulletBody(b.md)}<button type="button" class="go" data-go="${bid}">본문 ↗</button></div>` : ''; }).join(''));
      if (c.usage.length) h += pkSec('기본 사용형태', `<pre><code data-lang="${esc(c.usage[0].lang)}">${esc(c.usage.map(u => u.line).join('\n'))}</code></pre>`);
      if (c.examples.length) h += pkSec(`예제 코드 ${c.examples.length}`, `<div class="cbtn-col">${codeBtns(c.examples)}</div>`);
      const co = {}; c.bullets.forEach(bid => { const b = findBullet(bid); b && b.concepts.forEach(x => { if (x !== id) co[x] = (co[x] || 0) + 1; }); });
      const coIds = Object.keys(co).sort((a, b) => co[b] - co[a]).slice(0, 8);
      if (coIds.length) h += pkSec('함께 나오는 개념', `<div class="rel">${coIds.map(x => cChip(x)).join('')}</div>`);
      if (c.lecture.length) h += pkSec('강의에서 언급', `<div class="rel">${c.lecture.map(t => `<button type="button" class="cnt" data-go="L${t}">${esc(L.topics[t].title)} ↗</button>`).join('')}</div>`);
      if (c.quiz.length) h += pkSec(`연관 문제 ${c.quiz.length}`, `<div class="rel">${c.quiz.map(q => `<button type="button" class="cnt" data-go="q${q}">Q${q + 1} ↗</button>`).join('')}</div>`);
      h += `<p class="pk-foot"><a href="#/search?q=${encodeURIComponent(shortName(c))}">다른 교시에서 찾기 →</a></p>`;
    } else {
      const t = L.topics[id]; if (!t) return closePeek();
      const xi = exInfo(t);
      title = xi.name ? `📄 ${esc(xi.name)}` : esc(xi.bare);
      h += exInfoBox(t);
      const bs = LK.sections.filter(x => x.blocks).flatMap(x => x.blocks.filter(b => b.bullet && b.examples.includes(+id)));
      if (bs.length) h += pkSec('이어지는 핵심 내용', bs.map(b => `<div class="pk-item">${bulletBody(b.md)}<button type="button" class="go" data-go="${b.id}">본문 ↗</button></div>`).join(''));
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
    const cb = e.target.closest('[data-code-open],[data-code-back]');
    if (cb) {
      if (cb.dataset.codeBack !== undefined) { CODE.view = 'list'; rpSet(true, 'code'); return; }
      const [cs_, cd, cn, ct] = cb.dataset.codeOpen.split('|'); openCode(cs_, +cd, +cn, +ct); return;
    }
    const b = e.target.closest('[data-peek-c],[data-peek-x],[data-go],[data-pk-back],[data-pk-close],[data-view]'); if (!b || !CX) return;
    if (b.dataset.peekC) { e.preventDefault(); openPeek('c', b.dataset.peekC); }
    else if (b.dataset.peekX !== undefined) { e.preventDefault(); if (narrowDict()) closePeek(); openCode(CX.s, CX.d, CX.n, +b.dataset.peekX); }
    else if (b.dataset.go) revealInMain(b.dataset.go);
    else if (b.dataset.pkBack !== undefined) { peekStack.pop(); renderPeek(); }
    else if (b.dataset.pkClose !== undefined) closePeek();
    else if (b.dataset.view) renderNotesBody(b.dataset.view);                      // 이 페이지에서만 바꾼다 (기억하지 않음)
  });
  document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if (!peekEl.hidden) closePeek();
    else if (!dictPanel.hidden && narrowDict()) rpSet(false);
  });
  const narrowDict = () => matchMedia('(max-width: 900px)').matches;
  /* 키워드 → 이 페이지 안에서 언급된 곳 (노트 · 강의 원문 · 문항) */
  let KW = { hits: [], i: -1, k: -1 };
  const KW_STOP = new Set(['검사', '사용', '함정', '관용구', '규칙', '방법', '차이', '구분', '초기화', '반환', '주소', '파일', '함수', '변수', '값', '두', '배', '수', '읽기', '쓰기']);
  function kwClear() {
    $$('mark.kwm', main).forEach(m => m.replaceWith(document.createTextNode(m.textContent)));
    main.normalize();
    KW = { hits: [], i: -1, k: -1 };
    $$('.kwc.on', main).forEach(b => b.classList.remove('on'));
    const nav = $('.kwnav', main); if (nav) nav.hidden = true;
  }
  function kwRoots() {
    return [...$$('#notes-body > section, #notes-body > .ccard', main), ...$$('#lecture details, #quiz .q', main)];
  }
  function kwMark(term) {
    const lo = term.toLowerCase(); const out = [];
    const ascii = /^[\x20-\x7e]+$/.test(term) && /^\w/.test(term) && /\w$/.test(term);
    const re = ascii ? new RegExp('(^|[^A-Za-z0-9_])(' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')(?![A-Za-z0-9_])', 'gi') : null;
    kwRoots().forEach(root => {
      const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode: n => n.parentElement.closest('summary .chk, button.cnt, .rel, script, style') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT });
      const nodes = []; while (tw.nextNode()) nodes.push(tw.currentNode);
      nodes.forEach(nd => {
        const t = nd.nodeValue; const spans = [];
        if (re) { re.lastIndex = 0; let m; while ((m = re.exec(t))) spans.push([m.index + m[1].length, m[2].length]); }
        else { let j = 0, low = t.toLowerCase(); while ((j = low.indexOf(lo, j)) >= 0) { spans.push([j, lo.length]); j += lo.length; } }
        if (!spans.length) return;
        const frag = document.createDocumentFragment(); let p = 0;
        spans.forEach(([a, l]) => {
          frag.appendChild(document.createTextNode(t.slice(p, a)));
          const mk = document.createElement('mark'); mk.className = 'kwm'; mk.textContent = t.slice(a, a + l); frag.appendChild(mk); out.push(mk);
          p = a + l;
        });
        frag.appendChild(document.createTextNode(t.slice(p)));
        nd.parentNode.replaceChild(frag, nd);
      });
    });
    return out;
  }
  function kwCandidates(k) {
    const c = [k.trim()];
    // "죽은 스택 주소 반환", "모드 r w a r+", "int main(void)" 처럼 문장에 그대로 없을 때 쓸 조각들 (긴 것부터)
    const toks = k.split(/[\s,/·()（）=]+/).map(x => x.trim()).filter(x => x.length >= 2 && !KW_STOP.has(x));
    toks.sort((a, b) => b.length - a.length).forEach(x => { if (!c.includes(x)) c.push(x); });
    return c;
  }
  function kwGo(i) {
    const hits = KW.hits; if (!hits.length) return;
    KW.i = (i + hits.length) % hits.length;
    hits.forEach(h => h.classList.remove('cur'));
    const m = hits[KW.i]; m.classList.add('cur');
    let p = m; while (p) { if (p.tagName === 'DETAILS') p.open = true; p = p.parentElement; }
    m.scrollIntoView({ block: 'center', behavior: 'smooth' });
    $('.kwpos', main).textContent = `${KW.i + 1}/${hits.length}`;
  }
  function kwFind(ki) {
    const k = CX.L.keywords[ki];
    if (KW.k === ki && KW.hits.length) return kwGo(KW.i + 1);     // 같은 칩을 다시 누르면 다음 위치
    kwClear(); KW.k = ki;
    $(`.kwc[data-kw="${ki}"]`, main).classList.add('on');
    let used = '';
    for (const c of kwCandidates(k)) { const h = kwMark(c); if (h.length) { KW.hits = h; used = c; break; } }
    const nav = $('.kwnav', main); nav.hidden = false;
    if (!KW.hits.length) { $('.kwpos', main).innerHTML = `이 페이지엔 없음 · <a href="#/search?q=${encodeURIComponent(k)}">전체 검색</a>`; return; }
    nav.title = used !== k ? `“${k}” 그대로는 없어서 “${used}” 로 찾음` : '';
    kwGo(0);
  }
  document.addEventListener('click', e => {
    if (!CX) return;
    const c = e.target.closest('.kwc'); if (c) return kwFind(+c.dataset.kw);
    const n = e.target.closest('[data-kwnav]'); if (!n) return;
    const v = +n.dataset.kwnav; if (v === 0) kwClear(); else kwGo(KW.i + v);
  });

  function renderNotesBody(view) {
    const box = $('#notes-body', main); if (!box) return;
    kwClear();
    box.innerHTML = notesHtml(view);
    $$('[data-view]', main).forEach(x => x.classList.toggle('on', x.dataset.view === view));
    const vn = $('#viewName', main); if (vn) vn.textContent = view === 'concept' ? '개념별 묶어 보기' : '노트 순서';
    const nt = $('#notes', main); if (nt && nt.getBoundingClientRect().top < 0) nt.scrollIntoView();
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
    const my = ROUTE_N;
    let data;
    try { data = await loadSubject(s); } catch (e) { if (my === ROUTE_N) main.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
    if (my !== ROUTE_N) return;
    const D = data.days.find(x => x.n === d); const L = D && D.lessons.find(x => x.n === n);
    if (!L) return notFound();
    const exTopics = L.topics.filter(t => t.blocks.some(b => b.kind !== 'out'));
    const readKey = `${s}|${d}|${n}`; const readSet = new Set(store.get('read', {})[readKey] || []);
    CX = { s, d, n, L, D, LK: window.WikiLink.build(L, n) }; peekStack = []; closePeek(); KW = { hits: [], i: -1, k: -1 };
    const anchor = params.get('a');
    // 기본은 개념별 묶어 보기. 검색에서 노트 섹션(s3 등)으로 오면 그 섹션이 있는 노트 순서로
    const view = params.get('v') || (anchor && /^s\d/.test(anchor) ? 'flow' : 'concept');
    let h = `<article class="lesson sc-${s}"><div class="l-head"><div class="crumb"><a href="#/">홈</a> › <a href="#/${s}">${esc(S.name)}</a> › Day ${d} ${esc(D.title)}</div>
      <div class="l-title"><h1>${n}교시 · ${esc(L.title)}</h1>${L.anim ? `<a class="btn anim-btn" href="${esc(L.anim)}" target="_blank" rel="noopener" title="이 교시의 강의 애니메이션을 새 탭에서 엽니다">🎬 애니메이션 보기</a>` : ''}</div><div class="tags"><span class="tag s">${esc(S.short)} Day ${d}</span>
      <span class="tag ${L.status}">${L.status === 'done' ? '정리됨' : '정리 전'}</span>
      ${L.lectureTitle && L.lectureTitle !== L.title ? `<span class="tag">강의: ${esc(L.lectureTitle)}</span>` : ''}</div>
      ${L.goals.length ? `<ul class="goals"><b>학습 목표 (강의)</b>${L.goals.map(g => `<li>${esc(g)}</li>`).join('')}</ul>` : ''}</div>
      <nav class="jump"><a href="#notes" data-jump>정리 노트</a><button type="button" class="jbtn" data-code-list>💻 예제 코드 ${exTopics.length}</button><a href="#lecture" data-jump>강의 원문 ${L.topics.length}</a><a href="#quiz" data-jump>연관 문제 ${L.quiz.length}</a>${L.sections.length ? `<span class="view-tg" role="group" aria-label="정리 노트 보기 방식"><button type="button" data-view="flow" class="${view === 'flow' ? 'on' : ''}">노트 순서</button><button type="button" data-view="concept" class="${view === 'concept' ? 'on' : ''}">개념별 묶어 보기</button></span>` : ''}${L.keywords.length ? `<div class="kwbar" id="s-kw"><span class="kwbar-l">키워드</span><div class="kwbar-chips">${L.keywords.map((k, i) => `<button type="button" class="kwc" data-kw="${i}">${esc(k)}</button>`).join('')}</div><span class="kwnav" hidden><button type="button" class="icon-btn sm" data-kwnav="-1" aria-label="이전 위치">‹</button><span class="kwpos"></span><button type="button" class="icon-btn sm" data-kwnav="1" aria-label="다음 위치">›</button><button type="button" class="icon-btn sm" data-kwnav="0" aria-label="강조 지우기">✕</button></span></div>` : ''}</nav>`;

    h += `<section class="part" id="notes"><h2>정리 노트 <small id="viewName">${view === 'concept' ? '개념별 묶어 보기' : '노트 순서'}</small></h2>`;
    if (L.sections.length) h += `<p class="part-note">점선 칩(개념)·▶ 칩(예제)을 누르면 관련 정의·핵심 내용·코드·문항이 옆에 모입니다.</p><div id="notes-body">${notesHtml(view)}</div>`;
    else h += `<div class="empty">아직 정리 전인 교시입니다. 공부하면서 <code>summary/.src/${{ c: 'C', linux: 'LINUX', mcu: 'MCU', eio: 'EIO' }[s]}_day${d}.md</code> 에 <b>키워드 · 개요 · 핵심 용어 · 새로 나온 함수 · 예제 코드가 하는 일 · 핵심 내용 · 기본 사용형태 · 연결</b> 순서로 채우면 여기에 나타납니다. 그 전까지는 아래 강의 개념 목록과 예제 코드로 먼저 훑어보세요.</div>`;
    h += '</section>';

    h += `<section class="part" id="lecture"><h2>강의 원문 <small>강의 자료를 주제별로. 읽고 정리했으면 체크 · 예제 코드는 오른쪽 💻 코드 패널</small></h2>
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
    const pk = (params.get('p') || '').match(/^([cx]):(\w+)$/);   // #/c/4/1?p=c:t3 → 개념 패널, ?p=x:2 → 코드 패널
    if (pk) { if (pk[1] === 'x') openCode(s, d, n, +pk[2]); else openPeek('c', pk[2]); }
    const xa = (anchor || '').match(/^X(\d+)$/);                     // 검색의 코드 결과 → 코드 패널로
    if (xa) { openCode(s, d, n, +xa[1]); window.scrollTo(0, 0); return; }
    const kq = params.get('k');                                      // #/c/2/7?k=realloc → 그 키워드 위치로
    if (kq) { const ki = L.keywords.findIndex(x => x === kq); if (ki >= 0) { kwFind(ki); return; } }
    jumpTo(anchor && CX.LK.alias[anchor] && !document.getElementById(anchor) ? CX.LK.alias[anchor] : anchor, params.get('q'));
  }

  function jumpTo(anchor, q) {
    if (!anchor) { window.scrollTo(0, 0); return; }
    let el = document.getElementById(anchor);
    if (!el && /^[tf]\d+$/.test(anchor)) {
      // 개념별 보기에는 용어·함수 표가 따로 없다 → 그 용어·함수가 들어 있는 카드의 줄로, 카드에 없으면 노트 순서 보기로
      const id = (CX && CX.LK.alias[anchor]) || anchor;
      const chip = $(`#notes-body .gcard .gsec [data-peek-c="${id}"]`, main) || $(`#notes-body [data-peek-c="${id}"]`, main);
      el = chip ? (chip.closest('.gfn, .gterms > div, li, .gcard')) : null;
      if (el && CX.LK.byId[id]) openPeek('c', id);                        // 정의는 옆 패널에
      if (!el) { renderNotesBody('flow'); el = document.getElementById(anchor); }
    }
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
    const html = top.map(x => hitHtml(x.r, q, '')).join('') + `<a class="more" href="#/search?q=${encodeURIComponent(q)}">전체 결과 ${hits.length}건 · 처음 나온 곳 보기 (Enter)</a>`;
    if (html !== lastDrop || drop.hidden) { drop.innerHTML = html; lastDrop = html; }      // 같은 결과면 다시 그리지 않음
    drop.hidden = false; dropSel = -1;
  }
  // 입력마다 바로 그리지 않고 프레임당 한 번만, 한글 조합 중(ㅍ→포→폰)에는 조합이 끝날 때까지 기다린다
  let dropRaf = 0, lastDrop = '';
  const scheduleDrop = () => { if (!dropRaf) dropRaf = requestAnimationFrame(() => { dropRaf = 0; updateDrop(); }); };
  qIn.addEventListener('input', e => { if (!e.isComposing) scheduleDrop(); });
  qIn.addEventListener('compositionend', scheduleDrop);
  qIn.addEventListener('focus', () => { lastDrop = ''; scheduleDrop(); });
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
  const stripTags = t => String(t || '').replace(/[`*]/g, '').replace(/<[^>]+>/g, '');
  function dictPage(kind, params) {
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
    const subjOf = e => [...new Set(e.src.map(r => r.s))];
    main.innerHTML = `<div class="page"><h1>${isT ? '📖 용어 사전' : '🔧 함수·명령 사전'}</h1>
      <p class="sub">교시 노트의 <b>${isT ? '핵심 용어' : '새로 나온 함수/명령'}</b>에서 자동으로 모은 ${list.length}개 · 처음 나온 순서. 이름을 누르면 그 교시로 이동</p>
      <div class="filter-row"><input class="filter" type="search" placeholder="걸러 보기… (이름·설명)" autocomplete="off" spellcheck="false"><span class="fcount"></span></div>
      <div class="chips">${['all', ...SUBJ.map(s => s.id)].map(k => `<button class="chip${k === 'all' ? ' on' : ''}" data-s="${k}">${k === 'all' ? '전체' : sById(k).short}</button>`).join('')}</div>
      <div class="dict">${list.map((e, i) => `<div data-i="${i}"><span class="nm">${isT ? esc(e.name) : `<code>${esc(e.name)}</code>`}</span><span class="df">${mdInline(e.def)}</span><span class="src">${e.src.map(r => `<a href="${href(r.s, r.d, r.n, r.a)}">${loc(r.s, r.d, r.n)}</a>`).join('')}</span></div>`).join('') || '<div class="empty">아직 없습니다.</div>'}</div>
      <div class="empty fnone" hidden>맞는 항목이 없습니다. <a class="fsearch" href="#/search">전체 검색으로 찾기 →</a></div></div>`;
    highlightIn(main);
    // 행·검색 문자열·텍스트 노드는 한 번만 만들어 둔다. 입력마다 DOM 은 건드리지 않고(보이기 토글 + CSS 강조만)
    const HL = typeof CSS !== 'undefined' && CSS.highlights && typeof Highlight !== 'undefined';
    const textNodes = root => { const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT), a = []; while (tw.nextNode()) a.push(tw.currentNode); return a; };
    const rowsEl = $$('.dict > div[data-i]', main).map(el => {
      const e = list[+el.dataset.i];
      const r = { el, subj: subjOf(e), nmLen: e.name.length, key: (e.name + ' ' + stripTags(e.def)).toLowerCase(), nm: $('.nm', el), df: $('.df', el), nmH: null, dfH: null, vis: true, ord: '', tn: null };
      r.nmH = r.nm.innerHTML; r.dfH = r.df.innerHTML;
      if (HL) r.tn = textNodes(el).filter(n => !n.parentNode.closest('.src')).map(n => ({ n, lo: n.nodeValue.toLowerCase() }));
      return r;
    });
    const input = $('.filter', main), cnt = $('.fcount', main), none = $('.fnone', main), fsearch = $('.fsearch', main);
    let sf = 'all', marked = [], raf = 0, lastKey = null;
    const hlRanges = words => {
      const ranges = [];
      for (const r of rowsEl) {
        if (!r.vis) continue;
        for (const { n, lo } of r.tn) {
          let i = 0;
          while (true) {
            let best = -1, bw = 0;
            for (const w of words) { const j = lo.indexOf(w, i); if (j >= 0 && (best < 0 || j < best)) { best = j; bw = w.length; } }
            if (best < 0) break;
            const rg = new Range(); rg.setStart(n, best); rg.setEnd(n, best + bw); ranges.push(rg); i = best + bw;
          }
        }
      }
      return ranges;
    };
    const apply = () => {
      raf = 0;
      const v = input.value.trim().toLowerCase();
      const key = sf + '\n' + v;
      if (key === lastKey) return;                       // 같은 조건이면 아무것도 안 함
      lastKey = key;
      const words = v.split(/\s+/).filter(Boolean);
      if (!HL) { marked.forEach(r => { r.nm.innerHTML = r.nmH; r.df.innerHTML = r.dfH; }); marked = []; }
      let shown = 0;
      for (const r of rowsEl) {
        const ok = (sf === 'all' || r.subj.includes(sf)) && words.every(w => r.key.includes(w));
        if (ok !== r.vis) { r.vis = ok; r.el.hidden = !ok; }
        if (!ok) continue;
        // 이름이 입력어로 시작 → 이름에 포함 → 설명에만 있음 순서 (같은 순위 안에서는 처음 나온 순서)
        const nmLo = r.key.slice(0, r.nmLen);
        const ord = !words.length ? '' : nmLo.startsWith(words[0]) ? '0' : words.every(w => nmLo.includes(w)) ? '1' : '2';
        if (ord !== r.ord) { r.ord = ord; r.el.style.order = ord; }
        shown++;
        if (!HL && words.length && shown <= 80) { markText(r.nm, v); markText(r.df, v); marked.push(r); }
      }
      if (HL) { CSS.highlights.delete('dictmark'); if (words.length) CSS.highlights.set('dictmark', new Highlight(...hlRanges(words))); }
      cnt.textContent = words.length || sf !== 'all' ? `${shown} / ${rowsEl.length}` : '';
      none.hidden = shown > 0 || !rowsEl.length;
      fsearch.href = `#/search?q=${encodeURIComponent(input.value.trim())}`;
    };
    const schedule = () => { if (!raf) raf = requestAnimationFrame(apply); };
    input.addEventListener('input', e => { if (!e.isComposing) schedule(); });
    input.addEventListener('compositionend', schedule);                    // 한글 조합이 끝났을 때
    input.addEventListener('keydown', e => { if (e.key === 'Escape') { input.value = ''; schedule(); } });
    $$('.chip', main).forEach(c => c.addEventListener('click', () => { sf = c.dataset.s; $$('.chip', main).forEach(x => x.classList.toggle('on', x === c)); schedule(); }));
    if (params && params.get('f')) { input.value = params.get('f'); apply(); }
    input.focus();
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
    const s = sById(params.get('s')) ? params.get('s') : 'c', dsel = params.get('d') || 'all', mode = params.get('m') || 'rand';
    main.innerHTML = '<div class="empty">문항 불러오는 중…</div>';
    const my = ROUTE_N;
    let data; try { data = await loadSubject(s); } catch (e) { if (my === ROUTE_N) main.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
    if (my !== ROUTE_N) return;
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
  let ROUTE_N = 0;                                                             // 늦게 끝난 이전 페이지가 화면을 덮어쓰지 않게
  function route() {
    ROUTE_N++;
    const raw = location.hash.replace(/^#/, '') || '/';
    const [path, qs] = raw.split('?'); const params = new URLSearchParams(qs || '');
    const seg = path.split('/').filter(Boolean);
    document.body.classList.remove('menu', 'side-peek'); $('#scrim').hidden = true;
    const fo = $('.figov'); if (fo) fo.remove();
    closePeek(); CX = null;
    if (typeof CSS !== 'undefined' && CSS.highlights) CSS.highlights.delete('dictmark');
    $$('.top-nav a').forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#/' + (seg[0] || '')));
    let cur = null;
    if (!seg.length) renderHome();
    else if (seg[0] === 'search') { qIn.value = params.get('q') || ''; renderSearch(params.get('q') || '', params.get('k')); }
    else if (seg[0] === 'terms' || seg[0] === 'funcs') dictPage(seg[0], params);
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
  const narrow = () => matchMedia('(max-width: 900px)').matches;
  /* 좌측 목차: 데스크톱은 ☰ 또는 목차 위에 마우스가 있을 때만 펼침 (본문 위에 겹쳐 뜸)
   * ☰ → 목차로 옮겨 가는 사이 닫히지 않게 잠깐 기다린다. 터치는 ☰ 클릭으로 열고 닫기, 모바일은 서랍 */
  const menuBtn = $('#menuBtn'); let sideTimer = 0;
  const sideShow = () => { if (narrow()) return; clearTimeout(sideTimer); document.body.classList.add('side-peek'); menuBtn.setAttribute('aria-expanded', 'true'); };
  const sideHide = (delay = 250) => { if (narrow()) return; clearTimeout(sideTimer); sideTimer = setTimeout(() => { document.body.classList.remove('side-peek'); menuBtn.setAttribute('aria-expanded', 'false'); }, delay); };
  [menuBtn, side].forEach(el => { el.addEventListener('mouseenter', sideShow); el.addEventListener('mouseleave', () => sideHide()); });
  side.addEventListener('focusin', sideShow);
  side.addEventListener('focusout', e => { if (!side.contains(e.relatedTarget)) sideHide(0); });
  menuBtn.addEventListener('click', () => {
    if (narrow()) { const o = document.body.classList.toggle('menu'); $('#scrim').hidden = !o; return; }
    const touch = matchMedia('(hover: none)').matches;                   // 마우스가 없으면 클릭으로 열고 닫기
    if (touch && document.body.classList.contains('side-peek')) sideHide(0); else sideShow();
  });
  menuBtn.setAttribute('aria-expanded', 'false');
  try { localStorage.removeItem('wiki.sideOff'); localStorage.removeItem('wiki.noteView'); } catch (e) { /* 예전 상태 정리 */ }
  $('#dictBtn').addEventListener('click', () => (dictIsOpen() && rTab === 'dict') ? rpSet(false) : rpSet(true, 'dict', true));
  $('#codeBtn').addEventListener('click', () => (dictIsOpen() && rTab === 'code') ? rpSet(false) : rpSet(true, 'code'));
  document.addEventListener('click', e => { if (e.target.closest('[data-code-list]')) { CODE.view = 'list'; rpSet(true, 'code'); } });
  if (store.get('dictOn', false)) rpSet(true);
  $('#scrim').addEventListener('click', () => { document.body.classList.remove('menu'); $('#scrim').hidden = true; });
  $('#themeBtn').addEventListener('click', () => {
    const r = document.documentElement; const dark = r.dataset.theme ? r.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    r.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('wiki.theme', r.dataset.theme); } catch (e) { /* 무시 */ }
  });
  route();
})();
