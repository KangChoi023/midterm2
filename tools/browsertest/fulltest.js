// node fulltest.js <url> [w] [h] [lessons|code|misc]  — 위키 전체를 돌며 예외 · 이상을 모은다
const [,, url, W = '1360', H = '900', mode = 'lessons'] = process.argv;
const H_ = `
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const until = async (f, ms = 4000) => { const t0 = Date.now(); while (Date.now() - t0 < ms) { try { if (f()) return true; } catch (e) {} await sleep(25); } return false; };
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const click = el => el && el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
  const out = { problems: [] }; const P = m => { if (out.problems.length < 60) out.problems.push(m); };
  try { localStorage.clear(); } catch (e) {}`;
const T = {
  lessons: `
    let n = 0;
    for (const S of window.WIKI_INDEX.subjects) for (const D of S.days) for (const L of D.lessons) {
      location.hash = '#/' + S.id + '/' + D.n + '/' + L.n;
      if (!await until(() => { const h = $('.lesson h1'); return h && h.textContent.startsWith(L.n + '교시'); })) { P('not rendered ' + location.hash); continue; }
      n++;
      if (document.documentElement.scrollWidth > innerWidth + 1) P('hscroll ' + location.hash);
      for (const v of ['flow', 'concept']) { const b = $('[data-view="' + v + '"]'); if (b) { click(b); await sleep(15); if (!$('#notes-body') || !$('#notes-body').children.length) P('empty ' + v + ' ' + location.hash); } }
      const kw = $('.kwc'); if (kw) { click(kw); await sleep(5); click($('[data-kwnav="0"]')); }
      const c = $('#notes-body [data-peek-c]'); if (c) { click(c); await sleep(5); if ($('#peek').hidden) P('peek ' + location.hash); click($('[data-pk-close]')); }
      for (const q of $$('#quiz .q')) { if ($('[data-check], [data-show]', q)) continue; const o = $('.opt', q); if (o) { click(o); if ($('.q-ans', q).hidden) P('quiz answer ' + location.hash + ' ' + q.id); } }
    }
    out.n = n; return out;`,
  code: `
    let n = 0, steps = 0;
    location.hash = '#/c/1/1'; await until(() => $('.lesson h1'));
    for (const r of window.WIKI_INDEX.code) {
      const b = document.createElement('button'); b.dataset.codeOpen = r.slice(0, 4).join('|'); document.body.appendChild(b); click(b); b.remove();
      if (!await until(() => $('#sideCode .code-head'))) { P('code ' + r[4]); continue; }
      n++;
      if (r[9]) { await until(() => $$('#sideCode .gband').length === r[9], 1500) || P('bands ' + r[4]);
        for (const band of $$('#sideCode .gband')) { const rc = band.getBoundingClientRect(); click(band.parentElement); band.parentElement.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: rc.left + 40, clientY: rc.top + Math.min(rc.height / 2, 8) })); steps++;
          const p = $('#sideCode .gpop'); if (!p || !p.textContent.startsWith((+band.dataset.gi + 1) + '.')) P('pop ' + r[4] + ' #' + band.dataset.gi); } }
      const h = $('#sideCode [data-hlc], #sideCode [data-hl-word]'); if (h) { click(h); if (!$('#sideCode .hlbar')) P('hl ' + r[4]); click($('#sideCode [data-hlnav="0"]')); }
    }
    out.n = n; out.steps = steps; return out;`,
  misc: `
    location.hash = '#/linux/2/4'; await until(() => $('.lesson h1'));
    click($('#dictBtn')); await sleep(50); if ($('#dictPanel').hidden) P('dict open');
    const q = $('#sideDict .dict-q'); q.value = 'sched_setaffinity'; q.dispatchEvent(new Event('input', { bubbles: true })); await sleep(80);
    if (!$$('#sideDict .dent').length) P('dict search');
    click($('.rtabs [data-rt="code"]')); await sleep(50); if (!$$('#sideCode .citem').length) P('code list');
    click($('#dictClose'));
    for (const w of ['malloc', 'mutex', 'pwm', 'I2C', '포인터', 'affinity']) {
      location.hash = '#/search?q=' + encodeURIComponent(w); await until(() => $('.page h1') && $('.page h1').textContent.includes(w));
      if (!$$('.hit').length) P('search ' + w);
    }
    for (const k of ['terms', 'funcs']) { location.hash = '#/' + k; await until(() => $('.dict')); if (!$$('.dict > div').length) P(k); }
    for (const s of ['c', 'linux', 'mcu', 'eio']) for (const m of ['rand', 'all']) {
      location.hash = '#/quiz?s=' + s + '&m=' + m + '&r=' + Math.random(); await until(() => $('#qs') && $('#qs').value === s);
      await sleep(100); if (!$$('.q').length) P('quiz ' + s + ' ' + m);
    }
    location.hash = '#/zzz'; await sleep(100); if (!$('.empty')) P('notFound');
    return out;`,
};
require('./lib').open(url, +W, +H, +W < 600).then(async p => {
  console.log(JSON.stringify(await p.run(`(async () => { ${H_} ${T[mode]} })()`)));
  console.log('ERRORS', p.errs.length); p.errs.slice(0, 30).forEach(e => console.log(' ', e)); p.close();
});
