// Chrome(9333)에 붙어 새로 불러온 페이지에서 식을 평가 (about:blank 를 거쳐 매번 새로 읽음)
exports.open = async (url, w = 1360, h = 900, mobile = false) => {
  const list = await (await fetch('http://127.0.0.1:9333/json/list')).json();
  const ws = new WebSocket(list.find(p => p.type === 'page').webSocketDebuggerUrl); let id = 0; const wait = {}; const errs = [];
  const send = (method, params = {}) => new Promise(r => { const i = ++id; wait[i] = r; ws.send(JSON.stringify({ id: i, method, params })); });
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.id && wait[m.id]) { wait[m.id](m); delete wait[m.id]; return; }
    if (m.method === 'Runtime.exceptionThrown') { const d = m.params.exceptionDetails; errs.push('EXC ' + ((d.exception && d.exception.description) || d.text).split('\n').slice(0, 3).join(' | ')); }
    if (m.method === 'Runtime.consoleAPICalled' && /error|warning/.test(m.params.type)) errs.push('CON ' + m.params.args.map(a => a.value || a.description).join(' '));
    if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') errs.push('LOG ' + m.params.entry.text + ' ' + (m.params.entry.url || ''));
  };
  await new Promise(r => ws.onopen = r);
  await send('Runtime.enable'); await send('Log.enable'); await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile });
  await send('Page.navigate', { url: 'about:blank' }); await new Promise(r => setTimeout(r, 300));
  await send('Page.navigate', { url }); await new Promise(r => setTimeout(r, 3000));
  const run = async expr => {
    const res = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true, timeout: 900000 });
    if (res.result.exceptionDetails) return 'EVAL-ERR ' + JSON.stringify(res.result.exceptionDetails).slice(0, 400);
    return res.result.result.value;
  };
  return { send, run, errs, close: () => ws.close() };
};
