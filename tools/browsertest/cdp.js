// node cdp.js <url> <js식>
const [,, url, expr, w, h] = process.argv;
require('./lib').open(url, +(w || 1360), +(h || 900)).then(async p => { console.log(await p.run(expr)); if (p.errs.length) console.log('ERRORS', p.errs); p.close(); });
