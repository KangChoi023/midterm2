// node shot.js <url> <js식> <out.png> [w] [h]
const fs = require('fs'); const [,, url, expr, out, w, h] = process.argv;
require('./lib').open(url, +(w || 1360), +(h || 900)).then(async p => {
  const v = await p.run(expr); if (v !== undefined) console.log(v);
  await new Promise(r => setTimeout(r, 500));
  const s = await p.send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(out, Buffer.from(s.result.data, 'base64')); if (p.errs.length) console.log('ERRORS', p.errs); p.close();
});
