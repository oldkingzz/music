// Syntax check of the inline scripts of an HTML page (no DOM, nothing runs): node check_page_js_v1.js <page.html>
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const re = /<script>([\s\S]*?)<\/script>/g;
let m, n = 0, bad = 0;
while ((m = re.exec(html)) !== null) {
  n += 1;
  try { new vm.Script(m[1], {filename: `inline-${n}.js`}); }
  catch (e) { bad += 1; console.log(`inline script ${n}: ${e.name}: ${e.message}`); }
}
console.log(`${process.argv[2].split('/').pop()}: ${n} inline script(s), ${bad} with syntax errors`);
process.exit(bad ? 1 : 0);
