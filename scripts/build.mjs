import { cpSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
const source = fileURLToPath(new URL('../frontend/', import.meta.url));
const target = fileURLToPath(new URL('../dist/', import.meta.url));
mkdirSync(target, { recursive: true });
cpSync(source, target, { recursive: true });
let html = readFileSync(target + 'index.html', 'utf8');
for (const asset of ['app.js', 'styles.css']) {
  const hash = createHash('sha256').update(readFileSync(target + asset)).digest('hex').slice(0,12);
  html = html.replace(new RegExp('/' + asset.replace('.', '\\.') + '(?:\\?v=[^"\\s]+)?'), '/' + asset + '?v=' + hash);
}
writeFileSync(target + 'index.html', html);
console.log('GENCO Studio static build: dist/ (no remote CDN dependencies)');
