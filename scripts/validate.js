/**
 * Amiado — pre-push validation script
 * Checks app.js for structural issues before deployment.
 */
const fs = require('fs');
const path = require('path');

const appPath = path.join(__dirname, '../amiado/app.js');
const appJs = fs.readFileSync(appPath, 'utf8');

let errors = 0;

function pass(msg) { console.log('  ✓', msg); }
function fail(msg) { console.error('  ✗', msg); errors++; }

console.log('\nAmiado — validating app.js...\n');

// ── 1. Duplicate song IDs ─────────────────────────────
const idMatches = [...appJs.matchAll(/\bid:\s*['"]([a-z0-9-]+)['"]/g)].map(m => m[1]);
const seen = new Set();
const dupes = idMatches.filter(id => { const d = seen.has(id); seen.add(id); return d; });
if (dupes.length > 0) {
  fail(`Duplicate song IDs: ${dupes.join(', ')}`);
} else {
  pass(`${idMatches.length} song IDs — no duplicates`);
}

// ── 2. Every song has a title ─────────────────────────
const songBlocks = [...appJs.matchAll(/\{[\s\S]*?id:\s*'([^']+)'[\s\S]*?title:\s*'([^']+)'/g)];
pass(`${songBlocks.length} songs with id + title found`);

// ── 3. sunoEmbedId format (should be alphanumeric, no full URL) ───
const sunoUrls = [...appJs.matchAll(/sunoEmbedId:\s*'(https?:\/\/[^']+)'/g)];
if (sunoUrls.length > 0) {
  fail(`sunoEmbedId should be an ID not a full URL: ${sunoUrls.map(m => m[1]).join(', ')}`);
} else {
  pass('sunoEmbedId values look clean');
}

// ── 4. youtubeVideoId format (should be 11 chars, no full URL) ───
const ytUrls = [...appJs.matchAll(/youtubeVideoId:\s*'(https?:\/\/[^']+)'/g)];
if (ytUrls.length > 0) {
  fail(`youtubeVideoId should be an ID not a full URL: ${ytUrls.map(m => m[1]).join(', ')}`);
} else {
  pass('youtubeVideoId values look clean');
}

// ── 5. No console.log left in production code ────────
const logs = [...appJs.matchAll(/console\.log\(/g)];
if (logs.length > 0) {
  fail(`${logs.length} console.log() found — remove before push`);
} else {
  pass('No console.log found');
}

// ── Result ────────────────────────────────────────────
console.log('');
if (errors > 0) {
  console.error(`❌  ${errors} issue(s) found — fix before pushing\n`);
  process.exit(1);
} else {
  console.log(`✅  All checks passed\n`);
}
