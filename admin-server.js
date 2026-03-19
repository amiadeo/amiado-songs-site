// ─────────────────────────────────────────
// Amiado Admin Server — Local only
// Run: node admin-server.js
// Then open admin-tool.html in browser
// ─────────────────────────────────────────
const http = require('http');
const fs   = require('fs');
const path = require('path');

const APP_JS = path.join(__dirname, 'amiado', 'app.js');
const PORT   = 3333;

const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  // Health check
  if (req.method === 'GET' && req.url === '/ping') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true }));
    return;
  }

  // Add song
  if (req.method === 'POST' && req.url === '/add-song') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const { code } = JSON.parse(body);

        if (!code || !code.trim()) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'No code provided' }));
          return;
        }

        let content = fs.readFileSync(APP_JS, 'utf8');

        // Find the end of the SONGS array.
        // The array ends with the last song followed by  }]; or  },\n];
        // We look for the last occurrence of the SONGS closing bracket.
        // Marker: a line that is exactly "];  " or "];" that closes the SONGS const.
        // Strategy: find "const SONGS = [", then find the matching "];"
        const songsStart = content.indexOf('const SONGS = [');
        if (songsStart === -1) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Could not find SONGS array in app.js' }));
          return;
        }

        // Find the closing ]; of the SONGS array by counting brackets
        let depth = 0;
        let i = content.indexOf('[', songsStart);
        let closeIdx = -1;
        while (i < content.length) {
          if (content[i] === '[') depth++;
          else if (content[i] === ']') {
            depth--;
            if (depth === 0) { closeIdx = i; break; }
          }
          i++;
        }

        if (closeIdx === -1) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Could not find end of SONGS array' }));
          return;
        }

        // Insert the new song code before the closing ]
        // Trim the code and make sure it ends with a comma
        let songCode = code.trim();
        if (!songCode.endsWith(',')) songCode += ',';

        const before = content.slice(0, closeIdx);
        const after  = content.slice(closeIdx);

        // Add a newline before if needed
        const newContent = before.trimEnd() + '\n  ' + songCode + '\n' + after;

        // Backup the original file
        fs.writeFileSync(APP_JS + '.bak', content, 'utf8');

        // Write updated file
        fs.writeFileSync(APP_JS, newContent, 'utf8');

        console.log(`✓ Song added successfully`);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true, message: 'Song added to app.js' }));

      } catch (e) {
        console.error('Error:', e.message);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  res.writeHead(404);
  res.end();
});

server.listen(PORT, () => {
  console.log('');
  console.log('  ♪  Amiado Admin Server');
  console.log(`  →  http://localhost:${PORT}`);
  console.log('');
  console.log('  פתח את admin-tool.html בדפדפן');
  console.log('  Ctrl+C להפסקה');
  console.log('');
});
