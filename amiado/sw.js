const CACHE = 'amiado-v16';
const STATIC = [
  './',
  './index.html',
  './style.css',
  './app.js',
  './aologo.svg',
  './songs/index.json'
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(STATIC)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  // Only handle same-origin GET requests
  if (e.request.method !== 'GET' || !e.request.url.startsWith(self.location.origin)) return;

  e.respondWith(
    caches.match(e.request).then(cached => {
      if (cached) return cached;
      return fetch(e.request).then(res => {
        // Cache song covers and audio metadata on the fly
        if (res.ok && (e.request.url.includes('.jpeg') || e.request.url.includes('.jpg') || e.request.url.includes('.svg'))) {
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return res;
      }).catch(() => {
        // Offline fallback: return cached index for navigation
        if (e.request.mode === 'navigate') return caches.match('./index.html');
      });
    })
  );
});
