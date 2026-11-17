// Guarda el juego en el teléfono para que funcione sin internet.
const CACHE = 'kuriri-v5';
const ARCHIVOS = ['./', './index.html', './manifest.webmanifest', './icono-180.png', './icono-192.png', './icono-512.png'];
self.addEventListener('install', e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(ARCHIVOS)).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(
  caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  // Solo archivos del juego: el ranking (Supabase) va siempre directo a la red.
  if (e.request.method !== 'GET' || new URL(e.request.url).origin !== location.origin) return;
  // Red primero (para recibir versiones nuevas), caché si no hay internet.
  e.respondWith(fetch(e.request).then(r => {
    if (r.ok && new URL(e.request.url).origin === location.origin) { const copia = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copia)); }
    return r;
  }).catch(() => caches.match(e.request).then(r => r || caches.match('./index.html'))));
});
