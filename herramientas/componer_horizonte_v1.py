"""«Horizonte»: tema original de Kuriri aburrido en el espacio (120 BPM, Re menor).

Electrónica cinemática de los 90: arranca suave (reloj, percusión lejana), entran coros sintetizados,
sube con bajo y arpegios, explota en una parte electrónica épica, se frena en el horizonte de eventos
(dilatación del tiempo) y termina en un clímax que vuelve al inicio sin cortes.

Todo se sintetiza aquí: no usa grabaciones ni muestras de terceros.
Uso:  python componer_horizonte.py salida.wav
"""
import sys
import numpy as np
from scipy.signal import butter, lfilter, fftconvolve
from scipy.io import wavfile

SR = 44100
BPM = 120
PASO = 60 / BPM / 4                 # semicorchea
COMPAS = 16 * PASO
rng = np.random.default_rng(42)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ================================================================= osciladores (polyBLEP, sin aliasing)
def _blep(t, dt):
    y = np.zeros_like(t)
    a = t < dt
    x = t[a] / dt[a]
    y[a] = x + x - x * x - 1
    b = t > 1 - dt
    x = (t[b] - 1) / dt[b]
    y[b] = x * x + x + x + 1
    return y


def sierra(f, n, fase0=None):
    f = np.broadcast_to(np.asarray(f, float), (n,))
    dt = f / SR
    ph = ((rng.random() if fase0 is None else fase0) + np.cumsum(dt)) % 1.0
    return 2 * ph - 1 - _blep(ph, dt)


def cuadrada(f, n, ciclo=0.5):
    f = np.broadcast_to(np.asarray(f, float), (n,))
    dt = f / SR
    ph = (rng.random() + np.cumsum(dt)) % 1.0
    y = np.where(ph < ciclo, 1.0, -1.0)
    y += _blep(ph, dt)
    y -= _blep((ph + 1 - ciclo) % 1.0, dt)
    return y


def supersierra(m, n, voces=7, ancho=0.18, vib=0.0):
    """Varias sierras desafinadas: el sonido grueso de los sintetizadores de trance de los 90."""
    t = np.arange(n) / SR
    y = np.zeros(n)
    for v in range(voces):
        d = (v - (voces - 1) / 2) / ((voces - 1) / 2) * ancho if voces > 1 else 0
        f = hz(m + d) * (2 ** (vib * np.sin(2 * np.pi * 5.2 * t + v) / 12) if vib else 1)
        y += sierra(f, n)
    return y / voces


def env(n, a=0.01, d=0.2, s=0.7, r=0.2, nota=None):
    nota = n if nota is None else max(1, min(nota, n))
    i = np.arange(n)
    A, D, R = max(1, int(a * SR)), max(1, int(d * SR)), max(1, int(r * SR))
    e = np.where(i < A, i / A, s + (1 - s) * np.exp(-(i - A) / D))
    nivel = e[nota - 1]
    e[nota:] = nivel * np.exp(-(i[nota:] - nota) / R)
    return e


def lp(x, fc, o=2):
    return lfilter(*butter(o, np.clip(fc, 20, SR / 2 - 200) / (SR / 2), 'low'), x)


def hp(x, fc, o=2):
    return lfilter(*butter(o, fc / (SR / 2), 'high'), x)


def bp(x, f1, f2, o=2):
    return lfilter(*butter(o, [f1 / (SR / 2), f2 / (SR / 2)], 'band'), x)


def resonante(x, f0, q):
    """Pasa banda resonante (formante)."""
    w = 2 * np.pi * f0 / SR
    al = np.sin(w) / (2 * q)
    b = np.array([al, 0, -al]); a = np.array([1 + al, -2 * np.cos(w), 1 - al])
    return lfilter(b / a[0], a / a[0], x)


def lp_barrido(x, f_ini, f_fin, bloque=2048):
    """Filtro que se abre o cierra en el tiempo (por bloques)."""
    y = np.zeros_like(x)
    nb = max(1, len(x) // bloque + 1)
    zi = None
    for k in range(nb):
        i, j = k * bloque, min(len(x), (k + 1) * bloque)
        if i >= j:
            break
        fc = f_ini * (f_fin / f_ini) ** (k / max(1, nb - 1))
        b, a = butter(2, np.clip(fc, 30, SR / 2 - 300) / (SR / 2), 'low')
        if zi is None:
            zi = np.zeros(max(len(a), len(b)) - 1)
        y[i:j], zi = lfilter(b, a, x[i:j], zi=zi)
    return y


# ================================================================= instrumentos
def coro(m, dur, vocal='a'):
    """Voces de coro sintetizadas: sierras desafinadas filtradas por formantes de vocal + aliento."""
    n = int((dur + 1.2) * SR)
    t = np.arange(n) / SR
    fuente = np.zeros(n)
    for v in range(5):
        f = hz(m + (v - 2) * 0.09) * 2 ** (0.12 * np.sin(2 * np.pi * (4.6 + v * 0.3) * t + v) / 12)
        fuente += sierra(f, n)
    fuente += rng.normal(0, 0.15, n)                      # aliento
    F = {'a': [(750, 1.0, 6), (1150, 0.6, 7), (2600, 0.25, 9)],
         'o': [(480, 1.0, 6), (820, 0.6, 7), (2550, 0.18, 9)]}[vocal]
    y = sum(g * resonante(fuente, f0, q) for f0, g, q in F)
    return lp(y, 4200) * env(n, 0.6, 0.8, 0.9, 1.0, int(dur * SR))


def tambor_grave(g=1.0):
    """Tambor épico (tipo taiko): cuerpo grave con caída de tono + golpe de ruido."""
    n = int(1.6 * SR); t = np.arange(n) / SR
    f = 52 + 70 * np.exp(-t * 18)
    cuerpo = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.2)
    golpe = lp(rng.uniform(-1, 1, n), 1800) * np.exp(-t * 28) * 0.6
    return np.tanh((cuerpo + golpe) * 1.8) * g


def bombo909():
    n = int(0.55 * SR); t = np.arange(n) / SR
    f = 44 + 140 * np.exp(-t * 28)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6.5)
    y[:90] += hp(rng.uniform(-1, 1, 90), 3000) * np.linspace(0.5, 0, 90)
    return np.tanh(y * 2.2)


def palmas():
    n = int(0.35 * SR); t = np.arange(n) / SR
    base = bp(rng.uniform(-1, 1, n), 900, 5200)
    e = np.zeros(n)
    for k, off in enumerate((0, 0.011, 0.022)):                # tres palmas casi juntas
        i = int(off * SR)
        e[i:] += np.exp(-(t[i:] - off) * (60 if k < 2 else 16))
    return base * e


def caja_grande():
    n = int(0.5 * SR); t = np.arange(n) / SR
    r = bp(rng.uniform(-1, 1, n), 700, 8000) * np.exp(-t * 11)
    c = np.sin(2 * np.pi * (180 + 40 * np.exp(-t * 40)) * t) * np.exp(-t * 18)
    return np.tanh((r * 1.2 + c * 0.8) * 1.4)


def hat(abierto=False):
    n = int((0.35 if abierto else 0.06) * SR); t = np.arange(n) / SR
    r = sum(cuadrada(f, n) for f in (317, 421, 587, 733, 889, 1123)) / 6       # metálico, estilo 808
    r = hp(r + rng.uniform(-0.5, 0.5, n), 7000)
    return r * np.exp(-t * (9 if abierto else 70))


def shaker():
    n = int(0.09 * SR); t = np.arange(n) / SR
    return bp(rng.uniform(-1, 1, n), 4000, 12000) * np.sin(np.pi * t / t[-1]) ** 2


def tic(alto):
    n = int(0.03 * SR); t = np.arange(n) / SR
    return np.sin(2 * np.pi * (2400 if alto else 1800) * t) * np.exp(-t * 260)


def platillo_largo():
    n = int(3.5 * SR); t = np.arange(n) / SR
    return hp(rng.uniform(-1, 1, n), 5000) * np.exp(-t * 1.4)


def caida_grave(dur=2.0):
    """Caída de tono en el grave (sub-drop): Kuriri cae."""
    n = int(dur * SR); t = np.arange(n) / SR
    f = 30 + 70 * np.exp(-t * 2.2)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.2)


def subida(dur):
    """Ruido que sube de tensión (riser)."""
    n = int(dur * SR); t = np.arange(n) / SR
    r = lp_barrido(rng.uniform(-1, 1, n), 300, 12000)
    tono = sierra(hz(50) * 2 ** (t / dur * 3), n) * 0.25
    return (r + lp(tono, 3000)) * (t / dur) ** 2


# ================================================================= forma y armonía
ACORDES = {'Dm': (50, [0, 3, 7]), 'Bb': (46, [0, 4, 7]), 'F': (53, [0, 4, 7]), 'C': (48, [0, 4, 7]),
           'Gm': (55, [0, 3, 7]), 'A': (57, [0, 4, 7]), 'Am': (57, [0, 3, 7])}
P1 = ['Dm', 'Bb', 'F', 'C']
P2 = ['Dm', 'Bb', 'Gm', 'A']
SECCIONES = [            # (nombre, compases, progresión)
    ('duerme', 8, P1 + P1),
    ('sueno', 8, P1 + P2),
    ('despega', 8, P1 + P2),
    ('espacio', 16, P1 + P2 + P1 + P2),
    ('horizonte', 8, ['Gm', 'Dm', 'Bb', 'A', 'Gm', 'Dm', 'Bb', 'A']),
    ('climax', 16, P1 + P2 + P1 + P2),
    ('regreso', 4, P1),
]
# Melodía principal (paso, duración en pasos, nota MIDI) por compás; 8 compases.
MELODIA = [
    [(0, 6, 69), (6, 4, 74), (10, 2, 76), (12, 4, 77)],
    [(0, 8, 74), (8, 4, 77), (12, 4, 82)],
    [(0, 10, 81), (10, 2, 79), (12, 2, 77), (14, 2, 76)],
    [(0, 8, 76), (8, 4, 72), (12, 2, 76), (14, 2, 79)],
    [(0, 6, 81), (6, 2, 77), (8, 4, 74), (12, 4, 81)],
    [(0, 6, 82), (6, 2, 81), (8, 4, 77), (12, 4, 74)],
    [(0, 4, 79), (4, 4, 81), (8, 4, 82), (12, 4, 86)],
    [(0, 8, 85), (8, 4, 81), (12, 4, 76)],
]
# Contramelodía del clímax (más aguda, en blancas)
CONTRA = [[(0, 8, 86), (8, 8, 89)], [(0, 8, 89), (8, 8, 86)], [(0, 8, 84), (8, 8, 88)], [(0, 16, 84)],
          [(0, 8, 86), (8, 8, 89)], [(0, 8, 89), (8, 8, 94)], [(0, 8, 91), (8, 8, 94)], [(0, 16, 93)]]


def poner(pista, x, t, g=1.0):
    i = int(t * SR)
    if i >= len(pista):
        return
    j = min(len(pista), i + len(x))
    pista[i:j] += x[:j - i] * g


def reverb(x, seg=3.2, oscuro=6000, semilla=1):
    r = np.random.default_rng(semilla)
    n = int(seg * SR); t = np.arange(n) / SR
    ir = r.normal(0, 1, n) * np.exp(-t * 6.9 / seg)
    ir = lp(ir, oscuro)
    ir[:int(0.02 * SR)] = 0                                 # pre-delay
    ir /= np.sqrt((ir ** 2).sum())
    return fftconvolve(x, ir)[:len(x)]


def eco_pingpong(x, pasos=3, retro=0.42):
    d = int(pasos * PASO * SR)
    L, R = np.zeros_like(x), np.zeros_like(x)
    for k in range(1, 8):
        s = np.zeros_like(x); s[d * k:] = x[:-d * k] * retro ** k
        (L if k % 2 else R)[:] += s
    return lp(L, 3800), lp(R, 3800)


def frenar(x, i, j):
    """Dilatación del tiempo: el tramo [i, j) se frena como una cinta que se detiene."""
    seg = x[i:j].copy()
    n = len(seg)
    vel = np.linspace(1.0, 0.0, n) ** 0.7
    pos = np.cumsum(vel)
    pos = pos / pos[-1] * (n * 0.55)
    x[i:j] = np.interp(pos, np.arange(n), seg) * np.linspace(1, 0.2, n)


# ================================================================= composición
def componer():
    tl = []                                             # (sección, índice en sección, acorde)
    for nombre, nc, prog in SECCIONES:
        for k in range(nc):
            tl.append((nombre, k, prog[k % len(prog)], nc))
    total = len(tl) * COMPAS
    N = int((total + 4.5) * SR)
    pista = {k: np.zeros(N) for k in ('bat', 'tambor', 'bajo', 'coro', 'pad', 'arp', 'lead', 'fx', 'sub')}

    K909, CLAP, CAJA, HAT, HATA, SHK = bombo909(), palmas(), caja_grande(), hat(), hat(True), shaker()
    TAMB, CRASH = tambor_grave(), platillo_largo()

    for c, (sec, k, ac, nc) in enumerate(tl):
        t0 = c * COMPAS
        raiz, tipo = ACORDES[ac]
        ultimo = k == nc - 1
        P = lambda p: t0 + p * PASO

        # ---------------- percusión
        if sec == 'duerme':
            if k < 6:
                for p in range(0, 16, 4):
                    poner(pista['fx'], tic(p % 8 == 0), P(p), 0.25 * (1 - k / 7))
            for p in (0, 10):
                poner(pista['bat'], lp(K909, 400), P(p), 0.35 + 0.05 * k)
            for p in range(2, 16, 4):
                poner(pista['bat'], SHK, P(p), 0.25 + 0.03 * k)
            if k >= 4:
                poner(pista['tambor'], TAMB, P(0), 0.45)
        elif sec == 'sueno':
            poner(pista['tambor'], TAMB, P(0), 0.75)
            poner(pista['tambor'], TAMB, P(10), 0.45)
            if k % 2:
                poner(pista['tambor'], TAMB, P(14), 0.35)
            for p in (0, 10):
                poner(pista['bat'], K909, P(p), 0.5)
            for p in range(2, 16, 2):
                poner(pista['bat'], SHK, P(p), 0.3)
            poner(pista['bat'], CLAP, P(12), 0.35)
        elif sec in ('despega', 'espacio', 'climax'):
            fuerte = sec != 'despega'
            for p in (0, 4, 8, 12):
                poner(pista['bat'], K909, P(p), 0.95 if fuerte else 0.7)
            if fuerte or k >= 4:
                for p in (4, 12):
                    poner(pista['bat'], CLAP, P(p), 0.6)
                    poner(pista['bat'], CAJA, P(p), 0.3 if fuerte else 0.2)
            for p in range(16):
                if p % 4 == 2:
                    poner(pista['bat'], HATA, P(p), 0.28)
                elif fuerte or p % 2 == 0:
                    poner(pista['bat'], HAT, P(p), 0.2 if p % 2 else 0.26)
            if sec == 'climax' or (sec == 'espacio' and k >= 8):
                poner(pista['tambor'], TAMB, P(0), 0.6)
                if k % 4 == 3:
                    for p in (8, 11, 14):
                        poner(pista['tambor'], TAMB, P(p), 0.5)
            if k == 0 and sec != 'despega':
                poner(pista['fx'], CRASH, P(0), 0.5)
                poner(pista['sub'], caida_grave(2.2), P(0), 0.8)
        elif sec == 'horizonte':
            if k % 2 == 0:
                poner(pista['tambor'], TAMB, P(0), 0.5)
            poner(pista['tambor'], TAMB, P(8), 0.2)
        elif sec == 'regreso':
            for p in (0, 10):
                poner(pista['bat'], lp(K909, 500), P(p), 0.5 - 0.08 * k)
            for p in range(2, 16, 4):
                poner(pista['bat'], SHK, P(p), 0.3)
            poner(pista['fx'], tic(True), P(0), 0.12 * k)

        # redobles y subidas al final de las secciones que suben
        if ultimo and sec in ('despega', 'espacio', 'sueno'):
            for p in range(8, 16):
                poner(pista['bat'], CAJA, P(p), 0.15 + 0.06 * (p - 8))
        if sec in ('despega', 'horizonte') and k == nc - 4:
            poner(pista['fx'], subida(4 * COMPAS), t0, 0.35)

        # ---------------- bajo
        if sec in ('despega', 'espacio', 'climax'):
            for p in range(16):
                if sec == 'despega' and k < 4 and p % 2:
                    continue
                m = raiz - 12 + (12 if p % 4 == 2 else 0)
                n = int(PASO * 1.05 * SR)
                x = sierra(hz(m), n) * 0.7 + cuadrada(hz(m - 0.05), n) * 0.3
                fc = 2500 if p % 4 == 2 else 900
                x = lp_barrido(x, fc, 180, 512) * env(n, 0.002, 0.07, 0.35, 0.02)
                x += np.sin(2 * np.pi * hz(m) * np.arange(n) / SR) * env(n, 0.002, 0.1, 0.6, 0.02) * 0.7
                poner(pista['bajo'], x, P(p), 0.75 if p % 4 else 0.0)        # el bombo ocupa el tiempo 1
        elif sec in ('horizonte',):
            m = raiz - 24
            n = int(COMPAS * SR)
            x = np.sin(2 * np.pi * hz(m) * np.arange(n) / SR) * env(n, 0.3, 1, 0.9, 0.4)
            poner(pista['sub'], x, t0, 0.45)
        elif sec in ('duerme', 'sueno', 'regreso'):
            m = raiz - 12
            n = int((COMPAS + 0.5) * SR)
            x = lp(supersierra(m, n, 3, 0.1), 220) * env(n, 1.0, 1, 0.9, 0.6, int(COMPAS * SR))
            poner(pista['pad'], x, t0, 0.55 if sec != 'duerme' else 0.35 + 0.04 * k)

        # ---------------- coro
        if sec in ('sueno', 'despega', 'espacio', 'horizonte', 'climax'):
            voces = [raiz + 12 + tipo[0], raiz + 12 + tipo[1], raiz + 12 + tipo[2], raiz + 24 + tipo[0]]
            if sec == 'climax':
                voces.append(raiz + 24 + tipo[2])
            vocal = 'o' if sec in ('horizonte',) else 'a'
            g = {'sueno': 0.45 + 0.03 * k, 'despega': 0.4, 'espacio': 0.3, 'horizonte': 0.36, 'climax': 0.42}[sec]
            for m in voces:
                poner(pista['coro'], coro(m, COMPAS, vocal), t0, g / len(voces) * 2.2)

        # ---------------- arpegio espacial
        if sec in ('despega', 'espacio', 'climax', 'regreso'):
            tonos = [raiz + 24 + i for i in tipo] + [raiz + 36 + tipo[0]]
            orden = [0, 1, 2, 3, 2, 1, 0, 2, 1, 3, 2, 0, 3, 2, 1, 2]
            for p in range(16):
                n = int(PASO * 1.4 * SR)
                x = sierra(hz(tonos[orden[p]]), n) * 0.6 + cuadrada(hz(tonos[orden[p]] + 0.07), n, 0.3) * 0.4
                if sec == 'despega':
                    abre = 500 + 3500 * (k + p / 16) / 8
                elif sec == 'regreso':
                    abre = 2500 - 500 * k
                else:
                    abre = 3200
                x = lp_barrido(x, abre * 2.2, abre * 0.6, 512) * env(n, 0.002, 0.09, 0.2, 0.03)
                poner(pista['arp'], x, P(p), 0.5)

        # ---------------- pad de sierras (ancho)
        if sec in ('espacio', 'climax', 'horizonte'):
            for i in tipo:
                m = raiz + 12 + i
                n = int((COMPAS + 0.6) * SR)
                x = lp(supersierra(m, n, 7, 0.22), 2200 if sec != 'horizonte' else 900)
                poner(pista['pad'], x * env(n, 0.3, 1, 0.85, 0.5, int(COMPAS * SR)), t0, 0.2)

        # ---------------- melodía
        if sec in ('espacio', 'climax'):
            for p, d, m in MELODIA[k % 8]:
                n = int((d * PASO + 0.25) * SR)
                x = supersierra(m, n, 5, 0.12, vib=0.18 if d >= 6 else 0.0)
                x = 0.75 * x + 0.35 * cuadrada(hz(m - 12), n, 0.35)
                x = lp(x, 4800 if sec == 'climax' else 3800) * env(n, 0.02, 0.4, 0.75, 0.18, int(d * PASO * SR))
                poner(pista['lead'], x, P(p), 0.6)
            if sec == 'climax':
                for p, d, m in CONTRA[k % 8]:
                    n = int((d * PASO + 0.3) * SR)
                    x = lp(supersierra(m, n, 5, 0.1, vib=0.15), 5000)
                    poner(pista['lead'], x * env(n, 0.15, 0.6, 0.8, 0.25, int(d * PASO * SR)), P(p), 0.25)

    # ================================================================= mezcla
    # Bombeo: pads, coros y arpegios se agachan con cada bombo en las partes electrónicas
    bombeo = np.ones(N)
    for c, (sec, k, ac, nc) in enumerate(tl):
        if sec in ('espacio', 'climax', 'despega'):
            for p in (0, 4, 8, 12):
                i = int((c * COMPAS + p * PASO) * SR)
                L = int(PASO * 3.2 * SR)
                curva = 1 - 0.55 * np.exp(-np.arange(L) / (0.09 * SR))
                bombeo[i:i + L] = np.minimum(bombeo[i:i + L], curva[:len(bombeo[i:i + L])])
    for k in ('pad', 'coro', 'arp', 'bajo'):
        pista[k] *= bombeo

    rev = reverb(pista['coro'] * 0.9 + pista['lead'] * 0.35 + pista['tambor'] * 0.4 + pista['fx'] * 0.3 +
                 pista['bat'] * 0.08 + pista['pad'] * 0.2, 3.6, 5500, 1)
    rev2 = reverb(pista['coro'] * 0.9 + pista['lead'] * 0.35 + pista['tambor'] * 0.4 + pista['fx'] * 0.3 +
                  pista['bat'] * 0.08 + pista['pad'] * 0.2, 3.6, 5500, 2)
    arpL, arpR = eco_pingpong(pista['arp'], 3, 0.45)
    leadL, leadR = eco_pingpong(pista['lead'], 3, 0.3)

    seco = (pista['bat'] * 0.55 + pista['tambor'] * 0.5 + pista['bajo'] * 0.55 + pista['sub'] * 0.6 +
            pista['coro'] * 0.5 + pista['pad'] * 0.35 + pista['lead'] * 0.42 + pista['fx'] * 0.5)
    L = seco + pista['arp'] * 0.16 + arpL * 0.14 + leadL * 0.12 + rev * 0.55
    R = seco + pista['arp'] * 0.16 + arpR * 0.14 + leadR * 0.12 + rev2 * 0.55

    # Horizonte de eventos: el último compás se frena como si el tiempo se detuviera
    ini = sum(nc for nombre, nc, _ in SECCIONES[:4])
    i, j = int((ini + 7) * COMPAS * SR), int((ini + 8) * COMPAS * SR)
    frenar(L, i, j); frenar(R, i, j)

    # loop sin costuras: la cola se suma al inicio
    fin = int(total * SR)
    for x in (L, R):
        x[:N - fin] += x[fin:].copy()
    st = np.stack([L[:fin], R[:fin]], axis=1)
    st = hp(st.T, 28).T                                  # limpia graves inaudibles
    st /= np.max(np.abs(st))
    st = np.tanh(st * 1.4) / np.tanh(1.4) * 0.89
    return st


if __name__ == '__main__':
    salida = sys.argv[1] if len(sys.argv) > 1 else 'horizonte.wav'
    audio = componer()
    wavfile.write(salida, SR, (audio * 32767).astype(np.int16))
    print('listo:', salida, f'{len(audio) / SR:.1f} s')
