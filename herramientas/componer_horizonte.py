"""«Horizonte» v2: tema original de Kuriri aburrido en el espacio (Re menor, tempo que acelera).

Referencias de estilo (no de notas): la electrónica cinemática de Jazz Jackrabbit y el sonido
italo-disco de Giorgio Moroder: bajo secuenciado en semicorcheas saltando de octava, bombo en cada
tiempo, platillos abiertos a contratiempo, tambores electrónicos «piuu» (syndrums) y flanger.

El mareo de caer al agujero negro:
  - acelerando continuo (104 → 128 BPM), frenazo en el horizonte (dilatación del tiempo) y clímax a 140;
  - arpegios que giran alrededor de la cabeza cada vez más rápido;
  - flanger en platillos y pads, y una afinación que ondula cada vez más.

Todo se sintetiza aquí (instrumentos base en componer_horizonte_v1.py). Sin grabaciones de terceros.
Uso:  python componer_horizonte.py salida.wav
"""
import sys
import numpy as np
from scipy.io import wavfile
from componer_horizonte_v1 import (SR, hz, sierra, cuadrada, supersierra, env, lp, hp, bp, resonante,
                                   lp_barrido, coro, tambor_grave, bombo909, palmas, caja_grande, hat,
                                   shaker, tic, platillo_largo, caida_grave, subida, reverb, frenar, rng)

# ================================================================= forma, armonía y tempo
ACORDES = {'Dm': (50, [0, 3, 7]), 'Bb': (46, [0, 4, 7]), 'F': (53, [0, 4, 7]), 'C': (48, [0, 4, 7]),
           'Gm': (55, [0, 3, 7]), 'A': (57, [0, 4, 7])}
P1 = ['Dm', 'Bb', 'F', 'C']
P2 = ['Dm', 'Bb', 'Gm', 'A']
SECCIONES = [  # (nombre, compases, progresión, bpm al empezar, bpm al terminar)
    ('duerme', 8, P1 + P1, 104, 108),
    ('sueno', 8, P1 + P2, 108, 114),
    ('despega', 8, P1 + P2, 114, 120),
    ('espacio', 16, P1 + P2 + P1 + P2, 120, 128),
    ('horizonte', 8, ['Gm', 'Dm', 'Bb', 'A'] * 2, 128, 96),
    ('climax', 16, P1 + P2 + P1 + P2, 130, 140),
    ('regreso', 4, P1, 136, 112),
]
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
CONTRA = [[(0, 8, 86), (8, 8, 89)], [(0, 8, 89), (8, 8, 86)], [(0, 8, 84), (8, 8, 88)], [(0, 16, 84)],
          [(0, 8, 86), (8, 8, 89)], [(0, 8, 89), (8, 8, 94)], [(0, 8, 91), (8, 8, 94)], [(0, 16, 93)]]
# Secuencia propia del bajo (semitonos sobre la raíz, 16 pasos): octavas con un giro al final del compás.
SEC_A = [0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 10, 12]
SEC_B = [0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 7, 12, 3, 15, 7, 19]

TL = [(n, k, prog[k % len(prog)], nc, b0, b1) for n, nc, prog, b0, b1 in SECCIONES for k in range(nc)]
_bpm = np.concatenate([np.linspace(b0, b1, nc * 16, endpoint=False) for _, nc, _, b0, b1 in SECCIONES])
DUR = 60 / _bpm / 4                                    # duración de cada semicorchea (cambia con el tempo)
T = np.concatenate([[0], np.cumsum(DUR)])              # instante en que empieza cada semicorchea


def t_de(c, p):
    return T[c * 16 + p]


def paso(c, p=0):
    return DUR[min(len(DUR) - 1, c * 16 + p)]


def compas(c):
    return T[(c + 1) * 16] - T[c * 16]


# ================================================================= sonidos nuevos
def syndrum(f0=900, f1=110, dur=0.45):
    """Tambor electrónico de los 70: un «piuu» que cae de tono."""
    n = int(dur * SR); t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t * 9)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
    y += lp(rng.uniform(-1, 1, n), 3000) * np.exp(-t * 60) * 0.3
    return np.tanh(y * 1.5)


def laser(dur=0.3):
    n = int(dur * SR); t = np.arange(n) / SR
    f = 160 + 2600 * np.exp(-t * 14)
    return lp(cuadrada(f, n), 5000) * np.exp(-t * 8)


def flanger(x, ini, fin, prof=0.0025, base=0.0012, vel=0.25):
    """Efecto de «avión»: suma la señal con una copia retrasada que oscila."""
    seg = x[ini:fin]
    n = len(seg); t = np.arange(n) / SR
    d = (base + prof * (0.5 + 0.5 * np.sin(2 * np.pi * vel * t))) * SR
    y = np.interp(np.arange(n) - d, np.arange(n), seg, left=0)
    x[ini:fin] = 0.7 * seg + 0.6 * y


def ondular(x, prof_ms, vel):
    """Desafina suavemente (como una cinta que se estira): el mareo. prof_ms puede variar en el tiempo."""
    n = len(x); t = np.arange(n) / SR
    d = (np.asarray(prof_ms) / 1000) * SR * (1 + np.sin(2 * np.pi * vel * t)) + 2
    return np.interp(np.arange(n) - d, np.arange(n), x, left=0)


def girar(x, vel_hz):
    """Paneo circular: el sonido da vueltas alrededor de la cabeza; vel_hz puede crecer en el tiempo."""
    fase = 2 * np.pi * np.cumsum(np.broadcast_to(vel_hz, x.shape)) / SR
    return x * (0.5 + 0.45 * np.cos(fase)), x * (0.5 - 0.45 * np.cos(fase))


def poner(pista, x, t, g=1.0):
    i = int(t * SR)
    if i >= len(pista):
        return
    j = min(len(pista), i + len(x))
    pista[i:j] += x[:j - i] * g


def eco_pingpong(x, d_seg, retro=0.42):
    d = int(d_seg * SR)
    L, R = np.zeros_like(x), np.zeros_like(x)
    for k in range(1, 8):
        s = np.zeros_like(x); s[d * k:] = x[:-d * k] * retro ** k
        (L if k % 2 else R)[:] += s
    return lp(L, 3800), lp(R, 3800)


# ================================================================= composición
def componer():
    total = T[-1]
    N = int((total + 4.5) * SR)
    pista = {k: np.zeros(N) for k in ('bat', 'hats', 'syn', 'tambor', 'bajo', 'coro', 'pad', 'arp', 'lead', 'fx', 'sub')}
    K909, CLAP, CAJA, HAT, HATA, SHK = bombo909(), palmas(), caja_grande(), hat(), hat(True), shaker()
    TAMB, CRASH = tambor_grave(), platillo_largo()
    SYN_A, SYN_B, SYN_C = syndrum(1100, 140), syndrum(800, 100), syndrum(600, 70)

    for c, (sec, k, ac, nc, _, _) in enumerate(TL):
        t0, raiz, tipo = t_de(c, 0), *ACORDES[ac]
        ultimo = k == nc - 1
        P = lambda p: t_de(c, p)
        disco = sec in ('despega', 'espacio', 'climax')

        # ---------------- percusión
        if sec == 'duerme':
            if k < 6:
                for p in range(0, 16, 4):
                    poner(pista['fx'], tic(p % 8 == 0), P(p), 0.25 * (1 - k / 7))
            for p in (0, 8):
                poner(pista['bat'], lp(K909, 400), P(p), 0.35 + 0.05 * k)
            for p in range(2, 16, 4):
                poner(pista['hats'], SHK, P(p), 0.22 + 0.03 * k)
            if k >= 4:
                poner(pista['tambor'], TAMB, P(0), 0.45)
        elif sec == 'sueno':
            poner(pista['tambor'], TAMB, P(0), 0.45)
            for p in (0, 4, 8, 12):
                poner(pista['bat'], lp(K909, 900), P(p), 0.35 + 0.03 * k)
            for p in range(2, 16, 4):                                     # el «tss» a contratiempo
                poner(pista['hats'], HATA, P(p), 0.12 + 0.015 * k)
            if k % 2:
                poner(pista['syn'], SYN_B, P(14), 0.3)
        elif disco:
            fuerte = sec != 'despega'
            for p in (0, 4, 8, 12):
                poner(pista['bat'], K909, P(p), 0.95 if fuerte else 0.75)
            for p in (4, 12):
                poner(pista['bat'], CLAP, P(p), 1.0 if fuerte else 0.5)
                if fuerte:
                    poner(pista['bat'], CAJA, P(p), 0.45)
            for p in range(16):
                if p % 4 == 2:
                    poner(pista['hats'], HATA, P(p), 0.42)
                elif fuerte:
                    poner(pista['hats'], HAT, P(p), 0.18 if p % 2 else 0.28)
            # syndrums: «piuu» a contratiempo y una cascada cada 4 compases
            if fuerte:
                poner(pista['syn'], SYN_A if k % 2 else SYN_B, P(7), 0.35)
                if k % 4 == 3:
                    for p, s in ((10, SYN_A), (12, SYN_B), (13, SYN_B), (14, SYN_C), (15, SYN_C)):
                        poner(pista['syn'], s, P(p), 0.45)
                if k % 8 == 5:
                    poner(pista['fx'], laser(), P(6), 0.25)
            if sec == 'climax' and k % 2 == 0:
                poner(pista['tambor'], TAMB, P(0), 0.5)
            if k == 0 and fuerte:
                poner(pista['fx'], CRASH, P(0), 0.5)
                poner(pista['sub'], caida_grave(2.2), P(0), 0.8)
        elif sec == 'horizonte':
            if k % 2 == 0:
                poner(pista['tambor'], TAMB, P(0), 0.5)
            poner(pista['tambor'], TAMB, P(8), 0.2)
            if k in (3, 7):
                poner(pista['syn'], syndrum(500, 45, 1.2), P(12), 0.4)    # piuu lento y grave
        elif sec == 'regreso':
            for p in (0, 8):
                poner(pista['bat'], lp(K909, 500), P(p), 0.5 - 0.08 * k)
            for p in range(2, 16, 4):
                poner(pista['hats'], SHK, P(p), 0.3)
            poner(pista['fx'], tic(True), P(0), 0.12 * k)

        if ultimo and sec in ('despega', 'espacio', 'sueno'):
            for p in range(8, 16):
                poner(pista['bat'], CAJA, P(p), 0.15 + 0.06 * (p - 8))
        if sec in ('despega', 'horizonte') and k == nc - 4:
            dur_sub = T[(c + 4) * 16] - t0
            poner(pista['fx'], subida(dur_sub), t0, 0.35)

        # ---------------- bajo secuenciado (desde el sueño, con el filtro abriéndose)
        if sec in ('sueno', 'despega', 'espacio', 'climax', 'regreso'):
            secuencia = SEC_B if k % 4 == 3 else SEC_A
            if sec == 'sueno':
                corte = 260 + 60 * k
            elif sec == 'despega':
                corte = 700 + 220 * k
            elif sec == 'regreso':
                corte = 1400 - 300 * k
            else:
                corte = 2600 if sec == 'espacio' else 3200
            for p in range(16):
                m = raiz - 12 + secuencia[p]
                pl = paso(c, p)
                n = int(pl * 1.0 * SR)
                x = sierra(hz(m), n) * 0.6 + cuadrada(hz(m + 0.04), n, 0.45) * 0.4
                acento = 1.0 if p % 2 == 0 else 0.85
                x = lp_barrido(x, max(500, corte * acento), max(260, corte * 0.4), 256)
                x += resonante(x, max(300, corte * 0.5), 3) * 0.25            # «chirrido» del filtro
                x = np.tanh(x * 2.4) / np.tanh(2.4)                           # saturación: armónicos que se oyen
                x *= env(n, 0.002, 0.12, 0.72, 0.012)                         # notas más sostenidas
                cuerpo = np.sin(2 * np.pi * hz(m) * np.arange(n) / SR)        # peso en la fundamental
                x += cuerpo * env(n, 0.002, 0.12, 0.75, 0.012) * 0.7
                crece = {'sueno': 0.35 + 0.03 * k, 'despega': 0.6 + 0.04 * k, 'regreso': 0.5 - 0.08 * k}.get(sec, 1.0)
                poner(pista['bajo'], x, P(p), 0.9 * acento * crece)
        elif sec == 'horizonte':
            n = int(compas(c) * SR)
            x = np.sin(2 * np.pi * hz(raiz - 24) * np.arange(n) / SR) * env(n, 0.3, 1, 0.9, 0.4)
            poner(pista['sub'], x, t0, 0.75)
        if sec == 'duerme':
            n = int((compas(c) + 0.5) * SR)
            x = lp(supersierra(raiz - 12, n, 3, 0.1), 220) * env(n, 1.0, 1, 0.9, 0.6, int(compas(c) * SR))
            poner(pista['pad'], x, t0, 0.35 + 0.04 * k)

        # ---------------- coro
        if sec in ('sueno', 'despega', 'espacio', 'horizonte', 'climax'):
            voces = [raiz + 12 + tipo[0], raiz + 12 + tipo[1], raiz + 12 + tipo[2], raiz + 24 + tipo[0]]
            if sec == 'climax':
                voces.append(raiz + 24 + tipo[2])
            g = {'sueno': 0.26 + 0.02 * k, 'despega': 0.32, 'espacio': 0.28, 'horizonte': 0.62, 'climax': 0.4}[sec]
            for m in voces:
                poner(pista['coro'], coro(m, compas(c), 'o' if sec == 'horizonte' else 'a'), t0, g / len(voces) * 2.2)

        # ---------------- arpegio que gira
        if sec in ('despega', 'espacio', 'climax'):
            tonos = [raiz + 24 + i for i in tipo] + [raiz + 36 + tipo[0]]
            orden = [0, 1, 2, 3, 2, 1, 0, 2, 1, 3, 2, 0, 3, 2, 1, 2]
            for p in range(16):
                n = int(paso(c, p) * 1.4 * SR)
                m = tonos[orden[p]]
                x = sierra(hz(m), n) * 0.6 + cuadrada(hz(m + 0.07), n, 0.3) * 0.4
                abre = 500 + 3500 * (k + p / 16) / 8 if sec == 'despega' else 3200
                x = lp_barrido(x, abre * 2.2, abre * 0.6, 512) * env(n, 0.002, 0.09, 0.2, 0.03)
                poner(pista['arp'], x, P(p), 0.5)

        # ---------------- cuerdas sintéticas (pad ancho)
        if sec in ('espacio', 'climax', 'horizonte', 'despega'):
            for i in tipo:
                n = int((compas(c) + 0.6) * SR)
                x = lp(supersierra(raiz + 12 + i, n, 7, 0.22), 900 if sec == 'horizonte' else 2400)
                poner(pista['pad'], x * env(n, 0.3, 1, 0.85, 0.5, int(compas(c) * SR)), t0,
                      0.12 if sec == 'despega' else 0.2)

        # ---------------- melodía
        if sec in ('espacio', 'climax'):
            for p, d, m in MELODIA[k % 8]:
                dur = T[c * 16 + p + d] - P(p) if c * 16 + p + d < len(T) else d * paso(c, p)
                n = int((dur + 0.25) * SR)
                x = supersierra(m, n, 5, 0.12, vib=0.18 if d >= 6 else 0.0)
                x = 0.75 * x + 0.35 * cuadrada(hz(m - 12), n, 0.35)
                x = lp(x, 4800 if sec == 'climax' else 3800) * env(n, 0.02, 0.4, 0.75, 0.18, int(dur * SR))
                poner(pista['lead'], x, P(p), 0.6)
            if sec == 'climax':
                for p, d, m in CONTRA[k % 8]:
                    dur = T[min(len(T) - 1, c * 16 + p + d)] - P(p)
                    n = int((dur + 0.3) * SR)
                    x = lp(supersierra(m, n, 5, 0.1, vib=0.15), 5000)
                    poner(pista['lead'], x * env(n, 0.15, 0.6, 0.8, 0.25, int(dur * SR)), P(p), 0.25)

    # ================================================================= efectos de mareo
    i_seccion = {}
    for c, (sec, *_r) in enumerate(TL):
        i_seccion.setdefault(sec, c)
    s_ = lambda nombre: int(t_de(i_seccion[nombre], 0) * SR)
    # flanger en platillos y pads en las partes que suben
    for a, b in (('despega', 'espacio'), ('climax', 'regreso')):
        flanger(pista['hats'], s_(a), s_(b), vel=0.18)
        flanger(pista['pad'], s_(a), s_(b), prof=0.003, vel=0.12)
    # la afinación ondula cada vez más (de 0 a ~1,6 ms de vaivén) en pads y coros
    progreso = np.clip(np.arange(N) / (T[-1] * SR), 0, 1)
    pista['pad'] = ondular(pista['pad'], 0.2 + 1.4 * progreso, 0.35)
    pista['coro'] = ondular(pista['coro'], 0.1 + 0.9 * progreso, 0.27)

    # bombeo con el bombo en las partes disco
    bombeo = np.ones(N)
    for c, (sec, *_r) in enumerate(TL):
        if sec in ('espacio', 'climax', 'despega'):
            for p in (0, 4, 8, 12):
                i = int(t_de(c, p) * SR); L = int(paso(c, p) * 3.2 * SR)
                curva = 1 - 0.55 * np.exp(-np.arange(L) / (0.09 * SR))
                bombeo[i:i + L] = np.minimum(bombeo[i:i + L], curva[:len(bombeo[i:i + L])])
    for k in ('pad', 'coro', 'arp'):
        pista[k] *= bombeo
    pista['bajo'] *= 0.75 + 0.25 * bombeo

    # arpegio girando alrededor de la cabeza, cada vez más rápido (0,15 → 1,2 vueltas por segundo)
    arpL, arpR = girar(pista['arp'], 0.15 + 1.05 * progreso)
    synL, synR = girar(pista['syn'], 0.5)
    pe = 3 * 0.125
    ecoL, ecoR = eco_pingpong(pista['arp'] + pista['syn'] * 0.5, pe, 0.45)
    leadL, leadR = eco_pingpong(pista['lead'], pe, 0.3)
    envio = (pista['coro'] * 0.9 + pista['lead'] * 0.35 + pista['tambor'] * 0.4 + pista['fx'] * 0.3 +
             pista['bat'] * 0.06 + pista['pad'] * 0.2 + pista['syn'] * 0.35)
    revL, revR = reverb(envio, 3.6, 5500, 1), reverb(envio, 3.6, 5500, 2)

    # batería con saturación: golpea más sin subir los picos
    bateria = np.tanh(pista['bat'] * 1.6) / np.tanh(1.6)
    seco = (bateria * 0.95 + pista['hats'] * 1.4 + pista['tambor'] * 0.6 + pista['bajo'] * 0.95 +
            pista['sub'] * 0.6 + pista['coro'] * 0.45 + pista['pad'] * 0.28 + pista['lead'] * 0.42 + pista['fx'] * 0.5)
    L = seco + arpL * 0.3 + synL * 0.8 + ecoL * 0.14 + leadL * 0.12 + revL * 0.55
    R = seco + arpR * 0.3 + synR * 0.8 + ecoR * 0.14 + leadR * 0.12 + revR * 0.55

    # horizonte de eventos: el último compás se frena hasta casi detenerse
    c_fin = i_seccion['horizonte'] + 7
    i, j = int(t_de(c_fin, 0) * SR), int(t_de(c_fin + 1, 0) * SR)
    frenar(L, i, j); frenar(R, i, j)

    fin = int(total * SR)
    for x in (L, R):
        x[:N - fin] += x[fin:].copy()
    st = np.stack([L[:fin], R[:fin]], axis=1)
    st = hp(st.T, 28).T
    st /= np.max(np.abs(st))
    st = np.tanh(st * 1.4) / np.tanh(1.4) * 0.89
    return st


if __name__ == '__main__':
    salida = sys.argv[1] if len(sys.argv) > 1 else 'horizonte.wav'
    audio = componer()
    wavfile.write(salida, SR, (audio * 32767).astype(np.int16))
    print('listo:', salida, f'{len(audio) / SR:.1f} s | tempo {_bpm[0]:.0f} → {_bpm.max():.0f} BPM')
