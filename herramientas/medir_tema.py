"""Mide un tema por secciones: energía total, cuánto pesa el grave y cuánto la percusión.

Uso:  python medir_tema.py tema.wav
"""
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
import componer_horizonte as h

sr, x = wavfile.read(sys.argv[1])
m = x.astype(float).mean(1) / 32767
graves = sosfilt(butter(4, 120, 'low', fs=sr, output='sos'), m)            # bajo y bombo
cuerpo = sosfilt(butter(4, [100, 400], 'band', fs=sr, output='sos'), m)  # cuerpo del bajo
golpes = sosfilt(butter(4, [2000, 12000], 'band', fs=sr, output='sos'), m)  # palmas, cajas, platillos
rms = lambda s: np.sqrt((s ** 2).mean())
print(f"{'sección':10s} energía  graves%  cuerpo%  agudos%")
for i, (n, k, _, nc, _, _) in enumerate(h.TL):
    if k == 0:
        a = int(h.T[i * 16] * sr)
    if k == nc - 1:
        b = int(h.T[(i + 1) * 16] * sr)
        e = rms(m[a:b])
        print(f'{n:10s} {e:6.2f}  {100 * rms(graves[a:b]) / e:6.0f}  {100 * rms(cuerpo[a:b]) / e:6.0f}  {100 * rms(golpes[a:b]) / e:6.0f}')
