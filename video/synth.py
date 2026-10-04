"""Procedural instruments and sound effects shared by the audio mixes."""
import numpy as np
from scipy import signal

SR = 44100
rng = np.random.default_rng(42)

def T(n): return np.arange(n) / SR
def env_exp(n, tau): return np.exp(-T(n) / tau)
def sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output="sos")
def filt(x, kind, f, order=2): return signal.sosfilt(sos(kind, f, order), x)

def add(buf, x, t, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i >= buf.shape[1] or i + len(x) <= 0:
        return
    j0 = max(0, -i); i = max(0, i)
    x = x[j0:j0 + buf.shape[1] - i]
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buf[0, i:i + len(x)] += x * gain * l * 1.414
    buf[1, i:i + len(x)] += x * gain * r * 1.414

# ------------------------------------------------------------- instruments
def kick():
    n = int(0.45 * SR); t = T(n)
    f = 45 + 110 * np.exp(-t / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * env_exp(n, 0.16)
    x[:200] += rng.normal(0, 0.6, 200) * np.linspace(1, 0, 200)
    return np.tanh(x * 2.2) * 0.9

def clap():
    n = int(0.25 * SR)
    x = rng.normal(0, 1, n)
    e = np.zeros(n)
    for d in (0, 0.009, 0.018):
        k = int(d * SR); e[k:] += env_exp(n - k, 0.012 if d < 0.018 else 0.07)
    x = filt(x * e, "bandpass", [900, 4500])
    return x * 1.6

def hat(open_=False):
    n = int((0.18 if open_ else 0.05) * SR)
    x = filt(rng.normal(0, 1, n), "highpass", 7000, 4) * env_exp(n, 0.06 if open_ else 0.012)
    return x * 0.6

def bass808(freq, dur):
    n = int(dur * SR); t = T(n)
    f = freq * (1 + 1.0 * np.exp(-t / 0.02))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.clip(np.exp(-t / (dur * 0.7)), 0, 1)
    x *= np.minimum(1, (n - np.arange(n)) / 400)
    return np.tanh(x * 2.8) * 0.75

def cowbell(freq):
    n = int(0.22 * SR); t = T(n)
    x = signal.square(2 * np.pi * freq * t) * 0.5 + signal.square(2 * np.pi * freq * 1.48 * t) * 0.5
    x = filt(x, "bandpass", [freq * 0.8, freq * 4])
    x *= env_exp(n, 0.07)
    return np.tanh(x * 1.6) * 0.5

def impact():
    n = int(1.6 * SR); t = T(n)
    f = 30 + 70 * np.exp(-t / 0.08)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(n, 0.5)
    nz = filt(rng.normal(0, 1, n), "lowpass", 3000) * env_exp(n, 0.12)
    return np.tanh((sub * 1.3 + nz * 0.7) * 1.5)

def crash():
    n = int(2.0 * SR)
    x = filt(rng.normal(0, 1, n), "highpass", 4000) * env_exp(n, 0.6)
    return x * 0.35

def whoosh(dur=0.35, up=True):
    n = int(dur * SR)
    x = rng.normal(0, 1, n)
    out = np.zeros(n); seg = 256
    for k in range(0, n, seg):
        p = k / n
        fc = 400 + (5000 if up else 3000) * (p if up else 1 - p)
        out[k:k + seg] = filt(x[k:k + seg + 512], "bandpass", [fc * 0.6, fc * 1.6])[:len(out[k:k + seg])]
    e = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
    return out * e * 0.9

def riser(dur):
    n = int(dur * SR); t = T(n)
    p = t / dur
    x = rng.normal(0, 1, n)
    out = np.zeros(n); seg = 512
    for k in range(0, n, seg):
        fc = 300 + 7000 * (k / n) ** 2
        out[k:k + seg] = filt(x[k:k + seg + 512], "bandpass", [fc * 0.7, fc * 1.4])[:len(out[k:k + seg])]
    tone = np.sin(2 * np.pi * np.cumsum(200 + 1200 * p ** 2) / SR) * 0.25
    return (out * 0.8 + tone) * p ** 1.5

def scratch():
    n = int(0.35 * SR); t = T(n)
    f = 900 * (1 - t / 0.35) ** 2 + 60
    x = signal.sawtooth(2 * np.pi * np.cumsum(f) / SR) * 0.4 + rng.normal(0, 0.3, n)
    return filt(x, "bandpass", [200, 3000]) * env_exp(n, 0.2)

def kaching():
    n = int(1.0 * SR); t = T(n)
    x = sum(np.sin(2 * np.pi * f * t) * env_exp(n, d) * a for f, d, a in
            [(2093, 0.35, 0.5), (2637, 0.3, 0.4), (3136, 0.25, 0.3), (4186, 0.2, 0.2)])
    k = np.zeros(n); k[:int(0.05 * SR)] = filt(rng.normal(0, 1, int(0.05 * SR)), "bandpass", [1500, 6000])
    d = int(0.08 * SR)
    return np.concatenate([k[:d], x[:n - d]]) * 0.8 + k * 0.6

def ting(freq=1800):
    n = int(0.5 * SR); t = T(n)
    return (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * freq * 2.76 * t)) * env_exp(n, 0.12) * 0.5

def tick():
    n = int(0.015 * SR)
    return filt(rng.normal(0, 1, n), "bandpass", [2000, 8000]) * env_exp(n, 0.003)

