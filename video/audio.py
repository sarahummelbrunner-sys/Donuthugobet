"""Procedural phonk beat + sound design, mixed under the voiceover."""
import os, wave
import numpy as np
from scipy import signal
from timeline import EV, DUR, BUILD

SR = 44100
rng = np.random.default_rng(42)
N = int(DUR * SR)

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

# ------------------------------------------------------------- arrangement
def build_music():
    mus = np.zeros((2, N))
    bpm = 140; b = 60 / bpm
    drop = EV["drop"]
    t0 = drop - 16 * b            # grid start so a bar lands exactly on the drop
    K, C, Hc, Ho = kick(), clap(), hat(), hat(True)
    notes = {"A1": 55.0, "F1": 43.65, "G1": 49.0, "C2": 65.4, "E1": 41.2}
    prog = ["A1", "F1", "G1", "E1"]
    cow = [880, 880, 1046, 880, 784, 880, 1318, 1174, 880, 880, 1046, 1174, 1318, 1174, 1046, 784]

    def bar_pattern(tb, bar_idx, full=True):
        # kick pattern (phonk-ish)
        for s in (0, 3, 6, 10, 11) if full else (0, 6, 10):
            add(mus, K, tb + s * b / 4, 0.95)
        for s in (4, 12):
            add(mus, C, tb + s * b / 4, 0.55)
        for s in range(16):
            if full and bar_idx % 2 == 1 and s >= 12:
                for r in range(3):
                    add(mus, Hc, tb + s * b / 4 + r * b / 12, 0.22, pan=0.3)
            else:
                add(mus, Hc, tb + s * b / 4, 0.28 if s % 2 == 0 else 0.18, pan=0.25)
        add(mus, Ho, tb + 14 * b / 4, 0.18, pan=-0.3)
        note = notes[prog[bar_idx % 4]]
        add(mus, bass808(note, b * 2.4), tb, 0.85)
        add(mus, bass808(note, b * 1.4), tb + 10 * b / 4, 0.7)
        if full:
            for s, f in enumerate(cow):
                add(mus, cowbell(f * (1 if prog[bar_idx % 4] == "A1" else 0.94)), tb + s * b / 4, 0.32, pan=-0.2 if s % 2 else 0.2)

    # intro groove (muffled) up to "WRONG"
    intro = np.zeros((2, N)); save = mus
    mus = intro
    bar = 4 * b; tb = t0; i = 0
    while tb < EV["wrong"]:
        bar_pattern(tb, i, full=False); tb += bar; i += 1
    intro[:, int(EV["wrong"] * SR):] = 0
    intro = signal.sosfilt(sos("lowpass", 700), intro, axis=1) * 0.85
    mus = save + intro

    # main section from drop until "stop", and final section after "final"
    def section(a, z, start_bar=0):
        nonlocal mus
        tmp = np.zeros((2, N)); keep = mus; mus = tmp
        tb = a; k = start_bar
        while tb < z:
            bar_pattern(tb, k); tb += bar; k += 1
        tmp[:, int(z * SR):] = 0
        fade = int(0.03 * SR)
        tmp[:, int(z * SR) - fade:int(z * SR)] *= np.linspace(1, 0, fade)
        mus = keep + tmp
    section(drop, EV["stop"] + 0.18)
    # tape-stop on "so stop grinding": pitch the last 0.45s down
    a = int((EV["stop"] - 0.27) * SR); z = int((EV["stop"] + 0.18) * SR)
    seg = mus[:, a:z].copy(); n = z - a
    pos = np.cumsum(np.linspace(1, 0.05, n)); pos = pos / pos[-1] * (n - 1) * 0.55
    for c in range(2):
        mus[c, a:z] = np.interp(pos, np.arange(n), seg[c]) * np.linspace(1, 0.2, n)
    section(EV["final"], DUR - 0.2)
    fo = int(1.4 * SR)
    mus[:, N - fo:] *= np.linspace(1, 0, fo) ** 1.5
    return mus

def build_sfx():
    s = np.zeros((2, N))
    add(s, impact(), 0.0, 0.5)
    add(s, scratch(), EV["wrong"] - 0.12, 0.55)
    add(s, impact(), EV["wrong"], 0.9)
    add(s, riser(EV["drop"] - EV["secret"]), EV["secret"], 0.55)
    add(s, impact(), EV["drop"], 1.0); add(s, crash(), EV["drop"], 0.8)
    for k in ("donut", "hugo", "coin", "multi", "balance", "instant", "goto"):
        add(s, whoosh(0.3), EV[k] - 0.15, 0.45, pan=rng.uniform(-0.4, 0.4))
    add(s, whoosh(0.25, up=False), EV["you_are"] - 0.1, 0.35)
    add(s, whoosh(0.3), EV["instant2"] - 0.15, 0.35)
    for k in range(6):
        add(s, ting(1700 + 300 * (k % 2)), EV["coin"] + 0.08 + k * 0.1, 0.15)
    add(s, kaching(), EV["multi"] - 0.12, 0.45)
    t = EV["balance"] + 0.1
    while t < EV["explode"]:
        add(s, tick(), t, 0.35, pan=rng.uniform(-0.5, 0.5)); t += 0.045
    add(s, impact(), EV["explode"], 0.9); add(s, kaching(), EV["explode"] + 0.05, 0.6)
    add(s, ting(2400), EV["instant"] + 0.25, 0.25); add(s, ting(2400), EV["instant2"] + 0.25, 0.25)
    add(s, impact(), EV["stop_word"], 0.5)
    add(s, riser(EV["final"] - EV["goto"] + 0.1), EV["goto"] - 0.1, 0.6)
    add(s, impact(), EV["final"], 1.0); add(s, crash(), EV["final"], 0.8)
    for k in range(16):
        add(s, tick(), EV["final"] + 0.35 + k * 0.055, 0.25)
    add(s, whoosh(0.3), EV["tagline"] - 0.12, 0.35)
    add(s, kaching(), EV["tagline"] + 0.55, 0.35)
    return s

def load_voice():
    with wave.open(os.path.join(BUILD, "voice.wav")) as w:
        sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768.0
    x = signal.resample_poly(x, SR, sr)
    x = filt(x, "highpass", 90)
    pres = filt(x, "bandpass", [2500, 6000])
    x = x + 0.35 * pres
    x = np.tanh(x * 1.8) / np.tanh(1.8)
    v = np.zeros(N); v[:min(N, len(x))] = x[:N]
    return v

def main():
    v = load_voice()
    mus = build_music(); sfx = build_sfx()
    env = np.abs(v)
    env = signal.sosfilt(sos("lowpass", 8), env)
    env = env / (env.max() + 1e-9)
    duck = 1 - 0.55 * np.clip(env * 3, 0, 1)
    mix = mus * 0.42 * duck[None] + sfx * 0.55 + v[None] * 0.95
    mix = np.tanh(mix * 1.15)
    mix /= np.abs(mix).max() / 0.95
    out = (mix.T * 32767).astype(np.int16)
    with wave.open(os.path.join(BUILD, "mix.wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(out.tobytes())
    print("audio ok", DUR)

if __name__ == "__main__":
    main()
