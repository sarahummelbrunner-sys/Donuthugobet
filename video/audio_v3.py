"""v3 soundtrack: clean modern tech-ad track (no voiceover) + UI sound design."""
import os, wave
import numpy as np
from scipy import signal

from synth import SR, add, filt, sos, env_exp, T, kick, clap, hat, impact, whoosh, riser, tick

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get("BUILD_DIR", os.path.join(HERE, "build_v3"))
from timeline_v3 import EV, DUR

N = int(DUR * SR)
rng = np.random.default_rng(3)
BPM = 112
B = 60 / BPM
BAR = 4 * B

def midi(m): return 440.0 * 2 ** ((m - 69) / 12)

CHORDS = [  # (bass, pad voicing) Am9 | Fmaj9 | Cadd9 | G6
    (45, [60, 64, 67, 71]),
    (41, [57, 60, 64, 67]),
    (48, [64, 67, 72, 74]),
    (43, [59, 62, 64, 67]),
]

def saw_bl(f, n, harmonics=10, phase=0.0):
    t = T(n)
    x = np.zeros(n)
    for k in range(1, harmonics + 1):
        if f * k > 12000:
            break
        x += np.sin(2 * np.pi * f * k * t + phase * k) / k
    return x

def pad_note(f, dur):
    n = int(dur * SR)
    x = np.zeros((2, n))
    for d, pan in ((-0.09, -0.8), (0.0, 0.0), (0.08, 0.8), (-0.04, 0.4), (0.05, -0.4)):
        v = saw_bl(f * 2 ** (d / 12), n, 8, rng.uniform(0, 6.28))
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        x[0] += v * l; x[1] += v * r
    x = signal.sosfilt(sos("lowpass", 1800), x, axis=1)
    a = int(0.35 * SR); rl = int(0.5 * SR)
    e = np.ones(n); e[:a] = np.linspace(0, 1, a) ** 2; e[-rl:] *= np.linspace(1, 0, rl) ** 2
    return x * e * 0.08

def pluck(f, dur=0.9):
    n = int(dur * SR)
    p = int(SR / f)
    buf = rng.uniform(-1, 1, p)
    out = np.zeros(n)
    for i in range(n):
        out[i] = buf[i % p]
        buf[i % p] = 0.5 * (buf[i % p] + buf[(i + 1) % p]) * 0.996
    out = filt(out, "highpass", 200)
    return out * env_exp(n, 0.35) * 0.5

_PL = {}
def pluck_c(m):
    if m not in _PL:
        _PL[m] = pluck(midi(m))
    return _PL[m]

def bass(f, dur):
    n = int(dur * SR); t = T(n)
    x = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
    e = np.minimum(1, t / 0.01) * np.exp(-t / (dur * 1.2))
    e[-400:] *= np.linspace(1, 0, 400)
    return np.tanh(x * e * 1.4) * 0.55

def soft_kick():
    n = int(0.5 * SR); t = T(n)
    f = 48 + 90 * np.exp(-t / 0.035)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(n, 0.22)
    return np.tanh(x * 1.8) * 0.85

def shaker():
    n = int(0.06 * SR)
    x = filt(rng.normal(0, 1, n), "bandpass", [5000, 12000]) * env_exp(n, 0.018)
    return x * 0.35

def chime(freqs, dur=1.4):
    n = int(dur * SR); t = T(n)
    x = sum(np.sin(2 * np.pi * f * t) * np.exp(-t / (0.5 - 0.06 * i)) / (1 + 0.4 * i) for i, f in enumerate(freqs))
    return x * np.minimum(1, t / 0.004) * 0.35

def ui_click():
    n = int(0.05 * SR); t = T(n)
    return np.sin(2 * np.pi * 1800 * t) * env_exp(n, 0.008) * 0.5 + filt(rng.normal(0, 1, n), "highpass", 4000) * env_exp(n, 0.003) * 0.3

def reverb(x, secs=2.6, wet=0.32):
    n = int(secs * SR)
    ir = np.zeros((2, n))
    for c in range(2):
        ir[c] = rng.normal(0, 1, n) * np.exp(-np.arange(n) / (SR * secs / 6.5))
        ir[c] = filt(ir[c], "lowpass", 6000)
    ir /= np.abs(ir).sum(axis=1, keepdims=True) ** 0.5 * 18
    y = np.stack([signal.fftconvolve(x[c], ir[c])[:x.shape[1]] for c in range(2)])
    return x * (1 - wet) + y * wet * 3.2

def main():
    pads = np.zeros((2, N)); plk = np.zeros((2, N)); bs = np.zeros((2, N)); drm = np.zeros((2, N)); sfx = np.zeros((2, N))
    t0 = EV["beat"] - 2 * BAR          # grid so a bar starts exactly when the beat drops
    while t0 > 0:
        t0 -= BAR
    k = 0; tb = t0
    while tb < DUR:
        root, voicing = CHORDS[k % 4]
        if tb + BAR > 0:
            for m in voicing:
                pn = pad_note(midi(m), BAR + 0.5)
                i = int(tb * SR)
                if i < N:
                    j0 = max(0, -i); i = max(0, i)
                    seg = pn[:, j0:j0 + N - i]
                    pads[:, i:i + seg.shape[1]] += seg
            # pluck arpeggio (8ths) from chord tones an octave up
            arp = [voicing[0] + 12, voicing[2] + 12, voicing[1] + 12, voicing[3] + 12, voicing[2] + 12, voicing[0] + 12, voicing[3], voicing[1] + 12]
            for s, m in enumerate(arp):
                ts = tb + s * B / 2
                g = 0.22 if (EV["beat"] <= ts < EV["outro"]) else 0.13
                add(plk, pluck_c(m), ts, g, pan=0.35 if s % 2 else -0.35)
            if EV["beat"] <= tb < EV["outro"]:
                for s in range(4):
                    add(bs, bass(midi(root), B * 0.9), tb + s * B, 0.9)
                for s in (0, 2):
                    add(drm, soft_kick(), tb + s * B, 1.0)
                add(drm, soft_kick(), tb + 2.5 * B, 0.6)
                for s in (1, 3):
                    add(drm, clap(), tb + s * B, 0.28)
                for s in range(16):
                    add(drm, shaker(), tb + s * B / 4, 0.55 if s % 2 else 0.3, pan=0.3)
        tb += BAR; k += 1

    # sidechain pump from the kick grid
    sc = np.ones(N)
    tb = t0
    while tb < DUR:
        for s in (0, 2):
            ts = tb + s * B
            if EV["beat"] <= ts < EV["outro"]:
                i = int(ts * SR); n = int(0.28 * SR)
                if 0 <= i < N:
                    seg = 1 - 0.55 * np.exp(-np.arange(min(n, N - i)) / (0.07 * SR))
                    sc[i:i + len(seg)] = np.minimum(sc[i:i + len(seg)], seg)
        tb += BAR
    # intro: open the pad filter over the first seconds; outro: close it
    lp_env = np.clip(np.arange(N) / (SR * EV["beat"]), 0.15, 1.0)
    pads_f = signal.sosfilt(sos("lowpass", 700), pads, axis=1)
    pads = pads_f * (1 - lp_env) + pads * lp_env
    music = pads * sc + plk * sc * 0.9 + bs * sc + drm
    fi = int(0.8 * SR); music[:, :fi] *= np.linspace(0, 1, fi) ** 2

    # sound design
    add(sfx, riser(EV["beat"] - 1.6), 1.6, 0.25)
    add(sfx, chime([1318.5, 1975.5]), EV["notif"], 0.5)
    add(sfx, impact(), EV["beat"], 0.45)
    for key in ("app", "cards", "live", "wallet"):
        add(sfx, whoosh(0.45), EV[key] - 0.22, 0.25, pan=rng.uniform(-0.3, 0.3))
    add(sfx, ui_click(), EV["toggle"], 0.5)
    add(sfx, chime([987.8, 1318.5, 1975.5]), EV["toggle"] + 0.05, 0.22)
    for i in range(5):
        add(sfx, ui_click(), EV["cards"] + 0.6 + i * 0.09, 0.12)
    add(sfx, chime([1568, 2093]), EV["cards"] + 1.45, 0.3)
    tt = EV["focus"] + 0.2
    while tt < EV["focus"] + 1.9:
        add(sfx, tick(), tt, 0.12, pan=rng.uniform(-0.4, 0.4)); tt += 0.07
    add(sfx, chime([1318.5, 1760, 2637]), EV["focus"] + 1.95, 0.35)
    for i in range(6):
        add(sfx, ui_click(), EV["live"] + 0.4 + i * 0.42, 0.18, pan=rng.uniform(-0.4, 0.4))
    add(sfx, ui_click(), EV["wallet"] + 0.9, 0.5)
    add(sfx, chime([1046.5, 1568, 2093, 2637], 2.0), EV["wallet"] + 1.75, 0.4)
    add(sfx, whoosh(0.6), EV["outro"] - 0.3, 0.2)
    add(sfx, riser(EV["logo"] - EV["outro"]), EV["outro"], 0.18)
    add(sfx, impact(), EV["logo"], 0.4)
    add(sfx, chime([659.3, 987.8, 1318.5, 1975.5], 3.0), EV["logo"], 0.35)

    mix = reverb(music * 0.9 + sfx * 0.8, 2.6, 0.28)
    fo = int(2.0 * SR); mix[:, N - fo:] *= np.linspace(1, 0, fo) ** 1.5
    mix = np.tanh(mix * 1.6)
    mix /= np.abs(mix).max() / 0.93
    os.makedirs(BUILD, exist_ok=True)
    with wave.open(os.path.join(BUILD, "mix.wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix.T * 32767).astype(np.int16).tobytes())
    print("audio v3 ok", DUR)

if __name__ == "__main__":
    main()
