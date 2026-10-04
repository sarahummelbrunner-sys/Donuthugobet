"""Procedural phonk beat + sound design, mixed under the voiceover."""
import os, wave
import numpy as np
from scipy import signal
from timeline import EV, DUR, BUILD

from synth import *
from synth import rng

N = int(DUR * SR)

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
