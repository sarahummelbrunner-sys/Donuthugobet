"""v2 mix: cleaner trap/phonk bed with a drop on the logo reveal."""
import os, wave
import numpy as np
from scipy import signal
from timeline_v2 import EV, DUR, BUILD
from synth import *
from synth import rng

N = int(DUR * SR)

def main():
    mus = np.zeros((2, N))
    bpm = 140; b = 60 / bpm; bar = 4 * b
    drop = EV["drop"]
    t0 = drop - 12 * b
    K, C, Hc, Ho = kick(), clap(), hat(), hat(True)
    notes = {"A1": 55.0, "F1": 43.65, "G1": 49.0, "E1": 41.2}
    prog = ["A1", "F1", "G1", "E1"]
    cow = [880, 0, 1046, 880, 0, 880, 1318, 0, 1174, 0, 1046, 880, 1318, 1174, 1046, 784]

    def bar_pattern(buf, tb, k, full):
        for s in ((0, 3, 6, 10, 11) if full else (0, 10)):
            add(buf, K, tb + s * b / 4, 0.9)
        if full:
            for s in (4, 12):
                add(buf, C, tb + s * b / 4, 0.5)
        for s in range(16):
            if full and k % 2 == 1 and s >= 12:
                for r in range(3):
                    add(buf, Hc, tb + s * b / 4 + r * b / 12, 0.2, pan=0.3)
            elif full or s % 2 == 0:
                add(buf, Hc, tb + s * b / 4, 0.25 if s % 2 == 0 else 0.16, pan=0.25)
        note = notes[prog[k % 4]]
        add(buf, bass808(note, b * 2.4), tb, 0.8)
        add(buf, bass808(note, b * 1.4), tb + 10 * b / 4, 0.65)
        if full:
            for s, f in enumerate(cow):
                if f:
                    add(buf, cowbell(f * (1 if prog[k % 4] == "A1" else 0.94)), tb + s * b / 4, 0.26, pan=-0.2 if s % 2 else 0.2)

    # intro: filtered groove until the drop
    intro = np.zeros((2, N)); tb = t0 - 4 * bar; k = 0
    while tb < drop:
        if tb + bar > 0:
            bar_pattern(intro, tb, k, False)
        tb += bar; k += 1
    intro[:, int(drop * SR):] = 0
    intro = signal.sosfilt(sos("lowpass", 900), intro, axis=1)
    fi = int(0.4 * SR); intro[:, :fi] *= np.linspace(0, 1, fi)
    mus += intro * 0.9
    main_ = np.zeros((2, N)); tb = drop; k = 0
    while tb < DUR:
        bar_pattern(main_, tb, k, True); tb += bar; k += 1
    # short breakdown before the end card, then back in on the URL
    a, z = int((EV["final"] - 2 * b) * SR), int(EV["final"] * SR)
    main_[:, a:z] = signal.sosfilt(sos("lowpass", 500), main_[:, a:z], axis=1) * np.linspace(1, 0.3, z - a)
    mus += main_
    fo = int(1.6 * SR); mus[:, N - fo:] *= np.linspace(1, 0, fo) ** 1.6

    s = np.zeros((2, N))
    add(s, riser(drop - EV["smarter"]), EV["smarter"], 0.45)
    add(s, impact(), drop, 0.95); add(s, crash(), drop, 0.7)
    for key in ("smarter", "donut", "hugo", "coin", "multi", "instant", "end"):
        add(s, whoosh(0.32), EV[key] - 0.16, 0.38, pan=rng.uniform(-0.4, 0.4))
    for kk in range(3):
        add(s, tick(), 0.55 + kk * 0.33, 0.25)
    for kk in range(5):
        add(s, ting(1700 + 300 * (kk % 2)), EV["coin"] + 0.3 + kk * 0.09, 0.12)
    add(s, kaching(), EV["coin"] + 0.85, 0.35)
    t = EV["multi"] + 0.2
    while t < EV["instant"] - 0.25:
        add(s, tick(), t, 0.25, pan=rng.uniform(-0.5, 0.5)); t += 0.06
    add(s, kaching(), EV["instant"] - 0.3, 0.3)
    add(s, ting(2400), EV["instant"] + 0.75, 0.22); add(s, ting(2600), EV["instant2"] + 0.75, 0.22)
    add(s, riser(EV["final"] - EV["end"]), EV["end"], 0.35)
    add(s, impact(), EV["final"], 0.85); add(s, crash(), EV["final"], 0.6)
    for kk in range(16):
        add(s, tick(), EV["final"] + 0.45 + kk * 0.05, 0.18)
    add(s, whoosh(0.3), EV["tagline"] - 0.12, 0.3)

    with wave.open(os.path.join(BUILD, "voice.wav")) as w:
        sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768.0
    x = signal.resample_poly(x, SR, sr)
    x = filt(x, "highpass", 90); x = x + 0.3 * filt(x, "bandpass", [2500, 6000])
    x = np.tanh(x * 1.7) / np.tanh(1.7)
    v = np.zeros(N); v[:min(N, len(x))] = x[:N]
    env = signal.sosfilt(sos("lowpass", 8), np.abs(v)); env /= env.max() + 1e-9
    duck = 1 - 0.6 * np.clip(env * 3, 0, 1)
    mix = mus * 0.38 * duck[None] + s * 0.5 + v[None] * 0.95
    mix = np.tanh(mix * 1.1); mix /= np.abs(mix).max() / 0.95
    with wave.open(os.path.join(BUILD, "mix.wav"), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix.T * 32767).astype(np.int16).tobytes())
    print("audio v2 ok", DUR)

if __name__ == "__main__":
    main()
