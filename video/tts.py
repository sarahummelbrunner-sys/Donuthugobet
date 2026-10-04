"""Voiceover: synthesizes each script line with Piper and extracts per-word timings."""
import json, sys, wave
import numpy as np
from piper import PiperVoice, SynthesisConfig

VOICE = sys.argv[1]           # path to .onnx voice
OUT_WAV, OUT_JSON = sys.argv[2], sys.argv[3]

# Each line: list of (caption_text, spoken_text). Pause (s) after the line.
SCRIPT_V1 = [
    ([("if", "If"), ("you're", "you're"), ("still", "still"), ("grinding", "grinding"),
      ("money", "money"), ("on", "on"), ("DonutSMP", "Donut S M P"), ("like", "like"), ("this", "this")], 0.10),
    ([("you", "you"), ("are", "are"), ("WRONG", "WRONG!")], 0.35),
    ([("let", "Let"), ("me", "me"), ("show", "show"), ("you", "you"), ("the", "the"), ("secret", "secret.")], 0.15),
    ([("it's", "It's"), ("called", "called"), ("DONUTHUGOBET.NET", "donut hugo bet dot net.")], 0.30),
    ([("gamble", "Gamble"), ("with", "with"), ("your", "your"), ("DONUT", "Donut"), ("money", "money,")], 0.12),
    ([("or", "or"), ("your", "your"), ("HUGO", "Hugo"), ("money", "money.")], 0.25),
    ([("flip", "Flip"), ("a", "a"), ("coin", "coin,"), ("hit", "hit"), ("a", "a"), ("multiplier", "multiplier,")], 0.05),
    ([("and", "and"), ("watch", "watch"), ("your", "your"), ("balance", "balance"), ("EXPLODE", "explode.")], 0.25),
    ([("instant", "Instant"), ("deposits", "deposits."), ("instant", "Instant"), ("withdrawals", "withdrawals.")], 0.25),
    ([("so", "So"), ("stop", "stop"), ("grinding", "grinding,"), ("and", "and"), ("go", "go"), ("to", "to"),
      ("DONUTHUGOBET.NET", "donut hugo bet dot net!")], 0.0),
]

# v2: cleaner, ad-style read for the motion-design cut
SCRIPT_V2 = [
    ([("still", "Still"), ("grinding", "grinding"), ("money", "money"), ("on", "on"), ("DonutSMP", "Donut S M P?")], 0.40),
    ([("there's", "There's"), ("a", "a"), ("smarter", "smarter"), ("way", "way.")], 0.45),
    ([("meet", "Meet"), ("DONUTHUGOBET.NET", "donut hugo bet dot net.")], 0.55),
    ([("gamble", "Gamble"), ("with", "with"), ("your", "your"), ("DONUT", "Donut"), ("money", "money,")], 0.12),
    ([("or", "or"), ("your", "your"), ("HUGO", "Hugo"), ("money", "money.")], 0.45),
    ([("flip", "Flip"), ("a", "a"), ("coin", "coin.")], 0.18),
    ([("ride", "Ride"), ("the", "the"), ("multiplier", "multiplier.")], 0.40),
    ([("instant", "Instant"), ("deposits", "deposits.")], 0.12),
    ([("instant", "Instant"), ("withdrawals", "withdrawals.")], 0.45),
    ([("play", "Play"), ("now", "now"), ("at", "at"), ("DONUTHUGOBET.NET", "donut hugo bet dot net.")], 0.0),
]
import os
SCRIPT, LS = (SCRIPT_V2, 0.92) if os.environ.get("SCRIPT") == "v2" else (SCRIPT_V1, 0.86)

voice = PiperVoice.load(VOICE, include_alignments=True)
sr = voice.config.sample_rate
cfg = SynthesisConfig(length_scale=LS, noise_scale=0.6, noise_w_scale=0.7)

lead = 0.45  # silence before first word
audio = [np.zeros(int(lead * sr), np.float32)]
t = lead
lines_out = []
for words, pause in SCRIPT:
    text = " ".join(s for _, s in words)
    chunks = list(voice.synthesize(text, syn_config=cfg, include_alignments=True))
    a = np.concatenate([c.audio_float_array for c in chunks])
    # Word boundaries from phoneme alignments (space phonemes separate words).
    spans, cur, pos, started = [], None, 0, False
    for c in chunks:
        for al in c.phoneme_alignments or []:
            n = al.num_samples
            if al.phoneme in ("^", "$", "_"):
                pos += n; continue
            if al.phoneme == " " or al.phoneme in ",.!?;:":
                if cur is not None:
                    spans.append((cur, pos)); cur = None
                pos += n; continue
            if cur is None:
                cur = pos
            pos += n
    if cur is not None:
        spans.append((cur, pos))
    n_spoken = sum(len(s.split()) for _, s in words)
    if len(spans) != n_spoken:
        print(f"WARN alignment {len(spans)} vs {n_spoken} for: {text}", file=sys.stderr)
        # proportional fallback by character count
        tot = sum(len(s) for _, s in words); acc = 0; spans = []
        for _, s in words:
            for w in s.split():
                st = acc / tot * len(a); acc += len(w) + 1
                spans.append((int(st), int(acc / tot * len(a))))
    # Map spoken tokens -> caption words
    out, k = [], 0
    for cap, s in words:
        m = len(s.split())
        st, en = spans[k][0], spans[k + m - 1][1]
        out.append({"w": cap, "t0": round(t + st / sr, 3), "t1": round(t + en / sr, 3)})
        k += m
    lines_out.append(out)
    audio.append(a)
    t += len(a) / sr
    audio.append(np.zeros(int(pause * sr), np.float32))
    t += pause

audio.append(np.zeros(int(0.2 * sr), np.float32))
full = np.concatenate(audio)
with wave.open(OUT_WAV, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((np.clip(full, -1, 1) * 32000).astype(np.int16).tobytes())
json.dump({"sr": sr, "duration": len(full) / sr, "lines": lines_out}, open(OUT_JSON, "w"), indent=1)
print("duration", len(full) / sr)
