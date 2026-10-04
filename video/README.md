# DonutHugoBet promo edit

"Gamble with Donut money or Hugo money" – 1080x1920 TikTok/Reels/Shorts edit (~22 s, 30 fps).
Everything is generated from code: Minecraft-style voxel scenes, voiceover, phonk beat and sound design.

## Build

```bash
pip install pillow numpy scipy piper-tts onnx
# English neural voice for the voiceover (Piper)
curl -L -o voice.tgz https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-en-us-ryan-high.tar.gz && tar xzf voice.tgz

mkdir -p build
python3 tts.py en-us-ryan-high.onnx build/voice.wav build/timings.json   # voice + word timings
python3 audio.py                                                         # beat + sfx + voice -> build/mix.wav
python3 make_video.py build/donuthugobet_promo.mp4                       # frames + encode
```

`FRAMES=0,120,300 python3 make_video.py` renders single stills for preview.

## v2 – motion-design cut

Cleaner ad style: kinetic typography (Unbounded + Inter), aurora background, glass UI cards,
phone mockup with coinflip/crash screens, perspective panels with the voxel scenes, motion blur.

```bash
mkdir -p build_v2
SCRIPT=v2 python3 tts.py en-us-ryan-high.onnx build_v2/voice.wav build_v2/timings.json
python3 audio_v2.py
python3 make_video_v2.py build_v2/donuthugobet_v2.mp4      # SUBFRAMES=1 for a fast draft
```

## v3 – product-ad cut (current)

Apple/Shopify-style: black background with one brand glow, 3D phone (lockscreen -> app),
floating UI cards with rack focus, live bets panel, instant withdrawal, wordmark end card.
No voiceover – music + UI sound design only.

```bash
python3 audio_v3.py
python3 make_video_v3.py build_v3/donuthugobet_v3.mp4
```

Drop the real logo at `video/assets/logo.png` (transparent PNG) and it replaces the text wordmark.

## Files

- `tts.py` – script/voiceover text (edit `SCRIPT` to change what is said; captions follow automatically)
- `timeline.py` – scene cut points derived from the word timings
- `audio.py` – procedural phonk beat, risers, impacts, ka-ching etc., ducked under the voice
- `voxel.py`, `worlds.py` – small numpy voxel raycaster + farm / donut / Hugo coin / tunnel worlds
- `make_video.py` – v1 scenes, captions, logo, effects and ffmpeg encode
- `make_video_v2.py`, `audio_v2.py`, `timeline_v2.py` – v2 motion-design cut
- `synth.py` – shared instruments / sound effects
