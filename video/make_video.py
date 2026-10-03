"""DonutHugoBet promo edit: 1080x1920 TikTok-style video.

Pipeline: tts.py (voice + word timings) -> audio.py (beat + sfx mix) -> this file (frames + encode).
"""
import math, os, subprocess, sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from timeline import EV, DUR, L, BUILD
from voxel import render, look_at
import worlds

W, H, FPS = 1080, 1920, 30
HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.environ.get("FONT_DIR", os.path.join(HERE, "fonts"))

PINK = (255, 95, 170)
GOLD = (255, 196, 50)
GREEN = (60, 255, 120)
RED = (255, 40, 50)
WHITE = (255, 255, 255)
DARK = (18, 10, 31)

# ----------------------------------------------------------------- worlds
FARM, G_FARM = worlds.farm_world()
DONUT, DONUT_C = worlds.donut_world()
HUGO, HUGO_C = worlds.hugo_world()
TUNNEL = worlds.tunnel_world()

def sky_dark(d):
    return np.tile(np.array([8, 6, 18], np.float32), (len(d), 1))

# ----------------------------------------------------------------- helpers
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def lerp(a, b, k): return a + (b - a) * k
def ease_out(k): k = clamp(k); return 1 - (1 - k) ** 3
def ease_in(k): k = clamp(k); return k ** 3
def ease_io(k): k = clamp(k); return 3 * k * k - 2 * k * k * k
def ease_back(k, s=1.9):
    k = clamp(k) - 1
    return k * k * ((s + 1) * k + s) + 1

@lru_cache(None)
def font(size, weight="ExtraBold", fam="Montserrat"):
    if fam == "Montserrat":
        f = ImageFont.truetype(os.path.join(FONTS, "Montserrat.ttf"), size)
        f.set_variation_by_name(weight)
        return f
    return ImageFont.truetype(os.path.join(FONTS, fam), size)

def pixel_font(size): return font(size, fam="SilkscreenBold.ttf")

def text_rgba(text, size, fill=WHITE, stroke=10, stroke_fill=(0, 0, 0), weight="ExtraBold",
              shadow=True, glow=None, fnt=None, spacing=0):
    f = fnt or font(size, weight)
    pad = stroke + 40
    if spacing:
        widths = [f.getbbox(c)[2] - f.getbbox(c)[0] if c != " " else size * 0.3 for c in text]
        tw = int(sum(widths) + spacing * (len(text) - 1))
    else:
        bb = f.getbbox(text, stroke_width=stroke); tw = bb[2] - bb[0]
    asc, desc = f.getmetrics()
    th = asc + desc
    img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad + stroke), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    def draw(dd, col, sc):
        if spacing:
            x = pad
            for c, wdt in zip(text, widths):
                dd.text((x, pad), c, font=f, fill=col, stroke_width=stroke, stroke_fill=sc)
                x += wdt + spacing
        else:
            dd.text((pad - bb[0], pad), text, font=f, fill=col, stroke_width=stroke, stroke_fill=sc)
    if glow:
        gl = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw(ImageDraw.Draw(gl), glow + (255,), glow + (255,))
        gl = gl.filter(ImageFilter.GaussianBlur(18))
        img = Image.alpha_composite(img, gl)
    if shadow:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw(ImageDraw.Draw(sh), (0, 0, 0, 170), (0, 0, 0, 170))
        sh = sh.filter(ImageFilter.GaussianBlur(6))
        base = Image.new("RGBA", img.size, (0, 0, 0, 0))
        base.paste(sh, (0, 8), sh)
        img = Image.alpha_composite(img, base)
    d = ImageDraw.Draw(img)
    draw(d, fill + (255,) if len(fill) == 3 else fill, stroke_fill)
    return img

def paste_center(base, im, cx, cy, scale=1.0, alpha=1.0, rot=0.0):
    if scale <= 0.01 or alpha <= 0.01:
        return
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BICUBIC)
    if rot:
        im = im.rotate(rot, Image.BICUBIC, expand=True)
    if alpha < 1:
        a = im.getchannel("A").point(lambda v: int(v * alpha))
        im = im.copy(); im.putalpha(a)
    base.alpha_composite(im, (int(cx - im.width / 2), int(cy - im.height / 2))) if (
        0 <= int(cx - im.width / 2) and 0 <= int(cy - im.height / 2) and
        int(cx - im.width / 2) + im.width <= base.width and int(cy - im.height / 2) + im.height <= base.height) \
        else base.paste(im, (int(cx - im.width / 2), int(cy - im.height / 2)), im)

def rounded(size, radius, fill, outline=None, width=0):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius, fill=fill,
                                         outline=outline, width=width)
    return im

def glass_card(size, radius=48, accent=PINK):
    w, h = size; pad = 40
    im = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    glow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).rounded_rectangle([pad, pad, pad + w, pad + h], radius, outline=accent + (255,), width=10)
    glow = glow.filter(ImageFilter.GaussianBlur(16))
    im.alpha_composite(glow)
    ImageDraw.Draw(im).rounded_rectangle([pad, pad, pad + w, pad + h], radius, fill=(22, 12, 38, 228),
                                         outline=accent + (255,), width=4)
    hl = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(hl).rounded_rectangle([pad + 6, pad + 6, pad + w - 6, pad + h // 2], radius - 6, fill=(255, 255, 255, 14))
    im.alpha_composite(hl)
    return im

# ----------------------------------------------------------------- brand art
@lru_cache(None)
def donut_icon(size):
    S = size * 4
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = S / 2
    d.ellipse([c - 0.47 * S, c - 0.47 * S + 0.03 * S, c + 0.47 * S, c + 0.47 * S + 0.03 * S], fill=(150, 80, 30, 255))
    d.ellipse([c - 0.47 * S, c - 0.47 * S, c + 0.47 * S, c + 0.47 * S], fill=(222, 150, 75, 255))
    pts_o, pts_i = [], []
    for k in range(361):
        th = math.radians(k)
        ro = (0.40 + 0.025 * math.sin(7 * th) + 0.018 * max(0, math.sin(3 * th + 1)) ** 4) * S
        ri = (0.20 + 0.012 * math.sin(5 * th)) * S
        pts_o.append((c + ro * math.cos(th), c + ro * math.sin(th)))
        pts_i.append((c + ri * math.cos(th), c + ri * math.sin(th)))
    d.polygon(pts_o, fill=PINK + (255,))
    # glossy highlight
    hl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(hl).arc([c - 0.33 * S, c - 0.33 * S, c + 0.33 * S, c + 0.33 * S], 200, 290, fill=(255, 200, 230, 200), width=int(0.04 * S))
    im.alpha_composite(hl.filter(ImageFilter.GaussianBlur(S * 0.006)))
    r = np.random.default_rng(5)
    cols = [(255, 255, 255), (255, 230, 60), (90, 200, 255), (120, 240, 120), (255, 255, 255), (170, 110, 255)]
    for k in range(26):
        th = r.uniform(0, 2 * math.pi); rr = r.uniform(0.25, 0.37) * S
        x, y = c + rr * math.cos(th), c + rr * math.sin(th)
        a = r.uniform(0, math.pi); l = 0.035 * S; w = 0.012 * S
        dx, dy = math.cos(a) * l, math.sin(a) * l
        d.line([x - dx, y - dy, x + dx, y + dy], fill=cols[k % len(cols)] + (255,), width=int(w * 2))
    # hole
    mask = Image.new("L", (S, S), 255)
    ImageDraw.Draw(mask).ellipse([c - 0.17 * S, c - 0.17 * S, c + 0.17 * S, c + 0.17 * S], fill=0)
    d.ellipse([c - 0.205 * S, c - 0.205 * S, c + 0.205 * S, c + 0.205 * S], fill=(222, 150, 75, 255))
    d.ellipse([c - 0.185 * S, c - 0.185 * S + 0.012 * S, c + 0.185 * S, c + 0.185 * S + 0.012 * S], fill=(150, 80, 30, 255))
    im.putalpha(Image.fromarray(np.minimum(np.array(im.getchannel("A")), np.array(mask))))
    return im.resize((size, size), Image.LANCZOS)

@lru_cache(None)
def hugo_coin(size):
    S = size * 4
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = S / 2
    d.ellipse([c - 0.48 * S, c - 0.48 * S, c + 0.48 * S, c + 0.48 * S], fill=(170, 105, 10, 255))
    d.ellipse([c - 0.45 * S, c - 0.45 * S, c + 0.45 * S, c + 0.45 * S], fill=GOLD + (255,))
    d.ellipse([c - 0.37 * S, c - 0.37 * S, c + 0.37 * S, c + 0.37 * S], outline=(200, 130, 15, 255), width=int(0.02 * S))
    f = font(int(0.5 * S), "Black")
    bb = d.textbbox((0, 0), "H", font=f)
    d.text((c - (bb[0] + bb[2]) / 2, c - (bb[1] + bb[3]) / 2 + 0.01 * S), "H", font=f, fill=(200, 125, 10, 255))
    d.text((c - (bb[0] + bb[2]) / 2 - 0.012 * S, c - (bb[1] + bb[3]) / 2 - 0.004 * S), "H", font=f, fill=(255, 240, 160, 255))
    hl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(hl).arc([c - 0.41 * S, c - 0.41 * S, c + 0.41 * S, c + 0.41 * S], 190, 260, fill=(255, 255, 230, 200), width=int(0.03 * S))
    im.alpha_composite(hl.filter(ImageFilter.GaussianBlur(S * 0.006)))
    return im.resize((size, size), Image.LANCZOS)

@lru_cache(None)
def donut_coin(size):
    S = size
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([S * 0.02, S * 0.02, S * 0.98, S * 0.98], fill=(150, 30, 90, 255))
    d.ellipse([S * 0.05, S * 0.05, S * 0.95, S * 0.95], fill=(255, 140, 195, 255))
    ic = donut_icon(int(S * 0.72))
    im.alpha_composite(ic, (int(S * 0.14), int(S * 0.14)))
    return im

@lru_cache(None)
def wordmark():
    a = text_rgba("DONUT", 150, PINK, stroke=8, weight="Black", glow=(255, 60, 150))
    b = text_rgba("HUGO", 150, GOLD, stroke=8, weight="Black", glow=(255, 170, 0))
    top = Image.new("RGBA", (a.width + b.width - 80, max(a.height, b.height)), (0, 0, 0, 0))
    top.alpha_composite(a, (0, 0)); top.alpha_composite(b, (a.width - 80, 0))
    # BET pill
    f = font(128, "Black")
    bb = f.getbbox("BET.NET")
    pw, ph = bb[2] - bb[0] + 110, 190
    pill = Image.new("RGBA", (pw + 80, ph + 80), (0, 0, 0, 0))
    glow = Image.new("RGBA", pill.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).rounded_rectangle([40, 40, 40 + pw, 40 + ph], 95, fill=(255, 255, 255, 140))
    pill.alpha_composite(glow.filter(ImageFilter.GaussianBlur(20)))
    ImageDraw.Draw(pill).rounded_rectangle([40, 40, 40 + pw, 40 + ph], 95, fill=(255, 255, 255, 255))
    dd = ImageDraw.Draw(pill)
    x0 = 40 + 55 - bb[0]
    fb = font(128, "Black")
    dd.text((x0, 40 + ph / 2 - (bb[1] + bb[3]) / 2), "BET", font=fb, fill=DARK + (255,))
    wb = fb.getbbox("BET")[2]
    dd.text((x0 + wb, 40 + ph / 2 - (bb[1] + bb[3]) / 2), ".NET", font=fb, fill=(200, 60, 140, 255))
    out = Image.new("RGBA", (max(top.width, pill.width), top.height + pill.height - 70), (0, 0, 0, 0))
    out.alpha_composite(top, ((out.width - top.width) // 2, 0))
    out.alpha_composite(pill, ((out.width - pill.width) // 2, top.height - 70))
    if out.width > 1000:
        k = 1000 / out.width
        out = out.resize((int(out.width * k), int(out.height * k)), Image.LANCZOS)
    return out

@lru_cache(None)
def rays_img(col):
    S = 2600
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = S / 2
    for k in range(18):
        a0 = k * 2 * math.pi / 18; a1 = a0 + math.pi / 36
        d.polygon([(c, c), (c + S * math.cos(a0), c + S * math.sin(a0)), (c + S * math.cos(a1), c + S * math.sin(a1))],
                  fill=col + (40,))
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).ellipse([c - S * 0.48, c - S * 0.48, c + S * 0.48, c + S * 0.48], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(300))
    a = np.array(im.getchannel("A")).astype(np.float32) * (np.array(m) / 255.0)
    im.putalpha(Image.fromarray(a.astype(np.uint8)))
    return im.filter(ImageFilter.GaussianBlur(3))

# ----------------------------------------------------------------- post fx
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIGNETTE = (1 - 0.42 * (((_xx - W / 2) / (W * 0.72)) ** 2 + ((_yy - H / 2) / (H * 0.72)) ** 2)).clip(0.45, 1)[..., None]
del _yy, _xx
GRAIN = [np.random.default_rng(i).normal(0, 3.2, (H // 2, W // 2, 1)).astype(np.float32) for i in range(6)]

def zoom(img, s, cx=W / 2, cy=H / 2):
    if abs(s - 1) < 1e-3:
        return img
    w, h = W / s, H / s
    x0 = clamp(cx - w / 2, 0, W - w); y0 = clamp(cy - h / 2, 0, H - h)
    return img.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + w, y0 + h))

def zoom_blur(img, amount, n=6):
    if amount < 0.004:
        return img
    acc = np.zeros((H, W, 3), np.float32)
    for i in range(n):
        acc += np.asarray(zoom(img, 1 + amount * i / (n - 1)), np.float32)
    return Image.fromarray((acc / n).astype(np.uint8))

def shake(t, amp, freq=38):
    return amp * math.sin(t * freq * 1.3 + 1.1) * math.cos(t * freq * 0.7), amp * math.sin(t * freq + 0.3)

def offset(img, dx, dy, s=1.0):
    if abs(dx) < 0.5 and abs(dy) < 0.5 and s == 1.0:
        return img
    s = max(s, 1 + (abs(dx) * 2 + 4) / W, 1 + (abs(dy) * 2 + 4) / H)
    return zoom(img, s, W / 2 - dx, H / 2 - dy)

def chroma(arr, px):
    px = int(px)
    if px <= 0:
        return arr
    out = arr.copy()
    out[..., 0] = np.roll(arr[..., 0], px, axis=1)
    out[..., 2] = np.roll(arr[..., 2], -px, axis=1)
    return out

def glitch(arr, t, strength):
    if strength <= 0:
        return arr
    r = np.random.default_rng(int(t * 1000))
    out = arr.copy()
    for _ in range(int(8 * strength) + 2):
        y = r.integers(0, H - 60); h = r.integers(10, 120)
        out[y:y + h] = np.roll(out[y:y + h], int(r.normal(0, 90 * strength)), axis=1)
    return chroma(out, 12 * strength)

def grade(arr, t, sat=1.12, contrast=1.08):
    a = arr.astype(np.float32)
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    a = lum[..., None] + (a - lum[..., None]) * sat
    a = (a - 128) * contrast + 128
    a *= VIGNETTE
    g = GRAIN[int(t * FPS) % len(GRAIN)]
    a += np.repeat(np.repeat(g, 2, 0), 2, 1)
    return np.clip(a, 0, 255).astype(np.uint8)

# ----------------------------------------------------------------- voxel shots
def vox(world, pos, yaw, pitch, fov=72, roll=0.0, res=1, sky=None, fog=110.0, fog_col=None, bloom=0.8):
    w, h = (360, 640) if res == 1 else (180, 320)
    kw = {}
    if sky is not None:
        kw["sky"] = sky
    col, em = render(world, pos, yaw, pitch, W=w, H=h, fov=fov, fog_dist=fog, fog_col=fog_col, roll=roll, **kw)
    small = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8))
    img = small.resize((W, H), Image.NEAREST if res == 1 else Image.BILINEAR)
    if bloom and em.max() > 0:
        b = Image.fromarray(np.clip(col * em[..., None] * 1.4, 0, 255).astype(np.uint8))
        b = b.filter(ImageFilter.GaussianBlur(w / 40)).resize((W, H), Image.BILINEAR)
        a = np.asarray(img, np.float32) + np.asarray(b, np.float32) * bloom
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return img

def farm_cam(shot, u):
    g = G_FARM
    if shot == "walk":
        p = (47.5, g + 3.6 + 0.07 * math.sin(u * 13), 20 + 6.5 * u)
        return dict(pos=p, yaw=0.05 * math.sin(u * 2), pitch=-0.26, fov=72)
    if shot == "side":
        p = (24, g + 6.5, 36 + 9 * u)
        return dict(pos=p, yaw=math.pi / 2 - 0.25, pitch=-0.32, fov=70)
    if shot == "aerial":
        a = -0.7 + 0.35 * u
        p = (56 + 32 * math.sin(a), g + 22, 58 - 32 * math.cos(a))
        y, pt = look_at(p, (56, g, 58))
        return dict(pos=p, yaw=y, pitch=pt, fov=74)
    if shot == "close":
        p = (47.5 + 0.3 * u, g + 3.3, 40 + 0.8 * u)
        return dict(pos=p, yaw=0.35, pitch=-0.5, fov=70 - 18 * u)
    raise ValueError(shot)

def farm_shot(shot, u, res=1):
    c = farm_cam(shot, u)
    return vox(FARM, c["pos"], c["yaw"], c["pitch"], c["fov"], res=res, fog=150)

def donut_shot(a, dist, height, fov=62, res=1, target_dy=-2):
    cx, cy, cz = DONUT_C
    p = (cx + dist * math.sin(a), cy + height, cz - dist * math.cos(a))
    y, pt = look_at(p, (cx, cy + target_dy, cz))
    return vox(DONUT, p, y, pt, fov, res=res, fog=170)

def hugo_shot(a, dist, height, fov=62, res=1, target=None):
    cx, cy, cz = HUGO_C
    p = (cx + dist * math.sin(a), cy + height, cz - dist * math.cos(a))
    y, pt = look_at(p, target or (cx, cy - 3, cz))
    return vox(HUGO, p, y, pt, fov, res=res, fog=170, bloom=0.5)

def tunnel_shot(z, roll=0.0, fov=80):
    return vox(TUNNEL, (10.5, 10.5 + 0.2 * math.sin(z * 0.3), z), 0.0, 0.0, fov, roll=roll,
               sky=sky_dark, fog=80, fog_col=(8, 6, 18), bloom=1.4)

# ----------------------------------------------------------------- overlays
def scoreboard(base, money, plus_age):
    """Minecraft-style scoreboard (DonutSMP vibe) in the top-left."""
    f_t = pixel_font(40); f = pixel_font(36)
    box = Image.new("RGBA", (470, 230), (0, 0, 0, 0))
    ImageDraw.Draw(box).rectangle([0, 0, 469, 229], fill=(0, 0, 0, 120))
    ImageDraw.Draw(box).rectangle([0, 0, 469, 56], fill=(0, 0, 0, 90))
    d = ImageDraw.Draw(box)
    d.text((235, 30), "DONUTSMP", font=f_t, fill=(85, 255, 255), anchor="mm")
    d.text((24, 80), "Money:", font=f, fill=(255, 255, 255))
    d.text((24 + 175, 80), f"${money:,}", font=f, fill=(85, 255, 85))
    d.text((24, 135), "Shards:", font=f, fill=(255, 255, 255))
    d.text((24 + 205, 135), "3", font=f, fill=(255, 85, 255))
    d.text((24, 185), "Farming...", font=pixel_font(28), fill=(170, 170, 170))
    base.alpha_composite(box, (60, 230))
    if plus_age is not None and plus_age < 0.6:
        im = text_rgba("+$2", 54, (85, 255, 85), stroke=5, fnt=pixel_font(54), shadow=False)
        paste_center(base, im, 610, 340 - 90 * plus_age, alpha=1 - plus_age / 0.6)

def watermark(base, alpha=0.9):
    im = _watermark()
    paste_center(base, im, W / 2, 150, alpha=alpha)

@lru_cache(None)
def _watermark():
    f = font(40, "ExtraBold")
    tw = f.getbbox("donuthugobet.net")[2]
    pill = rounded((tw + 120, 76), 38, (10, 6, 20, 150), outline=(255, 255, 255, 60), width=2)
    pill.alpha_composite(donut_icon(50), (18, 13))
    ImageDraw.Draw(pill).text((84, 38), "donuthugobet.net", font=f, fill=(255, 255, 255, 235), anchor="lm")
    return pill

class Particles:
    def __init__(self, seed, n, t0, x, y, speed=(900, 2200), kinds=("coin", "spr")):
        r = np.random.default_rng(seed)
        self.t0, self.x, self.y = t0, x, y
        a = r.uniform(0, 2 * np.pi, n); s = r.uniform(*speed, n)
        self.vx, self.vy = np.cos(a) * s, np.sin(a) * s - 600
        self.rot0 = r.uniform(0, 360, n); self.vr = r.uniform(-700, 700, n)
        self.kind = r.choice(kinds, n); self.size = r.uniform(0.6, 1.3, n)
        self.col = r.integers(0, 6, n)

    def draw(self, base, t):
        u = t - self.t0
        if u < 0 or u > 1.6:
            return
        cols = [(255, 255, 255), (255, 230, 60), (90, 200, 255), (120, 240, 120), PINK, (170, 110, 255)]
        for i in range(len(self.vx)):
            x = self.x + self.vx[i] * u * (1 - 0.35 * u)
            y = self.y + self.vy[i] * u + 1400 * u * u
            if not (-100 < x < W + 100 and -100 < y < H + 100):
                continue
            k = self.kind[i]; s = self.size[i]
            rot = self.rot0[i] + self.vr[i] * u
            if k == "coin":
                im = hugo_coin(int(90 * s))
                sq = abs(math.cos(math.radians(rot))) * 0.8 + 0.2
                im = im.resize((max(2, int(im.width * sq)), im.height))
            elif k == "donut":
                im = donut_icon(int(100 * s))
            elif k == "em":
                im = _emerald(int(60 * s))
            elif k == "cash":
                im = _cash(int(110 * s))
            else:
                im = rounded((int(46 * s), int(16 * s)), int(8 * s), cols[self.col[i]] + (255,))
            paste_center(base, im, x, y, rot=rot if k != "coin" else 0, alpha=clamp(1.6 - u))

@lru_cache(None)
def _emerald(s):
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(s / 2, 0), (s, s / 2), (s / 2, s), (0, s / 2)], fill=(40, 210, 100, 255), outline=(10, 110, 50, 255))
    d.polygon([(s / 2, s * 0.2), (s * 0.8, s / 2), (s / 2, s * 0.5), (s * 0.3, s * 0.45)], fill=(150, 255, 180, 255))
    return im

@lru_cache(None)
def _cash(s):
    w, h = s, int(s * 0.5)
    im = rounded((w, h), 6, (60, 170, 80, 255), outline=(20, 90, 40, 255), width=4)
    d = ImageDraw.Draw(im)
    d.ellipse([w / 2 - h * 0.3, h * 0.2, w / 2 + h * 0.3, h * 0.8], outline=(200, 255, 200, 255), width=3)
    d.text((w / 2, h / 2), "$", font=font(int(h * 0.5), "Black"), fill=(220, 255, 220, 255), anchor="mm")
    return im

# ----------------------------------------------------------------- captions
SPECIAL = {"WRONG": None, "DONUTHUGOBET.NET": None,
           "DONUT": dict(fill=PINK, scale=1.3, glow=(255, 60, 150)),
           "HUGO": dict(fill=GOLD, scale=1.3, glow=(255, 160, 0)),
           "EXPLODE": dict(fill=GREEN, scale=1.35, glow=(0, 255, 90)),
           "DonutSMP": dict(fill=(85, 255, 255), scale=1.0),
           "multiplier": dict(fill=GREEN, scale=1.0)}

def build_chunks():
    chunks = []
    for li, line in enumerate(L):
        cur = []
        for wi, w in enumerate(line):
            sp = w["w"] in SPECIAL
            if cur and (sp or cur[-1]["w"] in SPECIAL or len(cur) >= 2 or
                        sum(len(x["w"]) for x in cur) + len(w["w"]) > 9 or w["t0"] - cur[-1]["t1"] > 0.22):
                chunks.append(cur); cur = []
            cur.append(w)
        if cur:
            chunks.append(cur)
    out = []
    for i, c in enumerate(chunks):
        t0 = c[0]["t0"] - 0.04
        nxt = chunks[i + 1][0]["t0"] - 0.04 if i + 1 < len(chunks) else 1e9
        t1 = min(nxt, c[-1]["t1"] + 0.3)
        text = " ".join(x["w"] for x in c)
        out.append(dict(text=text, t0=t0, t1=t1, key=c[0]["w"] if len(c) == 1 else None))
    return out

CHUNKS = build_chunks()

@lru_cache(None)
def caption_img(text, key):
    st = SPECIAL.get(key) or {}
    size = int(124 * st.get("scale", 1.0))
    disp = text if key in SPECIAL else text.lower()
    im = text_rgba(disp, size, st.get("fill", WHITE), stroke=11, glow=st.get("glow"))
    if im.width > 1010:
        k = 1010 / im.width
        im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
    return im

def caption_y(t):
    lo = [("coin", "balance"), ("balance", "stop"), ("stop", "goto")]
    for a, b in lo:
        if EV[a] <= t < EV[b]:
            return 1470
    return 1080

def draw_caption(base, t):
    for c in CHUNKS:
        if c["t0"] <= t < c["t1"]:
            if c["key"] in SPECIAL and SPECIAL[c["key"]] is None:
                return
            u = t - c["t0"]
            s = 0.55 + 0.45 * ease_back(u / 0.16, 2.6) if u < 0.16 else 1.0
            dy = 0
            if c["key"] == "EXPLODE":
                dx, dy = shake(t, 10)
            paste_center(base, caption_img(c["text"], c["key"]), W / 2, caption_y(t) + dy, scale=s)
            return

# ----------------------------------------------------------------- scenes
def bg_blur(img, radius=14, dark=0.55, tint=(40, 10, 60)):
    b = img.filter(ImageFilter.GaussianBlur(radius))
    a = np.asarray(b, np.float32) * dark + np.array(tint, np.float32) * (1 - dark)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def scene_farm(t):
    if t < EV["wrong"]:
        bounds = [(0, "walk", L[0][3]["t0"]), (L[0][3]["t0"], "side", L[0][6]["t0"]),
                  (L[0][6]["t0"], "aerial", L[0][7]["t0"]), (L[0][7]["t0"], "close", 99)]
        for a, shot, b in bounds:
            if a <= t < b:
                break
        u = t - a
        if t >= EV["you_are"]:
            img = farm_shot("close", (EV["you_are"] - a))
            k = (t - EV["you_are"]) / (EV["wrong"] - EV["you_are"])
            img = zoom_blur(zoom(img, 1 + 0.6 * ease_in(k)), 0.06 * k)
            arr = np.asarray(img, np.float32)
            arr = arr * (1 - 0.45 * k) + np.array([255, 20, 30], np.float32) * 0.45 * k
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        else:
            img = farm_shot(shot, u)
            punch = 1 + 0.12 * (1 - ease_out(u / 0.18))
            if shot == "close":
                punch *= 1 + 0.35 * ease_io(u / (EV["you_are"] - a))
            img = zoom(img, punch)
        base = img.convert("RGBA")
        money = 1204 + 2 * int(t / 0.55)
        scoreboard(base, money, (t % 0.55) if t > 0.3 else None)
        return base
    # WRONG
    u = t - EV["wrong"]
    img = farm_shot("close", EV["you_are"] - L[0][7]["t0"])
    arr = np.asarray(zoom(img, 1.6), np.float32)
    gray = arr.mean(-1, keepdims=True)
    arr = np.repeat(gray * 0.18 + 255 * 0.82, 3, -1)
    base = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    s = 0.75 + 0.25 * ease_back(u / 0.14, 3.0) if u < 0.14 else 1.0 + 0.04 * u
    im = _wrong()
    dx, dy = shake(t, 22 * max(0, 1 - u / 0.4) + 4)
    paste_center(base, im, W / 2 + dx, 1000 + dy, scale=s, rot=-4)
    return base

@lru_cache(None)
def _wrong():
    im = text_rgba("WRONG", 230, RED, stroke=16, weight="Black", glow=(255, 0, 0))
    k = 940 / im.width
    return im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)

def scene_tunnel(t, a, b, z0, z1, pw=2.2):
    p = (t - a) / (b - a)
    z = z0 + (z1 - z0) * p ** pw
    img = tunnel_shot(z, roll=0.12 * math.sin(p * 3.5), fov=78 + 24 * p ** 2)
    img = zoom_blur(img, 0.02 + 0.10 * p ** 2)
    if p > 0.82:
        arr = np.asarray(img, np.float32)
        k = (p - 0.82) / 0.18
        arr = arr * (1 - k) + 255 * k
        img = Image.fromarray(arr.astype(np.uint8))
    return img.convert("RGBA")

LOGO_PARTS = Particles(1, 46, 0, W / 2, 820, kinds=("coin", "spr", "spr", "donut"))
FINAL_PARTS = Particles(2, 46, 0, W / 2, 700, kinds=("coin", "spr", "spr", "donut"))

def logo_block(base, u, cy, t):
    # light rays
    rays = rays_img(PINK if int(t * 2) % 2 == 0 else PINK)
    rr = rays.rotate(u * 12, Image.BILINEAR)
    paste_center(base, rr, W / 2, cy, alpha=clamp(u / 0.2) * 0.9)
    s = 2.2 - 1.2 * ease_back(u / 0.32, 1.4) if u < 0.32 else 1 + 0.015 * math.sin(u * 4)
    fl = math.sin(u * 3) * 10
    paste_center(base, donut_icon(400), W / 2, cy - 250 + fl, scale=s, rot=u * 25)
    wm = wordmark()
    paste_center(base, wm, W / 2, cy + 160 + fl, scale=s)
    # shine sweep
    if 0.45 < u < 1.0:
        k = (u - 0.45) / 0.55
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        x = -300 + (W + 600) * k
        ImageDraw.Draw(sh).polygon([(x, cy - 50), (x + 90, cy - 50), (x - 30, cy + 330), (x - 120, cy + 330)], fill=(255, 255, 255, 90))
        sh = sh.filter(ImageFilter.GaussianBlur(10))
        a = np.asarray(sh.getchannel("A"), np.float32)
        m = np.zeros((H, W), np.float32)
        wa = np.asarray(wm.getchannel("A"), np.float32) / 255
        ox, oy = int(W / 2 - wm.width / 2), int(cy + 160 + fl - wm.height / 2)
        m[oy:oy + wm.height, ox:ox + wm.width] = wa
        sh.putalpha(Image.fromarray((a * m).astype(np.uint8)))
        base.alpha_composite(sh)

_BG_CACHE = {}
def bg_still(key):
    if key not in _BG_CACHE:
        if key == "donut":
            _BG_CACHE[key] = bg_blur(donut_shot(0.2, 60, 34, res=2), 12, 0.5)
        elif key == "hugo":
            _BG_CACHE[key] = bg_blur(hugo_shot(-0.25, 44, 6, res=2), 12, 0.5, tint=(50, 25, 5))
        elif key == "tunnel":
            _BG_CACHE[key] = bg_blur(tunnel_shot(120), 10, 0.6, tint=(30, 5, 50))
    return _BG_CACHE[key]

def kenburns(img, u, s0=1.05, s1=1.15):
    return zoom(img, lerp(s0, s1, clamp(u)))

def scene_logo(t):
    u = t - EV["drop"]
    bg = kenburns(bg_still("donut"), u / 2, 1.0, 1.1).convert("RGBA")
    LOGO_PARTS.t0 = EV["drop"]
    LOGO_PARTS.draw(bg, t)
    logo_block(bg, u, 820, t)
    return bg

def scene_donut(t):
    u = t - EV["donut"]; d = EV["hugo"] - EV["donut"]
    p = u / d
    img = donut_shot(-0.95 + 0.75 * ease_io(p), 54 - 9 * p, 28 - 4 * p, fov=60)
    w = [x for x in L[4] if x["w"] == "DONUT"][0]["t0"]
    s = 1 + 0.12 * (1 - ease_out(u / 0.2))
    if t >= w:
        s *= 1 + 0.10 * (1 - ease_out((t - w) / 0.25))
    return zoom(img, s).convert("RGBA")

def scene_hugo(t):
    u = t - EV["hugo"]; d = EV["coin"] - EV["hugo"]
    p = u / d
    img = hugo_shot(0.75 - 0.6 * ease_io(p), 46 - 6 * p, -2 + 6 * p, fov=60)
    w = [x for x in L[5] if x["w"] == "HUGO"][0]["t0"]
    s = 1 + 0.12 * (1 - ease_out(u / 0.2))
    if t >= w:
        s *= 1 + 0.10 * (1 - ease_out((t - w) / 0.25))
    return zoom(img, s).convert("RGBA")

@lru_cache(None)
def _coin_card():
    c = glass_card((900, 980), accent=PINK)
    d = ImageDraw.Draw(c)
    d.text((c.width / 2, 130), "COINFLIP", font=font(70, "Black"), fill=WHITE, anchor="mm")
    d.text((c.width / 2, 195), "DONUT  vs  HUGO", font=font(36, "Bold"), fill=(200, 180, 230), anchor="mm")
    return c

def scene_coin(t):
    u = t - EV["coin"]; d = EV["multi"] - EV["coin"]
    base = kenburns(bg_still("hugo"), u / d).convert("RGBA")
    cin = ease_back(u / 0.25, 1.6)
    card_y = 860
    paste_center(base, _coin_card(), W / 2, card_y + (1 - cin) * 300, scale=0.9 + 0.1 * cin, alpha=clamp(u / 0.12))
    land = d * 0.72
    k = clamp(u / land)
    ang = (1 - (1 - k) ** 2.2) * math.pi * 9
    cs = math.cos(ang)
    face = donut_coin(440) if cs >= 0 else hugo_coin(440)
    hop = math.sin(k * math.pi) * 170
    fw = max(4, int(440 * abs(cs)))
    paste_center(base, face.resize((fw, 440), Image.BICUBIC), W / 2, card_y + 60 - hop + (1 - cin) * 300, scale=0.95)
    if u > land:
        v = u - land
        win = _won_img()
        paste_center(base, win, W / 2, card_y + 380, scale=0.6 + 0.4 * ease_back(v / 0.15, 2.5))
    return base

@lru_cache(None)
def _won_img():
    return text_rgba("WIN  x2", 96, GREEN, stroke=8, weight="Black", glow=(0, 255, 90))

def scene_multi(t):
    u = t - EV["multi"]; d = EV["balance"] - EV["multi"]
    base = kenburns(bg_still("donut"), u / d).convert("RGBA")
    card = glass_card((900, 980), accent=GREEN)
    cx0, cy0 = W / 2 - card.width / 2, 860 - card.height / 2
    cin = ease_back(u / 0.2, 1.6)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer.alpha_composite(card, (int(cx0), int(cy0)))
    dd = ImageDraw.Draw(layer)
    gx0, gy0, gx1, gy1 = 160, 560, 920, 1240
    for i in range(6):
        y = gy0 + (gy1 - gy0) * i / 5
        dd.line([gx0, y, gx1, y], fill=(255, 255, 255, 30), width=2)
    p = clamp(u / (d * 0.95))
    m = math.exp(2.25 * p * 1.0)
    mmax = math.exp(2.25)
    pts = []
    for i in range(60):
        q = p * i / 59
        mv = math.exp(2.25 * q)
        x = gx0 + (gx1 - gx0) * q
        y = gy1 - (gy1 - gy0) * (mv - 1) / (mmax - 1)
        pts.append((x, y))
    if len(pts) > 1:
        poly = pts + [(pts[-1][0], gy1), (gx0, gy1)]
        fill = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(fill).polygon(poly, fill=(60, 255, 120, 60))
        layer.alpha_composite(fill)
        glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(glow).line(pts, fill=(60, 255, 120, 255), width=22, joint="curve")
        layer.alpha_composite(glow.filter(ImageFilter.GaussianBlur(12)))
        dd.line(pts, fill=(200, 255, 210, 255), width=10, joint="curve")
        paste_center(layer, _emerald(70), pts[-1][0], pts[-1][1])
    dd.text((W / 2, 420), "MULTIPLIER", font=font(44, "Bold"), fill=(200, 230, 210, 255), anchor="mm")
    mt = text_rgba(f"{m:.2f}x", 150, GREEN, stroke=8, weight="Black", glow=(0, 255, 90))
    paste_center(layer, mt, W / 2, 495 - 70 + 140, scale=1.0)
    paste_center(base, layer, W / 2, H / 2 + (1 - cin) * 300, scale=0.9 + 0.1 * cin)
    return base

BAL_PARTS = Particles(3, 60, 0, W / 2, 820, speed=(1000, 2600), kinds=("cash", "em", "coin", "cash"))

def scene_balance(t):
    u = t - EV["balance"]; d = EV["instant"] - EV["balance"]
    p = u / d
    img = hugo_shot(-0.25 + 0.2 * p, 40 - 4 * p, 4 + 6 * p, fov=64)
    ex = EV["explode"]
    s = 1 + 0.1 * (1 - ease_out(u / 0.2))
    if t >= ex:
        s *= 1 + 0.18 * (1 - ease_out((t - ex) / 0.4))
    base = bg_blur(zoom(img, s), 5, 0.7, tint=(20, 40, 10)).convert("RGBA")
    if t < ex:
        k = clamp((t - EV["balance"]) / (ex - EV["balance"]))
        val = int(1_250_000 * (39 ** (k ** 1.6)))
        col = WHITE
    else:
        val = 48_750_000; col = GOLD
    card = rounded((900, 330), 40, (0, 0, 0, 150), outline=(255, 255, 255, 50), width=3)
    ImageDraw.Draw(card).text((450, 70), "BALANCE", font=font(46, "Bold"), fill=(210, 210, 210, 255), anchor="mm")
    paste_center(base, card, W / 2, 830)
    num = text_rgba(f"${val:,}", 112, col, stroke=8, fnt=pixel_font(112),
                    glow=(0, 255, 90) if t < ex else (255, 170, 0))
    ns = 1.0
    if t >= ex:
        ns = 1.0 + 0.3 * (1 - ease_out((t - ex) / 0.35))
    dx, dy = shake(t, 18 * max(0, 1 - (t - ex) / 0.5)) if t >= ex else (0, 0)
    paste_center(base, num, W / 2 + dx, 880 + dy, scale=min(ns, 1010 / num.width * ns))
    # floating +gains
    if t < ex:
        for i in range(4):
            ph = (u * 1.8 + i / 4) % 1
            gx = (230, 820, 330, 740)[i]
            gy = (560, 600, 1130, 1170)[i]
            gi = _gain(i)
            paste_center(base, gi, gx, gy - 160 * ph, alpha=math.sin(ph * math.pi))
    BAL_PARTS.t0 = ex
    BAL_PARTS.draw(base, t)
    if t >= ex:
        fl = 1 - clamp((t - ex) / 0.25)
        if fl > 0:
            wht = Image.new("RGBA", (W, H), (255, 255, 255, int(220 * fl)))
            base.alpha_composite(wht)
    return base

@lru_cache(None)
def _gain(i):
    return text_rgba(f"+${(250, 500, 750, 1000)[i]}K", 58, GREEN, stroke=5, fnt=pixel_font(58), shadow=False)

@lru_cache(None)
def _bolt(s):
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([0, 0, s - 1, s - 1], fill=GOLD + (255,))
    pts = [(0.56, 0.12), (0.28, 0.55), (0.48, 0.55), (0.40, 0.88), (0.72, 0.42), (0.52, 0.42), (0.60, 0.12)]
    d.polygon([(x * s, y * s) for x, y in pts], fill=DARK + (255,))
    return im

@lru_cache(None)
def _check(s):
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([0, 0, s - 1, s - 1], fill=(40, 220, 100, 255))
    d.line([(0.27 * s, 0.52 * s), (0.44 * s, 0.68 * s), (0.74 * s, 0.34 * s)], fill=WHITE + (255,), width=int(s * 0.11), joint="curve")
    return im

def instant_card(base, u, cy, title, sub, accent):
    if u < 0:
        return
    k = ease_back(u / 0.28, 1.5)
    card = glass_card((900, 300), radius=44, accent=accent)
    d = ImageDraw.Draw(card)
    card.alpha_composite(_bolt(120), (78, 95))
    fs = 60
    while font(fs, "Black").getlength(title) > 545:
        fs -= 2
    d.text((235, 125), title, font=font(fs, "Black"), fill=WHITE, anchor="lm")
    d.text((235, 190), sub, font=font(36, "SemiBold"), fill=(205, 190, 230), anchor="lm")
    bar_k = clamp((u - 0.15) / 0.4)
    d.rounded_rectangle([235, 245, 235 + 545, 263], 9, fill=(255, 255, 255, 40))
    if bar_k > 0:
        d.rounded_rectangle([235, 245, 235 + int(545 * bar_k), 263], 9, fill=accent + (255,))
    if u > 0.58:
        c = _check(110)
        sc = ease_back((u - 0.58) / 0.15, 2.5)
        cc = c.resize((max(1, int(110 * sc)), max(1, int(110 * sc))))
        card.alpha_composite(cc, (int(872 - cc.width / 2), int(190 - cc.height / 2)))
    paste_center(base, card, W / 2 + (1 - k) * 1100, cy)

def scene_instant(t):
    u = t - EV["instant"]; d = EV["stop"] - EV["instant"]
    base = kenburns(bg_still("tunnel"), u / d, 1.05, 1.2).convert("RGBA")
    instant_card(base, u, 700, "INSTANT DEPOSITS", "Donut & Hugo money", PINK)
    instant_card(base, t - EV["instant2"], 1060, "INSTANT WITHDRAWALS", "Straight to your account", GOLD)
    return base

def scene_stop(t):
    u = t - EV["stop"]
    img = farm_shot("walk", 0.4 + u * 0.5)
    arr = np.asarray(img, np.float32)
    gray = arr.mean(-1, keepdims=True)
    arr = gray * 0.75 + arr * 0.25
    arr *= 0.75
    base = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    scoreboard(base, 1204 + 2 * int((t - 10) / 0.55), None)
    v = t - EV["stop_word"]
    if v >= 0:
        s = 1.8 - 0.8 * ease_back(v / 0.18, 2.0) if v < 0.18 else 1.0
        dx, dy = shake(t, 20 * max(0, 1 - v / 0.35))
        paste_center(base, _xmark(), W / 2 + dx, 850 + dy, scale=s)
    return base

@lru_cache(None)
def _xmark():
    s = 640
    im = Image.new("RGBA", (s + 80, s + 80), (0, 0, 0, 0))
    gl = Image.new("RGBA", im.size, (0, 0, 0, 0))
    for layer, wdt, col in ((gl, 150, (255, 0, 0, 200)), (im, 110, RED + (255,))):
        d = ImageDraw.Draw(layer)
        d.line([(80, 80), (s, s)], fill=(0, 0, 0, 255) if layer is im else col, width=wdt + (28 if layer is im else 0))
        d.line([(s, 80), (80, s)], fill=(0, 0, 0, 255) if layer is im else col, width=wdt + (28 if layer is im else 0))
    gl = gl.filter(ImageFilter.GaussianBlur(30))
    d = ImageDraw.Draw(im)
    d.line([(80, 80), (s, s)], fill=RED + (255,), width=110)
    d.line([(s, 80), (80, s)], fill=RED + (255,), width=110)
    out = Image.alpha_composite(gl, im)
    return out

def scene_final(t):
    u = t - EV["final"]
    bg = donut_shot(0.15 + 0.12 * u, 66, 30, res=2)
    base = bg_blur(bg, 10, 0.45).convert("RGBA")
    FINAL_PARTS.t0 = EV["final"]
    FINAL_PARTS.draw(base, t)
    logo_block(base, u, 640, t)
    # URL bar typing
    url = "donuthugobet.net"
    k = clamp((u - 0.35) / 0.8)
    n = int(len(url) * k)
    bar = _url_bar(n, (int(u * 3) % 2 == 0) or k < 1)
    ba = clamp((u - 0.25) / 0.2)
    paste_center(base, bar, W / 2, 1110, alpha=ba, scale=0.9 + 0.1 * ease_back(ba))
    v = t - EV["tagline"]
    if v >= 0:
        l1, l2 = _tagline()
        paste_center(base, l1, W / 2, 1330, scale=0.6 + 0.4 * ease_back(v / 0.2, 2.2))
        if v > 0.25:
            paste_center(base, l2, W / 2, 1460, scale=0.6 + 0.4 * ease_back((v - 0.25) / 0.2, 2.2))
    if u > 0.6:
        paste_center(base, _disclaimer(), W / 2, 1720, alpha=clamp((u - 0.6) / 0.3) * 0.85)
    fl = 1 - clamp(u / 0.3)
    if fl > 0:
        base.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(255 * fl))))
    return base

@lru_cache(None)
def _url_bar(n, cursor):
    f = font(58, "Bold")
    bar = Image.new("RGBA", (880, 170), (0, 0, 0, 0))
    gl = Image.new("RGBA", bar.size, (0, 0, 0, 0))
    ImageDraw.Draw(gl).rounded_rectangle([30, 30, 850, 140], 55, fill=(255, 255, 255, 120))
    bar.alpha_composite(gl.filter(ImageFilter.GaussianBlur(16)))
    d = ImageDraw.Draw(bar)
    d.rounded_rectangle([30, 30, 850, 140], 55, fill=(255, 255, 255, 255))
    # lock
    d.rounded_rectangle([78, 78, 118, 112], 6, fill=(40, 180, 90, 255))
    d.arc([84, 58, 112, 94], 180, 360, fill=(40, 180, 90, 255), width=7)
    txt = "donuthugobet.net"[:n]
    d.text((145, 85), txt, font=f, fill=DARK + (255,), anchor="lm")
    if cursor:
        x = 145 + (f.getbbox(txt)[2] if txt else 0) + 6
        d.rectangle([x, 60, x + 5, 110], fill=(200, 60, 140, 255))
    return bar

@lru_cache(None)
def _tagline():
    def line(parts, size):
        f = font(size, "Black"); st = 9; pad = 50
        tw = sum(f.getlength(txt) for txt, _, _ in parts)
        asc, desc = f.getmetrics()
        im = Image.new("RGBA", (int(tw) + 2 * pad, asc + desc + 2 * pad), (0, 0, 0, 0))
        gl = Image.new("RGBA", im.size, (0, 0, 0, 0))
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        x = pad
        for txt, col, glc in parts:
            if glc:
                ImageDraw.Draw(gl).text((x, pad), txt, font=f, fill=glc + (255,), stroke_width=st, stroke_fill=glc + (255,))
            ImageDraw.Draw(sh).text((x, pad + 8), txt, font=f, fill=(0, 0, 0, 170), stroke_width=st, stroke_fill=(0, 0, 0, 170))
            x += f.getlength(txt)
        im.alpha_composite(gl.filter(ImageFilter.GaussianBlur(16)))
        im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6)))
        x = pad
        for txt, col, _ in parts:
            ImageDraw.Draw(im).text((x, pad), txt, font=f, fill=col + (255,), stroke_width=st, stroke_fill=(0, 0, 0, 255))
            x += f.getlength(txt)
        if im.width > 1040:
            k = 1040 / im.width
            im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
        return im
    l1 = line([("GAMBLE WITH ", WHITE, None), ("DONUT", PINK, (255, 60, 150)), (" MONEY", WHITE, None)], 92)
    l2 = line([("OR ", WHITE, None), ("HUGO", GOLD, (255, 160, 0)), (" MONEY", WHITE, None)], 110)
    return l1, l2

@lru_cache(None)
def _disclaimer():
    return text_rgba("18+  |  Play responsibly", 34, (220, 220, 220), stroke=3, weight="SemiBold", shadow=False)

# ----------------------------------------------------------------- frame
def frame(i):
    t = i / FPS
    fx = dict(chroma=0, glitch=0, flash=0, shake=(0, 0))
    if t < EV["secret"]:
        img = scene_farm(t)
        if t >= EV["wrong"]:
            u = t - EV["wrong"]
            fx["chroma"] = 10 * max(0, 1 - u / 0.3) + 2
            fx["flash"] = max(0, 1 - u / 0.12) * 0.8
            if EV["secret"] - t < 0.1:
                fx["glitch"] = 1.0
    elif t < EV["drop"]:
        img = scene_tunnel(t, EV["secret"], EV["drop"], 6, 150)
    elif t < EV["donut"]:
        img = scene_logo(t)
        u = t - EV["drop"]
        fx["flash"] = max(0, 1 - u / 0.22)
        fx["shake"] = shake(t, 26 * max(0, 1 - u / 0.45))
        fx["chroma"] = 8 * max(0, 1 - u / 0.3)
    elif t < EV["hugo"]:
        img = scene_donut(t)
    elif t < EV["coin"]:
        img = scene_hugo(t)
    elif t < EV["multi"]:
        img = scene_coin(t)
    elif t < EV["balance"]:
        img = scene_multi(t)
    elif t < EV["instant"]:
        img = scene_balance(t)
        if t >= EV["explode"]:
            fx["chroma"] = 10 * max(0, 1 - (t - EV["explode"]) / 0.35)
    elif t < EV["stop"]:
        img = scene_instant(t)
    elif t < EV["goto"]:
        img = scene_stop(t)
        if t >= EV["stop_word"]:
            fx["chroma"] = 8 * max(0, 1 - (t - EV["stop_word"]) / 0.3)
    elif t < EV["final"]:
        img = scene_tunnel(t, EV["goto"], EV["final"], 40, 150, pw=1.6)
    else:
        img = scene_final(t)
        u = t - EV["final"]
        fx["shake"] = shake(t, 24 * max(0, 1 - u / 0.45))
        fx["chroma"] = 8 * max(0, 1 - u / 0.3)

    # every new scene gets a short glitch-in for snappy cuts
    for k in ("donut", "hugo", "coin", "multi", "balance", "instant", "stop"):
        if 0 <= t - EV[k] < 2 / FPS:
            fx["glitch"] = max(fx["glitch"], 0.6)
            fx["chroma"] = max(fx["chroma"], 6)

    if EV["drop"] + 0.6 < t < EV["final"] - 0.6 and not (EV["stop"] <= t < EV["final"]):
        watermark(img, 0.85)
    draw_caption(img, t)
    img = img.convert("RGB")
    if fx["shake"] != (0, 0):
        img = offset(img, *fx["shake"])
    arr = np.asarray(img)
    if fx["glitch"]:
        arr = glitch(arr, t, fx["glitch"])
    if fx["chroma"]:
        arr = chroma(arr, fx["chroma"])
    arr = grade(arr, t)
    if fx["flash"] > 0:
        arr = (arr.astype(np.float32) * (1 - fx["flash"]) + 255 * fx["flash"]).astype(np.uint8)
    return arr.tobytes()

def prewarm():
    for k in ("donut", "hugo", "tunnel"):
        bg_still(k)
    wordmark(); donut_icon(400); _tagline()

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BUILD, "donuthugobet_promo.mp4")
    only = os.environ.get("FRAMES")  # e.g. "0,90,200" for stills
    prewarm()
    if only:
        for i in map(int, only.split(",")):
            Image.frombytes("RGB", (W, H), frame(i)).save(os.path.join(BUILD, f"still_{i:04d}.jpg"), quality=90)
        return
    n = int(DUR * FPS)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", os.path.join(BUILD, "mix.wav"), "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
           "-shortest", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(int(os.environ.get("JOBS", os.cpu_count()))) as pool:
        for k, buf in enumerate(pool.imap(frame, range(n), chunksize=2)):
            enc.stdin.write(buf)
            if k % 30 == 0:
                print(f"frame {k}/{n}", flush=True)
    enc.stdin.close(); enc.wait()
    print("wrote", out)

if __name__ == "__main__":
    main()
