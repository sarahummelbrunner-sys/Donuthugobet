"""DonutHugoBet v2: premium motion-design cut (1080x1920, 30 fps, motion blur).

Pipeline: SCRIPT=v2 tts.py -> audio_v2.py -> this file.
"""
import math, os, subprocess, sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from timeline_v2 import EV, DUR, L, BUILD
from voxel import render, look_at
import worlds

W, H, FPS = 1080, 1920, 30
SUB = int(os.environ.get("SUBFRAMES", 3))       # temporal samples per frame (motion blur)
HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "fonts")

PINK = (255, 79, 160)
PINK2 = (255, 150, 200)
GOLD = (255, 190, 40)
GOLD2 = (255, 225, 130)
GREEN = (52, 230, 120)
CYAN = (90, 230, 255)
WHITE = (255, 255, 255)
MUTED = (185, 172, 210)
BG = (10, 6, 20)
INK = (16, 10, 28)

FARM, G_FARM = worlds.farm_world()
DONUT, DONUT_C = worlds.donut_world()
HUGO, HUGO_C = worlds.hugo_world()

# ================================================================= math / easing
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def lerp(a, b, k): return a + (b - a) * k
def prog(t, a, d): return clamp((t - a) / d)
def eo(k): k = clamp(k); return 1 - (1 - k) ** 3                 # ease out cubic
def eo5(k): k = clamp(k); return 1 - (1 - k) ** 5                # ease out quint
def ei(k): k = clamp(k); return k ** 3
def eio(k): k = clamp(k); return 4 * k ** 3 if k < 0.5 else 1 - (-2 * k + 2) ** 3 / 2
def eback(k, s=1.7):
    k = clamp(k) - 1
    return k * k * ((s + 1) * k + s) + 1
def spring(k, f=4.5, d=6.0):
    k = max(0.0, k)
    return 1 - math.exp(-d * k) * math.cos(f * 2 * math.pi * k * 0.5)

# ================================================================= fonts / text
@lru_cache(None)
def F(size, weight="Black", fam="Unbounded"):
    f = ImageFont.truetype(os.path.join(FONTS, f"{fam}.ttf"), int(size))
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f

def _fill_img(size, fill):
    if isinstance(fill, tuple) and len(fill) == 2 and isinstance(fill[0], tuple):
        a, b = fill                                   # vertical gradient
        g = np.linspace(0, 1, size[1])[:, None, None]
        arr = np.array(a, np.float32) * (1 - g) + np.array(b, np.float32) * g
        arr = np.repeat(arr, size[0], 1)
        return Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    return Image.new("RGBA", size, fill[:3] + (255,))

@lru_cache(None)
def text(s, size, fill=WHITE, weight="Black", fam="Unbounded", track=0, outline=0, shadow=True, glow=None):
    """Tight RGBA text image. fill: rgb or ((top),(bottom)) gradient. outline>0 => stroke-only text."""
    f = F(size, weight, fam)
    pad = int(size * 0.35) + outline
    xs = [0.0]
    for i, ch in enumerate(s):
        xs.append(xs[-1] + f.getlength(ch) + (track if i < len(s) - 1 else 0))
    tw = int(xs[-1])
    asc, desc = f.getmetrics()
    size_px = (tw + 2 * pad, asc + desc + 2 * pad)
    mask = Image.new("L", size_px, 0)
    md = ImageDraw.Draw(mask)
    for i, ch in enumerate(s):
        if outline:
            md.text((pad + xs[i], pad), ch, font=f, fill=0, stroke_width=outline, stroke_fill=255)
        else:
            md.text((pad + xs[i], pad), ch, font=f, fill=255)
    if outline:  # remove the glyph interior so only the stroke remains
        inner = Image.new("L", size_px, 0)
        idr = ImageDraw.Draw(inner)
        for i, ch in enumerate(s):
            idr.text((pad + xs[i], pad), ch, font=f, fill=255)
        mask = Image.fromarray(np.clip(np.asarray(mask, np.int16) - np.asarray(inner, np.int16), 0, 255).astype(np.uint8))
    im = _fill_img(size_px, fill)
    im.putalpha(mask)
    out = Image.new("RGBA", size_px, (0, 0, 0, 0))
    if glow:
        g = Image.new("RGBA", size_px, glow + (0,))
        g.putalpha(mask.filter(ImageFilter.GaussianBlur(size * 0.18)).point(lambda v: min(255, int(v * 1.6))))
        out.alpha_composite(g)
    if shadow:
        sh = Image.new("RGBA", size_px, (0, 0, 0, 0))
        sm = mask.filter(ImageFilter.GaussianBlur(size * 0.06)).point(lambda v: int(v * 0.55))
        sh.putalpha(sm)
        tmp = Image.new("RGBA", size_px, (0, 0, 0, 0))
        tmp.alpha_composite(sh, (0, int(size * 0.05)))
        out.alpha_composite(tmp)
    out.alpha_composite(im)
    return out

def glyphs(s, size, fills, weight="Black", fam="Unbounded", track=0):
    """Per-glyph images + x offsets (for staggered letter animation)."""
    f = F(size, weight, fam)
    out, x = [], 0.0
    for i, ch in enumerate(s):
        if ch != " ":
            out.append((text(ch, size, fills[i], weight, fam), x))
        x += f.getlength(ch) + track
    return out, x - track

# ================================================================= compositing
def comp(base, im, x, y, alpha=1.0):
    """Alpha-composite im with top-left at (x, y), clipped to base."""
    x, y = int(round(x)), int(round(y))
    if alpha <= 0.003:
        return
    if alpha < 0.997:
        im = im.copy()
        a = np.asarray(im.getchannel("A"), np.float32) * alpha
        im.putalpha(Image.fromarray(a.astype(np.uint8)))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(base.width, x + im.width), min(base.height, y + im.height)
    if x1 <= x0 or y1 <= y0:
        return
    if (x0, y0, x1, y1) != (x, y, x + im.width, y + im.height):
        im = im.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    base.alpha_composite(im, (x0, y0))

def place(base, im, cx, cy, scale=1.0, alpha=1.0, rot=0.0):
    if scale <= 0.01:
        return
    if abs(scale - 1) > 0.002:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BICUBIC)
    if rot:
        im = im.rotate(rot, Image.BICUBIC, expand=True)
    comp(base, im, cx - im.width / 2, cy - im.height / 2, alpha)

def reveal(base, im, x, y, k, anchor="c", rise=1.0, alpha_in=True):
    """Masked slide-up reveal: im rises from below its own baseline box. k: 0..1 (eased inside)."""
    if k <= 0:
        return
    e = eo5(k)
    if anchor == "c":
        x = x - im.width / 2
    elif anchor == "r":
        x = x - im.width
    y = y - im.height / 2
    shift = int((1 - e) * im.height * 0.75 * rise)
    vis = im.crop((0, 0, im.width, im.height - shift)) if shift > 0 else im
    if vis.height > 0:
        comp(base, vis, x, y + shift, alpha=clamp(k * 3) if alpha_in else 1)

def hide(base_t, t, a, d=0.25):
    """Exit factor (1 -> 0) starting at a."""
    return 1 - eo(prog(t, a, d))

@lru_cache(None)
def aa_rrect(w, h, r, fill, outline=None, ow=0):
    s = 3
    im = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w * s - 1, h * s - 1], r * s, fill=fill,
                                         outline=outline, width=ow * s)
    return im.resize((w, h), Image.LANCZOS)

@lru_cache(None)
def aa_circle(d, fill, outline=None, ow=0):
    s = 3
    im = Image.new("RGBA", (d * s, d * s), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([0, 0, d * s - 1, d * s - 1], fill=fill, outline=outline, width=ow * s)
    return im.resize((d, d), Image.LANCZOS)

@lru_cache(None)
def soft_shadow(w, h, r, blur=40, a=150):
    pad = blur * 2
    im = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([pad, pad, pad + w, pad + h], r, fill=(0, 0, 0, a))
    return im.filter(ImageFilter.GaussianBlur(blur))

@lru_cache(None)
def rr_mask(w, h, r):
    return aa_rrect(w, h, r, (255, 255, 255, 255)).getchannel("A")

def ring(base, cx, cy, radius, width, col, alpha):
    if alpha <= 0 or radius <= 0:
        return
    R = int(radius + width + 3)
    x0, y0 = int(cx - R), int(cy - R)
    yy, xx = np.mgrid[0:2 * R, 0:2 * R].astype(np.float32)
    d = np.abs(np.hypot(xx + x0 - cx, yy + y0 - cy) - radius) - width / 2
    a = np.clip(0.5 - d, 0, 1) * 255 * alpha
    im = Image.new("RGBA", (2 * R, 2 * R), col + (0,))
    im.putalpha(Image.fromarray(a.astype(np.uint8)))
    comp(base, im, x0, y0)

def persp(im, ry=0.0, rx=0.0, f=2200.0):
    """Rotate a flat card in 3D (radians) and project with perspective."""
    w, h = im.size
    pts = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], np.float64)
    cy, sy, cx, sx = math.cos(ry), math.sin(ry), math.cos(rx), math.sin(rx)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    p = pts @ (Rx @ Ry).T
    proj = p[:, :2] * (f / (f + p[:, 2:3]))
    mn = proj.min(0); mx = proj.max(0)
    ow, oh = int(math.ceil(mx[0] - mn[0])) + 4, int(math.ceil(mx[1] - mn[1])) + 4
    dst = proj - mn + 2
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    c = np.linalg.solve(np.array(A), np.array(B))
    return im.transform((ow, oh), Image.PERSPECTIVE, tuple(c), Image.BICUBIC)

# ================================================================= background
def _blob(col, r):
    S = 96
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    g = np.exp(-(((xx - S / 2) ** 2 + (yy - S / 2) ** 2) / (2 * (S * r) ** 2)))
    return g[..., None] * np.array(col, np.float32)[None, None]

BW, BH = 270, 480
_by, _bx = np.mgrid[0:BH, 0:BW].astype(np.float32)

@lru_cache(None)
def dot_grid():
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(30, H, 60):
        for x in range(30, W, 60):
            d.ellipse([x - 1.6, y - 1.6, x + 1.6, y + 1.6], fill=(255, 255, 255, 22))
    return im

VIG = None
def vignette():
    global VIG
    if VIG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        VIG = np.clip(1 - 0.5 * (((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2), 0.35, 1)[..., None]
    return VIG

def aurora(t, mood):
    """Animated soft gradient light. mood: dict col->intensity."""
    acc = np.zeros((BH, BW, 3), np.float32) + np.array(BG, np.float32)
    blobs = [
        (PINK, 0.28 + 0.06 * math.sin(t * 0.7), 0.25 + 0.1 * math.sin(t * 0.5), 0.22, mood.get("pink", 0.5)),
        (GOLD, 0.78 + 0.05 * math.cos(t * 0.6), 0.72 + 0.08 * math.sin(t * 0.4 + 1), 0.24, mood.get("gold", 0.4)),
        ((120, 60, 255), 0.65 + 0.1 * math.sin(t * 0.3 + 2), 0.2 + 0.05 * math.cos(t * 0.5), 0.26, mood.get("violet", 0.45)),
        (PINK, 0.15 + 0.05 * math.cos(t * 0.45), 0.85, 0.2, mood.get("pink2", 0.25)),
    ]
    for col, bx, by, r, inten in blobs:
        if inten <= 0:
            continue
        sig = r * BW * 1.25
        g = np.exp(-(((_bx - bx * BW) ** 2 + (_by - by * BH) ** 2) / (2 * sig ** 2)))
        acc += g[..., None] * np.array(col, np.float32) * inten * 0.55
    img = Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR).convert("RGBA")
    img.alpha_composite(dot_grid())
    return img

# ================================================================= brand art
@lru_cache(None)
def donut_icon(size):
    S = size * 4
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = S / 2
    d.ellipse([c - 0.47 * S, c - 0.44 * S, c + 0.47 * S, c + 0.50 * S], fill=(150, 80, 30, 255))
    d.ellipse([c - 0.47 * S, c - 0.47 * S, c + 0.47 * S, c + 0.47 * S], fill=(225, 155, 80, 255))
    pts = []
    for k in range(361):
        th = math.radians(k)
        ro = (0.405 + 0.022 * math.sin(7 * th) + 0.02 * max(0, math.sin(3 * th + 1)) ** 4) * S
        pts.append((c + ro * math.cos(th), c + ro * math.sin(th)))
    d.polygon(pts, fill=PINK + (255,))
    hl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(hl).arc([c - 0.32 * S, c - 0.32 * S, c + 0.32 * S, c + 0.32 * S], 200, 290,
                           fill=(255, 210, 235, 210), width=int(0.045 * S))
    im.alpha_composite(hl.filter(ImageFilter.GaussianBlur(S * 0.008)))
    r = np.random.default_rng(5)
    cols = [(255, 255, 255), (255, 225, 70), (100, 205, 255), (130, 240, 140), (255, 255, 255), (175, 120, 255)]
    for k in range(26):
        th = r.uniform(0, 2 * math.pi); rr = r.uniform(0.25, 0.37) * S
        x, y = c + rr * math.cos(th), c + rr * math.sin(th)
        a = r.uniform(0, math.pi); l = 0.034 * S
        dx, dy = math.cos(a) * l, math.sin(a) * l
        d.line([x - dx, y - dy, x + dx, y + dy], fill=cols[k % 6] + (255,), width=int(0.024 * S))
    d.ellipse([c - 0.205 * S, c - 0.205 * S, c + 0.205 * S, c + 0.205 * S], fill=(225, 155, 80, 255))
    d.ellipse([c - 0.185 * S, c - 0.175 * S, c + 0.185 * S, c + 0.195 * S], fill=(150, 80, 30, 255))
    mask = Image.new("L", (S, S), 255)
    ImageDraw.Draw(mask).ellipse([c - 0.17 * S, c - 0.17 * S, c + 0.17 * S, c + 0.17 * S], fill=0)
    im.putalpha(Image.fromarray(np.minimum(np.asarray(im.getchannel("A")), np.asarray(mask))))
    return im.resize((size, size), Image.LANCZOS)

@lru_cache(None)
def hugo_coin(size):
    S = size * 4
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = S / 2
    d.ellipse([c - 0.48 * S, c - 0.46 * S, c + 0.48 * S, c + 0.50 * S], fill=(150, 90, 5, 255))
    d.ellipse([c - 0.48 * S, c - 0.48 * S, c + 0.48 * S, c + 0.48 * S], fill=(215, 145, 20, 255))
    d.ellipse([c - 0.43 * S, c - 0.43 * S, c + 0.43 * S, c + 0.43 * S], fill=GOLD + (255,))
    d.ellipse([c - 0.36 * S, c - 0.36 * S, c + 0.36 * S, c + 0.36 * S], outline=(215, 145, 20, 255), width=int(0.018 * S))
    f = F(int(0.42 * S), "Black")
    bb = d.textbbox((0, 0), "H", font=f)
    ox, oy = c - (bb[0] + bb[2]) / 2, c - (bb[1] + bb[3]) / 2
    d.text((ox, oy + 0.012 * S), "H", font=f, fill=(190, 115, 5, 255))
    d.text((ox, oy), "H", font=f, fill=(255, 240, 175, 255))
    hl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(hl).arc([c - 0.40 * S, c - 0.40 * S, c + 0.40 * S, c + 0.40 * S], 195, 260,
                           fill=(255, 255, 235, 220), width=int(0.03 * S))
    im.alpha_composite(hl.filter(ImageFilter.GaussianBlur(S * 0.008)))
    return im.resize((size, size), Image.LANCZOS)

@lru_cache(None)
def donut_coin(size):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    im.alpha_composite(aa_circle(size, (150, 30, 90, 255)))
    inner = aa_circle(int(size * 0.92), (255, 140, 195, 255))
    im.alpha_composite(inner, (int(size * 0.04), int(size * 0.04)))
    ic = donut_icon(int(size * 0.72))
    im.alpha_composite(ic, (int(size * 0.14), int(size * 0.14)))
    return im

def coin_face(size, side, squash):
    im = donut_coin(size) if side == "donut" else hugo_coin(size)
    w = max(2, int(size * abs(squash)))
    return im.resize((w, size), Image.BICUBIC)

@lru_cache(None)
def bet_pill(scale=1.0):
    fs = int(96 * scale)
    t1 = text("BET", fs, INK, shadow=False); t2 = text(".NET", fs, (215, 50, 130), shadow=False)
    pad = int(fs * 0.35)
    w = t1.width + t2.width - 4 * pad + int(fs * 1.0); h = int(fs * 1.62)
    pill = Image.new("RGBA", (w + 80, h + 80), (0, 0, 0, 0))
    gl = Image.new("RGBA", pill.size, (0, 0, 0, 0))
    ImageDraw.Draw(gl).rounded_rectangle([40, 40, 40 + w, 40 + h], h // 2, fill=(255, 255, 255, 110))
    pill.alpha_composite(gl.filter(ImageFilter.GaussianBlur(22)))
    pill.alpha_composite(aa_rrect(w, h, h // 2, (255, 255, 255, 255)), (40, 40))
    x = 40 + int(fs * 0.5) - pad
    y = 40 + h // 2 - t1.height // 2 + int(fs * 0.04)
    pill.alpha_composite(t1, (x, y))
    pill.alpha_composite(t2, (x + t1.width - 2 * pad, y))
    return pill

def logo(base, t, t0, cx, cy, scale=1.0):
    """Animated logo build: rings + icon spring + staggered letters + pill."""
    u = t - t0
    if u < 0:
        return
    for i in range(3):
        k = prog(u, i * 0.08, 0.9)
        if 0 < k < 1:
            ring(base, cx, cy - 250 * scale, lerp(120, 760, eo(k)) * scale, (10 - 6 * k) * scale,
                 (PINK, GOLD, WHITE)[i], (1 - k) * 0.9)
    s_icon = spring(u / 0.9) if u < 0.9 else 1.0
    bob = math.sin(u * 2.2) * 8 * scale
    ic = donut_icon(int(380 * scale))
    place(base, ic, cx, cy - 250 * scale + bob, scale=max(0.01, s_icon), rot=(1 - eo(u / 0.8)) * 120 + u * 6)
    fills = [PINK] * 5 + [GOLD] * 4
    gl, tw = _wordmark_glyphs(scale)
    x0 = cx - tw / 2
    for i, (g, gx) in enumerate(gl):
        k = prog(u, 0.12 + i * 0.035, 0.45)
        reveal(base, g, x0 + gx + g.width / 2 - int(int(104 * scale) * 0.35), cy + 40 * scale + bob, k)
    k = prog(u, 0.5, 0.5)
    if k > 0:
        place(base, bet_pill(scale), cx, cy + 190 * scale + bob, scale=0.6 + 0.4 * eback(k, 2.2), alpha=clamp(k * 4))

@lru_cache(None)
def _wordmark_glyphs(scale):
    fills = tuple([((255, 120, 185), (235, 50, 140))] * 5 + [((255, 220, 110), (245, 160, 20))] * 4)
    return glyphs("DONUTHUGO", int(104 * scale), fills, track=int(1 * scale))

# ================================================================= voxel panels
_VOX_CACHE = {}
def vox_frame(key, fi, fn):
    """Voxel renders are expensive: one per output frame, shared across motion-blur subframes."""
    k = (key, fi)
    if k not in _VOX_CACHE:
        if len(_VOX_CACHE) > 8:
            _VOX_CACHE.clear()
        _VOX_CACHE[k] = fn()
    return _VOX_CACHE[k]

def vox_img(world, pos, yaw, pitch, fov, w=300, h=400, fog=150, bloom=0.5):
    col, em = render(world, pos, yaw, pitch, W=w, H=h, fov=fov, fog_dist=fog)
    img = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8))
    if em.max() > 0 and bloom:
        b = Image.fromarray(np.clip(col * em[..., None] * 1.3, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))
        img = Image.fromarray(np.clip(np.asarray(img, np.float32) + np.asarray(b, np.float32) * bloom, 0, 255).astype(np.uint8))
    return img

def panel(src, w, h, r=56, border=True):
    """Rounded media panel with soft shadow + hairline border."""
    img = src.resize((w, h), Image.NEAREST) if src.width < w else src.resize((w, h), Image.LANCZOS)
    img = img.convert("RGBA")
    img.putalpha(rr_mask(w, h, r))
    if border:
        img.alpha_composite(aa_rrect(w, h, r, (0, 0, 0, 0), (255, 255, 255, 70), 3))
    sh = soft_shadow(w, h, r)
    out = Image.new("RGBA", sh.size, (0, 0, 0, 0))
    out.alpha_composite(sh, (0, 30))
    p = (sh.width - w) // 2
    out.alpha_composite(img, (p, p))
    return out

def farm_view(fi, t):
    def f():
        u = t
        p = (47.5, G_FARM + 4.2 + 0.06 * math.sin(u * 9), 18 + 4.5 * u)
        return vox_img(FARM, p, 0.08 * math.sin(u * 0.8), -0.3, 66)
    return vox_frame("farm", fi, f)

def donut_view(fi, p):
    def f():
        cx, cy, cz = DONUT_C
        a = -0.8 + 0.9 * p
        pos = (cx + 50 * math.sin(a), cy + 24, cz - 50 * math.cos(a))
        y, pt = look_at(pos, (cx, cy - 3, cz))
        return vox_img(DONUT, pos, y, pt, 58, fog=170)
    return vox_frame("donut", fi, f)

def hugo_view(fi, p):
    def f():
        cx, cy, cz = HUGO_C
        a = 0.55 - 0.8 * p
        pos = (cx + 44 * math.sin(a), cy + 2 + 4 * p, cz - 44 * math.cos(a))
        y, pt = look_at(pos, (cx, cy - 4, cz))
        return vox_img(HUGO, pos, y, pt, 58, fog=170)
    return vox_frame("hugo", fi, f)

# ================================================================= reusable UI bits
@lru_cache(None)
def chip(label, fg=WHITE, bg=(255, 255, 255, 26), fs=40, weight="SemiBold", fam="Inter", icon=None, border=(255, 255, 255, 60)):
    f = F(fs, weight, fam)
    tw = int(f.getlength(label)); h = int(fs * 2.0)
    iw = h - 16 if icon is not None else 0
    w = tw + int(fs * 1.4) + iw
    im = aa_rrect(w, h, h // 2, bg, border, 2).copy()
    if icon is not None:
        im.alpha_composite(donut_icon(iw - 12), (14, 14))
    ImageDraw.Draw(im).text((int(fs * 0.7) + iw, h / 2), label, font=f, fill=fg + (255,), anchor="lm")
    return im

@lru_cache(None)
def check_icon(s, col=GREEN):
    im = aa_circle(s, col + (255,)).copy()
    big = Image.new("RGBA", (s * 3, s * 3), (0, 0, 0, 0))
    ImageDraw.Draw(big).line([(0.28 * s * 3, 0.52 * s * 3), (0.44 * s * 3, 0.67 * s * 3), (0.73 * s * 3, 0.35 * s * 3)],
                             fill=WHITE + (255,), width=int(s * 0.11 * 3), joint="curve")
    im.alpha_composite(big.resize((s, s), Image.LANCZOS))
    return im

@lru_cache(None)
def bolt_icon(s, col=GOLD):
    big = Image.new("RGBA", (s * 3, s * 3), (0, 0, 0, 0))
    pts = [(0.58, 0.04), (0.18, 0.56), (0.46, 0.56), (0.36, 0.96), (0.82, 0.40), (0.54, 0.40), (0.66, 0.04)]
    ImageDraw.Draw(big).polygon([(x * s * 3, y * s * 3) for x, y in pts], fill=col + (255,))
    return big.resize((s, s), Image.LANCZOS)

@lru_cache(None)
def arrow_icon(s, up, col):
    im = aa_circle(s, col + (40,), col + (255,), 3).copy()
    big = Image.new("RGBA", (s * 3, s * 3), (0, 0, 0, 0))
    d = ImageDraw.Draw(big); S = s * 3
    if up:
        d.line([(S / 2, S * 0.72), (S / 2, S * 0.3)], fill=col + (255,), width=int(S * 0.08))
        d.line([(S * 0.32, S * 0.46), (S / 2, S * 0.28), (S * 0.68, S * 0.46)], fill=col + (255,), width=int(S * 0.08), joint="curve")
    else:
        d.line([(S / 2, S * 0.28), (S / 2, S * 0.70)], fill=col + (255,), width=int(S * 0.08))
        d.line([(S * 0.32, S * 0.54), (S / 2, S * 0.72), (S * 0.68, S * 0.54)], fill=col + (255,), width=int(S * 0.08), joint="curve")
    im.alpha_composite(big.resize((s, s), Image.LANCZOS))
    return im

@lru_cache(None)
def glass(w, h, r=44, accent=None, a=200):
    pad = 50
    im = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    im.alpha_composite(soft_shadow(w, h, r, 30, 120), (pad - 60, pad - 60 + 20))
    if accent:
        gl = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(gl).rounded_rectangle([pad, pad, pad + w, pad + h], r, outline=accent + (255,), width=8)
        im.alpha_composite(gl.filter(ImageFilter.GaussianBlur(18)))
    im.alpha_composite(aa_rrect(w, h, r, (28, 18, 46, a), (accent or (255, 255, 255)) + (150 if accent else 45,), 2), (pad, pad))
    top = aa_rrect(w - 4, h // 2, r - 2, (255, 255, 255, 12))
    im.alpha_composite(top, (pad + 2, pad + 2))
    return im

def headline_stack(base, t, words, x, y0, lh, anchor="l"):
    """words: list of (img, t_reveal). Lines stacked from y0."""
    for i, (im, tr) in enumerate(words):
        reveal(base, im, x if anchor != "c" else x, y0 + i * lh, prog(t, tr, 0.42),
               anchor="c" if anchor == "c" else "l")

def money(v):
    return f"${v:,.0f}"

# ================================================================= phone mockup
PW, PH = 640, 1300
SW, SH = PW - 36, PH - 36

@lru_cache(None)
def phone_frame():
    im = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))
    im.alpha_composite(aa_rrect(PW, PH, 96, (40, 36, 52, 255)))
    im.alpha_composite(aa_rrect(PW - 8, PH - 8, 92, (8, 6, 14, 255)), (4, 4))
    return im

@lru_cache(None)
def phone_overlay():
    im = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))
    im.alpha_composite(aa_rrect(170, 46, 23, (0, 0, 0, 255)), (PW // 2 - 85, 34))
    gl = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))
    ImageDraw.Draw(gl).polygon([(0, 0), (PW * 0.55, 0), (0, PH * 0.45)], fill=(255, 255, 255, 14))
    m = rr_mask(PW, PH, 96)
    gl.putalpha(Image.fromarray(np.minimum(np.asarray(gl.getchannel("A")), np.asarray(m))))
    im.alpha_composite(gl)
    return im

@lru_cache(None)
def app_header(balance):
    im = Image.new("RGBA", (SW, 190), (0, 0, 0, 0))
    im.alpha_composite(donut_icon(64), (34, 96))
    d = ImageDraw.Draw(im)
    f = F(30, "Bold", "Unbounded")
    d.text((110, 128), "DONUT", font=f, fill=PINK + (255,), anchor="lm")
    d.text((110 + f.getlength("DONUT"), 128), "HUGO", font=f, fill=GOLD + (255,), anchor="lm")
    bp = aa_rrect(220, 64, 32, (255, 255, 255, 22), (255, 255, 255, 50), 2)
    im.alpha_composite(bp, (SW - 254, 96))
    d.text((SW - 254 + 110, 128), balance, font=F(28, "Bold", "Inter"), fill=GREEN + (255,), anchor="mm")
    return im

def screen_coinflip(u, fi):
    sc = Image.new("RGBA", (SW, SH), (14, 9, 24, 255))
    land = 0.85
    won = u > land
    bal = "$2.5M" if not won else "$3.0M"
    sc.alpha_composite(app_header(bal))
    d = ImageDraw.Draw(sc)
    d.text((40, 250), "Coinflip", font=F(52, "Black"), fill=WHITE + (255,), anchor="lm")
    d.text((40, 305), "Double or nothing", font=F(28, "Medium", "Inter"), fill=MUTED + (255,), anchor="lm")
    card = aa_rrect(SW - 60, 560, 40, (28, 18, 46, 255), (255, 255, 255, 30), 2)
    sc.alpha_composite(card, (30, 360))
    k = clamp(u / land)
    ang = (1 - (1 - k) ** 2.4) * math.pi * 8
    cs = math.cos(ang)
    hop = math.sin(k * math.pi) * 120
    face = coin_face(300, "donut" if cs >= 0 else "hugo", cs)
    comp(sc, face, SW / 2 - face.width / 2, 490 - hop)
    # shadow
    sh = Image.new("RGBA", (300, 60), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([0, 0, 300, 60], fill=(0, 0, 0, int(120 - 60 * hop / 120)))
    sh = sh.filter(ImageFilter.GaussianBlur(10))
    comp(sc, sh.resize((int(300 * (1 - hop / 260)), 40)), SW / 2 - 150 * (1 - hop / 260), 830)
    # side buttons
    for i, (lab, col) in enumerate((("DONUT", PINK), ("HUGO", GOLD))):
        sel = i == 0
        b = aa_rrect(255, 96, 30, col + (255,) if sel else (255, 255, 255, 16), col + (255,), 3)
        comp(sc, b, 40 + i * 285, 950)
        ImageDraw.Draw(sc).text((40 + i * 285 + 127, 998), lab, font=F(32, "Black"),
                                fill=(INK if sel else col) + (255,), anchor="mm")
    d = ImageDraw.Draw(sc)
    fld = aa_rrect(SW - 80, 96, 30, (255, 255, 255, 14), (255, 255, 255, 40), 2)
    sc.alpha_composite(fld, (40, 1070))
    d.text((70, 1118), "Bet", font=F(30, "Medium", "Inter"), fill=MUTED + (255,), anchor="lm")
    d.text((SW - 70, 1118), "$500,000", font=F(34, "Bold", "Inter"), fill=WHITE + (255,), anchor="rm")
    btn = aa_rrect(SW - 80, 104, 34, PINK + (255,))
    sc.alpha_composite(btn, (40, 1185))
    d.text((SW / 2, 1237), "FLIP", font=F(38, "Black"), fill=WHITE + (255,), anchor="mm")
    if won:
        v = u - land
        tst = toast("+ $500,000", "You won  x2")
        comp(sc, tst, SW / 2 - tst.width / 2, lerp(-tst.height, 150, eback(v / 0.3, 1.4)))
    return sc

@lru_cache(None)
def toast(big, small):
    w, h = SW - 60, 150
    im = aa_rrect(w, h, 40, (20, 60, 38, 250), GREEN + (255,), 3).copy()
    im.alpha_composite(check_icon(84), (30, 33))
    d = ImageDraw.Draw(im)
    d.text((136, 58), big, font=F(44, "Black"), fill=GREEN + (255,), anchor="lm")
    d.text((136, 108), small, font=F(28, "Medium", "Inter"), fill=(200, 240, 210, 255), anchor="lm")
    return im

def screen_crash(u):
    sc = Image.new("RGBA", (SW, SH), (14, 9, 24, 255))
    dur = EV["instant"] - EV["multi"]
    k = clamp(u / (dur * 0.86))
    m = math.exp(1.85 * k)
    cashed = k >= 1
    sc.alpha_composite(app_header("$3.0M" if not cashed else "$9.5M"))
    d = ImageDraw.Draw(sc)
    d.text((40, 250), "Crash", font=F(52, "Black"), fill=WHITE + (255,), anchor="lm")
    d.text((40, 305), "Cash out before it crashes", font=F(28, "Medium", "Inter"), fill=MUTED + (255,), anchor="lm")
    gx0, gy0, gx1, gy1 = 50, 600, SW - 50, 1080
    card = aa_rrect(SW - 60, 800, 40, (28, 18, 46, 255), (255, 255, 255, 30), 2)
    sc.alpha_composite(card, (30, 360))
    for i in range(5):
        y = gy0 + (gy1 - gy0) * i / 4
        d.line([gx0, y, gx1, y], fill=(255, 255, 255, 22), width=2)
    mmax = math.exp(1.85)
    pts = []
    for i in range(70):
        q = k * i / 69
        mv = math.exp(1.85 * q)
        pts.append((gx0 + (gx1 - gx0) * q, gy1 - (gy1 - gy0) * (mv - 1) / (mmax - 1)))
    lay = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lay)
    if k > 0.01:
        ld.polygon(pts + [(pts[-1][0], gy1), (gx0, gy1)], fill=(52, 230, 120, 50))
        gl = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
        ImageDraw.Draw(gl).line(pts, fill=GREEN + (255,), width=18, joint="curve")
        lay.alpha_composite(gl.filter(ImageFilter.GaussianBlur(10)))
        ImageDraw.Draw(lay).line(pts, fill=(210, 255, 225, 255), width=8, joint="curve")
        place(lay, aa_circle(34, WHITE + (255,), GREEN + (255,), 6), pts[-1][0], pts[-1][1])
    sc.alpha_composite(lay)
    mt = text(f"{m:.2f}x", 112, GREEN if not cashed else GOLD, shadow=False,
              glow=(0, 200, 90) if not cashed else (255, 160, 0))
    pop = 1 + 0.15 * (1 - eo((u - dur * 0.86) / 0.3)) if cashed else 1
    place(sc, mt, SW / 2, 480, scale=pop)
    btn = aa_rrect(SW - 80, 110, 36, (GREEN if not cashed else GOLD) + (255,))
    sc.alpha_composite(btn, (40, 1180))
    ImageDraw.Draw(sc).text((SW / 2, 1235), "CASHED OUT  6.36x" if cashed else f"CASH OUT  {money(500000 * m)}",
                            font=F(32 if not cashed else 34, "Black"), fill=INK + (255,), anchor="mm")
    return sc

def phone(base, screen, cx, cy, ry, rx, scale=1.0, alpha=1.0):
    ph = phone_frame().copy()
    scr = screen.copy()
    scr.putalpha(Image.fromarray(np.minimum(np.asarray(scr.getchannel("A")), np.asarray(rr_mask(SW, SH, 80)))))
    ph.alpha_composite(scr, (18, 18))
    ph.alpha_composite(phone_overlay())
    pad = 90
    big = Image.new("RGBA", (PW + 2 * pad, PH + 2 * pad), (0, 0, 0, 0))
    big.alpha_composite(soft_shadow(PW, PH, 96, 45, 170), (pad - 90, pad - 90 + 40))
    big.alpha_composite(ph, (pad, pad))
    if scale != 1:
        big = big.resize((int(big.width * scale), int(big.height * scale)), Image.BICUBIC)
    pim = persp(big, ry, rx)
    comp(base, pim, cx - pim.width / 2, cy - pim.height / 2, alpha)

# ================================================================= scenes
def s_hook(base, t, fi):
    out = hide(None, t, EV["smarter"] - 0.05, 0.3)
    # media panel
    k = eo5(prog(t, 0.0, 0.8))
    img = farm_view(fi, t)
    pn = panel(img, 880, 960, 60)
    pn = persp(pn, ry=(1 - k) * 0.35 + 0.04 * math.sin(t * 0.9), rx=(1 - k) * -0.12)
    place(base, pn, W / 2, 1270 - 120 * (1 - out) + (1 - k) * 160, scale=(0.94 + 0.06 * k) * (1 - 0.1 * (1 - out)),
          alpha=clamp(t / 0.25) * out)
    # in-panel money chip, ticking slowly
    m = 1204 + 2 * int(max(0, t - 0.4) / 0.6)
    c = _money_chip(m)
    place(base, c, 330, 1665 - 120 * (1 - out), alpha=clamp((t - 0.4) / 0.3) * out)
    ph = (t - 0.4) % 0.6
    if t > 0.9:
        place(base, _plus2(), 590, 1655 - 70 * ph - 120 * (1 - out), alpha=(1 - ph / 0.6) * out)
    # kinetic headline
    w = L[0]
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    reveal(lay, _hl("STILL", 118), 90, 320, prog(t, w[0]["t0"] - 0.06, 0.45), anchor="l")
    reveal(lay, _hl("GRINDING", 118, ((255, 255, 255), (205, 195, 230))), 90, 455, prog(t, w[1]["t0"] - 0.06, 0.45), anchor="l")
    reveal(lay, _hl("MONEY?", 118), 90, 590, prog(t, w[2]["t0"] - 0.06, 0.45), anchor="l")
    kc = prog(t, w[3]["t0"] - 0.05, 0.4)
    if kc > 0:
        place(lay, _smp_chip(), 90 + _smp_chip().width / 2, 715, scale=0.85 + 0.15 * eback(kc, 2), alpha=clamp(kc * 3))
    # strike through GRINDING
    ks = prog(t, w[4]["t1"] - 0.1, 0.35)
    if ks > 0:
        gw = _hl("GRINDING", 118).width - 90
        ImageDraw.Draw(lay).rounded_rectangle([130, 452, 130 + int(gw * eo5(ks)), 470], 9, fill=PINK + (255,))
    comp(base, lay, 0, -60 * (1 - out), alpha=out)

@lru_cache(None)
def _hl(s, size, fill=WHITE):
    return text(s, size, fill)

@lru_cache(None)
def _smp_chip():
    return chip("on DonutSMP", CYAN, (20, 60, 80, 200), 46, "Bold", "Inter", border=(90, 230, 255, 200))

@lru_cache(None)
def _money_chip(m):
    return chip(f"Money  ${m:,}", WHITE, (0, 0, 0, 150), 40, "Bold", "Inter")

@lru_cache(None)
def _plus2():
    return text("+$2", 50, GREEN, "Bold", "Inter", shadow=False)

class Floaters:
    """Depth-layered drifting coins and donuts with depth blur (parallax)."""
    def __init__(self, seed, n):
        r = np.random.default_rng(seed)
        self.items = []
        for i in range(n):
            z = r.uniform(0.35, 1.0)
            yy = r.uniform(0.04, 0.27) if i % 2 == 0 else r.uniform(0.70, 0.95)
            self.items.append(dict(x=r.uniform(0.05, 0.95) * W, y=yy * H, z=z,
                                   kind="donut" if i % 2 else "coin", ph=r.uniform(0, 6.28),
                                   vy=r.uniform(-60, -25), spin=r.uniform(1.2, 2.6)))
        self.items.sort(key=lambda d: d["z"])

    def draw(self, base, t, t0, alpha=1.0, spread=1.0):
        u = t - t0
        for it in self.items:
            z = it["z"]
            size = int(70 + 170 * z)
            blur = int((1 - z) * 10) // 3 * 3
            x = W / 2 + (it["x"] - W / 2) * spread + math.sin(u * 0.8 + it["ph"]) * 20 * z
            y = H / 2 + (it["y"] - H / 2) * spread + it["vy"] * u * z
            if it["kind"] == "coin":
                sq = math.cos(u * it["spin"] + it["ph"])
                im = _float_img("coin" if sq >= 0 else "donutcoin", size, blur)
                im = im.resize((max(2, int(im.width * (0.15 + 0.85 * abs(sq)))), im.height), Image.BICUBIC)
            else:
                im = _float_img("donut", size, blur)
                im = im.rotate(u * 30 * it["spin"] + it["ph"] * 50, Image.BICUBIC)
            place(base, im, x, y, alpha=alpha * (0.45 + 0.55 * z))

@lru_cache(None)
def _float_img(kind, size, blur):
    im = hugo_coin(size) if kind == "coin" else donut_coin(size) if kind == "donutcoin" else donut_icon(size)
    if blur:
        pad = blur * 3
        big = Image.new("RGBA", (size + 2 * pad, size + 2 * pad), (0, 0, 0, 0))
        big.alpha_composite(im, (pad, pad))
        im = big.filter(ImageFilter.GaussianBlur(blur))
    return im

FLOAT = Floaters(4, 14)

def s_smarter(base, t, fi):
    t0 = EV["smarter"]; out = hide(None, t, EV["drop"] - 0.12, 0.2)
    ein = eo5(prog(t, t0, 0.8))
    FLOAT.draw(base, t, t0, alpha=ein * out, spread=lerp(1.4, 1.0, ein) * lerp(1, 1.6, 1 - out))
    w = L[1]
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    reveal(lay, _hl2("THERE'S A"), W / 2, 760, prog(t, w[0]["t0"] - 0.05, 0.45))
    reveal(lay, _hl("SMARTER", 150, (GOLD2, PINK)), W / 2, 905, prog(t, w[2]["t0"] - 0.06, 0.45))
    reveal(lay, _hl("WAY.", 150), W / 2, 1075, prog(t, w[3]["t0"] - 0.06, 0.45))
    kl = prog(t, w[3]["t0"], 0.5)
    if kl > 0:
        ww = int(560 * eo5(kl))
        ImageDraw.Draw(lay).rounded_rectangle([W / 2 - ww / 2, 1180, W / 2 + ww / 2, 1192], 6, fill=GOLD + (255,))
    s = 1 + 0.25 * (1 - out)
    place(base, lay, W / 2, H / 2, scale=s, alpha=out)

@lru_cache(None)
def _hl2(s):
    return text(s, 50, MUTED, "SemiBold", "Inter", track=10, shadow=False)

def s_logo(base, t, fi):
    t0 = EV["drop"]
    out = hide(None, t, EV["donut"] - 0.05, 0.3)
    kin = prog(t, EV["meet"], 0.4)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    reveal(lay, _hl2("MEET"), W / 2, 420, kin)
    logo(lay, t, t0, W / 2, 960, 1.0)
    place(base, lay, W / 2, H / 2, scale=1 + 0.18 * (1 - out) + 0.02 * prog(t, t0, 2), alpha=out)

def s_money(base, t, fi, which):
    if which == "donut":
        t0, t1, li, col, grad, word = EV["donut"], EV["hugo"], 3, PINK, ((255, 130, 190), (230, 40, 130)), "DONUT"
    else:
        t0, t1, li, col, grad, word = EV["hugo"], EV["coin"], 4, GOLD, ((255, 230, 130), (240, 150, 10)), "HUGO"
    u = t - t0; d = t1 - t0
    kin = eo5(prog(t, t0, 0.55))
    kout = ei(prog(t, t1 - 0.2, 0.2))
    dx = (1 - kin) * W * 0.9 - kout * W * 0.9
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    # giant outline word behind
    ow = _outline(word, col)
    comp(lay, ow, W / 2 - ow.width / 2 + u * -40, 300 - ow.height / 2)
    # voxel panel
    img = donut_view(fi, u / d) if which == "donut" else hugo_view(fi, u / d)
    pn = panel(img, 720, 880, 56)
    pn = persp(pn, ry=-0.12 + 0.18 * (u / d), rx=0.05)
    place(lay, pn, W / 2, 920)
    # text: small line + big filled word + MONEY
    w = L[li]
    small = "GAMBLE WITH YOUR" if which == "donut" else "OR YOUR"
    reveal(lay, _hl2(small), W / 2, 1440, prog(t, w[0]["t0"] - 0.06, 0.4))
    wi = [x["w"] for x in w].index(word)
    kw = prog(t, w[wi]["t0"] - 0.05, 0.4)
    reveal(lay, _hl(word, 150, grad), W / 2, 1570, kw)
    km = prog(t, w[wi + 1]["t0"] - 0.05, 0.35)
    if km > 0:
        place(lay, _money_tag(which), W / 2, 1700, scale=0.8 + 0.2 * eback(km, 2), alpha=clamp(km * 3))
    comp(base, lay, dx, 0)

@lru_cache(None)
def _outline(word, col):
    return text(word, 300, col, outline=4, shadow=False)

@lru_cache(None)
def _money_tag(which):
    col = PINK if which == "donut" else GOLD
    return chip("MONEY", col, col + (30,), 48, "Black", "Unbounded", border=col + (160,))

def s_phone(base, t, fi):
    t0 = EV["coin"]; tm = EV["multi"]; t1 = EV["instant"]
    kin = eo5(prog(t, t0, 0.7))
    out = hide(None, t, t1 - 0.12, 0.25)
    # screen content: coinflip, then push to crash
    push = eio(prog(t, tm - 0.05, 0.4))
    if push <= 0:
        scr = screen_coinflip(t - t0 - 0.15, fi)
    elif push >= 1:
        scr = screen_crash(t - tm)
    else:
        a = screen_coinflip(t - t0 - 0.15, fi); b = screen_crash(t - tm)
        scr = Image.new("RGBA", (SW, SH), (14, 9, 24, 255))
        comp(scr, a, -SW * push, 0); comp(scr, b, SW * (1 - push), 0)
    ry = (1 - kin) * -0.6 + 0.10 * math.sin((t - t0) * 0.9) - 0.06
    rx = (1 - kin) * 0.25 + 0.04
    phone(base, scr, W / 2 + (1 - kin) * 200, 1110 + (1 - kin) * 500 + 200 * (1 - out), ry, rx, scale=0.92, alpha=out)
    # headlines above phone
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hk = 1 - eo(prog(t, tm - 0.1, 0.25))
    w5, w6 = L[5], L[6]
    if hk > 0:
        reveal(lay, _hl("FLIP A COIN", 96), W / 2, 330, prog(t, w5[0]["t0"] - 0.08, 0.45))
    else:
        reveal(lay, _hl2("RIDE THE"), W / 2, 270, prog(t, w6[0]["t0"] - 0.06, 0.4))
        reveal(lay, _hl("MULTIPLIER", 104, ((140, 255, 180), (40, 200, 100))), W / 2, 380, prog(t, w6[2]["t0"] - 0.06, 0.45))
    comp(base, lay, 0, 0, alpha=hk if hk > 0 else out)

def s_instant(base, t, fi):
    t0 = EV["instant"]; out = hide(None, t, EV["end"] - 0.1, 0.25)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    w7, w8 = L[7], L[8]
    kb = prog(t, w7[0]["t0"] - 0.1, 0.5)
    if kb > 0:
        place(lay, bolt_icon(150), W / 2, 520, scale=0.5 + 0.5 * eback(kb, 2.4), alpha=clamp(kb * 3))
    reveal(lay, _hl("INSTANT", 170, (GOLD2, GOLD)), W / 2, 740, prog(t, w7[0]["t0"] - 0.06, 0.45))
    pulse = prog(t, w8[0]["t0"] - 0.05, 0.3)
    if 0 < pulse < 1:
        ring(lay, W / 2, 740, 300 + 400 * eo(pulse), 6, GOLD, (1 - pulse) * 0.6)
    _tx_card(lay, t - (w7[1]["t0"] - 0.12), 1060, "Deposits", "Donut & Hugo money in", False, PINK, "0.4s")
    _tx_card(lay, t - (w8[1]["t0"] - 0.12), 1340, "Withdrawals", "Straight to your account", True, GOLD, "0.6s")
    place(base, lay, W / 2, H / 2, scale=1 + 0.15 * (1 - out), alpha=out)

def _tx_card(base, u, cy, title, sub, up, col, tm):
    if u < 0:
        return
    k = eo5(u / 0.5)
    g = glass(900, 230, 44, col)
    card = g.copy()
    ox = 50
    d = ImageDraw.Draw(card)
    card.alpha_composite(arrow_icon(120, up, col), (ox + 40, ox + 55))
    d.text((ox + 195, ox + 82), title, font=F(54, "Black"), fill=WHITE + (255,), anchor="lm")
    d.text((ox + 195, ox + 140), sub, font=F(30, "Medium", "Inter"), fill=MUTED + (255,), anchor="lm")
    bk = eo(clamp((u - 0.2) / 0.45))
    d.rounded_rectangle([ox + 195, ox + 180, ox + 195 + 520, ox + 192], 6, fill=(255, 255, 255, 30))
    if bk > 0:
        d.rounded_rectangle([ox + 195, ox + 180, ox + 195 + int(520 * bk), ox + 192], 6, fill=col + (255,))
    if u > 0.65:
        ck = eback((u - 0.65) / 0.25, 2.6)
        c = check_icon(100).resize((max(1, int(100 * ck)), max(1, int(100 * ck))), Image.LANCZOS)
        card.alpha_composite(c, (int(ox + 815 - c.width / 2), int(ox + 85 - c.height / 2)))
        tt = text(tm, 30, GREEN, "Bold", "Inter", shadow=False)
        card.alpha_composite(tt, (int(ox + 815 - tt.width / 2), int(ox + 165 - tt.height / 2)))
    place(base, card, W / 2 + (1 - k) * 260, cy, alpha=clamp(u / 0.2))

def s_end(base, t, fi):
    t0 = EV["end"]
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    w9 = L[9]
    big_out = 1 - eo(prog(t, EV["final"] - 0.08, 0.25))
    if big_out > 0:
        bl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        reveal(bl, _hl("PLAY", 190), W / 2, 820, prog(t, w9[0]["t0"] - 0.06, 0.45))
        reveal(bl, _hl("NOW", 190, (GOLD2, PINK)), W / 2, 1030, prog(t, w9[1]["t0"] - 0.06, 0.45))
        comp(lay, bl, 0, -260 * (1 - big_out), alpha=big_out)
    reveal(lay, _hl2("PLAY NOW AT"), W / 2, 330, prog(t, EV["final"] - 0.05, 0.45))
    logo(lay, t, EV["final"], W / 2, 780, 0.92)
    # url bar typing
    u = t - EV["final"]
    ka = prog(u, 0.55, 0.35)
    if ka > 0:
        n = int(len("donuthugobet.net") * clamp((u - 0.65) / 0.75))
        bar = _url(n, (int(u * 2.5) % 2 == 0) or n < 16)
        place(lay, bar, W / 2, 1185, scale=0.9 + 0.1 * eback(ka, 2), alpha=clamp(ka * 3))
    v = t - EV["tagline"]
    if v > 0:
        reveal(lay, _tag1(), W / 2, 1390, prog(v, 0, 0.45))
        reveal(lay, _tag2(), W / 2, 1490, prog(v, 0.12, 0.45))
        place(lay, _disc(), W / 2, 1745, alpha=clamp((v - 0.4) / 0.4) * 0.8)
    fade = 1 - prog(t, DUR - 0.35, 0.35)
    comp(base, lay, 0, 0, alpha=fade)

@lru_cache(None)
def _url(n, cursor):
    w, h = 820, 128
    im = Image.new("RGBA", (w + 80, h + 80), (0, 0, 0, 0))
    gl = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(gl).rounded_rectangle([40, 40, 40 + w, 40 + h], h // 2, fill=(255, 79, 160, 120))
    im.alpha_composite(gl.filter(ImageFilter.GaussianBlur(24)))
    im.alpha_composite(aa_rrect(w, h, h // 2, (255, 255, 255, 255)), (40, 40))
    d = ImageDraw.Draw(im)
    lock = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lock)
    ld.rounded_rectangle([12, 26, 48, 54], 7, fill=(40, 185, 95, 255))
    ld.arc([18, 6, 42, 40], 180, 360, fill=(40, 185, 95, 255), width=6)
    im.alpha_composite(lock, (40 + 42, 40 + 32))
    f = F(50, "Bold", "Inter")
    s = "donuthugobet.net"[:n]
    d.text((40 + 122, 40 + h / 2), s, font=f, fill=INK + (255,), anchor="lm")
    if cursor:
        x = 40 + 122 + f.getlength(s) + 6
        d.rounded_rectangle([x, 40 + 34, x + 5, 40 + h - 34], 2, fill=PINK + (255,))
    btn = aa_rrect(150, 92, 46, PINK + (255,))
    im.alpha_composite(btn, (40 + w - 168, 40 + 18))
    d.text((40 + w - 93, 40 + h / 2), "PLAY", font=F(32, "Black"), fill=WHITE + (255,), anchor="mm")
    return im

@lru_cache(None)
def _tag1():
    return _seg_line([("GAMBLE WITH ", WHITE), ("DONUT", PINK)], 66)

@lru_cache(None)
def _tag2():
    return _seg_line([("OR ", WHITE), ("HUGO", GOLD), (" MONEY", WHITE)], 66)

def _seg_line(parts, size):
    ims = [text(p, size, c) for p, c in parts]
    pad = int(size * 0.35)
    w = sum(i.width - 2 * pad for i in ims) + 2 * pad
    out = Image.new("RGBA", (w, ims[0].height), (0, 0, 0, 0))
    x = 0
    for i in ims:
        out.alpha_composite(i, (x, 0)); x += i.width - 2 * pad
    return out

@lru_cache(None)
def _disc():
    return text("18+  ·  Play responsibly", 30, (200, 190, 220), "Medium", "Inter", shadow=False)

# ================================================================= frame
MOODS = [
    (0.0, dict(pink=0.25, gold=0.15, violet=0.35, pink2=0.1)),
    (EV["smarter"], dict(pink=0.45, gold=0.55, violet=0.4, pink2=0.2)),
    (EV["drop"], dict(pink=0.9, gold=0.6, violet=0.5, pink2=0.4)),
    (EV["donut"], dict(pink=0.9, gold=0.15, violet=0.45, pink2=0.5)),
    (EV["hugo"], dict(pink=0.2, gold=0.95, violet=0.4, pink2=0.15)),
    (EV["coin"], dict(pink=0.55, gold=0.5, violet=0.5, pink2=0.3)),
    (EV["instant"], dict(pink=0.4, gold=0.85, violet=0.45, pink2=0.25)),
    (EV["end"], dict(pink=0.8, gold=0.6, violet=0.5, pink2=0.4)),
]

def mood_at(t):
    cur = MOODS[0][1]
    for i, (ts, m) in enumerate(MOODS):
        if t >= ts:
            prev = MOODS[i - 1][1] if i else m
            k = eio(prog(t, ts - 0.2, 0.6))
            cur = {c: lerp(prev[c], m[c], k) for c in m}
    return cur

def compose(t, fi):
    base = aurora(t, mood_at(t))
    if t < EV["smarter"] + 0.3:
        s_hook(base, t, fi)
    if EV["smarter"] - 0.1 <= t < EV["drop"]:
        s_smarter(base, t, fi)
    if EV["meet"] <= t < EV["donut"] + 0.3:
        s_logo(base, t, fi)
    if EV["donut"] <= t < EV["hugo"]:
        s_money(base, t, fi, "donut")
    if EV["hugo"] - 0.02 <= t < EV["coin"]:
        s_money(base, t, fi, "hugo")
    if EV["coin"] - 0.05 <= t < EV["instant"] + 0.15:
        s_phone(base, t, fi)
    if EV["instant"] - 0.05 <= t < EV["end"] + 0.2:
        s_instant(base, t, fi)
    if t >= EV["end"]:
        s_end(base, t, fi)
    if EV["drop"] + 1.2 < t < EV["end"] - 0.2:
        place(base, _wm(), W / 2, 120, alpha=0.9 * clamp((t - EV["drop"] - 1.2) / 0.3) * clamp((EV["end"] - 0.2 - t) / 0.2))
    a = np.asarray(base.convert("RGB"), np.float32)
    # flash on the drop and the final logo
    for ts, amt in ((EV["drop"], 0.85), (EV["final"], 0.6)):
        u = t - ts
        if 0 <= u < 0.35:
            f = amt * (1 - eo(u / 0.35))
            a = a * (1 - f) + 255 * f
    return a

@lru_cache(None)
def _wm():
    return chip("donuthugobet.net", WHITE, (10, 6, 20, 140), 32, "SemiBold", "Inter", icon="donut")

GRAIN = [np.random.default_rng(i).normal(0, 2.6, (H // 2, W // 2, 1)).astype(np.float32) for i in range(6)]

def frame(i):
    t = i / FPS
    if SUB > 1:
        offs = [(k / (SUB - 1) - 0.5) * 0.5 / FPS for k in range(SUB)]
        acc = sum(compose(t + o, i) for o in offs) / SUB
    else:
        acc = compose(t, i)
    acc *= vignette()
    lum = acc @ np.array([0.299, 0.587, 0.114], np.float32)
    acc = lum[..., None] + (acc - lum[..., None]) * 1.06
    acc += np.repeat(np.repeat(GRAIN[i % 6], 2, 0), 2, 1)
    return np.clip(acc, 0, 255).astype(np.uint8).tobytes()

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BUILD, "donuthugobet_v2.mp4")
    only = os.environ.get("FRAMES")
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
        for k, buf in enumerate(pool.imap(frame, range(n), chunksize=3)):
            enc.stdin.write(buf)
            if k % 30 == 0:
                print(f"frame {k}/{n}", flush=True)
    enc.stdin.close(); enc.wait()
    print("wrote", out)

if __name__ == "__main__":
    main()
