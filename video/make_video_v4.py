"""DonutHugoBet v4: v3 product-ad style with the server branding (blue/red logo) and a "Join DC in bio" end card.

Pipeline: TIMELINE=timeline_v4 BUILD_DIR=build_v4 python3 audio_v3.py -> this file.
"""
import math, os, subprocess, sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from timeline_v4 import EV, DUR

W, H, FPS = 1080, 1920, 30
SUB = int(os.environ.get("SUBFRAMES", 3))
HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "fonts")
BUILD = os.environ.get("BUILD_DIR", os.path.join(HERE, "build_v4"))
LOGO_PATH = os.path.join(HERE, "assets", "server_logo.png")

def _hex(v): v = v.lstrip("#"); return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))
# Brand colours: override with the website's exact values, e.g. BRAND_PRIMARY=#2f6bff BRAND_SECONDARY=#ff2a3d
BLUE = _hex(os.environ.get("BRAND_PRIMARY", "#3478ff"))      # Donut money
RED = _hex(os.environ.get("BRAND_SECONDARY", "#f0303a"))     # Hugo money
def tint(c, k): return tuple(min(255, int(v + (255 - v) * k)) for v in c)     # towards white
def shade(c, k): return tuple(int(v * (1 - k)) for v in c)                     # towards black
BLUE_L = tint(BLUE, 0.45)
GREEN = (48, 209, 88)
TXT = (245, 245, 247)
SUBT = (150, 150, 160)
CARD = (28, 28, 32)
CARD2 = (40, 40, 46)

# ================================================================= basics
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def lerp(a, b, k): return a + (b - a) * k
def lerpc(a, b, k): return tuple(int(lerp(x, y, k)) for x, y in zip(a, b))
def prog(t, a, d): return clamp((t - a) / d)
def eo(k): k = clamp(k); return 1 - (1 - k) ** 3
def eo5(k): k = clamp(k); return 1 - (1 - k) ** 5
def eio(k): k = clamp(k); return 4 * k ** 3 if k < 0.5 else 1 - (-2 * k + 2) ** 3 / 2
def eios(k): k = clamp(k); return 0.5 - 0.5 * math.cos(math.pi * k)
def eback(k, s=1.5):
    k = clamp(k) - 1
    return k * k * ((s + 1) * k + s) + 1

@lru_cache(None)
def F(size, weight="SemiBold", fam="Inter"):
    f = ImageFont.truetype(os.path.join(FONTS, f"{fam}.ttf"), int(size))
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f

def _fill(size, fill):
    if isinstance(fill, tuple) and len(fill) == 3 and isinstance(fill[0], str):
        _, a, b = fill                       # ("h", rgb, rgb) horizontal gradient
        g = np.linspace(0, 1, size[0])[None, :, None]
        arr = np.array(a, np.float32) * (1 - g) + np.array(b, np.float32) * g
        arr = np.repeat(arr, size[1], 0)
        return Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    return Image.new("RGBA", size, fill[:3] + (255,))

@lru_cache(None)
def text(s, size, fill=TXT, weight="SemiBold", fam="Inter", track=0.0):
    f = F(size, weight, fam)
    pad = int(size * 0.3)
    xs = [0.0]
    for i, ch in enumerate(s):
        xs.append(xs[-1] + f.getlength(ch) + (track if i < len(s) - 1 else 0))
    asc, desc = f.getmetrics()
    sz = (int(xs[-1]) + 2 * pad, asc + desc + 2 * pad)
    m = Image.new("L", sz, 0)
    d = ImageDraw.Draw(m)
    if track == 0:
        d.text((pad, pad), s, font=f, fill=255)
    else:
        for i, ch in enumerate(s):
            d.text((pad + xs[i], pad), ch, font=f, fill=255)
    im = _fill(sz, fill)
    im.putalpha(m)
    return im

def comp(base, im, x, y, alpha=1.0):
    x, y = int(round(x)), int(round(y))
    if alpha <= 0.003:
        return
    if alpha < 0.997:
        im = im.copy()
        im.putalpha(Image.fromarray((np.asarray(im.getchannel("A"), np.float32) * alpha).astype(np.uint8)))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(base.width, x + im.width), min(base.height, y + im.height)
    if x1 <= x0 or y1 <= y0:
        return
    if (x0, y0, x1, y1) != (x, y, x + im.width, y + im.height):
        im = im.crop((x0 - x, y0 - y, x1 - x, y1 - y))
    base.alpha_composite(im, (x0, y0))

def place(base, im, cx, cy, scale=1.0, alpha=1.0, blur=0.0):
    if scale <= 0.01 or alpha <= 0.003:
        return
    if abs(scale - 1) > 0.002:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BICUBIC)
    if blur > 0.4:
        im = blurred(im, blur)
    comp(base, im, cx - im.width / 2, cy - im.height / 2, alpha)

def blurred(im, r):
    p = int(r * 2.5) + 2
    big = Image.new("RGBA", (im.width + 2 * p, im.height + 2 * p), (0, 0, 0, 0))
    big.alpha_composite(im, (p, p))
    # premultiplied blur avoids dark fringes
    a = np.asarray(big, np.float32)
    pm = a.copy(); pm[..., :3] *= a[..., 3:4] / 255
    img = Image.fromarray(pm.astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))
    b = np.asarray(img, np.float32)
    al = b[..., 3:4]
    b[..., :3] = np.where(al > 0, b[..., :3] * 255 / np.maximum(al, 1), 0)
    return Image.fromarray(np.clip(b, 0, 255).astype(np.uint8))

@lru_cache(None)
def rrect(w, h, r, fill, outline=None, ow=0):
    s = 3
    im = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([0, 0, w * s - 1, h * s - 1], r * s, fill=fill, outline=outline, width=ow * s)
    return im.resize((w, h), Image.LANCZOS)

@lru_cache(None)
def circle(d, fill, outline=None, ow=0):
    s = 3
    im = Image.new("RGBA", (d * s, d * s), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([0, 0, d * s - 1, d * s - 1], fill=fill, outline=outline, width=ow * s)
    return im.resize((d, d), Image.LANCZOS)

@lru_cache(None)
def rmask(w, h, r):
    return rrect(w, h, r, (255, 255, 255, 255)).getchannel("A")

@lru_cache(None)
def shadow(w, h, r, blur=50, a=170):
    p = blur * 2
    im = Image.new("RGBA", (w + 2 * p, h + 2 * p), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle([p, p, p + w, p + h], r, fill=(0, 0, 0, a))
    return im.filter(ImageFilter.GaussianBlur(blur))

def grad_rrect(w, h, r, a, b, angle_h=True):
    g = np.linspace(0, 1, w if angle_h else h)
    g = g[None, :, None] if angle_h else g[:, None, None]
    arr = np.array(a, np.float32) * (1 - g) + np.array(b, np.float32) * g
    arr = np.broadcast_to(arr, (h, w, 3)).copy()
    im = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    im.putalpha(rmask(w, h, r))
    return im

def homography(src, dst):
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B.append(v)
    return tuple(np.linalg.solve(np.array(A, float), np.array(B, float)))

def rot(ry, rx, rz):
    cy, sy, cx, sx, cz, sz = math.cos(ry), math.sin(ry), math.cos(rx), math.sin(rx), math.cos(rz), math.sin(rz)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry

def card3d(im, ry=0.0, rx=0.0, rz=0.0, f=2400.0):
    w, h = im.size
    P = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], float) @ rot(ry, rx, rz).T
    pr = P[:, :2] * (f / (f + P[:, 2:3]))
    mn = pr.min(0)
    ow, oh = (pr.max(0) - mn).astype(int) + 4
    dst = pr - mn + 2
    return im.transform((int(ow), int(oh)), Image.PERSPECTIVE, homography([(0, 0), (w, 0), (w, h), (0, h)], dst), Image.BICUBIC)

# ================================================================= background glow
BW, BH = 216, 384
_gy, _gx = np.mgrid[0:BH, 0:BW].astype(np.float32)
GRAIN = [np.random.default_rng(i).normal(0, 2.2, (H // 2, W // 2, 1)).astype(np.float32) for i in range(8)]

def glow_bg(glows):
    acc = np.zeros((BH, BW, 3), np.float32)
    for col, cx, cy, r, inten in glows:
        if inten <= 0.002:
            continue
        g = np.exp(-(((_gx - cx / W * BW) ** 2 + ((_gy - cy / H * BH) * 1.0) ** 2) / (2 * (r / W * BW) ** 2)))
        acc += g[..., None] * np.array(col, np.float32) * inten
    return Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC).convert("RGBA")

# ================================================================= brand
@lru_cache(None)
def server_logo(width):
    lg = Image.open(LOGO_PATH).convert("RGBA")
    k = width / lg.width
    return lg.resize((int(lg.width * k), int(lg.height * k)), Image.LANCZOS)

@lru_cache(None)
def wordmark(size):
    a = text("Donut", size, BLUE, "ExtraBold", track=-size * 0.02)
    b = text("Hugo", size, RED, "ExtraBold", track=-size * 0.02)
    c = text("Bet", size, TXT, "ExtraBold", track=-size * 0.02)
    pad = int(size * 0.3)
    w = a.width + b.width + c.width - 4 * pad
    out = Image.new("RGBA", (w, a.height), (0, 0, 0, 0))
    x = 0
    for im in (a, b, c):
        out.alpha_composite(im, (x, 0)); x += im.width - 2 * pad
    return out

@lru_cache(None)
def badge(kind, d):
    """Casino chip in the style of the server logo: striped rim, inner ring, letter."""
    col = BLUE if kind == "donut" else RED
    dark = tuple(int(c * 0.55) for c in col)
    S = d * 4
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.ellipse([0, 0, S - 1, S - 1], fill=dark + (255,))
    dr.ellipse([S * 0.03, S * 0.03, S * 0.97, S * 0.97], fill=col + (255,))
    for i in range(8):                                   # white rim stripes
        a0 = i * 45 - 9
        dr.pieslice([S * 0.03, S * 0.03, S * 0.97, S * 0.97], a0, a0 + 18, fill=(245, 248, 255, 255))
    dr.ellipse([S * 0.17, S * 0.17, S * 0.83, S * 0.83], fill=dark + (255,))
    dr.ellipse([S * 0.19, S * 0.19, S * 0.81, S * 0.81], fill=(245, 248, 255, 255))
    dr.ellipse([S * 0.22, S * 0.22, S * 0.78, S * 0.78], fill=col + (255,))
    dr.arc([S * 0.26, S * 0.26, S * 0.74, S * 0.74], 200, 290, fill=(255, 255, 255, 120), width=int(S * 0.03))
    im = im.resize((d, d), Image.LANCZOS)
    t = text("D" if kind == "donut" else "H", int(d * 0.36), (255, 255, 255), "ExtraBold")
    im.alpha_composite(t, (d // 2 - t.width // 2, d // 2 - t.height // 2))
    return im

@lru_cache(None)
def app_icon(d):
    im = rrect(d, d, int(d * 0.23), (14, 18, 34, 255)).copy()
    lg = server_logo(int(d * 0.92))
    im.alpha_composite(lg, ((d - lg.width) // 2, (d - lg.height) // 2))
    return im

def fmt(v): return f"{int(v):,}"

# ================================================================= phone
SW, SH = 560, 1200            # screen
PW, PH, PR, TH = 600, 1240, 100, 30

def status_bar(sc, col=TXT):
    d = ImageDraw.Draw(sc)
    d.text((70, 48), "9:41", font=F(30, "SemiBold"), fill=col + (255,), anchor="lm")
    for i in range(4):
        d.rounded_rectangle([SW - 150 + i * 11, 58 - 6 - i * 4, SW - 143 + i * 11, 58], 2, fill=col + (255,))
    d.rounded_rectangle([SW - 95, 38, SW - 52, 58], 6, outline=col + (200,), width=2)
    d.rounded_rectangle([SW - 92, 41, SW - 62, 55], 3, fill=col + (255,))

@lru_cache(None)
def lock_bg():
    sc = Image.new("RGBA", (SW, SH), (6, 5, 10, 255))
    g = glow_bg_small(SW, SH, [(tuple(int(c * 0.5) for c in BLUE), SW * 0.3, SH * 0.85, SW * 0.7, 1.0),
                               (tuple(int(c * 0.5) for c in RED), SW * 0.9, SH * 1.0, SW * 0.5, 0.8)])
    sc.alpha_composite(g)
    lg = server_logo(430)
    comp(sc, blurred(lg, 3), SW / 2 - lg.width / 2 - 8, 760 - lg.height / 2 - 8, alpha=0.55)
    return sc

def glow_bg_small(w, h, glows):
    yy, xx = np.mgrid[0:h // 4, 0:w // 4].astype(np.float32) * 4
    acc = np.zeros((h // 4, w // 4, 3), np.float32)
    for col, cx, cy, r, inten in glows:
        g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)))
        acc += g[..., None] * np.array(col, np.float32) * inten
    return Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8)).resize((w, h), Image.BICUBIC).convert("RGBA")

def screen_lock(u):
    sc = lock_bg().copy()
    status_bar(sc)
    d = ImageDraw.Draw(sc)
    d.text((SW / 2, 175), "Sunday, October 4", font=F(30, "Medium"), fill=(235, 235, 240, 230), anchor="mm")
    d.text((SW / 2, 300), "9:41", font=F(176, "SemiBold"), fill=(245, 245, 250, 240), anchor="mm")
    k = prog(u, EV["notif"], 0.6)
    if k > 0:
        n = _notif()
        comp(sc, n, SW / 2 - n.width / 2, lerp(-n.height, 470, eback(k, 1.2)), alpha=clamp(k * 3))
    return sc

@lru_cache(None)
def _notif():
    w, h = 520, 150
    im = rrect(w, h, 36, (52, 50, 58, 235)).copy()
    im.alpha_composite(app_icon(84), (22, 33))
    d = ImageDraw.Draw(im)
    d.text((124, 52), "DonutHugoBet", font=F(28, "SemiBold"), fill=TXT + (255,), anchor="lm")
    d.text((w - 26, 52), "now", font=F(24, "Regular"), fill=(190, 190, 200, 255), anchor="rm")
    d.text((124, 96), "You won 2,000,000 Donut money", font=F(25, "Regular"), fill=(230, 230, 235, 255), anchor="lm")
    return im

def screen_app(u):
    """Home screen; u = scene time. Currency toggles Donut -> Hugo at EV['toggle']."""
    sc = Image.new("RGBA", (SW, SH), (9, 9, 11, 255))
    k = eio(prog(u, EV["toggle"], 0.5))
    accent = lerpc(BLUE, RED, k)
    sc.alpha_composite(glow_bg_small(SW, SH, [(accent, SW * 0.5, 330, 300, 0.22)]))
    status_bar(sc)
    d = ImageDraw.Draw(sc)
    lg = server_logo(70)
    comp(sc, lg, 30, 128 - lg.height / 2)
    wm = wordmark(30)
    comp(sc, wm, 108 - int(30 * 0.3), 128 - wm.height / 2)
    sc.alpha_composite(circle(56, (50, 50, 56, 255)), (SW - 96, 100))
    d.text((SW - 68, 128), "S", font=F(26, "SemiBold"), fill=TXT + (255,), anchor="mm")
    # segmented control
    sx, sy, sw_, sh_ = 30, 192, SW - 60, 72
    sc.alpha_composite(rrect(sw_, sh_, 36, (34, 34, 38, 255)), (sx, sy))
    kx = sx + 6 + (sw_ / 2 - 6) * k
    sc.alpha_composite(rrect(int(sw_ / 2 - 6), sh_ - 12, 30, accent + (255,)), (int(kx), sy + 6))
    for i, lab in enumerate(("Donut money", "Hugo money")):
        on = (1 - k) if i == 0 else k
        col = lerpc((170, 170, 180), (255, 255, 255), on)
        d.text((sx + sw_ / 4 + i * sw_ / 2, sy + sh_ / 2), lab, font=F(26, "SemiBold"), fill=col + (255,), anchor="mm")
    # balance card
    cy0 = 296
    ca = grad_rrect(SW - 60, 270, 40, tint(BLUE, 0.12), shade(BLUE, 0.3))
    cb = grad_rrect(SW - 60, 270, 40, tint(RED, 0.15), shade(RED, 0.25))
    comp(sc, ca, 30, cy0, 1 - k); comp(sc, cb, 30, cy0, k)
    lab_col = lerpc(tint(BLUE, 0.85), tint(RED, 0.85), k)
    num_col = (255, 255, 255)
    d.text((64, cy0 + 50), "Balance", font=F(26, "Medium"), fill=lab_col + (255,), anchor="lm")
    v = lerp(2_500_000, 1_840_000, k)
    d.text((64, cy0 + 125), fmt(v), font=F(66, "Bold"), fill=num_col + (255,), anchor="lm")
    bd = badge("donut" if k < 0.5 else "hugo", 64)
    sc.alpha_composite(bd, (SW - 30 - 34 - 64, cy0 + 26))
    d.text((64, cy0 + 178), "Donut money" if k < 0.5 else "Hugo money", font=F(24, "Medium"), fill=lab_col + (255,), anchor="lm")
    for i, lab in enumerate(("Deposit", "Withdraw")):
        b = rrect(190, 58, 29, (255, 255, 255, 240))
        sc.alpha_composite(b, (64 + i * 206, cy0 + 196))
        d.text((64 + i * 206 + 95, cy0 + 225), lab, font=F(24, "SemiBold"),
               fill=(lerpc(shade(BLUE, 0.3), shade(RED, 0.25), k)) + (255,), anchor="mm")
    # games
    d.text((36, 625), "Games", font=F(34, "Bold"), fill=TXT + (255,), anchor="lm")
    for i, (name, sub) in enumerate((("Coinflip", "x2 payout"), ("Crash", "up to 1000x"))):
        x = 30 + i * 258
        sc.alpha_composite(rrect(242, 220, 34, CARD + (255,)), (x, 660))
        if i == 0:
            sc.alpha_composite(badge("donut", 70), (x + 24, 686))
            sc.alpha_composite(badge("hugo", 70), (x + 70, 686))
        else:
            ic = Image.new("RGBA", (120, 70), (0, 0, 0, 0))
            ImageDraw.Draw(ic).line([(4, 64), (40, 52), (70, 34), (100, 14)], fill=GREEN + (255,), width=7, joint="curve")
            sc.alpha_composite(ic, (x + 24, 686))
        d.text((x + 24, 800), name, font=F(30, "Bold"), fill=TXT + (255,), anchor="lm")
        d.text((x + 24, 838), sub, font=F(22, "Medium"), fill=SUBT + (255,), anchor="lm")
    # recent wins
    d.text((36, 935), "Recent wins", font=F(34, "Bold"), fill=TXT + (255,), anchor="lm")
    rows = [("M", "Max", "Crash 6.1x", "+3,050,000"), ("L", "Lena", "Coinflip", "+800,000"), ("J", "Jonas", "Crash 2.4x", "+1,200,000")]
    for i, (a, n, g, p) in enumerate(rows):
        y = 975 + i * 86
        sc.alpha_composite(rrect(SW - 60, 74, 22, CARD + (255,)), (30, y))
        sc.alpha_composite(circle(46, (60, 60, 68, 255)), (46, y + 14))
        d.text((69, y + 37), a, font=F(22, "SemiBold"), fill=TXT + (255,), anchor="mm")
        d.text((108, y + 26), n, font=F(24, "SemiBold"), fill=TXT + (255,), anchor="lm")
        d.text((108, y + 52), g, font=F(20, "Medium"), fill=SUBT + (255,), anchor="lm")
        d.text((SW - 52, y + 37), p, font=F(24, "SemiBold"), fill=GREEN + (255,), anchor="rm")
    return sc

def phone_front(screen, gloss):
    im = rrect(PW, PH, PR, (14, 14, 16, 255), (90, 90, 100, 255), 3).copy()
    scr = screen.copy()
    scr.putalpha(Image.fromarray(np.minimum(np.asarray(scr.getchannel("A")), np.asarray(rmask(SW, SH, PR - 20)))))
    im.alpha_composite(scr, ((PW - SW) // 2, (PH - SH) // 2))
    im.alpha_composite(rrect(150, 44, 22, (0, 0, 0, 255)), (PW // 2 - 75, 42))
    # moving glass reflection
    g = Image.new("L", (PW, PH), 0)
    x = lerp(-PW * 0.6, PW * 1.2, gloss)
    ImageDraw.Draw(g).polygon([(x, 0), (x + 260, 0), (x - 380, PH), (x - 640, PH)], fill=28)
    g = g.filter(ImageFilter.GaussianBlur(40))
    g = Image.fromarray(np.minimum(np.asarray(g), np.asarray(rmask(PW, PH, PR))))
    wl = Image.new("RGBA", (PW, PH), (255, 255, 255, 0)); wl.putalpha(g)
    im.alpha_composite(wl)
    return im

@lru_cache(None)
def _outline_pts(n=144):
    pts = []
    hw, hh, r = PW / 2, PH / 2, PR
    centers = [(hw - r, -hh + r, -90), (hw - r, hh - r, 0), (-hw + r, hh - r, 90), (-hw + r, -hh + r, 180)]
    per = n // 4
    for cx, cy, a0 in centers:
        for i in range(per):
            a = math.radians(a0 + 90 * i / per)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a), math.cos(a), math.sin(a)))
    return pts

def phone3d(base, screen, cx, cy, ry, rx, rz, scale, gloss=0.5, alpha=1.0, blur=0.0):
    f = 2600.0
    R = rot(ry, rx, rz)
    pts = _outline_pts()
    P = np.array([[x, y, 0] for x, y, _, _ in pts]) * scale
    Q = np.array([[x, y, TH] for x, y, _, _ in pts]) * scale
    Nn = np.array([[nx, ny, 0] for _, _, nx, ny in pts]) @ R.T
    Pr = P @ R.T; Qr = Q @ R.T
    pp = Pr[:, :2] * (f / (f + Pr[:, 2:3])); qq = Qr[:, :2] * (f / (f + Qr[:, 2:3]))
    corners = np.array([[-PW / 2, -PH / 2, 0], [PW / 2, -PH / 2, 0], [PW / 2, PH / 2, 0], [-PW / 2, PH / 2, 0]]) * scale @ R.T
    cc = corners[:, :2] * (f / (f + corners[:, 2:3]))
    allp = np.vstack([pp, qq, cc])
    mn = allp.min(0) - 6; mx = allp.max(0) + 6
    cw, ch = int(mx[0] - mn[0]), int(mx[1] - mn[1])
    S = 2
    side = Image.new("RGBA", (cw * S, ch * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(side)
    light = np.array([-0.5, -0.55, -0.67]); light /= np.linalg.norm(light)
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        nr = (Nn[i] + Nn[j]) / 2
        if nr[2] > 0.05:
            continue
        dif = max(0.0, float(nr @ light))
        spec = dif ** 18
        v = 40 + 70 * dif + 160 * spec
        col = (int(v), int(v), int(v * 1.06), 255)
        poly = [tuple((pp[i] - mn) * S), tuple((pp[j] - mn) * S), tuple((qq[j] - mn) * S), tuple((qq[i] - mn) * S)]
        d.polygon(poly, fill=col)
    side = side.resize((cw, ch), Image.LANCZOS)
    fr = phone_front(screen, gloss)
    dst = cc - mn
    fw = fr.transform((cw, ch), Image.PERSPECTIVE, homography([(0, 0), (PW, 0), (PW, PH), (0, PH)], dst), Image.BICUBIC)
    side.alpha_composite(fw)
    off = 0
    if blur > 0.4:
        side = blurred(side, blur)
        off = (side.width - cw) / 2          # blurred() pads symmetrically
    comp(base, side, cx + mn[0] - off, cy + mn[1] - off, alpha)

# ================================================================= floating cards
def card_coinflip(u):
    w, h = 700, 560
    im = rrect(w, h, 44, (24, 24, 28, 250), (255, 255, 255, 30), 2).copy()
    d = ImageDraw.Draw(im)
    d.text((44, 66), "Coinflip", font=F(40, "Bold"), fill=TXT + (255,), anchor="lm")
    d.text((44, 110), "Double or nothing", font=F(26, "Medium"), fill=SUBT + (255,), anchor="lm")
    land = 1.25
    k = clamp((u - 0.35) / land)
    ang = (1 - (1 - k) ** 2.3) * math.pi * 8
    cs = math.cos(ang)
    hop = math.sin(k * math.pi) * 70
    face = badge("donut" if cs >= 0 else "hugo", 210)
    fw = max(3, int(210 * abs(cs)))
    sh = Image.new("RGBA", (220, 40), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([0, 0, 219, 39], fill=(0, 0, 0, 110))
    im.alpha_composite(sh.filter(ImageFilter.GaussianBlur(8)), (w // 2 - 110, 420))
    comp(im, face.resize((fw, 210), Image.BICUBIC), w / 2 - fw / 2, 205 - hop)
    if u > 0.35 + land:
        v = u - 0.35 - land
        t = _won()
        comp(im, t, w / 2 - t.width / 2, lerp(500, 455, eo5(v / 0.4)), alpha=clamp(v / 0.25))
    return im

@lru_cache(None)
def _won():
    b = rrect(380, 72, 36, (24, 70, 40, 255), GREEN + (255,), 2).copy()
    ImageDraw.Draw(b).text((190, 36), "You won  +500,000", font=F(28, "SemiBold"), fill=(140, 245, 170, 255), anchor="mm")
    return b

def card_crash(u):
    w, h = 700, 640
    im = rrect(w, h, 44, (24, 24, 28, 250), (255, 255, 255, 30), 2).copy()
    d = ImageDraw.Draw(im)
    d.text((44, 66), "Crash", font=F(40, "Bold"), fill=TXT + (255,), anchor="lm")
    d.text((44, 110), "Cash out before it crashes", font=F(26, "Medium"), fill=SUBT + (255,), anchor="lm")
    k = clamp((u - 0.15) / 1.8)
    m = math.exp(1.435 * k)
    gx0, gy0, gx1, gy1 = 50, 250, w - 50, 560
    for i in range(4):
        y = gy0 + (gy1 - gy0) * i / 3
        d.line([gx0, y, gx1, y], fill=(255, 255, 255, 18), width=2)
    pts = []
    mmax = math.exp(1.435)
    for i in range(60):
        q = k * i / 59
        pts.append((gx0 + (gx1 - gx0) * q, gy1 - (gy1 - gy0) * (math.exp(1.435 * q) - 1) / (mmax - 1)))
    if k > 0.02:
        lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(lay).polygon(pts + [(pts[-1][0], gy1), (gx0, gy1)], fill=(48, 209, 88, 40))
        gl = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(gl).line(pts, fill=GREEN + (255,), width=14, joint="curve")
        lay.alpha_composite(gl.filter(ImageFilter.GaussianBlur(8)))
        ImageDraw.Draw(lay).line(pts, fill=(190, 255, 205, 255), width=6, joint="curve")
        im.alpha_composite(lay)
        im.alpha_composite(circle(26, (255, 255, 255, 255), GREEN + (255,), 5), (int(pts[-1][0] - 13), int(pts[-1][1] - 13)))
    done = k >= 1
    mt = text(f"{m:.2f}x", 96, GREEN if not done else BLUE_L, "Bold")
    comp(im, mt, w - 44 - mt.width + int(96 * 0.3), 150 - mt.height / 2 + 10)
    if done:
        v = u - 1.95
        c = _cashed()
        comp(im, c, w / 2 - c.width / 2, lerp(330, 300, eo5(v / 0.4)), alpha=clamp(v / 0.2))
    return im

@lru_cache(None)
def _cashed():
    b = rrect(360, 72, 36, shade(BLUE, 0.65) + (255,), BLUE + (255,), 2).copy()
    ImageDraw.Draw(b).text((180, 36), "Cashed out at 4.20x", font=F(27, "SemiBold"), fill=BLUE_L + (255,), anchor="mm")
    return b

LIVE_ROWS = [
    ("K", "Kevin", "Crash", "250,000", "3.10x", "+775,000", "hugo"),
    ("A", "Anna", "Coinflip", "1,000,000", "2.00x", "+2,000,000", "donut"),
    ("T", "Tim", "Crash", "400,000", "1.85x", "+740,000", "donut"),
    ("S", "Sara", "Coinflip", "750,000", "2.00x", "+1,500,000", "hugo"),
    ("L", "Luca", "Crash", "2,000,000", "5.40x", "+10,800,000", "hugo"),
    ("N", "Nina", "Coinflip", "300,000", "2.00x", "+600,000", "donut"),
    ("P", "Paul", "Crash", "900,000", "2.25x", "+2,025,000", "donut"),
]

def panel_live(u):
    w, h = 880, 980
    im = rrect(w, h, 48, (20, 20, 24, 252), (255, 255, 255, 30), 2).copy()
    d = ImageDraw.Draw(im)
    d.text((48, 74), "Live bets", font=F(46, "Bold"), fill=TXT + (255,), anchor="lm")
    blink = 0.6 + 0.4 * math.sin(u * 6)
    im.alpha_composite(circle(18, (255, 69, 58, int(255 * blink))), (300, 65))
    d.text((328, 74), "LIVE", font=F(24, "Bold"), fill=(255, 105, 97, 255), anchor="lm")
    for i, lab in enumerate(("Player", "Bet", "Multi", "Payout")):
        d.text(((116, 364, 554, 832)[i], 140), lab, font=F(22, "SemiBold"), fill=SUBT + (255,),
               anchor="lm" if i < 3 else "rm")
    n_vis = 7
    shown = int(u / 0.42) + 3
    rh = 108
    lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    frac = eo5((u % 0.42) / 0.3)
    for slot in range(n_vis + 1):
        idx = shown - slot
        if idx < 0:
            continue
        row = LIVE_ROWS[idx % len(LIVE_ROWS)]
        y = 175 + (slot - 1 + frac) * rh
        if y < 150 or y > h - 40:
            continue
        a = clamp((y - 150) / 60) if slot == 0 else 1
        r = _live_row(row)
        comp(lay, r, 24, y, alpha=a * (1 - clamp((y - (h - 160)) / 120)))
    im.alpha_composite(lay)
    return im

@lru_cache(None)
def _live_row(row):
    a, n, g, bet, mul, pay, cur = row
    im = rrect(832, 92, 26, (32, 32, 38, 255)).copy()
    d = ImageDraw.Draw(im)
    im.alpha_composite(circle(56, (62, 62, 70, 255)), (18, 18))
    d.text((46, 46), a, font=F(24, "SemiBold"), fill=TXT + (255,), anchor="mm")
    d.text((92, 34), n, font=F(26, "SemiBold"), fill=TXT + (255,), anchor="lm")
    d.text((92, 64), g, font=F(20, "Medium"), fill=SUBT + (255,), anchor="lm")
    im.alpha_composite(badge(cur, 30), (300, 31))
    d.text((340, 46), bet, font=F(24, "SemiBold"), fill=TXT + (255,), anchor="lm")
    d.text((530, 46), mul, font=F(24, "SemiBold"), fill=(BLUE_L if float(mul[:-1]) > 3 else TXT) + (255,), anchor="lm")
    d.text((808, 46), pay, font=F(24, "Bold"), fill=GREEN + (255,), anchor="rm")
    return im

def card_wallet(u):
    w, h = 820, 860
    im = rrect(w, h, 48, (22, 22, 26, 252), (255, 255, 255, 30), 2).copy()
    d = ImageDraw.Draw(im)
    d.text((48, 76), "Withdraw", font=F(46, "Bold"), fill=TXT + (255,), anchor="lm")
    d.text((48, 126), "Straight to your account", font=F(26, "Medium"), fill=SUBT + (255,), anchor="lm")
    im.alpha_composite(rrect(w - 96, 150, 32, (34, 34, 40, 255)), (48, 180))
    d.text((84, 228), "Amount", font=F(24, "Medium"), fill=SUBT + (255,), anchor="lm")
    d.text((84, 283), "5,000,000", font=F(60, "Bold"), fill=TXT + (255,), anchor="lm")
    im.alpha_composite(rrect(250, 64, 32, RED + (40,), RED + (255,), 2), (w - 48 - 36 - 250, 223))
    im.alpha_composite(badge("hugo", 40), (w - 48 - 36 - 250 + 14, 235))
    d.text((w - 48 - 36 - 250 + 64, 255), "Hugo money", font=F(24, "SemiBold"), fill=tint(RED, 0.45) + (255,), anchor="lm")
    # button
    press = 1 - 0.05 * math.sin(math.pi * prog(u, 0.85, 0.2))
    bw, bh = int((w - 96) * press), int(110 * press)
    state = 0 if u < 0.95 else (1 if u < 1.75 else 2)
    col = (255, 255, 255) if state == 0 else ((48, 48, 56) if state == 1 else GREEN)
    b = rrect(bw, bh, bh // 2, col + (255,))
    comp(im, b, w / 2 - bw / 2, 380 + (110 - bh) / 2)
    if state == 0:
        d.text((w / 2, 435), "Withdraw now", font=F(34, "Bold"), fill=(20, 20, 24, 255), anchor="mm")
    elif state == 1:
        a0 = (u - 0.95) * 540
        sp = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
        ImageDraw.Draw(sp).arc([20, 20, 160, 160], a0, a0 + 270, fill=(255, 255, 255, 255), width=14)
        sp = sp.resize((60, 60), Image.LANCZOS)
        im.alpha_composite(sp, (int(w / 2 - 150), 405))
        d.text((w / 2 + 20, 435), "Processing…", font=F(32, "SemiBold"), fill=TXT + (255,), anchor="mm")
    else:
        d.text((w / 2, 435), "Sent", font=F(34, "Bold"), fill=(10, 40, 20, 255), anchor="mm")
    # receipt rows
    k = eo5(prog(u, 1.8, 0.5))
    if k > 0:
        ck = _check(120)
        s = 0.6 + 0.4 * eback(prog(u, 1.75, 0.35), 2.2)
        cimg = ck.resize((int(120 * s), int(120 * s)), Image.LANCZOS)
        comp(im, cimg, w / 2 - cimg.width / 2, 590 - cimg.height / 2)
        d.text((w / 2, 700), "Withdrawal complete", font=F(36, "Bold"), fill=TXT + (int(255 * k),), anchor="mm")
        d.text((w / 2, 750), "Arrived in 0.6 seconds", font=F(26, "Medium"), fill=SUBT + (int(255 * k),), anchor="mm")
    return im

@lru_cache(None)
def _check(s):
    im = circle(s, GREEN + (255,)).copy()
    big = Image.new("RGBA", (s * 3, s * 3), (0, 0, 0, 0))
    S = s * 3
    ImageDraw.Draw(big).line([(0.28 * S, 0.52 * S), (0.44 * S, 0.67 * S), (0.73 * S, 0.35 * S)], fill=(255, 255, 255, 255),
                             width=int(S * 0.1), joint="curve")
    im.alpha_composite(big.resize((s, s), Image.LANCZOS))
    return im

# ================================================================= typography (Apple-style)
def words_in(base, parts, t, t_in, t_out, cy, size=72, weight="SemiBold", stagger=0.07):
    """parts: list of (word, color). Each word fades/unblurs/rises in; all fade+blur out at t_out."""
    if t < t_in - 0.01 or t > t_out + 0.5:
        return
    f = F(size, weight)
    space = f.getlength(" ")
    widths = [f.getlength(wd) for wd, _ in parts]
    total = sum(widths) + space * (len(parts) - 1)
    x = W / 2 - total / 2
    pad = int(size * 0.3)
    ko = prog(t, t_out, 0.4)
    for i, (wd, col) in enumerate(parts):
        k = prog(t, t_in + i * stagger, 0.7)
        if k > 0:
            e = eo5(k)
            im = text(wd, size, col, weight)
            a = clamp(k * 1.6) * (1 - eo(ko))
            bl = (1 - e) * 14 + eo(ko) * 10
            yy = cy + (1 - e) * 34 - eo(ko) * 10
            place(base, im, x - pad + im.width / 2, yy, alpha=a, blur=bl)
        x += widths[i] + space

# ================================================================= scenes
def phone_state(t):
    """Phone pose over the first two scenes."""
    k1 = eo5(prog(t, 0.25, 2.4))
    k2 = eio(prog(t, EV["app"], 1.1))
    ry = lerp(-1.05, -0.34, k1); rx = lerp(0.42, 0.14, k1); rz = lerp(-0.16, -0.05, k1)
    ry = lerp(ry, 0.0, k2) + 0.05 * math.sin((t - EV["app"]) * 0.8) * k2
    rx = lerp(rx, 0.0, k2) + 0.03 * math.sin((t - EV["app"]) * 0.6 + 1) * k2
    rz = lerp(rz, 0.0, k2)
    y = lerp(1560, 1120, k1); y = lerp(y, 1110, k2)
    s = lerp(0.92, 1.06, k1); s = lerp(s, 1.15, k2)
    # exit at cards
    k3 = eio(prog(t, EV["cards"] - 0.15, 0.7))
    ry += 0.6 * k3; y += 900 * k3 * k3; s *= 1 - 0.15 * k3
    return ry, rx, rz, y, s

def screen_at(t):
    if t < EV["app"] + 0.35:
        sc = screen_lock(t) if t >= 1.25 else Image.new("RGBA", (SW, SH), (4, 4, 6, 255))
        if 1.25 <= t < 1.6:   # screen wake
            k = prog(t, 1.25, 0.35)
            blk = Image.new("RGBA", (SW, SH), (4, 4, 6, int(255 * (1 - k))))
            sc.alpha_composite(blk)
        if t >= EV["app"]:
            k = eio(prog(t, EV["app"], 0.35))
            app = screen_app(t)
            out = sc.copy()
            ls = sc.resize((int(SW * (1 + 0.15 * k)), int(SH * (1 + 0.15 * k))), Image.BICUBIC)
            out.paste((0, 0, 0, 255), (0, 0, SW, SH))
            comp(out, ls, SW / 2 - ls.width / 2, SH / 2 - ls.height / 2, 1 - k)
            ap = app.resize((int(SW * (0.9 + 0.1 * k)), int(SH * (0.9 + 0.1 * k))), Image.BICUBIC)
            comp(out, ap, SW / 2 - ap.width / 2, SH / 2 - ap.height / 2, k)
            return out
        return sc
    return screen_app(t)

def s_phone(base, t):
    ry, rx, rz, y, s = phone_state(t)
    a = clamp((t - 0.1) / 0.5) * (1 - prog(t, EV["cards"] + 0.2, 0.35))
    gloss = 0.35 + 0.5 * eo(prog(t, 0.3, 3.0)) + 0.1 * prog(t, EV["app"], 3)
    phone3d(base, screen_at(t), W / 2, y, ry, rx, rz, s, gloss, alpha=a)

def s_cards(base, t):
    u = t - EV["cards"]
    if u < -0.2 or t > EV["live"] + 0.5:
        return
    kin = eo5(prog(t, EV["cards"], 0.9))
    kout = eio(prog(t, EV["live"] - 0.2, 0.6))
    f = eios(prog(t, EV["focus"] - 0.2, 0.6))     # 0: coinflip sharp, 1: crash sharp
    drift = u * 0.06
    # coinflip card (upper-left), crash card (lower-right)
    c1 = card_coinflip(u)
    c1 = card3d(c1, ry=0.22 - drift, rx=0.08, rz=-0.03)
    c2 = card_crash(t - EV["focus"] + 0.1)
    c2 = card3d(c2, ry=-0.2 + drift, rx=0.06, rz=0.025)
    out_s = 1 - 0.12 * kout
    place(base, c1, lerp(-200, 430, kin) - 120 * f, 760 - 60 * f - 400 * kout, scale=lerp(1.0, 0.86, f) * out_s,
          alpha=lerp(1, 0.55, f) * (1 - kout), blur=f * 12 + kout * 10)
    place(base, c2, lerp(1300, 640, kin), 1200 - 50 * f - 300 * kout, scale=lerp(0.84, 1.02, f) * out_s,
          alpha=lerp(0.6, 1.0, f) * kin * (1 - kout), blur=(1 - f) * 12 + kout * 10)

def s_live(base, t):
    u = t - EV["live"]
    if u < -0.3 or t > EV["wallet"] + 0.5:
        return
    kin = eo5(prog(t, EV["live"] - 0.1, 0.9))
    kout = eio(prog(t, EV["wallet"] - 0.2, 0.6))
    p = panel_live(max(0, u))
    p = card3d(p, ry=-0.12 + 0.05 * u, rx=lerp(0.35, 0.12, kin), rz=0.0)
    place(base, p, W / 2, lerp(1400, 1080, kin) - 200 * kout, scale=lerp(0.9, 1.0, kin) * (1 - 0.1 * kout),
          alpha=clamp(kin * 1.5) * (1 - kout), blur=(1 - kin) * 10 + kout * 12)

def s_wallet(base, t):
    u = t - EV["wallet"]
    if u < -0.3 or t > EV["outro"] + 0.6:
        return
    kin = eo5(prog(t, EV["wallet"] - 0.1, 0.9))
    kout = eio(prog(t, EV["outro"] - 0.1, 0.6))
    c = card_wallet(max(0, u))
    c = card3d(c, ry=0.14 - 0.05 * u, rx=lerp(-0.3, -0.08, kin))
    place(base, c, W / 2, lerp(1350, 1060, kin) - 150 * kout, scale=lerp(0.9, 1.0, kin) * (1 - 0.1 * kout),
          alpha=clamp(kin * 1.5) * (1 - kout), blur=(1 - kin) * 10 + kout * 12)

def s_logo(base, t):
    u = t - EV["logo"]
    if u < 0:
        return
    fade = 1 - prog(t, DUR - 0.6, 0.6)
    kc = eio(prog(t, EV["cta"] - 0.1, 0.8))          # shift layout up for the CTA
    k = eo5(prog(u, 0, 1.1))
    sp = 1 + 0.06 * math.sin(min(u, 1.2) / 1.2 * math.pi) * (1 - prog(u, 1.2, 0.01))
    lg = server_logo(640)
    ly = lerp(740, 600, kc) + math.sin(u * 1.6) * 8
    place(base, lg, W / 2, ly + (1 - k) * 60, scale=lerp(0.7, 1.0, k) * sp * lerp(1, 0.86, kc),
          alpha=clamp(u / 0.4) * fade, blur=(1 - k) * 18)
    kw = eo5(prog(u, 0.35, 0.9))
    wm = wordmark(104)
    wy = lerp(1150, 1000, kc)
    if kw > 0:
        place(base, wm, W / 2, wy + (1 - kw) * 30, alpha=clamp(kw * 1.5) * fade, blur=(1 - kw) * 12)
    if 0.9 < u < 2.0:
        sw = (u - 0.9) / 1.1
        m = Image.new("L", (wm.width, wm.height), 0)
        x = lerp(-200, wm.width + 200, eio(sw))
        ImageDraw.Draw(m).polygon([(x, 0), (x + 80, 0), (x - 40, wm.height), (x - 120, wm.height)], fill=150)
        m = m.filter(ImageFilter.GaussianBlur(14))
        m = Image.fromarray(np.minimum(np.asarray(m), np.asarray(wm.getchannel("A"))))
        sh = Image.new("RGBA", wm.size, (255, 255, 255, 0)); sh.putalpha(m)
        place(base, sh, W / 2, wy, alpha=fade)
    ku = eo5(prog(u, 0.8, 0.8))
    if ku > 0:
        place(base, _url_chip(), W / 2, lerp(1290, 1110, kc) + (1 - ku) * 24, alpha=clamp(ku * 1.5) * fade,
              blur=(1 - ku) * 10)
    # CTA
    v = t - EV["cta"]
    if v > 0:
        ka = prog(v, 0.15, 0.6)
        s = 0.7 + 0.3 * eback(ka, 1.8)
        pulse = 1 + 0.025 * math.sin(v * 5.0) * prog(v, 0.8, 0.3)
        place(base, _cta_glow(), W / 2, 1330, alpha=clamp(ka * 1.4) * fade * (0.75 + 0.25 * math.sin(v * 5.0)))
        place(base, _cta(), W / 2, 1330, scale=s * pulse, alpha=clamp(ka * 2) * fade)
        kar = prog(v, 0.6, 0.5)
        if kar > 0:
            for i in range(3):
                bob = (math.sin(v * 5.0 - i * 0.7) + 1) * 0.5
                place(base, _chevron(), W / 2, 1490 + i * 34 + bob * 10, alpha=clamp(kar * 2) * (0.35 + 0.65 * bob) * fade)
    kd = prog(u, 1.4, 0.8)
    if kd > 0:
        place(base, _disc(), W / 2, 1660, alpha=clamp(kd) * 0.75 * fade)

@lru_cache(None)
def _cta():
    f = F(62, "ExtraBold")
    label = "JOIN DC IN BIO"
    tw = int(f.getlength(label))
    ic = 70
    w, h = tw + ic + 150, 138
    im = rrect(w, h, h // 2, (255, 255, 255, 255)).copy()
    # chat-bubble icon
    S = ic * 3
    bub = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bub)
    bd.rounded_rectangle([0, S * 0.08, S, S * 0.78], S * 0.26, fill=BLUE + (255,))
    bd.polygon([(S * 0.22, S * 0.7), (S * 0.18, S * 0.98), (S * 0.48, S * 0.74)], fill=BLUE + (255,))
    for cx in (0.3, 0.5, 0.7):
        bd.ellipse([S * cx - S * 0.06, S * 0.37, S * cx + S * 0.06, S * 0.49], fill=(255, 255, 255, 255))
    im.alpha_composite(bub.resize((ic, ic), Image.LANCZOS), (58, (h - ic) // 2 + 2))
    ImageDraw.Draw(im).text((58 + ic + 24, h / 2 + 2), label, font=f, fill=(12, 14, 24, 255), anchor="lm")
    return im

@lru_cache(None)
def _cta_glow():
    c = _cta()
    p = 80
    im = Image.new("RGBA", (c.width + 2 * p, c.height + 2 * p), (0, 0, 0, 0))
    g = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(g).rounded_rectangle([p, p, p + c.width, p + c.height], c.height // 2, fill=BLUE + (230,))
    return g.filter(ImageFilter.GaussianBlur(36))

@lru_cache(None)
def _chevron():
    S = 3
    im = Image.new("RGBA", (60 * S, 30 * S), (0, 0, 0, 0))
    ImageDraw.Draw(im).line([(6 * S, 6 * S), (30 * S, 24 * S), (54 * S, 6 * S)], fill=(255, 255, 255, 255), width=6 * S, joint="curve")
    return im.resize((60, 30), Image.LANCZOS)

@lru_cache(None)
def _url_chip():
    f = F(40, "SemiBold")
    tw = int(f.getlength("donuthugobet.net"))
    w, h = tw + 110, 96
    im = rrect(w, h, h // 2, (255, 255, 255, 22), (255, 255, 255, 70), 2).copy()
    ImageDraw.Draw(im).text((w / 2, h / 2), "donuthugobet.net", font=f, fill=TXT + (255,), anchor="mm")
    return im

@lru_cache(None)
def _disc():
    return text("18+  ·  Play responsibly", 28, (170, 170, 180), "Medium")

# ================================================================= glow + text timeline
def glows_at(t):
    ka = eio(prog(t, EV["toggle"], 0.6))
    kw = eio(prog(t, EV["wallet"] - 0.2, 0.8))
    kl = eio(prog(t, EV["live"] - 0.2, 0.8))
    kc = eio(prog(t, EV["cards"] - 0.2, 0.8))
    ko = eio(prog(t, EV["outro"] - 0.2, 0.8))
    col = lerpc(BLUE, RED, ka)
    col = lerpc(col, BLUE, kc)
    col = lerpc(col, GREEN, kl * 0.0)
    col = lerpc(col, RED, kw)
    col = lerpc(col, BLUE, ko)
    intro = eo(prog(t, 0.0, 1.6))
    pulse = 1 + 0.08 * math.sin(t * 1.7)
    y = 1080 if t < EV["outro"] else lerp(1080, 920, ko)
    y = y if t < EV["logo"] else lerp(y, 900, eio(prog(t, EV["logo"], 1.0)))
    kl = eio(prog(t, EV["logo"], 1.0))
    g = [(col, W / 2, y, 310 * pulse, 0.55 * intro * (1 - 0.45 * kl)),
         (lerpc(col, (255, 255, 255), 0.15), W / 2, y + 40, 150, 0.16 * intro)]
    if t > EV["cards"] - 0.5:
        g.append((RED if col != RED else BLUE, W * 0.85, 1500, 320, 0.12 * kc * (1 - ko)))
    if t > EV["logo"]:
        u = t - EV["logo"]
        g.append((BLUE, W * 0.3, 720, 300, 0.22 * eo(prog(u, 0, 1))))
        g.append((RED, W * 0.72, 720, 280, 0.20 * eo(prog(u, 0, 1))))
    fade = 1 - prog(t, DUR - 0.6, 0.6)
    return [(c, x, y, r, i * fade) for c, x, y, r, i in g]

TEXTS = [
    ([("Stop", TXT), ("grinding.", TXT)], 0.9, 3.1),
    ([("Gamble", TXT), ("with", TXT), ("Donut", BLUE), ("money", TXT)], 4.2, 6.5),
    ([("or", TXT), ("Hugo", RED), ("money.", TXT)], 6.75, 8.75),
    ([("Coinflip.", TXT), ("Crash.", TXT), ("Big", TXT), ("wins.", BLUE_L)], 9.3, 13.0),
    ([("Every", TXT), ("bet.", TXT), ("Live.", (255, 105, 97))], 13.6, 15.9),
    ([("Instant", TXT), ("withdrawals.", BLUE_L)], 16.4, 19.0),
]

def compose(t):
    base = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    base.alpha_composite(glow_bg(glows_at(t)))
    if t < EV["cards"] + 0.6:
        s_phone(base, t)
    s_cards(base, t)
    s_live(base, t)
    s_wallet(base, t)
    for parts, a, b in TEXTS:
        words_in(base, parts, t, a, b, 270, 70)
    # outro tagline, centered
    words_in(base, [("Gamble", TXT), ("with", TXT), ("Donut", BLUE), ("money", TXT)], t, EV["outro"] + 0.2, EV["logo"] - 0.45, 880, 72)
    words_in(base, [("or", TXT), ("Hugo", RED), ("money.", TXT)], t, EV["outro"] + 0.6, EV["logo"] - 0.4, 980, 72)
    s_logo(base, t)
    return np.asarray(base.convert("RGB"), np.float32)

def frame(i):
    t = i / FPS
    if SUB > 1:
        offs = [(k / (SUB - 1) - 0.5) * 0.5 / FPS for k in range(SUB)]
        acc = sum(compose(t + o) for o in offs) / SUB
    else:
        acc = compose(t)
    acc += np.repeat(np.repeat(GRAIN[i % len(GRAIN)], 2, 0), 2, 1)
    return np.clip(acc, 0, 255).astype(np.uint8).tobytes()

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BUILD, "donuthugobet_v3.mp4")
    only = os.environ.get("FRAMES")
    if only:
        for i in map(int, only.split(",")):
            Image.frombytes("RGB", (W, H), frame(i)).save(os.path.join(BUILD, f"still_{i:04d}.jpg"), quality=92)
        return
    n = int(DUR * FPS)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", os.path.join(BUILD, "mix.wav"), "-c:v", "libx264", "-preset", "slow", "-crf", "17",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
           "-shortest", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(int(os.environ.get("JOBS", os.cpu_count()))) as pool:
        for k, buf in enumerate(pool.imap(frame, range(n), chunksize=3)):
            enc.stdin.write(buf)
            if k % 60 == 0:
                print(f"frame {k}/{n}", flush=True)
    enc.stdin.close(); enc.wait()
    print("wrote", out)

if __name__ == "__main__":
    main()
