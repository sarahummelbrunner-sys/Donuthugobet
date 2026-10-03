"""Tiny vectorized voxel raycaster with procedural Minecraft-style 16x16 textures."""
import numpy as np

T = 16
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- textures
def _noise(base, var, seed=None, size=T):
    r = np.random.default_rng(seed)
    n = r.random((size, size, 1))
    c = np.array(base, np.float32)[None, None, :] * (1 - var / 2 + var * n)
    return c

def _rgba(rgb, alpha=None):
    a = np.full(rgb.shape[:2] + (1,), 255.0) if alpha is None else alpha[..., None] * 255.0
    return np.concatenate([np.clip(rgb, 0, 255), a], -1).astype(np.uint8)

def tex_grass_top(): return _rgba(_noise((95, 159, 53), 0.45, 1))
def tex_dirt(): return _rgba(_noise((134, 96, 67), 0.5, 2))
def tex_grass_side():
    d = _noise((134, 96, 67), 0.5, 3)
    g = _noise((95, 159, 53), 0.45, 4)
    r = np.random.default_rng(5)
    depth = 3 + (r.random(T) * 3).astype(int)
    for x in range(T):
        d[: depth[x], x] = g[: depth[x], x]
    return _rgba(d)
def tex_stone(): return _rgba(_noise((125, 125, 125), 0.35, 6))
def tex_deepslate():
    c = _noise((70, 70, 78), 0.35, 7)
    c[::4] *= 0.8
    return _rgba(c)
def tex_cobble():
    c = _noise((122, 122, 122), 0.3, 8)
    r = np.random.default_rng(9)
    for _ in range(14):
        x, y = r.integers(0, T, 2)
        c[y:y + 1, x:x + 4] *= 0.6
    return _rgba(c)
def tex_planks():
    c = _noise((162, 130, 78), 0.25, 10)
    c[3::4] *= 0.72
    for y in range(0, T, 4):
        c[y:y + 3, (y * 5) % T] *= 0.75
    return _rgba(c)
def tex_log_side():
    c = _noise((102, 81, 49), 0.35, 11)
    c[:, ::3] *= 0.8
    return _rgba(c)
def tex_log_top():
    c = _noise((162, 130, 78), 0.2, 12)
    yy, xx = np.mgrid[0:T, 0:T]
    rr = np.hypot(xx - 7.5, yy - 7.5)
    c[(rr.astype(int) % 3) == 0] *= 0.8
    c[rr > 6.8] = np.array([102, 81, 49]) * 0.9
    return _rgba(c)
def tex_leaves():
    c = _noise((60, 130, 40), 0.6, 13)
    a = (np.random.default_rng(14).random((T, T)) > 0.22).astype(float)
    return _rgba(c, a)
def tex_farmland():
    c = _noise((88, 58, 36), 0.35, 15)
    c[::4] *= 0.7
    return _rgba(c)
def tex_wheat():
    c = np.zeros((T, T, 3)); a = np.zeros((T, T))
    r = np.random.default_rng(16)
    for x in range(1, T, 3):
        h = r.integers(9, 15)
        col = np.array([200, 170, 60]) * (0.8 + 0.3 * r.random())
        c[T - h:, x] = col; a[T - h:, x] = 1
        c[T - h:T - h + 4, x - 1:x + 2] = np.array([215, 185, 70]); a[T - h:T - h + 4, x - 1:x + 2] = 1
    c[T - 5:, :] = np.where(a[T - 5:, :, None] > 0, np.array([110, 150, 50]), 0)
    return _rgba(c, a)
def tex_water(): return _rgba(_noise((50, 90, 220), 0.25, 17))
def tex_gold():
    c = _noise((250, 210, 55), 0.15, 18)
    c[0, :] = c[:, 0] = (255, 250, 160); c[-1, :] = c[:, -1] = (190, 130, 20)
    c[2:4, 2:6] = (255, 255, 200)
    return _rgba(c)
def tex_gold_dark():
    c = _noise((200, 140, 25), 0.15, 19)
    c[0, :] = c[:, 0] = (240, 200, 70); c[-1, :] = c[:, -1] = (140, 90, 10)
    return _rgba(c)
def _ore(base_tex, col, seed):
    t = base_tex[..., :3].astype(float)
    r = np.random.default_rng(seed)
    for _ in range(6):
        x, y = r.integers(1, T - 3, 2)
        t[y:y + 2, x:x + 3] = col; t[y + 1, x + 1] = np.array(col) * 1.25
    return _rgba(t)
def tex_glow():
    c = _noise((250, 200, 110), 0.45, 20)
    return _rgba(c)
def tex_dough():
    c = _noise((205, 145, 80), 0.25, 21)
    return _rgba(c)
def tex_frosting():
    c = _noise((255, 125, 185), 0.15, 22)
    return _rgba(c)
def tex_solid(col, var=0.12, seed=23): return _rgba(_noise(col, var, seed))
def tex_emerald():
    c = _noise((40, 200, 90), 0.2, 24)
    c[0, :] = c[:, 0] = (140, 255, 170); c[-1, :] = c[:, -1] = (10, 120, 50)
    return _rgba(c)
def tex_diamond():
    c = _noise((90, 230, 225), 0.2, 25)
    c[0, :] = c[:, 0] = (200, 255, 255); c[-1, :] = c[:, -1] = (30, 150, 150)
    return _rgba(c)

# block ids
AIR, GRASS, DIRT, STONE, DEEP, COBBLE, PLANKS, LOG, LEAVES, FARM, WHEAT, WATER, GOLD, GOLDD, \
    DIA_ORE, EM_ORE, GOLD_ORE, GLOW, DOUGH, FROST, SPR_R, SPR_Y, SPR_B, SPR_W, SPR_G, EMERALD, DIAMOND, \
    OBSID, WOOL_R = range(29)
NB = 29
EMISSIVE = {GLOW: 1.0, GOLD: 0.25, EMERALD: 0.2, DIAMOND: 0.2}

def build_atlas():
    A = np.zeros((NB, 3, T, T, 4), np.uint8)   # [block, face(side,top,bottom)]
    def put(b, side, top=None, bot=None):
        A[b, 0] = side; A[b, 1] = side if top is None else top; A[b, 2] = (A[b, 1] if bot is None else bot)
    st = tex_stone(); dp = tex_deepslate()
    put(GRASS, tex_grass_side(), tex_grass_top(), tex_dirt())
    put(DIRT, tex_dirt()); put(STONE, st); put(DEEP, dp); put(COBBLE, tex_cobble())
    put(PLANKS, tex_planks()); put(LOG, tex_log_side(), tex_log_top()); put(LEAVES, tex_leaves())
    put(FARM, tex_dirt(), tex_farmland()); put(WHEAT, tex_wheat(), np.zeros((T, T, 4), np.uint8))
    put(WATER, tex_water()); put(GOLD, tex_gold()); put(GOLDD, tex_gold_dark())
    put(DIA_ORE, _ore(dp, (90, 230, 225), 30)); put(EM_ORE, _ore(dp, (40, 210, 90), 31))
    put(GOLD_ORE, _ore(st, (250, 210, 60), 32)); put(GLOW, tex_glow())
    put(DOUGH, tex_dough()); put(FROST, tex_frosting())
    put(SPR_R, tex_solid((235, 50, 60))); put(SPR_Y, tex_solid((255, 230, 60)))
    put(SPR_B, tex_solid((70, 140, 255))); put(SPR_W, tex_solid((250, 250, 250)))
    put(SPR_G, tex_solid((90, 220, 90))); put(EMERALD, tex_emerald()); put(DIAMOND, tex_diamond())
    put(OBSID, tex_solid((30, 20, 45), 0.4, 33)); put(WOOL_R, tex_solid((200, 40, 40), 0.2, 34))
    return A

ATLAS = build_atlas()
EMIS = np.zeros(NB, np.float32)
for k, v in EMISSIVE.items():
    EMIS[k] = v

# ---------------------------------------------------------------- raycaster
FACE_SHADE = np.array([0.80, 1.00, 0.55, 0.66], np.float32)  # x-side, top, bottom, z-side

def sky_default(d):
    up = np.clip(d[:, 1], -1, 1)
    top = np.array([96, 150, 255], np.float32); hor = np.array([190, 215, 255], np.float32)
    k = np.clip(up * 2.2, 0, 1)[:, None]
    return hor * (1 - k) + top * k

def render(world, pos, yaw, pitch, W=360, H=640, fov=75, sky=sky_default, fog_dist=70.0,
           max_steps=220, fog_col=None, roll=0.0):
    """world: uint8 [X,Y,Z]. Returns float32 HxWx3 (0..255) and emissive mask HxW."""
    X, Y, Z = world.shape
    pos = np.asarray(pos, np.float64)
    asp = W / H
    tf = np.tan(np.radians(fov) / 2)
    ys, xs = np.mgrid[0:H, 0:W]
    px = ((xs + 0.5) / W * 2 - 1) * tf * asp
    py = (1 - (ys + 0.5) / H * 2) * tf
    if roll:
        cr, sr = np.cos(roll), np.sin(roll)
        px, py = px * cr - py * sr, px * sr + py * cr
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    fwd = np.array([sy * cp, sp, cy * cp]); right = np.array([cy, 0, -sy]); up = np.cross(fwd, right)
    d = fwd[None] + px.reshape(-1, 1) * right[None] + py.reshape(-1, 1) * up[None]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    N = d.shape[0]
    out = sky(d).astype(np.float32)
    emis = np.zeros(N, np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.sign(d).astype(np.int64)
        tdel = np.where(d != 0, np.abs(1.0 / d), np.inf)
        ip = np.floor(pos).astype(np.int64)[None].repeat(N, 0)
        nxt = ip + (step > 0)
        tmax = np.where(d != 0, (nxt - pos[None]) / d, np.inf)
    idx = np.arange(N)
    sz = np.array([X, Y, Z])
    for _ in range(max_steps):
        if idx.size == 0:
            break
        ax = np.argmin(tmax, axis=1)
        r = np.arange(idx.size)
        t = tmax[r, ax]
        ip[r, ax] += step[r, ax]
        tmax[r, ax] += tdel[r, ax]
        inb = np.all((ip >= 0) & (ip < sz), axis=1)
        b = np.zeros(idx.size, np.uint8)
        b[inb] = world[ip[inb, 0], ip[inb, 1], ip[inb, 2]]
        hit = b > 0
        if hit.any():
            h = np.nonzero(hit)[0]
            hp = pos[None] + d[idx[h]] * t[h, None]
            a = ax[h]
            fr = hp - np.floor(hp)
            fr = np.clip(fr, 0, 0.9999)
            u = np.where(a == 0, fr[:, 2], fr[:, 0])
            v = np.where(a == 1, fr[:, 2], 1 - fr[:, 1])
            top_face = (a == 1) & (step[h, 1] < 0)
            bot_face = (a == 1) & (step[h, 1] > 0)
            face = np.where(top_face, 1, np.where(bot_face, 2, 0))
            tu = np.minimum((u * T).astype(int), T - 1); tv = np.minimum((v * T).astype(int), T - 1)
            texel = ATLAS[b[h], face, tv, tu]
            solid = texel[:, 3] > 0
            hs = h[solid]
            if hs.size:
                sh = np.where(a[solid] == 0, FACE_SHADE[0], np.where(a[solid] == 2, FACE_SHADE[3],
                              np.where(top_face[solid], FACE_SHADE[1], FACE_SHADE[2])))
                bid = b[hs]
                sh = sh * _ao(world, ip[hs], a[solid], step[hs], fr[solid], sz)
                e = EMIS[bid]
                sh = sh * (1 - e) + e * 1.15
                col = texel[solid, :3].astype(np.float32) * sh[:, None]
                dist = t[hs]
                fk = np.clip((dist / fog_dist) ** 1.6, 0, 1)[:, None]
                fc = out[idx[hs]] if fog_col is None else np.asarray(fog_col, np.float32)[None]
                out[idx[hs]] = col * (1 - fk) + fc * fk
                emis[idx[hs]] = e * (1 - fk[:, 0])
            hit[h[solid]] = True
            hit[h[~solid]] = False
        # dead = hit or left world
        leaving = ~inb & np.any(((ip < 0) & (step < 0)) | ((ip >= sz) & (step > 0)), axis=1)
        keep = ~(hit | leaving)
        if not keep.all():
            idx, ip, tmax, step, tdel = idx[keep], ip[keep], tmax[keep], step[keep], tdel[keep]
    return out.reshape(H, W, 3), emis.reshape(H, W)

def _ao(world, cell, a, step, fr, sz):
    """Cheap ambient occlusion from the 8 voxels around the visible face."""
    n = len(a)
    r = np.arange(n)
    front = cell.copy()
    front[r, a] -= step[r, a]
    t1 = np.where(a == 0, 1, 0); t2 = np.where(a == 2, 1, 2)
    f1 = fr[r, t1]; f2 = fr[r, t2]
    def occ(o1, o2):
        c = front.copy()
        c[r, t1] += o1; c[r, t2] += o2
        ok = np.all((c >= 0) & (c < sz), axis=1)
        v = np.zeros(n, bool)
        v[ok] = world[c[ok, 0], c[ok, 1], c[ok, 2]] > 0
        return v
    wp1 = np.clip(f1 * 2 - 1, 0, 1) ** 2; wn1 = np.clip(1 - f1 * 2, 0, 1) ** 2
    wp2 = np.clip(f2 * 2 - 1, 0, 1) ** 2; wn2 = np.clip(1 - f2 * 2, 0, 1) ** 2
    s_p1, s_n1, s_p2, s_n2 = occ(1, 0), occ(-1, 0), occ(0, 1), occ(0, -1)
    o = s_p1 * wp1 + s_n1 * wn1 + s_p2 * wp2 + s_n2 * wn2
    o = o + occ(1, 1) * (~s_p1 & ~s_p2) * wp1 * wp2 + occ(1, -1) * (~s_p1 & ~s_n2) * wp1 * wn2 \
          + occ(-1, 1) * (~s_n1 & ~s_p2) * wn1 * wp2 + occ(-1, -1) * (~s_n1 & ~s_n2) * wn1 * wn2
    return 1 - 0.45 * np.clip(o, 0, 1)

def look_at(pos, target):
    d = np.asarray(target, float) - np.asarray(pos, float)
    yaw = np.arctan2(d[0], d[2])
    pitch = np.arctan2(d[1], np.hypot(d[0], d[2]))
    return yaw, pitch
