"""Voxel worlds for each scene."""
import numpy as np
from voxel import *

def _tree(w, x, y, z, h=5):
    w[x, y:y + h, z] = LOG
    for dy in range(h - 2, h + 2):
        rad = 2 if dy < h else 1
        for dx in range(-rad, rad + 1):
            for dz in range(-rad, rad + 1):
                if abs(dx) + abs(dz) <= rad + 1 and w[x + dx, y + dy, z + dz] == AIR:
                    w[x + dx, y + dy, z + dz] = LEAVES

def farm_world():
    """Flat grassland with a big wheat farm (water channels) and some trees."""
    X, Y, Z = 112, 32, 112
    w = np.zeros((X, Y, Z), np.uint8)
    g = 8
    w[:, :g - 3] = STONE; w[:, g - 3:g] = DIRT; w[:, g] = GRASS
    r = np.random.default_rng(3)
    # farm plot
    x0, x1, z0, z1 = 30, 82, 24, 92
    w[x0:x1, g, z0:z1] = FARM
    w[x0:x1, g + 1, z0:z1] = WHEAT
    for xx in range(x0 + 4, x1, 9):
        w[xx, g, z0:z1] = WATER; w[xx, g + 1, z0:z1] = AIR
    # fence-ish border of planks
    w[x0 - 1, g + 1, z0 - 1:z1 + 1] = PLANKS; w[x1, g + 1, z0 - 1:z1 + 1] = PLANKS
    w[x0 - 1:x1 + 1, g + 1, z0 - 1] = PLANKS; w[x0 - 1:x1 + 1, g + 1, z1] = PLANKS
    w[x0 + 20:x0 + 23, g + 1, z0 - 1] = AIR
    for _ in range(40):
        x, z = r.integers(4, X - 4), r.integers(4, Z - 4)
        if x0 - 4 < x < x1 + 4 and z0 - 4 < z < z1 + 4:
            continue
        _tree(w, x, g + 1, z, r.integers(4, 7))
    return w, g

def donut_world():
    """Floating giant voxel donut with pink frosting and sprinkles above a grass island."""
    X, Y, Z = 136, 80, 136
    w = np.zeros((X, Y, Z), np.uint8)
    g = 6
    w[:, :g] = DIRT; w[:, g] = GRASS
    r = np.random.default_rng(11)
    for _ in range(70):
        x, z = r.integers(4, X - 4), r.integers(4, Z - 4)
        if 44 < x < 92 and 44 < z < 92:
            continue
        _tree(w, x, g + 1, z, r.integers(4, 7))
    cx, cy, cz = 68, 28, 68
    R, rr = 13.0, 6.0
    xs, ys, zs = np.mgrid[0:X, 0:Y, 0:Z].astype(np.float32)
    # donut lies tilted towards camera a bit: rotate around x axis
    ang = np.radians(-8)
    dy, dz = ys - cy, zs - cz
    ry = dy * np.cos(ang) - dz * np.sin(ang)
    rz = dy * np.sin(ang) + dz * np.cos(ang)
    rx = xs - cx
    q = np.sqrt(rx ** 2 + rz ** 2) - R
    dist = np.sqrt(q ** 2 + ry ** 2)
    inside = dist < rr
    w[inside] = DOUGH
    frost = inside & (ry > -1.0 + 1.6 * np.sin(np.arctan2(rz, rx) * 7) * 0.6)
    w[frost] = FROST
    # sprinkles on frosting surface
    surf = frost & (dist > rr - 1.3)
    cand = np.argwhere(surf)
    pick = cand[r.random(len(cand)) < 0.07]
    cols = np.array([SPR_R, SPR_Y, SPR_B, SPR_W, SPR_G], np.uint8)
    w[pick[:, 0], pick[:, 1], pick[:, 2]] = cols[r.integers(0, 5, len(pick))]
    return w, (cx, cy, cz)

def hugo_world():
    """Giant standing gold coin with an 'H' on a pile of gold/emerald blocks."""
    X, Y, Z = 136, 56, 136
    w = np.zeros((X, Y, Z), np.uint8)
    g = 6
    w[:, :g] = DIRT; w[:, g] = GRASS
    r = np.random.default_rng(12)
    cx, cy, cz = 68, 26, 68
    # treasure pile
    xs, ys, zs = np.mgrid[0:X, 0:Y, 0:Z].astype(np.float32)
    pile = (ys - g) < (12 - 0.012 * ((xs - cx) ** 2 + (zs - cz) ** 2) * 1.0 + r.random((X, Y, Z)) * 1.5)
    pile &= ys > g
    w[pile] = GOLD
    pts = np.argwhere(pile)
    sel = pts[r.random(len(pts)) < 0.08]
    w[sel[:, 0], sel[:, 1], sel[:, 2]] = EMERALD
    sel = pts[r.random(len(pts)) < 0.04]
    w[sel[:, 0], sel[:, 1], sel[:, 2]] = DIAMOND
    # coin: disk in x-y plane, thickness 3 in z
    rad = 14
    disk = (np.hypot(xs - cx, ys - cy) < rad) & (np.abs(zs - cz) <= 1.5)
    w[disk] = GOLD
    rim = disk & (np.hypot(xs - cx, ys - cy) > rad - 1.6)
    w[rim] = GOLDD
    # H letter embossed on both faces (raised by one block)
    H = ["X....X", "X....X", "X....X", "XXXXXX", "X....X", "X....X", "X....X"]
    s = 2
    for j, row in enumerate(H):
        for i, ch in enumerate(row):
            if ch == "X":
                x = cx + (i - 3) * s; y = cy + (3 - j) * s
                for zz in (cz - 2, cz + 2):
                    w[x:x + s, y - s + 1:y + 1, zz] = GOLDD
    for _ in range(70):
        x, z = r.integers(4, X - 4), r.integers(4, Z - 4)
        if 40 < x < 96 and 40 < z < 96:
            continue
        _tree(w, x, g + 1, z, r.integers(4, 7))
    return w, (cx, cy, cz)

def tunnel_world():
    """Long dark deepslate tunnel with ores and a glowing portal at the end (along +z)."""
    X, Y, Z = 21, 21, 160
    w = np.full((X, Y, Z), DEEP, np.uint8)
    r = np.random.default_rng(13)
    xs, ys = np.mgrid[0:X, 0:Y]
    hole = (np.abs(xs - 10) <= 3) & (np.abs(ys - 10) <= 3)
    w[hole, :] = AIR
    shell = (np.abs(xs - 10) <= 4) & (np.abs(ys - 10) <= 4) & ~hole
    for z in range(Z):
        m = shell & (r.random((X, Y)) < 0.10)
        w[m, z] = r.choice([DIA_ORE, EM_ORE, GOLD_ORE, STONE], size=m.sum())
        if z % 12 == 0:
            w[10, 14, z] = GLOW; w[6, 10, z] = GLOW; w[14, 10, z] = GLOW
    w[:, :, Z - 4:] = GLOW
    w[hole, Z - 4:] = GLOW
    w[:, :, 0] = DEEP
    return w
