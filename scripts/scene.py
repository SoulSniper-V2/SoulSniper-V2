"""Paints the night: every layer is a luminance field, dithered to one bit."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 420, 236
MOON = (292, 76, 54)          # cx, cy, r
HORIZON = 140
MUZZLE = (171, 100)
HS = 1.5                      # hunter scale
HO = (50, 114)                # hunter origin
rng = np.random.default_rng(7)

yy, xx = np.mgrid[0:H, 0:W].astype(float)


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def vnoise(x, y, seed):
    r = np.random.default_rng(seed)
    g = r.random((256, 256))
    xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - xi, y - yi
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    q = lambda a, b: g[(yi + b) & 255, (xi + a) & 255]
    return (q(0, 0) * (1 - fx) + q(1, 0) * fx) * (1 - fy) + (q(0, 1) * (1 - fx) + q(1, 1) * fx) * fy


def fbm(x, y, seed, o=5):
    s, a, t = 0, 0.5, 0
    for i in range(o):
        s = s + a * vnoise(x * 2 ** i + 17 * i, y * 2 ** i + 5 * i, seed + i)
        t += a
        a *= 0.5
    return s / t


# ---------- dithering ----------
def atkinson(L):
    """Atkinson error diffusion: blown highlights, crushed blacks, the old Mac look."""
    a = np.pad(L.astype(float).copy(), ((0, 2), (2, 2)))
    out = np.zeros(L.shape, bool)
    for y in range(L.shape[0]):
        d = 1 if y % 2 == 0 else -1
        xs = range(L.shape[1]) if d == 1 else range(L.shape[1] - 1, -1, -1)
        for x in xs:
            v = a[y, x + 2]
            on = v > 0.5
            out[y, x] = on
            e = (v - on) / 8
            a[y, x + 2 + d] += e
            a[y, x + 2 + 2 * d] += e
            a[y + 1, x + 2 - d] += e
            a[y + 1, x + 2] += e
            a[y + 1, x + 2 + d] += e
            a[y + 2, x + 2] += e
    return out


def _bayer(n):
    if n == 1:
        return np.array([[0]])
    b = _bayer(n // 2)
    return np.block([[4 * b, 4 * b + 2], [4 * b + 3, 4 * b + 1]])


B8 = (_bayer(8) + 0.5) / 64


def bayer(L):
    h, w = L.shape
    return L > np.tile(B8, (h // 8 + 1, w // 8 + 1))[:h, :w]


# ---------- shapes ----------
def ridge(base, amp, scale, seed, o=5):
    x = np.arange(W) / scale
    n = fbm(x, np.zeros(W) + 3.3, seed, o)
    return base - amp * (np.abs(n * 2 - 1) * -1 + 1)


def poly_mask(pts, ss=4):
    im = Image.new("L", (W * ss, H * ss))
    ImageDraw.Draw(im).polygon([(x * ss, y * ss) for x, y in pts], fill=255)
    return np.asarray(im.resize((W, H), Image.BOX)) > 127


def cliff_top():
    """Height of the near cliff for each column (H+1 where there is none)."""
    x = np.arange(W)
    pts = [(-1, 96), (12, 103), (26, 112), (40, 116), (60, 117), (90, 118), (122, 119), (146, 121), (156, 126),
           (161, 137), (163, 150), (171, 161), (175, 178), (186, 192), (197, 208), (216, 222), (240, 240)]
    top = np.interp(x, [p[0] for p in pts], [p[1] for p in pts])
    top = top + (fbm(x / 6, np.zeros(W), 91, 4) - 0.5) * 5 * np.where((x > 34) & (x < 150), 0.35, 1.3)
    return top


CLIFF = cliff_top()


def terrain(flash=0.0, t=0):
    """Return luminance of the whole static scene. flash lights it from the muzzle."""
    mx, my, mr = MOON
    d = np.hypot(xx - mx, yy - my)
    # sky: a falloff of light round the moon, torn by cloud
    L = 0.03 + 0.46 * np.exp(-np.maximum(d - mr, 0) / 42) + 0.10 * np.exp(-np.maximum(d - mr, 0) / 130)
    c = fbm(xx / 70 + 3, yy / 17, 33, 5)
    streak = smooth(0.50, 0.74, c)
    lit = 0.12 + 0.75 * np.exp(-np.maximum(d - mr, 0) / 70)
    L = L * (1 - 0.35 * smooth(0.35, 0.5, c)) + streak * lit * 0.42 * smooth(8, 60, yy)
    L += 0.10 * smooth(95, HORIZON, yy)
    L += 0.50 * np.exp(-((xx - 135) / 78) ** 2 - ((yy - 108) / 34) ** 2)
    L += 0.50 * np.exp(-((xx - 146) / 44) ** 2 - ((yy - 100) / 17) ** 2)

    # stars, kept off the moonlight
    for _ in range(150):
        sx, sy = rng.integers(0, W), rng.integers(0, 110)
        if np.hypot(sx - mx, sy - my) > mr + 46:
            L[sy, sx] = 1.3

    # mountains, three ranges deep
    for base, amp, scale, seed, tone, lo in [(142, 52, 62, 12, 0.30, 0.10), (144, 36, 40, 27, 0.17, 0.05), (145, 17, 24, 41, 0.07, 0.02)]:
        top = ridge(base, amp, scale, seed)
        m = (yy > top[None, :]) & (yy < HORIZON + 4)
        depth = yy - top[None, :]
        slope = np.gradient(top)[None, :]
        n = fbm(xx / 9 + depth / 14, yy / 11, seed + 3, 4)
        face = np.sign(mx - xx) * slope
        sh = tone * (0.35 + 1.5 * smooth(-0.6, 1.0, face + (n - 0.5) * 3.2)) * np.exp(-depth / 26) + lo
        sh += 0.16 * smooth(112, HORIZON, yy) * np.exp(-((xx - mx) / 150) ** 2)   # mist in the valley
        L = np.where(m, sh, L)
        sh = np.where(xx < 205, np.maximum(sh, 0.74 * np.exp(-((xx - 140) / 62) ** 2 - ((yy - 102) / 24) ** 2)), sh)
        rim = (depth >= 0) & (depth < 1.2) & m
        L = np.where(rim, L + 0.35 * np.exp(-np.abs(xx - mx) / 120), L)

    # lake
    ly = yy - HORIZON
    lake = yy >= HORIZON
    persp = 1 / (np.maximum(ly, 0) + 4)
    wav = fbm(xx * (2.6 * persp) + 40, 60 * persp * 4, 61, 3) * 0.55 + fbm(xx / 11, yy / 1.7, 64, 3) * 0.45 * (1 - 0.0) + 0.05 * np.sin(6.283 * t / 6 + xx / 7 + vnoise(xx / 20, yy / 2, 70) * 9)
    width = 20 + ly * 0.75
    col = np.exp(-((xx - mx) / width) ** 2)
    glit = smooth(0.60 - 0.22 * col, 0.70, wav + 0.0) * (0.10 + 1.15 * col)
    lk = 0.035 + 0.10 * np.exp(-((xx - mx) / 120) ** 2) + glit
    lk = np.where(ly < 1.5, 0.75 * np.exp(-((xx - mx) / 110) ** 2) + 0.1, lk)
    L = np.where(lake, lk, L)

    # far shore pines
    r2 = np.random.default_rng(5)
    for x0 in range(150, W + 6, 3):
        edge = smooth(330, 420, x0) + smooth(205, 150, x0)
        if r2.random() > 0.25 + 0.75 * edge:
            continue
        h = r2.integers(4, 9) + int(16 * smooth(350, 420, x0) * (0.6 + 0.6 * r2.random()))
        base = HORIZON + 1 + int(5 * smooth(370, 420, x0))
        for k in range(h):
            half = (k % 4) * 0.55 + k * 0.17
            y = base - h + k
            L[y, max(0, int(x0 - half)):max(0, min(W, int(x0 + half) + 1))] = 0.01

    # near cliff
    cm = yy > CLIFF[None, :]
    depth = yy - CLIFF[None, :]
    rock = fbm(xx / 5 + yy / 30, yy / 13 + xx / 40, 77, 5)
    strata = fbm(xx / 40, yy / 3.0, 79, 3)
    gx, gy = np.gradient(fbm(xx / 13 + yy / 60, yy / 22 + xx / 70, 201, 5))[::-1]
    facet = np.clip(0.5 + 13 * (gx * 0.9 - gy * 0.8), 0, 1)                   # lit from the moon side
    crack = np.abs(fbm(xx / 16 + yy / 50, yy / 34, 210, 4) - 0.5)
    tow = 0.22 + 1.15 * smooth(20, 185, xx + (yy - 120) * 0.35)                 # nearer the moon, brighter
    cl = (0.03 + 0.62 * smooth(0.42, 0.95, facet)) * tow * np.exp(-np.maximum(depth - 8, 0) / 75)
    cl -= 0.5 * (crack < 0.022) + 0.22 * smooth(0.6, 0.75, strata) * 0.6
    edge = np.interp(yy[:, 0], np.maximum.accumulate(CLIFF[150:]), np.arange(150, W))[:, None] - xx
    cl += 0.65 * smooth(9, 0, edge) * smooth(0.40, 0.55, rock) * (yy > 126)
    cl = np.where(depth < 1.1, 0.40 + 0.5 * smooth(20, 140, xx), cl)
    L = np.where(cm, cl, L)

    if flash:
        fd = np.hypot(xx - MUZZLE[0], yy - MUZZLE[1])
        L = L + flash * np.exp(-fd / 60) * (0.75 + 0.5 * rock)
    return L


def moon_tex():
    mx, my, mr = MOON
    u, v = (xx - mx) / mr, (yy - my) / mr
    rr = np.sqrt(u * u + v * v)
    z = np.sqrt(np.clip(1 - rr * rr, 0, 1))
    m = rr <= 1
    sx, sy = u / (0.35 + 0.65 * z + 1e-3), v / (0.35 + 0.65 * z + 1e-3)
    n = fbm(sx * 2.1 + 9, sy * 2.1 + 4, 101, 5)
    mare = smooth(0.46, 0.60, n)
    L = 0.98 - 0.52 * mare - 0.22 * smooth(0.45, 0.7, fbm(sx * 6, sy * 6, 120, 4))
    r3 = np.random.default_rng(22)
    for _ in range(38):
        a, q = r3.random() * 6.283, math.sqrt(r3.random()) * 0.92
        cx, cy, cr = q * math.cos(a), q * math.sin(a), 0.03 + 0.11 * r3.random() ** 2
        dd = np.hypot((u - cx) / max(0.35, math.sqrt(1 - q * q)), v - cy)
        L += 0.30 * np.exp(-((dd - cr) / 0.02) ** 2) - 0.55 * smooth(cr, cr * 0.4, dd) * smooth(-0.2, 1, -(u - cx) / cr)
    light = np.clip(0.80 * z + 0.55 * u * -0.45 - 0.55 * v * 0.5 + 0.25, 0, 1.1)
    return L * np.maximum(0.52, 0.22 + 0.80 * light), m, u, v


def eye_layer(openness, gaze=(0.0, 0.0)):
    """The moon with its lid open 0..1. Iris is drawn separately so it can look around."""
    L, m, u, v = moon_tex()
    if openness <= 0:
        return L, m, None
    half = 0.50 * openness
    shape = half * np.sqrt(np.clip(1 - (u / 0.86) ** 2, 0, 1)) ** 1.35
    inside = (np.abs(v + 0.10) < shape) & (np.abs(u) < 0.86)
    lid = (np.abs(v + 0.10) < shape + 0.055) & (np.abs(u) < 0.92) & ~inside
    fold = np.exp(-((np.abs(v + 0.10) - shape - 0.17) / 0.035) ** 2) * (np.abs(u) < 0.8)
    L = L - 0.55 * fold * openness
    white = 1.15 - 0.75 * smooth(0.3, 0.9, np.abs(u)) - 0.75 * smooth(0.10, 0.0, shape - np.abs(v + 0.10) + 0.02) \
        - 0.5 * smooth(0.0, -0.5, v + half * 0.35)
    veins = fbm(u * 7, v * 9 + u * 3, 133, 4)
    white -= 0.8 * (np.abs(veins - 0.5) < 0.018) * smooth(0.25, 0.8, np.abs(u))
    gx, gy = gaze[0], gaze[1] - 0.10
    q = np.hypot(u - gx, v - gy)
    ang = np.arctan2(v - gy, u - gx)
    spokes = fbm(ang * 7 + 30, q * 5, 140, 3)
    iris = 0.18 + 0.75 * smooth(0.40, 0.68, spokes) * smooth(0.16, 0.30, q)
    iris = np.where(q > 0.345, 0.0, iris)
    iris = np.where(q < 0.17, 0.0, iris)
    iris = np.where(np.hypot(u - gx + 0.10, v - gy + 0.12) < 0.065, 1.4, iris)
    white = np.where(q < 0.385, iris, white)
    L = np.where(inside, white, L)
    L = np.where(lid, 0.0, L)
    return L, m, inside


# ---------- the hunter ----------
def hunter():
    """Prone on the ledge. Returns (body, rim) masks at native size."""
    ss = 6
    im = Image.new("L", (W * ss, H * ss))
    d = ImageDraw.Draw(im)
    P = lambda pts: d.polygon([((HO[0] + x * HS) * ss, (HO[1] + y * HS) * ss) for x, y in pts], fill=255)
    L_ = lambda pts, w: d.line([((HO[0] + x * HS) * ss, (HO[1] + y * HS) * ss) for x, y in pts], fill=255, width=int(w * HS * ss), joint="curve")
    C = lambda x, y, r: d.ellipse([0, 0, 0, 0], fill=255)
    # legs, coat, back
    P([(-8, 2), (-7, -1.5), (8, -4), (22, -8), (33, -10.5), (40, -9.5), (44, -4), (44, 3), (20, 4), (-6, 4)])
    P([(-12, 3), (-13, -3), (-10, -2), (-6, 0), (-6, 3)])                      # boots
    P([(6, -3.5), (20, -8.5), (30, -11), (34, -5), (18, 2), (-2, 2)])           # coat
    # hood
    P([(33, -10), (35.5, -15.5), (40, -18), (45.2, -16.8), (47.4, -13), (46.2, -9.2), (41, -7), (35, -6)])
    # arms
    L_([(38, -7), (45, -2.2), (52, -7.6)], 3.6)
    L_([(36, -6), (42, 1), (47, -1), (49.4, -5)], 3.4)
    # rifle
    P([(40, -8.6), (45, -10.2), (52, -10), (52, -6.8), (47, -6.6), (44, -4.4), (40.5, -4.6)])   # stock
    P([(50, -10), (63, -10), (63, -7.6), (50, -7)])                              # receiver
    L_([(62, -8.8), (79, -8.8)], 1.25)                                           # barrel
    P([(76.6, -10), (81, -10), (81, -7.6), (76.6, -7.6)])                        # brake
    P([(49, -13.7), (61, -13.7), (63, -14.4), (65.5, -14.4), (65.5, -11), (63, -11), (61, -11.8), (49, -11.8)])  # scope
    L_([(54, -12), (54, -9)], 1.2); L_([(59, -12), (59, -9)], 1.2)
    L_([(55, -7.2), (54.2, -3.4)], 1.8)                                          # magazine
    L_([(66, -8), (63.4, 3)], 0.95); L_([(67, -8), (70.6, 3.6)], 0.95)           # bipod
    body = np.asarray(im.resize((W, H), Image.BOX)) > 118
    up = np.zeros_like(body); up[1:] = body[:-1]
    rim = body & ~up & (xx < 116)                                                            # top-lit edge
    return body, rim


def scarf(frame, n=8):
    """Scarf streaming behind the hunter: one native-resolution mask per frame."""
    m = np.zeros((H, W), bool)
    ph = frame / n * 2 * math.pi
    for i in range(0, 46):
        t = i / 46
        x = 106 - i * 1.15
        y = 99 - 9 * t - 4 * t * t + math.sin(t * 8.5 + ph) * (0.6 + 5.0 * t)
        th = 4.2 - 2.2 * t
        for k in range(int(round(th)) + 1):
            yi = int(round(y - th / 2 + k))
            if 0 <= yi < H:
                m[yi, int(x)] = m[yi, int(x) - 1] = True
    return m


def grass(frame, n=6):
    """Foreground reeds, bending. Returns (solid, lit-edge) masks."""
    r = np.random.default_rng(404)
    ss = 4
    im = Image.new("L", (W * ss, H * ss)); d = ImageDraw.Draw(im)
    hi = Image.new("L", (W * ss, H * ss)); dh = ImageDraw.Draw(hi)
    ph = frame / n * 2 * math.pi
    top = lambda x: 224 - 15 * smooth(250, 420, x) - 5 * math.sin(x / 31) - 8 * smooth(90, 0, x)
    d.polygon([(0, H * ss)] + [(x * ss, top(x) * ss) for x in range(0, W + 4, 4)] + [(W * ss, H * ss)], fill=255)
    for _ in range(330):
        x = r.uniform(-6, W + 6); b = float(top(x)) + 4
        h = r.uniform(9, 22) + 16 * float(smooth(300, 420, x)) * r.random() + 9 * float(smooth(70, 0, x)) * r.random()
        lean = r.uniform(-5, 7); sway = math.sin(ph + x / 24 + r.uniform(0, 1)) * (1.1 + h / 13)
        pts = [((x + (lean + sway) * t * t) * ss, (b - h * t) * ss) for t in np.linspace(0, 1, 7)]
        d.line(pts, fill=255, width=int(ss * r.uniform(0.9, 1.7)))
        if r.random() < 0.34:
            dh.line(pts[3:], fill=255, width=int(ss * 1.0))
        if h > 27 and r.random() < 0.6:                                         # cattail
            ex, ey = pts[-1]
            for q in (d, dh if r.random() < .4 else d):
                q.line([(ex, ey), (ex - 1, ey + 6 * ss)], fill=255, width=int(2.1 * ss))
    solid = np.asarray(im.resize((W, H), Image.BOX)) > 96
    lit = (np.asarray(hi.resize((W, H), Image.BOX)) > 110) & solid
    return solid, lit


def preview(path, openness=0.0, flash=0.0):
    L = terrain(flash)
    bg = atkinson(L)
    ML, m, inside = eye_layer(openness)
    mb = atkinson(np.where(m, ML, 0))
    sky_front = occluder()
    bg = np.where(m & ~sky_front, mb, bg)
    body, rim = hunter()
    sc = scarf(0)
    bg[body] = False; bg[rim] = True
    s, l = grass(0)
    bg[s] = l[s]
    img = np.where(bg[..., None], np.array([240, 234, 216]), np.array([13, 17, 23])).astype(np.uint8)
    img[sc & ~body] = [255, 48, 38]
    Image.fromarray(img).resize((W * 3, H * 3), Image.NEAREST).save(path)


def occluder():
    """Everything in front of the moon (nothing is, unless the ridges reach it)."""
    t = np.full(W, 999.0)
    for base, amp, scale, seed in [(142, 52, 62, 12), (144, 36, 40, 27), (145, 17, 24, 41)]:
        t = np.minimum(t, ridge(base, amp, scale, seed))
    return (yy > t[None, :]) | (yy > CLIFF[None, :])


if __name__ == "__main__":
    import sys
    preview(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0, float(sys.argv[3]) if len(sys.argv) > 3 else 0)
