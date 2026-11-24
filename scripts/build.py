"""Builds every SVG in ../assets. Run: python scripts/build.py (needs numpy, pillow)."""
import base64, io, math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scene as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets")
W, H = S.W, S.H
FPS = 12
RED = "#ff3026"
THEMES = {"dark": ((240, 234, 216), (13, 17, 23)), "light": ((255, 255, 255), (20, 22, 28))}   # (lit, unlit)

# ---------- cast ----------
# '#' lit, 'o' dark. The hem is added by code so it can ripple.
CAST = [
    dict(key="snapstate", name="SnapState", at=(204, 74), hem=True, say="Puts every Mac window back where it was.",
         art=["###############", "#ooooooooooooo#", "#o#o#o#oooooo##", "#ooooooooooooo#", "###############", "###############",
              "###oo#####oo###", "###oo#####oo###", "###oo#####oo###", "###############", "######ooo######", "###############", "###############"]),
    dict(key="exhibit", name="Exhibit", at=(206, 150), hem=True, say="Turns your texts into PDF exhibits.",
         art=["###########....", "############o...", "############oo..", "############ooo.", "###############.", "###oo####oo####.", "###oo####oo####.",
              "###oo####oo####.", "###############.", "##ooooooooooo##.", "###############.", "##oooooooo#####.", "###############."]),
    dict(key="seed", name="Seed", at=(256, 178), hem=True, say="A coding agent that grows into your tool.",
         art=["...##.....##...", "..####...####..", "...####.####...", ".....##.##.....", ".......#.......", ".......#.......", ".....#####.....",
              "...#########...", "..###########..", ".#############.", ".###oo###oo###.", ".###oo###oo###.", ".#############.", ".######o######.", ".#############."]),
    dict(key="flick", name="Flick", at=(326, 160), hem=True, say="Lets an agent drive a real iPhone.",
         art=["............##.", "............##.", "............##.", "....#####...##.", "..#########.##.", ".##############", ".##############", "###oo###oo#####",
              "###oo###oo####.", "#############..", "####ooooo####..", "#############..", "#############.."]),
    dict(key="relay", name="Agent Relay", at=(384, 130), hem=True, say="Lets your agent talk to someone else's.",
         art=["###############", "#o###########o#", "##o#########o##", "###oo#####oo###", "#####oo#oo#####", "#######o#######", "###############",
              "###oo#####oo###", "###oo#####oo###", "###############", "######ooo######", "###############", "###############"]),
    dict(key="hoplite", name="hoplite-ios", at=(384, 48), hem=True, say="Unofficial iPhone app for cloud agents.",
         art=["....#########....", "..#############..", ".####.......####.", ".##..#######..##.", ".#.###########.#.", "..#############..", "..#############..", "..##ooo###ooo##..",
              "..##oooo#oooo##..", "..###ooo#ooo###..", "..######o######..", "..####ooooo####..", "..#####ooo#####..", "..#####ooo#####.."]),
]
LINKS_AT = {}


def epx(g):
    """Scale2x: doubles pixel art and rounds its corners."""
    p = np.pad(g, 1, constant_values=".")
    A, B, C, D = p[:-2, 1:-1], p[1:-1, 2:], p[1:-1, :-2], p[2:, 1:-1]
    o = np.repeat(np.repeat(g, 2, 0), 2, 1)
    o[0::2, 0::2] = np.where((C == A) & (C != D) & (A != B), A, g)
    o[0::2, 1::2] = np.where((A == B) & (A != C) & (B != D), B, g)
    o[1::2, 0::2] = np.where((D == C) & (D != B) & (C != A), C, g)
    o[1::2, 1::2] = np.where((B == D) & (B != A) & (D != C), D, g)
    return o


def sprite(c, frame, scared=False):
    """Return (lit, dark) boolean masks, padded by one for the outline."""
    w = max(map(len, c["art"]))
    rows = [r.ljust(w, ".") for r in c["art"]]
    grid = [list(r) for r in rows]
    if c["hem"]:
        last = "".join("#" if ch in "#o" else "." for ch in rows[-1])
        for k in range(5):
            r = []
            for x in range(w):
                wave = 2.6 + 2.1 * math.sin(x * 1.25 + frame * math.pi / 2)
                r.append("#" if last[x] == "#" and k < wave else ".")
            grid.append(r)
    g = epx(np.array(grid))
    h, w = g.shape
    lit = np.zeros((h + 2, w + 4), bool); drk = lit.copy()
    for y in range(h):
        sh = 0 if y < 2 * len(rows) else int(round(math.sin(frame * math.pi / 2 + y / 2) * 0.9))
        for x in range(w):
            if g[y, x] == "#": lit[y + 1, x + 2 + sh] = True
            if g[y, x] == "o": drk[y + 1, x + 2 + sh] = True
    body = lit | drk
    o = np.zeros_like(body)
    o[1:] |= body[:-1]; o[:-1] |= body[1:]; o[:, 1:] |= body[:, :-1]; o[:, :-1] |= body[:, 1:]
    return lit, drk | (o & ~body)


# ---------- png / svg helpers ----------
def png(lit=None, drk=None, red=None, theme="dark"):
    shp = next(a for a in (lit, drk, red) if a is not None).shape
    idx = np.zeros(shp, np.uint8)
    if drk is not None: idx[drk] = 1
    if lit is not None: idx[lit] = 2
    if red is not None: idx[red] = 3
    L, D = THEMES[theme]
    im = Image.fromarray(idx, "P")
    im.putpalette([255, 0, 255, *D, *L, 255, 48, 38])
    b = io.BytesIO()
    im.save(b, "PNG", transparency=0, optimize=True, bits=2)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def img(uri, x, y, w, h, extra=""):
    return f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{uri}" {extra}/>'


def stack(masks):
    return np.concatenate(masks, axis=0)


def strip(uri, x, y, w, h, n, fps, begin=0):
    vals = ";".join(str(-h * i) for i in range(n))
    return (f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="0 0 {w} {h}" overflow="hidden">'
            f'<image width="{w}" height="{h * n}" href="{uri}"><animate attributeName="y" values="{vals}" calcMode="discrete" '
            f'dur="{n / fps:.3f}s" begin="{begin}s" repeatCount="indefinite"/></image></svg>')


def keyed(attr, frames, dur, tag="animate", extra=""):
    """Discrete animation with runs collapsed. frames: one value per 1/FPS s."""
    ks, vs = [], []
    for i, v in enumerate(frames):
        if not vs or v != vs[-1]:
            vs.append(v); ks.append(i / len(frames))
    if len(vs) == 1:
        return f'<set attributeName="{attr}" to="{vs[0]}"/>' if tag == "animate" else \
            f'<animateTransform attributeName="transform" type="translate" values="{vs[0]}" dur="{dur}s" repeatCount="indefinite"/>'
    kt = ";".join(f"{k:.4f}".rstrip("0").rstrip(".") or "0" for k in ks)
    return (f'<{tag} attributeName="{attr}" {extra} values="{";".join(vs)}" keyTimes="{kt}" '
            f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')


def show(on, N, dur):
    """visibility track from a set of (start, end) seconds."""
    fr = ["hidden"] * N
    for a, b in on:
        for i in range(max(0, round(a * FPS)), min(N, round(b * FPS))):
            fr[i] = "visible"
    return keyed("visibility", fr, dur)


FONT = {"title": ("Jacquard24-Regular", None), "name": ("Jacquard12-Regular", None), "body": ("PixelifySans", 12)}
SIZES = {"Jacquard24-Regular": 43, "Jacquard12-Regular": 21, "PixelifySans": 11}   # one font cell = one pixel


def text(s, kind, scale=1):
    """Bitmap text, sampled on the font's own grid, trimmed."""
    fn, _ = FONT[kind]
    k = 8
    f = ImageFont.truetype(f"{HERE}/fonts/{fn}.ttf", SIZES[fn] * k)
    cols, x = [], 0
    canvas = np.zeros((40, 700), bool)
    for ch in s:
        t = Image.new("L", (60 * k, 40 * k)); ImageDraw.Draw(t).text((8 * k, 2 * k), ch, font=f, fill=255)
        a = np.asarray(t.resize((60, 40), Image.BOX)) > 127
        adv = max(1, round(f.getlength(ch) / k))
        yy, xx = np.nonzero(a)
        for y0, x0 in zip(yy, xx):
            xp = x + x0 - 8 + 2
            if 0 <= xp < 700: canvas[y0, xp] = True
        x += adv
    ys, xs = np.nonzero(canvas)
    out = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.kron(out, np.ones((scale, scale), bool)).astype(bool) if scale > 1 else out


def outlined(m, pad=2):
    a = np.pad(m, pad)
    o = a.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            o |= np.roll(np.roll(a, dy, 0), dx, 1)
    return a, o & ~a


def wrap(s, n):
    out, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 > n and cur: out.append(cur); cur = w
        else: cur = (cur + " " + w).strip()
    return out + [cur]


def ease(t):
    t = min(1, max(0, t))
    return 1 - (1 - t) ** 3


# ---------- effects ----------
def halo(r, k):
    y, x = np.mgrid[-r:r, -r:r] + 0.5
    return S.bayer(0.62 * np.clip(1 - np.hypot(x, y) / r, 0, 1) ** 1.6 * k)


def burst(frames=8, r=26):
    g = np.random.default_rng(3)
    parts = [(g.uniform(0, 6.283), g.uniform(0.35, 1), g.integers(1, 3)) for _ in range(60)]
    out = []
    for f in range(frames):
        m = np.zeros((2 * r, 2 * r), bool)
        t = (f + 1) / frames
        for a, v, s in parts:
            if t > 0.45 + 0.55 * v: continue
            d = (r - 2) * v * ease(t)
            px, py = int(r + math.cos(a) * d), int(r + math.sin(a) * d + 5 * t * t)
            m[max(0, py):py + (s if t < .5 else 1), max(0, px):px + (s if t < .5 else 1)] = True
        y, x = np.mgrid[-r:r, -r:r] + 0.5
        if f < 4: m |= np.abs(np.hypot(x, y) - (5 + f * 5)) < (1.6 - f * 0.35)
        out.append(m)
    return out


def edge_mask():
    d = np.minimum.reduce([S.xx, W - 1 - S.xx, S.yy, H - 1 - S.yy])
    n = S.fbm(S.xx / 9, S.yy / 9, 300, 3)
    return ~S.bayer(S.smooth(-1.5, 11, d + (n - 0.5) * 9))


# ---------- hero ----------
def hero(theme):
    T0, STEP = 2.0, 3.0
    n = len(CAST)
    END = T0 + STEP * n                       # moon wakes up
    DUR = END + 9.0
    N = round(DUR * FPS)
    mx, my, mr = S.MOON
    occ = S.occluder()

    base = S.terrain()
    moon, mask, _ = S.eye_layer(0)
    vis = mask & ~occ
    bg = np.where(vis, S.atkinson(np.where(mask, moon, 0)), S.atkinson(base))
    fl = S.atkinson(S.terrain(1.35)); fl = np.where(vis, bg, fl)
    edge = edge_mask()

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W * 2}" height="{H * 2}" role="img" '
           f'aria-label="A sniper on a cliff hunts six ghosts over a moonlit lake. Each ghost is a project. Then the moon opens its eye.">',
           '<style>image{image-rendering:pixelated}@media(max-width:560px){image{image-rendering:auto}}'
           '.still{display:none}@media(prefers-reduced-motion:reduce){.live{display:none}.still{display:inline}}</style>',
           img(png(bg, ~bg, theme=theme), 0, 0, W, H)]

    # lake shimmer
    y0 = S.HORIZON + 2; lh = H - y0
    keep = (~occ_lake())[y0:H]
    fr = [S.atkinson(S.terrain(0, t))[y0:H] for t in range(6)]
    lit = stack([f & keep for f in fr]); drk = stack([~f & keep for f in fr])
    out.append('<g class="live">' + strip(png(lit, drk, theme=theme), 0, y0, W, lh, 6, 6) + '</g>')

    # flash of each shot
    shots = [T0 + STEP * i + 1.25 for i in range(n)]
    out.append(f'<g class="live"><g visibility="hidden">{img(png(fl, ~fl, theme=theme), 0, 0, W, H)}'
               + show([(s, s + 2 / FPS) for s in shots], N, DUR) + '</g></g>')

    # moon's eye
    x0, x1, yy0, yy1 = mx - mr - 1, mx + mr + 2, my - mr - 1, my + mr + 2
    look = (-0.30, 0.10)
    frames = {"a": (0.28, (0, 0)), "b": (0.65, (0, 0)), "c": (1, (0, 0)), "d": (1, look), "e": (0.55, look), "f": (1.12, look), "g": (0.22, look)}
    E = END + 1.9
    sched = {"a": [(E, E + .17)], "b": [(E + .17, E + .34)], "c": [(E + .34, E + 1.15)],
             "d": [(E + 1.15, E + 2.3), (E + 2.65, E + 3.1), (E + 3.9, E + 4.6)], "e": [(E + 2.3, E + 2.4), (E + 2.55, E + 2.65), (E + 4.6, E + 4.75)],
             "g": [(E + 2.4, E + 2.55), (E + 4.75, E + 4.95)], "f": [(E + 3.1, E + 3.9)]}
    for k, (op, gz) in frames.items():
        L, m, _ = S.eye_layer(op, gz)
        b = S.atkinson(np.where(m, L, 0))
        c = slice(yy0, yy1), slice(x0, x1)
        out.append(f'<g class="live"><g visibility="hidden">{img(png((b & vis)[c], (~b & vis)[c], theme=theme), x0, yy0, x1 - x0, yy1 - yy0)}{show(sched[k], N, DUR)}</g></g>')

    # cast
    JAR = (10, 82)
    hl = [halo(34, k) for k in (1.0, 0.82)]
    live, still = [], []
    burst_uri = png(stack(burst()), theme=theme)
    orb_l = np.zeros((7, 7), bool); orb_l[1:6, 2:5] = True; orb_l[2:5, 1:6] = True
    orb_a, orb_o = orb_l, np.zeros((7, 7), bool)
    ring = np.zeros((7, 7), bool); ring[0, 2:5] = ring[6, 2:5] = ring[2:5, 0] = ring[2:5, 6] = True
    ring[1, 1] = ring[1, 5] = ring[5, 1] = ring[5, 5] = True
    orb = png(orb_l, ring, theme=theme)
    pos = {}
    for i, c in enumerate(CAST):
        fr = [sprite(c, f) for f in range(4)]
        sh, sw = fr[0][0].shape
        uri = png(stack([f[0] for f in fr]), stack([f[1] for f in fr]), theme=theme)
        cx, cy = c["at"]
        shot = shots[i]
        lock = shot - 0.55
        back = DUR - 2.2 + i * 0.28
        track = []
        for k in range(N):
            t = k / FPS
            ph = i * 1.7
            # every wobble completes whole cycles per loop, so the loop has no seam
            w1, w2 = 2 * math.pi * round(DUR / 5.5) / DUR, 2 * math.pi * round(DUR / 3.2) / DUR
            x = cx + 7 * math.sin(w1 * t + ph); y = cy + 4.5 * math.sin(w2 * t + ph * 1.3)
            if lock <= t < shot:                     # it noticed
                x += (-1) ** k; y += 0
            if t >= back - 0.01:
                y += 26 * (1 - ease((t - back) / 0.9))
            track.append((round(x), round(y)))
        pos[i] = track
        tr = keyed("transform", [f"{x - sw // 2} {y - sh // 2}" for x, y in track], DUR, "animateTransform", 'type="translate"')
        vis_on = [(0, shot)] + [(back, DUR)]
        g = (f'<g>{show(vis_on, N, DUR)}<g transform="translate({cx - sw // 2} {cy - sh // 2})">{tr}'
             + img(png(hl[0], theme=theme), sw // 2 - 34, sh // 2 - 34, 68, 68)
             + strip(uri, 0, 0, sw, sh, 4, 7, -i * 0.13) + '</g></g>')
        live.append(g)
        still.append(img(png(hl[0], theme=theme), cx - 34, cy - 34, 68, 68)
                     + img(png(fr[0][0], fr[0][1], theme=theme), cx - sw // 2, cy - sh // 2, sw, sh))
        # death: burst, name, orb to the jar
        sx, sy = track[round(shot * FPS)]
        vals = ";".join(str(-52 * f) for f in range(8))
        kt = ";".join(f"{min(1, (shot + f * 0.085) / DUR):.5f}" for f in range(8))
        live.append(f'<g visibility="hidden">{show([(shot, shot + .68)], N, DUR)}<svg x="{sx - 26}" y="{sy - 26}" width="52" height="52" viewBox="0 0 52 52" overflow="hidden">'
                    f'<image width="52" height="416" href="{burst_uri}"><animate attributeName="y" values="0;{vals}" keyTimes="0;{kt}" calcMode="discrete" dur="{DUR}s" repeatCount="indefinite"/></image></svg></g>')
        a = np.pad(text(c["name"], "name"), ((4, 4), (6, 6)))
        o = ~a; o[0, 0] = o[0, -1] = o[-1, 0] = o[-1, -1] = False
        tw, th = a.shape[1], a.shape[0]
        tx = int(min(W - tw - 8, max(8, sx - tw // 2)))
        ny = [f"{tx} {sy - th // 2 - int(8 * ease((k / FPS - shot - .25) / .6))}" for k in range(N)]
        live.append(f'<g visibility="hidden">{show([(shot + .25, shot + 2.15)], N, DUR)}<g>{keyed("transform", ny, DUR, "animateTransform", "type=" + chr(34) + "translate" + chr(34))}'
                    f'{img(png(a, o, theme=theme), 0, 0, tw, th)}</g></g>')
        op = []
        a0, a1 = shot + .35, shot + 1.35
        for k in range(N):
            t = min(1, max(0, (k / FPS - a0) / (a1 - a0))); e = t * t * (3 - 2 * t)
            ox = sx + (JAR[0] + 13 - sx) * e; oy = sy + (JAR[1] + 14 - sy) * e - 46 * math.sin(math.pi * e)
            op.append(f"{round(ox) - 3} {round(oy) - 3}")
        live.append(f'<g visibility="hidden">{show([(a0, a1)], N, DUR)}<g>{keyed("transform", op, DUR, "animateTransform", "type=" + chr(34) + "translate" + chr(34))}{img(orb, 0, 0, 7, 7)}</g></g>')
        # its light in the jar
        jx, jy = JAR[0] + 7 + (i % 3) * 5 - (i // 3), JAR[1] + 22 - (i // 3) * 6
        live.append(f'<g visibility="hidden">{show([(a1, END + 5.3)], N, DUR)}<rect x="{jx}" y="{jy}" width="3" height="3" fill="rgb{THEMES[theme][0]}">'
                    f'<animate attributeName="y" values="{jy};{jy - 1};{jy}" calcMode="discrete" dur="{1.1 + .17 * i:.2f}s" repeatCount="indefinite"/></rect></g>')

    # laser: one dot, one beam
    aim, on = [], []
    cur = (196, 150)
    for k in range(N):
        t = k / FPS
        tgt, vis_l, snap = cur, False, 0.30
        for i, s in enumerate(shots):
            if s - 1.25 <= t < s:
                tgt = pos[i][k]; vis_l = True
                q = (t - (s - 1.25)) / 0.7
                snap = 0.22 if q < 1 else 0.8
        if END + .7 <= t < E + 2.0:
            tgt = (mx, my - 4) if t < E + 1.15 else (mx - 16, my + 1); vis_l = True
        cur = (cur[0] + (tgt[0] - cur[0]) * snap, cur[1] + (tgt[1] - cur[1]) * snap)
        j = 0 if not vis_l else math.sin(k * 1.9) * 0.6
        if E + 1.15 <= t < E + 2.0: j = (-1) ** k * 2.5       # shaking
        aim.append((round(cur[0] + j), round(cur[1] + j * .5)))
        on.append("visible" if vis_l else "hidden")
    ax = keyed("x2", [str(a) for a, _ in aim], DUR); ay = keyed("y2", [str(b) for _, b in aim], DUR)
    dot = keyed("transform", [f"{a} {b}" for a, b in aim], DUR, "animateTransform", 'type="translate"')
    v = keyed("visibility", on, DUR)
    live.append(f'<g visibility="hidden" shape-rendering="crispEdges">{v}<line x1="{S.MUZZLE[0]}" y1="{S.MUZZLE[1]}" x2="250" y2="90" stroke="{RED}" stroke-width=".5" opacity=".85">{ax}{ay}</line>'
                f'<g>{dot}<path d="M-1-1h3v3h-3z" fill="{RED}"/><path d="M0-5h1v2h-1zM0 4h1v2h-1zM-5 0h2v1h-2zM4 0h2v1h-2z" fill="{RED}"/></g></g>')
    out.append('<g class="live">' + "".join(live) + '</g><g class="still">' + "".join(still) + '</g>')

    # hunter + jar, muzzle flash
    body, rim = S.hunter()
    ys, xs = np.nonzero(body); a, b, c_, d = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    glow = np.zeros_like(body)                                        # moonlight catching the near edges
    for dx, dy in ((1, 0), (1, -1), (0, -1)): glow |= np.roll(np.roll(body, dx, 1), dy, 0)
    glow &= ~body & (S.yy < S.CLIFF[None, :]) & (S.xx > 98)
    a -= 1; d += 1; rim &= S.xx <= 98
    live_rim = png(rim[a:b, c_:d] | glow[a:b, c_:d], (body & ~rim)[a:b, c_:d], theme=theme)
    out.append(img(live_rim, c_, a, d - c_, b - a))
    sc = [S.scarf(f) & ~body for f in range(8)]
    out.append(strip(png(red=stack([s[80:125, 40:115] for s in sc]), theme=theme), 40, 80, 75, 45, 8, 9))
    jl, jd = jar_small()
    out.append(img(png(jl, jd, theme=theme), JAR[0], JAR[1], jl.shape[1], jl.shape[0]))
    star = np.zeros((19, 25), bool)
    for (dx, dy, l) in [(1, 0, 22), (1, .5, 11), (1, -.5, 11), (.35, 1, 8), (.35, -1, 8)]:
        for q in range(l):
            px, py = int(1 + dx * q), int(9 + dy * q)
            if 0 <= px < 25 and 0 <= py < 19:
                star[py, px] = True
                if q < l * .5 and py + 1 < 19: star[py + 1, px] = True
    out.append(f'<g class="live"><g visibility="hidden">{img(png(drk=star, theme=theme), S.MUZZLE[0], S.MUZZLE[1] - 9, 25, 19)}{show([(s, s + 1 / FPS) for s in shots], N, DUR)}</g>')
    # tracer rounds
    for i, s in enumerate(shots):
        tx, ty = pos[i][round(s * FPS)]
        out.append(f'<line visibility="hidden" x1="{S.MUZZLE[0]}" y1="{S.MUZZLE[1]}" x2="{tx}" y2="{ty}" stroke="rgb{THEMES[theme][1]}" stroke-width="1.2" shape-rendering="crispEdges">{show([(s, s + 2 / FPS)], N, DUR)}</line>')
    # the hunter's reaction
    ex = np.array([[c == "#" for c in r] for r in ["###", "###", "###", "###", ".#.", "...", "###", "###"]])
    out.append(f'<g visibility="hidden">{img(png(red=ex, theme=theme), 111, 78, 3, 8)}{show([(E + 1.35, E + 2.2)], N, DUR)}</g></g>')

    # reeds
    gy = 168; gh = H - gy
    gr = [S.grass(f) for f in range(6)]
    gl, gd = stack([g[1][gy:] for g in gr]), stack([(g[0] & ~g[1])[gy:] for g in gr])
    out.append(strip(png(gl, gd, theme=theme), 0, gy, W, gh, 6, 5))

    # wordmark
    t = text("SoulSniper", "title")
    ta, to = outlined(t, 2)
    out.append(img(png(ta, to, theme=theme), 14, 14, ta.shape[1], ta.shape[0]))
    print(theme, "title", ta.shape, "dur", DUR)

    page = {"dark": "dark", "light": "light"}[theme]
    L, D = THEMES[theme]
    e_uri = png(edge if theme == "light" else None, edge if theme == "dark" else None, theme=theme)
    out.append(img(e_uri, 0, 0, W, H))
    out.append("</svg>")
    return "".join(out)


def occ_lake():
    """Pixels in the lake band that belong to land."""
    r = np.zeros((H, W), bool)
    r |= S.yy > S.CLIFF[None, :] - 1
    r |= (S.xx > 355) & (S.yy < S.HORIZON + 9)
    return r


def jar_small():
    rows = ["..#########..", "..#ooooooo#..", "..#########..", "...#ooooo#...", "..#ooooooo#..", ".#ooooooooo#.", ".#o#ooooooo#.", ".#o#ooooooo#.", ".#o#ooooooo#.",
            ".#ooooooooo#.", ".#ooooooooo#.", ".#ooooooooo#.", ".#ooooooooo#.", ".#ooooooooo#.", "..#########.."]
    g = np.array([list(r) for r in rows])
    g = epx(g)
    body = g != "."
    # inside stays clear so the captured lights, drawn beneath, show through
    inner = np.zeros_like(body); inner[12:28, 6:20] = True
    return (g == "#"), (g == "o") & ~inner


# ---------- shelf ----------
TW, TH = 210, 92


def jar_big():
    """Glass outline and highlights, 46x66."""
    ss = 4; w, h = 46, 66
    def outline(draw):
        im = Image.new("L", (w * ss, h * ss)); draw(ImageDraw.Draw(im)); return np.asarray(im.resize((w, h), Image.BOX)) > 127
    full = outline(lambda d: d.rounded_rectangle([1 * ss, 12 * ss, 45 * ss - 1, 65 * ss], 9 * ss, fill=255))
    neck = outline(lambda d: d.rectangle([8 * ss, 8 * ss, 38 * ss - 1, 14 * ss], fill=255))
    lid = outline(lambda d: d.rounded_rectangle([6 * ss, 1 * ss, 40 * ss - 1, 9 * ss - 1], 1 * ss, fill=255))
    g = full | neck
    inn = g.copy()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)): inn &= np.roll(np.roll(g, dx, 1), dy, 0)
    lit = g & ~inn
    yy, xx = np.mgrid[0:h, 0:w]
    lit |= inn & (xx == 5) & (yy > 22) & (yy < 46); lit |= inn & (xx == 5) & (yy > 49) & (yy < 53)
    lit |= inn & (xx == 7) & (yy > 22) & (yy < 34)
    lit |= inn & (xx > 36) & S.bayer(np.where(xx > 36, 0.10 + (xx - 36) * 0.035, 0))
    lit |= inn & (yy > 60) & S.bayer(np.full((h, w), .3))
    cork = S.bayer(np.where(lid, 0.30 + 0.4 * (xx / w) + 0.25 * (S.fbm(xx / 3., yy / 2., 9, 3) - .5), 0)) & lid
    li = lid.copy()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)): li &= np.roll(np.roll(lid, dx, 1), dy, 0)
    cork |= lid & ~li
    return lit | cork, (lid | g) & ~(lit | cork)


def shelf_plank(seed):
    h = 13
    yy, xx = np.mgrid[0:h, 0:TW].astype(float)
    grain = S.fbm((xx % TW) / 34, yy / 1.6, seed, 4)
    L = 0.20 + 0.75 * S.smooth(0.50, 0.62, grain)
    L = np.where(yy < 1, 1.2, np.where(yy > 8, 0.0, L * (1 - yy / 16)))
    return S.atkinson(L)


def tile(c, i, theme):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {TW} {TH}" width="{TW * 2}" height="{TH * 2}" role="img" aria-label="{c["name"]}: {c["say"]}">',
           '<style>image{image-rendering:pixelated}@media(max-width:230px){image{image-rendering:auto}}'
           '@media(prefers-reduced-motion:reduce){.k{display:none}}.s{display:none}@media(prefers-reduced-motion:reduce){.s{display:inline}}</style>']
    flip = theme == "light"                 # on white paper the ink draws the glass
    def P(lit=None, drk=None, red=None):
        return png(drk, lit, red, theme) if flip else png(lit, drk, red, theme)
    jx, jy = 10, TH - 13 - 66
    glass_l, glass_d = jar_big()
    inside = glass_d.copy()
    # soul
    fr = [sprite(c, f) for f in range(4)]
    sh, sw = fr[0][0].shape
    u = png(stack([f[0] for f in fr]), stack([f[1] for f in fr]), theme=theme)
    cx, cy = jx + 23, jy + 38
    hr = 21
    hh = [halo(hr, k) for k in (1.25, 0.95)]
    out.append(f'<svg x="{cx - hr}" y="{cy - hr}" width="{2 * hr}" height="{2 * hr}" viewBox="0 0 {2 * hr} {2 * hr}" overflow="hidden"><image width="{2 * hr}" height="{4 * hr}" href="{P(stack(hh))}">'
               f'<animate attributeName="y" values="0;{-2 * hr}" calcMode="discrete" dur="{1.3 + i * .11:.2f}s" repeatCount="indefinite"/></image></svg>')
    bob = [f"{cx - sw // 2 + round(2 * math.sin(2 * math.pi * k / 60 + i))} {cy - sh // 2 + round(2 * math.sin(4 * math.pi * k / 60 + i * 2))}" for k in range(60)]
    out.append(f'<g class="k"><g transform="translate({cx - sw // 2} {cy - sh // 2})">{keyed("transform", bob, 5, "animateTransform", "type=" + chr(34) + "translate" + chr(34))}{strip(u, 0, 0, sw, sh, 4, 7, -i * .2)}</g></g>')
    out.append(f'<g class="s">{img(png(fr[0][0], fr[0][1], theme=theme), cx - sw // 2, cy - sh // 2, sw, sh)}</g>')
    out.append(img(P(glass_l), jx, jy, 46, 66))
    pl = shelf_plank(55 + 9 * i)
    out.append(img(P(pl), 0, TH - 13, TW, 13))
    nm = text(c["name"], "name")
    out.append(img(P(nm), 68, 22, nm.shape[1], nm.shape[0]))
    y = 22 + 23
    for ln in wrap(c["say"], 22):
        t = text(ln, "body")
        out.append(img(P(t), 68, y + top_of(ln), t.shape[1], t.shape[0]))
        if t.shape[1] + 68 > TW: print("OVERFLOW", c["key"], ln, t.shape)
        y += 12
    if nm.shape[1] + 68 > TW: print("OVERFLOW", c["key"], nm.shape)
    out.append("</svg>")
    return "".join(out)


def top_of(ln):
    tall = any(ch in "bdfhkltijABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for ch in ln)
    return 0 if tall else 2


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for th in THEMES:
        open(f"{OUT}/hero-{th}.svg", "w").write(hero(th))
        for i, c in enumerate(CAST):
            open(f"{OUT}/{c['key']}-{th}.svg", "w").write(tile(c, i, th))
