"""
inspect_network.py — Visualise the topology and weights of a saved veteran network.

Usage:
    python inspect_network.py               # opens file picker
    python inspect_network.py <path>.pkl    # load directly

Controls (viewer):
  Scroll wheel    -> zoom in / out
  Left-drag       -> pan
  Click node      -> show node details in sidebar
  Click edge      -> show weight in sidebar
  T / Shift+T     -> raise / lower weight-display threshold
  L               -> toggle weight labels on edges
  R               -> reset camera
  ESC / Q         -> back to file picker
"""

import sys
import os
import pickle
import math
from collections import defaultdict

import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "evolution"))

from network import Network, Node          # noqa: E402
from profiles import PROFILES              # noqa: E402

# ── Window ────────────────────────────────────────────────────────────────────

WIN_W    = 1400
WIN_H    = 840
SIDE_W   = 300
CANVAS_W = WIN_W - SIDE_W

# ── Colours ───────────────────────────────────────────────────────────────────

BG       = (18,  18,  26)
BG_SIDE  = (26,  26,  38)
RULE     = (55,  55,  75)
FG       = (230, 230, 242)
DIM      = (145, 145, 165)
ACCENT   = (70,  200, 255)

C_IN     = (80,  145, 255)     # input node  — blue
C_HID    = (255, 170,  45)     # hidden node — amber
C_OUT    = (55,  225,  95)     # output node — green
C_SEL    = (255, 240,  55)     # selection highlight

C_POS    = (40,  230,  90)     # strong positive weight
C_NEG    = (240,  60,  60)     # strong negative weight
C_WEAK   = (110, 110, 135)     # near-zero weight (neutral)
C_SEL_E  = (255, 215,  40)     # selected edge

# ── Labels ────────────────────────────────────────────────────────────────────

_RAW_LABELS = (
    [f"G({y},{x})" for y in range(10) for x in range(10)]
    + ["dir:R", "dir:U", "dir:L", "dir:D"]
    + ["food_t", "bomb_t", "explode", "len"]
)
_OUT_LABELS = ["turn left", "turn right"]


def input_labels(profile_name: str) -> list:
    if profile_name == "raw":
        return _RAW_LABELS
    profile = PROFILES.get(profile_name, {})
    return [k for k, w in profile.items() if w > 0]


def act_name(fn) -> str:
    return getattr(fn, "__name__", "?") if fn is not None else "none"


def safe_float(v) -> str:
    try:
        return f"{float(v):+.6f}"
    except Exception:
        return str(v)


# ── Layout ────────────────────────────────────────────────────────────────────

def compute_positions(network: Network) -> dict:
    """
    World-space (x, y) per id(node).
    Column = topological depth from inputs (BFS longest-path).
    Row    = evenly distributed within each column.
    """
    by_id = {id(n): n for n in network.nodes}

    # Longest-path BFS
    depth = {id(n): 0 for n in network.input_nodes}
    changed = True
    while changed:
        changed = False
        for c in network.connections:
            fid = id(c.from_node)
            tid = id(c.to_node)
            if fid in depth:
                tnode = by_id.get(tid)
                if tnode and tnode.type != "input":
                    nd = depth[fid] + 1
                    if depth.get(tid, -1) < nd:
                        depth[tid] = nd
                        changed = True

    # Output nodes always at the last column
    hid_depths = [d for nid, d in depth.items()
                  if by_id.get(nid) and by_id[nid].type == "hidden"]
    max_hid = max(hid_depths, default=0)
    for n in network.output_nodes:
        depth[id(n)] = max_hid + 1

    # Nodes not reachable from any input
    for n in network.nodes:
        if id(n) not in depth:
            depth[id(n)] = max_hid + 1

    max_d = max(depth.values(), default=1)

    columns: dict[int, list] = defaultdict(list)
    for n in network.nodes:
        columns[depth[id(n)]].append(n)

    max_col = max(len(v) for v in columns.values())
    COL_W   = 220
    ROW_H   = max(9, min(44, 880 / max(max_col, 1)))

    pos = {}
    for d, nodes in columns.items():
        wx    = (d - max_d / 2) * COL_W
        count = len(nodes)
        for i, node in enumerate(nodes):
            wy         = (i - (count - 1) / 2) * ROW_H
            pos[id(node)] = (wx, wy)

    return pos


# ── Camera ────────────────────────────────────────────────────────────────────

def w2s(wx, wy, cam_x, cam_y, zoom):
    return (
        int((wx - cam_x) * zoom + CANVAS_W / 2),
        int((wy - cam_y) * zoom + WIN_H   / 2),
    )


def s2w(sx, sy, cam_x, cam_y, zoom):
    return (
        (sx - CANVAS_W / 2) / zoom + cam_x,
        (sy - WIN_H   / 2) / zoom + cam_y,
    )


# ── Hit testing ───────────────────────────────────────────────────────────────

def _base_r(node: Node) -> float:
    return 5.0 if node.type == "input" else 10.0


def hit_node(network, pos, wx, wy, zoom):
    for n in network.nodes:
        if id(n) not in pos:
            continue
        nx, ny = pos[id(n)]
        r = _base_r(n) * 2.2 / max(zoom, 0.05)
        if (wx - nx) ** 2 + (wy - ny) ** 2 <= r ** 2:
            return n
    return None


def hit_conn(network, pos, wx, wy):
    best, best_d = None, 8.0
    for c in network.connections:
        if id(c.from_node) not in pos or id(c.to_node) not in pos:
            continue
        fx, fy = pos[id(c.from_node)]
        tx, ty = pos[id(c.to_node)]
        dx, dy = tx - fx, ty - fy
        L2 = dx * dx + dy * dy
        if L2 == 0:
            continue
        t  = max(0.0, min(1.0, ((wx - fx) * dx + (wy - fy) * dy) / L2))
        px, py = fx + t * dx, fy + t * dy
        d  = math.hypot(wx - px, wy - py)
        if d < best_d:
            best_d, best = d, c
    return best


# ── Drawing ───────────────────────────────────────────────────────────────────

def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _conn_color(weight: float, max_w: float) -> tuple:
    """Interpolate: near-zero -> C_WEAK, strong positive -> C_POS, strong negative -> C_NEG."""
    t = min(abs(weight) / max(max_w, 1e-6), 1.0)
    target = C_POS if weight >= 0 else C_NEG
    return _lerp(C_WEAK, target, t)


def draw_network(surface, font_sm, network, pos, cam_x, cam_y, zoom,
                 sel_node, sel_conn, threshold, show_labels, in_lbls=None):

    max_w = max((abs(c.weight) for c in network.connections), default=1.0)

    # ── Connections (single SRCALPHA surface, one blit) ───────────────────────
    conn_surf = pygame.Surface((CANVAS_W, WIN_H), pygame.SRCALPHA)

    for c in network.connections:
        # Guard: skip if either endpoint has no computed position
        if id(c.from_node) not in pos or id(c.to_node) not in pos:
            continue
        w = c.weight if c.weight is not None else 0.0
        if abs(w) < threshold:
            continue

        fx, fy = pos[id(c.from_node)]
        tx, ty = pos[id(c.to_node)]
        sx1, sy1 = w2s(fx, fy, cam_x, cam_y, zoom)
        sx2, sy2 = w2s(tx, ty, cam_x, cam_y, zoom)

        is_sel = c is sel_conn
        color  = C_SEL_E if is_sel else _conn_color(w, max_w)
        # Alpha: minimum 80 so even weak connections are visible; max 255
        alpha  = 255 if is_sel else int(80 + 175 * abs(w) / max(max_w, 1e-6))
        width  = max(1, int(abs(w) / max(max_w, 1e-6) * 4 * max(zoom, 0.25)))
        if is_sel:
            width = max(width + 1, 2)

        pygame.draw.line(conn_surf, (*color, alpha), (sx1, sy1), (sx2, sy2), width)

        if show_labels and zoom > 0.6:
            mx, my = (sx1 + sx2) // 2, (sy1 + sy2) // 2
            lbl = font_sm.render(f"{w:+.2f}", True, (*DIM, 200))
            conn_surf.blit(lbl, (mx - lbl.get_width() // 2, my - 6))

    surface.blit(conn_surf, (0, 0))

    # ── Nodes ─────────────────────────────────────────────────────────────────
    node_idx      = {id(n): i  for i, n in enumerate(network.nodes)}
    input_node_i  = {id(n): i  for i, n in enumerate(network.input_nodes)}
    labels        = in_lbls or []

    for n in network.nodes:
        if id(n) not in pos:
            continue
        nx, ny = pos[id(n)]
        sx, sy = w2s(nx, ny, cam_x, cam_y, zoom)
        if sx < -20 or sx > CANVAS_W + 20 or sy < -20 or sy > WIN_H + 20:
            continue

        r     = max(3, int(_base_r(n) * zoom))
        color = C_IN if n.type == "input" else C_OUT if n.type == "output" else C_HID
        idx   = node_idx.get(id(n), "?")

        if n is sel_node:
            pygame.draw.circle(surface, C_SEL, (sx, sy), r + 4)

        pygame.draw.circle(surface, color, (sx, sy), r)
        pygame.draw.circle(surface, FG,    (sx, sy), r, 1)

        # Activation function abbreviation inside hidden / output nodes
        if zoom > 0.5 and n.type != "input":
            act_lbl = font_sm.render(act_name(n.squash)[:4], True, BG)
            surface.blit(act_lbl, (sx - act_lbl.get_width() // 2,
                                   sy - act_lbl.get_height() // 2))

        # Index number below the node
        show_idx = zoom > 0.22 if n.type != "input" else zoom > 0.45
        if show_idx:
            idx_lbl = font_sm.render(str(idx), True, color)
            surface.blit(idx_lbl, (sx - idx_lbl.get_width() // 2, sy + r + 2))

        # Sensor name to the LEFT of input nodes
        if n.type == "input" and zoom > 0.28 and labels:
            i_in = input_node_i.get(id(n), -1)
            if 0 <= i_in < len(labels):
                lbl_txt = labels[i_in]
                lbl_img = font_sm.render(lbl_txt, True, C_IN)
                surface.blit(lbl_img, (sx - r - lbl_img.get_width() - 6,
                                       sy - lbl_img.get_height() // 2))


def draw_sidebar(surface, fonts, network, sel_node, sel_conn,
                 profile_name, in_lbls, threshold, show_labels):

    font_big, font, font_sm = fonts
    x0 = CANVAS_W

    pygame.draw.rect(surface, BG_SIDE, (x0, 0, SIDE_W, WIN_H))
    pygame.draw.line(surface, RULE, (x0, 0), (x0, WIN_H), 1)

    y = 10

    def text(s, color=FG, f=None):
        nonlocal y
        img = (f or font_sm).render(str(s), True, color)
        surface.blit(img, (x0 + 10, y))
        y += img.get_height() + 4

    def hrule():
        nonlocal y
        y += 5
        pygame.draw.line(surface, RULE, (x0 + 10, y), (x0 + SIDE_W - 10, y))
        y += 8

    n_hid = sum(1 for n in network.nodes if n.type == "hidden")

    text("Network Inspector", ACCENT, font_big)
    text(f"profile: {profile_name}", DIM)
    text(f"nodes:  {len(network.nodes)}  "
         f"({len(network.input_nodes)} in / {n_hid} hid / {len(network.output_nodes)} out)", DIM)
    text(f"conns:  {len(network.connections)}", DIM)
    text(f"longevity: {getattr(network, 'longevity', '?')} gens", DIM)

    hrule()

    try:
        # ── Selected connection ───────────────────────────────────────────────
        if sel_conn is not None:
            fn, tn = sel_conn.from_node, sel_conn.to_node
            try:
                fi = network.nodes.index(fn)
            except ValueError:
                fi = "?"
            try:
                ti = network.nodes.index(tn)
            except ValueError:
                ti = "?"

            text("-- Connection --", ACCENT)
            text(f"from: {fn.type}[{fi}]")
            text(f"to:   {tn.type}[{ti}]")
            w      = sel_conn.weight if sel_conn.weight is not None else 0.0
            wcolor = C_POS if w >= 0 else C_NEG
            text(f"weight: {w:+.6f}", wcolor)

            # Weight bar  [-2 … 0 … +2]
            bw = SIDE_W - 30
            bx = x0 + 15
            pygame.draw.rect(surface, RULE, (bx, y, bw, 10))
            mid  = bx + bw // 2
            frac = min(abs(w) / 2.0, 1.0) * (bw // 2)
            if w >= 0:
                pygame.draw.rect(surface, C_POS, (mid, y, int(frac), 10))
            else:
                left = mid - int(frac)
                pygame.draw.rect(surface, C_NEG, (left, y, mid - left, 10))
            y += 18
            hrule()

        # ── Selected node ─────────────────────────────────────────────────────
        if sel_node is not None:
            try:
                idx = network.nodes.index(sel_node)
            except ValueError:
                idx = "?"

            text(f"-- {sel_node.type.upper()} Node [{idx}] --", ACCENT)
            text(f"bias:   {safe_float(sel_node.bias)}")
            text(f"squash: {act_name(sel_node.squash)}")

            if sel_node.type == "input" and in_lbls:
                try:
                    ii  = network.input_nodes.index(sel_node)
                    lbl = in_lbls[ii] if ii < len(in_lbls) else "?"
                except (ValueError, IndexError):
                    lbl = "?"
                text(f"sensor: {lbl}", ACCENT)

            if sel_node.type == "output":
                try:
                    oi  = network.output_nodes.index(sel_node)
                    lbl = _OUT_LABELS[oi] if oi < len(_OUT_LABELS) else "?"
                except (ValueError, IndexError):
                    lbl = "?"
                text(f"output: {lbl}", ACCENT)

            y += 4
            conns_in  = getattr(sel_node, "connections_in",  [])
            conns_out = getattr(sel_node, "connections_out", [])

            if conns_in:
                text(f"in ({len(conns_in)}):", DIM)
                for c in conns_in[:10]:
                    try:
                        fi = network.nodes.index(c.from_node)
                    except ValueError:
                        fi = "?"
                    w  = c.weight if c.weight is not None else 0.0
                    text(f"  [{fi}]  {w:+.4f}", C_POS if w >= 0 else C_NEG)
                if len(conns_in) > 10:
                    text(f"  ... +{len(conns_in) - 10} more", DIM)

            if conns_out:
                text(f"out ({len(conns_out)}):", DIM)
                for c in conns_out[:10]:
                    try:
                        ti = network.nodes.index(c.to_node)
                    except ValueError:
                        ti = "?"
                    w  = c.weight if c.weight is not None else 0.0
                    text(f"  [{ti}]  {w:+.4f}", C_POS if w >= 0 else C_NEG)
                if len(conns_out) > 10:
                    text(f"  ... +{len(conns_out) - 10} more", DIM)

    except Exception as e:
        text(f"[display error: {e}]", C_NEG)

    # ── Controls ──────────────────────────────────────────────────────────────
    hints = [
        "Scroll: zoom",
        "Drag: pan",
        "Click: select node / edge",
        f"T / Shift+T: threshold ({threshold:.2f})",
        f"L: labels ({'on' if show_labels else 'off'})",
        "R: reset view",
        "ESC / Q: file picker",
    ]
    hy = WIN_H - len(hints) * 17 - 12
    pygame.draw.line(surface, RULE, (x0 + 10, hy - 6), (x0 + SIDE_W - 10, hy - 6))
    for h in hints:
        img = font_sm.render(h, True, DIM)
        surface.blit(img, (x0 + 10, hy))
        hy += 17


# ── Viewer main loop ──────────────────────────────────────────────────────────

def run_viewer(window, network: Network, profile_name: str, path: str):
    pygame.display.set_caption(f"Network Inspector -- {os.path.basename(path)}")

    font_big = pygame.font.SysFont(None, 26)
    font     = pygame.font.SysFont(None, 20)
    font_sm  = pygame.font.SysFont(None, 15)
    fonts    = (font_big, font, font_sm)

    in_lbls = input_labels(profile_name)
    pos     = compute_positions(network)

    cam_x, cam_y = 0.0, 0.0
    zoom         = 1.0
    sel_node     = None
    sel_conn     = None
    threshold    = 0.0
    show_labels  = False
    dragging     = False
    drag_start   = (0, 0)
    cam_start    = (0.0, 0.0)
    clock        = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return
                elif event.key == pygame.K_r:
                    cam_x, cam_y, zoom = 0.0, 0.0, 1.0
                elif event.key == pygame.K_l:
                    show_labels = not show_labels
                elif event.key == pygame.K_t:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        threshold = max(0.0, round(threshold - 0.05, 3))
                    else:
                        threshold = min(2.0, round(threshold + 0.05, 3))

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if mx >= CANVAS_W:
                    continue
                if event.button == 1:
                    dragging   = True
                    drag_start = (mx, my)
                    cam_start  = (cam_x, cam_y)
                elif event.button in (4, 5):
                    factor = 1.15 if event.button == 4 else 1 / 1.15
                    wx, wy = s2w(mx, my, cam_x, cam_y, zoom)
                    zoom   = max(0.05, min(12.0, zoom * factor))
                    sx2, sy2 = w2s(wx, wy, cam_x, cam_y, zoom)
                    cam_x += (sx2 - mx) / zoom
                    cam_y += (sy2 - my) / zoom

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and dragging:
                    mx, my   = event.pos
                    dx       = mx - drag_start[0]
                    dy       = my - drag_start[1]
                    dragging = False
                    if abs(dx) < 5 and abs(dy) < 5 and mx < CANVAS_W:
                        wx, wy = s2w(mx, my, cam_x, cam_y, zoom)
                        node   = hit_node(network, pos, wx, wy, zoom)
                        if node:
                            sel_node = node
                            sel_conn = None
                        else:
                            c = hit_conn(network, pos, wx, wy)
                            sel_conn = c
                            if c is None:
                                sel_node = None

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mx, my = event.pos
                    cam_x  = cam_start[0] - (mx - drag_start[0]) / zoom
                    cam_y  = cam_start[1] - (my - drag_start[1]) / zoom

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if mx < CANVAS_W:
                    factor = 1.15 ** event.y
                    wx, wy = s2w(mx, my, cam_x, cam_y, zoom)
                    zoom   = max(0.05, min(12.0, zoom * factor))
                    sx2, sy2 = w2s(wx, wy, cam_x, cam_y, zoom)
                    cam_x += (sx2 - mx) / zoom
                    cam_y += (sy2 - my) / zoom

        window.fill(BG)
        draw_network(window, font_sm, network, pos, cam_x, cam_y, zoom,
                     sel_node, sel_conn, threshold, show_labels, in_lbls)
        draw_sidebar(window, fonts, network, sel_node, sel_conn,
                     profile_name, in_lbls, threshold, show_labels)
        pygame.display.flip()
        clock.tick(60)


# ── File picker ───────────────────────────────────────────────────────────────

def find_veterans() -> list:
    found = []
    for root, _, files in os.walk(_HERE):
        for f in files:
            if f.startswith("veteran") and f.endswith(".pkl"):
                found.append(os.path.join(root, f))
    return sorted(found, key=os.path.getmtime, reverse=True)


def run_selector(window) -> str | None:
    pygame.display.set_caption("Network Inspector -- Select veteran")
    font_big = pygame.font.SysFont(None, 28)
    font_sm  = pygame.font.SysFont(None, 16)

    veterans = find_veterans()
    scroll   = 0
    LINE_H   = 26
    LIST_TOP = 72
    clock    = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_UP:
                    scroll = max(0, scroll - LINE_H)
                elif event.key == pygame.K_DOWN:
                    max_s  = max(0, len(veterans) * LINE_H - (WIN_H - LIST_TOP))
                    scroll = min(max_s, scroll + LINE_H)
            elif event.type == pygame.MOUSEWHEEL:
                max_s  = max(0, len(veterans) * LINE_H - (WIN_H - LIST_TOP))
                scroll = max(0, min(max_s, scroll - event.y * LINE_H * 2))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                idx    = (my - LIST_TOP + scroll) // LINE_H
                if 0 <= idx < len(veterans):
                    return veterans[idx]

        window.fill(BG)
        title = font_big.render("Network Inspector -- select a veteran", True, ACCENT)
        window.blit(title, (20, 16))
        hint = font_sm.render("Click to load   |   ESC / Q to quit", True, DIM)
        window.blit(hint, (20, 46))
        pygame.draw.line(window, RULE, (0, 66), (WIN_W, 66), 1)

        if not veterans:
            msg = font_sm.render("No veteran_*.pkl files found.", True, DIM)
            window.blit(msg, (20, LIST_TOP + 20))
        else:
            for i, path in enumerate(veterans):
                y = LIST_TOP + i * LINE_H - scroll
                if not (LIST_TOP - LINE_H < y < WIN_H):
                    continue
                rel   = os.path.relpath(path, _HERE)
                color = FG if y >= LIST_TOP - 4 else DIM
                lbl   = font_sm.render(rel, True, color)
                window.blit(lbl, (20, y + 5))

        pygame.display.flip()
        clock.tick(60)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_veteran(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data["network"], data.get("profile_name", "full")
    return data, "full"


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    pygame.init()
    window = pygame.display.set_mode((WIN_W, WIN_H))

    if len(sys.argv) > 1:
        path = sys.argv[1]
        network, profile_name = load_veteran(path)
        run_viewer(window, network, profile_name, path)
    else:
        while True:
            path = run_selector(window)
            if path is None:
                break
            network, profile_name = load_veteran(path)
            run_viewer(window, network, profile_name, path)

    pygame.quit()


if __name__ == "__main__":
    main()
