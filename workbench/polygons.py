"""Binary mask -> exact pixel-boundary polygons (outer contours + holes).

The delivered mask is piecewise constant on 20 nm pixels, so its material
boundary is a union of axis-aligned closed loops on the pixel-corner lattice.
This module traces those loops EXACTLY (no marching-squares interpolation):

  * every boundary edge separates a material pixel (left of travel direction)
    from a non-material pixel, so loops around material come out CCW and loops
    around holes come out CW (signed area decides);
  * collinear runs are merged, coordinates are centered (x in [-Lx/2, Lx/2]);
  * components are cut at the unit-cell boundary (the solver's periodic BCs
    stitch them; geometry inside the cell is identical).

Verification: rasterizing the polygons back onto the pixel grid must reproduce
the input mask bit-for-bit (checked in export()).
"""
import numpy as np


def _trace_loops(mask):
    """All boundary loops of a binary mask, as pixel-corner vertex lists."""
    ny, nx = mask.shape
    pad = np.zeros((ny + 2, nx + 2), dtype=bool)
    pad[1:-1, 1:-1] = mask.astype(bool)

    # directed boundary edges: material strictly on the LEFT of travel.
    # corner lattice: (iy, ix), iy in 0..ny, ix in 0..nx (unpadded frame)
    edges = {}                                    # start corner -> list of end corners
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            below = pad[iy, ix + 1]               # pixel (iy-1, ix) in unpadded
            above = pad[iy + 1, ix + 1]           # pixel (iy,   ix)
            # horizontal edge from (iy, ix) to (iy, ix+1) exists if the two
            # pixels it separates differ; direction chosen so material is left
            if ix < nx:
                lo = pad[iy, ix + 1]              # pixel below edge row iy
                hi = pad[iy + 1, ix + 1]          # pixel above edge row iy
                if lo != hi:
                    if hi:                        # material above -> travel +x
                        edges.setdefault((iy, ix), []).append((iy, ix + 1))
                    else:                         # material below -> travel -x
                        edges.setdefault((iy, ix + 1), []).append((iy, ix))
            if iy < ny:
                le = pad[iy + 1, ix]              # pixel left of edge col ix
                ri = pad[iy + 1, ix + 1]          # pixel right of edge col ix
                if le != ri:
                    if le:                        # material left -> travel +y
                        edges.setdefault((iy, ix), []).append((iy + 1, ix))
                    else:                         # material right -> travel -y
                        edges.setdefault((iy + 1, ix), []).append((iy, ix))

    loops = []
    while edges:
        start = next(iter(edges))
        loop = [start]
        cur = start
        prev_dir = None
        while True:
            nxts = edges[cur]
            if len(nxts) == 1:
                nxt = nxts.pop()
            else:
                # corner with two outgoing edges (checkerboard touch):
                # prefer the left turn to keep loops simple
                dy, dx = prev_dir
                # left turn of (dy,dx) is (dx,-dy) in (iy,ix) convention
                want = (cur[0] + dx, cur[1] - dy)
                if want in nxts:
                    nxts.remove(want)
                    nxt = want
                else:
                    nxt = nxts.pop()
            if not edges[cur]:
                del edges[cur]
            prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            if nxt == start:
                break
            loop.append(nxt)
            cur = nxt
        loops.append(loop)
    return loops


def _simplify(loop):
    """Merge collinear runs; keep closed orientation."""
    out = []
    n = len(loop)
    for i in range(n):
        a, b, c = loop[i - 1], loop[i], loop[(i + 1) % n]
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            out.append(b)
    return out


def _signed_area(vertices_xy):
    x = np.array([v[0] for v in vertices_xy])
    y = np.array([v[1] for v in vertices_xy])
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def mask_to_polygons(mask, pixel_nm=(20.0, 20.0), lattice_nm=(2400.0, 700.0)):
    """Return (outers, holes): lists of (n,2) float arrays in CENTERED nm."""
    dy, dx = pixel_nm
    Lx, Ly = lattice_nm
    outers, holes = [], []
    for loop in _trace_loops(np.asarray(mask, dtype=bool)):
        simp = _simplify(loop)
        xy = [(ix * dx - Lx / 2.0, iy * dy - Ly / 2.0) for iy, ix in simp]
        area = _signed_area(xy)
        (outers if area > 0 else holes).append(np.array(xy))
    return outers, holes


def rasterize(outers, holes, shape, pixel_nm=(20.0, 20.0),
              lattice_nm=(2400.0, 700.0)):
    """Point-in-polygon on pixel centers; verifies the export is exact."""
    ny, nx = shape
    dy, dx = pixel_nm
    Lx, Ly = lattice_nm
    xs = (np.arange(nx) + 0.5) * dx - Lx / 2.0
    ys = (np.arange(ny) + 0.5) * dy - Ly / 2.0
    X, Y = np.meshgrid(xs, ys)
    acc = np.zeros(shape, dtype=int)

    def inside(poly, X, Y):
        n = len(poly)
        cnt = np.zeros(X.shape, dtype=int)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            cond = ((y1 > Y) != (y2 > Y))
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = x1 + (Y - y1) * (x2 - x1) / (y2 - y1 + 1e-300)
            cnt += (cond & (X < xint)).astype(int)
        return cnt % 2 == 1

    for p in outers:
        acc += inside(p, X, Y).astype(int)
    for p in holes:
        acc -= inside(p, X, Y).astype(int)
    return acc > 0


def export(mask, pixel_nm=(20.0, 20.0), lattice_nm=(2400.0, 700.0)):
    """Polygons + exactness proof.  Raises if rasterization mismatches."""
    outers, holes = mask_to_polygons(mask, pixel_nm, lattice_nm)
    back = rasterize(outers, holes, np.asarray(mask).shape, pixel_nm,
                     lattice_nm)
    mism = int(np.sum(back != np.asarray(mask, dtype=bool)))
    if mism:
        raise AssertionError("polygon export not exact: %d pixels differ"
                             % mism)
    return outers, holes
