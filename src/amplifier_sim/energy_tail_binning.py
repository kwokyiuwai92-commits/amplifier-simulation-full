"""Input-only tail quadrature with native central samples and conserved mass."""
from __future__ import annotations

import numpy as np

from .spectral_reporting import coordinate_bin_edges


def energy_tail_groups(
    coordinate: np.ndarray,
    mass: np.ndarray,
    *,
    coverage: float,
    tail_points: int,
    bin_edges: np.ndarray | None = None,
) -> dict:
    """Keep an equal-tail energy interval; distribute a TOTAL tail budget.

    Each nonempty tail gets one group first. Additional groups go to the
    tail with the greater energy per assigned group (lower tail breaks ties).
    A group never crosses the resolved centre or joins the two tails.
    """
    x = np.asarray(coordinate, dtype=float)
    e = np.asarray(mass, dtype=float)
    if (x.ndim != 1 or e.shape != x.shape or not len(x)
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(e))
            or np.any(e < 0) or (len(x) > 1 and not np.all(np.diff(x) > 0))):
        raise ValueError('Tail binning requires increasing coordinates and finite nonnegative mass')
    if not np.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError('Tail coverage must be in (0, 1]')
    if isinstance(tail_points, bool) or not isinstance(tail_points, (int, np.integer)) or tail_points < 2:
        raise ValueError('Total tail points must be an integer of at least two')
    edges = (
        coordinate_bin_edges(x) if bin_edges is None and len(x) > 1
        else None if bin_edges is None else np.asarray(bin_edges, dtype=float)
    )
    if edges is not None and (
        edges.shape != (len(x) + 1,) or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
        or np.any(x < edges[:-1]) or np.any(x > edges[1:])
    ):
        raise ValueError('Bin edges must increase and enclose each original coordinate')
    total = float(e.sum())
    if total == 0 or coverage == 1:
        lo, hi = 0, len(e)-1
    else:
        cumulative = np.cumsum(e) / total
        tail = (1-coverage)/2
        lo = int(np.searchsorted(cumulative, tail, side='left'))
        hi = min(int(np.searchsorted(cumulative, 1-tail, side='left')), len(e)-1)
    tails = [np.arange(lo), np.arange(hi+1, len(e))]
    counts = [int(bool(len(t))) for t in tails]
    while sum(counts) < tail_points:
        eligible = [i for i in range(2) if counts[i] < len(tails[i])]
        if not eligible:
            break
        side = max(eligible, key=lambda i: float(e[tails[i]].sum())/max(counts[i], 1))
        counts[side] += 1
    guards = [[g for g in np.array_split(t, n) if len(g)] if n else []
              for t, n in zip(tails, counts)]
    centre = [np.array([i]) for i in range(lo, hi+1)]
    groups = tuple(guards[0] + centre + guards[1])
    weights = np.array([e[g].sum() for g in groups])
    representatives = np.array([
        x[g[0]] if len(g) == 1 else
        np.average(x[g], weights=e[g]) if weights[i] > np.finfo(float).tiny else x[g].mean()
        for i, g in enumerate(groups)
    ])
    grouped_edges = (
        None if edges is None else edges[[groups[0][0], *[g[-1] + 1 for g in groups]]]
    )
    return dict(groups=groups, coordinate=representatives, mass=weights,
                bin_edges=grouped_edges,
                central_indices=(lo, hi), central_count=hi-lo+1,
                tail_counts=tuple(counts), total_groups=len(groups), source_count=len(e),
                central_energy_fraction=float(e[lo:hi+1].sum()/total) if total else 1.)
