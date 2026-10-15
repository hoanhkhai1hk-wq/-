"""Class pixel counts and per-field connected-component tables (pixel units)."""

from collections import deque

import numpy as np

NODATA_CLASS = 255


def class_counts(mask):
    """Counts per class value present in {0,1,2,255}."""
    return {
        "background_0": int(np.count_nonzero(mask == 0)),
        "field_interior_1": int(np.count_nonzero(mask == 1)),
        "boundary_2": int(np.count_nonzero(mask == 2)),
        "nodata_255": int(np.count_nonzero(mask == NODATA_CLASS)),
    }


def field_table(mask):
    """4-connected components of class 1.

    Returns rows of dicts: field_id (1-based), pixels, min_row, min_col,
    max_row, max_col. Pure pixel units; NOT verified square metres.
    """
    region = mask == 1
    h, w = region.shape
    visited = np.zeros((h, w), dtype=bool)
    rows = []
    field_id = 0
    for r0 in range(h):
        for c0 in range(w):
            if region[r0, c0] and not visited[r0, c0]:
                field_id += 1
                q = deque([(r0, c0)])
                visited[r0, c0] = True
                pixels = 0
                min_r = max_r = r0
                min_c = max_c = c0
                while q:
                    r, c = q.popleft()
                    pixels += 1
                    if r < min_r:
                        min_r = r
                    if r > max_r:
                        max_r = r
                    if c < min_c:
                        min_c = c
                    if c > max_c:
                        max_c = c
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < h and 0 <= cc < w and region[rr, cc] and not visited[rr, cc]:
                            visited[rr, cc] = True
                            q.append((rr, cc))
                rows.append(
                    {
                        "field_id": field_id,
                        "pixels": pixels,
                        "min_row": int(min_r),
                        "min_col": int(min_c),
                        "max_row": int(max_r),
                        "max_col": int(max_c),
                    }
                )
    return rows
