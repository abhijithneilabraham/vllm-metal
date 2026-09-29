# SPDX-License-Identifier: Apache-2.0
"""Tiled prefill and the sliding window: a tile left of a row's window.

The KV tile loop starts at the window of the threadgroup's first query row,
but the window mask is applied per row.  Rows later in the threadgroup can
see their first tile fully masked.  Such a tile must not update the row's
running max: pinned at 0, the row's in-window keys, all scoring far below 0,
are weighted down to nothing while the reference softmax is a near-uniform
average of V.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from tests.attention.test_bidi_prefill_kernel import (
    BLOCK,
    DTYPE,
    HD,
    HEADS,
    KV_HEADS,
    _kernel,
    _setup,
)

TILE_KV = 32  # TileConfig for head sizes 64, 96 and 128
assert HD in (64, 96, 128), "TILE_KV above is the tile size for these head sizes"
ATOL, RTOL = 1.5e-2, 1e-2


def _reference(query, key_cache, value_cache, table_row, *, n, seq_len, window):
    q = np.array(query.astype(mx.float32))
    kc = np.array(key_cache.astype(mx.float32))
    vc = np.array(value_cache.astype(mx.float32))
    k = np.stack([kc[table_row[p // BLOCK], p % BLOCK] for p in range(seq_len)])
    v = np.stack([vc[table_row[p // BLOCK], p % BLOCK] for p in range(seq_len)])
    n_rep = HEADS // KV_HEADS
    k = np.repeat(k, n_rep, axis=1)
    v = np.repeat(v, n_rep, axis=1)
    q_lo = seq_len - n
    qi = np.arange(q_lo, q_lo + n)[:, None]
    ki = np.arange(seq_len)[None, :]
    allowed = (ki <= qi) & ((qi - ki) < window)
    scores = np.einsum("qhd,khd->hqk", q, k) * HD**-0.5
    scores = np.where(allowed[None], scores, -1e30)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", probs, v)


@pytest.mark.parametrize("window_start_in_tile", [0, 1, 25])
def test_rows_whose_first_tile_is_fully_masked_stay_neutral(
    window_start_in_tile,
) -> None:
    """Rows from index TILE_KV - offset onward see their first tile fully
    masked when the threadgroup's window starts `offset` tokens into a
    tile; offset 0 is the control where no row does."""
    n, window, magnitude = 2 * TILE_KV, 96, 2.0
    window_start = 600 - n + 1 - window
    seq_len = 600 + (window_start_in_tile - window_start) % TILE_KV
    key_cache, value_cache, _, table = _setup(0, n=n, seq_len=seq_len)
    mx.random.seed(3)
    query = (mx.ones((n, HEADS, HD)) * magnitude).astype(DTYPE)
    key_cache = (
        -magnitude * mx.ones(key_cache.shape) + 0.05 * mx.random.normal(key_cache.shape)
    ).astype(DTYPE)
    mx.eval(query, key_cache)
    got = _kernel(
        query, key_cache, value_cache, table, n=n, seq_len=seq_len, window=window
    )
    ref = _reference(
        query,
        key_cache,
        value_cache,
        np.array(table[0]).tolist(),
        n=n,
        seq_len=seq_len,
        window=window,
    )
    np.testing.assert_allclose(np.array(got), ref, atol=ATOL, rtol=RTOL)
