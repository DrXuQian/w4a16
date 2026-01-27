# Marlin `init_slice()` Examples (How CTA Stripes Are Assigned)

This note explains what `init_slice()` computes in `homemade_marlin/marlin_standalone.cu` and gives several concrete,
numeric examples so you can sanity-check the scheduling.

The core idea: each CTA gets a 1D **stripe** of length `iters` over the flattened `(slice_col_par, slice_row)` grid.
If a stripe crosses a column boundary, the CTA will process it as multiple **slices** (one per column).
If multiple CTAs touch the same output N tile, the code assigns `slice_idx`/`slice_count` so they can serialize a
cross-CTA reduction via `locks[slice_col]`.

---

## Key Definitions

Per kernel instance:

- Tile sizes (in matrix elements):
  - `tile_m = 16 * thread_m_blocks`
  - `tile_n = 16 * thread_n_blocks`
  - `tile_k = 16 * thread_k_blocks`
- Tile grid sizes:
  - `k_tiles = prob_k / tile_k = prob_k / (16 * thread_k_blocks)`
  - `n_tiles = prob_n / tile_n = prob_n / (16 * thread_n_blocks)`
- Grid:
  - `blocks = gridDim.x`
  - `iters = ceildiv(k_tiles * n_tiles * parallel, blocks)`
  - (grouped quant) if `group_blocks != -1`, `iters` is rounded up to a multiple of
    `GROUP_PIPES = group_blocks / thread_k_blocks` (so slice boundaries align with scale groups).

Flattened coordinate system:

- `t = iters * blockIdx.x`
- `slice_col_par = t / k_tiles` (column id over **N tiles**, extended by `parallel`)
- `slice_row     = t % k_tiles` (row id over **K tiles** within that column)

Stripe length:

- CTA `blockIdx.x` owns flattened range:
  - `[t, t + iters)`
- But each column has only `k_tiles` rows, so a stripe may cross into the next column(s).

What `init_slice()` sets (values used outside the lambda):

- `slice_iters`: how many K-tiles this CTA will do **in the current column**.
- `slice_count`: how many CTAs contribute to the same output N tile (`slice_col`) and thus must be reduced.
- `slice_idx`: this CTA’s rank `0..slice_count-1` in the reduction order (used with `locks[slice_col]`).
- It may also wrap `slice_col` and advance `A/C/locks` when `slice_col == n_tiles` (crossing a `parallel` boundary).

---

## Example 1: One CTA Per Column (No Reduction)

Assume:

- `prob_n = 2048, prob_k = 2048`
- `thread_n_blocks = 8`  => `tile_n = 128` => `n_tiles = 2048 / 128 = 16`
- `thread_k_blocks = 8`  => `tile_k = 128` => `k_tiles = 2048 / 128 = 16`
- `parallel = 1`
- `blocks = 16`

Then:

- `iters = ceildiv(16 * 16 * 1, 16) = 16`
- So `iters == k_tiles`, meaning each CTA owns exactly one full column (one N tile) end-to-end.

Selected CTAs:

| blockIdx.x | t=iters*b | slice_col_par | slice_row | slice_iters | slice_count | slice_idx |
|-----------:|----------:|--------------:|----------:|------------:|------------:|----------:|
| 0 | 0  | 0 | 0 | 16 | 1 | 0 |
| 1 | 16 | 1 | 0 | 16 | 1 | 0 |
| 2 | 32 | 2 | 0 | 16 | 1 | 0 |

Interpretation:

- Each CTA computes one full `(tile_m x tile_n)` output tile for its `slice_col`.
- `slice_count=1` => no cross-CTA reduction is needed.

---

## Example 2: Multiple CTAs Per Column (Cross-CTA Reduction + Cross-Column Stripes)

Same shape/config as Example 1, but with more CTAs:

- `blocks = 48`
- `k_tiles = n_tiles = 16`, `parallel = 1`

Then:

- `iters = ceildiv(16 * 16 * 1, 48) = ceildiv(256, 48) = 6`
- Since `iters < k_tiles`, a column is split across multiple CTAs => `slice_count > 1`.
- Some CTAs start near the end of a column (`slice_row` large), so their stripe crosses into the next column.

### 2.1 Per-CTA First Slice (and Carry-Over Slice)

| blockIdx.x | t | (col_par,row) | slice_iters | slice_count | slice_idx | notes |
|-----------:|--:|--------------:|------------:|------------:|----------:|------|
| 0 | 0  | (0, 0)  | 6 | 3 | 2 | col0 rows 0..5 |
| 1 | 6  | (0, 6)  | 6 | 3 | 1 | col0 rows 6..11 |
| 2 | 12 | (0,12)  | 4 | 3 | 0 | col0 rows 12..15, then carries |
| 2 (next) | — | (1, 0) | 2 | 4 | 3 | col1 rows 0..1 (carry-over) |
| 3 | 18 | (1, 2)  | 6 | 4 | 2 | col1 rows 2..7 |
| 4 | 24 | (1, 8)  | 6 | 4 | 1 | col1 rows 8..13 |
| 5 | 30 | (1,14)  | 2 | 4 | 0 | col1 rows 14..15, then carries |
| 5 (next) | — | (2, 0) | 4 | 3 | 2 | col2 rows 0..3 (carry-over) |
| 6 | 36 | (2, 4)  | 6 | 3 | 1 | col2 rows 4..9 |
| 7 | 42 | (2,10)  | 6 | 3 | 0 | col2 rows 10..15 |

Notes:

- For `block2`, the stripe is length 6 starting at `(col0,row12)`:
  - it can only do 4 rows in col0 (12..15) => `slice_iters=4`
  - the remaining 2 rows land in the next column col1 at row 0..1 (that’s why the “next” slice has `slice_iters=2`)
- For reduction, a smaller `slice_idx` runs earlier in the serialized `locks[slice_col]` order:
  - `slice_idx==0` is “first” (`first=true`) in `global_reduce()`
  - `slice_idx==slice_count-1` is “last” (`last=true`) and performs `write_result()`

### 2.2 Visual Stripe Coverage (First Few Columns)

Each column has `k_tiles=16` rows (0..15). Here’s how the first few columns are covered:

```
col0: [ 0.. 5]=b0  [ 6..11]=b1  [12..15]=b2
col1: [ 0.. 1]=b2  [ 2.. 7]=b3  [ 8..13]=b4  [14..15]=b5
col2: [ 0.. 3]=b5  [ 4.. 9]=b6  [10..15]=b7
```

### 2.3 Why `slice_idx` Looks “Reversed”

For `col0`, the contributors are:

- `b2` covers rows 12..15 => `slice_idx=0` (runs first)
- `b1` covers rows  6..11 => `slice_idx=1`
- `b0` covers rows  0.. 5 => `slice_idx=2` (runs last)

So the reduction order is roughly from larger `slice_row` to smaller `slice_row`. The code computes this order using
`col_first/col_off/delta_first` so that stripes that cross boundaries (“carry-over” slices) also get a consistent rank.

---

## Example 3: `iters > k_tiles` (A CTA Spans Multiple Columns, Still No Reduction)

Keep `k_tiles=n_tiles=16`, `parallel=1`, but set:

- `blocks = 8`

Then:

- `iters = ceildiv(16 * 16 * 1, 8) = 32`
- A stripe is longer than one column (`k_tiles=16`), so each CTA will process multiple full columns.
- But each column is still only touched by one CTA => `slice_count=1`.

Selected CTAs (showing the first two slices per CTA):

| blockIdx.x | start (col_par,row) | first slice_iters | next (col_par,row) | next slice_iters | slice_count |
|-----------:|---------------------:|-----------------:|-------------------:|----------------:|------------:|
| 0 | (0,0) | 16 | (1,0) | 16 | 1 |
| 1 | (2,0) | 16 | (3,0) | 16 | 1 |
| 2 | (4,0) | 16 | (5,0) | 16 | 1 |

Interpretation:

- Each CTA covers two consecutive N tiles (because `iters=2*k_tiles`).
- No reduction is needed because stripes do not overlap on any column.

---

## Example 4: `parallel > 1` (M-Slicing Inside the Kernel)

This happens when `prob_m > 16 * thread_m_blocks` inside the kernel:

```cpp
parallel = prob_m / (16 * thread_m_blocks);
prob_m = 16 * thread_m_blocks;
```

Meaning:

- The kernel internally treats the work as `parallel` independent M-slices.
- `slice_col_par` ranges over `0 .. n_tiles*parallel-1`.
- When `slice_col_par >= n_tiles`, the code offsets `A/C/locks` to the next M-slice and sets:
  - `slice_col = slice_col_par % n_tiles`

Concrete numbers (same `k_tiles=n_tiles=16`, but `parallel=4`, `blocks=48`):

- Total tiles = `16 * 16 * 4 = 1024`
- `iters = ceildiv(1024, 48) = 22`

Selected CTAs:

| blockIdx.x | t | slice_col_par | par_idx = slice_col_par/n_tiles | slice_col = slice_col_par%n_tiles | slice_row |
|-----------:|--:|--------------:|--------------------------------:|----------------------------------:|----------:|
| 11 | 242 | 15 | 0 | 15 | 2 |
| 12 | 264 | 16 | 1 |  0 | 8 |
| 13 | 286 | 17 | 1 |  1 | 14 |

Interpretation:

- `block12` has `slice_col_par=16`, so `par_idx=1` and `slice_col=0`:
  it computes N tile 0, but for the *second* M-slice (A/C pointers advanced accordingly).

---

## Quick Sanity Rules

- If `iters >= k_tiles`, a CTA will usually cover whole columns (possibly multiple columns), and `slice_count` tends to be 1.
- If `iters < k_tiles`, multiple CTAs will touch the same column => `slice_count > 1` and `locks[slice_col]` is used.
- If `slice_row + iters > k_tiles`, the CTA will get `slice_iters < iters` for that column and then “carry over” to the next column.

