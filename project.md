# MultiAmr2Cube Optimisation

## Context
Dyablo cosmological simulation snapshot (`star_iter0070000`), `bx=4`, `level_min=6`, `level_max=8` → adjusted cell levels lmin=8, lmax=10.

`MultiAmr2Cube` reprojects AMR data onto a regular `2^lcube` Cartesian grid.
Two cell paths:
- **Fine** (`lcell >= lcube`): vectorised numpy scatter write
- **Coarse** (`lcell < lcube`): fills a `zoom^3` sub-cube per cell — was a Python loop, the bottleneck

All work lives in `claude/`. Existing files in the parent directory are untouched.

## Files
| File | Role |
|---|---|
| `multiamr2cube_ref.py` | Reference single-field implementation + `extract_snapshot_data()` |
| `multiamr2cube_numpy.py` | Batched coarse fill via per-level numpy broadcasting |
| `multiamr2cube_numba.py` | `@njit(parallel=True)` coarse fill kernel only |
| `multiamr2cube_numba_full.py` | Fully-numba single-pass kernel: index compute + fine scatter + coarse fill |
| `benchmark.py` | CLI benchmark (loads data once, times all variants, writes `results.csv`) |
| `compare.ipynb` | Interactive notebook: timing bars, slice plots, diff maps |

## Results (from `timing.png`)
| lcube | grid | ref | numpy | numba | numba_full |
|---|---|---|---|---|---|
| 8 | 256³ | 0.6s | 0.6s (1.0x) | 0.6s (1.0x) | 0.055s (**11x**) |
| 9 | 512³ | 9.5s | 1.1s (9.5x) | 0.7s (13.4x) | 0.21s (**46x**) |
| 10 | 1024³ | 17.5s | 11s (1.5x) | 2.4s (7.2x) | 2.4s (**7.4x**) |

## Key findings

**lcube=8 (fine cells only):**
All variants except `numba_full` are identical at ~0.6s. The bottleneck was never the coarse loop but the index computation: numpy allocates ~15 temporary arrays of 27M float64 (~2 GB total) for `2**lcell`, `i/j/k`, `idxi/idxj/idxk`, etc. `numba_full` fuses everything into one pass with bit shifts → ~1 GB total memory traffic vs ~30 GB → 11x speedup.

**lcube=9 (first coarse level, sweet spot):**
`numba_full` wins 46x. Two bottlenecks eliminated simultaneously: Python loop overhead (~1μs/cell × millions of coarse cells) AND temp array allocations. Output (1 GB) is still small enough that memory bandwidth is not limiting.

**lcube=10 (heavier coarse load):**
`numba_full` and `numba` converge (~7x). Output is 8 GB — both are now memory-bandwidth-bound writing the large array. `numpy` collapses to 1.5x due to huge intermediate array allocations (`bases[:,None]+offsets[None,:]`).

**Scaling analysis (ref time decomposition):**
```
ref   ≈ temp_array_cost (~0.6s, constant)
      + Python_loop_cost (∝ N_coarse, grows with lcube)
      + memory_BW_cost (∝ 8^lcube, grows exponentially)

numba_full ≈ memory_BW_cost only
```
Speedup = (sum of all three) / memory_BW_cost → collapses toward 1 as lcube → ∞.
lcube=9 is the sweet spot because Python loop dominates ref while output is still small.
lcube=11 (2048³ = 64 GB) is likely infeasible on most machines.

## Conclusion
**`numba_full` is the winner across all cases** and should be the production implementation.
The next natural step would be multi-field support (straightforward extension of the single-pass kernel).
