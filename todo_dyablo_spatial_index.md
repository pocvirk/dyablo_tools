# TODO: Add spatial index to Dyablo HDF5 output

## Motivation

`load_subcube_h5py.py` currently streams the full connectivity dataset (~1.7 GB)
to find which cells belong to a sub-cube. This pass is independent of sub-cube
size — a 1/64-volume sub-cube costs the same ~3s as the full box.

With a small metadata dataset storing per-rank Morton intervals, pass 1 could
jump directly to the relevant cells, reducing scan time proportionally to the
sub-cube volume fraction.

## The change

**File:** `dyablo/core/src/io/IOManager_hdf5.cpp`

At write time, each MPI rank already knows:
- `first_local_oct` — its global octant offset (used for HDF5 hyperslab writes)
- its Morton key interval `[morton_begin, morton_end)` — computed in
  `AMRmesh::loadBalance()` and available via `morton_intervals`

Add a small collective write after the main datasets, writing two datasets
of size `nranks`:

```cpp
// Gather first_local_oct from all ranks -> array of size nranks+1
// (last entry = total_octs, gives the end of the last rank's range)
std::vector<uint64_t> rank_offsets(mpi_size + 1);
mpi_comm.MPI_Allgather(&first_local_oct, rank_offsets.data(), 1);
rank_offsets[mpi_size] = total_num_octs;

// Gather Morton intervals -> array of size nranks+1
// (already computed as morton_intervals in loadBalance)

// Write as HDF5 datasets (only rank 0 needs to write these)
if (mpi_rank == 0) {
    write_dataset(file, "metadata/rank_cell_offsets",  rank_offsets);
    write_dataset(file, "metadata/rank_morton_intervals", morton_intervals);
}
```

**Result:** two tiny datasets (nranks+1 uint64 values each, a few KB) that
allow a reader to:
1. Read `metadata/rank_morton_intervals` (few KB)
2. Compute which ranks' Morton intervals overlap the requested sub-cube
3. Read `metadata/rank_cell_offsets` to get the HDF5 hyperslab for those ranks
4. Read connectivity/coordinates/fields only for those hyperslabs

## Expected impact

For a sub-cube covering fraction f of the volume on nranks MPI ranks:
- Current:  always reads full connectivity (~1.7 GB) — O(1) in f
- With index: reads ~f * nranks_overlap/nranks of connectivity — O(f)

Example: 1/64 sub-cube on 256 ranks → ~4 rank hyperslabs → 4/256 of
connectivity read instead of all of it. Pass 1 drops from ~3s to ~0.05s.

## Notes
- Requires finding the right branch where `morton_intervals` is accessible
  from `IOManager_hdf5.cpp` (it lives in `AMRmesh::PData`)
- The change is backward-compatible: readers that don't use the index
  (pyablo, old load_subcube_h5py) are unaffected
- Morton interval → sub-cube overlap test: for a power-of-2 aligned sub-cube,
  compute its Morton key range and check for interval intersection
