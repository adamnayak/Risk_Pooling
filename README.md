# NFIP decentralization pipeline (refactored)

Refactor of the ten analysis notebooks into a `src/` package (`nfip`) plus thin
notebooks that call into it. The scientific logic is a **verbatim** move from the
notebooks — same parameters, same operations, same order — so results should be
identical; this pass changes *structure*, not behavior.

## Layout

```
src/nfip/
  config.py          paths, constants, lookup tables  (single source of truth)
  data.py            file loaders (geospatial, claims, CPI, premiums, ENSO, results)
  preprocess.py      claim shaping, CPI adjustment, event relabeling, balance sheet
  st_clustering.py   temporal + ST-DBSCAN cluster-update engine
  hurdat.py          HURDAT2 parsing + storm-track SVM splitting
  dtw_clustering.py  DTW / consensus / wavelet state clustering
  simulation.py      stepwise reinsurance coverage + pool balance simulation
  plotting.py        shared figure helpers (drawdowns, stackplot, grouped bar, ...)
  figures/           (note only — figure composition lives in the notebooks)
notebooks/
  01_cluster_update.ipynb   <- 2025_Cluster_Update
  02_splitting.ipynb        <- Splitting
  03_dtw_clustering.ipynb   <- Clustering
  04_simulation.ipynb       <- NEW_Current_Rules_Simulation
  05_fig1.ipynb .. 09_fig4_si.ipynb  <- Fig_1 .. Fig_4_SI
```

Original notebook → package mapping:

| Original                         | Notebook                  | Main module(s)                     |
|----------------------------------|---------------------------|------------------------------------|
| `2025_Cluster_Update`            | `01_cluster_update`       | `st_clustering`, `preprocess`, `data` |
| `Splitting`                      | `02_splitting`            | `hurdat`, `data`                   |
| `Clustering` / `Clustering-Historic` | `03_dtw_clustering`   | `dtw_clustering`, `preprocess`, `data` |
| `NEW_Current_Rules_Simulation`   | `04_simulation`           | `simulation`, `preprocess`, `data` |
| `Fig_1` … `Fig_4_SI`             | `05_fig1` … `09_fig4_si`  | `data`, `preprocess`, `plotting`   |

## Install & run

```bash
pip install -e .           # or: pip install -r requirements.txt
cd notebooks && jupyter lab
```

Point the whole pipeline at your data root in one line (defaults reproduce the
original `../Local_Data`, `../2_Low_Return_Period`, `Results/` layout):

```python
from nfip.config import Paths
PATHS = Paths(base="/path/to/data")
```

The DTW (`tslearn`) and ST-DBSCAN (`st_dbscan`) modules are imported lazily, so
you can run e.g. the simulation without installing every dependency.

## What changed vs. the notebooks (structure only)

- The data-load / CPI-adjust / premium / event-relabel blocks that were
  copy-pasted across the simulation and all five figure notebooks now live once
  in `data.py` / `preprocess.py`.
- All hard-coded constants and tables (`BASE_CPI`, reinsurance/ILS vectors, CRS,
  FIPS dictionaries, GDP/income/poverty tables, event-relabel maps) are in
  `config.py`.
- The cluster-update and HURDAT function definitions moved into modules
  unchanged; the notebook script cells that were pure procedure are wrapped into
  functions with the loose globals promoted to explicit parameters.
- Figure notebooks are thin: the shared load prefix is replaced by `nfip` calls,
  and **every figure-composition cell is verbatim** from the original.

## Two things to know

1. **External dependency — your `ST_Cluster.py`.** `01_cluster_update` still does
   `from ST_Cluster import sensitivity_analysis` (the county-analysis / sensitivity
   step). That function — plus `assemble_temporal_labels` / `assemble_final_st_labels`
   used by a superseded code path — lives in your existing `src/ST_Cluster.py`,
   which was not part of this refactor. Keep it importable on `sys.path`.
   `EARTH_KM` (previously defined there and inline) is now in `nfip.config`.

2. **Not executed end-to-end here.** The refactor was validated by moving logic
   verbatim and by import/functional smoke tests (HURDAT parsing incl. lat/lon
   sign handling, stepwise coverage, wavelet energy, event relabeling,
   sociodemographic build). It was **not** run against the full datasets (they
   aren't included). Recommended check: run one config (e.g.
   `test_case='block'`, `input_clustering='_new'`) through the old and new
   notebooks and diff the four `Results/*.csv` and the figure PNGs.

Superseded / dead cells from the originals (e.g. the alternate
`assemble_temporal_labels` merge path in the cluster-update notebook, and the
non-DTW / wavelet exploratory panels in `Clustering-Historic`) were left out of
the thin callers; say the word and I can fold them back in as optional cells.
