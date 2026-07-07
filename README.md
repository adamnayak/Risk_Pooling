# NFIP Pooling Decentralization Simulation

Code accompanying:

> Nayak, A., et al. 2026., 2026. *AFEMA Phase-Out? Catastrophic Extremes Challenge Decentralization of U.S. Flood Insurance.*
> *Proceedings of the National Academies of Sciences* (accepted).
> DOI: `<add on publication>`
>
> Preprint available at: https://eartharxiv.org/repository/view/10790/

This repository reproduces the spatiotemporal flood-loss clustering, tropical
cyclone attribution, state-balance clustering, and pool-solvency simulation used
to evaluate decentralization of the U.S. National Flood Insurance Program (NFIP)
into state and regional risk pools.

## Overview

The analysis proceeds in four stages, backed by a small Python package
(`nfip`) that holds the shared logic:

1. **Claim clustering.** NFIP claims are grouped into spatiotemporal flood
   events using a temporal DBSCAN pass within quiet-period splits followed by
   ST-DBSCAN on event proxy points.
2. **Storm attribution.** Clusters are matched to Atlantic tropical cyclone
   tracks (HURDAT2), and claims in clusters spanning more than one storm are
   partitioned with a space-time support vector classifier.
3. **State-balance clustering.** Simulated per-state pool balances are grouped
   with DTW-based hierarchical clustering and a co-association consensus across
   simulations.
4. **Pool simulation.** A year-resampled (historic, block-bootstrap, or random)
   balance-sheet simulation propagates event losses through federal and
   decentralized pool structures, with optional reinsurance and
   insurance-linked-securities (ILS) coverage.

## Repository structure

```
.
├── src/nfip/               # analysis package
│   ├── config.py           # paths, constants, and lookup tables
│   ├── data.py             # data loaders
│   ├── preprocess.py       # claim shaping, CPI adjustment, balance sheet
│   ├── st_clustering.py    # temporal + ST-DBSCAN claim clustering
│   ├── hurdat.py           # HURDAT2 parsing and storm-track attribution
│   ├── dtw_clustering.py   # DTW / consensus / wavelet state clustering
│   ├── simulation.py       # pool balance simulation with reinsurance and ILS
│   └── plotting.py         # shared figure helpers
├── notebooks/
│   ├── 01_cluster_update.ipynb   # claim clustering
│   ├── 02_splitting.ipynb        # storm attribution and splitting
│   ├── 03_dtw_clustering.ipynb   # state-balance consensus clustering
│   ├── 04_simulation.ipynb       # pool simulation
│   ├── 05_fig1.ipynb             # Figure 1
│   ├── 06_fig2.ipynb             # Figure 2
│   ├── 07_fig3.ipynb             # Figure 3
│   └── 08_fig4.ipynb             # Figure 4
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/[USER]/[REPO].git
cd [REPO]
pip install -e .
```

The DTW (`tslearn`) and ST-DBSCAN (`st-dbscan`) dependencies are imported only
where used, so the simulation and figure notebooks run without them if the
clustering notebooks are not needed.

## Data

The input datasets are not redistributed here. All are publicly available:

| Input | Source |
|-------|--------|
| NFIP redacted claims and policy/premium records | OpenFEMA |
| County, state, urban-area, and river geometries | U.S. Census TIGER / Cartographic Boundary files |
| Consumer Price Index (CPIAUCSL) | U.S. Bureau of Labor Statistics |
| Niño 3.4 anomaly index | NOAA / HadISST |
| Atlantic tropical cyclone best-track (HURDAT2) | NOAA National Hurricane Center (downloaded by `02_splitting`) |

By default the notebooks expect the directory layout used during development
(`../Local_Data`, `../2_Low_Return_Period`, and a local `Results/`). To point
the pipeline at a different root, override the path object once:

```python
from nfip.config import Paths
PATHS = Paths(base="/path/to/data")
```

## Reproducing the analysis

Run the notebooks in dependency order:

1. `01_cluster_update` and `02_splitting` produce the clustered, storm-attributed
   claim set.
2. `04_simulation` runs the pool simulation and writes the balance-sheet outputs
   to `Results/`.
3. `03_dtw_clustering` clusters the simulated state balances and writes the
   consensus labels used in Figure 2.
4. `05_fig1` through `08_fig4` render the main-text figures from the simulation
   outputs and consensus clusters.

`01_cluster_update` additionally imports `sensitivity_analysis` from a local
`ST_Cluster.py`; place that module on the path (for example in `src/`) before
running the county-analysis cell, or remove that cell if it is not needed.

## Package notes

Constants and lookup tables (inflation reference, reinsurance and ILS
attachment structures, coordinate systems, FIPS mappings, and the
sociodemographic tables) are collected in `config.py`. Loaders in `data.py`
return raw frames; derived quantities such as CPI-adjusted losses, event
relabeling, and the state balance sheet are built in `preprocess.py`.
```

## License

[LICENSE]. See `LICENSE` for details.

## Contact

Adam Nayak, Columbia University, an3232@columbia.edu
