"""nfip pipeline for the NFIP decentralization analysis.

Subpackages / modules
---------------------
config          paths, constants, lookup tables (single source of truth)
data            file loaders (geospatial, claims, CPI, premiums, ENSO, results)
preprocess      claim shaping, CPI adjustment, event relabeling, balance sheet
st_clustering   temporal + ST-DBSCAN cluster-update engine
hurdat          HURDAT2 parsing + storm-track SVM splitting
dtw_clustering  DTW / consensus / wavelet state clustering
simulation      stepwise reinsurance coverage + pool balance simulation
plotting        shared figure helpers
figures         per-figure composition (thin notebooks call these)

Typical import style::

    from nfip import config, data, preprocess, simulation
    from nfip.config import PATHS
"""

import importlib

# Lightweight modules imported eagerly (no exotic dependencies).
from . import config, data, plotting, preprocess, simulation  # noqa: F401

# ``st_clustering`` (needs st_dbscan) and ``dtw_clustering`` (needs tslearn) and
# ``hurdat`` (needs shapely/requests) imported lazily.
_LAZY = {"st_clustering", "dtw_clustering", "hurdat", "figures"}


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "config", "data", "preprocess", "st_clustering", "hurdat",
    "dtw_clustering", "simulation", "plotting",
]

__version__ = "0.1.0"
