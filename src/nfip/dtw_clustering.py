"""DTW / consensus / wavelet clustering of state balance series.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pywt
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import cdist_dtw
from tqdm import tqdm


# Per-simulation DTW + hierarchical clustering
def cluster_one_simulation(pivot_df: pd.DataFrame, k: int = 4,
                           linkage_method: str = "average") -> tuple[list, pd.Index]:
    """Normalize, DTW-distance, and hierarchically cluster one simulation.

    ``pivot_df`` is years x states.  Returns ``(labels, state_columns)`` where
    ``labels[i]`` is the cluster of ``state_columns[i]``.
    """
    normalized = StandardScaler().fit_transform(pivot_df)
    normalized_df = pd.DataFrame(normalized, index=pivot_df.index, columns=pivot_df.columns)

    data = normalized_df.T.values
    dtw_distances = cdist_dtw(data)

    condensed_dist = squareform(dtw_distances)
    Z = linkage(condensed_dist, method=linkage_method)
    labels = fcluster(Z, k, criterion="maxclust")
    return labels, normalized_df.columns


def run_simulation_clustering(nfip_balances: pd.DataFrame, k: int = 4) -> dict:
    """Cluster every simulation and collect per-state label vectors.

    ``nfip_balances`` must have ['simulation','year','state_abbrev','nfip_balance'].
    Returns ``cluster_records`` = {state: [label_per_simulation, ...]}.
    """
    simulations = nfip_balances["simulation"].unique()
    nfip_balances = nfip_balances.dropna(subset=["state_abbrev"])
    nfip_balances["state_abbrev"] = nfip_balances["state_abbrev"].astype(str)

    cluster_records = defaultdict(list)
    for sim in tqdm(simulations, desc="Clustering simulations"):
        df_sim = nfip_balances[nfip_balances["simulation"] == sim]
        pivot_df = df_sim.pivot_table(
            index="year", columns="state_abbrev", values="nfip_balance", fill_value=0
        )
        labels, columns = cluster_one_simulation(pivot_df, k=k)
        for state, cluster_label in zip(columns, labels):
            cluster_records[state].append(cluster_label)
    return cluster_records


def cluster_records_to_wide(cluster_records: dict) -> pd.DataFrame:
    """Turn ``cluster_records`` into the wide 'state x sim_i' frame the notebook saves."""
    df_wide = pd.DataFrame(cluster_records).T
    df_wide.columns = [f"sim_{i}" for i in range(df_wide.shape[1])]
    df_wide.index.name = "state"
    return df_wide.reset_index()


def wide_to_cluster_records(df_wide: pd.DataFrame) -> dict:
    """Inverse of :func:`cluster_records_to_wide` (Clustering cell 19 ``else`` branch)."""
    df_wide_indexed = df_wide.set_index("state")
    return df_wide_indexed.apply(lambda row: row.tolist(), axis=1).to_dict()


# Consensus clustering via co-association
def build_consensus_clusters(cluster_records: dict, k: int = 3,
                             linkage_method: str = "average"):
    """Co-association matrix -> dissimilarity -> hierarchical consensus clusters.

    Returns ``(cluster_df, Z, coassoc_matrix)`` where ``cluster_df`` has
    ['state','consensus_cluster'].
    """
    states = sorted(cluster_records.keys())
    n_states = len(states)
    n_sims = len(next(iter(cluster_records.values())))

    coassoc_matrix = np.zeros((n_states, n_states))
    for sim in tqdm(range(n_sims), desc="Building co-association matrix"):
        sim_labels = {state: cluster_records[state][sim] for state in states}
        for i, state_i in enumerate(states):
            for j, state_j in enumerate(states):
                if sim_labels[state_i] == sim_labels[state_j]:
                    coassoc_matrix[i, j] += 1
    coassoc_matrix /= n_sims

    dissimilarity = 1 - coassoc_matrix
    Z = linkage(dissimilarity, method=linkage_method)
    final_labels = fcluster(Z, k, criterion="maxclust")

    cluster_df = pd.DataFrame({"state": states, "consensus_cluster": final_labels})
    return cluster_df, Z, coassoc_matrix


# Single-series DTW + wavelet energy
def normalize_series_matrix(multivariate: pd.DataFrame):
    """z-score each column and return (normalized_df, data, labels)."""
    normalized_df = pd.DataFrame(
        StandardScaler().fit_transform(multivariate),
        index=multivariate.index,
        columns=multivariate.columns,
    )
    data = normalized_df.T.values
    labels = normalized_df.columns.tolist()
    return normalized_df, data, labels


def dtw_linkage(data: np.ndarray, linkage_method: str = "average"):
    """DTW distance matrix + average-linkage tree (Clustering-Historic cells 18-20)."""
    dtw_distances = cdist_dtw(data)
    condensed_dist = squareform(dtw_distances)
    Z = linkage(condensed_dist, method=linkage_method)
    return dtw_distances, Z


def extract_wavelet_energy(series, wavelet: str = "db4", level: int = 4):
    """Per-band wavelet energy of a 1-D series (verbatim, Clustering-Historic cell 31)."""
    coeffs = pywt.wavedec(series, wavelet=wavelet, level=level)
    energy = [np.sum(c ** 2) for c in coeffs]
    return energy


def wavelet_energy_features(normalized_df: pd.DataFrame, wavelet: str = "db4", level: int = 4):
    """Build the states x band-energy feature matrix (Clustering-Historic cell 31)."""
    energy_features = []
    labels = normalized_df.columns.tolist()
    for col in labels:
        series = normalized_df[col].values
        energy_features.append(extract_wavelet_energy(series, wavelet=wavelet, level=level))
    energy_df = pd.DataFrame(energy_features, index=labels)
    energy_df.columns = [f"energy_L{lvl}" for lvl in range(len(energy_df.columns))]
    return energy_df
