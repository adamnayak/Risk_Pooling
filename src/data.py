"""File loaders.
"""

from __future__ import annotations

import io
from typing import Optional

import geopandas as gpd
import pandas as pd

from . import config
from .config import PATHS, Paths


# Geospatial
def load_counties(paths: Paths = PATHS, crs: Optional[str] = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(paths.counties_shp)
    return gdf.to_crs(crs) if crs else gdf


def load_states(paths: Paths = PATHS, crs: Optional[str] = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(paths.states_shp)
    return gdf.to_crs(crs) if crs else gdf


def load_rivers(paths: Paths = PATHS) -> gpd.GeoDataFrame:
    return gpd.read_file(paths.rivers_shp)


def load_urban(paths: Paths = PATHS) -> gpd.GeoDataFrame:
    return gpd.read_file(paths.urban_shp)


def try_load_counties(paths: Paths = PATHS, crs: Optional[str] = None):
    """Best-effort loader used by Splitting (returns None if the file is absent)."""
    try:
        gdf = gpd.read_file(paths.counties_shp)
        return gdf.to_crs(crs) if crs else gdf
    except Exception:
        return None


def try_load_states(paths: Paths = PATHS, crs: Optional[str] = None):
    try:
        gdf = gpd.read_file(paths.states_shp)
        return gdf.to_crs(crs) if crs else gdf
    except Exception:
        return None


# Claims / clusters
def load_claims_by_year(paths: Paths = PATHS) -> pd.DataFrame:
    """Annual FIMA NFIP claims."""
    return pd.read_csv(paths.all_claims_by_year_csv, low_memory=False)

def load_raw_claims(paths: Paths = PATHS) -> pd.DataFrame:
    """Raw FIMA NFIP claims (used as the entry point of the cluster-update pipeline)."""
    return pd.read_csv(paths.raw_claims_csv, low_memory=False)

def load_clustered_claims(input_clustering: str = "_new", paths: Paths = PATHS):
    """Load clustered claims and return ``(df, optimal_cluster_column)``.

    Mirrors the branch that appears verbatim in Simulation/Fig_* :
      * ``'_new'`` -> ``new_clusters_9.24.25.csv``, cluster col ``st_cluster_final_gid``
      * else        -> ``clustered_claims_sensitivity.csv`` (caller's cluster col)
    """
    if input_clustering == "_new":
        df = pd.read_csv(paths.clusters_new_csv)
        optimal_cluster = "st_cluster_final_gid"
    else:
        df = pd.read_csv(paths.clusters_sensitivity_csv)
        optimal_cluster = "st_cluster_3_5_7"
    df["countyCode"] = df["countyCode"].apply(lambda x: str(x).zfill(5))
    df["stateCode"] = df["countyCode"].str[:2]
    return df, optimal_cluster


# CPI (BLS)
def load_cpi_annual(paths: Paths = PATHS) -> pd.DataFrame:
    """Annual-average CPI with columns ``['Year', 'CPI']`` (US_BLS_CPIAUCSL.csv)."""
    cpi = pd.read_csv(paths.cpi_csv, parse_dates=["DATE"])
    cpi["CPIAUCSL"] = pd.to_numeric(cpi["CPIAUCSL"], errors="coerce")
    cpi["CPIAUCSL"].fillna(method="ffill", inplace=True)
    cpi["CPIAUCSL"].fillna(method="bfill", inplace=True)
    cpi["Year"] = cpi["DATE"].dt.year
    cpi = cpi.groupby("Year")["CPIAUCSL"].mean().reset_index()
    return cpi.rename(columns={"CPIAUCSL": "CPI"})


# ENSO / Nino 3.4
def load_enso_annual(paths: Paths = PATHS) -> pd.DataFrame:
    """Annual ENSO index with nino/nina sums and normalized columns (Simulation cells 19-20)."""
    nino34 = pd.read_csv(
        paths.nino34_csv,
        skiprows=1,
        header=None,
        names=["datetime", "ANOM"],
        parse_dates=["datetime"],
    )
    nino34 = nino34.iloc[:-12]
    nino34["YR"] = nino34["datetime"].dt.year
    nino34["ANOM"] = pd.to_numeric(nino34["ANOM"], errors="coerce")

    grouped = nino34.groupby("YR")
    nino = grouped.apply(lambda g: g[g["ANOM"] > 0]["ANOM"].sum())
    nina = grouped.apply(lambda g: g[g["ANOM"] < 0]["ANOM"].sum())

    enso = pd.DataFrame({"YR": nino.index, "nino": nino.values, "nina": nina.values}).reset_index(drop=True)
    enso["nina_norm"] = enso["nina"] / enso["nina"].min()
    enso["nino_norm"] = enso["nino"] / enso["nino"].max()
    return enso


# NFIP policies / premiums
def load_premiums(premium_type: str = "", paths: Paths = PATHS) -> pd.DataFrame:
    """Aggregated risk-policy premiums, branching on ``premium_type``.

    Returns ``aggregated_risk_policies`` exactly as the notebooks built it
    (grouping keys differ by branch, matching the originals).
    """
    if premium_type == "_Full":
        rp = pd.read_excel(paths.nfip_full_xlsx, sheet_name="2024")
        rp["County Code"] = rp["County Code"].astype(int).astype(str)
        rp["County Code"] = rp["County Code"].apply(lambda x: str(x).zfill(5))
        rp["State"] = rp["State Names"].str.strip().str.upper()
        rp["Policies in Force"] = rp["Policy Count"]
        return rp.groupby(["County Code", "State"]).agg({
            "Policies in Force": "sum",
            "Total Written Premium + FPF": "sum",
            "Total Annual Payment": "sum",
        }).reset_index()

    if premium_type == "_Discount":
        rp = pd.read_csv(paths.nfip_discount_csv)
        rp["countyCode"] = rp["countyCode"].astype(int).astype(str)
        rp["countyCode"] = rp["countyCode"].apply(lambda x: str(x).zfill(5))
        rp["State"] = rp["State Name"].str.strip().str.upper()
        rp["Policies in Force"] = rp["TotalPolicies"]
        rp["Total Written Premium + FPF"] = rp["actualPremium"] + rp["federalPolicyFee"]
        return rp.groupby(["countyCode", "State"]).agg({
            "Policies in Force": "sum",
            "Total Written Premium + FPF": "sum",
            "Total Annual Payment": "sum",
        }).reset_index()

    rp = pd.read_excel(paths.nfip_pif_xlsx, sheet_name="PIF")
    rp["County"] = rp["County"].str.strip()
    rp["State"] = rp["State"].str.strip()
    return rp.groupby(["County", "State"]).agg({
        "Policies in Force": "sum",
        "Total Written Premium + FPF": "sum",
        "Total Annual Payment": "sum",
    }).reset_index()


def load_state_claims(paths: Paths = PATHS) -> pd.DataFrame:
    """Mean-by-state historical claim totals (All_Claims_by_Year.csv)."""
    sc = pd.read_csv(paths.all_claims_by_year_csv)
    sc = sc.groupby("State").aggregate(
        {"Total Claim Dollars Paid": "mean", "Total Paid Claims": "mean"}
    ).reset_index()
    sc["State"] = sc["State"].str.upper()
    return sc


# Simulation outputs
def _results_name(kind: str, test_case: str, input_clustering: str,
                  premium_type: str, base_case: str, lean: str = "") -> str:
    return f"{lean}Results/{kind}_{test_case}{input_clustering}{premium_type}{base_case}.csv"


def load_simulation_results(test_case: str, input_clustering: str = "_new",
                            premium_type: str = "", base_case: str = "", lean: str = "",
                            historic_ref: str = ""):
    """Load the four simulation output CSVs plus the historic state-balance CSV.

    Returns a dict keyed by
    ``{'state_balance', 'final_balances', 'cluster_state', 'balance_transition',
       'state_balance_hist'}`` matching the Fig_1 load block.
    """
    out = {
        "state_balance": pd.read_csv(_results_name("state_balance", test_case, input_clustering, premium_type, base_case, lean)),
        "final_balances": pd.read_csv(_results_name("final_balances", test_case, input_clustering, premium_type, base_case, lean)),
        "cluster_state": pd.read_csv(_results_name("cluster_state", test_case, input_clustering, premium_type, base_case, lean)),
        "balance_transition": pd.read_csv(_results_name("balance_transition", test_case, input_clustering, premium_type, base_case, lean)),
        "state_balance_hist": pd.read_csv(f"Results/state_balance_historic{input_clustering}{historic_ref}.csv"),
    }
    return out


# Sociodemographic
def build_sociodemographic() -> pd.DataFrame:
    """Merged GDP / median-income / poverty table (values transcribed in config)."""
    df_gdp = pd.DataFrame({"State": config._STATES_51, "GDP_2023_Million": config.GDP_2023_MILLION})
    df_income = pd.DataFrame({"State": config._STATES_51, "Median_Income_2023": config.MEDIAN_INCOME_2023})
    df_poverty = pd.DataFrame({"State": config._STATES_51, "Poverty_Rate_Percent": config.POVERTY_RATE_PERCENT})
    df_merged = df_gdp.merge(df_income, on="State").merge(df_poverty, on="State")
    return df_merged.sort_values("State").reset_index(drop=True)
