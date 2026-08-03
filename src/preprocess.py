"""Claim shaping and derived quantities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .config import BASE_CPI


# Cluster-update entry preprocessing (2025_Cluster_Update cell 20)
def preprocess_raw_claims(claims: pd.DataFrame, damage_only: bool = True) -> pd.DataFrame:
    """Clean raw FIMA claims and add ``daysSinceStart``, ``date``, ``index``.
    """
    claims = claims.dropna(subset=["dateOfLoss"])
    claims = claims.dropna(subset=["latitude", "longitude"])
    claims = claims[(claims["latitude"] != "") & (claims["longitude"] != "")]

    if damage_only:
        claims = claims[claims["buildingDamageAmount"] >= 1]

    claims["dateOfLoss"] = pd.to_datetime(claims["dateOfLoss"]).dt.tz_localize(None)
    origin_date = claims["dateOfLoss"].min()
    claims["daysSinceStart"] = (claims["dateOfLoss"] - origin_date).dt.days
    claims["date"] = pd.to_datetime(claims["daysSinceStart"], unit="D", origin=origin_date)

    claims["index"] = claims.index
    claims = claims.dropna(subset=["countyCode"])
    claims["countyCode"] = claims["countyCode"].astype(int).astype(str)
    claims["countyCode"] = claims["countyCode"].apply(lambda x: str(x).zfill(5))

    b = config.CONUS_BOUNDS
    claims = claims.dropna(subset=["latitude", "longitude"]).loc[
        (claims["longitude"] >= b["min_lon"]) &
        (claims["longitude"] <= b["max_lon"]) &
        (claims["latitude"] >= b["min_lat"]) &
        (claims["latitude"] <= b["max_lat"])
    ]
    return claims


# Derived claim fields (Clustering / Simulation / Fig_* )
def add_claim_fields(claims: pd.DataFrame) -> pd.DataFrame:
    """Add ``percentageBuildingDamageAmount`` + ``totalClaimPaid``; drop state-owned."""
    claims = claims.copy()
    claims["percentageBuildingDamageAmount"] = (
        (claims["buildingDamageAmount"] / claims["buildingPropertyValue"]) * 100
    ).clip(upper=100)
    claims["totalClaimPaid"] = (
        claims["amountPaidOnBuildingClaim"].fillna(0)
        + claims["amountPaidOnContentsClaim"].fillna(0)
    )
    claims = claims[claims["stateOwnedIndicator"] != True]  # noqa: E712 (matches original)
    return claims


def cpi_adjust_claims(claims: pd.DataFrame, cpi_annual: pd.DataFrame,
                      base_cpi: float = BASE_CPI) -> pd.DataFrame:
    """Merge annual CPI and add ``adjustedClaim`` (= base_cpi / CPI * totalClaimPaid).

    Returns the merged frame (the notebooks reassign ``clustered_claims`` to it).
    """
    claims = claims.copy()
    claims["dateOfLoss"] = pd.to_datetime(claims["dateOfLoss"])
    claims.loc[:, "yearOfLoss"] = claims["dateOfLoss"].dt.year
    merged = pd.merge(claims, cpi_annual, left_on=["yearOfLoss"], right_on=["Year"])
    merged["adjustedClaim"] = (base_cpi / merged["CPI"]) * merged["totalClaimPaid"]
    merged = merged.drop(index=1428782) # longitude error in NJ
    return merged


def relabel_events(claims: pd.DataFrame, optimal_cluster: str,
                   relabel: dict = None) -> pd.DataFrame:
    """Collapse raw cluster ids into canonical named-event ids (legacy input only)."""
    if relabel is None:
        relabel = config.EVENT_RELABEL
    claims = claims.copy()
    for canonical, members in relabel.items():
        members_set = set(members)
        claims[optimal_cluster] = claims[optimal_cluster].apply(
            lambda x: canonical if x in members_set else x
        )
    return claims


def annotate_is_ts(claims: pd.DataFrame, optimal_cluster: str,
                   input_clustering: str = '_new') -> pd.DataFrame:
    """Add boolean ``is_TS`` flag and (for '_new') copy ``Year`` -> ``year``."""
    claims = claims.copy()
    hurricanes = config.COVERED_NEW
    claims["year"] = claims["Year"]
    claims["is_TS"] = claims[optimal_cluster].isin(hurricanes)
    return claims


# State balance sheet (Simulation cells 23-26)
def build_state_balance_sheet(gdf_states, aggregated_risk_policies: pd.DataFrame,
                              state_claims: pd.DataFrame):
    """Merge premiums/claims onto the states GeoDataFrame and derive pool metrics.

    Adds Contribution %, Benefit/OG_Benefit, Withdrawal thresholds and drops
    Alaska/Hawaii, exactly as the simulation notebook does before the main loop.
    """
    state_df = pd.DataFrame(list(config.FIPS_TO_NAME.items()), columns=["State FIPS", "State Name"])

    sc = state_claims.merge(state_df, left_on="State", right_on="State Name", how="left")
    state_policies = aggregated_risk_policies.groupby("State").aggregate({
        "Policies in Force": "sum",
        "Total Written Premium + FPF": "sum",
        "Total Annual Payment": "sum",
    }).reset_index()
    state_merged = sc.merge(state_policies, left_on="State", right_on="State", how="left")
    state_merged["meanLoss"] = (
        state_merged["Total Written Premium + FPF"] - state_merged["Total Claim Dollars Paid"]
    )

    gdf_states = gdf_states.merge(state_merged, left_on="GEOID", right_on="State FIPS", how="left")

    total_premium = gdf_states["Total Written Premium + FPF"].sum()
    gdf_states["Contribution %"] = 100 * gdf_states["Total Written Premium + FPF"] / total_premium
    gdf_states["Benefit"] = 100 / gdf_states["Contribution %"]
    gdf_states["OG_Benefit"] = gdf_states["Benefit"]
    og_gain_balance = np.min(gdf_states["Benefit"])
    gdf_states["Withdrawal_threshold"] = gdf_states["Total Written Premium + FPF"] * og_gain_balance
    gdf_states["OG_Withdrawal_threshold"] = gdf_states["Withdrawal_threshold"]

    gdf_states = gdf_states[~gdf_states["State"].isin(["ALASKA", "HAWAII"])]
    return gdf_states, total_premium
