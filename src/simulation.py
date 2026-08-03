"""NFIP pool simulation.

Year-resampled (historic / block-bootstrap / random) balance-sheet simulation
with optional reinsurance + ILS stepwise coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def apply_stepwise_coverage(loss, thresholds, payout_rates):
    covered = 0
    previous_thresh = 0
    
    for thresh, rate in zip(thresholds, payout_rates):
        if loss <= thresh:
            covered += (loss - previous_thresh) * rate
            break
        else:
            covered += (thresh - previous_thresh) * rate
            previous_thresh = thresh

    return covered


def run_simulation(clustered_claims, gdf_states, optimal_cluster,
                   test_case="random", steps=100, block_size=10, simulations=1000,
                   use_reinsurance=False,
                   re_thres_vec=None, re_payout_vec=None,
                   ils_thres_vec=None, ils_payout_vec=None,
                   pls_print=False):
    """Run the pool simulation and return the four output DataFrames.

    Returns dict with keys: cluster_state_df, state_balance_df,
    final_balances_df, balance_transition_df.

    ``clustered_claims`` must already be CPI-adjusted and carry ``is_TS`` and a
    ``year`` column; ``gdf_states`` must be the built balance sheet (with
    'Total Written Premium + FPF').  Defaults for the reinsurance/ILS vectors
    come from :mod:`nfip.config`.
    """
    if re_thres_vec is None:
        re_thres_vec = config.RE_THRES_VEC
    if re_payout_vec is None:
        re_payout_vec = config.RE_PAYOUT_VEC
    if ils_thres_vec is None:
        ils_thres_vec = config.ILS_THRES_VEC
    if ils_payout_vec is None:
        ils_payout_vec = config.ILS_PAYOUT_VEC

    # Initialization
    years_arr = np.array(sorted(clustered_claims["dateOfLoss"].dt.year.unique()), dtype=int)
    min_year, max_year = years_arr.min(), years_arr.max()

    # For block bootstrap, keep the horizon equal to the span you're resampling over
    steps = len(years_arr) if test_case == "historic" else steps

    # Global containers
    cluster_state_output = []     # Cluster-level payouts per state-year
    state_balance_output = []     # State-level financial summaries
    final_balance_sheets = []     # End-of-simulation state balances

    # Initialize transition tracker and sign memory
    balance_transitions = []
    previous_positive = {}  # Keyed by (simulation, state): True/False

    for n in range(simulations):
        print(f'\nSimulation {n + 1}/{simulations}')

        # Year sampling
        if test_case == "random":
            sampled_years = np.random.choice(years_arr, size=steps, replace=True).tolist()
            synthetic_years = list(range(min_year, min_year + steps))

        elif test_case == "block":
            n_blocks = int(np.ceil(steps / block_size))
        
            # Get all years in historical record
            years = sorted(clustered_claims["dateOfLoss"].dt.year.unique())
            min_year, max_year = min(years), max(years)
        
            # All valid block start years
            start_years = list(range(min_year, max_year - block_size + 2))
        
            # Sample block start years with replacement
            sampled_starts = np.random.choice(start_years, size=n_blocks, replace=True)
        
            # Build sampled (historical) years from the blocks
            sampled_years = []
            for start in sampled_starts:
                block = list(range(start, start + block_size))
                sampled_years.extend(block)
        
            # Truncate to exact desired length
            sampled_years = sampled_years[:steps]
        
            # Synthetic timeline (same length as steps, starting from 1978 or other baseline)
            synthetic_years = list(range(min_year, min_year + steps))

        else:
            sampled_years = years_arr.tolist()
            synthetic_years = years_arr.tolist()

        # Initialize balance sheet for this simulation
        balance_sheet = gdf_states.copy()
        balance_sheet["STATEFP"] = balance_sheet["STATEFP"].astype(str)
        balance_sheet["contribution"] = 0
        balance_sheet["nfip_balance"] = 0

        for i in range(steps):
            sample_year = sampled_years[i]
            display_year = synthetic_years[i]

            if pls_print:
                print(f'  Processing year {sample_year}')

            # Subset claims for current year
            year_claims = clustered_claims[clustered_claims["year"] == sample_year]

            # Compute state-cluster claims
            state_cluster_damage = (
                year_claims
                .groupby(["stateCode", optimal_cluster])["adjustedClaim"]
                .sum()
                .reset_index()
            )
            state_cluster_damage["stateCode"] = state_cluster_damage["stateCode"].astype(str)

            # Merge with base balance sheet
            temp = balance_sheet.merge(
                state_cluster_damage,
                how="left",
                left_on="STATEFP",
                right_on="stateCode"
            )
            temp["adjustedClaim"] = temp["adjustedClaim"].fillna(0)

            # Initialize reinsurance and ILS columns
            temp["re_covered"] = 0.0
            temp["ils_covered"] = 0.0
            temp["total_covered"] = 0.0

            if use_reinsurance:
                # Compute total loss per cluster and find high-loss clusters
                cluster_damage = (
                    year_claims
                    .groupby(optimal_cluster)["adjustedClaim"]
                    .sum()
                    .reset_index()
                    .merge(
                        clustered_claims[[optimal_cluster, "is_TS"]].drop_duplicates(),
                        on=optimal_cluster,
                        how="left"
                    )
                )
                min_thresh = min(re_thres_vec[0], ils_thres_vec[0])
                high_loss_clusters = cluster_damage[cluster_damage["adjustedClaim"] > min_thresh]
    
                # Allocate coverage only to affected state-cluster pairs
                for _, row in high_loss_clusters.iterrows():
                    cluster = row[optimal_cluster]
                    total_loss = row["adjustedClaim"]
                    is_ts = row["is_TS"]
    
                    # State-wise breakdown of this cluster
                    state_breakdown = (
                        year_claims[year_claims[optimal_cluster] == cluster]
                        .groupby("stateCode")["adjustedClaim"]
                        .sum()
                        .reset_index()
                    )
                    state_breakdown["proportion"] = state_breakdown["adjustedClaim"] / total_loss
    
                    re_covered = apply_stepwise_coverage(total_loss, re_thres_vec, re_payout_vec)
                    ils_covered = apply_stepwise_coverage(total_loss, ils_thres_vec, ils_payout_vec) if is_ts else 0
                    total_cluster_covered = re_covered + ils_covered
    
                    # In-place update of temp for each state in the cluster
                    for _, s_row in state_breakdown.iterrows():
                        state = s_row["stateCode"]
                        prop = s_row["proportion"]
    
                        mask = (temp["STATEFP"] == state) & (temp[optimal_cluster] == cluster)
                        temp.loc[mask, "re_covered"] += prop * re_covered
                        temp.loc[mask, "ils_covered"] += prop * ils_covered
                        temp.loc[mask, "total_covered"] += prop * total_cluster_covered

            # Annotate metadata
            temp["year"] = display_year
            temp["simulation"] = n

            # Store cluster-level results
            cluster_state_output.append(
                temp[[
                    "STATEFP", "stateCode", optimal_cluster, "adjustedClaim",
                    "re_covered", "ils_covered", "total_covered",
                    "year", "simulation"
                ]]
            )

            # State-level financial tracking
            state_summary = (
                temp.groupby("STATEFP")[["adjustedClaim", "re_covered", "ils_covered", "total_covered"]]
                .sum()
                .reset_index()
            )

            # Add premiums and calculate contributions
            state_summary = state_summary.merge(
                balance_sheet[["STATEFP", "Total Written Premium + FPF"]],
                on="STATEFP", how="left"
            ).rename(columns={"Total Written Premium + FPF": "premium"}).fillna({"premium": 0})

            state_summary["nfip_payout"] = state_summary["adjustedClaim"] - state_summary["total_covered"]
            state_summary["contribution"] = state_summary["premium"] - state_summary["nfip_payout"]
            state_summary["year"] = display_year
            state_summary["simulation"] = n
            state_summary = state_summary.merge(balance_sheet[["STATEFP", "nfip_balance"]], on="STATEFP", how="left")

            # Merge contributions, ensuring all states retained
            balance_sheet = balance_sheet.merge(
                state_summary[["STATEFP", "contribution"]],
                on="STATEFP", how="left",
                suffixes=('', '_new')
            )
        
            # Safely fill and update contribution column
            if "contribution_new" in balance_sheet.columns:
                balance_sheet["contribution"] = balance_sheet["contribution_new"].fillna(0.0)
                balance_sheet = balance_sheet.drop(columns=["contribution_new"])
            else:
                balance_sheet["contribution"] = 0.0
        
            # Update running balance
            balance_sheet["nfip_balance"] += balance_sheet["contribution"]

            # Track balance transitions from positive to negative
            for _, row in balance_sheet.iterrows():
                state = row["STATEFP"]
                sim_key = (n, state)
                current_balance = row["nfip_balance"]
                was_positive = previous_positive.get(sim_key, True)  # Default to True at start
        
                # Check for transition from positive to negative
                if was_positive and current_balance < 0:
                    # Find highest-loss cluster for this state-year
                    state_clusters = temp[temp["STATEFP"] == state]
                    if not state_clusters.empty:
                        max_row = state_clusters.loc[state_clusters["adjustedClaim"].idxmax()]
                        balance_transitions.append({
                            "simulation": n,
                            "year": display_year,
                            "STATEFP": state,
                            "top_cluster": max_row[optimal_cluster],
                            "cluster_loss": max_row["adjustedClaim"]
                        })
        
                # Update current balance sign
                previous_positive[sim_key] = current_balance >= 0

            # Store state-year summary
            state_balance_output.append(state_summary)

        # Store final balance sheet for this simulation
        balance_sheet["simulation"] = n
        final_balance_sheets.append(balance_sheet.copy())

    cluster_state_df = pd.concat(cluster_state_output, ignore_index=True)
    state_balance_df = pd.concat(state_balance_output, ignore_index=True)
    final_balances_df = pd.concat(final_balance_sheets, ignore_index=True)
    balance_transition_df = pd.DataFrame(balance_transitions)
    return {
        "cluster_state_df": cluster_state_df,
        "state_balance_df": state_balance_df,
        "final_balances_df": final_balances_df,
        "balance_transition_df": balance_transition_df,
    }
