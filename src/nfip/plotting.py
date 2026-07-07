"""Shared plotting helpers used across the figure notebooks.
"""

from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import get_cmap

def average_ratio(df, flag_col, flag_val, label):
    return (
        df[df[flag_col] == flag_val]
        .groupby("STATEFP")["contribution_ratio"]
        .mean()
        .reset_index()
        .rename(columns={"contribution_ratio": label})
    )

def pooled_drawdown(row):
    contrib, bal = row['contribution'], row['nfip_balance']
    if contrib >= 0:
        return 0
    elif bal >= 0:
        return abs(min(0, bal + contrib))
    else:
        return abs(contrib)

def state_drawdown(row):
    contrib, bal = row['contribution'], row['nfip_balance']
    if contrib >= 0:
        return 0
    elif bal >= 0:
        return abs(min(0, bal + contrib))
    else:
        return abs(contrib)

def prep_state_stats(df, selected_states, state_abbrev):
    """
    Normalize STATEFP, compute cumulative balances per simulation/state, map to abbreviations,
    filter selected states, and aggregate stats across simulations by (State, year).
    Returns a DataFrame with columns: State, year, min, max, mean, q25, q75
    """
    df = df.copy()

    # Normalize STATEFP to zero-padded 2-char strings
    df["STATEFP"] = (
        df["STATEFP"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(2)
    )

    # cumulative balance per simulation & state
    df["cumulative_balance"] = (
        df.groupby(["simulation", "STATEFP"])["contribution"]
          .cumsum()
    )

    # Add abbreviation and filter
    df["State"] = df["STATEFP"].map(state_abbrev)
    df = df[df["State"].isin(selected_states)].copy()

    # Aggregate across simulations: min, max, mean, IQR
    stats = (
        df.groupby(["State", "year"])["cumulative_balance"]
          .agg(
              min="min",
              max="max",
              mean="mean",
              q25=lambda x: x.quantile(0.25),
              q75=lambda x: x.quantile(0.75),
          )
          .reset_index()
    )

    # Sort for clean plotting
    stats = stats.sort_values(["State", "year"])
    return stats

def grouped_bar(tidy_df, x, y, hue, title, ylabel, order=None, hue_order=None, rotate_xticks=True, height=4, width=10):
    plt.figure(figsize=(width, height))
    ax = sns.barplot(
        data=tidy_df,
        x=x, y=y, hue=hue,
        order=order, hue_order=hue_order,
        palette=PALETTE, alpha=BAR_ALPHA
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    if rotate_xticks:
        plt.xticks(rotation=90)
    ax.legend(title="")
    plt.tight_layout()
    plt.show()

def fmt_musd(x, _pos):
    return f"${x:,.0f}M"

def fmt_ratio(x, _pos):
    return f"{x:.2f}"

def barplot(ax, data, x, y, hue=None, order=None, hue_order=None, title="", ylabel="", rotate=False):
    if hue is None:
        sns.barplot(ax=ax, data=data, x=x, y=y, color=PALETTE.get(data[x].iloc[0], "#7f7f7f") if x=="Scenario" and len(data[x].unique())==1 else "#7f7f7f", alpha=BAR_ALPHA)
    else:
        sns.barplot(ax=ax, data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order, palette=PALETTE, alpha=BAR_ALPHA)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    if rotate:
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
            tick.set_ha("center")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    return ax

def add_panel_labels(axs, labels, x=-0.02, y=1.12):
    for ax, lab in zip(axs, labels):
        ax.text(x, y, lab, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")

def custom_stackplot(ax, df, title, ylabel, xlabel, label_text, label_pos, colors_dict):
    if df is None or df.empty or df.shape[1] == 0:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(label_pos[0], label_pos[1], label_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=8, fontweight='bold')
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center', fontsize=11)
        return

    # Use the columns of THIS df, not a global list
    cols = list(df.columns)

    # Build a color list that matches the number of series exactly
    cmap_local = cm.get_cmap("Spectral", len(cols))
    if colors_dict is None:
        color_list = [cmap_local(i) for i in range(len(cols))]
    else:
        color_list = [colors_dict.get(c, cmap_local(i)) for i, c in enumerate(cols)]

    x = df.index
    y = [df[c].values for c in cols]

    # Unpack series with *y
    ax.stackplot(x, *y, labels=cols, colors=color_list, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(label_pos[0], label_pos[1], label_text, transform=ax.transAxes,
            ha='left', va='top', fontsize=8, fontweight='bold')

def _norm_fips_list(lst):
    # Handles strings like '9', '05', ints, etc.
    return [str(int(x)).zfill(2) for x in lst]

def plot_choropleth(gdf, column, title, cmap='viridis'):
    fig, ax = plt.subplots(figsize=(12, 8))
    gdf.plot(
        column=column,
        cmap=cmap,
        legend=True,
        edgecolor="black",
        linewidth=0.4,
        ax=ax
    )
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    # Bounding box for CONUS
    xlim = [-130, -65]  # Longitude
    ylim = [24, 50]     # Latitude
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()
