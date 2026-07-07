"""HURDAT2 parsing and storm-track attribution.

Pipeline: parse the Atlantic best-track file -> space-time match claim clusters
to storm fixes -> for clusters overlapping >=2 storms, split claims between
storms with a space-time SVM.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, Point
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CRS_EQD, CRS_LL, HURDAT2_URL

# Helpers
def parse_hurdat2(text: str) -> pd.DataFrame:
    """
    Parse Atlantic HURDAT2 into tidy six-hourly records.
    Columns: ['storm_id','storm_name','iso_time','year','lat','lon','status','max_wind_kt','min_mslp_mb']
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    i = 0
    while i < len(lines):
        m = re.match(r'^([A-Z]{2}\d{6}),\s*([^,]+),\s*(\d+)', lines[i])
        if not m:
            i += 1
            continue
        storm_id, storm_name, n = m.group(1), m.group(2).strip(), int(m.group(3))
        i += 1
        for _ in range(n):
            parts = [p.strip() for p in lines[i].split(",")]
            ymd, hm = parts[0], parts[1]
            status  = parts[3] if len(parts) > 3 else ""
            lat_s   = parts[4] if len(parts) > 4 else ""
            lon_s   = parts[5] if len(parts) > 5 else ""
            wind_s  = parts[6] if len(parts) > 6 else ""
            mslp_s  = parts[7] if len(parts) > 7 else ""

            t = datetime.strptime(ymd + hm, "%Y%m%d%H%M")

            def parse_ll(s, neg=("S","W")):
                s = s.upper()
                if s and s[-1] in ("N","S","E","W"):
                    sign = -1 if s[-1] in neg else 1
                    return sign * float(s[:-1])
                return float(s) if s else None

            def to_int(x):
                try: return int(x)
                except: return None

            rows.append({
                "storm_id":     storm_id,
                "storm_name":   storm_name,
                "iso_time":     t,
                "year":         t.year,
                "lat":          parse_ll(lat_s),
                "lon":          parse_ll(lon_s),
                "status":       status,       # TD/TS/HU/SS/SD/EX/LO/DB, etc.
                "max_wind_kt":  to_int(wind_s),
                "min_mslp_mb":  to_int(mslp_s),
            })
            i += 1
    df = pd.DataFrame(rows)
    # Keep only fixes with coords/time
    return df.dropna(subset=["iso_time","lat","lon"]).reset_index(drop=True)

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance (km)."""
    R = 6371.0088
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2.0)**2)
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

#  Helper: build “closest-track” pseudo-labels for a cluster
def seed_labels_by_closest_fix(msub, speed_kmh=30.0):
    """
    msub: matches for ONE cluster_id (each row is a (claim x fix) pair)
          must have: ['dateOfLoss','claim_lat','claim_lon','storm_id',
                      'time_delta_hours','distance_km']
    """
    s = msub.copy()
    # Build a per-claim key
    s["claim_key"] = list(zip(
        s["dateOfLoss"].values,
        s["claim_lat"].round(5).values,
        s["claim_lon"].round(5).values
    ))
    # Space–time score in km (time converted via speed)
    s["score"] = s["distance_km"] + np.abs(s["time_delta_hours"]) * speed_kmh
    idx = s.groupby("claim_key")["score"].idxmin()
    best = s.loc[idx, ["claim_key","storm_id"]].set_index("claim_key")["storm_id"]
    return best

#  Helper: make features (x,y,t) in projected coords + scaled time
def make_features(claims_sub, speed_kmh=30.0):
    gdf = gpd.GeoDataFrame(
        claims_sub,
        geometry=gpd.points_from_xy(claims_sub["longitude"], claims_sub["latitude"]),
        crs=CRS_LL
    ).to_crs(CRS_EQD)

    x_km = gdf.geometry.x.values / 1000.0
    y_km = gdf.geometry.y.values / 1000.0
    t0 = claims_sub["dateOfLoss"].median()
    t_h = (claims_sub["dateOfLoss"] - t0).dt.total_seconds().values / 3600.0
    t_km = t_h * speed_kmh

    X = pd.DataFrame({"x_km": x_km, "y_km": y_km, "t_km": t_km})
    return X, gdf

def load_hurdat(url: str = HURDAT2_URL) -> pd.DataFrame:
    """Download and parse the HURDAT2 best-track file (Splitting cell 12 source)."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return parse_hurdat2(resp.text)


def build_tracks_table(hurdat: pd.DataFrame) -> pd.DataFrame:
    """Tidy per-fix track table with a floored date, used for matching."""
    tracks = hurdat[["storm_id", "storm_name", "iso_time", "year", "lat", "lon"]].dropna().copy()
    tracks["date_floor"] = pd.to_datetime(tracks["iso_time"]).dt.floor("D")
    return tracks

def match_claims_to_tracks(claims: pd.DataFrame, tracks: pd.DataFrame,
                           r_km: float, t_hours: float) -> pd.DataFrame:
    """Space-time match: claims within ``r_km`` and ``+/- t_hours`` of a fix.

    ``claims`` must have columns
    ['cluster_id','dateOfLoss','claim_lat','claim_lon','date_floor'].
    """
    matches = []
    for day_offset in range(-int(t_hours // 24) - 1, int(t_hours // 24) + 2):
        ctmp = claims.copy()
        ctmp["date_shifted"] = ctmp["date_floor"] + pd.Timedelta(days=day_offset)
        merged = tracks.merge(
            ctmp, left_on="date_floor", right_on="date_shifted",
            how="inner", suffixes=("", "_c"),
        )
        if merged.empty:
            continue

        dt_hours = (merged["iso_time"] - merged["dateOfLoss"]).abs().dt.total_seconds() / 3600.0
        dist_km = haversine_km(merged["lat"], merged["lon"], merged["claim_lat"], merged["claim_lon"])

        keep = (dt_hours <= t_hours) & (dist_km <= r_km)
        if keep.any():
            m = merged.loc[keep, [
                "storm_id", "storm_name", "iso_time", "year", "lat", "lon",
                "cluster_id", "dateOfLoss", "claim_lat", "claim_lon",
            ]].copy()
            m["time_delta_hours"] = dt_hours[keep].values
            m["distance_km"] = dist_km[keep].values
            matches.append(m)

    matches = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(
        columns=["storm_id", "storm_name", "iso_time", "year", "lat", "lon",
                 "cluster_id", "dateOfLoss", "claim_lat", "claim_lon",
                 "time_delta_hours", "distance_km"]
    )
    matches.drop_duplicates(
        subset=["storm_id", "iso_time", "cluster_id", "dateOfLoss", "claim_lat", "claim_lon"],
        inplace=True,
    )
    return matches


def find_multi_track_clusters(matches: pd.DataFrame, exclude_cluster=-1):
    """Return (valid_matches, list_of_cluster_ids_with_>=2_storms)."""
    valid_matches = matches[matches["cluster_id"] != exclude_cluster].copy()
    tracks_per_cluster = (valid_matches.groupby("cluster_id")["storm_id"]
                          .nunique().reset_index(name="n_tracks"))
    multi = tracks_per_cluster.loc[tracks_per_cluster["n_tracks"] >= 2, "cluster_id"].tolist()
    return valid_matches, multi


def build_track_lines(hurdat: pd.DataFrame):
    """Per-storm LineString GeoDataFrame in CRS_EQD (Splitting cell 18)."""
    hurdat_ll = gpd.GeoDataFrame(
        hurdat[["storm_id", "storm_name", "iso_time", "lat", "lon"]].dropna(),
        geometry=gpd.points_from_xy(hurdat["lon"], hurdat["lat"]),
        crs=CRS_LL,
    )
    if getattr(hurdat_ll, "crs", None) is None:
        hurdat_ll = hurdat_ll.set_crs(CRS_LL, allow_override=True)

    lines_df = (
        hurdat_ll.sort_values(["storm_id", "iso_time"])
        .groupby("storm_id")
        .agg(geometry=("geometry", lambda s: LineString([pt for pt in s.tolist()]) if len(s) > 1 else None))
        .dropna(subset=["geometry"])
        .reset_index()
    )
    gdf_tracks = gpd.GeoDataFrame(lines_df, geometry="geometry", crs=hurdat_ll.crs).to_crs(CRS_EQD)
    return hurdat_ll, gdf_tracks

def split_multitrack_clusters(clustered_claims, optimal_cluster, valid_matches,
                              multi_track_clusters, hurdat_ll, gdf_tracks,
                              gdf_counties=None, gdf_states=None,
                              point_size=10, plot=True):
    """Per-cluster space-time SVM split (verbatim body of Splitting cell 20).

    Returns ``assignments`` (list of per-cluster DataFrames).  ``multi_track_clusters``
    here is the list of cluster ids to split (from ``find_multi_track_clusters``).
    """
    POINT_SIZE = point_size
    assignments = []
    if "id" not in clustered_claims.columns:
        clustered_claims["id"] = np.arange(len(clustered_claims), dtype=np.int64)

    # Per-cluster SVM split & plotting
    # Storage for final assignments
    assignments = []
    if "id" not in clustered_claims.columns:
        clustered_claims["id"] = np.arange(len(clustered_claims), dtype=np.int64)

    for cid in multi_track_clusters:
        claims_sub = clustered_claims.loc[
            clustered_claims[optimal_cluster] == cid,
            ["id", optimal_cluster, "dateOfLoss", "latitude", "longitude"]
        ].dropna().copy()

        if claims_sub.empty:
            continue

        # Candidate matches for this cluster
        msub = valid_matches.loc[valid_matches["cluster_id"] == cid].copy()
        cand_storms = sorted(msub["storm_id"].unique().tolist())
        if len(cand_storms) < 2:
            continue

        # Seed labels from nearest fix (space–time score)
        seeds = seed_labels_by_closest_fix(msub)

        # Build features and align with seeds
        # Create stable claim keys to join seeds back
        claims_sub["claim_key"] = list(zip(claims_sub["dateOfLoss"].values,
                                           claims_sub["latitude"].round(5).values,
                                           claims_sub["longitude"].round(5).values))
        y_seed = claims_sub["claim_key"].map(seeds)  # storm_id or NaN

        # Inside the per-cluster loop, replacing your fallback
        if y_seed.isna().any():
            tmin = claims_sub["dateOfLoss"].min() - pd.Timedelta(days=5)
            tmax = claims_sub["dateOfLoss"].max() + pd.Timedelta(days=5)
            fixes = hurdat_ll.loc[
                (hurdat_ll["storm_id"].isin(cand_storms)) &
                (hurdat_ll["iso_time"].between(tmin, tmax))
            ].copy()
    
            # Project once for distances
            fixes_proj = fixes.to_crs(CRS_EQD)
            storm_fix_groups = {sid: df for sid, df in fixes_proj.groupby("storm_id")}
            storm_time_groups = {sid: fixes.loc[fixes["storm_id"]==sid, "iso_time"].values
                                 for sid in cand_storms}
    
            SPEED_KMH = 30.0  # tune 25–40
    
            def nearest_storm_spacetime(row):
                p = gpd.GeoSeries([Point(row["longitude"], row["latitude"])], crs=CRS_LL)\
                      .to_crs(CRS_EQD).iloc[0]
                t = row["dateOfLoss"].to_datetime64()
    
                best_sid, best_score = None, np.inf
                for sid in cand_storms:
                    fp = storm_fix_groups.get(sid)
                    if fp is None or fp.empty:
                        continue
                    # Find the fix closest in time to the claim
                    times = storm_time_groups[sid]
                    i = np.argmin(np.abs(times - t))
                    # Spatial distance to that fix
                    d_space = fp.geometry.iloc[i].distance(p) / 1000.0  # meters -> km
                    # Time penalty in km
                    dt_h = abs((pd.Timestamp(times[i]).to_pydatetime() - row["dateOfLoss"]).total_seconds())/3600.0
                    score = d_space + dt_h * SPEED_KMH
                    if score < best_score:
                        best_score, best_sid = score, sid
                return best_sid
    
            y_seed = y_seed.fillna(claims_sub.apply(nearest_storm_spacetime, axis=1))

        X, gdf_claims_proj = make_features(claims_sub, speed_kmh=30.0)
        feat_cols = ["x_km","y_km","t_km"]
    
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=False),  # center but DON'T rescale away weights
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)
        )
        clf.fit(X[feat_cols], y_seed)

        # Predict assignments + confidence
        y_pred = clf.predict(X)
        proba = clf.predict_proba(X).max(axis=1)

        claims_sub["svm_storm_id"] = y_pred
        claims_sub["svm_confidence"] = proba
        assignments.append(
            claims_sub[["id", optimal_cluster, "dateOfLoss", "latitude", "longitude"]]
                .assign(svm_storm_id=y_pred, svm_confidence=proba)
                .copy()
        )

        # Plot the outcome for this cluster
        # Build a per-storm color map
        uniq_sids = sorted(pd.unique(y_pred))
        cmap = plt.get_cmap("tab10")
        colors = {sid: cmap(i % 10) for i, sid in enumerate(uniq_sids)}

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

        # Basemaps
        if gdf_counties is not None:
            gdf_counties.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.2, alpha=0.6)
        if gdf_states is not None:
            gdf_states.boundary.plot(ax=ax, color='black', linewidth=0.4)

        # Plot track lines for candidate storms
        gdf_tracks.loc[gdf_tracks["storm_id"].isin(cand_storms)].plot(
            ax=ax, linewidth=2.0, alpha=0.8,
            color=gdf_tracks["storm_id"].map(lambda sid: colors.get(sid, "k"))
        )

        # Plot claims colored by predicted storm
        for sid in uniq_sids:
            pts = gdf_claims_proj.loc[claims_sub["svm_storm_id"] == sid]
            pts.plot(ax=ax, markersize=POINT_SIZE, alpha = 0.3, color=colors[sid], label=f"{sid}")

        # Extent: pad around points
        if not gdf_claims_proj.empty:
            xmin, ymin, xmax, ymax = gdf_claims_proj.total_bounds
            dx = (xmax - xmin) * 0.2 if xmax > xmin else 1e5
            dy = (ymax - ymin) * 0.2 if ymax > ymin else 1e5
            ax.set_xlim(xmin - dx, xmax + dx)
            ax.set_ylim(ymin - dy, ymax + dy)

        ax.set_title(f"Cluster {cid}: SVM split by storm (points) + tracks (lines)")
        ax.axis("off")
        ax.legend(title="storm_id", loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.show()
    return assignments
