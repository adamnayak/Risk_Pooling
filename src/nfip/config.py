"""Central configuration: paths, constants, and lookup tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
#
@dataclass(frozen=True)
class Paths:
    base: Path = Path("..")

    # --- geospatial -------------------------------------------------------
    counties_shp: Path = field(init=False)
    states_shp: Path = field(init=False)
    rivers_shp: Path = field(init=False)
    urban_shp: Path = field(init=False)

    # --- BLS / climate ----------------------------------------------------
    cpi_csv: Path = field(init=False)
    nino34_csv: Path = field(init=False)

    # --- claims / clusters ------------------------------------------------
    raw_claims_csv: Path = field(init=False)
    clusters_new_csv: Path = field(init=False)
    clusters_sensitivity_csv: Path = field(init=False)

    # --- NFIP policy / premium data --------------------------------------
    nfip_full_xlsx: Path = field(init=False)
    nfip_discount_csv: Path = field(init=False)
    nfip_pif_xlsx: Path = field(init=False)
    all_claims_by_year_csv: Path = field(init=False)

    # --- outputs ----------------------------------------------------------
    results_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)

    def __post_init__(self):
        b = Path(self.base)
        object.__setattr__(self, "counties_shp", b / "Local_Data/Geospatial/tl_2019_us_county.shp")
        object.__setattr__(self, "states_shp", b / "Local_Data/Geospatial/cb_2018_us_state_20m.shp")
        object.__setattr__(self, "rivers_shp", b / "Local_Data/Rivers/rs16my07.shp")
        object.__setattr__(self, "urban_shp", b / "Local_Data/Geospatial/cb_2018_us_ua10_500k.shp")
        object.__setattr__(self, "cpi_csv", b / "Local_Data/BLS_Data/US_BLS_CPIAUCSL.csv")
        object.__setattr__(self, "nino34_csv", b / "Local_Data/Climatological_Index/nino34.long.anom.hadisst.csv")
        object.__setattr__(self, "raw_claims_csv", b / "3_Failure_Modes/FimaNfipClaims_Aug2025.csv")
        object.__setattr__(self, "clusters_new_csv", b / "2_Low_Return_Period/new_clusters_9.24.25.csv")
        object.__setattr__(self, "clusters_sensitivity_csv", b / "2_Low_Return_Period/Clusters/2025_all/clustered_claims_sensitivity.csv")
        object.__setattr__(self, "nfip_full_xlsx", b / "Local_Data/NFIP_Data/updatedNFIPdata_2025.xlsx")
        object.__setattr__(self, "nfip_discount_csv", b / "Local_Data/NFIP_Data/UpdatedNFIPData_2024_V4.csv")
        object.__setattr__(self, "nfip_pif_xlsx", b / "Local_Data/NFIP_Data/nfip_policy-information-by-state_20240531.xlsx")
        object.__setattr__(self, "all_claims_by_year_csv", b / "Local_Data/NFIP_Data/All_Claims_by_Year.csv")
        object.__setattr__(self, "results_dir", Path("Results"))
        object.__setattr__(self, "plots_dir", Path("Plots"))


# A ready-to-use default instance matching the original notebook layout.
PATHS = Paths()

# HURDAT2 Atlantic best-track file (downloaded over the network in Splitting).
HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt"

# --------------------------------------------------------------------------
# Economic / actuarial constants
# --------------------------------------------------------------------------
BASE_CPI = 313.3  # CPI-U used as the inflation reference, April 2024

# Reinsurance & insurance-linked-securities attachment structure used by the
# stepwise-coverage function in the simulation.
RE_THRES_VEC = [7_000_000_000, 9_000_000_000, 11_000_000_000]
RE_PAYOUT_VEC = [0.120334, 0.258584, 0]
ILS_THRES_VEC = [6_000_000_000, 7_000_000_000, 8_000_000_000, 9_000_000_000, 10_000_000_000, 11_000_000_000]
ILS_PAYOUT_VEC = [0.025, 0.10, 0.2625, 0.35, 0.2375, 0]

# --------------------------------------------------------------------------
# Coordinate reference systems
# --------------------------------------------------------------------------
CRS_LL = "EPSG:4326"   # lat/lon
CRS_EQD = "EPSG:5070"  # CONUS Albers equal-area

# Contiguous-US bounding box used to filter claims.
CONUS_BOUNDS = {"min_lon": -130, "max_lon": -65, "min_lat": 24, "max_lat": 50}

EARTH_KM = 6371.0088  # mean Earth radius, used for haversine / BallTree work

# --------------------------------------------------------------------------
# Named-event relabeling (Clustering / Simulation notebooks)
# --------------------------------------------------------------------------
# Maps a canonical cluster id -> the list of raw cluster ids that should be
# collapsed into it.
EVENT_RELABEL = {
    4: [4520, 1707, 1561, 1246, 2187, 1481],          # Katrina
    87: [1603, 1507, 1252],                           # Sandy
    327: [3430, 5719, 5614, 5712, 5678, 5611, 1456, 1736],  # Harvey
    166: [3907, 4050],                                # Ian
    8: [222, 1559, 1419],                             # Ike
    295: [3731, 2086, 2053, 3617, 3557],              # LA 2016 floods
    6: [159, 364, 4003, 4459, 164, 143, 434],         # Ivan
    190: [195],                                       # Helene
}

# Canonical event id -> display name.
CLUSTER_NAMES = {
    4: "Katrina", 87: "Sandy", 327: "Harvey", 166: "Ian",
    8: "Ike", 295: "LA 2016 Floods", 6: "Ivan", 190: "Helene",
}

# Storms treated as tropical (is_TS) depending on the clustering input.
HURRICANES_LEGACY = [4, 87, 327]        # input_clustering == ''
HURRICANES_NEW = [1734, 2214, 2578]     # input_clustering == '_new'  (Katrina, Sandy, Harvey)

# --------------------------------------------------------------------------
# FIPS lookup tables
# --------------------------------------------------------------------------
FIPS_TO_ABBREV = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
    '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
    '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
    '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
    '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
    '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
    '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
    '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
    '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
    '56': 'WY',
}

FIPS_TO_NAME = {
    "01": "ALABAMA", "02": "ALASKA", "04": "ARIZONA", "05": "ARKANSAS",
    "06": "CALIFORNIA", "08": "COLORADO", "09": "CONNECTICUT", "10": "DELAWARE",
    "11": "DISTRICT OF COLUMBIA", "12": "FLORIDA", "13": "GEORGIA", "15": "HAWAII",
    "16": "IDAHO", "17": "ILLINOIS", "18": "INDIANA", "19": "IOWA", "20": "KANSAS",
    "21": "KENTUCKY", "22": "LOUISIANA", "23": "MAINE", "24": "MARYLAND",
    "25": "MASSACHUSETTS", "26": "MICHIGAN", "27": "MINNESOTA", "28": "MISSISSIPPI",
    "29": "MISSOURI", "30": "MONTANA", "31": "NEBRASKA", "32": "NEVADA",
    "33": "NEW HAMPSHIRE", "34": "NEW JERSEY", "35": "NEW MEXICO", "36": "NEW YORK",
    "37": "NORTH CAROLINA", "38": "NORTH DAKOTA", "39": "OHIO", "40": "OKLAHOMA",
    "41": "OREGON", "42": "PENNSYLVANIA", "44": "RHODE ISLAND", "45": "SOUTH CAROLINA",
    "46": "SOUTH DAKOTA", "47": "TENNESSEE", "48": "TEXAS", "49": "UTAH",
    "50": "VERMONT", "51": "VIRGINIA", "53": "WASHINGTON", "54": "WEST VIRGINIA",
    "55": "WISCONSIN", "56": "WYOMING",
}

# --------------------------------------------------------------------------
# Sociodemographic tables
# --------------------------------------------------------------------------
_STATES_51 = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
    "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
]

GDP_2023_MILLION = [
    304936, 68056, 522767, 178606, 3870379, 529627, 345912, 98069,
    176502, 1600811, 831828, 110265, 120958, 1098346, 499503, 254032,
    228232, 279707, 314989, 93270, 515607, 736296, 673818, 483162,
    151147, 430114, 73255, 181285, 245979, 114101, 806665,
    135010, 2172010, 788103, 76043, 884834, 256689, 318884,
    976361, 77574, 327420, 74034, 523032, 2583866, 281329,
    43534, 719897, 807865, 102152, 428447, 51991,
]

MEDIAN_INCOME_2023 = [
    62248, 88696, 77158, 58748, 95473, 92790, 91477, 81615,
    104643, 73283, 74521, 96716, 74859, 80346, 69458, 71662,
    70316, 61099, 58273, 73463, 98568, 99750, 69097, 85070,
    54386, 68484, 70939, 74727, 76332, 97031, 99716,
    62266, 82052, 70838, 77346, 67873, 62120, 80061,
    73826, 83518, 67988, 72794, 67651, 75778, 93030,
    80626, 89864, 94553, 55875, 74671, 73558,
]

POVERTY_RATE_PERCENT = [
    11.3, 6.8, 8.9, 11.5, 8.4, 5.9, 6.8, 7.3,
    10.7, 8.9, 9.9, 6.9, 7.0, 8.2, 8.4, 6.9,
    7.7, 11.8, 14.2, 6.5, 6.3, 6.6, 8.8, 5.5,
    14.3, 8.4, 7.1, 6.7, 9.0, 4.4, 7.0,
    13.7, 9.8, 9.4, 6.2, 9.2, 11.1, 7.3,
    8.1, 7.0, 10.1, 7.4, 9.9, 10.5, 5.7,
    5.7, 6.8, 6.4, 11.9, 6.6, 7.1,
]
