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
# Named-event labeling (Clustering / Simulation notebooks)
# --------------------------------------------------------------------------

CLUSTER_EVENT_MAP = {
    1734: "Katrina\n2005",
    2214: "Sandy\n2012",
    2578: "Harvey\n2017",
    3042: "Milton/Helene\n2024",
    2939: "Ian\n2022",
    1923: "Gustav/Ike\n2008",
    2506: "LA Floods\n2016",
    1676: "Frances/Ivan/Jeanne\n2004",
    2138: "Irene/Lee\n2011",
    1455: "ALLISON 2001",
    2590: "IRMA",
    1060: "1995 MISSISSIPPI RIVER FLOODS",
    1355: "Floyd/Irene\n1999",
    2653: "FLORENCE",
    747: "HUGO",
    2475: "2016 NORTH AMERICAN STORM COMPLEX",
    1074: "OPAL",
    2519: "MATTHEW",
    1289: "GEORGE/MITCH",
    1614: "ISABEL",
    2865: "FRED/IDA/NICHOLAS",
    2731: "BARRY",
    923: "DEC 1992 NOR'EASTER",
    2202: "ISAAC",
    2871: "IDA",
    2410: "TX/OK Floods\n& Tornadoes 2015",
    107: "CLAUDETTE",
    941: "The Great Flood\n1993",
    1742: "WILMA",
    61: "1979 EASTER FLOOD + STORM?",
    1033: "ROSA",
    1895: "Midwest Floods\n2008",
    1144: "FRAN",
    1545: "HANNA/ISIDORE/LILI",
    119: "DAVID/FREDERIC",
    940: "1993 SUPERSTORM",
    2980: "IDALIA",
    1773: "2006 MID-ATLANTIC US FLOOD",
    1101: "North American\nBlizzard 1996",
    2040: "TN Floods\n2010",
    2471: "2016 LOUISIANA HEAVY RAINS",
    2112: "MS River Floods\n2011",
    144: "1980 LOUISIANA HEAVY RAINS",
    1828: "APRIL 2007 NOR'EASTER",
    412: "ALICIA",
    2808: "SALLY",
    858: "1991 PERFECT STORM",
    393: "1983 LOWER MISSISSIPPI FLOODS",
    1171: "Red River Flood\n1997",
    1168: "1997 SPRING FLOOD",
    909: "ANDREW",
    1150: "JOSEPHINE",
    2687: "Arkansas River Floods\n2019",
    2666: "MICHAEL",
    2834: "MAY 2021 SOUTH CENTRAL FLOODS",
    2034: "MAR 2010 NOR'EASTER",
    1417: "GORDON/LESLIE",
    544: "JUAN",
    1956: "2009 SEVERE STORMS",
    2956: "2023 FT LAUDERDALE FLOODS",
    534: "ELENA",
    2324: "2014 FLASH FLOOD",
    1716: "DENNIS",
    2444: "2015 NORTH AMERICAN STORM COMPLEX",
    15: "1978 NEW ORLEANS FLASH FLOOD",
    822: "1991 LA/MS FLOODS",
    2003: "2009 CATASTROPHIC ATLANTA FLOOD",
    1232: "WINTER 1998 BLIZZARD",
    1727: "KATRINA - FL",
    1927: "IKE - MIDWEST",
    718: "1989 TX/LA FLASH FLOODS",
    387: "1982/83 MISSISSIPPI RIVER FLOODS 2",
    2238: "2013 MIDWESTERN FLOODS",
    1706: "2005 DELAWARE RIVER FLOODS",
    0: "1978 BLIZZARD",
    2013: "NOR'IDA",
    449: "1984 NORTHEAST FLOODS",
    364: "1982/83 MISSISSIPPI RIVER FLOODS 1",
    546: "Election Day\nFloods/Juan 1985",
    1046: "1995 ATM RIVER 1",
    1748: "2005/06 ATM RIVER",
    1741: "2005 NE FLOODS / TAMMY",
    726: "ALLISON 1989",
    1885: "MS River Floods\n2008",
    1161: "1996/97 ATM RIVER 1",
    451: "1984 KY, WV, TN FLOODS",
    1105: "1996 ATM RIVER",
    755: "1989 NEW ORLEANS FLOOD",
    2547: "2017 FLOOD/TORNADO OUTBREAKS",
    2815: "ETA",
    1525: "2002 TX FLOODS",
    604: "1986 MIDWEST FLOODS",
    2460: "2015 MISSISSIPPI RIVER FLOODS",
    851: "BOB",
    2813: "LAURA",
    871: "1992 TX FLOODS",
    2513: "HERMINE",
    1238: "1998 ATM RIVER",
    1835: "TX/OK Floods\n& Tornados 2007",
    3040: "DEBBY",
    372: "1983 ATM RIVER",
    559: "1986 ATM RIVER",
    1055: "1995 ATM RIVER 2",
    138: "1980 ATM RIVER",
    276: "1981/82 ATM RIVER",
    2944: "2022/23 ATM RIVER",
    2681: "2019 ATM RIVER",
    926: "1992/93 ATM RIVER 1",
    2536: "2017 ATM RIVER 2",
    1693: "2004/05 ATM RIVER",
    2530: "2016/17 ATM RIVER 1",
    1229: "1997 ATM RIVER 2",
    4: "1978 ATM RIVER 2",
    868: "1992 ATM RIVER 2",
    3002: "2024 ATM RIVER",
    2951: "2023 ATM RIVER 2",
    2103: "2010 ATM RIVER",
    669: "1988 ATM RIVER",
    1558: "2002 ATM RIVER",
    2: "1978 ATM RIVER 1",
    2486: "WV Downpour\n2016",
    462: "Tulsa Memorial Day\nFlood 1984",
}

# Storms treated as tropical (is_TS) and meet cover thresholds depending on the clustering input.
COVERED_NEW = [1734, 2214, 2578]     # input_clustering == '_new'  (Katrina, Sandy, Harvey)

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
    "AL","AK","AZ","AR","CA","CO","CT","DE",
    "DC","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN",
    "MS","MO","MT","NE","NV","NH","NJ","NM",
    "NY","NC","ND","OH","OK","OR","PA","RI",
    "SC","SD","TN","TX","UT","VT","VA","WA",
    "WV","WI","WY"
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
