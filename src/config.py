ID_COLS = ["CNT", "CNTSCHID", "CNTSTUID", "OECD", "STRATUM"]

WEIGHT_COLS = ["W_FSTUWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]

SCORE_COLS = (
    [f"PV{i}MATH" for i in range(1, 11)] +
    [f"PV{i}READ" for i in range(1, 11)] +
    [f"PV{i}SCIE" for i in range(1, 11)]
)

EQUITY_COLS = [
    "ST004D01T", "IMMIG", "ESCS", "HISEI",
    "PAREDINT", "HOMEPOS", "REPEAT", "LANGN",
]
CONTEXT_COLS = [
    "BELONG", "MATHMOT", "ST062Q01TA", "ST062Q03TA",
    "ST259Q01JA", "AGE", "GRADE",
]
SCHOOL_COLS = ["SC001Q01TA", "SCHLTYPE"]
KEEP_COLS = ID_COLS + WEIGHT_COLS + SCORE_COLS + EQUITY_COLS + CONTEXT_COLS + SCHOOL_COLS

SUBJECTS = {"MATH": "Mathematics", "READ": "Reading", "SCIE": "Science"}

PERCENTILES_COARSE = [10, 25, 50, 75, 90]
PERCENTILES_FINE   = list(range(5, 96, 5))

GENDER_MAP  = {1.0: "Female", 2.0: "Male"}
IMMIG_MAP   = {1.0: "Native", 2.0: "2nd-gen immigrant", 3.0: "1st-gen immigrant"}
LOC_MAP = {
    1.0: "Village", 
    2.0: "Small town", 
    3.0: "Town",
    4.0: "City", 
    5.0: "Large city",
    6.0: "Megacity"
}
SCHLTYPE_MAP = {1: "Independent private", 2: "Govt-dep. private", 3: "Public"}

GROUP_OPTIONS = {
    "Gender":             ("ST004D01T",  GENDER_MAP),
    "Immigration status": ("IMMIG",      IMMIG_MAP),
    "School location":    ("SC001Q01TA", LOC_MAP),
    "School type":        ("SCHLTYPE",   SCHLTYPE_MAP),
    "Socioeconomic status": ("ESCS",       None),
}

# Okabe-Ito colour palette for deuteranopia, protanopia, and tritanopia.
OKABE_ITO = {
    "orange":     "#E69F00",
    "sky_blue":   "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "pink":       "#CC79A7",
    "black":      "#000000",
}

COUNTRY_COLORS = {
    "CAN": OKABE_ITO["blue"],      
    "USA": OKABE_ITO["vermillion"], 
}

YEAR_COLORS = {
    2015: "#888780",                 
    2018: OKABE_ITO["sky_blue"],     
    2022: OKABE_ITO["blue"],        
}

PALETTE = [
    OKABE_ITO["vermillion"], 
    OKABE_ITO["orange"],      
    OKABE_ITO["sky_blue"],   
    OKABE_ITO["blue"],      
]

MIN_GROUP_N = 30  # minimum rows to compute a stat; below this return NaN

SYMBOLS_COARSE = ["triangle-down", "square", "diamond", "circle", "triangle-up"]

# Lookup dictionary for all PISA country codes
COUNTRY_NAMES = {
    "ALB": "Albania", 
    "ARE": "United Arab Emirates", 
    "ARG": "Argentina", 
    "AUS": "Australia",
    "AUT": "Austria", 
    "BEL": "Belgium", 
    "BGR": "Bulgaria", 
    "BIH": "Bosnia and Herzegovina",
    "BLR": "Belarus",
    "BRA": "Brazil",
    "BRN": "Brunei", 
    "CAN": "Canada",
    "CHE": "Switzerland", 
    "CHL": "Chile", 
    "COL": "Colombia", 
    "CRI": "Costa Rica",
    "CZE": "Czech Republic", 
    "DEU": "Germany", 
    "DNK": "Denmark", 
    "DOM": "Dominican Republic",
    "DZA": "Algeria",
    "ESP": "Spain", 
    "EST": "Estonia", 
    "FIN": "Finland", 
    "FRA": "France",
    "GBR": "United Kingdom", 
    "GEO": "Georgia", 
    "GRC": "Greece", 
    "GTM": "Guatemala",
    "HKG": "Hong Kong",
    "HRV": "Croatia", 
    "HUN": "Hungary", 
    "IDN": "Indonesia", 
    "IRL": "Ireland",
    "ISL": "Iceland", 
    "ISR": "Israel", 
    "ITA": "Italy", 
    "JAM": "Jamaica",
    "JOR": "Jordan",
    "JPN": "Japan", 
    "KAZ": "Kazakhstan", 
    "KHM": "Cambodia",
    "KOR": "South Korea", 
    "KSV": "Kosovo",
    "LBN": "Lebanon",
    "LTU": "Lithuania", 
    "LUX": "Luxembourg", 
    "LVA": "Latvia", 
    "MAC": "Macao",
    "MAR": "Morocco", 
    "MDA": "Moldova", 
    "MEX": "Mexico", 
    "MKD": "North Macedonia",
    "MLT": "Malta", 
    "MNE": "Montenegro", 
    "MNG": "Mongolia",
    "MYS": "Malaysia", 
    "NLD": "Netherlands",
    "NOR": "Norway", 
    "NZL": "New Zealand", 
    "PAN": "Panama", 
    "PER": "Peru",
    "PHL": "Philippines", 
    "POL": "Poland", 
    "PRT": "Portugal", 
    "PRY": "Paraguay",
    "PSE": "Palestinian Authority",
    "QAR": "Buenos Aires (Argentina)",
    "QAT": "Qatar",
    "QAZ": "Baku (Azerbaijan)",
    "QCH": "B-S-J-G (China)",
    "QCI": "B-S-J-Z (China)", 
    "QCN": "China", 
    "QES": "Spain (Regions)",
    "QMR": "Moscow Region (Russia)",
    "QRS": "Serbia", 
    "QRT": "Tatarstan (Russia)",
    "QUC": "Massachusettes (USA)", 
    "QUD": "Puerto Rico (USA)",
    "QUE": "North Carolina (USA)",
    "QUR": "Ukrainian regions (18 of 27)",
    "ROM": "Romania", 
    "ROU": "Romania", 
    "RUS": "Russia", 
    "SAU": "Saudi Arabia",
    "SGP": "Singapore", 
    "SLV": "El Salvador",
    "SRB": "Serbia", 
    "SVK": "Slovakia", 
    "SVN": "Slovenia",
    "SWE": "Sweden", 
    "TAP": "Chinese Taipei", 
    "THA": "Thailand", 
    "TTO": "Trinidad and Tobago",
    "TUN": "Tunisia",
    "TUR": "Turkey",
    "UKR": "Ukraine", 
    "URY": "Uruguay", 
    "USA": "United States", 
    "UZB": "Uzbekistan",
    "VNM": "Vietnam"
}