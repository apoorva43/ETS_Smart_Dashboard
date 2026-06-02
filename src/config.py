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
    1.0: "Village (<3k)", 
    2.0: "Small town (3-15k)", 
    3.0: "Town (15-100k)",
    4.0: "City (100k-1M)", 
    5.0: "Large city (>1M)",
    6.0: "Megacity (>10M)"
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