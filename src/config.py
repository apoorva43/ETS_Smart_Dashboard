ID_COLS = ["CNT", "CNTSCHID", "CNTSTUID", "OECD", "STRATUM"]

WEIGHT_COLS = ["W_FSTUWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]

SCORE_COLS = (
    [f"PV{i}MATH" for i in range(1, 11)] +
    [f"PV{i}READ" for i in range(1, 11)] +
    [f"PV{i}SCIE" for i in range(1, 11)]
)

KEEP_COLS = ID_COLS + WEIGHT_COLS + SCORE_COLS

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
LOC_MAP     = {
    1.0: "Village (<3k)", 2.0: "Small town (3-15k)", 3.0: "Town (15-100k)",
    4.0: "City (100k-1M)", 5.0: "Large city (>1M)",
}
SCHLTYPE_MAP = {1: "Public", 2: "Govt-dep. private", 3: "Independent private"}

GROUP_OPTIONS = {
    "Gender":             ("ST004D01T",  GENDER_MAP),
    "Immigration status": ("IMMIG",      IMMIG_MAP),
    "School location":    ("SC001Q01TA", LOC_MAP),
    "School type":        ("SCHLTYPE",   SCHLTYPE_MAP),
}

COUNTRY_COLORS = {"CAN": "#185FA5", "USA": "#D85A30"}
YEAR_COLORS    = {2015: "#888780",  2018: "#BA7517", 2022: "#185FA5"}
PALETTE        = ["#D85A30", "#BA7517", "#1D9E75", "#185FA5"]

MIN_GROUP_N = 30  # minimum rows to compute a stat; below this return NaN