ID_COLS = ["CNT", "CNTSCHID", "CNTSTUID", "OECD", "STRATUM"]

WEIGHT_COLS = ["W_FSTUWT"] + [f"W_FSTURWT{i}" for i in range(1, 81)]

SCORE_COLS = (
    [f"PV{i}MATH" for i in range(1, 11)] +
    [f"PV{i}READ" for i in range(1, 11)] +
    [f"PV{i}SCIE" for i in range(1, 11)]
)

KEEP_COLS = ID_COLS + WEIGHT_COLS + SCORE_COLS

MIN_GROUP_N = 30  # minimum rows to compute a stat; below this return NaN