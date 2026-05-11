"""
PISA Data Downloader Script.

This script orchestrates the downloading of raw PISA student questionnaire 
datasets for the specified cycle years (2022, 2018, 2015). It validates the 
target URLs before attempting to download and extract the zip archives into 
the raw data directory.

Notes
-----
This script modifies the system path (`sys.path`) at runtime to ensure 
that internal modules from the `src` directory can be imported successfully, 
regardless of the directory from which the script is executed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import download_pisa_year, validate_url, PISA_URLS

for year in [2022, 2018, 2015]:
    if not validate_url(PISA_URLS.get(year, ""), year):
        continue

    raw_dir = download_pisa_year(year)