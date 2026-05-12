import pandas as pd
import numpy as np
import requests
from pathlib import Path
from src.config import KEEP_COLS

# Student questionnaire files (SPSS format).
# NOTE: Update here if OECD changes hosting.
PISA_URLS = {
    2022: "https://webfs.oecd.org/pisa2022/STU_QQQ_SPSS.zip",
    2018: "https://webfs.oecd.org/pisa2018/SPSS_STU_QQQ.zip",
    2015: "https://webfs.oecd.org/pisa/PUF_SPSS_COMBINED_CMB_STU_QQQ.zip",
}

# School questionnaire files
PISA_SCHOOL_URLS = {
    2022: "https://webfs.oecd.org/pisa2022/SCH_QQQ_SPSS.zip",
    2018: "https://webfs.oecd.org/pisa2018/SPSS_SCH_QQQ.zip",
    2015: "https://webfs.oecd.org/pisa/PUF_SPSS_COMBINED_CMB_SCH_QQQ.zip",
}

def validate_url(url: str, year: int) -> bool:
    """
    Performs a HEAD request to confirm the URL is reachable before starting download.

    Parameters
    ----------
    url : str
        The target URL to validate.
    year : int
        The PISA cycle year associated with the URL, used for logging.

    Returns
    -------
    bool
        True if the URL returns a 200 status code, False otherwise.
    """
    if not url:
        print(f" {year}: URL not set in PISA_URLS -- skipping")
        return False
    r = requests.head(url)
    if r.status_code == 200:
        size_mb = int(r.headers.get("content-length", 0)) / 1e6
        print(f" {year}: URL valid ({size_mb:.0f} MB)")
        return True
    print(f" {year}: URL returned {r.status_code} -- skipping")
    return False


def download_pisa_year(year: int, raw_dir: str = "data/raw") -> Path:
    """
    Download and unzip the PISA student questionnaire zip for a given year.
    
    Checks local files to skip the download if a complete zip already exists 
    based on the expected content length from the server.

    Parameters
    ----------
    year : int
        The PISA cycle year to download (e.g., 2022, 2018).
    raw_dir : str, optional
        The root directory where raw data should be stored. Defaults to "data/raw".

    Returns
    -------
    Path
        The path to the directory containing the unzipped files.

    Raises
    ------
    requests.exceptions.HTTPError
        If the download request returns an unsuccessful status code.
    KeyError
        If the provided year does not exist in the PISA_URLS dictionary.
    """
    import zipfile

    dest_dir = Path(raw_dir) / str(year)
    dest_dir.mkdir(parents=True, exist_ok=True)

    url = PISA_URLS[year]
    zip_path = dest_dir / f"pisa_{year}.zip"

    if zip_path.exists():
        actual_size = zip_path.stat().st_size
        expected_size = int(requests.head(url).headers.get("content-length", 0))
        if expected_size and actual_size >= expected_size * 0.99:
            print(f" {year}: already downloaded ({actual_size/1e6:.0f} MB), skipping")
            return dest_dir
        print(f" {year}: incomplete download ({actual_size/1e6:.0f} MB), re-downloading")
        zip_path.unlink()

    print(f" Downloading {year} from {url}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    downloaded = 0

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                mb_done = downloaded / 1_000_000
                mb_total = total / 1_000_000
                print(f"\r {pct:.1f}% ({mb_done:.0f} / {mb_total:.0f} MB)",
                      end="", flush=True)
    print()

    print(f" Unzipping...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

    print(f" Done: {dest_dir}")
    return dest_dir