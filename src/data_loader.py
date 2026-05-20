import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import Union
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

HEAD_TIMEOUT = 10
DOWNLOAD_TIMEOUT = 60 * 30      # 30 minutes for large files
CHUNK_SIZE = 1024 * 1024        # 1 MB chunks
S3_BASE_URL = "https://pisa-dashboard-data.s3.ca-central-1.amazonaws.com"

def validate_url(url: str, year: int) -> bool:
    """
    Performs a HEAD request to confirm the URL is reachable before downloading.

    Gracefully handles network timeouts and connection errors by returning False 
    rather than crashing the script.

    Parameters
    ----------
    url : str
        The target URL to validate.
    year : int
        The PISA cycle year associated with the URL, used for logging.

    Returns
    -------
    bool
        True if the URL returns a 200 status code, False if the URL is missing, 
        unreachable, or times out.
    """
    if not url:
        print(f" {year}: URL not set in PISA_URLS -- skipping")
        return False
    try:
        r = requests.head(url, timeout=HEAD_TIMEOUT)
    except requests.exceptions.Timeout:
        print(f" {year}: HEAD request timed out after {HEAD_TIMEOUT}s -- skipping")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f" {year}: connection error during HEAD request -- {e}")
        return False
    if r.status_code == 200:
        size_mb = int(r.headers.get("content-length", 0)) / 1e6
        print(f" {year}: URL valid ({size_mb:.0f} MB)")
        return True
    print(f" {year}: URL returned {r.status_code} -- skipping")
    return False


def download_pisa_year(year: int, raw_dir: Union[str, Path] = "data/raw") -> Path:
    """
    Download and unzip the PISA student questionnaire zip for a given year.
    
    Checks local files to skip the download if a complete zip already exists 
    based on the expected content length from the server.

    Parameters
    ----------
    year : int
        The PISA cycle year to download (e.g., 2022, 2018).
    raw_dir : Union[str, Path], optional
        The root directory where raw data should be stored. Defaults to "data/raw".

    Returns
    -------
    Path
        The path to the directory containing the unzipped files.

    Raises
    ------
    ValueError
        If the requested year is not found in the PISA_URLS dictionary.
    RuntimeError
        If the download times out or encounters a network failure.
    """
    import zipfile

    url = PISA_URLS.get(year)
    if url is None:
        raise ValueError(
            f"Year {year} not found in PISA_URLS. "
            f"Available years: {list(PISA_URLS.keys())}"
        )

    dest_dir = Path(raw_dir) / str(year)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"pisa_{year}.zip"

    if zip_path.exists():
        actual_size = zip_path.stat().st_size
        try:
            expected_size = int(
                requests.head(url, timeout=HEAD_TIMEOUT)
                .headers.get("content-length", 0)
            )
        except requests.exceptions.RequestException:
            expected_size = 0

        if expected_size and actual_size >= expected_size * 0.99:
            print(f" {year}: already downloaded ({actual_size/1e6:.0f} MB), skipping")
            return dest_dir
        print(f" {year}: incomplete download ({actual_size/1e6:.0f} MB), re-downloading")
        zip_path.unlink()

    print(f" Downloading {year} from {url}...")
    try:
        r = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Download timed out after {DOWNLOAD_TIMEOUT}s for year {year}. "
            "Try again or increase DOWNLOAD_TIMEOUT."
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Download failed for year {year}: {e}")

    total = int(r.headers.get("content-length", 0))
    downloaded = 0

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
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

def sav_to_parquet(sav_path: Union[str, Path],
    parquet_path: Union[str, Path],
    year: int,
    keep_cols: list = KEEP_COLS,
) -> None:
    """
    Converts a PISA SPSS (.sav) file to a filtered Parquet file.

    Reads column metadata first to avoid loading the full dataset into memory. 
    Only columns present in the `keep_cols` list are loaded; expected columns 
    missing from the source file are populated with NaN to ensure schema consistency.

    Parameters
    ----------
    sav_path : Union[str, Path]
        The file path to the input raw SPSS (.sav) file.
    parquet_path : Union[str, Path]
        The destination file path for the output Parquet file.
    year : int
        The PISA cycle year associated with the dataset, appended as a new column.
    keep_cols : list, optional
        The list of exact column names to extract. Defaults to the global KEEP_COLS.

    Returns
    -------
    None
        Saves the processed data directly to disk.
    """
    import pyreadstat

    sav_path = Path(sav_path)
    parquet_path = Path(parquet_path)

    print(f" Reading column list from {sav_path.name}...")
    _, meta = pyreadstat.read_sav(str(sav_path), row_limit=0)
    available = [c for c in keep_cols if c in meta.column_names]
    missing = [c for c in keep_cols if c not in meta.column_names]

    if missing:
        print(f" {len(missing)} expected columns not in file: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    print(f" Loading {len(available)} columns from {len(meta.column_names)} total "
          f"({sav_path.name})...")
    print(f" This may take several minutes for a large file...")

    df, _ = pyreadstat.read_sav(str(sav_path), usecols=available)
    df["YEAR"] = year

    if missing:
        df[missing] = np.nan

    df = df[keep_cols + ["YEAR"]]

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    size_mb = parquet_path.stat().st_size / 1e6
    print(f" Saved: {parquet_path}  ({len(df):,} rows, {size_mb:.0f} MB)")


def load_sample_csv(path: Union[str, Path], keep_cols: list = KEEP_COLS,) -> pd.DataFrame:
    """
    Loads a sample CSV dataset for rapid development without requiring full Parquet files.

    Reads the CSV header first to determine available columns, filters them 
    against the `keep_cols` list, and loads the data efficiently. Automatically 
    injects a `YEAR` column into the resulting dataframe.

    Parameters
    ----------
    path : Union[str, Path]
        The file path to the sample CSV dataset.
    keep_cols : list, optional
        The list of columns to retain. Defaults to the global KEEP_COLS.

    Returns
    -------
    pd.DataFrame
        The filtered dataset containing only the specified columns and the 
        appended `YEAR` column.
    """
    path = Path(path)
    all_cols = pd.read_csv(path, nrows=0).columns.tolist()
    available = [c for c in keep_cols if c in all_cols]
    df = pd.read_csv(path, usecols=available, low_memory=False)
    df["YEAR"] = 2022
    return df

def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a single processed Parquet dataset into memory.

    Parameters
    ----------
    path : Union[str, Path]
        The file path to the target Parquet file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    return pd.read_parquet(Path(path))


def load_all_years(processed_dir: Union[str, Path] = "data/processed",
    years: list = None,
    optimize_memory: bool = True,
    profile_memory: bool = False
) -> pd.DataFrame:
    """
    Loads and concatenates available yearly Parquet files into a unified dataframe.

    Iterates through the target cycle years and attempts to load their corresponding 
    processed Parquet files. Missing years are skipped gracefully with a console warning.
    Can optionally convert string columns to categorical types to drastically reduce 
    memory footprint.

    Parameters
    ----------
    processed_dir : Union[str, Path], optional
        The directory containing the processed Parquet files. Defaults to "data/processed".
    years : list, optional
        The specific PISA cycle years to load. Defaults to [2022, 2018, 2015].
    optimize_memory : bool, optional
        If True, converts low-cardinality object/string columns to pandas 'category' 
        dtypes after concatenation. Defaults to True.
    profile_memory : bool, optional
        If True, performs a deep memory inspection before and after optimization and 
        prints the results. Noticeably impacts load time; keep False in production. Defaults to False.

    Returns
    -------
    pd.DataFrame
        A unified dataframe containing all loaded years concatenated vertically.

    Raises
    ------
    FileNotFoundError
        If no Parquet files are found for any of the target years in the specified directory.
    """
    if years is None:
        years = [2022, 2018, 2015]

    processed_dir = Path(processed_dir)
    frames = []
    
    for year in years:
        path = processed_dir / f"pisa_{year}.parquet"
        if path.exists():
            frames.append(load_parquet(path))
        else:
            print(f"  Warning: {path} not found, skipping year {year}")
            
    if not frames:
        raise FileNotFoundError(f"No parquet files found in {processed_dir} for years: {years}")
        
    df = pd.concat(frames, ignore_index=True)

    if profile_memory and not optimize_memory:
        print("Warning: profile_memory=True has no effect when optimize_memory=False")

    if optimize_memory:
        before_mb = 0.0
        after_mb = 0.0
        
        if profile_memory:
            before_mb = df.memory_usage(deep=True).sum() / 1e6
            
        object_cols = df.select_dtypes(include=["object"]).columns
        for col in object_cols:
            if df[col].nunique() < 100:
                df[col] = df[col].astype("category")
                
        if profile_memory:
            after_mb = df.memory_usage(deep=True).sum() / 1e6
            print(f"Memory optimization: {before_mb:.0f} MB to {after_mb:.0f} MB "
                  f"(saved {before_mb - after_mb:.0f} MB)")

    return df

def load_all_years_s3(base_url: str = S3_BASE_URL,
    years: list = None,
    optimize_memory: bool = True        
) -> pd.DataFrame:
    """
    Load and concatenate PISA parquet files from public S3 URLs.

    No credentials required - bucket must be publically readable.

    Parameters
    ----------
    base_url : str
        Base HTTPS URL of the S3 bucket, without trailing slash.
    years : list, optional
        PISA cycle years to load. Defaults to [2015, 2018, 2022].
    optimize_memory : bool, optional
        Convert low-cardinality string columns to categoricals. Defaults to True.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe for all available years.

    Raises
    ------
    Runtime Error
        If no parquet files could be loaded from any of the target years.
    """
    import io

    if years is None:
        years = [2015, 2018, 2022]

    frames = []
    for year in years:
        url = f"{base_url}/pisa_{year}.parquet"
        try:
            df_year = pd.read_parquet(url)
            df_year["YEAR"] = year
            frames.append(df_year)
        except Exception as e:
            print(f" Warning: unexpected error loading {year} - {e}")

    if not frames:
        raise RuntimeError(
            f"No parquet files loaded from {base_url}."
            "Check the bucket URL and that files are publically accessible."
        )
    
    df = pd.concat(frames, ignore_index=True)

    if optimize_memory:
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() < 100:
                df[col] = df[col].astype("category")

    return df