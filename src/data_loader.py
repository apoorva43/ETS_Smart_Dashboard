import pandas as pd
import numpy as np
import requests
import zipfile
from pathlib import Path
from typing import Union
from src.config import KEEP_COLS, SCHOOL_COLS

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


def download_pisa_zip(year: int, raw_dir: Union[str, Path] = "data/raw") -> Path:
    """
    Downloads the PISA zip archive, skipping if a complete file already exists.

    Checks local files based on the expected content length from the server to 
    prevent re-downloading existing data.

    Parameters
    ----------
    year : int
        The PISA cycle year to download (e.g., 2022, 2018).
    raw_dir : Union[str, Path], optional
        The root directory where raw data should be stored. Defaults to "data/raw".

    Returns
    -------
    Path
        The file path to the downloaded zip archive.

    Raises
    ------
    ValueError
        If the requested year is not found in the PISA_URLS dictionary.
    RuntimeError
        If the download times out or encounters a network failure.
    """
    url = PISA_URLS.get(year)
    if url is None:
        raise ValueError(f"Year {year} not found in PISA_URLS.")

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
            print(f" {year}: complete zip already exists ({actual_size/1e6:.0f} MB), skipping.")
            return zip_path
        
        print(f" {year}: incomplete zip ({actual_size/1e6:.0f} MB), re-downloading...")
        zip_path.unlink()

    print(f" Downloading {year} from {url}...")
    try:
        r = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Download timed out after {DOWNLOAD_TIMEOUT}s for year {year}.")
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
                print(f"\r {pct:.1f}% ({downloaded/1e6:.0f} / {total/1e6:.0f} MB)", end="", flush=True)
                
    print("\n Download complete.")
    return zip_path

def unzip_pisa_data(zip_path: Union[str, Path]) -> Path:
    """
    Unzips a downloaded PISA archive into its parent directory.

    Parameters
    ----------
    zip_path : Union[str, Path]
        The file path to the target zip archive.

    Returns
    -------
    Path
        The directory containing the unzipped files.
    """
    zip_path = Path(zip_path)
    dest_dir = zip_path.parent
    
    print(f" Unzipping {zip_path.name} into {dest_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
        
    print(f" Unzip complete.")
    return dest_dir

def sav_to_parquet(
    student_sav:  Union[str, Path],
    school_sav:   Union[str, Path],
    parquet_path: Union[str, Path],
    year:         int,
    keep_cols:    list = KEEP_COLS,
) -> None:
    """
    Merges student and school SPSS files and converts them to an optimized Parquet file.

    Parameters
    ----------
    student_sav : Union[str, Path]
        File path to the raw student SPSS (.sav) file.
    school_sav : Union[str, Path]
        File path to the raw school SPSS (.sav) file.
    parquet_path : Union[str, Path]
        Destination file path for the output Parquet file.
    year : int
        The PISA cycle year.
    keep_cols : list, optional
        The exact columns to extract and retain. Defaults to KEEP_COLS.
    """
    import pyreadstat

    student_sav = Path(student_sav)
    school_sav = Path(school_sav)
    parquet_path = Path(parquet_path)

    # 1. Process School Data
    print(f" Reading school metadata from {school_sav.name}...")
    _, school_meta = pyreadstat.read_sav(str(school_sav), row_limit=0)
    
    school_target_cols = ["CNTSCHID"] + SCHOOL_COLS
    avail_school = [c for c in school_target_cols if c in school_meta.column_names]
    df_school, _ = pyreadstat.read_sav(str(school_sav), usecols=avail_school)

    # 2. Process Student Data
    print(f" Reading student metadata from {student_sav.name}...")
    _, student_meta = pyreadstat.read_sav(str(student_sav), row_limit=0)
    
    # We exclude school columns from the student extraction to prevent merge collisions
    student_target_cols = [c for c in keep_cols if c not in school_target_cols]
    avail_student = [c for c in student_target_cols if c in student_meta.column_names]
    
    print(f" Loading student data (this may take several minutes)...")
    df_student, _ = pyreadstat.read_sav(str(student_sav), usecols=avail_student)
    df_student["YEAR"] = year

    # 3. Merge Datasets
    print(f"  Merging school data on CNTSCHID...")
    # Some older SPSS files store CNTSCHID as string or float; ensure matching types before merge
    if "CNTSCHID" in df_student.columns and "CNTSCHID" in df_school.columns:
        df_student["CNTSCHID"] = df_student["CNTSCHID"].astype(float)
        df_school["CNTSCHID"] = df_school["CNTSCHID"].astype(float)
        
    df = df_student.merge(df_school, on="CNTSCHID", how="left")

    # 4. Fill completely missing columns to maintain strictly consistent schema across years
    missing = [c for c in keep_cols if c not in df.columns]
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