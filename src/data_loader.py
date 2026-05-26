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


def download_pisa_zip(year: int, dataset_type: str = "student", raw_dir: Union[str, Path] = "data/raw") -> Path:
    """
    Downloads the PISA zip archive for either student or school data.

    Parameters
    ----------
    year : int
        The PISA cycle year to download (e.g., 2022, 2018).
    dataset_type : str, optional
        Either 'student' or 'school'. Defaults to 'student'.
    raw_dir : Union[str, Path], optional
        The root directory where raw data should be stored. Defaults to "data/raw".

    Returns
    -------
    Path
        The file path to the downloaded zip archive.
    """
    if dataset_type == "school":
        url = PISA_SCHOOL_URLS.get(year)
        filename = f"pisa_school_{year}.zip"
    else:
        url = PISA_URLS.get(year)
        filename = f"pisa_{year}.zip"

    if url is None:
        raise ValueError(f"Year {year} not found for {dataset_type} URLs.")

    # Isolate student and school data into separate subdirectories
    dest_dir = Path(raw_dir) / str(year) / dataset_type
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / filename

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
            print(f" {year} ({dataset_type}): complete zip already exists, skipping.")
            return zip_path
        
        print(f" {year} ({dataset_type}): incomplete zip, re-downloading...")
        zip_path.unlink()

    print(f" Downloading {year} {dataset_type} data from {url}...")
    try:
        r = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Download failed for {dataset_type} {year}: {e}")

    total = int(r.headers.get("content-length", 0))
    downloaded = 0

    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r {pct:.1f}% ({downloaded/1e6:.0f} / {total/1e6:.0f} MB)", end="", flush=True)
                
    print(f"\n {dataset_type.capitalize()} download complete.")
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
    student_target_cols = [c for c in keep_cols if c not in SCHOOL_COLS]
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


def merge_parquets(processed_dir: Union[str, Path] = "data/processed",
                   years: list = None
                   ) -> Path:
    """
    Function to merge individual yearly parquets into one optimized pisa_all.parquet.
    Applies float32 downcasting and categorical encoding before saving.

    Parameters
    ----------
    processed_dir : Union[str, Path], optional
        The directory containing the processed Parquet files. Defaults to "data/processed".
    years: list, optional
        The specific PISA cycle years to load. Defaults to [2015, 2018, 2022].

    Returns
    -------
    Path
        Path to the merged output file.

    Raises
    ------
    FileNotFoundError
        If no Parquet files are found for any of the target years in the specified directory.
    """
    if years is None:
        years = [2015, 2018, 2022]

    processed_dir = Path(processed_dir)
    frames = []

    for year in years:
        path = processed_dir / f"pisa_{year}.parquet"
        if not path.exists():
            print(f"Warning: {path} not found, skipping {year}")
            continue
        df = pd.read_parquet(path)
        df["YEAR"] = year
        # Downcast floats before concat to reduce peak memory
        f64 = df.select_dtypes("float64").columns
        df[f64] = df[f64].astype("float32")
        frames.append(df)
        print(f"Loaded {year}: {df.shape}")

    if not frames:
        raise FileNotFoundError(f"No parquets found in {processed_dir}")

    print("Concatenating...")
    combined = pd.concat(frames, ignore_index=True)
    del frames

    # Categorical encoding for low-cardinality string columns
    for col in combined.select_dtypes("object").columns:
        if combined[col].nunique() < 3000:
            combined[col] = combined[col].astype("category")

    # YEAR as int16
    combined["YEAR"] = combined["YEAR"].astype("int16")

    out_path = processed_dir / "pisa_all.parquet"
    combined.to_parquet(out_path, compression="zstd", index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved: {out_path} ({combined.shape}, {size_mb:.0f} MB on disk)")
    return out_path


def build_country_stats(processed_dir: Union[str, Path] = "data/processed") -> Path:
    """
    Function to build a tiny country-year aggregated parquet from pisa_all.parquet.
    Used by the dashboard sidebar and get_meta() - loads in milliseconds.

    Parameters
    ----------
    processed_dir : Union[str, Path], optional
        The directory containing the processed Parquet files. Defaults to "data/processed".

    Returns
    -------
    Path
        Path to the country stats output file.

    Raises
    ------
    FileNotFoundError
        If no merged Parquet files are found.
    """
    processed_dir = Path(processed_dir)
    source = processed_dir / "pisa_all.parquet"

    if not source.exists():
        raise FileNotFoundError(f"{source} not found. Run merge first.")

    print(f"Loading {source}...")
    df = pd.read_parquet(source)
    records = []

    for (cnt, year), grp in df.groupby(["CNT", "YEAR"], observed=True):
        w = grp["W_FSTUWT"]
        row = {"CNT": str(cnt), "YEAR": int(year), "n": len(grp)}

        for subj, prefix in [("math","MATH"), ("read","READ"), ("scie","SCIE")]:
            pv_cols = [f"PV{i}{prefix}" for i in range(1, 11)
                       if f"PV{i}{prefix}" in grp.columns]
            if pv_cols:
                pv_means = grp[pv_cols].multiply(w, axis=0).sum() / w.sum()
                row[f"score_{subj}"]    = round(float(pv_means.mean()), 2)
                row[f"score_{subj}_se"] = round(float(pv_means.std()), 4)

        for col in ["ESCS","HISEI","PAREDINT","HOMEPOS",
                    "BELONG","MATHMOT","AGE","GRADE"]:
            if col in grp.columns:
                valid = grp[col].notna()
                row[col] = round(float(
                    np.average(grp.loc[valid, col], weights=w[valid])
                ), 4) if valid.any() else None

        if "REPEAT" in grp.columns:
            row["repeat_rate"] = round(float((grp["REPEAT"] == 1).mean()), 4)
        if "IMMIG" in grp.columns:
            row["immig_rate"] = round(float((grp["IMMIG"] == 1).mean()), 4)
        if "OECD" in grp.columns:
            row["OECD"] = int(grp["OECD"].iloc[0])

        records.append(row)

    stats = pd.DataFrame(records)
    out_path = processed_dir / "pisa_country_stats.parquet"
    stats.to_parquet(out_path, compression="zstd", index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved: {out_path} ({stats.shape}, {size_mb:.2f} MB on disk)")
    return out_path