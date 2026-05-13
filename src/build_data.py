"""
PISA Data Processing Orchestrator.

This script manages the end-to-end data pipeline for the PISA student 
questionnaire datasets across specified cycle years. It handles URL validation, 
downloading, unarchiving, identifying the primary SPSS (.sav) data file 
amongst supplementary files, and triggering the conversion to an optimized 
Parquet format.

Notes
-----
This script modifies the system path (`sys.path`) at runtime to ensure 
that internal modules from the `src` directory can be imported successfully, 
regardless of the directory from which the script is executed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import download_pisa_year, sav_to_parquet, validate_url, PISA_URLS

TARGET_YEARS = [2022, 2018, 2015]


def main():
    for year in TARGET_YEARS:
        parquet_path = Path(f"data/processed/pisa_{year}.parquet")

        if parquet_path.exists():
            print(f" {year}: parquet already exists, skipping")
            continue

        if not validate_url(PISA_URLS.get(year, ""), year):
            continue

        try:
            raw_dir = download_pisa_year(year)
            sav_files = list(raw_dir.rglob("*.sav")) + list(raw_dir.rglob("*.SAV"))

            if not sav_files:
                print(f" No .sav file found anywhere under {raw_dir}. Contents:")
                for f in raw_dir.rglob("*"):
                    if f.is_file():
                        print(f" {f.relative_to(raw_dir)}  "
                              f"({f.stat().st_size / 1e6:.0f} MB)")
                raise FileNotFoundError(
                    f"Could not find .sav file under {raw_dir}"
                )

            # NOTE: Use the largest .sav, since it the main student file.
            # Supplementary files should not be concatenated.
            sav_file = max(sav_files, key=lambda f: f.stat().st_size)

            if len(sav_files) > 1:
                others = [f.name for f in sav_files if f != sav_file]
                print(f" Found {len(sav_files)} .sav files -- "
                      f"using largest, ignoring: {others}")

            print(f" Using: {sav_file.relative_to(raw_dir)}  "
                  f"({sav_file.stat().st_size / 1e6:.0f} MB)")
            sav_to_parquet(sav_file, parquet_path, year)

        except FileNotFoundError as e:
            print(f" {year}: file not found -- {e} -- skipping")
        except RuntimeError as e:
            print(f" {year}: download failed -- {e} -- skipping")
        except Exception as e:
            print(f" {year}: unexpected error -- {e} -- skipping")


if __name__ == "__main__":
    main()