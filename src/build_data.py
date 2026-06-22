"""
PISA Data Processing CLI.

This script acts as the command-line interface for the end-to-end data pipeline,
orchestrating the downloading, unzipping, and Parquet conversion of PISA student 
and school questionnaire datasets. It is designed to be triggered via modular 
Makefile commands.
"""

import sys
import click
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    download_pisa_zip, 
    unzip_pisa_data, 
    sav_to_parquet, 
    validate_url, 
    merge_parquets,
    build_country_stats,
    build_se_stats,
    PISA_URLS,
    PISA_SCHOOL_URLS
)

@click.group()
def cli():
    """PISA Data ETL Pipeline CLI."""
    pass

@cli.command()
@click.option("--year", type=int, required=True, help="PISA cycle year.")
@click.option("--dataset", type=click.Choice(["student", "school"]), default="student")
def download(year, dataset):
    """Step 1: Download the raw zip archive."""
    url_dict = PISA_SCHOOL_URLS if dataset == "school" else PISA_URLS
    if not validate_url(url_dict.get(year, ""), year):
        click.secho(f"URL validation failed for {year} {dataset}", fg="red")
        sys.exit(1)
        
    try:
        download_pisa_zip(year, dataset_type=dataset)
    except Exception as e:
        click.secho(f"Download failed: {e}", fg="red")
        sys.exit(1)

@cli.command()
@click.option("--year", type=int, required=True, help="PISA cycle year.")
@click.option("--dataset", type=click.Choice(["student", "school"]), default="student")
def unzip(year, dataset):
    """Step 2: Extract the downloaded archive."""
    prefix = "pisa_school" if dataset == "school" else "pisa"
    zip_path = Path(f"data/raw/{year}/{dataset}/{prefix}_{year}.zip")
    
    if not zip_path.exists():
        click.secho(f"Zip file not found: {zip_path}. Run download step first.", fg="red")
        sys.exit(1)
        
    try:
        unzip_pisa_data(zip_path)
    except Exception as e:
        click.secho(f"Unzip failed: {e}", fg="red")
        sys.exit(1)

@cli.command()
@click.option("--year", type=int, required=True, help="PISA cycle year.")
def convert(year):
    """Step 3: Merge and convert to Parquet."""
    student_dir = Path(f"data/raw/{year}/student")
    school_dir  = Path(f"data/raw/{year}/school")
    parquet_path = Path(f"data/processed/pisa_{year}.parquet")
    
    stu_savs = list(student_dir.rglob("*.sav")) + list(student_dir.rglob("*.SAV"))
    sch_savs = list(school_dir.rglob("*.sav")) + list(school_dir.rglob("*.SAV"))
    
    if not stu_savs or not sch_savs:
        click.secho("Missing .sav files. Ensure both student and school data are unzipped.", fg="red")
        sys.exit(1)

    # Extract largest files to ignore supplementary questionnaires
    stu_sav = max(stu_savs, key=lambda f: f.stat().st_size)
    sch_sav = max(sch_savs, key=lambda f: f.stat().st_size)
    
    click.echo(f"  Student File: {stu_sav.name}")
    click.echo(f"  School File:  {sch_sav.name}")
    
    try:
        sav_to_parquet(stu_sav, sch_sav, parquet_path, year)
    except Exception as e:
        click.secho(f"Conversion failed: {e}", fg="red")
        sys.exit(1)

@cli.command()
@click.option("--years", default="2015,2018,2022",
              help="Comma-separated years to merge. Default: 2015, 2018, 2022")
@click.option("--processed-dir", default="data/processed",
              help="Directory containing yearly parquets.")
def merge(years, processed_dir):
    """Step 4: Merge yearly parquets into one optimized pisa_all.parquet."""
    year_list = [int(y.strip()) for y in years.split(",")]
    click.echo(f"Merging years: {year_list}")
    try:
        out = merge_parquets(processed_dir=processed_dir, years=year_list)
    except Exception as e:
        click.secho(f"Merge failed: {e}", fg="red")
        sys.exit(1)


@cli.command()
@click.option("--processed-dir", default="data/processed",
              help="Directory containing pisa_all.parquet.")
def country_stats(processed_dir):
    """Step 5: Build country-year aggregated stats parquet for the dashboard sidebar."""
    try:
        out = build_country_stats(processed_dir=processed_dir)
    except Exception as e:
        click.secho(f"Country stats failed: {e}", fg="red")
        sys.exit(1)


@cli.command()
@click.option("--processed-dir", default="data/processed",
              help="Directory containing pisa_all.parquet.")
@click.option("--out-path", default="data/processed/pisa_precomputed.parquet",
              help="Output path for precomputed stats.")
def precompute(processed_dir, out_path):
    """Step 6: Precompute group-level percentile stats for fast dashboard loading."""
    from precompute import build_precomputed
    try:
        build_precomputed(processed_dir=processed_dir, out_path=out_path)
    except Exception as e:
        click.secho(f"Precompute failed: {e}", fg="red")
        sys.exit(1)


@cli.command()
@click.option("--processed-dir", default="data/processed",
              help="Directory containing pisa_all.parquet.")
def se_stats(processed_dir):
    """Step 7: Precompute weighted SE + 95% CI per country/year/subject."""
    try:
        out = build_se_stats(processed_dir=processed_dir)
        click.secho(f"Wrote {out}")
    except Exception as e:
        click.secho(f"SE stats failed: {e}", fg="red")
        sys.exit(1)

if __name__ == "__main__":
    cli()