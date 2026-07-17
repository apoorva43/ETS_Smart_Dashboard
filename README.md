# ETS Smart Dashboard

**Gaurang Ahuja, Apoorva Srivastava, Sidharth Malik, Gloria Yi**

An interactive equity-focused dashboard for exploring PISA (Programme for International Student Assessment) score distributions across 81 countries and three cycles (2015, 2018, 2022), built for the Educational Testing Service (ETS).

---

## Quick Links

 - **Live dashboard** - [Posit Cloud deployment](https://019e4677-6a04-aa54-3548-1eae51bbdb21.share.connect.posit.cloud/) 
- **Full documentation** - [GitHub Pages](https://apoorva43.github.io/ETS_Smart_Dashboard/) 
- **Final report** - [`reports/final/final_report.pdf`](reports/final/final_report.pdf) 

---

## What this does
 
Rather than reporting country averages alone, the dashboard surfaces the **full score distribution** - broken down by percentile, socioeconomic background, immigration status, school type, and location - for any combination of country, subject, and year.
 
Two modes:
 
- **Data Story** - guided narrative that walks policymakers through key patterns with contextual framing
- **Explore** - lets analysts and researchers interact with the data directly

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/apoorva43/ETS_Smart_Dashboard
cd ETS_Smart_Dashboard
```

### 2. Set up the environment

```bash
conda env create -f environment.yml
conda activate ets_capstone
pip install -e .
```

### 3. Build the data pipeline

The repository includes a multi-step pipeline that automatically downloads, merges, and optimizes PISA student and school data.
To build the full multi-year dataset (2015, 2018, 2022):

```bash
make data
```

The pipeline runs in order:

1. **Download** - fetches raw SPSS zip archives from the OECD for each year
2. **Unzip** - extracts student and school questionnaire files
3. **Convert** - merges student and school files and writes a per-year parquet to `data/processed`
4. **Merge** - concatenates all years into a single optimized `pisa_all.parquet` with float32 downcastings and zstd compression
5. **Country stats** - aggregates to one row per country-year and writes `pisa_country_stats.parquet`, used by the dashboard sidebar
6. **Precompute KDE and percentiles** - replaces runtime KDE fitting by precomputing KDE arrays and weighted percentiles (P10, P25, etc.) per group
7. **Precompute standard errors** - precomputes Fay BRR standard error and 95% CI indexed by `(COUNTRY, YEAR, SUBJECT)`

> **Note on data access and speed:**  
> If you do not want to run the full download and preprocessing pipeline (which can take several minutes), you can directly use the precomputed datasets:
>
> - [pisa_all.parquet](https://drive.google.com/file/d/14agrHgJC03yvsuj5nI936EAuwVCUSb2Z/view?usp=sharing)
> - [pisa_country_stats.parquet](https://drive.google.com/file/d/1QNSRkj7o9AFi4T0_SqHi6Wz8WGf9AMBi/view?usp=sharing)
> - [pisa_precomputed.parquet](https://drive.google.com/file/d/1cnu-5s6SNMumJ_0UMoqdxKJalDpLllHq/view?usp=sharing)
> - [pisa_se_stats.parquet](https://drive.google.com/file/d/1jPE7wx6gagaCUPDStSFr1OXbW9BMShwX/view?usp=sharing)
>
> Place these files in:
> ```
> data/processed/
> ```
>
> Alternatively, for a lightweight demo, a smaller sample dataset (Canada + US, 2022 only) is available [here](
> https://drive.google.com/file/d/1xBM1twCzTA8ikwzmEGn7psqxDJBU0X_t/view).
>
> Place it in:
> ```
> data/raw/sampledat.csv
> ``` 

---

## Run the App

### Run Locally

From the root directory of the repository, run: 
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Please ensure that you've run the `make data` command before this. 

### Live Demo

A hosted version of the app is available on [Posit Cloud](https://019e4677-6a04-aa54-3548-1eae51bbdb21.share.connect.posit.cloud/).

> **Note on data hosting (Cloud CDN):**  
> The processed parquet files are currently hosted via a DigitalOcean Spaces CDN at `https://mds26-ets-capstone.sfo3.cdn.digitaloceanspaces.com`.
>
> If local data files are not found in the `data/processed/` directory, the application will automatically fall back to querying these public cloud endpoints. No manual configuration or API keys are required.
>
> **If the hosting endpoint ever changes:** You can migrate the processed `.parquet` files to any public S3-compatible bucket or CDN. To point the dashboard to the new location, simply update the `S3_BASE_URL` variable in these three files:
>
> - `app.py`
> - `src/data_loader.py`
> - `src/precompute.py`
>
> No other code changes are required - only updating the base URL to point to the new bucket.

---

## Testing

Tests live in the `tests/` folder and use synthetic data - no real PISA files needed. These tests are intended for development and validation purposes only, and do not affect end-user dashboard usage or runtime behavior. 

### Install test dependencies

If not already installed via `requirements.txt`: 

```bash
pip install pytest pytest-cov
```

### Run all unit tests

From the project root directory, run:
```bash
pytest tests/ \
  --cov=src \
  --cov-report=term-missing \
  -v \
  --ignore=tests/test_integration.py
```

### Run integration tests separately

```bash
pytest tests/test_integration.py -v
```

### Run a specific test file

```bash
pytest tests/<test_file>.py -v
```

Replace `<test_file>` with any file in the `tests/` directory. 

### Continuous Integration (CI)

All tests are automatically executed via GitHub Actions:

- on every push to `main` and `dev`
- on every pull request targeting `main` or `dev`

The CI pipeline runs unit tests, and executes integration tests separately on merge-related events. 

---

## Generate Final Report

Before rendering the report, ensure the environment is fully installed and the processed parquet files are available in `data/processed`:

```
cd reports/final
quarto render final_report.qmd --to pdf
```