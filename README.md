# ETS Smart Dashboard

**Gaurang Ahuja, Apoorva Srivastava, Sidharth Malik, Gloria Yi**

An interactive equity-focused dashboard for exploring PISA (Programme for International Student Assessment) score distributions across 81 countries and three cycles (2015, 2018, 2022), built for the Educational Testing Service (ETS).

---

## Quick Links

 - **Live dashboard** - [Posit Cloud deployment](https://019e4677-6a04-aa54-3548-1eae51bbdb21.share.connect.posit.cloud/) 
- **Full documentation** - [GitHub Pages](https://apoorva43.github.io/ETS_Smart_Dashboard/) 
- **Final report** - [`reports/final/final_report.pdf`](reports/final/final_report.pdf) *(update URL)*

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

The repository includes a seven-step pipeline that automatically downloads, merges, and optimizes PISA student and school data.
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

>Note: A smaller sample of the data (countries: Canada, US; year: 2022) is available [here](https://drive.google.com/file/d/1xBM1twCzTA8ikwzmEGn7psqxDJBU0X_t/view) and can be placed in `data/raw/sampledat.csv` to run the dashboard instead.  

---

## Run the App

### Run Locally

```bash
PYTHONPATH=. streamlit run app.py
```

The app will open at `http://localhost:8501`. Please ensure that you've run the `make data` command before this. 

### Live Demo

A hosted version of the app is available on [Posit Cloud](https://019e4677-6a04-aa54-3548-1eae51bbdb21.share.connect.posit.cloud/).

>Note: The current S3 bucket at `pisa-dashboard-data.s3.ca-central-1.amazonaws.com` is valid until 10 August 2026. It can be updated by changing the `S3_BASE_URL` constant in both `app.py` and `src/data_loader.py`.

---

## Testing

Tests live in the `tests/` folder and use synthetic data - no real PISA files needed.

### Install test dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

### Run all unit tests

```bash
PYTHONPATH=. pytest tests/ \
  --cov=src \
  --cov-report=term-missing \
  -v \
  --ignore=tests/test_integration.py
```

### Run integration tests separately

```bash
PYTHONPATH=. pytest tests/test_integration.py -v
```

### Run a specific test file

```bash
PYTHONPATH=. pytest tests/test_data_loader.py -v
```