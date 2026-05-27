# ETS Smart Dashboard

**Gaurang Ahuja, Apoorva Srivastava, Sidharth Malik, Gloria Yi**

---

## Reproducing the Report

### 1. Clone the repository

```bash
git clone https://github.com/apoorva43/ETS_Smart_Dashboard
cd ETS_Smart_Dashboard
```

### 2. Add the sample data

The raw data is not tracked by git. Download it [here](https://drive.google.com/file/d/1xBM1twCzTA8ikwzmEGn7psqxDJBU0X_t/view?usp=sharing) 
and place it in: `data/raw/sampledat.csv`
> NOTE: The data file is large and hence hosted outside of GitHub.

### 3. Set up the environment

```bash
conda env create -f environment.yml
conda activate ets_capstone
pip install -e .
```

### 4. Download the Data

The repository includes a five-step pipeline that automatically downloads, merges, and optimizes PISA student and school data.
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

### 5. Run the full pipeline for report generation

```bash
make
```

This runs three steps in order:
1. `python src/compute_stats.py` - computes weighted statistics and saves to `data/processed/stats.json`
2. `python src/figures.py` - generates all three report figures and saves to `data/images`
3. `quarto render reports/proposal/proposal_report.qmd --to pdf` - renders the final PDF report

---

## Run the App
![running app example](data/images/running_app.png)

### Live Demo
A hosted version of the app is available on [Posit Cloud](https://019e4677-6a04-aa54-3548-1eae51bbdb21.share.connect.posit.cloud/).

### Run Locally
The Streamlit App could be opened by using:
```bash
streamlit run app.py
```

Run it in dev mode for showing memory savings:
```bash
PISA_PROFILE_MEMORY=1 streamlit run app.py
```

The app uses a two-tier data loading strategy:

- **Sidebar** loads instantly from `pisa_country_stats.parquet` (~40 KB)
- **Charts** query only the rows and columns they need from `pisa_all.parquet` via DuckDB

If neither parquet is found locally, the app falls back to the public S3 copy of the parquet files. 
