# ETS Smart Dashboard

**Gaurang Ahuja, Apoorva Srivastava, Sidharth Malik, Gloria Yi**

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

### 3. Download the Data

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

---

## Run the App

### Live Demo
A hosted version of the app is available on [Posit Cloud](https://019e4677-6a04-aa54-3548-1eae51bbdb21.share.connect.posit.cloud/).

### Run Locally
The Streamlit App could be opened by using:
```bash
streamlit run app.py
```

The app uses a two-tier data loading strategy:

- **Sidebar** loads instantly from `pisa_country_stats.parquet` (~40 KB)
- **Charts** query only the rows and columns they need from `pisa_all.parquet` via DuckDB

If neither parquet is found locally, the app falls back to the public S3 copy of the parquet files, available at [https://pisa-dashboard-data.s3.ca-central-1.amazonaws.com](https://pisa-dashboard-data.s3.ca-central-1.amazonaws.com/pisa_all.parquet)
