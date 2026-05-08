# ETS Smart Dashboard

**Gaurang Ahuja, Apoorva Srivastava, Sidharth Malik, Gloria Yi**

---

## Reproducing the Report

### 1. Clone the repository

```bash
git clone https://github.com/apoorva43/ETS_Smart_Dashboard
cd ETS_Smart_Dashboard
```

### 2. Add the data

The raw data is not tracked by git. Download and place it as: `data/raw/sampledat.csv`

### 3. Set up the environment

```bash
conda env create -f environment.yml
conda activate ets_capstone
```

### 4. Run the full pipeline

```bash
make
```

This runs three steps in order:
1. `python src/compute_stats.py` - computes weighted statistics and saves to `data/processed/stats.json`
2. `python src/figures.py` - generates all three report figures and saves to `data/images`
3. `quarto render reports/proposal/proposal_report.qmd --to pdf` - renders the final PDF report