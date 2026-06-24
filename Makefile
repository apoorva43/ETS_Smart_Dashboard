.PHONY: all stats figures clean data setup clean-raw clean-data report-proposal report-final test test-integration run-local run-posit

# Variables for the data pipeline
YEARS = 2022 2018 2015
RAW_DIR = data/raw
PROCESSED_DIR = data/processed
PARQUETS = $(foreach y,$(YEARS),$(PROCESSED_DIR)/pisa_$(y).parquet)

# Default target - runs full data pipeline, stats, figures, and final report
all: stats figures report-final

# --- 1. Data Pipeline ---
data: $(PARQUETS) \
	  $(PROCESSED_DIR)/pisa_all.parquet \
	  $(PROCESSED_DIR)/pisa_country_stats.parquet \
	  $(PROCESSED_DIR)/pisa_precomputed.parquet \
	  $(PROCESSED_DIR)/pisa_se_stats.parquet

$(RAW_DIR)/%/student/pisa_%.zip:
	@echo "\n--- Downloading PISA $* (Student) ---"
	python src/build_data.py download --year $* --dataset student

$(RAW_DIR)/%/school/pisa_school_%.zip:
	@echo "\n--- Downloading PISA $* (School) ---"
	python src/build_data.py download --year $* --dataset school

$(RAW_DIR)/%/student/.unzipped: $(RAW_DIR)/%/student/pisa_%.zip
	@echo "\n--- Unzipping PISA $* (Student) ---"
	python src/build_data.py unzip --year $* --dataset student
	@touch $@

$(RAW_DIR)/%/school/.unzipped: $(RAW_DIR)/%/school/pisa_school_%.zip
	@echo "\n--- Unzipping PISA $* (School) ---"
	python src/build_data.py unzip --year $* --dataset school
	@touch $@

$(PROCESSED_DIR)/pisa_%.parquet: $(RAW_DIR)/%/student/.unzipped $(RAW_DIR)/%/school/.unzipped
	@echo "\n--- Merging & Converting PISA $* to Parquet ---"
	python src/build_data.py convert --year $*

$(PROCESSED_DIR)/pisa_all.parquet: $(PARQUETS)
	@echo "\n--- Merging all years into pisa_all.parquet ---"
	python src/build_data.py merge --years 2015,2018,2022

$(PROCESSED_DIR)/pisa_country_stats.parquet: $(PROCESSED_DIR)/pisa_all.parquet
	@echo "\n--- Building country stats parquet ---"
	python src/build_data.py country-stats

$(PROCESSED_DIR)/pisa_precomputed.parquet: $(PROCESSED_DIR)/pisa_all.parquet
	@echo "\n--- Precomputing group-level KDE + percentile stats ---"
	python -W ignore src/build_data.py precompute 

$(PROCESSED_DIR)/pisa_se_stats.parquet: $(PROCESSED_DIR)/pisa_all.parquet
	@echo "\n--- Computing standard errors ---"
	python src/build_data.py standarderror

# --- 2. Stats & Figures ---
stats: data/processed/stats.json

data/processed/stats.json: src/compute_stats.py data/raw/sampledat.csv
	python src/compute_stats.py

figures: data/images/fig1_math_dist.png \
		 data/images/fig2_gender_gap.png \
		 data/images/fig3_escs_dist.png 

data/images/fig1_math_dist.png \
data/images/fig2_gender_gap.png \
data/images/fig3_escs_dist.png: src/figures.py data/raw/sampledat.csv 
	python src/figures.py

# --- 3. Reports ---
report-proposal: reports/proposal/proposal_report.pdf

reports/proposal/proposal_report.pdf: \
	reports/proposal/proposal_report.qmd \
	reports/proposal/references.bib \
	data/processed/stats.json \
	data/images/fig1_math_dist.png \
	data/images/fig2_gender_gap.png \
	data/images/fig3_escs_dist.png
	cd reports/proposal && quarto render proposal_report.qmd --to pdf

report-final: reports/final/final_report.pdf

reports/final/final_report.pdf: \
	reports/final/final_report.qmd \
	reports/final/references.bib
	cd reports/final && quarto render final_report.qmd --to pdf

# --- 4. Testing ---
test:
	pytest tests/ --cov=src --cov-report=term-missing -v --ignore=tests/test_integration.py

test-integration:
	pytest tests/test_integration.py -v

# --- 5. App Execution ---
run-local:
	streamlit run app.py

# --- 6. Utilities ---
clean:
	rm -f data/processed/stats.json 
	rm -f data/images/fig1_math_dist.png
	rm -f data/images/fig2_gender_gap.png
	rm -f data/images/fig3_escs_dist.png
	rm -f reports/proposal/proposal_report.pdf
	rm -f reports/final/final_report.pdf

clean-raw:
	rm -rf $(RAW_DIR)/*

clean-data:
	rm -f $(PROCESSED_DIR)/*.parquet

setup:
	conda env create -f environment.yml
	pip install -e .