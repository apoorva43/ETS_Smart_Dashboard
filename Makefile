.PHONY: all stats figures report clean data setup clean-raw

# Variables for the data pipeline
YEARS = 2022 2018 2015
RAW_DIR = data/raw
PROCESSED_DIR = data/processed
PARQUETS = $(foreach y,$(YEARS),$(PROCESSED_DIR)/pisa_$(y).parquet)

# Default target - runs full pipeline
all: stats figures report

data: $(PARQUETS)

# Step 1: Download Rules
$(RAW_DIR)/%/student/pisa_%.zip:
	@echo "\n--- Downloading PISA $* (Student) ---"
	python src/build_data.py download --year $* --dataset student

$(RAW_DIR)/%/school/pisa_school_%.zip:
	@echo "\n--- Downloading PISA $* (School) ---"
	python src/build_data.py download --year $* --dataset school

# Step 2: Unzip Rules
$(RAW_DIR)/%/student/.unzipped: $(RAW_DIR)/%/student/pisa_%.zip
	@echo "\n--- Unzipping PISA $* (Student) ---"
	python src/build_data.py unzip --year $* --dataset student
	@touch $@

$(RAW_DIR)/%/school/.unzipped: $(RAW_DIR)/%/school/pisa_school_%.zip
	@echo "\n--- Unzipping PISA $* (School) ---"
	python src/build_data.py unzip --year $* --dataset school
	@touch $@

# Step 3: Convert to Parquet (Depends on BOTH unzipped markers)
$(PROCESSED_DIR)/pisa_%.parquet: $(RAW_DIR)/%/student/.unzipped $(RAW_DIR)/%/school/.unzipped
	@echo "\n--- Merging & Converting PISA $* to Parquet ---"
	python src/build_data.py convert --year $*

# Step 1: Compute inline statistics
stats: data/processed/stats.json

data/processed/stats.json: src/compute_stats.py data/raw/sampledat.csv
	python src/compute_stats.py

# Step 2: Generate figures
figures: data/images/fig1_math_dist.png \
		 data/images/fig2_gender_gap.png \
		 data/images/fig3_escs_dist.png 

data/images/fig1_math_dist.png \
data/images/fig2_gender_gap.png \
data/images/fig3_escs_dist.png: src/figures.py data/raw/sampledat.csv 
	python src/figures.py

# Step 3: Render quarto report
report: reports/proposal/proposal_report.pdf

reports/proposal/proposal_report.pdf: \
	reports/proposal/proposal_report.qmd \
	reports/proposal/references.bib \
	data/processed/stats.json \
	data/images/fig1_math_dist.png \
	data/images/fig2_gender_gap.png \
	data/images/fig3_escs_dist.png
	cd reports/proposal && quarto render proposal_report.qmd --to pdf

# Remove all generated files
clean:
	rm -f data/processed/stats.json 
	rm -f data/images/fig1_math_dist.png
	rm -f data/images/fig2_gender_gap.png
	rm -f data/images/fig3_escs_dist.png
	rm -f reports/proposal/proposal_report.pdf

clean-raw:
	rm -rf $(RAW_DIR)/*

setup:
	conda env create -f environment.yml
	pip install -e .