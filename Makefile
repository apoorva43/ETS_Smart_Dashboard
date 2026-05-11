.PHONY: all stats figures report clean data setup

# Default target - runs full pipeline
all: stats figures report

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

data:
	python build_data.py

setup:
	conda env create -f environment.yml
	pip install -e .