# SEO Content Quality & Duplicate Detector

## Project Overview
*A machine learning pipeline that analyzes web content for SEO quality assessment and duplicate detection. The system processes HTML content, extracts NLP features, detects near-duplicate content using similarity algorithms, and classifies content quality. Includes both Jupyter notebook analysis and an interactive Streamlit web application.*

## Setup Instructions

go to **bash**
* git clone https://github.com/umang710/seo-content-detector
* cd seo-content-detector
* pip install -r requirements.txt
* jupyter notebook notebooks/seo_pipeline.ipynb

## Quick Start
* Place your dataset **(data.csv with url and html_content columns)** in the **../data/** folder

* Run the **seo_pipeline.ipynb** notebook sequentially

* All intermediate results will be saved as CSV files in the **../data/** folder

* For real-time analysis, use the **analyze_url()** function in the notebook

## Deployed Streamlit App
https://seo-content-detector-umang710.streamlit.app/

## Key Decisions
**Libraries:** BeautifulSoup (HTML parsing), scikit-learn (ML), textstat (readability)

**HTML Parsing:** Multi-selector approach focusing on semantic tags (article, main, p) with graceful fallbacks

**Similarity Threshold:** 0.8 cosine similarity for duplicate detection based on industry standards

**Model Selection:** Random Forest classifier for interpretability and handling mixed feature types

**Feature Engineering:** Combined traditional metrics (word count, readability) with TF-IDF keywords

## Results Summary
**Model Performance:** 92% accuracy (vs 64% baseline) - 44% improvement

**Dataset:** 81 web pages analyzed

**Duplicate Detection:** 15 duplicate pairs identified

**Thin Content:** 35 pages (43.2%) flagged with word count < 500

**Quality Distribution:** 51 Low, 23 Medium, 7 High quality pages

## Limitations
* Web scraping may be blocked by anti-bot protection on some sites

* Readability scores less accurate for highly technical content

* Model performance dependent on quality of synthetic labeling

* Small dataset size limits generalization to all web content types