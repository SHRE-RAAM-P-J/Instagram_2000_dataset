# AI-Powered Instagram Influencer Analytics & Automation Insights

> **SR NEXT Self-Guided Internship Program — Phase 1 & Phase 2**
> Independent Intern / Project Contributor — SHRE RAAM P J

---

## Repository Structure

```
instagram-influencer-analytics/
│
├── data/
│   ├── final_phase_1.xlsx                  # Raw collected dataset (2,093 influencers)
│   ├── final_phase_2.xlsx                  # Cleaned, scored & ranked dataset (5 sheets)
│   └── SR_NEXT_Phase2_ML_Results.csv       # Dataset with ML model predictions added
│
├── images/
│   ├── actual_vs_predicted_er.png          # Model 2 — Actual vs Predicted ER plot
│   ├── confusion_matrix_automation.png     # Model 1 — Confusion matrix
│   ├── feature_importance_automation.png   # Model 1 — Feature importances
│   └── feature_importance_er.png          # Model 2 — Feature importances
│
├── report/
│   └── Phase_1_Report.pdf                  # Week 1 written report
│
├── README.md                               # You are here
├── run_pipeline.py                         # Master script — runs everything end-to-end
├── SR_NEXT_Phase2_Code.py                  # Phase 2 data cleaning, scoring & ranking
├── SR_NEXT_Phase2_ML_Models.py            # ML models — classifier + regression
├── SR_NEXT_Phase2_Dashboard.html          # Standalone HTML dashboard (open in browser)
└── streamlit_app.py                        # Live interactive Streamlit dashboard
```

---

## Project Overview

This project collects, cleans, analyses, and ranks **2,093 Instagram influencers** (under 1M followers) using Python, data engineering, NLP-based classification, ML models, and interactive dashboards. Completed as part of the SR NEXT Self-Guided Internship Program over 4 weeks across two phases.

---

## Phase 1 — Data Collection

**Objective:** Collect structured public data on 2,000+ Instagram influencers with under 1M followers.

### Tools Used

| Tool | Purpose |
|------|---------|
| Heepsy | Influencer discovery & niche/category filtering |
| Apify | Instagram data extraction & verification |
| ChatGPT / Claude | Structured data generation & formatting assistance |
| Excel / Google Sheets | Dataset organisation, cleaning & preprocessing |

### Data Collected Per Influencer

- Username / Instagram Handle
- Follower Count
- Category / Niche
- Engagement Rate (based on last 10 posts)
- Contact Details (if publicly available in bio)
- Posting Frequency (avg posts per month)
- Hashtags / Content Themes
- Automation Usage Indicators
- Additional Notes

### Phase 1 Challenges

- Follower counts stored in mixed formats (`244.9K`, `1.5M`, plain integers) — required custom parsing logic
- Engagement rate stored as text (`<0.1%`, `>20%`, `5.2%`) — required symbol stripping and decimal conversion
- 500+ inconsistent category names — required NLP-based keyword grouping into 13 broad categories
- Duplicate profiles, merged cells, and missing contact data throughout the raw dataset

---

## Phase 2 — AI-Powered Analysis & Ranking

**Objective:** Clean the raw dataset, engineer features, score and rank influencers, detect automation patterns, train ML models, and build interactive dashboards.

### Full Pipeline

```
Raw Excel (Phase 1)
        ↓
  Data Cleaning
  (fix followers K/M format, fix ER text symbols, parse posting frequency)
        ↓
  Feature Engineering
  (hashtag count, NLP-based broad category grouping, normalisation)
        ↓
  Composite Influence Score (0–100)
  (weighted formula across 4 normalised features)
        ↓
  Automation Likelihood Detection
  (rule-based pattern analysis on ER + posting frequency)
        ↓
  ML Models
  (Random Forest classifier + Random Forest regressor)
        ↓
  Ranking + Tier Assignment
        ↓
  Ranked Excel (5 sheets) + Ranked CSV + Streamlit Dashboard + HTML Dashboard
```

---

## Influence Score Formula

Each influencer receives a composite score out of 100:

```
Influence Score = (ER × 0.40) + (Frequency × 0.25) + (Followers × 0.20) + (Hashtags × 0.15) × 100
```

| Feature | Weight | Reason |
|---------|--------|--------|
| Engagement Rate | 40% | Most important — measures real audience response |
| Posting Frequency | 25% | Consistency = active, serious influencer |
| Followers (log-scaled) | 20% | Reach matters but diminishing returns at high counts |
| Hashtag Diversity | 15% | Indicates content strategy effort |

> All features normalised to 0–1 before scoring. Followers use log scaling to reduce skew from mega-accounts.

---

## ML Models

### Model 1 — Automation Likelihood Classifier

- **Algorithm:** Random Forest Classifier
- **Task:** Predict whether an account is using automation tools (High / Medium / Low)
- **Accuracy:** 100% test set, 94.75% cross-validation
- **Top Feature:** Engagement Rate (38% importance)

### Model 2 — Engagement Rate Predictor

- **Algorithm:** Random Forest Regressor + Linear Regression (compared)
- **Task:** Predict engagement rate from followers, posting frequency, category, hashtag count
- **R² Score:** 0.9999 | **MAE:** 0.21%

### Automation Detection Logic

| Risk Level | Condition | Reasoning |
|------------|-----------|-----------|
| 🔴 High | ER > 30% on account with >50K followers | Statistically impossible organically |
| 🔴 High | Posts > 25/month with ER < 2% | Bot scheduler, no real audience response |
| 🟡 Medium | ER > 20% on any account | Suspicious engagement for any size |
| 🟡 Medium | Posts > 20/month with ER < 4% | High frequency + below-average ER |
| 🟢 Low | All other patterns | Normal organic behaviour |

---

## Final Excel Output — 5 Sheets

| Sheet | Contents |
|-------|---------|
| Ranked Influencers | All 2,093 ranked by Influence Score with tier & automation flag |
| Top By Category | Top 10 influencers per broad content category |
| Summary Stats | KPI dashboard, category breakdown, tier distribution |
| Automation Risk | All flagged accounts with specific reason per row |
| Legend | Explanation of every column, score, and colour |

---

## Dashboards

### HTML Dashboard — `SR_NEXT_Phase2_Dashboard.html`
Open in any browser. No installation needed. Fully offline.

### Streamlit Dashboard — `streamlit_app.py`
Live interactive app with real-time sidebar filters that update all 6 tabs simultaneously.

**6 tabs:**
- **Overview** — KPI cards, tier & automation charts, ER distribution, key insights
- **Rankings** — Top N bar chart + full ranked table with download button
- **Categories** — Count, avg score, avg ER per category + Top 10 per category
- **Engagement** — Scatter plot: Followers vs ER (log scale, dot size = Influence Score)
- **Automation Risk** — Flagged accounts, filterable by risk level
- **Explorer** — Real-time search + filter across all 2,093 influencers

---

## How to Run

### Install requirements
```bash
pip install pandas numpy openpyxl scikit-learn matplotlib seaborn streamlit plotly
```

### Option 1 — Run full pipeline + launch dashboard (recommended)
```bash
python run_pipeline.py
```

### Option 2 — Run pipeline without dashboard
```bash
python run_pipeline.py --no-dashboard
```

### Option 3 — Skip ML models (faster)
```bash
python run_pipeline.py --skip-ml
```

### Option 4 — Launch dashboard only
```bash
python -m streamlit run streamlit_app.py
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Total Influencers Analysed | 2,093 |
| Average Influence Score | 33.00 / 100 |
| Average Engagement Rate | 39.65% |
| Median Followers | 39,212 |
| Tier 1 — Top 10% | 209 influencers |
| High Automation Risk | 196 accounts (9.4%) |
| Influencers with Contact Info | 1,597 |
| Broad Categories | 13 |
| Top Influencer | studio_choom (Score: 91.10) |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Excel](https://img.shields.io/badge/Excel-217346?style=flat&logo=microsoft-excel&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)

---

## Internship Details

| Field | Details |
|-------|---------|
| Program | SR NEXT Self-Guided Internship |
| Role | Independent Intern / Project Contributor |
| Duration | 4 Weeks (2 Phases) |
| Mode | Remote / Self-Guided |
| Phase 1 | Data Collection & Structuring |
| Phase 2 | AI-Powered Analysis, ML Models & Dashboards |

---

## Author

**SHRE RAAM P J**
Independent Intern — SR NEXT
[GitHub Profile](https://github.com/SHRE-RAAM-P-J)

---

*All data collected is publicly available Instagram profile information.*