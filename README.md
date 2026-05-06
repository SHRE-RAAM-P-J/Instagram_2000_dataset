# AI-Powered Instagram Influencer Analytics & Automation Insights

> **SR NEXT Self-Guided Internship Program — Phase 1 & Phase 2**  
> Independent Intern / Project Contributor — SHRE RAAM P J

---

## Project Overview

This project collects, cleans, analyses, and ranks **2,093 Instagram influencers** (under 1M followers) using Python, data engineering techniques, and AI-powered scoring. It was completed as part of the SR NEXT Self-Guided Internship Program over 4 weeks.

The project is split into two phases:

- **Phase 1** — Large-scale data collection, cleaning, and structuring
- **Phase 2** — Feature engineering, influence scoring, automation detection, and interactive dashboard

---

## Repository Structure

```
Instagram_2000_dataset/
│
├── Final_phase_1.xlsx          # Raw collected dataset (2,093 influencers)
├── Final_phase_2.xlsx          # Cleaned, scored & ranked dataset (5 sheets)
├── SR_NEXT_Phase2_Dashboard.html  # Interactive Plotly dashboard (open in browser)
├── phase2_analysis.py          # Full Python pipeline — cleaning, scoring, ranking
├── Phase_1_Report.pdf          # Week 1 written report
└── README.md                   # You are here
```

---

## Phase 1 — Data Collection

**Objective:** Collect structured data on 2,000+ Instagram influencers with under 1M followers.

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

### Challenges Faced
- Follower counts stored in mixed formats (`244.9K`, `1.5M`, plain integers) — required custom parsing
- Engagement rate stored as text (`<0.1%`, `>20%`, `5.2%`) — required symbol stripping and conversion
- 500+ inconsistent category names — required NLP-based grouping
- Duplicate profiles, merged cells, and missing contact data throughout

---

## Phase 2 — AI-Powered Analysis & Ranking

**Objective:** Clean the raw dataset, engineer features, score influencers, detect automation patterns, and visualise insights.

### Pipeline Overview

```
Raw Excel (Phase 1)
        ↓
  Data Cleaning
  (fix followers, ER, posting freq)
        ↓
  Feature Engineering
  (hashtag count, broad category, normalisation)
        ↓
  Composite Influence Score (0–100)
  (weighted formula across 4 features)
        ↓
  Automation Likelihood Detection
  (rule-based pattern analysis)
        ↓
  Ranking + Tier Assignment
        ↓
  Ranked Excel (5 sheets) + CSV + Dashboard
```

---

## Influence Score Formula

Each influencer receives a **composite score out of 100** calculated as:

```
Influence Score = (ER_Score × 0.40) + (Freq_Score × 0.25) + (Followers_Score × 0.20) + (Hash_Score × 0.15) × 100
```

| Feature | Weight | Reason |
|---------|--------|--------|
| Engagement Rate | 40% | Most important — measures real audience response |
| Posting Frequency | 25% | Consistency = active, serious influencer |
| Followers (log-scaled) | 20% | Reach matters but diminishing returns at high counts |
| Hashtag Diversity | 15% | Indicates content strategy effort |

> All features are normalised to a 0–1 scale before scoring. Followers use log scaling to reduce skew from mega-accounts.

---

## Automation Likelihood Detection

Accounts are flagged based on engagement and posting pattern anomalies:

| Risk Level | Condition | Reasoning |
|------------|-----------|-----------|
| 🔴 High | ER > 30% on account with >50K followers | Statistically impossible organically for large accounts |
| 🔴 High | Posts >25/month with ER < 2% | Bot scheduler posting with no real audience response |
| 🟡 Medium | ER > 20% on any account | Suspicious engagement level regardless of size |
| 🟡 Medium | Posts >20/month with ER < 4% | High frequency, below-average engagement |
| 🟢 Low | All other patterns | Normal organic engagement behaviour |

---

## Final Excel Output (5 Sheets)

| Sheet | Contents |
|-------|---------|
| Ranked Influencers | All 2,093 influencers ranked by Influence Score with tier & automation flag |
| Top By Category | Top 10 influencers per broad content category |
| Summary Stats | KPI dashboard, category breakdown, tier distribution |
| Automation Risk | All flagged accounts with specific reason per row |
| Legend | Explanation of every column, score, and colour |

---

## Interactive Dashboard

The `SR_NEXT_Phase2_Dashboard.html` file is a **fully interactive analytics dashboard** — open it in any browser, no installation needed.

**6 pages:**
- **Overview** — KPI cards, tier & automation donut charts, ER & frequency distributions, key insights
- **Rankings** — Top 20 bar chart + Top 100 scrollable table
- **Categories** — Count, avg score, avg ER by category + Top 10 per category with pill navigation
- **Engagement** — Scatter plot: Followers vs ER (log scale, dot size = Influence Score)
- **Automation Risk** — Flagged accounts with filter by High/Medium risk
- **Explorer** — Search + filter all 2,093 influencers by tier, risk level, category

---

## How to Run the Python Code

### Requirements
```bash
pip install pandas numpy openpyxl
```

### Run
```bash
python phase2_analysis.py
```

**Input:** `cleaned_influencers_sorted.xlsx`  
**Outputs:**
- `SR_NEXT_Phase2_Ranked_Influencers.csv` — ranked influencer list
- Console summary with top 10 influencers and dataset stats

---

## Key Results

| Metric | Value |
|--------|-------|
| Total Influencers Analysed | 2,093 |
| Average Influence Score | ~42 / 100 |
| Average Engagement Rate | ~4.2% |
| Median Followers | ~18,000 |
| Tier 1 (Top 10%) | 209 influencers |
| High Automation Risk | 196 accounts (9.4%) |
| Influencers with Contact Info | 1,596 |
| Broad Categories | 13 |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
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
| Phase 2 | AI-Powered Analysis & Ranking |

---

## Author

**SHRE RAAM P J**  
Independent Intern — SR NEXT  
[GitHub Profile](https://github.com/SHRE-RAAM-P-J)

---

*This project was completed as part of the SR NEXT Self-Guided Internship Program. All data collected is publicly available Instagram profile information.*