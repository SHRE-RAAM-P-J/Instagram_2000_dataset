# =============================================================================
# SR NEXT Self-Guided Internship — Phase 2
# END-TO-END PIPELINE — Run everything with one command
# Student: SHRE RAAM P J
#
# USAGE:
#   python run_pipeline.py                  → full pipeline + launch dashboard
#   python run_pipeline.py --no-dashboard   → full pipeline, skip dashboard
#   python run_pipeline.py --skip-ml        → skip ML models (faster)
# =============================================================================
# INSTALL:
#   pip install pandas numpy openpyxl scikit-learn matplotlib seaborn streamlit plotly
# =============================================================================

import sys
import os
import time
import re
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# ── Parse command line arguments ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description='SR NEXT Influencer Analytics Pipeline')
parser.add_argument('--no-dashboard', action='store_true', help='Skip launching Streamlit dashboard')
parser.add_argument('--skip-ml',      action='store_true', help='Skip ML model training')
args = parser.parse_args()


# =============================================================================
# HELPERS
# =============================================================================
def log(step, msg):
    """Print a clean progress log line."""
    print(f"\n{'='*60}")
    print(f"  STEP {step}: {msg}")
    print(f"{'='*60}")

def ok(msg):
    print(f"  {msg}")

def info(msg):
    print(f"  → {msg}")

def find_input_file():
    """Search for the input Excel file in known locations."""
    candidates = [
        'data/cleaned_influencers_sorted.xlsx',
        'data/Final_phase_1.xlsx',
        'cleaned_influencers_sorted.xlsx',
        'Final_phase_1.xlsx',
        'data/Final_phase_2.xlsx',
        'Excel/Final_phase_2.xlsx',
        'Final_phase_2.xlsx',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# =============================================================================
# STEP 1 — FIND INPUT FILE
# =============================================================================
log(1, "Locating input dataset")

input_file = find_input_file()
if input_file is None:
    print("\n   ERROR: Could not find input Excel file.")
    print("  Please make sure one of these files exists:")
    print("    • data/Final_phase_2.xlsx")
    print("    • data/cleaned_influencers_sorted.xlsx")
    sys.exit(1)

ok(f"Found input file: {input_file}")


# =============================================================================
# STEP 2 — LOAD & CLEAN DATA
# =============================================================================
log(2, "Loading and cleaning dataset")

start = time.time()
# Try reading normally first, then with header=1 if columns look wrong
df = pd.read_excel(input_file)
if 'Username' not in df.columns:
    # Formatted output Excel has a title banner row — skip it
    df = pd.read_excel(input_file, header=1)
    info("Detected formatted Excel — skipped title row, re-read with header=1")

# If still no Username column, try finding which row has it
if 'Username' not in df.columns:
    for skip in range(2, 6):
        df_try = pd.read_excel(input_file, header=skip)
        if 'Username' in df_try.columns:
            df = df_try
            info(f"Found headers at row {skip}")
            break

if 'Username' not in df.columns:
    print("   ERROR: Could not find 'Username' column in the file.")
    print(f"     Columns found: {df.columns.tolist()[:5]}")
    print("     Please use the raw Phase 1 Excel file, not the formatted output.")
    sys.exit(1)

info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
info(f"Columns: {df.columns.tolist()}")

# Rename hashtag column if needed
if 'Hastags' in df.columns:
    df.rename(columns={'Hastags': 'Hashtags'}, inplace=True)

df = df.dropna(subset=['Username'])

# ── Fix Followers ─────────────────────────────────────────────────────────────
def parse_followers(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().replace(',', '')
    try: return int(float(s))
    except:
        s = s.upper()
        m = re.match(r'([\d.]+)\s*([KMB]?)', s)
        if m:
            n, u = float(m.group(1)), m.group(2)
            if u == 'K': return int(n * 1_000)
            if u == 'M': return int(n * 1_000_000)
            if u == 'B': return int(n * 1_000_000_000)
            return int(n)
        return np.nan

# ── Fix Engagement Rate ───────────────────────────────────────────────────────
def parse_er(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    try: return float(s)
    except:
        c = s.replace('%','').replace('<','').replace('>','').strip()
        try:
            v = float(c)
            return v/100 if ('%' in s or v > 1) else v
        except: return np.nan

# ── Fix Posting Frequency ─────────────────────────────────────────────────────
def parse_freq(val):
    if pd.isna(val): return 8.0
    s = str(val).lower()
    if 'daily' in s: return 30.0
    w = re.search(r'(\d+)\s*posts?/week', s)
    if w: return round(int(w.group(1)) * 4.3, 1)
    nums = list(map(int, re.findall(r'\d+', s)))
    return round(sum(nums)/len(nums), 1) if nums else 8.0

df['Followers']       = df['Followers'].apply(parse_followers)
df['Engagement Rate'] = df['Engagement Rate'].apply(parse_er)
df['Posts_Per_Month'] = df['Posting Frequency'].apply(parse_freq)

nan_before = df['Followers'].isna().sum()
df['Followers']       = df['Followers'].fillna(df['Followers'].median()).astype(int)
df['Engagement Rate'] = df['Engagement Rate'].fillna(df['Engagement Rate'].median())
df['Category']        = df['Category'].fillna('Unknown')
df['Contact']         = df['Contact'].fillna('')
df['Hashtags']        = df['Hashtags'].fillna('')
df['Notes']           = df['Notes'].fillna('')

ok(f"Cleaned {len(df):,} influencers")
ok(f"Fixed {nan_before} NaN follower values")
ok(f"Followers range: {df['Followers'].min():,} – {df['Followers'].max():,}")
ok(f"ER range: {df['Engagement Rate'].min():.4f} – {df['Engagement Rate'].max():.4f}")


# =============================================================================
# STEP 3 — FEATURE ENGINEERING
# =============================================================================
log(3, "Feature Engineering")

# Hashtag count
df['Hashtag_Count'] = df['Hashtags'].apply(
    lambda v: len(re.findall(r'#\w+', str(v)))
)

# Broad category via NLP keyword classification
def broad_cat(cat):
    c = str(cat).lower()
    if any(x in c for x in ['fashion','beauty','style','skincare','makeup']): return 'Fashion & Beauty'
    if any(x in c for x in ['fitness','health','yoga','wellness','gym','sport']): return 'Health & Fitness'
    if any(x in c for x in ['food','cook','recipe','kitchen','drink','baking']): return 'Food & Cooking'
    if any(x in c for x in ['travel','adventure','explore','tourism']): return 'Travel'
    if any(x in c for x in ['tech','gadget','software','digital','gaming']): return 'Technology & Gaming'
    if any(x in c for x in ['music','band','song','artist']): return 'Music'
    if any(x in c for x in ['art','illustrat','design','creative','photo']): return 'Art & Design'
    if any(x in c for x in ['entertainment','comedy','meme','funny','film']): return 'Entertainment'
    if any(x in c for x in ['education','exam','history','learn','news','media']): return 'Education & News'
    if any(x in c for x in ['business','market','brand','entrepreneur']): return 'Business'
    if any(x in c for x in ['lifestyle','motivat','inspir','faith']): return 'Lifestyle'
    if any(x in c for x in ['politic','government','social','activist']): return 'Politics & Social'
    return 'Other'

df['Broad_Category'] = df['Category'].apply(broad_cat)

ok(f"Hashtag Count added (avg: {df['Hashtag_Count'].mean():.1f} per account)")
ok(f"Broad Category classified: {df['Broad_Category'].nunique()} categories from {df['Category'].nunique()} original labels")


# =============================================================================
# STEP 4 — INFLUENCE SCORING & RANKING
# =============================================================================
log(4, "Calculating Influence Scores and Rankings")

# Normalise features to 0-1
df['Followers_Score'] = np.log1p(df['Followers']) / np.log1p(df['Followers'].max())
er_cap = df['Engagement Rate'].quantile(0.99)
df['ER_Score']   = df['Engagement Rate'].clip(upper=er_cap) / er_cap
df['Freq_Score'] = df['Posts_Per_Month'].clip(upper=30) / 30
df['Hash_Score'] = df['Hashtag_Count'].clip(upper=8) / 8

# Composite Influence Score (0-100)
df['Influence_Score'] = (
    df['ER_Score']        * 0.40 +
    df['Freq_Score']      * 0.25 +
    df['Followers_Score'] * 0.20 +
    df['Hash_Score']      * 0.15
) * 100
df['Influence_Score'] = df['Influence_Score'].round(2)

# Rank
df['Rank'] = df['Influence_Score'].rank(ascending=False, method='min').astype(int)
df = df.sort_values('Rank').reset_index(drop=True)

total = len(df)
def assign_tier(r):
    p = r/total
    if p <= 0.10:  return 'Tier 1 – Top 10%'
    elif p <= 0.30: return 'Tier 2 – Top 30%'
    elif p <= 0.60: return 'Tier 3 – Mid'
    return 'Tier 4 – Lower'
df['Tier'] = df['Rank'].apply(assign_tier)

# Automation likelihood
def auto_flag(row):
    er, f, freq = row['Engagement Rate'], row['Followers'], row['Posts_Per_Month']
    if er > 0.30 and f > 50000: return 'High'
    if freq > 25 and er < 0.02: return 'High'
    if er > 0.20 or (freq > 20 and er < 0.04): return 'Medium'
    return 'Low'

def auto_reason(row):
    er, f, freq = row['Engagement Rate'], row['Followers'], row['Posts_Per_Month']
    if er > 0.30 and f > 50000: return 'Unusually high ER (>30%) for large account'
    if freq > 25 and er < 0.02: return 'Very high post frequency + near-zero ER'
    if er > 0.20:               return 'High ER – potential inorganic engagement'
    if freq > 20 and er < 0.04: return 'High frequency + below-average ER'
    return 'Normal organic pattern'

df['Automation_Likelihood'] = df.apply(auto_flag, axis=1)
df['Auto_Reason']           = df.apply(auto_reason, axis=1)

ok(f"Influence Scores calculated (range: {df['Influence_Score'].min():.1f} – {df['Influence_Score'].max():.1f})")
ok(f"Rankings assigned (1 to {len(df):,})")
ok(f"Tier distribution:")
for tier, cnt in df['Tier'].value_counts().items():
    info(f"  {tier}: {cnt} ({cnt/len(df)*100:.1f}%)")
ok(f"Automation flags: High={( df['Automation_Likelihood']=='High').sum()}, Medium={(df['Automation_Likelihood']=='Medium').sum()}, Low={(df['Automation_Likelihood']=='Low').sum()}")


# =============================================================================
# STEP 5 — ML MODELS (optional, skip with --skip-ml)
# =============================================================================
if not args.skip_ml:
    log(5, "Training ML Models")

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

        le = LabelEncoder()
        df['Category_Encoded'] = le.fit_transform(df['Broad_Category'].astype(str))

        FEATURES = ['Followers', 'Engagement Rate', 'Posts_Per_Month',
                    'Hashtag_Count', 'Category_Encoded']
        X = df[FEATURES].fillna(df[FEATURES].median())

        # ── Model 1: Automation Classifier ───────────────────────────────────
        y_class = df['Automation_Likelihood']
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
        )
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        cv  = cross_val_score(clf, X, y_class, cv=5).mean()

        df['ML_Automation_Prediction'] = clf.predict(X)
        df['ML_Automation_Confidence'] = clf.predict_proba(X).max(axis=1).round(3)

        ok(f"Automation Classifier trained — Accuracy: {acc*100:.2f}%, CV: {cv*100:.2f}%")

        # ── Model 2: ER Predictor ─────────────────────────────────────────────
        er_cap2 = df['Engagement Rate'].quantile(0.99)
        df_reg  = df[df['Engagement Rate'] <= er_cap2].copy()
        y_reg   = df_reg['Engagement Rate']
        X_reg   = df_reg[FEATURES].fillna(df_reg[FEATURES].median())

        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        reg.fit(Xr_tr, yr_tr)
        r2  = r2_score(yr_te, reg.predict(Xr_te))
        mae = mean_absolute_error(yr_te, reg.predict(Xr_te))

        df.loc[df_reg.index, 'ML_ER_Prediction'] = reg.predict(X_reg).round(4)

        ok(f"ER Predictor trained — R²: {r2:.4f}, MAE: {mae*100:.2f}%")

    except ImportError:
        print("     scikit-learn not installed — skipping ML models")
        print("     Run: pip install scikit-learn")
else:
    info("Skipping ML models (--skip-ml flag set)")


# =============================================================================
# STEP 6 — EXPORT OUTPUTS
# =============================================================================
log(6, "Exporting outputs")

os.makedirs('data', exist_ok=True)

# Ranked CSV
output_cols = [
    'Rank', 'Username', 'Followers', 'Broad_Category', 'Category',
    'Engagement Rate', 'Posts_Per_Month', 'Hashtag_Count',
    'Influence_Score', 'Tier', 'Automation_Likelihood', 'Auto_Reason', 'Contact',
    'Hashtags', 'Notes'
]
# Add ML columns if they exist
for col in ['ML_Automation_Prediction', 'ML_Automation_Confidence', 'ML_ER_Prediction']:
    if col in df.columns:
        output_cols.append(col)

output_cols = [c for c in output_cols if c in df.columns]
df[output_cols].to_csv('SR_NEXT_Phase2_Ranked_Influencers.csv', index=False)
ok(f"Ranked CSV saved: SR_NEXT_Phase2_Ranked_Influencers.csv ({len(df):,} rows)")


# =============================================================================
# STEP 7 — FINAL SUMMARY
# =============================================================================
log(7, "Pipeline Complete — Summary")

elapsed = time.time() - start
print(f"""
  ┌─────────────────────────────────────────────┐
  │         SR NEXT Phase 2 — Results           │
  ├─────────────────────────────────────────────┤
  │  Total Influencers      : {len(df):>8,}         │
  │  Avg Influence Score    : {df['Influence_Score'].mean():>8.2f} / 100    │
  │  Avg Engagement Rate    : {df['Engagement Rate'].mean()*100:>7.2f}%         │
  │  Median Followers       : {int(df['Followers'].median()):>8,}         │
  │  Tier 1 (Top 10%)       : {(df['Tier']=='Tier 1 – Top 10%').sum():>8,}         │
  │  High Automation Risk   : {(df['Automation_Likelihood']=='High').sum():>8,}         │
  │  With Contact Info      : {(df['Contact']!='').sum():>8,}         │
  ├─────────────────────────────────────────────┤
  │  ⏱  Pipeline ran in {elapsed:.1f}s                  │
  └─────────────────────────────────────────────┘

  Top 10 Influencers:
""")
print(df[['Rank','Username','Followers','Engagement Rate',
          'Influence_Score','Tier']].head(10).to_string(index=False))


# =============================================================================
# STEP 8 — LAUNCH STREAMLIT DASHBOARD
# =============================================================================
if not args.no_dashboard:
    if os.path.exists('streamlit_app.py'):
        print(f"\n{'='*60}")
        print("  STEP 8: Launching Streamlit Dashboard")
        print(f"{'='*60}")
        print("  → Opening dashboard at http://localhost:8501")
        print("  → Press Ctrl+C to stop the dashboard\n")
        os.system('python -m streamlit run streamlit_app.py')
    else:
        print("\n   streamlit_app.py not found — skipping dashboard launch")
        print("     Make sure streamlit_app.py is in the same folder")
else:
    print("\n  Dashboard launch skipped (--no-dashboard flag set)")
    print("  To launch manually: python -m streamlit run streamlit_app.py")