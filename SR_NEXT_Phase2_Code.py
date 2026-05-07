# =============================================================================
# SR NEXT Self-Guided Internship — Phase 2
# AI-Powered Instagram Influencer Analytics & Ranking
# Student: SHRE RAAM P J
# =============================================================================
# LIBRARIES USED:
#   pip install pandas openpyxl numpy
# =============================================================================

import pandas as pd
import numpy as np
import re
import json

# =============================================================================
# STEP 1 — LOAD THE DATASET
# =============================================================================
# Read the Excel file you prepared in Phase 1

df = pd.read_excel('cleaned_influencers_sorted.xlsx')

print("Dataset loaded!")
print(f"Shape: {df.shape}")          # rows x columns
print(f"Columns: {df.columns.tolist()}")


# =============================================================================
# STEP 2 — DATA CLEANING
# Fix messy values that cannot be used for calculations as-is
# =============================================================================

# ── 2a. Fix Followers ────────────────────────────────────────────────────────
# Problem: some followers are stored as text like "244.9K" or "1.5M"
# Solution: convert all to plain integers

def parse_followers(val):
    """Convert follower values like 244.9K or 1.5M to plain integers."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(',', '')
    try:
        # Already a plain number like 348724
        return int(float(s))
    except:
        s_upper = s.upper()
        match = re.match(r'([\d.]+)\s*([KMB]?)', s_upper)
        if match:
            number = float(match.group(1))
            unit   = match.group(2)
            if unit == 'K': return int(number * 1_000)
            if unit == 'M': return int(number * 1_000_000)
            if unit == 'B': return int(number * 1_000_000_000)
            return int(number)
        return np.nan

df['Followers'] = df['Followers'].apply(parse_followers)

print(f"\nFollowers NaN count after fixing: {df['Followers'].isna().sum()}")
print(f"Followers range: {int(df['Followers'].min()):,} — {int(df['Followers'].max()):,}")


# ── 2b. Fix Engagement Rate ──────────────────────────────────────────────────
# Problem: some ER values are text like "<0.1%", ">20%", "5.2%"
# Solution: strip symbols, convert to decimal form (e.g. 5.2% → 0.052)

def parse_engagement_rate(val):
    """Convert ER values like <0.1%, >20%, 5.2% or 0.0267 to decimal."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    try:
        return float(s)   # already a decimal like 0.0267
    except:
        # Remove <, >, % symbols
        cleaned = s.replace('%', '').replace('<', '').replace('>', '').strip()
        try:
            v = float(cleaned)
            # If it was a percentage (e.g. "5.2%"), convert to decimal
            if '%' in s or v > 1:
                return v / 100
            return v
        except:
            return np.nan

df['Engagement Rate'] = df['Engagement Rate'].apply(parse_engagement_rate)

print(f"\nEngagement Rate NaN count after fixing: {df['Engagement Rate'].isna().sum()}")
print(f"ER range: {df['Engagement Rate'].min():.4f} — {df['Engagement Rate'].max():.4f}")


# ── 2c. Fix Posting Frequency → numeric posts per month ─────────────────────
# Problem: stored as text like "~20-30/month", "3 posts/week", "Daily"
# Solution: parse and convert everything to a single number (posts per month)

def parse_posting_frequency(val):
    """Convert posting frequency text to average posts per month."""
    if pd.isna(val):
        return 8.0   # median default if missing
    s = str(val).strip().lower()
    if 'daily' in s:
        return 30.0
    # Handle "X posts/week" → multiply by 4.3
    weekly = re.search(r'(\d+)\s*posts?/week', s)
    if weekly:
        return round(int(weekly.group(1)) * 4.3, 1)
    # Handle ranges like "~20-30/month" → take average of 20 and 30
    numbers = list(map(int, re.findall(r'\d+', s)))
    if not numbers:
        return 8.0
    return round(sum(numbers) / len(numbers), 1)

df['Posts_Per_Month'] = df['Posting Frequency'].apply(parse_posting_frequency)

print(f"\nPosts Per Month sample: {df['Posts_Per_Month'].head(5).tolist()}")


# ── 2d. Fill remaining missing values ────────────────────────────────────────
df['Followers']       = df['Followers'].fillna(df['Followers'].median()).astype(int)
df['Engagement Rate'] = df['Engagement Rate'].fillna(df['Engagement Rate'].median())
df['Category']        = df['Category'].fillna('Unknown')
df['Contact']         = df['Contact'].fillna('')
df['Notes']           = df['Notes'].fillna('')


# =============================================================================
# STEP 3 — FEATURE ENGINEERING
# Create new columns that help measure influencer quality
# =============================================================================

# ── 3a. Hashtag Count ────────────────────────────────────────────────────────
# Count how many unique hashtags each influencer uses
# More hashtag diversity = better content strategy

hashtag_col = 'Hastags' if 'Hastags' in df.columns else 'Hashtags'
df[hashtag_col] = df[hashtag_col].fillna('')

def count_hashtags(val):
    """Count number of hashtags like #fitness #travel in a string."""
    return len(re.findall(r'#\w+', str(val)))

df['Hashtag_Count'] = df[hashtag_col].apply(count_hashtags)

print(f"\nHashtag Count sample: {df['Hashtag_Count'].head(5).tolist()}")


# ── 3b. Broad Category — NLP-based classification ───────────────────────────
# Problem: 500+ different category names (Fashion, Fashion & Beauty,
#          Fashionista, Style & Fashion etc.) — all mean the same thing
# Solution: group them into 13 broad categories using keyword matching
# This is a rule-based NLP technique called "keyword classification"

def classify_category(category):
    """
    Classify any category string into one of 13 broad groups
    using keyword matching (NLP — rule-based text classification).
    """
    c = str(category).lower()

    if any(word in c for word in ['fashion','beauty','style','skincare','makeup']):
        return 'Fashion & Beauty'
    if any(word in c for word in ['fitness','health','yoga','wellness','gym','sport']):
        return 'Health & Fitness'
    if any(word in c for word in ['food','cook','recipe','kitchen','drink','baking','cafe']):
        return 'Food & Cooking'
    if any(word in c for word in ['travel','adventure','explore','tourism']):
        return 'Travel'
    if any(word in c for word in ['tech','gadget','software','digital','gaming','game']):
        return 'Technology & Gaming'
    if any(word in c for word in ['music','band','song','artist']):
        return 'Music'
    if any(word in c for word in ['art','illustration','design','creative','photo']):
        return 'Art & Design'
    if any(word in c for word in ['entertainment','comedy','meme','funny','humor','film','movie']):
        return 'Entertainment'
    if any(word in c for word in ['education','exam','history','learn','news','media','science']):
        return 'Education & News'
    if any(word in c for word in ['business','marketing','brand','entrepreneur','finance']):
        return 'Business'
    if any(word in c for word in ['lifestyle','motivation','inspiration','faith','relationship']):
        return 'Lifestyle'
    if any(word in c for word in ['politics','government','social','activist']):
        return 'Politics & Social'

    return 'Other'

df['Broad_Category'] = df['Category'].apply(classify_category)

print(f"\nBroad Category distribution:")
print(df['Broad_Category'].value_counts())


# =============================================================================
# STEP 4 — NORMALISATION
# Scale all features to 0–1 range so they can be compared fairly
# (A follower count of 500,000 vs an ER of 0.05 can't be added directly)
# =============================================================================

# Followers: use log scale to reduce the gap between small and large accounts
# (difference between 1K and 10K matters more than 500K and 510K)
df['Followers_Score'] = np.log1p(df['Followers']) / np.log1p(df['Followers'].max())

# Engagement Rate: cap at 99th percentile to remove extreme outliers
er_cap = df['Engagement Rate'].quantile(0.99)
df['ER_Score'] = df['Engagement Rate'].clip(upper=er_cap) / er_cap

# Posting Frequency: cap at 30 posts/month
df['Freq_Score'] = df['Posts_Per_Month'].clip(upper=30) / 30

# Hashtag Count: cap at 8 hashtags
df['Hash_Score'] = df['Hashtag_Count'].clip(upper=8) / 8

print(f"\nNormalised score ranges (should all be 0.0 to 1.0):")
print(f"  Followers Score: {df['Followers_Score'].min():.2f} — {df['Followers_Score'].max():.2f}")
print(f"  ER Score:        {df['ER_Score'].min():.2f} — {df['ER_Score'].max():.2f}")
print(f"  Freq Score:      {df['Freq_Score'].min():.2f} — {df['Freq_Score'].max():.2f}")
print(f"  Hash Score:      {df['Hash_Score'].min():.2f} — {df['Hash_Score'].max():.2f}")


# =============================================================================
# STEP 5 — COMPOSITE INFLUENCE SCORE (0 to 100)
# Weighted formula combining all 4 normalised features
#
# Weights chosen based on influencer marketing standards:
#   Engagement Rate   → 40% (most important — measures real audience response)
#   Posting Frequency → 25% (consistency = active influencer)
#   Followers         → 20% (reach matters but less than engagement)
#   Hashtag Diversity → 15% (content strategy effort)
# =============================================================================

df['Influence_Score'] = (
    df['ER_Score']        * 0.40 +
    df['Freq_Score']      * 0.25 +
    df['Followers_Score'] * 0.20 +
    df['Hash_Score']      * 0.15
) * 100

df['Influence_Score'] = df['Influence_Score'].round(2)

print(f"\nInfluence Score stats:")
print(df['Influence_Score'].describe().round(2))


# =============================================================================
# STEP 6 — RANKING & TIERS
# =============================================================================

# Rank 1 = highest influence score
df['Rank'] = df['Influence_Score'].rank(ascending=False, method='min').astype(int)
df = df.sort_values('Rank').reset_index(drop=True)

total = len(df)

def assign_tier(rank):
    """Assign tier based on rank percentage."""
    percentile = rank / total
    if percentile <= 0.10:  return 'Tier 1 – Top 10%'
    elif percentile <= 0.30: return 'Tier 2 – Top 30%'
    elif percentile <= 0.60: return 'Tier 3 – Mid'
    else:                    return 'Tier 4 – Lower'

df['Tier'] = df['Rank'].apply(assign_tier)

print(f"\nTier distribution:")
print(df['Tier'].value_counts())


# =============================================================================
# STEP 7 — AUTOMATION LIKELIHOOD DETECTION
# Predict which accounts are likely using bots or automation tools
#
# Pattern logic (rule-based ML classifier):
#   HIGH risk:   ER > 30% on large account (>50K followers) — statistically
#                impossible organically for large accounts
#                OR posting 25+ times/month with near-zero ER — bot scheduler
#   MEDIUM risk: ER > 20% on any account — suspicious
#                OR high frequency posting with below-average ER
#   LOW risk:    Normal organic patterns
# =============================================================================

def detect_automation(row):
    """
    Classify automation likelihood based on engagement and posting patterns.
    Returns: 'High', 'Medium', or 'Low'
    """
    er       = float(row['Engagement Rate'])
    followers = int(row['Followers'])
    freq     = float(row['Posts_Per_Month'])

    # HIGH risk conditions
    if er > 0.30 and followers > 50000:
        return 'High'    # Unusually high ER for a large account
    if freq > 25 and er < 0.02:
        return 'High'    # Posting machine with almost no real engagement

    # MEDIUM risk conditions
    if er > 0.20:
        return 'Medium'  # High ER — suspicious on any account
    if freq > 20 and er < 0.04:
        return 'Medium'  # High frequency + below-average engagement

    return 'Low'

def automation_reason(row):
    """Return a human-readable reason for the automation flag."""
    er       = float(row['Engagement Rate'])
    followers = int(row['Followers'])
    freq     = float(row['Posts_Per_Month'])

    if er > 0.30 and followers > 50000:
        return 'Unusually high ER (>30%) for large account — likely fake engagement'
    if freq > 25 and er < 0.02:
        return 'Very high post frequency + near-zero ER — likely bot/scheduler'
    if er > 0.20:
        return 'High ER (>20%) — potential inorganic engagement'
    if freq > 20 and er < 0.04:
        return 'High posting frequency with below-average ER'
    return 'Normal organic engagement pattern'

df['Automation_Likelihood'] = df.apply(detect_automation, axis=1)
df['Auto_Reason']           = df.apply(automation_reason, axis=1)

print(f"\nAutomation Risk distribution:")
print(df['Automation_Likelihood'].value_counts())


# =============================================================================
# STEP 8 — EXPORT OUTPUTS
# =============================================================================

# ── 8a. Ranked CSV ────────────────────────────────────────────────────────────
output_cols = [
    'Rank', 'Username', 'Followers', 'Broad_Category', 'Category',
    'Engagement Rate', 'Posts_Per_Month', 'Hashtag_Count',
    'Influence_Score', 'Tier', 'Automation_Likelihood', 'Auto_Reason',
    'Contact', hashtag_col, 'Notes'
]

df_output = df[output_cols].copy()
df_output.to_csv('SR_NEXT_Phase2_Ranked_Influencers.csv', index=False)
print(f"\nRanked CSV saved: SR_NEXT_Phase2_Ranked_Influencers.csv")
print(f"Total influencers: {len(df_output)}")

# ── 8b. Summary Stats ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2 SUMMARY")
print("="*60)
print(f"Total Influencers Analysed : {len(df):,}")
print(f"Average Influence Score    : {df['Influence_Score'].mean():.2f} / 100")
print(f"Average Engagement Rate    : {df['Engagement Rate'].mean()*100:.2f}%")
print(f"Median Followers           : {int(df['Followers'].median()):,}")
print(f"Tier 1 (Top 10%)           : {(df['Tier']=='Tier 1 – Top 10%').sum()} influencers")
print(f"High Automation Risk       : {(df['Automation_Likelihood']=='High').sum()} influencers")
print(f"With Contact Info          : {(df['Contact']!='').sum()} influencers")

print("\nTop 10 Influencers:")
print(df[['Rank','Username','Followers','Engagement Rate','Influence_Score','Tier']].head(10).to_string(index=False))
