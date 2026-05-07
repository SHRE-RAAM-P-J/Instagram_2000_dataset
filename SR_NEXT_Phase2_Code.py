# =============================================================================
# SR NEXT Self-Guided Internship — Phase 2
# Streamlit Dashboard: AI-Powered Instagram Influencer Analytics
# Student: SHRE RAAM P J
# =============================================================================
# HOW TO RUN:
#   pip install streamlit pandas numpy plotly scikit-learn openpyxl
#   streamlit run streamlit_app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIG — must be the very first Streamlit command
# =============================================================================
st.set_page_config(
    page_title="SR NEXT — Influencer Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS — dark professional theme
# =============================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #e6edf3; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
    }

    /* Headers */
    h1, h2, h3 { color: #e6edf3 !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border-radius: 8px; }

    /* Divider */
    hr { border-color: #30363d; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom-color: #58a6ff !important;
    }

    /* Info boxes */
    .insight-box {
        background-color: #1c2128;
        border-left: 4px solid #58a6ff;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
    }
    .insight-box.warn  { border-left-color: #d29922; }
    .insight-box.good  { border-left-color: #3fb950; }
    .insight-box.danger{ border-left-color: #f85149; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING & CLEANING — cached so it only runs once
# =============================================================================
@st.cache_data
def load_and_process():
    """Load raw Excel, clean it, engineer features, score and rank."""

    # ── Try to load the ranked CSV first (faster), else raw Excel ────────────
    import os

    # Search for CSV in multiple possible locations
    csv_paths = [
        'SR_NEXT_Phase2_Ranked_Influencers.csv',
        'data/SR_NEXT_Phase2_Ranked_Influencers.csv',
        'SR_NEXT_Phase2_ML_Results.csv',
        'data/SR_NEXT_Phase2_ML_Results.csv',
    ]
    for path in csv_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df

    # ── Load raw Excel and process ────────────────────────────────────────────
    excel_paths = [
        'data/Final_phase_2.xlsx',
        'Excel/Final_phase_2.xlsx',
        'Final_phase_2.xlsx',
        'data/cleaned_influencers_sorted.xlsx',
        'cleaned_influencers_sorted.xlsx',
    ]
    df = None
    for path in excel_paths:
        if os.path.exists(path):
            df = pd.read_excel(path)
            break
    if df is None:
        raise FileNotFoundError(
            "Could not find data file. Please place SR_NEXT_Phase2_Ranked_Influencers.csv "
            "or Final_phase_2.xlsx in the same folder as streamlit_app.py"
        )

    # Fix column name
    if 'Hastags' in df.columns:
        df.rename(columns={'Hastags': 'Hashtags'}, inplace=True)

    df = df.dropna(subset=['Username'])

    # Fix Followers
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
                return int(n)
            return np.nan

    # Fix Engagement Rate
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

    # Fix Posting Frequency
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

    df['Followers']       = df['Followers'].fillna(df['Followers'].median()).astype(int)
    df['Engagement Rate'] = df['Engagement Rate'].fillna(df['Engagement Rate'].median())
    df['Category']        = df['Category'].fillna('Unknown')
    df['Contact']         = df['Contact'].fillna('')
    df['Hashtags']        = df['Hashtags'].fillna('')
    df['Notes']           = df['Notes'].fillna('')

    df['Hashtag_Count'] = df['Hashtags'].apply(lambda v: len(re.findall(r'#\w+', str(v))))

    def broad_cat(cat):
        c = str(cat).lower()
        if any(x in c for x in ['fashion','beauty','style','skincare','makeup']): return 'Fashion & Beauty'
        if any(x in c for x in ['fitness','health','yoga','wellness','gym','sport']): return 'Health & Fitness'
        if any(x in c for x in ['food','cook','recipe','kitchen','drink']): return 'Food & Cooking'
        if any(x in c for x in ['travel','adventure','explore','tourism']): return 'Travel'
        if any(x in c for x in ['tech','gadget','software','digital','gaming']): return 'Technology & Gaming'
        if any(x in c for x in ['music','band','song','artist']): return 'Music'
        if any(x in c for x in ['art','illustrat','design','creative','photo']): return 'Art & Design'
        if any(x in c for x in ['entertainment','comedy','meme','funny']): return 'Entertainment'
        if any(x in c for x in ['education','exam','history','learn','news','media']): return 'Education & News'
        if any(x in c for x in ['business','market','brand','entrepreneur']): return 'Business'
        if any(x in c for x in ['lifestyle','motivat','inspir','faith']): return 'Lifestyle'
        if any(x in c for x in ['politic','government','social','activist']): return 'Politics & Social'
        return 'Other'

    df['Broad_Category'] = df['Category'].apply(broad_cat)

    df['Followers_Score'] = np.log1p(df['Followers']) / np.log1p(df['Followers'].max())
    er_cap = df['Engagement Rate'].quantile(0.99)
    df['ER_Score']   = df['Engagement Rate'].clip(upper=er_cap) / er_cap
    df['Freq_Score'] = df['Posts_Per_Month'].clip(upper=30) / 30
    df['Hash_Score'] = df['Hashtag_Count'].clip(upper=8) / 8

    df['Influence_Score'] = (
        df['ER_Score'] * 0.40 + df['Freq_Score'] * 0.25 +
        df['Followers_Score'] * 0.20 + df['Hash_Score'] * 0.15
    ) * 100
    df['Influence_Score'] = df['Influence_Score'].round(2)

    df['Rank'] = df['Influence_Score'].rank(ascending=False, method='min').astype(int)
    df = df.sort_values('Rank').reset_index(drop=True)

    total = len(df)
    def tier(r):
        p = r/total
        if p <= 0.10: return 'Tier 1 – Top 10%'
        elif p <= 0.30: return 'Tier 2 – Top 30%'
        elif p <= 0.60: return 'Tier 3 – Mid'
        return 'Tier 4 – Lower'
    df['Tier'] = df['Rank'].apply(tier)

    def auto_flag(row):
        er, f, freq = row['Engagement Rate'], row['Followers'], row['Posts_Per_Month']
        if er > 0.30 and f > 50000: return 'High'
        if freq > 25 and er < 0.02: return 'High'
        if er > 0.20 or (freq > 20 and er < 0.04): return 'Medium'
        return 'Low'
    df['Automation_Likelihood'] = df.apply(auto_flag, axis=1)

    return df


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def fmt_num(n):
    """Format number: 1500000 → 1.5M, 25000 → 25K"""
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(int(n))

TIER_COLORS = {
    'Tier 1 – Top 10%': '#d4a017',
    'Tier 2 – Top 30%': '#a8a9ad',
    'Tier 3 – Mid':     '#8b6914',
    'Tier 4 – Lower':   '#4a4a6a',
}
AUTO_COLORS = {'High': '#f85149', 'Medium': '#d29922', 'Low': '#3fb950'}

PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e6edf3', family='Arial'),
    xaxis=dict(gridcolor='#30363d', zerolinecolor='#30363d'),
    yaxis=dict(gridcolor='#30363d', zerolinecolor='#30363d'),
)


# =============================================================================
# LOAD DATA
# =============================================================================
with st.spinner("Loading and processing influencer data..."):
    df = load_and_process()


# =============================================================================
# SIDEBAR — Global Filters
# =============================================================================
with st.sidebar:
    st.markdown("## Global Filters")
    st.markdown("Filters apply to all pages")
    st.divider()

    # Category filter
    all_cats = ['All Categories'] + sorted(df['Broad_Category'].unique().tolist())
    sel_cat  = st.selectbox("Category", all_cats)

    # Tier filter
    all_tiers = ['All Tiers'] + sorted(df['Tier'].unique().tolist())
    sel_tier  = st.selectbox("Tier", all_tiers)

    # Automation filter
    sel_auto = st.selectbox("Automation Risk", ['All', 'Low', 'Medium', 'High'])

    # Follower range slider
    st.markdown("**Follower Range**")
    min_f, max_f = int(df['Followers'].min()), int(df['Followers'].max())
    follower_range = st.slider(
        "Followers", min_value=min_f, max_value=max_f,
        value=(min_f, max_f), format="%d"
    )

    # Engagement rate slider
    st.markdown("**Min Engagement Rate (%)**")
    min_er = st.slider("Min ER", 0.0, 20.0, 0.0, step=0.1)

    # Influence Score slider
    st.markdown("**Min Influence Score**")
    min_score = st.slider("Min Score", 0, 100, 0)

    st.divider()
    st.markdown("**SR NEXT Internship — Phase 2**")
    st.markdown("*SHRE RAAM P J*")

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df.copy()
if sel_cat  != 'All Categories': filtered = filtered[filtered['Broad_Category'] == sel_cat]
if sel_tier != 'All Tiers':      filtered = filtered[filtered['Tier'] == sel_tier]
if sel_auto != 'All':            filtered = filtered[filtered['Automation_Likelihood'] == sel_auto]
filtered = filtered[
    (filtered['Followers'] >= follower_range[0]) &
    (filtered['Followers'] <= follower_range[1]) &
    (filtered['Engagement Rate'] >= min_er/100) &
    (filtered['Influence_Score'] >= min_score)
]


# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown("# SR NEXT — Instagram Influencer Analytics")
st.markdown(f"**Phase 2 · AI-Powered Analysis · {len(filtered):,} influencers shown** (of {len(df):,} total)")
st.divider()


# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Rankings",
    "Categories",
    "Engagement",
    "Automation",
    "Explorer"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Dashboard Overview")

    # KPI row
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    empty = len(filtered) == 0
    c1.metric("Total Influencers",   f"{len(filtered):,}")
    c2.metric("Avg Influence Score", f"{filtered['Influence_Score'].mean():.1f}/100" if not empty else "—")
    c3.metric("Avg Engagement Rate", f"{filtered['Engagement Rate'].mean()*100:.2f}%" if not empty else "—")
    c4.metric("Median Followers",    fmt_num(int(filtered['Followers'].dropna().median())) if not empty else "—")
    c5.metric("Tier 1 Count",        len(filtered[filtered['Tier']=='Tier 1 – Top 10%']))
    c6.metric("High Auto Risk",      len(filtered[filtered['Automation_Likelihood']=='High']))
    c7.metric("Have Contact",        len(filtered[filtered['Contact']!='']))

    st.divider()

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        # Tier donut
        tier_counts = filtered['Tier'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        fig_tier = px.pie(
            tier_counts, names='Tier', values='Count', hole=0.45,
            color='Tier',
            color_discrete_map=TIER_COLORS,
            title='Tier Distribution'
        )
        fig_tier.update_layout(**PLOTLY_THEME, height=320)
        fig_tier.update_traces(textinfo='label+percent')
        st.plotly_chart(fig_tier, use_container_width=True)

    with col2:
        # Automation donut
        auto_counts = filtered['Automation_Likelihood'].value_counts().reset_index()
        auto_counts.columns = ['Risk', 'Count']
        fig_auto = px.pie(
            auto_counts, names='Risk', values='Count', hole=0.45,
            color='Risk',
            color_discrete_map=AUTO_COLORS,
            title='Automation Risk Breakdown'
        )
        fig_auto.update_layout(**PLOTLY_THEME, height=320)
        fig_auto.update_traces(textinfo='label+value+percent')
        st.plotly_chart(fig_auto, use_container_width=True)

    # Charts row 2
    col3, col4 = st.columns(2)

    with col3:
        # ER distribution
        er_data = filtered[filtered['Engagement Rate'] <= 1.0].copy()
        er_data['ER_Bucket'] = pd.cut(
            er_data['Engagement Rate'],
            bins=[0, 0.01, 0.03, 0.05, 0.10, 1.0],
            labels=['0-1%', '1-3%', '3-5%', '5-10%', '10%+']
        )
        er_counts = er_data['ER_Bucket'].value_counts().sort_index().reset_index()
        er_counts.columns = ['ER Range', 'Count']
        fig_er = px.bar(
            er_counts, x='ER Range', y='Count',
            title='Engagement Rate Distribution',
            color='Count', color_continuous_scale='Blues',
            text='Count'
        )
        fig_er.update_layout(**PLOTLY_THEME, height=300, showlegend=False)
        fig_er.update_traces(textposition='outside')
        st.plotly_chart(fig_er, use_container_width=True)

    with col4:
        # Posts/month distribution
        freq_bins  = [0, 5, 10, 15, 20, 30, 100]
        freq_labels = ['1-5', '6-10', '11-15', '16-20', '21-30', '30+']
        filtered_copy = filtered.copy()
        filtered_copy['Freq_Bucket'] = pd.cut(
            filtered_copy['Posts_Per_Month'], bins=freq_bins, labels=freq_labels
        )
        freq_counts = filtered_copy['Freq_Bucket'].value_counts().sort_index().reset_index()
        freq_counts.columns = ['Posts/Month', 'Count']
        fig_freq = px.bar(
            freq_counts, x='Posts/Month', y='Count',
            title='Posting Frequency Distribution',
            color_discrete_sequence=['#58a6ff'],
            text='Count'
        )
        fig_freq.update_layout(**PLOTLY_THEME, height=300)
        fig_freq.update_traces(textposition='outside')
        st.plotly_chart(fig_freq, use_container_width=True)

    # Key Insights
    st.markdown("### Key Insights")
    top1 = filtered.iloc[0] if len(filtered) > 0 else None
    if empty:
        st.warning('No influencers match the current filter combination. Try adjusting the sidebar filters.')
        st.stop()

    insights = [
        ("good",   f"**Top Performer:** {top1['Username'] if top1 is not None else '—'} leads with Influence Score {top1['Influence_Score'] if top1 is not None else '—'} and {fmt_num(int(top1['Followers'])) if top1 is not None else '—'} followers."),
        ("",       f"**Most Common Niche:** {filtered['Broad_Category'].mode()[0] if len(filtered)>0 else '—'} dominates with {filtered['Broad_Category'].value_counts().iloc[0] if len(filtered)>0 else 0} influencers."),
        ("warn",   f"**Automation Warning:** {len(filtered[filtered['Automation_Likelihood']=='High'])} accounts flagged as High risk — suspicious ER or posting patterns detected."),
        ("good",   f"**Contact Coverage:** {len(filtered[filtered['Contact']!='']):,} of {len(filtered):,} influencers have publicly available contact info."),
        ("danger", f"**Data Challenge:** Follower counts were in mixed formats (244.9K, 1.5M) — required custom parsing pipeline to convert accurately."),
    ]
    for style, text in insights:
        st.markdown(f'<div class="insight-box {style}">{text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RANKINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Influencer Rankings")
    st.caption("Ranked by composite Influence Score: ER×40% + Frequency×25% + Followers×20% + Hashtags×15%")

    top_n = st.slider("Show Top N influencers in chart", 10, 50, 20, step=5)
    top_data = filtered.head(top_n).copy()

    # Horizontal bar chart
    fig_rank = px.bar(
        top_data.sort_values('Influence_Score'),
        x='Influence_Score', y='Username',
        orientation='h',
        color='Influence_Score',
        color_continuous_scale='Viridis',
        hover_data=['Followers', 'Engagement Rate', 'Tier', 'Broad_Category'],
        title=f'Top {top_n} Influencers by Influence Score',
        text='Influence_Score'
    )
    fig_rank.update_layout(**PLOTLY_THEME, height=max(400, top_n * 22))
    fig_rank.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    st.plotly_chart(fig_rank, use_container_width=True)

    st.divider()

    # Table
    st.markdown(f"### Full Ranked Table ({len(filtered):,} influencers)")
    display_cols = ['Rank', 'Username', 'Followers', 'Broad_Category',
                    'Engagement Rate', 'Posts_Per_Month', 'Influence_Score',
                    'Tier', 'Automation_Likelihood', 'Contact']
    display_cols = [c for c in display_cols if c in filtered.columns]

    show_df = filtered[display_cols].copy()
    show_df['Followers']       = show_df['Followers'].apply(fmt_num)
    show_df['Engagement Rate'] = (show_df['Engagement Rate'] * 100).round(2).astype(str) + '%'
    show_df['Posts_Per_Month'] = show_df['Posts_Per_Month'].round(1)

    st.dataframe(show_df, use_container_width=True, height=450)
    st.download_button(
        "⬇Download Filtered Data as CSV",
        filtered[display_cols].to_csv(index=False),
        file_name="filtered_influencers.csv",
        mime="text/csv"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CATEGORIES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Category Analysis")

    cat_stats = filtered.groupby('Broad_Category').agg(
        Count=('Username', 'count'),
        Avg_ER=('Engagement Rate', 'mean'),
        Avg_Score=('Influence_Score', 'mean'),
        Avg_Followers=('Followers', 'mean'),
        Top_Influencer=('Username', 'first')
    ).reset_index().sort_values('Avg_Score', ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        fig_cat_count = px.bar(
            cat_stats.sort_values('Count', ascending=True),
            x='Count', y='Broad_Category', orientation='h',
            title='Influencer Count by Category',
            color='Count', color_continuous_scale='Blues',
            text='Count'
        )
        fig_cat_count.update_layout(**PLOTLY_THEME, height=380, showlegend=False)
        fig_cat_count.update_traces(textposition='outside')
        st.plotly_chart(fig_cat_count, use_container_width=True)

    with col2:
        fig_cat_score = px.bar(
            cat_stats.sort_values('Avg_Score', ascending=True),
            x='Avg_Score', y='Broad_Category', orientation='h',
            title='Avg Influence Score by Category',
            color='Avg_Score', color_continuous_scale='Viridis',
            text=cat_stats.sort_values('Avg_Score', ascending=True)['Avg_Score'].round(1)
        )
        fig_cat_score.update_layout(**PLOTLY_THEME, height=380, showlegend=False)
        fig_cat_score.update_traces(textposition='outside')
        st.plotly_chart(fig_cat_score, use_container_width=True)

    # Avg ER by category
    fig_cat_er = px.bar(
        cat_stats.sort_values('Avg_ER', ascending=False),
        x='Broad_Category', y='Avg_ER',
        title='Avg Engagement Rate by Category',
        color='Avg_ER', color_continuous_scale='RdYlGn',
        text=(cat_stats.sort_values('Avg_ER', ascending=False)['Avg_ER'] * 100).round(2).astype(str) + '%'
    )
    fig_cat_er.update_layout(**PLOTLY_THEME, height=320, showlegend=False)
    fig_cat_er.update_traces(textposition='outside')
    st.plotly_chart(fig_cat_er, use_container_width=True)

    # Category stats table
    st.markdown("### Category Summary Table")
    cat_display = cat_stats.copy()
    cat_display['Avg_ER']        = (cat_display['Avg_ER'] * 100).round(2).astype(str) + '%'
    cat_display['Avg_Score']     = cat_display['Avg_Score'].round(2)
    cat_display['Avg_Followers'] = cat_display['Avg_Followers'].apply(lambda x: fmt_num(int(x)))
    cat_display.columns          = ['Category', 'Count', 'Avg ER', 'Avg Score', 'Avg Followers', 'Top Influencer']
    st.dataframe(cat_display, use_container_width=True)

    # Top 10 per selected category
    st.divider()
    st.markdown("### Top 10 by Specific Category")
    sel_cat2 = st.selectbox(
        "Pick a category",
        sorted(filtered['Broad_Category'].unique().tolist()),
        key='cat_top10'
    )
    sub = filtered[filtered['Broad_Category'] == sel_cat2].head(10)
    sub_display = sub[['Rank','Username','Followers','Engagement Rate',
                        'Posts_Per_Month','Influence_Score','Tier','Contact']].copy()
    sub_display['Followers']       = sub_display['Followers'].apply(fmt_num)
    sub_display['Engagement Rate'] = (sub_display['Engagement Rate']*100).round(2).astype(str)+'%'
    st.dataframe(sub_display, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ENGAGEMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Engagement Analysis")

    # Scatter plot
    scatter_df = filtered[filtered['Engagement Rate'] <= 1.0].copy()
    scatter_df['ER_Pct'] = (scatter_df['Engagement Rate'] * 100).round(2)

    fig_scatter = px.scatter(
        scatter_df,
        x='Followers', y='ER_Pct',
        color='Broad_Category',
        size='Influence_Score',
        size_max=20,
        hover_name='Username',
        hover_data={'Followers': True, 'ER_Pct': True,
                    'Influence_Score': True, 'Tier': True},
        title='Followers vs Engagement Rate (dot size = Influence Score)',
        log_x=True,
        labels={'ER_Pct': 'Engagement Rate (%)', 'Followers': 'Followers (log scale)'}
    )
    fig_scatter.update_layout(**PLOTLY_THEME, height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('<div class="insight-box warn"><b>Inverse Trend:</b> As follower count increases, engagement rate generally drops — larger audiences are less intimate. Micro-influencers (5K–50K) typically show the strongest ER.</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box good"><b>Sweet Spot:</b> Accounts with 10K–100K followers and ER above 5% are the highest-value targets for brand partnerships.</div>', unsafe_allow_html=True)

    st.divider()

    # Influence Score distribution
    col1, col2 = st.columns(2)
    with col1:
        fig_score_hist = px.histogram(
            filtered, x='Influence_Score', nbins=30,
            title='Influence Score Distribution',
            color_discrete_sequence=['#58a6ff']
        )
        fig_score_hist.update_layout(**PLOTLY_THEME, height=300)
        st.plotly_chart(fig_score_hist, use_container_width=True)

    with col2:
        fig_er_hist = px.histogram(
            filtered[filtered['Engagement Rate'] <= 0.5],
            x='Engagement Rate', nbins=30,
            title='Engagement Rate Distribution',
            color_discrete_sequence=['#3fb950']
        )
        fig_er_hist.update_layout(**PLOTLY_THEME, height=300)
        st.plotly_chart(fig_er_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AUTOMATION RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## Automation Risk Analysis")
    st.caption("Accounts flagged based on ER anomalies and posting frequency patterns")

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 High Risk",   len(filtered[filtered['Automation_Likelihood']=='High']))
    col2.metric("🟡 Medium Risk", len(filtered[filtered['Automation_Likelihood']=='Medium']))
    col3.metric("🟢 Low Risk",    len(filtered[filtered['Automation_Likelihood']=='Low']))

    # Detection logic explanation
    with st.expander("How Automation is Detected"):
        st.markdown("""
| Risk Level | Condition | Reasoning |
|---|---|---|
| 🔴 **High** | ER > 30% on account with >50K followers | Statistically impossible organically |
| 🔴 **High** | Posts >25/month with ER < 2% | Bot scheduler, no real audience response |
| 🟡 **Medium** | ER > 20% on any account | Suspicious engagement for any account size |
| 🟡 **Medium** | Posts >20/month with ER < 4% | High frequency + below-average engagement |
| 🟢 **Low** | All other patterns | Normal organic behaviour |
        """)

    # ER vs Followers scatter coloured by risk
    fig_auto_scatter = px.scatter(
        filtered[filtered['Engagement Rate'] <= 1.0],
        x='Followers', y='Engagement Rate',
        color='Automation_Likelihood',
        color_discrete_map=AUTO_COLORS,
        hover_name='Username',
        title='Automation Risk — Followers vs Engagement Rate',
        log_x=True,
        labels={'Engagement Rate': 'Engagement Rate (decimal)'}
    )
    fig_auto_scatter.update_layout(**PLOTLY_THEME, height=420)
    st.plotly_chart(fig_auto_scatter, use_container_width=True)

    # Flagged table
    st.markdown("### Flagged Accounts")
    risk_filter = st.selectbox("Filter by risk level", ['All', 'High', 'Medium'], key='auto_risk')
    flagged = filtered[filtered['Automation_Likelihood'] != 'Low'].copy()
    if risk_filter != 'All':
        flagged = flagged[flagged['Automation_Likelihood'] == risk_filter]

    flagged_display = flagged[['Rank','Username','Followers','Engagement Rate',
                                'Posts_Per_Month','Influence_Score',
                                'Automation_Likelihood','Auto_Reason']].copy()
    flagged_display['Followers']       = flagged_display['Followers'].apply(fmt_num)
    flagged_display['Engagement Rate'] = (flagged_display['Engagement Rate']*100).round(2).astype(str)+'%'
    st.dataframe(flagged_display, use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## Influencer Explorer")
    st.caption("Search and explore all influencers in real time")

    # Search
    search = st.text_input("Search by username or category", "")

    exp_df = filtered.copy()
    if search:
        mask = (
            exp_df['Username'].str.contains(search, case=False, na=False) |
            exp_df['Broad_Category'].str.contains(search, case=False, na=False) |
            exp_df['Category'].str.contains(search, case=False, na=False)
        )
        exp_df = exp_df[mask]

    st.markdown(f"**{len(exp_df):,} influencers match your search**")

    # Display columns
    exp_display = exp_df[[
        'Rank','Username','Followers','Broad_Category','Category',
        'Engagement Rate','Posts_Per_Month','Influence_Score',
        'Tier','Automation_Likelihood','Contact'
    ]].copy()
    exp_display['Followers']       = exp_display['Followers'].apply(fmt_num)
    exp_display['Engagement Rate'] = (exp_display['Engagement Rate']*100).round(2).astype(str)+'%'
    exp_display['Posts_Per_Month'] = exp_display['Posts_Per_Month'].round(1)

    st.dataframe(exp_display, use_container_width=True, height=500)

    # Download filtered results
    st.download_button(
        "Download These Results as CSV",
        exp_df.to_csv(index=False),
        file_name=f"influencers_search_{search or 'all'}.csv",
        mime="text/csv"
    )