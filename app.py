import pandas as pd
import numpy as np
import re
import logging
import time
from datetime import timedelta
from itertools import combinations
import streamlit as st
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import os
import shutil

# --- Clear Streamlit Cache on Startup ---
def clear_streamlit_cache():
    cache_dir = ".streamlit/cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        st.info("✅ Streamlit cache cleared. Running fresh code.")
clear_streamlit_cache()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Config ---
CONFIG = {
    "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
    "bertrend": {"min_cluster_size": 3},
    "analysis": {"time_window": "48H"},
    "coordination_detection": {"threshold": 0.85, "max_features": 5000}
}

# --- Groq Setup ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    import groq
    client = groq.Groq(api_key=GROQ_API_KEY)
except Exception as e:
    logger.warning(f"Groq API key not found or client error: {e}")
    client = None

# --- URLs (CLEANED: no trailing spaces!) ---
CFA_LOGO_URL = "https://opportunities.codeforafrica.org/wp-content/uploads/sites/5/2015/11/1-Zq7KnTAeKjBf6eENRsacSQ.png"

# --- Helper Functions ---
def safe_llm_call(prompt, max_tokens=2048):
    if client is None:
        logger.warning("Groq client not initialized. LLM call skipped.")
        return None
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens
        )
        try:
            content = response.choices[0].message.content.strip()
        except (AttributeError, KeyError, TypeError):
            content = str(response)
        return content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None

def translate_text(text, target_lang="en"):
    if client is None:
        return text
    try:
        prompt = f"Translate the following text to {target_lang}:\n{text}"
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=512
        )
        if response and hasattr(response, 'choices'):
            return response.choices[0].message.content.strip()
        return text
    except Exception:
        return text

def infer_platform_from_url(url):
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return "Unknown"
    url = url.lower()
    platforms = {
        "tiktok.com": "TikTok", "facebook.com": "Facebook", "fb.watch": "Facebook",
        "twitter.com": "X", "x.com": "X", "youtube.com": "YouTube", "youtu.be": "YouTube",
        "instagram.com": "Instagram", "telegram.me": "Telegram", "t.me": "Telegram"
    }
    for key, val in platforms.items():
        if key in url:
            return val
    media_domains = ["nytimes.com", "bbc.com", "cnn.com", "reuters.com", "theguardian.com", "aljazeera.com", "lemonde.fr", "dw.com"]
    if any(domain in url for domain in media_domains):
        return "News/Media"
    return "Media"

def extract_original_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    cleaned = re.sub(r'^(RT|rt|QT|qt)\s+@\w+:\s*', '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'@\w+', '', cleaned).strip()
    cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned).strip()
    cleaned = re.sub(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', cleaned)
    cleaned = re.sub(r'\b\d{4}\b', '', cleaned)
    cleaned = re.sub(r"[\n\r\t]", " ", cleaned).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.lower()
    
def is_original_post(text):
    if pd.isna(text) or not isinstance(text, str):
        return False
    text_lower = text.strip().lower()
    return not (text_lower.startswith('rt @') or ' rt @' in text_lower)

def parse_timestamp_robust(timestamp):
    if pd.isna(timestamp):
        return pd.NaT
    ts_str = str(timestamp).strip()
    ts_str = re.sub(r'\s+GMT$', '', ts_str, flags=re.IGNORECASE)
    try:
        parsed = pd.to_datetime(ts_str, errors='coerce', utc=True, dayfirst=True)
        if pd.notna(parsed):
            return parsed
    except Exception:
        pass
    date_formats = [
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M',
        '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M',
        '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M',
        '%b %d, %Y %H:%M', '%d %b %Y %H:%M',
        '%A, %d %b %Y %H:%M:%S',
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'
    ]
    for fmt in date_formats:
        try:
            parsed = pd.to_datetime(ts_str, format=fmt, errors='coerce', utc=True)
            if pd.notna(parsed):
                return parsed
        except Exception:
            continue
    return pd.NaT

# --- Combine Datasets (now supports TikTok) ---
def combine_social_media_data(meltwater_df, civicsignals_df, tiktok_df=None):
    combined_dfs = []
    def get_col(df, cols):
        df_cols = [c.lower().strip() for c in df.columns]
        for col in cols:
            if col.lower() in df_cols:
                return df.iloc[:, df_cols.index(col.lower())]
        return pd.Series([np.nan]*len(df), index=df.index)
    
    if meltwater_df is not None and not meltwater_df.empty:
        mw = pd.DataFrame()
        mw['account_id'] = get_col(meltwater_df, ['influencer'])
        mw['content_id'] = get_col(meltwater_df, ['tweet id', 'post id'])
        mw['object_id'] = get_col(meltwater_df, ['hit sentence', 'opening text', 'headline'])
        mw['URL'] = get_col(meltwater_df, ['url'])
        mw_primary_dt = get_col(meltwater_df, ['date'])
        mw_alt_date = get_col(meltwater_df, ['alternate date format'])
        mw_time = get_col(meltwater_df, ['time'])
        if not mw_primary_dt.empty and len(mw_primary_dt)==len(meltwater_df):
            mw['timestamp_share'] = mw_primary_dt
        elif not mw_alt_date.empty and not mw_time.empty and len(mw_alt_date)==len(meltwater_df):
            mw['timestamp_share'] = mw_alt_date.astype(str)+' '+mw_time.astype(str)
        else:
            mw['timestamp_share'] = mw_alt_date
        mw['source_dataset'] = 'Meltwater'
        combined_dfs.append(mw)
    
    if civicsignals_df is not None and not civicsignals_df.empty:
        cs = pd.DataFrame()
        cs['account_id'] = get_col(civicsignals_df, ['author', 'username', 'user'])
        cs['content_id'] = get_col(civicsignals_df, ['post_id', 'id', 'content_id'])
        cs['object_id'] = get_col(civicsignals_df, ['text', 'content', 'body', 'message'])
        cs['URL'] = get_col(civicsignals_df, ['url', 'link', 'post_url'])
        cs['timestamp_share'] = get_col(civicsignals_df, ['timestamp', 'date', 'created_at'])
        cs['source_dataset'] = 'Civicsignal'
        combined_dfs.append(cs)
    
    if tiktok_df is not None and not tiktok_df.empty:
        tt = pd.DataFrame()
        tt['account_id'] = get_col(tiktok_df, ['authorMeta.name', 'username', 'creator', 'author'])
        tt['content_id'] = get_col(tiktok_df, ['id', 'video_id', 'post_id', 'itemId'])
        tt['object_id'] = get_col(tiktok_df, ['text', 'caption', 'description', 'content'])
        tt['URL'] = get_col(tiktok_df, ['webVideoUrl', 'link', 'video_url', 'url'])
        tt['timestamp_share'] = get_col(tiktok_df, ['createTimeISO', 'timestamp', 'date', 'created_time', 'createTime'])
        tt['source_dataset'] = 'TikTok'
        combined_dfs.append(tt)

    if not combined_dfs:
        return pd.DataFrame()
    return pd.concat(combined_dfs, ignore_index=True)

def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    if df.empty:
        return pd.DataFrame(columns=[
            'account_id','content_id','object_id','URL','timestamp_share',
            'Platform','original_text','Outlet','Channel','cluster',
            'source_dataset','Sentiment'
        ])
    df_processed = df.copy()
    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan','').fillna('')
    df_processed = df_processed[df_processed['object_id'].str.strip()!=""].copy()
    if coordination_mode=="Text Content":
        df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    else:
        df_processed['original_text'] = df_processed['URL'].astype(str).replace('nan','').fillna('')
    df_processed = df_processed[df_processed['original_text'].str.strip()!=""].reset_index(drop=True)
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)
    df_processed['Outlet'] = np.nan
    df_processed['Channel'] = np.nan
    df_processed['cluster'] = -1
    if 'Sentiment' not in df_processed.columns:
        df_processed['Sentiment'] = np.nan
    columns_to_keep = ['account_id','content_id','object_id','URL','timestamp_share',
                       'Platform','original_text','Outlet','Channel','cluster',
                       'source_dataset','Sentiment']
    df_processed = df_processed[[c for c in columns_to_keep if c in df_processed.columns]].copy()
    return df_processed
    
@st.cache_data(show_spinner=False)
def cached_clustering(df, eps, min_samples, max_features, data_source_key):
    if df.empty or 'original_text' not in df.columns:
        return pd.DataFrame()
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(3,5), max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(df['original_text'])
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    df = df.copy()
    df['cluster'] = clustering.fit_predict(tfidf_matrix)
    return df

def assign_virality_tier(post_count):
    if post_count>=500:
        return "Tier 4: Viral Emergency"
    elif post_count>=100:
        return "Tier 3: High Spread"
    elif post_count>=20:
        return "Tier 2: Moderate"
    else:
        return "Tier 1: Limited"

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Summarize Cluster ---
def summarize_cluster(texts, urls, cluster_data, min_ts, max_ts):
    joined = "\n".join(texts[:50])
    url_context = "\nRelevant post links:\n" + "\n".join(urls[:5]) if urls else ""
    prompt = f"""
Generate a structured IMI intelligence report on online narratives related to election.
Focus on pre and post election tensions and emerging narratives, including:
- Allegations of political suppression
- Electoral Commission corruption or bias
- Economic distress or state fund misuse
- Hate speech, tribalism, xenophobia
- Gender-based attacks
- Foreign interference ("Western puppet", anti-EU, etc.)
- Marginalization of minorities
- Claims of election fraud, rigging, tally center issues
- Calls for protests or civic resistance
- Viral slogans or hashtags

**Strict Instructions:**
- Only report claims **explicitly present** in the provided posts.
- Identify **originators**: accounts that first posted the core claim.
- Note **amplification**: how widely it spread.
- Do NOT invent, assume, or fact-check.
- Summarize clearly.

**Output Format:**
- **Narrative Title**: [Short title]
- **Core Claim(s)**: [Bullet points]
- **Originator(s)**: [Account IDs or "Unknown"]
- **Amplification**: [Total posts]
- **First Detected**: {min_ts}
- **Last Updated**: {max_ts}

Documents:
{joined}{url_context}
"""    
    response = safe_llm_call(prompt, max_tokens=2048)
    raw_summary = ""
    if response:
        try:
            raw_summary = response.strip()
        except Exception:
            raw_summary = str(response).strip()
    evidence_urls = re.findall(r"(https?://[^\s\)\]]+)", raw_summary)
    cleaned_summary = re.sub(r'\*\*Here is a concise.*?\*\*', '', raw_summary, flags=re.IGNORECASE | re.DOTALL)
    cleaned_summary = re.sub(r'\*\*Here are a few options.*?\*\*', '', cleaned_summary, flags=re.IGNORECASE | re.DOTALL)
    cleaned_summary = re.sub(r'"[^"]*"', '', cleaned_summary)
    cleaned_summary = cleaned_summary.strip()
    if evidence_urls:
        url_links = [f'<a href="{u}" target="_blank">{u}</a>' for u in evidence_urls[:5]]
        cleaned_summary += "<br><br>Sources: " + ", ".join(url_links)
    return cleaned_summary, evidence_urls

# --- Main App ---
# CLEANED URLs: no trailing spaces!
MELTWATER_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/Co%CC%82te_dIvoire_Sep_Oct16.csv"
CIVICSIGNALS_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/cote-d-ivoire-or-ivory-all-story-urls-20251019081557.csv"
TIKTOK_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/TIKTOK_cot_oct20%20-%20Sheet1.csv"

def main():
    st.set_page_config(layout="wide", page_title="Côte d’Ivoire Election Monitoring Dashboard")
    
    col_logo, col_title = st.columns([1,5])
    with col_logo:
        st.image(CFA_LOGO_URL, width=120)
    with col_title:
        st.markdown("## 🇨🇮 Côte d’Ivoire Election Monitoring Dashboard")

    # --- Load datasets ---
    with st.spinner("📥 Loading Meltwater data..."):
        meltwater_df = pd.DataFrame()
        try:
            meltwater_df = pd.read_csv(MELTWATER_URL, sep='\t', low_memory=False, on_bad_lines='skip')
            logger.info("Meltwater loaded with default encoding")
        except Exception as e:
            try:
                meltwater_df = pd.read_csv(MELTWATER_URL, encoding='utf-16', sep='\t', low_memory=False, on_bad_lines='skip')
                logger.info("Meltwater loaded with utf-16")
            except Exception as e:
                st.error(f"❌ Meltwater failed to load: {e}")

    civicsignals_df = pd.DataFrame()
    if CIVICSIGNALS_URL:
        with st.spinner("📥 Loading Civicsignal data..."):
            try:
                civicsignals_df = pd.read_csv(CIVICSIGNALS_URL, low_memory=False, on_bad_lines='skip')
                logger.info(f"Civicsignal loaded: {civicsignals_df.shape}")
            except Exception as e:
                logger.warning(f"Civicsignal load failed: {e}")

    tiktok_df = pd.DataFrame()
    if TIKTOK_URL:
        with st.spinner("📥 Loading TikTok data..."):
            try:
                tiktok_df = pd.read_csv(TIKTOK_URL, low_memory=False, on_bad_lines='skip')
                logger.info(f"TikTok loaded: {tiktok_df.shape}")
            except Exception as e:
                logger.warning(f"TikTok load failed: {e}")

    combined_raw_df = combine_social_media_data(meltwater_df, civicsignals_df, tiktok_df)
    if combined_raw_df.empty:
        st.error("❌ No data after combining datasets.")
        st.stop()

    # --- Preprocessing ---
    df_full = final_preprocess_and_map_columns(combined_raw_df, coordination_mode="Text Content")
    if df_full.empty:
        st.error("❌ No valid data after preprocessing.")
        st.stop()
    df_full['timestamp_share'] = df_full['timestamp_share'].apply(parse_timestamp_robust)

    # Original posts only
    df_original = df_full[df_full['object_id'].apply(is_original_post)].copy()
    if df_original.empty:
        st.warning("⚠️ No original posts found. Coordination analysis may be limited.")

    # --- Date filter ---
    valid_dates = df_full['timestamp_share'].dropna()
    if valid_dates.empty:
        st.error("❌ No valid timestamps found in the dataset.")
        st.stop()
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    selected_date_range = st.sidebar.date_input(
        "Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date
    )
    if len(selected_date_range) == 2:
        start_date = pd.Timestamp(selected_date_range[0], tz='UTC')
        end_date = pd.Timestamp(selected_date_range[1], tz='UTC') + pd.Timedelta(days=1)
    else:
        start_date = pd.Timestamp(selected_date_range[0], tz='UTC')
        end_date = start_date + pd.Timedelta(days=1)

    filtered_df_global = df_full[(df_full['timestamp_share'] >= start_date) & (df_full['timestamp_share'] < end_date)].copy()
    filtered_original = df_original[
        (df_original['timestamp_share'] >= start_date) & 
        (df_original['timestamp_share'] < end_date)
    ].copy() if not df_original.empty else pd.DataFrame()

    df_clustered = cached_clustering(filtered_original, eps=0.3, min_samples=2, max_features=5000, data_source_key="report") if not filtered_original.empty else pd.DataFrame()

    # --- Top clusters ---
    top_15_clusters = []
    if 'cluster' in df_clustered.columns and not df_clustered.empty:
        cluster_sizes = df_clustered[df_clustered['cluster'] != -1].groupby('cluster').size()
        top_15_clusters = cluster_sizes.nlargest(15).index.tolist()

    # --- Enhanced Summaries ---
    all_summaries = []
    for cluster_id in top_15_clusters:
        original_cluster = df_clustered[df_clustered['cluster'] == cluster_id]
        original_urls = original_cluster['URL'].dropna().unique().tolist()
        originators = original_cluster['account_id'].dropna().unique().tolist()

        if not df_full.empty and original_urls:
            all_matching_posts = df_full[df_full['URL'].isin(original_urls)]
            all_texts = all_matching_posts['object_id'].astype(str).apply(extract_original_text).tolist()
            all_texts = [t for t in all_texts if len(t.strip()) > 10]
            amplifiers = all_matching_posts['account_id'].dropna().unique().tolist()
            total_reach = len(all_matching_posts)
        else:
            all_texts = original_cluster['original_text'].tolist()
            amplifiers = []
            total_reach = len(original_cluster)

        if not all_texts:
            continue

        min_ts_str = original_cluster['timestamp_share'].min().strftime('%Y-%m-%d')
        max_ts_str = original_cluster['timestamp_share'].max().strftime('%Y-%m-%d')
        joined = "\n".join(all_texts[:100])
        url_context = "\nRelevant post links:\n" + "\n".join(original_urls[:5]) if original_urls else ""

        prompt = f"""
You are an IMI election monitoring analyst. Generate a structured intelligence report on online narratives.

Focus on:
- Allegations of political suppression
- Electoral Commission corruption or bias
- Economic distress or state fund misuse
- Hate speech, tribalism, xenophobia
- Gender-based attacks
- Foreign interference ("Western puppet", anti-EU, etc.)
- Marginalization of minorities
- Claims of election fraud, rigging, tally center issues
- Calls for protests or civic resistance
- Viral slogans or hashtags

**Strict Instructions:**
- Only report claims **explicitly present** in the provided posts.
- Identify **originators**: accounts that first posted the core claim.
- Note **amplification**: how widely it spread.
- Do NOT invent, assume, or fact-check.
- Summarize clearly.

**Output Format:**
- **Narrative Title**: [Short title]
- **Core Claim(s)**: [Bullet points]
- **Originator(s)**: [Account IDs or "Unknown"]
- **Amplification**: [Total posts]
- **First Detected**: {min_ts_str}
- **Last Updated**: {max_ts_str}

Documents:
{joined}{url_context}
"""
        raw_response = safe_llm_call(prompt, max_tokens=2048)
        if not raw_response:
            raw_response = "⚠️ Summary generation failed due to API error."
        cleaned_summary = raw_response.strip()
        virality = assign_virality_tier(total_reach)

        all_summaries.append({
            "cluster_id": cluster_id,
            "Context": cleaned_summary,
            "Originators": ", ".join([str(a) for a in originators[:5]]) if originators else "Unknown",
            "Amplifiers_Count": len(amplifiers),
            "Total_Reach": total_reach,
            "Emerging Virality": virality,
        })

    # --- Metrics ---
    total_posts = len(df_full)
    valid_clusters_count = len(top_15_clusters)
    top_platform = df_full['Platform'].mode()[0] if not df_full['Platform'].mode().empty else "—"
    high_virality_count = len([s for s in all_summaries if "Tier 4" in s.get("Emerging Virality","")])
    last_update_time = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')

    # Tabs
    tabs = st.tabs([
        "🏠 Dashboard Overview",
        "📈 Data Insights",
        "🔍 Coordination Analysis",
        "⚠️ Risk Assessment",
        "📰 Trending Narratives"
    ])
    
    # TAB 0: Dashboard Overview
    with tabs[0]:
        st.markdown(f"""
        This dashboard provides **daily monitoring of trending narratives** related to the 2025 elections in Côte d’Ivoire.
        The primary purpose is to support **transparent, evidence-based election observation** by:
        1. **Detecting Emerging Narratives**
        2. **Tracking Virality**
        3. **Providing Evidence**
        Data is updated daily. Last updated: **{last_update_time}**
        """)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Posts Analyzed", f"{total_posts:,}")
        col2.metric("Active Narratives", valid_clusters_count)
        col3.metric("Top Platform", top_platform)
        col4.metric("Alert Level", "🚨 High" if high_virality_count > 5 else "⚠️ Medium" if high_virality_count > 0 else "✅ Low")
    
    # TAB 1: Data Insights
    with tabs[1]:
        st.markdown("### 🔬 Data Insights")
        st.markdown(f"**Total Rows:** `{len(filtered_df_global):,}` | **Date Range:** {selected_date_range[0]} to {selected_date_range[-1]}")
        if not filtered_df_global.empty:
            top_influencers = filtered_df_global['account_id'].value_counts().head(10)
            fig_src = px.bar(top_influencers, title="Top 10 Influencers")
            st.plotly_chart(fig_src, use_container_width=True, key="top_influencers")
            platform_counts = filtered_df_global['Platform'].value_counts()
            fig_platform = px.bar(platform_counts, title="Post Distribution by Platform")
            st.plotly_chart(fig_platform, use_container_width=True, key="platform_dist")
            social_media_df = filtered_df_global[~filtered_df_global['Platform'].isin(['Media', 'News/Media'])].copy()
            if not social_media_df.empty and 'object_id' in social_media_df.columns:
                social_media_df['hashtags'] = social_media_df['object_id'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
                all_hashtags = [tag for tags_list in social_media_df['hashtags'] if isinstance(tags_list, list) for tag in tags_list]
                if all_hashtags:
                    hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                    fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags (Social Media Only)", labels={'value': 'Frequency', 'index': 'Hashtag'})
                    st.plotly_chart(fig_ht, use_container_width=True, key="top_hashtags")
                    st.markdown("**Top 10 Hashtags (Social Media Only)**")
            plot_df = filtered_df_global.copy()
            plot_df = plot_df.set_index('timestamp_share')
            time_series = plot_df.resample('D').size()
            fig_ts = px.area(time_series, title="Daily Post Volume")
            st.plotly_chart(fig_ts, use_container_width=True, key="daily_volume")
    
    # TAB 2: Coordination Analysis
    with tabs[2]:
        coordination_groups = []
        if 'cluster' in df_clustered.columns:
            from collections import defaultdict
            grouped = df_clustered[df_clustered['cluster'] != -1].groupby('cluster')
            for cluster_id, group in grouped:
                if len(group) < 2:
                    continue
                clean_df = group[['account_id', 'timestamp_share', 'Platform', 'URL', 'original_text']].copy()
                clean_df = clean_df.rename(columns={'original_text': 'text'})
                vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(3, 5), max_features=5000)
                try:
                    tfidf_matrix = vectorizer.fit_transform(clean_df['text'])
                    cosine_sim = cosine_similarity(tfidf_matrix)
                    adj = defaultdict(list)
                    for i in range(len(clean_df)):
                        for j in range(i + 1, len(clean_df)):
                            if cosine_sim[i, j] >= 0.85:
                                adj[i].append(j)
                                adj[j].append(i)
                    visited = set()
                    for i in range(len(clean_df)):
                        if i not in visited:
                            group_indices = []
                            q = [i]
                            visited.add(i)
                            while q:
                                u = q.pop(0)
                                group_indices.append(u)
                                for v in adj[u]:
                                    if v not in visited:
                                        visited.add(v)
                                        q.append(v)
                            if len(group_indices) > 1 and len(clean_df.iloc[group_indices]['account_id'].unique()) > 1:
                                max_sim = round(cosine_sim[np.ix_(group_indices, group_indices)].max(), 3)
                                num_accounts = len(clean_df.iloc[group_indices]['account_id'].unique())
                                if max_sim > 0.95:
                                    coord_type = "High Text Similarity"
                                elif num_accounts >= 3:
                                    coord_type = "Multi-Account Amplification"
                                else:
                                    coord_type = "Potential Coordination"
                                coordination_groups.append({
                                    "posts": clean_df.iloc[group_indices].to_dict('records'),
                                    "num_posts": len(group_indices),
                                    "num_accounts": num_accounts,
                                    "max_similarity_score": max_sim,
                                    "coordination_type": coord_type
                                })
                except Exception:
                    continue
        if coordination_groups:
            st.success(f"Found {len(coordination_groups)} coordinated groups.")
            for i, group in enumerate(coordination_groups):
                st.markdown(f"### Group {i+1}: {group['coordination_type']}")
                st.write(f"**Posts:** {group['num_posts']} | **Accounts:** {group['num_accounts']} | **Max similarity:** {group['max_similarity_score']}")
                posts_df = pd.DataFrame(group['posts'])
                posts_df['Timestamp'] = posts_df['timestamp_share']
                posts_df['URL'] = posts_df['URL'].apply(
                    lambda x: f'<a href="{x}" target="_blank">{x}</a>' if pd.notna(x) else ""
                )
                st.markdown(posts_df.to_html(escape=False, index=False, columns=['account_id', 'Platform', 'Timestamp', 'URL']), unsafe_allow_html=True)
        else:
            st.info("No coordinated groups found.")

    # TAB 3: Risk Assessment
    with tabs[3]:
        st.subheader("⚠️ Risk & Influence Assessment")
        st.markdown("""
        This tab ranks accounts by **coordination activity**.
        High-risk accounts are potential **amplifiers or originators**.
        """)
        if df_clustered.empty or 'cluster' not in df_clustered.columns:
            st.info("No data available for risk assessment.")
        else:
            clustered_accounts = df_clustered[df_clustered['cluster'] != -1].dropna(subset=['account_id'])
            account_risk = clustered_accounts.groupby('account_id').size().reset_index(name='Coordination_Count')
            account_risk = account_risk.merge(
                df_clustered[['account_id', 'Platform']].drop_duplicates(subset=['account_id']),
                on='account_id',
                how='left'
            )
            account_risk = account_risk.sort_values('Coordination_Count', ascending=False).head(20)
            if account_risk.empty:
                st.info("No high-risk accounts detected.")
            else:
                st.markdown("#### Top 20 Accounts by Coordination Activity")
                st.dataframe(account_risk, use_container_width=True)
                risk_csv = convert_df_to_csv(account_risk)
                st.download_button(
                    "📥 Download Risk Assessment CSV",
                    risk_csv,
                    "risk_assessment.csv",
                    "text/csv"
                )

    # TAB 4: Trending Narratives (Interactive Cards)
    with tabs[4]:
        st.subheader("📖 Trending Narrative Summaries")
        
        if not all_summaries:
            st.info("No narrative summaries available.")
        else:
            all_reaches = [s["Total_Reach"] for s in all_summaries if s["Total_Reach"] > 0]
            median_reach = np.median(all_reaches) if all_reaches else 1
            sorted_summaries = sorted(all_summaries, key=lambda x: x["Total_Reach"], reverse=True)

            for summary in sorted_summaries:
                cluster_id = summary["cluster_id"]
                total_reach = summary["Total_Reach"]
                if total_reach < 10:
                    continue

                original_cluster = df_clustered[df_clustered['cluster'] == cluster_id]
                original_urls = original_cluster['URL'].dropna().unique().tolist()
                all_matching_posts = df_full[df_full['URL'].isin(original_urls)] if not df_full.empty and original_urls else original_cluster

                # Platform distribution for header
                platform_dist = all_matching_posts['Platform'].value_counts()
                top_platforms = ", ".join([f"{p} ({c})" for p, c in platform_dist.head(2).items()])
                relative_virality = total_reach / median_reach if median_reach > 0 else 1.0

                # Expander title includes originators and key info
                virality_emoji = "🔥" if "Tier 4" in summary['Emerging Virality'] else "📢" if "Tier 3" in summary['Emerging Virality'] else "💬"
                originators_display = summary['Originators'] if summary['Originators'] != "Unknown" else "Unknown originator(s)"
                card_title = f"{virality_emoji} Cluster {cluster_id} · {summary['Emerging Virality']} · {originators_display}"

                with st.expander(card_title, expanded=False):
                    # Clean narrative details in plain text
                    st.markdown(f"**Amplification:** {total_reach} posts ({relative_virality:.1f}x median activity)")
                    st.markdown(f"**Platforms:** {top_platforms}")
                    st.markdown(f"**First Detected:** {original_cluster['timestamp_share'].min().strftime('%Y-%m-%d')}")
                    st.markdown(f"**Last Updated:** {original_cluster['timestamp_share'].max().strftime('%Y-%m-%d')}")
                    
                    st.markdown("### Summary")
                    st.markdown(summary['Context'], unsafe_allow_html=True)

                    # Timeline chart (with unique key and future-proof width)
                    if not all_matching_posts.empty and 'timestamp_share' in all_matching_posts.columns:
                        timeline_df = all_matching_posts[['timestamp_share']].copy().dropna()
                        if not timeline_df.empty:
                            timeline_df = timeline_df.set_index('timestamp_share').resample('6H').size().reset_index(name='count')
                            if timeline_df['count'].sum() > 0:
                                fig = px.bar(
                                    timeline_df, x='timestamp_share', y='count',
                                    height=200,
                                    labels={'timestamp_share': 'Time', 'count': 'Posts'}
                                )
                                fig.update_layout(
                                    showlegend=False,
                                    margin=dict(l=0, r=0, t=30, b=0),
                                    xaxis_title=None,
                                    yaxis_title=None
                                )
                                st.plotly_chart(fig, width='stretch', key=f"timeline_{cluster_id}")

                    # Sample posts
                    if not all_matching_posts.empty:
                        st.markdown("### Sample Posts")
                        sample_posts = all_matching_posts[['account_id', 'Platform', 'object_id']].head(3)
                        st.dataframe(sample_posts, use_container_width=True, hide_index=True)
    # Global download
    report_df = pd.DataFrame(all_summaries)
    csv_data = convert_df_to_csv(report_df)
    st.download_button("📥 Download Full Report (CSV)", csv_data, "Trending_narrative_report.csv", "text/csv")
    
if __name__ == '__main__':
    main()
