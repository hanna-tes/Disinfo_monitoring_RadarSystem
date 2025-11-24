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
import textwrap

# --- Clear Streamlit Cache on Startup ---
def clear_streamlit_cache():
    cache_dir = ".streamlit/cache"
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            st.info("✅ Streamlit cache cleared. Running fresh code.")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

clear_streamlit_cache()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Config ---
CONFIG = {
    "model_id": "llama-3.3-70b-versatile",
    "bertrend": {"min_cluster_size": 3},
    "analysis": {"time_window": "48H"},
    "coordination_detection": {"threshold": 0.85, "max_features": 5000}
}

# --- Groq Setup ---
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    if GROQ_API_KEY:
        import groq
        client = groq.Groq(api_key=GROQ_API_KEY)
    else:
        logger.warning("GROQ_API_KEY not found in st.secrets. LLM functions disabled.")
        client = None
except Exception as e:
    logger.warning(f"Groq client setup failed: {e}")
    client = None

# --- URLs ---
CFA_LOGO_URL = "https://opportunities.codeforafrica.org/wp-content/uploads/sites/5/2015/11/1-Zq7KnTAeKjBf6eENRsacSQ.png"
MELTWATER_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/Co%CC%82te_dIvoire_OR_Ivory_Coast_OR_Abidjan_OR_Ivoirien%20-%20Nov%205%2C%202025%20-%203%2058%2048%20PM.csv"
CIVICSIGNALS_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/cote-d-ivoire-or-ivory-all-story-urls-20251105131200.csv"
TIKTOK_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/TikTok_Oct_Nov.csv"
OPENMEASURES_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/open-measures-data%20(4).csv"

# --- Helper Functions ---
def load_data_robustly(url, name, default_sep=','):
    df = pd.DataFrame()
    if not url:
        return df
    attempts = [
        (',', 'utf-8'),
        (',', 'utf-8-sig'),
        ('\t', 'utf-8'),
        (';', 'utf-8'),
        ('\t', 'utf-16'),
        (',', 'latin-1'),
    ]
    for sep, enc in attempts:
        try:
            df = pd.read_csv(url, sep=sep, low_memory=False, on_bad_lines='skip', encoding=enc)
            if not df.empty and len(df.columns) > 1:
                logger.info(f"✅ {name} loaded successfully (Sep: '{sep}', Enc: '{enc}', Shape: {df.shape})")
                return df
        except Exception:
            pass
    logger.error(f"❌ {name} failed to load with all combinations.")
    return pd.DataFrame()

def safe_llm_call(prompt, max_tokens=2048):
    if client is None:
        logger.warning("Groq client not initialized. LLM call skipped.")
        return None
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
            model="llama-3.1-8b-instant",
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
        "tiktok.com": "TikTok", 
        "vt.tiktok.com": "TikTok",
        "facebook.com": "Facebook", 
        "fb.watch": "Facebook",
        "twitter.com": "X", 
        "x.com": "X", 
        "youtube.com": "YouTube", 
        "youtu.be": "YouTube",
        "instagram.com": "Instagram", 
        "telegram.me": "Telegram", 
        "t.me": "Telegram"
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
    """
    Returns True ONLY if the post is original (not a retweet, quote, repost, or share).
    Uses robust regex to catch edge cases like 'reposted.', '🔁', 'RT @user', etc.
    """
    if pd.isna(text) or not isinstance(text, str):
        return False
    lower_text = text.strip().lower()
    if not lower_text:
        return False
    # Patterns to exclude — designed to match even with punctuation or spacing
    exclude_patterns = [
        r'^(rt|qt)\s*@',                     # RT @user, QT @user
        r'^//',                              # // repost style
        r'(\b|_)(repost|reshare|retweet|quote\s*tweet)(s|ed|ing)?(\b|_)',  # repost, reposted, retweets, etc.
        r'(\b|_)(shared|forwarded|credit|via)\s+(by\s+)?@?\w*',            # shared by @, via @, forwarded from
        r'(\b|_)(by|cc)\s+@',
        r'[🔁↪️➡️🔄♻️]',                      # repost emojis
        r'\b(re\s*post|re\s*tweet)\b'        # "re post", "re tweet"
    ]
    for pattern in exclude_patterns:
        if re.search(pattern, lower_text, flags=re.IGNORECASE):
            return False
    return True

def parse_timestamp_robust(timestamp):
    if pd.isna(timestamp):
        return pd.NaT
    ts_str = str(timestamp).strip()
    ts_str = re.sub(r'\s+GMT$', '', ts_str, flags=re.IGNORECASE)
    try:
        parsed = pd.to_datetime(ts_str, errors='coerce', utc=True)
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
            dayfirst = '%d' in fmt and ('%m' in fmt) and (fmt.startswith('%d'))
            parsed = pd.to_datetime(ts_str, format=fmt, errors='coerce', utc=True, dayfirst=dayfirst)
            if pd.notna(parsed):
                return parsed
        except Exception:
            continue
    return pd.NaT

# --- Combine Datasets ---
def combine_social_media_data(meltwater_df, civicsignals_df, tiktok_df=None, openmeasures_df=None):
    combined_dfs = []
    def get_col(df, cols):
        df_cols = [c.lower().strip() for c in df.columns]
        for col in cols:
            normalized_col = col.lower().strip()
            if normalized_col in df_cols:
                return df[df.columns[df_cols.index(normalized_col)]]
        return pd.Series([np.nan]*len(df), index=df.index)
    if meltwater_df is not None and not meltwater_df.empty:
        mw = pd.DataFrame()
        mw['account_id'] = get_col(meltwater_df, ['influencer'])
        mw['content_id'] = get_col(meltwater_df, ['tweet id', 'post id', 'id'])
        mw['object_id'] = get_col(meltwater_df, ['hit sentence', 'opening text', 'headline', 'article body', 'text', 'content']) 
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
        cs['account_id'] = get_col(civicsignals_df, ['media_name', 'author', 'username', 'user'])
        cs['content_id'] = get_col(civicsignals_df, ['stories_id', 'post_id', 'id', 'content_id'])
        cs['object_id'] = get_col(civicsignals_df, ['title', 'text', 'content', 'body', 'message', 'description', 'caption'])
        cs['URL'] = get_col(civicsignals_df, ['url', 'link', 'post_url'])
        cs['timestamp_share'] = get_col(civicsignals_df, ['publish_date', 'timestamp', 'date', 'created_at', 'post_date'])
        cs['source_dataset'] = 'Civicsignal'
        combined_dfs.append(cs)
    if tiktok_df is not None and not tiktok_df.empty:
        tt = pd.DataFrame()
        tt['object_id'] = get_col(tiktok_df, ['text', 'Transcript', 'caption', 'description', 'content'])
        tt['account_id'] = get_col(tiktok_df, ['authorMeta.name', 'username', 'creator', 'author'])
        tt['content_id'] = get_col(tiktok_df, ['id', 'video_id', 'post_id', 'itemId'])
        tt['URL'] = get_col(tiktok_df, ['webVideoUrl', 'TikTok Link', 'link', 'video_url', 'url'])
        tt['timestamp_share'] = get_col(tiktok_df, ['createTimeISO', 'timestamp', 'date', 'created_time', 'createTime'])
        tt['source_dataset'] = 'TikTok'
        combined_dfs.append(tt)
    if openmeasures_df is not None and not openmeasures_df.empty:
        om = pd.DataFrame()
        om['account_id'] = get_col(openmeasures_df, ['context_name'])
        om['content_id'] = get_col(openmeasures_df, ['id'])
        om['object_id'] = get_col(openmeasures_df, ['text'])
        om['URL'] = get_col(openmeasures_df, ['url'])
        om['timestamp_share'] = get_col(openmeasures_df, ['created_at'])
        om['source_dataset'] = 'OpenMeasure'
        combined_dfs.append(om)
    if not combined_dfs:
        return pd.DataFrame()
    return pd.concat(combined_dfs, ignore_index=True)

def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    if df.empty:
        return pd.DataFrame(columns=['account_id','content_id','object_id','URL','timestamp_share',
                                     'Platform','original_text','Outlet','Channel','cluster',
                                     'source_dataset','Sentiment'])
    df_processed = df.copy()
    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan','').fillna('')
    df_processed = df_processed[df_processed['object_id'].str.strip()!=""]
    if coordination_mode=="Text Content":
        df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    else:
        df_processed['original_text'] = df_processed['URL'].astype(str).replace('nan','').fillna('')
    df_processed = df_processed[df_processed['original_text'].str.strip()!=""].reset_index(drop=True)
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)
    df_processed.loc[df_processed['source_dataset'] == 'OpenMeasure', 'Platform'] = 'Telegram'
    df_processed.loc[df_processed['source_dataset'] == 'TikTok', 'Platform'] = 'TikTok'
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
def cached_clustering(df, eps, min_samples, max_features):
    if df.empty or 'original_text' not in df.columns:
        return pd.DataFrame()
    df_filtered = df[df['original_text'].str.len() > 10].copy()
    if df_filtered.empty:
        return df
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(3,5), max_features=max_features)
    try:
        tfidf_matrix = vectorizer.fit_transform(df_filtered['original_text'])
    except ValueError:
        logger.warning("Not enough documents to cluster after filtering.")
        return df
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    df_filtered['cluster'] = clustering.fit_predict(tfidf_matrix)
    df = df.copy()
    df['cluster'] = -1
    df.loc[df_filtered.index, 'cluster'] = df_filtered['cluster']
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
Generate a structured IMI intelligence report on online narratives related to election in Côte d’Ivoire.
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
**Output Format (Use simple titles for normal font size):**
Narrative Title: [Short title]
Core Claim(s): [Bullet points]
Originator(s): [Account IDs or "Unknown"]
Amplification: [Total posts]
First Detected: {min_ts}
Last Updated: {max_ts}
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
    return cleaned_summary, evidence_urls

def get_summaries_for_platform(df_clustered, filtered_df_global, platform_filter=None):
    if df_clustered.empty or 'cluster' not in df_clustered.columns:
        return []
    cluster_sizes = df_clustered[df_clustered['cluster'] != -1].groupby('cluster').size()
    top_15_clusters = cluster_sizes.nlargest(15).index.tolist()
    all_summaries = []
    for cluster_id in top_15_clusters:
        original_cluster = df_clustered[df_clustered['cluster'] == cluster_id]
        original_urls = original_cluster['URL'].dropna().unique().tolist()
        originators = original_cluster['account_id'].dropna().unique().tolist()
        if original_urls:
            all_matching_posts = filtered_df_global[filtered_df_global['URL'].isin(original_urls)]
        else:
            all_matching_posts = original_cluster.copy()
        all_texts = all_matching_posts['object_id'].astype(str).apply(extract_original_text).tolist()
        all_texts = [t for t in all_texts if len(t.strip()) > 10]
        amplifiers = all_matching_posts['account_id'].dropna().unique().tolist()
        total_reach = len(all_matching_posts)
        if not all_texts:
            continue
        min_ts = original_cluster['timestamp_share'].min()
        max_ts = original_cluster['timestamp_share'].max()
        min_ts_str = min_ts.strftime('%Y-%m-%d') if pd.notna(min_ts) else 'N/A'
        max_ts_str = max_ts.strftime('%Y-%m-%d') if pd.notna(max_ts) else 'N/A'
        raw_response, evidence_urls = summarize_cluster(all_texts, original_urls, original_cluster, min_ts_str, max_ts_str)
        virality = assign_virality_tier(total_reach)
        platform_dist = all_matching_posts['Platform'].value_counts()
        top_platforms = ", ".join([f"{p} ({c})" for p, c in platform_dist.head(3).items()])
        all_summaries.append({
            "cluster_id": cluster_id,
            "Context": raw_response,
            "Originators": ", ".join([str(a) for a in originators[:5]]) if originators else "Unknown",
            "Amplifiers_Count": len(amplifiers),
            "Total_Reach": total_reach,
            "Emerging Virality": virality,
            "Top_Platforms": top_platforms,
            "Min_TS": min_ts,
            "Max_TS": max_ts,
            "Posts_Data": all_matching_posts
        })
    return all_summaries

# --- Main App ---
def main():
    st.set_page_config(layout="wide", page_title="Côte d’Ivoire Election Monitoring Dashboard")
    col_logo, col_title = st.columns([1,5])
    with col_logo:
        st.image(CFA_LOGO_URL, width=120)
    with col_title:
        st.markdown("## 🇨🇮 Côte d’Ivoire Election Monitoring Dashboard")

    with st.spinner("📥 Loading Meltwater (X) data..."):
        meltwater_df = load_data_robustly(MELTWATER_URL, "Meltwater")
    with st.spinner("📥 Loading Civicsignal (Media) data..."):
        civicsignals_df = load_data_robustly(CIVICSIGNALS_URL, "Civicsignal")
    with st.spinner("📥 Loading TikTok data (using transcripts)..."):
        tiktok_df = load_data_robustly(TIKTOK_URL, "TikTok")
    with st.spinner("📥 Loading OpenMeasures Telegram data..."):
        openmeasures_df = load_data_robustly(OPENMEASURES_URL, "OpenMeasures")

    combined_raw_df = combine_social_media_data(meltwater_df, civicsignals_df, tiktok_df, openmeasures_df)
    if combined_raw_df.empty:
        st.error("❌ No data after combining datasets. Please check CSV formats/URLs.")
        st.stop()

    st.sidebar.markdown("### Data Sources (Raw Count)")
    source_counts = combined_raw_df['source_dataset'].value_counts()
    st.sidebar.dataframe(
        source_counts.reset_index().rename(columns={'index':'Source', 'source_dataset':'Posts'}), 
        use_container_width=True, 
        hide_index=True
    )

    df_full = final_preprocess_and_map_columns(combined_raw_df, coordination_mode="Text Content")
    if df_full.empty:
        st.error("❌ No valid data after preprocessing (content or URL missing). This means all posts were filtered out.")
        st.stop()

    df_full['timestamp_share'] = df_full['timestamp_share'].apply(parse_timestamp_robust)
    # ✅ CRITICAL: Apply robust is_original_post HERE
    df_original = df_full[df_full['object_id'].apply(is_original_post)].copy()

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

    st.sidebar.markdown("### Platform Breakdown (Filtered Count)")
    st.sidebar.markdown("*(Platforms inferred from URL/Source)*")
    platform_counts_filtered = filtered_df_global['Platform'].value_counts()
    st.sidebar.dataframe(
        platform_counts_filtered.reset_index().rename(columns={'index':'Platform', 'Platform':'Posts'}), 
        use_container_width=True, 
        hide_index=True
    )

    # ✅ TABS 2 & 3: ONLY ORIGINAL POSTS
    df_clustered_original = cached_clustering(filtered_original, eps=0.3, min_samples=2, max_features=5000) if not filtered_original.empty else pd.DataFrame()

    # ✅ TAB 4: ALL POSTS
    df_clustered_all_narratives = cached_clustering(filtered_df_global, eps=0.3, min_samples=2, max_features=5000) if not filtered_df_global.empty else pd.DataFrame()
    all_summaries = get_summaries_for_platform(df_clustered_all_narratives, filtered_df_global)

    # --- Coordination Analysis: ONLY on ORIGINAL posts ---
    coordination_groups = []
    if not df_clustered_original.empty:
        from collections import defaultdict
        grouped = df_clustered_original[df_clustered_original['cluster'] != -1].groupby('cluster')
        for cluster_id, group in grouped:
            if len(group) < 2:
                continue
            clean_df = group[['account_id', 'timestamp_share', 'Platform', 'URL', 'original_text']].copy()
            clean_df = clean_df.rename(columns={'original_text': 'text'})
            if len(clean_df['text'].unique()) < 2:
                continue 
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(3, 5), max_features=5000)
            try:
                tfidf_matrix = vectorizer.fit_transform(clean_df['text'])
                cosine_sim = cosine_similarity(tfidf_matrix)
                adj = defaultdict(list)
                for i in range(len(clean_df)):
                    for j in range(i + 1, len(clean_df)):
                        if cosine_sim[i, j] >= CONFIG["coordination_detection"]["threshold"]: 
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
            except Exception as e:
                logger.error(f"Error in coordination analysis for cluster {cluster_id}: {e}")
                continue

    total_posts = len(filtered_df_global)
    valid_clusters_count = len([s for s in all_summaries if s["Total_Reach"] >= 10])
    top_platform = filtered_df_global['Platform'].mode()[0] if not filtered_df_global['Platform'].mode().empty else "—"
    high_virality_count = len([s for s in all_summaries if "Tier 4" in s.get("Emerging Virality","")])
    last_update_time = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')

    tabs = st.tabs([
        "🏠 Dashboard Overview",
        "📈 Data Insights",
        "🔍 Coordination Analysis",
        "⚠️ Risk Assessment",
        "📰 Trending Narratives"
    ])

    with tabs[0]:
        st.markdown(f"""
        This dashboard supports the early detection of information manipulation and disinformation campaigns during election periods that seek to distort public opinion by:
        1. **Detecting Emerging Narratives**
        2. **Tracking Virality**
        3. **Providing Evidence**
        Data is updated weekly. Last updated: **{last_update_time}**
        """)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Posts Analyzed", f"{total_posts:,}")
        col2.metric("Active Narratives", valid_clusters_count)
        col3.metric("Top Platform", top_platform)
        col4.metric("Alert Level", "🚨 High" if high_virality_count > 5 else "⚠️ Medium" if high_virality_count > 0 else "✅ Low")

    with tabs[1]:
        st.markdown("### 🔬 Data Insights")
        st.markdown(f"**Total Rows:** `{len(filtered_df_global):,}` | **Date Range:** {selected_date_range[0]} to {selected_date_range[-1]}")
        if not filtered_df_global.empty:
            top_influencers = filtered_df_global['account_id'].value_counts().head(10)
            fig_src = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Post Count', 'index': 'Account ID'})
            st.plotly_chart(fig_src, use_container_width=True, key="top_influencers")
            platform_counts = filtered_df_global['Platform'].value_counts()
            fig_platform = px.bar(platform_counts, title="Post Distribution by Platform", labels={'value': 'Post Count', 'index': 'Platform'})
            st.plotly_chart(fig_platform, use_container_width=True, key="platform_dist")
            social_media_df = filtered_df_global[~filtered_df_global['Platform'].isin(['Media', 'News/Media'])].copy()
            if not social_media_df.empty and 'object_id' in social_media_df.columns:
                social_media_df['hashtags'] = social_media_df['object_id'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
                all_hashtags = [tag for tags_list in social_media_df['hashtags'] if isinstance(tags_list, list) for tag in tags_list]
                if all_hashtags:
                    hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                    fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags (Social Media Only)", labels={'value': 'Frequency', 'index': 'Hashtag'})
                    st.plotly_chart(fig_ht, use_container_width=True, key="top_hashtags")
            plot_df = filtered_df_global.copy()
            plot_df = plot_df.set_index('timestamp_share')
            time_series = plot_df.resample('D').size()
            fig_ts = px.area(time_series, title="Daily Post Volume", labels={'value': 'Total Posts', 'timestamp_share': 'Date'})
            st.plotly_chart(fig_ts, use_container_width=True, key="daily_volume")

    # ✅ TAB 2: Coordination Analysis — ONLY ORIGINAL POSTS
    with tabs[2]:
        st.subheader("🔍 Coordination Analysis (Cross-Platform)")
        st.markdown("Identifies groups of accounts sharing near-identical content, potentially indicating coordinated activity.")
        if coordination_groups:
            st.success(f"Found {len(coordination_groups)} coordinated groups.")
            for i, group in enumerate(coordination_groups):
                st.markdown(f"### Group {i+1}: {group['coordination_type']}")
                st.write(f"**Posts:** {group['num_posts']} | **Accounts:** {group['num_accounts']} | **Max similarity:** {group['max_similarity_score']}")
                posts_df = pd.DataFrame(group['posts'])
                posts_df['Timestamp'] = posts_df['timestamp_share'].dt.strftime('%Y-%m-%d %H:%M:%S')
                posts_df['URL'] = posts_df['URL'].apply(
                    lambda x: f'<a href="{x}" target="_blank">Link</a>' if pd.notna(x) else ""
                )
                posts_df['Text Snippet'] = posts_df['text'].str[:100] + '...'
                st.markdown(posts_df.to_html(escape=False, index=False, columns=['account_id', 'Platform', 'Timestamp', 'Text Snippet', 'URL']), unsafe_allow_html=True)
        else:
            st.info("No coordinated groups found based on the current threshold.")

    # ✅ TAB 3: Risk Assessment — ONLY ORIGINAL POSTS
    with tabs[3]:
        st.subheader("⚠️ Risk & Influence Assessment")
        st.markdown("""
        This tab ranks accounts by **coordination activity**.
        High-risk accounts are potential **amplifiers or originators** involved in clustered content sharing.
        """)
        if df_clustered_original.empty or 'cluster' not in df_clustered_original.columns:
            st.info("No data available for risk assessment.")
        else:
            clustered_accounts = df_clustered_original[df_clustered_original['cluster'] != -1].dropna(subset=['account_id'])
            account_risk = clustered_accounts.groupby('account_id').size().reset_index(name='Coordination_Count')
            total_post_counts = filtered_df_global.groupby('account_id').size().reset_index(name='Total_Posts')
            account_risk = account_risk.merge(
                filtered_df_global[['account_id', 'Platform']].drop_duplicates(subset=['account_id']),
                on='account_id',
                how='left'
            ).merge(
                total_post_counts,
                on='account_id',
                how='left'
            )
            account_risk['Risk_Ratio'] = account_risk['Coordination_Count'] / account_risk['Total_Posts']
            account_risk = account_risk.sort_values(['Coordination_Count', 'Risk_Ratio'], ascending=[False, False]).head(20)
            if account_risk.empty:
                st.info("No high-risk accounts detected.")
            else:
                st.markdown("#### Top 20 Accounts by Coordination Activity (Cross-Platform)")
                st.dataframe(account_risk[['account_id', 'Platform', 'Coordination_Count', 'Total_Posts', 'Risk_Ratio']], use_container_width=True)
                risk_csv = convert_df_to_csv(account_risk)
                st.download_button(
                    "📥 Download Risk Assessment CSV",
                    risk_csv,
                    "risk_assessment.csv",
                    "text/csv"
                )

    # ✅ TAB 4: Trending Narratives — ALL POSTS
    with tabs[4]:
        st.subheader("📖 Trending Narrative Summaries")
        def render_summaries_as_cards(summaries_list, title):
            if not summaries_list:
                st.info(f"No active narratives found in the {title} data.")
                return
            all_reaches = [s["Total_Reach"] for s in summaries_list if s["Total_Reach"] > 0]
            median_reach = np.median(all_reaches) if all_reaches else 1
            sorted_summaries = sorted(summaries_list, key=lambda x: x["Total_Reach"], reverse=True)
            for i, summary in enumerate(sorted_summaries):
                cluster_id = summary["cluster_id"]
                total_reach = summary["Total_Reach"]
                all_matching_posts = summary["Posts_Data"]
                if total_reach < 10:
                    continue
                relative_virality = total_reach / median_reach if median_reach > 0 else 1.0
                virality_emoji = "🔥" if "Tier 4" in summary['Emerging Virality'] else "📢" if "Tier 3" in summary['Emerging Virality'] else "💬"
                originators_display = summary['Originators'] if summary['Originators'] != "Unknown" else "Unknown originator(s)"
                card_title = f"{virality_emoji} Cluster {cluster_id} · {summary['Emerging Virality']} · {originators_display}"
                col_img, col_text = st.columns([1, 3])
                with col_img:
                    image_url = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/main/trending_topic.png"
                    st.markdown(
                        f"""
                        <div style="width:100%; height:150px; border-radius:8px; overflow:hidden; display:flex; align-items:center; justify-content:center; background:#333; border:1px solid #555;">
                            <img src="{image_url}" style="max-width:90%; max-height:90%; object-fit:contain;" onerror="this.style.display='none'; this.parentNode.innerHTML='<span style=&quot;color:white; font-size:28px;&quot;>📈</span>'" />
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_text:
                    st.markdown(f"### {card_title}")
                    min_ts_str = summary['Min_TS'].strftime('%Y-%m-%d') if pd.notna(summary['Min_TS']) else 'N/A'
                    max_ts_str = summary['Max_TS'].strftime('%Y-%m-%d') if pd.notna(summary['Max_TS']) else 'N/A'
                    st.caption(f"📅 First Detected: {min_ts_str} | Last Updated: {max_ts_str}")
                    st.caption(f"📈 Amplification: {total_reach} posts ({relative_virality:.1f}x median activity)")
                    st.caption(f"🌐 Platforms: {summary['Top_Platforms']}")
                    raw_context = summary['Context'] if summary['Context'] else ""
                    if raw_context.strip():
                        cleaned_lines = [line.strip() for line in raw_context.split('\n')]
                        while cleaned_lines and not cleaned_lines[0]:
                            cleaned_lines.pop(0)
                        while cleaned_lines and not cleaned_lines[-1]:
                            cleaned_lines.pop()
                        narrative_text = '\n'.join(cleaned_lines)
                    else:
                        narrative_text = "No summary available."
                    st.markdown("#### Narrative Summary")
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #1e1e1e;
                            color: white;
                            padding: 12px;
                            border-radius: 6px;
                            font-family: monospace;
                            white-space: pre-wrap;
                            line-height: 1.5;
                            font-size: 14px;
                            max-height: 300px;
                            overflow-y: auto;
                            margin-bottom: 10px;
                            text-align: left;
                        ">{narrative_text}</div>
                        """,
                        unsafe_allow_html=True
                    )
                    if not all_matching_posts.empty and 'timestamp_share' in all_matching_posts.columns:
                        plot_df_time = all_matching_posts.set_index('timestamp_share').resample('h').size().reset_index(name='Count')
                        fig_timeline = px.line(
                            plot_df_time,
                            x='timestamp_share',
                            y='Count',
                            title=f"Time Series Activity for Cluster {cluster_id}",
                            labels={'Count': 'Post Volume', 'timestamp_share': 'Time'}
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True, key=f"{title.replace(' ', '_')}_timeline_{cluster_id}")
                    st.markdown("**Example Posts (Top 5)**")
                    example_posts = all_matching_posts[['source_dataset', 'Platform', 'account_id', 'object_id', 'URL']].head(5).copy()
                    example_posts['URL'] = example_posts['URL'].astype(str).str.strip()
                    example_posts.loc[~example_posts['URL'].str.startswith('http', na=False), 'URL'] = None
                    example_posts['object_id'] = example_posts['object_id'].astype(str).str[:100] + "..."
                    st.dataframe(
                        example_posts,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "URL": st.column_config.LinkColumn(
                                "Post Link",
                                display_text="View Post",
                                help="Click to open original post"
                            ),
                            "object_id": st.column_config.TextColumn(
                                "Content Snippet",
                                width="medium"
                            )
                        }
                    )
                st.markdown("---")
        render_summaries_as_cards(all_summaries, "Cross-Platform")

        st.markdown("---")
        st.subheader("📱 Top TikTok Videos (Latest & Most Substantial)")
        tiktok_posts = filtered_df_global[filtered_df_global['source_dataset'] == 'TikTok'].copy()
        if tiktok_posts.empty:
            st.warning("No TikTok data available for the selected date range.")
        else:
            tiktok_posts = tiktok_posts.sort_values(
                by=['timestamp_share', 'object_id'],
                key=lambda col: col.str.len() if col.name == 'object_id' else col,
                ascending=[False, False]
            ).head(15).reset_index(drop=True)
            def simple_virality(text):
                if pd.isna(text):
                    return "Low"
                length = len(str(text))
                if length > 200:
                    return "High Engagement Potential"
                elif length > 100:
                    return "Moderate"
                else:
                    return "Basic"
            display_df = tiktok_posts[['timestamp_share', 'object_id', 'URL']].copy()
            display_df.columns = ['Posted At', 'Transcript / Text', 'Video Link']
            display_df['Virality Level'] = display_df['Transcript / Text'].apply(simple_virality)
            display_df['Video Link'] = display_df['Video Link'].astype(str).str.strip()
            display_df.loc[~display_df['Video Link'].str.startswith('http', na=False), 'Video Link'] = None
            display_df = display_df[['Posted At', 'Virality Level', 'Transcript / Text', 'Video Link']]
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Video Link": st.column_config.LinkColumn(
                        "Video Link",
                        display_text="Watch on TikTok",
                        help="Click to open TikTok video"
                    )
                }
            )

        st.markdown("---")
        st.subheader("📡 Top Telegram Posts")
        telegram_posts = filtered_df_global[filtered_df_global['source_dataset'] == 'OpenMeasure'].copy()
        if telegram_posts.empty:
            st.warning("No Telegram data available for the selected date range.")
        else:
            raw_om = load_data_robustly(OPENMEASURES_URL, "OpenMeasures (metrics)")
            if not raw_om.empty:
                raw_om = raw_om[['id', 'engagement_views']].copy()
                raw_om['engagement_views'] = pd.to_numeric(raw_om['engagement_views'], errors='coerce')
                telegram_posts = telegram_posts.merge(
                    raw_om.rename(columns={'id': 'content_id'}),
                    on='content_id',
                    how='left'
                )
            else:
                telegram_posts['engagement_views'] = np.nan
            telegram_posts['engagement_views'] = telegram_posts['engagement_views'].fillna(0)
            telegram_posts['text_length'] = telegram_posts['object_id'].astype(str).str.len()
            telegram_posts = telegram_posts.sort_values(
                by=['engagement_views', 'text_length', 'timestamp_share'],
                ascending=[False, False, False]
            ).head(15).reset_index(drop=True)
            def telegram_virality(views):
                if views >= 10000:
                    return "High Reach"
                elif views >= 1000:
                    return "Moderate Reach"
                else:
                    return "Limited Reach"
            display_telegram = telegram_posts[['timestamp_share', 'account_id', 'object_id', 'URL', 'engagement_views']].copy()
            display_telegram.columns = ['Posted At', 'Channel', 'Message Text', 'Post Link', 'Views']
            display_telegram['Virality Level'] = display_telegram['Views'].apply(telegram_virality)
            display_telegram['Post Link'] = display_telegram['Post Link'].astype(str).str.strip()
            display_telegram.loc[~display_telegram['Post Link'].str.startswith('http', na=False), 'Post Link'] = None
            display_telegram = display_telegram[['Posted At', 'Channel', 'Virality Level', 'Views', 'Message Text', 'Post Link']]
            st.dataframe(
                display_telegram,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Post Link": st.column_config.LinkColumn(
                        "Post Link",
                        display_text="View on Telegram",
                        help="Open original Telegram post"
                    ),
                    "Views": st.column_config.NumberColumn("Views", format="%d")
                }
            )

if __name__ == '__main__':
    main()
