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
    # Use st.secrets to safely retrieve API key
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    import groq
    client = groq.Groq(api_key=GROQ_API_KEY)
except Exception as e:
    logger.warning(f"Groq API key not found or client error: {e}")
    client = None

# --- URLs ---
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
        # Safely extract content
        try:
            content = response.choices[0].message['content'].strip()
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

def parse_timestamp_robust(timestamp):
    """
    Robust timestamp parsing for Meltwater/CSV data.
    Always returns pandas Timestamp in UTC or NaT if unparseable.
    """
    if pd.isna(timestamp):
        return pd.NaT

    ts_str = str(timestamp).strip()

    # Remove trailing GMT if present
    ts_str = re.sub(r'\s+GMT$', '', ts_str, flags=re.IGNORECASE)

    # Try pandas automatic parsing with UTC
    try:
        parsed = pd.to_datetime(ts_str, errors='coerce', utc=True, dayfirst=True)
        if pd.notna(parsed):
            return parsed
    except Exception:
        pass

    # Fallback formats
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

# --- Combine Datasets ---
def combine_social_media_data(meltwater_df, civicsignals_df):
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

    # Ensure 'Sentiment' exists for downstream aggregation
    if 'Sentiment' not in df_processed.columns:
        df_processed['Sentiment'] = np.nan

    # Select columns safely
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
- Allegations of political suppression: opposition figures being silenced, arrested, or excluded from governance before voting.
- Allegations of corruption, bias, or manipulation within the **Electoral Commission** (tally centers, vote transmission, fraud, rigging).
- Economic distress, cost of living, or corruption involving state funds.
- Hate speech, ethnic slurs, tribalism, sectarianism, xenophobia.
- Gender-based attacks, misogyny, sexist remarks.
- Foreign interference: anti-Western, anti-EU, colonialism, imperialism, "Western puppet" narratives.
- Marginalization of minority communities.
- *Narratives undermining voting process: fraud, rigged elections, tally center issues, system failures*.
- *Mentions of protests or civic resistance being planned or mobilized in anticipation of the election*.
- *Lists of viral content, hashtags, or slogans promoting civic action, voter turnout, or anti-government sentiment*.
**Strict Instructions:**
- Only summarize content that is **directly present in the posts provided**.
- Do **not** invent claims — only document what is explicitly stated in posts.
- For every claim, **only use a URL that explicitly contains that exact claim**.
- Do **not** repeat the same claim with different wording.
- Do not add outside knowledge, fact-checking, or assumptions.
**Output Format:**
- Start each cluster with a bold title: **Narrative Title Here**
- Summarize factually in short narrative paragraphs.
- Include post URLs for every claim or reused message.
- End with the narrative lifecycle:
  - First Detected: {min_ts}
  - Last Updated: {max_ts}
Documents:
{joined}{url_context}
"""    
    response = safe_llm_call(prompt, max_tokens=2048)

    raw_summary = ""
    if response:
        try:
            # Try standard ChatCompletion format
            raw_summary = response.choices[0].message.content.strip()
        except (AttributeError, IndexError):
            # Fallback if the object is different
            raw_summary = str(response).strip()
    
    # Then do your cleaning
    evidence_urls = re.findall(r"(https?://[^\s\)\]]+)", raw_summary)
    cleaned_summary = re.sub(r'\*\*Here is a concise.*?\*\*', '', raw_summary, flags=re.IGNORECASE | re.DOTALL)
    cleaned_summary = re.sub(r'\*\*Here are a few options.*?\*\*', '', cleaned_summary, flags=re.IGNORECASE | re.DOTALL)
    cleaned_summary = re.sub(r'"[^"]*"', '', cleaned_summary)
    cleaned_summary = cleaned_summary.strip()
    
    if evidence_urls:
        cleaned_summary += "\n\nSources: " + ", ".join(evidence_urls[:5])
    
    return cleaned_summary, evidence_urls
# --- Main App ---
# --- GitHub Raw CSV URL (predefined) ---
MELTWATER_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/Co%CC%82te_dIvoire_Sep_Oct16.csv"

def main():
    st.set_page_config(layout="wide", page_title="Côte d’Ivoire Election Monitoring Dashboard")
    
    # --- Header ---
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
            logger.info("Meltwater loaded with default encoding, sep='\t'")
        except Exception as e:
            try:
                meltwater_df = pd.read_csv(MELTWATER_URL, encoding='utf-16', sep='\t', low_memory=False, on_bad_lines='skip')
                logger.info("Meltwater loaded with utf-16, sep='\t'")
            except Exception as e:
                st.error(f"❌ Meltwater failed to load: {e}")

    combined_raw_df = combine_social_media_data(meltwater_df, pd.DataFrame())
    if combined_raw_df.empty:
        st.error("❌ No data after combining datasets.")
        st.stop()

    # --- Preprocessing ---
    df = final_preprocess_and_map_columns(combined_raw_df, coordination_mode="Text Content")
    if df.empty:
        st.error("❌ No valid data after preprocessing.")
        st.stop()
    df['timestamp_share'] = df['timestamp_share'].apply(parse_timestamp_robust)

    # --- Date filter ---
    valid_dates = df['timestamp_share'].dropna()
    if valid_dates.empty:
        st.error("❌ No valid timestamps found in the dataset.")
        st.stop()
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    selected_date_range = st.sidebar.date_input(
        "Date Range", value=[min_date,max_date], min_value=min_date, max_value=max_date
    )
    if len(selected_date_range)==2:
        start_date = pd.Timestamp(selected_date_range[0], tz='UTC')
        end_date = pd.Timestamp(selected_date_range[1], tz='UTC') + pd.Timedelta(days=1)
    else:
        start_date = pd.Timestamp(selected_date_range[0], tz='UTC')
        end_date = start_date + pd.Timedelta(days=1)

    filtered_df_global = df[(df['timestamp_share']>=start_date)&(df['timestamp_share']<end_date)].copy()

    # --- Clustering ---
    df_clustered = cached_clustering(filtered_df_global, eps=0.3, min_samples=2, max_features=5000, data_source_key="report")

    # --- Top clusters ---
    top_15_clusters = []
    if 'cluster' in df_clustered.columns and not df_clustered.empty:
        cluster_sizes = df_clustered[df_clustered['cluster'] != -1].groupby('cluster').size()
        top_15_clusters = cluster_sizes.nlargest(15).index.tolist()

    # --- Summaries ---
    all_summaries = []
    for cluster_id in top_15_clusters:
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        texts = cluster_data['original_text'].tolist()
        urls = cluster_data['URL'].dropna().unique().tolist()
        min_ts_str = cluster_data['timestamp_share'].min().strftime('%Y-%m-%d')
        max_ts_str = cluster_data['timestamp_share'].max().strftime('%Y-%m-%d')

        summary, evidence_urls = summarize_cluster(texts, urls, cluster_data, min_ts_str, max_ts_str)

        post_count = len(cluster_data)
        virality = assign_virality_tier(post_count)

        sentiment_counts = cluster_data['Sentiment'].value_counts().to_dict() if 'Sentiment' in cluster_data.columns else {"Negative":0,"Neutral":0,"Positive":0}

        all_summaries.append({
            "Evidence": ", ".join(evidence_urls[:5]),
            "Context": summary,
            "URLs": str(urls),
            "Emerging Virality": virality,
            "Post Count": post_count,
            "Negative Count": sentiment_counts.get("Negative", 0),
            "Neutral Count": sentiment_counts.get("Neutral", 0),
            "Positive Count": sentiment_counts.get("Positive", 0)
        })

    report_df = pd.DataFrame(all_summaries)

    # --- Metrics ---
    total_posts = len(df)
    valid_clusters_count = len(top_15_clusters)
    top_platform = df['Platform'].mode()[0] if not df['Platform'].mode().empty else "—"
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
        #st.markdown("### 🎯 Aim and Purpose")
        st.markdown(f"""
        This dashboard provides **daily monitoring of trending narratives** related to the 2025 elections in Côte d’Ivoire.
    
        The primary purpose is to support **transparent, evidence-based election observation** by:
    
        1. **Detecting Emerging Narratives**: Identify rapidly spreading disinformation, hate speech, and coordinated messaging.
        2. **Tracking Virality**: Assess spread and influence of high-risk content across multiple platforms.
        3. **Providing Evidence**: Offer timely, actionable intelligence to stakeholders for early intervention.
    
        Data is updated daily. Last updated: **{last_update_time}**
        """)
        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Posts Analyzed", f"{total_posts:,}")
        col2.metric("Active Narratives", valid_clusters_count)
        col3.metric("Top Platform", top_platform)
        col4.metric("Alert Level", "🚨 High" if high_virality_count>5 else "⚠️ Medium" if high_virality_count>0 else "✅ Low")
    
    # TAB 1: Data Insights
    with tabs[1]:
        st.markdown("### 🔬 Data Insights")
        st.markdown(f"**Total Rows:** `{len(df):,}` | **Date Range:** {selected_date_range[0]} to {selected_date_range[-1]}")
        if not filtered_df_global.empty:
            top_influencers = filtered_df_global['account_id'].value_counts().head(10)
            fig_src = px.bar(top_influencers, title="Top 10 Influencers")
            st.plotly_chart(fig_src, use_container_width=True)
    
            platform_counts = filtered_df_global['Platform'].value_counts()
            fig_platform = px.bar(platform_counts, title="Post Distribution by Platform")
            st.plotly_chart(fig_platform, use_container_width=True)
            
            # --- TOP HASHTAGS PLOT ---
            social_media_df = filtered_df_global[~filtered_df_global['Platform'].isin(['Media', 'News/Media'])].copy()
            if not social_media_df.empty and 'object_id' in social_media_df.columns:
                social_media_df['hashtags'] = social_media_df['object_id'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
                all_hashtags = [tag for tags_list in social_media_df['hashtags'] if isinstance(tags_list, list) for tag in tags_list]
                if all_hashtags:
                    hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                    fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags (Social Media Only)", labels={'value': 'Frequency', 'index': 'Hashtag'})
                    st.plotly_chart(fig_ht, use_container_width=True)
                    st.markdown("**Top 10 Hashtags (Social Media Only)**: Highlights the most frequently used hashtags on social platforms.")
            
            # --- END HASHTAGS PLOT ---
    
            plot_df = filtered_df_global.copy()
            plot_df = plot_df.set_index('timestamp_share')
            time_series = plot_df.resample('D').size()
            fig_ts = px.area(time_series, title="Daily Post Volume")
            st.plotly_chart(fig_ts, use_container_width=True)
    
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
                                # Determine coordination type based on similarity / size heuristic
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
                st.write(f"**Posts:** {group['num_posts']} | **Accounts involved:** {group['num_accounts']} | **Max similarity:** {group['max_similarity_score']}")
                
                posts_df = pd.DataFrame(group['posts'])
                posts_df['Timestamp'] = posts_df['timestamp_share']
                
                # Make URLs clickable
                posts_df['URL'] = posts_df['URL'].apply(lambda x: f"[Link]({x})" if pd.notna(x) else "")
                st.dataframe(posts_df[['account_id', 'Platform', 'Timestamp', 'URL']], use_container_width=True)
        else:
            st.info("No coordinated groups found.")
    # TAB 3: Risk Assessment
    with tabs[3]:
        st.subheader("⚠️ Risk & Influence Assessment")
        st.markdown("""
        This tab ranks accounts by **coordination activity** — how many coordinated groups they appear in.
        High-risk accounts are potential **amplifiers or originators** of coordinated disinformation.
        """)
    
        if df_clustered.empty or 'cluster' not in df_clustered.columns:
            st.info("No data available for risk assessment.")
        else:
            clustered_accounts = df_clustered[df_clustered['cluster'] != -1].dropna(subset=['account_id'])
            account_risk = clustered_accounts.groupby('account_id').size().reset_index(name='Coordination_Count')
    
            # Merge Platform info
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
    
                # Download CSV
                risk_csv = convert_df_to_csv(account_risk)
                st.download_button(
                    "📥 Download Risk Assessment CSV",
                    risk_csv,
                    "risk_assessment.csv",
                    "text/csv"
                )
    # TAB 4
    with tabs[4]:
        if report_df.empty:
            st.info("No narratives to display.")
        else:
            report_df = report_df.sort_values('Post Count', ascending=False)
            for idx, row in report_df.iterrows():
                context = row.get('Context', 'No narrative available')
                urls = row.get('URLs', '')
                if isinstance(urls, str):
                    url_list = [u.strip() for u in urls.strip("[]").split(',') if u.strip().startswith('http')]
                else:
                    url_list = []
                virality = row['Emerging Virality']
                if "Tier 4" in str(virality):
                    badge = '<span style="background-color: #ffebee; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #c62828;">🚨 Viral Emergency</span>'
                elif "Tier 3" in str(virality):
                    badge = '<span style="background-color: #fff3e0; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #e65100;">🔥 High Spread</span>'
                elif "Tier 2" in str(virality):
                    badge = '<span style="background-color: #e8f5e9; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #2e7d32;">📈 Moderate</span>'
                else:
                    badge = '<span style="background-color: #f5f5f5; padding: 4px 8px; border-radius: 6px; color: #555;">ℹ️ Limited</span>'
                title_preview = context.split('\n')[0][:120] + ("..." if len(context) > 120 else "")
                with st.expander(f"**{title_preview}**"):
                    st.markdown("### 📖 Narrative Summary")
                    st.markdown(context)
                    st.markdown("### ⚠️ Virality Level")
                    st.markdown(badge, unsafe_allow_html=True)
                    if url_list:
                        st.markdown("### 🔗 Supporting Evidence")
                        for url in url_list[:5]:
                            st.markdown(f"- [{url}]({url})")
            csv_data = convert_df_to_csv(report_df)
            st.download_button("📥 Download Full Report (CSV)", csv_data, "imi_narrative_report.csv", "text/csv")

if __name__ == '__main__':
    main()
