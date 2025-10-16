import pandas as pd
import numpy as np
import re
import logging
import time
from datetime import timedelta
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
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
    logger.warning(f"Groq API key not found: {e}")
    client = None

GITHUB_DATA_URL = "https://raw.githubusercontent.com/hanna-tes/Disinfo_monitoring_RadarSystem/refs/heads/main/Co%CC%82te_dIvoire_Sep_Oct16.csv"
CFA_LOGO_URL = "https://opportunities.codeforafrica.org/wp-content/uploads/sites/5/2015/11/1-Zq7KnTAeKjBf6eENRsacSQ.png"
# --- Helper Functions ---
def safe_llm_call(prompt, max_tokens=2048):
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens
        )
        return response
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None

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
    if pd.isna(timestamp):
        return None
    if isinstance(timestamp, (int, float)):
        if 0 < timestamp < 253402300800:
            return int(timestamp)
        return None
    try:
        parsed = pd.to_datetime(timestamp, errors='coerce', utc=True)
        if pd.notna(parsed):
            return int(parsed.timestamp())
    except:
        pass
    return None

# --- Combine Multiple Datasets with Robust Column Mapping ---
def combine_social_media_data(
    meltwater_df=None,
    civicsignals_df=None,
    openmeasure_df=None,
    meltwater_object_col='hit sentence',
    civicsignals_object_col='title',
    openmeasure_object_col='text'
):
    """
    Combines datasets from Meltwater, CivicSignals, and OpenMeasure (optional).
    Allows specification of which column to use as 'object_id' for coordination analysis.
    Returns a clean DataFrame with standardized columns and UNIX timestamps.
    """
    combined_dfs = []

    def get_column_safe(df, possible_cols):
        """
        Returns the first column from possible_cols found in df (case-insensitive),
        or a Series of NaNs if none exist.
        """
        df_cols_lower = [c.lower().strip() for c in df.columns]
        for col in possible_cols:
            if col.lower() in df_cols_lower:
                return df.iloc[:, df_cols_lower.index(col.lower())]
        return pd.Series([np.nan] * len(df), index=df.index)

    # --- Process Meltwater ---
    if meltwater_df is not None and not meltwater_df.empty:
        mw = pd.DataFrame()
        mw['account_id'] = get_column_safe(meltwater_df, ['influencer'])
        mw['content_id'] = get_column_safe(meltwater_df, ['tweet id', 'post id', 'id'])
        mw['object_id'] = get_column_safe(meltwater_df, [meltwater_object_col])
        mw['URL'] = get_column_safe(meltwater_df, ['url', 'link'])
        mw['timestamp_share'] = get_column_safe(meltwater_df, ['date', 'published_at'])
        mw['source_dataset'] = 'Meltwater'
        combined_dfs.append(mw)

    # --- Process CivicSignals ---
    if civicsignals_df is not None and not civicsignals_df.empty:
        cs = pd.DataFrame()
        cs['account_id'] = get_column_safe(civicsignals_df, ['media_name', 'account'])
        cs['content_id'] = get_column_safe(civicsignals_df, ['stories_id', 'id'])
        cs['object_id'] = get_column_safe(civicsignals_df, [civicsignals_object_col])
        cs['URL'] = get_column_safe(civicsignals_df, ['url', 'link'])
        cs['timestamp_share'] = get_column_safe(civicsignals_df, ['publish_date', 'date'])
        cs['source_dataset'] = 'CivicSignals'
        combined_dfs.append(cs)

    # --- Process OpenMeasure ---
    if openmeasure_df is not None and not openmeasure_df.empty:
        om = pd.DataFrame()
        om['account_id'] = get_column_safe(openmeasure_df, ['actor_username', 'user'])
        om['content_id'] = get_column_safe(openmeasure_df, ['id'])
        om['object_id'] = get_column_safe(openmeasure_df, [openmeasure_object_col])
        om['URL'] = get_column_safe(openmeasure_df, ['url', 'link'])
        om['timestamp_share'] = get_column_safe(openmeasure_df, ['created_at', 'date'])
        om['source_dataset'] = 'OpenMeasure'
        combined_dfs.append(om)

    if not combined_dfs:
        return pd.DataFrame()

    combined = pd.concat(combined_dfs, ignore_index=True)

    # --- Clean text columns ---
    for col in ['account_id', 'content_id', 'object_id', 'URL']:
        combined[col] = combined[col].astype(str).replace('nan', '').fillna('')
    combined = combined[combined['object_id'].str.strip() != ""].copy()
    combined = combined.drop_duplicates(subset=['account_id','content_id','object_id','timestamp_share']).reset_index(drop=True)

    # --- Convert timestamps ---
    combined['timestamp_share'] = combined['timestamp_share'].apply(parse_timestamp_robust)
    combined = combined.dropna(subset=['timestamp_share']).reset_index(drop=True)
    combined['timestamp_share'] = combined['timestamp_share'].astype('Int64')

    return combined

# --- Final Preprocessing Function ---
def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    """
    Prepares combined dataset for clustering/analysis.
    - Uses 'object_id' for text or 'URL' for URL coordination.
    - Adds 'original_text', 'Platform', 'Outlet', 'Channel'.
    """
    if df.empty:
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=['account_id','content_id','object_id','URL','timestamp_share',
                                     'Platform','original_text','Outlet','Channel','cluster','source_dataset'])

    df_processed = df.copy()
    df_processed.rename(columns={'original_url': 'URL'}, inplace=True)

    # Filter empty object_id
    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan','').fillna('')
    df_processed = df_processed[df_processed['object_id'].str.strip() != ""].copy()

    if coordination_mode == "Text Content":
        df_processed['object_id'] = df_processed['object_id'].apply(extract_original_text)
        df_processed = df_processed[df_processed['object_id'].str.strip() != ""].reset_index(drop=True)

    elif coordination_mode == "Shared URLs":
        df_processed['object_id'] = df_processed['URL'].astype(str).replace('nan','').fillna('')
        df_processed = df_processed[df_processed['object_id'].str.strip() != ""].reset_index(drop=True)

    df_processed['original_text'] = df_processed['object_id']

    # Infer platform from URL
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)
    df_processed['Outlet'] = df_processed.get('Outlet', np.nan)
    df_processed['Channel'] = df_processed.get('Channel', np.nan)
    df_processed['cluster'] = -1
    df_processed['source_dataset'] = df_processed.get('source_dataset', 'Unknown')

    return df_processed

@st.cache_data(show_spinner=False, ttl=3600)
def load_data_from_github(url):
    try:
        df = pd.read_csv(url, encoding='utf-8-sig', sep='\t', low_memory=False)
        st.success(f"✅ Loaded {len(df):,} posts from GitHub using UTF-8-SIG and tab separator.")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data from GitHub: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def cached_clustering(df, eps, min_samples, max_features, data_source_key):
    if df.empty or 'original_text' not in df.columns:
        return pd.DataFrame()
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(3, 5), max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(df['original_text'])
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    df = df.copy()
    df['cluster'] = clustering.fit_predict(tfidf_matrix)
    return df

def assign_virality_tier(post_count):
    if post_count >= 500:
        return "Tier 4: Viral Emergency"
    elif post_count >= 100:
        return "Tier 3: High Spread"
    elif post_count >= 20:
        return "Tier 2: Moderate"
    else:
        return "Tier 1: Limited"

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Summarize Cluster (required for narrative generation) ---
def summarize_cluster(texts, urls, cluster_data, min_ts, max_ts):
    joined = "\n".join(texts[:50])
    url_context = "\nRelevant post links:\n" + "\n".join(urls[:5]) if urls else ""
    prompt = f"""
Generate a structured IMI intelligence report on online narratives related to election.
Focus on pre and post election tensions and emerging narratives, including:
- Allegations of political suppression: opposition figures being silenced, arrested, or excluded from governance before voting.
- Allegations of corruption, bias, or manipulation within the **Electoral Commission** (tally centers, vote transmission, fraud, rigging).
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
- Do not include URLs that do NOT contain the claim.
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
    if response:
        raw_summary = response.choices[0].message.content.strip()
        evidence_urls = re.findall(r"(https?://[^\s\)\]]+)", raw_summary)
        cleaned_summary = re.sub(r'\*\*Here is a concise.*?\*\*', '', raw_summary, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned_summary = re.sub(r'\*\*Here are a few options.*?\*\*', '', cleaned_summary, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned_summary = re.sub(r'"[^"]*"', '', cleaned_summary).strip()
        return cleaned_summary, evidence_urls
    else:
        return "Summary generation failed.", []

# --- Main App ---
def main():
    st.set_page_config(layout="wide", page_title="Côte d’Ivoire Election Monitor")

    # Header with Logo
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image(CFA_LOGO_URL, width=120)
    with col_title:
        st.markdown("## 🇨🇮 Côte d’Ivoire Election Integrity Monitor")
        # 👇 REMOVED: st.caption("Powered by **Code for Africa** | ...")

    # Load data
    df_raw = load_data_from_github(GITHUB_DATA_URL)
    if df_raw.empty:
        st.stop()

    # Preprocess
    df = final_preprocess_and_map_columns(df_raw, coordination_mode="Text Content")
    if df.empty:
        st.error("❌ No valid data after preprocessing.")
        st.stop()

    # Global Filters
    min_ts = df['timestamp_share'].min()
    max_ts = df['timestamp_share'].max()
    min_date = pd.to_datetime(min_ts, unit='s').date() if pd.notna(min_ts) else pd.Timestamp.now().date()
    max_date = pd.to_datetime(max_ts, unit='s').date() if pd.notna(max_ts) else pd.Timestamp.now().date()
    selected_date_range = st.sidebar.date_input("Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)

    if len(selected_date_range) == 2:
        start_ts = int(pd.Timestamp(selected_date_range[0], tz='UTC').timestamp())
        end_ts = int((pd.Timestamp(selected_date_range[1], tz='UTC') + timedelta(days=1) - timedelta(microseconds=1)).timestamp())
    else:
        start_ts = int(pd.Timestamp(selected_date_range[0], tz='UTC').timestamp())
        end_ts = start_ts + 86400 - 1

    filtered_df_global = df[
        (df['timestamp_share'] >= start_ts) &
        (df['timestamp_share'] <= end_ts)
    ].copy()

    # Cluster
    df_clustered = cached_clustering(filtered_df_global, 0.3, 2, 5000, "report")

    # Identify top 15 clusters by post count
    top_15_clusters = []
    if 'cluster' in df_clustered.columns:
        cluster_sizes = df_clustered[df_clustered['cluster'] != -1].groupby('cluster').size()
        top_15_clusters = cluster_sizes.nlargest(15).index.tolist()

    # Generate report for top 15 only
    all_summaries = []
    for cluster_id in top_15_clusters:
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        texts = cluster_data['original_text'].tolist()
        urls = cluster_data['URL'].dropna().unique().tolist()
        min_ts_str = pd.to_datetime(cluster_data['timestamp_share'].min(), unit='s').strftime('%Y-%m-%d')
        max_ts_str = pd.to_datetime(cluster_data['timestamp_share'].max(), unit='s').strftime('%Y-%m-%d')
        summary, evidence_urls = summarize_cluster(texts, urls, cluster_data, min_ts_str, max_ts_str)
        post_count = len(cluster_data)
        virality = assign_virality_tier(post_count)
        all_summaries.append({
            "Evidence": ", ".join(evidence_urls[:5]),
            "Context": summary,
            "URLs": str(urls),
            "Emerging Virality": virality,
            "Post Count": post_count
        })
    report_df = pd.DataFrame(all_summaries)

    # Compute metrics for overview
    total_posts = len(df)
    valid_clusters_count = len(top_15_clusters)
    top_platform = df['Platform'].mode()[0] if not df['Platform'].mode().empty else "—"
    high_virality_count = len([s for s in all_summaries if "Tier 4" in s.get("Emerging Virality", "")])
    last_update_time = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')
    
    # Tabs
    tabs = st.tabs([
        "🏠 Dashboard Overview",
        "📈 Data Insights",
        "🔍 Coordination Analysis",
        "🕸️ Network Analysis",
        "📰 Trending Narratives"
    ])

    # ===== TAB 0: Dashboard Overview =====
    with tabs[0]:
        st.markdown("""
        This dashboard provides regularly updated insights into emerging narratives and information manipulation** related to the 2025 elections in Côte d’Ivoire.
        
        We track recurring claims, coordinated amplification, and viral content across digital platforms — presenting **what is being said**, **how widely it spreads**, and **who is sharing it** — to support transparent, evidence-based election observation.
        
        **Monitored themes include**:
        - Claims about electoral procedures, voter suppression, or fraud
        - Hate speech, ethnic or regional tensions
        - Coordinated inauthentic behavior
        - Foreign interference narratives
        - Mobilization for protests or civic action
        
        Data is updated daily. Last updated: **{}**
        """.format(last_update_time))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Posts Analyzed", f"{total_posts:,}")
        col2.metric("Active Narratives", valid_clusters_count)
        col3.metric("Top Platform", top_platform)
        col4.metric("Alert Level", "🚨 High" if high_virality_count > 5 else "⚠️ Medium" if high_virality_count > 0 else "✅ Low")

        st.markdown(">")
        st.markdown("_This tool supports transparent, evidence-based election observation in Côte d’Ivoire._")

    # ===== TAB 1: Data Insights =====
    with tabs[1]:
        st.markdown("### 🔬 Data Insights")
        st.markdown(f"**Total Rows:** `{len(df):,}` | **Date Range:** {selected_date_range[0]} to {selected_date_range[-1]}")
        
        if not filtered_df_global.empty:
            # Top Influencers
            top_influencers = filtered_df_global['account_id'].value_counts().head(10)
            fig_src = px.bar(top_influencers, title="Top 10 Influencers")
            st.plotly_chart(fig_src, use_container_width=True)
            st.caption("**Top 10 Influencers**: Most active accounts by post volume.")

            # Platform Distribution
            if 'Platform' in filtered_df_global.columns:
                platform_counts = filtered_df_global['Platform'].value_counts()
                fig_platform = px.bar(platform_counts, title="Post Distribution by Platform")
                st.plotly_chart(fig_platform, use_container_width=True)
                st.caption("**Post Distribution by Platform**: Where content is appearing.")

            # Top Hashtags
            social_media_df = filtered_df_global[~filtered_df_global['Platform'].isin(['Media', 'News/Media', 'Unknown'])].copy()
            if not social_media_df.empty and 'original_text' in social_media_df.columns:
                social_media_df['hashtags'] = social_media_df['original_text'].astype(str).str.findall(r'#\w+')
                all_hashtags = [tag.lower() for tags_list in social_media_df['hashtags'] if isinstance(tags_list, list) for tag in tags_list]
                if all_hashtags:
                    hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                    if not hashtag_counts.empty:
                        fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags (Social Media Only)")
                        st.plotly_chart(fig_ht, use_container_width=True)
                        st.caption("**Top 10 Hashtags**: Trending topics on social platforms.")

            # Daily Post Volume
            plot_df = filtered_df_global.copy()
            if 'timestamp_share' in plot_df.columns:
                plot_df['datetime'] = pd.to_datetime(plot_df['timestamp_share'], unit='s', utc=True)
                plot_df = plot_df.set_index('datetime')
                time_series = plot_df.resample('D').size()
                if not time_series.empty:
                    fig_ts = px.area(time_series, title="Daily Post Volume")
                    st.plotly_chart(fig_ts, use_container_width=True)
                    st.caption("**Daily Post Volume**: Spikes may indicate emerging narratives.")

            # High Reposts Table
            if 'original_text' in filtered_df_global.columns:
                repost_counts = filtered_df_global.groupby('original_text').filter(lambda x: len(x) > 1)
                if not repost_counts.empty:
                    repost_summary = repost_counts.groupby('original_text').agg(
                        repost_count=('account_id', 'size'),
                        unique_accounts=('account_id', 'nunique'),
                        platforms=('Platform', lambda x: ', '.join(x.unique())),
                        sample_url=('URL', 'first')
                    ).reset_index().sort_values('repost_count', ascending=False).head(10)
                    if not repost_summary.empty:
                        st.markdown("### 🔁 High Reposts")
                        st.dataframe(repost_summary[['repost_count', 'unique_accounts', 'platforms', 'sample_url']], use_container_width=True)
                        st.caption("**High Reposts**: Content shared by multiple accounts.")
        # Tab 2: Coordination
        with tab2:
            if not is_preprocessed_mode and 'cluster' in df_clustered.columns:
                from collections import defaultdict
                grouped = df_clustered[df_clustered['cluster'] != -1].groupby('cluster')
                for cluster_id, group in grouped:
                    if len(group) < 2: continue
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
                                    adj[i].append(j); adj[j].append(i)
                        visited = set()
                        for i in range(len(clean_df)):
                            if i not in visited:
                                group_indices = []
                                q = [i]; visited.add(i)
                                while q:
                                    u = q.pop(0); group_indices.append(u)
                                    for v in adj[u]:
                                        if v not in visited: visited.add(v); q.append(v)
                                if len(group_indices) > 1 and len(clean_df.iloc[group_indices]['account_id'].unique()) > 1:
                                    coordination_groups.append({
                                        "posts": clean_df.iloc[group_indices].to_dict('records'),
                                        "num_posts": len(group_indices),
                                        "num_accounts": len(clean_df.iloc[group_indices]['account_id'].unique()),
                                        "max_similarity_score": round(cosine_sim[np.ix_(group_indices, group_indices)].max(), 3),
                                        "coordination_type": "TBD"
                                    })
                    except Exception:
                        continue

            if coordination_groups:
                st.info(f"Found {len(coordination_groups)} coordinated groups.")
                for i, group in enumerate(coordination_groups):
                    st.markdown(f"#### Group {i+1}: {group['coordination_type']}")
                    st.write(f"**Posts:** {group['num_posts']} | **Accounts:** {group['num_accounts']}")
                    posts_df = pd.DataFrame(group['posts'])
                    posts_df['Timestamp'] = pd.to_datetime(posts_df['timestamp_share'], unit='s', utc=True)
                    st.dataframe(posts_df[['account_id', 'Platform', 'Timestamp', 'URL']], width="stretch")
            else:
                st.info("No coordinated groups found.")

        # Tab 3: Network An
        with tab3:
            st.markdown("Use the slider below to limit the number of accounts displayed in the network graph.")
            if 'max_nodes_to_display' not in st.session_state:
                st.session_state.max_nodes_to_display = 40
            st.session_state.max_nodes_to_display = st.slider(
                "Maximum Nodes to Display in Graph",
                min_value=10, max_value=200, value=st.session_state.max_nodes_to_display, step=10,
                help="Limit the graph to the top N most central accounts to improve visibility and focus on key influencers."
            )
            st.markdown("---")
            st.markdown("This visualization shows a network of accounts involved in coordinated activity. A link between two accounts means they posted similar content or shared the same URL.")
            
            if not df_for_analysis.empty:
                if coordination_mode == "Text Content":
                    df_for_graph = df_for_analysis
                    with st.spinner("🗂️ Pre-processing data for network graph..."):
                        clustered_df_for_graph = cached_clustering(df_for_graph, eps=0.3, min_samples=2, max_features=5000, data_source_key="graph")
                    G, pos, cluster_map = build_user_interaction_graph(clustered_df_for_graph, coordination_type="text")
                    st.info(f"Displaying a network of the top {st.session_state.max_nodes_to_display} most connected accounts.")
                    st.info("Nodes are accounts, colored by content cluster. Edges show co-participation in a cluster.")
                elif coordination_mode == "Shared URLs":
                    df_for_graph = df_for_analysis
                    G, pos, cluster_map = build_user_interaction_graph(df_for_graph, coordination_type="url")
                    st.info(f"Displaying a network of the top {st.session_state.max_nodes_to_display} most connected accounts.")
                    st.info("Nodes are accounts, colored by a grouping of shared URLs. Edges show co-sharing of URLs.")

                if not G.nodes():
                    st.warning("No coordinated activity detected to build a network graph.")
                else:
                    fig_net = go.Figure()
                    edge_x, edge_y = [], []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'))
                    node_x, node_y, node_text, node_color = [], [], [], []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        hover_text = f"User: {node}<br>Platform: {G.nodes[node].get('platform', 'N/A')}"
                        node_text.append(hover_text)
                        cluster_id = cluster_map.get(node)
                        if isinstance(cluster_id, str):
                            node_color.append(hash(cluster_id) % 100)
                        elif cluster_id not in [-1, -2]:
                            node_color.append(cluster_id)
                        else:
                            node_color.append(-1)
                    nodes_df = pd.DataFrame({
                        'x': node_x, 'y': node_y, 'text': node_text, 'color': node_color,
                        'size': [G.degree(node) for node in G.nodes()]
                    })
                    fig_net.add_trace(go.Scatter(
                        x=nodes_df['x'], y=nodes_df['y'], mode='markers', hoverinfo='text', text=nodes_df['text'],
                        marker=dict(showscale=False, colorscale='Viridis', size=nodes_df['size'] * 1.5 + 5, color=nodes_df['color'], line_width=2, opacity=0.8)
                    ))
                    fig_net.update_layout(
                        title='Network of Coordinated Accounts', showlegend=True, hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=700
                    )
                    st.plotly_chart(fig_net, width="stretch")

                    st.markdown("### Risk & Influence Assessment")
                    st.markdown("""
                    **Centrality Analysis**: Accounts with high centrality (many connections) are key nodes in the network, potentially acting as amplifiers or originators of a message.
                    - **Degree Centrality**: The number of connections a node has. High degree means an account is co-participating with many others.
                    """)
                    degree_centrality = nx.degree_centrality(G)
                    risk_df = pd.DataFrame(degree_centrality.items(), columns=['Account', 'Degree Centrality'])
                    risk_df = risk_df.sort_values(by='Degree Centrality', ascending=False).reset_index(drop=True)
                    risk_df['Risk Score'] = (risk_df['Degree Centrality'] / risk_df['Degree Centrality'].max()) * 100
                    risk_df = risk_df.merge(filtered_df_global[['account_id', 'Platform']].drop_duplicates(), left_on='Account', right_on='account_id', how='left').drop(columns='account_id')
                    risk_df = risk_df.head(20)
                    if not risk_df.empty:
                        st.markdown("#### Top 20 Most Central Accounts (by Degree Centrality)")
                        st.dataframe(risk_df, width="stretch")
                        risk_csv = convert_df_to_csv(risk_df)
                        st.download_button("Download Risk Assessment CSV", risk_csv, "risk_assessment.csv", "text/csv")
                    else:
                        st.warning("No network data available for risk assessment.")
            else:
                st.info("No data available to generate a network graph.")

        # Tab 4: Summary
        with tab4:
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
