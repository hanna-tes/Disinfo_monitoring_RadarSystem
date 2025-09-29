import pandas as pd
import numpy as np
import re
import logging
import time
import random
from datetime import timedelta
from itertools import combinations
import streamlit as st
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

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

# --- Helper Functions ---
def safe_llm_call(prompt, max_tokens=2048):
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens
        )
        return response
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None

def summarize_cluster(texts, urls, cluster_data):
    joined = "\n".join(texts[:50])
    url_context = "\nRelevant post links:\n" + "\n".join(urls[:5]) if urls else ""
    cluster_data = cluster_data.copy()
    cluster_data['Timestamp'] = pd.to_datetime(cluster_data['timestamp_share'], unit='s', utc=True)
    min_ts = cluster_data['Timestamp'].min().strftime('%Y-%m-%d %H:%M')
    max_ts = cluster_data['Timestamp'].max().strftime('%Y-%m-%d %H:%M')
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
        if evidence_urls:
            urls_section = "\n\nSources: " + ", ".join(evidence_urls[:5])
            cleaned_summary += urls_section
        return cleaned_summary, evidence_urls
    else:
        return "Summary generation failed.", []

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
    cleaned = re.sub(r'\b\d{1,2}\s+(january|...|dec)\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', cleaned)
    cleaned = re.sub(r'\b\d{4}\b', '', cleaned)
    cleaned = re.sub(r"\n|\r|\t", " ", cleaned).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.lower()

def parse_timestamp_robust(timestamp):
    if pd.isna(timestamp):
        return None
    if isinstance(timestamp, (int, float)):
        if 0 < timestamp < 253402300800:
            return int(timestamp)
        return None
    date_formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%b %d, %Y @ %H:%M:%S.%f', '%d-%b-%Y %I:%M%p',
        '%A, %d %b %Y %H:%M:%S', '%b %d, %I:%M%p', '%d %b %Y %I:%M%p',
        '%Y-%m-%d', '%m/%d/%Y', '%d %b %Y',
    ]
    try:
        parsed = pd.to_datetime(timestamp, errors='coerce', utc=True)
        if pd.notna(parsed):
            return int(parsed.timestamp())
    except:
        pass
    for fmt in date_formats:
        try:
            parsed = pd.to_datetime(timestamp, format=fmt, errors='coerce', utc=True)
            if pd.notna(parsed):
                return int(parsed.timestamp())
        except (ValueError, TypeError):
            continue
    return None

def process_preprocessed_data(preprocessed_df):
    if preprocessed_df.empty:
        return pd.DataFrame()
    preprocessed_df.columns = preprocessed_df.columns.str.strip().str.lower()
    col_map = {
        'country': 'country',
        'evidence': 'evidence',
        'context': 'context',
        'urls': 'urls',
        'emerging virality': 'emerging_virality',
        'emerging_virality': 'emerging_virality'
    }
    df_out = pd.DataFrame()
    def safe_get_col(df, col_name, fallback=""):
        if col_name in df.columns:
            return df[col_name]
        for original_col in df.columns:
            if original_col.lower().replace(' ', '_') == col_name:
                return df[original_col]
        return pd.Series([fallback] * len(df))
    df_out['original_text'] = safe_get_col(preprocessed_df, 'context', 'No Context Provided').astype(str)
    df_out['URL'] = safe_get_col(preprocessed_df, 'urls', '').astype(str)
    df_out['Emerging Virality'] = safe_get_col(preprocessed_df, 'emerging_virality', 'Unknown').astype(str)
    df_out['Country'] = safe_get_col(preprocessed_df, 'country', 'Uganda').astype(str)
    df_out['account_id'] = 'Summary_Author'
    df_out['object_id'] = df_out['original_text']
    df_out['content_id'] = 'SUMMARY_' + preprocessed_df.index.astype(str)
    df_out['timestamp_share'] = int(pd.Timestamp.now(tz='UTC').timestamp())
    df_out['timestamp_share'] = df_out['timestamp_share'].astype('Int64')
    df_out['source_dataset'] = 'Preprocessed_Summary'
    df_out['Platform'] = 'Report'
    df_out['Outlet'] = 'Report'
    df_out['Channel'] = 'Summary'
    df_out['is_preprocessed_summary'] = True
    df_out['cluster'] = preprocessed_df.index
    return df_out[df_out['original_text'].str.strip() != ""].reset_index(drop=True)

def read_uploaded_file(uploaded_file, file_name):
    if not uploaded_file:
        return pd.DataFrame()
    bytes_data = uploaded_file.getvalue()
    for enc in ['utf-8-sig', 'utf-16le', 'utf-16be', 'utf-16', 'latin1', 'cp1252']:
        try:
            decoded_content = bytes_data.decode(enc)
            break
        except (UnicodeDecodeError, AttributeError):
            continue
    else:
        st.error(f"❌ Failed to decode {file_name}")
        return pd.DataFrame()
    sep = '\t' if '\t' in decoded_content.strip().splitlines()[0] else ','
    try:
        return pd.read_csv(StringIO(decoded_content), sep=sep, low_memory=False)
    except Exception as e:
        st.error(f"❌ Parse error in {file_name}: {e}")
        return pd.DataFrame()

def combine_social_media_data(meltwater_df, civicsignals_df, openmeasure_df=None, meltwater_object_col='hit sentence', civicsignals_object_col='title', openmeasure_object_col='text'):
    combined_dfs = []
    def get_col(df, col):
        return df[col] if col in df.columns else pd.Series([np.nan]*len(df))
    for source_df, obj_col, name in [(meltwater_df, meltwater_object_col, 'Meltwater'), (civicsignals_df, civicsignals_object_col, 'CivicSignals'), (openmeasure_df, openmeasure_object_col, 'OpenMeasure')]:
        if source_df is not None and not source_df.empty:
            df = source_df.copy()
            df.columns = df.columns.str.lower()
            out = pd.DataFrame()
            out['account_id'] = get_col(df, 'influencer' if name=='Meltwater' else 'media_name' if name=='CivicSignals' else 'actor_username')
            out['content_id'] = get_col(df, 'tweet id' if name=='Meltwater' else 'stories_id' if name=='CivicSignals' else 'id')
            out['object_id'] = get_col(df, obj_col.lower())
            out['original_url'] = get_col(df, 'url')
            out['timestamp_share'] = get_col(df, 'date' if name=='Meltwater' else 'publish_date' if name=='CivicSignals' else 'created_at')
            out['source_dataset'] = name
            combined_dfs.append(out)
    if not combined_dfs:
        return pd.DataFrame()
    combined = pd.concat(combined_dfs, ignore_index=True)
    combined = combined.dropna(subset=['account_id', 'content_id', 'timestamp_share', 'object_id'])
    combined['account_id'] = combined['account_id'].astype(str).fillna('Unknown_User').replace('nan', 'Unknown_User')
    combined['content_id'] = combined['content_id'].astype(str).str.replace('"', '', regex=False).str.strip()
    combined['original_url'] = combined['original_url'].astype(str).fillna('').replace('nan', '')
    combined['object_id'] = combined['object_id'].astype(str).fillna('').replace('nan', '')
    combined['timestamp_share'] = combined['timestamp_share'].apply(parse_timestamp_robust)
    combined = combined.dropna(subset=['timestamp_share']).reset_index(drop=True)
    combined['timestamp_share'] = combined['timestamp_share'].astype('Int64')
    combined = combined[combined['object_id'].str.strip() != ""].copy()
    combined = combined.drop_duplicates(subset=['account_id', 'content_id', 'object_id', 'timestamp_share']).reset_index(drop=True)
    return combined

def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    df_processed = df.copy()
    df_processed.columns = [c.lower() for c in df_processed.columns]
    df_processed.rename(columns={
        'original_url': 'URL',
        'permalink': 'URL',
        'post_link': 'URL',
        'user_id': 'account_id',
        'author_id': 'account_id',
        'author': 'account_id',
        'hit_sentence': 'object_id',
        'title': 'object_id',
        'text': 'object_id',
        'time_stamp': 'timestamp_share',
        'created_at': 'timestamp_share',
        'date_posted': 'timestamp_share'
    }, inplace=True)
    for col in ['account_id', 'content_id', 'object_id', 'timestamp_share', 'URL']:
        if col not in df_processed.columns:
            df_processed[col] = np.nan
    df_processed['timestamp_share'] = df_processed['timestamp_share'].apply(parse_timestamp_robust)
    df_processed = df_processed.dropna(subset=['timestamp_share']).reset_index(drop=True)
    df_processed['timestamp_share'] = df_processed['timestamp_share'].astype('Int64')
    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan', '').fillna('')
    df_processed['URL'] = df_processed['URL'].astype(str).replace('nan', '').fillna('')
    df_processed = df_processed[df_processed['object_id'].str.strip() != ""].copy()
    if coordination_mode == "Text Content":
        df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    else:
        df_processed['original_text'] = df_processed['URL'].astype(str).replace('nan', '').fillna('')
    df_processed = df_processed[df_processed['original_text'].str.strip() != ""].reset_index(drop=True)
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)
    if 'cluster' not in df_processed.columns: df_processed['cluster'] = -1
    if 'source_dataset' not in df_processed.columns: df_processed['source_dataset'] = 'Uploaded_File'
    for col in ['Outlet', 'Channel']:
        if col not in df_processed.columns: df_processed[col] = np.nan
    return df_processed[['account_id', 'content_id', 'object_id', 'URL', 'timestamp_share', 'Platform', 'original_text', 'Outlet', 'Channel', 'cluster', 'source_dataset']].copy()

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

def build_user_interaction_graph(df, coordination_type="text"):
    G = nx.Graph()
    influencer_column = 'account_id'

    if coordination_type == "text":
        if 'cluster' not in df.columns:
            return G, {}, {}
        grouped = df.groupby('cluster')
        for cluster_id, group in grouped:
            if cluster_id == -1 or len(group[influencer_column].unique()) < 2:
                for user in group[influencer_column].dropna().unique():
                    if user not in G:
                        G.add_node(user, cluster=cluster_id)
                continue
            users_in_cluster = group[influencer_column].dropna().unique().tolist()
            for u1, u2 in combinations(users_in_cluster, 2):
                if G.has_edge(u1, u2):
                    G[u1][u2]['weight'] += 1
                else:
                    G.add_edge(u1, u2, weight=1)

    elif coordination_type == "url":
        if 'URL' not in df.columns:
            return G, {}, {}
        url_groups = df.groupby('URL')
        for url_shared, group in url_groups:
            if pd.isna(url_shared) or url_shared.strip() == "":
                continue
            users_sharing_url = group[influencer_column].dropna().unique().tolist()
            if len(users_sharing_url) < 2:
                for user in users_sharing_url:
                    if user not in G:
                        G.add_node(user)
                continue
            for u1, u2 in combinations(users_sharing_url, 2):
                if G.has_edge(u1, u2):
                    G[u1][u2]['weight'] += 1
                else:
                    G.add_edge(u1, u2, weight=1)

    all_influencers = df[influencer_column].dropna().unique().tolist()
    influencer_platform_map = df.groupby(influencer_column)['Platform'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').to_dict()

    for inf in all_influencers:
        if inf not in G.nodes():
            G.add_node(inf)
        G.nodes[inf]['platform'] = influencer_platform_map.get(inf, 'Unknown')
        if coordination_type == "text":
            clusters = df[df[influencer_column] == inf]['cluster'].dropna()
            G.nodes[inf]['cluster'] = clusters.mode()[0] if not clusters.empty else -2
        elif coordination_type == "url":
            shared_urls = df[(df[influencer_column] == inf) & df['URL'].notna() & (df['URL'].str.strip() != '')]['URL'].unique()
            G.nodes[inf]['cluster'] = f"SharedURL_Group_{hash(tuple(sorted(shared_urls))) % 100}" if len(shared_urls) > 0 else "NoSharedURL"

    if G.nodes():
        node_degrees = dict(G.degree())
        sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)
        top_n_nodes = sorted_nodes[:st.session_state.max_nodes_to_display]
        subgraph = G.subgraph(top_n_nodes)
        pos = nx.kamada_kawai_layout(subgraph)
        cluster_map = {node: G.nodes[node].get('cluster', -2) for node in subgraph.nodes()}
        return subgraph, pos, cluster_map
    else:
        return G, {}, {}

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def virality_badge(val):
    s = str(val).lower()
    if "tier 4" in s or "emergency" in s:
        return '<span style="background-color: #ffebee; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #c62828;">🚨 Viral Emergency</span>'
    elif "tier 3" in s or "high" in s:
        return '<span style="background-color: #fff3e0; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #e65100;">🔥 High Virality</span>'
    elif "tier 2" in s or "medium" in s:
        return '<span style="background-color: #e8f5e9; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #2e7d32;">📈 Medium Virality</span>'
    else:
        return '<span style="background-color: #f5f5f5; padding: 4px 8px; border-radius: 6px; color: #555;">ℹ️ Low/Unknown</span>'

# --- Main App ---
def main():
    st.set_page_config(layout="wide", page_title="Election Monitoring Dashboard")
    st.title("Election Monitoring Dashboard")

    # Early exit: no files uploaded
    st.sidebar.header("⚙️ Configuration")
    coordination_mode = st.sidebar.selectbox("Coordination Type", ["Text Content", "Shared URLs"])
    data_source = st.sidebar.selectbox("Data Source", ["Upload CSV Files"])

    st.sidebar.info("Upload your data below.")
    uploaded_preprocessed = st.sidebar.file_uploader("Upload Preprocessed Summary CSV", type=["csv"], key="preprocessed_upload")
    uploaded_meltwater = st.sidebar.file_uploader("Upload Meltwater CSV", type=["csv"], key="meltwater_upload")
    uploaded_civicsignals = st.sidebar.file_uploader("Upload CivicSignals CSV", type=["csv"], key="civicsignals_upload")
    uploaded_openmeasure = st.sidebar.file_uploader("Upload Open-Measure CSV", type=["csv"], key="openmeasure_upload")

    if not any([uploaded_preprocessed, uploaded_meltwater, uploaded_civicsignals, uploaded_openmeasure]):
        st.info("📤 Please upload data using the sidebar to begin analysis.")
        st.stop()

    # Process uploads
    combined_raw_df = pd.DataFrame()
    is_preprocessed_mode = False

    preprocessed_df = read_uploaded_file(uploaded_preprocessed, "Preprocessed Summary")
    meltwater_df = read_uploaded_file(uploaded_meltwater, "Meltwater")
    civicsignals_df = read_uploaded_file(uploaded_civicsignals, "CivicSignals")
    openmeasure_df = read_uploaded_file(uploaded_openmeasure, "Open-Measure")

    if not preprocessed_df.empty:
        with st.spinner("Processing preprocessed summary..."):
            combined_raw_df = process_preprocessed_data(preprocessed_df)
        is_preprocessed_mode = True
    else:
        with st.spinner("Combining raw datasets..."):
            obj_map = {
                "meltwater": "hit sentence" if coordination_mode == "Text Content" else "url",
                "civicsignals": "title" if coordination_mode == "Text Content" else "url",
                "openmeasure": "text" if coordination_mode == "Text Content" else "url"
            }
            combined_raw_df = combine_social_media_data(
                meltwater_df, civicsignals_df, openmeasure_df,
                meltwater_object_col=obj_map["meltwater"],
                civicsignals_object_col=obj_map["civicsignals"],
                openmeasure_object_col=obj_map["openmeasure"]
            )

    if combined_raw_df.empty:
        st.error("❌ No valid data could be loaded from the uploaded files.")
        st.stop()

    # Preprocessing
    if is_preprocessed_mode:
        df = combined_raw_df.copy()
    else:
        with st.spinner("Preprocessing data..."):
            df = final_preprocess_and_map_columns(combined_raw_df, coordination_mode=coordination_mode)
        if df.empty:
            st.error("❌ No valid data after preprocessing.")
            st.stop()

    # Global Filters
    if 'timestamp_share' not in df.columns or df['timestamp_share'].isnull().all():
        min_date = max_date = pd.Timestamp.now().date()
    else:
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
    ].copy() if 'timestamp_share' in df.columns else pd.DataFrame()

    max_posts = st.sidebar.number_input("Limit Posts for Analysis (0 for all)", min_value=0, value=0, step=1000)
    df_for_analysis = filtered_df_global.sample(n=max_posts, random_state=42).copy() if max_posts > 0 and len(filtered_df_global) > max_posts else filtered_df_global.copy()

    # Analysis & Report Generation
    df_clustered = df_for_analysis.copy()
    coordination_groups = []
    if not is_preprocessed_mode and not df_for_analysis.empty:
        df_clustered = cached_clustering(df_for_analysis, 0.3, 2, 5000, "report")

    # Report Generation
    if is_preprocessed_mode:
        report_df = df_clustered[['Country', 'original_text', 'URL', 'Emerging Virality']].rename(columns={'original_text': 'Context', 'URL': 'URLs'})
        report_df['Evidence'] = report_df['URLs'].apply(lambda x: ", ".join(eval(x)[:5]) if isinstance(x, str) and x.startswith('[') else x)
        report_df = report_df[['Country', 'Evidence', 'Context', 'URLs', 'Emerging Virality']]
    else:
        all_summaries = []
        if 'cluster' in df_clustered.columns:
            valid_clusters = df_clustered[df_clustered['cluster'] != -1]['cluster'].unique()[:50]
            for cluster_id in valid_clusters:
                cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                texts = cluster_data['original_text'].tolist()
                urls = cluster_data['URL'].dropna().unique().tolist()
                summary, evidence_urls = summarize_cluster(texts, urls, cluster_data)
                all_summaries.append({
                    "Evidence": ", ".join(evidence_urls[:5]),
                    "Context": summary,
                    "URLs": str(urls),
                    "Emerging Virality": "Tier 4: Viral Emergency (Requires immediate response)"
                })
        report_df = pd.DataFrame(all_summaries)

    # Tabs
    if is_preprocessed_mode:
        st.header("📊 Narrative Insights from Preprocessed IMI Report")
        if report_df.empty:
            st.info("No data to display.")
        else:
            for idx, row in report_df.iterrows():
                context = row.get('Context', 'No narrative available')
                urls = row.get('URLs', '')
                if isinstance(urls, str):
                    if urls.startswith('['):
                        try:
                            url_list = eval(urls)
                        except:
                            url_list = [u.strip() for u in urls.split(',') if u.strip().startswith('http')]
                    else:
                        url_list = [u.strip() for u in urls.split(',') if u.strip().startswith('http')]
                else:
                    url_list = urls if isinstance(urls, list) else []

                virality = row['Emerging Virality']
                if "tier 4" in str(virality).lower():
                    badge = '<span style="background-color: #ffebee; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #c62828;">🚨 Viral Emergency</span>'
                elif "tier 3" in str(virality).lower():
                    badge = '<span style="background-color: #fff3e0; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #e65100;">🔥 High Virality</span>'
                elif "tier 2" in str(virality).lower():
                    badge = '<span style="background-color: #e8f5e9; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #2e7d32;">📈 Medium Virality</span>'
                else:
                    badge = '<span style="background-color: #f5f5f5; padding: 4px 8px; border-radius: 6px; color: #555;">ℹ️ Low/Unknown</span>'

                title_preview = context.split('\n')[0][:120] + ("..." if len(context) > 120 else "")
                with st.expander(f"**{title_preview}**"):
                    st.markdown("### 📖 Narrative Summary")
                    st.markdown(context)
                    st.markdown("### ⚠️ Virality Level")
                    st.markdown(badge, unsafe_allow_html=True)
                    if url_list:
                        st.markdown("### 🔗 Supporting Evidence (Click to Open)")
                        for url in url_list[:10]:
                            st.markdown(f"- <a href='{url}' target='_blank' style='text-decoration: underline; color: #1f77b4;'>{url}</a>", unsafe_allow_html=True)
                    else:
                        st.markdown("### 🔗 Supporting Evidence\n- No URLs available.")
                st.markdown("---")
            st.download_button("📥 Download Report", convert_df_to_csv(report_df), "imi_report.csv", "text/csv")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Coordination", "🕸️ Network", "📝 Summary"])

        # Tab 1: Overview
        with tab1:
            st.markdown("### 🔬 Preprocessed Data Sample")
            st.markdown(f"**Data Source:** `{data_source}` | **Coordination Mode:** `{coordination_mode}` | **Total Rows:** `{len(df):,}`")
            display_cols = ['account_id', 'content_id', 'object_id', 'timestamp_share', 'URL', 'Platform']
            existing_cols = [col for col in display_cols if col in df.columns]
            if existing_cols:
                st.dataframe(df[existing_cols].head(10), width="stretch")
            else:
                st.info("No data available to display.")

            if not filtered_df_global.empty:
                # Top Influencers
                top_influencers = filtered_df_global['account_id'].value_counts().head(10)
                fig_src = px.bar(top_influencers, title="Top 10 Influencers")
                st.plotly_chart(fig_src, width="stretch")
                st.markdown("**Top 10 Influencers**: Shows the most active accounts by post volume.")

                # Platform Distribution
                if 'Platform' in filtered_df_global.columns:
                    platform_counts = filtered_df_global['Platform'].value_counts()
                    fig_platform = px.bar(platform_counts, title="Post Distribution by Platform")
                    st.plotly_chart(fig_platform, width="stretch")
                    st.markdown("**Post Distribution by Platform**: Breakdown of content sources across social media and news outlets.")

                # Top Hashtags
                social_media_df = filtered_df_global[~filtered_df_global['Platform'].isin(['Media', 'News/Media', 'Unknown', 'Report'])].copy()
                if not social_media_df.empty and 'original_text' in social_media_df.columns:
                    social_media_df['hashtags'] = social_media_df['original_text'].astype(str).str.findall(r'#\w+')
                    all_hashtags = [tag.lower() for tags_list in social_media_df['hashtags'] if isinstance(tags_list, list) for tag in tags_list]
                    if all_hashtags:
                        hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                        if not hashtag_counts.empty:
                            fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags (Social Media Only)")
                            st.plotly_chart(fig_ht, width="stretch")
                            st.markdown("**Top 10 Hashtags**: Most frequently used hashtags on social platforms, indicating trending topics.")

                # Daily Post Volume
                plot_df = filtered_df_global.copy()
                if 'timestamp_share' in plot_df.columns:
                    plot_df['datetime'] = pd.to_datetime(plot_df['timestamp_share'], unit='s', utc=True)
                    plot_df = plot_df.set_index('datetime')
                    time_series = plot_df.resample('D').size()
                    if not time_series.empty:
                        fig_ts = px.area(time_series, title="Daily Post Volume")
                        st.plotly_chart(fig_ts, width="stretch")
                        st.markdown("**Daily Post Volume**: Tracks the number of posts per day to identify spikes in activity or emerging narratives.")

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

        # Tab 3: Network
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
                st.info("No narratives to display. Try adjusting filters or generating a new report.")
            else:
                report_df = report_df.copy()
                def virality_sort_key(val):
                    s = str(val).lower()
                    if "tier 4" in s: return 4
                    elif "tier 3" in s: return 3
                    elif "tier 2" in s: return 2
                    else: return 1
                report_df['virality_score'] = report_df['Emerging Virality'].apply(virality_sort_key)
                report_df = report_df.sort_values('virality_score', ascending=False).drop(columns='virality_score')
                for idx, row in report_df.iterrows():
                    context = row.get('Context', row.get('original_text', 'No narrative available'))
                    urls = row.get('URLs', row.get('URL', ''))
                    if isinstance(urls, str):
                        if urls.startswith('['):
                            try:
                                url_list = eval(urls)
                            except:
                                url_list = [u.strip() for u in urls.split(',') if u.strip().startswith('http')]
                        else:
                            url_list = [u.strip() for u in urls.split(',') if u.strip().startswith('http')]
                    else:
                        url_list = urls if isinstance(urls, list) else []

                    virality = row['Emerging Virality']
                    if "tier 4" in str(virality).lower():
                        badge = '<span style="background-color: #ffebee; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #c62828;">🚨 Viral Emergency</span>'
                    elif "tier 3" in str(virality).lower():
                        badge = '<span style="background-color: #fff3e0; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #e65100;">🔥 High Virality</span>'
                    elif "tier 2" in str(virality).lower():
                        badge = '<span style="background-color: #e8f5e9; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #2e7d32;">📈 Medium Virality</span>'
                    else:
                        badge = '<span style="background-color: #f5f5f5; padding: 4px 8px; border-radius: 6px; color: #555;">ℹ️ Low/Unknown</span>'

                    title_preview = context.split('\n')[0][:120] + ("..." if len(context) > 120 else "")
                    with st.expander(f"**{title_preview}**"):
                        st.markdown("### 📖 Narrative Summary")
                        st.markdown(context)
                        st.markdown("### ⚠️ Virality Level")
                        st.markdown(badge, unsafe_allow_html=True)
                        if url_list:
                            st.markdown("### 🔗 Supporting Evidence (Click to Open)")
                            for url in url_list[:10]:
                                st.markdown(f"- <a href='{url}' target='_blank' style='text-decoration: underline; color: #1f77b4;'>{url}</a>", unsafe_allow_html=True)
                        else:
                            st.markdown("### 🔗 Supporting Evidence\n- No URLs available.")
                        if 'First Detected' in row and 'Last Updated' in row:
                            first_ts = row['First Detected'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['First Detected']) else "N/A"
                            last_ts = row['Last Updated'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['Last Updated']) else "N/A"
                            st.markdown(f"### 📅 Narrative Lifecycle\n- **First Detected:** {first_ts}\n- **Last Updated:** {last_ts}")
                    st.markdown("---")
                csv_data = convert_df_to_csv(report_df)
                st.download_button("📥 Download Full Report (CSV)", csv_data, "imi_narrative_report.csv", "text/csv")

if __name__ == '__main__':
    main()
