import pandas as pd
import numpy as np
import re
import logging
import time
import random
from datetime import timedelta
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
    "bertrend": {
        "min_cluster_size": 3,
    },
    "analysis": {
        "time_window": "48H",
    },
    "coordination_detection": {
        "threshold": 0.85,
        "max_features": 5000
    }
}

# --- Groq Setup (via Streamlit Secrets) ---
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")  
    if GROQ_API_KEY:
        import groq
        client = groq.Groq(api_key=GROQ_API_KEY)
    else:
        logger.warning("GROQ_API_KEY not found in st.secrets. Set client to None.")
        client = None
except Exception as e:
    logger.error(f"Error initializing Groq client: {e}")
    client = None

# --- Helper: Safe LLM call ---
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

# --- IMI Summary Generator ---
def summarize_cluster(texts, urls, cluster_data):
    joined = "\n".join(texts[:50])
    url_context = "\nRelevant post links:\n" + "\n".join(urls[:5]) if urls else ""
    
    # Convert UNIX timestamp_share to UTC datetime for display
    cluster_data = cluster_data.copy()
    cluster_data['Timestamp'] = pd.to_datetime(
        cluster_data['timestamp_share'], 
        unit='s', 
        utc=True
    )
    min_timestamp_str = cluster_data['Timestamp'].min().strftime('%Y-%m-%d %H:%M')
    max_timestamp_str = cluster_data['Timestamp'].max().strftime('%Y-%m-%d %H:%M')

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
  - First Detected: {min_timestamp_str}
  - Last Updated: {max_timestamp_str}
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
        logger.error(f"LLM call failed for cluster")
        return "Summary generation failed.", []

# --- Generate IMI Report (CSV Output) ---
@st.cache_data(show_spinner="Generating Narrative Summaries...")
def generate_imi_report(clustered_df, data_source_key):
    if clustered_df.empty or 'cluster' not in clustered_df.columns:
        return pd.DataFrame()
    
    # Create a copy and add Timestamp for processing
    df = clustered_df.copy()
    df['Timestamp'] = pd.to_datetime(df['timestamp_share'], unit='s', utc=True)
    df = df.dropna(subset=['Timestamp'])

    report_data = []
    valid_clusters = df[df['cluster'] != -1]['cluster'].unique()[:50]  # Limit to 50

    for cluster_id in valid_clusters:
        cluster_data = df[df['cluster'] == cluster_id]
        texts = cluster_data['original_text'].tolist()
        urls = cluster_data['URL'].dropna().unique().tolist()
        summary, evidence_urls = summarize_cluster(texts, urls, cluster_data)

        # Build CSV row
        report_data.append({
            "Country": "Uganda",
            "Evidence": ", ".join(evidence_urls[:5]),
            "Context": summary,
            "URLs": str(urls),  # Full list as string
            "Emerging Virality": "Tier 4: Viral Emergency (Requires immediate response)"
        })

    return pd.DataFrame(report_data)

# --- Core Helper Functions ---
def infer_platform_from_url(url):
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return "Unknown"
    url = url.lower()
    if "tiktok.com" in url:
        return "TikTok"
    elif "facebook.com" in url or "fb.watch" in url:
        return "Facebook"
    elif "twitter.com" in url or "x.com" in url:
        return "X"
    elif "youtube.com" in url or "youtu.be" in url:
        return "YouTube"
    elif "instagram.com" in url:
        return "Instagram"
    elif "telegram.me" in url or "t.me" in url:
        return "Telegram"
    elif any(domain in url for domain in ["nytimes.com", "bbc.com", "cnn.com", "reuters.com", "theguardian.com", "aljazeera.com", "lemonde.fr", "dw.com"]):
        return "News/Media"
    else:
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
    cleaned = re.sub(r"\\n|\\r|\\t", " ", cleaned).strip()
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
    df_out = pd.DataFrame()
    df_out['original_text'] = preprocessed_df.get('Context', 'No Context').astype(str)
    df_out['URL'] = preprocessed_df.get('URLs', '').astype(str)
    df_out['Emerging Virality'] = preprocessed_df.get('Emerging Virality', 'Unknown').astype(str)
    df_out['Country'] = preprocessed_df.get('Country', 'Uganda').astype(str)
    df_out['account_id'] = 'Summary_Author'
    df_out['object_id'] = df_out['original_text']
    df_out['content_id'] = 'SUMMARY_' + preprocessed_df.index.astype(str)
    df_out['timestamp_share'] = int(pd.Timestamp.now(tz='UTC').timestamp())  # Dummy UNIX
    df_out['timestamp_share'] = df_out['timestamp_share'].astype('Int64')
    df_out['source_dataset'] = 'Preprocessed_Summary'
    df_out['Platform'] = 'Report'
    df_out['Outlet'] = 'Report'
    df_out['Channel'] = 'Summary'
    df_out['cluster'] = preprocessed_df.index
    df_out['is_preprocessed_summary'] = True
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

@st.cache_data(show_spinner="Detecting Coordinated Groups...")
def find_coordinated_groups(df, threshold, max_features):
    text_col = 'original_text'
    social_media_platforms = {'TikTok', 'Facebook', 'X', 'YouTube', 'Instagram', 'Telegram'}
    coordination_groups = {}
    clustered_groups = df[df['cluster'] != -1].groupby('cluster')
    for cluster_id, group in clustered_groups:
        if len(group) < 2: continue
        clean_df = group[['account_id', 'timestamp_share', 'Platform', 'URL', text_col]].copy()
        clean_df = clean_df.rename(columns={text_col: 'text'})
        clean_df = clean_df.reset_index(drop=True)
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(3, 5), max_features=max_features)
        try:
            tfidf_matrix = vectorizer.fit_transform(clean_df['text'])
        except Exception: continue
        cosine_sim = cosine_similarity(tfidf_matrix)
        adj = {i: [] for i in range(len(clean_df))}
        for i in range(len(clean_df)):
            for j in range(i + 1, len(clean_df)):
                if cosine_sim[i, j] >= threshold:
                    adj[i].append(j); adj[j].append(i)
        visited = set()
        group_id_counter = 0
        for i in range(len(clean_df)):
            if i not in visited:
                group_indices = []
                q = [i]; visited.add(i)
                while q:
                    u = q.pop(0); group_indices.append(u)
                    for v in adj[u]:
                        if v not in visited: visited.add(v); q.append(v)
                if len(group_indices) > 1:
                    group_posts = clean_df.iloc[group_indices].copy()
                    if len(group_posts['account_id'].unique()) > 1:
                        group_sim_scores = cosine_sim[np.ix_(group_indices, group_indices)]
                        max_sim = group_sim_scores.max() if group_sim_scores.size > 0 else 0.0
                        coordination_groups[f"group_{group_id_counter}"] = {
                            "posts": group_posts.to_dict('records'),
                            "num_posts": len(group_posts),
                            "num_accounts": len(group_posts['account_id'].unique()),
                            "max_similarity_score": round(max_sim, 3),
                            "coordination_type": "TBD"
                        }
                        group_id_counter += 1
    final_groups = []
    for group_id, group_data in coordination_groups.items():
        posts_df = pd.DataFrame(group_data['posts'])
        platforms = posts_df['Platform'].unique()
        social_media_platforms_in_group = [p for p in platforms if p in social_media_platforms]
        media_platforms_in_group = [p for p in platforms if p in {'News/Media', 'Media'}]
        if len(media_platforms_in_group) > 1 and len(social_media_platforms_in_group) == 0:
            coordination_type = "Syndication (Media Outlets)"
        elif len(social_media_platforms_in_group) > 1 and len(media_platforms_in_group) == 0:
            coordination_type = "Coordinated Amplification (Social Media)"
        elif len(social_media_platforms_in_group) > 0 and len(media_platforms_in_group) > 0:
            coordination_type = "Media-to-Social Replication"
        else:
            coordination_type = "Other / Uncategorized"
        group_data['coordination_type'] = coordination_type
        final_groups.append(group_data)
    return final_groups

@st.cache_data(show_spinner="Clustering data...")
def perform_clustering(df, coordination_mode):
    if df.empty or 'original_text' not in df.columns:
        return df
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    try:
        tfidf_matrix = vectorizer.fit_transform(df['original_text'])
    except ValueError:
        df['cluster'] = -1
        return df
    dbscan = DBSCAN(eps=0.5, min_samples=CONFIG["bertrend"]["min_cluster_size"], metric='cosine')
    df['cluster'] = dbscan.fit_predict(tfidf_matrix)
    return df

@st.cache_data(show_spinner="Building Network Graph...")
def cached_network_graph(df, coordination_type, data_source_key):
    G = nx.Graph()
    if coordination_type == "text" and 'cluster' in df.columns:
        for cluster_id in df[df['cluster'] != -1]['cluster'].unique():
            cluster_df = df[df['cluster'] == cluster_id]
            accounts = cluster_df['account_id'].unique().tolist()
            if len(accounts) > 1:
                for i in range(len(accounts)):
                    for j in range(i + 1, len(accounts)):
                        G.add_edge(accounts[i], accounts[j], weight=1)
    elif coordination_type == "url" and 'URL' in df.columns:
        for url in df['URL'].unique():
            url_df = df[df['URL'] == url]
            accounts = url_df['account_id'].unique().tolist()
            if len(accounts) > 1:
                for i in range(len(accounts)):
                    for j in range(i + 1, len(accounts)):
                        G.add_edge(accounts[i], accounts[j], weight=1)
    pos = nx.spring_layout(G, seed=42) if G.nodes() else {}
    return G, pos

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Main App ---
def main():
    st.set_page_config(layout="wide", page_title="Election monitoring Dashboard")
    st.title("Election Monitoring Dashboard")
    
    if 'is_preprocessed_data_mode' not in st.session_state:
        st.session_state['is_preprocessed_data_mode'] = False

    # --- Sidebar ---
    st.sidebar.header("⚙️ Configuration")
    coordination_mode = st.sidebar.selectbox("Coordination Type", ["Text Content", "Shared URLs"])
    data_source = st.sidebar.selectbox("Data Source", ["Upload CSV Files"])

    # --- File Upload ---
    combined_raw_df = pd.DataFrame()
    is_preprocessed_data_mode = False
    if data_source == "Upload CSV Files":
        st.sidebar.info("Upload your CSV files below.")
        uploaded_preprocessed = st.sidebar.file_uploader("Upload Preprocessed Summary CSV", type=["csv"], key="preprocessed_upload")
        uploaded_meltwater = st.sidebar.file_uploader("Upload Meltwater CSV", type=["csv"], key="meltwater_upload")
        uploaded_civicsignals = st.sidebar.file_uploader("Upload CivicSignals CSV", type=["csv"], key="civicsignals_upload")
        uploaded_openmeasure = st.sidebar.file_uploader("Upload Open-Measure CSV", type=["csv"], key="openmeasure_upload")

        preprocessed_df = read_uploaded_file(uploaded_preprocessed, "Preprocessed Summary")
        meltwater_df = read_uploaded_file(uploaded_meltwater, "Meltwater")
        civicsignals_df = read_uploaded_file(uploaded_civicsignals, "CivicSignals")
        openmeasure_df = read_uploaded_file(uploaded_openmeasure, "Open-Measure")

        if not preprocessed_df.empty:
            with st.spinner("Processing preprocessed summary..."):
                combined_raw_df = process_preprocessed_data(preprocessed_df)
            is_preprocessed_data_mode = True
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

    st.session_state['is_preprocessed_data_mode'] = is_preprocessed_data_mode

    # --- Preprocessing ---
    df = pd.DataFrame()
    if not combined_raw_df.empty:
        if is_preprocessed_data_mode:
            df = combined_raw_df.copy()
        else:
            with st.spinner("Preprocessing data..."):
                df = final_preprocess_and_map_columns(combined_raw_df, coordination_mode=coordination_mode)

    if df.empty:
        st.warning("No valid data after preprocessing.")
        return

    # --- Global Filters ---
    st.sidebar.header("🔍 Global Filters")
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

    # --- Sampling ---
    max_posts = st.sidebar.number_input("Limit Posts for Analysis (0 for all)", min_value=0, value=0, step=1000)
    df_for_analysis = filtered_df_global.sample(n=max_posts, random_state=42).copy() if max_posts > 0 and len(filtered_df_global) > max_posts else filtered_df_global.copy()

    # --- Analysis ---
    df_clustered = df_for_analysis.copy()
    coordination_groups = []
    if not is_preprocessed_data_mode and not df_for_analysis.empty:
        df_clustered = perform_clustering(df_for_analysis, coordination_mode)
        coordination_groups = find_coordinated_groups(
            df_clustered,
            CONFIG["coordination_detection"]["threshold"],
            CONFIG["coordination_detection"]["max_features"]
        )

    # --- Report Generation ---
    if is_preprocessed_data_mode:
        report_df = df_clustered[['Country', 'original_text', 'URL', 'Emerging Virality']].rename(columns={'original_text': 'Context', 'URL': 'URLs'})
        report_df['Evidence'] = report_df['URLs'].apply(lambda x: ", ".join(eval(x)[:5]) if isinstance(x, str) and x.startswith('[') else x)
        report_df = report_df[['Country', 'Evidence', 'Context', 'URLs', 'Emerging Virality']]
    else:
        report_df = generate_imi_report(df_clustered, data_source)

    # --- Tabs ---
    if is_preprocessed_data_mode:
        tabs = st.tabs(["📊 Narrative Insights"])
        with tabs[0]:
            if report_df.empty:
                st.info("No data to display.")
            else:
                for idx, row in report_df.iterrows():
                    with st.expander(f"**{row['Context'][:100]}...**"):
                        st.markdown("### Narrative Summary")
                        st.markdown(row['Context'])
                        st.markdown("### Virality")
                        st.markdown(row['Emerging Virality'])
                        st.markdown("### Evidence URLs")
                        urls = eval(row['URLs']) if isinstance(row['URLs'], str) and row['URLs'].startswith('[') else [row['URLs']]
                        for url in urls[:5]:
                            st.markdown(f"- [{url}]({url})")
                st.download_button("📥 Download Report", convert_df_to_csv(report_df), "imi_report.csv", "text/csv")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Coordination", "🕸️ Network", "📝 Summary"])

        # Tab 1: Overview
        with tab1:
            st.markdown(f"**Data Source:** `{data_source}` | **Coordination Mode:** `{coordination_mode}` | **Total Rows:** `{len(df):,}`")
            if not filtered_df_global.empty:
                top_influencers = filtered_df_global['account_id'].value_counts().head(10)
                fig_src = px.bar(top_influencers, title="Top 10 Influencers")
                st.plotly_chart(fig_src, width="stretch")

                if 'Platform' in filtered_df_global.columns:
                    platform_counts = filtered_df_global['Platform'].value_counts()
                    fig_platform = px.bar(platform_counts, title="Post Distribution by Platform")
                    st.plotly_chart(fig_platform, width="stretch")

                plot_df = filtered_df_global.copy()
                if 'timestamp_share' in plot_df.columns:
                    plot_df['datetime'] = pd.to_datetime(plot_df['timestamp_share'], unit='s', utc=True)
                    plot_df = plot_df.set_index('datetime')
                    time_series = plot_df.resample('D').size()
                    if not time_series.empty:
                        fig_ts = px.area(time_series, title="Daily Post Volume")
                        st.plotly_chart(fig_ts, width="stretch")

        # Tab 2: Coordination
        with tab2:
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
            if not df_for_analysis.empty:
                G, pos = cached_network_graph(df_for_analysis, "text" if coordination_mode == "Text Content" else "url", "network")
                if G.nodes():
                    edge_x, edge_y = [], []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none'))
                    node_x, node_y, node_text = [], [], []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(f"User: {node}")
                    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', text=node_text, hoverinfo='text'))
                    fig.update_layout(title='Network of Coordinated Accounts', showlegend=False, height=700)
                    st.plotly_chart(fig, width="stretch")

        # Tab 4: Summary
        with tab4:
            if st.button("Generate IMI Report"):
                if report_df.empty:
                    st.error("No report generated.")
                else:
                    st.dataframe(report_df, width="stretch")
                    st.download_button("📥 Download Report", convert_df_to_csv(report_df), "imi_report.csv", "text/csv")

if __name__ == '__main__':
    main()
