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
    }
}

# --- Groq Setup (via Streamlit Secrets) ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    import groq
    client = groq.Groq(api_key=GROQ_API_KEY)
except Exception as e:
    logger.warning(f"Groq API key not found: {e}")
    client = None

# --- Helper: Safe LLM call (no retry, just throttle) ---
def safe_llm_call(prompt, max_tokens=2048):
    """
    Call LLM safely without retry logic.
    Use time.sleep(0.06 + jitter) to stay under 1,000 RPM.
    """
    if client is None:
        logger.error("Groq client not initialized.")
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

# --- IMI Summary Generator (YOUR EXACT VERSION) ---
def summarize_cluster(texts, urls, cluster_data):
    # Truncate to top 50 posts to reduce token load
    joined = "\n".join(texts[:50])
    url_context = "\nRelevant post links:\n" + "\n".join(urls[:5]) if urls else ""

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
  - Do **not** include URLs that do NOT contain the claim.
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

        # Extract URLs from the summary (if any)
        evidence_urls = re.findall(r"(https?://[^\s\)\]]+)", raw_summary)

        # Clean up and ensure URLs are included
        cleaned_summary = re.sub(r'\*\*Here is a concise.*?\*\*', '', raw_summary, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned_summary = re.sub(r'\*\*Here are a few options.*?\*\*', '', cleaned_summary, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned_summary = re.sub(r'"[^"]*"', '', cleaned_summary).strip()

        # Ensure URLs are at the end
        if evidence_urls:
            urls_section = "\n\nSources: " + ", ".join(evidence_urls[:5])
            cleaned_summary += urls_section

        return cleaned_summary, evidence_urls
    else:
        logger.error(f"LLM call failed for cluster {cluster_data['cluster'].iloc[0]}")
        return "Summary generation failed.", []

# --- Helper Functions (Your Exact Versions) ---
def infer_platform_from_url(url):
    """Infers the social media or news platform from a given URL."""
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
    elif url.startswith("https://") or url.startswith("http://"):
        media_domains = ["nytimes.com", "bbc.com", "cnn.com", "reuters.com", "theguardian.com", "aljazeera.com", "lemonde.fr", "dw.com"]
        if any(domain in url for domain in media_domains):
            return "News/Media"
        return "Media"
    else:
        return "Unknown"

def extract_original_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    cleaned = re.sub(r'^(RT|rt|QT|qt)\s+@\w+:\s*', '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'@\w+', '', cleaned).strip()
    cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned).strip()
    cleaned = re.sub(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b', '', cleaned, flags=re.IGNORECASE)
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
        else:
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

def combine_social_media_data(
    meltwater_df,
    civicsignals_df,
    openmeasure_df=None,
    meltwater_object_col='hit sentence',
    civicsignals_object_col='title',
    openmeasure_object_col='text'
):
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
    if df_processed.empty:
        df_processed['URL'] = pd.Series([], dtype='object')
        df_processed['Platform'] = pd.Series([], dtype='object')
        df_processed['original_text'] = pd.Series([], dtype='object')
        df_processed['Outlet'] = pd.Series([], dtype='object')
        df_processed['Channel'] = pd.Series([], dtype='object')
        df_processed['object_id'] = pd.Series([], dtype='object')
        df_processed['timestamp_share'] = pd.Series([], dtype='Int64')
        return df_processed
    df_processed.rename(columns={'original_url': 'URL'}, inplace=True)
    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan', '').fillna('')
    df_processed = df_processed[df_processed['object_id'].str.strip() != ""].copy()
    def clean_text_for_display(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r"\\n|\\r|\\t", " ", text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    if coordination_mode == "Text Content":
        df_processed['object_id'] = df_processed['object_id'].apply(clean_text_for_display)
        df_processed = df_processed[df_processed['object_id'].str.len() > 0].reset_index(drop=True)
        df_processed = df_processed[
            ~df_processed['object_id'].str.lower().str.startswith('rt @') &
            ~df_processed['object_id'].str.lower().str.startswith('qt @')
        ].reset_index(drop=True)
    if coordination_mode == "Text Content":
        df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    elif coordination_mode == "Shared URLs":
        df_processed['original_text'] = df_processed['URL'].astype(str).replace('nan', '').fillna('')
    df_processed = df_processed[df_processed['original_text'].str.strip() != ""].reset_index(drop=True)
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)
    if 'Outlet' not in df_processed.columns:
        df_processed['Outlet'] = np.nan
    if 'Channel' not in df_processed.columns:
        df_processed['Channel'] = np.nan
    return df_processed

# --- Coordination Detection ---
def find_coordinated_groups(df, threshold, max_features):
    text_col = 'original_text'
    social_media_platforms = {'TikTok', 'Facebook', 'X', 'YouTube', 'Instagram', 'Telegram'}
    coordination_groups = {}
    clustered_groups = df[df['cluster'] != -1].groupby('cluster')
    for cluster_id, group in clustered_groups:
        if len(group) < 2:
            continue
        clean_df = group[['account_id', 'timestamp_share', 'Platform', 'URL', text_col]].copy()
        clean_df = clean_df.rename(columns={text_col: 'text', 'timestamp_share': 'Timestamp'})
        clean_df = clean_df.reset_index(drop=True)
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(3, 5), max_features=max_features)
        try:
            tfidf_matrix = vectorizer.fit_transform(clean_df['text'])
        except Exception:
            continue
        cosine_sim = cosine_similarity(tfidf_matrix)
        adj = {i: [] for i in range(len(clean_df))}
        for i in range(len(clean_df)):
            for j in range(i + 1, len(clean_df)):
                if cosine_sim[i, j] >= threshold:
                    adj[i].append(j)
                    adj[j].append(i)
        visited = set()
        group_id_counter = 0
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

@st.cache_data(show_spinner=False)
def cached_network_graph(df, coordination_type, data_source):
    G = nx.Graph()
    cluster_map = {}
    if coordination_type == "text" and 'cluster' in df.columns:
        for cluster_id in df[df['cluster'] != -1]['cluster'].unique():
            cluster_df = df[df['cluster'] == cluster_id]
            accounts = cluster_df['account_id'].unique().tolist()
            if len(accounts) > 1:
                for i in range(len(accounts)):
                    for j in range(i + 1, len(accounts)):
                        G.add_edge(accounts[i], accounts[j], weight=1)
                for account in accounts:
                    cluster_map[account] = cluster_id
    elif coordination_type == "url" and 'URL' in df.columns:
        for url in df['URL'].unique():
            url_df = df[df['URL'] == url]
            accounts = url_df['account_id'].unique().tolist()
            if len(accounts) > 1:
                for i in range(len(accounts)):
                    for j in range(i + 1, len(accounts)):
                        G.add_edge(accounts[i], accounts[j], weight=1)
                for account in accounts:
                    cluster_map[account] = hash(url)
    pos = nx.spring_layout(G, seed=42) if G.nodes() else {}
    return G, pos, cluster_map

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Main App ---
def main_election_monitoring():
    st.set_page_config(layout="wide", page_title="Election Monitoring Dashboard")
    st.sidebar.title("Configuration")

    data_source = st.sidebar.selectbox("Select Data Source", ["Upload CSV Files", "Upload Preprocessed Report"])

    if data_source == "Upload Preprocessed Report":
        # === MODE 2: ONLY ONE TAB ===
        st.header("📊 Narrative Insights from Preprocessed IMI Report")
        uploaded_report = st.file_uploader("Upload your preprocessed IMI report (CSV)", type=["csv"], key="imi_report_upload")
        if uploaded_report:
            try:
                report_df = pd.read_csv(uploaded_report)
                report_df.columns = report_df.columns.str.strip()
                required_cols = {"Context", "Evidence", "URLs", "Emerging Virality"}
                if not required_cols.issubset(set(report_df.columns)):
                    st.error(f"❌ Missing required columns. Expected: {required_cols}")
                else:
                    report_df = report_df.dropna(subset=["Context"]).reset_index(drop=True)
                    
                    # --- URL Extraction (from Evidence + URLs) ---
                    def extract_urls(text):
                        return re.findall(r'https?://[^\s\)\]\,\"\'\<\>]+', str(text)) if pd.notna(text) else []
                    def parse_urls_column(val):
                        if pd.isna(val) or not isinstance(val, str):
                            return []
                        try:
                            parsed = eval(val)
                            if isinstance(parsed, list):
                                return [u.strip() for u in parsed if isinstance(u, str) and u.startswith("http")]
                        except:
                            pass
                        return extract_urls(val)
                    
                    report_df["Evidence_URLs"] = report_df["Evidence"].apply(extract_urls)
                    report_df["Full_URLs"] = report_df["URLs"].apply(parse_urls_column)
                    report_df["All_URLs"] = report_df.apply(
                        lambda row: list(dict.fromkeys(row["Evidence_URLs"] + row["Full_URLs"]))[:10], axis=1
                    )

                    # --- Virality Badge ---
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
                    
                    report_df["Virality_HTML"] = report_df["Emerging Virality"].apply(virality_badge)

                    # --- Display ---
                    for idx, row in report_df.iterrows():
                        context = row["Context"].strip()
                        if not context: continue
                        title_preview = context.split('\n')[0][:100] + ("..." if len(context) > 100 else "")
                        with st.expander(f"**{title_preview}**"):
                            st.markdown("### Narrative Summary")
                            st.markdown(context)
                            st.markdown("### Virality Level")
                            st.markdown(row["Virality_HTML"], unsafe_allow_html=True)
                            urls = row["All_URLs"]
                            if urls:
                                st.markdown("### Supporting Evidence (Sample URLs)")
                                for url in urls:
                                    st.markdown(f"- <a href='{url}' target='_blank' style='text-decoration: underline; color: #1f77b4;'>{url}</a>", unsafe_allow_html=True)
                            else:
                                st.markdown("### Supporting Evidence\n- No URLs available.")
                            st.markdown("---")

                    # Download
                    output_df = report_df[["Country", "Context", "Evidence", "URLs", "Emerging Virality"]].copy()
                    st.download_button(
                        "📥 Download Cleaned Report",
                        convert_df_to_csv(output_df),
                        "imi_narrative_insights.csv",
                        "text/csv"
                    )
            except Exception as e:
                st.error(f"❌ Failed to process report: {e}")
        else:
            st.info("📤 Please upload a preprocessed IMI report in CSV format (with columns: Context, Evidence, URLs, Emerging Virality).")

    else:
        # === MODE 1: FULL PIPELINE (Tabs 1-4) ===
        coordination_mode = st.sidebar.selectbox("Coordination Mode", ["Text Content", "Shared URLs"])
        st.sidebar.info("Upload your CSV files below.")
        uploaded_meltwater = st.sidebar.file_uploader("Upload Meltwater CSV", type=["csv"], key="meltwater_upload")
        uploaded_civicsignals = st.sidebar.file_uploader("Upload CivicSignals CSV", type=["csv"], key="civicsignals_upload")
        uploaded_openmeasure = st.sidebar.file_uploader("Upload Open-Measure CSV", type=["csv"], key="openmeasure_upload")

        def read_uploaded_file(uploaded_file, file_name):
            if not uploaded_file:
                return pd.DataFrame()
            bytes_data = uploaded_file.getvalue()
            encodings = ['utf-8-sig', 'utf-16le', 'utf-16be', 'utf-16', 'latin1', 'cp1252']
            decoded_content = None
            for enc in encodings:
                try:
                    decoded_content = bytes_data.decode(enc)
                    break
                except (UnicodeDecodeError, AttributeError):
                    continue
            if decoded_content is None:
                st.error(f"❌ Failed to decode {file_name}")
                return pd.DataFrame()
            sample_line = decoded_content.strip().splitlines()[0]
            sep = '\t' if '\t' in sample_line else ','
            try:
                df = pd.read_csv(StringIO(decoded_content), sep=sep, low_memory=False)
                return df
            except Exception as e:
                st.error(f"❌ Parse error in {file_name}: {e}")
                return pd.DataFrame()

        meltwater_df_upload = read_uploaded_file(uploaded_meltwater, "Meltwater")
        civicsignals_df_upload = read_uploaded_file(uploaded_civicsignals, "CivicSignals")
        openmeasure_df_upload = read_uploaded_file(uploaded_openmeasure, "Open-Measure")

        with st.spinner("📥 Combining uploaded datasets..."):
            obj_map = {
                "meltwater": "hit sentence" if coordination_mode == "Text Content" else "url",
                "civicsignals": "title" if coordination_mode == "Text Content" else "url",
                "openmeasure": "text" if coordination_mode == "Text Content" else "url"
            }
            combined_raw_df = combine_social_media_data(
                meltwater_df_upload,
                civicsignals_df_upload,
                openmeasure_df_upload,
                meltwater_object_col=obj_map["meltwater"],
                civicsignals_object_col=obj_map["civicsignals"],
                openmeasure_object_col=obj_map["openmeasure"]
            )

        if combined_raw_df.empty:
            st.warning("No data loaded from uploaded files.")
            df = pd.DataFrame()
        else:
            with st.spinner("⏳ Preprocessing and mapping combined data..."):
                df = final_preprocess_and_map_columns(combined_raw_df, coordination_mode=coordination_mode)

        # --- Global Filters ---
        if df.empty or 'timestamp_share' not in df.columns:
            filtered_df_global = pd.DataFrame()
        else:
            min_ts = df['timestamp_share'].min()
            max_ts = df['timestamp_share'].max()
            if pd.isna(min_ts) or pd.isna(max_ts):
                min_date = max_date = pd.Timestamp.now().date()
            else:
                min_date = pd.to_datetime(min_ts, unit='s').date()
                max_date = pd.to_datetime(max_ts, unit='s').date()

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

        # --- Sampling ---
        max_posts_for_analysis = st.sidebar.number_input("Limit Posts for Analysis (0 for all)", min_value=0, value=0, step=1000)
        if max_posts_for_analysis > 0 and len(filtered_df_global) > max_posts_for_analysis:
            df_for_analysis = filtered_df_global.sample(n=max_posts_for_analysis, random_state=42).copy()
            st.sidebar.warning(f"⚠️ Analyzing a random sample of **{len(df_for_analysis):,}** posts.")
        else:
            df_for_analysis = filtered_df_global.copy()
            st.sidebar.info(f"✅ Analyzing all **{len(df_for_analysis):,}** posts.")

        # --- Download Buttons ---
        if not df.empty:
            core_cols = ['account_id', 'content_id', 'object_id', 'timestamp_share']
            if all(col in df.columns for col in core_cols):
                csv_data = convert_df_to_csv(df[core_cols])
                st.sidebar.download_button("Download Preprocessed Data", csv_data, "preprocessed.csv", "text/csv")

        if not filtered_df_global.empty:
            csv_filtered = convert_df_to_csv(filtered_df_global)
            st.sidebar.download_button("Download Filtered Data", csv_filtered, "filtered.csv", "text/csv")

        # --- Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "🔍 Similarity & Coordination",
            "🕸️ Network Graph",
            "📝 Summary"
        ])

        # === TAB 1: Overview ===
        with tab1:
            st.header("📊 Overview")
            if not df.empty:
                st.markdown(f"**Data Source:** `{data_source}` | **Coordination Mode:** `{coordination_mode}` | **Total Rows:** `{len(df):,}`")
                display_cols = ['account_id', 'content_id', 'object_id', 'timestamp_share']
                existing_cols = [col for col in display_cols if col in df.columns]
                if existing_cols:
                    st.dataframe(df[existing_cols].head(10))

            if not filtered_df_global.empty:
                # Top Influencers
                top_influencers = filtered_df_global['account_id'].value_counts().head(10)
                fig_src = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Posts', 'index': 'Account'})
                st.plotly_chart(fig_src, use_container_width=True)

                # Platform Distribution
                if 'Platform' in filtered_df_global.columns:
                    platform_counts = filtered_df_global['Platform'].value_counts()
                    fig_platform = px.bar(platform_counts, title="Post Distribution by Platform", labels={'value': 'Posts', 'index': 'Platform'})
                    st.plotly_chart(fig_platform, use_container_width=True)

                # Top Hashtags (Social Media Only)
                social_media_df = filtered_df_global[~filtered_df_global['Platform'].isin(['Media', 'News/Media'])].copy()
                if not social_media_df.empty and 'object_id' in social_media_df.columns:
                    social_media_df['hashtags'] = social_media_df['object_id'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
                    all_hashtags = [tag for tags_list in social_media_df['hashtags'] if isinstance(tags_list, list) for tag in tags_list]
                    if all_hashtags:
                        hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
                        fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags (Social Media Only)", labels={'value': 'Frequency', 'index': 'Hashtag'})
                        st.plotly_chart(fig_ht, use_container_width=True)

                # Daily Post Volume
                plot_df = filtered_df_global.copy()
                if 'timestamp_share' in plot_df.columns:
                    plot_df['datetime'] = pd.to_datetime(plot_df['timestamp_share'], unit='s', utc=True)
                    plot_df = plot_df.set_index('datetime')
                    time_series = plot_df.resample('D').size()
                    if not time_series.empty:
                        fig_ts = px.area(time_series, title="Daily Post Volume")
                        st.plotly_chart(fig_ts, use_container_width=True)

        # === TAB 2: Coordination ===
        with tab2:
            if coordination_mode == "Text Content":
                eps = st.sidebar.slider("DBSCAN eps", 0.1, 1.0, 0.3, 0.05)
                min_samples = st.sidebar.slider("Min Samples", 2, 10, 2, 1)
                max_features = st.sidebar.slider("TF-IDF Max Features", 1000, 10000, 5000, 1000)
                threshold_sim = st.slider("Similarity Threshold", 0.75, 0.99, 0.90, 0.01)

                if not df_for_analysis.empty:
                    clustered_df = cached_clustering(df_for_analysis, eps, min_samples, max_features, f"{coordination_mode}_{eps}_{min_samples}")
                    coordinated_groups = find_coordinated_groups(clustered_df, threshold_sim, max_features)
                    if coordinated_groups:
                        st.info(f"Found {len(coordinated_groups)} coordinated groups.")
                        for i, group in enumerate(coordinated_groups):
                            st.markdown(f"#### Group {i+1}: {group['coordination_type']}")
                            st.write(f"**Posts:** {group['num_posts']} | **Accounts:** {group['num_accounts']} | **Max Sim:** {group['max_similarity_score']}")
                            posts_df = pd.DataFrame(group['posts'])
                            posts_df['Timestamp'] = pd.to_datetime(posts_df['Timestamp'], unit='s')
                            st.dataframe(posts_df[['account_id', 'Platform', 'Timestamp', 'URL']], use_container_width=True)
                            st.markdown("---")
                    else:
                        st.warning("No coordinated groups found.")
                else:
                    st.info("No data for analysis.")
            else:
                st.info("Shared URLs analysis coming soon.")

        # === TAB 3: Network Graph ===
        with tab3:
            st.subheader("🕸️ Network Graph of Coordinated Activity")
            if 'max_nodes_to_display' not in st.session_state:
                st.session_state.max_nodes_to_display = 40
            st.session_state.max_nodes_to_display = st.slider(
                "Maximum Nodes to Display in Graph",
                min_value=10, max_value=200, value=st.session_state.max_nodes_to_display, step=10
            )
            st.markdown("---")

            if not df_for_analysis.empty and coordination_mode == "Text Content":
                eps = 0.3
                min_samples = 2
                max_features = 5000
                with st.spinner("🗂️ Pre-processing data for network graph..."):
                    clustered_df_for_graph = cached_clustering(df_for_analysis, eps=eps, min_samples=min_samples, max_features=max_features, data_source="graph")
                G, pos, cluster_map = cached_network_graph(clustered_df_for_graph, coordination_type="text", data_source_key="graph")
                if not G.nodes():
                    st.warning("No coordinated activity detected.")
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
                        title='Network of Coordinated Accounts',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=700
                    )
                    st.plotly_chart(fig_net, use_container_width=True)

                    # Risk Assessment
                    degree_centrality = nx.degree_centrality(G)
                    risk_df = pd.DataFrame(degree_centrality.items(), columns=['Account', 'Degree Centrality'])
                    risk_df = risk_df.sort_values(by='Degree Centrality', ascending=False).reset_index(drop=True)
                    risk_df['Risk Score'] = (risk_df['Degree Centrality'] / risk_df['Degree Centrality'].max()) * 100
                    if 'Platform' in filtered_df_global.columns:
                        risk_df = risk_df.merge(filtered_df_global[['account_id', 'Platform']].drop_duplicates(), left_on='Account', right_on='account_id', how='left').drop(columns='account_id')
                    risk_df = risk_df.head(20)
                    if not risk_df.empty:
                        st.markdown("#### Top 20 Most Central Accounts")
                        st.dataframe(risk_df, use_container_width=True)
                        risk_csv = convert_df_to_csv(risk_df)
                        st.download_button("Download Risk Assessment CSV", risk_csv, "risk_assessment.csv", "text/csv")
            else:
                st.info("Network graph only available for 'Text Content' mode.")

        # === TAB 4: Summary (WITH YOUR FULL LOGIC) ===
        with tab4:
            st.subheader("📝 Generate IMI Narrative Report")
            if st.button("Generate IMI Report"):
                if df_for_analysis.empty:
                    st.error("❌ No data available for report generation.")
                else:
                    # Ensure 'cluster' column exists
                    eps = 0.3
                    min_samples = 2
                    max_features = 5000
                    with st.spinner("🗂️ Clustering posts..."):
                        clustered_df = cached_clustering(
                            df_for_analysis, 
                            eps=eps, 
                            min_samples=min_samples, 
                            max_features=max_features, 
                            data_source_key="report"
                        )
                    if 'cluster' not in clustered_df.columns:
                        st.error("❌ Clustering failed.")
                        st.stop()

                    # Fix timestamp
                    clustered_df['Timestamp'] = pd.to_datetime(clustered_df['timestamp_share'], unit='s', utc=True)

                    # Step 1: Pre-filter clusters by size (≥ min_cluster_size)
                    cluster_sizes = clustered_df.groupby('cluster').size()
                    min_cluster_size = CONFIG['bertrend']['min_cluster_size']
                    valid_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index
                    logger.info(f"Reduced from {len(clustered_df['cluster'].unique())} to {len(valid_clusters)} valid clusters (min_size={min_cluster_size})")

                    # Step 2: Limit max clusters to 50
                    max_clusters = 50
                    if len(valid_clusters) > max_clusters:
                        valid_clusters = valid_clusters[:max_clusters]
                        logger.info(f"Clamped cluster count to {max_clusters} (was {len(valid_clusters)})")
                    else:
                        logger.info(f"Cluster count already under {max_clusters}: {len(valid_clusters)}")

                    all_summaries = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Step 3: Loop only over valid clusters
                    for i, cluster_id in enumerate(valid_clusters):
                        status_text.text(f"Processing Cluster {i+1}/{len(valid_clusters)}")
                        progress_bar.progress((i + 1) / len(valid_clusters))
                        cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
                        texts = cluster_data['original_text'].tolist()
                        urls = cluster_data['URL'].dropna().unique().tolist()

                        try:
                            raw_summary, evidence_urls = summarize_cluster(texts, urls, cluster_data)

                            cleaned_summary = re.sub(r'\*\*Here is a concise.*?\*\*', '', raw_summary, flags=re.IGNORECASE | re.DOTALL).strip()
                            cleaned_summary = re.sub(r'\*\*Here are a few options.*?\*\*', '', cleaned_summary, flags=re.IGNORECASE | re.DOTALL).strip()
                            cleaned_summary = re.sub(r'"[^"]*"', '', cleaned_summary).strip()

                            # Ensure URL section is added
                            if evidence_urls:
                                urls_section = "\n\nSources: " + ", ".join(evidence_urls[:5])
                                cleaned_summary += urls_section

                        except Exception as e:
                            logger.error(f"Failed to summarize cluster {cluster_id}: {e}")
                            cleaned_summary = f"**Narrative Summary Failed**\n\nError: {str(e)}"
                            evidence_urls = []

                        all_urls = list(dict.fromkeys(evidence_urls + [u for u in urls if u]))

                        all_summaries.append({
                            "cluster_id": cluster_id,
                            "main_text": cleaned_summary,
                            "narrative_summary": cleaned_summary,
                            "urls": all_urls,
                            "first_detected": cluster_data['Timestamp'].min(),
                            "last_detected": cluster_data['Timestamp'].max(),
                            "post_count": len(cluster_data)
                        })

                        # Safe rate: ~16 requests/sec → under 1,000 RPM
                        time.sleep(0.06 + random.uniform(0, 0.02))

                    progress_bar.empty()
                    status_text.empty()
                    st.success("✅ IMI Report generation complete!")

                    # Reduce to ~50 Clusters (Key Fix)
                    logger.info("Reducing cluster count to ~50...")
                    all_summaries.sort(key=lambda x: x['post_count'], reverse=True)
                    n_groups = min(50, len(all_summaries))
                    final_summaries = all_summaries[:n_groups]
                    logger.info(f"Final cluster count: {len(final_summaries)}")

                    # Display the summaries in the UI
                    for summary in final_summaries:
                        st.markdown(f"**Cluster ID:** `{summary['cluster_id']}` | **Post Count:** `{summary['post_count']}`")
                        st.markdown(summary['narrative_summary'])
                        st.markdown("---")

                    # Create downloadable report
                    if final_summaries:
                        report_text = ""
                        for summary in final_summaries:
                            report_text += f"**Narrative Summary:**\n{summary['narrative_summary']}\n"
                            report_text += f"First Detected: {summary['first_detected'].strftime('%Y-%m-%d %H:%M')}\n"
                            report_text += f"Last Updated: {summary['last_detected'].strftime('%Y-%m-%d %H:%M')}\n"
                            report_text += f"Post Count: {summary['post_count']}\n"
                            report_text += "-" * 50 + "\n\n"

                        st.download_button(
                            label="📥 Download Full IMI Report",
                            data=report_text,
                            file_name="IMI_Narrative_Report_Uganda_2026.txt",
                            mime="text/plain"
                        )
                    else:
                        st.info("No valid summaries were generated for the report.")

if __name__ == "__main__":
    main_election_monitoring()
