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


# --- Helper: Safe LLM call (Unchanged) ---
def safe_llm_call(prompt, max_tokens=2048):
    """Call LLM safely."""
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=CONFIG["model_id"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens
        )
        return response
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None

# --- IMI Summary Generator (Full Prompt Maintained) ---
def summarize_cluster(texts, urls, cluster_data):
    # Truncate to top 50 posts to reduce token load
    joined = "\n".join(texts[:50])
    url_context = "\nRelevant post links:\n" + "\n".join(urls[:5]) if urls else ""

    # Handle timestamps
    if 'Timestamp' not in cluster_data.columns or cluster_data['Timestamp'].dtype != 'datetime64[ns]':
        cluster_data['Timestamp'] = pd.to_datetime(cluster_data['timestamp_share'], unit='s', errors='coerce')
        
    cluster_data = cluster_data.dropna(subset=['Timestamp']) 
        
    if cluster_data.empty:
        return "Summary generation failed: No valid timestamps.", [], "Failed to Summarize"

    min_timestamp_str = cluster_data['Timestamp'].min().strftime('%Y-%m-%d %H:%M')
    max_timestamp_str = cluster_data['Timestamp'].max().strftime('%Y-%m-%d %H:%M')

    # --- FULL, UN-CUT PROMPT ---
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
    # --- END: FULL, UN-CUT PROMPT ---

    response = safe_llm_call(prompt, max_tokens=2048)

    if response:
        raw_summary = response.choices[0].message.content.strip()
        
        # Title Extraction and cleanup logic
        title_match = re.search(r'\*\*(.*?)\*\*', raw_summary, re.DOTALL)
        title = title_match.group(1).strip() if title_match else "Un-Titled Narrative"
        evidence_urls = re.findall(r"(https?://[^\s\)\]]+)", raw_summary)
        cleaned_summary = re.sub(r'\*\*(.*?)\*\*', '', raw_summary, count=1, flags=re.DOTALL).strip()
        cleaned_summary = re.sub(r'First Detected: .*?Last Updated: .*', '', cleaned_summary, flags=re.DOTALL).strip()
        
        return cleaned_summary, evidence_urls, title
    else:
        logger.error(f"LLM call failed for cluster {cluster_data['cluster'].iloc[0] if 'cluster' in cluster_data.columns else 'N/A'}")
        return "Summary generation failed.", [], "Failed to Summarize"

# --- Generate IMI Report (Unchanged) ---
@st.cache_data(show_spinner="Generating Narrative Summaries...")
def generate_imi_report(clustered_df, data_source_key):
    # Ensure Timestamp exists and is datetime
    if 'Timestamp' not in clustered_df.columns or clustered_df['Timestamp'].dtype != 'datetime64[ns]':
        clustered_df['Timestamp'] = pd.to_datetime(clustered_df['timestamp_share'], unit='s', errors='coerce')

    if clustered_df.empty or 'cluster' not in clustered_df.columns:
        return pd.DataFrame()

    report_data = []
    clustered_df = clustered_df.dropna(subset=['Timestamp'])

    unique_clusters = clustered_df[clustered_df['cluster'] != -1]['cluster'].unique()

    for i, cluster_id in enumerate(unique_clusters[:5]):
        cluster_data = clustered_df[clustered_df['cluster'] == cluster_id].copy()
        
        texts = cluster_data['original_text'].tolist()
        urls = cluster_data['URL'].tolist()
        
        summary, evidence_urls, title = summarize_cluster(texts, urls, cluster_data)

        min_ts = cluster_data['Timestamp'].min()
        max_ts = cluster_data['Timestamp'].max()
        posts_count = len(cluster_data)
        accounts_count = cluster_data['account_id'].nunique()
        platforms_list = cluster_data['Platform'].unique().tolist()
        datasets_list = cluster_data['source_dataset'].unique().tolist()
        
        time_span_hours = (max_ts - min_ts).total_seconds() / 3600
        posts_per_hour = posts_count / time_span_hours if time_span_hours > 0 else posts_count
        
        virality = "Tier 1: Low"
        if posts_per_hour > 50:
            virality = "Tier 4: Emergency"
        elif posts_per_hour > 20:
            virality = "Tier 3: High"
        elif posts_per_hour > 5:
            virality = "Tier 2: Medium"

        report_data.append({
            'ID': cluster_id,
            'Title': title,
            'Posts': posts_count,
            'Accounts': accounts_count,
            'Platforms': ", ".join(platforms_list),
            'First Detected': min_ts,
            'Last Updated': max_ts,
            'Context': summary,
            'URLs': evidence_urls,
            'Source Datasets': ", ".join(datasets_list),
            'Emerging Virality': virality
        })

    return pd.DataFrame(report_data)


# --- CORE HELPER FUNCTIONS ---

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
    """
    Cleans text by removing RT/QT prefixes, @mentions, URLs, and normalizing spaces.
    Removes dates and years to prevent them from dominating the narrative summary.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    cleaned = re.sub(r'^(RT|rt|QT|qt)\s+@\w+:\s*', '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'@\w+', '', cleaned).strip()
    cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned).strip()
    
    # --- NEW: Remove dates, years, and common month names ---
    # Matches patterns like "June 17", "17 June", "17/06/2025", "2025"
    cleaned = re.sub(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', cleaned)
    cleaned = re.sub(r'\b\d{4}\b', '', cleaned)
    
    cleaned = re.sub(r"\\n|\\r|\\t", " ", cleaned).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.lower()

# --- Robust Timestamp Parser: Returns UNIX Timestamp (Integer) ---
def parse_timestamp_robust(timestamp):
    """
    Converts a timestamp string to a UNIX timestamp (integer seconds since epoch).
    Returns None if parsing fails.
    """
    if pd.isna(timestamp):
        return None
    if isinstance(timestamp, (int, float)):
        if 0 < timestamp < 253402300800:  # Valid range: 1970–9999
            return int(timestamp)
        else:
            return None

    # List of common timestamp formats
    date_formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%b %d, %Y @ %H:%M:%S.%f', '%d-%b-%Y %I:%M%p',
        '%A, %d %b %Y %H:%M:%S', '%b %d, %I:%M%p', '%d %b %Y %I:%M%p',
        '%Y-%m-%d', '%m/%d/%Y', '%d %b %Y',
    ]

    # Try direct parsing
    try:
        parsed = pd.to_datetime(timestamp, errors='coerce', utc=True)
        if pd.notna(parsed):
            return int(parsed.timestamp())
    except:
        pass

    # Try each format
    for fmt in date_formats:
        try:
            parsed = pd.to_datetime(timestamp, format=fmt, errors='coerce', utc=True)
            if pd.notna(parsed):
                return int(parsed.timestamp())
        except (ValueError, TypeError):
            continue
    return None

# --- NEW: Helper for Preprocessed Data ---
def process_preprocessed_data(preprocessed_df):
    """
    Maps columns from the summary (preprocessed) data to the expected final 
    dashboard columns, filling required columns with placeholders to prevent 
    app crashes, and adds the preprocessed flag.
    """
    if preprocessed_df.empty:
        return pd.DataFrame()

    # Standardize column names
    preprocessed_df.columns = preprocessed_df.columns.str.lower().str.replace(' ', '_', regex=False)
    
    df_out = pd.DataFrame()
    
    # 1. Map core content columns
    df_out['original_text'] = preprocessed_df.get('context', 'No Context Provided').astype(str)
    df_out['URL'] = preprocessed_df.get('urls', '').astype(str)
    
    # 2. Map summary-specific columns for display
    df_out['Emerging Virality'] = preprocessed_df.get('emerging_virality', 'Unknown').astype(str)
    df_out['Country'] = preprocessed_df.get('country', 'Unknown Country').astype(str)
    
    # 3. Fill mandatory core columns with placeholder values
    df_out['account_id'] = 'Summary_Author'
    df_out['object_id'] = df_out['original_text'] 
    df_out['content_id'] = 'SUMMARY_' + preprocessed_df.index.to_series().astype(str)
    
    # Use a recent dummy timestamp for date filtering to work
    current_timestamp = int(pd.Timestamp.now(tz='UTC').timestamp())
    df_out['timestamp_share'] = current_timestamp 
    df_out['timestamp_share'] = df_out['timestamp_share'].astype('Int64')
    
    # 4. Set source and platform
    df_out['source_dataset'] = 'Preprocessed_Summary'
    df_out['Platform'] = 'Report_Summary' 
    df_out['Outlet'] = 'Report'
    df_out['Channel'] = 'Summary'
    
    # 5. Add the flag to control downstream application behavior
    df_out['is_preprocessed_summary'] = True
    df_out['cluster'] = df_out.index # Give each summary its own "cluster"
    
    # Filter out empty entries
    df_out = df_out[df_out['original_text'].str.strip() != ""].reset_index(drop=True)

    return df_out

# --- File Reading Helper ---
def read_uploaded_file(uploaded_file, file_name):
    if not uploaded_file:
        return pd.DataFrame()
        
    bytes_data = uploaded_file.getvalue()
    encodings = ['utf-8-sig', 'utf-16le', 'utf-16be', 'utf-16', 'latin1', 'cp1252']
    decoded_content = None
    detected_enc = None

    for enc in encodings:
        try:
            decoded_content = bytes_data.decode(enc)
            detected_enc = enc
            logger.info(f"✅ {file_name}: Decoded using '{enc}'")
            break
        except (UnicodeDecodeError, AttributeError):
            continue
        
    if decoded_content is None:
        st.error(f"❌ Failed to read {file_name} CSV: Could not decode with any supported encoding.")
        return pd.DataFrame()

    # Attempt to determine the separator
    sample_line = decoded_content.strip().splitlines()[0]
    sep = '\t' if '\t' in sample_line else ','
    
    try:
        df = pd.read_csv(StringIO(decoded_content), sep=sep, low_memory=False)
        logger.info(f"✅ {file_name}: Loaded {len(df)} rows (sep='{sep}', enc='{detected_enc}')")
        return df
    except Exception as e:
        st.error(f"❌ Failed to parse {file_name} CSV after decoding: {e}")
        return pd.DataFrame()

# --- Combine Multiple Datasets with Flexible Object Column ---
def combine_social_media_data(
    meltwater_df,
    civicsignals_df,
    openmeasure_df=None,
    meltwater_object_col='hit sentence',
    civicsignals_object_col='title',
    openmeasure_object_col='text'
):
    """
    Combines datasets from Meltwater, CivicSignals, and Open-Measure (optional).
    Allows specification of which column to use as 'object_id' for coordination analysis.
    Returns timestamp as UNIX integer.
    """
    combined_dfs = []

    def get_specific_col(df, col_name_lower):
        if col_name_lower in df.columns:
            return df[col_name_lower]
        return pd.Series([np.nan] * len(df), index=df.index)

    # Process Meltwater
    if meltwater_df is not None and not meltwater_df.empty:
        meltwater_df.columns = meltwater_df.columns.str.lower()
        mw = pd.DataFrame()
        mw['account_id'] = get_specific_col(meltwater_df, 'influencer')
        mw['content_id'] = get_specific_col(meltwater_df, 'tweet id')
        mw['object_id'] = get_specific_col(meltwater_df, meltwater_object_col.lower())
        mw['original_url'] = get_specific_col(meltwater_df, 'url')
        mw['timestamp_share'] = get_specific_col(meltwater_df, 'date')
        mw['source_dataset'] = 'Meltwater'
        combined_dfs.append(mw)

    # Process CivicSignals
    if civicsignals_df is not None and not civicsignals_df.empty:
        civicsignals_df.columns = civicsignals_df.columns.str.lower()
        cs = pd.DataFrame()
        cs['account_id'] = get_specific_col(civicsignals_df, 'media_name')
        cs['content_id'] = get_specific_col(civicsignals_df, 'stories_id')
        cs['object_id'] = get_specific_col(civicsignals_df, civicsignals_object_col.lower())
        cs['original_url'] = get_specific_col(civicsignals_df, 'url')
        cs['timestamp_share'] = get_specific_col(civicsignals_df, 'publish_date')
        cs['source_dataset'] = 'CivicSignals'
        combined_dfs.append(cs)

    # Process Open-Measure
    if openmeasure_df is not None and not openmeasure_df.empty:
        openmeasure_df.columns = openmeasure_df.columns.str.lower()
        om = pd.DataFrame()
        om['account_id'] = get_specific_col(openmeasure_df, 'actor_username')
        om['content_id'] = get_specific_col(openmeasure_df, 'id')
        om['object_id'] = get_specific_col(openmeasure_df, openmeasure_object_col.lower())
        om['original_url'] = get_specific_col(openmeasure_df, 'url')
        om['timestamp_share'] = get_specific_col(openmeasure_df, 'created_at')
        om['source_dataset'] = 'OpenMeasure'
        combined_dfs.append(om)

    if not combined_dfs:
        return pd.DataFrame()

    combined = pd.concat(combined_dfs, ignore_index=True)
    # Ensure required columns exist before dropping NaNs
    for col in ['account_id', 'content_id', 'object_id', 'original_url', 'timestamp_share']:
        if col not in combined.columns:
            combined[col] = np.nan
            
    combined = combined.dropna(subset=['account_id', 'content_id', 'timestamp_share', 'object_id']).copy()
    combined['account_id'] = combined['account_id'].astype(str).replace('nan', 'Unknown_User').fillna('Unknown_User')
    combined['content_id'] = combined['content_id'].astype(str).str.replace('"', '', regex=False).str.strip()
    combined['original_url'] = combined['original_url'].astype(str).replace('nan', '').fillna('')
    combined['object_id'] = combined['object_id'].astype(str).replace('nan', '').fillna('')

    # Convert timestamp to UNIX using the robust parser
    combined['timestamp_share'] = combined['timestamp_share'].apply(parse_timestamp_robust)
    combined = combined.dropna(subset=['timestamp_share']).reset_index(drop=True)
    combined['timestamp_share'] = combined['timestamp_share'].astype('Int64')  # Nullable integer

    combined = combined[combined['object_id'].str.strip() != ""].copy()
    combined = combined.drop_duplicates(subset=['account_id', 'content_id', 'object_id', 'timestamp_share']).reset_index(drop=True)
    return combined

# --- Final Preprocessing Function ---
def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    """
    Performs final preprocessing steps on the combined DataFrame.
    Respects coordination_mode: uses text or URL as object_id.
    Ensures timestamp_share is UNIX integer.
    """
    df_processed = df.copy()

    # Standardize column names (basic attempt for uncombined upload)
    df_processed.columns = [c.lower() for c in df_processed.columns]
    
    # Core mapping and renaming for single/raw upload flexibility
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
    
    # Fill in potential missing columns for consistency with multi-source structure
    required_cols = ['account_id', 'content_id', 'object_id', 'timestamp_share', 'URL', 'source_dataset']
    for col in required_cols:
        if col not in df_processed.columns:
            df_processed[col] = np.nan
            
    # Process object_id and URL to ensure string type
    df_processed['object_id'] = df_processed['object_id'].astype(str).replace('nan', '').fillna('')
    df_processed['URL'] = df_processed['URL'].astype(str).replace('nan', '').fillna('')
    df_processed = df_processed[df_processed['object_id'].str.strip() != ""].copy()

    # Text cleaning helper for display/initial object_id processing
    def clean_text_for_display(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r"\\n|\\r|\\t", " ", text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    # Apply text extraction based on coordination mode
    if coordination_mode == "Text Content":
        df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    elif coordination_mode == "Shared URLs":
        df_processed['original_text'] = df_processed['URL'].astype(str).replace('nan', '').fillna('')

    df_processed = df_processed[df_processed['original_text'].str.strip() != ""].reset_index(drop=True)
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)

    if 'cluster' not in df_processed.columns: df_processed['cluster'] = -1 
    if 'source_dataset' not in df_processed.columns: df_processed['source_dataset'] = 'Uploaded_File' 

    for col in ['Outlet', 'Channel']:
        if col not in df_processed.columns: df_processed[col] = np.nan

    if df_processed.empty:
        st.error("❌ No valid data after final preprocessing.")
        
    return df_processed[['account_id', 'content_id', 'object_id', 'URL', 'timestamp_share', 'Platform', 'original_text', 'Outlet', 'Channel', 'cluster', 'source_dataset']].copy()


# --- Coordination Detection, Network, and Display functions ---

@st.cache_data(show_spinner="Detecting Coordinated Groups...")
def find_coordinated_groups(df, threshold, max_features):
    # Logic remains the same
    text_col = 'original_text'
    social_media_platforms = {'TikTok', 'Facebook', 'X', 'YouTube', 'Instagram', 'Telegram'}
    coordination_groups = {}
    clustered_groups = df[df['cluster'] != -1].groupby('cluster')
    
    for cluster_id, group in clustered_groups:
        if len(group) < 2: continue
        clean_df = group[['account_id', 'timestamp_share', 'Platform', 'URL', text_col]].copy()
        clean_df = clean_df.rename(columns={text_col: 'text', 'timestamp_share': 'Timestamp'})
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
                            "posts": group_posts.to_dict('records'), "num_posts": len(group_posts),
                            "num_accounts": len(group_posts['account_id'].unique()),
                            "max_similarity_score": round(max_sim, 3), "coordination_type": "TBD"
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
    
    texts = df['original_text'].tolist()
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1, 3))
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        df['cluster'] = -1
        return df
        
    dbscan = DBSCAN(eps=0.5, min_samples=CONFIG["bertrend"]["min_cluster_size"], metric='cosine')
    clusters = dbscan.fit_predict(tfidf_matrix)
    df['cluster'] = clusters
    return df

@st.cache_data(show_spinner="Building Network Graph...")
def cached_network_graph(df, coordination_type, data_source_key):
    # Logic remains the same
    G = nx.Graph()
    if coordination_type == "text" and 'cluster' in df.columns:
        valid_clusters = df[df['cluster'] != -1].groupby('cluster')
        for _, cluster_df in valid_clusters:
            accounts = cluster_df['account_id'].unique().tolist()
            if len(accounts) > 1:
                for i in range(len(accounts)):
                    for j in range(i + 1, len(accounts)):
                        if G.has_edge(accounts[i], accounts[j]):
                            G[accounts[i]][accounts[j]]['weight'] += 1
                        else:
                            G.add_edge(accounts[i], accounts[j], weight=1)
    
    elif coordination_type == "url" and 'URL' in df.columns:
        for url in df['URL'].unique():
            url_df = df[df['URL'] == url]
            accounts = url_df['account_id'].unique().tolist()
            if len(accounts) > 1:
                for i in range(len(accounts)):
                    for j in range(i + 1, len(accounts)):
                        if G.has_edge(accounts[i], accounts[j]):
                            G[accounts[i]][accounts[j]]['weight'] += 1
                        else:
                            G.add_edge(accounts[i], accounts[j], weight=1)
                            
    if G.nodes():
        try:
            influence_scores = nx.betweenness_centrality(G, weight='weight')
            amplification_scores = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        except Exception as e:
            logger.error(f"Error calculating centrality: {e}")
            influence_scores = {node: 0.1 for node in G.nodes()}
            amplification_scores = {node: 0.1 for node in G.nodes()}

        for node in G.nodes():
            G.nodes[node]['influence'] = influence_scores.get(node, 0)
            G.nodes[node]['amplification'] = amplification_scores.get(node, 0)
            G.nodes[node]['total_posts'] = df[df['account_id'] == node].shape[0]

    pos = nx.spring_layout(G, seed=42) if G.nodes() else {}
    return G, pos

# --- Display Helper Functions (Added for completeness) ---

def display_imi_report_visuals(report_df):
    """Function to display the report data interactively."""
    st.markdown("### 📰 Narrative Summaries")
    st.info("This is the interactive summary display.")
    
    if 'Emerging Virality' in report_df.columns:
        report_df = report_df.sort_values(by='Emerging Virality', ascending=False, key=lambda x: x.str.split(':', expand=True)[0] if x.dtype == object else x).reset_index(drop=True)
    
    for i, row in report_df.iterrows():
        title = row.get('Title', f"Summary {i+1}")
        # Use Context or original_text depending on data source
        narrative_text = row.get('original_text', row.get('Context', 'No narrative context provided.'))
        
        st.markdown(f"#### **{title}**")
        
        col1, col2 = st.columns([1, 4])
        col1.metric("Virality", row.get('Emerging Virality', 'N/A'))
        if 'Country' in row: col1.metric("Country", row['Country'])
        
        col2.markdown(narrative_text)

        with st.expander("Details"):
            urls = row.get('URLs', row.get('URL', 'N/A'))
            if isinstance(urls, list): urls = ", ".join(urls)
            st.markdown(f"**Source URL(s):** {urls}")
            if 'Posts' in row.index:
                st.markdown(f"**Metrics:** Posts: {row.get('Posts', 1)}, Accounts: {row.get('Accounts', 1)}, Platforms: {row.get('Platforms', 'N/A')}")
            if 'First Detected' in row.index and 'Last Updated' in row.index:
                first_ts = row['First Detected'].strftime('%Y-%m-%d %H:%M')
                last_ts = row['Last Updated'].strftime('%Y-%m-%d %H:%M')
                st.markdown(f"**Lifecycle:** {first_ts} to {last_ts}")
        st.markdown("---")

def plot_network_graph(G, pos, coordination_mode):
    """Function for plotting the network graph."""
    st.markdown("### Network Graph Visualization")
    if not G.nodes():
        st.warning("No connections found.")
        return

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    influence_values = [G.nodes[node]['influence'] for node in G.nodes()]
    amplification_values = [G.nodes[node]['amplification'] for node in G.nodes()]
    node_size = np.array(influence_values) * 50 + 5 

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        account_id = adjacencies[0]
        num_connections = len(list(adjacencies[1].keys()))
        total_posts = G.nodes[account_id].get('total_posts', 0)
        
        node_adjacencies.append(num_connections)
        node_text.append(
            f'Account: {account_id}<br>'
            f'Connections: {num_connections}<br>'
            f'Total Posts: {total_posts}<br>'
            f'Influence Score: {G.nodes[account_id]["influence"]:.4f}<br>'
            f'Amplification Score: {G.nodes[account_id]["amplification"]:.4f}')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=node_text,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=amplification_values,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Amplification Score',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=f'<br>Network of Accounts Sharing {coordination_mode}',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Connections between accounts sharing the same clustered text or URL.",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    st.plotly_chart(fig, use_container_width=True)


# --- Tab Functions ---

def tab1_summary_statistics(df, filtered_df_global, data_source, coordination_mode):
    # ... Tab 1 logic ...
    st.subheader("📌 Summary Statistics")
    st.markdown("### 🔬 Preprocessed Data Sample")
    st.markdown(f"**Data Source:** `{data_source}` | **Coordination Mode:** `{coordination_mode}` | **Total Rows:** `{len(df):,}`")
    
    display_cols_overview = ['account_id', 'content_id', 'object_id', 'timestamp_share', 'URL', 'Platform']
    existing_cols = [col for col in df.columns if col in display_cols_overview]
    
    if not df.empty and existing_cols:
        st.dataframe(df[existing_cols].head(10))
    else:
        st.info("No data available to display in this tab.")

    if filtered_df_global.empty: return

    if 'source_dataset' in filtered_df_global.columns:
        st.markdown("### 📊 Data Sources in Filtered Data")
        source_counts = filtered_df_global['source_dataset'].value_counts()
        st.dataframe(source_counts)

    top_influencers = filtered_df_global['account_id'].value_counts().head(10)
    fig_src = px.bar(top_influencers, title="Top 10 Influencers", labels={'value': 'Posts', 'index': 'account_id'})
    st.plotly_chart(fig_src, use_container_width=True)
    st.markdown("**Top 10 Influencers**: Shows the most active accounts based on number of posts.")

    if 'Platform' in filtered_df_global.columns:
        all_platforms_counts = filtered_df_global['Platform'].value_counts()
        fig_platform = px.bar(all_platforms_counts, title="Post Distribution by Platform", labels={'value': 'Posts', 'index': 'Platform'})
        st.plotly_chart(fig_platform, use_container_width=True)
        st.markdown("**Post Distribution by Platform**: Visualizes how posts are distributed across different social and media platforms.")
    
    plot_df = filtered_df_global.copy()
    if 'timestamp_share' in plot_df.columns:
        plot_df['timestamp_share'] = pd.to_numeric(plot_df['timestamp_share'], errors='coerce')
        valid_mask = (plot_df['timestamp_share'] >= 946684800) & (plot_df['timestamp_share'] <= 4102444800)
        plot_df = plot_df[valid_mask]
        
        if not plot_df.empty:
            plot_df['datetime'] = pd.to_datetime(plot_df['timestamp_share'], unit='s', utc=True)
            plot_df = plot_df.set_index('datetime')
            time_series = plot_df.resample('D').size()
            
            if not time_series.empty:
                fig_ts = px.area(time_series, title="Daily Post Volume", labels={'value': 'Number of Posts', 'datetime': 'Date'}, markers=True)
                fig_ts.update_layout(xaxis_title="Date", yaxis_title="Number of Posts")
                st.plotly_chart(fig_ts, use_container_width=True)
                st.markdown("**Daily Post Volume**: Visualizes the volume of posts over time to identify spikes or trends.")

def tab2_narrative_intelligence(report_df):
    """Displays the Narrative Report using the common helper, handling both raw and preprocessed data."""
    st.header("🧠 Narrative Summary Report")
    st.markdown("**Focus:** Summaries of clustered, high-similarity content generated by the LLM, presented in a non-technical, intelligence-ready format, or the display of preprocessed summaries.")
    
    if report_df.empty:
        st.info("No narrative reports available. Please ensure data is uploaded and narratives have been generated.")
    else:
        # Calls the function that handles display logic
        display_imi_report_visuals(report_df)

def tab3_coordination_detection(df, coordination_groups):
    # ... Tab 2 logic ...
    st.header("🔗 Text Similarity & Coordination Analysis")
    st.markdown("**Focus:** Groups of posts from different accounts with very high text similarity, indicating potential coordinated activity.")

    if not coordination_groups:
        st.info("No coordinated groups were detected based on the current clustering or similarity threshold.")
        st.caption(f"Detection Threshold: {CONFIG['coordination_detection']['threshold']} (Text Similarity)")
        return

    st.markdown(f"**Total Coordinated Groups Detected:** **{len(coordination_groups)}**")
    st.caption(f"Detection Threshold: {CONFIG['coordination_detection']['threshold']} (Text Similarity)")
    
    summary_data = []
    for i, group in enumerate(coordination_groups):
        summary_data.append({
            'Group ID': i + 1, 'Type': group['coordination_type'], 'Posts': group['num_posts'],
            'Accounts': group['num_accounts'], 'Max Sim. Score': group['max_similarity_score'],
            'Top Platform': pd.DataFrame(group['posts'])['Platform'].mode()[0] if not pd.DataFrame(group['posts']).empty else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values(by=['Posts', 'Accounts'], ascending=False).reset_index(drop=True)
    st.markdown("### Group Summary Table")
    st.dataframe(summary_df)
    
    st.markdown("---")
    st.markdown("### Detailed Group Analysis")
    
    for i, group in enumerate(coordination_groups[:10]):
        with st.expander(f"Group {i+1}: {group['coordination_type']} | Posts: {group['num_posts']} | Accounts: {group['num_accounts']}"):
            posts_df = pd.DataFrame(group['posts'])
            posts_df = posts_df.sort_values('Timestamp').reset_index(drop=True)
            posts_df['Timestamp'] = pd.to_datetime(posts_df['Timestamp'], unit='s', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

            st.markdown("**Coordinated Posts (First 10)**")
            st.dataframe(posts_df[['Timestamp', 'account_id', 'Platform', 'text']].head(10))

def tab4_network_analysis(df, coordination_mode):
    # ... Tab 3 logic ...
    st.header("🕸️ Account Network Analysis")
    st.markdown(f"**Focus:** Visualizing the connections between accounts based on shared **{coordination_mode.lower()}** (text similarity or shared URLs).")

    if df.empty:
        st.info("Please upload and process data to generate the network graph.")
        return

    network_type = "text" if coordination_mode == "Text Content" else "url"
    # Use a simple unique key for caching
    data_source_key = st.session_state.get('data_source', 'default_data') + str(len(df)) + str(coordination_mode)

    G, pos = cached_network_graph(df, network_type, data_source_key)

    if not G.nodes():
        st.warning("The network graph could not be generated. Check if content similarity clustering was performed and accounts are sharing the selected content type.")
    else:
        plot_network_graph(G, pos, coordination_mode)
        
        st.markdown("---")
        st.markdown("### Key Metrics Explained")
        st.markdown(
            """
            * **Node Size (Key Influencer Score):** Based on **Betweenness Centrality**.
            * **Node Color (Loud Amplifier Score):** Based on **Eigenvector Centrality**.
            """
        )

# --- Main Streamlit App ---

def main():
    st.set_page_config(layout="wide", page_title="Election monitoring Dashboard")
    st.title(" Election monitoring Dashboard")
    
    # --- Session State Initialization ---
    if 'data_source' not in st.session_state:
        st.session_state['data_source'] = "Upload CSV Files"
    if 'coordination_mode' not in st.session_state:
        st.session_state['coordination_mode'] = "Text Content"
    if 'is_preprocessed_data_mode' not in st.session_state:
        st.session_state['is_preprocessed_data_mode'] = False
        
    # --- Sidebar Input (Configuration) ---
    st.sidebar.header("⚙️ Configuration")
    
    # Use session state for coordination mode and source
    coordination_mode = st.sidebar.selectbox(
        "Coordination Type (Raw Data Only)",
        ["Text Content", "Shared URLs"],
        key='coordination_mode_select'
    )
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Upload CSV Files", "Load Sample Data (Not Implemented)"],
        key='data_source_select'
    )
    st.session_state['coordination_mode'] = coordination_mode
    st.session_state['data_source'] = data_source
    
    # --- Data Upload and Processing ---
    combined_raw_df = pd.DataFrame()
    is_preprocessed_data_mode = False

    if data_source == "Upload CSV Files":
        st.sidebar.info("Upload your CSV files below. Uploading a **Preprocessed Summary** will bypass detailed analysis.")
        
        # File Upload Widgets
        uploaded_preprocessed_file = st.sidebar.file_uploader("Upload Preprocessed Summary CSV (Optional)", type=["csv"], key="preprocessed_upload")
        uploaded_meltwater_file = st.sidebar.file_uploader("Upload Meltwater CSV", type=["csv"], key="meltwater_upload")
        uploaded_civicsignals_file = st.sidebar.file_uploader("Upload CivicSignals CSV", type=["csv"], key="civicsignals_upload")
        uploaded_openmeasure_file = st.sidebar.file_uploader("Upload Open-Measure CSV", type=["csv"], key="openmeasure_upload")

        # 1. Read all files
        preprocessed_df_upload = read_uploaded_file(uploaded_preprocessed_file, "Preprocessed Summary")
        meltwater_df_upload = read_uploaded_file(uploaded_meltwater_file, "Meltwater")
        civicsignals_df_upload = read_uploaded_file(uploaded_civicsignals_file, "CivicSignals")
        openmeasure_df_upload = read_uploaded_file(uploaded_openmeasure_file, "Open-Measure")

        # 2. Conditional Processing
        if not preprocessed_df_upload.empty:
            # **Mode: Preprocessed Summary**
            with st.spinner("📥 Processing uploaded preprocessed summary data..."):
                combined_raw_df = process_preprocessed_data(preprocessed_df_upload)
            st.sidebar.success(f"✅ Loaded {len(combined_raw_df)} summary entries. **Analysis tabs will be limited to a single Report View.**")
            is_preprocessed_data_mode = True 
        else:
            # **Mode: Raw Data Combination**
            with st.spinner("📥 Combining uploaded raw datasets..."):
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
            
            # Fallback logic for single file upload (to handle cases where only one raw file is provided)
            if combined_raw_df.empty:
                 raw_dfs = [meltwater_df_upload, civicsignals_df_upload, openmeasure_df_upload]
                 raw_dfs = [d for d in raw_dfs if not d.empty]
                 if raw_dfs:
                    combined_raw_df = raw_dfs[np.argmax([len(d) for d in raw_dfs])].copy()
                    st.sidebar.info(f"Using fallback: Largest raw file ({len(combined_raw_df)} rows).")

            if combined_raw_df.empty:
                st.warning("No data loaded from uploaded raw files.")
            else:
                st.sidebar.success(f"✅ Combined {len(combined_raw_df)} posts from uploaded datasets.")

    st.session_state['is_preprocessed_data_mode'] = is_preprocessed_data_mode

    # --- Final Preprocess ---
    df = pd.DataFrame()
    if not combined_raw_df.empty:
        if is_preprocessed_data_mode:
            # Data is already processed and mapped by process_preprocessed_data
            df = combined_raw_df.copy()
            st.sidebar.info("Data treated as Preprocessed Summary, skipping raw data cleaning steps.")
        else:
            with st.spinner("⏳ Preprocessing and mapping combined data..."):
                df = final_preprocess_and_map_columns(combined_raw_df, coordination_mode=coordination_mode)
    
    if df.empty:
        st.warning("No valid data after final preprocessing.")

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Global Filters (Apply to all tabs)")
    
    # Timestamp and Date Range Logic
    if 'timestamp_share' not in df.columns or df['timestamp_share'].dtype != 'Int64' or df['timestamp_share'].isnull().all():
        min_ts = int(pd.Timestamp('2024-01-01', tz='UTC').timestamp())
        max_ts = int(pd.Timestamp.now(tz='UTC').timestamp())
        min_date = pd.to_datetime(min_ts, unit='s').date()
        max_date = pd.to_datetime(max_ts, unit='s').date()
    else:
        min_ts_df = df['timestamp_share'].min()
        max_ts_df = df['timestamp_share'].max()
        min_date = pd.to_datetime(min_ts_df, unit='s', errors='coerce').date() if pd.notna(min_ts_df) else pd.Timestamp.now().date()
        max_date = pd.to_datetime(max_ts_df, unit='s', errors='coerce').date() if pd.notna(max_ts_df) else pd.Timestamp.now().date()
        min_date = min(min_date, max_date)
        max_date = max(min_date, max_date)
    
    try:
        selected_date_range = st.sidebar.date_input("Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
    except Exception:
        selected_date_range = [min_date, max_date]

    if len(selected_date_range) == 2:
        start_ts = int(pd.Timestamp(selected_date_range[0], tz='UTC').timestamp())
        end_ts = int((pd.Timestamp(selected_date_range[1], tz='UTC') + timedelta(days=1) - timedelta(microseconds=1)).timestamp())
    elif len(selected_date_range) == 1:
        start_ts = int(pd.Timestamp(selected_date_range[0], tz='UTC').timestamp())
        end_ts = start_ts + 86400 - 1
    else:
        start_ts = min_ts
        end_ts = max_ts

    if 'timestamp_share' in df.columns:
        filtered_df_global = df[
            (df['timestamp_share'] >= start_ts) &
            (df['timestamp_share'] <= end_ts)
        ].copy()
    else:
        filtered_df_global = pd.DataFrame()

    if 'Platform' in filtered_df_global.columns:
        platform_options = filtered_df_global['Platform'].dropna().unique().tolist()
        selected_platforms = st.sidebar.multiselect("Filter by Platform", options=platform_options, default=platform_options)
        filtered_df_global = filtered_df_global[filtered_df_global['Platform'].isin(selected_platforms)].copy()
    
    # Add new control to limit posts
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏩ Performance Controls")
    max_posts_for_analysis = st.sidebar.number_input(
        "Limit Posts for Analysis (0 for all)",
        min_value=0,
        value=50000,
        step=1000,
        help="To speed up analysis on large datasets, enter a number to process a random sample of posts. Set to 0 to use all posts."
    )
    st.sidebar.markdown(f"**Filtered Posts:** `{len(filtered_df_global):,}`")

    # Apply sampling if requested
    if max_posts_for_analysis > 0 and len(filtered_df_global) > max_posts_for_analysis:
        df_for_analysis = filtered_df_global.sample(n=max_posts_for_analysis, random_state=42).copy()
        st.sidebar.warning(f"⚠️ Analyzing a random sample of **{len(df_for_analysis):,}** posts to improve performance.")
    else:
        df_for_analysis = filtered_df_global.copy()
        st.sidebar.info(f"✅ Analyzing all **{len(df_for_analysis):,}** posts.")
        
    if df_for_analysis.empty:
        st.warning("No data remains after applying filters and performance limits.")
    
    # --- Download Combined Data ---
    @st.cache_data
    def convert_df_to_csv(data_frame):
        return data_frame.to_csv(index=False).encode('utf-8')
        
    st.sidebar.markdown("### 💾 Download Combined & Preprocessed Data")
    download_df_columns = ['account_id', 'content_id', 'object_id', 'timestamp_share']
    if is_preprocessed_data_mode:
        download_df_columns = ['Country', 'Emerging Virality', 'original_text', 'URL', 'timestamp_share']

    downloadable_df = df[download_df_columns].copy() if all(col in df.columns for col in download_df_columns) else pd.DataFrame()

    if not downloadable_df.empty:
        combined_preprocessed_csv = convert_df_to_csv(downloadable_df)
        st.sidebar.download_button(
            "Download Preprocessed Dataset (Core Columns)",
            combined_preprocessed_csv,
            f"preprocessed_combined_core_data_{coordination_mode.replace(' ', '_').lower()}.csv",
            "text/csv",
            help="Downloads the data after all preprocessing and column mapping. 'object_id' contains either text or URL based on your selection."
        )
    
    # Export filtered data
    st.sidebar.markdown("### 📄 Export Filtered Results")
    if not filtered_df_global.empty:
        filtered_csv_data = convert_df_to_csv(filtered_df_global)
        st.sidebar.download_button("Download Filtered Data (All Columns)", filtered_csv_data, "filtered_dashboard_data.csv", "text/csv")


    # --- Analysis Steps ---
    df_clustered = df_for_analysis.copy()
    coordination_groups = []

    if not is_preprocessed_data_mode and not df_for_analysis.empty:
        # Only run clustering/coordination for RAW data
        df_clustered = perform_clustering(df_for_analysis, coordination_mode)
        coordination_groups = find_coordinated_groups(
            df_clustered, 
            CONFIG["coordination_detection"]["threshold"], 
            CONFIG["coordination_detection"]["max_features"]
        )

    # --- Report Generation ---
    if is_preprocessed_data_mode:
        # Map preprocessed data columns to report_df structure for tab display
        report_df = pd.DataFrame({
            'ID': df_clustered.index,
            'Title': df_clustered['Country'] + ' - ' + df_clustered['Emerging Virality'],
            'Posts': 1, 
            'Accounts': 1, 
            'Platforms': df_clustered['Platform'],
            'First Detected': pd.to_datetime(df_clustered['timestamp_share'], unit='s', errors='coerce'),
            'Last Updated': pd.to_datetime(df_clustered['timestamp_share'], unit='s', errors='coerce'),
            'Context': df_clustered['original_text'],
            'URLs': df_clustered['URL'],
            'Source Datasets': df_clustered['source_dataset'],
            'Emerging Virality': df_clustered['Emerging Virality'],
            # Include these for display helper to function properly
            'original_text': df_clustered['original_text'], 
            'is_preprocessed_summary': df_clustered['is_preprocessed_summary'], 
            'Country': df_clustered['Country']
        })
    else:
        # Generate report from clustered raw data
        report_df = generate_imi_report(df_clustered, data_source)


    # --- Conditional Tab Rendering (The core fix) ---

    if is_preprocessed_data_mode:
        # If preprocessed, only show the summary/report tab
        tab_titles = ["Summary Report"]
        # Lambda function to call the report tab function with the preprocessed report_df
        tab_fns = [lambda: tab2_narrative_intelligence(report_df)] 
    else:
        # If raw data, show all tabs
        tab_titles = ["1. Summary & Overview", "2. Coordination Detection", "3. Network Analysis", "4. Narrative Report"]
        tab_fns = [
            lambda: tab1_summary_statistics(df, filtered_df_global, data_source, coordination_mode),
            lambda: tab3_coordination_detection(df_clustered, coordination_groups),
            lambda: tab4_network_analysis(df_clustered, coordination_mode),
            lambda: tab2_narrative_intelligence(report_df)
        ]

    # Create and run tabs
    if not df.empty:
        tabs = st.tabs(tab_titles)
        for tab, tab_fn in zip(tabs, tab_fns):
            with tab:
                tab_fn()
    else:
        st.warning("Please upload valid data using the sidebar controls to begin analysis.")


# --- Run Main ---
if __name__ == '__main__':
    # Initialize Streamlit session state variables if not already set (needed for widget stability)
    if 'data_source' not in st.session_state:
        st.session_state['data_source'] = "Upload CSV Files"
    if 'coordination_mode' not in st.session_state:
        st.session_state['coordination_mode'] = "Text Content"
    if 'is_preprocessed_data_mode' not in st.session_state:
        st.session_state['is_preprocessed_data_mode'] = False
        
    main()
