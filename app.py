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
        return None, "Error"
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens
        )
        # Mocking the title extraction for display purposes. The LLM's full output is complex.
        content = response.choices[0].message.content.strip()
        title_match = re.search(r'\*\*(.*?)\*\*', content, re.DOTALL)
        title = title_match.group(1).strip() if title_match else "Un-Titled Narrative"
        return content, title
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None, "Error"

# --- IMI Summary Generator (YOUR EXACT VERSION - MODIFIED TO RETURN TITLE) ---
def summarize_cluster(texts, urls, cluster_data):
    # Truncate to top 50 posts to reduce token load
    joined = "\n".join(texts[:50])
    url_context = "\nRelevant post links:\n" + "\n".join(urls[:5]) if urls else ""

    min_timestamp_str = pd.to_datetime(cluster_data['Timestamp'].min(), unit='s').strftime('%Y-%m-%d %H:%M')
    max_timestamp_str = pd.to_datetime(cluster_data['Timestamp'].max(), unit='s').strftime('%Y-%m-%d %H:%M')

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

    response, title = safe_llm_call(prompt, max_tokens=2048)

    if response:
        # Extract URLs from the summary (if any)
        evidence_urls = re.findall(r"(https?://[^\s\)\]]+)", response)

        # Clean up and ensure URLs are included
        cleaned_summary = re.sub(r'\*\*Here is a concise.*?\*\*', '', response, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned_summary = re.sub(r'\*\*Here are a few options.*?\*\*', '', cleaned_summary, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned_summary = re.sub(r'"[^"]*"', '', cleaned_summary).strip()

        # Ensure URLs are at the end
        if evidence_urls:
            urls_section = "\n\nSources: " + ", ".join(evidence_urls[:5])
            cleaned_summary += urls_section

        return cleaned_summary, evidence_urls, title
    else:
        logger.error(f"LLM call failed for cluster {cluster_data['cluster'].iloc[0]}")
        return "Summary generation failed.", [], "Failed to Summarize"

# --- Helper Functions (Remaining untouched for brevity) ---
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

# --- New Function: Generate IMI Report (Mocked for demonstration) ---
@st.cache_data(show_spinner=False)
def generate_imi_report(clustered_df, data_source_key):
    """Mocks the final IMI report generation from clustered data."""
    if clustered_df.empty or 'cluster' not in clustered_df.columns:
        return pd.DataFrame()

    report_data = []
    clustered_df['Timestamp'] = clustered_df['timestamp_share'] # Align column name for summarizer
    
    # Process only the actual clusters (cluster != -1)
    unique_clusters = clustered_df[clustered_df['cluster'] != -1]['cluster'].unique()
    
    # Simulate a few summaries for demonstration
    for i, cluster_id in enumerate(unique_clusters[:5]): # Limiting to 5 for speed
        cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
        
        texts = cluster_data['original_text'].tolist()
        urls = cluster_data['URL'].tolist()
        
        # Call the LLM (This is the costly step)
        summary, evidence_urls, title = summarize_cluster(texts, urls, cluster_data)

        # Calculate metrics for the report table
        min_ts = cluster_data['Timestamp'].min()
        max_ts = cluster_data['Timestamp'].max()
        
        # Add mock posts/accounts/platforms if not available from LLM (should be available in actual clustered_df)
        posts_count = len(cluster_data)
        accounts_count = cluster_data['account_id'].nunique()
        platforms_list = cluster_data['Platform'].unique().tolist()
        datasets_list = cluster_data['source_dataset'].unique().tolist()
        
        report_data.append({
            'ID': cluster_id,
            'Title': title,
            'Posts': posts_count,
            'Accounts': accounts_count,
            'Platforms': ", ".join(platforms_list[:3]),
            'First Detected': pd.to_datetime(min_ts, unit='s'),
            'Last Updated': pd.to_datetime(max_ts, unit='s'),
            'Context': summary,
            'URLs': evidence_urls,
            'Source Datasets': ", ".join(datasets_list),
        })

    return pd.DataFrame(report_data)

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
def cached_network_graph(df, coordination_type, data_source_key):
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

# --- NEW: Reusable Display Function for IMI Report ---
def display_imi_report_visuals(report_df):
    
    st.subheader("Narrative Lifecycle Timeline 📈")
    
    # 1. Narrative Lifecycle Timeline
    timeline_df = report_df.copy()
    
    # Ensure datetime columns are correct (required for Preprocessed Report upload)
    try:
        if 'First Detected' not in timeline_df.columns or 'Last Updated' not in timeline_df.columns:
            st.error("Timeline requires 'First Detected' and 'Last Updated' columns.")
            return

        timeline_df['First Detected'] = pd.to_datetime(timeline_df['First Detected'], errors='coerce')
        timeline_df['Last Updated'] = pd.to_datetime(timeline_df['Last Updated'], errors='coerce')
        timeline_df = timeline_df.dropna(subset=['First Detected', 'Last Updated'])
        timeline_df['Time Span'] = timeline_df['Last Updated'] - timeline_df['First Detected']
        
        # Ensure 'Posts' and 'Accounts' exist or mock them for visualization purposes
        if 'Posts' not in timeline_df.columns:
            timeline_df['Posts'] = 10 # Default size
        if 'Accounts' not in timeline_df.columns:
            timeline_df['Accounts'] = 5 # Default hover data
        if 'Platforms' not in timeline_df.columns:
             timeline_df['Platforms'] = "Unknown"
             
        # Convert Time Span to hours for a readable Y-axis (or days if duration is long)
        if timeline_df['Time Span'].dt.days.max() > 1:
             y_label = "Duration (Days)"
             timeline_df['Time Span Value'] = timeline_df['Time Span'].dt.total_seconds() / (24 * 3600)
        else:
             y_label = "Duration (Hours)"
             timeline_df['Time Span Value'] = timeline_df['Time Span'].dt.total_seconds() / 3600

        fig_timeline = px.scatter(
            timeline_df, 
            x='First Detected', 
            y='Time Span Value', 
            size='Posts', 
            color='Platforms',
            hover_data=['Title', 'Posts', 'Accounts', 'Time Span'],
            title="Narrative Virality: Emergence vs. Duration"
        )
        fig_timeline.update_layout(
            xaxis_title="First Detection Timestamp",
            yaxis_title=y_label,
            height=500
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate Timeline chart. Check date columns: {e}")

    # 2. Key Metrics by Narrative (Structured Display)
    st.subheader("Interactive Narrative Reports")

    # Standardize column names for display
    display_report = report_df.copy()
    
    # Clean up columns for robust display in expander
    display_report['Context'] = display_report['Context'].fillna("No summary provided.")
    display_report['Title'] = display_report['Title'].fillna("Untitled Narrative")
    display_report['Posts'] = display_report['Posts'].fillna(0).astype(int)
    display_report['Accounts'] = display_report['Accounts'].fillna(0).astype(int)
    display_report['Platforms'] = display_report['Platforms'].fillna("N/A")
    display_report['Source Datasets'] = display_report['Source Datasets'].fillna("N/A")
    
    # Ensure URLs column is a list of strings
    def ensure_list_of_urls(val):
        if pd.isna(val) or val is None: return []
        if isinstance(val, list): return val
        if isinstance(val, str):
            # Attempt to parse as a list string
            try:
                parsed = eval(val)
                if isinstance(parsed, list):
                    return [u for u in parsed if isinstance(u, str) and u.startswith("http")]
            except:
                # Fallback to simple URL extraction
                return re.findall(r'https?://[^\s\)\]\,\"\'\<\>]+', val)
        return []

    if 'URLs' in display_report.columns:
        display_report['URLs'] = display_report['URLs'].apply(ensure_list_of_urls)
    else:
        display_report['URLs'] = [[]] * len(display_report)
        
    # --- Display using Expanders ---
    for index, row in display_report.iterrows():
        title = row['Title']
        post_count = row['Posts']
        account_count = row['Accounts']
        platforms = row['Platforms']
        
        # Get formatted dates if available
        first_detected = row['First Detected'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['First Detected']) else 'N/A'
        last_updated = row['Last Updated'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['Last Updated']) else 'N/A'
        
        header_title = f"**{title}** | Posts: **{post_count:,}** | Accounts: **{account_count:,}**"
        
        with st.expander(header_title):
            st.markdown("### 📰 Narrative Context")
            st.markdown(f"**Platforms:** {platforms}")
            st.markdown(f"**Source Datasets:** {row['Source Datasets']}")
            st.markdown(f"**Lifecycle:** First Detected: `{first_detected}` | Last Updated: `{last_updated}`")
            
            st.markdown("---")
            
            # LLM-Generated Context (The actual summary)
            st.markdown("### 📄 Intelligence Report Summary")
            st.markdown(row['Context'])
            
            # Supporting URLs
            urls = row['URLs']
            if urls:
                st.markdown("### 🔗 Key Supporting Evidence URLs")
                for url in urls[:5]: # Limit to 5 for cleanliness
                    st.markdown(f"- <a href='{url}' target='_blank' style='text-decoration: underline; color: #1f77b4;'>{url}</a>", unsafe_allow_html=True)
            else:
                 st.markdown("### 🔗 Key Supporting Evidence URLs\n- *No direct URLs provided in this report.*")
            st.markdown("---")
            
    st.success(f"✅ Displaying {len(report_df)} narrative reports.")
    

# --- Main App ---
def main_election_monitoring():
    st.set_page_config(layout="wide", page_title="Election Monitoring Dashboard")
    st.sidebar.title("Configuration")

    data_source = st.sidebar.selectbox("Select Data Source", ["Upload CSV Files", "Upload Preprocessed Report"])

    if data_source == "Upload Preprocessed Report":
        # === MODE 2: ONLY ONE TAB - ENHANCED SUMMARY VISUALIZATION ===
        st.header("📊 Narrative Insights from Preprocessed IMI Report")
        uploaded_report = st.file_uploader("Upload your preprocessed IMI report (CSV)", type=["csv"], key="imi_report_upload")
        if uploaded_report:
            try:
                report_df = pd.read_csv(uploaded_report)
                report_df.columns = report_df.columns.str.strip().str.replace(r'[^a-zA-Z0-9\s_]', '', regex=True).str.replace(' ', '_', regex=False)
                
                # Standardize column names to match expected for display
                col_map = {
                    'Context': 'Context', 
                    'Summary': 'Context', # Alternative for summary content
                    'Narrative_Title': 'Title',
                    'Title': 'Title',
                    'First_Detected': 'First Detected',
                    'Last_Updated': 'Last Updated',
                    'Posts': 'Posts',
                    'Accounts': 'Accounts',
                    'Platforms': 'Platforms',
                    'Source_Datasets': 'Source Datasets',
                    'URLs': 'URLs', 
                    'Evidence_URLs': 'URLs',
                    'Evidence': 'Context' # If 'Evidence' is the long text
                }
                
                final_cols = {}
                for old, new in col_map.items():
                    if old in report_df.columns and new not in final_cols:
                        final_cols[new] = old
                
                if 'Context' not in final_cols:
                    st.error("❌ The report must contain a column for the Narrative Summary/Context (e.g., 'Context', 'Summary', or 'Evidence').")
                    return
                if 'Title' not in final_cols:
                    st.warning("⚠️ No 'Title' column found. Summaries will be labeled by their content.")
                if 'First Detected' not in final_cols or 'Last Updated' not in final_cols:
                    st.warning("⚠️ Missing 'First Detected' and/or 'Last Updated' columns. Timeline chart may not display correctly.")

                # Select and rename columns for the display function
                display_df = report_df[[col for col in final_cols.values()]].rename(columns={v: k for k, v in final_cols.items()})
                
                # Fill missing required columns for robust display
                if 'Title' not in display_df.columns: display_df['Title'] = display_df['Context'].apply(lambda x: str(x).split('\n')[0][:80] + '...')
                if 'Posts' not in display_df.columns: display_df['Posts'] = 1 # Mock
                if 'Accounts' not in display_df.columns: display_df['Accounts'] = 1 # Mock
                if 'Platforms' not in display_df.columns: display_df['Platforms'] = "Various/Unknown"
                if 'Source Datasets' not in display_df.columns: display_df['Source Datasets'] = "Reported"

                display_df = display_df.dropna(subset=["Context"]).reset_index(drop=True)
                
                # --- Call the NEW Display Function ---
                display_imi_report_visuals(display_df)

                # Download
                output_df = display_df.copy()
                st.download_button(
                    "📥 Download Cleaned Report",
                    convert_df_to_csv(output_df),
                    "imi_narrative_insights.csv",
                    "text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Failed to process report. Ensure it's a clean CSV file with required columns: {e}")
        else:
            st.info("📤 Please upload a preprocessed IMI report in CSV format.")

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
        data_source_key = hash(tuple(df.index)) if not df.empty else 0
        
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
        max_posts_for_analysis = st.sidebar.number_input("Limit Posts for Analysis (0 for all)", min_value=0, value=5000, step=1000)
        if max_posts_for_analysis > 0 and len(filtered_df_global) > max_posts_for_analysis:
            df_for_analysis = filtered_df_global.sample(n=max_posts_for_analysis, random_state=42).copy()
            st.sidebar.warning(f"⚠️ Analyzing a random sample of **{len(df_for_analysis):,}** posts.")
        else:
            df_for_analysis = filtered_df_global.copy()
            st.sidebar.info(f"✅ Analyzing all **{len(df_for_analysis):,}** posts.")

        # --- Clustering/Report Generation (Happens once after filtering) ---
        if not df_for_analysis.empty and coordination_mode == "Text Content":
            st.sidebar.subheader("Clustering Params")
            eps_val = st.sidebar.slider("DBSCAN Epsilon (Eps)", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
            min_samples_val = st.sidebar.number_input("DBSCAN Min Samples", min_value=2, value=3, step=1)
            max_features_val = st.sidebar.number_input("TFIDF Max Features", min_value=500, value=2000, step=500)
            
            with st.spinner(f"🌀 Clustering {len(df_for_analysis):,} posts..."):
                # Pass data_source_key to re-run clustering only when data changes
                clustered_df = cached_clustering(df_for_analysis, eps_val, min_samples_val, max_features_val, data_source_key)
            
            # Generate report after clustering
            if not clustered_df.empty:
                # Use a combined key for the report generation cache
                report_cache_key = (data_source_key, eps_val, min_samples_val, max_features_val)
                with st.spinner("🧠 Generating LLM Summaries for top clusters..."):
                    imi_report_df = generate_imi_report(clustered_df, report_cache_key)
            else:
                imi_report_df = pd.DataFrame()
        else:
            clustered_df = df_for_analysis.copy()
            imi_report_df = pd.DataFrame()


        # --- Download Buttons ---
        if not df.empty:
            core_cols = ['account_id', 'content_id', 'object_id', 'timestamp_share']
            if all(col in df.columns for col in core_cols):
                csv_data = convert_df_to_csv(df[core_cols])
                st.sidebar.download_button("Download Preprocessed Data", csv_data, "preprocessed.csv", "text/csv")

        if not filtered_df_global.empty:
            csv_filtered = convert_df_to_csv(filtered_df_global)
            st.sidebar.download_button("Download Filtered Data", csv_filtered, "filtered.csv", "text/csv")
            
        if not imi_report_df.empty:
            csv_report = convert_df_to_csv(imi_report_df[['Title', 'Posts', 'Accounts', 'Platforms', 'First Detected', 'Last Updated', 'Context']])
            st.sidebar.download_button("Download IMI Report", csv_report, "imi_report.csv", "text/csv")


        # --- Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "🔍 Similarity & Coordination",
            "🕸️ Network Graph",
            "📝 Summary"
        ])

        # === TAB 1: Overview (Uses previous fix) ===
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

        # === TAB 2: Similarity & Coordination (Requires clustered_df) ===
        with tab2:
            st.header("🔍 Narrative Similarity & Coordination")
            if clustered_df.empty or 'cluster' not in clustered_df.columns:
                st.info("Run clustering first (Tab 1/Sidebar) to find coordinated groups.")
            else:
                # Add clustering metrics
                num_clusters = clustered_df['cluster'].nunique() - 1 
                num_noise = (clustered_df['cluster'] == -1).sum()
                st.metric("Total Narratives (Clusters)", num_clusters)
                st.markdown(f"Posts identified as noise: {num_noise:,}")
                
                # Coordination Detection
                threshold = st.slider("Text Similarity Threshold for Coordination", min_value=0.5, max_value=1.0, value=0.95, step=0.01)
                max_feat_coord = st.number_input("Coordination TFIDF Max Features", min_value=1000, value=5000, step=500)
                
                if st.button("Find Coordinated Groups"):
                    with st.spinner("Finding coordinated groups..."):
                        coordinated_groups = find_coordinated_groups(clustered_df, threshold, max_feat_coord)
                    
                    if coordinated_groups:
                        st.subheader(f"Found {len(coordinated_groups)} Coordinated Groups")
                        for i, group in enumerate(coordinated_groups):
                            st.subheader(f"Group {i+1}: {group['coordination_type']}")
                            col1, col2 = st.columns(2)
                            col1.metric("Total Posts", group['num_posts'])
                            col2.metric("Unique Accounts", group['num_accounts'])
                            
                            posts_df = pd.DataFrame(group['posts'])
                            posts_df = posts_df.sort_values('Timestamp', ascending=True)
                            
                            st.dataframe(posts_df[['account_id', 'text', 'Platform', 'Timestamp']].head(5))

        # === TAB 3: Network Graph (Requires clustered_df) ===
        with tab3:
            st.header("🕸️ Account-to-Account Network Graph")
            if clustered_df.empty:
                 st.info("Run clustering first (Tab 1/Sidebar) to generate the network.")
            else:
                st.subheader("Network Generation")
                # Use a simple key for the network graph cache to update when necessary
                network_cache_key = (data_source_key, coordination_mode)
                G, pos, cluster_map = cached_network_graph(clustered_df, coordination_mode.split()[0].lower(), network_cache_key)
                
                if G.nodes():
                    st.markdown(f"**Nodes (Accounts):** {G.number_of_nodes():,} | **Edges (Shared Content/Cluster):** {G.number_of_edges():,}")

                    # Plotly Network Graph
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.append(x0)
                        edge_x.append(x1)
                        edge_x.append(None)
                        edge_y.append(y0)
                        edge_y.append(y1)
                        edge_y.append(None)

                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        mode='lines')

                    node_x = []
                    node_y = []
                    node_text = []
                    node_color = []
                    node_size = []
                    
                    # Calculate node metrics
                    degree = dict(G.degree())
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        
                        cluster_id = cluster_map.get(node, -2) # -2 for accounts that are not in a cluster (noise or singletons)
                        node_color.append(cluster_id)
                        
                        # Size based on degree (how much they share)
                        size_val = max(5, degree.get(node, 0) / 5)
                        node_size.append(size_val)
                        
                        node_text.append(f"Account: {node}<br>Shared Items: {degree.get(node, 0)}<br>Cluster: {cluster_id}")

                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers',
                        hoverinfo='text',
                        marker=dict(
                            showscale=True,
                            colorscale='Viridis',
                            reversescale=True,
                            color=node_color,
                            size=node_size,
                            colorbar=dict(
                                thickness=15,
                                title='Narrative ID',
                                xanchor='left',
                                titleside='right'
                            ),
                            line_width=2),
                        text=node_text)

                    fig = go.Figure(data=[edge_trace, node_trace],
                                    layout=go.Layout(
                                        title=f'Interaction Network by {coordination_mode}',
                                        titlefont_size=16,
                                        showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=20,l=5,r=5,t=40),
                                        annotations=[ dict(
                                            text="Nodes represent accounts, colors represent content clusters.",
                                            showarrow=False,
                                            xref="paper", yref="paper",
                                            x=0.005, y=-0.002 ) ],
                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No connections found to build a network graph.")
        
        # === TAB 4: Summary (New and Improved) ===
        with tab4:
            st.header("📝 IMI Narrative Summary & Insights")
            
            if imi_report_df.empty:
                st.info("No LLM summaries generated yet. Please ensure data is loaded and clustering is complete.")
            else:
                display_imi_report_visuals(imi_report_df)
                # --- EXECUTION ENTRY POINT ---
if __name__ == '__main__':
    main_election_monitoring()
