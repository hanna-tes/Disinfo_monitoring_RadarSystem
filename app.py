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
    # Use st.secrets in a Streamlit app environment
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") 
    if GROQ_API_KEY:
        import groq
        client = groq.Groq(api_key=GROQ_API_KEY)
    else:
        logger.warning("GROQ_API_KEY not found in st.secrets.")
        client = None
except Exception as e:
    logger.warning(f"Error initializing Groq client: {e}")
    client = None

# --- Helper: Safe LLM call (no retry, just throttle) ---
def safe_llm_call(prompt, max_tokens=2048):
    """
    Call LLM safely without retry logic.
    """
    if client is None:
        logger.error("Groq client not initialized.")
        return None, "Error"
    try:
        response = client.chat.completions.create(
            model=CONFIG["model_id"],
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

# --- Helper Functions ---
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
    # Build the graph based on shared cluster content or shared URL
    if coordination_type == "text" and 'cluster' in df.columns:
        valid_clusters = df[df['cluster'] != -1].groupby('cluster')
        for cluster_id, cluster_df in valid_clusters:
            accounts = cluster_df['account_id'].unique().tolist()
            if len(accounts) > 1:
                for i in range(len(accounts)):
                    for j in range(i + 1, len(accounts)):
                        # Add edge, weight is the number of shared messages/clusters
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
    
    # Calculate Centrality Metrics for Node Visualization
    if G.nodes():
        # Influence (Key Player) -> Size
        try:
            influence_scores = nx.betweenness_centrality(G, weight='weight')
        except Exception:
            # Fallback for unconnected or small graphs
            influence_scores = {node: 0.1 for node in G.nodes()} 
        
        # Amplification (Loudness) -> Color
        try:
            amplification_scores = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        except Exception:
            amplification_scores = {node: 0.1 for node in G.nodes()}

        for node in G.nodes():
            G.nodes[node]['influence'] = influence_scores.get(node, 0)
            G.nodes[node]['amplification'] = amplification_scores.get(node, 0)
            G.nodes[node]['degree'] = G.degree(node)
            G.nodes[node]['total_posts'] = df[df['account_id'] == node].shape[0]

    pos = nx.spring_layout(G, seed=42) if G.nodes() else {}
    return G, pos

# --- NETWORK VISUALIZATION FUNCTION (FIXED & IMPROVED) ---
def plot_network_graph(G, pos, coordination_mode):
    if not G.nodes():
        st.warning("No connections detected to build the network graph.")
        return

    # --- 1. Prepare Nodes and Edges Data for Plotly ---
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Edges Trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Connections'
    )

    node_x = []
    node_y = []
    node_text = []
    node_influence = [] # Mapped to Size
    node_amplification = [] # Mapped to Color

    # Scale the influence score to a displayable size
    raw_influence = [G.nodes[node].get('influence', 0) for node in G.nodes()]
    max_raw_influence = max(raw_influence) if raw_influence else 1
    # Base size + Scaled influence
    base_size = 10 
    max_size = 40
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Calculate size and color metrics
        influence = G.nodes[node].get('influence', 0)
        amplification = G.nodes[node].get('amplification', 0)
        
        # Sizing (Key Influencer)
        scaled_size = base_size + (max_size - base_size) * (influence / max_raw_influence if max_raw_influence > 0 else 0)
        node_influence.append(scaled_size)
        
        # Coloring (Loud Amplifier)
        node_amplification.append(amplification)

        # Hover Text (Journalist-friendly)
        hover_text = (
            f"**Account:** {node}<br>"
            f"**Key Influencer Score (Size):** {influence:.4f}<br>"
            f"**Loud Amplifier Score (Color):** {amplification:.4f}<br>"
            f"Total Posts in Shared Content: {G.nodes[node].get('total_posts', 0)}"
        )
        node_text.append(hover_text)


    # Nodes Trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hovertext=node_text,
        hoverinfo='text',
        name='Network Nodes',
        marker=dict(
            # COLOR MAPPING: Measures Amplification (Loudness)
            colorscale='Hot',          
            color=node_amplification, 
            showscale=True,
            
            # --- FIXED COLORBAR CONFIGURATION ---
            colorbar=dict(
                title=dict(
                    text='Loudness / Amplification Score',
                    side='right'
                ),
                thickness=20,
                len=0.7,
                outlinewidth=1
            ),
            # ------------------------------------

            # SIZE MAPPING: Measures Influence (Key Player Status)
            size=node_influence,
            line=dict(width=1, color='DarkSlateGrey')
        )
    )

    # --- 2. Create the Figure ---
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'Account Network: Key Influencers (Size) & Loud Amplifiers (Color)<br><sup>Based on Shared {coordination_mode.split()[0]}</sup>',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- NEW HELPER: Hashtag Extraction ---
def extract_hashtags_from_text(text):
    """Extracts all hashtags from a text string."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    # Regex to find words starting with #
    return re.findall(r'#(\w+)', text.lower())

# --- NEW FUNCTION: Plot Top Hashtags (Incorporating the Fix) ---
@st.cache_data(show_spinner=False)
def plot_top_hashtags(df, data_source_key, top_n=10):
    if df.empty or 'original_text' not in df.columns:
        # Avoid showing warning in the sidebar when data is not yet loaded
        return None

    # 1. Extract all hashtags into a single list/Series
    all_hashtags = df['original_text'].apply(extract_hashtags_from_text).explode()
    
    # Drop NaNs/empty strings that might result from explode
    all_hashtags = all_hashtags.dropna()
    
    if all_hashtags.empty:
        return None

    # 2. Calculate the value counts and convert the Series to a DataFrame
    #    The original index (the hashtag text) becomes a new column.
    top_hashtags_df = all_hashtags.value_counts().head(top_n).reset_index()

    # 3. Rename the columns for clarity (CRITICAL STEP for plotting)
    top_hashtags_df.columns = ['Hashtag', 'Frequency']
    
    # 4. Create the Plotly Bar Chart, correctly mapping the 'Hashtag' column to the x-axis.
    fig_hashtags = px.bar(
        top_hashtags_df,
        x='Hashtag',         # <-- THE FIX: Uses the text column for labels
        y='Frequency',
        title=f'Top {top_n} Hashtags',
        color='Frequency',   # Color by frequency for visual depth
        color_continuous_scale=px.colors.sequential.Agsunset 
    )

    # 5. Improve readability by rotating the X-axis labels (optional, but good practice).
    fig_hashtags.update_layout(
        xaxis_tickangle=-45,
        xaxis_title="Hashtag",
        yaxis_title="Count (Frequency)",
        title_x=0.5
    )
    
    return fig_hashtags
    

# --- Reusable Display Function for IMI Report ---
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
        # === MODE 2: ENHANCED SUMMARY VIEW (Report Visualization Only) ===
        st.title("IMI Narrative Intelligence Dashboard 📰 (Report Viewer)")
        st.markdown("Upload a preprocessed report CSV file to visualize narrative intelligence.")
        
        uploaded_report_file = st.file_uploader("Upload IMI Report CSV", type="csv")
        
        if uploaded_report_file is not None:
            try:
                report_df = pd.read_csv(uploaded_report_file)
                st.success("Report loaded successfully.")
                display_imi_report_visuals(report_df)
                st.info("Raw data is required for Network Analysis and Top Hashtag charts.")

            except Exception as e:
                st.error(f"Error reading the uploaded CSV file: {e}")
                logger.error(f"Report file reading error: {e}")
        else:
            st.info("Please upload a CSV file to begin.")

    elif data_source == "Upload CSV Files":
        # === MODE 1: FULL WORKFLOW WITH RAW DATA (Election Monitoring Dashboard) ===
        st.title("Election Monitoring Dashboard: Full Analysis Workflow 🕸️")
        
        # --- File Uploads ---
        st.sidebar.subheader("Raw Data Files (CSV)")
        meltwater_file = st.sidebar.file_uploader("Upload Meltwater CSV", type="csv")
        civicsignals_file = st.sidebar.file_uploader("Upload CivicSignals CSV", type="csv")
        # openmeasure_file = st.sidebar.file_uploader("Upload OpenMeasure CSV (Optional)", type="csv")

        # --- Analysis Parameters ---
        st.sidebar.subheader("Analysis Parameters")
        coordination_mode = st.sidebar.selectbox("Coordination Based On", ["Text Content", "Shared URLs"])
        eps = st.sidebar.slider("DBSCAN Epsilon (Similarity Threshold)", 0.05, 0.9, 0.25)
        min_samples = st.sidebar.slider("DBSCAN Min Samples (Min Cluster Size)", 2, 20, 3)
        max_features = st.sidebar.slider("TF-IDF Max Features", 1000, 10000, 3000)
        
        if meltwater_file and civicsignals_file:
            st.sidebar.success("Files ready for processing.")
            
            # --- Data Loading and Combining ---
            @st.cache_data(show_spinner="Loading and Combining Data...")
            def load_and_combine_data(m_file, c_file, mode):
                meltwater_df = pd.read_csv(m_file)
                civicsignals_df = pd.read_csv(c_file)
                combined_df = combine_social_media_data(meltwater_df, civicsignals_df)
                return final_preprocess_and_map_columns(combined_df, mode)

            final_df = load_and_combine_data(meltwater_file, civicsignals_file, coordination_mode)
            data_source_key = f"{meltwater_file.name}-{civicsignals_file.name}-{coordination_mode}" # Cache Key

            if final_df.empty:
                st.error("The combined and preprocessed dataset is empty. Check your input files and column mappings.")
                return

            st.info(f"Total Unique Posts for Analysis: {len(final_df):,}")

            # --- Tabbed Interface for Results ---
            tab_summary, tab_clustering, tab_network, tab_report = st.tabs(["Summary Metrics", "Clustering & Coordination", "Network Graph", "IMI Report"])

            with tab_summary:
                st.header("1. Summary Metrics & Trends")
                
                # --- Hashtag Plotting (FIX IS APPLIED HERE) ---
                st.subheader("Top Hashtags 📊")
                hashtag_chart = plot_top_hashtags(final_df, data_source_key) 
                if hashtag_chart:
                    st.plotly_chart(hashtag_chart, use_container_width=True)
                else:
                    st.info("No hashtags found in the dataset for analysis.")
                    
                st.subheader("Platform Distribution")
                platform_counts = final_df['Platform'].value_counts().reset_index()
                platform_counts.columns = ['Platform', 'Count']
                fig_platform = px.bar(platform_counts, x='Platform', y='Count', title="Post Distribution by Platform")
                st.plotly_chart(fig_platform, use_container_width=True)
                
                st.subheader("Temporal Distribution")
                final_df['Timestamp_DT'] = pd.to_datetime(final_df['timestamp_share'], unit='s')
                time_series = final_df.set_index('Timestamp_DT').resample('H').size().reset_index(name='Count')
                fig_time = px.line(time_series, x='Timestamp_DT', y='Count', title="Posts Over Time (Hourly)")
                st.plotly_chart(fig_time, use_container_width=True)


            with tab_clustering:
                st.header("2. Narrative Clustering (DBSCAN) and Coordinated Groups")
                
                # --- Clustering ---
                clustered_df = cached_clustering(final_df, eps, min_samples, max_features, data_source_key)
                
                if 'cluster' in clustered_df.columns:
                    num_clusters = clustered_df['cluster'].max() + 1
                    noise_count = (clustered_df['cluster'] == -1).sum()
                    st.success(f"Clustering complete. Found **{num_clusters}** narratives and **{noise_count}** unclustered posts.")
                    
                    st.dataframe(clustered_df.sort_values(by=['cluster', 'timestamp_share']).head(20), use_container_width=True)
                    st.download_button(
                        label="Download Clustered Data as CSV",
                        data=convert_df_to_csv(clustered_df),
                        file_name=f'clustered_data_{num_clusters}_narratives.csv',
                        mime='text/csv',
                    )

                    # --- Coordination Detection ---
                    st.subheader("Coordinated Activity Detection")
                    group_data = find_coordinated_groups(clustered_df, threshold=0.9, max_features=max_features)
                    
                    st.markdown(f"Found **{len(group_data)}** potential coordinated groups based on shared content similarity (threshold 0.9, min 2 accounts).")
                    
                    if group_data:
                        for i, group in enumerate(group_data):
                            group_type = group['coordination_type']
                            num_posts = group['num_posts']
                            num_accounts = group['num_accounts']
                            max_sim = group['max_similarity_score']
                            
                            header = f"Group {i+1} | Type: **{group_type}** | Posts: {num_posts} | Accounts: {num_accounts} | Max Similarity: {max_sim}"
                            with st.expander(header):
                                group_df = pd.DataFrame(group['posts'])
                                st.dataframe(group_df[['account_id', 'Platform', 'text', 'Timestamp', 'URL']], use_container_width=True)
                    else:
                        st.info("No highly coordinated groups detected.")

            with tab_network:
                st.header("3. Network Graph Analysis")
                st.markdown("Visualizes connections between accounts based on shared clustered content.")
                
                if 'cluster' in locals() and num_clusters > 0:
                    coordination_type_network = st.selectbox(
                        "Network Edge Type:", 
                        ["Shared Content Cluster (Text)", "Shared URL"],
                        key='network_type_select'
                    )
                    
                    network_key = "text" if coordination_type_network.startswith("Shared Content") else "url"
                    
                    # Need clustered_df to be defined, which it is from tab_clustering
                    G, pos = cached_network_graph(clustered_df, network_key, data_source_key)
                    
                    if G.nodes():
                        st.info(f"Graph created with **{G.number_of_nodes()}** nodes and **{G.number_of_edges()}** edges.")
                        plot_network_graph(G, pos, coordination_type_network)
                    else:
                        st.warning("No connections found to build a network graph.")
                else:
                    st.warning("Please run Clustering first to generate the necessary data for network analysis.")


            with tab_report:
                st.header("4. IMI Intelligence Report Generation")
                st.warning("⚠️ LLM calls are expensive and time-consuming. Only the top 5 clusters will be summarized for demo purposes.")
                
                if 'clustered_df' in locals() and not clustered_df.empty:
                    if st.button("Generate IMI Report (LLM Summarization)", key='generate_report_btn'):
                        with st.spinner("Generating reports using LLM..."):
                            imi_report_df = generate_imi_report(clustered_df, data_source_key)
                        
                        if not imi_report_df.empty:
                            st.session_state['imi_report_df'] = imi_report_df
                            st.success(f"Successfully generated {len(imi_report_df)} narrative reports.")
                            
                        else:
                            st.error("Failed to generate any narrative reports.")
                
                if 'imi_report_df' in st.session_state and not st.session_state['imi_report_df'].empty:
                    display_imi_report_visuals(st.session_state['imi_report_df'])
                    
                    st.download_button(
                        label="Download IMI Report (CSV)",
                        data=convert_df_to_csv(st.session_state['imi_report_df']),
                        file_name='imi_narrative_intelligence_report.csv',
                        mime='text/csv',
                    )
                else:
                    st.info("Click the 'Generate IMI Report' button to begin summarization.")

        else:
            st.info("Please upload both Meltwater and CivicSignals CSV files to begin the full analysis workflow.")
            
# --- Run the App ---
if __name__ == "__main__":
    main_election_monitoring()
