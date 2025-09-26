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
    # Attempt to load Groq API key (assuming it is configured in .streamlit/secrets.toml)
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

# --- IMI Summary Generator (FIXED with FULL PROMPT) ---
def summarize_cluster(texts, urls, cluster_data):
    # Truncate to top 50 posts to reduce token load
    joined = "\n".join(texts[:50])
    url_context = "\nRelevant post links:\n" + "\n".join(urls[:5]) if urls else ""

    # Handle timestamps
    if cluster_data['timestamp_share'].dtype != 'datetime64[ns]':
        cluster_data['Timestamp'] = pd.to_datetime(cluster_data['timestamp_share'], unit='s', errors='coerce')
    else:
        cluster_data['Timestamp'] = cluster_data['timestamp_share']
        
    cluster_data = cluster_data.dropna(subset=['Timestamp']) 
            
    if cluster_data.empty:
        return "Summary generation failed: No valid timestamps.", [], "Failed to Summarize"

    min_timestamp_str = cluster_data['Timestamp'].min().strftime('%Y-%m-%d %H:%M')
    max_timestamp_str = cluster_data['Timestamp'].max().strftime('%Y-%m-%d %H:%M')

    # --- FULL, UN-CUT PROMPT INSERTION ---
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
    # --- END: FULL, UN-CUT PROMPT INSERTION ---

    response = safe_llm_call(prompt, max_tokens=2048)

    if response:
        raw_summary = response.choices[0].message.content.strip()
        
        # --- Title Extraction ---
        title_match = re.search(r'\*\*(.*?)\*\*', raw_summary, re.DOTALL)
        title = title_match.group(1).strip() if title_match else "Un-Titled Narrative"
        
        # Extract URLs from the summary (if any)
        evidence_urls = re.findall(r"(https?://[^\s\)\]]+)", raw_summary)

        # Clean up the summary (removing title and lifecycle added by the LLM)
        cleaned_summary = re.sub(r'\*\*(.*?)\*\*', '', raw_summary, count=1, flags=re.DOTALL).strip()
        cleaned_summary = re.sub(r'First Detected: .*?Last Updated: .*', '', cleaned_summary, flags=re.DOTALL).strip()
        
        return cleaned_summary, evidence_urls, title
    else:
        logger.error(f"LLM call failed for cluster {cluster_data['cluster'].iloc[0] if 'cluster' in cluster_data.columns else 'N/A'}")
        return "Summary generation failed.", [], "Failed to Summarize"

# --- Generate IMI Report (Unchanged) ---
@st.cache_data(show_spinner="Generating Narrative Summaries...")
def generate_imi_report(clustered_df, data_source_key):
    """Generates the IMI report (summarization) from clustered data."""
    if clustered_df.empty or 'cluster' not in clustered_df.columns:
        return pd.DataFrame()

    report_data = []
    clustered_df['Timestamp'] = pd.to_datetime(clustered_df['timestamp_share'], unit='s', errors='coerce')
    clustered_df = clustered_df.dropna(subset=['Timestamp'])

    unique_clusters = clustered_df[clustered_df['cluster'] != -1]['cluster'].unique()

    for i, cluster_id in enumerate(unique_clusters[:5]):
        cluster_data = clustered_df[clustered_df['cluster'] == cluster_id]
        
        texts = cluster_data['original_text'].tolist()
        urls = cluster_data['URL'].tolist()
        
        # Call the LLM with the full prompt
        summary, evidence_urls, title = summarize_cluster(texts, urls, cluster_data)

        min_ts = cluster_data['Timestamp'].min()
        max_ts = cluster_data['Timestamp'].max()
        posts_count = len(cluster_data)
        accounts_count = cluster_data['account_id'].nunique()
        platforms_list = cluster_data['Platform'].unique().tolist()
        datasets_list = cluster_data['source_dataset'].unique().tolist()
        
        # Simple heuristic for Virality
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

# --- Coordination Detection (Unchanged) ---
@st.cache_data(show_spinner="Detecting Coordinated Groups...")
def find_coordinated_groups(df, threshold, max_features):
    # This function remains as previously defined
    text_col = 'original_text'
    social_media_platforms = {'TikTok', 'Facebook', 'X', 'YouTube', 'Instagram', 'Telegram'}
    coordination_groups = {}
    clustered_groups = df[df['cluster'] != -1].groupby('cluster')
    
    # ... [Implementation remains the same] ...
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

# --- Network Graph Logic (Unchanged) ---
@st.cache_data(show_spinner="Building Network Graph...")
def cached_network_graph(df, coordination_type, data_source_key):
    # This function remains as previously defined
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

# --- NETWORK VISUALIZATION FUNCTION (Unchanged) ---
def plot_network_graph(G, pos, coordination_mode):
    # This function remains as previously defined
    if not G.nodes():
        st.warning("No connections detected to build the network graph.")
        return

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Connections'
    )

    node_x, node_y, node_text = [], [], []
    node_influence, node_amplification = [], []

    raw_influence = [G.nodes[node].get('influence', 0) for node in G.nodes()]
    max_raw_influence = max(raw_influence) if raw_influence else 1
    base_size = 10 
    max_size = 40
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        influence = G.nodes[node].get('influence', 0)
        amplification = G.nodes[node].get('amplification', 0)
        
        scaled_size = base_size + (max_size - base_size) * (influence / max_raw_influence if max_raw_influence > 0 else 0)
        node_influence.append(scaled_size)
        node_amplification.append(amplification)

        hover_text = (
            f"**Account:** {node}<br>"
            f"**Key Influencer Score (Size):** {influence:.4f}<br>"
            f"**Loud Amplifier Score (Color):** {amplification:.4f}<br>"
            f"Total Posts: {G.nodes[node].get('total_posts', 0)}"
        )
        node_text.append(hover_text)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hovertext=node_text,
        hoverinfo='text',
        name='Network Nodes',
        marker=dict(
            colorscale='YlOrRd', 
            color=node_amplification, 
            showscale=True,
            colorbar=dict(
                title=dict(text='Loud Amplifier Score', side='right'),
                thickness=20,
                len=0.7,
                outlinewidth=1
            ),
            size=node_influence, 
            line=dict(width=1, color='DarkSlateGrey')
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'Account Network: Key Influencers (Size) & Loud Amplifiers (Color)<br><sup>Connections based on Shared {coordination_mode.split()[0]}</sup>',
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


# --- Unified Display Function for IMI Report (Unchanged) ---
def display_imi_report_visuals(report_df):
    # This function remains as previously defined
    st.subheader("Interactive Narrative Reports")

    def virality_badge(val):
        s = str(val).lower()
        if "tier 4" in s or "emergency" in s:
            return '<span style="background-color: #ffebee; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #c62828;">🚨 Tier 4: Viral Emergency</span>'
        elif "tier 3" in s or "high" in s:
            return '<span style="background-color: #fff3e0; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #e65100;">🔥 Tier 3: High Virality</span>'
        elif "tier 2" in s or "medium" in s:
            return '<span style="background-color: #e8f5e9; padding: 4px 8px; border-radius: 6px; font-weight: bold; color: #2e7d32;">📈 Tier 2: Medium Virality</span>'
        else:
            return '<span style="background-color: #f5f5f5; padding: 4px 8px; border-radius: 6px; color: #555;">ℹ️ Tier 1: Low/Unknown</span>'
            
    def parse_urls_column(val):
        if pd.isna(val) or val is None: return []
        if isinstance(val, list): return val
        if isinstance(val, str):
            try: 
                parsed = eval(val)
                if isinstance(parsed, list):
                    return [u.strip() for u in parsed if isinstance(u, str) and u.startswith("http")]
            except: 
                return re.findall(r'https?://[^\s\)\]\,\"\'\<\>]+', val)
        return []

    standard_df = pd.DataFrame()
    standard_df['Title'] = report_df.get('Title', report_df.get('ID', report_df.get('Context', pd.Series())).astype(str).str.split('\n').str[0].str[:70])
    standard_df['Context'] = report_df.get('Context', "")
    standard_df['Emerging Virality'] = report_df.get('Emerging Virality', "Tier 1: Low")
    standard_df['Posts'] = report_df.get('Posts', report_df.get('Post Count', 0)).fillna(0).astype(int)
    standard_df['Accounts'] = report_df.get('Accounts', report_df.get('Account Count', 0)).fillna(0).astype(int)
    standard_df['Platforms'] = report_df.get('Platforms', report_df.get('Platform Distribution', "N/A"))
    standard_df['Source Datasets'] = report_df.get('Source Datasets', report_df.get('Source Dataset', "N/A"))
    
    standard_df['First Detected'] = pd.to_datetime(report_df.get('First Detected', pd.NaT), errors='coerce')
    standard_df['Last Updated'] = pd.to_datetime(report_df.get('Last Updated', pd.NaT), errors='coerce')

    if 'URLs' in report_df.columns:
        standard_df['All_URLs'] = report_df['URLs'].apply(parse_urls_column)
    elif 'All_URLs' in report_df.columns:
        standard_df['All_URLs'] = report_df['All_URLs'].apply(parse_urls_column)
    else:
        standard_df['All_URLs'] = [[]] * len(standard_df)


    # --- Display using Expanders ---
    for index, row in standard_df.iterrows():
        title = row['Title'].strip()
        post_count = row['Posts']
        account_count = row['Accounts']
        platforms = row['Platforms']
        
        first_detected = row['First Detected'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['First Detected']) else 'N/A'
        last_updated = row['Last Updated'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['Last Updated']) else 'N/A'
        
        header_title = f"**{title}** | Posts: **{post_count:,}** | Accounts: **{account_count:,}**"
        
        with st.expander(header_title):
            st.markdown("### 📰 Narrative Context & Metrics")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("**Virality Level**")
                st.markdown(virality_badge(row['Emerging Virality']), unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Platforms:** `{platforms}`")
                st.markdown(f"**Source Datasets:** `{row['Source Datasets']}`")
                st.markdown(f"**Lifecycle:** First Detected: `{first_detected}` | Last Updated: `{last_updated}`")
            
            st.markdown("---")
            
            st.markdown("### 📄 Intelligence Report Summary")
            st.markdown(row['Context'])
            
            urls = row['All_URLs']
            if urls:
                st.markdown("### 🔗 Key Source Links (Evidence)")
                for i, url in enumerate(urls[:5]):
                    st.markdown(f"- [[Link {i+1}]]({url})")
                if len(urls) > 5:
                    st.markdown(f"... and {len(urls) - 5} more source links.")

# --- Tab Functions (Unchanged) ---
def tab1_summary_statistics(df, filtered_df_global, data_source, coordination_mode):
    # This function remains as previously defined (Your requested Tab 1 content)
    st.subheader("📌 Summary Statistics")
    st.markdown("### 🔬 Preprocessed Data Sample")
    st.markdown(f"**Data Source:** `{data_source}` | **Coordination Mode:** `{coordination_mode}` | **Total Rows:** `{len(df):,}`")
    
    display_cols_overview = ['account_id', 'content_id', 'object_id', 'timestamp_share']
    existing_cols = [col for col in df.columns if col in display_cols_overview]
    
    if not df.empty and existing_cols:
        st.dataframe(df[existing_cols].head(10))
    else:
        st.info("No data available to display in this tab.")

    if filtered_df_global.empty:
        return

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

    if 'Outlet' in filtered_df_global.columns and filtered_df_global['Outlet'].notna().any():
        top_outlets = filtered_df_global['Outlet'].value_counts().head(10)
        fig_outlet = px.bar(top_outlets, title="Top 10 Media Outlets/Channels", labels={'value': 'Posts', 'index': 'Outlet'})
        st.plotly_chart(fig_outlet, use_container_width=True)
        st.markdown("**Top 10 Media Outlets/Channels**: Ranks traditional and digital media sources by volume of coverage.")
    elif 'Channel' in filtered_df_global.columns and filtered_df_global['Channel'].notna().any():
        top_channels = filtered_df_global['Channel'].value_counts().head(10)
        fig_chan = px.bar(top_channels, title="Top 10 Channels", labels={'value': 'Posts', 'index': 'Channel'})
        st.plotly_chart(fig_chan, use_container_width=True)
        st.markdown("**Top 10 Channels**: Displays the most active YouTube or social media channels.")

    social_media_df = filtered_df_global[~filtered_df_global['Platform'].isin(['Media', 'News/Media'])].copy()
    if not social_media_df.empty and 'object_id' in social_media_df.columns:
        social_media_df['hashtags'] = social_media_df['object_id'].astype(str).str.findall(r'#\w+').apply(lambda x: [tag.lower() for tag in x])
        all_hashtags = [tag for tags_list in social_media_df['hashtags'] if isinstance(tags_list, list) for tag in tags_list]
        if all_hashtags:
            hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
            fig_ht = px.bar(hashtag_counts, title="Top 10 Hashtags (Social Media Only)", labels={'value': 'Frequency', 'index': 'Hashtag'})
            st.plotly_chart(fig_ht, use_container_width=True)
            st.markdown("**Top 10 Hashtags (Social Media Only)**: Highlights the most frequently used hashtags on social platforms.")

    plot_df = filtered_df_global.copy()
    if 'timestamp_share' not in plot_df.columns:
        st.warning("⚠️ 'timestamp_share' column not found. Cannot plot time series.")
    else:
        plot_df['timestamp_share'] = pd.to_numeric(plot_df['timestamp_share'], errors='coerce')
        valid_mask = (plot_df['timestamp_share'] >= 946684800) & (plot_df['timestamp_share'] <= 4102444800)
        plot_df = plot_df[valid_mask]
        
        if plot_df.empty:
            st.info("No valid timestamps available for time series.")
        else:
            plot_df['datetime'] = pd.to_datetime(plot_df['timestamp_share'], unit='s', utc=True)
            plot_df = plot_df.set_index('datetime')
            time_series = plot_df.resample('D').size()
            
            if time_series.empty:
                st.info("No data to display in time series.")
            else:
                fig_ts = px.area(time_series,
                                 title="Daily Post Volume",
                                 labels={'value': 'Number of Posts', 'datetime': 'Date'},
                                 markers=True)
                fig_ts.update_layout(xaxis_title="Date", yaxis_title="Number of Posts")
                st.plotly_chart(fig_ts, use_container_width=True)
                st.markdown("**Daily Post Volume**: Visualizes the volume of posts over time to identify spikes or trends.")

def tab2_narrative_intelligence(report_df):
    """Renders the Narrative Intelligence (IMI Report) tab (Your requested Tab 4 content)."""
    st.header("🧠 Narrative Summary Report")
    st.markdown("**Focus:** Summaries of clustered, high-similarity content generated by the LLM, presented in a non-technical, intelligence-ready format.")
    
    if report_df.empty:
        st.info("No narrative reports available. Please ensure data is uploaded and narratives have been generated.")
    else:
        display_imi_report_visuals(report_df)

def tab3_coordination_detection(df, coordination_groups):
    """Renders the Coordination Detection tab (Your requested Tab 2 content)."""
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
            'Group ID': i + 1,
            'Type': group['coordination_type'],
            'Posts': group['num_posts'],
            'Accounts': group['num_accounts'],
            'Max Sim. Score': group['max_similarity_score'],
            'Top Platform': pd.DataFrame(group['posts'])['Platform'].mode()[0] if not pd.DataFrame(group['posts']).empty else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values(by=['Posts', 'Accounts'], ascending=False).reset_index(drop=True)
    st.markdown("### Group Summary Table")
    st.dataframe(summary_df)
    
    st.markdown("---")
    st.markdown("### Detailed Group Analysis")
    
    for i, group in enumerate(coordination_groups[:10]):
        with st.expander(f"Group {i+1}: {group['coordination_type']} | Posts: {group['num_posts']} | Accounts: {group['num_accounts']}"):
            st.markdown(f"**Max Similarity Score:** `{group['max_similarity_score']}`")
            st.markdown(f"**Coordination Type:** `{group['coordination_type']}`")
            
            posts_df = pd.DataFrame(group['posts'])
            posts_df = posts_df.sort_values('Timestamp').reset_index(drop=True)
            posts_df['Timestamp'] = pd.to_datetime(posts_df['Timestamp'], unit='s', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

            st.markdown("**Coordinated Posts (First 10)**")
            st.dataframe(posts_df[['Timestamp', 'account_id', 'Platform', 'text']].head(10))
            
            st.markdown("**Distribution**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("Platforms:")
                st.dataframe(posts_df['Platform'].value_counts())
            with col2:
                st.markdown("Top Accounts:")
                st.dataframe(posts_df['account_id'].value_counts().head(5))

def tab4_network_analysis(df, coordination_mode):
    """Renders the Network Analysis tab (Your requested Tab 3 content)."""
    st.header("🕸️ Account Network Analysis")
    st.markdown(f"**Focus:** Visualizing the connections between accounts based on shared **{coordination_mode.lower()}** (text similarity or shared URLs).")

    if df.empty:
        st.info("Please upload and process data to generate the network graph.")
        return

    network_type = "text" if coordination_mode == "Text Content" else "url"
    data_source_key = st.session_state.get('data_source', 'default_data') + str(len(df))

    G, pos = cached_network_graph(df, network_type, data_source_key)

    if not G.nodes():
        st.warning("The network graph could not be generated. Check if content similarity clustering was performed and accounts are sharing the selected content type.")
    else:
        plot_network_graph(G, pos, coordination_mode)
        
        st.markdown("---")
        st.markdown("### Key Metrics Explained")
        st.markdown(
            """
            * **Node Size (Key Influencer Score):** Based on **Betweenness Centrality**. Nodes with larger sizes act as **bridges** connecting different clusters or groups.
            * **Node Color (Loud Amplifier Score):** Based on **Eigenvector Centrality**. Nodes with brighter/darker colors (higher score) are connected to other **well-connected** accounts.
            """
        )

# --- General Helper Functions (Data Processing - Unchanged) ---
def infer_platform_from_url(url):
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"): return "Unknown"
    url = url.lower()
    if "tiktok.com" in url: return "TikTok"
    elif "facebook.com" in url or "fb.watch" in url: return "Facebook"
    elif "twitter.com" in url or "x.com" in url: return "X"
    elif "youtube.com" in url or "youtu.be" in url: return "YouTube"
    elif "instagram.com" in url: return "Instagram"
    elif "telegram.me" in url or "t.me" in url: return "Telegram"
    elif any(domain in url for domain in ["nytimes.com", "bbc.com", "cnn.com"]): return "News/Media"
    return "Media"

def extract_original_text(text):
    if pd.isna(text) or not isinstance(text, str): return ""
    cleaned = re.sub(r'^(RT|rt|QT|qt)\s+@\w+:\s*', '', text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'@\w+', '', cleaned).strip()
    cleaned = re.sub(r'http\S+|www\S+|https\S+', '', cleaned).strip()
    cleaned = re.sub(r"\\n|\\r|\\t", " ", cleaned).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.lower()

def final_preprocess_and_map_columns(df, coordination_mode="Text Content"):
    df_processed = df.copy()
    if df_processed.empty:
        cols = ['URL', 'Platform', 'original_text', 'Outlet', 'Channel', 'object_id', 'timestamp_share', 'cluster', 'source_dataset']
        for col in cols:
            if col not in df_processed.columns:
                df_processed[col] = pd.Series([], dtype='object' if col not in ['timestamp_share', 'cluster'] else 'Int64')
        return df_processed
    
    df_processed.rename(columns={'original_url': 'URL'}, inplace=True)
    df_processed['object_id'] = df_processed.get('object_id', df_processed.get('text', '')).astype(str).replace('nan', '').fillna('')

    if coordination_mode == "Text Content":
        df_processed['original_text'] = df_processed['object_id'].apply(extract_original_text)
    elif coordination_mode == "Shared URLs":
        df_processed['original_text'] = df_processed.get('URL', '').astype(str).replace('nan', '').fillna('')
    
    df_processed = df_processed[df_processed['original_text'].str.strip() != ""].reset_index(drop=True)
    df_processed['Platform'] = df_processed['URL'].apply(infer_platform_from_url)
    
    if 'cluster' not in df_processed.columns: df_processed['cluster'] = -1 
    if 'source_dataset' not in df_processed.columns: df_processed['source_dataset'] = 'Uploaded_File' 

    for col in ['Outlet', 'Channel']:
        if col not in df_processed.columns: df_processed[col] = np.nan
    
    return df_processed

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

# --- Main Streamlit App ---

def main():
    st.set_page_config(layout="wide", page_title="IMI Coordination Dashboard")
    st.title("🛡️ Information Manipulation Insight Dashboard")
    st.markdown("Upload raw social media data or a pre-processed dataset to begin analysis.")
    
    # --- Sidebar and Data Upload ---
    with st.sidebar:
        st.header("Upload & Settings")
        uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])
        
        coordination_mode = st.radio(
            "Coordination Mode (Clustering/Network Focus)",
            options=["Text Content", "Shared URLs"],
            index=0,
            help="Select the basis for identifying coordinated groups and building the account network."
        )
        
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            try:
                if uploaded_file.name.endswith('.csv'):
                    raw_df = pd.read_csv(uploaded_file)
                else: 
                    raw_df = pd.read_excel(uploaded_file)

                if st.button("Start Analysis"):
                    st.session_state['data_loaded'] = True
                    with st.spinner("Step 1: Preprocessing Data..."):
                        temp_df = raw_df.copy()
                        temp_df.columns = temp_df.columns.str.lower()
                        
                        standard_df = pd.DataFrame()
                        for target, default_source in zip(['account_id', 'content_id', 'object_id', 'timestamp_share', 'original_url'], temp_df.columns):
                             standard_df[target] = temp_df[default_source]
                        
                        df_preprocessed = final_preprocess_and_map_columns(standard_df, coordination_mode)
                        st.session_state['df'] = df_preprocessed.copy()
                    
                    with st.spinner("Step 2: Performing Clustering..."):
                        df_clustered = perform_clustering(df_preprocessed.copy(), coordination_mode)
                        st.session_state['filtered_df_global'] = df_clustered.copy()
                        
                    with st.spinner("Step 3: Generating Narrative Reports (LLM Call)..."):
                        report_df = generate_imi_report(df_clustered.copy(), uploaded_file.name)
                        st.session_state['report_df'] = report_df.copy()
                        
                    with st.spinner("Step 4: Detecting Coordination Groups..."):
                        coordination_groups = find_coordinated_groups(df_clustered.copy(), 
                                                                    CONFIG['coordination_detection']['threshold'], 
                                                                    CONFIG['coordination_detection']['max_features'])
                        st.session_state['coordination_groups'] = coordination_groups
                    
                    st.success("Analysis Complete!")
                    st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
                logger.error(f"File processing error: {e}")
                st.session_state['df'] = pd.DataFrame() 

        st.caption(f"LLM Model: {CONFIG['model_id']}")
        if client is None:
             st.warning("⚠️ Groq API key is missing. LLM summarization (Tab 4) will be unavailable.")

    # --- Main Content Tabs ---
    
    if 'df' not in st.session_state or st.session_state['df'].empty:
        st.info("Upload a dataset in the sidebar and click 'Start Analysis' to view the dashboard.")
        return

    # Retrieve data from session state
    df = st.session_state.get('df', pd.DataFrame())
    filtered_df_global = st.session_state.get('filtered_df_global', pd.DataFrame())
    report_df = st.session_state.get('report_df', pd.DataFrame())
    coordination_groups = st.session_state.get('coordination_groups', [])
    data_source = uploaded_file.name if uploaded_file else 'In-Session Data'

    # 4-Tab Layout - CORRECTED ORDER PER USER REQUEST
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Summary Statistics", 
        "🔗 Text Similarity & Coordination Analysis", 
        "🕸️ Network Analysis", 
        "🧠 Narrative Summary Report"
    ])

    with tab1:
        tab1_summary_statistics(df, filtered_df_global, data_source, coordination_mode)

    with tab2:
        tab3_coordination_detection(filtered_df_global, coordination_groups) # Your requested Tab 2 content

    with tab3:
        tab4_network_analysis(filtered_df_global, coordination_mode) # Your requested Tab 3 content

    with tab4:
        tab2_narrative_intelligence(report_df) # Your requested Tab 4 content

if __name__ == '__main__':
    if 'df' not in st.session_state: st.session_state['df'] = pd.DataFrame()
    if 'filtered_df_global' not in st.session_state: st.session_state['filtered_df_global'] = pd.DataFrame()
    if 'report_df' not in st.session_state: st.session_state['report_df'] = pd.DataFrame()
    if 'coordination_groups' not in st.session_state: st.session_state['coordination_groups'] = []
    
    main()
