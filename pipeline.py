# -*- coding: utf-8 -*-
##pipeline.ipynb

# Package Loading
import openai
import streamlit as st
import numpy as np
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.collocations import*
from nltk import word_tokenize
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer, word_tokenize, TweetTokenizer
from nltk.tag import StanfordNERTagger
from nltk.stem import WordNetLemmatizer
from time import sleep
from nltk import download
from tenacity import retry, wait_exponential, stop_after_attempt
from fastdtw import fastdtw
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import Dataset, DataLoader
from matplotlib.dates import DateFormatter
# Download necessary NLTK datasets (if not already installed)
#download('punkt')
#download('stopwords')
#download('wordnet')
#download('punkt_tab')
import string
import nltk.corpus
from nltk import SnowballStemmer
from sklearn.metrics import silhouette_samples, silhouette_score
import re
from stop_words import get_stop_words
from wordcloud import WordCloud, STOPWORDS
import tweepy
import logging
from sklearn import preprocessing
from sklearn.cluster import KMeans
from numpy import argmax
from transformers import pipeline, AutoTokenizer, BertTokenizer, BartTokenizer, BartForConditionalGeneration, AutoModelForTokenClassification, BertModel
import networkx as nx
from datetime import timedelta
from sentence_transformers import SentenceTransformer, util
from bertopic import BERTopic
from transformers import MarianMTModel, MarianTokenizer, GPT2Tokenizer
import random
# Initialize the tokenizer to manage token limits
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import cv2
from skimage.metrics import structural_similarity as ssim
#from flair.data import Sentence
#from flair.models import SequenceTagger
import groq
import concurrent.futures
import time
import groq
from groq import Groq
import random
import multiprocessing
import os
import re
import groq
from multiprocessing import Manager, cpu_count, Pool
import torch
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import TextGeneration, OpenAI, KeyBERTInspired
import annoy
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Configuration (now using Streamlit secrets)
CONFIG = {
    "api_key":st.secrets["groq_api_key"], # REPLACED WITH SECRETS
    "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
    "gpu_params": {
        "batch_size": 512,  # Increased batch size
        "max_seq_length": 128,
        "num_workers": 4,
        "fp16": True  # Enable mixed precision
    },
    "bertrend": {
        "model_name": "bert-base-multilingual-cased",  # Explicitly set model name
        "temporal_weight": 0.3,
        "cluster_threshold": 0.2,  # Adjusted to a reasonable value
        "min_cluster_size": 4,
        "growth_threshold": 1.2,  # Adjusted to a reasonable value
        "pca_components": 64,  # Increased PCA components
        "chunk_size": 500,  # Increased chunk size
        "ann_neighbors": 50,  # Increased ANN neighbors
        "time_window_hours": 48  # Reduced time window
    },
    "analysis": {  # Corrected indentation here
        "time_window": "48H",  # Reduced time window
        "min_sources": 2,
        "decay_factor": 0.01,  # Slightly slower decay
        "decay_power": 1.5,  # Adjusted decay power
        "visualization": {
            "plot_size": (16, 10),  # Larger plot size
            "palette": "viridis",
            "max_display_clusters": 20  # Increased max clusters
        }
    }
}

# Initialize Groq client with Streamlit secrets
def get_groq_client():
    return Groq(api_key=st.secrets.groq.api_key)

# Load BERT model to GPU
tokenizer = BertTokenizer.from_pretrained(CONFIG["bertrend"]["model_name"])
bert_model = BertModel.from_pretrained(CONFIG["bertrend"]["model_name"]).to(device)

# GPU-optimized Dataset with Pre-batching
class DRCDataset(Dataset):
    def __init__(self, texts):
        # Filter out invalid or empty entries
        self.texts = [str(t).strip() for t in texts if isinstance(t, (str, bytes)) and len(str(t).strip()) > 0]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx >= len(self):
            return ""
        return self.texts[idx]

    def collate_fn(self, batch):
        # Filter empty strings and None values
        batch = [text for text in batch if isinstance(text, (str, bytes)) and len(text) > 0]
        # Handle empty batches
        if not batch:
            return {"input_ids": torch.zeros((0, CONFIG["gpu_params"]["max_seq_length"]), dtype=torch.long),
                    "attention_mask": torch.zeros((0, CONFIG["gpu_params"]["max_seq_length"]), dtype=torch.long)}
        # Tokenize with error handling
        try:
            return tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=CONFIG["gpu_params"]["max_seq_length"],
                return_attention_mask=True
            )
        except Exception as e:
            logger.error(f"Tokenizer error: {str(e)}")
            return {"input_ids": torch.zeros((len(batch), CONFIG["gpu_params"]["max_seq_length"]), dtype=torch.long),
                    "attention_mask": torch.zeros((len(batch), CONFIG["gpu_params"]["max_seq_length"]), dtype=torch.long)}

# Turbo-charged BERT Embeddings Generator
def get_bert_embeddings(texts):
    dataset = DRCDataset(texts)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["gpu_params"]["batch_size"],
        num_workers=0,  # Disable multiprocessing for stability
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        shuffle=False
    )
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            # Skip empty batches
            if batch["input_ids"].shape[0] == 0:
                continue
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = bert_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu())
            torch.cuda.empty_cache()  # Free GPU memory
    full_embeddings = torch.cat(embeddings)
    pca = PCA(n_components=CONFIG["bertrend"]["pca_components"])
    return pca.fit_transform(full_embeddings.cpu().numpy())

# GPU-accelerated Temporal-Spatial Distance Calculator
def temporal_distance_matrix(embeddings, timestamps):
    """Hybrid distance calculation with temporal constraints"""
    emb_tensor = torch.tensor(embeddings, device=device)
    time_tensor = torch.tensor(timestamps, dtype=torch.float64, device=device)
    time_diff = torch.abs(time_tensor[:, None] - time_tensor[None, :]) / 3.6e9
    time_mask = (time_diff < CONFIG["bertrend"]["time_window_hours"]).float()
    semantic_dists = torch.cdist(emb_tensor, emb_tensor, p=2)
    combined_dists = (
        CONFIG["bertrend"]["temporal_weight"] * time_diff +
        (1 - CONFIG["bertrend"]["temporal_weight"]) * semantic_dists
    ) * time_mask
    return combined_dists.cpu().numpy()

# Hyper-optimized BERTrend Analysis
def bertrend_analysis(df):
    """GPU-powered clustering pipeline with temporal constraints"""
    st.write("Input DataFrame to bertrend_analysis:")
    st.write(df)
    st.write(f"Input DataFrame shape: {df.shape}")
    st.write(f"Input DataFrame is empty: {df.empty}")
    try:
        logger.info("Generating turbo-charged BERT embeddings...")
        embeddings = get_bert_embeddings(df['text'].tolist())
        timestamps = df['Timestamp'].astype(np.int64).values
        ann_index = annoy.AnnoyIndex(embeddings.shape[1], 'euclidean')
        for i, emb in enumerate(embeddings):
            ann_index.add_item(i, emb)
        ann_index.build(20)  # More trees for better accuracy
        clusters = np.full(len(df), -1, dtype=int)
        current_cluster = 0
        chunk_size = CONFIG["bertrend"]["chunk_size"]
        for i in range(0, len(embeddings), chunk_size):
            chunk_end = min(i + chunk_size, len(embeddings))
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(embeddings)//chunk_size)+1}")
            chunk_times = timestamps[i:chunk_end]
            time_mask = (timestamps >= chunk_times[0]) & (timestamps <= chunk_times[-1])
            neighbor_candidates = set()
            for idx in range(i, chunk_end):
                neighbors = ann_index.get_nns_by_item(
                    idx,
                    CONFIG["bertrend"]["ann_neighbors"],
                    search_k=100000  # Higher search effort
                )
                neighbor_candidates.update(neighbors)
            candidate_indices = np.array(list(neighbor_candidates))
            candidate_indices = candidate_indices[time_mask[candidate_indices]]
            if len(candidate_indices) == 0:
                continue
            sub_emb = embeddings[candidate_indices]
            sub_ts = timestamps[candidate_indices]
            dist_matrix = temporal_distance_matrix(sub_emb, sub_ts)
            dist_matrix = dist_matrix.astype(np.double)
            clusterer = HDBSCAN(
                min_cluster_size=CONFIG["bertrend"]["min_cluster_size"],
                metric="precomputed",
                cluster_selection_epsilon=CONFIG["bertrend"]["cluster_threshold"],
                core_dist_n_jobs=4
            )
            chunk_clusters = clusterer.fit_predict(dist_matrix)
            valid_mask = chunk_clusters != -1
            chunk_clusters[valid_mask] += current_cluster
            clusters[candidate_indices] = chunk_clusters
            current_cluster = chunk_clusters[valid_mask].max() + 1 if valid_mask.any() else current_cluster
        df['Cluster'] = clusters
        df = df[df['Cluster'] != -1]
        st.write("DataFrame after clustering:")
        st.write(df)
        st.write(f"DataFrame after clustering shape: {df.shape}")
        st.write(f"DataFrame after clustering is empty: {df.empty}")
        if df.empty:
            logger.error("No valid clusters found after clustering.")
            st.error("❌ No valid clusters found after clustering.")
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.error(f"Error in bertrend_analysis: {e}")
        st.error(f"Error in bertrend_analysis: {e}")
        return pd.DataFrame()

# Optimized Momentum Calculator
def calculate_trend_momentum(clustered_df):
    """Optimized momentum calculation with source tracking"""
    df = clustered_df.copy()
    df['time_window'] = df['Timestamp'].dt.floor(CONFIG["analysis"]["time_window"])
    grouped = df.groupby(['Cluster', 'time_window']).agg(
        count=('text', 'size'),
        sources=('Source', 'unique'),
        last_time=('Timestamp', 'max')
    ).reset_index()
    emerging = []
    momentum_states = {}
    for cluster, cluster_group in grouped.groupby('Cluster'):
        if cluster == -1:
            continue
        cluster_group = cluster_group.sort_values('time_window')
        cumulative_sources = set()
        momentum = 0
        last_update = None
        for idx, row in cluster_group.iterrows():
            if last_update is not None:
                delta_hours = (row['last_time'] - last_update).total_seconds() / 3600
                decay = np.exp(-CONFIG["analysis"]["decay_factor"] *
                             (delta_hours ** CONFIG["analysis"]["decay_power"]))
                momentum *= decay
            momentum += row['count']
            cumulative_sources.update(row['sources'])
            momentum_score = momentum * len(cumulative_sources) * np.log1p(row['count'])
            if (momentum_score > CONFIG["bertrend"]["growth_threshold"] and
                len(cumulative_sources) >= CONFIG["analysis"]["min_sources"]):
                emerging.append((cluster, momentum_score))
            last_update = row['last_time']
        momentum_states[cluster] = {
            'momentum': momentum,
            'last_update': last_update,
            'sources': cumulative_sources
        }
    return sorted(emerging, key=lambda x: -x[1]), momentum_states

# Visualizations
def visualize_trends(clustered_df, momentum_states):
    """Generate interactive trend visualizations"""
    if clustered_df is None or clustered_df.empty:
        st.error("❌ No data available for visualization.")
        return None
    st.write("DataFrame before pivot_table:")
    st.write(clustered_df)
    st.write(f"DataFrame shape: {clustered_df.shape}")
    st.write(f"Is DataFrame empty? {clustered_df.empty}")
    plt.figure(figsize=CONFIG["analysis"]["visualization"]["plot_size"])
    plt.subplot(2, 1, 1)
    for cluster in list(momentum_states.keys())[:CONFIG["analysis"]["visualization"]["max_display_clusters"]]:
        cluster_data = clustered_df[clustered_df['Cluster'] == cluster]
        timeline = cluster_data.groupby(pd.Grouper(key='Timestamp', freq='6H'))['text'].count().cumsum()
        plt.plot(timeline.index, timeline, label=f"Cluster {cluster}", lw=2, alpha=0.8)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
    plt.title("Narrative Momentum Timeline")
    plt.ylabel("Cumulative Momentum")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 1, 2)
    try:
        heatmap_data = clustered_df.pivot_table(
            index=pd.Grouper(key='Timestamp', freq='6H'),
            columns='Cluster',
            values='text',
            aggfunc='count',
            fill_value=0
        ).iloc[:, :CONFIG["analysis"]["visualization"]["max_display_clusters"]]
        st.write("Heatmap Data:")
        st.write(heatmap_data)
        st.write(f"Heatmap Data Shape: {heatmap_data.shape}")
        st.write(f"Heatmap Data Contains NaNs? {heatmap_data.isnull().values.any()}")
        if heatmap_data.empty or heatmap_data.isnull().all().all():
            st.error("❌ Heatmap data is empty or contains only NaN values.")
            return None
        sns.heatmap(
            heatmap_data.T,
            cmap=CONFIG["analysis"]["visualization"]["palette"],
            cbar_kws={'label': 'Activity Level'}
        )
        plt.title("Cluster Activity Patterns")
        plt.xlabel("Time Windows")
        plt.ylabel("Cluster ID")
    except Exception as e:
        st.error(f"❌ Error generating heatmap: {e}")
        return None
    plt.tight_layout()
    plt.savefig("trend_visualization.png", bbox_inches='tight')
    st.pyplot(plt)
    plt.close()
    st.success("✅ Visualizations Generated")
    return "trend_visualization.png"

# Report Generation
# Report Generation
def generate_investigative_report(cluster_data, momentum_states, cluster_id, max_tokens=1024):
    """Generate report with top 3 documents and their URLs"""
    client = Groq(api_key=st.secrets["groq_api_key"]) # REPLACED WITH SECRETS
    try:
        metrics = momentum_states.get(cluster_id, {})
        sample_docs = cluster_data[['text', 'URL', 'Timestamp']].values.tolist()
        random.shuffle(sample_docs)
        Country = "Gabon"
        selected_docs = []
        total_tokens = 0
        for doc in sample_docs:
            doc_tokens = len(tokenizer.encode(doc[0]))
            if total_tokens + doc_tokens <= max_tokens:
                selected_docs.append(doc)
                total_tokens += doc_tokens
            else:
                break
        response = client.chat.completions.create(
            model=CONFIG["model_id"],
            messages=[{
                "role": "system",
                "content": f"""
                Generate {Country} structured Foreign/domestic Information Manipulation and Interference (FIMI) intelligence report related to the upcoming presidential elections:
                - Provide general context and identify key narratives with reference documents and URLs as evidence.
                - Map these narratives lifecycle: First Detected {cluster_data['Timestamp'].min().strftime('%Y-%m-%d %H:%M')} → Last Updated {cluster_data['Timestamp'].max().strftime('%Y-%m-%d %H:%M')}.
                - Identify narratives with mentions of anti-West, anti-France, pro/anti-ECOWAS, pro/anti-AES, pro-Russia, or pro-China sentiment.
                - Highlight toxic incitement, trigger lexicons, and provide URLs as evidence.
                - Identify coordinated networks, crossposting clusters, reused media, and instances of repeated content.
                - Suggest 2-3 investigative leads based on documented evidence.
                Exclude speculation, history, or general claims. Reference only exact documented evidence with URLs.
                """
            }, {
                "role": "user",
                "content": "\n".join([f"Document {i+1}: {doc[0]}\nURL: {doc[1]}\n[TIMESTAMP]: {doc[2]}" for i, doc in enumerate(selected_docs)])
            }],
            temperature=0.6,
            max_tokens=800
        )
        return {
            "report": response.choices[0].message.content,
            "metrics": metrics,
            "sample_texts": [doc[0] for doc in selected_docs],
            "sample_urls": [doc[1] for doc in selected_docs],
            "Time": [doc[2] for doc in selected_docs],
            "all_urls": cluster_data['URL'].head(20).tolist(),
            "source_count": cluster_data['Source'].nunique(),
            "momentum_score": cluster_data['momentum_score'].iloc[0]
        }
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return {"error": str(e)}

# Threat Categorization
def categorize_momentum(score):
    score = float(score)
    if score <= 150:
        return 'Tier 1: Ambient Noise (Normal baseline activity)'
    elif score <= 500:
        return 'Tier 2: Emerging Narrative (Potential story development)'
    elif score <= 2000:
        return 'Tier 3: Coordinated Activity (Organized group behavior)'
    else:
        return 'Tier 4: Viral Emergency (Requires immediate response)'
                
