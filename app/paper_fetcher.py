import arxiv
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Bio import Entrez
from lxml import etree
import asyncio
import aiohttp
import nest_asyncio
from typing import List, Dict

nest_asyncio.apply()

# Initialize model once
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

async def process_query(query: str, max_results: int = 15) -> List[Dict]:
    # Parallel fetching
    arxiv_papers = fetch_arxiv_papers(query, max_results)
    pubmed_task = fetch_pubmed_papers(query, max_results)
    semantic_task = fetch_semantic_scholar_papers(query, max_results//2)
    
    papers = await asyncio.gather(
        arxiv_papers, pubmed_task, semantic_task
    )
    
    # Flatten results
    combined = [paper for source in papers for paper in source]
    
    # Deduplication and ranking
    df = pd.DataFrame(combined)
    df = process_and_rank(df, query)
    
    return df.to_dict('records')

def process_and_rank(df: pd.DataFrame, query: str) -> pd.DataFrame:
    # Your existing processing logic
    df['text'] = df['Title'] + ". " + df['Abstract']
    texts = df['text'].tolist()
    
    # Batch processing
    embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    
    # Similarity calculation
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    df['Similarity'] = np.round(similarities, 2)
    
    # Filtering and sorting
    df = df[df['Similarity'] > 0.3]
    df = df.sort_values('Similarity', ascending=False).head(10)
    
    return df[['Title', 'Abstract', 'URL', 'Source', 'Authors', 'Similarity']]

# Keep your existing fetch_arxiv_papers, fetch_pubmed_papers, 
# and fetch_semantic_scholar_papers functions here
# (Remove Jupyter-specific print statements)