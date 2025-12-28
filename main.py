import pandas as pd
import numpy as np
from google_play_scraper import reviews, Sort
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime, timedelta
import openai
import os
import time
import json

APP_ID = 'in.swiggy.android'
START_DATE = datetime(2024, 6, 1).date()
TODAY = datetime.now().date()
OPENAI_API_KEY = "YOUR_API_KEY_HERE"

client = openai.OpenAI(api_key=OPENAI_API_KEY)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def fetch_historical_reviews(app_id, target_start_date):
    print(f"Fetching reviews for {app_id} back to {target_start_date}...")
    all_reviews = []
    continuation_token = None
    
    while True:
        result, continuation_token = reviews(
            app_id,
            lang='en',
            country='in',
            sort=Sort.NEWEST,
            count=1000,
            continuation_token=continuation_token
        )
        
        batch_df = pd.DataFrame(result)
        if batch_df.empty:
            break
            
        batch_df['at'] = pd.to_datetime(batch_df['at']).dt.date
        min_date = batch_df['at'].min()
        
        all_reviews.extend(result)
        print(f"Fetched {len(result)} reviews. Oldest: {min_date}")
        
        if min_date < target_start_date:
            break
            
        time.sleep(1)

    df = pd.DataFrame(all_reviews)
    df['date'] = pd.to_datetime(df['at']).dt.date
    df = df[df['date'] >= target_start_date]
    print(f"Total relevant reviews: {len(df)}")
    return df

class TaxonomyAgent:
    def __init__(self):
        self.taxonomy = [
            "Delivery issue", 
            "App crash/bugs", 
            "Food quality", 
            "Refund/Payment issue"
        ]
        
    def refine_topics(self, daily_review_clusters):
        if not daily_review_clusters:
            return {}

        cluster_descriptions = []
        for cid, texts in daily_review_clusters.items():
            sample_text = " | ".join(texts[:5])
            cluster_descriptions.append(f"Cluster {cid}: {sample_text}")
            
        prompt = f"""
        You are a Senior Product Analyst Agent.
        
        Current Topic Taxonomy: {json.dumps(self.taxonomy)}
        
        New Review Clusters (Daily Batch):
        {json.dumps(cluster_descriptions)}
        
        TASK:
        1. Classify each Cluster into an existing topic from the Taxonomy OR create a NEW topic name if it represents a new trend.
        2. MERGE synonyms. If a cluster is "Delivery guy rude" and taxonomy has "Delivery partner rude", map it to "Delivery partner rude".
        3. BE SPECIFIC. Avoid generic topics like "Good app". Use "Bolt delivery request" or "Map accuracy" if evident.
        
        OUTPUT JSON FORMAT ONLY:
        {{
            "cluster_mappings": {{ "Cluster 0": "Topic Name", ... }},
            "new_taxonomy_additions": ["New Topic 1", ...]
        }}
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are a JSON-speaking API."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            
            if "new_taxonomy_additions" in data:
                new_topics = [t for t in data['new_taxonomy_additions'] if t not in self.taxonomy]
                self.taxonomy.extend(new_topics)
                
            return data.get("cluster_mappings", {})
            
        except Exception as e:
            print(f"Agent Error: {e}")
            return {}

def process_batches(df):
    agent = TaxonomyAgent()
    trend_data = []
    
    dates = sorted(df['date'].unique())
    print("Agent starting daily batch processing...")
    
    for current_date in dates:
        print(f"Processing batch: {current_date}")
        daily_df = df[df['date'] == current_date].copy()
        
        if daily_df.empty:
            continue
            
        reviews_text = daily_df['content'].tolist()
        
        embeddings = embedder.encode(reviews_text)
        
        num_clusters = max(1, len(reviews_text) // 5)
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=1.5,
            metric='euclidean', 
            linkage='ward'
        )
        cluster_labels = clustering.fit_predict(embeddings)
        
        daily_clusters = {}
        for idx, label in enumerate(cluster_labels):
            daily_clusters.setdefault(str(label), []).append(reviews_text[idx])
            
        mappings = agent.refine_topics(daily_clusters)
        
        daily_counts = {}
        for idx, label in enumerate(cluster_labels):
            cluster_key = str(label)
            topic = mappings.get(cluster_key, "Uncategorized")
            daily_counts[topic] = daily_counts.get(topic, 0) + 1
            
        for topic, count in daily_counts.items():
            trend_data.append({
                "date": current_date,
                "topic": topic,
                "count": count
            })
            
    return trend_data, agent.taxonomy

def generate_report(trend_data):
    if not trend_data:
        print("No data to report.")
        return

    df_trends = pd.DataFrame(trend_data)
    
    pivot_df = df_trends.pivot_table(
        index='topic', 
        columns='date', 
        values='count', 
        fill_value=0
    )
    
    end_date = df_trends['date'].max()
    start_date = end_date - timedelta(days=30)
    date_range = pd.date_range(start=start_date, end=end_date).date
    
    pivot_df = pivot_df.reindex(columns=date_range, fill_value=0)
    
    os.makedirs("output", exist_ok=True)
    file_path = "output/trend_analysis_report.csv"
    pivot_df.to_csv(file_path)
    print(f"Final Report Generated: {file_path}")
    print(pivot_df.iloc[:5, -5:])

if __name__ == "__main__":
    df_reviews = fetch_historical_reviews(APP_ID, START_DATE)
    trend_data, final_taxonomy = process_batches(df_reviews)
    generate_report(trend_data)
    print("\nFinal Taxonomy Discovered by Agent:")
    print(final_taxonomy)