# Pulse-AI Assignment: Agentic Trend Analysis System

## 📋 Project Overview

This system is an **Agentic AI solution** designed to analyze Google Play Store reviews for high-volume apps (e.g., Swiggy) and generate a daily trend analysis report.

Unlike traditional clustering (LDA/TopicBERT), this project utilizes a **Hybrid Agentic Workflow**. It combines the speed of vector embeddings with the reasoning capabilities of an LLM (GPT-4o) to autonomously discover, name, and evolve topics over time.

## 🚀 Key Features

* **Agentic Topic Discovery:** Autonomously identifies new, evolving trends (e.g., "Bolt delivery request") without manual seed lists.


* **Smart Deduplication:** Uses semantic reasoning to merge variations like *"Delivery guy rude"* and *"Delivery partner behaved badly"* into a single canonical topic.


* **Daily Batch Simulation:** Processes data strictly in daily batches (June 2024 – Present) to simulate real-world data ingestion.


* **High-Performance "Turbo" Mode:** Implements a "Vector-First, Agent-Fallback" architecture to minimize API costs and latency while maintaining high recall.



---

## 🛠️ Architecture

The system operates in three stages:

1. **Ingestion & Caching:** Scrapes thousands of reviews efficiently and caches them locally to allow rapid iteration.
2. **Vector-First Classification:** Uses `all-MiniLM-L6-v2` embeddings to instantly classify reviews into known topics (Efficiency Layer).
3. **Agentic Reasoning Loop:**
* Buffers "Unknown" reviews that don't match existing topics.
* When the buffer fills, the **Discovery Agent** (GPT-4o) analyzes the cluster.
* If a new trend is found, the Agent updates the global Taxonomy and re-indexes the vector space dynamically.



---

## ⚙️ Installation

### 1. Prerequisites

* Python 3.8+
* OpenAI API Key

### 2. Install Dependencies

Create a `requirements.txt` file and run the installer:

```bash
pip install pandas numpy google-play-scraper sentence-transformers scikit-learn openai tqdm

```

### 3. Configuration

Open `main.py` and update the following configuration variables:

```python
APP_ID = 'in.swiggy.android'       # Target App ID
OPENAI_API_KEY = "sk-..."          # Your OpenAI API Key
[cite_start]START_DATE = datetime(2024, 6, 1)  # Analysis start date [cite: 5]

```

---

## 🏃 Usage

Run the main script:

```bash
python main.py

```

### What happens during execution?

1. **Scraping:** The system fetches reviews back to June 1st, 2024.
2. **Caching:** Data is saved to `reviews_cache.csv` for speed.
3. **Processing:** You will see a progress bar as the Agent iterates through each day:
* *Matches known topics...*
* *Buffers unknowns...*
* *🚀 DISCOVERED NEW TREND: [Topic Name]* (triggered when the Agent identifies a new issue).


4. **Reporting:** A final CSV is generated.

---

## 📊 Output

The system generates a file at `output/trend_analysis_report.csv`.

**Format:**

* 
**Rows:** Discovered Topics (Issues, Requests, Feedback).


* 
**Columns:** Dates (Daily frequency counts).


* 
**Values:** Volume of reviews per topic per day.



**Example Output:**
| Topic | 2024-12-01 | 2024-12-02 | 2024-12-03 | ... |
| :--- | :--- | :--- | :--- | :--- |
| Delivery Issue | 45 | 50 | 12 | ... |
| **Bolt Delivery Request** | 0 | 5 | 18 | ... |
| **App Crash (Login)** | 12 | 14 | 11 | ... |

*(Note: "Bolt Delivery Request" would be a topic discovered autonomously by the Agent).*

---

## 🧩 How This Fulfills Requirements

| Requirement | Implementation Details |
| --- | --- |
| **Agentic Approach** | Uses LLM as a reasoning engine to maintain and update a dynamic taxonomy/ontology.

 |
| **Deduplication** | The Agent is explicitly prompted to "Merge synonyms" before creating new categories.

 |
| **Daily Batches** | The script sorts data by date and iterates day-by-day to build the trend history.

 |
| **Output Format** | Generates the exact T-30 matrix structure requested in the assignment.

 |
