# 🏆 Candidate Retrieval Engine

An optimized, end-to-end artificial intelligence pipeline designed to filter, rank, and retrieve the top 100 matching candidates from a large-scale talent dataset based on semantic job description mapping and behavioral signals. Built with a fast Python backend and an interactive Streamlit frontend web dashboard interface.

## 🚀 Performance Benchmarks
* **Execution Baseline Time:** ~2 minutes and 35 seconds.
* **Architecture:** Fully protected Windows multi-core sub-processing using isolated `ProcessPoolExecutor` blocks to prevent memory overflow and infinite fork loops.

---

## 🛠️ System Architecture

1. **Fast C-Engine Loader:** Safely caches the `candidates.jsonl` dataset into local system RAM using accelerated Pandas parsing layers.
2. **3-Level Multi-Core Filtering System:** Uses parallel background worker threads to truncate heavy textual inputs and quickly prune non-matching candidates.
3. **Semantic Embedding Matcher:** Compares job description criteria vectors with candidate vectors using PyTorch vector calculations and cosine distance evaluation scores.
4. **Dynamic Reasoning Engine:** Vectorized matching string synthesizer that builds a logical, custom, and rank-consistent justification string for every single profile matching the target format.

---

## 📋 Submission Layout Format
The engine processes internal raw signals down to the exact 4-column schema matching evaluation standards:
* `candidate_id`: Standardized unique hash pointer.
* `rank`: Monotonically consistent rank tracking sequence (1 to 100).
* `score`: Calculated rank performance confidence score, formatted precisely to 4 decimal places.
* `reasoning`: A unique custom sentence detailing the applicant's current title, years of experience, core AI skills, and recruiter response rates.

---

## 💻 Installation & Setup

Follow these steps to deploy and run the system locally on your environment:

### 1. Clone & Initialize the Project Environment
Ensure you have Python 3.10+ installed on your computer, open your terminal inside the root project directory, and initialize a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1