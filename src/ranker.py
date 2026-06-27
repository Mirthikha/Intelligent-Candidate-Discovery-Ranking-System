import json
import pandas as pd 
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.pipeline import three_level_architecture
import os
import docx
import torch
from datasketch import MinHash, MinHashLSH
import torch.ao.quantization as ao_quant
import streamlit as st
import time
from pathlib import Path

if Path("data/candidates.jsonl").exists():
    DATA_PATH = "data/candidates.jsonl"
else:
    DATA_PATH = "data/sample_candidates.json"

def update_ui_status(text):
    import sys
    
    if 'streamlit' in sys.modules:
        import streamlit as st
        print(text)

def parallel_processing_ranker(candidate_chunks):

    system_cores=os.cpu_count() or 4
    no_of_cores = min(10, max(1, system_cores - 2))

    chunk_size=5000
    
    sub_chunks=[candidate_chunks[i:i+chunk_size] for i in range(0, len(candidate_chunks),chunk_size)]

    processed_candidates=[]

    with ProcessPoolExecutor(max_workers=no_of_cores) as executor:

        results=executor.map(three_level_architecture, sub_chunks)

        for result in results:
            processed_candidates.extend(result)
    
    return processed_candidates

def candidate_profile_text(row):
    profile=row.get('profile',{})
    headline = profile.get('headline', '')
    summary = profile.get('summary', '')  
    
    skills_list = row.get('skills', [])
    skills_string = ", ".join([skill.get('name', '') for skill in skills_list])
    
    return f"Headline: {headline}\nSummary: {summary}\nSkills: {skills_string}"

@st.cache_data
def run_pipeline(ui_callback=None):
    start = time.perf_counter()
    
    print("Reading jsonl as python dicts...")
    raw_data=[]

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    
    print("Data loaded successfully and size is ", len(raw_data))

    #Parallel process function is called
    filtered_candidates=parallel_processing_ranker(raw_data)

    df=pd.DataFrame(filtered_candidates)
    print("Shape = ",df.shape)

    df=df.drop_duplicates(subset='candidate_id')

    print("3 Level filtered data size is ", df.shape)

    # Early Feature Assembly to avoid KeyErrors downstream
    df['combined_profile_text'] = df.apply(candidate_profile_text, axis=1)

    #Identify Duplicates
    lsh = MinHashLSH(threshold=0.85, num_perm=128)
    indices_to_drop = []

    minhashes = {}

    for idx, row in df.iterrows():
        text = str(row['combined_profile_text']).lower()
        words = text.split()

        shingles = [" ".join(words[i:i+3]) for i in range(len(words)-2)]

        m = MinHash(num_perm=128)
        for shingle in shingles:
            m.update(shingle.encode('utf-8'))

        result1 = lsh.query(m)
        if len(result1) > 0:
            indices_to_drop.append(idx)
        else:
            lsh.insert(f"cand_{idx}", m)

    df = df.drop(index=indices_to_drop).reset_index(drop=True)

    print("After removing duplicates shape is ", df.shape)

     # Set CPU optimization constraint
    total_cores = os.cpu_count() or 4

    if total_cores >= 12:
        optimal_threads = 8
        optimal_batch_size = 1024  
    elif total_cores >= 6:
        optimal_threads = 4
        optimal_batch_size = 512   
    else:
        optimal_threads = 2
        optimal_batch_size = 256

    print(f" -> Hardware Detected: {total_cores} logical cores.")
    print(f" -> Dynamically allocating {optimal_threads} compute threads.")
    print(f" -> Tuning matrix pipeline execution to batch_size={optimal_batch_size}.")

    torch.set_num_threads(optimal_threads)

    #Embedding job description 

    doc = docx.Document('data/job_description.docx')
    full_text=[]

    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)

    jd_text="\n".join(full_text)

    model = SentenceTransformer('src/ALL-MiniLM-l6-v2')

    jd_embedding=model.encode(jd_text).tolist()

    candidate_texts = df['combined_profile_text'].tolist()

    df = df.reset_index(drop=True)
    print(df.size)

    clean_candidate_texts = df['combined_profile_text'].tolist()

    model1 = ao_quant.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )

    clean_candidate_texts = [str(text)[:450] for text in clean_candidate_texts]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    df['embedding'] = model.encode(
        clean_candidate_texts, 
        batch_size=1024,         
        convert_to_numpy=True,
        show_progress_bar=True
    ).tolist()

    candidate_embeddings = np.array(df['embedding'].tolist())
    target_jd_vector = np.array(jd_embedding).reshape(1, -1)

    print( "Embeddings done ")

    df['base_similarity'] = cosine_similarity(candidate_embeddings, target_jd_vector).flatten()
    df['score'] = df['behavioral_multiplier'] * df['base_similarity']
    print("Cosine similarity calculated")

    df_ranked = df.sort_values(by='score', ascending=False).reset_index(drop=True)

    df_ranked['score'] = df_ranked['score'].apply(lambda x: f"{float(x):.4f}")
 
    df_ranked['rank'] = df_ranked.index + 1

    sim_percentages = (df_ranked['base_similarity'] * 100).round(1).astype(str)

    roles = df_ranked['profile'].apply(lambda x: x.get('current_title', 'Specialist') if isinstance(x, dict) else 'Specialist').astype(str)
    exps = df_ranked['profile'].apply(lambda x: x.get('years_of_experience', 5.0) if isinstance(x, dict) else 5.0).round(1).astype(str)

    skills = df_ranked['skills'].apply(lambda x: len(x) if isinstance(x, list) else 6).astype(str)
    
    response_rates = df_ranked['redrob_signals'].apply(lambda x: x.get('recruiter_response_rate', 0.75) if isinstance(x, dict) else 0.75).round(2).astype(str)

    df_ranked['reasoning'] = roles + " with " + exps + " yrs exp; " + skills + "  skills; response rate " + response_rates + "; " + sim_percentages  +  " % JD similarity."

    required_columns = ['candidate_id', 'rank', 'score', 'reasoning']
    final_df = df_ranked[required_columns]

    df_top_100 = final_df.head(100)
    output_filename = "top_100_ranked_candidates.csv"
    df_top_100.to_csv(output_filename, index=False)
    df_top_100.to_excel("top_100_ranked_candidates.xlsx", index=False)

    print(f"Execution time: {time.perf_counter() - start:.2f} seconds")

    return df_top_100

if __name__ == "__main__":
    result_df = run_pipeline()
    # FIX: Only print columns that actually exist inside your final 3-column filtered df!
    print(result_df.head())
    print("Done! Final Processed shape:", result_df.shape)