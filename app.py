import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# ✅ Set page config FIRST
st.set_page_config(page_title="SHL Recommender", layout="wide")

# ✅ Load SHL assessment data
with open("shl_data.json", "r") as f:
    assessments = json.load(f)

# ✅ Load model & prepare data
model = SentenceTransformer('all-MiniLM-L6-v2')
shl_texts = [a["description"] for a in assessments]
shl_embeddings = model.encode(shl_texts, convert_to_tensor=True)

# ✅ UI
st.title("🔍 SHL Assessment Recommendation System")
query = st.text_area("Paste your job description or search query here:")

if st.button("Recommend"):
    if not query.strip() or len(query.strip()) < 5:
        st.warning("Please enter a detailed query (at least 5 characters).")
    else:
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, shl_embeddings)[0]

        # Set similarity threshold (0.3 is a safe default)
        threshold = 0.3
        filtered_results = [
            (i, float(score)) for i, score in enumerate(scores) if score >= threshold
        ]

        if not filtered_results:
            st.error("No relevant assessments found. Please try refining your query.")
        else:
            # Sort by similarity score descending and pick top 10
            top_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:10]
            top_assessments = [assessments[i] for i, _ in top_results]

            df = pd.DataFrame(top_assessments)
            df["Assessment"] = df.apply(lambda row: f"[{row['name']}]({row['url']})", axis=1)

            df = df.rename(columns={
                "remote": "Remote",
                "adaptive": "Adaptive",
                "duration": "Duration",
                "type": "Test Type"
            })

            st.subheader("🔎 Top Relevant SHL Assessments:")
            st.dataframe(df[["Assessment", "Remote", "Adaptive", "Duration", "Test Type"]], use_container_width=True)
