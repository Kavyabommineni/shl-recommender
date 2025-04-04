from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, util
import json

app = FastAPI()

with open("shl_data.json", "r") as f:
    data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [d["description"] for d in data]
embeddings = model.encode(texts, convert_to_tensor=True)

@app.get("/recommend")
def recommend(query: str):
    q_embed = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(q_embed, embeddings)[0]
    top = scores.argsort(descending=True)[:10]
    return [data[i] for i in top]
