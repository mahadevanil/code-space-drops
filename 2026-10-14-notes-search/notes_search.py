"""
Semantic search over your own notes, in about 30 lines.

What this does: turns every note into a vector (a list of numbers that
captures its meaning), does the same for your search query, then returns
the notes whose vectors are closest to the query's vector. That's the
whole trick behind every "chat with your documents" tool.

Setup (free, runs locally, no API key needed):
    pip install sentence-transformers numpy

To use OpenAI's hosted embeddings instead (small cost, see the pricing
note at the bottom), swap embed_texts() for the commented-out version
below it.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # small, free, runs on CPU

# ---- 1. your notes -------------------------------------------------
notes = [
    "car wouldn't start this morning, might be the battery",
    "quarterly budget review pushed to next friday",
    "recipe idea: lemon garlic pasta with chili flakes",
    "need to get the vehicle checked, weird noise when braking",
    "team standup notes: ship the auth fix by wednesday",
]

# ---- 2. turn text into vectors --------------------------------------
model = SentenceTransformer(MODEL_NAME)


def embed_texts(texts):
    return model.encode(texts, normalize_embeddings=True)


# --- OpenAI version (uncomment to use instead) ---
# from openai import OpenAI
# client = OpenAI()
# def embed_texts(texts):
#     resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
#     vecs = np.array([d.embedding for d in resp.data])
#     return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


note_vectors = embed_texts(notes)


# ---- 3. search by meaning, not keywords -----------------------------
def search(query, top_k=3):
    query_vector = embed_texts([query])[0]
    # cosine similarity: how closely each note's vector points the
    # same direction as the query's vector (1.0 = identical meaning)
    scores = note_vectors @ query_vector
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(notes[i], float(scores[i])) for i in ranked]


if __name__ == "__main__":
    query = "car trouble"
    print(f'query: "{query}"\n')
    for note, score in search(query):
        print(f"  {score:.3f}  {note}")

    # try it: "car trouble" surfaces both the battery note and the
    # braking note, even though neither contains those exact words.

# Notes:
# - all-MiniLM-L6-v2 outputs 384-dim vectors and is free/local via
#   sentence-transformers.
# - OpenAI's text-embedding-3-small costs $0.02 per 1M tokens and
#   outputs 1536-dim vectors by default (openai.com pricing, Oct 2026).
# - For a few hundred notes, a plain numpy array like this is plenty.
#   Past a few thousand, look at Chroma or FAISS for faster search.
