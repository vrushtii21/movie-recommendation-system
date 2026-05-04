import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load data
with open("df.pkl", "rb") as f:
    df = pickle.load(f)

with open("indices.pkl", "rb") as f:
    indices = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

# Make case-insensitive
indices = {k.lower(): v for k, v in indices.items()}

# Recommendation function
def recommend(title, n=10):
    title = title.lower()

    if title not in indices:
        return ["Movie not found"]

    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_scores.argsort()[::-1][1:n+1]

    return df['title'].iloc[similar_idx].tolist()

# UI
st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter a movie name")

if st.button("Recommend"):
    if movie_name:
        results = recommend(movie_name)

        st.subheader("Recommended Movies:")
        for movie in results:
            st.write("•", movie)
    else:
        st.warning("Please enter a movie name")