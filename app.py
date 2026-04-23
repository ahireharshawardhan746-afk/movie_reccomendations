import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("movie_dataset.csv")
    return df

df = load_data()

# Features used
features = ['keywords', 'cast', 'genres', 'director']

# Fill NaN values
for feature in features:
    df[feature] = df[feature].fillna('')

# Combine features
def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']

df["combined_features"] = df.apply(combine_features, axis=1)

# Vectorization
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Similarity
cosine_sim = cosine_similarity(count_matrix)

# Helper functions
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

def get_recommendation(movie_name):
    try:
        movie_index = get_index_from_title(movie_name)
        similar_movies = list(enumerate(cosine_sim[movie_index]))
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]

        recommendations = []
        for movie in sorted_similar_movies:
            recommendations.append(get_title_from_index(movie[0]))

        return recommendations

    except:
        return ["Movie not found. Try another name."]

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_list = df["title"].values

selected_movie = st.selectbox("Select a movie:", movie_list)

if st.button("Recommend"):
    results = get_recommendation(selected_movie)

    st.subheader("Top Recommendations:")
    for movie in results:
        st.write(movie)