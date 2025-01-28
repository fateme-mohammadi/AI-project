import pandas as pd 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load datasets
movies = pd.read_csv("movies.csv")
links = pd.read_csv("links.csv")
tags = pd.read_csv("tags.csv")

# Merge movies with links to add IMDb ID
movies = movies.merge(links[['movieId', 'imdbId']], on='movieId', how='left')

# Preprocess movies data
def preprocess_movies(movies):
    movies['genres'] = movies['genres'].fillna('')  # Fill missing genres
    movies['clean_title'] = movies['title'].apply(lambda x: re.sub(r'\s\(\d{4}\)$', '', x).lower())  # Clean titles
    movies['imdb_link'] = movies['imdbId'].apply(lambda x: f"https://www.imdb.com/title/tt{int(x):07d}/" if not pd.isna(x) else None)  # Add IMDb links
    return movies

movies = preprocess_movies(movies)

# Merge tags with movies
tags = tags.merge(movies[['movieId', 'title']], on='movieId', how='left')

# Preprocess tags to join them into a single string for each movie
tags_grouped = tags.groupby('title')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# Merge with original movies dataframe
movies_with_tags = pd.merge(movies, tags_grouped, on='title', how='left')

# Combine genres and tags into a single feature for each movie
movies_with_tags['combined_features'] = movies_with_tags['genres'] + ' ' + movies_with_tags['tag'].fillna('')

# TF-IDF and similarity calculation
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_with_tags['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movies(movie_title, cosine_sim=cosine_sim):
    # Clean input title
    cleaned_title = re.sub(r'\s\(\d{4}\)$', '', movie_title).lower()

    # Check if the movie exists
    if cleaned_title not in movies_with_tags['clean_title'].values:
        return f"Error: '{movie_title}' not found in the dataset."

    # Get index of the movie
    idx = movies_with_tags[movies_with_tags['clean_title'] == cleaned_title].index[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the input movie and get top 10
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    # Return recommended movie titles and IMDb links
    return movies_with_tags.iloc[movie_indices][['title', 'imdb_link']]

# Streamlit Web App
st.title("Movie Recommendation System with Tags")

st.header("Get Movie Recommendations")
movie_title = st.text_input("Enter a movie title:")

if movie_title:
    recommendations = recommend_movies(movie_title)

    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.write("Recommended Movies:")
        for idx, row in enumerate(recommendations.itertuples(), start=1):
            st.write(f"{idx}. {row.title} - [IMDb Link]({row.imdb_link})")

st.markdown("---")
st.write("Ensure your `movies.csv`, `tags.csv`, and `links.csv` files are in the same directory and follow the correct format.")
