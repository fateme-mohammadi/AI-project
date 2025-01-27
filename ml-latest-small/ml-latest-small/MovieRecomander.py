import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")  

# Preprocess
movies['genres'] = movies['genres'].fillna('')  # Fill missing genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def recommend_movies(movie_title, cosine_sim=cosine_sim):
    # Remove the production date from the input movie title
    cleaned_title = re.sub(r'\s\(\d{4}\)$', '', movie_title)  # Removes " (YYYY)" at the end
    
    # Preprocess the dataset titles to ignore production dates
    movies['clean_title'] = movies['title'].apply(lambda x: re.sub(r'\s\(\d{4}\)$', '', x))
    
    # Check if the movie exists in the dataset
    if cleaned_title not in movies['clean_title'].values:
        return f"Error: '{movie_title}' not found in the dataset."
    
    # Get the index of the input movie
    idx = movies[movies['clean_title'] == cleaned_title].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the input movie and get the top 10
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the recommended movie titles
    return movies['title'].iloc[movie_indices]

# Example usage
print(recommend_movies("Toy Story"))