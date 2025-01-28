import pandas as pd

# Load the MovieLens files
movies = pd.read_csv("movies.csv")
tags = pd.read_csv("tags.csv")


# Optional: Merge datasets (e.g., movies and ratings)
merged_data = pd.merge(tags, movies, on="movieId")

# Preview the dataset
print(merged_data.head())

# Export to CSV
merged_data.to_csv("movie_tags.csv", index=False)

print("Dataset exported successfully to 'movie_tags.csv'")