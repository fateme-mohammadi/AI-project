import pandas as pd

# Load the MovieLens files
movies = pd.read_csv("ml-latest-small\ml-latest-small\movies.csv")
ratings = pd.read_csv(r"ml-latest-small\ml-latest-small\ratings.csv")


# Optional: Merge datasets (e.g., movies and ratings)
merged_data = pd.merge(ratings, movies, on="movieId")

# Preview the dataset
print(merged_data.head())

# Export to CSV
merged_data.to_csv("movie_lens_export.csv", index=False)

print("Dataset exported successfully to 'movie_lens_export.csv'")
