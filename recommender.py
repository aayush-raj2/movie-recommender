# recommender.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# ----------------------------
# 1. Load the dataset (local paths)
# ----------------------------
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets on the 'title' column
movies = movies.merge(credits, left_on='title', right_on='title')

# ----------------------------
# 2. Data Preprocessing
# ----------------------------
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:   # take top 3 actors
            L.append(i['name'])
            counter += 1
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# ----------------------------
# 3. Vectorization
# ----------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# ----------------------------
# 4. Recommendation Function
# ----------------------------
def recommend(movie):
    movie = movie.lower()
    if movie not in new_df['title'].str.lower().values:
        print("\n❌ Movie not found in database. Try another name.")
        return

    idx = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    print(f"\n🎬 Top 5 movies similar to '{new_df.iloc[idx].title}':")
    for i in movie_list:
        print("👉", new_df.iloc[i[0]].title)

# ----------------------------
# 5. Interactive Input
# ----------------------------
if __name__ == "__main__":
    print("✅ Movie Recommendation System is ready!")
    print("⚡ Debug: Script is running!")  # add this line
    while True:
        user_movie = input("\nEnter a movie name (or type 'exit' to quit): ")
        if user_movie.lower() == "exit":
            print("👋 Exiting... Have a nice day!")
            break
        recommend(user_movie)

