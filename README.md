# movie-recommender
The system demonstrates data preprocessing, text vectorization, and similarity computation in a real-world scenario. It converts movie metadata into a unified text feature, vectorizes it using CountVectorizer, and computes cosine similarity to find similar movies.
# Movie Recommendation System

## Description
This Python-based Movie Recommendation System suggests movies similar to a selected movie using **content-based filtering**.  
It analyzes movie attributes like **genres, keywords, cast, crew, and overview** to provide relevant recommendations.  

## Features
- Recommends **top 5 movies** similar to a given movie.  
- Processes movie metadata including genres, keywords, cast, and crew.  
- Interactive terminal input for entering movie names.  
- Handles missing data and text preprocessing for accurate results.  

## Technologies Used
- Python 3  
- Pandas, NumPy  
- scikit-learn (CountVectorizer, cosine similarity)  

## How to Run
1. Make sure **Python 3** is installed.  
2. Install dependencies:
```bash
pip3 install pandas numpy scikit-learn
Run the program:

bash
Copy code
python3 recommender.py
Type a movie name when prompted, or type exit to quit.

Example
mathematica
Copy code
Enter a movie name: Avatar

🎬 Top 5 movies similar to 'Avatar':
👉 Guardians of the Galaxy
👉 Aliens
👉 Star Trek
👉 John Carter
👉 Star Wars
Example Code Snippet
python
Copy code
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv")

# Simple text processing and vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['overview'].fillna('')).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)
Notes
Ensure tmdb_5000_movies.csv and tmdb_5000_credits.csv are in the same folder as recommender.py.

The system uses top 3 cast members and the director for more accurate recommendations.

yaml
Copy code

---

### ✅ How to use this README
1. Save it in your project folder as `README.md`.  
2. Preview in VS Code: `Cmd+Shift+V` (Mac) or `Ctrl+Shift+V` (Windows).  
3. Commit and push to GitHub:

```bash
git add README.md
git commit -m "Add project README"
git push
