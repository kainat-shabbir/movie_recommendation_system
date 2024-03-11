# movie_recommendation_system

import numpy as np
import pandas as pd
import ast
import pandas as pd

# Assuming 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in the same directory
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Display the shape of the 'movies' and credits DataFrame
movies.shape
credits.shape

print(type(movies))
print(type(credits))
movies['title'] = movies['title'].str.strip()
credits['title'] = credits['title'].str.strip()
# Assuming 'title' is a common column in both DataFrames

movies = movies.merge(credits, on='title', suffixes=('', ''))
# Display the first few rows of the merged DataFrame
movies.head(1)
# selection of relevant columns on which tags will be created and recommendation will be made
movies = movies[['movie_id', 'title','overview','genres','keywords', 'cast','crew']]
# checking for null values
movies.isnull().sum()
# dropping null values 
movies.dropna(inplace=True)
movies.isnull().sum()
movies.duplicated().sum()