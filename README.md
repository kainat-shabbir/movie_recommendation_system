# This ReadMe contains code that i executed on jupyter notebook as well as code that i executed on pycharm to provide this system a GUI. Also i have included link to google drive where all files can be found here:
https://drive.google.com/file/d/1Po0rdwQFdO-gBxFo3d_MXcI-Hnb-x8US/view?usp=sharing

## movie_recommendation_system - Jupyter Notebook Code

# 1. Setup and data loading
### 1. Import Libraries
import numpy as np <br>
import pandas as pd<br>
import ast <br>
### 1. Read Files
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

## Exploratory Data Analysis
### Display the shape of the 'movies' and 'credits' DataFrame
movies.shape
credits.shape

print(type(movies))
print(type(credits))
movies['title'] = movies['title'].str.strip()
credits['title'] = credits['title'].str.strip()
### Assuming 'title' is a common column in both DataFrames

movies = movies.merge(credits, on='title', suffixes=('', ''))
### Display the first few rows of the merged DataFrame
movies.head(1)
### selection of relevant columns on which tags will be created and recommendation will be made
movies = movies[['movie_id', 'title','overview','genres','keywords', 'cast','crew']]
### checking for null values
movies.isnull().sum()
### dropping null values 
movies.dropna(inplace=True)
movies.isnull().sum()
movies.duplicated().sum()
(movies.iloc[0].genres)
(movies.iloc[0].keywords)
### creating a function which will convert string of list into list and then extracting genre name from each dictionary
def convert(obj):
    L= []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()
### creating a function which will convert string of list into list and then extracting first 3 cast member names
def convertCast(obj):
    L= []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L
movies['cast'] = movies['cast'].apply(convertCast)
movies.head()
movies['crew'][0]

### creating a function which will convert string of list into list and then extracting director name from each dictionary
def fetch_director(obj):
    L= []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
movies['crew']= movies['crew'].apply(fetch_director)
movies.head()
### to split the string in overview column into words
movies['overview']= movies['overview'].apply(lambda x:x.split())
### remove spaces from genre, cast, keywords and crew column
movies['genres']= movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']= movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']= movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']= movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies.head()
### creating tags column by combining all 5 columns
movies['tags'] = movies['overview'] + movies['genres'] + movies['cast'] + movies['crew'] + movies['keywords']
movies.head()
### creating new dataFrame 
new_df = movies[['movie_id', 'title', 'tags']]
###  convert tags list into string
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df.head()
new_df['tags'][0]
### convert all letters into lower case
new_df['tags']= new_df['tags'].apply(lambda x:x.lower())
### import sklearn class CountVectorizer to do text vectorization of tags column (5000 most common words)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer (max_features = 5000, stop_words = 'english')
### converting scipy parse matrix into numpy array
vectors = cv.fit_transform(new_df['tags']).toarray()
vectors[0]
### importing nltk PorterStemmer Class which will provide the functionality to solve issues of similar words e.g, actor, actors
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
### function will take words and will extract their stem words, e.g. from dancing it will extract danc
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_df['tags']= new_df['tags'].apply(stem)
new_df['tags'][0]
cv.get_feature_names_out()
### for calculating cosine theta between vectors to see to what extinct two movies are similar, we are importing cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
### similarity matrix calculating distance of each movie with each other movie from 0 to 1
similarity = cosine_similarity(vectors)
similarity[14]
#### list(enumerate(similarity[0])): to create index and similarity value tuple
#### reverse=True: to print in descending order
####  key=lambda x:x[1]: to specify that we want to order on the base of second item of each tuple (not on index value)
sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x[1])[1:6]
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]        # fetch the index of entered movie
    distances = similarity[movie_index]                 # fetch the distances of found index
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)       # fetching the titles of the movie_list
recommend('Titanic')
import pickle
pickle.dump(new_df.to_dict(), open('movie_dict_pkl', 'wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

# Code for Website Creation


import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=089eee4360cb595f2f54f08a0471856d'.format(movie_id))
    data = response.json()
    return "http://image.tmdb.org/t/p/w500/" + data['poster_path']

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]  # fetch the index of entered movie
    distances = similarity[movie_index]  # fetch the distances of found index
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    recommended_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        # fetch poster from API
        recommended_movies.append(movies.iloc[i[0]].title)  # fetching the titles of the movie_list
        recommended_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_posters
# Load pickled movie dictionary
with open('movie_dict_pkl', 'rb') as file:
    movies_dict = pickle.load(file)
with open('similarity.pkl', 'rb') as file:
    similarity = pickle.load(file)
movies = pd.DataFrame(movies_dict)
print(type(movies))
st.title("Movie Recommendation System")

selected_movie_name = st.selectbox('How would you like to be collected?', movies['title'].values)

if st.button('Recommend'):
    names,posters = recommend(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.write(names[0])
        st.image(posters[0])

    with col2:
        st.write(names[1])
        st.image(posters[1])

    with col3:
        st.write(names[2])
        st.image(posters[2])

    with col4:
        st.write(names[3])
        st.image(posters[3])

    with col5:
        st.write(names[4])
        st.image(posters[4])

































