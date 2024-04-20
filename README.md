# This ReadMe contains code that i executed on jupyter notebook as well as code that i executed on pycharm to provide this system a GUI. Also i have included link to google drive where all files can be found here:
https://drive.google.com/file/d/1Po0rdwQFdO-gBxFo3d_MXcI-Hnb-x8US/view?usp=sharing

## movie_recommendation_system - Jupyter Notebook Code

# 1. Setup and data loading
### 1. Import Libraries
import numpy as np <br>
import pandas as pd<br>
import ast <br>
### 2. Read Files
movies = pd.read_csv('tmdb_5000_movies.csv') <br>
credits = pd.read_csv('tmdb_5000_credits.csv') <br>

# 2. Exploratory Data Analysis
### 1. Data Overview
movies.shape <br>
credits.shape <br>
print(type(movies)) <br>
print(type(credits)) <br>
movies['title'] = movies['title'].str.strip() <br>
credits['title'] = credits['title'].str.strip() <br>
### 2. Joining DataFrames
movies = movies.merge(credits, on='title', suffixes=('', '')) <br>
movies.head(1) <br>
### 3. Selection of relevant columns
movies = movies[['movie_id', 'title','overview','genres','keywords', 'cast','crew']] <br>
# 3. Data Cleaning and Preprocessing
### 1. Handling missing values
movies.isnull().sum() <br>
movies.dropna(inplace=True) <br>
movies.isnull().sum() <br>
### 2. Removing Duplicates
movies.duplicated().sum() <br>

(movies.iloc[0].genres) <br>
(movies.iloc[0].keywords) <br>
# 4. Feature Engineering
### 1. Extracting genre and keywords using convert function
def convert(obj): <br>
    L= [] <br>
    for i in ast.literal_eval(obj): <br>
        L.append(i['name']) <br>
    return L <br>
movies['genres'] = movies['genres'].apply(convert) <br>
movies['keywords'] = movies['keywords'].apply(convert) <br>
movies.head() <br>
### 2. Extracting cast using convertCast function
def convertCast(obj): <br>
    L= [] <br>
    counter = 0 <br>
    for i in ast.literal_eval(obj): <br>
        if counter !=3: <br>
            L.append(i['name']) <br>
            counter+=1 <br>
        else: <br>
            break <br>
    return L <br>
movies['cast'] = movies['cast'].apply(convertCast) <br>
movies.head() <br>
movies['crew'][0] <br>

### 3. Extracting director using fetch_director function
def fetch_director(obj): <br>
    L= [] <br>
    for i in ast.literal_eval(obj): <br>
        if i['job'] == 'Director': <br>
            L.append(i['name']) <br>
            break <br>
    return L <br>
movies['crew']= movies['crew'].apply(fetch_director) <br>
movies.head() <br>

### 4. Text Preprocessing
#### i. Splitting Overview Column
movies['overview']= movies['overview'].apply(lambda x:x.split()) <br>
#### ii. Removing Spaces
movies['genres']= movies['genres'].apply(lambda x:[i.replace(" ","") for i in x]) <br>
movies['cast']= movies['cast'].apply(lambda x:[i.replace(" ","") for i in x]) <br>
movies['crew']= movies['crew'].apply(lambda x:[i.replace(" ","") for i in x]) <br>
movies['keywords']= movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x]) <br>
movies.head() <br>
#### iii. Creating tags column
movies['tags'] = movies['overview'] + movies['genres'] + movies['cast'] + movies['crew'] + movies['keywords'] <br>
movies.head() <br>
# 5. Text Vectorization

### 1. Creating tag based dataframe 
new_df = movies[['movie_id', 'title', 'tags']] <br>
### 2. Convert tags list into string
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x)) <br>
new_df.head() <br>
new_df['tags'][0] <br>
### 3. Convert all letters into lower case
new_df['tags']= new_df['tags'].apply(lambda x:x.lower()) <br>
### 4. Import sklearn class and vectorizing text features
from sklearn.feature_extraction.text import CountVectorizer <br>
cv = CountVectorizer (max_features = 5000, stop_words = 'english') <br>
vectors = cv.fit_transform(new_df['tags']).toarray() <br>
vectors[0] <br>
### 5. Importing nltk porterStemmer class and stemming words
import nltk <br>
from nltk.stem.porter import PorterStemmer <br>
ps = PorterStemmer() <br>
def stem(text): <br>
    y=[] <br>
    for i in text.split(): <br>
        y.append(ps.stem(i)) <br>
    return " ".join(y) <br>
new_df['tags']= new_df['tags'].apply(stem) <br>
new_df['tags'][0] <br>
cv.get_feature_names_out() <br>
# 6. Similarity Calculation
### 1. Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity <br>
similarity = cosine_similarity(vectors) <br>
similarity[14] <br>
sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x[1])[1:6] <br>
# 7. Recommendation Function
def recommend(movie): <br>
    movie_index = new_df[new_df['title'] == movie].index[0]        # fetch the index of entered movie <br>
    distances = similarity[movie_index]                 # fetch the distances of found index <br>
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6] <br>

    for i in movies_list: <br>
        print(new_df.iloc[i[0]].title)       # fetching the titles of the movie_list <br>
recommend('Titanic')<br>
# 8. Model Saving
### Using Pickle
import pickle<br>
pickle.dump(new_df.to_dict(), open('movie_dict_pkl', 'wb')) <br>
pickle.dump(similarity,open('similarity.pkl','wb')) <br>

# 9. Streamlit website code
### 1. Loading imports
import streamlit as st <br>
import pickle <br>
import pandas as pd <br>
import requests <br>

### 2. Poster fetching function
def fetch_poster(movie_id): <br>
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=089eee4360cb595f2f54f08a0471856d'.format(movie_id)) <br>
    data = response.json() <br>
    return "http://image.tmdb.org/t/p/w500/" + data['poster_path'] <br>

### 3. Recommendation logic using pickle data
def recommend(movie): <br>
    movie_index = movies[movies['title'] == movie].index[0]  # fetch the index of entered movie <br>
    distances = similarity[movie_index]  # fetch the distances of found index <br>
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6] <br>
    recommended_movies = [] <br>
    recommended_posters = [] <br>
    for i in movies_list: <br>
        movie_id = movies.iloc[i[0]].movie_id <br>
        # fetch poster from API <br>
        recommended_movies.append(movies.iloc[i[0]].title)  # fetching the titles of the movie_list <br>
        recommended_posters.append(fetch_poster(movie_id)) <br>
    return recommended_movies, recommended_posters <br>
with open('movie_dict_pkl', 'rb') as file: <br>
    movies_dict = pickle.load(file) <br>
with open('similarity.pkl', 'rb') as file: <br>
    similarity = pickle.load(file) <br>
movies = pd.DataFrame(movies_dict) <br>
print(type(movies)) <br>

### 4. Streamlt UI setup
st.title("Movie Recommendation System") <br>
selected_movie_name = st.selectbox('How would you like to be collected?', movies['title'].values) <br>

if st.button('Recommend'): <br>
    names,posters = recommend(selected_movie_name) <br>

    col1, col2, col3, col4, col5 = st.columns(5) <br>

    with col1: <br>
        st.write(names[0]) <br>
        st.image(posters[0]) <br>

    with col2: <br>
        st.write(names[1]) <br>
        st.image(posters[1]) <br>

    with col3: <br>
        st.write(names[2]) <br>
        st.image(posters[2]) <br>

    with col4: <br>
        st.write(names[3]) <br>
        st.image(posters[3]) <br>

    with col5: <br>
        st.write(names[4]) <br>
        st.image(posters[4]) <br>

































