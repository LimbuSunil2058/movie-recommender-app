import streamlit as st 
import pandas as pd
import pickle
import requests

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("TMDB_API_KEY")

new_df = pickle.load(open('movies_dataframe.pkl', 'rb'))
movie_list = new_df['title'].values

#  Load cosinesimiliraty.pkl from Hugging Face(Because of its large file)
@st.cache_resource
def load_cosine_matrix():
    url = "https://huggingface.co/datasets/limbusunil/movie-recommendation-data/resolve/main/cosinesimiliraty.pkl"
    response = requests.get(url)
    with open("cosinesimiliraty.pkl", "wb") as f:
        f.write(response.content)
    with open("cosinesimiliraty.pkl", "rb") as f:
        return pickle.load(f)

cs = load_cosine_matrix()


def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    else:
        return "https://via.placeholder.com/300x450?text=No+Image"

def find_reccomendation(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distance = cs[movie_index]
    similar_movies = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:8]

    st.subheader("Top 7 Similar Movies:")
    
    cols = st.columns(3) 
    for i, sim_movie in enumerate(similar_movies):
        idx = sim_movie[0]
        title = new_df.loc[idx, 'title']
        movie_id = new_df.loc[idx, 'movie_id']  
        poster_url = fetch_poster(movie_id)

        with cols[i % 3]:
            st.image(poster_url, caption=title, use_container_width=True)


st.title("Movie Recommendation System")
mov = st.selectbox("Select your movie", movie_list)

#  Show poster of selected movie
def poster_id(new_df):
    matched_row = new_df[new_df['title'] == mov]
    if not matched_row.empty:
        return matched_row.iloc[0]['movie_id']
    return None

title_id = poster_id(new_df)
poster_url = fetch_poster(title_id)
st.image(poster_url, caption=f"Now Showing: {mov}", width=150)

click = st.button("Click for Recommendation")
if click:
    find_reccomendation(mov)
