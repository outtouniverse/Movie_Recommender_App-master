import streamlit as st
from PIL import Image
import json
from Classifier import KNearestNeighbours
from bs4 import BeautifulSoup
import requests, io
import PIL.Image
from urllib.request import urlopen

# Load data from JSON files
with open('./Data/movie_data.json', 'r+', encoding='utf-8') as f:
    data = json.load(f)
with open('./Data/movie_titles.json', 'r+', encoding='utf-8') as f:
    movie_titles = json.load(f)

hdr = {'User-Agent': 'Mozilla/5.0'}

def movie_poster_fetcher(imdb_link):
    """Fetch and return the movie poster from IMDb."""
    try:
        url_data = requests.get(imdb_link, headers=hdr).text
        s_data = BeautifulSoup(url_data, 'html.parser')
      
        imdb_dp = s_data.find("meta", property="og:image")
        if imdb_dp:
            movie_poster_link = imdb_dp.attrs.get('content')
            u = urlopen(movie_poster_link)
            raw_data = u.read()
            image = PIL.Image.open(io.BytesIO(raw_data))
          
            image = image.resize((474, 902))  
            return image
        else:
            st.warning("Movie poster not found.")
            return None
    except Exception as e:
        st.error(f"Error fetching movie poster: {e}")
        return None

def get_movie_info(imdb_link):
    """Fetch and return movie information from IMDb."""
    try:
        url_data = requests.get(imdb_link, headers=hdr).text
        s_data = BeautifulSoup(url_data, 'html.parser')
        
        rating = s_data.find("span", class_="sc-bde20123-1 iZlgcd")
        movie_rating = 'Total Rating count: ' + rating.text if rating else ''
        
        return movie_rating
    except Exception as e:
        st.error(f"Error fetching movie information: {e}")
        return ''

def KNN_Movie_Recommender(test_point, k):
    """Recommend movies using K-Nearest Neighbors."""
   
    target = [0 for _ in movie_titles]
   
    model = KNearestNeighbours(data, target, test_point, k=k)
    
    model.fit()
   
    table = []
    for i in model.indices:
        
        table.append([movie_titles[i][0], movie_titles[i][2], data[i][-1]])
    return table

st.set_page_config(
    page_title="Movie Recommender System",
)

def run():
    
    st.title("Movie Recommender System")
    st.markdown(
        '''<h4 style='text-align: left; color: #d73b5c;'>* Data is based on "IMDB 5000 Movie Dataset"</h4>''',
        unsafe_allow_html=True
    )
    
    movies = [title[0] for title in movie_titles]
    
    select_movie = st.selectbox('Select movie: (Recommendation will be based on this selection)', movies)
    
    if select_movie:
        no_of_reco = 7  # Fixed number of recommendations
        test_points = data[movies.index(select_movie)]
        table = KNN_Movie_Recommender(test_points, no_of_reco + 1)
        table.pop(0)  # Remove the selected movie from recommendations
        c = 0
        st.success('Here are some movie recommendations based on your selection:')
        
        # Use columns to display movies horizontally
        cols = st.columns(no_of_reco)

        for idx, (movie, link, ratings) in enumerate(table):
            with cols[idx]:
                st.markdown(f"**{movie}**")
                with st.spinner("Loading..."):
                    image = movie_poster_fetcher(link)
                    if image:
                        st.image(image, use_column_width=True)
                total_rat = get_movie_info(link)
                if total_rat:
                    st.markdown(total_rat)
                st.markdown('IMDB Rating: ' + str(ratings) + '⭐')

run()
