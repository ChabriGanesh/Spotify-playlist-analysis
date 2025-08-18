import streamlit as st
import requests
import pandas as pd
st.title("Spotify Playlist Recommendations")
seed = st.text_input("Enter seed track name:", "Shape of You")
num_songs = st.slider("Number of recommendations:", 1, 20, 10)
if st.button("Get Recommendations"):
    url = "http://127.0.0.1:8000/recommend"
    params = {"seed": seed, "n": num_songs}    
    with st.spinner("Fetching recommendations..."):
        response = requests.get(url, params=params)        
    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            st.table(df)
        else:
            st.warning("No recommendations found. Try another seed.")
    else:
        st.error(f"API error: {response.status_code}")
