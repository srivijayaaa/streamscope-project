import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# PAGE CONFIG

st.set_page_config(page_title="StreamScope", layout="wide")
st.title("🎬 StreamScope: Netflix Content Analysis & Recommendation System")


# LOAD DATA

@st.cache_data
def load_data():
    df = pd.read_csv("data/netflix_milestone3.csv")
    df = df.fillna("")
    return df

df = load_data()


# FIX COUNTRY FILTER (ONE-BY-ONE + SORTED)

all_countries = df['country'].str.split(',')
all_countries = all_countries.explode().str.strip()
all_countries = sorted(all_countries.unique())


# SIDEBAR FILTERS

st.sidebar.header("🔍 Filters")

countries = st.sidebar.multiselect("Country", all_countries)
genres = st.sidebar.multiselect(
    "Genre",
    sorted(df['listed_in'].str.split(',').explode().str.strip().unique())
)
ratings = st.sidebar.multiselect("Rating", sorted(df['rating'].unique()))

years = st.sidebar.slider(
    "Release Year",
    int(df['release_year'].min()),
    int(df['release_year'].max()),
    (2000, 2021)
)

# APPLY FILTERS

filtered_df = df.copy()

if countries:
    filtered_df = filtered_df[
        filtered_df['country'].str.contains('|'.join(countries), na=False)
    ]

if genres:
    filtered_df = filtered_df[
        filtered_df['listed_in'].str.contains('|'.join(genres), na=False)
    ]

if ratings:
    filtered_df = filtered_df[filtered_df['rating'].isin(ratings)]

filtered_df = filtered_df[
    (filtered_df['release_year'] >= years[0]) &
    (filtered_df['release_year'] <= years[1])
]

# KPI METRICS
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", len(filtered_df))
col2.metric("Movies", len(filtered_df[filtered_df['type'] == 'Movie']))
col3.metric("TV Shows", len(filtered_df[filtered_df['type'] == 'TV Show']))

top_genre = filtered_df['listed_in'].mode()
col4.metric("Top Genre", top_genre[0] if not top_genre.empty else "N/A")

# VISUALIZATIONS
colA, colB = st.columns(2)

with colA:
    fig1 = px.pie(filtered_df, names='type', title="Content Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    growth = filtered_df.groupby('year_added').size().reset_index(name='count')
    fig2 = px.line(growth, x='year_added', y='count', title="Growth Over Time")
    st.plotly_chart(fig2, use_container_width=True)

colC, colD = st.columns(2)

with colC:
    genre_counts = filtered_df['listed_in'].str.split(',').explode().str.strip().value_counts().head(10)
    fig3 = px.bar(genre_counts, title="Top Genres")
    st.plotly_chart(fig3, use_container_width=True)

with colD:
    fig4 = px.bar(filtered_df['rating'].value_counts(), title="Rating Distribution")
    st.plotly_chart(fig4, use_container_width=True)


# CLUSTER VISUALIZATION

st.subheader("🧠 Content Clusters")

fig_cluster = px.scatter(
    filtered_df,
    x='release_year',
    y='duration_numeric',
    color='cluster',
    hover_data=['title', 'listed_in']
)

st.plotly_chart(fig_cluster, use_container_width=True)


# BUILD SIMILARITY

@st.cache_resource
def build_similarity(df):
    df['combined'] = (
        df['title'] + " " +
        df['listed_in'] + " " +
        df['description']
    )

    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(df['combined'])

    return cosine_similarity(matrix)

similarity = build_similarity(df)


# SPELL CORRECTION

def correct_query(query, df):
    titles = df['title'].tolist()
    genres = df['listed_in'].str.split(',').explode().str.strip().unique()

    all_words = list(titles) + list(genres)

    match = difflib.get_close_matches(query, all_words, n=1, cutoff=0.6)

    return match[0] if match else query


# RECOMMEND FUNCTION

def recommend(query, df, similarity, top_n=10):

    query = correct_query(query.lower().strip(), df)

    # EXACT MATCH
    exact = df[df['title'].str.lower() == query]

    if not exact.empty:
        idx = exact.index[0]

        scores = list(enumerate(similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        indices = [i[0] for i in scores]

        return df.iloc[idx], df.iloc[indices]

    # GENRE SEARCH
    genre_matches = df[
        df['listed_in'].str.lower().str.contains(query, na=False)
    ]

    if not genre_matches.empty:
        return None, genre_matches.head(top_n)

    # DESCRIPTION SEARCH
    desc_matches = df[
        df['description'].str.lower().str.contains(query, na=False)
    ]

    if not desc_matches.empty:
        return None, desc_matches.head(top_n)

    return None, None


# SEARCH

st.subheader("🔎 Search Content")

search = st.text_input("Search by title")

if search:
    results = df[df['title'].str.contains(search, case=False, na=False)]

    if not results.empty:
        for _, row in results.head(5).iterrows():
            st.markdown(f"### 🎬 {row['title']}")
            st.write(f"🎭 {row['listed_in']}")
            st.write(f"⭐ {row['rating']}")
            st.write(f"🌍 {row['country']}")
            st.write(f"📝 {row['description']}")
            st.markdown("---")


# RECOMMENDATION SYSTEM

st.subheader("🎯 Smart Recommendation System")

query = st.text_input("Enter a movie name or genre")

if st.button("Recommend"):

    selected, recs = recommend(query, df, similarity)

    if selected is not None:
        st.subheader("🎬 Selected Content")
        st.write(selected[['title', 'listed_in', 'rating', 'release_year']])

    if recs is not None and not recs.empty:
        st.subheader("🔥 Recommended for You")

        for _, row in recs.iterrows():
            st.markdown(f"### 🎬 {row['title']}")
            st.write(f"🎭 {row['listed_in']}")
            st.write(f"⭐ {row['rating']}")
            st.write(f"🌍 {row['country']}")
            st.write(f"📝 {row['description']}")
            st.markdown("---")

    if selected is None and (recs is None or recs.empty):
        st.error("No matching title or genre found.")