import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.title("Netflix Content Analytics Dashboard")

# Load dataset
df = pd.read_csv("data/netflix_milestone3.csv")

# Load trained model
with open("models/netflix_model.pkl", "rb") as f:
    model = pickle.load(f)

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Content Type Distribution")

fig, ax = plt.subplots()
sns.countplot(data=df, x="type", ax=ax)

st.pyplot(fig)

st.subheader("Rating Distribution")

fig, ax = plt.subplots()

sns.countplot(
    data=df,
    x="rating",
    order=df["rating"].value_counts().index,
    ax=ax
)

plt.xticks(rotation=90)

st.pyplot(fig)

st.subheader("Netflix Content Clusters")

fig, ax = plt.subplots()

scatter = ax.scatter(
    df["duration_numeric"],
    df["rating_encoded"],
    c=df["cluster"],
    cmap="viridis"
)

ax.set_xlabel("Duration")
ax.set_ylabel("Rating Encoded")

st.pyplot(fig)

st.subheader("Predict Content Type")

rating = st.number_input("Rating Encoded Value", min_value=0)
duration = st.number_input("Duration (numeric)")
length = st.number_input("Length Category Encoded")
rating_cat = st.number_input("Rating Category Encoded")

if st.button("Predict"):

    prediction = model.predict([[rating, duration, length, rating_cat]])

    if prediction[0] == 0:
        st.success("Prediction: Movie")
    else:
        st.success("Prediction: TV Show")


