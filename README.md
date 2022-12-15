![alt text](https://mltic.my/wp-content/uploads/2022/08/1635783444_This-is-MANGA-Plus-an-app-to-read-comics-for-790x527.jpg)
# MISK Data Science Program
* [Slide](./Presentation.pptx)
* [Model](./model.ipynb)

# Overview
This project aims to build a reliable recommender system for manga, by taking one manga as an input and recommends 10 similar mangas. You can filter the recommendations by different features.

# Dataset

The dataset was obtained from this [github](https://github.com/goldbattle/MangadexRecomendations). The data contains total of 60000 manga, and total of 20 features.

# Similarity

The similarity is calculated using both the TF-IDf vectorizer and the cosine similarity on the description of the manga. Also different features in the data such as genres and formats have their influence on the similarity score.

# Web App

This is a basic web app for the project [link](https://alwaleed-7-dsi-manga-recommender-app-991svj.streamlit.app/)

# Refrences

[Mangadex](https://mangadex.org/)

[github](https://github.com/similar-manga/similar)
