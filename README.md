![alt text](https://mltic.my/wp-content/uploads/2022/08/1635783444_This-is-MANGA-Plus-an-app-to-read-comics-for-790x527.jpg)

# MangaMatch

## Overview

This project is an intelligent manga recommendation system that helps users discover new manga based on their preferences. The system takes one or more manga titles as input and recommends 10 similar manga, with advanced filtering options for genres, themes, and spin-offs.

## Key Features

- **Multi-Manga Input**: Get recommendations based on multiple manga titles simultaneously
- **Advanced Filtering**: Filter recommendations by genres, themes, and spin-off status
- **Comprehensive Dataset**: Built on a dataset of 60,000 manga with 20+ features
- **Intelligent Matching**: Uses TF-IDF vectorization and cosine similarity for accurate recommendations
- **Interactive Web App**: Streamlit-based interface with manga details and cover images
- **MangaDex Integration**: Direct links to read recommended manga on MangaDex

## How It Works

1. **Data Processing**:
   - Cleans and preprocesses manga descriptions using advanced text cleaning techniques
   - Handles multiple languages and special formatting
2. **Similarity Calculation**:
   - Uses TF-IDF vectorization to analyze manga descriptions
   - Calculates cosine similarity between manga
   - Incorporates additional features like genres, themes, and ratings
3. **Recommendation Engine**:
   - Combines similarity scores from multiple input manga
   - Applies user-selected filters
   - Ranks and presents top recommendations

## Web App Features

- **Search Functionality**: Find manga by title with instant results
- **Manga Details**: View comprehensive information including:
  - Rating
  - Status
  - Chapters
  - Genres
  - Themes
  - Description
- **Cover Images**: Display manga covers with fallback to default image
- **Interactive Filters**:
  - Genre selection
  - Theme selection
  - Spin-off inclusion/exclusion
- **Direct Links**: Quick access to read manga on MangaDex

## Dataset

The dataset was obtained from [MangadexRecomendations GitHub](https://github.com/goldbattle/MangadexRecomendations) and contains:

- 60,000 manga entries
- 20+ features including:
  - Title
  - Description
  - Rating
  - Genres
  - Themes
  - Demographic
  - Status
  - Author
  - Artist
  - URL

## Try It Out

Experience the manga recommender system through our interactive web app:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alwaleed-7-dsi-manga-recommender-app-991svj.streamlit.app/)

## References

- [Mangadex](https://mangadex.org/)
- [Similar Manga GitHub](https://github.com/similar-manga/similar)
- [MangadexRecomendations GitHub](https://github.com/goldbattle/MangadexRecomendations)

## Project Files

- [Presentation](./Presentation.pptx)
- [Model Code](./model.ipynb)
- [Web App Code](./app.py)
