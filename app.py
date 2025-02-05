import streamlit as st
import pandas as pd
from scipy.stats import uniform
from scipy.stats import randint
import re
import html
import os.path
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from functools import lru_cache
import time
import requests
from bs4 import BeautifulSoup
from surprise import Reader, Dataset, SVD
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from ratelimit import limits, sleep_and_retry
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from PIL import Image

def clean_string(str_raw, removeStopWords=False):
    # bbcodes that our description will have in it
    # https://github.com/CarlosEsco/Neko/blob/master/app/src/main/java/eu/kanade/tachiyomi/source/online/utils/MdUtil.kt
    descriptionLanguages = [
        "Russian / Русский",
        "[u]Russian",
        "[b][u]Russian",
        "[RUS]",
        "Russian / Русский",
        "Russian/Русский:",
        "Russia/Русское",
        "Русский",
        "RUS:",
        "[b][u]German / Deutsch",
        "German/Deutsch:",
        "Espa&ntilde;ol / Spanish",
        "Spanish / Espa&ntilde;ol",
        "Spanish / Espa & ntilde; ol",
        "Spanish / Espa&ntilde;ol",
        "[b][u]Spanish",
        "[Espa&ntilde;ol]:",
        "[b] Spanish: [/ b]",
        "정보",
        "Spanish/Espa&ntilde;ol",
        "Espa&ntilde;ol / Spanish",
        "Italian / Italiano",
        "Italian/Italiano",
        "\r\n\r\nItalian\r\n",
        "Pasta-Pizza-Mandolino/Italiano",
        "Persian /فارسی",
        "Farsi/Persian/",
        "Polish / polski",
        "Polish / Polski",
        "Polish Summary / Polski Opis",
        "Polski",
        "Portuguese (BR) / Portugu&ecirc;s",
        "Portuguese / Portugu&ecirc;s",
        "Português / Portuguese",
        "Portuguese / Portugu",
        "Portuguese / Portugu&ecirc;s",
        "Portugu&ecirc;s",
        "Portuguese (BR) / Portugu & ecirc;",
        "Portuguese (BR) / Portugu&ecirc;",
        "[PTBR]",
        "R&eacute;sume Fran&ccedil;ais",
        "R&eacute;sum&eacute; Fran&ccedil;ais",
        "[b][u]French",
        "French / Fran&ccedil;ais",
        "Fran&ccedil;ais",
        "[hr]Fr:",
        "French - Français:",
        "Turkish / T&uuml;rk&ccedil;e",
        "Turkish/T&uuml;rk&ccedil;e",
        "T&uuml;rk&ccedil;e",
        "[b][u]Chinese",
        "Arabic / العربية",
        "العربية",
        "[hr]TH",
        "[b][u]Vietnamese",
        "[b]Links:",
        "[b]Link[/b]",
        "Links:",
        "[b]External Links"
    ]
    englishDescriptionTags = [
        "[b][u]English:",
        "[b][u]English",
        "[English]:",
        "[B][ENG][/B]"
    ]
    bbcodes = [
        "[list]",
        "[/list]",
        "[*]",
        "[hr]",
        "[u]",
        "[/u]",
        "[b]",
        "[/b]"
    ]

    # Precompile regex patterns for better performance
    html_pattern = re.compile(r'<[^>]*>')
    url_pattern = re.compile(r'https?:\/\/\S+|http?:\/\/\S+')
    email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+')
    source_pattern = re.compile(r'\(source: [^)]*\)')
    bbcode_pattern = re.compile(r'\[.*?\]')
    symbol_pattern = re.compile(r'\n')
    
    # Predefined replacements
    replacements = {
        "isn't": "is not",
        "aren't": "are not",
        "ain't": "am not",
        "won't": "will not",
        "didn't": "did not",
        "shan't": "shall not",
        "haven't": "have not",
        "hadn't": "had not",
        "hasn't": "has not",
        "don't": "do not",
        "wasn't": "was not",
        "weren't": "were not",
        "doesn't": "does not",
        "'s": " is",
        "'re": " are",
        "'m": " am",
        "'d": " would",
        "'ll": " will"
    }

    # Stop words
    stops = {'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 
            'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 
            'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to'}

    # Clean non-English descriptions
    for tag in descriptionLanguages:
        str_raw = str_raw.split(tag, 1)[0]

    # Clean English tags
    for tag in englishDescriptionTags:
        str_raw = str_raw.replace(tag, "")

    # Basic cleaning
    str_raw = html.unescape(str_raw)
    str_raw = str_raw.lower()
    str_raw = " ".join(str_raw.split())

    # Remove bbcodes
    str_raw = bbcode_pattern.sub('', str_raw)

    # Remove source parentheses
    str_raw = source_pattern.sub('', str_raw)

    # Remove HTML tags
    str_raw = html_pattern.sub(' ', str_raw)

    # Remove URLs and emails
    str_raw = url_pattern.sub(' ', str_raw)
    str_raw = email_pattern.sub(' ', str_raw)

    # Replace contractions
    for pattern, replacement in replacements.items():
        str_raw = str_raw.replace(pattern, replacement)

    # Remove stop words if requested
    if removeStopWords:
        str_raw = ''.join(c for c in str_raw if c not in punctuation)
        str_raw = " ".join(w for w in str_raw.split() if w.lower() not in stops)

    # Final cleaning
    str_raw = symbol_pattern.sub(' ', str_raw)
    str_raw = " ".join(str_raw.split())

    return str_raw

@lru_cache(maxsize=1)
def load_data():
    df = pd.read_csv('Mangas.csv', encoding="utf8")
    df2 = pd.read_csv('Author.csv')
    df = pd.merge(df, df2, on='title')
    df = df.drop_duplicates(subset=["title", "description"], keep="first")
    df = df.reset_index(drop=True)
    
    # Clean descriptions
    list_ind = df.index[df['description'].isnull() == False].tolist()
    for i in list_ind:
        df.loc[i, 'description'] = clean_string(df.loc[i, 'description'])
    df = df.dropna(subset='description')
    return df.reset_index(drop=True)

df = load_data()

@lru_cache(maxsize=1)
def get_tfidf_matrix():
    return load_corpus_into_tfidf(df['description'])

author_score = 1.5
artist_score = 1.4
demographic_score = 1.0
theme_score = 0.8
genre_score = 0.8
format_score = 0.6
rating_weight = 0.5
status_weight = 0.3

format_manga = ['Award Winning',
 'Long Strip',
 'Oneshot',
 '4-Koma',
 'Web Comic',
 'Anthology',
 'Adaptation',
 'Full Color',
 'Official Colored',
 'Fan Colored']

theme_manga = [
 'Martial Arts',
 'Supernatural',
 'School Life',
 'Post-Apocalyptic',
 'Cooking',
 'Video Games',
 'Traditional Games',
 'Music',
 'Delinquents',
 'Magic',
 'Mafia',
 'Office Workers',
 'Military',
 'Survival',
 'Virtual Reality',
 'Police',
 'Ninja',
 'Time Travel',
 'Aliens',
 'Demons',
 'Animals',
 'Samurai',
 'Vampires',
 'Monsters',
 'Reincarnation',
 'Monster Girls',
 'Ghosts',
 'Zombies',
 'Villainess']

genre_manga = ['Action',
 'Comedy',
 'Drama',
 'Fantasy',
 'Adventure',
 'Romance',
 'Psychological',
 'Slice of Life',
 'Sports',
 'Horror',
 'Mystery',
 'Historical',
 'Tragedy',
 'Sci-Fi',
 'Mecha',
 'Medical',
 'Thriller',
 'Philosophical',
 'Crime',
 'Isekai']

def load_corpus_into_tfidf(corpus):
    # build a TF/IDF matrix for each paper
    # https://markhneedham.com/blog/2016/07/27/scitkit-learn-tfidf-and-cosine-similarity-for-computer-science-papers/
    # tf = TfidfVectorizer(strip_accents='ascii', analyzer='word', ngram_range=(1, 1),
    #                      min_df=0.10, max_df=1.0, stop_words='english', max_features=50000, sublinear_tf=False)
    tf = TfidfVectorizer(strip_accents='unicode', analyzer='word')
    x = tf.fit_transform(corpus)
    # print(tf.get_feature_names())
    # with open("./output/vocab.json", "w") as fp:
    #     json.dump(dict(zip(tf.get_feature_names(), x.toarray()[0])), fp, indent=2)
    return x

def find_similar_tfidf(tfidf_matrix, corpus_index):
    # top similar papers based on cosine similarity
    cosine_similarities = linear_kernel(tfidf_matrix[corpus_index:corpus_index + 1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != corpus_index]

    # return the matches (best matches to worst)
    tuple_vec = [(index, cosine_similarities[index]) for index in related_docs_indices]

    # convert to dictionary
    scores = {}
    for id1, score in tuple_vec:
        scores[id1] = score
    return scores

def get_recommendations(manga_title, n_recommendations=10):
    # Find the index of the selected manga
    manga_index = df[df['title'] == manga_title].index[0]
    
    # Get TF-IDF matrix
    tfidf_matrix = load_corpus_into_tfidf(df['description'])
    
    # Find similar manga
    similarity_scores = find_similar_tfidf(tfidf_matrix, manga_index)
    
    # Get top n recommendations
    recommendations = []
    for idx, score in sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]:
        manga = df.iloc[idx]
        recommendations.append({
            'title': manga['title'],
            'score': score,
            'rating': manga['rating'],
            'description': manga['description'],
            'url': manga['url'],
            'genre': manga['genre'],
            'theme': manga['theme']
        })
    
    return recommendations

# Move the search_manga function definition before it's used
def search_manga(query):
    return df[df['title'].str.contains(query, case=False)]['title'].tolist()

# Add show_manga_details function
def show_manga_details(index):
    # Ensure index is integer
    if not isinstance(index, int):
        index = int(index)
    manga = df.iloc[index]
    # Get chapter count from Author.csv
    author_data = pd.read_csv('Author.csv')
    manga_id = manga['id_y']
    chapter_count = author_data[author_data['id'] == manga_id]['chapters'].values[0]
    
    st.subheader(manga['title'])
    st.write(f"**Rating:** {manga['rating']}")
    st.write(f"**Chapters:** {chapter_count if not pd.isna(chapter_count) else 'Ongoing'}")
    st.write(f"**Demographic:** {manga['demographic']}")
    st.write(f"**Genres:** {', '.join(eval(manga['genre']))}")
    st.write(f"**Themes:** {', '.join(eval(manga['theme']))}")
    st.write(f"**Description:** {manga['description']}")
    st.markdown(f"[Read on MangaDex]({manga['url']})")

# Custom CSS for better styling
st.markdown("""
    <style>
        .manga-card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            background: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            gap: 20px;
            align-items: flex-start;
        }
        .manga-card img {
            border-radius: 5px;
            width: 150px;
            height: auto;
            flex-shrink: 0;
        }
        .manga-card .content {
            flex: 1;
            min-width: 0;
        }
        .manga-card h3 {
            margin-top: 0;
            color: #1e3a8a;
        }
        .manga-card a {
            color: #3b82f6;
            text-decoration: none;
        }
        .manga-card a:hover {
            text-decoration: underline;
        }
        .search-result {
            margin-bottom: 30px;
            display: flex;
            gap: 20px;
            align-items: flex-start;
        }
        .search-result img {
            width: 200px;
            border-radius: 5px;
            flex-shrink: 0;
        }
        .search-result .content {
            flex: 1;
            min-width: 0;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }
        .info-item {
            background: #ffffff;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .info-item strong {
            color: #1e3a8a;
            display: block;
            margin-bottom: 5px;
        }
        .info-item p {
            color: #333333;
            margin: 0;
        }
        .stMarkdown p {
            color: #333333;
        }
        .stMarkdown h2, .stMarkdown h3 {
            color: #1e3a8a;
        }
        .stMarkdown a {
            color: #3b82f6;
        }
        .info-text {
            color: #333333 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Main content
st.title("Manga Recommender System")
st.write("Discover new manga based on your favorites!")

# Search for manga
search_query = st.text_input("Search for a manga:", value="", help="Type to search for your favorite manga.")

def get_mangadex_cover(manga_id):
    """Fetch cover image from MangaDex API"""
    try:
        # MangaDex API endpoint for covers
        api_url = f"https://api.mangadex.org/cover?manga[]={manga_id}&limit=1"
        response = requests.get(api_url)
        data = response.json()
        
        if response.status_code == 200 and data['data']:
            cover_filename = data['data'][0]['attributes']['fileName']
            return f"https://uploads.mangadex.org/covers/{manga_id}/{cover_filename}"
        return None
    except Exception as e:
        print(f"Error fetching cover from MangaDex: {e}")
        return None

# Display search result
if search_query:
    search_results = df[df['title'].str.contains(search_query, case=False)]
    
    if not search_results.empty:
        # Get exact match first, then partial matches
        exact_match = df[df['title'].str.lower() == search_query.lower()]
        manga_details = exact_match.iloc[0] if not exact_match.empty else search_results.iloc[0]
        
        # Try to get cover image
        cover_url = None
        try:
            # First try local cover
            filename = str(manga_details['filename']).strip()
            if filename and filename.lower() != 'nan':
                if not filename.endswith('.jpg') and not filename.endswith('.png'):
                    filename += '.jpg'
                cover_path = f"mangas_cover/{filename}"
                if os.path.exists(cover_path):
                    cover_url = cover_path
            
            # If local cover not found, try MangaDex
            if not cover_url:
                manga_id = manga_details.get('id_y')
                if manga_id:
                    cover_url = get_mangadex_cover(manga_id)
            
            # Fallback to no image
            if not cover_url:
                cover_url = 'mangas_cover/no_img.jpg'
                
        except Exception as e:
            print(f"Error loading cover image: {e}")
            cover_url = 'mangas_cover/no_img.jpg'
        
        with st.container():
            st.markdown('<div class="search-result">', unsafe_allow_html=True)
            
            # Display manga information
            st.image(cover_url, width=200)
            
            st.markdown(f"""
                <div class="content">
                    <h2>{manga_details['title']}</h2>
                    <div class="info-grid">
                        <div class="info-item">
                            <strong>Rating:</strong><br>
                            <span class="info-text">{manga_details.get('rating', 'N/A')}</span>
                        </div>
                        <div class="info-item">
                            <strong>Status:</strong><br>
                            <span class="info-text">{manga_details.get('status', 'N/A')}</span>
                        </div>
                        <div class="info-item">
                            <strong>Genres:</strong><br>
                            <span class="info-text">{', '.join(manga_details['genre'] if isinstance(manga_details['genre'], list) else eval(manga_details.get('genre', '[]')))}</span>
                        </div>
                        <div class="info-item">
                            <strong>Themes:</strong><br>
                            <span class="info-text">{', '.join(manga_details['theme'] if isinstance(manga_details['theme'], list) else eval(manga_details.get('theme', '[]')))}</span>
                        </div>
                    </div>
                    <p><strong>Description:</strong> <span class="info-text">{manga_details.get('description', 'No description available')}</span></p>
                    <p><a href="{manga_details.get('url', '#')}" target="_blank">Read on MangaDex</a></p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No manga found with that title")

def is_spinoff(title, input_titles):
    """
    Check if a title is a spin-off of any of the input titles
    """
    title = title.lower()
    for input_title in input_titles:
        input_title = input_title.lower()
        # Check for common spin-off patterns
        patterns = [
            f"{input_title}:",
            f"{input_title} -",
            f"{input_title} (",
            f"{input_title} ~",
            f"{input_title} gaiden",
            f"{input_title} side story",
            f"{input_title} spin-off"
        ]
        if any(pattern in title for pattern in patterns):
            return True
    return False

# Recommendations section
with st.expander("Get Recommendations"):
    # Multi-select for multiple manga
    selected_manga = st.multiselect(
        "Select manga to base recommendations on:",
        df['title'].sort_values().unique(),
        help="Select one or more manga to get recommendations"
    )
    
    # Add filtering options
    col1, col2 = st.columns(2)
    with col1:
        selected_genres = st.multiselect(
            "Filter by genres:",
            genre_manga,
            help="Select genres to filter recommendations"
        )
    with col2:
        selected_themes = st.multiselect(
            "Filter by themes:",
            theme_manga,
            help="Select themes to filter recommendations"
        )
    
    # Add spin-off filter
    include_spinoffs = st.checkbox(
        "Include spin-offs",
        value=True,
        help="Include manga that are spin-offs of your selected titles"
    )
    
    if selected_manga and st.button("Show Recommendations", key="show_recommendations_button"):
        with st.spinner('Finding recommendations...'):
            start_time = time.time()
            
            # Get recommendations for each selected manga
            all_recommendations = []
            for manga_title in selected_manga:
                recommendations = get_recommendations(manga_title, 20)  # Get more recommendations for filtering
                all_recommendations.extend(recommendations)
            
            # Combine and sort recommendations
            combined_recommendations = {}
            for rec in all_recommendations:
                if rec['title'] in combined_recommendations:
                    combined_recommendations[rec['title']]['score'] += rec['score']
                else:
                    combined_recommendations[rec['title']] = rec
            
            # Convert to list and sort by combined score
            final_recommendations = sorted(
                combined_recommendations.values(),
                key=lambda x: x['score'],
                reverse=True
            )
            
            # Apply genre, theme, and spin-off filters
            if selected_genres or selected_themes or not include_spinoffs:
                filtered_recommendations = []
                for rec in final_recommendations:
                    manga_data = df[df['title'] == rec['title']].iloc[0]
                    
                    # Check genre filter
                    genres = eval(manga_data['genre']) if isinstance(manga_data['genre'], str) else manga_data['genre']
                    genre_match = not selected_genres or any(genre in genres for genre in selected_genres)
                    
                    # Check theme filter
                    themes = eval(manga_data['theme']) if isinstance(manga_data['theme'], str) else manga_data['theme']
                    theme_match = not selected_themes or any(theme in themes for theme in selected_themes)
                    
                    # Check spin-off filter
                    spinoff_match = include_spinoffs or not is_spinoff(rec['title'], selected_manga)
                    
                    if genre_match and theme_match and spinoff_match:
                        filtered_recommendations.append(rec)
                
                final_recommendations = filtered_recommendations[:10]  # Show top 10 after filtering
            
            st.session_state.recommendations = final_recommendations[:10]  # Show top 10 by default
            st.success(f"Found {len(final_recommendations)} recommendations in {time.time() - start_time:.2f} seconds")

# Display recommendations
if 'recommendations' in st.session_state:
    st.subheader("Recommended Manga")
    
    for rec in st.session_state.recommendations:
        try:
            # Get the correct manga entry from dataframe
            manga_entry = df[df['title'] == rec['title']].iloc[0]
            
            # Check if image exists
            url = f'mangas_cover/{manga_entry["filename"]}'
            if os.path.exists(url):
                result = url
            else:
                result = 'mangas_cover/no_img.jpg'
                
            # Create expandable section for each recommendation
            with st.expander(f"{rec['title']} (Rating: {rec['rating']})"):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.image(
                        result,
                        width=200,
                        use_column_width='always',
                        output_format="auto"
                    )
                
                with col2:
                    # Display detailed information
                    st.markdown(f"""
                        **Status:** {manga_entry.get('status', 'N/A')}  
                        **Chapters:** {manga_entry.get('chapters', 'N/A')}  
                        **Author:** {manga_entry.get('author', 'N/A')}  
                        **Artist:** {manga_entry.get('artist', 'N/A')}  
                        **Genres:** {', '.join(manga_entry['genre'] if isinstance(manga_entry['genre'], list) else eval(manga_entry.get('genre', '[]')))}  
                        **Themes:** {', '.join(manga_entry['theme'] if isinstance(manga_entry['theme'], list) else eval(manga_entry.get('theme', '[]')))}  
                        **Description:**  
                        {manga_entry.get('description', 'No description available')}
                    """)
                    
                    # Add external link if available
                    if manga_entry.get('url'):
                        st.markdown(f"[Read on MangaDex ↗]({manga_entry['url']})", unsafe_allow_html=True)
            
        except Exception as e:
            print(f"Error loading recommendation: {e}")
            st.image(
                'mangas_cover/no_img.jpg',
                caption=f"{rec['title']} (Rating: {rec['rating']})",
                width=400,
                use_column_width='always',
                output_format="auto"
            )
