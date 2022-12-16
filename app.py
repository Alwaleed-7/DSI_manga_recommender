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
df = pd.read_csv('Mangas.csv', encoding="utf8")
df2 = pd.read_csv('Author.csv')
df = pd.merge(df, df2, on='title')
df = df.drop_duplicates(subset=["title", "description"], keep="first")
df = df.reset_index(drop = True)


author_score = 1.5
artist_score = 1.4
demographic_score = 1.0
theme_score = 0.8
genre_score = 0.8
format_score = 0.6

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

    # remove all non-english descriptions
    # this assumes the english one is first
    for tag in descriptionLanguages:
        str_raw = str_raw.split(tag, 1)[0]

    # now remove all english tags which are no longer needed
    for tag in englishDescriptionTags:
        str_raw = str_raw.replace(tag, "")

    # convert all works to lower case
    # also remove multiple white space and replace with single
    str_raw = html.unescape(str_raw)
    str_raw = str_raw.lower()
    str_raw = " ".join(str_raw.split())

    # run a second time now, but with all lower case
    # for tag in descriptionLanguages:
    #     str_raw = str_raw.split(tag.lower(), 1)[-1]
    # for tag in englishDescriptionTags:
    #     str_raw = str_raw.replace(tag.lower(), "")

    # next clean the string from any bbcodes
    for tag in bbcodes:
        str_raw = str_raw.replace(tag, "")
    str_raw = re.sub('\[.*?]', '', str_raw)

    # remove source parentheses typical of anilist
    # Eg: (source: solitarycross), (source: eat manga)
    str_raw = re.sub(r'\(source: [^)]*\)', '', str_raw)

    # remove any html codes
    str_raw = re.sub(r'<[^>]*>', r' ', str_raw)

    # remove emails and urls
    str_raw = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', str_raw, flags=re.MULTILINE)
    str_raw = re.sub(r'^http?:\/\/.*[\r\n]*', ' ', str_raw, flags=re.MULTILINE)
    str_raw = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', str_raw, flags=re.MULTILINE)

    # Replace apostrophes with standard lexicons
    str_raw = str_raw.replace("isn't", "is not")
    str_raw = str_raw.replace("aren't", "are not")
    str_raw = str_raw.replace("ain't", "am not")
    str_raw = str_raw.replace("won't", "will not")
    str_raw = str_raw.replace("didn't", "did not")
    str_raw = str_raw.replace("shan't", "shall not")
    str_raw = str_raw.replace("haven't", "have not")
    str_raw = str_raw.replace("hadn't", "had not")
    str_raw = str_raw.replace("hasn't", "has not")
    str_raw = str_raw.replace("don't", "do not")
    str_raw = str_raw.replace("wasn't", "was not")
    str_raw = str_raw.replace("weren't", "were not")
    str_raw = str_raw.replace("doesn't", "does not")
    str_raw = str_raw.replace("'s", " is")
    str_raw = str_raw.replace("'re", " are")
    str_raw = str_raw.replace("'m", " am")
    str_raw = str_raw.replace("'d", " would")
    str_raw = str_raw.replace("'ll", " will")

    # now clean stop words which are not helpful
    # we want to basically just collect a bunch of words
    stops = ['the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'which', 'this', 'that', 'these',
             'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while',
             'during', 'to']

    # Remove punctuation and stop words
    if removeStopWords:
        str_raw = ''.join([c for c in str_raw if c not in punctuation])
        str_raw = " ".join([w for w in str_raw.split() if w.lower() not in stops])

    # Remove all symbols (clean to normal english)
    # str_raw = re.sub(r'[^A-Za-z0-9\s]', r' ', str_raw)
    str_raw = re.sub(r'\n', r' ', str_raw)
    # str_raw = re.sub(r'[0-9]', r'', str_raw)
    str_raw = " ".join(str_raw.split())

    # return the final cleaned string
    return str_raw

list_ind = df.index[df['description'].isnull() == False].tolist()
for i in list_ind:
    df['description'][i] = clean_string(df['description'][i])
df = df.dropna(subset='description')
df = df.reset_index(drop=True)

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


# Title
st.header("Manga Recommender System")

options = df['title'].unique().tolist()
manga = st.selectbox("Enter manga name ", options)
status = st.checkbox("Only shows completed ")
st.write('The state of the checkbox', status)
colored = st.checkbox("Only shows colored ")
st.write('The state of the checkbox', colored)


# If button is pressed
if st.button("Submit"):
    my_matrix = load_corpus_into_tfidf(df['description'])
    ignore_tfidf_score_above_this_val = 0.75
    ignore_tfidf_score_below_this_val = 0.05

    def similarity_score(manga, status, colored):
        manga_ind = df.index[df['title'] == manga].tolist()[0]
        scores = []
        tf_idf = find_similar_tfidf(my_matrix, manga_ind)
        for j in range(len(df)):
            if j != manga_ind:
                score = 0
                if df['author'][manga_ind] == df['author'][j]:
                    score += author_score
                if df['artist'][manga_ind] == df['artist'][j]:
                    score += artist_score
                for k in format_manga:
                    if df[k][manga_ind] == df[k][j]:
                        score += format_score
                for m in theme_manga:
                    if df[m][manga_ind] == df[m][j]:
                        score += theme_score
                for t in genre_manga:
                    if df[t][manga_ind] == df[t][j]:
                        score += genre_score
                if ignore_tfidf_score_above_this_val <tf_idf[j] < ignore_tfidf_score_below_this_val:
                    score += tf_idf[j] 
                score += df['rating'][j] - df['rating'][manga_ind]

                if status:
                    if df['status'][j] == 'completed':
                        scores.append([score, df['title'][j], df['status'][j], df['filename'][j], df['id_y'][j], df['rating'][j], j])
                else:
                    scores.append([score, df['title'][j], df['status'][j], df['filename'][j], df['id_y'][j], df['rating'][j], j])

                if colored:
                    if df['Full Color'][j] and df['Official Colored'][j] == 'False':
                        if [score, df['title'][j], df['status'][j], df['filename'][j], df['id_y'][j], df['rating'][j], j] in scores:
                            scores.remove([score, df['title'][j], df['status'][j], df['filename'][j], df['id_y'][j], df['rating'][j], j])

        scores.sort()
        scores.reverse()

        return scores[0:10]
    result = similarity_score(manga, status=status, colored=colored)
    manga_ind = df.index[df['title'] == manga].tolist()[0]
    img_filename = df['filename'][manga_ind]
    # st.image(f'mangas_cover/{img_filename}', width=1, use_column_width='always', clamp=False, channels="RGB", output_format="JPEG")
    st.text(f'These are mangas similar to {manga}:', )
    for i in result:
        recommendation = i[3]
        
        ind = i[4]
        url = f'mangas_cover/{recommendation}'
        if os.path.exists(url) == True:
            result = url
            st.image(result, caption = f'{i[1]} ({i[2]}) Rating: {i[5]}', width=400, use_column_width='always', output_format="auto")
        else:
            result = 'mangas_cover/no_img.jpg'
            st.image(result, caption = f'{i[1]} ({i[2]}) Rating: {i[5]}', width=400, use_column_width='always', output_format="auto")


