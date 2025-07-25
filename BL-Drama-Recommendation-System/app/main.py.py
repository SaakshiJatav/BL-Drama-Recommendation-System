import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process, fuzz
import streamlit as st

# =========================
# Load & Prepare Dataset
# =========================
df = pd.read_csv("BL_Drama_Recommendation.csv")

# Clean columns
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r"[()]", "", regex=True)

# Ensure numeric ratings
df["Personal_rating_out_of_10"] = pd.to_numeric(df["Personal_rating_out_of_10"], errors="coerce").fillna(0)

# Fill missing values
for col in ["Genres", "Mood_Tags", "Summary", "Main_Leads"]:
    if col in df.columns:
        df[col] = df[col].fillna("Not specified")

# =========================
# Emoji Mapping for Genres
# =========================
genre_emoji = {
    "Romance": "❤", "Drama": "🎭", "Comedy": "😂", "Medical": "🩺",
    "Action": "🔥", "Music": "🎵", "Office": "🏢", "School": "🏫",
    "Supernatural": "👻", "Sci-Fi": "👽", "Business": "💼", "Historical": "🏰",
    "Thriller": "😱", "Crime": "🕵", "Youth": "🧒", "Fantasy": "🦄",
    "Mystery": "🔍", "Life": "🌱", "Food": "🍜", "Sports": "⚽"
}

def add_emojis(text):
    if pd.isna(text):
        return "Not specified"
    return ", ".join([f"{genre_emoji.get(word.strip(), '')} {word.strip()}" for word in text.split(",")])

df["Genres"] = df["Genres"].apply(add_emojis)

# =========================
# Build TF-IDF Similarity
# =========================
df["Combined_Features"] = (
    df["Genres"].fillna('') + " " +
    df["Mood_Tags"].fillna('') + " " +
    df["Summary"].fillna('') + " " +
    df["Main_Leads"].fillna('')
)
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Combined_Features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# =========================
# Helper Functions
# =========================
def find_best_match(title):
    choices = df["Title"].tolist()
    match, score = process.extractOne(title, choices)
    return match if score >= 40 else None

# --- Improved Recommendation Function ---
def recommend(query=None, num_recommendations=5):
    query = str(query).strip().lower()
    if not query:
        return "Please enter a drama name or keyword.", []

    # Search in Title, Genres, and Mood Tags
    search_space = df[['Title', 'Genres', 'Mood_Tags']].fillna('').astype(str).apply(lambda x: ' '.join(x).lower(), axis=1)
    matches = [(i, fuzz.partial_ratio(query, text)) for i, text in enumerate(search_space)]
    matches = sorted(matches, key=lambda x: x[1], reverse=True)

    # If strong match (likely a drama title)
    if matches and matches[0][1] > 80:
        idx = matches[0][0]
        drama_title = df.iloc[idx]['Title']
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        recommended_indices = [i[0] for i in sim_scores]
        header = f"If you liked <b>{drama_title}</b>, you might also enjoy:"
    else:
        # No strong match → treat as keyword search
        filtered = df[search_space.str.contains(query)]
        if filtered.empty:
            return "No dramas found for your search. Try another keyword!", []
        recommended_indices = filtered.index[:num_recommendations]
        header = "Here are some dramas similar to what you searched for:"

    recommendations = [df.iloc[i].to_dict() for i in recommended_indices]
    return header, recommendations

def get_top_rated(page=1, per_page=5):
    sorted_df = df.sort_values("Personal_rating_out_of_10", ascending=False)
    start, end = (page-1)*per_page, page*per_page
    return sorted_df.iloc[start:end]

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="BL Drama Recommendation System", layout="wide")
st.markdown("<h1 style='text-align:center; color: #FF69B4;'>🌈 BL Drama Recommendation System</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; font-size:18px;'>Search for your favorite Thai BL dramas and discover similar ones!</p>", unsafe_allow_html=True)

# Search input
user_input = st.text_input("Enter a BL drama name")
num_recs = st.slider("Number of recommendations:", 1, 10, 5)

# CSS Styling for cards + hover effect
st.markdown("""
    <style>
        .recommend-card {
            background-color: #2C2C2C; 
            padding: 15px; 
            border-radius: 10px; 
            margin: 10px 0; 
            color: white;
            transition: transform 0.3s ease, background-color 0.3s ease;
        }
        .recommend-card:hover {
            background-color: #3D3D3D;
            transform: scale(1.02);
        }
        .card-title {
            color: #FF69B4; 
            font-size: 20px; 
            font-weight: bold;
        }
        .section-header {
            color: #FFA500; 
            font-size: 24px; 
            font-weight: bold; 
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# Show Recommendations
# =========================
if st.button("Recommend"):
    if user_input.strip():
        header, recs = recommend(user_input, num_recs)
        if not recs:
            st.warning(header)  # Show warning like "No dramas found..."
        else:
            st.markdown(f"<h3 class='section-header'>{header}</h3>", unsafe_allow_html=True)

            for r in recs:
                st.markdown(f"""
                    <div class="recommend-card">
                        <div class="card-title">{r['Title']}</div>
                        <b>Genres:</b> {r['Genres']}<br>
                        <b>Mood Tags:</b> {r['Mood_Tags']}<br>
                        <b>Year:</b> {r['Year']}<br>
                        <b>Main Leads:</b> {r['Main_Leads']}<br>
                        <b>Rating:</b> {r['Personal_rating_out_of_10']}/10<br>
                        <b>Summary:</b> {r['Summary']}
                    </div>
                """, unsafe_allow_html=True)

# =========================
# Top Rated Dramas Browser
# =========================
st.markdown("<h3 class='section-header'>🔥 Browse Top-Rated Dramas</h3>", unsafe_allow_html=True)
page = st.number_input("Page:", min_value=1, max_value=(len(df)//5)+1, value=1)
top_rated = get_top_rated(page, 5)

for _, r in top_rated.iterrows():
    st.markdown(f"""
        <div class="recommend-card">
            <div class="card-title">{r['Title']}</div>
            <b>Genres:</b> {r['Genres']}<br>
            <b>Mood Tags:</b> {r['Mood_Tags']}<br>
            <b>Year:</b> {r['Year']}<br>
            <b>Main Leads:</b> {r['Main_Leads']}<br>
            <b>Rating:</b> {r['Personal_rating_out_of_10']}/10<br>
            <b>Summary:</b> {r['Summary']}
        </div>
    """, unsafe_allow_html=True)
    
    # =========================
# Footer
# =========================
st.markdown("""
    <div style='text-align:center; color:#888; font-size:14px; margin-top:40px;'>
        Made with ❤ by <b>Saakshi Jatav</b><br>
        <a href='https://github.com/SaakshiJatav' target='_blank' style='color:#f97316; text-decoration:none;'>GitHub</a> |
        <a href='https://www.linkedin.com/in/saakshi-jatav' target='_blank' style='color:#f97316; text-decoration:none;'>LinkedIn</a><br>
        Powered by <b>Python & Streamlit</b>
    </div>
""", unsafe_allow_html=True)