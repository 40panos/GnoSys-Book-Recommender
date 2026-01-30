"""
Gnosys Ecosystem - Main Streamlit Application
---------------------------------------------
This is the primary user interface for the Gnosys Book Recommendation system.
It features a digital library for browsing and a custom AI Assistant powered
by GPT models for natural language book discovery.
"""

import streamlit as st
import pandas as pd
import time
import re
import streamlit.components.v1 as components
from collections import Counter

# --- CUSTOM MODULE IMPORTS ---
from data_preparation import load_datasets, prepare_tags, prepare_books
from models import demographic_filtering, build_tfidf_similarity, build_mixed_similarity
from ai_assistant import process_user_message

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==============================================================================
st.set_page_config(
    page_title="Gnosys Ecosystem",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme styling and button responsiveness
st.markdown("""
<style>
    /* Global Dark Theme Background */
    .stApp { background-color: #1a1e2e; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #2d3343;
    }
    
    /* Chat Interface Styling */
    .chat-header {
        color: white; font-size: 24px; font-weight: bold;
        padding-bottom: 20px; border-bottom: 1px solid #2d3343; margin-bottom: 20px;
    }
    .stChatMessage[data-testid="assistant"] {
         background-color: #222838 !important;
         border-radius: 12px; padding: 10px; margin-bottom: 10px; border: 1px solid #3d455c;
    }
     .stChatMessage[data-testid="user"] {
         margin-bottom: 10px; justify-content: flex-end; background-color: #2b3346;
    }
    
    /* --- CUSTOM RADIO BUTTONS (Sidebar) --- */
    div.stRadio > div[role="radiogroup"] > label > div:first-child {
        background-color: #222838 !important; 
        border: 2px solid #5c6bc0 !important;
    }
    
    div.stRadio > div[role="radiogroup"] > label > div:first-child[data-checked="true"] {
        background-color: #3949ab !important; 
        border-color: #3949ab !important;
    }
    
    div.stRadio > div[role="radiogroup"] > label p {
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #e0e0e0 !important;
    }

    /* --- LARGE CATEGORY BUTTON STYLING (Genres) --- */
    div.stButton > button p {
        font-size: 2.3rem !important; 
        font-weight: 700 !important;
        color: #e8e8e8 !important; 
        margin: 0 !important; padding: 0 !important;
    }
    div.stButton > button {
        background-color: transparent !important; 
        border: none !important;
        padding: 0 !important; margin: 0 !important; 
        text-align: left !important;
    }
    div.stButton > button:hover p {
        color: #5c6bc0 !important; 
    }

    /* --- SMALL BOOK BUTTON STYLING (Books Grid) --- */
    [data-testid="stColumn"] div.stButton > button p,
    [data-testid="column"] div.stButton > button p {
        font-size: 0.95rem !important; 
        font-weight: 600 !important;
        color: #ffffff !important; 
        text-transform: none !important;
        white-space: normal !important; 
        height: auto !important; 
        padding-top: 0px !important;
    }
    [data-testid="stColumn"] div.stButton > button:hover p,
    [data-testid="column"] div.stButton > button:hover p {
        color: #ff4b4b !important; 
        text-decoration: underline !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CACHED DATA LOADING
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    """
    Loads raw datasets and applies preprocessing logic.
    Filters out records with missing covers or descriptions for a better UI.
    """
    books, book_tags, tags, ratings = load_datasets()
    all_tags_df = prepare_tags(book_tags, tags)
    books_prepared = prepare_books(books, all_tags_df)

    # Handle image URL fallback logic
    if "image_url" in books_prepared.columns:
        books_prepared["image_final"] = books_prepared["image_url"]
    else:
        books_prepared["image_final"] = None
        
    if "small_image_url" in books_prepared.columns:
        books_prepared["image_final"] = books_prepared["image_final"].fillna(books_prepared["small_image_url"])

    # UI Optimization: Keep only books with covers and content
    books_prepared = books_prepared[books_prepared["image_final"].notna()]
    books_prepared = books_prepared[books_prepared["description"] != ""]
    books_prepared = books_prepared.reset_index(drop=True)
    return books_prepared

@st.cache_resource(show_spinner=False)
def load_similarity_matrix(df):
    """
    Builds the hybrid similarity matrix once per session to enable 
    instant 'More Like This' recommendations.
    """
    sim_desc = build_tfidf_similarity(df['description'])
    sim_tags = build_tfidf_similarity(df['all_tags'], min_df=5, max_df=0.40)
    sim_mixed = build_mixed_similarity([sim_desc, sim_tags], [0.6, 0.4])
    return sim_mixed

# Global Data Initialization
with st.spinner("🚀 Loading Gnosys..."):
    df_books = load_and_prepare_data()

# ==============================================================================
# 3. UI HELPER FUNCTIONS
# ==============================================================================
def clean_series_suffix(title: str) -> str:
    """Removes series numbering (e.g., '#1') from titles for cleaner display."""
    if not isinstance(title, str): return title
    return re.sub(r"\s*\([^)]*#\d+\)$", "", title).strip()

def get_top_similar(idx, sim_matrix, n=5):
    """Calculates top N similar items for a given index."""
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return [i[0] for i in sim_scores[1:n+1]]

def get_top_genres(books_prepared, top_k=5):
    """Extracts the most frequent genres from the dataset for the home screen."""
    all_genres = []
    if "genres" not in books_prepared.columns: return []
    for genres in books_prepared["genres"]:
        if isinstance(genres, list): all_genres.extend(genres)
    counter = Counter(all_genres)
    return [g for g, _ in counter.most_common(top_k)]

def update_genre_state(new_genre):
    """Callback to switch current genre view and trigger UI refresh."""
    st.session_state["genre_selector"] = new_genre
    st.session_state["selected_book"] = None
    st.session_state["scroll_top"] = True

@st.dialog("Book Details")
def show_book_details_modal(full_df, sim_matrix):
    """Displays a popup with detailed book info and similar recommendations."""
    if "selected_book" not in st.session_state or st.session_state["selected_book"] is None:
        return

    book = st.session_state["selected_book"]
    col_img, col_info = st.columns([1, 1.5], gap="medium")
    
    with col_img:
        img_url = book.get("image_final", None)
        if pd.notna(img_url): st.image(img_url, use_container_width=True)
        else: st.write("(No Cover)")
            
    with col_info:
        raw_title = str(book.get("title", "Unknown"))
        st.subheader(clean_series_suffix(raw_title))
        
        authors_val = book.get("authors", None)
        if isinstance(authors_val, list): st.markdown(f"**✍️ Author:** {', '.join([str(a) for a in authors_val])}")
        elif isinstance(authors_val, str): st.markdown(f"**✍️ Author:** {authors_val}")
            
        avg = book.get("average_rating", None)
        if pd.notna(avg): st.markdown(f"**⭐ Rating:** {avg:.2f}")

    st.divider()
    desc = book.get("description", None)
    if pd.notna(desc):
        st.write(re.sub('<[^<]+?>', '', str(desc)))

    st.divider()
    st.subheader("💡 You might also like:")
    
    try: current_idx = book.name
    except: current_idx = None
    
    if current_idx is not None:
        rec_indices = get_top_similar(current_idx, sim_matrix, n=5)
        recommendations = full_df.iloc[rec_indices]
        rec_cols = st.columns(5)
        for col, (_, rec_book) in zip(rec_cols, recommendations.iterrows()):
            with col:
                r_img = rec_book.get("image_final")
                if pd.notna(r_img): st.image(r_img, use_container_width=True)
                r_title = clean_series_suffix(str(rec_book.get("title", "")))
                short = r_title[:25] + "..." if len(r_title)>25 else r_title
                
                if st.button(short, key=f"rec_{current_idx}_{rec_book.name}"):
                    st.session_state["selected_book"] = rec_book
                    st.rerun()

def show_books_grid(title, books_df, max_books=50, cols_per_row=5, key_prefix="grid"):
    """Displays a grid of clickable book covers."""
    if books_df.empty: return
    if title: st.subheader(title)
    df = books_df.head(max_books).reset_index(drop=True)
    
    for i in range(0, len(df), cols_per_row):
        row = df.iloc[i:i + cols_per_row]
        cols = st.columns(len(row))
        for col, (_, book) in zip(cols, row.iterrows()):
            with col:
                img_url = book.get("image_final", None)
                if pd.notna(img_url): st.image(img_url, use_container_width=True)
                raw_title = str(book.get("title", "Unknown"))
                bt = clean_series_suffix(raw_title)
                
                if st.button(bt, key=f"{key_prefix}_{book.get('book_id', i)}_{bt[:5]}"):
                    st.session_state["selected_book"] = book
                    st.rerun()
                
                avg = book.get("average_rating", None)
                if pd.notna(avg): st.caption(f"⭐ {avg:.2f}")

# ==============================================================================
# 4. PAGE RENDERING LOGIC
# ==============================================================================
def render_library_page():
    """Renders the Digital Library browsing interface."""
    st.markdown("<h1 style='text-align: center; color:white;'>📚 Digital Library</h1>", unsafe_allow_html=True)
    
    sim_matrix = load_similarity_matrix(df_books)

    # State initialization
    if "selected_book" not in st.session_state: st.session_state["selected_book"] = None
    if "genre_selector" not in st.session_state: st.session_state["genre_selector"] = "All"

    # Scroll anchor logic
    st.markdown("<a id='top'></a>", unsafe_allow_html=True)
    if st.session_state.get("scroll_top", False):
        components.html("<meta http-equiv='refresh' content='0; url=#top'>", height=0, width=0)
        st.session_state["scroll_top"] = False

    # Dynamic Genre List
    all_genres = set()
    if "genres" in df_books.columns:
        for lst in df_books["genres"]:
            if isinstance(lst, list): all_genres.update(lst)
    sorted_genres = sorted(all_genres)
    display_genres = ["All"] + [g.replace("-", " ").title() for g in sorted_genres]

    # Search & Category Toolbar
    col_search, col_sel = st.columns([3, 1])
    with col_search:
        search_query = st.text_input("🔍 Search Library", placeholder="Type book title...")
    
    with col_sel:
        def on_dropdown_change(): st.session_state["selected_book"] = None
        selected_display = st.selectbox("Category", options=display_genres, key="genre_selector", on_change=on_dropdown_change)

    # Content Display Routing
    if search_query:
        mask = df_books["title"].str.contains(search_query, case=False, na=False)
        results = df_books[mask]
        if results.empty: st.warning("No books found.")
        else: show_books_grid(f"Search Results: '{search_query}'", results, key_prefix="search")

    elif selected_display == "All":
        top_idx = demographic_filtering(df_books, quantile=0.70)
        top_books = df_books.iloc[top_idx]
        show_books_grid("🔥 Trending Now (Top 10)", top_books, max_books=10, cols_per_row=10, key_prefix="top10")
        
        st.markdown("---")
        st.header("🎬 Popular Genres")
        top_gs = get_top_genres(df_books, 5)
        shown_ids = set(top_books.index)
        
        for g in top_gs:
            label = g.replace("-", " ").title()
            st.button(label, key=f"btn_h_{g}", on_click=update_genre_state, args=(label,))
            mask = df_books["genres"].apply(lambda l: isinstance(l, list) and g in l)
            gb = df_books[mask].copy()
            gb = gb[~gb.index.isin(shown_ids)].head(10)
            if not gb.empty:
                show_books_grid(None, gb, max_books=10, cols_per_row=10, key_prefix=f"row_{g}")
                shown_ids.update(gb.index)
    else:
        # Genre-specific view
        idx = display_genres.index(selected_display) - 1
        raw_genre = sorted_genres[idx]
        mask = df_books["genres"].apply(lambda l: isinstance(l, list) and raw_genre in l)
        gb = df_books[mask].copy()
        
        if gb.empty: st.info("No books found.")
        else:
            top_g_idx = demographic_filtering(gb, quantile=0.50)
            gb_sorted = gb.loc[top_g_idx]
            show_books_grid(f"📖 {selected_display} Collection", gb_sorted, max_books=50, cols_per_row=5, key_prefix="spec_genre")

    if st.session_state["selected_book"] is not None:
        show_book_details_modal(df_books, sim_matrix)

def render_assistant_page():
    """Renders the AI chat interface for natural language discovery."""
    st.markdown('<div class="chat-header">🤖 Gnosys AI Librarian</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am Gnosys. Tell me what you're in the mood for, and I'll find the perfect book for you!"}
        ]

    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "image_url" in message and message["image_url"]:
                st.image(message["image_url"], width=150)

    if prompt := st.chat_input("Ask Gnosys..."):
        st.chat_message("user", avatar="👤").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Gnosys is searching the stacks..."):
            result = process_user_message(prompt, df_books)
            response_text = result["text"]
            book_obj = result["book"]

        img_link = book_obj.get('image_final') if book_obj is not None else None

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response_text)
            if pd.notna(img_link): st.image(img_link, width=150)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "image_url": img_link if pd.notna(img_link) else None
        })

# ==============================================================================
# 5. MAIN ENTRY POINT
# ==============================================================================
def main():
    with st.sidebar:
        try: st.image("logo.png", use_container_width=True)
        except: pass
        
        st.markdown("---")
        st.subheader("Navigation")
        page = st.radio("Go to", ["Library", "AI Assistant"], label_visibility="collapsed")

    # App Routing
    if page == "Library": render_library_page()
    elif page == "AI Assistant": render_assistant_page()

if __name__ == "__main__":
    main()