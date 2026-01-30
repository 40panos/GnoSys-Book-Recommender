"""
Gnosys Library UI - Main Catalog Interface
------------------------------------------
This module defines the primary user interface for browsing the digital library.
It implements the book grid layout, category filtering, search functionality, 
and an interactive modal for viewing book details and related recommendations.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
from collections import Counter

# --- CUSTOM MODULE IMPORTS ---
from data_preparation import (
    load_datasets,
    prepare_tags,
    prepare_books,
)
from models import (
    demographic_filtering,
    build_tfidf_similarity,
    build_mixed_similarity  # Updated to reflect project's mixed similarity logic
)


# ==============================================================================
# 1. UI HELPER FUNCTIONS
# ==============================================================================

def clean_series_suffix(title: str) -> str:
    """Removes technical series markers from book titles for a cleaner UI."""
    if not isinstance(title, str):
        return title
    return re.sub(r"\s*\([^)]*#\d+\)$", "", title).strip()


def get_top_similar(idx, sim_matrix, n=5):
    """
    Retrieves the top N most similar items from the similarity matrix.
    
    Args:
        idx (int): The index of the currently selected book.
        sim_matrix (np.ndarray): Precomputed similarity matrix.
        n (int): Number of recommendations to return.
    """
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Skip index 0 as it refers to the item itself
    top_indices = [i[0] for i in sim_scores[1:n+1]]
    
    return top_indices


# ==============================================================================
# 2. INTERACTIVE MODAL (BOOK DETAILS)
# ==============================================================================

@st.dialog("Book Details")
def show_book_details_modal(full_df, sim_matrix):
    """
    Displays a pop-up dialog with book information and content-based recommendations.
    Uses st.session_state to track the currently selected book.
    """
    if "selected_book" not in st.session_state or st.session_state["selected_book"] is None:
        st.error("No book selected.")
        return

    book = st.session_state["selected_book"]

    # --- Header: Cover and Basic Info ---
    col_img, col_info = st.columns([1, 1.5], gap="medium")
    
    with col_img:
        img_url = book.get("image_final", None)
        if pd.notna(img_url):
            st.image(img_url, use_container_width=True)
        else:
            st.write("(No Cover Available)")
            
    with col_info:
        raw_title = str(book.get("title", "Unknown Title"))
        st.subheader(clean_series_suffix(raw_title))
        
        authors_val = book.get("authors", None)
        if isinstance(authors_val, list):
            st.markdown(f"**✍️ Author:** {', '.join([str(a) for a in authors_val])}")
        elif isinstance(authors_val, str):
            st.markdown(f"**✍️ Author:** {authors_val}")
            
        genres_val = book.get("genres", None)
        if isinstance(genres_val, list) and len(genres_val) > 0:
            st.markdown(f"**📚 Genres:** {', '.join(genres_val[:4])}")
            
        avg = book.get("average_rating", None)
        if pd.notna(avg):
            st.markdown(f"**⭐ Rating:** {avg:.2f}")

    st.divider()
    
    # --- Body: Description ---
    desc = book.get("description", None)
    if pd.notna(desc) and str(desc).strip():
        # Strip HTML tags from raw description strings
        clean_desc = re.sub('<[^<]+?>', '', str(desc))
        st.write(clean_desc)
    else:
        st.caption("No description available.")

    # --- Footer: Dynamic Recommendations ---
    st.divider()
    st.subheader("💡 You might also like:")

    try:
        # Locate the current book's index for similarity lookup
        current_idx = book.name
    except:
        current_id = book.get("goodreads_book_id")
        idx_list = full_df.index[full_df["goodreads_book_id"] == current_id].tolist()
        current_idx = idx_list[0] if idx_list else None

    if current_idx is None:
        st.warning("Could not load recommendations.")
        return

    try:
        rec_indices = get_top_similar(current_idx, sim_matrix, n=5)
        
        if not rec_indices:
            st.info("No similar items found.")
        else:
            recommendations = full_df.iloc[rec_indices]
            rec_cols = st.columns(5)
            
            for i, (col, (_, rec_book)) in enumerate(zip(rec_cols, recommendations.iterrows())):
                with col:
                    rec_img = rec_book.get("image_final", None)
                    if pd.notna(rec_img):
                        st.image(rec_img, use_container_width=True)

                    rec_title = clean_series_suffix(str(rec_book.get("title", "")))
                    short_title = rec_title[:30] + "..." if len(rec_title) > 30 else rec_title
                    
                    # Clicking a recommendation updates session state and refreshes the modal
                    if st.button(short_title, key=f"rec_{current_idx}_{rec_book.name}"):
                        st.session_state["selected_book"] = rec_book
                        st.rerun()

                    st.caption(f"⭐ {rec_book.get('average_rating', 0):.1f}")

    except Exception as e:
        st.error(f"Error loading recommendations: {e}")


# ==============================================================================
# 3. DATA LOADING & RESOURCE INITIALIZATION
# ==============================================================================

@st.cache_data(show_spinner=True)
def load_and_prepare_books():
    """Initializes the book dataset for the UI, ensuring covers and indices are ready."""
    books, book_tags, tags, ratings = load_datasets()
    all_tags_df = prepare_tags(book_tags, tags)
    books_prepared = prepare_books(books, all_tags_df)

    # Prioritize standard image URL, fallback to small_image_url
    if "image_url" in books_prepared.columns:
        books_prepared["image_final"] = books_prepared["image_url"]
    else:
        books_prepared["image_final"] = None

    if "small_image_url" in books_prepared.columns:
        books_prepared["image_final"] = books_prepared["image_final"].fillna(
            books_prepared["small_image_url"]
        )

    # Filter out items that cannot be displayed properly
    books_prepared = books_prepared[books_prepared["image_final"].notna()]
    books_prepared = books_prepared.reset_index(drop=True)
    return books_prepared


@st.cache_resource(show_spinner=True)
def load_sim_matrix(df):
    """Computes a Hybrid Similarity Matrix (60% Description, 40% Tags) for the UI."""
    sim_desc = build_tfidf_similarity(df['description'])
    sim_tags = build_tfidf_similarity(df['all_tags'], min_df=5, max_df=0.40)
    sim_mixed = build_mixed_similarity([sim_desc, sim_tags], [0.6, 0.4])
    return sim_mixed


def get_top_genres(books_prepared, top_k=5):
    """Calculates the most frequent genres to display on the landing page."""
    all_genres = []
    if "genres" not in books_prepared.columns:
        return []
    for genres in books_prepared["genres"]:
        if isinstance(genres, list):
            all_genres.extend(genres)
    counter = Counter(all_genres)
    return [g for g, _ in counter.most_common(top_k)]


# ==============================================================================
# 4. GRID DISPLAY ENGINE
# ==============================================================================

def show_books_grid(title, books_df, max_books=50, cols_per_row=5, key_prefix="grid"):
    """Renders a dynamic grid of book covers with interactive buttons."""
    if books_df.empty:
        st.warning("No books found.")
        return

    if title:
        st.subheader(title)

    df = books_df.head(max_books).reset_index(drop=True)
    n = len(df)

    for i in range(0, n, cols_per_row):
        row = df.iloc[i:i + cols_per_row]
        cols = st.columns(len(row))

        for col, (_, book) in zip(cols, row.iterrows()):
            with col:
                img_url = book.get("image_final", None)
                if pd.notna(img_url):
                    st.image(img_url, use_container_width=True)

                raw_title = str(book.get("title", "Unknown Title"))
                bt = clean_series_suffix(raw_title)
                
                # Selecting a book opens the detail modal
                if st.button(bt, key=f"{key_prefix}_btn_{book.get('book_id', i)}_{bt[:5]}"):
                    st.session_state["selected_book"] = book
                    st.rerun()

                avg = book.get("average_rating", None)
                if pd.notna(avg):
                    st.caption(f"⭐ {avg:.2f}")


# ==============================================================================
# 5. MAIN UI ENTRY POINT
# ==============================================================================

def reset_selection():
    """Utility to clear book selection when changing views."""
    st.session_state["selected_book"] = None

def main():
    st.set_page_config(page_title="Gnosys Library", page_icon="📚", layout="wide")

    # Global CSS injection for professional library aesthetics
    st.markdown("""
    <style>
    /* Category Headers */
    div.stButton > button p {
        font-size: 2.3rem !important; font-weight: 700 !important;
        color: #e8e8e8!important; margin: 0 !important; padding: 0 !important;
    }
    div.stButton > button {
        background-color: transparent !important; border: none !important;
        padding: 0 !important; margin: 0 !important; text-align: left !important;
    }
    div.stButton > button:hover p { color: #ffffff !important; }

    /* Book Titles in Grid */
    [data-testid="stColumn"] div.stButton > button p {
        font-size: 0.95rem !important; font-weight: 600 !important;
        color: #ffffff !important; white-space: normal !important; height: auto !important;
    }
    [data-testid="stColumn"] div.stButton > button:hover p {
        color: #ff4b4b !important; text-decoration: underline !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if "selected_book" not in st.session_state:
        st.session_state["selected_book"] = None

    # Handle automatic scrolling to top
    st.markdown("<a id='top'></a>", unsafe_allow_html=True)
    if st.session_state.get("scroll_top", False):
        components.html("<meta http-equiv='refresh' content='0; url=#top'>", height=0, width=0)
        st.session_state["scroll_top"] = False

    st.markdown("<h1 style='text-align: center;'>📚 Gnosys Digital Library</h1>", unsafe_allow_html=True)

    # --- Resources Initialization ---
    books_prepared = load_and_prepare_books()
    sim_matrix = load_sim_matrix(books_prepared)

    # --- Sidebar & Navigation ---
    all_genres = set()
    if "genres" in books_prepared.columns:
        for lst in books_prepared["genres"]:
            if isinstance(lst, list): all_genres.update(lst)
    all_genres = sorted(all_genres)
    display_genres = ["All"] + [g.replace("-", " ").title() for g in all_genres]

    # --- Search & Filter Bar ---
    col_search, col_sel = st.columns([3, 1])

    with col_search:
        search_query = st.text_input("🔍 Search Titles", placeholder="Enter book title...")

    with col_sel:
        if "selected_genre_from_click" in st.session_state:
            clicked = st.session_state["selected_genre_from_click"]
            default_index = display_genres.index(clicked) if clicked in display_genres else 0
            del st.session_state["selected_genre_from_click"]
        else:
            default_index = 0
            
        selected_display = st.selectbox("Browse Category", options=display_genres, index=default_index, on_change=reset_selection)

    # --- Dynamic Content Loading ---
    if search_query:
        mask = books_prepared["title"].str.contains(search_query, case=False, na=False)
        results = books_prepared[mask]
        show_books_grid(f"Search Results for: '{search_query}'", results, key_prefix="search_results")

    elif selected_display == "All":
        top_indices = demographic_filtering(books_prepared, quantile=0.70)
        show_books_grid("🔥 Trending Now (Global Top 10)", books_prepared.iloc[top_indices], max_books=10, cols_per_row=10, key_prefix="top_10_main")

        st.markdown("---")
        st.header("🎬 Popular Genres")
        top_genres = get_top_genres(books_prepared, top_k=5)
        for g in top_genres:
            label = g.replace("-", " ").title()
            if st.button(label, key=f"genre_btn_{g}"):
                st.session_state["selected_genre_from_click"] = label
                st.session_state["scroll_top"] = True
                reset_selection() 
                st.rerun()

            mask = books_prepared["genres"].apply(lambda lst: isinstance(lst, list) and g in lst)
            genre_top = books_prepared[mask].sort_values("ratings_count", ascending=False).head(10)
            show_books_grid(None, genre_top, max_books=10, cols_per_row=10, key_prefix=f"genre_row_{g}")
    
    else:
        # Filter by specific selected genre
        idx = display_genres.index(selected_display) - 1
        selected_genre = all_genres[idx]
        mask = books_prepared["genres"].apply(lambda lst: isinstance(lst, list) and selected_genre in lst)
        genre_books = books_prepared[mask].copy()
        
        top_genre_indices = demographic_filtering(genre_books, quantile=0.50)
        show_books_grid(f"📖 {selected_display} Collection", genre_books.loc[top_genre_indices], max_books=50, cols_per_row=5, key_prefix="genre_page")

    # Display Modal if a book is selected
    if st.session_state["selected_book"] is not None:
        show_book_details_modal(books_prepared, sim_matrix)


if __name__ == "__main__":
    main()