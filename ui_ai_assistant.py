"""
Gnosys AI Assistant UI - Standalone/Integrated Component
-------------------------------------------------------
This module defines the dedicated Chat Interface for the Gnosys AI Librarian.
It manages user session states, message history, and UI elements for a 
seamless conversational book discovery experience.
"""

import streamlit as st
import pandas as pd
import time

# --- CUSTOM IMPORTS ---
# Connecting the UI to the data pipeline and the AI logic
from data_preparation import load_datasets, prepare_tags, prepare_books
from ai_assistant import process_user_message

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Gnosys AI Librarian",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional dark-themed conversational UI
st.markdown("""
<style>
    .stApp { background-color: #1a1e2e; }
    [data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #2d3343;
    }
    [data-testid="stSidebarNav"] { display: none; }
    
    /* User Profile Box in Sidebar */
    .user-info-box {
        background-color: #222838; padding: 15px; border-radius: 12px;
        color: white; display: flex; align-items: center;
    }
    .user-avatar { font-size: 30px; margin-right: 15px; }
    .user-name { font-weight: bold; font-size: 16px; }
    .user-status { color: #4caf50; font-size: 13px; }
    
    /* Header Styling */
    .chat-header {
        color: white; font-size: 24px; font-weight: bold;
        padding-bottom: 20px; border-bottom: 1px solid #2d3343; margin-bottom: 20px;
    }
    
    /* Message Bubble Customization */
    .stChatMessage[data-testid="assistant"] {
         background-color: #222838 !important;
         border-radius: 12px; padding: 10px; margin-bottom: 10px;
    }
     .stChatMessage[data-testid="user"] {
         margin-bottom: 10px; justify-content: flex-end;
    }
    
    /* Chat Input Styling */
    [data-testid="stChatInput"] { background-color: #1a1e2e; border-top: 1px solid #2d3343; }
    [data-testid="stChatInput"] textarea {
        background-color: #222838; color: white; border: 1px solid #3d455c; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA INITIALIZATION
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_data_for_chat():
    """
    Initializes data required for the chat interface.
    Ensures that book images and descriptions are pre-processed for display.
    """
    books, book_tags, tags, ratings = load_datasets()
    all_tags_df = prepare_tags(book_tags, tags)
    books_prepared = prepare_books(books, all_tags_df)

    # Resolve image URL priorities
    if "image_url" in books_prepared.columns:
        books_prepared["image_final"] = books_prepared["image_url"]
    else:
        books_prepared["image_final"] = None
        
    if "small_image_url" in books_prepared.columns:
        books_prepared["image_final"] = books_prepared["image_final"].fillna(books_prepared["small_image_url"])

    # Filter out records missing vital UI components
    books_prepared = books_prepared[books_prepared["image_final"].notna()]
    books_prepared = books_prepared[books_prepared["description"] != ""]
    books_prepared = books_prepared.reset_index(drop=True)
    return books_prepared

# Load pre-processed books dataset
df_books = load_data_for_chat()

# ==============================================================================
# 3. SIDEBAR COMPONENTS
# ==============================================================================

with st.sidebar:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        pass # Fallback if logo is missing
        
    st.markdown("---")
    # User profile section for a more "app-like" feel
    st.markdown("""
        <div class="sidebar-header" style="color: white; margin-bottom: 10px;">User Info</div>
        <div class="user-info-box">
            <div class="user-avatar">👤</div>
            <div>
                <div class="user-name">Library Guest</div>
                <div class="user-status">● Online</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. CHAT INTERFACE & INTERACTION LOGIC
# ==============================================================================

st.markdown('<div class="chat-header">Gnosys AI Librarian</div>', unsafe_allow_html=True)

# Initialize session-based chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Not sure what to read next? I'm Gnosys. Describe what you're looking for and I'll find the perfect match!"
        }
    ]

# Render existing chat history from the session state
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "image_url" in message and message["image_url"]:
            st.image(message["image_url"], width=150)

# Main Chat Input Logic
if prompt := st.chat_input("Type your message here..."):
    # 1. Capture and display user input
    st.chat_message("user", avatar="👤").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Process message via backend AI logic
    with st.spinner("Gnosys is thinking..."):
        # The process_user_message function bridges LLM intent and local DB search
        result = process_user_message(prompt, df_books)
        
        response_text = result["text"]
        book_obj = result["book"]

    # 3. Handle and display AI response
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response_text)
        img_link = None
        if book_obj is not None:
            img_link = book_obj.get('image_final')
            if pd.notna(img_link):
                st.image(img_link, width=150)
    
    # 4. Persistent storage of the interaction
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "image_url": img_link if pd.notna(img_link) else None
    })