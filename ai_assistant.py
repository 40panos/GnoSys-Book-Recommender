"""
Gnosys AI Assistant Module
--------------------------
This module serves as the 'brain' of the Gnosys Recommender. It integrates 
OpenAI's GPT models to analyze user intent, extract keywords for search, 
και generate friendly, context-aware responses in Greek.
"""

import pandas as pd
from openai import OpenAI
import utils

# ==============================================================================
# CONFIGURATION & INITIALIZATION
# ==============================================================================

# OpenAI API Key
config = utils.load_config()
API_KEY = config.get("api", {}).get("openai_key", "")

client = None
if API_KEY and "sk-" in API_KEY:
    client = OpenAI(api_key=API_KEY)

# ==============================================================================
# 1. INTENT & KEYWORD EXTRACTION
# ==============================================================================

def analyze_intent_and_extract(user_query):
    """
    Analyzes the user's input using LLM to determine if they are looking for 
    a book or just chatting.
    
    Returns:
        str: A list of English keywords for searching the database, 
             or "CHAT_ONLY" if no search is required.
    """
    if not client: return "CHAT_ONLY"

    # System prompt defines the AI's persona and decision logic
    system_prompt = """
    You are the brain of a Book Recommendation AI called 'Gnosys'.
    Analyze the user's input and decide the next step.

    Rules:
    1. DO NOT mention that a specific book is missing.
    2. Even if the user asked for a specific title and it is NOT in the list, 
       IGNORE the specific title and recommend the best available match from the list.
   
    CASE 1: BOOK SEARCH
    If the user describes a plot, genre, mood, or asks for a recommendation.
    -> OUTPUT: A list of 3-5 English keywords based on the request.

    CASE 2: CHIT-CHAT / GREETING / IRRELEVANT
    If the user says "Hello", asks "Who are you?", or asks about non-book topics.
    -> OUTPUT: "CHAT_ONLY"

    Do not output anything else. Just keywords or "CHAT_ONLY".
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0 # Set to 0 for consistent, deterministic keyword extraction
        )
        result = response.choices[0].message.content.strip()
        result = result.replace('"', '').replace("'", "")
        return result
    except Exception as e:
        print(f"Error in intent analysis: {e}")
        return "CHAT_ONLY"

# ==============================================================================
# 2. LOCAL SEARCH ENGINE
# ==============================================================================

def search_books_in_df(keywords_str, df, top_k=3):
    """
    Performs a weighted keyword search across the titles, authors, descriptions, 
    and tags in the local DataFrame.
    """
    keywords = keywords_str.split()
    if not keywords or df.empty: return pd.DataFrame()

    df = df.copy()
    df['match_score'] = 0
    
    search_cols = ['title', 'authors', 'description', 'all_tags']
    valid_cols = [c for c in search_cols if c in df.columns]

    # Assign higher weights to title matches
    for word in keywords:
        for col in valid_cols:
            weight = 2 if col == 'title' else 1
            df['match_score'] += df[col].astype(str).str.contains(word, case=False, na=False).astype(int) * weight
    
    # Filter results and sort by match score and global rating
    results = df[df['match_score'] > 0]
    results = results.sort_values(by=['match_score', 'average_rating'], ascending=[False, False])
    
    return results.head(top_k)

# ==============================================================================
# 3. RESPONSE GENERATION
# ==============================================================================

def generate_gnosys_response(user_query, found_books, intent_type):
    """
    Generates the final natural language response for the user in Greek.
    If books were found, it presents them as the 'Librarian' persona.
    """
    if not client: return "No API Key configured."

    if intent_type == "SEARCH_SUCCESS" and not found_books.empty:
        # Prepare context from search results for the LLM
        books_context = ""
        for i, row in found_books.iterrows():
            title = row.get('title', 'Unknown')
            auth = row.get('authors', 'Unknown')
            desc = str(row.get('description', ''))[:200]
            books_context += f"{i+1}. Title: {title}\n   Author: {auth}\n   Description: {desc}...\n\n"

        system_prompt = f"""
        You are Gnosys, an AI Librarian.
        User asked: "{user_query}"
        Found books context:
        {books_context}
        
        Task: Recommend the best match. Be friendly and concise. Answer in Greek.
        """
    else:
        # Default persona for chit-chat or fallback scenarios
        system_prompt = """
        You are Gnosys, an AI Librarian.
        Task:
        - Greet users warmly in Greek.
        - Explain that you are a smart book recommender.
        - Politely decline non-book related questions.
        - Answer exclusively in Greek.
        """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

# ==============================================================================
# 4. MAIN WRAPPER FUNCTION (ENTRY POINT)
# ==============================================================================

def process_user_message(user_input, df_books):
    """
    High-level function to process a user message and return both 
    the text response and the top recommended book object.
    """
    # Step 1: Analyze Intent
    intent_result = analyze_intent_and_extract(user_input)
    
    # Step 2: Handle Chat-only intent
    if intent_result == "CHAT_ONLY":
        ai_response = generate_gnosys_response(user_input, pd.DataFrame(), intent_type="CHAT")
        return {"text": ai_response, "book": None}
    
    # Step 3: Handle Search intent
    else:
        keywords = intent_result
        found_books = search_books_in_df(keywords, df_books)
        
        if found_books.empty:
            return {
                "text": "I searched the library based on your description but couldn't find a direct match. Try different keywords or genres!",
                "book": None
            }
        
        # Step 4: Generate contextual recommendation response
        ai_response = generate_gnosys_response(user_input, found_books, intent_type="SEARCH_SUCCESS")
        top_book = found_books.iloc[0]
        
        return {
            "text": ai_response,
            "book": top_book 
        }