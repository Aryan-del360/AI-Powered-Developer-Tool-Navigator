# app.py - Streamlit Web UI for AI-Powered Search

import streamlit as st
import os
from pymongo import MongoClient
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig, Part
import matplotlib.pyplot as plt
from collections import Counter
import io  # Needed to save matplotlib figure to a BytesIO object
from PIL import Image  # Needed to display image from BytesIO
import requests  # Needed for optional Hugging Face calls
import sys  # For clean exit on critical errors
import pandas as pd # Using pandas for easier data manipulation and display

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="BALA AI Debugging Assistant", # More specific & professional title
    page_icon="🤖", # A robot/AI icon for debugging assistant
    layout="wide", # Use wide layout for more space
    initial_sidebar_state="expanded" # Sidebar expanded by default
)

# --- Custom CSS for a vibrant, modern, Gen Z friendly style ---
st.markdown("""
<style>
    :root {
        --primary-blue: #3498db; /* A brighter, more vibrant blue */
        --accent-green: #2ecc71; /* A fresh green for success/accents */
        --light-bg: #f0f2f6; /* Very light grey background */
        --light-grey: #e9ecef; /* Light grey for sidebar/cards */
        --dark-text: #2c3e50; /* Dark charcoal for main text */
        --muted-text: #6c757d; /* Muted grey for secondary text */
        --border-color: #dee2e6; /* Subtle border color */
        --card-bg: #ffffff; /* White for card backgrounds */
    }

    /* Main page styling */
    .stApp {
        background-color: var(--light-bg);
        color: var(--dark-text);
        padding-top: 2rem; /* More padding at the top */
        font-family: 'Inter', 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; /* Modern font stack */
    }

    /* Sidebar styling */
    .st-emotion-cache-10q9u7y { /* Sidebar container */
        background-color: var(--light-grey);
        padding: 2rem; /* More generous padding */
        border-right: 1px solid var(--border-color);
        border-radius: 0.75rem;
        box-shadow: 4px 0px 10px rgba(0, 0, 0, 0.05); /* Stronger shadow for sidebar */
    }

    /* Style for expanders (search results) - making them look more like cards */
    .st-emotion-cache-ch5fjs, .st-emotion-cache-ch5fjs > div[data-baseweb="button"] { /* Expander header and content */
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 0.75rem; /* Slightly more rounded corners */
        padding: 1.2rem 1.5rem; /* Increased padding */
        margin-bottom: 1rem; /* More space between expanders */
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.08); /* Noticeable shadow for card effect */
        transition: all 0.2s ease-in-out; /* Smooth transition on hover */
    }
    .st-emotion-cache-ch5fjs:hover {
        border-color: var(--primary-blue); /* Highlight border on hover */
        box-shadow: 3px 3px 10px rgba(var(--primary-blue), 0.3); /* More prominent shadow on hover */
        transform: translateY(-2px); /* Subtle lift effect */
    }

    /* Style for info/success/warning boxes */
    .stAlert {
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        padding: 1rem 1.5rem;
    }

    /* Style for buttons */
    .stButton > button {
        border-radius: 0.75rem; /* Rounded corners */
        border: 1px solid var(--primary-blue);
        color: var(--primary-blue);
        background-color: var(--card-bg);
        padding: 0.6rem 1.8rem; /* Increased padding */
        font-size: 1rem;
        margin-top: 0.5rem;
        transition: all 0.2s ease-in-out;
        font-weight: 600; /* Bolder text */
    }

    .stButton > button:hover {
        color: var(--card-bg);
        background-color: var(--primary-blue);
        box-shadow: 2px 2px 6px rgba(var(--primary-blue), 0.4);
    }

    /* Style for link buttons (like "Go to URL") */
    .st-emotion-cache-1c7v0sq a { /* Target link button container (adjust class if needed) */
        color: var(--accent-green) !important; /* Green for links */
        text-decoration: none !important;
        font-weight: bold;
        transition: color 0.2s ease-in-out;
    }
    .st-emotion-cache-1c7v0sq a:hover {
        text-decoration: underline !important;
        color: #27ae60 !important; /* Darker green on hover */
    }

    /* Style for text input */
    .stTextInput > div > div > input {
        border-radius: 0.75rem;
        border: 1px solid var(--border-color);
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s ease-in-out;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 0.2rem rgba(var(--primary-blue), 0.25); /* Focus highlight */
    }

    /* Style for selectbox and multiselect */
    .stSelectbox > div > div > div, .stMultiSelect > div > div > div {
        border-radius: 0.75rem;
        border: 1px solid var(--border-color);
        padding: 0.6rem 1rem;
        transition: border-color 0.2s ease-in-out;
    }
    .stSelectbox > div > div > div:focus, .stMultiSelect > div > div > div:focus {
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 0.2rem rgba(var(--primary-blue), 0.25);
    }


    /* Style for slider */
    .stSlider > div > div > div > div {
        background-color: var(--primary-blue); /* Primary blue slider track */
    }

    /* Style for the main title */
    h1 {
        color: var(--dark-text); /* Darker text for title */
        font-size: 2.8rem; /* Larger title */
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Style for subheaders */
    h2, h3, h4 {
        color: var(--dark-text);
        margin-top: 2rem; /* More space above subheaders */
        margin-bottom: 1rem;
        font-weight: 600;
    }

    /* Style for markdown text */
    p, li {
        line-height: 1.7; /* Improve readability */
        font-size: 1.05rem;
    }

    /* Footer styling */
    .st-emotion-cache-h4c060 { /* Target footer container */
        text-align: center;
        margin-top: 4rem;
        color: var(--muted-text);
        font-size: 0.85rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-color);
    }
</style>
""", unsafe_allow_html=True)


# --- Configuration ---
# Use environment variables for sensitive info
MONGO_URI = os.getenv("MONGO_URI")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
HF_API_TOKEN = os.getenv("HF_API_TOKEN") # Optional

# MongoDB database, collection, and Vector Search index names
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "Meta_data")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "Test Data")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_index")

# Vertex AI model names
EMBEDDING_MODEL_NAME = "text-embedding-004"
GEMINI_TEXT_MODEL_ID = "gemini-2.0-flash"

# --- Initialize Google Cloud Services (for embeddings and Text) ---
@st.cache_resource(show_spinner="⚡ Connecting to Google Cloud AI...")
def initialize_google_cloud():
    """Initializes Google Cloud services and loads models."""
    if not GCP_PROJECT_ID:
        st.error("⚠️ CRITICAL ERROR: GCP_PROJECT_ID environment variable is not set. Cannot initialize Google Cloud AI.")
        st.stop()

    try:
        import google.auth
        credentials, project = google.auth.default()
        vertexai.init(project=GCP_PROJECT_ID, location="us-central1", credentials=credentials)

        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        gemini_text_model = GenerativeModel(GEMINI_TEXT_MODEL_ID)

        st.sidebar.success("✅ Google Cloud AI services initialized!")
        return embedding_model, gemini_text_model

    except Exception as e:
        st.sidebar.error(f"❌ Error initializing Google Cloud AI or loading models: {e}")
        st.sidebar.warning("💡 Please ensure:")
        st.sidebar.markdown("- 1. You are logged in via `gcloud auth application-default login`.")
        st.sidebar.markdown("- 2. `GCP_PROJECT_ID` env variable is correctly set.")
        st.sidebar.markdown("- 3. Vertex AI API is enabled in your GCP project.")
        st.sidebar.markdown("- 4. Models are enabled and available in `us-central1`.")
        st.stop()

embedding_model, gemini_text_model = initialize_google_cloud()


# --- MongoDB Connection ---
@st.cache_resource(show_spinner="📦 Connecting to MongoDB Atlas...")
def get_mongo_client():
    """Establishes and caches the MongoDB connection."""
    if not MONGO_URI:
        st.error("⚠️ CRITICAL ERROR: MONGO_URI environment variable is not set. Cannot initialize MongoDB connection.")
        st.stop()

    try:
        cleaned_mongo_uri = MONGO_URI.strip('"')
        client = MongoClient(cleaned_mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster') # Check connection
        st.sidebar.success("🎉 MongoDB Atlas connected!")
        return client
    except Exception as e:
        st.sidebar.error(f"❌ Error connecting to MongoDB Atlas: {e}")
        st.sidebar.warning("💡 Please check your `MONGO_URI` env variable and network connection (IP whitelisting).")
        st.stop()

mongo_client = get_mongo_client()
db = mongo_client[MONGO_DB_NAME]
collection = db[MONGO_COLLECTION_NAME]


# --- Function to Generate Embedding ---
def get_embedding(text_content: str) -> list[float]:
    """Generates embedding for a given text using Vertex AI."""
    try:
        max_chars = 25000
        if len(text_content) > max_chars:
            st.info(f"✨ Heads up: Your query is quite long ({len(text_content)} chars). Truncating for best embedding results.")
            text_content = text_content[:max_chars]

        embeddings = embedding_model.get_embeddings([text_content])
        return embeddings[0].values

    except Exception as e:
        st.error(f"Oops! Ran into an issue generating embedding for your query. Error: {e}")
        return None

# --- Function to Generate Topic Bar Chart ---
def generate_topic_chart(search_results: list, query_text: str) -> io.BytesIO | None:
    """Generates a bar chart of the most frequent topics from search results and returns it as bytes."""
    if not search_results:
        st.info("No results to chart. Try a different search!")
        return None

    st.subheader("📊 Topic Trends") # Catchy subheader
    
    all_topics = []
    for result in search_results:
        topics = result.get('topics', [])
        if isinstance(topics, list):
            for topic_item in topics:
                if isinstance(topic_item, dict) and 'name' in topic_item:
                    all_topics.append(topic_item['name'])
                elif isinstance(topic_item, str):
                    all_topics.append(topic_item)

    if not all_topics:
        st.info("No topics found in these results to visualize.")
        return None

    topic_counts = Counter(all_topics)
    top_n = min(10, len(topic_counts)) # Ensure top_n doesn't exceed available topics
    most_common_topics = topic_counts.most_common(top_n)

    if not most_common_topics:
        st.info("Not enough unique topics to generate a meaningful chart.")
        return None

    topics, counts = zip(*most_common_topics)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(topics, counts, color=st.get_option("color.primaryBg")) # Use Streamlit's primary color
    ax.set_xlabel("Topic Category", fontsize=12, color=st.get_option("color.secondaryText"))
    ax.set_ylabel("Frequency", fontsize=12, color=st.get_option("color.secondaryText"))
    ax.set_title(f"Top {len(most_common_topics)} Topics for '{query_text[:40]}...'", fontsize=14, color=st.get_option("color.textColor"))
    plt.xticks(rotation=45, ha='right', fontsize=10, color=st.get_option("color.secondaryText"))
    plt.yticks(fontsize=10, color=st.get_option("color.secondaryText"))
    ax.grid(axis='y', linestyle='--', alpha=0.6, color=st.get_option("color.secondaryText"))

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, int(yval), va='bottom', ha='center', fontsize=9, color=st.get_option("color.textColor"))

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, transparent=True) # High DPI and transparent background
    buf.seek(0)
    plt.close(fig)

    st.success("Chart generated!")
    return buf

# --- Function to Generate Explanation using Gemini API (via Vertex AI) ---
def generate_explanation(summary_text: str) -> str:
    """Generates an explanation using the Vertex AI Gemini Text Model based on the search summary."""
    if not summary_text or not summary_text.strip():
        st.info("Can't generate explanation without a summary.")
        return "No AI explanation available."

    st.subheader("🧠 AI-Powered Insight") # Engaging subheader
    
    try:
        explanation_prompt = f"Based on the following summary of search results, provide a brief, easy-to-understand explanation relevant to a developer, focusing on how these tools might help them build a SaaS product or improve development workflows. Keep it concise and actionable:\n\nSummary:\n{summary_text}\n\nExplanation:"

        response = gemini_text_model.generate_content(
            explanation_prompt,
            generation_config=GenerationConfig(
                temperature=0.7,
                max_output_tokens=350, # Increased token limit for more comprehensive response
            ),
        )

        if response.candidates and response.candidates[0].content.parts:
            generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text') and part.text is not None)
            st.info("Explanation crafted by Google Gemini AI.")
            return generated_text.strip()
        else:
            st.warning("Gemini couldn't generate a valid explanation. Try refining your query!")
            if hasattr(response, 'text'):
                st.text(f"Raw response: {response.text[:200]}...") # Display snippet of raw response
            return "Could not generate AI explanation."

    except Exception as e:
        st.error(f"Error connecting to Gemini for explanation: {e}")
        st.warning("💡 Double-check your GCP credentials and ensure the text model is enabled.")
        return "Error generating AI explanation."

# --- Session State Initialization ---
if 'search_results' not in st.session_state:
    st.session_state['search_results'] = []
if 'query_text' not in st.session_state:
    st.session_state['query_text'] = ""
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []
if 'favorite_tools' not in st.session_state:
    st.session_state['favorite_tools'] = {}

# --- Helper function to add/remove favorites ---
def toggle_favorite(result_dict, is_favorite):
    """Adds or removes a result from favorites, using a consistent unique key."""
    # Ensure all relevant fields are present or default to empty string for key generation
    tool_name = result_dict.get('name', 'Unnamed Tool') # Assuming a 'name' field might exist
    description_snippet = result_dict.get('description', '')[:50]
    url_snippet = result_dict.get('url', '')

    key = f"{tool_name}_{description_snippet}_{url_snippet}"

    if is_favorite:
        st.session_state['favorite_tools'][key] = result_dict
        st.toast("Added to your top picks! ❤️")
    else:
        if key in st.session_state['favorite_tools']:
            del st.session_state['favorite_tools'][key]
            st.toast("Removed from favorites 💔")

# --- Streamlit App UI ---

st.header("BALA AI Debugging Assistant 🤖") # Main header, modern, clear
st.markdown("Unlock the **best AI tools** to supercharge your development workflow and **debug faster**, powered by **MongoDB Atlas** and **Google Cloud AI**.")

# Main search input
query_text = st.text_input(
    "What kind of AI tools are you looking for? (e.g., 'tools for generating code', 'AI for frontend testing', 'libraries for large language models')",
    value=st.session_state['query_text'],
    placeholder="Type your query here..." # Placeholder for better UX
)

# Search button with a little flair
if st.button("🚀 Find My Tools"):
    if not query_text:
        st.warning("Heads up! Please drop your search query in the box above to get started. 🤓")
    else:
        # Add query to search history
        if query_text not in st.session_state['search_history']:
            st.session_state['search_history'].append(query_text)

        st.info(f"Searching for: **'{query_text}'**")

        with st.spinner(f"⚡ Generating embedding and searching MongoDB Atlas for '{query_text}'..."):
            query_embedding = get_embedding(query_text)

            if query_embedding is None:
                st.error("Uh oh! Failed to generate embedding for your query. Can't search without it! 😵‍💫")
                st.session_state['search_results'] = []
            else:
                pipeline = [
                    {
                        '$vectorSearch': {
                            "index": VECTOR_INDEX_NAME,
                            "path": "embedding",
                            "queryVector": query_embedding,
                            "numCandidates": 100,
                            "limit": 50
                        }
                    },
                    {
                        '$project': {
                            "_id": 0,
                            "name": "$name", # Assuming you have a 'name' field for the tool
                            "description": 1,
                            "topics": 1,
                            "url": 1,
                            "score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]

                try:
                    search_results_cursor = collection.aggregate(pipeline)
                    search_results_list = list(search_results_cursor)

                    if search_results_list:
                        st.session_state['search_results'] = search_results_list
                        st.session_state['query_text'] = query_text # Store query text
                        st.success(f"Found {len(search_results_list)} awesome tools! Check out the filters in the sidebar to refine. 🎉")

                    else:
                        st.info("Bummer! No tools found for your query. Maybe try a different keyword? 🤷‍♀️")
                        st.session_state['search_results'] = []
                        st.session_state['query_text'] = query_text
                        
                        # Generate AI explanation for no results
                        raw_summary_text = f"No search results were found for the query: '{query_text}'. The user is looking for AI tools for software development, debugging, or related tasks."
                        explanation = generate_explanation(raw_summary_text)
                        st.markdown("---") # Visual separator
                        st.markdown(explanation) # Display AI explanation immediately

                except Exception as e:
                    st.error(f"An error occurred during the search. Please ensure your MongoDB Atlas Vector Search index is 'Ready' and the collection has data. Error: {e} 😬")
                    st.session_state['search_results'] = []

# --- Display Filtered and Sorted Results ---
if 'search_results' in st.session_state and st.session_state['search_results']:
    current_results = st.session_state['search_results']
    current_query_text = st.session_state['query_text']

    # --- Sidebar Filters & Sort ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("⚙️ Refine Results")

        all_topics_in_results = set()
        for result in current_results:
            topics = result.get('topics', [])
            if isinstance(topics, list):
                for topic_item in topics:
                    if isinstance(topic_item, dict) and 'name' in topic_item:
                        all_topics_in_results.add(topic_item['name'])
                    elif isinstance(topic_item, str):
                        all_topics_in_results.add(topic_item)

        sorted_topics = sorted(list(all_topics_in_results))

        selected_topics = st.multiselect(
            "Filter by Topic:",
            options=sorted_topics,
            default=[],
            placeholder="Select topics..."
        )

        min_score = 0.0
        max_score = max(r.get('score', 0) for r in current_results) if current_results else 1.0 # Dynamic max score
        score_range = st.slider(
            "Filter by Relevance Score:",
            min_value=min_score,
            max_value=max_score + 0.01, # Add a tiny bit to max to ensure slider range includes max
            value=(min_score, max_score),
            step=0.001, # Finer granularity
            format="%.3f" # Display score with 3 decimal places
        )

        st.markdown("---")
        st.subheader("⬆️⬇️ Sort Options")

        sort_option = st.selectbox(
            "Order Results By:",
            options=["Relevance (Highest First)", "Relevance (Lowest First)"],
            help="Sort by the AI's calculated relevance score."
        )

        # --- Apply Filters ---
        filtered_results = []
        for result in current_results:
            score = result.get('score', 0.0)
            if not (score_range[0] <= score <= score_range[1]):
                continue

            if selected_topics:
                result_topics = result.get('topics', [])
                result_topic_names = [t['name'] if isinstance(t, dict) and 'name' in t else str(t) for t in result_topics if isinstance(t, (dict, str))]
                if not any(topic in result_topic_names for topic in selected_topics):
                    continue
            
            filtered_results.append(result)

        st.info(f"Showing **{len(filtered_results)}** matching tools after filtering.")

        # --- Apply Sorting ---
        if sort_option == "Relevance (Highest First)":
            sorted_results = sorted(filtered_results, key=lambda x: x.get('score', 0), reverse=True)
        else: # Relevance (Lowest First)
            sorted_results = sorted(filtered_results, key=lambda x: x.get('score', 0), reverse=False)
        
        # --- Sidebar Search History ---
        st.markdown("---")
        st.subheader("🕒 Your Recent Searches")
        if st.session_state['search_history']:
            for i, query in enumerate(reversed(st.session_state['search_history'])):
                if st.button(f"🔍 {query[:40]}...", key=f"history_query_{i}"):
                    st.session_state['query_text'] = query
                    st.experimental_rerun() # Use rerun to trigger search with history query
        else:
            st.info("Your search journey starts here!")

        # --- Sidebar Favorite Tools ---
        st.markdown("---")
        st.subheader("❤️ Your Top Picks")
        if st.session_state['favorite_tools']:
            for key, result in st.session_state['favorite_tools'].items():
                col_fav_text, col_fav_btn = st.columns([3, 1])
                with col_fav_text:
                    st.markdown(f"**[{result.get('name', 'Tool')}]({result.get('url', '#')})**")
                    st.caption(f"{result.get('description', '')[:70]}...")
                with col_fav_btn:
                    if st.button("💔", key=f"fav_sidebar_{key}"): # Smaller unfavorite button
                        toggle_favorite(result, False)
                        st.rerun() # Refresh to update favorites list
        else:
            st.info("No favorites yet. Click the ❤️ button to save tools!")


    # --- Display Main Results ---
    if sorted_results:
        top_n_to_display = 15 # Display top 15 results in main section
        
        # Determine number of main sections based on results
        if len(sorted_results) > 0:
            st.markdown("---")
            st.markdown("## ✨ Top Recommendations")
            st.write("Here are the AI tools tailored for your needs:")

            summary_parts_for_ai = [] # Collect descriptions for AI summary
            result_urls_for_ai = [] # Collect URLs for AI summary

            for i, result in enumerate(sorted_results[:top_n_to_display]):
                tool_name = result.get('name', 'AI Tool')
                score = result.get('score', 0.0)
                description = result.get('description', 'No description provided.')
                project_url = result.get('url', '')
                topics = result.get('topics', [])

                formatted_topics = []
                if isinstance(topics, list):
                    formatted_topics = [t['name'] if isinstance(t, dict) and 'name' in t else str(t) for t in topics if isinstance(t, (dict, str))]

                # Create a unique key for this result for favorites
                result_key = f"{tool_name}_{description[:50]}_{project_url}"
                is_favorite = result_key in st.session_state['favorite_tools']

                with st.expander(f"**{i+1}. {tool_name}** `Score: {score:.3f}`"):
                    col_desc, col_actions = st.columns([3, 1])

                    with col_desc:
                        st.write(f"**Description:** {description}")
                        if project_url:
                            st.link_button("🚀 Check it out!", url=project_url)
                        else:
                            st.write("URL: N/A")
                        
                        if formatted_topics:
                            st.write(f"**Key Topics:** {', '.join(formatted_topics)}")
                        else:
                            st.write("Topics: N/A")
                    
                    with col_actions:
                        # Add the Save to Favorites button
                        if st.button("❤️ Add to Favorites" if not is_favorite else "💔 Remove Favorite", key=f"fav_main_{result_key}"):
                            toggle_favorite(result, not is_favorite)
                            st.rerun() # Rerun the app to update the button state

                    st.markdown("---")
                    st.markdown("### 💡 Developer Insight")
                    st.write(f"This tool ({tool_name}) seems relevant for tasks like `{', '.join(formatted_topics[:2])}`. Its high relevance score suggests it aligns well with your search.")
                    if description:
                        st.write(f"**What it offers:** *{description[:200]}...*") # Snippet from description

                # Collect data for AI summary (from displayed results)
                summary_parts_for_ai.append(f"- Tool: {tool_name}\n  Description: {description[:100]}...\n  Topics: {', '.join(formatted_topics[:3])}")
                if project_url:
                    result_urls_for_ai.append(project_url)
            
            # Build raw text summary for the explanation model from displayed results
            raw_summary_text = f"Search results for '{current_query_text}' (Top {min(len(sorted_results), top_n_to_display)}):\n\n" + "\n".join(summary_parts_for_ai)
            if result_urls_for_ai:
                raw_summary_text += "\n\nReferenced URLs include: " + ", ".join(result_urls_for_ai[:5]) # Limit URLs for summary

            # --- Generate Explanation from the summarized results ---
            explanation = generate_explanation(raw_summary_text)
            st.markdown("---") # Visual separator before AI explanation
            st.markdown(explanation)
            st.markdown("---")

        if len(sorted_results) > top_n_to_display:
            st.info(f"There are **{len(sorted_results) - top_n_to_display}** more tools matching your filters and query. Adjust filters or refine your search to see more!")

        # --- Generate Topic Bar Chart from ALL Filtered/Sorted results ---
        chart_buffer = generate_topic_chart(sorted_results, current_query_text)
        if chart_buffer:
            st.image(chart_buffer, caption="Top Topics in Your Filtered Results")

    else:
        if st.session_state['query_text']: # Only show if a search was performed
            st.info("No results found after applying filters. Try adjusting your filter settings! 🤔")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by BALA for the GitLab AI Challenge 2025. Powered by Google Cloud AI & MongoDB Atlas.")
