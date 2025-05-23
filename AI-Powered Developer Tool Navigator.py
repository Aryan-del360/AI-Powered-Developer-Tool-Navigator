# app.py - Streamlit Web UI for AI-Powered Search (Hackathon Winner Polish - Corrected Rerun)

import streamlit as st
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, ConfigurationError
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig, Part
# import matplotlib.pyplot as plt # Removed, replaced by Plotly
from collections import Counter
# import io # Removed, not needed for Plotly fig object
from PIL import Image # Needed to display image from BytesIO
import requests # Needed for optional Hugging Face calls
from google.cloud import language_v1 # Added for NLP Insights
import pandas as pd # For Plotly chart DataFrame
import plotly.express as px # For interactive charts

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI-Powered Dev Tool Navigator", # More specific title
    page_icon="🛠️", # Developer tool icon
    layout="wide", # Use wide layout for more space
    initial_sidebar_state="expanded" # Sidebar expanded by default
)

# --- Custom CSS for a touch of style ---
# We'll use a slightly more refined color palette and spacing
st.markdown("""
<style>
<!-- IMPORTANT: Streamlit's auto-generated class names (e.g., .st-emotion-cache-xxxxxx) can change between Streamlit versions. If custom styles break after an update, these class names might need to be re-inspected and updated. -->
    /* Main page styling */
    .stApp {
        background-color: #f8f9fa; /* Very light grey background */
        color: #212529; /* Dark text for readability */
        padding-top: 1rem; /* Add some padding at the top */
    }

    /* Sidebar styling */
    /* Targeting specific Streamlit classes - these might change in future Streamlit versions */
    .st-emotion-cache-10q9u7y { /* Sidebar container */
        background-color: #e9ecef; /* Light grey sidebar background */
        padding: 1.5rem; /* More padding */
        border-right: 1px solid #dee2e6; /* Subtle border */
        border-radius: 0.75rem; /* Slightly more rounded corners */
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); /* Add a subtle shadow */
    }

    /* Style for expanders (search results) - making them look more like cards */
    .st-emotion-cache-ch5fjs { /* Expander header */
        background-color: #ffffff; /* White background for headers */
        border: 1px solid #dee2e6; /* Add a border */
        border-radius: 0.5rem; /* Rounded corners */
        padding: 1rem; /* Increased padding */
        margin-bottom: 0.75rem; /* More space between expanders */
        box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.08); /* Subtle shadow for card effect */
        transition: all 0.2s ease-in-out; /* Smooth transition on hover */
    }
     .st-emotion-cache-ch5fjs:hover {
         border-color: #007bff; /* Highlight border on hover */
         box-shadow: 2px 2px 6px rgba(0, 123, 255, 0.2); /* More prominent shadow on hover */
     }

    /* Style for info/success/warning boxes */
    .stAlert {
        border-radius: 0.5rem; /* Rounded corners */
        margin-bottom: 1rem; /* Add space below alerts */
    }

    /* Style for buttons */
    .stButton > button {
        border-radius: 0.5rem; /* Rounded corners */
        border: 1px solid #007bff; /* Primary blue border */
        color: #007bff; /* Primary blue text */
        background-color: #ffffff; /* White background */
        padding: 0.5rem 1.5rem; /* Increased padding */
        font-size: 1rem; /* Standard font size */
        margin-top: 0.5rem; /* Add space above button */
        transition: all 0.2s ease-in-out; /* Smooth transition */
    }

    .stButton > button:hover {
        color: #ffffff; /* White text on hover */
        background-color: #007bff; /* Primary blue background on hover */
        box-shadow: 1px 1px 3px rgba(0, 123, 255, 0.3); /* Subtle shadow on hover */
    }

    /* Style for link buttons */
    .st-emotion-cache-1c7v0sq a { /* Target link button */
         color: #17a2b8 !important; /* Teal color for links */
         text-decoration: none !important; /* No underline */
         font-weight: bold; /* Make links bold */
         transition: color 0.2s ease-in-out; /* Smooth color transition */
    }
     .st-emotion-cache-1c7v0sq a:hover {
         text-decoration: underline !important; /* Underline on hover */
         color: #138496 !important; /* Darker teal on hover */
     }

    /* Style for text input */
    .stTextInput > div > div > input {
        border-radius: 0.5rem; /* Rounded corners */
        border: 1px solid #ced4da; /* Light grey border */
        padding: 0.75rem 1rem; /* Increased padding */
        margin-bottom: 0.5rem; /* Space below input */
    }

    /* Style for selectbox */
    .stSelectbox > div > div > div {
         border-radius: 0.5rem; /* Rounded corners */
         border: 1px solid #ced4da; /* Light grey border */
         padding: 0.5rem 0.75rem; /* Increased padding */
    }

    /* Style for multiselect */
     .stMultiSelect > div > div > div {
         border-radius: 0.5rem; /* Rounded corners */
         border: 1px solid #ced4da; /* Light grey border */
         padding: 0.25rem 0.5rem; /* Padding */
     }

     /* Style for slider */
     .stSlider > div > div > div > div {
         background-color: #007bff; /* Primary blue slider track */
     }

    /* Style for the main title */
    h1 {
        color: #007bff; /* Primary blue for title */
    }

    /* Style for subheaders */
    h2, h3, h4 {
        color: #343a40; /* Dark grey for subheaders */
        margin-top: 1.5rem; /* Space above subheaders */
        margin-bottom: 1rem; /* Space below subheaders */
    }

    /* Style for markdown text */
    p {
        line-height: 1.6; /* Improve readability */
    }

    /* Style for the footer */
    .st-emotion-cache-h4c060 { /* Target footer container */
        text-align: center;
        margin-top: 3rem;
        color: #6c757d; /* Muted grey */
        font-size: 0.9rem;
    }
    /* Make text input fields generally a bit larger and more prominent */
    .stTextInput input {
        font-size: 1.1rem; /* Slightly larger font */
        padding: 0.8rem 1rem; /* Adjust padding for height */
    }
    .ai-explanation-box {
        background-color: #e6f7ff; /* A light blue background */
        border-left: 5px solid #007bff; /* Primary blue left border */
        padding: 10px 15px;
        margin-top: 10px;
        margin-bottom: 15px; /* More space after the box */
        border-radius: 0.3rem; /* Slightly more rounded corners than default alerts */
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* Subtle shadow */
    }
</style>
""", unsafe_allow_html=True)


# --- Configuration ---
# Use environment variables for sensitive info
# These will be read from the environment where you run 'streamlit run app.py'.
# You MUST set these variables BEFORE running the Streamlit app.
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
# This initialization needs to happen when the script runs
@st.cache_resource(show_spinner="Connecting to Google Cloud...") # Cache with spinner
def initialize_google_cloud():
    """Initializes Google Cloud services and loads models."""
    if not GCP_PROJECT_ID:
        st.error("CRITICAL ERROR: GCP_PROJECT_ID environment variable is not set. Cannot initialize Google Cloud AI.")
        st.stop() # Stop Streamlit execution

    try:
        import google.auth
        credentials, project = google.auth.default()
        # Use a specific region, ensure models are available there
        vertexai.init(project=GCP_PROJECT_ID, location="us-central1", credentials=credentials)

        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        gemini_text_model = GenerativeModel(GEMINI_TEXT_MODEL_ID)

        st.sidebar.markdown("<span style='color:green; font-size: 16px; vertical-align: middle;'>●</span> <span style='vertical-align: middle;'>Google Cloud: Services initialized and models loaded.</span>", unsafe_allow_html=True)
        return embedding_model, gemini_text_model

    except Exception as e:
        st.sidebar.error(f"Error initializing Google Cloud services or loading models: {e}") # Use sidebar for status
        st.sidebar.warning("Please ensure:")
        st.sidebar.warning("1. You have run 'gcloud auth application-default login' and are logged in to the correct account associated with your GCP Project ID.")
        st.sidebar.warning("2. The correct GCP_PROJECT_ID environment variable is set.")
        st.sidebar.warning("3. The Vertex AI API is enabled in your GCP project.")
        st.sidebar.warning(f"4. Necessary models are enabled and available in 'us-central1'.")
        st.stop() # Stop Streamlit execution

embedding_model, gemini_text_model = initialize_google_cloud()


# --- MongoDB Connection ---
# This connection needs to happen when the script runs
@st.cache_resource(show_spinner="Connecting to MongoDB Atlas...") # Cache with spinner
def get_mongo_client():
    """Establishes and caches the MongoDB connection."""
    if not MONGO_URI:
        st.error("CRITICAL ERROR: MONGO_URI environment variable is not set. Cannot initialize MongoDB connection.")
        st.stop() # Stop Streamlit execution

    try:
        # Strip potential surrounding quotes that Windows set command/PowerShell might add
        cleaned_mongo_uri = MONGO_URI.strip('"')
        client = MongoClient(cleaned_mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster') # Check connection
        st.sidebar.markdown("<span style='color:green; font-size: 16px; vertical-align: middle;'>●</span> <span style='vertical-align: middle;'>MongoDB Atlas: Connection successful!</span>", unsafe_allow_html=True)
        return client
    except ServerSelectionTimeoutError as e:
        st.sidebar.error(f"MongoDB Connection Error: Server selection timed out. {e}")
        st.sidebar.warning("This usually means the app could not find a suitable MongoDB server within the time limit (5 seconds). Check:")
        st.sidebar.warning("1. Your MONGO_URI is correct and the server/cluster is running.")
        st.sidebar.warning("2. Network connectivity (including VPNs or firewalls if applicable).")
        st.sidebar.warning("3. IP Whitelisting on MongoDB Atlas if you are using it.")
        st.sidebar.warning("4. If it's a replica set, ensure a primary is elected.")
        st.stop()
    except ConnectionFailure as e:
        st.sidebar.error(f"MongoDB Connection Error: Failed to connect. {e}")
        st.sidebar.warning("This can be due to various network issues (DNS, firewall, intermittent connectivity) or incorrect MONGO_URI.")
        st.sidebar.warning("Please verify your MONGO_URI and network settings.")
        st.stop()
    except ConfigurationError as e:
        st.sidebar.error(f"MongoDB Configuration Error: {e}")
        st.sidebar.warning("There's an issue with the MongoDB connection string or configuration options.")
        st.sidebar.warning("Please double-check your MONGO_URI format and any specified options.")
        st.stop()
    except Exception as e:  # Generic fallback
        st.sidebar.error(f"An unexpected error occurred connecting to MongoDB Atlas: {e}")
        st.sidebar.warning("Please check your MONGO_URI, network connection, and IP whitelisting if applicable.")
        st.stop()

mongo_client = get_mongo_client()
db = mongo_client[MONGO_DB_NAME]
collection = db[MONGO_COLLECTION_NAME]


# --- Function to Generate Embedding ---
def get_embedding(text_content: str) -> list[float]:
    """Generates embedding for a given text using Vertex AI."""
    try:
        # Max 2048 tokens for text-embedding-004. Using 7500 chars as a proxy.
        max_chars = 7500
        if len(text_content) > max_chars:
            st.warning(f"Warning: Text content for embedding is too long ({len(text_content)} chars). Truncating to {max_chars} chars to attempt to meet underlying token limits.")
            text_content = text_content[:max_chars]

        embeddings = embedding_model.get_embeddings([text_content])
        return embeddings[0].values

    except Exception as e:
        st.error(f"Error generating embedding for text: '{text_content[:100]}...' This could be due to the input text exceeding model token limits, or other API issues. Error: {e}")
        return None

# --- Function to Generate Topic Bar Chart ---
def generate_topic_chart(search_results: list, query_text: str) -> object | None:
    """Generates an interactive bar chart of the most frequent topics from search results using Plotly Express."""
    if not search_results:
        st.info("Skipping topic chart generation: No search results found.")
        return None

    # st.subheader("📊 Topic Distribution") # Subheader moved to where chart is displayed

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
        st.info("No topics found in search results for chart generation.")
        return None

    topic_counts = Counter(all_topics)
    top_n = 10 # Keep the top_n logic
    most_common_topics = topic_counts.most_common(top_n)

    if not most_common_topics:
        # This logic to show all if top_n is too small but topics exist
        if len(topic_counts) > 0:
             most_common_topics = topic_counts.most_common()
             st.info(f"Only {len(most_common_topics)} unique topics found. Showing all.")
        else:
             st.info("No topics found in search results for chart generation.") # Keep this
             return None # Return None if no topics

    df_topics = pd.DataFrame(most_common_topics, columns=['Topic', 'Frequency'])

    fig = px.bar(
        df_topics,
        x='Topic',
        y='Frequency',
        title=f"Top {len(df_topics)} Topics for Query: '{query_text[:50]}...'",
        color='Frequency',
        color_continuous_scale=px.colors.sequential.Blues_r, # Reversed Blues for darker high values
        text_auto=True # Display values on bars
    )
    fig.update_layout(
        xaxis_title="Topic",
        yaxis_title="Frequency",
        title_font_size=16,
        xaxis_tickangle=-45,
        # Optional: Add some margin for x-axis labels if they get cut off
        margin=dict(b=100) # b is bottom margin
    )
    fig.update_traces(textposition='outside') # Ensure text is outside bars if text_auto places it inside
    return fig

# --- Function to Generate Explanation using Gemini API (via Vertex AI) ---
def generate_explanation(summary_text: str) -> str:
    """Generates an explanation using the Vertex AI Gemini Text Model based on the search summary."""
    if not summary_text or not summary_text.strip():
        st.info("No valid summary text provided for explanation.")
        return "No explanation generated."

    # st.info(f"Attempting to generate explanation for summary: '{summary_text[:100]}...'") # Removed verbose info

    try:
        explanation_prompt = f"Based on the following summary of search results, provide a brief, easy-to-understand explanation relevant to a developer, focusing on how these tools might help them build a SaaS product:\n\n{summary_text}\n\nExplanation:" # Refined prompt

        response = gemini_text_model.generate_content(
            explanation_prompt,
            generation_config=GenerationConfig(
                temperature=0.7,
                max_output_tokens=300, # Increased token limit slightly
            ),
        )

        if response.candidates and response.candidates[0].content.parts:
            generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text') and part.text is not None)
            st.success("Explanation received successfully from Vertex AI Gemini Model.")
            return generated_text.strip()

        else:
             st.warning("Vertex AI Gemini Text Model call did not return a valid text response structure or candidates.")
             if hasattr(response, 'text'):
                  st.text(f"Full response as text: {response.text}") # Display raw response if available
             return "Could not generate explanation."

    except Exception as e:
        st.error(f"Error calling Vertex AI Gemini Text Model for explanation: {e}")
        st.warning("Details: Check your GCP credentials and ensure the text model is enabled and available in your Vertex AI region.")
        return "Error generating explanation."


# --- Function for NLP Insights ---
@st.cache_data(show_spinner=False) # Cache NLP results
def get_nlp_insights(text_content: str) -> dict:
    """
    Analyzes text for entities and sentiment using Google Cloud Natural Language API.
    """
    insights = {"entities": [], "document_sentiment": None, "error": None}
    if not text_content or not isinstance(text_content, str) or len(text_content.strip()) == 0:
        insights["error"] = "No text content provided for NLP analysis."
        return insights
    try:
        client = language_v1.LanguageServiceClient()
        document = language_v1.types.Document(content=text_content, type_=language_v1.types.Document.Type.PLAIN_TEXT)
        
        # Analyze entity sentiment
        # For more features, consider client.analyze_entities() and client.analyze_sentiment() separately
        response = client.analyze_entity_sentiment(document=document, encoding_type=language_v1.EncodingType.UTF8)
        
        # Document sentiment
        doc_sentiment = response.document_sentiment
        insights["document_sentiment"] = {"score": doc_sentiment.score, "magnitude": doc_sentiment.magnitude}
        
        # Entities
        for entity in response.entities:
            insights["entities"].append({
                "name": entity.name,
                "type": language_v1.types.Entity.Type(entity.type_).name,
                "salience": entity.salience,
                "sentiment_score": entity.sentiment.score,
                "sentiment_magnitude": entity.sentiment.magnitude
            })
        
        # Sort entities by salience (descending)
        insights["entities"].sort(key=lambda x: x["salience"], reverse=True)
        
    except Exception as e:
        insights["error"] = str(e)
        # st.warning(f"NLP processing failed for some text: {e}") # Silent for now
    return insights

# --- Keyword Lens Helper Functions ---
def get_query_keywords(query_text: str) -> list[str]:
    if not query_text: 
        return []
    # Simple keyword extraction: lowercase, split by space, ignore short words (<=2 chars)
    return [word.lower() for word in query_text.split() if len(word) > 2]

def count_keyword_mentions(text_content: str, keywords: list[str]) -> int:
    if not text_content or not keywords: 
        return 0
    text_lower = text_content.lower()
    text_words = text_lower.split() # Simple split, could be improved with regex for punctuation
    word_counts = Counter(text_words)
    
    count = 0
    for keyword in keywords:
        count += word_counts.get(keyword, 0)
    return count

# --- Session State Initialization ---
if 'search_results' not in st.session_state:
    st.session_state['search_results'] = []
if 'query_text' not in st.session_state:
    st.session_state['query_text'] = ""
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []
if 'favorite_tools' not in st.session_state:
    st.session_state['favorite_tools'] = {} # Using a dictionary to store favorites by a unique key (e.g., description + url)
if 'selected_lens' not in st.session_state: # For Keyword Focus Lens
    st.session_state['selected_lens'] = "Standard View"


# --- Helper function to add/remove favorites ---
def toggle_favorite(result, is_favorite):
    """Adds or removes a result from favorites."""
    # Create a simple unique key for the result
    key = f"{result.get('description', '')[:50]}_{result.get('url', '')}"
    if is_favorite:
        st.session_state['favorite_tools'][key] = result
        st.toast("Added to favorites! ❤️")
    else:
        if key in st.session_state['favorite_tools']:
            del st.session_state['favorite_tools'][key]
            st.toast("Removed from favorites 💔")


# --- Streamlit App UI ---

st.title("🛠️ AI-Powered Developer Tool Navigator") # Added emoji and refined title
st.markdown("Discover the **best AI tools** to speed up your development process, powered by **MongoDB Atlas** and **Google Cloud AI**.") # Refined description

# Get search query from user input
query_text = st.text_input("🔎 Enter your search query (e.g., 'AI tools for generating images', 'libraries for building chatbots'):", value=st.session_state['query_text']) # Set initial value from state

# Button to trigger the search
if st.button("Search 🚀"):
    if not query_text:
        st.warning("Please enter a search query.")
    else:
        # Add query to search history
        if query_text not in st.session_state['search_history']:
             st.session_state['search_history'].append(query_text)

        st.info(f"Searching for: {query_text}")

        # Use a spinner while searching
        with st.spinner(f"Generating embedding and searching MongoDB Atlas for '{query_text}'..."):
            # Generate embedding for the user's search query
            query_embedding = get_embedding(query_text)

            if query_embedding is None:
                st.error("Failed to generate embedding for the query. Cannot perform search.")
                st.session_state['search_results'] = [] # Clear previous results on error
            else:
                # Define the Atlas Vector Search aggregation pipeline
                pipeline = [
                    {
                        '$vectorSearch': {
                            "index": VECTOR_INDEX_NAME,
                            "path": "embedding",
                            "queryVector": query_embedding,
                            "numCandidates": 100, # Retrieve more candidates to allow for filtering
                            "limit": 50 # Limit the initial number of results to process
                        }
                    },
                    {
                        '$project': {
                            "_id": 0,
                            "description": 1,
                            "topics": 1,
                            "url": 1,
                            "score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]

                try:
                    # Execute the aggregation pipeline
                    search_results_cursor = collection.aggregate(pipeline)
                    search_results_list = list(search_results_cursor)

                    if search_results_list:
                        st.success(f"Initial search retrieved {len(search_results_list)} results. Use the sidebar to filter and sort.")
                        # Store results in session state to apply filters/sorting without re-searching
                        st.session_state['search_results'] = search_results_list
                        st.session_state['query_text'] = query_text # Store query text too

                    else:
                        st.info("No results found for your query.")
                        st.session_state['search_results'] = []
                        st.session_state['query_text'] = query_text
                        # Generate prompts/summary for the "no results" case
                        raw_summary_text = f"No search results were found for the query: '{query_text}'."
                        st.subheader("✨ AI Explanation") # Added emoji
                        explanation = generate_explanation(raw_summary_text)
                        st.markdown(f"<div class='ai-explanation-box'>{explanation}</div>", unsafe_allow_html=True)


                except Exception as e:
                    st.error(f"An error occurred during the initial search: {e}")
                    st.warning("Details: Ensure your MongoDB Atlas Vector Search index is 'Ready', the field paths ('embedding') are correct in the pipeline, and there are documents in the collection with embeddings.")
                    st.session_state['search_results'] = [] # Clear results on error


# --- Display Filtered and Sorted Results ---
# This section runs every time the app updates (e.g., filter change)
if 'search_results' in st.session_state and st.session_state['search_results']:
    # st.subheader("Filtered and Sorted Results") # Moved subheader down

    current_results = st.session_state['search_results']
    current_query_text = st.session_state['query_text']

    # --- Sidebar Filters & Lens ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Filter & View Options") # Updated subheader

    # Lens Selection
    lens_options = ["Standard View", "Keyword Focus Lens"]
    # Ensure default index is correctly found if selected_lens was not yet in session_state or was None
    default_lens_index = 0
    if 'selected_lens' in st.session_state and st.session_state.selected_lens in lens_options:
        default_lens_index = lens_options.index(st.session_state.selected_lens)
    
    # Update session state based on widget - Streamlit handles rerun on change
    st.session_state.selected_lens = st.sidebar.selectbox(
        "Select Data Lens:", 
        options=lens_options, 
        key="data_lens_selection_widget", # Unique key for the widget itself
        index=default_lens_index
    )


    # Extract all unique topics for the filter
    all_topics_in_results = set()
    for result in current_results:
        topics = result.get('topics', [])
        if isinstance(topics, list):
            for topic_item in topics:
                if isinstance(topic_item, dict) and 'name' in topic_item:
                    all_topics_in_results.add(topic_item['name'])
                elif isinstance(topic_item, str):
                    all_topics_in_results.add(topic_item)

    # Sort topics alphabetically for the multiselect
    sorted_topics = sorted(list(all_topics_in_results))

    # Topic filter
    selected_topics = st.sidebar.multiselect(
        "Filter by Topic:",
        options=sorted_topics,
        default=[] # No topics selected by default
    )

    # Score filter (assuming score is between 0 and 1)
    min_score = 0.0
    max_score = 1.0
    score_range = st.sidebar.slider(
        "Filter by Score Range:",
        min_value=min_score,
        max_value=max_score,
        value=(min_score, max_score), # Default to full range
        step=0.01 # Allow fine-grained selection
    )

    # --- Apply Filters ---
    filtered_results = []
    for result in current_results:
        score = result.get('score', 0.0) # Get score as float
        # Check score range
        if not (score_range[0] <= score <= score_range[1]):
            continue # Skip if outside score range

        # Check topic filter (if topics are selected)
        if selected_topics:
            result_topics = result.get('topics', [])
            # Extract topic names from result topics
            result_topic_names = [t['name'] if isinstance(t, dict) and 'name' in t else str(t) for t in result_topics if isinstance(t, (dict, str))]
            # Check if any selected topic is in the result's topics
            if not any(topic in result_topic_names for topic in selected_topics):
                continue # Skip if none of the selected topics are present

        # If the result passed all filters, add it
        filtered_results.append(result)

    st.sidebar.info(f"Showing {len(filtered_results)} results after filtering.")

    # --- Sidebar Sorting ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("⬆️⬇️ Sort Results") # Added emoji

    sort_option = st.sidebar.selectbox(
        "Sort By:",
        options=["Score (Descending)", "Score (Ascending)"]
    )

    # --- Apply Sorting ---
    if sort_option == "Score (Descending)":
        sorted_results = sorted(filtered_results, key=lambda x: x.get('score', 0), reverse=True)
    else: # Score (Ascending)
        sorted_results = sorted(filtered_results, key=lambda x: x.get('score', 0), reverse=False)

    # --- Display Filtered and Sorted Results ---
    if sorted_results:
        top_5_results = sorted_results[:5]
        other_relevant_results = sorted_results[5:15] # Display next 10 relevant results
        remaining_results_count = len(sorted_results) - len(top_5_results) - len(other_relevant_results)


        # --- Display Top 5 Results ---
        if top_5_results:
            st.markdown("## ⭐ Top 5 Relevant Tools") # Prominent heading with emoji
            st.write("Here are the top 5 tools based on your search query:")

            summary_parts_for_text = [] # Summary for AI Explanation (from top 5)
            result_urls = [] # URLs from top 5

            for i, result in enumerate(top_5_results):
                score = result.get('score', 0.0) # Get score as float
                description = result.get('description', 'N/A')
                project_url = result.get('url', 'N/A')
                topics = result.get('topics', [])

                formatted_topics = []
                if isinstance(topics, list):
                    formatted_topics = [t['name'] if isinstance(t, dict) and 'name' in t else str(t) for t in topics if isinstance(t, (dict, str))]

                # Create a unique key for this result for favorites
                result_key = f"{description[:50]}_{project_url}"
                is_favorite = result_key in st.session_state['favorite_tools']

                # Use expander for each top result for better organization (like a card)
                # Format score here for display
                with st.expander(f"**{i+1}. Score: {score:.4f}** - {description[:80]}..."): # Added formatting to header
                    # Placeholder for Icon/Logo (replace with actual if available)
                    # st.image("placeholder_logo.png", width=50) # Example if you had logos

                    # Use columns for layout within the expander
                    col1, col2 = st.columns([3, 1]) # Adjust column ratio as needed

                    with col1:
                        st.caption(f"Score: {score:.4f}")
                        st.write(f"**Description:** {description}")

                        if project_url and project_url != 'N/A':
                             st.link_button("🔗 Go to URL", url=project_url) # Added emoji to button
                        else:
                             st.write("**URL:** N/A")

                        if formatted_topics: # Use the initialized variable
                            st.caption(f"Topics: {', '.join(formatted_topics)}")
                        else:
                            st.caption("Topics: N/A")
                        
                        if st.session_state.get('selected_lens') == "Keyword Focus Lens":
                            query_keywords = get_query_keywords(st.session_state.get('query_text', ''))
                            if query_keywords:
                                keyword_count = count_keyword_mentions(description, query_keywords)
                                st.caption(f"Keyword Mentions (from query): {keyword_count}")


                    with col2:
                         # Add the Save to Favorites button
                         if st.button("❤️ Favorite" if not is_favorite else "💔 Unfavorite", key=f"fav_{result_key}"):
                             toggle_favorite(result, not is_favorite)
                             # Rerun the app to update the button state
                             st.rerun() # Corrected: Use st.rerun()


                    # Add a more detailed "Insight for Developers" based on available data
                    st.markdown("---") # Separator for clarity
                    st.markdown("💡 **Insight for Developers:**") # Added emoji
                    if description != 'N/A':
                        st.write(f"- **What it is:** *{description[:150]}...*") # Use italics for description snippet
                    if formatted_topics: # Use the initialized variable
                         st.write(f"- **Key Themes:** *{', '.join(formatted_topics[:5])}*") # Limit topics for insight, use italics
                    # Add a placeholder for "Why it's relevant to your query" - this would ideally involve more AI analysis
                    st.write(f"- **Potential Relevance:** *Based on your query and its similarity score ({score:.4f}), this tool is likely relevant for tasks involving [mention potential use case based on query/topics].*")
                    
                    # --- NLP Insights Integration ---
                    if description and description != 'N/A':
                        st.markdown("<h6>NLP Insights:</h6>", unsafe_allow_html=True)
                        nlp_insights = get_nlp_insights(description)
                        if nlp_insights.get("error"):
                            st.caption("NLP Insights: Could not be processed.")
                        else:
                            if nlp_insights.get("document_sentiment"):
                                sentiment_score = nlp_insights["document_sentiment"]["score"]
                                sentiment_label = "Positive" if sentiment_score > 0.25 else "Negative" if sentiment_score < -0.25 else "Neutral"
                                st.caption(f"Overall Sentiment: {sentiment_label} (Score: {sentiment_score:.2f}, Magnitude: {nlp_insights['document_sentiment']['magnitude']:.2f})")
                            
                            if nlp_insights.get("entities"):
                                st.caption("Key Entities (Salience | Sentiment Score):")
                                for entity_info in nlp_insights["entities"][:3]: # Display top 3
                                    st.caption(f"- {entity_info['name']} ({entity_info['type']}) | {entity_info['salience']:.2f} | {entity_info['sentiment_score']:.2f}")
                    st.markdown("---") # Final separator for the card


                # Collect data for summary for explanation (from top 5)
                summary_parts_for_text.append(f"- {description[:150]}...")
                if formatted_topics: # Use the initialized variable
                     summary_parts_for_text.append(f"  Topics: {', '.join(formatted_topics[:3])}")

                if project_url and project_url != 'N/A':
                    result_urls.append(project_url)

            # Build raw text summary for the explanation model from TOP 5 results
            raw_summary_text = f"Search results for '{current_query_text}' (Top 5):\n\n" + "\n".join(summary_parts_for_text)
            if result_urls:
                 raw_summary_text += "\n\nRelevant URLs include: " + ", ".join(result_urls[:3])

            # --- Generate Explanation from TOP 5 results ---
            st.subheader("✨ AI Explanation") # Added emoji
            explanation = generate_explanation(raw_summary_text)
            st.markdown(f"<div class='ai-explanation-box'>{explanation}</div>", unsafe_allow_html=True)


        # --- Display Other Relevant Results ---
        if other_relevant_results:
            st.markdown("## ➡️ Other Relevant Results") # Prominent heading with emoji
            st.write(f"Showing the next {len(other_relevant_results)} relevant results:")
            for i, result in enumerate(other_relevant_results):
                score = result.get('score', 0.0) # Get score as float
                description = result.get('description', 'N/A')
                project_url = result.get('url', 'N/A')
                topics = result.get('topics', [])

                formatted_topics = []
                if isinstance(topics, list):
                     formatted_topics = [t['name'] if isinstance(t, dict) and 'name' in t else str(t) for t in topics if isinstance(t, (dict, str))]

                # Create a unique key for this result for favorites
                result_key = f"{description[:50]}_{project_url}"
                is_favorite = result_key in st.session_state['favorite_tools']


                # Format score here for display
                st.markdown(f"**{len(top_5_results) + i + 1}.**")
                st.caption(f"Score: {score:.4f}")
                st.write(f"Description: {description[:150]}...") # Shorter description
                if project_url and project_url != 'N/A':
                     # Use columns for link button and favorite button on other results
                     link_col, fav_col = st.columns([3, 1])
                     with link_col:
                         st.link_button("🔗 Go to URL", url=project_url, key=f"url_{len(top_5_results) + i + 1}") # Added emoji, unique key
                     with fav_col:
                          if st.button("❤️ Favorite" if not is_favorite else "💔 Unfavorite", key=f"fav_{result_key}"):
                              toggle_favorite(result, not is_favorite)
                              st.rerun() # Corrected: Use st.rerun()

                else:
                     # If no URL, just show favorite button in a column
                     no_url_col, fav_col = st.columns([3, 1])
                     with no_url_col:
                         st.write("URL: N/A")
                     with fav_col:
                          if st.button("❤️ Favorite" if not is_favorite else "💔 Unfavorite", key=f"fav_{result_key}"):
                              toggle_favorite(result, not is_favorite)
                              st.rerun() # Corrected: Use st.rerun()


                if formatted_topics: # Use the initialized variable
                    st.caption(f"Topics: {', '.join(formatted_topics[:5])}") # Limit topics displayed
                else:
                     st.caption("Topics: N/A")
                
                if st.session_state.get('selected_lens') == "Keyword Focus Lens":
                    query_keywords = get_query_keywords(st.session_state.get('query_text', ''))
                    if query_keywords:
                        keyword_count = count_keyword_mentions(result.get('description', ''), query_keywords)
                        st.caption(f"Keyword Mentions (from query): {keyword_count}")
                
                st.markdown("---") # Separator


        if remaining_results_count > 0:
            st.info(f"There are {remaining_results_count} more results matching your filters and query.")


        # --- Generate Topic Bar Chart from ALL Filtered/Sorted results ---
        # Pass sorted_results (the full list after filtering/sorting) to the chart function
        st.subheader("📊 Topic Distribution") # Added emoji
        plotly_fig = generate_topic_chart(sorted_results, current_query_text)
        if plotly_fig:
            st.success("Interactive topic bar chart generated.") # Message before showing chart
            st.plotly_chart(plotly_fig, use_container_width=True)

        # --- Display Search History ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🕒 Search History") # Added emoji
        if st.session_state['search_history']:
            # Display history in reverse chronological order
            for i, query in enumerate(reversed(st.session_state['search_history'])):
                st.sidebar.text(f"{len(st.session_state['search_history']) - i}. {query[:30]}...") # Truncate long queries
        else:
            st.sidebar.info("No search history yet.")

        # --- Display Favorite Tools ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("❤️ Favorite Tools") # Added emoji
        if st.session_state['favorite_tools']:
            for key, result in st.session_state['favorite_tools'].items():
                 st.sidebar.text(f"- {result.get('description', 'Unnamed Tool')[:30]}...") # Display favorite name
                 # Optionally add a small button to view favorite details or remove from sidebar
                 # st.sidebar.button("View", key=f"view_fav_{key}") # Could add logic to show details
                 # if st.sidebar.button("Remove", key=f"remove_fav_{key}"):
                 #     toggle_favorite(result, False)
                 #     st.rerun() # Corrected: Use st.rerun()
        else:
            st.sidebar.info("No favorite tools yet.")


    else:
        st.info("No results match the selected filters.")
        # Generate prompts/summary for the "no results after filtering" case
        raw_summary_text = f"No search results match the filters for the query: '{current_query_text}'."
        st.subheader("✨ AI Explanation") # Added emoji
        explanation = generate_explanation(raw_summary_text)
        st.markdown(f"<div class='ai-explanation-box'>{explanation}</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("Built for the [Hackathon Name Here] using MongoDB Atlas and Google Cloud AI.") # Add your hackathon name

# Note: MongoDB client is automatically closed by Streamlit when the app stops.
# No explicit client.close() needed here like in the command-line script.
