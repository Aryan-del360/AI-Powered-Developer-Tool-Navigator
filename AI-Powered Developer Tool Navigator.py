import streamlit as st
import requests

# Tagline
st.title("💬 Chatbot: Find the right developer tools instantly with AI-driven semantic search")
st.write(
    "This is a simple chatbot that uses Google Gemini (Vertex AI) to generate responses. "
    "To use this app, you need to provide your Gemini API key. "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their Gemini API key via `st.text_input`.
gemini_api_key = st.text_input("Gemini API Key", type="password", value="")
if not gemini_api_key:
    st.info("Please add your Gemini API key to continue.", icon="🗝️")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare messages for Gemini API (only last user message for simplicity)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            # Combined into a single line using .get() with default values for robustness
            gemini_reply = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "[Error: Unexpected Gemini API response format.]")
        else:
            gemini_reply = f"[Error: Gemini API returned status {response.status_code}]"

        with st.chat_message("assistant"):
            st.markdown(gemini_reply)
        st.session_state.messages.append({"role": "assistant", "content": gemini_reply})

# core_logic.py - Core Backend Functions for AI-Powered Dev Tool Navigator

import os
from pymongo import MongoClient
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig

# Configuration
MONGO_URI = os.getenv("MONGO_URI")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")

MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "Meta_data")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "Test Data")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_index")

EMBEDDING_MODEL_NAME = "text-embedding-004"
GEMINI_TEXT_MODEL_ID = "gemini-2.0-flash"

# Initialize Google Cloud Services
def initialize_google_cloud_models():
    if not GCP_PROJECT_ID:
        raise ValueError("GCP_PROJECT_ID environment variable is not set. Cannot initialize Google Cloud AI.")

    vertexai.init(project=GCP_PROJECT_ID, location="us-central1")
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    gemini_text_model = GenerativeModel(GEMINI_TEXT_MODEL_ID)
    return embedding_model, gemini_text_model

# Initialize models once when this module is imported.
# Any errors here (e.g., missing GCP_PROJECT_ID) will propagate up to app.py.
embedding_model, gemini_text_model = initialize_google_cloud_models()


# MongoDB Connection
def get_mongo_collection():
    if not MONGO_URI:
        raise ValueError("MONGO_URI environment variable is not set. Cannot connect to MongoDB Atlas.")

    cleaned_mongo_uri = MONGO_URI.strip('"')
    client = MongoClient(cleaned_mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping') # Check connection
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    return collection

# Get MongoDB collection once when this module is imported.
# Any errors here (e.g., missing MONGO_URI) will propagate up to app.py.
mongo_collection = get_mongo_collection()

# Function to Generate Embedding for Text
def get_embedding(text_content: str) -> list[float]:
    # `embedding_model` should be initialized; an error would have been raised at import if not.
    max_chars = 25000
    if len(text_content) > max_chars:
        text_content = text_content[:max_chars]

    embeddings = embedding_model.get_embeddings([text_content])
    return embeddings[0].values

# Function to Perform Vector Search in MongoDB Atlas
def perform_vector_search(query_embedding: list[float]) -> list[dict]:
    # `mongo_collection` should be initialized; an error would have been raised at import if not.
    if query_embedding is None:
        return [] # No embedding, no search

    pipeline = [{'$vectorSearch': {"index": VECTOR_INDEX_NAME, "path": "embedding", "queryVector": query_embedding, "numCandidates": 100, "limit": 10}}, {'$project': {"_id": 0, "description": 1, "topics": 1, "url": 1, "score": {"$meta": "vectorSearchScore"}}}]
    
    try:
        results = list(mongo_collection.aggregate(pipeline))
        return results
    except Exception as e:
        # Catch errors specifically from the MongoDB aggregation (e.g., malformed pipeline, index issues)
        raise RuntimeError(f"Error during MongoDB Vector Search: {e}. Check index and data configuration.") from e


# Function to Generate AI Explanation using Gemini
def generate_ai_explanation(summary_text: str) -> str:
    # `gemini_text_model` should be initialized; an error would have been raised at import if not.
    if not summary_text or not summary_text.strip():
        return "No explanation generated due to empty summary."

    explanation_prompt = (
        f"Based on the following summary of search results, provide a brief, easy-to-understand "
        f"explanation relevant to a developer, focusing on how these tools might help them build a SaaS product:\n\n"
        f"{summary_text}\n\nExplanation:"
    )

    response = gemini_text_model.generate_content(
        explanation_prompt,
        generation_config=GenerationConfig(
            temperature=0.7,
            max_output_tokens=300,
        ),
    )

    if response.candidates and response.candidates[0].content.parts:
        generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text') and part.text is not None)
        return generated_text.strip()
    else:
        return "Could not generate explanation."
