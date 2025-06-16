# Core Backend Functions

import os
from pymongo import MongoClient
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig

MONGO_URI = os.getenv("MONGO_URI")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "Meta_data")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "Test Data")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_index")
EMBEDDING_MODEL_NAME = "text-embedding-004"
GEMINI_TEXT_MODEL_ID = "gemini-2.0-flash"

def initialize_google_cloud_models():
    if not GCP_PROJECT_ID:
        raise ValueError("GCP_PROJECT_ID environment variable is not set. Cannot initialize Google Cloud AI.")
    vertexai.init(project=GCP_PROJECT_ID, location="us-central1")
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    gemini_text_model = GenerativeModel(GEMINI_TEXT_MODEL_ID)
    return embedding_model, gemini_text_model

embedding_model, gemini_text_model = initialize_google_cloud_models()

def get_mongo_collection():
    if not MONGO_URI:
        raise ValueError("MONGO_URI environment variable is not set. Cannot connect to MongoDB Atlas.")
    cleaned_mongo_uri = MONGO_URI.strip('"')
    client = MongoClient(cleaned_mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    return collection
mongo_collection = get_mongo_collection()

def get_embedding(text_content: str) -> list[float]:
    max_chars = 25000
    if len(text_content) > max_chars:
        text_content = text_content[:max_chars]
    embeddings = embedding_model.get_embeddings([text_content])
    return embeddings[0].values

def perform_vector_search(query_embedding: list[float]) -> list[dict]:
    if query_embedding is None:
        return []
    pipeline = [{'$vectorSearch': {"index": VECTOR_INDEX_NAME, "path": "embedding", "queryVector": query_embedding, "numCandidates": 100, "limit": 10}}, {'$project': {"_id": 0, "description": 1, "topics": 1, "url": 1, "score": {"$meta": "vectorSearchScore"}}}]
    try:
        results = list(mongo_collection.aggregate(pipeline))
        return results
    except Exception as e:
        raise RuntimeError(f"Error during MongoDB Vector Search: {e}. Check index and data configuration.") from e

def generate_ai_explanation(summary_text: str) -> str:
    if not summary_text or not summary_text.strip():
        return "No explanation generated due to empty summary."
    explanation_prompt = ( # Parentheses enclose the multi-line string
        f"Based on the following summary of search results, provide a brief, easy-to-understand "
        f"explanation relevant to a developer, focusing on how these tools might help them build a SaaS product:\n\n"
        f"{summary_text}\n\nExplanation:"
    )
    response = gemini_text_model.generate_content(
        explanation_prompt,
        generation_config=GenerationConfig(temperature=0.7, max_output_tokens=300,) # All brackets on the same line
    )
    if response.candidates and response.candidates[0].content.parts:
        generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text') and part.text is not None)
        return generated_text.strip()
    else:
        return "Could not generate explanation."
