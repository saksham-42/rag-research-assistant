import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=768
)

def embed_texts(texts):
    return embedding_model.embed_documents(texts)

def embed_text(text):
    return embedding_model.embed_query(text)

if __name__ == "__main__":
    test_text = "Titanium Alloys have high strength to weight ratio"
    vector = embed_text(test_text)
    print(f"API Key Found : {bool(os.getenv('GEMINI_API_KEY'))}")
    print (f"Vector Dimensions : {len(vector)}")
    print (f"First 10 values : {vector[:10]}")
    