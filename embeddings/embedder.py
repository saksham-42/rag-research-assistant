import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embed_texts(text):
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text,
        config={"output_dimensionality":768}
    )
    return response.embeddings[0].values

if __name__ == "__main__":
    test_text = "Titanium Alloys have high strength to weight ratio"
    vector = embed_texts(test_text)
    print (f"Vector Dimensions : {len(vector)}")
    print (f"First 10 values : {vector[:10]}")
    