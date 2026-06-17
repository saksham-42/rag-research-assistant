import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embed_texts(texts):
    contents = [types.Content(parts=[types.Part.from_text(text=t)])
                for t in texts]
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=contents,
        config={"output_dimensionality":768}
    )
    return [e.values for e in response.embeddings]  

def embed_text(text):
    return embed_texts([text])[0]

if __name__ == "__main__":
    test_text = "Titanium Alloys have high strength to weight ratio"
    vector = embed_texts([test_text])
    print (f"Vector Dimensions : {len(vector[0])}")
    print (f"First 10 values : {vector[:10]}")
    