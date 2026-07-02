from google import genai
from dotenv import load_dotenv
import os
load_dotenv()

try:
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    result = client.models.embed_content(
        model='gemini-embedding-001',
        contents='test quota')
    print("Quota available!")
    print(f"Vector length: {len(result.embeddings[0].values)}")
except Exception as e:
    print(f"Quota exhausted: {e}")