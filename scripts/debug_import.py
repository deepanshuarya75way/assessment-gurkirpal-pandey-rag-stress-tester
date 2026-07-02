import sys
print("Python:", sys.executable)
print("Path:", sys.path[:3])

# test sentence_transformers directly
try:
    from sentence_transformers import SentenceTransformer
    print("SentenceTransformer imported OK!")
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    print("Model loaded!")
    vec = model.encode("test", normalize_embeddings=True)
    print(f"Vector: {len(vec)} dims")
except Exception as e:
    print(f"Error: {e}")