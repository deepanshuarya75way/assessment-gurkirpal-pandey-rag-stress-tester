import sys
print(f"Python: {sys.executable}")

try:
    from InstructorEmbedding import INSTRUCTOR
    print("InstructorEmbedding imported successfully!")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Other error: {e}")