# packages/models/constants.py

# These labels must match the training data categories
MEDICAL_LABELS = [
    "cardiology",
    "neurology",
    "oncology",
    "radiology",
    "immunology",
    "genetics",
    "infectious_diseases",
    "gastroenterology",
    "endocrinology",
    "dermatology"
]

# Special tokens or markers (if used)
SECTION_TOKENS = {
    "title": "<TITLE>",
    "abstract": "<ABSTRACT>",
    "conclusion": "<CONCLUSION>"
}

# Example use case: for RAG chunking or section-aware embedding
DEFAULT_CHUNK_SIZE = 300  # tokens
DEFAULT_CHUNK_OVERLAP = 50
