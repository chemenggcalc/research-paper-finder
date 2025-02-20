import os

class Config:
    PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "chemenggcalc@gmail.com")
    MAX_RESULTS = int(os.getenv("MAX_RESULTS", 15))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.3))