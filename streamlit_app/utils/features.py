import re
from textstat import flesch_reading_ease

def calculate_features(body_text, word_count):
    """Calculate basic features from text"""
    sentence_count = len(re.findall(r'[.!?]+', body_text)) if body_text else 0
    readability = flesch_reading_ease(body_text) if body_text and len(body_text.split()) > 10 else 0
    is_thin = word_count < 500
    
    return sentence_count, readability, is_thin