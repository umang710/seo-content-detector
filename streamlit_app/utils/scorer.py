import pandas as pd

def predict_quality(model, word_count, sentence_count, readability):
    """Predict quality using trained model"""
    features = pd.DataFrame([{
        'word_count': word_count,
        'sentence_count': sentence_count,
        'flesch_reading_ease': readability
    }])
    
    return model.predict(features)[0]