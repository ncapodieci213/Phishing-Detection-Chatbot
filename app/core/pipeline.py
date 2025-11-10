# End-to-end prediction pipeline

# Simplified example
from .preprocessor import clean_text
from .model_loader import load_model
from .phishing_rules import check_keywords, check_homoglyphs
# from .llm_response import generate_explanation

def run_pipeline(text: str) -> dict:
    text_clean = clean_text(text)
    features = check_keywords(text_clean)
    features.update(check_homoglyphs(text_clean))
    
    model = load_model()
    prob = model.predict_proba([text_clean])[0, 1]
    label = "Phishing" if prob > 0.5 else "Legitimate"
    
    explanation = (
        "This message contains patterns similar to known phishing attempts."
        if label == "Phishing"
        else "No major phishing indicators detected."
    )
    return {
        "label": label,
        "probability": prob,
        "explanation": explanation,
    }
