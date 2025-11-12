# Rule-based and keyword detection
def check_keywords(text, keywords):
    detected_keywords = [keyword for keyword in keywords if keyword in text.lower()]
    return detected_keywords

def check_homoglyphs(text, homoglyphs):
    detected_homoglyphs = [char for char in text if char in homoglyphs]
    return detected_homoglyphs  