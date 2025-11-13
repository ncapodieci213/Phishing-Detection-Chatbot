# Rule-based and keyword detection
import json
import os
import torch

def check_keywords(text, keywords):
    detected_keywords = [keyword for keyword in keywords if keyword in text.lower()]
    return detected_keywords

def load_json_safe(filename, default=None):
    """
    Safely load JSON file with fallback
    """
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filename}: {e}")
    return default or {}

def check_homoglyphs(text):
    """Check for homoglyph attacks"""
    found = []
    homoglyphs = load_json_safe('homoglyphs.json')
    for i, char in enumerate(text):
        if char in homoglyphs:
            found.append({
                'char': char,
                'looks_like': homoglyphs[char],
                'position': i
            })
    return found

def generate_response(messages, model, tokenizer, device, max_new_tokens=400, temperature=0.3):
    """
    Generate text using Qwen2.5-0.5B-Instruct
    """
    try:
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer([text], return_tensors="pt")

        # Move to appropriate device
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        print(f"Generating response (max {max_new_tokens} tokens)...")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode only new tokens
        input_length = inputs['input_ids'].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        print(" Generation complete")
        return response.strip()

    except Exception as e:
        print(f"Error in generation: {e}")
        raise# Interface to LLaMA or other LLM for explanations

def detect_phishing(message, model, tokenizer, device):
    """
    Analyze a message for phishing indicators
    """

    messages = [
        {
            "role": "system",
            "content": """You are a phishing detection expert. Analyze messages for phishing signs:
- Suspicious URLs or domains (typosquatting, unusual TLDs)
- Urgency tactics ("act now", "limited time")
- Requests for sensitive info (passwords, SSN, credit cards)
- Impersonation attempts (fake brands, officials)
- Poor grammar or spelling
- Generic greetings ("Dear customer")
- Suspicious attachments or links

If NO suspicious signs are found:
- Briefly state (1-2) sentences that the user is likely not being scammed.

Provide a clear analysis with:
1. Risk Level: Low/Medium/High
2. Key indicators found
3. Brief explanation (2-3 sentences)
"""
        },
        {
            "role": "user",
            "content": f"Analyze this message for phishing:\n\n{message}"
        }
    ]

    response = generate_response(messages, model, tokenizer, device)
    return response