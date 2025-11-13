# # End-to-end prediction pipeline
# import torch
# from .preprocessor import clean_text
# from .model_loader import load_model
# from .detection_helpers import check_homoglyphs, detect_phishing
# from .message_analysis import generate_response
# from .model_loader import device_check

# def analyze_message(message, include_technical=False):
#     """
#     Main analysis function
#     """
#     device = device_check()

#     if not message.strip():
#         return "Please enter a message to analyze."

#     try:
#         print(f"Analyzing message ({len(message)} chars)")

#         # Check homoglyphs
#         homoglyph_results = check_homoglyphs(message)
#         homoglyph_warning = ""
#         if homoglyph_results:
#             chars_list = ", ".join([f"'{h['char']}' ({h['looks_like']})" for h in homoglyph_results[:5]])
#             homoglyph_warning = f"\n\n Homoglyph Alert:\nFound {len(homoglyph_results)} suspicious character(s): {chars_list}"

#         # Get AI analysis
#         llm_analysis = detect_phishing(message)

#         # Combine results
#         full_analysis = f"AI Analysis:\n{llm_analysis}"

#         if homoglyph_warning:
#             full_analysis += f"\n\n---{homoglyph_warning}"

#         if include_technical:
#             device_info = f"{device.upper()}"
#             if device == "cuda":
#                 allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
#                 device_info += f" ({allocated:.2f}GB used)"
#             full_analysis += f"\n\n---\nTechnical:\n- Device: {device_info}\n- Model: Qwen2.5-0.5B-Instruct\n- Length: {len(message)} chars"

#         return full_analysis

#     except Exception as e:
#         error_msg = f" Error: {str(e)}"
#         print(error_msg)
#         return error_msg

# def run_pipeline(text: str) -> dict:
#     text_clean = clean_text(text)
#     # features = check_keywords(text_clean)
#     # features.update(check_homoglyphs(text_clean))
    
#     model, tokenizer, device = load_model()

#     response = detect_phishing(text_clean, response_function=generate_response)
