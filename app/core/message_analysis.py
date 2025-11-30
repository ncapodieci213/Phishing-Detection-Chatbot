import torch
from .detection_helpers import check_homoglyphs, detect_phishing
from .model_loader import load_model
from ..utils.url_checker import analyze_text_for_urls_and_homoglyphs

def analyze_message(message, include_technical=False):
    """
    Main analysis function
    """
    model, tokenizer, device = load_model()

    if not message.strip():
        return "Please enter a message to analyze."

    try:
        print(f"Analyzing message ({len(message)} chars)")

        # Check homoglyphs in text and urls/domains
        url_analysis = analyze_text_for_urls_and_homoglyphs(message)
        homoglyph_warning = ""

        # Text-level homoglyphs
        text_homoglyphs = url_analysis.get('text_homoglyphs', [])
        if text_homoglyphs:
            chars_list = ", ".join([f"'{h['char']}' ({h['looks_like']})" for h in text_homoglyphs[:5]])
            homoglyph_warning = f"\n\nHomoglyph Alert (text): Found {len(text_homoglyphs)} suspicious character(s): {chars_list}"

        # Domain-level homoglyphs
        domain_warnings = []
        for entry in url_analysis.get('domains', []):
            domain = entry.get('domain')
            labels = entry.get('homoglyphs', {}).get('labels', [])
            problematic = [l for l in labels if l.get('issues')]
            if problematic:
                label_summaries = []
                for lab in problematic:
                    label_summaries.append(f"{lab['label']} -> {lab['normalized']}")
                domain_warnings.append(f"{domain} ({'; '.join(label_summaries)})")

        if domain_warnings:
            domain_list = ", ".join(domain_warnings[:5])
            homoglyph_warning += f"\n\nHomoglyph Alert (domains): Suspicious domain labels: {domain_list}"

        # Get AI analysis
        llm_analysis = detect_phishing(message, model, tokenizer, device)

        # Combine results
        full_analysis = f"AI Analysis:\n{llm_analysis}"

        if homoglyph_warning:
            full_analysis += f"\n\n---{homoglyph_warning}"

        if include_technical:
            device_info = f"{device.upper()}"
            if device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
                device_info += f" ({allocated:.2f}GB used)"
            full_analysis += f"\n\n---\nTechnical:\n- Device: {device_info}\n- Model: Qwen2.5-0.5B-Instruct\n- Length: {len(message)} chars"

        return full_analysis

    except Exception as e:
        error_msg = f" Error: {str(e)}"
        print(error_msg)
        return error_msg