"""
Application: Phishing Detection Chatbot using Qwen2.5-0.5B-Instruct and Gradio
Deploys locally in web browser
Compatible with both CPU/GPU
"""
# libraries for LLM implementation
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import warnings

# libraries for UI implementation
import gradio as gr

# libraries for URL and homoglyph detection
import re
from urllib.parse import urlparse
from typing import List, Dict

warnings.filterwarnings('ignore')

print("Initializing phishing detection system...")

# Check GPU availability
if torch.cuda.is_available():
    print(f" CUDA available: {torch.cuda.get_device_name(0)}")
    print(f" CUDA version: {torch.version.cuda}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"  GPU Memory: {total_vram:.2f} GB")
    device = "cuda"
else:
    print(" CUDA not available, using CPU (slower)")
    device = "cpu"

# Load model and tokenizer: Qwen2.5-0.5B-Instruct
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")

try:
    if device == "cuda":
        # Attempt to load model in FP16 first
        print("Attempting to load model in FP16 on GPU...")

        # Clear cache first
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",  # Automatically handles GPU/CPU split if GPU VRAM is not enough
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "5GB", "cpu": "8GB"}  # Reserve 5GB for GPU, rest for CPU
        )

        print(" Model loaded successfully on GPU")
        print(f" Device map: {model.hf_device_map}")

        # Check VRAM usage
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        print(f"  VRAM allocated: {allocated:.2f} GB")
        print(f"  VRAM reserved: {reserved:.2f} GB")

    else:
        # CPU loading
        print("Loading model on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model = model.to(device)
        print(" Model loaded on CPU")

except Exception as e:
    print(f" Error loading model: {e}")
    print("\nAttempting fallback: CPU offloading...")

    try:
        # Fallback: More CPU offloading
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "4GB", "cpu": "12GB"} if device == "cuda" else None
        )
        print(" Model loaded with CPU offloading")
    except Exception as e2:
        print(f" Fallback failed: {e2}")
        raise

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


# Load homoglyphs mapping
homoglyphs = load_json_safe('homoglyphs.json')

# URL extraction and domain analysis
URL_REGEX = re.compile(r"((?:https?://|http://|www\.)[^\s,;]+)", re.IGNORECASE)

def extract_urls(text: str) -> List[str]:
    """
    Return a list of URL-like substrings found in `text`.
    """
    return URL_REGEX.findall(text or "")

def _normalize_netloc(netloc: str) -> str:
    """
    Remove credentials and port from netloc
    """
    if '@' in netloc:
        netloc = netloc.split('@', 1)[1]
    if ':' in netloc:
        netloc = netloc.split(':', 1)[0]
    return netloc.strip().strip('.')


def get_domain_from_url(url: str) -> str:
    """
    Parse a URL-like string and return the domain/netloc.
    If the input lacks a scheme, a scheme will be assumed so parsing works.
    """
    if not url.lower().startswith(('http://', 'https://')):
        url = 'http://' + url
    try:
        p = urlparse(url)
        return _normalize_netloc(p.netloc)
    except Exception:
        return url


def detect_homoglyphs_in_text(text: str, mapping: dict = None) -> List[Dict]:
    """
    Detect homoglyph characters anywhere in a text string.
    Returns list of dicts: `char`, `looks_like`, `position`.
    """
    if mapping is None:
        mapping = homoglyphs

    if not mapping:
        return []

    found = []
    for i, ch in enumerate(text or ""):
        if ch in mapping:
            found.append({
                'char': ch,
                'looks_like': mapping[ch],
                'position': i
            })
    return found


def detect_homoglyphs_in_domain(domain: str, mapping: dict = None) -> Dict:
    """
    Check each label in `domain` for homoglyphs and return a structured result.
    Returned dict example:
    {
      'domain': 'xn--exmple-9ta.com',
      'labels': [
         { 'label': 'exаmple', 'issues': [ {char, looks_like, pos}, ... ], 'normalized': 'example' }
      ]
    }
    """
    if mapping is None:
        mapping = homoglyphs
    labels = []

    if not domain:
        return {'domain': domain, 'labels': labels}

    for label in domain.split('.'):
        issues = []
        normalized_chars = []
        for i, ch in enumerate(label):
            if ch in mapping:
                issues.append({'char': ch, 'looks_like': mapping[ch], 'position': i})
                normalized_chars.append(mapping[ch])
            else:
                normalized_chars.append(ch)
        labels.append({
            'label': label,
            'issues': issues,
            'normalized': ''.join(normalized_chars)
        })

    return {'domain': domain, 'labels': labels}


def analyze_text_for_urls_and_homoglyphs(text: str, mapping: dict = None) -> Dict:
    """
    High-level helper: extract URLs and run homoglyph checks on domains.
    Returns dict with `urls` and `domains` results.
    """
    if mapping is None:
        mapping = homoglyphs
    urls = extract_urls(text)
    domains = []

    for u in urls:
        domain = get_domain_from_url(u)
        domains.append({
            'url': u,
            'domain': domain,
            'homoglyphs': detect_homoglyphs_in_domain(domain, mapping)
        })

    # Also check plaintext for single-character homoglyphs
    text_homoglyphs = detect_homoglyphs_in_text(text, mapping)

    return {'urls': urls, 'domains': domains, 'text_homoglyphs': text_homoglyphs}

# LLM generation
def generate_response(messages, max_new_tokens=300, temperature=0.1):
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
        raise


def verify_claims_in_text(response: str, original_message: str) -> str:
    """
    post processing: verify that claims in the AI response are actually present in the original message.
    Flags suspicious claims that might be hallucinations. Attempts to reduce hallucinations.
    """
    # Keywords that indicate specific data requests (common hallucinations)
    sensitive_keywords = [
        'password', 'ssn', 'social security', 'credit card', 'bank account',
        'pin', 'cvv', 'routing number', 'account number', 'passport'
    ]

    original_lower = original_message.lower()
    response_lower = response.lower()

    warnings = []

    # Check if response mentions sensitive data requests not in original
    for keyword in sensitive_keywords:
        if keyword in response_lower and keyword not in original_lower:
            warnings.append(f"⚠ AI mentioned '{keyword}' but this word is NOT in the original message")

    if warnings:
        verification_note = "\n\n---\n**VERIFICATION ALERT:**\n" + "\n".join(warnings)
        verification_note += "\n\n*Note: The AI may have made assumptions. Always verify against the actual message text.*"
        return response + verification_note

    return response


def detect_phishing(message, url_analysis):
    """
    Analyze a message for phishing indicators
    """
    # Build context about URLs if any were found
    technical_findings = []

    if url_analysis['urls']:
        technical_findings.append(f"- Found {len(url_analysis['urls'])} URL(s)")
        for domain_info in url_analysis['domains']:
            technical_findings.append(f"- URL: {domain_info['url']}")
            technical_findings.append(f"  Domain: {domain_info['domain']}")

            # Check for homoglyphs in domain
            homoglyph_data = domain_info['homoglyphs']
            for label_info in homoglyph_data['labels']:
                if label_info['issues']:
                    technical_findings.append(f"  HOMOGLYPH DETECTED: '{label_info['label']}' mimics '{label_info['normalized']}'")

    if url_analysis['text_homoglyphs']:
        technical_findings.append(f"- Found {len(url_analysis['text_homoglyphs'])} suspicious character(s) in text")

    url_context = ""
    if technical_findings:
        url_context = "\n\n=== TECHNICAL ANALYSIS ===\n" + "\n".join(technical_findings)

    messages = [
        {
            "role": "system",
            "content": """You are a phishing detection expert. Analyze ONLY what is actually present in the message.

CRITICAL RULES:
1. ONLY mention details that are explicitly present in the message text
2. DO NOT invent or assume requests for information that aren't there
3. Base your analysis on the TECHNICAL ANALYSIS section if provided
4. If homoglyphs are detected, this is a HIGH risk indicator

Analyze for these signs:
- Suspicious URLs (check technical analysis)
- Urgency language ("act now", "limited time", "urgent")
- Actual requests for sensitive info (only if explicitly mentioned)
- Impersonation attempts
- Poor grammar/spelling
- Generic greetings

OUTPUT FORMAT:
Risk Level: [Low/Medium/High]
Key Findings: [List only what's actually present]
Explanation: [2-3 sentences based on actual content]"""
        },
        {
            "role": "user",
            "content": f"Analyze this message for phishing.{url_context}\n\n=== MESSAGE TEXT ===\n{message}\n"
        }
    ]

    response = generate_response(messages, max_new_tokens=300, temperature=0.1)

    # post-processing to verify claims
    verified_response = verify_claims_in_text(response, message)

    return verified_response


def analyze_message(message, include_technical=False):
    """
    Main analysis function
    """
    if not message.strip():
        return "Please enter a message to analyze."

    try:
        print(f"Analyzing message ({len(message)} chars)")

        # Perform comprehensive URL and homoglyph analysis
        url_analysis = analyze_text_for_urls_and_homoglyphs(message, homoglyphs)

        # Build warning sections
        warnings = []

        # URL warnings
        if url_analysis['urls']:
            url_warning = f"📎 **URLs Detected:** {len(url_analysis['urls'])} URL(s) found\n"
            for domain_info in url_analysis['domains']:
                url_warning += f"  • {domain_info['url']}\n"
                url_warning += f"    Domain: {domain_info['domain']}\n"

                # Check for domain homoglyphs
                homoglyph_data = domain_info['homoglyphs']
                for label_info in homoglyph_data['labels']:
                    if label_info['issues']:
                        url_warning += f"     Homoglyph detected: '{label_info['label']}' → may look like '{label_info['normalized']}'\n"

            warnings.append(url_warning)

        # Text homoglyph warnings
        if url_analysis['text_homoglyphs']:
            chars_list = ", ".join([
                f"'{h['char']}' (looks like '{h['looks_like']}')"
                for h in url_analysis['text_homoglyphs'][:5]
            ])
            homoglyph_warning = f" **Homoglyph Alert:**\nFound {len(url_analysis['text_homoglyphs'])} suspicious character(s): {chars_list}"
            if len(url_analysis['text_homoglyphs']) > 5:
                homoglyph_warning += f"\n...and {len(url_analysis['text_homoglyphs']) - 5} more"
            warnings.append(homoglyph_warning)

        # Get AI analysis with URL context (now includes post-processing)
        llm_analysis = detect_phishing(message, url_analysis)

        # Combine results
        full_analysis = f"**AI Analysis:**\n{llm_analysis}"

        if warnings:
            full_analysis += "\n\n---\n" + "\n\n".join(warnings)

        if include_technical:
            device_info = f"{device.upper()}"
            if device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
                device_info += f" ({allocated:.2f}GB used)"

            tech_info = f"\n\n---\n**Technical Details:**\n"
            tech_info += f"- Device: {device_info}\n"
            tech_info += f"- Model: Qwen2.5-0.5B-Instruct\n"
            tech_info += f"- Message length: {len(message)} chars\n"
            tech_info += f"- URLs extracted: {len(url_analysis['urls'])}\n"
            tech_info += f"- Homoglyphs found: {len(url_analysis['text_homoglyphs'])}\n"
            tech_info += f"- Temperature: 0.1 \n"
            tech_info += f"- Post-processing: Enabled"

            full_analysis += tech_info

        return full_analysis

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg

# gradio UI
with gr.Blocks(theme=gr.themes.Default(), title="Phishing Detector") as demo:
    gr.Markdown("""
    # AI Phishing Detection Assistant
    ### Powered by Qwen2.5-0.5B-Instruct
    """)

    # Status indicator
    device_status = f"{device.upper()}"
    if device == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        device_status += f" ({allocated:.2f}GB VRAM used)"
    status_box = gr.Markdown(f"**Status:** ✓ Model loaded on **{device_status}**")

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Message to Analyze",
                placeholder="Paste suspicious message here...",
                lines=8
            )

            with gr.Row():
                analyze_btn = gr.Button(" Analyze", variant="primary")
                clear_btn = gr.ClearButton([input_text], value="Clear")

            technical_checkbox = gr.Checkbox(label="Show technical details", value=False)

        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Analysis Results", lines=15)

    gr.Examples(
        examples=[
            ["Your account has been compromised! Click here: http://paypa1.com/verify"],
            ["Dear customer, confirm your SSN and password for security verification."],
            ["URGENT: You won $1,000,000! Send credit card details to claim NOW!"],
            ["Hi John, the project deadline is next Friday. Let me know if you need help."],
            ["Please verify your account at www.g00gle.com/secure"],
        ],
        inputs=input_text
    )

    analyze_btn.click(
        fn=analyze_message,
        inputs=[input_text, technical_checkbox],
        outputs=output_text
    )

if __name__ == "__main__":
    print(" Starting Gradio...")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
