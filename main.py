"""
Phishing Detection Chatbot using Qwen2.5-0.5B-Instruct and Gradio
Optimized for smart GPU/CPU offloading
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import json
import os

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

# Load model and tokenizer - Qwen2.5-0.5B-Instruct
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


# Load JSON knowledge bases
"""
EDIT THIS DEF WITH JSON FILE PORTION
"""
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


homoglyphs = load_json_safe('homoglyphs.json')


def generate_response(messages, max_new_tokens=400, temperature=0.3):
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


def detect_phishing(message):
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

    response = generate_response(messages, max_new_tokens=400, temperature=0.3)
    return response


def check_homoglyphs(text):
    """Check for homoglyph attacks"""
    found = []
    for i, char in enumerate(text):
        if char in homoglyphs:
            found.append({
                'char': char,
                'looks_like': homoglyphs[char],
                'position': i
            })
    return found


def analyze_message(message, include_technical=False):
    """
    Main analysis function
    """
    if not message.strip():
        return "Please enter a message to analyze."

    try:
        print(f"Analyzing message ({len(message)} chars)")

        # Check homoglyphs
        homoglyph_results = check_homoglyphs(message)
        homoglyph_warning = ""
        if homoglyph_results:
            chars_list = ", ".join([f"'{h['char']}' ({h['looks_like']})" for h in homoglyph_results[:5]])
            homoglyph_warning = f"\n\n Homoglyph Alert:\nFound {len(homoglyph_results)} suspicious character(s): {chars_list}"

        # Get AI analysis
        llm_analysis = detect_phishing(message)

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


# Gradio interface
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
    status_box = gr.Markdown(f"**Status:** Model loaded on **{device_status}**")

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
