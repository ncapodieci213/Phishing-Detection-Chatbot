# Visiting http://localhost:7860/ opens the Gradio UI.
# Hitting http://localhost:7860/api/predict accesses the FastAPI JSON API

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import uvicorn
import torch
from app.core.model_loader import device_check
from app.core.message_analysis import analyze_message

# from app.api import router as api_router

# ------------------------------------------------------------
# Initialize FastAPI app
# ------------------------------------------------------------
app = FastAPI(
    title="Anti-Phishing Chatbot API",
    description="API + Web UI for phishing detection and explanation",
    version="1.0.0",
)

# Enable CORS to host frontend separately
# We may not need this in the end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST routes
# app.include_router(api_router, prefix="/api")

# ------------------------------------------------------------
# Mount Gradio as a sub-application
# ------------------------------------------------------------
device = device_check()

with gr.Blocks(theme=gr.themes.Default(), title="Phishing Detector") as gradio_interface:
    
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

app = gr.mount_gradio_app(app, gradio_interface, path="/")

# ------------------------------------------------------------
# Run both services
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)
