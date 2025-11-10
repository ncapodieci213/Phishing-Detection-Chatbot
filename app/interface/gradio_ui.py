# Integration with Gradio for web-based user interface
import gradio as gr
from app.core.pipeline import run_pipeline

def create_gradio_interface():
    def predict_fn(text):
        result = run_pipeline(text)
        label = result["label"]
        prob = result["probability"]
        explanation = result["explanation"]
        return f"**Prediction:** {label}\n**Confidence:** {prob:.2f}\n\n{explanation}"

    with gr.Blocks(title="Anti-Phishing Chatbot") as demo:
        gr.Markdown("## 🛡️ Anti-Phishing Chatbot")
        gr.Markdown(
            "Paste an email or text message below to check if it might be a scam."
        )
        
        user_input = gr.Textbox(
            label="Enter suspicious text",
            placeholder="e.g., 'Your bank account has been suspended. Click this link...'",
            lines=5,
        )
        
        output = gr.Markdown(label="Result")
        submit_btn = gr.Button("Analyze")

        submit_btn.click(fn=predict_fn, inputs=user_input, outputs=output)
    return demo
