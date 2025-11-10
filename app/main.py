# Visiting http://localhost:7860/ opens the Gradio UI.
# Hitting http://localhost:7860/api/predict accesses the FastAPI JSON API

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import uvicorn

from app.api import router as api_router
from app.interface.gradio_ui import create_gradio_interface

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
app.include_router(api_router, prefix="/api")

# ------------------------------------------------------------
# Mount Gradio as a sub-application
# ------------------------------------------------------------
gradio_interface = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_interface, path="/")

# ------------------------------------------------------------
# Run both services
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)
