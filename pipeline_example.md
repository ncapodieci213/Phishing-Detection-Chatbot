anti-phishing-chatbot/
│
├── app/                          # Main application code
│   ├── __init__.py
│   ├── main.py                   # Entry point (FastAPI/Gradio interface)
│   ├── api.py                    # FastAPI endpoints (if serving backend)
│   ├── interface/                # Frontend layer
│   │   ├── gradio_ui.py          # Gradio UI layout & callbacks
│   │   ├── components/           # (optional) custom Gradio components
│   │   └── assets/               # logos, icons, CSS overrides
│   │
│   ├── core/                     # Core ML and logic
│   │   ├── pipeline.py           # End-to-end prediction pipeline
│   │   ├── model_loader.py       # Load or cache ML & LLM models
│   │   ├── preprocessor.py       # Text cleaning, normalization, tokenization
│   │   ├── phishing_rules.py     # Rule-based and keyword detection
│   │   └── llm_response.py       # Interface to LLaMA or other LLM for explanations
│   │
│   ├── utils/                    # Helpers
│   │   ├── logger.py             # Logging and monitoring
│   │   ├── url_checker.py        # Extract/validate URLs
│   │   └── config_loader.py      # Load JSON configs and environment vars
│   │
│   └── models/                   # Trained models & vectorizers
│       ├── phishing_classifier.pkl
│       └── vectorizer.pkl
│
├── knowledge_base/               # JSON knowledge files for reference
│   ├── homoglyphs.json
│   ├── suspicious_keywords.json
│   ├── phishing_examples.json
│   ├── domain_blacklist.json
│   └── url_patterns.json
│
├── tests/                        # Unit/integration tests
│   ├── test_pipeline.py
│   ├── test_phishing_rules.py
│   ├── test_ui.py
│   └── test_api.py
│
├── notebooks/                    # Jupyter notebooks for model training or EDA
│   ├── train_model.ipynb
│   └── data_exploration.ipynb
│
├── data/                         # Raw and processed data (not versioned if large)
│   ├── raw/
│   └── processed/
│
├── scripts/                      # CLI utilities
│   ├── run_server.sh
│   ├── evaluate_model.py
│   └── update_kb.py              # Update knowledge base files
│
├── requirements.txt              # Python dependencies
├── config.yaml                   # Global config (model paths, thresholds, etc.)
├── .env                          # Environment variables (API keys, model paths)
├── .gitignore
├── README.md
└── LICENSE
