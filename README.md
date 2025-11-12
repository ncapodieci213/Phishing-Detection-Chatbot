# Phishing Detection Chatbot

Chatbot to detect whether a user is getting phished.

## Description

A text-to-text chatbot that can assist users with evaluations of possible phishing attempts. 
User pastes in text, email, or message that they believe may be phishing.
Chatbot responds with risk level, key indicators in evaluation, and brief explanation.
For demo purposes, chatbot is downloaded and localized to user's computer; data is not sent back to servers.

## Built With
* [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](#)
* [![Gradio](https://img.shields.io/badge/Gradio-F97316?logo=Gradio&logoColor=white)](#)
* [![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)

## Getting Started



We recommend installing dependencies in a Python virtual environment using Python 3.10
```
pip install -r requirements.txt
```

After dependencies are installed, run the program with:
```
python main.py
```
This will create a localhost link for your web browser. 
Click on the link to interact with the chatbot.

## Usage

The chatbot is a text-to-text and will analyze pasted messages for signs of phishing.
This is not a multi-turn or conversational application.

## Authors

Contributors:\
Eghosa Awo-Osagie\
Justin Burr\
Noelle Capodieci\
Viswamohan Komati\
Kiana Vu
