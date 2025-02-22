## Overview
A real-time voice-enabled AI assistant that transcribes speech, processes queries using LLMs, and responds with synthesized speech. It supports dynamic emotional intelligence to adapt responses based on user sentiment and can escalate to a human agent when needed.

## Features:
- 🎤 Speech-to-Text using OpenAI Whisper

- 🧠 LLM Integration with GPT-4o / Groq for intelligent responses

- 🔊 Text-to-Speech using Google Text-to-Speech (gTTS) (currently)

- 🎭 Emotion Recognition for dynamic response adaptation

- 📞 Human Escalation for unresolved issues

- 📝 Chat Memory to retain conversation context

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sravyach136/Voice-Enabled-AI-Assistant.git
   
2. Navigate to the project directory:
    
3. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
4. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. Create a `.env` file with OpenAI API KEY/Groq KEY
   
2. Run the main application script:
    ```bash
    streamlit run src/app.py




