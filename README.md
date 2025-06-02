# FAQ Chatbot with LangChain and Streamlit

This is a simple chatbot built using Streamlit, LangChain, and Google's Gemini model. It lets you upload a document (PDF, TXT, or Markdown) and ask questions based on the content.

This project was created for learning purposes to explore how document-based question answering works using embeddings, vector stores, and LLMs.

## Live Demo

[https://faq-chatbot-lc.streamlit.app](https://faq-chatbot-lc.streamlit.app)

> **Note:** The deployed demo is not fully functional because it requires additional database setup. Some features may not work as expected.
## Setup

1. Clone the repository or copy the script.

2. Create a `.env` file in the root folder and add your API key:

    ```ini
    GOOGLE_API_KEY=your_google_generative_ai_key_here
    ```

3. Run the app with:

    ```bash
    streamlit run app.py
    ```

## How It Works

- Upload a supported file (pdf, txt, md)
- The text is split into chunks
- Chunks are embedded and stored in a temporary vector database
- When you ask a question, similar chunks are retrieved
- The model generates an answer based on the context