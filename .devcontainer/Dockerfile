FROM python:3.10-slim

# Set up working directory
WORKDIR /workspace

# Install required packages
RUN pip install --upgrade pip && \
    pip install langchain chromadb sentence-transformers fpdf \
    streamlit pandas altair transformers PyMuPDF pyttsx3 SpeechRecognition

# Optional: Expose a Streamlit port
EXPOSE 8501
