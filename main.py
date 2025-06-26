import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from dotenv import load_dotenv
load_dotenv()

# --- Secure Hugging Face Authentication ---
from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "Hugging Face token not found.\n"
        "1. Get token at: https://huggingface.co/settings/tokens\n"
        "2. Create a .env file with: HF_TOKEN=your_token_here\n"
        "3. Add .env to .gitignore"
    )
login(token=HF_TOKEN)

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
try:
    from fpdf import FPDF
except ImportError:
    try:
        from fpdf2 import FPDF
    except ImportError:
        st.error("PDF generation requires fpdf2. Install with: pip install fpdf2")
        FPDF = None
import datetime
import tempfile
import speech_recognition as sr
import pyttsx3
import pandas as pd
import altair as alt
from pydantic_settings import BaseSettings
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Enhanced Model Loading ---
@st.cache_resource
def load_llm():
    """Load Phi-3-mini with secure authentication"""
    cache_path = "./models"
    os.makedirs(cache_path, exist_ok=True)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        ) if device == "cuda" else None
        
        model_id = "microsoft/phi-3-mini-4k-instruct"  # Lowercase corrected
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            token=HF_TOKEN,  # Using env var
            cache_dir=cache_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_kwargs = {
            "token": HF_TOKEN,  # Using env var
            "cache_dir": cache_path,
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True
        }
        
        if device == "cuda" and bnb_config:
            model_kwargs.update({
                "device_map": "auto",
                "quantization_config": bnb_config
            })
        
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        
        logger.info("Model loaded successfully")
        return pipe
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error(f"""
        Model loading error: {str(e)}
        
        Troubleshooting:
        1. Verify token in .env is valid
        2. Check model access at: https://huggingface.co/{model_id}
        3. Ensure required packages are installed
        """)
        st.stop()

# --- Rest of your original script continues unchanged ---
[... ALL YOUR EXISTING CODE FOR VECTORSTORE, UI, CHAT FUNCTIONS, ETC ...]

# --- Updated QA Function with Auth ---
def ask_specbot(query):
    """Now uses the pre-authenticated token"""
    try:
        formatted_query = f"""<|system|>
You are a GHA specifications expert.<|end|>
<|user|>
{query.strip()}<|end|>
<|assistant|>"""
        
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": st.session_state.get("top_k_chunks", 4),
                "fetch_k": 8,
                "lambda_mult": 0.7
            }
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "memory": st.session_state.conversation_memory
            }
        )
        
        with st.spinner("🔍 Analyzing GHA specifications..."):
            result = qa_chain({"query": formatted_query})
            
        [...]  # Rest of your existing ask_specbot() function

# --- New Auth Status Indicator ---
with st.sidebar:
    [... your existing sidebar code ...]
    st.caption(f"🔒 Auth Status: {'✅' if HF_TOKEN else '❌'} Hugging Face")

[... ALL OTHER ORIGINAL CODE REMAINS EXACTLY THE SAME ...]
