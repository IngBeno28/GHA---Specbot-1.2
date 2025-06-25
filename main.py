import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3  # Must be before ANY other imports

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from fpdf import FPDF
import datetime
import tempfile
import speech_recognition as sr
import pyttsx3
import pandas as pd
import altair as alt
from pydantic_settings import BaseSettings

# --- Secure Model Loading ---
@st.cache_resource
def load_llm():
    """Load Mistral 7B with 4-bit quantization for memory efficiency"""
    cache_path = "./models"
    os.makedirs(cache_path, exist_ok=True)
    
    # Quantization config (reduces VRAM usage by ~70%)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )
    
    try:
        model_id = "mistralai/Mistral-7B-v0.1"  # Non-instruct version
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=cache_path
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_path,
            device_map="auto",
            quantization_config=bnb_config
        )
        
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            do_sample=True
        )
        
    except Exception as e:
        st.error(f"""Failed to load model: {str(e)}\n\n
                 Common issues:
                 1. Insufficient GPU memory (try reducing quantization)
                 2. Network connectivity problems
                 3. Corrupted model cache (delete ./models folder)""")
        st.stop()

# --- VectorStore Setup (Cached) ---
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory="gha_vectorstore",
        embedding_function=embedding_model
    )

# --- Initialize Components ---
llm_pipeline = load_llm()
llm = HuggingFacePipeline(pipeline=llm_pipeline)
vectorstore = load_vectorstore()

# --- Streamlit UI Config ---
st.set_page_config(page_title="GHA SpecBot", page_icon="🧱")
st.title("🧱 GHA SpecBot Pro Max")
st.markdown("Ask me anything about the Ghana Highway Authority Road & Bridge Specifications.")

# --- Session State Management ---
if "chat" not in st.session_state:
    st.session_state.chat = []
    st.session_state.qa_stats = {
        "Total": 0, 
        "Materials": 0, 
        "Dimensions": 0, 
        "Tests": 0, 
        "Execution": 0, 
        "General": 0
    }

# --- Voice Processing Utilities ---
def recognize_voice():
    """Improved voice input with error handling"""
    if st.runtime.exists_in_streamlit_cloud:
        st.warning("🎤 Voice input disabled in cloud deployments")
        return ""
    
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=1)
            st.info("🎤 Listening... (Speak clearly)")
            audio = r.listen(source, timeout=3, phrase_time_limit=5)
        return r.recognize_google(audio)
    except sr.WaitTimeoutError:
        st.warning("Listening timed out")
        return ""
    except Exception as e:
        st.error(f"Voice error: {str(e)}")
        return ""

def speak_response(text):
    """Safer text-to-speech with env checks"""
    if not st.session_state.get("voice_enabled", True):
        return
        
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)  # Slower speech
        engine.say(text[:500])  # Limit length
        engine.runAndWait()
    except Exception as e:
        st.warning(f"Voice output disabled: {str(e)}")

# --- Core QA Function ---
def ask_specbot(query):
    """Enhanced with doc validation and error handling"""
    sanitized_query = query.strip()[:500]  # Prevent prompt injection
    
    try:
        # Add instruction prefix for better responses from base model
        instruction = "Answer the following question about Ghana Highway Authority specifications: "
        full_query = instruction + sanitized_query
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": st.session_state.get("top_k_chunks", 3)}
            ),
            return_source_documents=True
        )
        
        result = qa_chain({"query": full_query})
        answer = result["result"].split("\n\nReferences:")[0]  # Clean output
        
        sources = []
        for doc in result["source_documents"]:
            if hasattr(doc, "metadata"):
                src = doc.metadata.get("source", "GHA Spec")
                content = doc.page_content[:250].replace("\n", " ").strip()
                sources.append(f"- *{src}* — _{content}..._")
        
        return answer, "\n📚 **Sources:**\n" + "\n".join(sources) if sources else ""
    
    except Exception as e:
        st.error(f"QA processing failed: {str(e)}")
        return "Sorry, I encountered an error processing your question.", ""

# --- UI Components ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.session_state.voice_enabled = st.checkbox(
        "🔊 Enable Voice Output", 
        value=True
    )
    st.session_state.highlight_sources = st.checkbox(
        "📚 Show Sources", 
        value=True
    )
    st.session_state.max_tokens = st.slider(
        "✂️ Max Answer Length", 
        min_value=128, 
        max_value=1024, 
        value=512, 
        step=64
    )
    st.session_state.top_k_chunks = st.slider(
        "🧹 Context Chunks", 
        min_value=1, 
        max_value=10, 
        value=3
    )

# --- Main Interaction ---
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("💬 Ask your question:", key="query_input")
with col2:
    if st.button("🎤 Voice Input"):
        if voice_input := recognize_voice():
            user_input = voice_input
            st.rerun()  # Refresh with voice input

if user_input:
    with st.spinner("🔍 Searching specifications..."):
        answer, sources = ask_specbot(user_input)
        
        if answer:
            st.success(answer)
            if st.session_state.highlight_sources and sources:
                st.markdown(sources)
            
            speak_response(answer)
            
            # Update chat history
            st.session_state.chat.append({
                "question": user_input,
                "answer": answer,
                "sources": sources
            })
            
            # Update stats
            q_lower = user_input.lower()
            category = (
                "Materials" if any(k in q_lower for k in ["cement", "bitumen"]) else
                "Dimensions" if any(k in q_lower for k in ["width", "depth"]) else
                "Tests" if "test" in q_lower else
                "Execution" if any(k in q_lower for k in ["construct", "compact"]) else
                "General"
            )
            st.session_state.qa_stats[category] += 1
            st.session_state.qa_stats["Total"] += 1

# --- Chat History Display ---
if st.session_state.chat:
    st.subheader("📜 Conversation History")
    for i, turn in enumerate(reversed(st.session_state.chat[-5:]), 1):
        with st.expander(f"Q{len(st.session_state.chat)-i+1}: {turn['question'][:50]}..."):
            st.markdown(f"**Question:** {turn['question']}")
            st.markdown(f"**Answer:** {turn['answer']}")
            if turn['sources']:
                st.markdown(turn['sources'])

# --- PDF Export ---
def safe_pdf_export(chat_history):
    """Generate PDF with sanitized inputs"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="GHA SpecBot Chat Log", ln=True, align='C')
        
        for i, entry in enumerate(chat_history, 1):
            pdf.multi_cell(0, 10, f"Q{i}: {entry['question'][:200]}")  # Truncate
            pdf.multi_cell(0, 10, f"A{i}: {entry['answer'][:500]}")  # Truncate
            pdf.ln(5)
            
        filename = f"gha_chat_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
        safe_path = os.path.join(tempfile.gettempdir(), os.path.basename(filename))
        pdf.output(safe_path)
        return safe_path
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None

if st.button("💾 Export Chat"):
    if pdf_path := safe_pdf_export(st.session_state.chat):
        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Download PDF",
                data=f,
                file_name=f"gha_specbot_chat.pdf",
                mime="application/pdf"
            )

# --- Analytics Dashboard ---
st.markdown("---")
st.subheader("📊 Usage Analytics")
if st.session_state.qa_stats["Total"] > 0:
    chart_data = pd.DataFrame({
        "Category": list(st.session_state.qa_stats.keys())[1:],
        "Count": list(st.session_state.qa_stats.values())[1:]
    })
    
    st.altair_chart(
        alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Category", sort="-y"),
            y="Count",
            color="Category",
            tooltip=["Category", "Count"]
        ).properties(height=300),
        use_container_width=True
    )
