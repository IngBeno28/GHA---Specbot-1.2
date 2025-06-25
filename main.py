import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA  # Added missing import
from fpdf import FPDF
import datetime
import tempfile
import speech_recognition as sr
import pyttsx3
import pandas as pd
import altair as alt
from pydantic_settings import BaseSettings

# --- Setup ---
persist_dir = "gha_vectorstore"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

# Load Mistral 7B (adjust this block if using HF Inference)
cache_path = "./models"
os.makedirs(cache_path, exist_ok=True)
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="./models, token=True")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models")

# Load LLM
max_tokens = 512
llm_pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=max_tokens, 
    temperature=0.2
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Retrieval Setup
top_k_chunks = 3
retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_chunks})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# --- Streamlit Config ---
st.set_page_config(page_title="GHA SpecBot", page_icon="🧱")
st.title("🧱 GHA SpecBot Pro Max")
st.markdown("Ask me anything about the Ghana Highway Authority Road & Bridge Specifications.")

# --- Sidebar Settings ---
st.sidebar.title("⚙️ Settings")
voice_enabled = st.sidebar.checkbox("🔊 Enable Voice Output", value=True)
highlight_sources = st.sidebar.checkbox("📚 Show Source Highlights", value=True)
max_tokens = st.sidebar.slider("✂️ Max Answer Length", min_value=128, max_value=1024, value=512, step=64)

top_k_chunks = st.sidebar.slider(
    "🧹 Number of Chunks to Retrieve",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

st.sidebar.caption("🔍 Higher values = more context, slower but smarter answers.")

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Session State Init ---
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

# --- Voice Input ---
def recognize_voice():
    if st.runtime.exists_in_streamlit_cloud:
        st.warning("🎤 Voice input is not supported in Streamlit Cloud.")
        return ""
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now")
            audio = r.listen(source, timeout=5)
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        st.warning("Sorry, I couldn't understand that.")
        return ""
    except sr.RequestError:
        st.error("Voice service failed.")
        return ""
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return ""

# --- Text-to-Speech ---
def speak_response(text):
    if voice_enabled and not st.runtime.exists_in_streamlit_cloud:
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            st.warning(f"Text-to-speech not available in this environment: {e}")

# --- Ask SpecBot ---
def ask_specbot(query):
    try:
        result = qa_chain({"query": query})  
        docs = result["source_documents"]
        answer = result["result"]
        source_info = "\n\n📚 **Sources:**\n" + "\n".join([
            f"- *{doc.metadata.get('source', 'GHA Spec')}* — _\"{doc.page_content[:250].strip()}...\"_" 
            for doc in docs
        ])
        return answer, source_info, docs
    except Exception as e:
        st.error(f"Error processing your question: {e}")
        return "Sorry, I encountered an error processing your question.", "", []

# --- Input Box & Voice Button ---
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("💬 Ask your question:")
with col2:
    if st.button("🎤 Voice Input"):
        spoken = recognize_voice()
        if spoken:
            st.session_state.last_voice_input = spoken
            user_input = spoken
            st.success(f"You said: {spoken}")

# --- Handle Input ---
if user_input:
    with st.spinner("Thinking..."):
        response, sources, docs = ask_specbot(user_input)
        if response:  # Only proceed if we got a valid response
            st.success(response)
            if highlight_sources and sources:
                st.markdown(sources)
            speak_response(response)

            st.session_state.chat.append({
                "question": user_input, 
                "answer": response, 
                "sources": sources
            })

            q_lower = user_input.lower()
            if any(k in q_lower for k in ["cement", "bitumen", "aggregate", "binder"]):
                st.session_state.qa_stats["Materials"] += 1
            elif any(k in q_lower for k in ["width", "depth", "height", "thickness"]):
                st.session_state.qa_stats["Dimensions"] += 1
            elif any(k in q_lower for k in ["test", "strength", "slump", "cb", "lab"]):
                st.session_state.qa_stats["Tests"] += 1
            elif any(k in q_lower for k in ["lay", "compact", "cure", "construct"]):
                st.session_state.qa_stats["Execution"] += 1
            else:
                st.session_state.qa_stats["General"] += 1
            st.session_state.qa_stats["Total"] += 1

# --- Chat Log Display ---
if st.session_state.chat:
    st.subheader("📿 Chat Log")
    for i, turn in enumerate(st.session_state.chat):
        st.markdown(f"**Q{i+1}:** {turn['question']}")
        st.markdown(f"**A{i+1}:** {turn['answer']}")
        if highlight_sources and turn['sources']:
            st.markdown(f"{turn['sources']}")

# --- Export to PDF ---
def export_to_pdf(chat):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="GHA SpecBot Chat Log", ln=True, align='C')
        pdf.ln(10)
        for i, turn in enumerate(chat):
            pdf.multi_cell(0, 10, f"Q{i+1}: {turn['question']}")
            pdf.multi_cell(0, 10, f"A{i+1}: {turn['answer']}")
            if turn['sources']:
                pdf.multi_cell(0, 10, f"{turn['sources']}")
            pdf.ln(5)
        filename = f"specbot_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        pdf.output(temp_path)
        return temp_path
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

if st.button("🗕️ Export Chat to PDF"):
    if st.session_state.chat:
        pdf_file = export_to_pdf(st.session_state.chat)
        if pdf_file:
            with open(pdf_file, "rb") as f:
                st.download_button(
                    label="📄 Download Chat Log", 
                    data=f, 
                    file_name="GHA_SpecBot_Chat.pdf"
                )
    else:
        st.warning("No chat history to export")

# --- Usage Dashboard ---
st.markdown("---")
st.subheader("📊 Usage Dashboard")
qa_data = pd.DataFrame({
    "Category": list(st.session_state.qa_stats.keys())[1:],
    "Questions": list(st.session_state.qa_stats.values())[1:]
})
st.markdown(f"🧲 **Total Questions Asked:** {st.session_state.qa_stats['Total']}")
if not qa_data.empty:
    chart = alt.Chart(qa_data).mark_bar().encode(
        x=alt.X("Category", sort="-y"),
        y="Questions",
        color="Category"
    ).properties(
        width=600,
        height=300
    )
    st.altair_chart(chart, use_container_width=True)
