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
        
@st.cache_resource
def load_vectorstore():
    """Load vectorstore with better error handling and configuration"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Better similarity computation
        )
        
        vectorstore = Chroma(
            persist_directory="gha_vectorstore",
            embedding_function=embedding_model
        )
        
        # Verify vectorstore has documents
        collection_count = vectorstore._collection.count()
        if collection_count == 0:
            st.warning("⚠️ No documents found in vectorstore. Please ensure your documents are properly indexed.")
            logger.warning("Empty vectorstore detected")
        else:
            logger.info(f"Vectorstore loaded with {collection_count} documents")
            
        return vectorstore
        
    except Exception as e:
        st.error(f"Failed to load vectorstore: {str(e)}")
        logger.error(f"Vectorstore loading error: {str(e)}")
        st.stop()

# --- Initialize Components with Error Handling ---
try:
    llm_pipeline = load_llm()
    llm = HuggingFacePipeline(
        pipeline=llm_pipeline,
        model_kwargs={"temperature": 0.2, "max_length": 512}
    )
    vectorstore = load_vectorstore()
except Exception as e:
    st.error(f"Critical initialization error: {str(e)}")
    st.stop()

# --- Enhanced Streamlit UI Config ---
st.set_page_config(
    page_title="GHA SpecBot", 
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧱 GHA SpecBot Pro Max")
st.markdown("""
**Ask me anything about the Ghana Highway Authority Road & Bridge Specifications.**
*Powered by Microsoft Phi-3 and advanced RAG technology.*
""")

# --- Enhanced Session State Management ---
if "chat" not in st.session_state:
    st.session_state.chat = []
    st.session_state.qa_stats = {
        "Total": 0, 
        "Materials": 0, 
        "Dimensions": 0, 
        "Tests": 0, 
        "Execution": 0, 
        "Safety": 0,
        "General": 0
    }
    st.session_state.voice_enabled = True
    st.session_state.show_sources = True
    st.session_state.conversation_memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

# --- Enhanced Voice Processing ---
def recognize_voice():
    """Improved voice input with better error handling and user feedback"""
    try:
        # Check if running in cloud environment
        if hasattr(st, 'runtime') and hasattr(st.runtime, 'exists'):
            st.warning("🎤 Voice input may not work in cloud deployments")
            return ""
        
        r = sr.Recognizer()
        with sr.Microphone() as source:
            with st.spinner("🎤 Adjusting for background noise..."):
                r.adjust_for_ambient_noise(source, duration=1)
            
            st.info("🎤 **Listening...** Speak clearly and concisely")
            
            # Record audio with timeout
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            
            with st.spinner("🧠 Processing speech..."):
                # Try Google first, fallback to other services
                try:
                    text = r.recognize_google(audio)
                    st.success(f"🎯 Heard: *{text}*")
                    return text
                except sr.UnknownValueError:
                    try:
                        text = r.recognize_sphinx(audio)  # Offline fallback
                        st.success(f"🎯 Heard: *{text}*")
                        return text
                    except:
                        st.warning("Could not understand speech. Please try again.")
                        return ""
                        
    except sr.WaitTimeoutError:
        st.warning("⏰ Listening timed out. No speech detected.")
        return ""
    except sr.RequestError as e:
        st.error(f"Speech service error: {str(e)}")
        return ""
    except Exception as e:
        st.error(f"Voice input error: {str(e)}")
        return ""

def speak_response(text):
    """Enhanced text-to-speech with better error handling"""
    if not st.session_state.get("voice_enabled", False):
        return
        
    try:
        # Initialize TTS engine with error handling
        engine = pyttsx3.init()
        
        # Configure voice properties
        voices = engine.getProperty('voices')
        if voices:
            # Prefer female voice if available
            for voice in voices:
                if 'female' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        
        engine.setProperty('rate', 160)  # Moderate speech rate
        engine.setProperty('volume', 0.8)  # 80% volume
        
        # Clean and limit text for speech
        clean_text = text.replace("*", "").replace("_", "").replace("#", "")
        speech_text = clean_text[:300]  # Limit to prevent long speeches
        
        if len(clean_text) > 300:
            speech_text += "... and more details are shown above."
        
        engine.say(speech_text)
        engine.runAndWait()
        
    except Exception as e:
        logger.warning(f"TTS error: {str(e)}")
        # Silently fail for TTS errors to not disrupt user experience

# --- Updated QA Function with Auth ---
def ask_specbot(query):
    """Now uses the pre-authenticated token"""
    try:
             # Sanitize the input query first
        sanitized_query = query.strip()
        
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
            
       
                # Process answer
        raw_answer = result.get("result", "")
        
        # Clean up the answer
        answer = raw_answer.split("<|end|>")[0].strip()  # Remove any template artifacts
        if not answer:
            answer = "I couldn't find specific information about that in the GHA specifications."
        
        # Process sources
        sources = []
        source_docs = result.get("source_documents", [])
        
        for i, doc in enumerate(source_docs[:3]):  # Limit to top 3 sources
            if hasattr(doc, "metadata") and hasattr(doc, "page_content"):
                source_name = doc.metadata.get("source", f"GHA Document {i+1}")
                page_num = doc.metadata.get("page", "")
                page_info = f" (Page {page_num})" if page_num else ""
                
                # Clean and truncate content
                content = doc.page_content.replace("\n", " ").strip()
                content_preview = content[:200] + "..." if len(content) > 200 else content
                
                sources.append(f"📄 **{source_name}**{page_info}\n   _{content_preview}_")
        
        source_text = "\n\n📚 **Sources:**\n" + "\n\n".join(sources) if sources else ""
        
        # Update conversation memory
        st.session_state.conversation_memory.chat_memory.add_user_message(sanitized_query)
        st.session_state.conversation_memory.chat_memory.add_ai_message(answer)
        
        return answer, source_text
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return "I encountered an error while processing your question. Please try rephrasing it.", ""

# --- Enhanced UI Components ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Voice settings
    st.subheader("🔊 Voice Settings")
    st.session_state.voice_enabled = st.checkbox(
        "Enable Voice Output", 
        value=st.session_state.get("voice_enabled", True),
        help="Enable text-to-speech for answers"
    )
    
    # Display settings
    st.subheader("📱 Display Options")
    st.session_state.show_sources = st.checkbox(
        "Show Source Documents", 
        value=st.session_state.get("show_sources", True),
        help="Display source documents used for answers"
    )
    
    show_stats = st.checkbox(
        "Show Usage Statistics", 
        value=True,
        help="Display analytics dashboard"
    )
    
    # Advanced settings
    st.subheader("🔧 Advanced Settings")
    st.session_state.max_tokens = st.slider(
        "Max Answer Length", 
        min_value=128, 
        max_value=1024, 
        value=512, 
        step=64,
        help="Maximum length of generated answers"
    )
    
    st.session_state.top_k_chunks = st.slider(
        "Context Chunks", 
        min_value=2, 
        max_value=8, 
        value=4,
        help="Number of document chunks to use for context"
    )
    
    # System info
    st.subheader("ℹ️ System Info")
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    st.info(f"**Device:** {device_info}")
    
    if st.session_state.qa_stats["Total"] > 0:
        st.info(f"**Questions Asked:** {st.session_state.qa_stats['Total']}")

# --- Enhanced Main Interface ---
st.subheader("💬 Ask Your Question")

# Create input columns
col1, col2, col3 = st.columns([6, 1, 1])

with col1:
    user_input = st.text_input(
        "Enter your question about GHA specifications:",
        key="query_input",
        placeholder="e.g., What are the concrete strength requirements for bridges?"
    )

with col2:
    voice_button = st.button("🎤 Voice", help="Click to use voice input")

with col3:
    clear_button = st.button("🗑️ Clear", help="Clear conversation history")

# Handle voice input
if voice_button:
    if voice_input := recognize_voice():
        user_input = voice_input
        st.rerun()

# Handle clear button
if clear_button:
    st.session_state.chat = []
    st.session_state.conversation_memory.clear()
    st.session_state.qa_stats = {k: 0 for k in st.session_state.qa_stats}
    st.success("Conversation history cleared!")
    st.rerun()

# Process user input
if user_input:
    with st.spinner("🔍 Searching GHA specifications..."):
        answer, sources = ask_specbot(user_input)
        
        if answer:
            # Display answer
            st.success("**Answer:**")
            st.markdown(answer)
            
            # Display sources if enabled
            if st.session_state.show_sources and sources:
                with st.expander("📚 View Sources", expanded=False):
                    st.markdown(sources)
            
            # Voice output
            if st.session_state.voice_enabled:
                speak_response(answer)
            
            # Update chat history
            st.session_state.chat.append({
                "timestamp": datetime.datetime.now(),
                "question": user_input,
                "answer": answer,
                "sources": sources
            })
            
            # Update statistics
            q_lower = user_input.lower()
            category = (
                "Materials" if any(k in q_lower for k in ["cement", "concrete", "bitumen", "asphalt", "steel", "aggregate"]) else
                "Dimensions" if any(k in q_lower for k in ["width", "depth", "thickness", "diameter", "length", "size"]) else
                "Tests" if any(k in q_lower for k in ["test", "testing", "quality", "inspection", "standard"]) else
                "Execution" if any(k in q_lower for k in ["construct", "build", "install", "compact", "place", "cure"]) else
                "Safety" if any(k in q_lower for k in ["safety", "hazard", "protection", "risk", "secure"]) else
                "General"
            )
            st.session_state.qa_stats[category] += 1
            st.session_state.qa_stats["Total"] += 1

# --- Enhanced Chat History ---
if st.session_state.chat:
    st.markdown("---")
    st.subheader("📜 Conversation History")
    
    # Show recent conversations
    recent_chats = list(reversed(st.session_state.chat[-5:]))
    
    for i, turn in enumerate(recent_chats):
        timestamp = turn.get("timestamp", datetime.datetime.now()).strftime("%H:%M")
        question_preview = turn['question'][:60] + "..." if len(turn['question']) > 60 else turn['question']
        
        with st.expander(f"**{timestamp}** - {question_preview}"):
            st.markdown(f"**❓ Question:** {turn['question']}")
            st.markdown(f"**✅ Answer:** {turn['answer']}")
            if turn.get('sources') and st.session_state.show_sources:
                st.markdown(turn['sources'])

# --- Enhanced Export Functionality ---
def generate_detailed_pdf(chat_history):
    """Generate comprehensive PDF report with proper error handling"""
    if FPDF is None:
        st.error("PDF generation unavailable. Please install fpdf2.")
        return None, None
        
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        
        # Title
        pdf.cell(0, 10, "GHA SpecBot Consultation Report", 0, 1, 'C')
        pdf.ln(5)
        
        # Metadata
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 5, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
        pdf.cell(0, 5, f"Total Questions: {len(chat_history)}", 0, 1)
        pdf.ln(10)
        
        # Chat content
        pdf.set_font("Arial", '', 11)
        for i, entry in enumerate(chat_history, 1):
            # Question
            pdf.set_font("Arial", 'B', 11)
            pdf.multi_cell(0, 6, f"Q{i}: {entry['question']}")
            pdf.ln(2)
            
            # Answer
            pdf.set_font("Arial", '', 10)
            answer_text = entry['answer'][:1000]  # Limit length
            if len(entry['answer']) > 1000:
                answer_text += "... (truncated)"
            pdf.multi_cell(0, 5, f"Answer: {answer_text}")
            pdf.ln(5)
            
        # Save to temp file
        filename = f"gha_specbot_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        safe_path = os.path.join(tempfile.gettempdir(), filename)
        pdf.output(safe_path)
        
        return safe_path, filename
        
    except Exception as e:
        logger.error(f"PDF generation error: {str(e)}")
        return None, None

# Export section
if st.session_state.chat:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Export to PDF", help="Download conversation as PDF"):
            pdf_path, filename = generate_detailed_pdf(st.session_state.chat)
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=f.read(),
                        file_name=filename,
                        mime="application/pdf"
                    )
                st.success("✅ PDF generated successfully!")
    
    with col2:
        if st.button("📊 Export Statistics", help="Download usage statistics"):
            stats_df = pd.DataFrame([
                {"Category": k, "Count": v} 
                for k, v in st.session_state.qa_stats.items()
            ])
            csv = stats_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Statistics",
                data=csv,
                file_name=f"gha_specbot_stats_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# --- Enhanced Analytics Dashboard ---
if show_stats and st.session_state.qa_stats["Total"] > 0:
    st.markdown("---")
    st.subheader("📊 Usage Analytics")
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Questions", st.session_state.qa_stats["Total"])
    with col2:
        most_asked = max(
            [(k, v) for k, v in st.session_state.qa_stats.items() if k != "Total"],
            key=lambda x: x[1], default=("None", 0)
        )
        st.metric("Top Category", f"{most_asked[0]} ({most_asked[1]})")
    with col3:
        avg_per_session = st.session_state.qa_stats["Total"] / max(len(st.session_state.chat), 1)
        st.metric("Avg Q/Session", f"{avg_per_session:.1f}")
    with col4:
        if st.session_state.chat:
            session_duration = (
                st.session_state.chat[-1]["timestamp"] - st.session_state.chat[0]["timestamp"]
            ).total_seconds() / 60
            st.metric("Session (min)", f"{session_duration:.1f}")
    
    # Create chart
    chart_data = pd.DataFrame({
        "Category": [k for k in st.session_state.qa_stats.keys() if k != "Total" and st.session_state.qa_stats[k] > 0],
        "Questions": [v for k, v in st.session_state.qa_stats.items() if k != "Total" and v > 0]
    })
    
    if not chart_data.empty:
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Questions:Q", title="Number of Questions"),
            y=alt.Y("Category:N", sort="-x", title="Question Category"),
            color=alt.Color("Category:N", legend=None),
            tooltip=["Category:N", "Questions:Q"]
        ).properties(
            title="Questions by Category",
            height=300
        )
        
        st.altair_chart(chart, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    🧱 GHA SpecBot Pro Max | Powered by Microsoft Phi-3 | 
    Built for Ghana Highway Authority Specifications
</div>
""", unsafe_allow_html=True)

# --- New Auth Status Indicator ---
with st.sidebar:
    st.caption(f"🔒 Auth Status: {'✅' if HF_TOKEN else '❌'} Hugging Face")
