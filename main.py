import streamlit as st
import PyPDF2
import docx2txt
import io
import os
from groq import Groq
import time
from typing import List, Dict
import hashlib

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip loading .env file
    pass

# Initialize Groq client with environment variable
@st.cache_resource
def get_groq_client():
    # Try to get API key from environment variables first, then from Streamlit secrets
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables or Streamlit secrets!")
        st.info("Please set the GROQ_API_KEY environment variable or add it to your Streamlit secrets.")
        st.stop()
    
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {str(e)}")
        st.stop()

# Document processing functions
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(uploaded_file):
    """Extract text from uploaded DOCX file"""
    try:
        # Create a temporary file
        temp_file = f"temp_{hashlib.md5(uploaded_file.name.encode()).hexdigest()}.docx"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        text = docx2txt.process(temp_file)
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_txt(uploaded_file):
    """Extract text from uploaded TXT file"""
    try:
        # Convert bytes to string
        text = str(uploaded_file.read(), "utf-8")
        return text
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return ""

def process_documents(uploaded_files):
    """Process multiple uploaded documents"""
    all_texts = []
    file_info = []
    
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            text = extract_text_from_docx(uploaded_file)
        elif file_extension == 'txt':
            text = extract_text_from_txt(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue
            
        if text.strip():
            all_texts.append(text)
            file_info.append({
                'name': uploaded_file.name,
                'size': uploaded_file.size,
                'type': file_extension,
                'word_count': len(text.split())
            })
    
    return all_texts, file_info

def chunk_text(text, chunk_size=3000, overlap=200):
    """Split text into overlapping chunks for better context"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

@st.cache_data
def get_answer_from_groq(text_chunks, question, _groq_client):
    """Get answer using Groq API with context from document chunks"""
    try:
        # Combine relevant chunks (first few for context)
        context = "\n".join(text_chunks[:3])  # Use first 3 chunks for context
        
        prompt = f"""Based on the following document content, please answer the question accurately and concisely.

Document Content:
{context}

Question: {question}

Please provide a detailed answer based only on the information provided in the document. If the answer cannot be found in the document, please state that clearly."""

        response = _groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",  # or "llama2-70b-4096"
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided documents. Be accurate and cite specific information when possible."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def main():
    # Page configuration
    st.set_page_config(
        page_title="Professional Document Chat Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/document-chat',
            'Report a bug': 'https://github.com/yourusername/document-chat/issues',
            'About': 'Professional Document Chat Assistant powered by Groq AI'
        }
    )

    # Custom CSS for professional styling
    st.markdown("""
    <style>
        .stApp {
            background: url("https://images.unsplash.com/photo-1618820830674-35aa0e70dbfc?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxfDB8MXxyYW5kb218MHx8fHx8fHx8MTcwOTA0NDk4MA&ixlib=rb-4.0.3&q=80&utm_campaign=api-credit&utm_medium=referral&utm_source=unsplash_source&w=1080");
            background-size: cover;
            background-attachment: fixed;
        }
        
        .main-header {
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .chat-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .file-info {
            background: rgba(245, 245, 245, 0.9);
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        
        .answer-box {
            background: rgba(240, 248, 255, 0.95);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
            margin: 1rem 0;
        }
        
        .stats-container {
            background: rgba(255, 255, 255, 0.9);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Professional Document Chat Assistant</h1>
        <p>Upload your documents and ask questions - powered by advanced AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Groq client
    groq_client = get_groq_client()

    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose your documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files to chat with"
        )
        
        st.markdown("---")
        
        # API Key Status
        st.header("🔑 API Status")
        if os.getenv("GROQ_API_KEY"):
            st.success("✅ API Key loaded from environment")
        elif st.secrets.get("GROQ_API_KEY"):
            st.success("✅ API Key loaded from secrets")
        else:
            st.error("❌ No API Key found")
        
        st.markdown("---")
        
        # Features
        st.header("✨ Features")
        st.markdown("""
        - 📄 **Multi-format Support**: PDF, DOCX, TXT
        - 🤖 **AI-Powered**: Advanced language model
        - 🔍 **Smart Search**: Context-aware answers
        - 📊 **Document Analytics**: File statistics
        - 🎨 **Professional UI**: Modern interface
        - ⚡ **Fast Processing**: Optimized performance
        """)
        
        st.markdown("---")
        
        # Setup Instructions
        st.header("⚙️ Setup")
        with st.expander("Environment Variable Setup"):
            st.markdown("""
            **Option 1: Environment Variables**
            ```bash
            export GROQ_API_KEY="your_api_key_here"
            ```
            
            **Option 2: .env File**
            Create a `.env` file in your project root:
            ```
            GROQ_API_KEY=your_api_key_here
            ```
            Then install python-dotenv:
            ```bash
            pip install python-dotenv
            ```
            
            **Option 3: Streamlit Cloud Secrets**
            Add `GROQ_API_KEY` to your app's secrets in the Streamlit Cloud dashboard.
            
            **Requirements.txt should include:**
            ```
            streamlit
            groq
            PyPDF2
            docx2txt
            python-dotenv  # Optional for .env file support
            ```
            """)
        
        # Dependencies Info
        with st.expander("📦 Dependencies"):
            st.markdown("""
            **Required packages:**
            - `streamlit`
            - `groq`
            - `PyPDF2`
            - `docx2txt`
            
            **Optional:**
            - `python-dotenv` (for .env file support)
            
            **Install command:**
            ```bash
            pip install streamlit groq PyPDF2 docx2txt python-dotenv
            ```
            """)
        
        st.markdown("---")
        
        # About
        st.header("👨‍💻 About")
        st.markdown("""
        **Developed by:** Maria Nadeem  
        **GitHub:** [marianadeem755](https://github.com/marianadeem755)  
        **LinkedIn:** [Maria Nadeem](https://www.linkedin.com/in/maria-nadeem-4994122aa/)  
        **Email:** [marianadeem755@gmail.com](mailto:marianadeem755@gmail.com)
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not uploaded_files:
            st.markdown("""
            <div class="chat-container">
                <h3>🚀 Get Started</h3>
                <ol>
                    <li>Upload your documents using the sidebar</li>
                    <li>Wait for processing to complete</li>
                    <li>Ask questions about your documents</li>
                    <li>Get intelligent, context-aware answers</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Process documents
            with st.spinner("🔄 Processing documents..."):
                all_texts, file_info = process_documents(uploaded_files)
            
            if all_texts:
                st.success(f"✅ Successfully processed {len(file_info)} documents")
                
                # Combine all texts and create chunks
                combined_text = "\n\n".join(all_texts)
                text_chunks = chunk_text(combined_text)
                
                # Chat interface
                st.markdown("""
                <div class="chat-container">
                    <h3>💬 Ask Questions About Your Documents</h3>
                </div>
                """, unsafe_allow_html=True)
                
                question = st.text_input(
                    "Enter your question:",
                    placeholder="What is the main topic of these documents?",
                    help="Ask any question about the content of your uploaded documents"
                )
                
                col_ask, col_clear = st.columns([1, 4])
                with col_ask:
                    ask_button = st.button("🔍 Ask Question", type="primary")
                
                if question and ask_button:
                    with st.spinner("🤔 Thinking..."):
                        answer = get_answer_from_groq(text_chunks, question, groq_client)
                    
                    st.markdown(f"""
                    <div class="answer-box">
                        <h4>💡 Answer:</h4>
                        <p>{answer}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Copy button
                    if st.button("📋 Copy Answer"):
                        st.code(answer)
                        st.success("Answer ready to copy!")
                
                # Document preview
                with st.expander("📖 Document Preview"):
                    preview_text = combined_text[:1000] + "..." if len(combined_text) > 1000 else combined_text
                    st.text_area("Content Preview:", preview_text, height=200)
            
            else:
                st.error("❌ No text could be extracted from the uploaded files.")
    
    with col2:
        if uploaded_files and file_info:
            st.markdown("""
            <div class="stats-container">
                <h3>📊 Document Stats</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # File statistics
            for info in file_info:
                st.markdown(f"""
                <div class="file-info">
                    <strong>📄 {info['name']}</strong><br>
                    <small>Type: {info['type'].upper()}</small><br>
                    <small>Size: {info['size']:,} bytes</small><br>
                    <small>Words: {info['word_count']:,}</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Overall stats
            total_words = sum(info['word_count'] for info in file_info)
            total_size = sum(info['size'] for info in file_info)
            
            st.markdown(f"""
            <div class="stats-container">
                <h4>📈 Total Statistics</h4>
                <p><strong>Files:</strong> {len(file_info)}</p>
                <p><strong>Total Words:</strong> {total_words:,}</p>
                <p><strong>Total Size:</strong> {total_size:,} bytes</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
