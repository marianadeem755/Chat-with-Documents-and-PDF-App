import streamlit as st
import PyPDF2
import docx2txt
import io
from typing import List, Optional
import re

# Configure page
st.set_page_config(
    page_title="Professional Document Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
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
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .content-box {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
    }
    
    .document-preview {
        max-height: 300px;
        overflow-y: auto;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }
    
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
    }
    
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.8);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file) -> str:
    """Extract text from uploaded DOCX file"""
    try:
        return docx2txt.process(docx_file)
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def process_uploaded_files(uploaded_files) -> dict:
    """Process multiple uploaded files and return a dictionary with filename and content"""
    documents = {}
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_extension = file_name.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            text = extract_text_from_docx(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {file_name}")
            continue
        
        if text:
            documents[file_name] = text
    
    return documents

def simple_qa_system(text: str, question: str) -> str:
    """Simple rule-based Q&A system without external APIs"""
    text_lower = text.lower()
    question_lower = question.lower()
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Extract key words from question
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    
    # Remove common stop words
    stop_words = {'what', 'where', 'when', 'why', 'how', 'who', 'is', 'are', 'was', 'were', 
                  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    question_words = question_words - stop_words
    
    if not question_words:
        return "Please ask a more specific question."
    
    # Score sentences based on keyword matches
    scored_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for word in question_words if word in sentence_lower)
        if score > 0:
            scored_sentences.append((score, sentence))
    
    if not scored_sentences:
        return "I couldn't find relevant information in the document to answer your question."
    
    # Sort by score and return top sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    
    # Return top 3 most relevant sentences
    answer_sentences = [sentence for score, sentence in scored_sentences[:3]]
    return " ".join(answer_sentences)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="text-align: center; color: #333; margin-bottom: 0.5rem;">
            📚 Professional Document Chat Application
        </h1>
        <p style="text-align: center; color: #666; font-size: 1.1rem;">
            Upload your documents and ask questions - No API keys required!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF or DOCX files",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        help="You can upload multiple PDF and DOCX files at once"
    )
    
    if not uploaded_files:
        st.markdown("""
        <div class="upload-section">
            <h4>👆 Please upload your documents to get started</h4>
            <p>Supported formats: PDF, DOCX</p>
            <p>You can upload multiple files at once</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sidebar info
        with st.sidebar:
            st.markdown("### 📋 How to Use")
            st.markdown("""
            1. **Upload Documents**: Click 'Browse files' and select PDF or DOCX files
            2. **Select Document**: Choose which document to analyze
            3. **Ask Questions**: Type your question about the document
            4. **Get Answers**: Receive instant responses based on document content
            """)
            
            st.markdown("### ✨ Features")
            st.markdown("""
            - 📄 Multiple file format support
            - 🔍 Intelligent text extraction
            - 💬 Interactive Q&A system
            - 📱 Professional responsive design
            - 🚀 No API keys required
            """)
            
            st.markdown("---")
            st.markdown("### 👨‍💻 Author: Maria Nadeem")
            st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/marianadeem755)")
            st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/maria-nadeem-4994122aa/)")
            st.markdown("📧 marianadeem755@gmail.com")
        
        return
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process uploaded files
    with st.spinner('Processing uploaded documents...'):
        documents = process_uploaded_files(uploaded_files)
    
    if not documents:
        st.error("No valid documents were processed. Please check your files and try again.")
        return
    
    # Display processed documents
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.markdown("### 📋 Processed Documents")
    st.success(f"Successfully processed {len(documents)} document(s)")
    
    # Document selection
    selected_doc = st.selectbox(
        "Select a document to analyze:",
        list(documents.keys()),
        help="Choose which document you want to ask questions about"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if selected_doc:
        # Document preview
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("### 👁️ Document Preview")
        
        document_text = documents[selected_doc]
        preview_text = document_text[:1000] + "..." if len(document_text) > 1000 else document_text
        
        st.markdown(f'<div class="document-preview">{preview_text}</div>', unsafe_allow_html=True)
        
        with st.expander("📊 Document Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", len(document_text))
            with col2:
                st.metric("Words", len(document_text.split()))
            with col3:
                st.metric("Lines", len(document_text.split('\n')))
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Question and Answer section
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("### 💬 Ask a Question")
        
        question = st.text_input(
            "What would you like to know about this document?",
            placeholder="e.g., What is the main topic discussed?",
            help="Ask specific questions about the document content"
        )
        
        if question:
            with st.spinner('🤔 Analyzing document and generating answer...'):
                answer = simple_qa_system(document_text, question)
            
            st.markdown(f"""
            <div class="answer-box">
                <h4>🎯 Answer:</h4>
                <p style="font-size: 1.1rem; line-height: 1.6;">{answer}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Copy to clipboard button (Note: This requires user interaction in most browsers)
            if st.button("📋 Copy Answer", help="Copy the answer to your clipboard"):
                st.code(answer, language=None)
                st.info("💡 Tip: You can select and copy the text from the code block above")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Upload your PDF or DOCX files
        2. Select a document from the dropdown
        3. Preview the document content
        4. Ask questions about the document
        5. Get instant answers!
        """)
        
        if documents:
            st.markdown("### 📄 Uploaded Documents")
            for doc_name in documents.keys():
                st.write(f"✅ {doc_name}")
        
        st.markdown("---")
        st.markdown("### 🛠️ Technical Info")
        st.info("""
        This app uses:
        - PyPDF2 for PDF processing
        - docx2txt for Word documents  
        - Rule-based Q&A system
        - No external APIs required
        """)
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Created by Maria Nadeem")
        st.markdown("[🌐 GitHub](https://github.com/marianadeem755)")
        st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/maria-nadeem-4994122aa/)")
        st.markdown("📧 marianadeem755@gmail.com")

if __name__ == "__main__":
    main()
