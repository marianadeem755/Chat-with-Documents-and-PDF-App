import streamlit as st
import os
import PyPDF2
import docx2txt
import pandas as pd
from pathlib import Path
import tempfile
from datetime import datetime
import re
import json
from typing import List, Dict, Optional
import hashlib

# Removed OpenAI dependency - using local processing instead

class DocumentProcessor:
    """Enhanced document processing with better text extraction and chunking"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF with better error handling"""
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text_parts = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(f"\n--- Page {page_num} ---\n{page_text}")
                    except Exception as e:
                        st.warning(f"Error reading page {page_num}: {str(e)}")
                        continue
                
                return "\n".join(text_parts)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_docx(docx_path: str) -> str:
        """Extract text from DOCX with better formatting"""
        try:
            text = docx2txt.process(docx_path)
            return text if text else "No readable content found in document."
        except Exception as e:
            st.error(f"Error processing DOCX: {str(e)}")
            return ""

    @staticmethod
    def extract_text_from_txt(txt_path: str) -> str:
        """Extract text from TXT files with encoding detection"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(txt_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, read as binary and decode with errors='ignore'
            with open(txt_path, 'rb') as file:
                return file.read().decode('utf-8', errors='ignore')
        except Exception as e:
            st.error(f"Error processing TXT: {str(e)}")
            return ""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better context preservation"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

class SimpleRAG:
    """Simple RAG implementation without external APIs"""
    
    def __init__(self):
        self.documents = {}
        self.processed_chunks = {}
    
    def add_document(self, doc_name: str, content: str):
        """Add a document to the knowledge base"""
        self.documents[doc_name] = content
        self.processed_chunks[doc_name] = DocumentProcessor.chunk_text(content)
    
    def simple_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Simple keyword-based search through documents"""
        query_words = set(query.lower().split())
        results = []
        
        for doc_name, chunks in self.processed_chunks.items():
            for i, chunk in enumerate(chunks):
                chunk_words = set(chunk.lower().split())
                # Calculate simple overlap score
                overlap = len(query_words.intersection(chunk_words))
                if overlap > 0:
                    results.append({
                        'document': doc_name,
                        'chunk_id': i,
                        'content': chunk,
                        'score': overlap / len(query_words)
                    })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer based on retrieved context (simple rule-based approach)"""
        if not context_chunks:
            return "I couldn't find relevant information in the documents to answer your question."
        
        # Combine context
        context = "\n\n".join([chunk['content'] for chunk in context_chunks])
        
        # Simple answer generation (in a real RAG, this would use LLM)
        answer_parts = [
            f"Based on the documents, here's what I found:\n",
            f"Context from {len(context_chunks)} relevant sections:\n"
        ]
        
        for i, chunk in enumerate(context_chunks, 1):
            preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            answer_parts.append(f"\n{i}. From {chunk['document']}:\n{preview}")
        
        answer_parts.append(f"\n\nNote: This is a simple keyword-based search. For better results, consider using an AI model.")
        
        return "\n".join(answer_parts)

def get_supported_file_types():
    """Return supported file types"""
    return {
        'PDF': ['.pdf'],
        'Word Document': ['.docx', '.doc'],
        'Text File': ['.txt', '.md']
    }

def scan_directory(directory_path: str) -> Dict[str, List[str]]:
    """Scan directory for supported documents"""
    supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md'}
    found_files = {'pdf': [], 'docx': [], 'txt': []}
    
    try:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                
                if file_ext == '.pdf':
                    found_files['pdf'].append(file_path)
                elif file_ext in ['.docx', '.doc']:
                    found_files['docx'].append(file_path)
                elif file_ext in ['.txt', '.md']:
                    found_files['txt'].append(file_path)
    
    except Exception as e:
        st.error(f"Error scanning directory: {str(e)}")
    
    return found_files

def create_download_link(content: str, filename: str):
    """Create a download link for content"""
    import base64
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'

def main():
    # Page configuration
    st.set_page_config(
        page_title="Professional Document RAG Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for professional styling
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #2E86AB;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 0.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .sub-header {
            text-align: center;
            color: #A23B72;
            font-size: 1.2em;
            margin-bottom: 2em;
        }
        .stApp {
            background: url("https://images.unsplash.com/photo-1618820830674-35aa0e70dbfc?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxfDB8MXxyYW5kb218MHx8fHx8fHx8MTcwOTA0NDk4MA&ixlib=rb-4.0.3&q=80&utm_campaign=api-credit&utm_medium=referral&utm_source=unsplash_source&w=1080");
            background-size: cover;
            background-attachment: fixed;
        }
        .content-box {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 5px;
        }
        .document-preview {
            max-height: 300px;
            overflow-y: auto;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #2E86AB;
        }
        .answer-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🤖 Professional Document RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Document Analysis & Question Answering System</p>', unsafe_allow_html=True)

    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = SimpleRAG()
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = set()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Document Management")
        
        # Method selection
        input_method = st.radio(
            "Choose input method:",
            ["📂 Browse Directory", "📎 Upload Files"],
            key="input_method"
        )
        
        st.markdown("### 📊 Statistics")
        total_docs = len(st.session_state.processed_docs)
        st.metric("Documents Processed", total_docs)
        st.metric("Chat History", len(st.session_state.chat_history))
        
        # Supported formats
        st.markdown("### 📋 Supported Formats")
        formats = get_supported_file_types()
        for format_name, extensions in formats.items():
            st.write(f"• {format_name}: {', '.join(extensions)}")
        
        # Clear data button
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.rag_system = SimpleRAG()
            st.session_state.processed_docs = set()
            st.session_state.chat_history = []
            st.success("All data cleared!")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("### 📂 Document Input")
        
        if input_method == "📂 Browse Directory":
            directory_path = st.text_input(
                "📁 Enter directory path:",
                placeholder="e.g., C:/Documents or /home/user/documents",
                help="Enter the full path to the directory containing your documents"
            )
            
            if directory_path and os.path.exists(directory_path):
                found_files = scan_directory(directory_path)
                total_files = sum(len(files) for files in found_files.values())
                
                if total_files > 0:
                    st.success(f"Found {total_files} supported files!")
                    
                    # Display file counts
                    col_pdf, col_docx, col_txt = st.columns(3)
                    with col_pdf:
                        st.metric("PDFs", len(found_files['pdf']))
                    with col_docx:
                        st.metric("Word Docs", len(found_files['docx']))
                    with col_txt:
                        st.metric("Text Files", len(found_files['txt']))
                    
                    # File selection
                    all_files = []
                    for file_type, file_list in found_files.items():
                        all_files.extend([(f, file_type) for f in file_list])
                    
                    if all_files:
                        selected_files = st.multiselect(
                            "Select files to process:",
                            options=[f[0] for f in all_files],
                            format_func=lambda x: os.path.basename(x),
                            default=[f[0] for f in all_files[:3]]  # Select first 3 by default
                        )
                        
                        if st.button("🔄 Process Selected Files", type="primary"):
                            process_files(selected_files)
                else:
                    st.warning("No supported files found in the specified directory.")
            elif directory_path:
                st.error("Directory does not exist. Please check the path.")
        
        else:  # Upload files
            uploaded_files = st.file_uploader(
                "📎 Upload documents:",
                type=['pdf', 'docx', 'doc', 'txt', 'md'],
                accept_multiple_files=True,
                help="Upload PDF, Word, or Text documents"
            )
            
            if uploaded_files:
                st.success(f"Uploaded {len(uploaded_files)} files!")
                
                if st.button("🔄 Process Uploaded Files", type="primary"):
                    process_uploaded_files(uploaded_files)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("### 💬 Document Chat")
        
        if st.session_state.processed_docs:
            # Display processed documents
            with st.expander("📚 Processed Documents", expanded=False):
                for doc_name in st.session_state.processed_docs:
                    st.write(f"• {doc_name}")
            
            # Question input
            question = st.text_input(
                "❓ Ask a question about your documents:",
                placeholder="e.g., What is the main topic discussed in these documents?",
                key="question_input"
            )
            
            # Search parameters
            with st.expander("🔧 Search Settings", expanded=False):
                top_k = st.slider("Number of relevant chunks to retrieve:", 1, 10, 3)
                show_sources = st.checkbox("Show source information", value=True)
            
            if question:
                with st.spinner("🔍 Searching documents..."):
                    # Perform RAG search
                    relevant_chunks = st.session_state.rag_system.simple_search(question, top_k=top_k)
                    answer = st.session_state.rag_system.generate_answer(question, relevant_chunks)
                
                # Display answer
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown("### 🎯 Answer")
                st.write(answer)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show sources if enabled
                if show_sources and relevant_chunks:
                    with st.expander("📖 Source Information", expanded=True):
                        for i, chunk in enumerate(relevant_chunks, 1):
                            st.write(f"**Source {i}:** {chunk['document']}")
                            st.write(f"**Relevance Score:** {chunk['score']:.2f}")
                            st.write(f"**Content Preview:** {chunk['content'][:200]}...")
                            st.write("---")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'question': question,
                    'answer': answer,
                    'sources': len(relevant_chunks)
                })
                
                # Download answer
                if st.button("💾 Save Answer"):
                    filename = f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    content = f"Question: {question}\n\nAnswer: {answer}\n\nGenerated on: {datetime.now()}"
                    st.download_button(
                        label="Download Answer",
                        data=content,
                        file_name=filename,
                        mime="text/plain"
                    )
        
        else:
            st.info("👆 Please process some documents first to start chatting!")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat History Section
    if st.session_state.chat_history:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("### 📜 Chat History")
        
        # Display recent chats
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            with st.expander(f"Chat {len(st.session_state.chat_history) - i + 1}: {chat['question'][:50]}...", expanded=False):
                st.write(f"**Time:** {chat['timestamp']}")
                st.write(f"**Question:** {chat['question']}")
                st.write(f"**Answer:** {chat['answer'][:200]}...")
                st.write(f"**Sources Used:** {chat['sources']}")
        
        # Export chat history
        if st.button("📤 Export Chat History"):
            chat_data = []
            for chat in st.session_state.chat_history:
                chat_data.append({
                    'Timestamp': chat['timestamp'],
                    'Question': chat['question'],
                    'Answer': chat['answer'],
                    'Sources': chat['sources']
                })
            
            df = pd.DataFrame(chat_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Chat History (CSV)",
                data=csv,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👩‍💻 Developer")
        st.markdown("**Maria Nadeem**")
        st.markdown("[GitHub](https://github.com/marianadeem755) | [LinkedIn](https://www.linkedin.com/in/maria-nadeem-4994122aa/)")
    
    with col2:
        st.markdown("### 🚀 Features")
        st.markdown("• Multi-format support")
        st.markdown("• Smart text chunking")
        st.markdown("• Chat history")
        st.markdown("• Export capabilities")
    
    with col3:
        st.markdown("### 📧 Contact")
        st.markdown("[marianadeem755@gmail.com](mailto:marianadeem755@gmail.com)")
        st.markdown("Built with ❤️ using Streamlit")

def process_files(file_paths):
    """Process selected files from directory"""
    processor = DocumentProcessor()
    progress_bar = st.progress(0)
    
    for i, file_path in enumerate(file_paths):
        try:
            file_name = os.path.basename(file_path)
            
            if file_path.lower().endswith('.pdf'):
                content = processor.extract_text_from_pdf(file_path)
            elif file_path.lower().endswith(('.docx', '.doc')):
                content = processor.extract_text_from_docx(file_path)
            elif file_path.lower().endswith(('.txt', '.md')):
                content = processor.extract_text_from_txt(file_path)
            else:
                continue
            
            if content.strip():
                st.session_state.rag_system.add_document(file_name, content)
                st.session_state.processed_docs.add(file_name)
                st.success(f"✅ Processed: {file_name}")
            else:
                st.warning(f"⚠️ No content extracted from: {file_name}")
        
        except Exception as e:
            st.error(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(file_paths))

def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    processor = DocumentProcessor()
    progress_bar = st.progress(0)
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Process based on file type
            if uploaded_file.name.lower().endswith('.pdf'):
                content = processor.extract_text_from_pdf(tmp_path)
            elif uploaded_file.name.lower().endswith(('.docx', '.doc')):
                content = processor.extract_text_from_docx(tmp_path)
            elif uploaded_file.name.lower().endswith(('.txt', '.md')):
                content = processor.extract_text_from_txt(tmp_path)
            else:
                continue
            
            if content.strip():
                st.session_state.rag_system.add_document(uploaded_file.name, content)
                st.session_state.processed_docs.add(uploaded_file.name)
                st.success(f"✅ Processed: {uploaded_file.name}")
            else:
                st.warning(f"⚠️ No content extracted from: {uploaded_file.name}")
            
            # Clean up temporary file
            os.unlink(tmp_path)
        
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))

if __name__ == "__main__":
    main()
