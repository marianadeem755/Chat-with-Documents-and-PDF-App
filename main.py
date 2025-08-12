import streamlit as st
import os
import json
import re
from typing import List, Dict, Optional, Tuple
import hashlib
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import zipfile

# Document processing imports
import PyPDF2
import docx2txt
import openpyxl
import csv
from io import StringIO, BytesIO

# Text processing and embeddings
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class DocumentProcessor:
    """Advanced document processing class with multiple format support"""
    
    SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt', '.csv', '.xlsx', '.json', '.md'}
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF with error handling"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                return "\n".join(text)
        except Exception as e:
            st.error(f"Error reading PDF {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX with error handling"""
        try:
            return docx2txt.process(file_path)
        except Exception as e:
            st.error(f"Error reading DOCX {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            st.error(f"Error reading TXT {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_csv(file_path: str) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df.to_string(index=False)
        except Exception as e:
            st.error(f"Error reading CSV {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_xlsx(file_path: str) -> str:
        """Extract text from Excel file"""
        try:
            df = pd.read_excel(file_path)
            return df.to_string(index=False)
        except Exception as e:
            st.error(f"Error reading Excel {file_path}: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_json(file_path: str) -> str:
        """Extract text from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return json.dumps(data, indent=2)
        except Exception as e:
            st.error(f"Error reading JSON {file_path}: {str(e)}")
            return ""
    
    @classmethod
    def extract_text(cls, file_path: str) -> Tuple[str, Dict]:
        """Main text extraction method with metadata"""
        file_ext = Path(file_path).suffix.lower()
        file_size = os.path.getsize(file_path)
        
        metadata = {
            'filename': os.path.basename(file_path),
            'file_type': file_ext,
            'file_size': file_size,
            'processed_at': datetime.now().isoformat()
        }
        
        if file_ext == '.pdf':
            text = cls.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            text = cls.extract_text_from_docx(file_path)
        elif file_ext == '.txt' or file_ext == '.md':
            text = cls.extract_text_from_txt(file_path)
        elif file_ext == '.csv':
            text = cls.extract_text_from_csv(file_path)
        elif file_ext == '.xlsx':
            text = cls.extract_text_from_xlsx(file_path)
        elif file_ext == '.json':
            text = cls.extract_text_from_json(file_path)
        else:
            st.warning(f"Unsupported file format: {file_ext}")
            return "", metadata
        
        metadata['word_count'] = len(text.split()) if text else 0
        metadata['char_count'] = len(text) if text else 0
        
        return text, metadata

class RAGSystem:
    """Advanced RAG system with TF-IDF based retrieval"""
    
    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.is_fitted = False
    
    def add_document(self, text: str, metadata: Dict) -> None:
        """Add document to the RAG system"""
        self.documents.append({
            'text': text,
            'metadata': metadata,
            'id': len(self.documents)
        })
        self.is_fitted = False
    
    def fit(self) -> None:
        """Fit the TF-IDF vectorizer on all documents"""
        if not self.documents:
            return
        
        texts = [doc['text'] for doc in self.documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        if not self.is_fitted:
            self.fit()
        
        if not self.documents:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include documents with some similarity
                results.append({
                    'document': self.documents[idx],
                    'similarity': similarities[idx]
                })
        
        return results
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer based on context documents (rule-based approach)"""
        if not context_docs:
            return "I couldn't find relevant information in the documents to answer your question."
        
        # Combine context from top documents
        context = "\n\n".join([doc['document']['text'][:1000] for doc in context_docs])
        
        # Simple keyword-based answer generation
        query_words = set(word.lower() for word in word_tokenize(query) 
                         if word.lower() not in stopwords.words('english'))
        
        sentences = sent_tokenize(context)
        scored_sentences = []
        
        for sentence in sentences:
            sentence_words = set(word.lower() for word in word_tokenize(sentence))
            score = len(query_words.intersection(sentence_words))
            if score > 0:
                scored_sentences.append((sentence, score))
        
        # Sort by score and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if scored_sentences:
            answer_sentences = [sent[0] for sent in scored_sentences[:3]]
            return " ".join(answer_sentences)
        else:
            return "Based on the documents, I found relevant content but couldn't extract a specific answer to your question."

def get_file_icon(file_ext: str) -> str:
    """Get emoji icon for file type"""
    icons = {
        '.pdf': '📄',
        '.docx': '📝',
        '.txt': '📄',
        '.csv': '📊',
        '.xlsx': '📈',
        '.json': '🔧',
        '.md': '📋'
    }
    return icons.get(file_ext, '📄')

def create_dashboard(rag_system: RAGSystem) -> None:
    """Create analytics dashboard"""
    if not rag_system.documents:
        st.info("No documents loaded yet.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(rag_system.documents))
    
    with col2:
        total_words = sum(doc['metadata']['word_count'] for doc in rag_system.documents)
        st.metric("Total Words", f"{total_words:,}")
    
    with col3:
        file_types = [doc['metadata']['file_type'] for doc in rag_system.documents]
        unique_types = len(set(file_types))
        st.metric("File Types", unique_types)
    
    with col4:
        total_size = sum(doc['metadata']['file_size'] for doc in rag_system.documents)
        st.metric("Total Size", f"{total_size/1024/1024:.1f} MB")
    
    # File type distribution
    st.subheader("📊 Document Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file_type_counts = pd.Series(file_types).value_counts()
        fig = px.pie(values=file_type_counts.values, names=file_type_counts.index,
                     title="Document Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        word_counts = [doc['metadata']['word_count'] for doc in rag_system.documents]
        filenames = [doc['metadata']['filename'] for doc in rag_system.documents]
        
        fig = px.bar(x=filenames, y=word_counts, title="Word Count by Document")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Professional Document RAG System",
        page_icon="🚀",
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
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .document-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin: 0.5rem 0;
        }
        .answer-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .sidebar .sidebar-content {
            background: rgba(255, 255, 255, 0.95);
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.9);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Professional Document RAG System</h1>
        <p>Advanced Retrieval-Augmented Generation for Document Analysis</p>
        <p><em>Created by Maria Nadeem</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Document Management")
        
        # File upload option
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files", 
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'json', 'md']
        )
        
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.processed_files:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Process the file
                        text, metadata = DocumentProcessor.extract_text(tmp_file_path)
                        if text:
                            st.session_state.rag_system.add_document(text, metadata)
                            st.session_state.processed_files.append(uploaded_file.name)
                            st.success(f"✅ {uploaded_file.name}")
                        
                        # Clean up temp file
                        os.unlink(tmp_file_path)
        
        st.divider()
        
        # Folder path option
        st.subheader("📁 Local Folder Path")
        folder_path = st.text_input("Enter folder path:", placeholder="C:/Documents/MyFiles")
        
        if folder_path and st.button("🔍 Scan Folder"):
            if os.path.exists(folder_path):
                with st.spinner("Scanning folder..."):
                    files_found = []
                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            file_ext = Path(file_path).suffix.lower()
                            if file_ext in DocumentProcessor.SUPPORTED_FORMATS:
                                files_found.append(file_path)
                    
                    for file_path in files_found:
                        if os.path.basename(file_path) not in st.session_state.processed_files:
                            text, metadata = DocumentProcessor.extract_text(file_path)
                            if text:
                                st.session_state.rag_system.add_document(text, metadata)
                                st.session_state.processed_files.append(os.path.basename(file_path))
                    
                    st.success(f"✅ Processed {len(files_found)} files from folder")
            else:
                st.error("❌ Folder path does not exist")
        
        st.divider()
        
        # Clear documents
        if st.button("🗑️ Clear All Documents"):
            st.session_state.rag_system = RAGSystem()
            st.session_state.processed_files = []
            st.session_state.chat_history = []
            st.success("✅ All documents cleared")
        
        st.divider()
        
        # Author info
        st.markdown("### 👩‍💻 Author: Maria Nadeem")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/marianadeem755)")
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/maria-nadeem-4994122aa/)")
        st.markdown("📧 marianadeem755@gmail.com")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Dashboard", "📚 Documents", "ℹ️ About"])
    
    with tab1:
        st.header("💬 Chat with Your Documents")
        
        if st.session_state.rag_system.documents:
            # Chat interface
            question = st.text_input("🤔 Ask a question about your documents:", 
                                    placeholder="What is the main topic discussed in these documents?")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                search_docs = st.button("🔍 Search & Answer", type="primary")
            with col2:
                if st.button("📋 Copy Last Answer"):
                    if st.session_state.chat_history:
                        last_answer = st.session_state.chat_history[-1]['answer']
                        st.code(last_answer)
                        st.success("Answer copied to display!")
            
            if question and search_docs:
                with st.spinner("🔍 Searching through documents..."):
                    # Search for relevant documents
                    results = st.session_state.rag_system.search(question, top_k=3)
                    
                    if results:
                        # Generate answer
                        answer = st.session_state.rag_system.generate_answer(question, results)
                        
                        # Display answer
                        st.markdown(f"""
                        <div class="answer-box">
                            <h4>🎯 Answer:</h4>
                            <p>{answer}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show relevant sources
                        st.subheader("📚 Relevant Sources:")
                        for i, result in enumerate(results):
                            with st.expander(f"📄 {result['document']['metadata']['filename']} (Similarity: {result['similarity']:.2f})"):
                                st.write(result['document']['text'][:500] + "...")
                        
                        # Save to chat history
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': answer,
                            'timestamp': datetime.now().isoformat(),
                            'sources': [r['document']['metadata']['filename'] for r in results]
                        })
                    else:
                        st.warning("❌ No relevant documents found for your question.")
            
            # Chat history
            if st.session_state.chat_history:
                st.subheader("💭 Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                    with st.expander(f"Q: {chat['question'][:50]}..."):
                        st.write(f"**Question:** {chat['question']}")
                        st.write(f"**Answer:** {chat['answer']}")
                        st.write(f"**Sources:** {', '.join(chat['sources'])}")
                        st.write(f"**Time:** {chat['timestamp']}")
        else:
            st.info("📁 Please upload documents or specify a folder path to start chatting!")
    
    with tab2:
        st.header("📊 Document Analytics Dashboard")
        create_dashboard(st.session_state.rag_system)
    
    with tab3:
        st.header("📚 Loaded Documents")
        
        if st.session_state.rag_system.documents:
            for doc in st.session_state.rag_system.documents:
                metadata = doc['metadata']
                file_icon = get_file_icon(metadata['file_type'])
                
                st.markdown(f"""
                <div class="document-card">
                    <h4>{file_icon} {metadata['filename']}</h4>
                    <p><strong>Type:</strong> {metadata['file_type']} | 
                       <strong>Size:</strong> {metadata['file_size']/1024:.1f} KB | 
                       <strong>Words:</strong> {metadata['word_count']:,}</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("Preview Content"):
                    st.text(doc['text'][:1000] + "..." if len(doc['text']) > 1000 else doc['text'])
        else:
            st.info("No documents loaded yet.")
    
    with tab4:
        st.header("ℹ️ About This RAG System")
        
        st.markdown("""
        ### 🎯 What is RAG (Retrieval-Augmented Generation)?
        
        This application implements a **Retrieval-Augmented Generation (RAG)** system that:
        
        1. **📤 Ingests** multiple document formats (PDF, DOCX, TXT, CSV, Excel, JSON, Markdown)
        2. **🔍 Indexes** content using TF-IDF vectorization
        3. **🎯 Retrieves** relevant document sections based on user queries
        4. **💡 Generates** contextual answers using the retrieved information
        
        ### 🚀 Key Features:
        
        - **Multi-format Support**: Process 7+ document types
        - **Advanced Search**: TF-IDF based semantic search
        - **Real-time Analytics**: Document statistics and visualizations
        - **Chat History**: Track your questions and answers
        - **Professional UI**: Modern, responsive interface
        - **No API Keys**: Completely local processing
        
        ### 🛠️ Technical Implementation:
        
        - **Text Processing**: NLTK for tokenization and preprocessing
        - **Vectorization**: Scikit-learn TF-IDF for document similarity
        - **Analytics**: Plotly for interactive visualizations
        - **UI Framework**: Streamlit for web interface
        
        ### 💡 How to Use:
        
        1. Upload documents or specify a folder path
        2. Ask questions about your documents
        3. View analytics and document insights
        4. Export or copy answers as needed
        
        This system works entirely **offline** and doesn't require any external APIs!
        """)

if __name__ == "__main__":
    main()
