import streamlit as st
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import docx
from io import StringIO
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data with error handling"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        return True
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        return False

class DocumentProcessor:
    """Enhanced document processor with better error handling"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {e}")
            return ""
    
    @staticmethod
    def process_file(uploaded_file) -> str:
        """Process uploaded file and extract text"""
        text = ""
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                text = DocumentProcessor.extract_text_from_pdf(tmp_file_path)
            elif file_extension == 'docx':
                text = DocumentProcessor.extract_text_from_docx(tmp_file_path)
            elif file_extension == 'txt':
                text = DocumentProcessor.extract_text_from_txt(tmp_file_path)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
        return text

class EnhancedRAGSystem:
    """Enhanced RAG system with improved text processing and fallback tokenization"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.documents = []
        self.vectorizer = None
        self.document_vectors = None
        self.nltk_available = download_nltk_data()
        
        # Initialize stopwords
        try:
            if self.nltk_available:
                self.stop_words = set(stopwords.words('english'))
            else:
                # Fallback stopwords
                self.stop_words = {
                    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                    'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
                    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                    'with', 'through', 'during', 'before', 'after', 'above', 'below',
                    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                    'further', 'then', 'once'
                }
        except Exception as e:
            logger.error(f"Error initializing stopwords: {e}")
            self.stop_words = set()
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text with fallback method"""
        try:
            if self.nltk_available:
                return word_tokenize(text.lower())
            else:
                # Fallback tokenization
                return re.findall(r'\b\w+\b', text.lower())
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            # Simple fallback
            return text.lower().split()
    
    def sentence_tokenize(self, text: str) -> List[str]:
        """Sentence tokenize with fallback method"""
        try:
            if self.nltk_available:
                return sent_tokenize(text)
            else:
                # Simple fallback sentence tokenization
                sentences = re.split(r'[.!?]+', text)
                return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.error(f"Error sentence tokenizing: {e}")
            # Very simple fallback
            return [s.strip() for s in text.split('.') if s.strip()]
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        sentences = self.sentence_tokenize(text)
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_size + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'word_count': current_size
                })
                
                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk.split()[-overlap:]) if overlap > 0 else ""
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_size = len(current_chunk.split())
            else:
                current_chunk += " " + sentence
                current_size += sentence_length
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'word_count': current_size
            })
        
        return chunks
    
    def add_document(self, text: str, filename: str = "Unknown"):
        """Add a document to the knowledge base"""
        if not text.strip():
            logger.warning(f"Empty document: {filename}")
            return
        
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Create chunks
        chunks = self.chunk_text(processed_text)
        
        # Add each chunk as a separate document
        for i, chunk in enumerate(chunks):
            self.documents.append({
                'text': chunk['text'],
                'filename': filename,
                'chunk_id': i,
                'word_count': chunk['word_count'],
                'original_text': text  # Keep original for reference
            })
        
        logger.info(f"Added {len(chunks)} chunks from {filename}")
    
    def build_index(self):
        """Build TF-IDF index for document retrieval"""
        if not self.documents:
            logger.warning("No documents to index")
            return
        
        try:
            # Extract text from documents
            doc_texts = [doc['text'] for doc in self.documents]
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            # Fit and transform documents
            self.document_vectors = self.vectorizer.fit_transform(doc_texts)
            
            logger.info(f"Built index for {len(self.documents)} document chunks")
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            st.error("Failed to build document index")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant document chunks"""
        if not self.vectorizer or self.document_vectors is None:
            logger.warning("Index not built yet")
            return []
        
        try:
            # Preprocess query
            processed_query = self.preprocess_text(query)
            
            # Transform query to vector
            query_vector = self.vectorizer.transform([processed_query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            
            # Get top-k most similar documents
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    results.append({
                        **self.documents[idx],
                        'similarity': similarities[idx]
                    })
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def generate_answer(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using Groq API"""
        if not relevant_chunks:
            return "I couldn't find relevant information in the uploaded documents to answer your question."
        
        try:
            # Prepare context
            context = "\n\n".join([
                f"Source: {chunk['filename']} (Chunk {chunk['chunk_id'] + 1})\n{chunk['text']}"
                for chunk in relevant_chunks
            ])
            
            # Create prompt
            prompt = f"""Based on the following document excerpts, please answer the user's question. If the answer cannot be found in the provided context, please say so clearly.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the information available in the context. If you need to make any inferences, please clearly indicate that."""
            
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": "llama-3.1-70b-versatile",
                "temperature": 0.1,
                "max_tokens": 1000,
                "top_p": 1,
                "stop": None
            }
            
            # Make API request
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data["choices"][0]["message"]["content"]
                return answer
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"Error generating response: API returned status code {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return "Error: Could not connect to the AI service. Please check your internet connection and API key."
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating response: {str(e)}"

def main():
    st.set_page_config(
        page_title="Chat with Documents",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📚 Chat with Your Documents")
    st.markdown("Upload documents and ask questions about their content!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key to use the chat functionality"
        )
        
        if not groq_api_key:
            st.warning("Please enter your Groq API key to start chatting with documents.")
            st.info("You can get a free API key from [Groq](https://console.groq.com/)")
            return
        
        # File upload
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files"
        )
        
        # Processing options
        st.header("Options")
        chunk_size = st.slider("Chunk Size (words)", 200, 1000, 500)
        top_k = st.slider("Number of relevant chunks", 3, 10, 5)
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = EnhancedRAGSystem(groq_api_key)
    else:
        st.session_state.rag_system.groq_api_key = groq_api_key
    
    # Process uploaded files
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Clear existing documents
                st.session_state.rag_system.documents = []
                
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Extract text from file
                        text = DocumentProcessor.process_file(uploaded_file)
                        
                        if text:
                            # Add to RAG system
                            st.session_state.rag_system.add_document(text, uploaded_file.name)
                            st.success(f"✓ Processed {uploaded_file.name}")
                        else:
                            st.error(f"✗ Could not extract text from {uploaded_file.name}")
                    
                    except Exception as e:
                        st.error(f"✗ Error processing {uploaded_file.name}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Build index
                if st.session_state.rag_system.documents:
                    with st.spinner("Building search index..."):
                        st.session_state.rag_system.build_index()
                    
                    st.success(f"🎉 Successfully processed {len(uploaded_files)} files and created {len(st.session_state.rag_system.documents)} searchable chunks!")
                    st.session_state.documents_processed = True
                else:
                    st.error("No valid documents were processed.")
    
    # Display document statistics
    if hasattr(st.session_state, 'documents_processed') and st.session_state.documents_processed:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", len(set(doc['filename'] for doc in st.session_state.rag_system.documents)))
        with col2:
            st.metric("Chunks", len(st.session_state.rag_system.documents))
        with col3:
            total_words = sum(doc['word_count'] for doc in st.session_state.rag_system.documents)
            st.metric("Total Words", total_words)
    
    # Chat interface
    st.header("Ask Questions")
    
    if not hasattr(st.session_state, 'documents_processed') or not st.session_state.documents_processed:
        st.info("Please upload and process documents first to start chatting.")
        return
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    question = st.text_input("Ask a question about your documents:", key="question_input")
    
    if st.button("Send") and question:
        with st.spinner("Searching documents and generating answer..."):
            # Retrieve relevant chunks
            results = st.session_state.rag_system.retrieve_relevant_chunks(question, top_k)
            
            if results:
                # Generate answer
                answer = st.session_state.rag_system.generate_answer(question, results)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': answer,
                    'sources': results
                })
            else:
                st.warning("No relevant information found in the documents.")
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:100]}...", expanded=(i == 0)):
                st.write("**Question:**")
                st.write(chat['question'])
                
                st.write("**Answer:**")
                st.write(chat['answer'])
                
                if chat.get('sources'):
                    st.write("**Sources:**")
                    for source in chat['sources']:
                        st.write(f"- {source['filename']} (Chunk {source['chunk_id'] + 1}) - Similarity: {source['similarity']:.3f}")

if __name__ == "__main__":
    main()
