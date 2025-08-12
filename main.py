import streamlit as st
import os
import openai
import PyPDF2
import pyperclip
import docx2txt
import io
from pathlib import Path
import logging
from typing import List, Optional
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Professional Document Chat App",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .section-header {
        color: #2E86AB;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .document-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #1e5f7a;
        transform: translateY(-2px);
        transition: all 0.3s;
    }
</style>
""", unsafe_allow_html=True)

class DocumentProcessor:
    """Professional document processing class"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from PDF file with error handling"""
        try:
            if isinstance(pdf_file, str):
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            else:
                # Handle uploaded file
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return f"Error reading PDF: {str(e)}"
    
    @staticmethod
    def extract_text_from_docx(docx_file) -> str:
        """Extract text from DOCX file with error handling"""
        try:
            if isinstance(docx_file, str):
                text = docx2txt.process(docx_file)
            else:
                # Handle uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                    tmp_file.write(docx_file.read())
                    tmp_file_path = tmp_file.name
                
                text = docx2txt.process(tmp_file_path)
                os.unlink(tmp_file_path)  # Clean up temp file
            
            return text.strip() if text else "No text found in document"
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            return f"Error reading DOCX: {str(e)}"
    
    @staticmethod
    def get_files_from_directory(directory: str, extensions: List[str]) -> List[str]:
        """Get files with specified extensions from directory"""
        try:
            if not os.path.exists(directory):
                return []
            
            files = []
            for filename in os.listdir(directory):
                if any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    files.append(os.path.join(directory, filename))
            return files
        except Exception as e:
            logger.error(f"Error accessing directory: {str(e)}")
            return []

class GPTProcessor:
    """Professional GPT interaction class"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
    
    def get_answer(self, text: str, question: str) -> str:
        """Get answer from GPT with better prompt engineering"""
        try:
            # Truncate text if too long (leave room for question and response)
            max_text_length = 3000
            if len(text) > max_text_length:
                text = text[:max_text_length] + "...[truncated]"
            
            prompt = f"""Based on the following document content, please answer the question accurately and concisely.

Document Content:
{text}

Question: {question}

Please provide a detailed and helpful answer based solely on the information provided in the document. If the answer cannot be found in the document, please state that clearly."""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            
            return response['choices'][0]['message']['content'].strip()
        
        except Exception as e:
            logger.error(f"Error getting GPT response: {str(e)}")
            return f"Error generating answer: {str(e)}. Please check your OpenAI API key and try again."

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Professional Document Chat App</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable document questioning"
        )
        
        if api_key:
            st.success("✅ API Key provided")
        else:
            st.warning("⚠️ Please provide your OpenAI API key")
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Upload files** or **specify folder path**
        2. **Select a document** from the list
        3. **Ask questions** about the content
        4. **Get AI-powered answers**
        """)
        
        st.markdown("---")
        st.markdown("### 👨‍💻 About")
        st.markdown("**Author:** Maria Nadeem")
        st.markdown("**GitHub:** [Profile](https://github.com/marianadeem755)")
        st.markdown("**LinkedIn:** [Profile](https://www.linkedin.com/in/maria-nadeem-4994122aa/)")
        st.markdown("**Contact:** [Email](mailto:marianadeem755@gmail.com)")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="section-header">📁 Document Source</h3>', unsafe_allow_html=True)
        
        # Choose input method
        input_method = st.radio(
            "Choose how to provide documents:",
            ["Upload Files", "Folder Path"],
            horizontal=True
        )
        
        document_files = []
        document_text = ""
        selected_file_name = ""
        
        if input_method == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload your documents",
                type=['pdf', 'docx'],
                accept_multiple_files=True,
                help="Upload PDF or DOCX files"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
                
                # Create a list of file names for selection
                file_names = [file.name for file in uploaded_files]
                selected_file_name = st.selectbox("Select a document:", file_names)
                
                # Find the selected file object
                selected_file = next(file for file in uploaded_files if file.name == selected_file_name)
                
                # Process the selected file
                processor = DocumentProcessor()
                
                with st.spinner("📖 Extracting text from document..."):
                    if selected_file_name.lower().endswith('.pdf'):
                        document_text = processor.extract_text_from_pdf(selected_file)
                    elif selected_file_name.lower().endswith('.docx'):
                        document_text = processor.extract_text_from_docx(selected_file)
        
        elif input_method == "Folder Path":
            folder_path = st.text_input(
                "📂 Enter folder path:",
                placeholder="e.g., C:\\Documents or /home/user/documents",
                help="Enter the full path to the folder containing your documents"
            )
            
            if folder_path:
                processor = DocumentProcessor()
                document_files = processor.get_files_from_directory(
                    folder_path, 
                    ['.pdf', '.docx']
                )
                
                if document_files:
                    st.success(f"✅ Found {len(document_files)} document(s)")
                    
                    # Display found files
                    file_names = [os.path.basename(file) for file in document_files]
                    selected_file_name = st.selectbox("Select a document:", file_names)
                    
                    # Get the full path of selected file
                    selected_file_path = next(
                        file for file in document_files 
                        if os.path.basename(file) == selected_file_name
                    )
                    
                    # Extract text from selected file
                    with st.spinner("📖 Extracting text from document..."):
                        if selected_file_path.lower().endswith('.pdf'):
                            document_text = processor.extract_text_from_pdf(selected_file_path)
                        elif selected_file_path.lower().endswith('.docx'):
                            document_text = processor.extract_text_from_docx(selected_file_path)
                
                else:
                    if os.path.exists(folder_path):
                        st.warning("⚠️ No PDF or DOCX files found in the specified folder")
                    else:
                        st.error("❌ Folder path does not exist. Please check the path and try again.")
    
    with col2:
        st.markdown('<h3 class="section-header">💬 Document Chat</h3>', unsafe_allow_html=True)
        
        if document_text and selected_file_name:
            # Display document info
            st.info(f"📄 **Current Document:** {selected_file_name}")
            
            # Show document preview
            with st.expander("👀 Document Preview"):
                preview_text = document_text[:1000] + "..." if len(document_text) > 1000 else document_text
                st.markdown(f'<div class="document-content">{preview_text}</div>', unsafe_allow_html=True)
            
            # Question input
            question = st.text_area(
                "❓ Ask a question about the document:",
                placeholder="e.g., What is the main topic of this document?",
                height=100
            )
            
            # Generate answer button
            if st.button("🚀 Get Answer", type="primary"):
                if not api_key:
                    st.error("❌ Please provide your OpenAI API key in the sidebar")
                elif not question.strip():
                    st.warning("⚠️ Please enter a question")
                else:
                    with st.spinner("🤔 Thinking... Please wait"):
                        gpt_processor = GPTProcessor(api_key)
                        answer = gpt_processor.get_answer(document_text, question)
                    
                    # Display answer
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.markdown("### 💡 Answer:")
                    st.write(answer)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Copy to clipboard button
                    col_a, col_b = st.columns([1, 4])
                    with col_a:
                        if st.button("📋 Copy Answer"):
                            try:
                                pyperclip.copy(answer)
                                st.success("✅ Copied!")
                            except:
                                st.error("❌ Copy failed")
        
        else:
            st.info("👆 Please select a document from the left panel to start chatting")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>🚀 Built with Streamlit • 🤖 Powered by OpenAI • 📚 Professional Document Processing</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
