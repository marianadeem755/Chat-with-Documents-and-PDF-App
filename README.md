# 📚 Chat-with-Documents-and-PDF-App

A Streamlit application that allows you to upload documents and chat with them using advanced AI. Upload PDF, DOCX, or TXT files and ask intelligent questions about their content.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Groq](https://img.shields.io/badge/Groq-API-green.svg)

## ✨ Features

- **📄 Multi-Format Support**: Upload and process PDF, DOCX, and TXT files
- **🤖 AI-Powered Responses**: Get intelligent answers using Groq's advanced language models
- **🔍 Smart Context Processing**: Automatic text chunking for better context understanding
- **📊 Document Analytics**: View detailed statistics about your uploaded files
- **🎨 Professional UI**: Modern, responsive interface with glassmorphism design
- **⚡ Fast Processing**: Optimized document processing and caching
- **🔐 Secure**: Environment-based API key management

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/marianadeem755/document-chat-assistant.git
   cd document-chat-assistant
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**
   
   Choose one of these methods:
   
   **Option A: Environment Variable**
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```
   
   **Option B: .env File**
   ```bash
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🖥️ Usage

1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or TXT files
2. **Wait for Processing**: The app will extract and process text from your documents
3. **Ask Questions**: Enter your questions in the chat interface
4. **Get Answers**: Receive AI-powered responses based on your document content
5. **View Analytics**: Check document statistics and preview content

### Example Questions
- "What is the main topic of this document?"
- "Summarize the key points discussed"
- "What are the conclusions mentioned?"
- "Find information about [specific topic]"

### Supported File Types

- **PDF** (.pdf) - Extracted using PyPDF2
- **Word Documents** (.docx) - Extracted using docx2txt
- **Text Files** (.txt) - Direct text reading

### Processing Limits

- **File Size**: No strict limit (depends on available memory)
- **File Count**: Multiple files supported
- **Text Length**: Automatically chunked for optimal processing
- 
## 🛠️ Development

### Setting up Development Environment

```bash
# Clone the repo
git clone https://github.com/marianadeem755/document-chat-assistant.git
cd document-chat-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```
## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web application framework.
- [ ] Chat history
- [ ] Document comparison
- [ ] Export functionality

---

*Made with ❤️ by Maria Nadeem*
