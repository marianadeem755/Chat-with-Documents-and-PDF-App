import streamlit as st
import os
import openai
import PyPDF2
import pyperclip
import docx2txt
from bs4 import BeautifulSoup

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = " ".join(page.extract_text() for page in pdf_reader.pages)
    return text

# Function to list PDF files in a directory
def list_pdf_files(directory):
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, filename))
    return pdf_files

# Function to list DOCX files in a directory
def list_docx_files(directory):
    docx_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.docx'):
            docx_files.append(os.path.join(directory, filename))
    return docx_files

# Function to generate answers using GPT-3.5 Turbo model
openai_api_key=st.secrets['OPENAI_API_KEY']
def get_answers_from_gpt(text, question, openai_api_key=openai_api_key):
    prompt = text[:4096] + "\nQuestion: " + question + "\nAnswer:"
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": prompt}], temperature=0.7, max_tokens=2000, api_key=openai_api_key)
    return response['choices'][0]['message']['content'].strip()

# Main function
def main():
    st.set_page_config(
        page_title="Chat with Documents and PDF App",
        page_icon="🔗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔗📔📒💬💫Chat with Documents App Created by Maria Nadeem")

    # Input API Key
    # openai_api_key = st.sidebar.text_input('OPENAI_API_KEY', key='password')

    document_folder = st.text_input("Enter the folder path containing documents (PDFs, DOCXs)")

    document_files = []

    if document_folder and os.path.isdir(document_folder):
        document_files.extend(list_pdf_files(document_folder))
        document_files.extend(list_docx_files(document_folder))

    if not document_files:
        st.warning('No document files found in the specified folder or the folder does not exist.')
    else:
        st.info(f"Number of document files found: {len(document_files)}")
        st.markdown('### Document Files')
        for file in document_files:
            st.write(os.path.basename(file))

        random_document = st.selectbox("Select a document file", document_files)

        if random_document.lower().endswith('.pdf'):
            text = extract_text_from_pdf(random_document)
        elif random_document.lower().endswith('.docx'):
            text = docx2txt.process(random_document)

        st.markdown('### Document Content')
        st.markdown('<style> .document-content { color: #000000; font-weight: bold; } </style>', unsafe_allow_html=True)
        st.info(text)  # Display only the first 500 characters for a brief summary


        

        question = st.text_input("Enter your question")

        if question:
            st.markdown('### Answer')
            with st.spinner('Generating answer...'):
                answer = get_answers_from_gpt(text, question, openai_api_key)
            st.write(answer)
            if st.button("Copy Answer Text"):
                pyperclip.copy(answer)
                st.success('Answer text copied to the clipboard')
    st.sidebar.markdown("---")
    # add author name and info
    st.sidebar.markdown("### Author: Maria Nadeem🎉🎊⚡")
    st.sidebar.markdown("### GitHub: [GitHub](https://github.com/marianadeem755)")
    st.sidebar.markdown("### Linkdin: [Linkdin Account](https://www.linkedin.com/in/maria-nadeem-4994122aa/)")
    st.sidebar.markdown("### Contact: [Email](mailto:marianadeem755@gmail.com)")

    # Background Image
                
    st.markdown("""
    <style>
        .stApp {
        background: url("https://images.unsplash.com/photo-1618820830674-35aa0e70dbfc?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxfDB8MXxyYW5kb218MHx8fHx8fHx8MTcwOTA0NDk4MA&ixlib=rb-4.0.3&q=80&utm_campaign=api-credit&utm_medium=referral&utm_source=unsplash_source&w=1080");
        background-size: cover;
        }
    </style>""", unsafe_allow_html=True)
    
    st.sidebar.markdown("---")

if __name__ == "__main__":
    main()
