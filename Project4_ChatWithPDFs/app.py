import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Retrieve the Google API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    # Loop through each uploaded PDF document
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # Extract text from each page in the PDF
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into smaller chunks
def get_text_chunks(text):
    # Split text into chunks of 10,000 characters with 1,000 character overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a vector store from text chunks
def get_vector_store(text_chunks):
    # Create embeddings using Google Generative AI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create a FAISS vector store from text chunks
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save the vector store locally
    vector_store.save_local("faiss_index")

# Function to create a conversational chain for Q&A
def get_conversational_chain():
    # Define a prompt template for the Q&A model
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # Load the Q&A model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    # Create a prompt with the defined template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Load the Q&A chain with the model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate a response
def user_input(user_question):
    # Create embeddings using Google Generative AI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Load the previously saved FAISS vector store with dangerous deserialization enabled
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Search for similar documents in the vector store based on the user's question
    docs = new_db.similarity_search(user_question)
    # Get the conversational chain for Q&A
    chain = get_conversational_chain()
    # Generate a response using the chain and the documents found
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    # Print the response to the console
    print(response)
    # Display the response in Streamlit
    st.write("Reply: ", response["output_text"])

# Main function to run the Streamlit app
def main():
    # Set the page configuration for the Streamlit app
    st.set_page_config(page_title="Chat PDF")
    # Display a header in the Streamlit app
    st.header("Chat with PDF using Gemini")
    # Create a text input box for the user to ask a question
    user_question = st.text_input("Ask a Question from the PDF Files")
    # If the user has entered a question, process it
    if user_question:
        user_input(user_question)
    # Create a sidebar for file uploading
    with st.sidebar:
        st.title("Menu:")
        # Allow the user to upload multiple PDF files
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        # Create a button for processing the uploaded PDF files
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Extract text from the uploaded PDF files
                raw_text = get_pdf_text(pdf_docs)
                # Split the extracted text into chunks
                text_chunks = get_text_chunks(raw_text)
                # Create and save a vector store from the text chunks
                get_vector_store(text_chunks)
                st.success("Done")

# Run the main function if this script is executed
if __name__ == "__main__":
    main()