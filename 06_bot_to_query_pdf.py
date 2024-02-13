import streamlit as st
import boto3
from langchain.llms.bedrock import Bedrock
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing_extensions import Concatenate
import os

temp_folder = "./data"
def bedrock_client():
    return boto3.client(
            service_name = "bedrock-runtime",
            region_name="us-east-1"
        )

def bot(model_id="anthropic.claude-instant-v1"):
    client = bedrock_client()
    llm = Bedrock(model_id=model_id,
                  client=client,
                  model_kwargs={"temperature":0.7, 
                                "max_tokens_to_sample": 1000})
    return llm

def read_pdf(path):
    pdfreader = PdfReader(path)
            
    # read text from pdf
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

def split_text_into_chunks(raw_text, 
                           chunk_size = 800,
                           chunk_overlap  = 200,
                           separator = "\n"):
    text_splitter = CharacterTextSplitter(
        separator = separator,
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    return texts

def create_vectorstore(texts,
                       model_id="amazon.titan-embed-text-v1"):
    client = bedrock_client()
    embeddings = BedrockEmbeddings(model_id=model_id,
                  client=client)
    vectore_store = FAISS.from_texts(texts, embeddings)
    return vectore_store


# Main Streamlit app
def main():
    file_path = os.path.join(temp_folder,"demo.pdf")
    st.title("PDF and Chatbot App")
    
    # Sidebar with options
    option = st.sidebar.radio("Select Option", ["Upload", "Query"])
    
    # Handle "Upload" option
    if option == "Upload":
        st.header("Upload PDF Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            with open(file_path, "wb") as file:
                file.write(uploaded_file.read())
            st.success("File uploaded successfully!")
            
    
    # Handle "Query" option
    elif option == "Query":
        st.header("Chatbot Interface")
        st.write("Type your question below:")
        # Text input for user's question
        user_question = st.text_input("Your Question:")
        
        # Generate and display response
        if user_question:
            with st.spinner("Analysing..."):
                raw_text = read_pdf(file_path)
                texts = split_text_into_chunks(raw_text)
                vectoreStore = create_vectorstore(texts)
                chain = load_qa_chain(bot(), chain_type="stuff")
                docs = vectoreStore.similarity_search(user_question)
                ans = chain.run(input_documents=docs, question=user_question)
            st.text("Chatbot Response:")
            st.write(ans)

# Run the app
if __name__ == "__main__":
    main()
