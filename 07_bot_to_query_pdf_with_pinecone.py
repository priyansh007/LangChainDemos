import streamlit as st
import boto3
from langchain.llms.bedrock import Bedrock
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from typing_extensions import Concatenate
import os
from pinecone import Pinecone as PCone
from dotenv import load_dotenv

#Put in .Env file
#PINECONE_API_KEY = ""
#PINECONE_INDEX_NAME = "" 

load_dotenv()


temp_folder = "./data"
def bedrock_client():
    return boto3.client(
            service_name = "bedrock-runtime",
            region_name="us-east-1"
        )

def pincone_db(doc, model_id):
    client = bedrock_client()
    embeddings = BedrockEmbeddings(model_id=model_id,
                  client=client)
    index_name="demotry"
    vectorstore = Pinecone.from_documents(doc, embeddings, index_name=index_name)
    return vectorstore


def bot(model_id="anthropic.claude-instant-v1"):
    client = bedrock_client()
    llm = Bedrock(model_id=model_id,
                  client=client,
                  model_kwargs={"temperature":0.7, 
                                "max_tokens_to_sample": 1000})
    return llm

def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs=text_splitter.split_documents(docs)
    return docs


def create_vectorstore(texts,
                       model_id="amazon.titan-embed-text-v1"):
    vectore_store = pincone_db(texts, model_id)
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
                docs = read_doc(temp_folder)
                texts = chunk_data(docs)
                vectoreStore = create_vectorstore(texts)
            with st.spinner("Analysing..."):
                chain = load_qa_chain(bot(), chain_type="stuff")
                docs = vectoreStore.similarity_search(user_question, k=2)
                print(docs)
                ans = chain.run(input_documents=docs, question=user_question)
            st.text("Chatbot Response:")
            st.write(ans)

# Run the app
if __name__ == "__main__":
    main()
