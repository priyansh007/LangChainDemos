import streamlit as st
import boto3
from langchain.llms.bedrock import Bedrock
from langchain.chains import load_summarize_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
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

def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs=text_splitter.split_documents(docs)
    return docs


# Main Streamlit app
def main():
    file_path = os.path.join(temp_folder,"demo.pdf")
    st.title("PDF and Summarizer App")
    
    
    st.header("Upload PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with open(file_path, "wb") as file:
            file.write(uploaded_file.read())
        
        st.success("File uploaded successfully!")
        option = st.radio("Select Chain Option", ["refine", "map_reduce"])
        with st.spinner("Analysing..."):
            docs = read_doc(temp_folder)
            texts = chunk_data(docs)
            chunks_prompt="""
            Please summarize the below speech:
            Speech:`{text}'
            Summary:
            """
            map_prompt_template=PromptTemplate(input_variables=['text'],
                                                template=chunks_prompt)
            final_combine_prompt='''
            Provide a final summary of the entire speech with these important points.
            Add a Generic Motivational Title,
            Start the precise summary with an introduction and provide the
            summary in number points for the speech.
            Speech: `{text}`
            '''
            final_combine_prompt_template=PromptTemplate(input_variables=['text'],
                                                        template=final_combine_prompt)
            summary_chain = load_summarize_chain(
                llm=bot(),
                chain_type=option,
                map_prompt=map_prompt_template,
                combine_prompt=final_combine_prompt_template,
                verbose=False
            )
            summary = summary_chain.run(texts)
            st.text("Summary")
            st.write(summary)
        
# Run the app
if __name__ == "__main__":
    main()
