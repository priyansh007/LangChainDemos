import streamlit as st

from langchain.llms.bedrock import Bedrock
import boto3


bedrock_client = boto3.client(
            service_name = "bedrock-runtime",
            region_name="us-east-1"
        )
llm = Bedrock(model_id="anthropic.claude-v2",
              client=bedrock_client,
              model_kwargs={"prompt": "\n\nHuman: {userQuestion}\n\nAssistant(Suppose you are a history teacher answer accordingly):",
                            "temperature":0.7, 
                            "max_tokens_to_sample": 1000})

st.title("Bedrock simple LLM")
input_text = st.text_input("Ask Questions to LLM")

if input_text:
    st.write(llm(input_text))