import streamlit as st

from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
import boto3


bedrock_client = boto3.client(
            service_name = "bedrock-runtime",
            region_name="us-east-1"
        )
llm = Bedrock(model_id="anthropic.claude-v2",
              client=bedrock_client,
              model_kwargs={"temperature":0.7, 
                            "max_tokens_to_sample": 1000})


template = PromptTemplate(input_variables=['name'],
                          template="You are a bot who has knowledge about all celebrities now give me information about {name}")
llm_chain = LLMChain(verbose=True, prompt=template, llm=llm)

st.title("Bedrock simple LLM - CelebrityBot")
input_text = st.text_input("Give name to LLM")

if input_text:
    st.write(llm_chain({"name":input_text})['text'])