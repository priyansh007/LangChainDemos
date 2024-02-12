import streamlit as st

from langchain.chains import LLMChain
from langchain.chains import SequentialChain
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


template1 = PromptTemplate(input_variables=['name'],
                          template="You are a bot who has knowledge about all celebrities now give me information about {name}")

template2 = PromptTemplate(input_variables=['person_details'],
                          template="from {person_details}, when was he/she born?")

template3 = PromptTemplate(input_variables=['date'],
                          template="What happened on day {date}")

llm_chain1 = LLMChain(verbose=True, prompt=template1, llm=llm, output_key='person_details')
llm_chain2 = LLMChain(verbose=True, prompt=template2, llm=llm, output_key='date')
llm_chain3 = LLMChain(verbose=True, prompt=template3, llm=llm, output_key='output')

final_chain = SequentialChain(chains=[llm_chain1,llm_chain2,llm_chain3],
                              input_variables=["name"],
                              output_variables=["person_details","date","output"],
                              verbose=True)

st.title("Bedrock simple LLM - CelebrityBot")
input_text = st.text_input("Give name to LLM")

if input_text:
    st.write(final_chain({"name":input_text}))