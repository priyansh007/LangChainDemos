import streamlit as st
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
import boto3


bedrock_client = boto3.client(
            service_name = "bedrock-runtime",
            region_name="us-east-1"
        )
llm = Bedrock(model_id="anthropic.claude-v2",
              client=bedrock_client,
              model_kwargs={"temperature":0.7, 
                            "max_tokens_to_sample": 1000})

examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

demo_template = """Word: {word}, "Antonym": {antonym}"""
template = PromptTemplate(input_variables=['word','antonym'],
                          template=demo_template)

few_shot_prompt = FewShotPromptTemplate(
    input_variables=["input"],
    examples=examples,
    example_prompt=template,
    # The prefix is some text that goes before the examples in the prompt.
    # Usually, this consists of intructions.
    prefix="Give the antonym of every input\n",
    # The suffix is some text that goes after the examples in the prompt.
    # Usually, this is where the user input will go
    suffix="Word: {input}\nAntonym: ",
    # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
    example_separator="\n",
)


llm_chain = LLMChain(verbose=True, prompt=few_shot_prompt, llm=llm)

st.title("Bedrock simple LLM - antonymBot")
input_text = st.text_input("Give word to LLM")

if input_text:
    st.write(llm_chain({"input":input_text})['text'])
