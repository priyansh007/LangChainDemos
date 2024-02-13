import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage,SystemMessage,AIMessage
import boto3


bedrock_client = boto3.client(
            service_name = "bedrock-runtime",
            region_name="us-east-1"
        )

llm = Bedrock(model_id="anthropic.claude-instant-v1",
              client=bedrock_client,
              model_kwargs={"temperature":0.7})

#Streamlit refreshes whole page so If you want to use memory make sure you save in session or cookie
if 'history' not in st.session_state:
    st.session_state['history']=[
        SystemMessage(content="You are a AI assitant who reply in Poem format")
    ]
   
llm_chain = ConversationChain(verbose=True, llm=llm)


st.title("Bedrock simple LLM")
input_text = st.text_input("Ask Question:")

if input_text:
    st.session_state['history'].append(HumanMessage(content=input_text))
    answer=llm_chain(st.session_state['history'])['response']
    st.session_state['history'].append(AIMessage(content=answer))
    st.write(answer)
    with st.expander('History'):
        st.info(st.session_state['history'])