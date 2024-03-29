import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
import boto3


bedrock_client = boto3.client(
            service_name = "bedrock-runtime",
            region_name="us-east-1"
        )

llm = Bedrock(model_id="anthropic.claude-instant-v1",
              client=bedrock_client,
              model_kwargs={"temperature":0.7})

#Streamlit refreshes whole page so If you want to use memory make sure you save in session or cookie
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory()
    


bot_memory = st.session_state['memory']
llm_chain = ConversationChain(verbose=True, llm=llm, memory=bot_memory)

st.title("Bedrock simple LLM")
input_text = st.text_input("Ask Question:")

if input_text:
    st.write(llm_chain(input_text)['response'])
    st.session_state['memory'] = bot_memory
    with st.expander('History'):
        st.info(bot_memory.buffer)