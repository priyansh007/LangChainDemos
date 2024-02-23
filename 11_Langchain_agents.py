import boto3
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from langchain.chains import LLMChain
from langchain.agents import initialize_agent
from langchain.tools import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor
def bedrock_client():
    return boto3.client(
            service_name = "bedrock-runtime",
            region_name="us-east-1"
        )

def bot(model_id="anthropic.claude-instant-v1"):
    client = bedrock_client()
    llm = Bedrock(model_id=model_id,
                  client=client,
                  model_kwargs={"temperature":0.6, 
                                "max_tokens_to_sample": 1000})
    return llm

class CalculatorInput(BaseModel):
    a: str = Field(description="numbers")


def multiply(a: str) -> int:
    """Multiply two numbers."""
    print(a)
    ans=a.split(',')
    return int(ans[0]) * int(ans[1])




def createAgentTools(llm):

    # initialize the math tool
    func_tool = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="multiply numbers but it should be in format like if multiply is of 7 and 8, it should be 7,8",
    args_schema=CalculatorInput,
    return_direct=True,
    )
    
    # when giving tools to LLM, we must pass as list of tools
    tools = [func_tool]
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Human: {query} \n AI:"
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # initialize the LLM tool
    llm_tool = Tool(
        name='Language Model',
        func=llm_chain.run,
        description='use this tool for general purpose queries'
    )
    tools.append(llm_tool)
    return tools

def agent(llm, tools):
    return initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
    )

llm = bot()
tools = createAgentTools(llm)
agent1 = agent(llm, tools)

print(agent1("what is two multiply by eight?"))
print(agent1("what is the capital of Norway?"))
