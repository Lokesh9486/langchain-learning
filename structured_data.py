from dotenv import load_dotenv


from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

load_dotenv()


class Source(BaseModel):
    """Search for a source used by the agent"""

    url: str = Field(description="The Url of the source")


class AgnetResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="Lsit of sorces used to generate the answer"
    )


# llm = ChatOllama(model="llama3.2")
llm = ChatOpenAI(model="gpt-5-nano")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgnetResponse)


def main():
    print("Hello from langchain cource")

    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Search for 3 job posting for an ai enigneer using langchain in the bay area on linkedin and list thier details"
                )
            ]
        }
    )
    print(response)


if __name__ == "__main__":
    main()
