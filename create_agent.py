import os

from dotenv import load_dotenv
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# tavily = TavilySearch()


# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches Over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Search {query}")
#     return tavily.search(query)
#     # return "Tokyo wether is sunny"


# llm = ChatOllama(temperature=0, model="gemma3:270m")
# llm = ChatOllama(temperature=0, model="llama3.2")
llm = ChatOpenAI(model="gpt-5-nano")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain cource")
    # result = agent.invoke({"messages":[HumanMessage(content="What is the weather in tokyo?")]})
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Search for 3 job posting for an ai enigneer using langchain in the bay area on linkedin and list thier details"
                )
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
