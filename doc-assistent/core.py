import os
from typing import Any, Dict
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = PineconeVectorStore(
    index_name="ingestion-langchain", embedding=embeddings
)
# ChatGoogleGenerativeAI(model="gemini-2.5-flash")
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve relevant context from the vector store based on the query.
    Args:
        query: The query to retrieve context for
    Returns:
        The relevant context from the vector store.
    """
    retriever_docs = vectorstore.as_retriever().invoke(query, k=4)
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source',"unknown")}\n\nContent:{doc.page_content}"
        )
        for doc in retriever_docs
    )
    return serialized, retriever_docs

def run_llm(query:str):
    systme_prompt = (
        "You are a helpful AI assistent that answers questions about Langchain doucmentation."
        "You have access to a tool that retrivers relevant documentation."
        "Use the tool to find relevent information before answering questions."
        "Always cite the sources you use in yout answers."
        "If you cannot find the answer in the retrieved documentaiton, say so."
    )
    agent = create_agent(model=model,tools = [retrieve_context], system_prompt=systme_prompt)
    message = [{"role":"user","content": query}]
    response =agent.invoke({"messages": message})
    print(response["messages"][-1].content)
    answer = response["messages"][-1].content
    content_docs = []
    for message in response["messages"]:
        print("🐍 File: doc-assistent/core.py | Line: 54 | run_llm ~ message",message)
        if isinstance(message,ToolMessage) and hasattr(message,"artifact"):
            content_docs.extend(message.artifact)
            
        return answer, content_docs


if __name__ == "__main__":
    query = "What deepagents"
    result = run_llm(query)
    print("🐍 File: doc-assistent/core.py | Line: 64 | undefined ~ result",result)