import os
from dotenv import load_dotenv

load_dotenv()

from langchain.messages import HumanMessage
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
vectorStore = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX"], embedding=embedding
)
retriver = vectorStore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Answer the question based only on the following context:
{context}

Question: {question}

Provide a detailed answer:""",
        )
    ]
)


def format_docs(docs):
    """
    Format retrieved documents into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """
    A simple retrieval chain that retrieves relevant documents and generates an answer without using LCE.
    """
    docs = retriver.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content
    # retrieved_docs = retriver.get_relevant_documents(query)
    # context = format_docs(retrieved_docs)
    # prompt = prompt_template.format(context=context, question=query)
    # response = llm.invoke([HumanMessage(content=prompt)])
    # return response.content


def create_retrieval_cahin_with_lcel():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriver | format_docs,
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retriving...")
    query = "What is  Pinecone in machine learing?"
    # =====================================================
    # Options 0: Raw LLM invocation (No Rag)
    # =====================================================
    result_raw = llm.invoke([HumanMessage(content=query)])
    print(result_raw.content)

    # =====================================================
    # Options 1: Without LCEL
    # =====================================================
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("result_without_lcel:", result_without_lcel)

    # =====================================================
    # Options 2: Without LCEL
    # =====================================================
    print("=" * 40)
    print("- IMPLEMENTATION 2: with LCEL - Better Approach")
    print("=" * 40)
    chain_with_lcel = create_retrieval_cahin_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
