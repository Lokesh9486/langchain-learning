import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl , TavilyExtract, TavilyMap
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

embeddings= GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",chunk_size= 50,retry_min_seconds= 10
    ) 
vectore_store = PineconeVectorStore(index_name=os.environ["PINECONE_INDEX"],embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(map_depth=5,max_breadth= 20,max_page=1000)
tavily_crawl = TavilyCrawl()
async def main():
    res = tavily_crawl.invoke({
        "url":"https://docs.langchain.com/oss/python/langchain/overview",
        "max_depth": 5,
        "extract_depth":"advanced"
    })
    print("🐍 File: doc-assistent/ingestion.py | Line: 32 | undefined ~ res",res)
    print("Hello, World!")


if __name__ == "__main__":
    asyncio.run(main())
