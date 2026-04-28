import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

load_dotenv()


if __name__ == "__main__":
    print("Ingesting data...")
    loader = TextLoader(os.path.join("", "mediumblog1.txt"), encoding="utf-8")
    document = loader.load()
    print("splitting data...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print("🐍 File: python-langchain/ingestion.py | Line: 19 | undefined ~ texts",texts)
    print(document[0].page_content)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
    )
    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ["PINECONE_INDEX"],
    )
    print("finished")
