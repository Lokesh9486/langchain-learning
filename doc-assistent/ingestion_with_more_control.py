import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyMap, TavilyExtract
from typing import List
import asyncio
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
tavily_map = TavilyMap(max_depth=3, max_breadth=15, limit=50)
tavily_extract = TavilyExtract()
# embedding = GoogleGenerativeAIEmbeddings(
#     model="gemini-embedding-2-preview",
# )
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
vectorStore = PineconeVectorStore(
    index_name="ingestion-langchain", embedding=embedding
)
retriver = vectorStore.as_retriever(search_kwargs={"k": 3})


def chank_urls(urls, chank_size):
    chanks = []
    for i in range(0, len(urls), chank_size):
        chanks.append(urls[i : i + chank_size])
    return chanks


async def extract_batch(urls: List[str]):
    print(
        "🐍 File: doc-assistent/ingestion_with_more_control.py | Line: 21 | undefined ~ urls",
        urls,
    )
    try:
        results = await asyncio.to_thread(tavily_extract.invoke, {"urls": urls})

        print(
            "🐍 File: doc-assistent/ingestion_with_more_control.py | Line: 19 | undefined ~ results",
            results,
        )
        return results
    except Exception as e:
        print("Error in extract_batch:", e)
        return []


async def async_extract_batch(urls_batch: List[List[str]]):
    print(
        "🐍 File: doc-assistent/ingestion_with_more_control.py | Line: 35 | undefined ~ urls_batch",
        urls_batch,
    )
    try:
        tasks = [extract_batch(batch) for i, batch in enumerate(urls_batch)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print(
            "🐍 File: doc-assistent/ingestion_with_more_control.py | Line: 34 | undefined ~ tasks",
            tasks,
        )
        all_pages = []
        failed_batches = 0
        for result in results:
            if isinstance(result, Exception):
                failed_batches += 1
            else:
                for extracted_page in result["results"]:
                    documents = Document(
                        page_content=extracted_page["raw_content"],
                        metadata={"source": extracted_page["url"]},
                    )
                    all_pages.append(documents)

        if failed_batches > 0:
            print(f"No of batchs failed {failed_batches}")

        print(
            "🐍 File: doc-assistent/ingestion_with_more_control.py | Line: 56 | undefined ~ failed_batches",
            failed_batches,
        )
        print(
            "🐍 File: doc-assistent/ingestion_with_more_control.py | Line: 59 | undefined ~ all_pages",
            all_pages,
        )
        return all_pages

    except Exception as e:
        print("Error results", e)
        return []


# async def index_documnets_async(documents:List[Document],batch_size:int=50):
#     batches = [
#         documents[i:i+batch_size] for i in range(0, len(documents), batch_size)
#     ]
#     print("🐍 File: doc-assistent/ingestion_with_more_control.py | Line: 100 | undefined ~ batches",batches)
#     async def add_batch(batch:list[Document],batch_num:int):
#         try:
#             await vectorStore.aadd_documents(batch)
#             print(f"📈 Batch {batch_num} indexed successfully.")
#         except Exception as e:
#             print(f"Error indexing batch {batch_num}: {e}")
#             return False
#         return True

#     tasks = [add_batch(batch, i) for i, batch in enumerate(batches)]
#     print("🐍 File: doc-assistent/ingestion_with_more_control.py | Line: 111 | undefined ~ tasks",tasks)
#     results = await asyncio.gather(*tasks,return_exceptions= True)

#     successful =sum (1 for result in results if result is True)
#     print("🐍 File: doc-assistent/ingestion_with_more_control.py | Line: 115 | undefined ~ successful",successful)
    
#     print(f"📈 {successful} out of {len(batches)} batches indexed successfully.")
#     if successful == len(batches):
#         print("All batches indexed successfully.")
#     else:
#         print(f"{successful} out of {len(batches)} batches indexed successfully.")


async def index_documnets_async(documents: List[Document], batch_size: int = 20):
    semaphore = asyncio.Semaphore(2)  # 👈 VERY IMPORTANT

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    async def add_batch(batch: list[Document], batch_num: int):
        async with semaphore:
            try:
                await asyncio.sleep(0.5)  # 👈 throttle
                await vectorStore.aadd_documents(batch)
                print(f"📈 Batch {batch_num} indexed successfully.")
                return True
            except Exception as e:
                print(f"Error indexing batch {batch_num}: {e}")
                return False

    results = []
    for i, batch in enumerate(batches):
        result = await add_batch(batch, i)  # 👈 sequential execution
        results.append(result)

    successful = sum(1 for r in results if r is True)

    print(f"📈 {successful} out of {len(batches)} batches indexed successfully.")


async def main():
    print("Hello, World!")
    base_url = "https://www.langchain.com/"
    site_map = tavily_map.invoke(base_url)
    print(
        "🐍 File: doc-assistent/ingestion_with_more_control.py | Line: 20 | undefined ~ site_map",
        site_map,
    )
    chank_url = chank_urls(urls=site_map.get("results", []), chank_size=3)
    print(
        "🐍 File: doc-assistent/ingestion_with_more_control.py | Line: 22 | undefined ~ chank_url",
        chank_url,
    )
    all_docs = await async_extract_batch(chank_url)
    print(
        "🐍 File: doc-assistent/ingestion_with_more_control.py | Line: 78 | undefined ~ all_docs",
        all_docs,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splitted_docs = text_splitter.split_documents(all_docs)
    print(
        "🐍 File: doc-assistent/ingestion_with_more_control.py | Line: 90 | undefined ~ splitted_docs",
        splitted_docs,
    )
    await index_documnets_async(splitted_docs, batch_size=50)


if __name__ == "__main__":
    asyncio.run(main())
