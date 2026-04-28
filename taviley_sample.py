import os
import asyncio
from dotenv import load_dotenv
from langchain_tavily import TavilyExtract, TavilyMap

load_dotenv()

tavily_map = TavilyMap(max_depth=3, max_breadth=15, limit=50)
tavily_extract = TavilyExtract()


def urls():
    search_result = tavily_map.invoke("https://www.langchain.com/")
    print(
        "🐍 File: python-langchain/taviley_sample.py | Line: 23 | undefined ~ search_result",
        search_result,
    )
    return search_result.get("results", [])


def chank_url(url, chank_size):
    chank_url = []
    for i in range(0, len(url), chank_size):
        chank = url[i : i + chank_size]
        chank_url.append(chank)
    return chank_url


async def extract_batch(urls):
    try:
        results = tavily_extract.invoke(input={"urls": urls})
        print("🐍 File: python-langchain/taviley_sample.py | Line: 32 | undefined ~ results",results)
        return results

    except Exception as e:
        print("Error in extract_batch:", e)


async def main():

    url_batch = chank_url(urls()[:9], 3)

    tasks = [extract_batch(batch) for batch in url_batch]
    results = await asyncio.gather(*tasks)
    print(
        "🐍 File: python-langchain/taviley_sample.py | Line: 44 | undefined ~ results",
        results,
    )


if __name__ == "__main__":

    asyncio.run(main())
    # main()
