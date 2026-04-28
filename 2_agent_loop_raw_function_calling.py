from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langsmith import traceable
import ollama

MAX_ITERATIONS = 10
MODEL = "gpt-oss:20b"
# MODEL = "llama3.2"


# MODEL = 'qwen3-vl:30b'
# MODEL = 'gemma3:270m'
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """
    Look up the price of the product in the catalog.
    """
    print(f"  >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers: bronze,sliver, gold,
    """
    print(f">> Executing apply_discount(price={price},discount_tier='{discount_tier}')")
    discount_precentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_precentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Product name like laptop, headphones, keyboard",
                    }
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply discount tier to price",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number"},
                    "discount_tier": {
                        "type": "string",
                        "enum": ["bronze", "silver", "gold"],
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tool_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant."
                "You have access to a product catalog tool"
                "and a discount tool. \n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price."
                "You Must call apply_discount After you have received price"
                "2. Only call apply_discount After you have received"
                "a price from get_product_price - do NOT pass a made-up number.\n"
                "3.NEVER calcuate discount tool.\n"
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier,"
                "ask them with tier to use - do NOT assume one."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration}---")
        response = ollama_chat(messages)
        ai_message = response.message
        # ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        print(f"tool_calls", tool_calls)
        print(f"ai_message", ai_message)
        if not tool_calls:
            print(f"\n Final Answer: {ai_message}")
            return ai_message.content

        tool_calls = tool_calls[0]
        tool_name = tool_calls.function.name
        tool_args = tool_calls.function.arguments
        print(f"Tool Selected {tool_name} with args: {tool_args}")
        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        # observation = tool_to_use.invoke(tool_args)
        observation = tool_to_use(**tool_args)
        print(f"[Tool result] {observation}")
        messages.append(ai_message)
        messages.append(
            {"role": "tool", "content": str(observation)}
            # ToolMessage(
            #     content=str(observation),
            #     tool_call_id=tool_call_id,
            # )
        )

    print("Max iteration reached without final output")
    return None


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")
    result = run_agent("What is the price of laptop after apply a gold discount")
