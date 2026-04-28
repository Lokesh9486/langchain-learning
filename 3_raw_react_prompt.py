import re
import inspect
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langsmith import traceable
import ollama

MAX_ITERATIONS = 10
# MODEL = "gpt-oss:20b"
MODEL = "llama3.2"


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


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


def get_tool_description(tools):
    descriptions = []
    for tool_name, tool_function in tools.items():
        original_functions = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_functions)
        docstring = inspect.getdoc(tool_function)
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)


tool_description = get_tool_description(tools)
tool_names = ", ".join(tools.keys())
react_prompt = f"""
STRICT RULES - you must follow these exactly:
1. NEVER guess or assume any product price.
You Must call apply_discount After you have received price
2. Only call apply_discount After you have received
a price from get_product_price - do NOT pass a made-up number.
3.NEVER calcuate discount tool.
Always use the apply_discount tool.
4. If the user does not specify a discount tier,
ask them with tier to use - do NOT assume one.

Answer the following questions as best you can using Chinese. You have access to the following tools:

{tool_description}

Use the following format:

Question: the input question you must answer\n
Thought: you should always think about what to do\n
Action: the action to take, should be one of [{tool_names}]\n
Action Input: the input to the action\n
Observation: the result of the action\n
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer\n
Final Answer: the final answer to the original input question\n
\n
Begin!

Question: {{question}}
Thought:"""


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_TRACED(model, options, messages):
    return ollama.chat(model=model, options=options, messages=messages)


@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    prompt = react_prompt.format(question=question)
    scratched = ""
    for iteration in range(1, MAX_ITERATIONS + 1):
        full_prompt = prompt + scratched
        print(f"\n--- Iteration {iteration}---")
        # response = ollama_chat(messages)
        response = ollama_chat_TRACED(
            model=MODEL,
            options={"temperature": 0},
            messages=[{"role": "user", "content": full_prompt}],
        )
        output = response.message.content
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"\n Final Answer: {final_answer}")
            return final_answer
        
        action_match = re.search(f"Action:\s*(.+)",output)
        action_input_match = re.search(r"Action Input:\s*(.+)",output)
        if not action_match or not action_input_match:
            print("No action found in the response, stopping.")
            break
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
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
