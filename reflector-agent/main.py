from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, BaseMessage

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from chain import reflection_chain, generate_chain


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflecion_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflecion_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) == 6:
        return END
    return REFLECT


builder.add_conditional_edges(
    GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT}
)

builder.add_edge(REFLECT, GENERATE)
graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().draw_mermaid_png(output_file_path="reflector-agent/flow.png")

if __name__ == "__main__":
    print("Hello langgraph")
    inputs = HumanMessage(
        content="""
                         Make this tweet better: @langchainAI
                         -newly Tool Calling feature is seriously underrated.
                         After along wait, its's here - making the implmentation of agent across different models with function calling
                         Made a video covering their newest blog post"""
    )
    response = graph.invoke({"messages": [inputs]})
