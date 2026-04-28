from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral titter influencer grading a tweet .Generate critique and recommendations afor  the user's tweet"
            "Always provide detailed recommendations,includeing requests for lenght,virality, stlyes,etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral techie influencer assistant task with writting excellent twitter posts. Generate the best twitter post possible for the user's request.If the user provide crtique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
