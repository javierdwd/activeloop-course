from packages.constants import CHAT_MODEL_GPT3

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(model_name=CHAT_MODEL_GPT3, temperature=0)

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to Spanish."),
    HumanMessage(
        content="Translate the following sentence: I love programming.")
]

print(chat(messages))
