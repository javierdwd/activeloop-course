import pprint
import packages.utils
from packages.constants import CHAT_MODEL_GPT3

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(model_name=CHAT_MODEL_GPT3, temperature=0)

batch_messages = [
    [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."),
        HumanMessage(
            content="Translate the following sentence: I love programming.")
    ],
    [
        SystemMessage(
            content="You are a helpful assistant that translates French to English."),
        HumanMessage(
            content="Translate the following sentence: J'aime la programmation.")
    ],
]

pprint.pprint(chat.generate(batch_messages).dict(), indent=4)
