from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from packages.constants import CHAT_MODEL_GPT3

llm = ChatOpenAI(model_name=CHAT_MODEL_GPT3, temperature=0)

summarization_template = "Summarize the following text to one sentence: {text}"
summarization_prompt = PromptTemplate(
    input_variables=["text"], template=summarization_template)
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)

text = "LangChain provides many modules that can be used to build language model applications. Modules can be combined to create more complex applications, or be used individually for simple applications. The most basic building block of LangChain is calling an LLM on some input. Let's walk through a simple example of how to do this. For this purpose, let's pretend we are building a service that generates a company name based on what the company makes."
summarized_text = summarization_chain.predict(text=text)

print(summarized_text)
print(
    f"\nOriginal text size: {len(text)} and for summarized is {len(summarized_text)}")
