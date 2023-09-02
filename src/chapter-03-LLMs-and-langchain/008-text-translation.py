from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from packages.constants import CHAT_MODEL_GPT3

llm = ChatOpenAI(model_name=CHAT_MODEL_GPT3, temperature=0)

translation_template = "Translate the following text from {source_language} to {target_language}: {text}"
translation_prompt = PromptTemplate(input_variables=[
                                    "source_language", "target_language", "text"], template=translation_template)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)

text = "LangChain provides many modules that can be used to build language model applications. Modules can be combined to create more complex applications, or be used individually for simple applications. The most basic building block of LangChain is calling an LLM on some input. Let's walk through a simple example of how to do this. For this purpose, let's pretend we are building a service that generates a company name based on what the company makes."
source_language = "English"
target_language = "Spanish"
translated_text = translation_chain.predict(
    source_language=source_language, target_language=target_language, text=text)

print(translated_text)
