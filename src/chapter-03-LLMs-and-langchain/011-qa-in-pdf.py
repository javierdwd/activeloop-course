from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

from langchain.callbacks import get_openai_callback

prompt = PromptTemplate(
    template="Question: {question}\nAnswer:", input_variables=["question"])

llm = OpenAI(model_name="text-davinci-003", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

with get_openai_callback() as cb:

    print(cb)
