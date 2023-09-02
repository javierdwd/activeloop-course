from langchain.llms import GPT4All
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

callback_manager = AsyncCallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(model="../models/ggml-model-q4_0.bin",
              callback_manager=callback_manager, verbose=True)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "How many cars I have if I have 5 cars?"
llm_chain.run(question)
