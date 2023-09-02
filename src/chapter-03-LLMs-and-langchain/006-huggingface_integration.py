from langchain import PromptTemplate
from langchain import HuggingFaceHub, LLMChain

template = """Question: {question}

Answer: """

multi_template = """Answer the following questions one at a time (each question is separated by the ascii new line character). Separate the answers with the ascii new line character. Do not stop until finished answering the complete list of questions.

Questions:
{question}

Answers:
"""

prompt = PromptTemplate(
    template=multi_template,
    input_variables=['question']
)

# user question
# question = "What is the capital city of France?"

# initialize Hub LLM
hub_llm = HuggingFaceHub(
    repo_id='google/flan-t5-large',
    model_kwargs={'temperature': 0.5}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm,
    verbose=True
)

# ask the user question about the capital of France
# print(llm_chain.run(question))

qa = [
    {'question': "What is the capital city of France?"},
    {'question': "What is the largest mammal on Earth?"},
    {'question': "Which gas is most abundant in Earth's atmosphere?"},
    {'question': "What color is a ripe banana?"}
]

print(llm_chain.generate(qa))

# qs_str = (
#     "What is the capital city of France?\n" +
#     "What is the largest mammal on Earth?\n" +
#     "Which gas is most abundant in Earth's atmosphere?\n" +
#     "What color is a ripe banana?\n"
# )

# print(llm_chain.run(qs_str)) # <<<< This didn't work.
