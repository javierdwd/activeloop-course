from packages.utils import load_config

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

# Envs.
load_config();

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
# Initialize LLM
llm = OpenAI(model_name="text-davinci-003", temperature=0)

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}?
"""
prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template,
)

# Create the LLMChain for the prompt
llm = OpenAI(model_name="text-davinci-003", temperature=0)

# Input data for the prompt
input_data = {"theme": "interstellar travel", "year": "3030"}

# Create LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the LLMChain to get the AI-generated song title
response = chain.run(input_data)

print("Theme: interstellar travel")
print("Year: 3030")
print("AI-generated song title:", response)