from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from packages.utils import load_config
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(
        name="words", description="A substitue word based on context"),
    ResponseSchema(
        name="reasons", description="the reasoning of why this word fits the context.")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)


load_config()


# Prepare the Prompt
template = """
Offer a unique suggestion to substitute the word '{target_word}' based the presented the following text: {context}.
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format(
    target_word="behaviour",
    context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

# Loading OpenAI API
model = OpenAI(model_name='text-davinci-003', temperature=0.0)

# Send the Request
output = model(model_input)
print(parser.parse(output))

# NOTE: Is not recommended to use this class, use PydanticOutputParser instead.
