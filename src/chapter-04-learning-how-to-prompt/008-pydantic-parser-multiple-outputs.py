from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from packages.utils import load_config
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import List

load_config()

template = """
Offer a list of suggestions to substitute the specified target_word based on the presented context and the reasoning for each word.
{format_instructions}
target_word={target_word}
context={context}
"""


class Suggestions(BaseModel):
    words: List[str] = Field(
        description="list of substitue words based on context")
    reasons: List[str] = Field(
        description="the reasoning of why this word fits the context")

    @field_validator('words')
    def not_start_with_number(cls, items: List[str]):
        for item in items:
            if item[0].isnumeric():
                raise ValueError("The word can not start with numbers!")
        return items

    @field_validator('reasons')
    def end_with_dot(cls, items: List[str]):
        for idx, item in enumerate(items):
            if item[-1] != ".":
                items[idx] += "."
        return items


parser = PydanticOutputParser(pydantic_object=Suggestions)

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format_prompt(
    target_word="behaviour",
    context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
model = OpenAI(model_name='text-davinci-003', temperature=0.0)

output = model(model_input.to_string())

print(output, end="\n===\n")
print(parser.parse(output))
l = parser.parse(output)
