from packages.utils import load_config

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from langchain.llms import OpenAI
from langchain.output_parsers import OutputFixingParser

load_config()

# Define your desired data structure.


class Suggestions(BaseModel):
    words: List[str] = Field(
        description="list of substitue words based on context")
    reasons: List[str] = Field(
        description="the reasoning of why this word fits the context")


parser = PydanticOutputParser(pydantic_object=Suggestions)

missformatted_output = '{"words": ["conduct", "manner"], "reasoning": ["refers to the way someone acts in a particular situation.", "refers to the way someone behaves in a particular situation."]}'
# error: reasoning >> reasons


# parser.parse(missformatted_output)

model = OpenAI(model_name='text-davinci-003', temperature=0.0)

outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
output = outputfixing_parser.parse(missformatted_output)

print(output)
