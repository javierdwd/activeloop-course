from typing import List
from pydantic import BaseModel, Field
from pydantic import validator
from langchain.output_parsers import PydanticOutputParser
from newspaper import Article
import requests
import os
import json
from dotenv import load_dotenv
from langchain.schema import (
    HumanMessage
)
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


load_dotenv()


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

session = requests.Session()


# create output parser class
class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(
        description="Bulleted list summary of the article")

    # validating whether the generated summary has at least three lines
    @validator('summary', allow_reuse=True)
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError(
                "Generated summary has less than three bullet points!")
        return list_of_lines


# set up output parser
parser = PydanticOutputParser(pydantic_object=ArticleSummary)

try:
    response = session.get(article_url, headers=headers, timeout=10)

    if response.status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

        # we get the article data from the scraping part
        article_title = article.title
        article_text = article.text

        # create prompt template
        # notice that we are specifying the "partial_variables" parameter
        template = """
        You are a very good assistant that summarizes online articles.

        Here's the article you want to summarize.

        ==================
        Title: {article_title}

        {article_text}
        ==================

        {format_instructions}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["article_title", "article_text"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()}
        )

        # Format the prompt using the article title and text obtained from scraping
        formatted_prompt = prompt.format_prompt(
            article_title=article_title, article_text=article_text)

        # load the model
        model = OpenAI(model_name="text-davinci-003", temperature=0.0)

        # generate summary
        output = model(formatted_prompt.to_string())
        parsed_output = parser.parse(output)

        print(parsed_output)
    else:
        print(f"Failed to fetch article at {article_url}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")
