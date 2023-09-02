import requests
from newspaper import Article


# LLM Imports
from packages.constants import CHAT_MODEL_GPT3
from langchain.schema import (
    HumanMessage
)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"

# session = requests.Session()

try:
    # response = session.get(article_url, headers=headers, timeout=10)

    status_code = 200
    if status_code == 200:
        article = Article(article_url)
        article.download()
        article.parse()

        article_title = article.title
        article_text = article.text

        # print(f"Title: {article.title}")
        # print(f"Text: {article.text}")

        # prepare template for prompt
        template = """You are a very good assistant that summarizes online articles.

        Here's the article you want to summarize.

        ==================
        Title: {article_title}

        {article_text}
        ==================

        Write a summary of the previous article in a single phrase with less than 100 characters and in a bullet list format.
        """

        prompt = template.format(
            article_title=article.title, article_text=article.text)

        messages = [HumanMessage(content=prompt)]

        from langchain.chat_models import ChatOpenAI

        # load the model
        chat = ChatOpenAI(model_name=CHAT_MODEL_GPT3, temperature=0)

        # generate summary
        summary = chat(messages)
        print(summary.content)
    else:
        print(f"Failed to fetch article at {article_url}")
except Exception as e:
    print(f"Error occurred while fetching article at {article_url}: {e}")
