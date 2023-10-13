from langchain.text_splitter import CharacterTextSplitter
from packages.utils import load_config
from langchain.document_loaders import TextLoader
from pathlib import Path
from langchain.embeddings import OpenAIEmbeddings

load_config()

# text to write to a local file
# taken from https://www.theverge.com/2023/3/14/23639313/google-ai-language-model-palm-api-challenge-openai
text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses “generate text, images, code, videos, audio, and more from
simple natural language prompts.”

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
PaLM is a flexible system that can potentially carry out all sorts of text generation and
editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
example, or you could use it for tasks like summarizing text or even writing code.
(It’s similar to features Google also announced today for its Workspace apps like Google
Docs and Gmail.)
"""

# write text to local file
# with open(Path("tmp/my_file.txt"), "w") as file:
#     file.write(text)

# use TextLoader to load text from local file
loader = TextLoader(Path("tmp/my_file.txt"))
docs_from_file = loader.load()

print(len(docs_from_file))
# 1


# create a text splitter
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# split documents into chunks
docs = text_splitter.split_documents(docs_from_file)

print(len(docs))
# 2


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
