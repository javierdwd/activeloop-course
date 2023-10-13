import os
from packages.utils import load_config
from langchain.vectorstores import DeepLake
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

load_config()

# Before executing the following code, make sure to have your
# Activeloop key saved in the “ACTIVELOOP_TOKEN” environment variable.

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002")

# use TextLoader to load text from local file
loader = TextLoader(Path("tmp/my_file.txt").as_posix())
docs_from_file = loader.load()

# create a text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=200, chunk_overlap=20, separator="\n|\.", is_separator_regex=True, )

# split documents into chunks
docs = text_splitter.split_documents(docs_from_file)


def cleanMetadata(doc: Document):
    return Document(page_content=doc.page_content)


# Patch: keeping metadata triggers a random python error
docs2 = list(map(cleanMetadata, docs))

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = "langchain_course_indexers_retrievers4"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

# add documents to our Deep Lake dataset

db.add_documents(docs2)
