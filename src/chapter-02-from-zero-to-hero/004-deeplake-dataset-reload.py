from configs.utils import db
from langchain.llms import OpenAI
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# llm = OpenAI(model="text-davinci-003", temperature=0)
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# load the existing Deep Lake dataset and specify the embedding function
# my_activeloop_org_id = "javiervb"
# my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
# dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
# db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# create new documents
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset
# db.add_documents(docs)
