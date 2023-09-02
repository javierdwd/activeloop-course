from dotenv import load_dotenv, find_dotenv

# Env
def load_config():

  load_dotenv(find_dotenv())

  config = load_dotenv('../.env')

  return config


# Langchain
# from langchain.vectorstores import DeepLake
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import DeepLake

# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# my_activeloop_org_id = "javiervb"
# my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
# dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

# db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

