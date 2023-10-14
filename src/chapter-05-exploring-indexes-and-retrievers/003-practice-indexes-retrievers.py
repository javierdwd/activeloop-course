from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from packages.utils import load_config, create_store
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback


load_config()

# Before executing the following code, make sure to have your
# Activeloop key saved in the “ACTIVELOOP_TOKEN” environment variable.

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002")

# create Deep Lake dataset
db = create_store("langchain_course_indexers_retrievers4",
                  embedding=embeddings)

# create retriever from db
retriever = db.as_retriever(search_kwargs={'k': 1})


# create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(model="text-davinci-003"),
    chain_type="stuff",
    retriever=retriever
)


with get_openai_callback() as cb:
    query = "How Google plans to challenge OpenAI?"
    response = qa_chain.run(query)
    print(response)
    print(cb)
