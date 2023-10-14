from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from packages.utils import load_config, create_store
from langchain.embeddings import OpenAIEmbeddings

from langchain.callbacks import get_openai_callback
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


load_config()

# Before executing the following code, make sure to have your
# Activeloop key saved in the “ACTIVELOOP_TOKEN” environment variable.

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002")

# create Deep Lake dataset
db = create_store("langchain_course_indexers_retrievers4",
                  embedding=embeddings)

# create retriever from db
retriever = db.as_retriever()


# create a retrieval chain

llm = OpenAI(model="text-davinci-003")

# Uncompress version
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever
# )
# response = qa_chain.run(query)

query = "How Google plans to challenge OpenAI?"

# create compressor for the retriever
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)


with get_openai_callback() as cb:
    # retrieving compressed documents
    retrieved_docs = compression_retriever.get_relevant_documents(
        "How Google plans to challenge OpenAI?"
    )
    print(retrieved_docs[0].page_content)

    print(cb)
