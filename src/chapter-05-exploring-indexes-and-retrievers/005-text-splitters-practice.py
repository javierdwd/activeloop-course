from packages.utils import load_config
from typing import Optional
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.llms.openai import OpenAI
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


load_config()

# loader = PyPDFLoader(Path("tmp/document.pdf").as_posix())
# pages = loader.load_and_split()

# text_splitter = CharacterTextSplitter(
#     chunk_size=100, chunk_overlap=50, separator="\n")
# texts = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings()
llm = OpenAI(model="text-davinci-003")


class Titular(BaseModel):
    nombre: str = Field(description="Nombre del titular")
    nacimiento: Optional[str] = Field(
        description="Fecha de nacimiento del titular", )


parser = PydanticOutputParser(pydantic_object=Titular)

with get_openai_callback() as cb:
    # store = FAISS.from_documents(documents=texts, embedding=embeddings)
    # store.save_local("tmp/faiss_index")
    store = FAISS.load_local("tmp/faiss_index", embeddings)
    query = "¿Cuales son el nombre y fecha de nacimiento del titular?"
    template = """
    {question}
    {context}
    {format_instructions}
    Answer:
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=['question', 'context', 'question'],
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(search_kwargs={'k': 3}),
        chain_type_kwargs={"prompt": prompt}
    )
    output = qa.run(query)
    print(output)

    parsed_output = parser.parse(output)
    print(parsed_output)
    print(cb)
