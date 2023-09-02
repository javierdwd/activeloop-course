# Import necessary modules
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback

# Initialize language model
llm = OpenAI(model_name="text-davinci-003", temperature=0)

# Load the summarization chain
summarize_chain = load_summarize_chain(llm)

# Load the document using PyPDFLoader
document_loader = PyPDFLoader(
    file_path="../../documents/The_One_Page_Linux_Manual.pdf")
document = document_loader.load()

# Summarize the document

with get_openai_callback() as cb:
    summary = summarize_chain(document)
    print(summary['output_text'])
    print(cb)
