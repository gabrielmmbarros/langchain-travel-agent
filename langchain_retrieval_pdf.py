import openai
from langchain_openai import AzureChatOpenAI
from langchain.globals import set_debug
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv

# This function loads environment variables from .env file into the application's environment,
# allowing for secure management of sensitive data
load_dotenv()

# Configure the OpenAI API with Azure credentials from environment variables
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_type = os.getenv("AZURE_OPENAI_API_TYPE")

# Initializing the language model with LangChain
# This creates a ChatOpenAI instance with specific parameters for generating responses
llm = AzureChatOpenAI(
    azure_deployment = "gpt-4.1-nano",
    openai_api_version = openai.api_version,
    temperature = 0.2
)

# Load PDF files using PyPDFLoader
loaders = [
    PyPDFLoader("GTB_standard_Nov23.pdf"),
    PyPDFLoader("GTB_gold_Nov23.pdf"),
    PyPDFLoader("GTB_platinum_Nov23.pdf")
]

# Read and combine all documents from the PDFs
documents = []
for loader in loaders:
    documents.extend(loader.load())

# Split the documents into smaller text chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(documents)
# print(texts)

# Create embeddings for the text chunks
embeddings = OpenAIEmbeddings()
# Store the embeddings in a FAISS vector database
db = FAISS.from_documents(texts, embeddings)

# Create a RetrievalQA chain for question answering
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

# Define the question to ask
question = "What should I do if I have a stolen purchased item?"

# Run the chain to get the answer
result = qa_chain.invoke({ "query" : question })
print(result)
