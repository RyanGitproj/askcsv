# chain.py

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, Runnable
from langchain.pydantic_v1 import BaseModel
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_data(file_path: str) -> List[Document]:
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    return documents

def create_chain(file_path: str) -> Runnable:
    data = load_data(file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(
        documents,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
    )

    retriever = RunnableLambda(vectorstore.similarity_search).bind(k=5)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )

    class Question(BaseModel):
        __root__: str

    chain = chain.with_types(input_type=Question)
    return chain
