from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain.pydantic_v1 import BaseModel
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class QuestionRequest(BaseModel):
    question: str
    file_path: str

class AnswerResponse(BaseModel):
    answer: str
    context: List[str]



def load_data(file_path: str) -> List[Document]:
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    return documents

def split_documents(data, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(data)
    return documents

def create_vectorstore(documents):
    vectorstore = Chroma.from_documents(
        documents,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
    )
    return vectorstore

def create_retriever(vectorstore):
    retriever = RunnableLambda(vectorstore.similarity_search).bind(k=5)
    return retriever

def create_llm():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
    return llm



if isinstance(input, dict):
    input = QuestionRequest(**input)
question = input.question
file_path = input.file_path
data = load_data(file_path)
documents = split_documents(data)
vectorstore = create_vectorstore(documents)
retriever = create_retriever(vectorstore)
llm = create_llm()

message = """
Veuillez fournir une réponse à la demande de l'utilisateur en respectant le contexte fourni. Veuillez suivre les règles suivantes :

- Assurez-vous que votre réponse est claire, concise et pertinente par rapport à la demande du client.
- Assurez la pertinence de la question de l'utilisateur.
- Utilisez les informations du contexte pour trouver la réponse la plus précise.

Question de l'utilisateur : {question}

Contexte: {context}
"""
prompt = ChatPromptTemplate.from_messages([("human", message)])
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

chain = rag_chain.with_types(input_type=QuestionRequest)