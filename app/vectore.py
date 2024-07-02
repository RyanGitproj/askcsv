from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from fastapi import FastAPI, File, UploadFile, Response

app = FastAPI()

@app.post("/csv/upload_file")
async def create_vector_store(file: UploadFile = File(...)):
    file_contents = await file.read()
    loader = CSVLoader(file_contents.decode('utf-8'))
    
    # Retourne un message de succès avec un code de statut 201
    return Response(status_code=201, content="Fichier uploadé avec succès")
    