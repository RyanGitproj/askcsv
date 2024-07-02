from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from app.chain import QuestionRequest, Rag_chain

app = FastAPI()

# Créez une instance de rag_chain
chain_instance = Rag_chain()

add_routes(app,
           chain_instance,
           path="/csv",
           input_type=QuestionRequest)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# @app.post("/receive_file")
# async def receive_file(file: UploadFile = File(...)):
#     upload_dir = "uploaded_files"
#     if not os.path.exists(upload_dir):
#         os.makedirs(upload_dir)
#     file_path = os.path.join(upload_dir, file.filename)
#     with open(file_path, 'wb') as f:
#         f.write(await file.read())
#     return "File uploaded successfully"

# file_path = "C:/Users/tsior/Downloads/prenoms.csv"
# file_path = None
# Ajoutez les routes à l'application FastAPI