# from fastapi import FastAPI, UploadFile, File
# from secrets import token_hex
# from fastapi.responses import RedirectResponse
# from langserve import add_routes
# # from app.api_ask import QuestionRequest, AnswerResponse,chain
# from app.chain import chain

# app = FastAPI()

# @app.get("/")
# async def redirect_root_to_docs():
#     return RedirectResponse("/docs")

# @app.post('/upload')
# async def upload(file :UploadFile = File(...)):
#     file_ext = file.filename.split(".").pop()
#     file_name = token_hex(10)
#     file_path = f"file/{file_name}.{file_ext}"
#     with open(file_path, "wb") as f:
#         content = await file.read()
#         f.write(content)
#     return {"success": True, "file_path": file_path, "message": "File upload successfully"}

# # Ajoutez les routes à l'application FastAPI
# add_routes(app,
#             chain, 
#             path="/csv")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)