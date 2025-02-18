from fastapi import FastAPI, UploadFile, File, Form
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import os
import tempfile
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = None
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

@app.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...)):
    global retriever
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    
    for file in files:
        temp_filepath = os.path.join(temp_dir.name, file.filename)
        with open(temp_filepath, "wb") as f:
            f.write(await file.read())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="bge-m3:latest")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    
    return {"message": "Files processed successfully"}

@app.post("/chat/")
async def chat(query: str = Form(...)):
    global retriever, memory
    if retriever is None:
        return {"error": "No documents uploaded yet."}
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0, streaming=False)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)
    
    system_prompt = """
    You are a knowledgeable shipbuilding engineer. 
    Think step by step to answer the question.
    Generate your answer including its reason (quote original sentences from the context).
    Do NOT repeat the same sentences in your answer.
    You MUST generate the answer in Korean (Hangul).
    """
    
    response = qa_chain.run(system_prompt + "\n" + query)
    
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
