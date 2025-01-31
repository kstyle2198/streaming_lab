import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# Display the chat history
def create_chat_area(chat_history):
    for chat in chat_history:
        role = chat['role']
        with st.chat_message(role):
            st.write(chat['content'])

    with st.expander("Retrieval Docs", expanded=True):
        for doc in st.session_state.retrieval_docs:
            with st.container(border=True):
                doc


# Generate chat responses using the OpenAI API
def rag(query, retrieval_docs, max_tokens, temperature=1, n=1, model="deepseek-r1-distill-llama-70b", stream=False):
    """
    A Retrieval-Augmented Generation (RAG) function that generates responses 
    based on the query and the retrieved documents.

    Arguments:
        query (str): The user's input query.
        retrieval_docs (list): List of retrieved documents relevant to the query.
        max_tokens (int): Maximum number of tokens in the generated response.
        temperature (float): Sampling temperature for response generation.
        n (int): Number of responses to generate.
        model (str): Model to use for generation.
        stream (bool): If True, stream the response chunks.

    Yields:
        str: Generated response chunks.
    """
    client = Groq()

    # Combine the retrieved documents into a single context string
    retrieval_docs = [doc.page_content for doc in retrieval_docs]
    context = "\n".join(retrieval_docs)

    # Messages for the chat completion
    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable AI assistant. Generate a reasonable and compact answer to the question, leveraging the provided context. Generate the answer in Korean (Han-gul)."
        },
        {
            "role": "system",
            "content": f"Relevant context: {context}"
        },
        {
            "role": "user",
            "content": f"{query}"
        }
    ]

    # Call the chat completion API
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        stream=stream
    )

    # Stream the response chunks
    for chunk in completion:
        try:
            yield chunk.choices[0].delta.content if chunk.choices[0].finish_reason != "stop" else ''
        except:
            yield 'error!'

# Main function to run the Streamlit app

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def main():
    # Streamlit settings
    st.markdown("""<style>.block-container{max-width: 66rem !important;}</style>""", unsafe_allow_html=True)
    st.title("Groq Streamlit Streaming RAG")
    st.markdown("#### Deepseek-r1-distill-llama-70b")
    st.markdown('---')

    with st.expander("📎:green[**Upload Your PDF**]", expanded=True):
        parent_dir = Path(__file__).parent
        base_dir = str(parent_dir) + "\data"
        uploaded_file = st.file_uploader("", type=['PDF', 'pdf'], key="sdfd1")
        btn1 = st.button("Make Retreiver", type='secondary', key="we234")
        try:
            with st.spinner("processing..."):
                if uploaded_file and btn1:
                    st.session_state.retrieval_docs = ""
                    if not os.path.exists(base_dir):
                        os.makedirs(base_dir)

                    files = os.listdir(base_dir)
                    for file in files:
                        file_path = os.path.join(base_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)

                    temp_dir = base_dir 
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    try:
                        vectordb = Chroma(persist_directory="./db", embedding_function=OllamaEmbeddings(base_url="http://localhost:11434", model="bge-m3:latest"))
                        vectordb._client.delete_collection(vectordb._collection.name)
                    except: pass
                    st.session_state.retriever1 = make_retriever_from_pdf(docs)
                st.info(st.session_state.retriever1)

        except Exception as e: 
            st.warning(e)

    # Session state initialization
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    if 'retrieval_docs' not in st.session_state:
        st.session_state.retrieval_docs = []
    if 'retriever1' not in st.session_state:
        st.session_state.retriever1 = ""

    run_chat_interface()





def make_retriever_from_pdf(pdf):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(pdf)
    embeddings_model = OllamaEmbeddings(base_url="http://localhost:11434", model="bge-m3:latest")
    vectorstore = Chroma.from_documents(docs, embeddings_model, persist_directory="./db")
    retriever1 = vectorstore.as_retriever(search_type="mmr",search_kwargs={'k': 3, 'fetch_k': 50})
    return retriever1

# Run the chat interface within Streamlit
def run_chat_interface():
    create_chat_area(st.session_state.chat_history)

    # Chat controls
    clear_button = st.button("Clear Chat History") if len(st.session_state.chat_history) > 0 else None
    user_input = st.chat_input("Ask something:")



    

    # Clear chat history
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.retrieval_docs = []
        st.rerun()

    # Handle user input and generate assistant response
    if user_input or st.session_state.streaming:
        process_user_input(user_input, st.session_state.retriever1)

def process_user_input(user_input, retriever1):
    if user_input:
        st.session_state.retrieval_docs = retriever1.invoke(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        gpt_answer = rag(st.session_state.chat_history, st.session_state.retrieval_docs, 1000, model="deepseek-r1-distill-llama-70b", stream=True)  ## 메모리 기능 포함됨
        st.session_state.generator = gpt_answer
        st.session_state.streaming = True
        st.session_state.chat_history.append({"role": "assistant", "content": ''})
        st.rerun()
    else:
        update_assistant_response()

def update_assistant_response():
    try:
        chunk = next(st.session_state.generator)
        st.session_state.chat_history[-1]["content"] += chunk
        st.rerun()
    except StopIteration:
        st.session_state.streaming = False
        st.rerun()

if __name__ == '__main__':
    main()