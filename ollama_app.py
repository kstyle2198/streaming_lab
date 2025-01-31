import os
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


# Display the chat history
def create_chat_area(chat_history):
    for chat in chat_history:
        role = chat['role']
        with st.chat_message(role):
            st.write(chat['content'])

# Generate chat responses using the OpenAI API
def chat(query):
    llm = OllamaLLM(base_url="http://localhost:11434", model="deepseek-r1:8b")
    prompt = PromptTemplate.from_template('''
                                      You are a Knowledgable AI Assistant. 
                                      Based on the user's query: "{query}", give a reasonable, compact response. 
                                      ''')
    response = (prompt | llm).stream({'query': query})
    for chunk in response:
        try:
            yield chunk
        except:
            yield 'error!'

# Main function to run the Streamlit app
def main():
    # Streamlit settings
    st.markdown("""<style>.block-container{max-width: 66rem !important;}</style>""", unsafe_allow_html=True)
    st.title("Ollama Streamlit Streaming Demo")
    st.markdown("#### deepseek-r1:8b")
    st.markdown('---')

    # Session state initialization
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False

    run_chat_interface()

# Run the chat interface within Streamlit
def run_chat_interface():
    create_chat_area(st.session_state.chat_history)

    # Chat controls
    clear_button = st.button("Clear Chat History") if len(st.session_state.chat_history) > 0 else None
    user_input = st.chat_input("Ask something:")

    # Clear chat history
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()

    # Handle user input and generate assistant response
    if user_input or st.session_state.streaming:
        process_user_input(user_input)

def process_user_input(user_input):
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        gpt_answer = chat(st.session_state.chat_history)    ## 메모리 기능 포함됨
        st.session_state.generator = gpt_answer
        st.session_state.streaming = True
        st.session_state.chat_history.append({"role": "assistant", "content": ''})
        st.rerun()
    else:
        update_assistant_response()

def update_assistant_response():
    try:
        chunk = next(st.session_state.generator)
        st.session_state.chat_history[-1]["content"] += chunk   # ollama
        st.rerun()
    except StopIteration:
        st.session_state.streaming = False
        st.rerun()

if __name__ == '__main__':
    main()
