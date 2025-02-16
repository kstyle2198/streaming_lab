from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_groq import ChatGroq
import streamlit as st

from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜", layout="wide")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b", streaming=True, callbacks=[stream_handler])
        response = llm.invoke(st.session_state.messages)  # 메모리 기능이 적용되어 있다.
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))