import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb  # Utility function to get a Streamlit callback handler with context



# st.title("StreamLit 🤝 LangGraph")
# st.markdown("#### Simple Chat Streaming")

# if "messages" not in st.session_state:
#     # default initial message to render in message state
#     st.session_state["messages"] = [{"role":"ai", "content": "How can I help you?"}]

# # Loop through all messages in the session state and render them as a chat on every st.refresh mech
# for msg in st.session_state.messages:
#     if msg['role'] == "ai":
#         st.chat_message("assistant").write(msg["content"])
#     if msg['role'] == "human":
#         st.chat_message("user").write(msg["content"])


# import requests
# def invoke_our_graph(messages:list):
#     url = "http://localhost:8000/invoke"
#     json={"messages": messages}
#     response = requests.post(url, json=json)
#     response = response.json()
#     return response


# # takes new input in chat box from user and invokes the graph
# if prompt := st.chat_input():
#     st.session_state.messages.append({"role":"human", "content": prompt})
#     st.chat_message("user").write(prompt)
#     st.session_state.messages = list(st.session_state.messages)

#     with st.chat_message("assistant"):
#         st_callback = get_streamlit_cb(st.container())
#         response = invoke_our_graph(st.session_state.messages)
#         st.session_state.messages.append({"role":"ai", "content": response['messages'][-1]['content']})
#         st.rerun()



### 이하는 스트리밍 테스트 코드 ###############
import streamlit as st
import requests

st.title("StreamLit 🤝 LangGraph")
st.markdown("#### Simple Chat Streaming")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "ai", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    role = "assistant" if msg['role'] == "ai" else "user"
    st.chat_message(role).write(msg["content"])

def stream_response(messages: list):
    """Generator function to stream response from backend"""
    url = "http://localhost:8000/invoke"  # Update to your streaming endpoint
    try:
        with requests.post(url, json={"messages": messages}, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except requests.exceptions.RequestException as e:
        yield f"Error connecting to server: {str(e)}"

if prompt := st.chat_input():
    # Add user message to history and display
    st.session_state.messages.append({"role": "human", "content": prompt})
    st.chat_message("user").write(prompt)

    # Prepare assistant message container
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in stream_response(st.session_state.messages):
            print(chunk)

            full_response += chunk
            response_placeholder.markdown(full_response + " ")  # Add typing indicator
        
        # Finalize the response display and update state
        response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "ai", "content": full_response})


st.session_state.messages