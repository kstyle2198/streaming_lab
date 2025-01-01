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

# Generate chat responses using the OpenAI API
def chat(query, max_tokens, temperature=1, n=1, model="llama-3.3-70b-versatile", stream=False):
    client = Groq()
    messages=[

        # Set an optional system message. This sets the behavior of the
        # assistant and can be used to provide specific instructions for
        # how it should behave throughout the conversation.

        {

            "role": "system",

            "content": "you are a Knowledgable AI Assistant. Generate a reasonable and compact answer to the question. and Generate the answer in Korean (Han-gul)"

        },

        # Set a user message for the assistant to respond to.

        {

            "role": "user",

            "content": f"{query}",

        }

    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        stream=stream
    )
    for chunk in completion:
        try:
            yield chunk.choices[0].delta.content if chunk.choices[0].finish_reason != "stop" else ''
        except:
            yield 'error!'

# Main function to run the Streamlit app
def main():
    # Streamlit settings
    st.markdown("""<style>.block-container{max-width: 66rem !important;}</style>""", unsafe_allow_html=True)
    st.title("Groq Streamlit Streaming Demo")
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
        gpt_answer = chat(st.session_state.chat_history, 1000, model="llama-3.1-70b-versatile", stream=True)
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