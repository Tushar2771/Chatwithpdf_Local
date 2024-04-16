import streamlit as st
from llm_chain import load_normal_chain , load_pdf_chat_chain
from langchain.memory import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder
from utils import save_chat_history_json ,get_timestamp,load_chat_history_json
from pdf_handler import add_documents_to_db
from html_templates import bot_template,user_template,css
import os
import yaml



with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def load_chain(chat_history):
    if st.session_state.pdf_chat:
        return load_pdf_chat_chain(chat_history)
    return load_normal_chain(chat_history)




def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""


def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == "new_session":
            st.session_state.new_session_key = get_timestamp() + ".json"
            save_chat_history_json(st.session_state.history,config["chat_history_path"] + st.session_state.new_session_key)
        else:
            save_chat_history_json(st.session_state.history,config["chat_history_path"] + st.session_state.session_key)



def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key 

def toggle_pdf_chat():
    st.session_state.pdf_chat = True


def main():
    st.title("Cdac gpt")
    st.write(css,unsafe_allow_html=True)
    chat_container = st.container()
    st.sidebar.title("Chat session")
    chat_session = ["new_session"] + os.listdir(config["chat_history_path"])


    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.new_session_key = None
        st.session_state.user_question = ""
        st.session_state.session_index_tracker = "new_session"

    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key 
        st.session_state.new_session_key = None

    index = chat_session.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("select a chat session", chat_session, key="session_key", index=index,on_change=track_index)
    st.sidebar.toggle("PDF chat",key = "pdf_chat", value=False)

    if st.session_state.session_key != "new_session" :
        st.session_state.history = load_chat_history_json(config["chat_history_path"]+st.session_state.session_key)
    else:
        st.session_state.history = []

    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)

    user_input = st.text_input("Input",key="user_input",on_change=set_send_input)


    upload_pdf = st.sidebar.file_uploader("Upload a pdf file", accept_multiple_files=True, key= "pdf_upload",type=["pdf"] ,on_change=toggle_pdf_chat)

    if upload_pdf:
        with st.spinner("processing pdf"):
            add_documents_to_db(upload_pdf)


    send_button = st.button("send", key ="send_button")



    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":

            with chat_container:
                llm_response = llm_chain.run(st.session_state.user_question)
                st.session_state.user_question == ""

    if chat_history.messages != []:
         with chat_container:
            st.write("Chat History:")
            for message in chat_history.messages:
                if message.type == "human":
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    save_chat_history()



if __name__ == "__main__":
    main()