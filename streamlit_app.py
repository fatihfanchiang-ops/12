"""
streamlit_app.py — a tiny chat UI for the trained model.

Run:
    streamlit run streamlit_app.py
"""
import streamlit as st

from model import predict

st.set_page_config(page_title="My Personal AI", page_icon="🤖")
st.title("🤖 My Personal AI Model")
st.write("Ask me anything about my name, birthday, height, weight, shoe size, favorite subject, hobby, or future goal.")

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    result = predict(user_input)
    reply = f"{result['response']}\n\n_intent: `{result['intent']}` · confidence: `{result['confidence']}`_"
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
