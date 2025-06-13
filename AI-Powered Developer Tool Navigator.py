import streamlit as st
import requests

# Tagline
st.title("💬 Chatbot: Find the right developer tools instantly with AI-driven semantic search")
st.write(
    "This is a simple chatbot that uses Google Gemini (Vertex AI) to generate responses. "
    "To use this app, you need to provide your Gemini API key. "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their Gemini API key via `st.text_input`.
gemini_api_key = st.text_input("Gemini API Key", type="password", value="")
if not gemini_api_key:
    st.info("Please add your Gemini API key to continue.", icon="🗝️")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare messages for Gemini API (only last user message for simplicity)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            # Combined into a single line using .get() with default values for robustness
            gemini_reply = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "[Error: Unexpected Gemini API response format.]")
        else:
            gemini_reply = f"[Error: Gemini API returned status {response.status_code}]"

        with st.chat_message("assistant"):
            st.markdown(gemini_reply)
        st.session_state.messages.append({"role": "assistant", "content": gemini_reply})
