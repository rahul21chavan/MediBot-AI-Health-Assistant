import streamlit as st
import google.generativeai as genai
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize chat model
model = genai.GenerativeModel("gemini-pro")


# Load custom dataset
def load_dataset():
    return pd.read_csv("hospital_chatbot.csv")


dataset = load_dataset()


def find_best_match(query):
    """Finds the best matching response from the dataset."""
    for _, row in dataset.iterrows():
        if query.lower() in row['Question'].lower():
            return row['Answer']
    return None


def generate_response(user_input, chat_history):
    """Generate chatbot response using Gemini API and custom dataset."""
    dataset_response = find_best_match(user_input)
    if dataset_response:
        return dataset_response

    chat_history.append({"role": "user", "content": user_input})
    response = model.generate_content([chat_history])
    chat_response = response.text
    chat_history.append({"role": "assistant", "content": chat_response})
    return chat_response


# Streamlit UI
def main():
    st.set_page_config(page_title="MediBot - AI Health Assistant", page_icon="🩺", layout="wide")

    st.markdown("""
        <style>
            .main { background-color: #d4edda; }
            .stChatMessage { border-radius: 10px; padding: 10px; margin-bottom: 10px; }
            .stChatMessage.user { background-color: #d1e7fd; }
            .stChatMessage.assistant { background-color: #e0f3db; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🩺 MediBot - AI Health Assistant")
    st.write("Welcome to MediBot, your AI-powered healthcare assistant!")

    # Display example questions with color styling
    st.subheader("💡 Example Questions:")
    example_questions = [
        "What are your hospital's visiting hours?",
        "How do I book an appointment?",
        "Do you accept insurance?",
        "Where is your hospital located?",
        "Do you offer telemedicine consultations?"
    ]

    for question in example_questions:
        st.markdown(
            f"<p style='background-color:#eaf5ff; padding:8px; border-radius:5px; color:black; font-weight:bold;'>🔹 {question}</p>",
            unsafe_allow_html=True)

    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI response
        response = generate_response(user_input, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()
