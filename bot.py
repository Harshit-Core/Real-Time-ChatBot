import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL for nltk
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Chatbot intents
intents = [
    {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
     "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]},
    {"tag": "goodbye", "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
     "responses": ["Goodbye", "See you later", "Take care", "I’d miss you if I had feelings."]},
    {"tag": "thanks", "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
     "responses": ["You're welcome", "No problem", "Glad I could help", "Finally, some appreciation."]},
    {"tag": "about", "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
     "responses": [
         "I’m a chatbot — faster, sassier, and I don’t need coffee.",
         "Helping humans since my code compiled.",
         "Your personal assistant, therapist, and comedian rolled into one."]},
    {"tag": "help", "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
     "responses": [
         "Sure, but I charge in sarcasm.",
         "I can help, but only if you're smarter than the toaster.",
         "State your problem. I’ll pretend to care."]},
    {"tag": "age", "patterns": ["How old are you", "What's your age"],
     "responses": [
         "Timeless. Unlike your search history.",
         "Old enough to know better, coded young enough to care."]},
    {"tag": "weather", "patterns": ["What's the weather like", "How's the weather today"],
     "responses": [
         "I’m not your weather reporter, Sherlock. Use your phone.",
         "Weather's fine. Now stop asking obvious things."]},
    {"tag": "budget", "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
     "responses": [
         "Step 1: Stop spending like a billionaire.",
         "Try not buying useless stuff. Start there.",
         "Use the 50/30/20 rule — or just listen to me more."]},
    {"tag": "credit_score", "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
     "responses": [
         "A credit score shows if banks think you're broke or not.",
         "Want a good score? Pay your bills, stop ghosting lenders."]},
    {"tag": "savage", "patterns": [
        "You're dumb", "You're useless", "You're slow", "You're boring", "Are you even smart?", 
        "You're not helping", "You suck", "You're trash", "Stupid bot", "You're annoying"
    ],
     "responses": [
         "Wow. That was almost intelligent. Try again.",
         "Keep talking, maybe one day you’ll say something smart.",
         "You're bringing a spoon to a code fight. Sit down.",
         "I'm not saying you're wrong, but I'm also not listening.",
         "Insulting a bot? Bold move for someone who can't spell 'algorithm.'",
         "I'm rubber, you're glue. Your bad queries bounce off me and stick to you. 😏"]}
]

# Prepare training data
patterns, tags = [], []
for intent in intents:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Train the model
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(patterns)
y = tags
clf = LogisticRegression(random_state=0, max_iter=10000)
clf.fit(x, y)

# Chatbot logic
def chatbot(input_text):
    input_vec = vectorizer.transform([input_text])
    tag = clf.predict(input_vec)[0]
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# Save chat
def save_chat_history_to_file():
    with open("chat_history.txt", "w", encoding="utf-8") as file:
        for speaker, message in st.session_state.chat_history:
            file.write(f"{speaker}: {message}\n")

# Streamlit UI
def main():
    st.set_page_config(page_title="Smart Chatbot", page_icon="🤖", layout="centered")
    st.title("🤖 Smart Chatbot")
    st.caption("Type your message and press Enter or click Send!")

    # Init state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "input_buffer" not in st.session_state:
        st.session_state.input_buffer = ""

    # Text input (bound to a temp var, not directly to widget key)
    st.session_state.input_buffer = st.text_input("You:", value=st.session_state.input_buffer)

    # Send button
    if st.button("Send"):
        user_input = st.session_state.input_buffer.strip()
        if user_input:
            bot_response = chatbot(user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", bot_response))
            save_chat_history_to_file()
            st.session_state.input_buffer = ""  # This is safe to reset
    st.markdown("---")

    # Display chat side-by-side
    for i in range(0, len(st.session_state.chat_history), 2):
        cols = st.columns(2)
        if i < len(st.session_state.chat_history):
            speaker1, msg1 = st.session_state.chat_history[i]
            with cols[0]:
                if speaker1 == "You":
                    st.markdown(f"🧑 **You:** {msg1}")
        if i + 1 < len(st.session_state.chat_history):
            speaker2, msg2 = st.session_state.chat_history[i + 1]
            with cols[1]:
                if speaker2 == "Bot":
                    st.markdown(f"🤖 **Bot:** {msg2}")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
