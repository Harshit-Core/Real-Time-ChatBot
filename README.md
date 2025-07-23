# Smart Chatbot 🤖

A simple, interactive chatbot web app built with Python and Streamlit. The bot uses machine learning (TF-IDF vectorizer + logistic regression) to classify user messages into predefined intents and responds with relevant, sometimes witty replies.

---

## Features

- **Conversational UI:** Chat with the bot in real time.
- **Intent Recognition:** Answers greetings, questions about itself, budgeting, weather, and more.
- **Humorous Replies:** Sassy and fun responses for insults and common queries.
- **Chat History:** See your conversation history in a side-by-side chat format.
- **Easy to Deploy:** Run locally or deploy for free on [Streamlit Cloud](https://streamlit.io/cloud).

---

## Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

### 3. Run the app

```sh
streamlit run bot.py
```

---

## Deploy on Streamlit Cloud

1. Push your code to a public GitHub repository.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with GitHub.
3. Click **"New app"**, select your repo and `bot.py` as the main file, then click **Deploy**.

---

## Requirements

- Python 3.7+
- streamlit
- scikit-learn
- nltk

---

## Description

This chatbot uses a simple intent classification model to match user input to a set of predefined intents and returns a random response from the matched intent. It is designed to be lightweight, fast, and fun to interact with.

---

## Example

![screenshot](https://user-images.githubusercontent.com/your-username/your-screenshot.png)

---

## License

MIT License

---

**Built with ❤️ using
