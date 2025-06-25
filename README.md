# 🧠 LLM Chatbot with Hugging Face & Streamlit

A simple interactive chatbot app built using [LangChain](https://www.langchain.com/), [Hugging Face](https://huggingface.co/), and [Streamlit](https://streamlit.io/).  
It uses the `mistralai/Mistral-7B-Instruct-v0.2` model via the `featherless-ai` provider.

---

## 🚀 Features

- Natural language chat interface
- Powered by Hugging Face models via LangChain
- Easily extensible and secure
- Lightweight Streamlit UI

---

## 🧩 How to Get Your Hugging Face API Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Set a name like `langchain-chatbot`, choose `read` access, and create it
4. Copy the token — you'll use it in your `.env` file

---

## 🔐 Environment Variables

Create a `.env` file in the root of your project:

```bash
touch .env
