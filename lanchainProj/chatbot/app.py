from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

#langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

if os.environ["OPENAI_API_KEY"]:
    print("OPENAI_API_KEY loaded successfully")
else:
    print("OPENAI_API_KEY is not set. Add it to the .env file.")


#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant. Respond to users query"),
        ("user","Question:{question}")
    ]
)

#streamlit framework
st.title("Langchain Demo with OpenAI API")
input_text = st.text_input("Search the topic you want")

#openAi LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text})) 