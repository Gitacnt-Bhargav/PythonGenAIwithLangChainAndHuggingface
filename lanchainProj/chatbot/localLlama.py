from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

#langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant. Respond to users query"),
        ("user","Question:{question}")
    ]
)

#streamlit framework
st.title("Langchain Demo with Ollama API")
input_text = st.text_input("Search the topic you want")

#ollama LLM
llm = Ollama(model="llama2")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    with st.spinner("Generating response..."):
        response = chain.invoke({"question": input_text})
    st.write(response)