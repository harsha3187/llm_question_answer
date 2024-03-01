import os
import sys
import datetime
import openai
import dotenv
import streamlit as st
from custom_qa_generation import *
import os
from audio_recorder_streamlit import audio_recorder

# import API key from .env file
dotenv.load_dotenv()
openai.api_key = os.getenv("OPEN_AI_KEY")
genre = st.radio(
    "Select your language",
    ["English", "Arabic"],
    captions = ["", ""],horizontal=True)
prompt = st.text_input("Please enter your question", value="", key="prompt")

qa_model = qa_llm()

def ret_openai_response(prompt):
    translate_response = openai.completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens = 350,
            temperature=0.3
        )
    return translate_response

if prompt!="":
    response = ""
    if genre == 'English':
        response = qa_model.generate_text(prompt)
    else:
        prompt = f"translate this text to english: {prompt}"
        translate_response = ret_openai_response(prompt)
        #response in english
        response = qa_model.generate_text(translate_response)
        #response in arabic
        prompt = f"translate this text to arabic: {response}"
        response = ret_openai_response(prompt)
    st.markdown(response)
