import os
import pandas as pd
import matplotlib.pyplot as plt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
#função que processa a entrada e devolve a string de resposta
def input_handler(prompt,db):
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())
    chat_history = []
    result = qa({"question": prompt, "chat_history": chat_history})
    return result["answer"]

def main():
    #OpenAI key
    os.environ["OPENAI_API_KEY"] = "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"

    #faz a leitura do PDF
    reader = PdfReader("./Procuradoria.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n" #texto que acumula toda a informação

    #criação dos chunks para o modelo
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1024,
        chunk_overlap  = 24,
    )
    chunks = text_splitter.create_documents([text])

    #o embedding model é da openAI
    embeddings = OpenAIEmbeddings()

    #cria o vetor database
    db = FAISS.from_documents(chunks, embeddings)

    #similaridade entre o input e a database
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")




    st.title("Unicamp 2024 ChatBot")

    #mensagem inicial do Bot
    with st.chat_message("assistant"):
        st.write("Bem-vindo ao ChatBox de dúvidas sobre o vestibular Unicamp 2024. Fique a vontade para fazer qualquer pergunta!")

    #inicializa o chat history
    if("messages" not in st.session_state):
        st.session_state.messages = []

    #imprime as mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #cria o prompt do usuário
    prompt = st.chat_input("Qual sua dúvida?")

    #quando recebe uma pergunta
    if prompt:
        #imprime a pergunta
        with st.chat_message("user"):
            st.markdown(prompt)
        #adiciona a pergunta ao histórico
        st.session_state.messages.append({"role":"user","content":prompt})
        response = input_handler(prompt,db)
        #imprime a resposta
        with st.chat_message("assistant"):
            st.markdown(response)
        #adiciona a resposta ao histórico
        st.session_state.messages.append({"role":"assistant","content":response})

main()