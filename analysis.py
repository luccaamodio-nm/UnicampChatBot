import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy

#constroi o hisograma
def histogram(similarity_scores):
    df = pd.DataFrame({'Scores': similarity_scores})
    df.hist( )
    plt.show()
#calcula a média dos scores
def media(similarity_scores):
    print("A  média é de: ",sum(similarity_scores)/len(similarity_scores))
    
def main():
    # URL do arquivo raw da planilha no GitHub
    url_raw = "https://raw.githubusercontent.com/amodiolucca/UnicampChatBot/main/Respostas%20ChatBot.csv"

    # Lê a planilha a partir do URL
    df = pd.read_csv(url_raw)

    if 'resposta esperada' in df.columns and 'resposta dada' in df.columns:
        similarity_scores = []
        nlp = spacy.load("pt_core_news_sm") #define o idioma como português
        for index, row in df.iterrows(): #passa pelos elementos da tabela
            resposta_esperada = str(row['resposta esperada']) #pega as informações da resposta esperada
            resposta_dada = str(row['resposta dada']) #pega as informações da resposta fornecida
            vector1= nlp(resposta_esperada) #vetoriza a resposta esperada
            vector2 = nlp(resposta_dada) #vetoriza a resposta fornecida
            similarity = vector1.similarity(vector2) #calcula a similaridade entre ambas
            similarity_scores.append(similarity)
        print(similarity_scores)
        histogram(similarity_scores)
        media(similarity_scores)
    else:
        print('As colunas "resposta esperada" e "resposta dada" não foram encontradas na planilha.')
main()