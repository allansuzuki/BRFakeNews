# Rotina para criar um dataset com as noticias e suas labels no seguinte schema:
# text | subject | date | label | target
# text: texto da noticia
# subject: assunto da noticia
# date: data no formato ano-mes-dia
# label: noticia 'FAKE' ou 'FATO'
# target: 1 := FAKE, 0 := FATO

# No diretorio preprocessed/pre-processed.csv:
# dataset preprocessado com as noticias 'fake' e 'fato'
# ordenadas de acordo com o seguinte schema:
# index,label,preprocessed_news
#
# é preciso adicionar o titulo, assunto e data e ordenar as colunas.
#
# No diretorio full_texts/fake-meta-information e full_texts/true-meta-information
# temos assunto (3a linha) e data (4a linha) e mais, de acordo com a documentacao:
#
#  author
#  link
#  category
#  date of publication
#  number of tokens
#  number of words without punctuation
#  number of types
#  number of links inside the news
#  number of words in upper case
#  number of verbs
#  number of subjuntive and imperative verbs
#  number of nouns
#  number of adjectives
#  number of adverbs
#  number of modal verbs (mainly auxiliary verbs)
#  number of singular first and second personal pronouns
#  number of plural first personal pronouns
#  number of pronouns
#  pausality
#  number of characters
#  average sentence length
#  average word length
#  percentage of news with speeling errors
#  emotiveness
#  diversity

#import
import os

import numpy as np
import pandas as pd

class Dados:
    #Criar um dataset com todas as metainfos:
    def __init__(self):
        self.metainfo = None
        self.dataset = None
        self.fulldataset = None
        self.features = None
        self.labels = None
        self.logerr = np.array([])

    #funcao para conseguir os metadados das noticias
    def _get_metainfos(self):
        """Routine to generate the metainfo and store it. The schema is:
        author
     link
     category
     date of publication
     number of tokens
     number of words without punctuation
     number of types
     number of links inside the news
     number of words in upper case
     number of verbs
     number of subjuntive and imperative verbs
     number of nouns
     number of adjectives
     number of adverbs
     number of modal verbs (mainly auxiliary verbs)
     number of singular first and second personal pronouns
     number of plural first personal pronouns
     number of pronouns
     pausality
     number of characters
     average sentence length
     average word length
     percentage of news with speeling errors
     emotiveness
     diversity
     index"""

        #criar o dataframe
        colunas = ['author', 'link', 'category', 'date-of-publication', 'number-of-tokens', 'number-of-words-without-punctuation', 'number-of-types', 'number-of-links-inside-the-news', 'number-of-words-in-upper-case', 'number-of-verbs', 'number-of-subjuntive-and-imperative-verbs', 'number-of-nouns', 'number-of-adjectives', 'number-of-adverbs', 'number-of-modal-verbs-(mainly-auxiliary-verbs)', 'number-of-singular-first-and-second-personal-pronouns', 'number-of-plural-first-personal-pronouns', 'number-of-pronouns', 'pausality', 'number-of-characters', 'average-sentence-length', 'average-word-length', 'percentage-of-news-with-speeling-errors', 'emotiveness', 'diversity','index']
        lmetas = []  #list a ser populado

        _replacespace = lambda x:x.replace(' ','-')  #funcao que troca todos os espacos por - (SOMENTE PARA O NOME DAS COLUNAS)
        cont = 0

        #Primeiro pegar todas as fakenews e depois as verdadeiras
        labels = ['fake','true']
        logerr = []
        for label in labels:
            dir2list = f'BRFakeCorpus/full_texts/{label}-meta-information'  #set diretorio para listar os arquivos
            for metatxt in os.listdir(dir2list):  # loop em todos os txt
                with open(os.path.join(dir2list,metatxt),'r',encoding='utf8') as fi:
                    lines = fi.readlines()
                    lines.append(str(cont))  #add o index nos dados
                    if len(lines) == len(colunas):
                        lmetas.append(list(map(str.strip,lines)))  #salvar os dados na lista
                    else:
                        logerr.append(f'diff num de campos do df: {dir2list}/{metatxt}')  #salva o erro
                        print(f'Erro: qtd dados em {dir2list}/{metatxt} apresenta problemas')
                    cont+=1  #
        #Salvar tudo no dataframe
        pdmetas = pd.DataFrame(lmetas,columns=colunas)

        #converter o index em numero
        pdmetas['index'] = pdmetas['index'].astype(int)

        self.metainfo = pdmetas
        self.logerr = np.append(self.logerr,logerr).flatten()

    #funcao para unir os textos das noticias com os metadados
    def _build_dataset(self):
        """"
        Unir as noticias com os metadados específicos
        """
        #Get metadados
        metainfo = self.metainfo

        #Get noticias
        csv2read = os.path.join(os.getcwd(),'BRFakeCorpus/preprocessed/pre-processed.csv')
        preproc_news = pd.read_csv(csv2read, sep=',', encoding='utf-8-sig')
        preproc_news['index'] = preproc_news['index'].astype(int)

        #Unir metadados['index','category'] e preproc_news
        cols2select = ['index', 'category']
        fulldataset = preproc_news.merge(metainfo, how='left', on='index')

        #criar um campo target
        fulldataset['target'] = [1 if lab=='fake' else 0 for lab in fulldataset['label']]

        #extrair os dados simplificados
        dataset = fulldataset[['index','preprocessed_news','category','label','target']]

        self.fulldataset = fulldataset
        self.dataset = dataset

    #funcao que carrega o dataset e separa os dados em feature e labels
    def load_dataset(self):
        print('loading dataset...')
        self._get_metainfos()
        self._build_dataset()
        self.features = self.dataset.drop(['label','target'],axis=1)
        self.labels = self.dataset[['label','target']]
        print('Dataset loaded!')

    def save_dataset(self):
        #se não existe o path, cria
        if not os.path.exists('data'): os.makedirs('data')

        self.metainfo.to_csv('data/metainfo.csv',sep=',',encoding='utf-8-sig',index=False)
        self.fulldataset.to_csv('data/fulldataset.csv',sep=',',encoding='utf-8-sig',index=False)
        self.dataset.to_csv('data/dataset.csv',sep=',',encoding='utf-8-sig',index=False)
        self.features.to_csv('data/features.csv',sep=',',encoding='utf-8-sig',index=False)
        self.labels.to_csv('data/labels.csv',sep=',',encoding='utf-8-sig',index=False)

