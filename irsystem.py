from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words
import json

import numpy as np
import math

class IrSystem:

    def __init__(self, alpha, dataset):

        self.alpha = alpha  
        self.searched = {}
        
        if dataset == '1':  
            with open('datasets\CRAN.ALL.json') as data:    
                self.dataset = json.load(data)
        else:
            with open('datasets\CISI.ALL.json') as data:    
                self.dataset = json.load(data)        

        self.data = {}

        for doc in self.dataset.values():
            self.data[doc['id']] = {
                'id' : doc['id'],
                'title' : word_tokenize(str(self.preprocess(doc['title']))) if 'title' in doc.keys() else '',
                'text' : word_tokenize(str(self.preprocess(doc['text']))) if 'text' in doc.keys() else ''
                }

        self.N = len(self.data)
        self.__df()
        self.__tf_idf()
            
    @staticmethod
    def __convert_lower_case(data):
        return np.char.lower(data)

    @staticmethod
    def __remove_stop_words(data):
        stop_words = stopwords.words('english')
        words = word_tokenize(str(data))
        new_text = ""
        for w in words:
            if w not in stop_words and len(w) > 1:
                new_text = new_text + " " + w
        return new_text

    @staticmethod
    def __remove_punctuation(data):
        symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
        for i in range(len(symbols)):
            data = np.char.replace(data, symbols[i], ' ')
            data = np.char.replace(data, "  ", " ")
        data = np.char.replace(data, ',', '')
        return data

    @staticmethod
    def __remove_apostrophe(data):
        return np.char.replace(data, "'", "")

    @staticmethod
    def __stemming(data):
        stemmer= PorterStemmer()
        
        tokens = word_tokenize(str(data))
        new_text = ""
        for w in tokens:
            new_text = new_text + " " + stemmer.stem(w)
        return new_text

    @staticmethod
    def __convert_numbers(data):
        tokens = word_tokenize(str(data))
        new_text = ""
        for w in tokens:
            try:
                w = num2words(int(w))
            except:
                a = 0
            new_text = new_text + " " + w
        new_text = np.char.replace(new_text, "-", " ")
        return new_text

    def doc_freq(self, word):
        c = 0
        try:
            c = self.word_frequency[word]
        except:
            pass
        return c

    def preprocess(self, data):
        data = IrSystem.__convert_lower_case(data)
        data = IrSystem.__remove_punctuation(data) #remove comma seperately
        data = IrSystem.__remove_apostrophe(data)
        data = IrSystem.__remove_stop_words(data)
        data = IrSystem.__convert_numbers(data)
        data = IrSystem.__stemming(data)
        data = IrSystem.__remove_punctuation(data)
        data = IrSystem.__convert_numbers(data)
        data = IrSystem.__stemming(data) #needed again as we need to stem the words
        data = IrSystem.__remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
        data = IrSystem.__remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
        return data

    def __df(self):
        self.word_frequency = {}

        for doc in self.data.values():
            for w in doc['title']:
                try:
                    self.word_frequency[w].add(int(doc['id']))
                except:
                    self.word_frequency[w] = {int(doc['id'])}

            for w in doc['text']:
                try:
                    self.word_frequency[w].add(int(doc['id']))
                except:
                    self.word_frequency[w] = {int(doc['id'])}

        for i in self.word_frequency:
            self.word_frequency[i] = len(self.word_frequency[i])
        
        self.total_vocab = [x for x in self.word_frequency]
        self.total_vocab_size = len(self.total_vocab)

    def __tf_idf(self):
        
        self.tf_idf = np.zeros((self.N, self.total_vocab_size))

        tf_idf = {}
        tf_idf_title = {}

        for doc in self.data.values():
                  
            counter = Counter(doc['text'])
            words_count = len(doc['text'])

            counter_title = Counter(doc['title'] + doc['text'])
            words_count_title = len(doc['title'] + doc['text'])
            
            for token in np.unique(doc['text']):
                
                tf = counter[token]/words_count
                df = self.doc_freq(token)
                idf = np.log((self.N+1)/(df+1))
                
                tf_idf[int(doc['id']), token] = tf*idf

                tf_title = counter_title[token]/words_count_title
                df_title = self.doc_freq(token)
                idf_title = np.log((self.N+1)/(df_title+1))
                
                tf_idf_title[int(doc['id']), token] = tf_title*idf_title

        for i in tf_idf:
            tf_idf[i] *= self.alpha
        
        for i in tf_idf_title:
            tf_idf[i] = tf_idf_title[i]

        
        for i in tf_idf:
            try:
                ind = self.total_vocab.index(i[1])
                self.tf_idf[i[0]][ind] = tf_idf[i]
            except:
                pass

            
    def __gen_query_vector(self, tokens):

        Q = np.zeros(self.total_vocab_size)
        
        counter = Counter(tokens)
        words_count = len(tokens)
        
        for token in np.unique(tokens):
            
            tf = counter[token]/words_count
            df = self.doc_freq(token)
            idf = math.log((self.N+1)/(df+1))

            try:
                ind = self.total_vocab.index(token)
                Q[ind] = tf*idf
            except:
                pass
        return Q

    def search(self, k, preview, query):
        print("\n---------- Ejecutando búsqueda -----------")
        
        if query in self.searched.keys():
            self.__print_search(self.searched[query], preview)
        
            return self.searched
        
        else:
            preprocessed_query = self.preprocess(query)
            tokens = word_tokenize(str(preprocessed_query))
        
            d_cosines = []
            
            query_vector = self.__gen_query_vector(tokens)
            
            for d in self.tf_idf:
                d_cosines.append(IrSystem.__cosine_sim(query_vector, d))
                
            self.searched[query] = np.array(d_cosines).argsort()[-k:][::-1]

            self.__print_search(self.searched[query], preview)

    @staticmethod
    def __cosine_sim(a, b):
        return 0 if not a.max() or not b.max() else np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def __print_search(self, out, preview):
        for doc in out:
            print(f"{doc} - { self.dataset[str(doc)]['title'] }\n\t{self.dataset[str(doc)]['text'][:preview]}")
            print()


if __name__ == '__main__':
    dataset = input('Elige un Dataset: \n1 - Cranfield \n2 - CISI \n-> ')
    irsystem = IrSystem(0.3, dataset)
    
    query = input("Escribe una consulta: ")
    irsystem.search(5, 500, query)
    
    while True:
        print('\nOpciones:')
        mode = input(f"1 - Hacer una nueva consulta \n2 - Aplicar Retroalimentación de Rocchio a una consulta \n3 - Analizar Rendimiento de una consulta \nEnter - Para terminar\n-> ")
        if mode == '1':
            query = input("\nEscribe una consulta: ")
            irsystem.search(5, 500, query)
        elif mode == '2':
            print('\nOpciones:')
            ask = [f'{i} - {query}\n' for i, query in enumerate(irsystem.searched.keys())]
            query = input("".join(ask) + '-> ')
            pass
        elif mode == '3':
            print('\nOpciones:')
            ask = [f'{i} - {query}\n' for i, query in enumerate(irsystem.searched.keys())]
            query = input("".join(ask) + '-> ')
            pass
        else:
            break

