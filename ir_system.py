from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import matplotlib.pyplot as plot
from numpy.lib.function_base import average
from scipy.sparse import data, lil_matrix, csr_matrix
import json

import numpy as np
import math

class IrSystem:

    def __init__(self, alpha, dataset):

        self.alpha = alpha  
        self.searched = {}
        
        if dataset == '1':  
            with open('datasets\Cranfield\CRAN.ALL.json') as data:    
                self.dataset = json.load(data)
            
            with open('datasets\Cranfield\CRAN.QRY.json') as data:    
                self.querys = json.load(data)

            with open('datasets\Cranfield\CRAN.REL.json') as data:    
                self.rel = json.load(data)
        elif dataset == '2':
            with open('datasets\Med\MED.ALL.json') as data:    
                self.dataset = json.load(data)  

            with open('datasets\Med\MED.QRY.json') as data:    
                self.querys = json.load(data)      
            
            with open('datasets\Med\MED.REL.json') as data:    
                self.rel = json.load(data)    
        else:
            return

        self.data = {}
        self.relevant_docs = int(average([len(queries.values()) for queries in self.rel.values()]))

        for doc in self.dataset.values():
            self.data[doc['id']] = {
                'id' : doc['id'],
                'title' : word_tokenize(str(self.__preprocess(doc['title']))) if 'title' in doc.keys() else [],
                'text' : word_tokenize(str(self.__preprocess(doc['abstract']))) if 'abstract' in doc.keys() else []
                }

        self.N = len(self.data)
        self.__df()
        self.__tf_idf()

        for query in self.querys.values():
            self.search(query['text'], query_id = query['id'])
            
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

    def __doc_freq(self, word):
        c = 0
        try:
            c = self.word_frequency[word]
        except:
            pass
        return c

    def __preprocess(self, data):
        data = IrSystem.__convert_lower_case(data)
        data = IrSystem.__remove_punctuation(data) #remove comma seperately
        data = IrSystem.__remove_apostrophe(data)
        data = IrSystem.__remove_stop_words(data)
        data = IrSystem.__stemming(data)
        data = IrSystem.__remove_punctuation(data)
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
        
        self.tf_idf = lil_matrix((self.N + 1, self.total_vocab_size))

        tf_idf = {}
        tf_idf_title = {}

        for doc in self.data.values():
                  
            counter = Counter(doc['text'])
            words_count = len(doc['text'])

            counter_title = Counter(doc['title'] + doc['text'])
            words_count_title = len(doc['title'] + doc['text'])
            
            for token in np.unique(doc['text']):
                
                tf = counter[token]/words_count
                df = self.__doc_freq(token)
                idf = np.log((self.N+1)/(df+1))
                
                tf_idf[int(doc['id']), token] = tf*idf

                tf_title = counter_title[token]/words_count_title
                df_title = self.__doc_freq(token)
                idf_title = np.log((self.N+1)/(df_title+1))
                
                tf_idf_title[int(doc['id']), token] = tf_title*idf_title

        for i in tf_idf:
            tf_idf[i] *= self.alpha
        
        for i in tf_idf_title:
            tf_idf[i] = tf_idf_title[i]

        
        for i in tf_idf:
            ind = self.total_vocab.index(i[1])
            self.tf_idf[i[0], ind] = tf_idf[i]
        
        self.tf_idf = csr_matrix(self.tf_idf)

            
    def __gen_query_vector(self, tokens, alpha):

        Q = lil_matrix((1, self.total_vocab_size))
        
        counter = Counter(tokens)
        words_count = len(tokens)
        
        for token in np.unique(tokens):
            
            tf = alpha + (1 - alpha) * (counter[token] / words_count)
            df = self.__doc_freq(token)
            if df:
                idf = math.log((self.N)/(df))
            else:
                idf = 0
            
            try:
                ind = self.total_vocab.index(token)
                Q[0, ind] = tf*idf
            except:
                pass
           
        return csr_matrix(Q)

    @staticmethod
    def __cosine_sim(a, b):
        return 0 if not a.max() or not b.max() else a.dot(b.transpose())/(IrSystem.__sparse_row_norm(a)*IrSystem.__sparse_row_norm(b))

    @staticmethod
    def __sparse_row_norm(A):
        out = np.zeros(A.shape[0])
        nz, = np.diff(A.indptr).nonzero()
        out[nz] = np.sqrt(np.add.reduceat(np.square(A.data),A.indptr[nz]))
        return out


    def __print_search(self, out, preview):
        for doc in out:
            print(f"{doc[0]} - { self.dataset[str(doc[0])]['title'] if self.dataset[str(doc[0])]['title'] != '' else 'Not Title'}\nText: {self.dataset[str(doc[0])]['abstract'][:preview]}")
            print()

    def search(self, query, alpha = 0.4, query_id = False, k = -1, preview = 500):
        if query_id and query_id in self.searched.keys():
            self.__print_search(self.searched[query_id][1], preview)
            return
        
        preprocessed_query = self.__preprocess(query)
            
        if (not query_id):
            print("\n---------- Ejecutando Búsqueda -----------\n")
        
        tokens = word_tokenize(str(preprocessed_query))
    
        d_cosines = []
        
        query_vector = self.__gen_query_vector(tokens, float(alpha))
        
        for d in self.tf_idf:
            d_cosines.append(IrSystem.__cosine_sim(d, query_vector))

        out = [(id, d_cosines[id].max()) for id in np.array(d_cosines).argsort()[-k:][::-1] if d_cosines[id] and d_cosines[id].max() > 0.0]

        if query_id:
            self.searched[query_id] = (query, out)
        else:
            self.__print_search(out[:self.relevant_docs], preview)

    def evaluate_query(self,query_id, show_output):
        if str(query_id) not in self.searched.keys():
            print("Consulta no encontrada")
            return

        if (show_output):
            print("\nConsulta: " + self.searched[str(query_id)][0]) 

        self.__evaluate(self.searched[query_id][1],self.rel[str(query_id)])

    def relevant_doc_retrieved(self, ranking, relevants_docs_query):
        true_positives = 0
        false_positives = 0
        for doc in ranking[:self.relevant_docs]:
           if str(doc[0]) in relevants_docs_query.keys():
                true_positives += 1
           else:
                false_positives += 1
        return true_positives,false_positives

    @staticmethod
    def __get_recall(true_positives,real_true_positives):
        recall=float(true_positives)/float(real_true_positives)
        return recall
    
    @staticmethod
    def __get_precision(true_positives,false_positives):
        relevant_items_retrieved=true_positives+false_positives
        precision=float(true_positives)/float(relevant_items_retrieved)
        return precision

    @staticmethod
    def __interpolate_precisions(recalls,precisions, recalls_levels):
        precisions_interpolated = np.zeros((len(recalls), len(recalls_levels)))
        i = 0
        while i < len(precisions):
            # use the max precision obtained for the topic for any actual recall level greater than or equal the recall_levels
            recalls_inter = np.where((recalls[i] > recalls_levels) == True)[0]
            for recall_id in recalls_inter:
                if precisions[i] > precisions_interpolated[i, recall_id]:
                    precisions_interpolated[i, recall_id] = precisions[i]
            i += 1

        mean_interpolated_precisions = np.mean(precisions_interpolated, axis=0)
        return mean_interpolated_precisions

    @staticmethod
    def __plot_results(recall, precision):
        plot.plot(recall, precision)
        plot.xlabel('Recobrado')
        plot.ylabel('Precisión')       
        plot.draw()
        plot.title('P/R')
        plot.show()

    def __evaluate(self,ranking,relevants_docs_query):
        
        [true_positives, false_positives] = self.relevant_doc_retrieved(ranking, relevants_docs_query)

        recall = IrSystem.__get_recall(true_positives,len(relevants_docs_query))
        precision = IrSystem.__get_precision(true_positives,false_positives)
        f1 = 2 / (1/precision + 1/recall)

        print(f"Precisión: {precision} \nRecobrado: {recall} \nMedida F1: {f1}")

        true_positives = 0
        false_positives = 0
        recall = []
        precision = []
        for doc in ranking:
            if str(doc[0]) in relevants_docs_query.keys():
                true_positives += 1
            else:
                false_positives += 1

            recall.append(self.__get_recall(true_positives,len(relevants_docs_query)))
            precision.append(self.__get_precision(true_positives,false_positives))


        recalls_levels = np.array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ]) 

        interpolated_precisions = self.__interpolate_precisions(recall,precision,recalls_levels)
        self.__plot_results(recalls_levels, interpolated_precisions)

if __name__ == '__main__':
    dataset = input('Elige un Dataset: \n1 - Cranfield \n2 - MED \nEnter - Para terminar\n-> ')
    if dataset == '1' or dataset == '2':
        irsystem = IrSystem(0.3, dataset)
    
        while True:
            print('\nOpciones:')
            mode = input(f"1 - Hacer una Consulta \n2 - Aplicar Retroalimentación de Rocchio a una Consulta \n3 - Analizar Rendimiento del Sistema \nEnter - Para terminar\n-> ")
            if mode == '1':
                query = input("\nEscribe una consulta: ")
                alpha = input("Escribe la Constante de Suavizado: ")
                irsystem.search(query, alpha)
            elif mode == '2':
                print('\nAplicar Retroalimentación de Rocchio a:')
                ask = [f'{query[0]} - {query[1][0]}\n' for query in enumerate(irsystem.searched.items())]
                query = input("".join(ask) + 'Elegir ID -> ')
                pass
            elif mode == '3':
                print("\n---------- Análisis del SRI -----------\n")
                while True:
                    mode = input(f"1 - Análisis General \n2 - Análisis de una Consulta \nEnter - Atrás\n-> ")
                    if mode == '1':
                        pass
                    elif mode == '2':
                        print('\nOpciones:')
                        ask = [f'{query[0]} - {query[1][0]}\n' for query in irsystem.searched.items()]
                        query = input("".join(ask) + 'Elegir ID -> ')
                        irsystem.evaluate_query(query, True)
                    else:
                        break
            else:
                break
