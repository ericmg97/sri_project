###################################################################################
## @file      information_retrieval_system.py
#  @brief     The information_retrieval_system.py is a basic information retrieval system  
#             implemented using Python, NLTK and GenSIM.
#  @authors   Yolanda de la Hoz Simon
###################################################################################
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora, models, similarities
from operator import itemgetter
import abc
import re
import numpy as np
###################################################################################
## @class   InformationRetrievalSystem
#  @brief   This class represents the InformationRetrievalSystem, i.e., basic methods 
#           used to preprocess and rank documents according to user queries.
###################################################################################
class IRSystem(object):
    
    def __init__(self, corpus, queries):
        self.ranking_query = []
        self.corpus=corpus
        self.queries=queries
  
    def preprocess_document(self,doc):
        stopset = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = wordpunct_tokenize(doc) # split text on whitespace and punctuation
        clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
        final = [stemmer.stem(word) for word in clean]
        return final

    def create_dictionary(self,docs):
        pdocs = [self.preprocess_document(doc) for doc in docs]
        dictionary = corpora.Dictionary(pdocs)
        dictionary.save('vsm.dict')
        return dictionary,pdocs
  
    def get_keyword_to_id_mapping(self,dictionary):
        print(dictionary.token2id)

    def docs2bows(self,corpus, dictionary, pdocs):
        vectors = [dictionary.doc2bow(doc) for doc in pdocs]
        corpora.MmCorpus.serialize('vsm_docs.mm', vectors) # Save the corpus in the Matrix Market format
        return vectors

    def ranking_function(self,corpus, q, query_id, mode):
        model, dictionary = self.create_documents_view(corpus, mode)
        loaded_corpus = corpora.MmCorpus('vsm_docs.mm')
        index = similarities.MatrixSimilarity(loaded_corpus, num_features=len(dictionary))
        vq=self.create_query_view(q,dictionary)
        if (mode == 1):
           self.query_weight  = [(w[0], 1 + np.log2(w[1])) for w in vq]
        else:
            self.query_weight = model[vq]
        sim = index[self.query_weight]
        ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
        self.ranking_query[query_id]=ranking # store the ranking of the query in a dict
        for doc, score in ranking:
            print ("[ Score = " + "%.3f" % round(score, 3) + "] " + corpus[doc])
      
      
    def create_query_view(self,query,dictionary):
        pq = self.preprocess_document(query)
        vq = dictionary.doc2bow(pq)
        return vq


    def create_documents_view(self,corpus, ir_mode):
        dictionary,pdocs = self.create_dictionary(corpus)
        bow = self.docs2bows(corpus, dictionary, pdocs)     
        loaded_corpus = corpora.MmCorpus('vsm_docs.mm') # Recover the corpus

        if ir_mode == 1:
             model = [[(w[0], 1 + np.log2(w[1])) for w in v] for v in bow] # TF model
        elif ir_mode == 2:
             model = models.TfidfModel(loaded_corpus) # TF IDF model
        
        return model, dictionary

    #################################################################################
    ## @brief   launch_query
    #  @details This method initializes the class with the parameters introduced by the user
    #           and execute the query. 
    #  @param   corpus Set of documents to be processed.
    #  @param   q Query, a document with the set of relevance words to the user.
    #################################################################################   
    def query_launcher(self,corpus, queries, mode):
        query_id=0
        if isinstance(queries, list): # launch queries
           for q in queries:
               print("\n-------------------------->Query = " + q ) 
               self.ranking_function(corpus,q,query_id,mode)
               query_id += 1
             
        else:
            print("\n-------------------------->Query = " + queries ) 
            self.ranking_function(corpus,queries,1,mode)
        return



class IRBoolean(IRSystem):

    def __init__(self,corpus,queries):
        IRSystem.__init__(self,corpus,queries)
        print("\n--------------------------Ejecutando el MRI Booleano--------------------------\n")
        self.ranking_query=dict()

        query_id=0
        if isinstance(queries, list): # launch queries
           for q in queries:
               print("\n-------------------------->Query = " + q ) 
               or_set,and_set = self.preprocess_query(q)
               dict_matches = self.process_operators(corpus,or_set,and_set,query_id)
               self.print_result(dict_matches)
               query_id += 1
        else:
             print("\n-------------------------->Query = " + queries ) 
             or_set,and_set = self.preprocess_query(queries)
             dict_matches = self.process_operators(corpus,or_set,and_set,1)
             self.print_result(dict_matches)

    def process_operators(self,corpus,or_set,and_set,query_id):   
        or_list = [val for sublist in or_set for val in sublist]     
        for or_txt in or_list: # assign score 1 to documents that match with either phrase with or
            dict_matches = self.document_matches(corpus,or_txt)
        if len(and_set) > 0: 
          and_list = [val for sublist in and_set for val in sublist]
          and_txt= ', '.join(and_list) # treat the and_set as a single query separated by commas
          dict_matches =  self.document_matches(corpus,and_txt)
        self.ranking_query[query_id]=dict_matches.items()
        return dict_matches

    def preprocess_corpus(self,corpus):
        dictionary,pdocs = self.create_dictionary(corpus)
        return dictionary, pdocs

    def preprocess_query(self,q):
        text=re.split(r'[^\w\s]',q) # detection of final of the OR operator, stop punctuation 
        or_set=[]
        and_set=[]
        for phrase in text:
            txt = re.split("or",phrase)
            if(len(txt)>1): # there are OR operators
               or_set.append(txt) 
            else: 
               and_set.append(txt) # it is an AND operator
        return or_set,and_set
        
    def document_matches(self,corpus, q):
        dictionary,pdocs = self.preprocess_corpus(corpus)
        vq = self.preprocess_document(q) # preprocess query
        dict_matches=dict((doc,0) for doc in dictionary) # Create a dictionary with documents and initial value score 0
        doc_number = 0 
        for doc in pdocs:
            intersection_list = list(set(doc) & set(vq)) 
            if len(intersection_list)==len(vq): # All terms are contained in the doc
               dict_matches[corpus[doc_number]]=1       
            doc_number += 1     
        return dict_matches
      
    def print_result(self,dict_matches):
        for keys,values in dict_matches.items():
            print("[ Score = " + str(values) + "] ")
            print("Document = " + str(keys))
          
################################################ Modelos en la Libreria Gensim ################################################

class IR_tf(IRSystem):

 def __init__(self,corpus,queries):
        IRSystem.__init__(self,corpus,queries)
        print("\n--------------------------Ejecutando el MRI TF--------------------------\n")
        self.ranking_query=dict()
        self.query_launcher(corpus,queries,1)


class IR_tf_idf(IRSystem):

    def __init__(self,corpus,queries):
        IRSystem.__init__(self,corpus,queries)
        print("\n--------------------------Ejecutando el MRI TF IDF--------------------------\n")
        self.ranking_query=dict()
        self.query_launcher(corpus,queries,2)        
