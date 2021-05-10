class RocchioAlgorithm(object):
    """description of class"""
    #################################################################################
    ## @brief   Constructor
    #  @details This method initializes the class with:
    #           corpus set of the documents to be searched
    #           relevance_judgments ID's of the relevant documents
    #################################################################################    
    def __init__(self,query,corpus,relevance_judgments,ir):
        dictionary,_ = ir.create_dictionary(corpus)
        queryVector = ir.create_query_view(query,dictionary)        
        query_mod = self.execute_rocchio(dictionary,relevance_judgments, queryVector, 1, .75, .15)
        self.new_query = self.getNewQuery(query, query_mod, dictionary)


    def getQueryVector(self,query,dictionary):
        queryVector = [0] * len(dictionary)
        for word in query.split():
            pos = dictionary[word]
            queryVector[pos] = queryVector[pos] + 1
        return queryVector	

    def execute_rocchio(self,dictionary,relevance, queryVector, alpha, beta, gamma):
        
        relDocs = [relevance[i] for i in range(len(relevance)) if relevance[i][1] > 0.1] 
        nonRelDocs = [relevance[i] for i in range(len(relevance)) if relevance[i][1] <= 0.1] # weights less than 0.1 are considered no relevant

        term1 = [alpha*i for i in queryVector]

        sumRelDocs=0
        for i in relDocs:
            sumRelDocs  += i[1]

        sumNonRelDocs=0
        for j in nonRelDocs:
            sumNonRelDocs  += j[1]

        term2 = [float(beta)/len(relDocs) * sumRelDocs]
        term3 = [-float(gamma)/len(nonRelDocs) * sumNonRelDocs]
        
        term1 = [list(i) for i in term1] # convert to a list    
                      
        pos=0   
        while pos < len(term1):
              term1[pos][1] = term1[pos][1] + term2[0] + term3[0]
              pos += 1  
        return term1

    def getKey(self,item):
        return item[1]

    def getNewQuery(self,query, queryMod, dictionary):
	    # find 2 new words with maximum value
        temp = queryMod[:]
        temp.sort(reverse = True)
        count = 0
        for element in temp:
            pos = queryMod.index(element)
            word = dictionary[pos]
            if word not in query:
               query = query + ' ' + word
               count = count + 1
               if count==2:
                   break

        return query

