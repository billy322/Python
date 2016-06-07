import sys,logging

from six import iteritems, itervalues, string_types
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray
from gensim import utils, matutils
from sklearn.metrics.pairwise import cosine_similarity
import numpy
#from gensim.models.word2vec import Word2Vec
from word2vec import Word2Vec
import io

def writeList2File(fname, resList):
    with io.open(fname, 'a', encoding='utf-8') as f:
        result = ' '.join(item for item in resList)
        f.write(result + u'\n')
        
def most_similar(self, positive=[], negative=[], topn=10, restrict_vocab=None):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words and the vectors for each word in the model.
        The method corresponds to the `word-analogy` and `distance` scripts in the original
        word2vec implementation.

        If topn is False, most_similar returns the vector of similarity scores.

        `restrict_vocab` is an optional integer which limits the range of vectors which
        are searched for most-similar values. For example, restrict_vocab=10000 would
        only check the first 10000 word vectors in the vocabulary order. (This may be
        meaningful if you've sorted the vocabulary by descending frequency.)

        Example::

          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]

        """
        #logger = logging
        if not topn: 
            topN=10  # force to return top10 for direction matching 
        else:
            topN=topn 
        
        self.init_sims()
        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in positive
        ]
        negative = [
            (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in negative
        ]
        # Retrivel the index, vector for b - a + c
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            elif word in self.vocab:
                #print word, weight ,len(self.syn0norm[self.vocab[word].index])
                mean.append(weight * self.syn0norm[self.vocab[word].index])    # mean =  algeria 1.0 vec, algiers -1.0 vec
                all_words.add(self.vocab[word].index) # the assessment item (a, b, c)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")

        # compute the weighted average of all words
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL) # normalise vector and find mean of 3 vector's distance (not sum!)
        limited = self.syn0norm if restrict_vocab is None else self.syn0norm[:restrict_vocab]  # store vector for the whole w2v model

        # compute sim(d,b-a+c) where d refer to the whole model
        dists = dot(limited, mean)
        
        # since we need top10 result for further processing 
#         if not topn:
#             return dists
        #for some reason, it only limit for 3 words
        #best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True) 
        best = matutils.argsort(dists, topn=topN , reverse=True) #dists contain sim score for whole w2v, get index from dists = word index 
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        writeList2File("distance.txt", [self.index2word[sim] for sim in best if sim not in all_words])
        logger.debug(" ") 
        logger.debug("Distance: %s", result) 
        #print result
        # restore a, c : b, d
#         a = negative[0][0]
#         b = positive[0][0]
#         d_c = [item[0] for item in result] + [positive[1][0]] # c is the last item 
        a = negative[0][0]
        b = positive[1][0]
        c = positive[0][0]
        d = [item[0] for item in result]
        fullVariable = [a] + [b] + [c] + d
        #d_b = d + [b] # c is the last item 
        d_c = d +[c]
        if [positive[1][0]] in [item[0] for item in result]:
            logger.debug("Contain: %s in %s", [positive[1][0]], \
                         [item[0] for item in result])
        # After checking direction, return best result 
        result_dir = most_similar_direction(self, positive = d_c, negative=[b, a], fullVariable=fullVariable, topn=len(result), vecModel=limited)
        #result_dir = most_similar_cosmul_direction(self, positive = d_c, negative=[b, a], topn=len(result), fullVariable=fullVariable, vecModel=limited)
        if result_dir is None:
            result_dir_sort = result
            if not topn:
                return dists
        else: 
            # make sure have correct index (based on the whole model) to be displayed 
            result_dir_sort =  matutils.argsort(result_dir, topn=topn + len(all_words), reverse=True)
            if not topn:
                return result_dir
        
        return result_dir_sort[:topn]

def most_similar_direction(self, positive=[], negative=[], fullVariable=[], topn=False, vecModel=None):
        """
        this compute pair Direction sim(d-b, c-a)

        """
        self.init_sims()
        #logger = logging
        #logger.debug("\t\t========direction===========")  
        #logger.debug("d, c: %s", positive)  
        #logger.debug("b, a: %s", negative)  
  
        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        WordDict = {}
        for word in positive: 
            if isinstance(word, string_types + (ndarray,)): # plus mean or type
                WordDict[word] = 1.0
        for word in negative:
            if isinstance(word, string_types + (ndarray,)):
                WordDict[word] = -1.0 

        all_words = set() # store question words
        #limited = [] # store d (i.e. topN words having closest distance with b - a + c from most_similiar)
        
        def getData(word):
            return WordDict[word], self.vocab[word].index, self.syn0norm[self.vocab[word].index] 
    
 
        wordD_list =[]
        for i, item in enumerate(fullVariable):
            if i == 0:
                wordA = fullVariable[i]
                wordA_weight, wordAIndex, vecA = getData(wordA)
                all_words.add(wordAIndex)
            elif i == 1:
                wordB = fullVariable[i]
                wordB_weight, wordBIndex, vecB = getData(wordB)
                all_words.add(wordBIndex)
            elif i == 2:
                wordC = fullVariable[i]
                wordC_weight, wordCIndex, vecC = getData(wordC)
                all_words.add(wordCIndex)
            else: 
                wordD_list.append(fullVariable[i])
        #print wordA, wordB, wordC
        # add assessment item  Validated version 
         ##get all (d - b) . checked 
        tempLimit = numpy.zeros(shape=(vecModel.shape)) #Quick Fix. Fake array to fit into Gensim format
        for word in wordD_list: # these are the predict ans (d) and b
              if isinstance(word, ndarray):
                  sys.exit() # not yet fixed
              elif word in self.vocab:
                  wordD_weight, wordDIndex, vecD = getData(word)
                  vecD_weight = wordD_weight * vecD
                  vecB_weight = wordB_weight * vecB
                  tempLimit[wordDIndex] = matutils.unitvec(array([ vecD_weight, vecB_weight ]).mean(axis=0)).astype(REAL)
              else:
                  raise KeyError("word '%s' not in vocabulary" % word)
                   
          # compute c - a 
        if isinstance(wordC, ndarray) and isinstance(wordA, ndarray): 
              sys.exit() # not yet fixed
        elif wordC in self.vocab and wordA in self.vocab:
             vecA_weight = wordA_weight * vecA 
             vecC_weight = wordC_weight * vecC
             mean = matutils.unitvec(array([ vecC_weight, vecA_weight ]).mean(axis=0)).astype(REAL)  # mean of c - a    
        else:
              raise KeyError("word '%s' not in vocabulary" % word) 
        dists1 = dot(tempLimit, mean)  # all row are zero, except those top10 word row  
        resIdx = numpy.nonzero(dists1)[0].tolist()
        vecDict = {}
        for i in resIdx:
            vecDict[self.index2word[i]] = dists1[i]
        from collections import OrderedDict
        sorted_vecDict = OrderedDict(sorted(vecDict.items(), key=lambda t: t[1], reverse=True))

        
        # debug version :        
#         tempLimit1 = numpy.zeros(shape=(vecModel.shape[0])) #Quick Fix. Fake array to fit into Gensim format
#         scoreSection= dict()
#         u = (vecC - vecA ) / (numpy.linalg.norm(vecC - vecA))
#         #print wordA, wordC, u
#         with open("temp.txt", "ab") as text_file:
#             text_file.write(wordC +"-"+ wordA +" "+ " ".join(str(item) for item in u) +"\n")
#             text_file.close()
#         B_C = dot(vecB,vecC)
#         B_A = dot(vecB,vecA)
#         for word in wordD_list: # these are the predict ans (d) and b
#              if isinstance(word, ndarray):
# #                  vecA = wordA
#                  #limited.append(matutils.unitvec(array([ vecD, vecB ]).mean(axis=0)).astype(REAL))    # mean =  algeria 1.0 vec, algiers -1.0 vec
#                  sys.exit() # not yet fixed
#              elif word in self.vocab:
#                  wordD_weight, wordDIndex, vecD = getData(word)
#                  #print wordD_weight, wordDIndex, vecD[1:4] , word
#                  D_C = dot(vecD,vecC)
#                  D_A = dot(vecD,vecA)
#                  D_B = dot(vecD,vecB)
#                  C_A = dot(vecC,vecA)
#                  totalScore =  (D_C - D_A - B_C + B_A) / (numpy.linalg.norm(vecD - vecB) * numpy.linalg.norm(vecC - vecA))
#                  totalScoreText = str(D_C) +" "+ str(D_B)+ " "+ str(D_A) +" "+\
#                   str(B_C) +" "+ str(C_A) +" " + str(B_A) +" " + str(totalScore)
#                  tempLimit1[self.vocab[word].index] = totalScore
#                  scoreSection[self.vocab[word].index] = totalScoreText
#              else:
#                  raise KeyError("word '%s' not in vocabulary" % word)
#         dists2 = tempLimit1
#         resIdx2 = numpy.nonzero(dists2)[0].tolist()
#              
#         vecDict2 = {}
#         for i in resIdx2:
#             vecDict2[self.index2word[i]] = scoreSection[i]
#         from collections import OrderedDict
#         sorted_vecDict2 = OrderedDict(sorted(vecDict2.items(), key=lambda t: t[1], reverse=True))
# 
# #         best_temp3 = matutils.argsort(dists2, topn=topn, reverse=True).tolist()
# #         best_temp0 = matutils.argsort(dists1, topn=topn, reverse=True).tolist()
# #         
# #         if best_temp0 != best_temp3:
# #             #print "different argmax result:", fullVariable
# #             print [item for item in best_temp0] 
# #             print [item for item in best_temp3] 
# #             sys.exit()
# #         else:
#         dists1 = dists2
#         sorted_vecDict = sorted_vecDict2
        
        # combine 
        best1 = matutils.argsort(dists1, topn=topn, reverse=True)
        #print best1
        if any(dists1[i] <= 0 for i in best1):
            logger.debug("Direction_Bad: %s", \
                     sorted_vecDict) 
            writeList2File("direction.txt", ["000"])
            return None
        
        # make sure result is a sorted version of the distance result , else quit 
        import collections
        if collections.Counter(wordD_list) != \
        collections.Counter([self.index2word[item] for item in best1]):
            logger.debug("Fail: %s\n %s\n", \
                     wordD_list, len(wordD_list),\
                     [self.index2word[item] for item in best1], len([self.index2word[item] for item in best1]))  
            sys.exit() 
        
        logger.debug("Direction: %s", \
                     [(self.index2word[sim], sorted_vecDict[self.index2word[sim]]) for sim in best1 if sim not in all_words]) 
        writeList2File("direction.txt", [self.index2word[sim] for sim in best1 if sim not in all_words])       
        return dists1

def most_similar_cosmul_direction(self, positive=[], negative=[], topn=10, fullVariable=[], vecModel=None):
        """
        .. [4] Omer Levy and Yoav Goldberg. Linguistic Regularities in Sparse and Explicit Word Representations, 2014.

        """
        self.init_sims()
        #logger = logging
        tempLimit = numpy.zeros(shape=(vecModel.shape[0])) #Quick Fix. Fake array to fit into Gensim format
        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar_cosmul('dog'), as a shorthand for most_similar_cosmul(['dog'])
            positive = [positive]

        all_words = set()

        def word_vec(word):
            if isinstance(word, ndarray):
                return word
            elif word in self.vocab:
                all_words.add(self.vocab[word].index)
                return self.syn0norm[self.vocab[word].index]
            else:
                raise KeyError("word '%s' not in vocabulary" % word)

#         negative = [word_vec(word) for word in negative]
#         if not positive:
#             raise ValueError("cannot compute similarity with no input")
        
        # equation (4) of Levy & Goldberg "Linguistic Regularities...",
        # with distances shifted to [0,1] per footnote (7)
        def positive_Cos(vecA, vecB):
            #result = matutils.unitvec(((1 + dot(vecA, vecB)) / 2)).astype(REAL)
            from sklearn.metrics.pairwise import cosine_similarity
            result = ((1 + cosine_similarity(vecA, vecB)) / 2).astype(REAL)
            #print result
            return result 
        
        wordA = fullVariable[0]
        wordB = fullVariable[1]
        wordC = fullVariable[2]
        vecA = self.syn0norm[self.vocab[wordA].index] 
        vecB = self.syn0norm[self.vocab[wordB].index]
        vecC = self.syn0norm[self.vocab[wordC].index] 
        B_C = positive_Cos(vecB, vecC)
        B_A = positive_Cos(vecB ,vecA)
        #print wordA, wordB, wordC, self.vocab[wordA].index, self.vocab[wordB].index, self.vocab[wordC].index
        # add assessment item
        all_words.add(self.vocab[wordA].index)
        all_words.add(self.vocab[wordB].index)
        all_words.add(self.vocab[wordC].index)

        # get all (d - b) . checked 
        for word in fullVariable[3:]: # these are the predict ans (d) and b
            if isinstance(word, ndarray):
                vecA = wordA
                vecB = wordB
                vecC = wordC
                vecD = word
                #limited.append(matutils.unitvec(array([ vecD, vecB ]).mean(axis=0)).astype(REAL))    # mean =  algeria 1.0 vec, algiers -1.0 vec
                sys.exit() # not yet fixed
            elif word in self.vocab:
                vecD = self.syn0norm[self.vocab[word].index]
                D_C = positive_Cos(vecC, vecD)
                D_A = positive_Cos(vecA, vecD)
                tempLimit[self.vocab[word].index] = prod([D_C, B_A], axis=0) / (prod([D_A, B_C], axis=0) + 0.000001)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)

        resIdx = numpy.nonzero(tempLimit)[0]
        vecDict = {}
        for i in resIdx:
            vecDict[self.index2word[i]] = tempLimit[i]
        from collections import OrderedDict
        sorted_vecDict = OrderedDict(sorted(vecDict.items(), key=lambda t: t[1], reverse=True))
        best1 = matutils.argsort(tempLimit, topn=topn, reverse=True)

        if any(tempLimit[i] < 0.5 for i in best1):
            logger.debug("Direction_Bad: %s", \
                     sorted_vecDict) 
            writeList2File("direction.txt", ["000"])
            return None
        import collections
        Dist_Result = [item for item in positive[:-1]]
        predict_result = [self.index2word[item] for item in best1]
        if collections.Counter(Dist_Result) != \
        collections.Counter(predict_result):
            logger.debug("Fail: %s\n %s\n" \
                    % (Dist_Result, predict_result))  
            sys.exit() 
        
        logger.debug("Direction: %s", \
                     [(self.index2word[sim], float(tempLimit[sim])) for sim in best1 if sim not in all_words]) 
        writeList2File("direction.txt", [self.index2word[sim] for sim in best1 if sim not in all_words])  

       
        # ignore (don't return) words from the input
        #result = [(self.index2word[sim], float(tempLimit[sim])) for sim in best if sim not in all_words]
        return tempLimit
    
if __name__ == "__main__":
    logger = logging.getLogger()
    #logger = logging.getLogger(__name__)
    #logging.basicConfig(stream=sys.stdout, level=logging.INFO) # debug can see comprehensive result: oov and wrong predict
    #logger = logging.getLogger(__name__)
    #Get the root logger
    logger = logging.getLogger()
    #Have to set the root logger level, it defaults to logging.WARNING
    logger.setLevel(logging.DEBUG)
    
    logging_handler_out = logging.StreamHandler(sys.stdout)
    logging_handler_out.setLevel(logging.INFO)
    #logging_handler_out.addFilter(LessThanFilter(logging.WARNING))
    logger.addHandler(logging_handler_out)
    
    logging_handler_err = logging.FileHandler("Result.log", mode='w')
    logging_handler_err.setLevel(logging.DEBUG)
    logger.addHandler(logging_handler_err)
    
    Clear = open("distance.txt",'w')
    Clear = open("direction.txt",'w')
    Clear = open("Result.txt",'w')
    
    model = Word2Vec.load_word2vec_format(sys.argv[1], binary=True, encoding='iso-8859-1')
    #model = Word2Vec.load('Google_News_Gensim',mmap='r')
    #accuracy = model.accuracy(sys.argv[2], use_lowercase=False)
#     print accuracy
    accuracy = model.accuracy(sys.argv[2], restrict_vocab=30000, most_similar=most_similar, use_lowercase=False) # list return as incorrect, section, correct
    accResult = [resline.split() \
                                  for resline in io.open("Result.txt", 'r', encoding='utf-8')]
            
    
    #accuracy = model.accuracy(sys.argv[2], restrict_vocab=30000)
#     for item in accuracy:
#          for a, b, c, d in item["incorrect"]:
#              if item['section'] != "total":
#                  test = model.most_similar(positive=[b, c], negative=[a], topn=10)
#                  print a,b,c,d, test