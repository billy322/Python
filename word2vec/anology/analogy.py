import sys,logging

from six import iteritems, itervalues, string_types
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray
from gensim import utils, matutils
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
        logger = logging
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
        # restore a, b : c, d
#         a = negative[0][0]
#         b = positive[0][0]
#         d_c = [item[0] for item in result] + [positive[1][0]] # c is the last item 
        a = negative[0][0]
        b = positive[0][0]
        c = positive[1][0]
        d = [item[0] for item in result]
        d_b = d + [b] # c is the last item 
        d_c = d +[c]
        if [positive[1][0]] in [item[0] for item in result]:
            logger.debug("Contain: %s in %s", [positive[1][0]], \
                         [item[0] for item in result])
#         delme = [self.vocab[word].index for word in d_c]
#         delme2 = [self.index2word[index] for index in delme]
#         print delme
#         print delme2
        
        # After checking direction, return best result 
        #result_dir = most_similar_direction(self, positive = d_b, negative=[c, a], topn=len(result), vecModel=limited)
        result_dir = most_similar_direction(self, positive = d_c, negative=[b, a], topn=len(result), vecModel=limited)
        #result_dir = most_similar_cosmul_direction(self, positive = d_c, negative=[b, a], topn=len(result), vecModel=limited)
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
        logger = logging
        #logger.debug("\t\t========direction===========")  
        #logger.debug("d, c: %s", positive)  
        #logger.debug("b, a: %s", negative)  
  
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
        positiveWordDict = {(positiveWordDict[word], 1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in positive}
        negativeWordDict = {(negativeWordDict[word], 1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in negative}
        all_words = set() # store question words
        #limited = [] # store d (i.e. topN words having closest distance with b - a + c from most_similiar)
        tempLimit = numpy.zeros(shape=(vecModel.shape)) #Quick Fix. Fake array to fit into Gensim format

        wordB = negative[0][0]
        wordB_weight = negative[0][1]
        # add assessment item
        all_words.add(self.vocab[wordB].index)
        #print wordB, self.vocab[wordB].index
        # get all (d - b) . checked 
        for word, weight in positive[:-1]: # these are the predict ans (d) and b
            if isinstance(word, ndarray):
                vecD = weight * word
                vecB = wordB_weight * wordB
                #limited.append(matutils.unitvec(array([ vecD, vecB ]).mean(axis=0)).astype(REAL))    # mean =  algeria 1.0 vec, algiers -1.0 vec
                sys.exit() # not yet fixed
            elif word in self.vocab:
                vecD = weight * self.syn0norm[self.vocab[word].index]
                vecB = wordB_weight * self.syn0norm[self.vocab[wordB].index]
                #limited.append(matutils.unitvec(array([ vecD, vecB ]).mean(axis=0)).astype(REAL))    # mean =  algeria 1.0 vec, algiers -1.0 vec 
                tempLimit[self.vocab[word].index] = matutils.unitvec(array([ vecD, vecB ]).mean(axis=0)).astype(REAL)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)

        #mean = c - a 
        wordC = positive[-1][0]
        wordC_weight = positive[-1][1]
        wordA = negative[1][0]
        wordA_weight = negative[1][1]
       # print wordC, self.vocab[wordC].index,wordA, self.vocab[wordA].index
        # add assessment item
        all_words.add(self.vocab[wordC].index)
        all_words.add(self.vocab[wordA].index)
        # compute c - a 
        if isinstance(wordC, ndarray) and isinstance(wordA, ndarray): 
            vecA = wordA_weight * wordA 
            vecC = wordC_weight * wordC
            mean = matutils.unitvec(array([ vecC, vecA ]).mean(axis=0)).astype(REAL)  # normalise vector and find mean of 3 vectors (distance between 3)
        elif wordC in self.vocab and wordA in self.vocab:
            vecA = wordA_weight * self.syn0norm[self.vocab[wordA].index] 
            vecC = wordC_weight * self.syn0norm[self.vocab[wordC].index] 
            mean = matutils.unitvec(array([ vecC, vecA ]).mean(axis=0)).astype(REAL)  # mean of c - a    
        else:
            raise KeyError("word '%s' not in vocabulary" % word)   
        
        #dists = dot(limited, mean)  # this is for normal query
        dists1 = dot(tempLimit, mean)  # all row are zero, except those top10 word row
        resIdx = numpy.nonzero(dists1)[0]
        vecDict = {}
        for i in resIdx:
            #print i 
            vecDict[self.index2word[i]] = dists1[i]
        from collections import OrderedDict
        sorted_vecDict = OrderedDict(sorted(vecDict.items(), key=lambda t: t[1], reverse=True))
        best1 = matutils.argsort(dists1, topn=topn, reverse=True)
        #print best1
        if any(dists1[i] == 0 for i in best1):
            logger.debug("Direction_Bad: %s", \
                     sorted_vecDict) 
            writeList2File("direction.txt", ["000"])
            return None
        import collections
        if collections.Counter([item[0] for item in positive[:-1]]) != \
        collections.Counter([self.index2word[item] for item in best1]):
            logger.debug("Fail: %s\n %s\n", \
                     [item[0] for item in positive[:-1]], len([item[0] for item in positive[:-1]]),\
                     [self.index2word[item] for item in best1], len([self.index2word[item] for item in best1]))  
            sys.exit() 
        
        logger.debug("Direction: %s", \
                     [(self.index2word[sim], float(dists1[sim])) for sim in best1 if sim not in all_words]) 
        writeList2File("direction.txt", [self.index2word[sim] for sim in best1 if sim not in all_words])       
        return dists1

def most_similar_cosmul_direction(self, positive=[], negative=[], topn=10, vecModel=None):
        """
        .. [4] Omer Levy and Yoav Goldberg. Linguistic Regularities in Sparse and Explicit Word Representations, 2014.

        """
        self.init_sims()
        logger = logging
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
            result = matutils.unitvec(((1 + dot(vecA, vecB)) / 2)).astype(REAL)
            #print result
            return result 
        
        wordA = negative[1]
        wordB = negative[0]
        wordC = positive[-1]
        vecA = self.syn0norm[self.vocab[wordA].index] 
        vecB = self.syn0norm[self.vocab[wordB].index]
        vecC = self.syn0norm[self.vocab[wordC].index] 
        B_C = positive_Cos(vecB, vecC)
        A_B = positive_Cos(vecA, vecB)
        #print wordA, wordB, wordC, self.vocab[wordA].index, self.vocab[wordB].index, self.vocab[wordC].index
        # add assessment item
        all_words.add(self.vocab[wordA].index)
        all_words.add(self.vocab[wordB].index)
        all_words.add(self.vocab[wordC].index)

        # get all (d - b) . checked 
        for word in positive[:-1]: # these are the predict ans (d) and b
            if isinstance(word, ndarray):
                vecA = wordA
                vecB = wordB
                vecC = wordC
                vecD = word
                #limited.append(matutils.unitvec(array([ vecD, vecB ]).mean(axis=0)).astype(REAL))    # mean =  algeria 1.0 vec, algiers -1.0 vec
                sys.exit() # not yet fixed
            elif word in self.vocab:
                vecD = self.syn0norm[self.vocab[word].index]
                C_D = positive_Cos(vecC, vecD)
                A_D = positive_Cos(vecA, vecD)
                tempLimit[self.vocab[word].index] = prod([C_D, A_B], axis=0) / (prod([A_D, B_C], axis=0) + 0.000001)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)

        resIdx = numpy.nonzero(tempLimit)[0]
        vecDict = {}
        for i in resIdx:
            #print i 
            vecDict[self.index2word[i]] = tempLimit[i]
        from collections import OrderedDict
        sorted_vecDict = OrderedDict(sorted(vecDict.items(), key=lambda t: t[1], reverse=True))
        best1 = matutils.argsort(tempLimit, topn=topn, reverse=True)

        if any(tempLimit[i] < 0.3 for i in best1):
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
    logging.basicConfig(stream=sys.stdout, level=logging.INFO) # debug can see comprehensive result: oov and wrong predict
    #logger = logging.getLogger(__name__)
    
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