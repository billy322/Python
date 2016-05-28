import sys,logging

from six import iteritems, itervalues, string_types
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray
from gensim import utils, matutils
import numpy
#from gensim.models.word2vec import Word2Vec
from word2vec import Word2Vec


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
        logger.debug(" ") 
        logger.debug("Distance: %s", result) 
        # restore a, b : c, d
        a = negative[0][0]
        b = positive[0][0]
        d_c = [item[0] for item in result] + [positive[1][0]] # c is the last item 
        if [positive[1][0]] in [item[0] for item in result]:
            logger.debug("Contain: %s in %s", [positive[1][0]], \
                         [item[0] for item in result])
#         delme = [self.vocab[word].index for word in d_c]
#         delme2 = [self.index2word[index] for index in delme]
#         print delme
#         print delme2
        
        # After checking direction, return best result 
        result_dir = most_similar_direction(self, positive = d_c, negative=[b, a], topn=len(result), restrict_vocab=restrict_vocab, vecModel=limited)
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

def most_similar_direction(self, positive=[], negative=[], topn=False, restrict_vocab=None, vecModel=None):
        """
        this compute pair Direction sim(d-b, c-a)

        """
        self.init_sims()
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
        dists1 = dot(tempLimit, mean)
        resIdx = numpy.nonzero(dists1)[0]
        vecDict = {}
        for i in resIdx:
            #print i 
            vecDict[self.index2word[i]] = dists1[i]
        from collections import OrderedDict
        sorted_vecDict = OrderedDict(sorted(vecDict.items(), key=lambda t: t[1], reverse=True))
        best1 = matutils.argsort(dists1, topn=topn, reverse=True)
        #print best1
        if any(dists1[i] < 0.3 for i in best1):
            logger.debug("Direction_Bad: %s", \
                     sorted_vecDict)  
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
        return dists1


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) # debug can see comprehensive result: oov and wrong predict
    #logger = logging.getLogger(__name__)
    logger = logging
    model = Word2Vec.load_word2vec_format(sys.argv[1], binary=True, encoding='iso-8859-1')
    #model = Word2Vec.load('Google_News_Gensim',mmap='r')
    #accuracy = model.accuracy(sys.argv[2])
#     print accuracy
    accuracy = model.accuracy(sys.argv[2], restrict_vocab=30000, most_similar=most_similar) # list return as incorrect, section, correct
    #accuracy = model.accuracy(sys.argv[2], restrict_vocab=30000)
#     for item in accuracy:
#          for a, b, c, d in item["incorrect"]:
#              if item['section'] != "total":
#                  test = model.most_similar(positive=[b, c], negative=[a], topn=10)
#                  print a,b,c,d, test