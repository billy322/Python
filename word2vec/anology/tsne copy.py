#
#  tsne.py
#  
# Implementation of t-SNE in Python. The implementation was tested on Python 2.5.1, and it requires a working 
# installation of NumPy. The implementation comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
# The example can be run by executing: ipython tsne.py -pylab
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.
from six import iteritems, itervalues, string_types
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray
from gensim import utils, matutils
import numpy
from matplotlib.backends.backend_pdf import PdfPages
import os  
import io  
from sklearn.decomposition import PCA

#import tempfile
#os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib 
#import matplotlib
#matplotlib.use('TkAgg')
#matplotlib.use('Agg') 
import pylab as Plot
import sys
import math
from word2vec import Word2Vec
#from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import logging
import sys

def Hbeta(D = numpy.array([]), beta = 1.0):
  """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""
  
  # Compute P-row and corresponding perplexity
  P = numpy.exp(-D.copy() * beta);
  sumP = sum(P);
  H = numpy.log(sumP) + beta * numpy.sum(D * P) / sumP;
  P = P / sumP;
  return H, P;
  
  
def x2p(X = numpy.array([]), tol = 1e-5, perplexity = 30.0):
  """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

  # Initialize some variables
  #print "Computing pairwise distances..."
  (n, d) = X.shape;
  sum_X = numpy.sum(numpy.square(X), 1);
  D = numpy.add(numpy.add(-2 * numpy.dot(X, X.T), sum_X).T, sum_X);
  P = numpy.zeros((n, n));
  beta = numpy.ones((n, 1));
  logU = numpy.log(perplexity);
    
  # Loop over all datapoints
  for i in range(n):
  
    # Print progress
    if i % 500 == 0:
                   pass
             #print "Computing P-values for point ", i, " of ", n, "..."
  
    # Compute the Gaussian kernel and entropy for the current precision
    betamin = -numpy.inf; 
    betamax =  numpy.inf;
    Di = D[i, numpy.concatenate((numpy.r_[0:i], numpy.r_[i+1:n]))];
    (H, thisP) = Hbeta(Di, beta[i]);
      
    # Evaluate whether the perplexity is within tolerance
    Hdiff = H - logU;
    tries = 0;
    while numpy.abs(Hdiff) > tol and tries < 50:
        
      # If not, increase or decrease precision
      if Hdiff > 0:
        betamin = beta[i];
        if betamax == numpy.inf or betamax == -numpy.inf:
          beta[i] = beta[i] * 2;
        else:
          beta[i] = (beta[i] + betamax) / 2;
      else:
        betamax = beta[i];
        if betamin == numpy.inf or betamin == -numpy.inf:
          beta[i] = beta[i] / 2;
        else:
          beta[i] = (beta[i] + betamin) / 2;
      
      # Recompute the values
      (H, thisP) = Hbeta(Di, beta[i]);
      Hdiff = H - logU;
      tries = tries + 1;
      
    # Set the final row of P
    P[i, numpy.concatenate((numpy.r_[0:i], numpy.r_[i+1:n]))] = thisP;
  
  # Return final P-matrix
  #print "Mean value of sigma: ", numpy.mean(numpy.sqrt(1 / beta))
  return P;
  
  
def pca(X = numpy.array([]), no_dims = 50):
  """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

  #print "Preprocessing the data using PCA..."
  (n, d) = X.shape;
  X = X - numpy.tile(numpy.mean(X, 0), (n, 1));
  (l, M) = numpy.linalg.eig(numpy.dot(X.T, X));
  Y = numpy.dot(X, M[:,0:no_dims]);
  return Y;


def tsne(X = numpy.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
  """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
  The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""
  
  # Check inputs
  if X.dtype != "float64":
    #print "Error: array X should have type float64.";
    return -1;
  #if no_dims.__class__ != "<type 'int'>":      # doesn't work yet!
  #  print "Error: number of dimensions should be an integer.";
  #  return -1;
  
  # Initialize variables
  X = pca(X, initial_dims);
  (n, d) = X.shape;
  max_iter = 1000;
  initial_momentum = 0.5;
  final_momentum = 0.8;
  eta = 500;
  min_gain = 0.01;
  Y = numpy.random.randn(n, no_dims);
  dY = numpy.zeros((n, no_dims));
  iY = numpy.zeros((n, no_dims));
  gains = numpy.ones((n, no_dims));
  
  # Compute P-values
  P = x2p(X, 1e-5, perplexity);
  P = P + numpy.transpose(P);
  P = P / numpy.sum(P);
  P = P * 4;                  # early exaggeration
  P = numpy.maximum(P, 1e-12);
  
  # Run iterations
  for iter in range(max_iter):
    
    # Compute pairwise affinities
    sum_Y = numpy.sum(numpy.square(Y), 1);    
    num = 1 / (1 + numpy.add(numpy.add(-2 * numpy.dot(Y, Y.T), sum_Y).T, sum_Y));
    num[range(n), range(n)] = 0;
    Q = num / numpy.sum(num);
    Q = numpy.maximum(Q, 1e-12);
    
    # Compute gradient
    PQ = P - Q;
    for i in range(n):
      dY[i,:] = numpy.sum(numpy.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);
      
    # Perform the update
    if iter < 20:
      momentum = initial_momentum
    else:
      momentum = final_momentum
    gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
    gains[gains < min_gain] = min_gain;
    iY = momentum * iY - eta * (gains * dY);
    Y = Y + iY;
    Y = Y - numpy.tile(numpy.mean(Y, 0), (n, 1));
    
    # Compute current value of cost function
    if (iter + 1) % 10 == 0:
      C = numpy.sum(P * numpy.log(P / Q));
      #print "Iteration ", (iter + 1), ": error is ", C
      
    # Stop lying about P-values
    if iter == 100:
      P = P / 4;
      
  # Return solution
  return Y;
  
def plot_words(wordVectors, set1p, set1n, imgFile):
  
  X = []
  labels = []
  for word in set1p+set1n:
    if word in wordVectors:
      labels.append(word)
      X.append(wordVectors[word])
    
  X = numpy.array(X)
  for r, row in enumerate(X):
    X[r] /= math.sqrt((X[r]**2).sum() + 1e-6) 
  
  Y = tsne(X, 2, 80, 20.0);
  Plot.scatter(Y[:,0], Y[:,1], s=0.1)
  for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    if label in set1p:
      Plot.annotate(
        label, color="green", xy = (x, y), xytext = None,
        textcoords = None, bbox = None, arrowprops = None, size=10
      )
    if label in set1n:
      Plot.annotate(
        label, color="red", xy = (x, y), xytext = None,
        textcoords = None, bbox = None, arrowprops = None, size=10
      )

  Plot.savefig(imgFile, bbox_inches='tight')
  Plot.close()
  
def getWordVecs(words, model, dim=300):
  vecs = []
  types = []
  for word in words:
      word = word.replace('\n', '')
      try:
          vecs.append(model[word].reshape((1,dim)))
          types.append(word)
      except KeyError:
      	print "OOV find", word
      	continue
  vecs = numpy.concatenate(vecs)
  return numpy.array(vecs, dtype='float'), numpy.asarray(types, dtype='<U2') #TSNE expects float type values

def write2file(string, fname):
	with io.open(fname, 'a', encoding='utf-8') as out_file:
		out_file.write(string)

def writeVec2file(wordList):
	f = open(sys.argv[4],'w')
	f = open(sys.argv[5],'w')
	import itertools
	mergedSet = list(set(list(itertools.chain.from_iterable(fullList)))) # flat a list of list and remove repeat
	testVec, testType = getWordVecs(mergedSet, model,dim=model.syn0.shape[1])
	print testVec.shape, len(testType)
	if testVec.shape[0] == testType.shape[0]:
		numpy.savetxt(sys.argv[4], testVec, delimiter=' ', newline='\n', header='', footer='')
    	numpy.savetxt(sys.argv[5], testType, fmt='%s',delimiter=' ', newline='\n', header='', footer='')

    	#headString = str(testVec.shape[0]) +" "+ str(testVec.shape[1])
    	
def combineW2V(type, vec):
	f = open('test.txt','w')
	# combine type and value to form vector in w2v format
	vType = numpy.loadtxt(type, delimiter=' ',dtype=str,comments="xsakjnaksdja")
	vVec = numpy.loadtxt(vec, delimiter=' ').astype(str)
	print vType.shape, vVec.shape,vType.dtype, vVec.dtype
	combine = numpy.column_stack((vType,vVec))
	headText = str(vVec.shape[0]).encode("utf-8").decode("utf-8") +" " + str(vVec.shape[1]).encode("utf-8").decode("utf-8") 
	#write2file(headText+u"\n",'test.txt')
	numpy.savetxt('test.txt', combine, fmt='%s',delimiter=' ', newline='\n', footer='')
	line_prepender('test.txt', headText)
	    		    	
def writeAnRes(accuracy, target):
	f = open("analogy_res.txt",'w')
	for item in accuracy:
 		for a, b, c, d in item[target]:
 			if item['section'] != "total":
 				test = model.most_similar(positive=[b, c], negative=[a], topn=10)
 				topNres = ' '.join([word[0] for word in test])
 				res = " ".join(["%s:%s %s %s %s " % (item['section'], a, b, c, d)]) + topNres+ "\n"
 				write2file(res, "analogy_res.txt")
 				
def line_prepender(fname, headTxt):
	with io.open(fname, "r+", encoding='utf-8') as f: 
		s = f.read(); 
		f.seek(0); 
		f.write(headTxt+u"\n" + s)
		


def runPCA(testVec, nCompoent):
	pca = PCA(nCompoent)
	reduced_vecs = pca.fit_transform(testVec)
	print 'Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)
	return reduced_vecs

def runTsne(testVec, dim):
	ts = TSNE(dim)
	reduced_vecs = ts.fit_transform(testItem) # using T-SNE to transform to x,y
	#reduced_vecs = tsne(testItem, 2, 80, 20.0);
	return reduced_vecs

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

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            elif word in self.vocab:
                mean.append(weight * self.syn0norm[self.vocab[word].index])
                all_words.add(self.vocab[word].index)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        limited = self.syn0norm if restrict_vocab is None else self.syn0norm[:restrict_vocab]
        dists = dot(limited, mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]
       
	        		
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO) # debug can see comprehensive result: oov and wrong predict
    test_words =[]
    import analogy as analogy
	# load model
 	#sys.argv 1: model 2: analogy question 3: res 4: word 5: vector
 	
 	# create analogy result
    model = Word2Vec.load_word2vec_format(sys.argv[1], binary=True, encoding='iso-8859-1')
    accuracy = model.accuracy(sys.argv[2], restrict_vocab=30000, most_similar=analogy.most_similar, use_lowercase=False) # list return as incorrect, section, correct # list return as incorrect, section, correct
    print accuracy[0]
    # write Analogy result to file
    writeAnRes(accuracy, 'incorrect')
   		
   	# read analogy result file
    fullList = []
    with io.open("analogy_res.txt", 'r', encoding='utf-8') as infile:
		for line in infile.readlines():	
			test_words = line.split(":", 1)[1].split()
   			fullList.append(test_words)
   	
  	#writeVec2file(fullList)
   	
  	# read type and vector to form word2vec
  	#combineW2V(sys.argv[4], sys.argv[5])
   	
#     # visualise vector
# 	#model = Word2Vec.load_word2vec_format(sys.argv[1], binary=True)
# 	#model.save('Google_News_Gensim')
# 	
# 	#later load the model
# 	model = Word2Vec.load('Google_News_Gensim',mmap='r')
    pdf = matplotlib.backends.backend_pdf.PdfPages("incorrect.pdf")
 	#for i in range(3):
    for i, ans in enumerate(fullList):
 		#figName.append("figure"+str(i))
 		plt.figure(i)
 		
 		#print fullList
 		testVec, testType = getWordVecs(fullList[i], model)
 		
 		print len(fullList[i]), len(testVec)
 		reduced_vecs = runPCA(testVec, 2)
        #reduced_vecs = runPCA(testVec, 50)
 
 		print len(fullList[i]), len(reduced_vecs)
 		#reduced_vecs = ts.fit_transform(np.concatenate((food_vecs, sports_vecs, weather_vecs)))
 		plt.scatter(reduced_vecs[:,0], reduced_vecs[:,1], s=0.1)
 		for t, item in enumerate(fullList[i]):
 			item = item
 			print item, reduced_vecs[t,0], reduced_vecs[t,1], t
 			if t <=  2: # question 
 				print "Case1"
 				plt.annotate(
 							item, color="green",
 							xy = (reduced_vecs[t,0], reduced_vecs[t,1]), xytext = None,
 							textcoords = None,
 							bbox = None,
 							arrowprops = None, size=10)
 			elif t == 3: # ground truth
 				# if arrow not match point color, mean this point also belong to the group of that color
 				print "Case2"
 				plt.annotate(
 							item, color="blue", 
 							xy = (reduced_vecs[t,0], reduced_vecs[t,1]), xytext = None,
 							textcoords = None,
 							bbox = None,
 							arrowprops = dict(facecolor='blue', shrink=0.05), size=10)  
 			elif t == 4: # predict answer
 				print "Case3"
 				plt.annotate(
 							item, color="black",
 							xy = (reduced_vecs[t,0], reduced_vecs[t,1]), xytext = None,
 							textcoords = None,
 							bbox = None,
 							arrowprops = None, size=10)
 			elif t > 4:
 				print "Case4" # topN result
 				plt.annotate(
 							item, color="red",
 							xy = (reduced_vecs[t,0], reduced_vecs[t,1]), xytext = None,
 							textcoords = None,
 							bbox = None,
 							arrowprops = None, size=10)
    for i in plt.get_fignums():
 		pdf.savefig( plt.figure(i) )
    pdf.close()
    #plt.show()
 	#figs = list(map(plt.figure, plt.get_fignums())
 	