import numpy
from gensim.models.word2vec import Word2Vec
import io, sys
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
from numpy import linalg as LA
from gensim import utils, matutils
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL, array

def read_word_vecs(filename):
    wordVectors = {}
    if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
    else: fileObject = io.open(filename, 'r', encoding='utf-8') #open(filename, 'r')
    for line in fileObject:
        line = line.strip()
        word = line.split()[0]
        wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
        tempVec = numpy.array([float(i) for i in line.split()[1:]])
        ''' normalize weight vector '''
        wordVectors[word] = tempVec / numpy.linalg.norm(tempVec)
    
    sys.stderr.write("Vectors read from: "+filename+" \n")
    return wordVectors

def concateWordVecs(words, model):
    vecs = []
    for word1 in words:
        word = word1.replace('\n', '').split("-")
        try:
            #print word[0], word[1]
            wordAvec = model[word[0]] * -1.0 #(model[word[0]] / numpy.linalg.norm(model[word[0]])) * -1.0
            wordCvec = model[word[1]] #(model[word[1]] / numpy.linalg.norm(model[word[1]]))
#             print "C", wordCvec[0:5]
#             print "A", wordAvec[0:5]
            #C_A_direction = wordCvec - wordAvec / (numpy.linalg.norm(wordCvec - wordAvec))
            C_A_direction = matutils.unitvec(array([ wordCvec, wordAvec ]).mean(axis=0)).astype(REAL)
            #C_A_direction = wordCvec + wordAvec
            #print C_A_direction[0:5]
            vecs.append(C_A_direction.reshape((1,C_A_direction.shape[0])))
            
        except KeyError:
            print "OOV find", word
            continue
    if vecs:
        vecs = numpy.concatenate(vecs)
        return numpy.array(vecs, dtype='float')
    else: 
        return None

def readQuestion(fname, model):
    with io.open(fname, 'r', encoding='utf-8') as f:
        count = 0
        fileLine = f.readlines()
        wordDict = dict.fromkeys([questionType.strip("\n").replace(" ","") for questionType in fileLine if ":" in questionType])
        wordList = []
        temp = []
        tempList_counter = -1
        print len(fileLine)
        print wordDict
        current = ""
        for i, line in enumerate(fileLine):
            if ":" in line or i== len(fileLine)-1:
                count = 0
                if ":" in line:
                    temp.append(line.strip("\n").replace(" ",""))
                if wordList:
                    print wordList
                    concatedVec = concateWordVecs(wordList, model)
                    del wordList[:] # empty the list for other category
                    if concatedVec is not None:
                        # skilearn don't have full scale SVD, truncated to min(M,N) of originl M
                        U, Sigma, VT = randomized_svd(concatedVec, n_components=min(concatedVec.shape[0],concatedVec.shape[1]),
                                                              n_iter=5,
                                                              random_state=0)
                        #U, Sigma, VT = numpy.linalg.svd(concatedVec, full_matrices=False)
                        # make sure the original matrix can be recovered
                        resume = numpy.dot(U.dot(numpy.diag(Sigma)), VT)
                        #print "U",U.shape, "Sigma", Sigma.shape, "VT", VT.shape
                        if not numpy.allclose(concatedVec,resume):
                            print "fail to recover concated Vector"
                            sys.exit()
                        # take the low-rank K vector:
                        new_Sigma = Sigma[0] 
                        new_U = U[:, 0] # (column 0 access)
                        new_VT = VT[0] # (row access)
                        k_U = new_U.dot(new_Sigma)
                        k_V = new_VT.dot(new_Sigma) #numpy.multiply(new_Sigma,new_VT)#new_VT.dot(new_Sigma)
                        print "Concated Vec", concatedVec[0:5]
                        print "kv", k_V[0:5], "new_Sigma", new_Sigma
                        print "put into dict item:", temp[tempList_counter]
                        wordDict[temp[tempList_counter]] = k_V # top singular right vector
                tempList_counter +=1
            elif count < 9: # get # training data
                word = line.split()
                indexText = word[0]+"-"+word[1]
                if indexText not in wordList:
                    wordList.append(indexText)
                    count +=1
            
    return wordDict

def write_dict2file(r, filename):    
    with open(filename, "a") as input_file:
        for key, value in r.items():
            if value is not None:
                input_file.write('%s %s\n' % (key.replace(" ",""), ' '.join(str(v) for v in value.tolist())))
            
if __name__ == "__main__":
    numpy.set_printoptions(suppress=False)
    model = Word2Vec.load_word2vec_format(sys.argv[1], binary=True)
    wordDict = readQuestion(sys.argv[2], model)
    print "len(wordDict)", len(wordDict), wordDict.keys()
    
    Clear = open("SVD.txt",'w')
    with io.open(sys.argv[2], 'r', encoding='utf-8') as f:
        fileLine = f.readlines()
        WordVecList = []
        for i, line in enumerate(fileLine):
            line = line.replace("\n", "")
            #print "line", line
            if ":" in line:
                #print line
                #print line.replace(" ","") in wordDict and wordDict[line.replace(" ","")] is not None
                if line.replace(" ","") in wordDict and wordDict[line.replace(" ","")] is not None:
                    keyVal = ' '.join(str(v) for v in wordDict[line.replace(" ","")].tolist())
                else:
                    keyVal = None 
            else:
                if keyVal is not None:
                    wordA = line.split()[0]
                    wordC = line.split()[1]
                    #wordB = line.split()[2]
                    vecValue = '%s %s\n' % (wordA+"-"+wordC, keyVal)
                    WordVecList.append(vecValue)
        uniqueWordVecList = set(WordVecList)
        with open("SVD.txt", "a") as input_file:
            #print '%s %s\n' % (wordA+"-"+wordC, keyVal)
            for item in uniqueWordVecList:
                #print "item2 write", item
                input_file.write(item)
                        
                    