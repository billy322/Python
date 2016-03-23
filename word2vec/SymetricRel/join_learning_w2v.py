'''
Created on 19 Jan 2016

@author: Billy
'''

import getopt,sys,os ,copy
from wvlib import wvlib as wv
from gensim.models import word2vec
from tools import utilities 

class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:],'hi:n:qf')
        opts = dict(opts)

        if '-h' in opts:
            self.printHelp()

        if len(args) == 3:
            self.inputFile = args[0]
            self.retroVector = args[1]
            self.outputfile = args[2]
        else:
            print >> sys.stderr, '\n*** ERROR: must specify precisely 3 arg input, retrofit vector, output***'
            self.printHelp()
            
            
        if '-f' in opts:
            self.fname = True

    def printHelp(self):
        help = __doc__.replace('<PROGNAME>',sys.argv[0],1)
        print >> sys.stderr, help
        exit()
        
if __name__ == "__main__":
    config = CommandLine()

    #from gensim.models import word2vec

    w2v1 = wv.load(config.inputFile).normalize()  # orginal w2v
    #print w2v1['his']
    w2v1= dict(w2v1)
    print len(w2v1.keys())
        
    w2v2 = wv.load(config.retroVector).normalize() # symetric w2v
    w2v2= dict(w2v2)
   
    #print "size of w2v 2:",len(w2v2)
            
    seta=set(w2v1.keys())
    setb=set(w2v2.keys())
        
    intersection = seta.intersection(setb)
        
    newWordvector = w2v1.copy()
        
    print len(intersection)
    for word in intersection:
        #print word
        #print "w2v1[word]", w2v1[word]
        #print "w2v2[word]", w2v2[word]
        #print "Before", newWordvector[word]
        newWordvector[word] = 0.7 * w2v1[word] + 0.3 * w2v2[word]
        #print "after", newWordvector[word]
           
      
    util = utilities()
    util.write_word_vecs(newWordvector, config.outputfile)  # output file
    model = wv.load(config.outputfile)
    print len(model.vocab.words())
    model.save_bin(config.outputfile.replace(".txt",".bin"))
    #w2v1 = wv.load(config.outputfile.replace(".txt",".bin")).normalize()
    #print w2v1['his']
