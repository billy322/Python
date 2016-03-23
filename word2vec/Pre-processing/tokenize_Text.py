'''
Created on 22 Mar 2016

@author: Billy
'''
from tools import utilities
import getopt,sys,os
class CommandLine:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:],'hi:n:qf')
        opts = dict(opts)

        if '-h' in opts:
            self.printHelp()

        if len(args) == 2:
            self.inputFile = args[0]
            self.outputfile = args[1]
        else:
            print >> sys.stderr, '\n*** ERROR: must specify precisely 2 arg input, output***'
            self.printHelp()
            
            
        if '-f' in opts:
            self.fname = True

    def printHelp(self):
        help = __doc__.replace('<PROGNAME>',sys.argv[0],1)
        print >> sys.stderr, help
        exit()
        
if __name__ == "__main__":
    config = CommandLine()
    util = utilities()
    N = 2
    
    #print re.escape("name=&quot;kgm&quot;/&gt;")
    for i, chunk in enumerate(util.read_in_chunks(config.inputFile, N)):
        SentenceList = []
        for item in chunk:
            #print item
            item = item.encode('utf-8')
            result = util.SentenceTokenize(item)
            SentenceList.append(result)
            #resText = " ".join(result)
        util.writeListOfList2File(config.outputfile, SentenceList)
            #util.writeText2File(config.outputfile, resText)
        