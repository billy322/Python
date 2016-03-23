from word2Vec import utilities as util
import sys , re, io



def exact_Match(phrase, word):
    word = re.escape(word)
    b = r'(\s|^|$)' 
    return re.match(b + word + b, phrase)

def getRel(text):
    processed = []
    l = text.split()[1:]
    #wordDict = {'and':['the', 'one'], 'or':['the','a'], 'nor':None}
    for i, token in enumerate(l):
        token = re.escape(token)
        #print token
        #token = token1.lower
        if exact_Match('and', token):
            if exact_Match("the", l[i+1]) or exact_Match("one", l[i+1]):
                #print "case X and the/and one Y:", l[i-1],l[i+2] 
                # capture X and the/and one Y
                processed.append([l[i-1],l[i+2]])
            else:
                #print "capture X and Y:", l[i-1],l[i+1]
                # capture X and Y 
                processed.append([l[i-1],l[i+1]])
                    
        elif exact_Match('or', token):
            if exact_Match('the', l[i+1]) or exact_Match('a', l[i+1]):
                # capture X or the/ or one / or a Y
                #print "capture X or the/ or one / or a Y:",l[i-1],l[i+2]
                processed.append([l[i-1],l[i+2]])
            else:
                #try:
#                     if l[i-1] == "either":
#                         processed.append[l[i],l[i+2]]
#                 except:
                    #print "capture X or Y/either X or Y" ,l[i-1],l[i+1]
                    #capture X or Y/either X or Y
                    processed.append([l[i-1],l[i+1]]) 
        elif exact_Match('nor', token ):
            #print "capture nor case:" ,l[i-1],l[i+1]
            #capture nor case
            processed.append([l[i-1],l[i+1]])
            
        elif exact_Match("rather" , token ) and exact_Match('than' , l[i+1]) :
            #print "capture X rather than Y:" ,l[i-1],l[i+2]
            # capture X rather than Y
            processed.append([l[i-1],l[i+2]]) 
                   
        elif exact_Match('as', token) and exact_Match('well',l[i+1]) and exact_Match('as',l[i+2]):
            #print "capture X as well as Y:",l[i-1],l[i+3]
            # capture X as well as Y
            processed.append([l[i-1],l[i+3]])
            
        elif exact_Match('from',token) and exact_Match('to', l[i+2]):
            #print "capture from X to Y:",l[i+1],l[i+3]
            # capture from X to Y
            processed.append([l[i+1],l[i+3]])
    #text = " ".join(processed)
    return processed

if __name__ == "__main__":
    
    N = 50000
    #print re.escape("name=&quot;kgm&quot;/&gt;")
    for i, chunk in enumerate(util.read_in_chunks(sys.argv[1], N)):
        if i > -1:
            util.appendFile(unicode(i), sys.argv[2]) # log
            
            text = " ".join(chunk)
    
            print "Get relationship"
            resultPair =  getRel(text)
            print resultPair
            # append to file
            with io.open(sys.argv[3], 'a', encoding='utf-8') as f:
                for x,y in (resultPair):
                    f.write("%s %s\n" % (unicode(util.removePunctuation(x)), unicode(util.removePunctuation(y))))
                    #f.write("%s %s\n" % (util.removePunctuation(y), util.removePunctuation(x)))
    