## ACL_eval_workshop

Here contain the scripts and code used in ACL 2016 paper:
Intrinsic Evaluation of Word Vectors Fails to Predict Extrinsic Performance

## API Package
word2vec: original word2vec from Mikolov (https://code.google.com/archive/p/word2vec/)
wvlib: lib to read word2vec file (https://github.com/spyysalo/wvlib)

## Scripts
createRawText.sh: download file for creating raw corpus
createCorpus.sh: Pre-process text (input: raw corpus directory)
createModel.sh: Create word2vec.bin file with different window size
intrinsicEva.sh: run intrinsic evaluation on 8 benchmark data-set (input: Dir. for testing vector)
ExtrinsicEva.sh: run extrinsic evaluation

## Code
Pre-processing:
tokenize_text.py: tokenized text (need NLTK installed)
sentence_spliter.py: segment sentence

Intrinsic evaluation:
evaluate.py: perform intrinisic evaluation

Extrinsic evaluation: (Keras folder: Need either tensorflow or theano installed):
mlp.py: simple feed-forward Neural Network
setting.py: parameter for the Neual Network
