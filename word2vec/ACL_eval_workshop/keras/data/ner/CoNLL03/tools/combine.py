#!/usr/bin/env python

# Combine CoNLL data with predictions for evaluation using conlleval.
# The predictions should be in a space-delimited format where the
# first field is the token text and the last the predicted label.
# Predicted IOBES tags (if any) are mapped to the IOB1 scheme used in
# CoNLL'03 data.

import sys

# Sentence separator used in predictions
SEPARATORS = ['</s>', 'PADDING']

def read_data(fn):
    data = []
    with open(fn) as f:
        for line in f:
            data.append(line.split())
    return data

def align_predictions(conll, pred):
    aligned = []
    i, j = 0, 0
    while i < len(conll) and j < len(pred):
        aligned.append(conll[i][:])
        if conll[i]:
            # CoNLL token. Confirm match and add predicted label.
            assert conll[i][0] == pred[j][0], 'token mismatch'
            aligned[-1].append(pred[j][-1])
            i += 1
            j += 1
        else:
            # CoNLL sentence separator. Skip and consume any number of
            # sentence separators in the predictions.
            i += 1
            while j < len(pred) and (not pred[j] or pred[j][0] in SEPARATORS):
                j += 1
    # Skip remaining separators and check that all data was consumed
    while i < len(conll) and not conll[i]:
        i += 1
    while j < len(pred) and (not pred[j] or pred[j][0] in SEPARATORS):
        j += 1
    assert i == len(conll), 'CoNLL data remains'
    assert j == len(pred), 'predictions remain'
    return aligned

def iobes_to_iob1(data):
    prev = None
    for fields in data:
        iobes = fields[-1]
        if iobes == 'O' or iobes[0] == 'I':
            iob1 = iobes
        elif iobes[0] == 'E':
            iob1 = 'I' + iobes[1:]
        else:
            assert iobes[0] in ('S', 'B'), 'invalid tag %s' % iobes
            # B only needed if previous was an "in" tag of same type
            if prev is None or prev[0] == 'O' or prev[1:] != iobes[1:]:
                iob1 = 'I' + iobes[1:]
            else:
                iob1 = 'B' + iobes[1:]
        prev = iobes
        fields[-1] = iob1
    return data

def main(argv):
    if len(argv) != 3:
        print >> sys.stderr, 'Usage: combine.py CONLLDATA PREDICTIONS'
        return 1

    conll = read_data(argv[1])
    pred = read_data(argv[2])

    pred = iobes_to_iob1(pred)

    aligned = align_predictions(conll, pred)

    for fields in aligned:
        print ' '.join(fields)
    
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
