import argparse
import os
import sys
import codecs
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import itertools
import subprocess

def main():

    sentence_data = []
    input_files, data_dir, lm_vocab_file, dims = parse_args()
    vocab_file = os.path.join(data_dir, u'vocab.txt')
    train_file = os.path.join(data_dir, u'train.ind')
    model_file = os.path.join(data_dir, u'model_{}'.format(dims))
   
    valid_words = set()
    with codecs.open(lm_vocab_file, u'r', u'utf-8') as vf:
        for line in vf:
            valid_words.add(line.strip().lower())
    
    for path in input_files:
        print u"Reading", path, u"..."
        with codecs.open(path, u'r', u'utf-8') as f:
            for line in f:
                tokens = line.strip().split(u' ')
                #if re.search(r'[\.\?\!\;\,]$', tokens[-1]):
                #    tokens[-1] = tokens[-1][:-1]
                tokens = [token.lower() for token in tokens 
                          if token in valid_words]
                sentence_counts = {}
                for token in tokens:
                    sentence_counts[token] = sentence_counts.get(token, 0) + 1
                if len(sentence_counts) > 0:
                    sentence_data.append(sentence_counts)
    vectorizer = DictVectorizer()
    tfidfer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=False,
                               sublinear_tf=False)

    print "Computing tfidf..."
    X = vectorizer.fit_transform(sentence_data)
    X = tfidfer.fit_transform(X)


    print "Found {} sentences and {} words".format(X.shape[0], X.shape[1])

    with codecs.open(vocab_file, u'w', u'utf-8') as vf:   
        for fname in vectorizer.get_feature_names():
            vf.write(fname)
            vf.write(u'\n')
            vf.flush()

    X = X.tocoo()    

    with open(train_file, u'w') as tf:
        tf.write('{} {}\n'.format(X.shape[0], X.shape[1]))
        c_row = 0
        c_words = []    
        for i,j,v in itertools.izip(X.row, X.col, X.data):
            if i != c_row:
                c_words.sort(key=lambda x: x[0])
                tf.write(str(len(c_words)))
                for index, tfidf in c_words:
                    tf.write(' {} {}'.format(index, tfidf))
                tf.write('\n')
                tf.flush()
                c_words = []
                c_row = i
                     
            c_words.append((j, v)) 
        tf.write(str(len(c_words)))
        c_words.sort(key=lambda x: x[0])
        for index, tfidf in c_words:
            tf.write(' {} {}'.format(index, tfidf))
        tf.write('\n')
        tf.flush()
   
    print u"Training model..."
    subprocess.call(["wtmf", "1", train_file, model_file, 
                     str(dims), "20", "0.01", "20"])

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--lm-inputs',
                        help=u'List of lm files',
                        type=unicode, nargs='+', required=True)
 
    parser.add_argument('-v', '--vocab-list',
                        help=u'Allowed vocabulary',
                        type=unicode, required=True)
   
    parser.add_argument('-d', '--data-dir',
                        help=u'Location to write data, model, and vocab',
                        type=unicode, required=True)

    parser.add_argument('--dims',
                        help=u'Dimensions of latent vectors',
                        type=int, required=True)

    
    args = parser.parse_args()
    infiles = args.lm_inputs
    data_dir = args.data_dir 
    vocab = args.vocab_list
    dims = args.dims

    assert dims > 0

    if data_dir != '' and not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return infiles, data_dir, vocab, dims

if __name__ == '__main__':
    main()
