import marisa_trie
import os
import gzip
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from cuttsum.srilm import Client

class SalienceFeatureSet(object):
    def __init__(self, features=None):
        
        self.character_features = False
        self.language_model_features = False
        self.frequency_features = False
        self.geographic_features = False
        self.query_features = False 
        if features is not None:
            self.activate_features(features)

    def __unicode__(self):
        return u'SalienceFeatureSet: '  \
            u'char[{}] lm[{}] freq[{}] geo[{}] query[{}]'.format(
                u'X' if self.character_features is True else u' ',
                u'X' if self.language_model_features is True else u' ',
                u'X' if self.frequency_features is True else u' ',
                u'X' if self.geographic_features is True else u' ',
                u'X' if self.query_features is True else u' ')

    def __str__(self):
        return unicode(self).decode(u'utf-8')

    def activate_features(self, features):

        for fset in features:
            if fset == u'all':
                self.character_features = True
                self.language_model_features = True
                self.frequency_features = True
                self.geographic_features = True
                self.query_features = True 
                break
            elif fset == u'character':            
                self.character_features = True
            elif fset == u'language model':            
                self.language_model_features = True
            elif fset == u'frequency':            
                self.frequency_features = True
            elif fset == u'geographic':            
                self.geographic_features = True
            elif fset == u'query':            
                self.query_features = True 

    def as_list(self):
        features = [] 
        if self.character_features is True:
            features.append(u'character')
        if self.language_model_features is True:
            features.append(u'language model')       
        if self.frequency_features is True:
            features.append(u'frequency')          
        if self.geographic_features is True:
            features.append(u'geographic')        
        if self.query_features is True:
            features.append(u'query')
        return features

    def as_set(self):
        return set(self.as_list())    
   
class TfIdfExtractor(object):

    def __init__(self, idf_path):
        self.trie_ = None
        if os.path.exists(idf_path):
            with gzip.open(idf_path, u'r') as f:
                trie = marisa_trie.RecordTrie("<d")
                trie.read(f)
                self.trie_ = trie

    def process_article(self, si, corpus):
        if u'article-clf' not in si.body.sentences:
            return list() 
        avg_tfidfs = list()
        sents = si.body.sentences[u'article-clf']
        tf_counts = self.make_tf_counts(sents)
        
        for sentence in sents:
            avg_tfidf = self.avg_tfidf(sentence, tf_counts)
            avg_tfidfs.append(avg_tfidf)

        features = []
        for avg_tfidf in avg_tfidfs:
            features.append({'avg_tfidf_t0': avg_tfidf})
        return features
            
    def make_tf_counts(self, sents):
        tf_counts = defaultdict(int)
        for sent in sents:
            for token in sent.tokens:
                tf_counts[token.token.decode(u'utf-8').lower()] += 1
        return tf_counts

    def avg_tfidf(self, sentence, tf_counts):
        
        if self.trie_ is None:
            return -float('inf')
        total_tfidf = 0
        n_terms = 0

        unique_words = set()
        for token in sentence.tokens:
            term = token.token.decode(u'utf-8').lower()
            unique_words.add(term)

        n_terms = len(unique_words)
        if n_terms == 0:
            return 0

        for word in unique_words:
            idf = self.trie_.get(word, None)
            if idf is None:
                idf = 0
            else: 
                # trie packs single items as a list of tuple, so we need to 
                # pull the actual data out.
                idf = idf[0][0]
            #print word, idf, tf_counts[word] * idf
            total_tfidf += tf_counts[word] * idf
        return total_tfidf / float(n_terms)    


class LMProbExtractor(object):
    def __init__(self, domain_port, domain_order,
                 gigaword_port, gigaword_order):
        self.tok_ = RegexpTokenizer(r'\w+')
        self.domain_lm_ = Client(domain_port, domain_order, True)
        self.gigaword_lm_ = Client(gigaword_port, gigaword_order, True)

    def process_article(self, si):
        if u'article-clf' not in si.body.sentences:
            return list()
        lm_scores = []

        for sentence in si.body.sentences[u'article-clf']:
            bytes_string = ' '.join(token.token for token in sentence.tokens)
            uni_string = bytes_string.decode(u'utf-8')
            uni_string = uni_string.lower()
            uni_tokens = self.tok_.tokenize(uni_string)
            uni_string = u' '.join(uni_tokens)
            bytes_string = uni_string.encode(u'utf-8')
            dmn_lp, dmn_avg_lp = self.domain_lm_.sentence_log_prob(
                bytes_string)
            gw_lp, gw_avg_lp = self.gigaword_lm_.sentence_log_prob(
                bytes_string)
            lm_scores.append(
                {u"domain lp": dmn_lp, u"domain avg lp": dmn_avg_lp,
                 u"gigaword lp": gw_lp, u"gigaword avg lp": gw_avg_lp})
        return lm_scores
