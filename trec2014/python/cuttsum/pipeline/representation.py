import marisa_trie
import os
import gzip
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from cuttsum.srilm import Client
from itertools import izip
import string

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

    def __init__(self, current_idf_path, prev_idf_paths):

        paths = [current_idf_path] + prev_idf_paths
        tries = []
        for path in paths:
            if os.path.exists(path):
                with gzip.open(path, u'r') as f:
                    trie = marisa_trie.RecordTrie("<dd")
                    trie.read(f)
                    tries.append(trie)
            else:
                tries.append(None)
        self.current_trie_ = tries[0]
        self.prev_tries_ = tries[1:]

        n_paths = len(paths)
        self.features = [u'TFID_FFEATS: avg tfidf, t0'] \
            + [u'TFIDF_FEATS: tfidf-delta, tm-{}'.format(i)
               for i in xrange(1, n_paths)]

    def tokenize(self, strings):
        sentences = []
        for string in strings:
            tokens = [token.lower() for token in string.split(' ')]
            sentences.append(tokens)
        return sentences

    def process_streamcorpus_strings(self, strings):
        sentences = self.tokenize(strings)

        tf_counts = self.make_tf_counts(sentences)
        feats = []
        for sentence in sentences:
            sent_feat = {}
            avg_tfidf_t0 = self.avg_tfidf(
                self.current_trie_, sentence, tf_counts)
            sent_feat[u'TFID_FFEATS: avg tfidf, t0'] = avg_tfidf_t0
            for i, trie in enumerate(self.prev_tries_, 1):
                delta_idf = avg_tfidf_t0 - self.avg_tfidf(
                    trie, sentence, tf_counts)
                label = u'TFIDF_FEATS: tfidf-delta, tm-{}'.format(i)
                sent_feat[label] = delta_idf

            feats.append(sent_feat)
           
        return feats
 

    def make_tf_counts(self, sentences):
        tf_counts = defaultdict(int)
        for sentence in sentences:
            for token in sentence:
                tf_counts[token] += 1
        return tf_counts


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
            

    def avg_tfidf(self, trie, sentence, tf_counts):
        
        if trie is None:
            return float('nan')
        total_tfidf = 0
        n_terms = 0

        unique_words = set()
        for token in sentence:
            unique_words.add(token)

        n_terms = len(unique_words)
        if n_terms == 0:
            return 0

        for word in unique_words:
            idf = trie.get(word, None)
            if idf is None:
                idf = 0
            else: 
                # trie packs single items as a list of tuple, so we need to 
                # pull the actual data out.
                idf = idf[0][0]
            #print word, idf, tf_counts[word] * idf
            total_tfidf += tf_counts[word] * idf
        return total_tfidf / float(n_terms)    

import re

class BasicFeaturesExtractor(object):

    def __init__(self):
        self.features = [
            u'BASIC_FEATS: sentence length',
            u'BASIC_FEATS: punc ratio',
            u'BASIC_FEATS: lower ratio',
            u'BASIC_FEATS: upper ratio',
            u'BASIC_FEATS: all caps ratio',
            u'BASIC_FEATS: person ratio', 
            u'BASIC_FEATS: location ratio', 
            u'BASIC_FEATS: organization ratio',
            u'BASIC_FEATS: date ratio', 
            u'BASIC_FEATS: time ratio', 
            u'BASIC_FEATS: duration ratio', 
            u'BASIC_FEATS: number ratio', 
            u'BASIC_FEATS: ordinal ratio', 
            u'BASIC_FEATS: percent ratio',
            u'BASIC_FEATS: money ratio', 
            u'BASIC_FEATS: set ratio', 
            u'BASIC_FEATS: misc ratio']
        self.ne_features = [
            u'BASIC_FEATS: person ratio', 
            u'BASIC_FEATS: location ratio', 
            u'BASIC_FEATS: organization ratio',
            u'BASIC_FEATS: date ratio', 
            u'BASIC_FEATS: time ratio', 
            u'BASIC_FEATS: duration ratio', 
            u'BASIC_FEATS: number ratio', 
            u'BASIC_FEATS: ordinal ratio', 
            u'BASIC_FEATS: percent ratio',
            u'BASIC_FEATS: money ratio', 
            u'BASIC_FEATS: set ratio', 
            u'BASIC_FEATS: misc ratio']



    def process_sentences(self, sc_strings, cnlp_strings):
        return [self.process_sentence(sc_string, cnlp_string)
                for sc_string, cnlp_string
                in izip(sc_strings, cnlp_strings)]
    
    def process_sentence(self, sc_string, cnlp_string):
        feats = {}
        cnlp_tokens = cnlp_string.split(' ')
        sc_tokens = sc_string.split(' ')
        
        feats[u'BASIC_FEATS: sentence length'] = len(cnlp_tokens)
       
        sc_no_space = sc_string.replace(' ', '')
        sc_no_space_punc = sc_no_space.translate(
            string.maketrans("",""), string.punctuation)
        
        punc_ratio = 1 - float(len(sc_no_space_punc)) / float(len(sc_no_space))
        feats[u'BASIC_FEATS: punc ratio'] = punc_ratio

        n_lower = len(re.findall(r'\b[a-z]', sc_string))
        n_upper = len(re.findall(r'\b[A-Z]', sc_string))
        n_all_caps = len(re.findall(r'\b[A-Z]+\b', sc_string))
        n_total = float(len(re.findall(r'\b[A-Za-z]', sc_string)))
        feats[u'BASIC_FEATS: lower ratio'] = n_lower / n_total
        feats[u'BASIC_FEATS: upper ratio'] = n_upper / n_total
        feats[u'BASIC_FEATS: all caps ratio'] = n_all_caps / n_total

        n_tokens = len(cnlp_tokens)
        ne_counts = {ne_feature: 0 for ne_feature in self.ne_features}
        for token in cnlp_tokens:
            if token.startswith('__') and token.endswith('__'):
                label = u'BASIC_FEATS: {} ratio'.format(token[2:-2])
                ne_counts[label] = ne_counts.get(label) + 1

        for token, count in ne_counts.iteritems():
            #label = u'BASIC_FEATS: {} ratio'.format(token[2:-2])
            feats[token] = \
                float(count) / n_tokens
        return feats

class LMProbExtractor(object):
    def __init__(self, domain_port, domain_order,
                 gigaword_port, gigaword_order):
        self.tok_ = RegexpTokenizer(r'\w+')
        self.domain_lm_ = Client(domain_port, domain_order, True)
        self.gigaword_lm_ = Client(gigaword_port, gigaword_order, True)
        self.features = [u"LM_FEATS: domain lp", 
                         u"LM_FEATS: domain avg lp",
                         u"LM_FEATS: gigaword lp", 
                         u"LM_FEATS: gigaword avg lp"]

    def process_corenlp_strings(self, strings):
        return [self.process_corenlp_string(string) for string in strings]

    def process_corenlp_string(self, string):
        dmn_lp, dmn_avg_lp = self.domain_lm_.sentence_log_prob(string)
        gw_lp, gw_avg_lp = self.gigaword_lm_.sentence_log_prob(string)
        return {u"LM_FEATS: domain lp": dmn_lp, 
                u"LM_FEATS: domain avg lp": dmn_avg_lp,
                u"LM_FEATS: gigaword lp": gw_lp, 
                u"LM_FEATS: gigaword avg lp": gw_avg_lp}

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



class QueryFeaturesExtractor(object):
    pass

class GeoFeaturesExtractor(object):
    pass
