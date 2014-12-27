from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import re
from pkg_resources import resource_filename
import os

class ArticleDetector:
    def __init__(self, event, vectorizer_pkl=None, clf_pkl=None):
        if vectorizer_pkl is None:
            vectorizer_pkl = resource_filename(
                u'cuttsum',
                os.path.join(u'models', u'article_vectorizer.pkl'))
        if clf_pkl is None:
            clf_pkl = resource_filename(
                u'cuttsum',
                os.path.join(u'models', u'article_clf.pkl'))
         
        self._vec = joblib.load(vectorizer_pkl)
        self._clf = joblib.load(clf_pkl)
        self.query_words = event.query
        self.len_max_cutoff = 45
        self.len_min_cutoff = 3
        
        safe_words = u'|'.join([re.escape(word) for word in event.query])
        self.patt = re.compile(r'({})'.format(safe_words), re.I)

    def find_articles(self, sentences):
        indices = []
        for arange in self.find_ranges(sentences):
            if self.contains_query(sentences, arange):
                for i in arange:
                    ntoks = len(sentences[i].tokens)
                    if ntoks <= self.len_max_cutoff:
                        if ntoks > self.len_min_cutoff:
                            indices.append(i)

        return indices

    def contains_query(self, sentences, arange):
        hits = {}
        for word in self.query_words:
            hits[word] = False
        for i in arange:
            s = sentences[i]
            if len(s.tokens) > self.len_max_cutoff:
                continue
            sstr = u' '.join(t.token.decode('utf-8') for t in s.tokens)
            for m in self.patt.findall(sstr):
                hits[m.lower()] = True
        for val in hits.itervalues():
            if val is False:
                return False
        return True

    def find_ranges(self, sentences):
        n_sents = len(sentences)
        data = []
        for sent_id, sent in enumerate(sentences):
            feats = self._make_feature_dict(sent, sent_id, n_sents)
            data.append(feats)
        if len(data) == 0:
            return []
        X = self._vec.transform(data)
        probs = self._clf.predict_proba(X)
#        end_probs = self._end_clf.predict_proba(X)
#        ends = np.where(end_probs[:,1] > .5)[0]
#        ranges = []
        ranges = []
        cur_range = []
        last_idx = -1
        for idx in np.where(probs[:,1] > .5)[0]:
            if last_idx < idx - 5:
                if len(cur_range) > 1:
                    ranges.append(cur_range)
                cur_range = []
            last_idx = idx 
            cur_range.append(idx)
        if len(cur_range) > 1:
            ranges.append(cur_range)

#        for arange in ranges:
#            for idx in arange:

#            valid_ends = np.where(ends > start)[0]
#            if valid_ends.size > 0:
#                ranges.append((start+1, ends[valid_ends[0]]))
#
        return ranges    

    def _make_feature_dict(self, sent, sent_id, n_sents):
        tokens = [token.token.decode(u'utf-8') for token in sent.tokens]
        feats = {}
        for t in tokens:
            feats[t] = feats.get(t, 0) + 1
        feats[u'__SID__'] = int(sent_id)
        feats[u'__POS__'] = float(sent_id) / float(n_sents)
        if len(tokens) >= 3:
            f = u'__LAST1_{}_{}_{}'.format(tokens[-3], tokens[-2], tokens[-1])
            feats[f] = 1
        if len(tokens) >= 2:
            f = u'__LAST1_{}_{}'.format(tokens[-2], tokens[-1])
            feats[f] = 1
        if len(tokens) >= 1:
            f = u'__LAST1_{}'.format(tokens[-1])
            feats[f] = 1

        return feats
