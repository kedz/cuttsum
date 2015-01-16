from ..data import get_resource_manager
import re
from itertools import izip
import scipy.cluster.hierarchy as hac
from sklearn.preprocessing import Normalizer
import numpy as np
import os
import gzip
from datetime import datetime, timedelta
from ..misc import ProgressBar
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def dt_cvrt(x): return datetime.utcfromtimestamp(int(x))

class HACSummarizer(object):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'hac-summaries')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)


    def get_txt_path(self, event, prefix=None, feature_set=None):
        return os.path.join(self.dir_,
            "hac-{}-txt.tsv.gz".format(event.fs_name()))


    def get_ssv_path(self, event, prefix=None, feature_set=None):
        return os.path.join(self.dir_,
            "hac-{}.ssv".format(event.fs_name()))

    def get_txt_dataframe(self, event, prefix=None, feature_set=None):
        txt_path = self.get_txt_path(event, prefix, feature_set)
        with gzip.open(txt_path, u'r') as f:
            df = pd.io.parsers.read_csv(
                f, sep='\t', quoting=3, header=None,
                converters={u'query id': lambda x: 'TS14.{}'.format(x),
                            u'update datetime': dt_cvrt},
                names=[u'query id', u'system id', u'run id', u'document id',
                       u'sentence id', u'update datetime', u'confidence', 
                       u'text'])
            df.insert(3, u'update id',
                df[u'document id'].map(str) + '-' + df[u'sentence id'].map(str))

            return df


    def make_summary(self, event, corpus, prefix, feature_set):
        string_res = get_resource_manager(u'SentenceStringsResource')
        lvec_res = get_resource_manager(u'SentenceLatentVectorsResource')
        txt_path = self.get_txt_path(event)
        ssv_path = self.get_ssv_path(event)
        epoch = datetime.utcfromtimestamp(0)
        Xcache = None

        with gzip.open(txt_path, u'w') as tf, open(ssv_path, u'w') as sf:

            hours = event.list_event_hours()
            n_hours = len(hours)
            for hour in event.list_event_hours():
                hp1 = hour + timedelta(hours=1)
                timestamp = str(int((hp1 - epoch).total_seconds()))

                string_df = string_res.get_dataframe(event, hour)
                lvec_df = lvec_res.get_dataframe(event, hour)      
                if string_df is None or lvec_df is None:
                    continue           

#                string_df.sort([u"stream id", u"sentence id"], inplace=True)
#                lvec_df.sort([u"stream id", u"sentence id"], inplace=True)
#
#                X = lvec_df.ix[:,2:].as_matrix()       
#                good_rows = np.where(X.any(axis=1))[0]
#                string_df = string_df.iloc[good_rows]
#                lvec_df = lvec_df.iloc[good_rows]
#                assert len(string_df) == len(lvec_df)
#                string_df = string_df.drop_duplicates(
#                    subset=[u'stream id', u'sentence id'])
#
#                lvec_df = lvec_df.drop_duplicates(
#                    subset=[u'stream id', u'sentence id'])
#                n_sents = len(string_df)
#                                   
#                for i in xrange(n_sents):
#                    assert string_df[u'stream id'].iloc[i] == \
#                        lvec_df[u'stream id'].iloc[i]
#                    assert string_df[u'sentence id'].iloc[i] == \
#                        lvec_df[u'sentence id'].iloc[i]
#
#                good_rows = []
#                for name, doc in string_df.groupby("stream id"):
#                    for rname, row in doc.iterrows():
#
#
#                        scstring = row["streamcorpus"]
#                        #scstring = doc.iloc[i]["streamcorpus"]
#                        words = len(re.findall(r'\b[^\W\d_]+\b', scstring))
#                        socs = len(re.findall(
#                            r'Digg|del\.icio\.us|Facebook|Kwoff|Myspace',
#                            scstring))  
#                        langs = len(re.findall(
#                            r'Flash|JavaScript|CSS', scstring, re.I))
#                        
#                        assert lvec_df.loc[rname][u'sentence id'] == \
#                            row[u'sentence id']
#                        assert lvec_df.loc[rname][u'stream id'] == \
#                            row[u'stream id']
#
#                        if words > 6 and len(doc) < 200 \
#                            and socs < 2 and langs < 2:
#                            
#                            good_rows.append(rname)
#                            #print lvec_df.loc[rname][2:].as_list()
#                    #print "\n--"
#                
#                lvec_df = lvec_df.loc[good_rows]
#                string_df = string_df.loc[good_rows]
#                n_sents = len(string_df)
#                
#                for i in xrange(n_sents):
#                    assert string_df[u'stream id'].iloc[i] == \
#                        lvec_df[u'stream id'].iloc[i]
#                    assert string_df[u'sentence id'].iloc[i] == \
#                        lvec_df[u'sentence id'].iloc[i]
#
#                X = lvec_df.ix[:,2:].as_matrix()       
#                if X.shape[0] < 10:
#                    continue

                string_df.sort([u"stream id", u"sentence id"], inplace=True)
                lvec_df.sort([u"stream id", u"sentence id"], inplace=True)
                #sal_df.sort([u"stream id", u"sentence id"], inplace=True)

                X = lvec_df.ix[:,2:].as_matrix()       
                good_rows = np.where(X.any(axis=1))[0]
                string_df = string_df.iloc[good_rows]
                lvec_df = lvec_df.iloc[good_rows]
                #sal_df = sal_df.iloc[good_rows]
                assert len(string_df) == len(lvec_df)
                #assert len(string_df) == len(sal_df)
                string_df = string_df.drop_duplicates(
                    subset=[u'stream id', u'sentence id'])

                lvec_df = lvec_df.drop_duplicates(
                    subset=[u'stream id', u'sentence id'])

                n_sents = len(string_df)
                                   
                for i in xrange(n_sents):
                    assert string_df[u'stream id'].iloc[i] == \
                        lvec_df[u'stream id'].iloc[i]
                    assert string_df[u'sentence id'].iloc[i] == \
                        lvec_df[u'sentence id'].iloc[i]

                lvec_df.reset_index(drop=True, inplace=True) 
                string_df.reset_index(drop=True, inplace=True) 
                good_rows = []
                for name, doc in string_df.groupby("stream id"):
                    for rname, row in doc.iterrows():
                        scstring = row["streamcorpus"]
                        words = len(re.findall(r'\b[^\W\d_]+\b', scstring))
                        socs = len(re.findall(
                            r'Digg|del\.icio\.us|Facebook|Kwoff|Myspace',
                            scstring))  
                        langs = len(re.findall(
                            r'Flash|JavaScript|CSS', scstring, re.I))

                        assert lvec_df.loc[rname][u'sentence id'] == \
                            row[u'sentence id']
                        assert lvec_df.loc[rname][u'stream id'] == \
                            row[u'stream id']

                        if words > 9 and len(doc) < 200 \
                            and socs < 2 and langs < 2:
                            
                            good_rows.append(rname)
                
                lvec_df = lvec_df.loc[good_rows]
                string_df = string_df.loc[good_rows]
                n_sents = len(string_df)
                if n_sents < 10:
                    continue

                for i in xrange(n_sents):
                    assert string_df[u'stream id'].iloc[i] == \
                        lvec_df[u'stream id'].iloc[i]
                    assert string_df[u'sentence id'].iloc[i] == \
                        lvec_df[u'sentence id'].iloc[i]

                X = lvec_df.ix[:,2:].as_matrix()       
                Xn = Normalizer().fit_transform(X)
                z = hac.linkage(Xn, method='average', metric='euclidean')        
                clusters = hac.fcluster(z, 1.35, 'distance') 
                II = np.arange(n_sents)
                #print set(clusters)
                for cluster_id, cluster in enumerate(set(clusters)):
                #    print cluster
                #    print (clusters == cluster).shape
                #    print II.shape
                    ii = II[clusters == cluster]
                    #print ii.shape
                    C = X[clusters == cluster,:]
                    if C.shape[0] <= 3:
                        #print "Too small"
                        continue
                    u = np.mean(C, axis=0)
                    #dist_2 = np.sum((C - u)**2, axis=1)
                    #cidx = np.argmin(dist_2)
                    cidx = np.argmax(cosine_similarity(C, u))
                    e = ii[cidx]
                    if Xcache is None:
                        Xcache = X[e]
                    else: 
                        if np.max(cosine_similarity(Xcache, X[e])) >= .5:
                            #print "Too similar"
                            continue
                        else:
                            Xcache = np.vstack((Xcache, X[e]))

                    stream_id = str(lvec_df.iloc[e][u'stream id'])
                    sentence_id = str(lvec_df.iloc[e][u'sentence id'])
                    scstring = \
                        string_df.iloc[ii[cidx]]['streamcorpus']
                    sf.write(' '.join(
                        [str(event.query_id).split(".")[1], "cunlp", "hac",
                         stream_id, sentence_id, timestamp, "1\n"]))
                    tf.write('\t'.join(
                        [str(event.query_id).split(".")[1], "cunlp", "hac",
                         stream_id, sentence_id, 
                         timestamp, "1", scstring + "\n"]))

__dt_cvrt = lambda x: datetime.utcfromtimestamp(int(x))
