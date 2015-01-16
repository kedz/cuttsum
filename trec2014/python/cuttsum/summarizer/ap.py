import os
from ..data import get_resource_manager
from ..pipeline.salience import SaliencePredictionAggregator
import gzip
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import AffinityPropagation
from datetime import datetime, timedelta
import pandas as pd

class APSummarizer(object):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'ap-summaries')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

#    def get_txt_dir(self, prefix, feature_set):
#        return os.path.join(self.dir_, prefix + "." + feature_set.fs_name())
#
#    def get_txt_path(self, event, prefix, feature_set):
#        txt_dir = self.get_txt_dir(prefix, feature_set)
#        return os.path.join(txt_dir,
#            "ap-sal-{}-txt.tsv.gz".format(event.fs_name()))
#
#    def get_ssv_dir(self, prefix, feature_set):
#        return os.path.join(self.dir_, prefix + "." + feature_set.fs_name())

    def get_tsv_path(self, event, prefix=None, feature_set=None):
        return os.path.join(self.dir_,
            "ap-{}.tsv.gz".format(event.fs_name()))

    def make_summary(self, event, corpus, prefix, feature_set):
        string_res = get_resource_manager(u'SentenceStringsResource')
        lvec_res = get_resource_manager(u'SentenceLatentVectorsResource')

        tsv_path = self.get_tsv_path(event)
        updates = []
        epoch = datetime.utcfromtimestamp(0)
        for hour in event.list_event_hours():
            hp1 = hour + timedelta(hours=1)
            timestamp = str(int((hp1 - epoch).total_seconds()))

            string_df = string_res.get_dataframe(event, hour)     
            lvec_df = lvec_res.get_dataframe(event, hour)   
            
            if string_df is None or lvec_df is None:
                continue           

            string_df = string_df.drop_duplicates(
                subset=[u'stream id', u'sentence id'])
            lvec_df = lvec_df.drop_duplicates(
                subset=[u'stream id', u'sentence id'])
            string_df.sort([u"stream id", u"sentence id"], inplace=True)
            lvec_df.sort([u"stream id", u"sentence id"], inplace=True)

            X = lvec_df.as_matrix()[:,2:].astype(np.float64)    
            good_rows = np.where(X.any(axis=1))[0]
            string_df = string_df.iloc[good_rows]
            lvec_df = lvec_df.iloc[good_rows]
            assert len(string_df) == len(lvec_df)

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
 
            X = lvec_df.as_matrix()[:,2:].astype(np.float64)    
            A = cosine_similarity(X)
            Aupper = A[np.triu_indices_from(A, k=1)]
            Amu = np.mean(Aupper)
            Astd = np.std(Aupper)

            A = (A - Amu) / Astd
            A = A - np.max(A[np.triu_indices_from(A, k=1)])

            af = AffinityPropagation(
                preference=None, affinity='precomputed', max_iter=100,
                damping=.7, verbose=False).fit(A)

            II = np.arange(n_sents)
            for cnum, cluster in enumerate(set(af.labels_)):
                e = af.cluster_centers_indices_[cluster]
                cluster_size = II[cluster == af.labels_].shape[0]
                   
                scstring = string_df.iloc[e][u'streamcorpus']
                stream_id = string_df.iloc[e][u'stream id']
                sentence_id = str(string_df.iloc[e][u'sentence id'])
                updates.append({"stream id": stream_id,
                                "sentence id": sentence_id,
                                "hour": hour, 
                                "timestamp": timestamp,
                                "cluster size": cluster_size,
                                "string": scstring})
        df = pd.DataFrame(updates, 
            columns=["stream id", "sentence id", "hour", "timestamp",
                     "cluster size", "string"])
        with gzip.open(tsv_path, u'w') as f:
            df.to_csv(f, sep='\t', index=False, index_label=False)

class APSalienceSummarizer(object):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'ap-sal-summaries')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_tsv_dir(self, prefix, feature_set):
        return os.path.join(self.dir_, prefix + "." + feature_set.fs_name())
#
#    def get_txt_path(self, event, prefix, feature_set):
#        txt_dir = self.get_txt_dir(prefix, feature_set)
#        return os.path.join(txt_dir,
#            "ap-sal-{}-txt.tsv.gz".format(event.fs_name()))
#
#    def get_ssv_dir(self, prefix, feature_set):
#        return os.path.join(self.dir_, prefix + "." + feature_set.fs_name())

    def get_tsv_path(self, event, prefix, feature_set):
        tsv_dir = self.get_tsv_dir(prefix, feature_set)
        return os.path.join(tsv_dir,
            "ap-sal-{}.tsv.gz".format(event.fs_name()))

    def make_summary(self, event, corpus, prefix, feature_set):
        string_res = get_resource_manager(u'SentenceStringsResource')
        lvec_res = get_resource_manager(u'SentenceLatentVectorsResource')
        spa = SaliencePredictionAggregator()

        tsv_path = self.get_tsv_path(event, prefix, feature_set)
        updates = []
        epoch = datetime.utcfromtimestamp(0)
        for hour in event.list_event_hours():
            hp1 = hour + timedelta(hours=1)
            timestamp = str(int((hp1 - epoch).total_seconds()))

            string_df = string_res.get_dataframe(event, hour)     
            lvec_df = lvec_res.get_dataframe(event, hour)   
            sal_df = spa.get_dataframe(event, hour, prefix, feature_set)   
            
            if string_df is None or lvec_df is None or sal_df is None:
                continue           

            string_df = string_df.drop_duplicates(
                subset=[u'stream id', u'sentence id'])

            lvec_df = lvec_df.drop_duplicates(
                subset=[u'stream id', u'sentence id'])

            sal_df = sal_df.drop_duplicates(
                subset=[u'stream id', u'sentence id'])

            string_df.sort([u"stream id", u"sentence id"], inplace=True)
            lvec_df.sort([u"stream id", u"sentence id"], inplace=True)
            sal_df.sort([u"stream id", u"sentence id"], inplace=True)

            X = lvec_df.as_matrix()[:,2:].astype(np.float64)    
            good_rows = np.where(X.any(axis=1))[0]
            string_df = string_df.iloc[good_rows]
            lvec_df = lvec_df.iloc[good_rows]
            sal_df = sal_df.iloc[good_rows]
            assert len(string_df) == len(lvec_df)
            assert len(string_df) == len(sal_df)
            
            n_sents = len(string_df)
                               
            for i in xrange(n_sents):
                assert string_df[u'stream id'].iloc[i] == \
                    lvec_df[u'stream id'].iloc[i]
                assert string_df[u'stream id'].iloc[i] == \
                    sal_df[u'stream id'].iloc[i]
                assert string_df[u'sentence id'].iloc[i] == \
                    lvec_df[u'sentence id'].iloc[i]
                assert string_df[u'sentence id'].iloc[i] == \
                    sal_df[u'sentence id'].iloc[i]

            lvec_df.reset_index(drop=True, inplace=True) 
            string_df.reset_index(drop=True, inplace=True) 
            sal_df.reset_index(drop=True, inplace=True) 
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

                    assert sal_df.loc[rname][u'sentence id'] == \
                        row[u'sentence id']
                    assert sal_df.loc[rname][u'stream id'] == \
                        row[u'stream id']

                    if words > 9 and len(doc) < 200 \
                        and socs < 2 and langs < 2:
                        
                        good_rows.append(rname)
            
            lvec_df = lvec_df.loc[good_rows]
            string_df = string_df.loc[good_rows]
            sal_df = sal_df.loc[good_rows]
            n_sents = len(string_df)
            if n_sents < 10:
                continue

            for i in xrange(n_sents):
                assert string_df[u'stream id'].iloc[i] == \
                    lvec_df[u'stream id'].iloc[i]
                assert string_df[u'stream id'].iloc[i] == \
                    sal_df[u'stream id'].iloc[i]
                assert string_df[u'sentence id'].iloc[i] == \
                    lvec_df[u'sentence id'].iloc[i]
                assert string_df[u'sentence id'].iloc[i] == \
                    sal_df[u'sentence id'].iloc[i]
 
            X = lvec_df.as_matrix()[:,2:].astype(np.float64) 
            S = sal_df.as_matrix()[:,2:].astype(np.float64)
            s = np.mean(S, axis=1)
            A = cosine_similarity(X)
            Aupper = A[np.triu_indices_from(A, k=1)]
            Amu = np.mean(Aupper)
            Astd = np.std(Aupper)

            A = (A - Amu) / Astd
            

            max_sim = np.max(
                (np.max(A[np.triu_indices_from(A, k=1)]), np.max(s)))
            A = A - max_sim
            P = s - max_sim
            
            
            #P = MinMaxScaler(feature_range=(-9, -5)).fit_transform(s)
            #A = MinMaxScaler(feature_range=(-3, -1)).fit_transform(A)
            
            #std_s = StandardScaler().fit_transform(s)
            #assert X.shape[0] == s.shape[0]

            #period = (((hour + timedelta(hours=6)) - \
            #    event.start).total_seconds() // (6 * 3600))
            #cutoff = 2. * period / (1. + period)
            af = AffinityPropagation(
                preference=P, affinity='precomputed', max_iter=100,
                damping=.7, verbose=False).fit(A)

            II = np.arange(n_sents)
            for cnum, cluster in enumerate(set(af.labels_)):
                e = af.cluster_centers_indices_[cluster]
                cluster_size = II[cluster == af.labels_].shape[0]

                scstring = string_df.iloc[e][u'streamcorpus']
                stream_id = string_df.iloc[e][u'stream id']
                sentence_id = str(string_df.iloc[e][u'sentence id'])
                updates.append({"stream id": stream_id,
                                "sentence id": sentence_id,
                                "hour": hour, 
                                "timestamp": timestamp,
                                "cluster size": cluster_size,
                                "string": scstring})
        df = pd.DataFrame(updates, 
            columns=["stream id", "sentence id", "hour", "timestamp",
                     "cluster size", "string"])
        with gzip.open(tsv_path, u'w') as f:
            df.to_csv(f, sep='\t', index=False, index_label=False)

                   
                #if std_s[e] > cutoff:
                #    if Xcache is None:
                #        Xcache = X[e]
               #     else: 
               #         if np.max(cosine_similarity(Xcache, X[e])) >= .5:
               #             continue
               #         else:
               #             Xcache = np.vstack((Xcache, X[e]))
               #     scstring = string_df.iloc[e][u'streamcorpus']
               #     sf.write(' '.join([
                #        str(event.query_id).split(".")[1],
                 #       "cunlp", "apsal",
                  #      string_df.iloc[e][u'stream id'],
                   #     str(string_df.iloc[e][u'sentence id']),
                   #     timestamp, str(s[e]) + "\n"])) 
                 #   tf.write('\t'.join([
                 #       str(event.query_id).split(".")[1],
                  #      "cunlp", "apsal",
                  #      string_df.iloc[e][u'stream id'],
                   #     str(string_df.iloc[e][u'sentence id']),
                    #    timestamp, str(s[e]), scstring + "\n"])) 
