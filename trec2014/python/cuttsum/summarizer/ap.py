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

class APSalienceSummarizer(object):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'ap-sal-summaries')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_txt_path(self, event):
        return os.path.join(self.dir_, "hac-{}-txt.tsv.gz".format(
            event.fs_name()))

    def get_ssv_path(self, event):
        return os.path.join(self.dir_, "hac-{}.ssv".format(
            event.fs_name()))


    def make_summary(self, event, corpus, prefix, feature_set):
        string_res = get_resource_manager(u'SentenceStringsResource')
        lvec_res = get_resource_manager(u'SentenceLatentVectorsResource')
        spa = SaliencePredictionAggregator()
        Xcache = None

        txt_path = self.get_txt_path(event)
        ssv_path = self.get_ssv_path(event)
        epoch = datetime.utcfromtimestamp(0)
        with gzip.open(txt_path, u'w') as tf, open(ssv_path, u'w') as sf:
            for hour in event.list_event_hours():
                hp1 = hour + timedelta(hours=1)
                timestamp = str(int((hp1 - epoch).total_seconds()))

                string_df = string_res.get_dataframe(event, hour)     
                lvec_df = lvec_res.get_dataframe(event, hour)   
                sal_df = spa.get_dataframe(event, hour, prefix, feature_set)   
                
                if string_df is None or lvec_df is None or sal_df is None:
                    continue           
                print hour


                string_df.sort([u"stream id", u"sentence id"], inplace=True)
                lvec_df.sort([u"stream id", u"sentence id"], inplace=True)
                sal_df.sort([u"stream id", u"sentence id"], inplace=True)

                X = lvec_df.ix[:,2:].as_matrix()       
                good_rows = np.where(X.any(axis=1))[0]
                string_df = string_df.iloc[good_rows]
                lvec_df = lvec_df.iloc[good_rows]
                sal_df = sal_df.iloc[good_rows]
                assert len(string_df) == len(lvec_df)
                assert len(string_df) == len(sal_df)
                string_df = string_df.drop_duplicates(
                    subset=[u'stream id', u'sentence id'])

                lvec_df = lvec_df.drop_duplicates(
                    subset=[u'stream id', u'sentence id'])

                sal_df = sal_df.drop_duplicates(
                    subset=[u'stream id', u'sentence id'])

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

                        if words > 7 and len(doc) < 200 \
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
     
                X = lvec_df.ix[:,2:].as_matrix()       
                S = sal_df.ix[:,2:].as_matrix()
                s = np.mean(S, axis=1)
                #pmax = np.minimum(-7 + np.exp(-np.log(n_sents/100.)), -2)
                #print pmax
                P = MinMaxScaler(feature_range=(-9, -5)).fit_transform(s)
                A = cosine_similarity(X)
                A = MinMaxScaler(feature_range=(-3, -1)).fit_transform(A)
                
                std_s = StandardScaler().fit_transform(s)
                assert X.shape[0] == s.shape[0]

                #cutoff = 1. / 
                period = (((hour + timedelta(hours=6)) - \
                    event.start).total_seconds() // (6 * 3600))
                cutoff = 2. * period / (1. + period)
                af = AffinityPropagation(
                    preference=P, affinity='precomputed', max_iter=100,
                    damping=.7, verbose=True).fit(A)

                #cache = pd.DataFrame(columns=string_df.columns)
                #cache = []
                II = np.arange(n_sents)
                for cnum, cluster in enumerate(set(af.labels_)):
                    e = af.cluster_centers_indices_[cluster]
                    if II[cluster == af.labels_].shape[0] <= 3:
                        continue
                    if std_s[e] > cutoff:
                        if Xcache is None:
                            Xcache = X[e]
                            #cache.loc[cnum] = string_df.iloc[e]
                            sf.write(' '.join([
                                str(event.query_id),
                                "cunlp", "apsal",
                                string_df.iloc[e][u'stream id'],
                                str(string_df.iloc[e][u'sentence id']),
                                timestamp, "1\n"])) 
                            tf.write('\t'.join([
                                str(event.query_id),
                                "cunlp", "apsal",
                                string_df.iloc[e][u'stream id'],
                                str(string_df.iloc[e][u'sentence id']),
                                timestamp, "1", scstring + "\n"])) 

                        else:
                            if np.max(cosine_similarity(Xcache, X[e])) < .5:
                                scstring = string_df.iloc[e][u'streamcorpus']
                                sf.write(' '.join([
                                    str(event.query_id),
                                    "cunlp", "apsal",
                                    string_df.iloc[e][u'stream id'],
                                    str(string_df.iloc[e][u'sentence id']),
                                    timestamp, "1\n"])) 
                                tf.write('\t'.join([
                                    str(event.query_id),
                                    "cunlp", "apsal",
                                    string_df.iloc[e][u'stream id'],
                                    str(string_df.iloc[e][u'sentence id']),
                                    timestamp, "1", scstring + "\n"])) 


                                Xcache = np.vstack((Xcache, X[e]))


                #cache.sort(["stream id", "sentence id"], inplace=True)
                #for idx, row in cache.iterrows():
                    #print row['streamcorpus']
                    #for i in II[cluster == af.labels_]:
                    #    print string_df.iloc[i]['streamcorpus']
                    #print 
                #for e in af.cluster_centers_indices_:
                #        print string_df.iloc[e]['streamcorpus']
                #print hour - event.start, cutoff
                #print np.exp((hour - event.start).total_seconds() / -(24. * 3600))
                #print
