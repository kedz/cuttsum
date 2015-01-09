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

class HACSummarizer(object):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'hac-summaries')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_txt_path(self, event):
        return os.path.join(self.dir_, "hac-{}-txt.tsv.gz".format(
            event.fs_name()))

    def get_ssv_path(self, event):
        return os.path.join(self.dir_, "hac-{}.ssv.gz".format(
            event.fs_name()))

    def make_summary(self, event, corpus):
        string_res = get_resource_manager(u'SentenceStringsResource')
        lvec_res = get_resource_manager(u'SentenceLatentVectorsResource')
        txt_path = self.get_txt_path(event)
        ssv_path = self.get_ssv_path(event)
        epoch = datetime.utcfromtimestamp(0)

        with gzip.open(txt_path, u'w') as tf, gzip.open(ssv_path, u'w') as sf:

            hours = event.list_event_hours()
            n_hours = len(hours)
            pb = ProgressBar(n_hours)
            for hour in event.list_event_hours():
                pb.update()
                hp1 = hour + timedelta(hours=1)
                timestamp = str(int((hp1 - epoch).total_seconds()))

                string_df = string_res.get_dataframe(event, hour)                 
                lvec_df = lvec_res.get_dataframe(event, hour)      
                if string_df is None or lvec_df is None:
                    continue           

                string_df.sort([u"stream id", u"sentence id"], inplace=True)
                lvec_df.sort([u"stream id", u"sentence id"], inplace=True)

                X = lvec_df.ix[:,2:].as_matrix()       
                good_rows = np.where(X.any(axis=1))[0]
                string_df = string_df.iloc[good_rows]
                lvec_df = lvec_df.iloc[good_rows]
                assert len(string_df) == len(lvec_df)
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

                good_rows = []
                for name, doc in string_df.groupby("stream id"):
                    for rname, row in doc.iterrows():


                        scstring = row["streamcorpus"]
                        #scstring = doc.iloc[i]["streamcorpus"]
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

                        if words > 6 and len(doc) < 200 \
                            and socs < 2 and langs < 2:
                            
                            good_rows.append(rname)
                            #print lvec_df.loc[rname][2:].as_list()
                    #print "\n--"
                
                lvec_df = lvec_df.loc[good_rows]
                string_df = string_df.loc[good_rows]
                n_sents = len(string_df)
                
                for i in xrange(n_sents):
                    assert string_df[u'stream id'].iloc[i] == \
                        lvec_df[u'stream id'].iloc[i]
                    assert string_df[u'sentence id'].iloc[i] == \
                        lvec_df[u'sentence id'].iloc[i]

                X = lvec_df.ix[:,2:].as_matrix()       
                if X.shape[0] < 10:
                    continue

                X = Normalizer().fit_transform(X)
                z = hac.linkage(X, method='average', metric='euclidean')        
                clusters = hac.fcluster(z, 1.35, 'distance') 
                #print len(set(clusters)), n_sents
                II = np.arange(n_sents)
                for cluster_id, cluster in enumerate(set(clusters)):
                    ii = II[clusters == cluster]
                    C = X[clusters == cluster,:]
                    if C.shape[0] == 1:
                        continue
                    u = np.mean(C, axis=0)
                    dist_2 = np.sum((C - u)**2, axis=1)
                    cidx = np.argmin(dist_2)
                  
 
                    #11 cunlp 1APSalRed 1326506940-433f8576b39b614c312c77c77f739ee3 157 1326510000 2.65524760187
                    stream_id = str(lvec_df.iloc[ii[cidx]][u'stream id'])
                    sentence_id = str(lvec_df.iloc[ii[cidx]][u'sentence id'])
                    scstring = \
                        string_df.iloc[ii[cidx]]['streamcorpus']
                    sf.write(' '.join(
                        [event.query_id, "cunlp", "hac",
                         stream_id, sentence_id, timestamp, "1\n"]))
                    tf.write('\t'.join(
                        [str(event.query_id), "cunlp", "hac",
                         stream_id, sentence_id, 
                         timestamp, "1", scstring + "\n"]))

                    #print "CLUSTER:", cluster_id
                    #print string_df.iloc[ii[cidx]]['streamcorpus']

