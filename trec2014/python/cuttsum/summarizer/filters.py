from cuttsum.sentsim import SentenceLatentVectorsResource
from cuttsum.summarizer.ap import APSummarizer, APSalienceSummarizer
from cuttsum.summarizer.baseline import HACSummarizer
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from cuttsum.salience import SaliencePredictionAggregator

class APFilteredSummary(object):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'ap-filtered-summaries')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_tsv_path(self, event, sim_cutoff, prefix=None, feature_set=None):
        return os.path.join(self.dir_,
            "ap-{}-sim_{}.tsv".format(event.fs_name(), sim_cutoff))

    def get_dataframe(self, event, sim_cutoff):
        tsv = self.get_tsv_path(event, sim_cutoff)
        if not os.path.exists(tsv):
            return None
        else:
            with open(tsv, u'r') as f:
                df = pd.io.parsers.read_csv(
                    f, sep='\t', quoting=3)
                df.columns = ['query id', 'system id', 'run id', 'stream id',
                              'sentence id', 'timestamp', 'conf', 'text']
            return df

    def make(self, event, min_cluster_size=2, sim_threshold=.2264):
        tsv_path = self.get_tsv_path(event, sim_threshold)
        lvecs = SentenceLatentVectorsResource()
        ap = APSummarizer()
        cluster_df = ap.get_dataframe(event)
       #for _, row in cluster_df.iterrows():
       #    print row['hour'], datetime.utcfromtimestamp(row['timestamp']) 
        updates = []
        Xcache = None
        timestamps = sorted(list(cluster_df['timestamp'].unique()))
        for timestamp in timestamps:
            hour = datetime.utcfromtimestamp(timestamp) - timedelta(hours=1)
            lvec_df = lvecs.get_dataframe(event, hour)
            lvec_df.drop_duplicates(['stream id', 'sentence id'], inplace=True)
            clusters = cluster_df[cluster_df['timestamp'] == timestamp].copy()
            clusters.sort(['stream id', 'sentence id'], inplace=True)
            


            for _, row in clusters.iterrows():
                if row['cluster size'] < min_cluster_size:
                    continue   
                vec = lvec_df.loc[
                    (lvec_df['stream id'] == row['stream id']) & \
                    (lvec_df['sentence id'] == row['sentence id'])].as_matrix()[:,2:].astype(np.float64)
                if Xcache is None:
                    Xcache = vec
                else:
                    if np.max(cosine_similarity(Xcache, vec)) >= sim_threshold:
                        continue
                    else:
                        Xcache = np.vstack((Xcache, vec))
                updates.append({
                    'query id': event.query_id[5:],
                    'system id': 'cunlp',
                    'run id': 'ap-sim_{}'.format(sim_threshold),
                    'stream id': row['stream id'], 
                    'sentence id': row['sentence id'],
                    'timestamp': timestamp,
                    'conf': 1.0,
                    'string': row['string']
                })

        df = pd.DataFrame(updates,
            columns=["query id", "system id", "run id",
                     "stream id", "sentence id", "timestamp", 
                     "conf", "string"])
        with open(tsv_path, u'w') as f:
            df.to_csv(
                f, sep='\t', index=False, index_label=False, header=False)

       #for timestamp, group in cluster_df.groupby(["timestamp"]):
       #    print timestamp

class APSalienceFilteredSummary(object):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'ap-sal-filtered-summaries')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_tsv_dir(self, prefix, feature_set):
        return os.path.join(self.dir_, prefix + "." + feature_set.fs_name())

    def get_tsv_path(self, event, prefix, feature_set, sal_cutoff, sim_cutoff):
        tsv_dir = self.get_tsv_dir(prefix, feature_set)
        return os.path.join(tsv_dir,
            "ap-sal-{}-sal_{}-sim_{}.tsv".format(
                event.fs_name(), sal_cutoff, sim_cutoff))

    def get_dataframe(self, event, prefix, feature_set, 
                      sal_cutoff, sim_cutoff):
        tsv = self.get_tsv_path(
            event, prefix, feature_set, sal_cutoff, sim_cutoff)
        if not os.path.exists(tsv):
            return None
        else:
            with open(tsv, u'r') as f:
                df = pd.io.parsers.read_csv(
                    f, sep='\t', quoting=3,
                    )
                df.columns = ['query id', 'system id', 'run id', 'stream id',
                              'sentence id', 'timestamp', 'conf', 'text']
            return df

    def make(self, event, prefix, feature_set, 
             min_cluster_size=2, sim_threshold=.2264, center_threshold=1.0):
        tsv_path = self.get_tsv_path(event, prefix, feature_set,
            center_threshold, sim_threshold)
        lvecs = SentenceLatentVectorsResource()
        spa = SaliencePredictionAggregator()
        apsal = APSalienceSummarizer()
        cluster_df = apsal.get_dataframe(event, prefix, feature_set)
       #for _, row in cluster_df.iterrows():
       #    print row['hour'], datetime.utcfromtimestamp(row['timestamp']) 
        updates = []
        Xcache = None
        timestamps = sorted(list(cluster_df['timestamp'].unique()))
        for timestamp in timestamps:
            hour = datetime.utcfromtimestamp(timestamp) - timedelta(hours=1)
            lvec_df = lvecs.get_dataframe(event, hour)
            lvec_df.drop_duplicates(['stream id', 'sentence id'], inplace=True)
            sal_df = spa.get_dataframe(event, hour, prefix, feature_set)            
            sal_df.drop_duplicates(['stream id', 'sentence id'], inplace=True)

            clusters = cluster_df[cluster_df['timestamp'] == timestamp].copy()
            clusters.sort(['stream id', 'sentence id'], inplace=True)

            salience = []
            for _, row in clusters.iterrows():
                sal_pred = sal_df.loc[
                    (sal_df['stream id'] == row['stream id']) & \
                    (sal_df['sentence id'] == row['sentence id'])].as_matrix()[:,2:].astype(np.float64).mean()
                salience.append(sal_pred)
            clusters['salience'] = salience
            clusters.sort(['salience'], ascending=False, inplace=True)

            sal_mean = np.mean(salience)
            sal_std = np.std(salience)


            #print clusters
            for _, row in clusters.iterrows():
                if row['cluster size'] < min_cluster_size:
                    continue   
                vec = lvec_df.loc[
                    (lvec_df['stream id'] == row['stream id']) & \
                    (lvec_df['sentence id'] == row['sentence id'])].as_matrix()[:,2:].astype(np.float64)

                sal_norm = (row['salience'] - sal_mean) / sal_std
                if sal_norm < center_threshold:
                    continue  

                if Xcache is None:
                    Xcache = vec
                else:
                    if np.max(cosine_similarity(Xcache, vec)) >= sim_threshold:
                        continue
                    else:
                        Xcache = np.vstack((Xcache, vec))
                updates.append({
                    'query id': event.query_id[5:],
                    'system id': 'cunlp',
                    'run id': 'apsal-sal_{}-sim_{}'.format(
                        center_threshold, sim_threshold),
                    'stream id': row['stream id'], 
                    'sentence id': row['sentence id'],
                    'timestamp': timestamp,
                    'conf': row['salience'],
                    'string': row['string']
                })

        df = pd.DataFrame(updates,
            columns=["query id", "system id", "run id",
                     "stream id", "sentence id", "timestamp", 
                     "conf", "string"])
        with open(tsv_path, u'w') as f:
            df.to_csv(
                f, sep='\t', index=False, index_label=False, header=False)


class HACFilteredSummary(object):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'hac-filtered-summaries')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_tsv_path(self, event, dist_cutoff, sim_cutoff):
        return os.path.join(self.dir_,
            "hac-{}-dist_{}-sim_{}.tsv".format(
                event.fs_name(), dist_cutoff, sim_cutoff))

    def get_dataframe(self, event, dist_cutoff, sim_cutoff):
        tsv = self.get_tsv_path(event, dist_cutoff, sim_cutoff)
        if not os.path.exists(tsv):
            return None
        else:
            with open(tsv, u'r') as f:
                df = pd.io.parsers.read_csv(
                    f, sep='\t', quoting=3,
                    )
                df.columns = ['query id', 'system id', 'run id', 'stream id',
                              'sentence id', 'timestamp', 'conf', 'text']
            return df

    def make(self, event, prefix, feature_set,
             min_cluster_size=2, sim_threshold=.2264,
             dist_cutoff=1.35):
        tsv_path = self.get_tsv_path(event, dist_cutoff, sim_threshold)
        lvecs = SentenceLatentVectorsResource()
        #spa = SaliencePredictionAggregator()
        hac = HACSummarizer()
        cluster_df = hac.get_dataframe(event, dist_cutoff)
       #for _, row in cluster_df.iterrows():
       #    print row['hour'], datetime.utcfromtimestamp(row['timestamp']) 
        updates = []
        Xcache = None
        timestamps = sorted(list(cluster_df['timestamp'].unique()))
        for timestamp in timestamps:
            hour = datetime.utcfromtimestamp(timestamp) - timedelta(hours=1)
            lvec_df = lvecs.get_dataframe(event, hour)
            lvec_df.drop_duplicates(['stream id', 'sentence id'], inplace=True)
            
#            sal_df = spa.get_dataframe(event, hour, prefix, feature_set)  
#            sal_df.drop_duplicates(['stream id', 'sentence id'], inplace=True)
            
            clusters = cluster_df[cluster_df['timestamp'] == timestamp].copy()
            clusters.sort(['stream id', 'sentence id'], inplace=True)

#            salience = []
#            for _, row in clusters.iterrows():
#                sal_pred = sal_df.loc[
#                    (sal_df['stream id'] == row['stream id']) & \
#                    (sal_df['sentence id'] == row['sentence id'])
#                    ].as_matrix()[:,2:].astype(np.float64).mean()
#                salience.append(sal_pred)
#            clusters['salience'] = salience
#            clusters.sort(['salience'], ascending=False, inplace=True)

#            sal_mean = np.mean(salience)
#            sal_std = np.std(salience)


            for _, row in clusters.iterrows():
                if row['cluster size'] < min_cluster_size:
                    continue   
                vec = lvec_df.loc[
                    (lvec_df['stream id'] == row['stream id']) & \
                    (lvec_df['sentence id'] == row['sentence id'])
                    ].as_matrix()[:,2:].astype(np.float64)

                #sal_norm = (row['salience'] - sal_mean) / sal_std
                #if sal_norm < center_threshold:
                #    continue

                if Xcache is None:
                    Xcache = vec
                else:
                    if np.max(cosine_similarity(Xcache, vec)) >= sim_threshold:
                        continue
                    else:
                        Xcache = np.vstack((Xcache, vec))
                updates.append({
                    'query id': event.query_id[5:],
                    'system id': 'cunlp',
                    'run id': 'hac-dist_{}-sim_{}'.format(
                        dist_cutoff, sim_threshold),
                    'stream id': row['stream id'], 
                    'sentence id': row['sentence id'],
                    'timestamp': timestamp,
                    'conf': 1.0,
                    'string': row['string']
                })

        df = pd.DataFrame(updates,
            columns=["query id", "system id", "run id",
                     "stream id", "sentence id", "timestamp", 
                     "conf", "string"])
        with open(tsv_path, u'w') as f:
            df.to_csv(
                f, sep='\t', index=False, index_label=False, header=False)

