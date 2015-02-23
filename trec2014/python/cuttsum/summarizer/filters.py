from cuttsum.sentsim import SentenceLatentVectorsResource
from cuttsum.summarizer.ap import APSummarizer, APSalienceSummarizer
from cuttsum.summarizer.baseline import HACSummarizer
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from cuttsum.salience import SaliencePredictionAggregator
from cuttsum.data import get_resource_manager
from cuttsum.misc import passes_simple_filter


class RankedSalienceFilteredSummary(object):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), 
            u'ranked-salience-filtered-summaries')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_tsv_path(self, event, sal_cutoff, sim_cutoff):
        return os.path.join(self.dir_, 
            "sal-ranked-{}-sal_{}-sim_{}.tsv".format(
            event.fs_name(), sal_cutoff, sim_cutoff))

    def get_dataframe(self, event, sal_cutoff, sim_cutoff):
        tsv = self.get_tsv_path(event, sal_cutoff, sim_cutoff)
        if not os.path.exists(tsv):
            return None
        else:
            with open(tsv, u'r') as f:
                df = pd.io.parsers.read_csv(
                    f, sep='\t', quoting=3)
                df.columns = ['query id', 'system id', 'run id', 'stream id',
                              'sentence id', 'timestamp', 'conf', 'text']
            return df

    def make(self, event, prefix, feature_set, sal_cutoff, sim_cutoff):
        tsv_path = self.get_tsv_path(event, sal_cutoff, sim_cutoff)
        lvecs = SentenceLatentVectorsResource()
        string_res = get_resource_manager(u'SentenceStringsResource')

        spa = SaliencePredictionAggregator()
        #cluster_df = hac.get_dataframe(event, dist_cutoff)
       #for _, row in cluster_df.iterrows():
       #    print row['hour'], datetime.utcfromtimestamp(row['timestamp']) 
        epoch = datetime.utcfromtimestamp(0)
        updates = []
        Xcache = None
        #timestamps = sorted(list(cluster_df['timestamp'].unique()))
        hours = event.list_event_hours()
        for hour in hours:
            #hour = datetime.utcfromtimestamp(timestamp) - timedelta(hours=1)
            hp1 = hour + timedelta(hours=1)
            timestamp = str(int((hp1 - epoch).total_seconds()))
            lvec_df = lvecs.get_dataframe(event, hour)
            sal_df = spa.get_dataframe(event, hour, prefix, feature_set)
            str_df = string_res.get_dataframe(event, hour)
            if lvec_df is None or sal_df is None or str_df is None:
                continue
            str_df.drop_duplicates(['stream id', 'sentence id'], inplace=True)
                
            lvec_df.drop_duplicates(['stream id', 'sentence id'], inplace=True)
            sal_df.drop_duplicates(['stream id', 'sentence id'], inplace=True)
            
            
            str_df.sort(['stream id', 'sentence id'], inplace=True)
            str_df.reset_index(drop=True, inplace=True)
            lvec_df.sort(['stream id', 'sentence id'], inplace=True)
            lvec_df.reset_index(drop=True, inplace=True)
            sal_df.sort(['stream id', 'sentence id'], inplace=True)
            sal_df.reset_index(drop=True, inplace=True)

            str_df = str_df.join(
                str_df.groupby('stream id')['sentence id'].agg('count'), 
                on='stream id', rsuffix='_r').rename(
                    columns={"sentence id_r": "document count"})
            good_sents = str_df.apply(
                lambda x: passes_simple_filter(
                    x['streamcorpus'], x['document count']), axis=1)
            #good_sents = good_sents.reset_index()

            str_df = str_df[good_sents] 
            lvec_df = lvec_df[good_sents] 
            sal_df = sal_df[good_sents]
             
            n_rows = len(sal_df)
            for i in xrange(n_rows):
                assert sal_df['stream id'].iloc[i] == \
                    lvec_df['stream id'].iloc[i]
                assert sal_df['sentence id'].iloc[i] == \
                    lvec_df['sentence id'].iloc[i]
                
                assert str_df['stream id'].iloc[i] == \
                    lvec_df['stream id'].iloc[i]
                assert str_df['sentence id'].iloc[i] == \
                    lvec_df['sentence id'].iloc[i]
             
            if n_rows == 0:
                continue

            Xsal = sal_df.as_matrix()[:,2:].astype(np.float64).mean(axis=1)
            mu_sal = np.mean(Xsal)
            sig_sal = np.std(Xsal)
            Xsal_norm = (Xsal - mu_sal) / sig_sal
            
            lvec_df = lvec_df[Xsal_norm > sal_cutoff]
            str_df = str_df[Xsal_norm > sal_cutoff]
            str_df = str_df.set_index(['stream id', 'sentence id'])
            lvec_df['salience'] = Xsal_norm[Xsal_norm > sal_cutoff]
            lvec_df.sort(['salience'], inplace=True, ascending=False)
            if Xcache is None:
                Xlvecs = lvec_df.as_matrix()[:, 2:-2].astype(np.float64)

                K = cosine_similarity(Xlvecs)
                K_ma = np.ma.array(K, mask=True)
                good_indexes = []
                for i, (_, row) in enumerate(lvec_df.iterrows()):
                    sim = K_ma[i,:].max(fill_value=0.0)    
                    if not isinstance(sim, np.float64):
                        sim = 0
                    if sim < sim_cutoff:
                        up_str = str_df.loc[
                            row['stream id'],
                            row['sentence id']]['streamcorpus']
                        updates.append({
                            'query id': event.query_num,
                            'system id': 'cunlp',
                            'run id': 'sal-ranked-sal_{}-sim_{}'.format(
                                sal_cutoff, sim_cutoff),
                            'stream id': row['stream id'], 
                            'sentence id': row['sentence id'],
                            'timestamp': timestamp,
                            'conf': row['salience'],
                            'string': up_str}) 
                        K_ma.mask[:,i] = False
                        good_indexes.append(i)
                    
                Xcache = Xlvecs[good_indexes].copy()
            else:
                xtmp = lvec_df.as_matrix()[:, 2:-2].astype(np.float64)

                Xlvecs = np.vstack([Xcache, xtmp])
                
                start_index = Xcache.shape[0]
                K = cosine_similarity(Xlvecs)
                K_ma = np.ma.array(K, mask=True)
                K_ma.mask.T[np.arange(0, start_index)] = False
                good_indexes = []
                for i, (_, row) in enumerate(lvec_df.iterrows(), start_index):
                    #print i
                    sim = K_ma[i,:].max(fill_value=0.0)    
                    #print sim
                    if sim < sim_cutoff:
                        up_str = str_df.loc[
                            row['stream id'],
                            row['sentence id']]['streamcorpus']
                        updates.append({
                            'query id': event.query_num,
                            'system id': 'cunlp',
                            'run id': 'sal-ranked-sal_{}-sim_{}'.format(
                                sal_cutoff, sim_cutoff),
                            'stream id': row['stream id'], 
                            'sentence id': row['sentence id'],
                            'timestamp': timestamp,
                            'conf': row['salience'],
                            'string': up_str}) 
                        K_ma.mask[:,i] = False
                        good_indexes.append(i)
                if len(good_indexes) > 0:
                    Xcache = np.vstack([Xcache, Xlvecs[good_indexes].copy()])
                #Xcache = Xlvecs[good_indexes].copy()

        if len(updates) == 0:
            updates.append({
                'query id': event.query_num,
                'system id': 'cunlp',
                'run id': 'sal-ranked-sal_{}-sim_{}'.format(
                    sal_cutoff, sim_cutoff),
                'stream id': 1111, 
                'sentence id': 1,
                'timestamp': timestamp,
                'conf': 0,
                'string': 'place holder'}) 

        df = pd.DataFrame(updates,
            columns=["query id", "system id", "run id",
                     "stream id", "sentence id", "timestamp", 
                     "conf", "string"])
        with open(tsv_path, u'w') as f:
            df.to_csv(
                f, sep='\t', index=False, index_label=False, header=False)






            #str_df['salience'] = Xsal_norm 
            
            
            #sal_norm = (row['salience'] - sal_mean) / sal_std
            #if sal_norm < center_threshold:
            #        continue  



            #salience = []
            #for _, row in clusters.iterrows():
            #    sal_pred = sal_df.loc[
            #        (sal_df['stream id'] == row['stream id']) & \
            #        (sal_df['sentence id'] == row['sentence id'])].as_matrix()[:,2:].astype(np.float64).mean()
            #    salience.append(sal_pred)
            #clusters['salience'] = salience
                 

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

class APSalTRankSalThreshFilteredSummary(object):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), 
            u'ap-sal-time-ranked-sal-thr-filtered-summaries')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_tsv_dir(self, prefix, feature_set):
        return os.path.join(self.dir_, prefix + "." + feature_set.fs_name())

    def get_tsv_path(self, event, prefix, feature_set, sal_cutoff, sim_cutoff):
        tsv_dir = self.get_tsv_dir(prefix, feature_set)
        return os.path.join(tsv_dir,
            "ap-sal-time-ranked-sal-thr-{}-sal_{}-sim_{}.tsv".format(
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
            #clusters.sort(['salie'], inplace=True)

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
                    'run id': 'apsal-time-ranked-sal_{}-sim_{}'.format(
                        center_threshold, sim_threshold),
                    'stream id': row['stream id'], 
                    'sentence id': row['sentence id'],
                    'timestamp': timestamp,
                    'conf': row['salience'],
                    'string': row['string']
                })

        print "Writing", tsv_path, "For ", center_threshold, sim_threshold
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

