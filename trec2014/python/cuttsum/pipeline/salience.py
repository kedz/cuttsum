import os
import re
import gzip
import pandas as pd
from ..data import get_resource_manager, MultiProcessWorker
from ..misc import ProgressBar
import random
import GPy
import numpy as np
from collections import defaultdict
import multiprocessing
import signal
import sys
import Queue
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from scipy import linalg

class SalienceModels(MultiProcessWorker):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'salience-models')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_model_dir(self, event, feature_set, prefix):
        return os.path.join(self.dir_, prefix + '.' + feature_set.fs_name(),
                            event.fs_name())

    def get_model_paths(self, event, feature_set, prefix, n_samples):
        model_dir = self.get_model_dir(event, feature_set, prefix)
        return [os.path.join(model_dir, 'model_{}.pkl'.format(ver))
                for ver in xrange(n_samples)]

    def check_coverage(self, event, corpus, 
                       feature_set, prefix, n_samples=10, **kwargs):

        model_dir = self.get_model_dir(event, feature_set, prefix)
        
        if n_samples <= 0:
            return 0        

        if not os.path.exists(model_dir):
            return 0
        
        n_covered = 0
        for model_path in self.get_model_paths(
            event, feature_set, prefix, n_samples):

            if os.path.exists(model_path):
                n_covered += 1
        return n_covered / float(n_samples)

    def train_models(self, event, corpus, feature_set, prefix, n_procs=1,
                     progress_bar=False, random_seed=42,
                     n_samples=10, sample_size=100, **kwargs):

        model_paths = self.get_model_paths(
            event, feature_set, prefix, n_samples)

        model_dir = self.get_model_dir(event, feature_set, prefix)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        jobs = []
        for ver, model_path in enumerate(model_paths):
            jobs.append((model_path, random_seed + ver))
        
        self.do_work(salience_train_worker_, jobs, n_procs, 
                     progress_bar, event=event,
                     corpus=corpus, feature_set=feature_set, 
                     sample_size=sample_size)

def salience_train_worker_(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    corpus = kwargs.get(u'corpus')
    event = kwargs.get(u'event')
    feature_set = kwargs.get(u'feature_set')
    sample_size = kwargs.get(u'sample_size')

    while not job_queue.empty():
        try:

            model_path, random_seed = job_queue.get(block=False)
            feats_df, sims_df = load_all_feats(
                event, feature_set, sample_size, random_seed)
            #sims_df = load_all_sims(event)

            n_points = len(feats_df)
            assert n_points == len(sims_df)
            assert n_points == sample_size

            fgroups = get_group_indices(feats_df)

            Y = []
            X = []
            for i in xrange(sample_size):
                assert feats_df.iloc[i][u'stream id'] == \
                    sims_df.iloc[i][u'stream id']
                assert feats_df.iloc[i][u'sentence id'] == \
                    sims_df.iloc[i][u'sentence id']
                
                Y.append(sims_df.iloc[i].values[2:])
                X.append(feats_df.iloc[i].values[2:])

            Y = np.array(Y, dtype=np.float64)
            #y = y[:, np.newaxis]
            X = np.array(X, dtype=np.float64)
            X += 1.e-2 * np.random.normal(0, 1, X.shape)
            #Xma = np.ma.masked_array(X, np.isnan(X)) 
            xrescaler = StandardScaler()

            yrescaler = StandardScaler()
            
            X = xrescaler.fit_transform(X)
            bad_cols = set(np.where(~X.any(axis=0))[0]) | \
                set(np.where(np.all(X == X[0,:], axis=0))[0])
            #print bad_cols
            #if len(bad_cols) > 0:
            #    result_queue.put(None)
            #    return 
            Y = yrescaler.fit_transform(Y)
            y = np.max(Y, axis=1)
            y = y[:, np.newaxis]

            kern_comb = None
            for key, indices in fgroups.items():
                good_indices = set(indices) - bad_cols
                indices = sorted(list(good_indices))
                kern = GPy.kern.RBF(input_dim=len(indices), 
                                    active_dims=indices, 
                                    ARD=True)
                                
                if kern_comb is None:
                    kern_comb = kern
                else:
                    kern_comb += kern
                kern_comb += GPy.kern.White(input_dim=len(indices),
                                            active_dims=indices)
            try:
                m = GPy.models.GPRegression(X, y, kern_comb)
                m.unconstrain('')
                m.constrain_positive('')
                #m['.*white'].constrain_fixed(1.)
                        
                m.optimize_restarts(
                    num_restarts=10, robust=False, verbose=False,
                    parallel=False, num_processes=1, max_iters=20)        

                joblib.dump((xrescaler, yrescaler, m), model_path)
            except linalg.LinAlgError, e:
                print e
                #print X
                #print y

            result_queue.put(None)
        except Queue.Empty:
            pass



def load_all_feats(event, fs, sample_size, random_seed):
    random.seed(random_seed)
 
    all_df = None 
    features = get_resource_manager(u'SentenceFeaturesResource')
    fregex = fs.get_feature_regex() + "|stream id|sentence id"

    resevoir = None
    current_index = 0
    hours = []
    res_sims = None

    for hour in event.list_event_hours():
        path = features.get_tsv_path(event, hour)
        #print path, current_index
        if not os.path.exists(path):
            continue
        with gzip.open(path, u'r') as f:
            df = pd.io.parsers.read_csv(
                f, sep='\t', quoting=3, header=0)
        df = df.filter(regex=fregex)
        
        if resevoir is None:
            resevoir = pd.DataFrame(columns=df.columns)

        for _, row in df.iterrows():
            if current_index < sample_size:
                resevoir.loc[current_index] = row
                hours.append(hour)
            else:
                r = random.randint(0, current_index)
                if r < sample_size:
                    resevoir.loc[r] = row
                    hours[r] = hour
            current_index += 1
       # if resevoir is None:
       #     resevoir = df.iloc[range(0, sample_size)]
       #     current_index = sample_size
       #     paths = [path for i in range(0, sample_size)]

    s = get_resource_manager(u'NuggetSimilaritiesResource')
    for hour in set(hours):
        path = s.get_tsv_path(event, hour)
        with gzip.open(path, u'r') as f:
            df = pd.io.parsers.read_csv(
                f, sep='\t', quoting=3, header=0)
        if res_sims is None:
            res_sims = pd.DataFrame([{} for x in range(sample_size)], 
                                    columns=df.columns)
        for idx, row_hour in enumerate(hours):
            if hour != row_hour:
                continue
            stream_id = resevoir.iloc[idx][u'stream id']
            sent_id = resevoir.iloc[idx][u'sentence id']
            res_sims.loc[idx] = df.loc[
                (df[u'stream id'] == stream_id) & \
                (df[u'sentence id'] == sent_id)].iloc[0]
             
    for i in range(sample_size):
        assert resevoir[u'sentence id'].iloc[i] == \
            res_sims[u'sentence id'].iloc[i]
            
        assert resevoir[u'stream id'].iloc[i] == \
            res_sims[u'stream id'].iloc[i]

    return resevoir, res_sims
    

def get_group_indices(feat_df):
    idxmap = defaultdict(list)
    for idx, feat in enumerate(feat_df.columns[2:]):
        idxmap[feat.split('_')[0]].append(idx)
    for feat in idxmap.keys():
        idxmap[feat].sort()
#        start = min(idxmap[feat])
#        end = max(idxmap[feat]) + 1
#        idxmap[feat] = (start, end)
    return idxmap

class SaliencePredictions(MultiProcessWorker):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'salience-predictions')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)


    def get_tsv_dir(self, event, prefix, feature_set):
        data_dir = os.path.join(self.dir_, 
                                prefix + "." + feature_set.fs_name(),
                                event.fs_name())
        return data_dir

    def get_tsv_path(self, event, hour, prefix, feature_set, model_path):
        
        parent_dir, model_num = os.path.split(model_path)
        model_name = os.path.split(parent_dir)[-1]
        key = model_name + "." + model_num
        data_dir = self.get_tsv_dir(event, prefix, feature_set)
        return os.path.join(data_dir, u'{}.{}.tsv.gz'.format(
            hour.strftime(u'%Y-%m-%d-%H'), key))

    def check_coverage(self, event, corpus, 
                       feature_set, prefix, model_events, n_samples=100, 
                       **kwargs):

        feats = get_resource_manager(u'SentenceFeaturesResource')
        
        n_feats = 0
        n_covered = 0
        sm = SalienceModels()
        model_paths = []
        for model_event in model_events:
            model_paths.extend(
                sm.get_model_paths(
                    model_event, feature_set, prefix, n_samples))
 
        for hour in event.list_event_hours():
            for model_path in model_paths: 
                feat_tsv_path = feats.get_tsv_path(event, hour)
                sal_tsv_path = self.get_tsv_path(
                    event, hour, prefix, feature_set, model_path)
                if os.path.exists(feat_tsv_path):
                    n_feats += 1
                    if os.path.exists(sal_tsv_path):
                        n_covered += 1
        if n_feats == 0:
            return 0
        return n_covered / float(n_feats)

    def predict_salience(self, event, corpus, feature_set, 
                         prefix, model_events, n_procs=1, n_samples=10,
                         progress_bar=False, **kwargs):

        data_dir = self.get_tsv_dir(event, prefix, feature_set)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        feats = get_resource_manager(u'SentenceFeaturesResource')
        sm = SalienceModels()
        model_paths = []
        for model_event in model_events:
            model_paths.extend(
                sm.get_model_paths(
                    model_event, feature_set, prefix, n_samples))
      
        jobs = []
#        for hour in event.list_event_hours():
        for model_path in model_paths:

#                feat_tsv_path = feats.get_tsv_path(event, hour)
#                sal_tsv_path = self.get_tsv_path(
 #                   event, hour, prefix, feature_set, model_path)
                #if os.path.exists(feat_tsv_path):
                jobs.append(model_path)
            
        self.do_work(salience_predict_worker_, jobs, n_procs, 
                     progress_bar, event=event,
                     feature_set=feature_set, prefix=prefix)


def salience_predict_worker_(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    event = kwargs.get(u'event')
    fs = kwargs.get(u'feature_set')
    prefix = kwargs.get(u'prefix')
    feats = get_resource_manager(u'SentenceFeaturesResource')
    sp = SaliencePredictions()
#    model_paths = kwargs.get(u'model_paths')
#    model_paths.sort()
    
#    model_keys = []
#    key2models = {}
#    n_models = len(model_paths) 
#    for i, model_path in enumerate(model_paths, 1):
#        print i, "/", n_models
        #if not os.path.exists(model_path):
        #    continue
#        parent_dir, model_num = os.path.split(model_path)
#        model_name = os.path.split(parent_dir)[-1]
#        key = model_name + "." + model_num
#        key2models[key] =  joblib.load(model_path)
#        model_keys.append(key)
        

   # n_model_paths = len(model_paths)
    fregex = fs.get_feature_regex() + "|stream id|sentence id"

    while not job_queue.empty():
        try:

            model_path = job_queue.get(block=False)
            parent_dir, model_num = os.path.split(model_path)
            model_name = os.path.split(parent_dir)[-1]
            model_key = model_name + "." + model_num
            xrescaler, yrescaler, m = joblib.load(model_path)

            for hour in event.list_event_hours():
                feat_tsv_path = feats.get_tsv_path(event, hour)
                sal_tsv_path = sp.get_tsv_path(
                    event, hour, prefix, fs, model_path)
                if not os.path.exists(feat_tsv_path):
                    continue
                if os.path.exists(sal_tsv_path):
                    continue
 
                with gzip.open(feat_tsv_path, u'r') as f:
                    feats_df = pd.io.parsers.read_csv(
                        f, sep='\t', quoting=3, header=0)
                    feats_df = feats_df.filter(regex=fregex)
     
                feats_df = feats_df.sort([u'stream id', u'sentence id'])   
                n_points = len(feats_df)

                sims = []
                X = []
                for i in xrange(n_points):
                    streamid = feats_df.iloc[i][u'stream id']
                    sentid = feats_df.iloc[i][u'sentence id']
                    x = feats_df.iloc[i].values[2:]
                    #print len(feats_df.columns), len(x), x
                    X.append(x)
                    sims.append({u'stream id': streamid, u'sentence id': sentid})


                X = np.array(X, dtype=np.float64)
               # print X.shape

                #for model_key, (xrescaler, yrescaler, m) in key2models.items():
                #    print model_key
                #for model_path in model_paths:
                #    parent_dir, model_num = os.path.split(model_path)
                #    model_name = os.path.split(parent_dir)[-1]
                #    model_key = model_name + "." + model_num 
                    #if not os.path.exists(model_path):
                    #    continue
                    #print model_path
                #xrescaler, yrescaler, m = joblib.load(model_path)
                    #print xrescaler.mean_.shape
                Xscale = xrescaler.transform(X)
                result = m.predict(Xscale)                
                    #print len(result)
                    ##print result[0].shape
                yp = result[0]
                for i, y in enumerate(yp):
                    sims[i][model_key] = y[0]
                                 
                sims_df = pd.DataFrame(
                    sims, columns=[u'stream id', u'sentence id', model_key])

                with gzip.open(sal_tsv_path, u'w') as f:
                    sims_df.to_csv(f, sep='\t', index=False, index_label=False)  

            result_queue.put(None)
        except Queue.Empty:
            pass

class SaliencePredictionAggregator(object):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'salience-predictions-agg')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)


    def check_coverage(self, event, corpus, feature_set,
                       prefix, model_events, n_samples=10, **kwargs):

        feats = get_resource_manager(u'SentenceFeaturesResource')
        n_hours = 0
        n_covered = 0

        for hour in event.list_event_hours():
            feats_tsv_path = feats.get_tsv_path(event, hour)
            sal_tsv_path = self.get_tsv_path(event, hour, prefix, feature_set)
            if os.path.exists(feats_tsv_path):
                n_hours += 1
                if os.path.exists(sal_tsv_path):
                    n_covered += 1
        if n_hours == 0:
            return 0
        else:
            return n_covered / float(n_hours)

    def get(self, event, corpus, feature_set,
            prefix, model_events, n_samples=10, **kwargs):

        feats = get_resource_manager(u'SentenceFeaturesResource')
        
        sm = SalienceModels()
        sp = SaliencePredictions()
        model_paths = []
        for model_event in model_events:
            model_paths.extend(
                sm.get_model_paths(
                    model_event, feature_set, prefix, n_samples))

        hours = event.list_event_hours()
        n_hours = len(hours)
        pb = ProgressBar(n_hours)
        for hour in hours:
            pb.update()
            tsv_paths = \
                [sp.get_tsv_path(event, hour, prefix, feature_set, model_path)
                 for model_path in model_paths]
            tsv_paths = [path for path in tsv_paths if os.path.exists(path)]
            if len(tsv_paths) == 0:
                continue
            data = []
            for tsv_path in tsv_paths:
                with gzip.open(tsv_path, u'r') as f:
                    df = pd.io.parsers.read_csv(
                        f, sep='\t', quoting=3, header=0)
                    df.set_index([u'stream id', u'sentence id'], inplace=True)
                    data.append(df)
            df = pd.concat(data, axis=1)
            agg_path = self.get_tsv_path(event, hour, prefix, feature_set)
            agg_dir = os.path.dirname(agg_path)
            if not os.path.exists(agg_dir):
                os.makedirs(agg_dir)

            df.columns=sorted(df.columns)
            with gzip.open(agg_path, u'w') as f:
                df.to_csv(f, sep='\t')  

    def get_tsv_dir(self, event, prefix, feature_set):
        data_dir = os.path.join(self.dir_, 
                                prefix + "." + feature_set.fs_name(),
                                event.fs_name())
        return data_dir

    def get_tsv_path(self, event, hour, prefix, feature_set):
        
        data_dir = self.get_tsv_dir(event, prefix, feature_set)
        return os.path.join(data_dir, u'{}.tsv.gz'.format(
            hour.strftime(u'%Y-%m-%d-%H')))

    def get_dataframe(self, event, hour, prefix, feature_set):
        tsv = self.get_tsv_path(event, hour, prefix, feature_set)
        if not os.path.exists(tsv):
            return None
        else:
            with gzip.open(tsv, u'r') as f:
                df = pd.io.parsers.read_csv(
                    f, sep='\t', quoting=3, header=0)
                return df

