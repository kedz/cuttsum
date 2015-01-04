import os
import re
import gzip
import pandas as pd
from ..data import get_resource_manager
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

class SalienceModels(object):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'salience-models')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_model_dir(self, event, feature_set, prefix):
        fname = event.fs_name() + '.' + feature_set.fs_name()
        return os.path.join(self.dir_, prefix, fname)

    def check_coverage(self, event, corpus, 
                       feature_set, prefix, n_samples=10, **kwargs):

        model_dir = self.get_model_dir(event, feature_set, prefix)
        
        if n_samples <= 0:
            return 0        

        if not os.path.exists(model_dir):
            return 0
        
        n_covered = 0
        for i in xrange(n_samples):
            model_path = os.path.join(model_dir, 'model_{}.pkl'.format(i))
            if os.path.exists(model_path):
                n_covered += 1
        return n_covered / float(n_samples)

    def train_models(self, event, corpus, feature_set, prefix, n_procs=1,
                     progress_bar=False, random_seed=42,
                     n_samples=10, sample_size=100, **kwargs):

        print n_samples, sample_size
        model_dir = self.get_model_dir(event, feature_set, prefix)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        jobs = []
        for version in xrange(n_samples):
            model_path = os.path.join(model_dir,
                'model_{}.pkl'.format(version))
            jobs.append((model_path, random_seed + version))
        
        self.do_work(worker, jobs, n_procs, progress_bar, event=event,
                     corpus=corpus, feature_set=feature_set, 
                     sample_size=sample_size)

    def do_work(self, worker, jobs, n_procs,
                progress_bar, result_handler=None, **kwargs):
        max_jobs = len(jobs)
        job_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        for job in jobs:
            job_queue.put(job)

        pool = []        
        for i in xrange(n_procs):
            p = multiprocessing.Process(
                target=worker, args=(job_queue, result_queue), kwargs=kwargs)
            p.start()
            pool.append(p)            

            pb = ProgressBar(max_jobs)
        try:
            for n_job in xrange(max_jobs):
                result = result_queue.get(block=True)
                if result_handler is not None:
                    result_handler(result)
                if progress_bar is True:
                    pb.update()

            for p in pool:
                p.join()

        except KeyboardInterrupt:
            pb.clear()
            print "Completing current jobs and shutting down!"
            while not job_queue.empty():
                job_queue.get()
            for p in pool:
                p.join()
            sys.exit()

def worker(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    corpus = kwargs.get(u'corpus')
    event = kwargs.get(u'event')
    feature_set = kwargs.get(u'feature_set')
    sample_size = kwargs.get(u'sample_size')

    while not job_queue.empty():
        try:

            model_path, random_seed = job_queue.get(block=False)
            feats_df = load_all_feats(event, feature_set)
            sims_df = load_all_sims(event)

            n_points = len(feats_df)
            assert n_points == len(sims_df)

            indices = [i for i in xrange(n_points)]
            random.seed(random_seed)
            random.shuffle(indices)
            indices = indices[:sample_size]

            fgroups = get_group_indices(feats_df)

            y = []
            X = []
            for i in indices:
                streamid = feats_df.iloc[i][u'stream id']
                sentid = feats_df.iloc[i][u'sentence id']
                sentsims = sims_df[
                    (sims_df[u'stream id'] == streamid) & \
                    (sims_df[u'sentence id'] == sentid)]
                max_sim = max(map(list, sentsims.values)[0][2:])
                y.append(max_sim)
                X.append(map(list, sentsims.values)[0][2:])

            y = np.array(y)
            y = y[:, np.newaxis]
            X = np.array(X)
            Xma = np.ma.masked_array(X, np.isnan(X)) 
            rescaler = StandardScaler()
            Xma = rescaler.fit_transform(Xma)

            kern_comb = None
            for key, indices in fgroups.items():
                
                kern = GPy.kern.RBF(input_dim=len(indices), 
                                    active_dims=indices, 
                                    ARD=True)
                if kern_comb is None:
                    kern_comb = kern
                else:
                    kern_comb += kern

            m = GPy.models.GPRegression(Xma, y, kern)
            m.unconstrain('')
            m.constrain_positive('')
                    
            m.optimize_restarts(
                num_restarts=20, robust=False, verbose=False,
                parallel=False, num_processes=1, max_iters=1000)        

            joblib.dump((rescaler, m), model_path)

            result_queue.put(None)
        except Queue.Empty:
            pass



def load_all_feats(event, fs):
   
    all_df = None 
    features = get_resource_manager(u'SentenceFeaturesResource')
    fregex = fs.get_feature_regex() + "|stream id|sentence id"
    for hour in event.list_event_hours():
        path = features.get_tsv_path(event, hour)
        if not os.path.exists(path):
            continue
        with gzip.open(path, u'r') as f:
            df = pd.io.parsers.read_csv(
                f, sep='\t', quoting=3, header=0)
      
        df = df.filter(regex=fregex)
        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat([all_df, df])
    return all_df

def load_all_sims(event):
   
    all_df = None 
    sims = get_resource_manager(u'NuggetSimilaritiesResource')
    for hour in event.list_event_hours():
        path = sims.get_tsv_path(event, hour)
        if not os.path.exists(path):
            continue
        with gzip.open(path, u'r') as f:
            df = pd.io.parsers.read_csv(
                f, sep='\t', quoting=3, header=0)
      
        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat([all_df, df])
    return all_df

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




class SaliencePredictions(object):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'salience-predictions')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)


    def get_tsv_path(self, event, hour, prefix):
        data_dir = os.path.join(self.dir_, prefix, event.fs_name())
        return os.path.join(data_dir, u'{}.tsv.gz'.format(
            hour.strftime(u'%Y-%m-%d-%H')))

    def check_coverage(self, event, corpus, 
                       feature_set, prefix, **kwargs):

        feats = get_resource_manager(u'SentenceFeaturesResource')
        
        n_feats = 0
        n_covered = 0

        for hour in event.list_event_hours():
            feat_tsv_path = feats.get_tsv_path(event, hour)
            sal_tsv_path = self.get_tsv_path(event, hour, prefix)
            if os.path.exists(feat_tsv_path):
                n_feats += 1
                if os.path.exists(sal_tsv_path):
                    n_covered += 1
        if n_feats == 0:
            return 0
        return n_covered / float(n_feats)


