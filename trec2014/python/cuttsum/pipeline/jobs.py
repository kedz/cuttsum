import cuttsum
import random
from cuttsum.pipeline.representation import SalienceFeatureSet
from cuttsum.salience import SalienceModels, SaliencePredictions, \
    SaliencePredictionAggregator
from cuttsum.summarizer.baseline import HACSummarizer
from cuttsum.summarizer.ap import APSummarizer, APSalienceSummarizer
from cuttsum.summarizer.filters import *
#    APSalienceFilteredSummary, HACFilteredSummary
import os
import multiprocessing
from multiprocessing import Pool
from cuttsum.misc import ProgressBar
from cuttsum.data import MultiProcessWorker
import signal
import Queue
import sys
import traceback
import numpy as np

def feature_ablation_jobs(key, per_train=.6, random_seed=42):
    def desc(train, test, features):

        all_features = set([u'character', u'language model', u'frequency',
                            u'geographic', u'query'])

        inactive_features = all_features - features.as_set()
        return u"feature ablation, -{}".format(
            u', '.join(inactive_features))

    for job in job_generator(
        None, leave_one_out_feature_selector, desc, key,
        per_train=per_train, random_seed=random_seed):
        yield job

def event_cross_validation_jobs(key):
    def desc(train, test, features):
        #test_event, corpus = test[0]
        #return u'{}/{}'.format(
        #    test_event.title, corpus.fs_name())
        return "event-cross-validation"

    for job in job_generator(
        None, all_feature_selector, desc, key,):
        yield job


### Event selectors

def random_event_splitter(events, **kwargs):
    per_train = kwargs.get(u'per_train', .6)
    random_seed = kwargs.get(u'random_seed', 42)
    assert per_train > 0 and per_train < 1
    cutoff = int(round(per_train * len(events)))
    random.seed(random_seed)
    random.shuffle(events)
    train_events = events[0:cutoff]
    test_events = events[cutoff:]
    return [(train_events, test_events)]

def event_n_fold_selector(events, **kwargs):
    splits = []
    for test_event in events:
        train_events = [event for event in events if event != test_event]
        splits.append((train_events, [test_event]))
    return splits

# Feature selectors

def leave_one_out_feature_selector(features, **kwargs):
    feature_groups = []
    for feature in features:
        feature_group = [selected for selected in features
                         if selected != feature]
        feature_groups.append(SalienceFeatureSet(feature_group))
    return feature_groups

def all_feature_selector(features, **kwargs):
    return [SalienceFeatureSet(features)]

def job_generator(event_selector, feature_selector,
    job_descriptor, key, **kwargs):
    """Make all possible jobs for a given evaluation format 
    e.g. leave-one-event-out or feature group ablations."""

    # Retrieve events.
    corpus_2013 = cuttsum.corpora.EnglishAndUnknown2013()
    events_2013 = [(event, corpus_2013)
                   for event in cuttsum.events.get_2013_events() 
                   if event.fs_name() != 'june_2012_north_american_derecho']

    corpus_2014 = cuttsum.corpora.FilteredTS2014()
    events_2014 = [(event, corpus_2014)
                   for event in cuttsum.events.get_2014_events()]

    events = events_2013 + events_2014
    if event_selector is None:
        event_splits = [events]

    # Generate event splits.
    #event_splits = event_selector(events, **kwargs)

    # Generate feature splits.
    features = [u'character', u'language model', u'frequency',
                u'geographic', u'query']
    feature_splits = feature_selector(features, **kwargs)

    job_num = 1
    # Yield the all combinations of event splits and feature splits.
    for event_split in event_splits:
        for feature_split in feature_splits:
            #train, test = event_split
            desc = job_descriptor(None, None, feature_split)
            job_key = u"{}.{}".format(key, job_num)
            job = PipelineJob(job_key, desc, events, feature_split)
            yield job
            job_num += 1

class PipelineJob(MultiProcessWorker):
    def __init__(self, key, description, 
        event_data, feature_set, seed=42):
        
        self.key = key
        self.description = description
        self.event_data = event_data
        self.feature_set = feature_set
        self.seed_ = seed

    def dev_events(self):
        ed = list(self.event_data)
        random.seed(self.seed_)
        random.shuffle(ed)
        return sorted(ed[0:3], key=lambda x: x[0].query_num)

    def eval_events(self):
        ed = list(self.event_data)
        random.seed(self.seed_)
        random.shuffle(ed)
        return sorted(ed[3:], key=lambda x: x[0].query_num)

    def __str__(self):
        return unicode(self).encode(u'utf-8')

    def __unicode__(self):
        strings = []
        strings.append(u"job key: {}".format(self.key))
        strings.append(u"job description: {}".format(self.description))
        #strings.append(u"training events:")
        for event, corpus in self.event_data:
            title = event.title
            if len(title) > 35:
                title = title[0:31] + u' ...'
            corpus_name = corpus.fs_name()
            if len(corpus_name) > 17:
                corpus_name = corpus_name[0:13] + u' ...'
            strings.append(
                u'  {:8s} {:14s} {:35s} {:17s}'.format(
                    event.query_id, event.type, title, corpus_name))
        #strings.append(u"test events:")
        #for event, corpus in self.testing_data:
        #    title = event.title
            #if len(title) > 35:
               # title = title[0:31] + u' ...'
           # corpus_name = corpus.fs_name()
           # if len(corpus_name) > 17:
           #     corpus_name = corpus_name[0:13] + u' ...'
           # strings.append(
           #     u'  {:8s} {:14s} {:35s} {:17s}'.format(
           #         event.query_id, event.type, title, corpus_name))

        strings.append(unicode(self.feature_set))

        return u'\n'.join(strings)


    def start(self, **kwargs):

#        print "  checking resources and models"
#        print "  ============================="
#        for event, corpus in self.event_data:
#                       
#            print event.fs_name(), corpus.fs_name()
#            has_all_resources = self.check_resource_pipeline(
#                event, corpus, **kwargs)
#            if not has_all_resources:
#                self.run_resource_pipeline(event, corpus, **kwargs)
#
#            print  "+  resource dependency checks met!"
#            
#            has_all_models = self.check_model_pipeline(
#                event, corpus, self.feature_set, **kwargs)
#            if not has_all_models:
#                self.train_models(event, corpus, self.feature_set, **kwargs)
#
#            print  "+  model dependency checks met!"
#
#
#        print "  checking predictions"
#        print "  ============================="
#
#        model_events = set(event for event, corpus in self.event_data)
#        for event, corpus in self.event_data:
# 
#            print event.fs_name(), "/", corpus.fs_name()
#            has_all_predictions = self.check_model_predictions(
#                event, corpus, self.feature_set,
#                self.key, model_events - set([event]), **kwargs)
#            if not has_all_predictions:
#                self.predict_salience(event, corpus, self.feature_set,
#                                      self.key, model_events - set([event]), 
#                                      **kwargs)
#            print  "+  model prediction checks met!"
#
#            spg = SaliencePredictionAggregator()
#            cov = spg.check_coverage(
#                event, corpus, self.feature_set,
#                self.key, model_events - set([event]), **kwargs)
#            if cov != 1:
#                spg.get(event, corpus, self.feature_set,
#                        self.key, model_events - set([event]), **kwargs)
#            print "+  model predictions aggregated!"
##
##   
##
        print "running summarizer jobs..."
        jobs = []

        dev_events = self.dev_events()
        eval_events = self.eval_events()


        if 'ablation' in self.key:
            print "I am here here" 
            if kwargs.get(u'tune', False) is True:
                self.tune_fa(dev_events, self.key, self.feature_set, **kwargs)
            else:
                self.make_fa_summaries(
                    eval_events, self.key, self.feature_set, **kwargs)


        elif 'cross' in self.key:
            if kwargs.get(u'tune', False) is True:
                self.tune(dev_events, self.key, self.feature_set, **kwargs)

            else:
                self.make_summaries(
                    eval_events, self.key, self.feature_set,
                    **kwargs)

    def check_resource_pipeline(self, event, corpus, **kwargs):
        sfeats = \
            cuttsum.data.get_resource_manager(u'SentenceFeaturesResource')
        nsims = \
            cuttsum.data.get_resource_manager(u'NuggetSimilaritiesResource')
        sf_satisfied = sfeats.check_coverage(event, corpus, **kwargs) == 1.0
        ns_satisfied = nsims.check_coverage(event, corpus, **kwargs) == 1.0
        return sf_satisfied and ns_satisfied
            
    def run_resource_pipeline(self, event, corpus, **kwargs):
        for group in cuttsum.data.get_sorted_dependencies(reverse=False):
            for resource in group:
                coverage = resource.check_coverage(event, corpus, **kwargs)
                print "    {:50s} : {:7.3f} %".format(
                    resource, 100. * coverage)
                if coverage != 1:
                    print kwargs
                    resource.get(event, corpus, **kwargs)
        print 
        if not self.check_resource_pipeline(event, corpus, **kwargs):
            raise Exception("Resource pipeline failed!") 

    def check_model_pipeline(self, event, corpus, feature_set, **kwargs):
        sm = SalienceModels()
        coverage = sm.check_coverage(event, corpus, feature_set, 
                                     self.key, **kwargs) 
        return coverage == 1
    
    def train_models(self, event, corpus, feature_set, **kwargs):
        print "-  training salience models..."
        sm = SalienceModels()
        sm.train_models(event, corpus, feature_set, self.key, **kwargs) 
        if not self.check_model_pipeline(event, corpus, feature_set, **kwargs):
            raise Exception("Model training failed!") 

    def check_model_predictions(self, event, corpus, 
                                feature_set, key, model_events, **kwargs):
        sp = SaliencePredictions()
        coverage = sp.check_coverage(event, corpus, feature_set, 
                                     key, model_events, **kwargs)
        return coverage == 1

    def predict_salience(self, event, corpus, feature_set, key,
                         model_events, **kwargs):

        print "-  predicting salience..."
        sp = SaliencePredictions()
        sp.predict_salience(event, corpus, feature_set, key, 
                            model_events, **kwargs)
        if not self.check_model_predictions(event, corpus, feature_set, 
                                            key, model_events, **kwargs):
            raise Exception("Model prediction failed!") 

    def tune_fa(self, dev_data, prefix, feature_set, 
                sal_min=-2.0, sal_max=2.0, sal_step=.1,
                sem_sim_min=.2, sem_sim_max=.7, sem_sim_step=.05, 
                **kwargs): 

        apsal = APSalienceSummarizer()
        sal_cutoffs = np.arange(
            sal_min, sal_max + sal_step, sal_step)
        sem_sim_cutoffs = np.arange(
            sem_sim_min, sem_sim_max + sem_sim_step, sem_sim_step)

        print "Tuning on dev data."
        
        ### Run clustering ###
        print "Generating AP+Salience Cluster\n\t(no params)"

        jobs = []                        
        for event, corpus in dev_data:
            print event.fs_name()
            apsal_tsv_dir = apsal.get_tsv_dir(prefix, feature_set)
            if not os.path.exists(apsal_tsv_dir):
                os.makedirs(apsal_tsv_dir)
            jobs.append((event, corpus, prefix, feature_set, apsal))

        self.do_work(cluster_worker, jobs, **kwargs)

        ### Run filtering ###
        print 
        print "Generating AP+Salience Summary"
        print "\tSal Threshold ({}, {}), step={}".format(
            sal_min, sal_max, sal_step)
        print "\tSim Threshold ({}, {}), step={}".format(
            sem_sim_min, sem_sim_max, sem_sim_step)
        print "\t{} jobs/event".format(
            sal_cutoffs.shape[0] * sem_sim_cutoffs.shape[0])

        jobs = []
        for event, corpus in dev_data:

            for sem_sim_cutoff in sem_sim_cutoffs:
                for sal_cutoff in sal_cutoffs:
                    jobs.append(
                        (APSalienceFilteredSummary(), 
                         (event, prefix, feature_set, sal_cutoff,
                          sem_sim_cutoff)))
        self.do_work(filter_worker, jobs, **kwargs)


    def tune(self, dev_data, prefix, feature_set, 
             hac_dist_min=.9, hac_dist_max=5.05, hac_dist_step=.05, 
             sal_min=-2.0, sal_max=2.0, sal_step=.1,
             sem_sim_min=.2, sem_sim_max=.7, sem_sim_step=.05, 
             rank_sim_min=.2, rank_sim_max=.4, rank_sim_step=.05, **kwargs): 

        ap = APSummarizer() 
        apsal = APSalienceSummarizer()
        hac = HACSummarizer()

        hac_dist_cutoffs = np.arange(
            hac_dist_min, hac_dist_max + hac_dist_step, hac_dist_step)
        sal_cutoffs = np.arange(
            sal_min, sal_max + sal_step, sal_step)
        sem_sim_cutoffs = np.arange(
            sem_sim_min, sem_sim_max + sem_sim_step, sem_sim_step)
        rank_sim_cutoffs = np.arange(
            rank_sim_min, rank_sim_max + rank_sim_step, rank_sim_step)

        print "Tuning on dev data."
        
        ### Run clustering ###
        print "Generating AP Cluster\n\t(no params)"
        print "Generating AP+Salience Cluster\n\t(no params)"
        print "Generating HAC Cluster"
        print "\tDist Threshold ({}, {}), step={} {} jobs/event".format(
            hac_dist_min, hac_dist_max, hac_dist_step,
            hac_dist_cutoffs.shape[0])

        jobs = []                        
        for event, corpus in dev_data:
            print event.fs_name()
            apsal_tsv_dir = apsal.get_tsv_dir(prefix, feature_set)
            if not os.path.exists(apsal_tsv_dir):
                os.makedirs(apsal_tsv_dir)
            if not os.path.exists(ap.dir_):
                os.makedirs(ap.dir_)        
            if not os.path.exists(hac.dir_):
                os.makedirs(hac.dir_)

            for cutoff in hac_dist_cutoffs:
                jobs.append(
                    (event, corpus, prefix, feature_set, hac, cutoff))   
            jobs.append((event, corpus, prefix, feature_set, ap))
            jobs.append((event, corpus, prefix, feature_set, apsal))

        self.do_work(cluster_worker, jobs, **kwargs)

        ### Run filtering ###
        print 
        print "Generating AP Summary"
        print "\tSim Threshold ({}, {}), step={}".format(
            sem_sim_min, sem_sim_max, sem_sim_step)
        print "\t{} jobs/event".format(
            sem_sim_cutoffs.shape[0])
        print "Generating AP+Salience Summary"
        print "\tSal Threshold ({}, {}), step={}".format(
            sal_min, sal_max, sal_step)
        print "\tSim Threshold ({}, {}), step={}".format(
            sem_sim_min, sem_sim_max, sem_sim_step)
        print "\t{} jobs/event".format(
            sal_cutoffs.shape[0] * sem_sim_cutoffs.shape[0])
        print "Generating HAC Summary"
        print "\tDist Threshold ({}, {}), step={}".format(
            hac_dist_min, hac_dist_max, hac_dist_step)
        print "\tSim Threshold ({}, {}), step={}".format(
            sem_sim_min, sem_sim_max, sem_sim_step)
        print "\t{} jobs/event".format(
            hac_dist_cutoffs.shape[0] * sem_sim_cutoffs.shape[0])

        rsfs = RankedSalienceFilteredSummary()  
        if not os.path.exists(rsfs.dir_):
            os.makedirs(rsfs.dir_)
        jobs = []
        for event, corpus in dev_data:

            for sem_sim_cutoff in sem_sim_cutoffs:
                    
                for dist_cutoff in hac_dist_cutoffs:
                    jobs.append(
                        (HACFilteredSummary(), 
                        (event, prefix, feature_set,
                         dist_cutoff, sem_sim_cutoff)))
        
                jobs.append(
                    (APFilteredSummary(), (event, sem_sim_cutoff)))
                for sal_cutoff in sal_cutoffs:
                    jobs.append(
                        (APSalienceFilteredSummary(), 
                         (event, prefix, feature_set, sal_cutoff,
                          sem_sim_cutoff)))
                    jobs.append(
                        (APSalTRankSalThreshFilteredSummary(),
                         (event, prefix, feature_set, sal_cutoff,
                          sem_sim_cutoff)))
            for rank_sim_cutoff in rank_sim_cutoffs:
                for sal_cutoff in sal_cutoffs:
                    jobs.append(
                        (RankedSalienceFilteredSummary(),
                         (event, prefix, feature_set, sal_cutoff,
                          rank_sim_cutoff)))
        self.do_work(filter_worker, jobs, **kwargs)

    def make_fa_summaries(self, eval_data, prefix, feature_set,
                          apsal_sal=.4, apsal_sim=.7, **kwargs):
    
        apsal = APSalienceSummarizer()
        jobs = []                        
        for event, corpus in eval_data:
            print event.fs_name()
            apsal_tsv_dir = apsal.get_tsv_dir(prefix, feature_set)
            if not os.path.exists(apsal_tsv_dir):
                os.makedirs(apsal_tsv_dir)
            jobs.append((event, corpus, prefix, feature_set, apsal))

        self.do_work(cluster_worker, jobs, **kwargs)

        print "Generating AP+Salience Summary"
        print "\tSal Threshold: {}".format(apsal_sal)
        print "\tSim Threshold: {}".format(apsal_sim)

        jobs = []
        for event, corpus in eval_data:
            jobs.append(
                (APSalienceFilteredSummary(), 
                 (event, prefix, feature_set, apsal_sal,
                  apsal_sim)))

        self.do_work(filter_worker, jobs, **kwargs)



    def make_summaries(self, eval_data, prefix, feature_set,
                       hac_dist=1.35, hac_sim=.7, ap_sim=.7,
                       apsal_sal=.4, apsal_sim=.7, 
                       apsal_tr_sal=.6, apsal_tr_sim=.6, 
                       sal_rank_sal=1.8, sal_rank_sim=.4, **kwargs):
                       
        
        ap = APSummarizer() 
        apsal = APSalienceSummarizer()
        hac = HACSummarizer()

        print "Running with optimal params on dev data."
        
        ### Run clustering ###
        print "Generating AP Cluster\n\t(no params)"
        print "Generating AP+Salience Cluster\n\t(no params)"
        print "Generating HAC Cluster\n\tdist-thresh: {}".format(hac_dist)

        jobs = []                        
        for event, corpus in eval_data:
            print event.fs_name()
            apsal_tsv_dir = apsal.get_tsv_dir(prefix, feature_set)
            if not os.path.exists(apsal_tsv_dir):
                os.makedirs(apsal_tsv_dir)
            if not os.path.exists(ap.dir_):
                os.makedirs(ap.dir_)        
            if not os.path.exists(hac.dir_):
                os.makedirs(hac.dir_)

            jobs.append(
                (event, corpus, prefix, feature_set, hac, hac_dist))   
            jobs.append((event, corpus, prefix, feature_set, ap))
            jobs.append((event, corpus, prefix, feature_set, apsal))

        self.do_work(cluster_worker, jobs, **kwargs)

        ### Run filtering ###
        print 
        print "Generating AP Summary"
        print "\tSim Threshold: {}".format(ap_sim)
        print "Generating AP+Salience Summary"
        print "\tSal Threshold: {}".format(apsal_sal)
        print "\tSim Threshold: {}".format(apsal_sim)
        print "Generating HAC Summary"
        print "\tDist Threshold: {}".format(hac_dist)
        print "\tSim Threshold: {}".format(hac_sim)
        print "Generating AP+Salience Time Ranked"
        print "\tSal Threshold: {}".format(apsal_tr_sal)
        print "\tSim Threshold: {}".format(apsal_tr_sim)
        print "Generating Salience Ranked Summary"
        print "\tSal Threshold: {}".format(sal_rank_sal)
        print "\tSim Threshold: {}".format(sal_rank_sim)
        

        jobs = []
        for event, corpus in eval_data:
            jobs.append(
                (HACFilteredSummary(), 
                (event, prefix, feature_set,
                 hac_dist, hac_sim)))
        
            jobs.append(
                (APFilteredSummary(), (event, ap_sim)))
            jobs.append(
                (APSalienceFilteredSummary(), 
                 (event, prefix, feature_set, apsal_sal,
                  apsal_sim)))
            jobs.append(
                (APSalTRankSalThreshFilteredSummary(),
                 (event, prefix, feature_set, apsal_tr_sal,
                  apsal_tr_sim)))
            jobs.append(
                (RankedSalienceFilteredSummary(),
                 (event, prefix, feature_set, sal_rank_sal,
                  sal_rank_sim)))

        self.do_work(filter_worker, jobs, **kwargs)


  
def cluster_worker(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while not job_queue.empty():
        try:
            args = job_queue.get(block=False)
            if isinstance(args[4], HACSummarizer):
                event, corpus, prefix, feature_set, summarizer, cutoff = args
                try:
                    tsv_path = summarizer.get_tsv_path(
                        event, cutoff)
                    if not os.path.exists(tsv_path):
                        summarizer.make_summary(
                            event, corpus, prefix, feature_set, cutoff)

                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    print "*** print_tb:"
                    traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                    print "*** print_exception:"
                    traceback.print_exception(exc_type, exc_value, 
                        exc_traceback, limit=15, file=sys.stdout)
                finally:
                    result_queue.put(None)

            else:
                event, corpus, prefix, feature_set, summarizer = args
                try:
                    tsv_path = summarizer.get_tsv_path(
                        event, prefix, feature_set)
                    if not os.path.exists(tsv_path):
                        summarizer.make_summary(
                            event, corpus, prefix, feature_set)
                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    print "*** print_tb:"
                    traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                    print "*** print_exception:"
                    traceback.print_exception(exc_type, exc_value, 
                        exc_traceback, limit=15, file=sys.stdout)
                finally:
                    result_queue.put(None)
        except Queue.Empty:
            pass
    return True 


def filter_worker(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import time
    time.sleep(1)
    while not job_queue.empty():
        try:
            summarizer, sum_args = job_queue.get(block=False)

            if isinstance(summarizer, APFilteredSummary):
                event, sim_cutoff = sum_args
                ap = APSummarizer()
                ap_tsv_path = ap.get_tsv_path(event)
                apf_tsv_path = summarizer.get_tsv_path(event, sim_cutoff)
                if os.path.exists(ap_tsv_path) and \
                    not os.path.exists(apf_tsv_path):
                    summarizer.make(event, sim_threshold=sim_cutoff)        

            elif isinstance(summarizer, HACFilteredSummary):
                event, prefix, feature_set, dist_cutoff, sim_cutoff = sum_args
                hac = HACSummarizer()
                hac_tsv_path = hac.get_tsv_path(event, dist_cutoff)
                hacf_tsv_path = summarizer.get_tsv_path(
                    event, dist_cutoff, sim_cutoff)
                if os.path.exists(hac_tsv_path) and \
                    not os.path.exists(hacf_tsv_path):
                    
                    try:
                        summarizer.make(event, prefix, feature_set, 
                                        dist_cutoff=dist_cutoff,
                                        sim_threshold=sim_cutoff)
                    except Exception, e:
                        print e
                        print hac_tsv_path

            elif isinstance(summarizer, APSalienceFilteredSummary) or \
                isinstance(summarizer, APSalTRankSalThreshFilteredSummary):
                event, prefix, feature_set, sal_cutoff, sim_cutoff = sum_args
                aps = APSalienceSummarizer()
                aps_tsv_path = aps.get_tsv_path(event, prefix, feature_set)
                apsf_tsv_path = summarizer.get_tsv_path(
                    event, prefix, feature_set, sal_cutoff, sim_cutoff)
                if os.path.exists(aps_tsv_path) and \
                    not os.path.exists(apsf_tsv_path):
                    summarizer.make(
                        event, prefix, feature_set,
                        min_cluster_size=2, center_threshold=sal_cutoff, 
                        sim_threshold=sim_cutoff)
            elif isinstance(summarizer, RankedSalienceFilteredSummary):
                event, prefix, feature_set, sal_cutoff, sim_cutoff = sum_args
                rsfs_tsv_path = summarizer.get_tsv_path(
                    event, sal_cutoff, sim_cutoff)
                if not os.path.exists(rsfs_tsv_path):
                    summarizer.make(
                        event, prefix, feature_set,
                        sal_cutoff, sim_cutoff)



            
            result_queue.put(None)
        except Queue.Empty:
            pass
    return True 
