import cuttsum
import random
from .representation import SalienceFeatureSet
from .salience import SalienceModels, SaliencePredictions, \
    SaliencePredictionAggregator
from cuttsum.summarizer.baseline import HACSummarizer
from cuttsum.summarizer.ap import APSummarizer, APSalienceSummarizer
import os
from multiprocessing import Pool
from ..misc import ProgressBar
from ..data import MultiProcessWorker
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

        
#        training_data = []
        print "  checking resources and models"
        print "  ============================="
        for event, corpus in self.event_data:
                       
            print event.fs_name(), corpus.fs_name()
            has_all_resources = self.check_resource_pipeline(
                event, corpus, **kwargs)
            if not has_all_resources:
                self.run_resource_pipeline(event, corpus, **kwargs)

            print  "+  resource dependency checks met!"
            
            has_all_models = self.check_model_pipeline(
                event, corpus, self.feature_set, **kwargs)
            if not has_all_models:
                self.train_models(event, corpus, self.feature_set, **kwargs)

            print  "+  model dependency checks met!"


        print "  checking predictions"
        print "  ============================="

        model_events = set(event for event, corpus in self.event_data)
        for event, corpus in self.event_data:
 
            print event.fs_name(), "/", corpus.fs_name()
            has_all_predictions = self.check_model_predictions(
                event, corpus, self.feature_set,
                self.key, model_events - set([event]), **kwargs)
            if not has_all_predictions:
                self.predict_salience(event, corpus, self.feature_set,
                                      self.key, model_events - set([event]), 
                                      **kwargs)
            print  "+  model prediction checks met!"

            spg = SaliencePredictionAggregator()
            cov = spg.check_coverage(
                event, corpus, self.feature_set,
                self.key, model_events - set([event]), **kwargs)
            if cov != 1:
                spg.get(event, corpus, self.feature_set,
                        self.key, model_events - set([event]), **kwargs)
            print "+  model predictions aggregated!"
#
#   
        ap = APSummarizer() 
        apsal = APSalienceSummarizer()
        hac = HACSummarizer()
#
        print "running summarizer jobs..."
        jobs = []

        import random
        random.seed(42)
        random.shuffle(self.event_data)
        dev_events = self.event_data[0:3]
        eval_events = self.event_data[3:]


        if 'ablation' in self.key:
            print "I am here"
            for event, corpus in eval_events:
                print event.fs_name()
                jobs.append((event, corpus, self.key, self.feature_set, apsal))
            apsal_tsv_dir = apsal.get_tsv_dir(self.key, self.feature_set)
            if not os.path.exists(apsal_tsv_dir):
                os.makedirs(apsal_tsv_dir)

        elif 'cross' in self.key:
            for event, corpus in dev_events:
                print event.fs_name()
                apsal_tsv_dir = apsal.get_tsv_dir(self.key, self.feature_set)
                if not os.path.exists(apsal_tsv_dir):
                    os.makedirs(apsal_tsv_dir)
                if not os.path.exists(ap.dir_):
                    os.makedirs(ap.dir_)        
                if not os.path.exists(hac.dir_):
                    os.makedirs(hac.dir_)

                for cutoff in np.arange(.9, 5.05, .05):
                    jobs.append(
                        (event, corpus, self.key, self.feature_set, hac, cutoff))
                
                jobs.append((event, corpus, self.key, self.feature_set, ap))
                jobs.append((event, corpus, self.key, self.feature_set, apsal))

            for event, corpus in eval_events:
                print event.fs_name()
                apsal_tsv_dir = apsal.get_tsv_dir(self.key, self.feature_set)
                if not os.path.exists(apsal_tsv_dir):
                    os.makedirs(apsal_tsv_dir)
                if not os.path.exists(ap.dir_):
                    os.makedirs(ap.dir_)        
                if not os.path.exists(hac.dir_):
                    os.makedirs(hac.dir_)

                for cutoff in np.arange(.9, 5.05, .1):
                    jobs.append(
                        (event, corpus, self.key,
                         self.feature_set, hac, cutoff))
                
                jobs.append((event, corpus, self.key, self.feature_set, ap))
                jobs.append((event, corpus, self.key, self.feature_set, apsal))



            
        n_procs = kwargs.get(u'n_procs', 1)
        n_jobs = len(jobs)
        #pool = Pool(n_procs)
        #pb = ProgressBar(n_jobs)
#        for job in jobs:
#            sum_worker(job)
       
        jobs.reverse() 
        self.do_work(sum_worker, jobs, **kwargs)
        #for job in jobs:
        #    sum_worker(job)
        #    pb.update()
        #for result in pool.imap_unordered(sum_worker, jobs):
        #    pb.update()
#

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

def sum_worker(job_queue, result_queue, **kwargs):
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

