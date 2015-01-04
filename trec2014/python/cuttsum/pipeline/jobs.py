import cuttsum
import random
from .representation import SalienceFeatureSet
from .salience import SalienceModels, SaliencePredictions


def feature_ablation_jobs(key, per_train=.6, random_seed=42):
    def desc(train, test, features):

        all_features = set([u'character', u'language model', u'frequency',
                            u'geographic', u'query'])

        inactive_features = all_features - features.as_set()
        return u"feature ablation, -{}".format(
            u', '.join(inactive_features))

    for job in job_generator(
        random_event_splitter, leave_one_out_feature_selector, desc, key,
        per_train=per_train, random_seed=random_seed):
        yield job

def event_cross_validation_jobs(key):
    def desc(train, test, features):
        test_event, corpus = test[0]
        return u'{}/{}'.format(
            test_event.title, corpus.fs_name())

    for job in job_generator(
        event_n_fold_selector, all_feature_selector, desc, key,):
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
                   for event in cuttsum.events.get_2013_events()]

    corpus_2014 = cuttsum.corpora.FilteredTS2014()
    events_2014 = [(event, corpus_2014)
                   for event in cuttsum.events.get_2014_events()]

    events = events_2013 + events_2014

    # Generate event splits.
    event_splits = event_selector(events, **kwargs)

    # Generate feature splits.
    features = [u'character', u'language model', u'frequency',
                u'geographic', u'query']
    feature_splits = feature_selector(features, **kwargs)

    job_num = 1
    # Yield the all combinations of event splits and feature splits.
    for event_split in event_splits:
        for feature_split in feature_splits:
            train, test = event_split
            desc = job_descriptor(train, test, feature_split)
            job_key = u"{}.{}".format(key, job_num)
            job = PipelineJob(job_key, desc, train, test, feature_split)
            yield job
            job_num += 1

class PipelineJob(object):
    def __init__(self, key, description, 
        training_data, testing_data, feature_set):
        
        self.key = key
        self.description = description
        self.training_data = training_data
        self.testing_data = testing_data
        self.feature_set = feature_set

    def __str__(self):
        return unicode(self).encode(u'utf-8')

    def __unicode__(self):
        strings = []
        strings.append(u"job key: {}".format(self.key))
        strings.append(u"job description: {}".format(self.description))
        strings.append(u"training events:")
        for event, corpus in self.training_data:
            title = event.title
            if len(title) > 35:
                title = title[0:31] + u' ...'
            corpus_name = corpus.fs_name()
            if len(corpus_name) > 17:
                corpus_name = corpus_name[0:13] + u' ...'
            strings.append(
                u'  {:8s} {:14s} {:35s} {:17s}'.format(
                    event.query_id, event.type, title, corpus_name))
        strings.append(u"test events:")
        for event, corpus in self.testing_data:
            title = event.title
            if len(title) > 35:
                title = title[0:31] + u' ...'
            corpus_name = corpus.fs_name()
            if len(corpus_name) > 17:
                corpus_name = corpus_name[0:13] + u' ...'
            strings.append(
                u'  {:8s} {:14s} {:35s} {:17s}'.format(
                    event.query_id, event.type, title, corpus_name))

        strings.append(unicode(self.feature_set))

        return u'\n'.join(strings)


    def start(self, **kwargs):

        print "  checking resources and models"
        print "  ============================="
        for event, corpus in self.testing_data:
            if event.fs_name() != u'2012_buenos_aires_rail_disaster':
                continue
       
                 
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

        for event, corpus in self.testing_data:
            if event.fs_name() != u'2012_buenos_aires_rail_disaster':
                continue
 
            has_all_predictions = self.check_model_predictions(
                event, corpus, self.feature_set, self.key, **kwargs)
            if not has_all_predictions:
                self.predict_salience(event, corpus, self.feature_set,
                                      self.key, **kwargs)


            import sys
            sys.exit()

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
                print "    {:50s} : {:7.3f} %".format(resource, 100. * coverage)
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
        print "  training salience models..."
        sm = SalienceModels()
        sm.train_models(event, corpus, feature_set, self.key, **kwargs) 
        if not self.check_model_pipeline(event, corpus, feature_set, **kwargs):
            raise Exception("Model training failed!") 

    def check_model_predictions(self, event, corpus, 
                                feature_set, key, **kwargs):
        sp = SaliencePredictions()
        coverage = sp.check_coverage(event, corpus, feature_set, 
                                     key, **kwargs)
        return coverage == 1

    def predict_salience(self, event, corpus, feature_set, key, **kwargs):
        sp = SaliencePredictions()
        print "Predicting!"
        if not self.check_model_predictions(event, corpus, feature_set, key,
                                            **kwargs):
            raise Exception("Model prediction failed!") 

