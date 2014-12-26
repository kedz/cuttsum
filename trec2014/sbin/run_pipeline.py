import argparse
import cuttsum
from cuttsum.data import LMResource, UrlListResource, SCChunkResource, \
    DomainLMInputResource, DomainLMResource
from cuttsum.pipeline import ArticlesResource, SentenceFeaturesResource
import pandas as pd

def print_report(resources=list(), query_ids=None, event_types=None,
                 fetch_all=False, fetch_sc2013=False, 
                 fetch_sc2014_ts=False, fetch_sc2014_serif=False, **kwargs):

    for resource_name in resources:
        resource = globals()[resource_name]()
        results = []

        if fetch_sc2013 is True or fetch_all is True:
            corpus = cuttsum.corpora.EnglishAndUnknown2013()
            for event in cuttsum.events.get_2013_events(
                by_query_ids=query_ids, by_event_types=event_types):
                coverage_per = resource.check_coverage(
                    event, corpus, **kwargs)
                results.append(
                    {u"coverage": coverage_per,
                     u"corpus": corpus.fs_name(),
                     u"query id": event.query_id})

        if fetch_sc2014_serif is True or fetch_all is True:
            corpus = cuttsum.corpora.SerifOnly2014()
            for event in cuttsum.events.get_2014_events(
                by_query_ids=query_ids, by_event_types=event_types):
                coverage_per = resource.check_coverage(
                    event, corpus, **kwargs)
                results.append(
                    {u"coverage": coverage_per,
                     u"corpus": corpus.fs_name(),
                     u"query id": event.query_id})

        if fetch_sc2014_ts is True or fetch_all is True:
            corpus = cuttsum.corpora.FilteredTS2014()
            for event in cuttsum.events.get_2014_events(
                by_query_ids=query_ids, by_event_types=event_types):
                coverage_per = resource.check_coverage(
                    event, corpus, **kwargs)
                results.append(
                    {u"coverage": coverage_per,
                     u"corpus": corpus.fs_name(),
                     u"query id": event.query_id})

        report = pd.DataFrame(
            results, columns=[u"query id", u"corpus", u"coverage"])
        print resource
        print report 

def get_resources(resources=list(), query_ids=None, event_types=None,
                  fetch_all=False, fetch_sc2013=False, 
                  fetch_sc2014_ts=False, fetch_sc2014_serif=False, **kwargs):

    for resource_name in resources:
        resource = globals()[resource_name]()

        if fetch_sc2013 is True or fetch_all is True:
            corpus = cuttsum.corpora.EnglishAndUnknown2013()
            for event in cuttsum.events.get_2013_events(
                by_query_ids=query_ids, by_event_types=event_types):
                get_resource(resource, event, corpus, **kwargs)

        if fetch_sc2014_serif is True or fetch_all is True:
            corpus = cuttsum.corpora.SerifOnly2014()
            for event in cuttsum.events.get_2014_events(
                by_query_ids=query_ids, by_event_types=event_types):
                get_resource(resource, event, corpus, **kwargs)

        if fetch_sc2014_ts is True or fetch_all is True:
            corpus = cuttsum.corpora.FilteredTS2014()
            for event in cuttsum.events.get_2014_events(
                by_query_ids=query_ids, by_event_types=event_types):
                get_resource(resource, event, corpus, **kwargs)

def get_resource(resource, event, corpus, **kwargs):

    print "Getting {} for".format(resource)
    print "\t", event.fs_name(), "/", corpus.fs_name()
    if kwargs.get(u'overwrite', False) is True:
        resource.get(event, corpus, **kwargs)
    else:    
        coverage = resource.check_coverage(event, corpus, **kwargs)
        if coverage != 1:
            resource.get(event, corpus, **kwargs)


def print_event_info(resources=list(), query_ids=None, event_types=None,
                     fetch_all=False, fetch_sc2013=False, 
                     fetch_sc2014_ts=False, fetch_sc2014_serif=False,
                     **kwargs):


    if fetch_sc2013 is True or fetch_all is True:
        corpus = cuttsum.corpora.EnglishAndUnknown2013()
        for event in cuttsum.events.get_2013_events(
            by_query_ids=query_ids, by_event_types=event_types):
            print event

    if fetch_sc2014_serif is True or fetch_all is True:
        corpus = cuttsum.corpora.SerifOnly2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=query_ids, by_event_types=event_types):
            print event

    if fetch_sc2014_ts is True or fetch_all is True:
        corpus = cuttsum.corpora.FilteredTS2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=query_ids, by_event_types=event_types):
            print event

def make_dependency_graph(**kwargs):
    import sys
    import os
    import pkgutil
    import inspect
    
    for loader, module_name, is_pkg in pkgutil.walk_packages(
        cuttsum.__path__, cuttsum.__name__ + ".", onerror=None):

        module = loader.find_module(module_name).load_module(module_name)
        clsmembers = inspect.getmembers(
            sys.modules[module.__name__], lambda x: inspect.isclass(x) and issubclass(x, cuttsum.data.Resource))  
        for name, clazz in clsmembers:
            if name == "Resource":
                continue
            print name
            for dep in clazz().dependencies():
                print "\t", dep.__name__

#   file, path, desc = imp.find_module(cuttsum)
   
#   for 

def main(**kwargs):

    #opts = sorted(kwargs.items(), key=lambda x: x[0])
    #for opt, val in opts:
    #    print opt, val

    if kwargs.get(u'dependency_graph', False) is True:
        make_dependency_graph(**kwargs)
    if kwargs.get(u'event_info', False) is True:
        print_event_info(**kwargs)
    elif kwargs.get(u'report', False) is True:
        print_report(**kwargs)
    else:
        get_resources(**kwargs)
        print u"Complete!"



if __name__ == u'__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(u'-q', u'--query-ids', nargs=u'+',
                        help=u'Event query ids, e.g. TS14.11',
                        default=None, required=False)

    parser.add_argument(u'-t', u'--event-types', nargs=u'+',
                        help=u'Event types e.g. earthquake',
                        default=None, required=False)

    parser.add_argument(u'-r', u'--resources', nargs=u'+',
                        help=u'List of resources to get or report.',
                        default=[u'ArticlesResource'], required=False)

    parser.add_argument(u'-d', u'--domains', nargs=u'+',
                        help=u'Chunk domain types to accept, eg \'news\'',
                        type=set,
                        default=set(['news', 'MAINSTREAM_NEWS']), 
                        required=False)

    parser.add_argument(u'-p', u'--n-procs', type=int,
                        help=u'Number of processes to run',
                        default=1, required=False)

    parser.add_argument(u'-w', u'--overwrite', action=u'store_true',
                        help=u'Overwrite previous files if any')

    parser.add_argument(u'--fetch-sc2013', action=u'store_true',
                        help=u'Fetch English and Unknown 2013 Stream Corpus')

    parser.add_argument(u'--fetch-sc2014-serif', action=u'store_true',
                        help=u'Fetch  Serif only 2014 Stream Corpus')

    parser.add_argument(u'--fetch-sc2014-ts', action=u'store_true',
                        help=u'Fetch TS Filtered 2014 Stream Corpus')

    parser.add_argument(u'--fetch-all', action=u'store_true',
                        help=u'Fetch links for all corpora.')
    
    parser.add_argument(u'--report', action=u'store_true',
                        help=u'Print report of resource coverage.')

    parser.add_argument(u'--dependency-graph', action=u'store_true',
                        help=u'Make png of dependency graph.')

    parser.add_argument(u'--event-info', action=u'store_true',
                        help=u'Print event info.')

    parser.add_argument(u'--hide-progress-bar', action=u'store_false', 
                        dest=u'progress_bar', default=True,
                        help=u'Hide progress bar when getting resource.')

    parser.add_argument(u'--preroll', type=int,
                        help=u'Extend event start time by PREROLL hours.',
                        default=0, required=False)

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
