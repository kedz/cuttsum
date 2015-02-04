import argparse
import cuttsum
import cuttsum.data
import pandas as pd


def print_reports(query_ids=None, event_types=None,
                  fetch_all=False, fetch_sc2013=False, 
                  fetch_sc2014_ts=False, fetch_sc2014_serif=False, **kwargs):
    
    if fetch_sc2013 is True or fetch_all is True:
        corpus = cuttsum.corpora.EnglishAndUnknown2013()
        for event in cuttsum.events.get_2013_events(
            by_query_ids=query_ids, by_event_types=event_types):
            
            print_report(event, corpus, **kwargs)

    if fetch_sc2014_ts is True or fetch_all is True:
        corpus = cuttsum.corpora.FilteredTS2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=query_ids, by_event_types=event_types):
            
            print_report(event, corpus, **kwargs)

def print_report(event, corpus, **kwargs):

    print event.query_id, event.fs_name(), "/", corpus.fs_name()
    for group in cuttsum.data.get_sorted_dependencies():
        for resource in group:
            print "    {:50s} : {:7.3f} %".format(
                resource, 
                100. * resource.check_coverage(event, corpus, **kwargs))
    print 

def run_event_pipelines(
    active_resources=None,
    query_ids=None, event_types=None, fetch_all=False, fetch_sc2013=False, 
    fetch_sc2014_ts=False, fetch_sc2014_serif=False, **kwargs):

    if active_resources is None:
        active_resources = set(cuttsum.data.get_resource_managers())
    else:
        active_resources = set([cuttsum.data.get_resource_manager(rstring)
                                for rstring in active_resources])
 
    if fetch_sc2013 is True or fetch_all is True:
        corpus = cuttsum.corpora.EnglishAndUnknown2013()
        for event in cuttsum.events.get_2013_events(
            by_query_ids=query_ids, by_event_types=event_types):

            run_pipeline(event, corpus, active_resources, **kwargs)

    if fetch_sc2014_ts is True or fetch_all is True:
        corpus = cuttsum.corpora.FilteredTS2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=query_ids, by_event_types=event_types):

            run_pipeline(event, corpus, active_resources, **kwargs)

def run_pipeline(event, corpus, active_resources, **kwargs):
    print event.query_id, event.fs_name(), "/", corpus.fs_name()
    for group in cuttsum.data.get_sorted_dependencies(reverse=False):
        for resource in group:
            coverage = resource.check_coverage(event, corpus, **kwargs)
            print "    {:50s} : {:7.3f} %".format(resource, 100. * coverage)
            if resource in active_resources and coverage != 1:
                resource.get(event, corpus, **kwargs)
    print 


#
#def make_dependency_graph(**kwargs):
#    import sys
#    import os
##    import pygraphviz as pgv
#   
#    kwargs['no_resolve'] = True
#    
#    corpus2013 = cuttsum.corpora.EnglishAndUnknown2013()
#    events2013 = [(event, corpus2013) for event
#                  in cuttsum.events.get_2013_events()]
#
#    corpus2014 = cuttsum.corpora.FilteredTS2014()
#    events2014 = [(event, corpus2014) for event
#                  in cuttsum.events.get_2014_events()] 
#
#    #events = events2013 + events2014
#    events = events2013
#
##    G=pgv.AGraph()
##    G.graph_attr.update(size="40,40")
##    G.node_attr['shape'] = 'box'
#
#
#    for event, corpus in events:
#        print event.fs_name(), "/", corpus.fs_name()
#        for resource in cuttsum.data.get_resource_managers():
#            print event.fs_name(), resource
#            deps = resource.check_unmet_dependencies(event, corpus, **kwargs)
#            print deps
#            resource.get_dependencies(event, corpus, deps, **kwargs)
#            print
#
#    sys.exit()
#    visited = set()
#    root_node = None
#    for node in G.iternodes():
#        if G.in_degree(node) == 0:
#            root_node = node
#
#
#    current_depth = 0
#    queue = [(root_node, 0)]
#    while len(queue) > 0:
#        node, depth = queue.pop(0)
#        if current_depth != depth:
#            print 
#            current_depth = depth
#        print depth, node, name2inst[name].check_coverage(event, corpus, **kwargs),
#        for tgt, src in G.edges(node):
#            if src not in visited:
#                queue.append((src, depth + 1))
#                visited.add(src)
#    
#
#    sys.exit()
#
#
#
#    G=pgv.AGraph()
#    G.graph_attr.update(size="40,40")
#    G.node_attr['shape'] = 'box'
#    for loader, module_name, is_pkg in pkgutil.walk_packages(
#        cuttsum.__path__, cuttsum.__name__ + ".", onerror=None):
#
#        module = loader.find_module(module_name).load_module(module_name)
#        clsmembers = inspect.getmembers(
#            sys.modules[module.__name__],
#            lambda x: inspect.isclass(x) and issubclass(x, cuttsum.data.Resource))  
#
#
#        event = events[0]
#
#
#        for name, clazz in clsmembers:
#            if name == "Resource":
#                continue
#            print name
#
#            for event in events:
#                cov_node_name = event[0].fs_name() + "." + name
#                G.add_edge(name, cov_node_name)
#                n = G.get_node(cov_node_name)
#                cov = round(clazz().check_coverage(event[0], event[1], **kwargs), 3)
#                n.attr['label'] = cov
#                if cov == 0:
#                    n.attr['color'] = 'red'
#                elif cov < 1:
#                    n.attr['color'] = 'yellow'
#                else:
#                    n.attr['color'] = 'green'
#
#                n.attr['fontsize'] = 8
#                n.attr['shape'] = 'circle'
#                n.attr['fixedsize'] = True
#                n.attr['height'] = .3
#                n.attr['width'] = .3
#
#            for dep in clazz().dependencies():
#                print "\t", dep.__name__
#                G.add_edge(name, dep.__name__)
#                    
#                
#
#
#    G.draw('test.png', prog='dot')
#
##   file, path, desc = imp.find_module(cuttsum)
#   
##   for 

def print_event_info(query_ids=None, event_types=None, **kwargs):

    events = []
    for event in cuttsum.events.get_2013_events(
        by_query_ids=query_ids, by_event_types=event_types):

        events.append(event)

    for event in cuttsum.events.get_2014_events(
        by_query_ids=query_ids, by_event_types=event_types):

        events.append(event)            

    for event in events:            
        title = event.title.encode('utf-8')
        print title
        print '=' * len(title)
        print "query id  :", event.query_id
        print "fs_name   :", event.fs_name()
        print "type      :", event.type
        print "start     :", event.start
        print "end       :", event.end
        print "duration  :", '{:5d} hrs'.format(
            len(event.list_event_hours()))
        print "query     :", event.query
        print  

def run_summarizer_jobs(feature_ablation=False, cross_fold=False, **kwargs):
    import cuttsum.pipeline.jobs as jobs

    from cuttsum.pipeline.salience import SalienceModels
    if feature_ablation:
        for jnum, job in enumerate(
            jobs.feature_ablation_jobs(u'feature-ablation')):

            if jnum != 2: 
                continue
            print job
            job.start(**kwargs)

    if cross_fold:
        for job in jobs.event_cross_validation_jobs("crossval"):
            print job
            job.start(**kwargs)

#        import sys
#        sys.exit()
#        sm = SalienceModel()
#        print job
#        for event, corpus in job.testing_data:
#            if event.fs_name() != '2012_buenos_aires_rail_disaster':
#                continue
#            print "fetching resources..."
#            run_pipeline(
#                event, corpus, 
#                set(cuttsum.data.get_resource_managers()), **kwargs)
#
#            sentence_features = \
#                cuttsum.data.get_resource_manager(u'SentenceFeaturesResource')
#            nugget_similarities = \
#                cuttsum.data.get_resource_manager(
#                    u'NuggetSimilaritiesResource')
#            assert sentence_features.check_coverage(event, corpus) == 1.0
#            assert nugget_similarities.check_coverage(event, corpus) == 1.0
#            print  "dependency checks met!"
#
#            fs = job.feature_set
#            if sm.check_coverage(event, corpus, fs, **kwargs) != 1:
#                sm.get(event, corpus, fs, **kwargs)
#
#        import sys
#        sys.exit()

             
def main(run_summarizer=False, report=False, event_info=False, run=False, 
         dependency_graph=False, **kwargs):
    if run_summarizer:
        del kwargs[u'active_resources']
        run_summarizer_jobs(**kwargs)
    if dependency_graph:
        make_dependency_graph(**kwargs)
    if event_info:
        print_event_info(**kwargs)
    if report:
        print_reports(**kwargs)
    if run:
        run_event_pipelines(**kwargs)
        print u"Complete!"

if __name__ == u'__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(u'--report', action=u'store_true',
                        help=u'Print report of resource coverage.')

    parser.add_argument(u'--dependency-graph', action=u'store_true',
                        help=u'Make png of dependency graph.')

    parser.add_argument(u'--event-info', action=u'store_true',
                        help=u'Print event info.')

    parser.add_argument(u'--run', action=u'store_true',
                        help=u'Run pipeline.')


    parser.add_argument(u'--run-summarizer', action=u'store_true',
                        dest=u'run_summarizer',
                        help=u'Train salience models and run summarizer.')

    parser.add_argument(u'--feature-ablation', action=u'store_true',
                        help=u'Run feature ablation jobs.')

    parser.add_argument(u'--cross-fold', action=u'store_true',
                        help=u'Run event cross fold jobs.')
   
    parser.add_argument(u'-q', u'--query-ids', nargs=u'+',
                        help=u'Filter task by query ids, e.g. TS14.11',
                        default=None, required=False)

    parser.add_argument(u'-t', u'--event-types', nargs=u'+',
                        help=u'Filter task by event types e.g. earthquake',
                        default=None, required=False)

    parser.add_argument(u'-d', u'--domains', nargs=u'+',
                        help=u'Chunk domain types to accept, eg \'news\'',
                        type=set,
                        default=set(['news', 'MAINSTREAM_NEWS']), 
                        required=False)

    parser.add_argument(u'--run-only', nargs=u'+', dest=u'active_resources',
                        help=u'Run only the listed components.',
                        default=None, required=False)

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
    
    parser.add_argument(u'--hide-progress-bar', action=u'store_false', 
                        dest=u'progress_bar', default=True,
                        help=u'Hide progress bar when getting resource.')

    parser.add_argument(u'--preroll', type=int,
                        help=u'Extend event start time by PREROLL hours.',
                        default=0, required=False)

    parser.add_argument(u'--n-samples', type=int,
                        help=u'Number of salience models',
                        default=10, required=False)

    parser.add_argument(u'--sample-size', type=int,
                        help=u'Sample size of salience models',
                        default=100, required=False)

    parser.add_argument(u'--tune', action=u'store_true', dest=u'tune',
                        default=False, 
                        help=u'Find optimal settings on dev set.')


    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
