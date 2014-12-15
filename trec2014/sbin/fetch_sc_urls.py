import argparse
import cuttsum.events
import cuttsum.kba
import cuttsum.data as data
import pandas as pd

def get_urls(event, corpus, overwrite=False):
    urllist = data.UrlListResource()
    if overwrite is True or urllist.check_coverage(event, corpus) != 1:
        print "Retrieving links for"
        print "\t", event.title, "/", corpus.fs_name()
        urllist.get(event, corpus, overwrite=overwrite)

def fetch_urls(args):

    if args.fetch_sc2013 is True or args.fetch_all is True:
        corpus = cuttsum.kba.EnglishAndUnknown2013()
        for event in cuttsum.events.get_2013_events(
            by_query_ids=args.query_ids):
                get_urls(event, corpus, args.overwrite)

    if args.fetch_sc2014_serif is True or args.fetch_all is True:
        corpus = cuttsum.kba.SerifOnly2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=args.query_ids):
                get_urls(event, corpus, args.overwrite)

    if args.fetch_sc2014_ts is True or args.fetch_all is True:
        corpus = cuttsum.kba.FilteredTS2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=args.query_ids):
                get_urls(event, corpus, args.overwrite)
          
def print_report(args):

    urllist = data.UrlListResource()
    results = []
    
    if args.fetch_sc2013 is True or args.fetch_all is True:
        corpus = cuttsum.kba.EnglishAndUnknown2013()
        for event in cuttsum.events.get_2013_events(
            by_query_ids=args.query_ids):
            paths = urllist.get_event_url_paths(event, corpus)
            results.append(
                {u"coverage": urllist.check_coverage(event, corpus),
                 u"corpus": corpus.fs_name(),
                 u"query id": event.query_id,
                 u"hour lists": len(paths),
                 u'event dur. (hrs)': event.duration_hours()})

    if args.fetch_sc2014_serif is True or args.fetch_all is True:
        corpus = cuttsum.kba.SerifOnly2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=args.query_ids):
            paths = urllist.get_event_url_paths(event, corpus)
            results.append(
                {u"coverage": urllist.check_coverage(event, corpus),
                 u"corpus": corpus.fs_name(),
                 u"query id": event.query_id,
                 u"hour lists": len(paths),
                 u'event dur. (hrs)': event.duration_hours()})

    if args.fetch_sc2014_ts is True or args.fetch_all is True:
        corpus = cuttsum.kba.FilteredTS2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=args.query_ids):
            paths = urllist.get_event_url_paths(event, corpus)
            results.append(
                {u"coverage": urllist.check_coverage(event, corpus),
                 u"corpus": corpus.fs_name(),
                 u"query id": event.query_id,
                 u"hour lists": len(paths),
                 u'event dur. (hrs)': event.duration_hours()})
    report = pd.DataFrame(
        results, 
        columns=[u"query id", u"corpus", u"coverage", u"hour lists",
                 u"event dur. (hrs)"])
    print report 

def main(args):

    if args.report is True:
        print_report(args)

    else:
        fetch_urls(args)
        print u"Complete!"

if __name__ == u'__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(u'-q', u'--query-ids', nargs=u'+',
                        help=u'Event query ids, e.g. TS14.11',
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
    
    parser.add_argument(u'--report', action=u'store_true',
                        default=False,
                        help=u'Print report of resource coverage.')

    args = parser.parse_args() 
    main(args)
