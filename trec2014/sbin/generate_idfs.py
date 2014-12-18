import argparse
import cuttsum.events
import cuttsum.kba
import cuttsum.data as data
import pandas as pd


def generate_idfs(event, corpus, domains=None,
                       overwrite=False, n_procs=1):

    print "Generating idf counts for"
    print "\t", event.fs_name(), "/", corpus.fs_name()
    if overwrite is True:
        idfs = data.IdfResource()
        idfs.get(event, corpus, domains=domains, 
            overwrite=overwrite, n_procs=n_procs, progress_bar=True)
    else:    
        idfs = data.IdfResource()
        coverage = idfs.check_coverage(
            event, corpus, domains=domains, 
            overwrite=overwrite, n_procs=n_procs, progress_bar=True)
        if coverage != 1:
            idfs.get(event, corpus, domains=domains, 
                overwrite=overwrite, n_procs=n_procs, progress_bar=True)

def print_report(args):

    idfs = data.IdfResource()
    results = []

    if args.fetch_sc2013 is True or args.fetch_all is True:
        corpus = cuttsum.kba.EnglishAndUnknown2013()
        for event in cuttsum.events.get_2013_events(
            by_query_ids=args.query_ids):
            coverage_per = idfs.check_coverage(
                event, corpus, domains=args.domains, 
                overwrite=args.overwrite, n_procs=args.n_procs,
                progress_bar=True)
            results.append(
                {u"coverage": coverage_per,
                 u"corpus": corpus.fs_name(),
                 u"query id": event.query_id})

    if args.fetch_sc2014_serif is True or args.fetch_all is True:
        corpus = cuttsum.kba.SerifOnly2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=args.query_ids):
            coverage_per = idfs.check_coverage(
                event, corpus, domains=args.domains, 
                overwrite=args.overwrite, n_procs=args.n_procs,
                progress_bar=True)
            results.append(
                {u"coverage": coverage_per,
                 u"corpus": corpus.fs_name(),
                 u"query id": event.query_id})

    if args.fetch_sc2014_ts is True or args.fetch_all is True:
        corpus = cuttsum.kba.FilteredTS2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=args.query_ids):
            coverage_per = idfs.check_coverage(
                event, corpus, domains=args.domains, 
                overwrite=args.overwrite, n_procs=args.n_procs,
                progress_bar=True)
            results.append(
                {u"coverage": coverage_per,
                 u"corpus": corpus.fs_name(),
                 u"query id": event.query_id})

    report = pd.DataFrame(
        results, columns=[u"query id", u"corpus", u"coverage"])
    print report 

def generate_all_idfs(args):

    if args.fetch_sc2013 is True or args.fetch_all is True:
        corpus = cuttsum.kba.EnglishAndUnknown2013()
        for event in cuttsum.events.get_2013_events(
            by_query_ids=args.query_ids):
            generate_idfs(
                event, corpus, args.domains,
                args.overwrite, args.n_procs)

    if args.fetch_sc2014_serif is True or args.fetch_all is True:
        corpus = cuttsum.kba.SerifOnly2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=args.query_ids):
            generate_idfs(
                event, corpus, args.domains,
                args.overwrite, args.n_procs)

    if args.fetch_sc2014_ts is True or args.fetch_all is True:
        corpus = cuttsum.kba.FilteredTS2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=args.query_ids):
            generate_idfs(
                event, corpus, args.domains,
                args.overwrite, args.n_procs)

def main(args):
    if args.report is True:
        print_report(args)
    else:
        generate_all_idfs(args)
        print u"Complete!"

if __name__ == u'__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(u'-q', u'--query-ids', nargs=u'+',
                        help=u'Event query ids, e.g. TS14.11',
                        default=None, required=False)

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

    args = parser.parse_args() 
    main(args)
