import argparse
import cuttsum.events
import cuttsum.kba
import cuttsum.data as data
import pandas as pd


def fetch_event_chunks(event, corpus, domains=None,
                       overwrite=False, n_procs=1):

    print "Fetching stream corpus chunks for"
    print "\t", event.fs_name(), "/", corpus.fs_name()
    if overwrite is True:
        chunks.get(event, corpus, domains=domains, 
            overwrite=overwrite, n_procs=n_procs, progress_bar=True)
    else:    
        chunks = data.KBAChunkResource()
        coverage = chunks.check_coverage(
            event, corpus, domains=domains, 
            overwrite=overwrite, n_procs=n_procs, progress_bar=True)
        if coverage != 1:
            chunks.get(event, corpus, domains=domains, 
                overwrite=overwrite, n_procs=n_procs, progress_bar=True)
         
#    if overwrite is True or chunks.check_coverage(event, corpus) != 1:
#        print "Retrieving links for"
#        print "\t", event.title, "/", corpus.fs_name()
#        chunks.get(event, corpus, overwrite=overwrite)

def print_report(args):

    chunks = data.KBAChunkResource()
    results = []

    if args.fetch_sc2013 is True or args.fetch_all is True:
        corpus = cuttsum.kba.EnglishAndUnknown2013()
        for event in cuttsum.events.get_2013_events(
            by_query_ids=args.query_ids):
            coverage_per = chunks.check_coverage(
                event, corpus, domains=args.domains, 
                overwrite=args.overwrite, n_procs=args.n_procs,
                progress_bar=True)
            n_chunks = len(chunks.get_chunk_info_paths_urls(event, corpus))
            results.append(
                {u"coverage": coverage_per,
                 u"corpus": corpus.fs_name(),
                 u"query id": event.query_id,
                 u"num chunks": n_chunks})

    if args.fetch_sc2014_serif is True or args.fetch_all is True:
        corpus = cuttsum.kba.SerifOnly2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=args.query_ids):
            coverage_per = chunks.check_coverage(
                event, corpus, domains=args.domains, 
                overwrite=args.overwrite, n_procs=args.n_procs,
                progress_bar=True)
            n_chunks = len(chunks.get_chunk_info_paths_urls(event, corpus))
            results.append(
                {u"coverage": coverage_per,
                 u"corpus": corpus.fs_name(),
                 u"query id": event.query_id,
                 u"num chunks": n_chunks})

    if args.fetch_sc2014_ts is True or args.fetch_all is True:
        corpus = cuttsum.kba.FilteredTS2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=args.query_ids):
            coverage_per = chunks.check_coverage(
                event, corpus, domains=args.domains, 
                overwrite=args.overwrite, n_procs=args.n_procs,
                progress_bar=True)
            n_chunks = len(chunks.get_chunk_info_paths_urls(event, corpus))
            results.append(
                {u"coverage": coverage_per,
                 u"corpus": corpus.fs_name(),
                 u"query id": event.query_id,
                 u"num chunks": n_chunks})

    report = pd.DataFrame(
        results, columns=[u"query id", u"corpus", u"coverage", u"num chunks"])
    print report 

def fetch_all_chunks(args):

    if args.fetch_sc2013 is True or args.fetch_all is True:
        corpus = cuttsum.kba.EnglishAndUnknown2013()
        for event in cuttsum.events.get_2013_events(
            by_query_ids=args.query_ids):
            fetch_event_chunks(
                event, corpus, args.domains,
                args.overwrite, args.n_procs)

    if args.fetch_sc2014_serif is True or args.fetch_all is True:
        corpus = cuttsum.kba.SerifOnly2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=args.query_ids):
            fetch_event_chunks(
                event, corpus, args.domains,
                args.overwrite, args.n_procs)

    if args.fetch_sc2014_ts is True or args.fetch_all is True:
        corpus = cuttsum.kba.FilteredTS2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=args.query_ids):
            fetch_event_chunks(
                event, corpus, args.domains,
                args.overwrite, args.n_procs)

def main(args):
    if args.report is True:
        print_report(args)
    else:
        fetch_all_chunks(args)
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

#from cuttsum import kba
#from cuttsum import events
#import os
#from multiprocessing import Pool
#import subprocess
#import re
#import time
#
#domains = set([u'news', u'MAINSTREAM_NEWS'])
#
#def validate_chunk_checksum(path):
#    fname = os.path.split(path)[-1]
#    check_sums = fname.split('.')[0].split('-')[2:]
#    #m = re.search(r'\w+-\d+-(.*?)-(.*?)\.sc', fname)
#    #if m is None: 
#    
#    #    return False
#
#    #else:
#    check_sum = check_sums[-1]
#        #check_sum = m.groups()[1]
#    sp_out = subprocess.check_output(
#        "gpg -d {} | xzcat | md5sum".format(path), shell=True)
#    actual_check_sum = sp_out.split(' ')[0]
#    return check_sum == actual_check_sum
#
#def worker(args):
#    path, dest, corpus = args
#    date, chunk = path.split('/')
#    tgt = os.path.join(dest, date, chunk)
#
#    if os.path.exists(tgt):
#        valid = validate_chunk_checksum(tgt)
#    
#        if valid:
#            return path
#        else:
#            print path, "is not correct downloading again."
#            os.remove(tgt)
#    
#    corpus.download_path(path, dest)
#    time.sleep(1)
#    return path
#
#def main(n_procs, trec_dir, 
#         download_2013=False, download_2014_filtered=False):
#    pool = Pool(n_procs)
#
#    if download_2013 is True:
#        corpus = kba.EnglishAndUnknown2013()
#        event_dir = u'2013_english_and_unknown'
#        dest = os.path.join(trec_dir, event_dir)
#        if not os.path.exists(dest):
#            os.makedirs(dest)
#
#        for event in events.get_2013_events():
#            print event 
#            jobs = [(path, dest, corpus) for ts, dmn, ct, path
#                    in corpus.paths(event.start, event.end, domains)]
#            for result in pool.imap(worker, jobs):
#                print result
#
#    if download_2014_filtered is True:
#        corpus = kba.FilteredTS2014()
#        event_dir = u'2014_filtered_ts'
#        dest = os.path.join(trec_dir, event_dir)
#        if not os.path.exists(dest):
#            os.makedirs(dest)
#
#        for event in events.get_2014_events():
#            print event 
#            jobs = [(path, dest, corpus) for ts, dmn, ct, path
#                    in corpus.paths(event.start, event.end, domains)]
#            
#            pool = Pool(n_procs)
#            for result in pool.imap(worker, jobs):
#                print result
#
#    corpus = kba.SerifOnly2014()
#    event_dir = u'2014_serif_only'
#    dest = os.path.join(trec_dir, event_dir)
#    if not os.path.exists(dest):
#        os.makedirs(dest)
#
#    for event in events.get_2014_events():
#        print event 
#        jobs = [(path, dest, corpus) for ts, dmn, ct, path
#                in corpus.paths(event.start, event.end, domains)]
#                
#        pool = Pool(n_procs)
#        for result in pool.imap(worker, jobs):
#            print result
#
#    print "Complete!"
#
#if __name__ == u'__main__':
#    n_procs = 10
#    trec_dir = os.getenv(u"TREC_DATA", None)
#    if trec_dir is None:
#        trec_dir = os.path.join(os.getenv(u'HOME', u'.'), u'trec-data')
#        print u'TREC_DATA env. var not set. Using {} to store data.'.format(
#            trec_dir)
#    
#    if not os.path.exists(trec_dir):
#        os.makedirs(trec_dir)
#    main(n_procs, trec_dir)
