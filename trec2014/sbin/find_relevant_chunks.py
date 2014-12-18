import cuttsum.events
import cuttsum.kba
from cuttsum.detector import ArticleDetector
import streamcorpus as sc
import os
import re
from multiprocessing import Pool
import argparse

domains = set([u'news', u'MAINSTREAM_NEWS'])

def make_dest_dir(event, trec_dir=None):
    if trec_dir is None:
        trec_dir = os.getenv(u'TREC_DATA', u'.')
    data_dir = os.path.join(trec_dir, u"relevant-chunks", event.fs_name())
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

def worker(args):
    data_dir, ts, paths, event, corpus, overwrite = args

    patt = event.regex_pattern()

    opath = os.path.join(
        data_dir, u'{}.sc.gz'.format(ts.strftime(u'%Y-%m-%d-%H')))
    if os.path.exists(opath):
        if overwrite is True:
            os.remove(opath)
        else:
            return opath

    artcl_detect = ArticleDetector(event)

    with sc.Chunk(path=opath, mode='wb', message=corpus.sc_msg()) as ochunk:
        for path in paths:
            for si in sc.Chunk(path=path, message=corpus.sc_msg()):
                if si.body.clean_visible is None:
                    continue
                elif patt.search(si.body.clean_visible, re.I):
                    if corpus.annotator() not in si.body.sentences:
                        continue
                    sent_idxs = artcl_detect.find_articles(
                        si, annotator=corpus.annotator())
                    if len(sent_idxs) > 0:
                        rel_sents = []
                        for sent_idx in sent_idxs:
                            rel_sents.append(
                                si.body.sentences[
                                    corpus.annotator()][sent_idx])
                        si.body.sentences[u'article-clf'] = rel_sents
                        ochunk.add(si)

    return opath

def find_relevant_chunks(event, corpus, n_procs, overwrite):
    data_dir = make_dest_dir(event)        

    jobs = []
    for ts, paths in corpus.chunk_paths(event.start, event.end, domains):
        jobs.append((data_dir, ts, paths, event, corpus, overwrite))
    
    if n_procs == 1:
        for job in jobs:
            print "Written", worker(job)
    else:
        pool = Pool(n_procs)
        for result in pool.imap(worker, jobs):
            print "Written", result

def main(args):

    if args.skip_sc2013 is False:
        corpus = cuttsum.kba.EnglishAndUnknown2013()
        for event in cuttsum.events.get_2013_events(
            by_query_ids=args.query_ids):
            find_relevant_chunks(event, corpus, n_procs, overwrite)

#    if args.skip_sc2014_serif is False:
#        corpus = cuttsum.kba.SerifOnly2014()
#        for event in cuttsum.events.get_2014_events(
#            by_query_ids=args.query_ids):
#            find_relevant_chunks(event, corpus, args.n_procs, args.overwrite)

    if args.skip_sc2014_ts is False:
        corpus = cuttsum.kba.FilteredTS2014()
        for event in cuttsum.events.get_2014_events(
            by_query_ids=args.query_ids):
            find_relevant_chunks(event, corpus, args.n_procs, args.overwrite)

    print "Complete!"

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

    parser.add_argument(u'--skip-sc2013', action=u'store_true',
                        help=u'Skip English and Unknown 2013 Stream Corpus')

    parser.add_argument(u'--skip-sc2014-serif', action=u'store_true',
                        help=u'Skip Serif only 2014 Stream Corpus')

    parser.add_argument(u'--skip-sc2014-ts', action=u'store_true',
                        help=u'Skip TS Filtered 2014 Stream Corpus')

    args = parser.parse_args()
    
    main(args)
