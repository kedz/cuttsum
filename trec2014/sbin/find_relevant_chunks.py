import cuttsum.events
import cuttsum.kba
import os
from collections import defaultdict
import streamcorpus as sc
import re
from cuttsum.detector import ArticleDetector
from pkg_resources import resource_listdir, resource_filename
from multiprocessing import Pool

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

def find_relevant_chunks(events, corpus, n_procs, overwrite):
    for e in events:
        print e
        data_dir = make_dest_dir(e)        

        jobs = []
        for ts, paths in corpus.chunk_paths(e.start, e.end, domains):
            jobs.append((data_dir, ts, paths, e, corpus, overwrite))
        
        if n_procs == 1:
            for job in jobs:
                print "Written", worker(job)
        else:
            pool = Pool(n_procs)
            for result in pool.imap(worker, jobs):
                print "Written", result

def main(n_procs, overwrite):

    corpus = cuttsum.kba.EnglishAndUnknown2013()
    events = [e for e in cuttsum.events.get_2013_events()]
    find_relevant_chunks(events, corpus, n_procs, overwrite)

if __name__ == u'__main__':
    n_procs = 24
    overwrite = True
    main(n_procs, overwrite)
