from cuttsum import kba
from cuttsum import events
import os
from multiprocessing import Pool

domains = set([u'news', u'MAINSTREAM_NEWS'])

def worker(args):
    path, dest, corpus = args
    date, chunk = path.split('/')
    if os.path.exists(os.path.join(dest, date, chunk)):
        return path
    else:
        corpus.download_path(path, dest)
        return path

def main(n_procs, trec_dir):
    pool = Pool(n_procs)
    corpus = kba.EnglishAndUnknown2013()
    for event in events.get_2013_events():
        print event 
        event_dir = os.path.join(
            u'2013_english_and_unknown', u'{}'.format(event.fs_safe_title()))
        dest = os.path.join(trec_dir, event_dir)
        if not os.path.exists(dest):
            os.makedirs(dest)
        jobs = [(path, dest, corpus) for ts, dmn, ct, path
                in corpus.paths(event.start, event.end, domains)]
        for result in pool.imap(worker, jobs):
            print result

    corpus = kba.FilteredTS2014()
    for event in events.get_2014_events():
        print event 
        event_dir = os.path.join(
            u'2014_filtered_ts', u'{}'.format(event.fs_safe_title()))
        dest = os.path.join(trec_dir, event_dir)
        if not os.path.exists(dest):
            os.makedirs(dest)
        jobs = [(path, dest, corpus) for ts, dmn, ct, path
                in corpus.paths(event.start, event.end, domains)]
        pool = Pool(n_procs)
        for result in pool.imap(worker, jobs):
            print result

    corpus = kba.SerifOnly2014()
    for event in events.get_2014_events():
        print event 
        event_dir = os.path.join(
            u'2014_serif_only', u'{}'.format(event.fs_safe_title()))
        dest = os.path.join(trec_dir, event_dir)
        if not os.path.exists(dest):
            os.makedirs(dest)
        jobs = [(path, dest, corpus) for ts, dmn, ct, path
                in corpus.paths(event.start, event.end, domains)]
        pool = Pool(n_procs)
        for result in pool.imap(worker, jobs):
            print result


if __name__ == u'__main__':
    n_procs = 24
    trec_dir = os.getenv(u"TREC_DATA", None)
    if trec_dir is None:
        trec_dir = os.path.join(os.getenv(u'HOME', u'.'), u'trec-data')
        print u'TREC_DATA env. var not set. Using {} to store data.'.format(
            trec_dir)
    
    if not os.path.exists(trec_dir):
        os.makedirs(trec_dir)
    main(n_procs, trec_dir)
