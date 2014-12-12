import os
import sys
import gzip
import streamcorpus as sc
import cuttsum.events
import cuttsum.kba
import cuttsum.geo

def worker(args):
    corpus, path = args

    locations = set()
    for si in sc.Chunk(path=path, message=corpus.sc_msg()):
        for sentence in si.body.sentences[u'article-clf']:
            for loc_seq in cuttsum.geo.get_loc_sequences(sentence):
                locations.add(loc_seq)
    return tuple(locations)

def get_corpus_paths(events, corpus, data_dir):
    corpus_paths = []
    for event in events:
        event_dir = os.path.join(data_dir, event.fs_name())
        if not os.path.exists(event_dir):
            continue
        for fname in os.listdir(event_dir):
            corpus_paths.append((corpus, os.path.join(event_dir, fname)))   
    return corpus_paths

def main(data_dir=None, n_procs=1):

    trec_dir = os.getenv(u'TREC_DATA', u'.')
    if data_dir is None:
        data_dir = os.path.join(trec_dir, u'relevant-chunks')

    jobs = []
    corpus = cuttsum.kba.EnglishAndUnknown2013()
    events = cuttsum.events.get_2013_events()
    jobs.extend(
        get_corpus_paths(events, corpus, data_dir))

    corpus = cuttsum.kba.FilteredTS2014()
    events = cuttsum.events.get_2014_events()
    jobs.extend(
        get_corpus_paths(events, corpus, data_dir))
   
    locs = set()
    n_jobs = float(len(jobs))
    stdout = sys.stdout
    if n_procs == 1:
        for i, job in enumerate(jobs, 1):
            stdout.write('%{:0.2f} complete...\r'.format(100. * i / n_jobs))
            stdout.flush()
            locs.update(worker(job))
    else:
        from multiprocessing import Pool
        pool = Pool(n_procs)
        for i, result in enumerate(pool.imap_unordered(worker, jobs), 1):
            locs.update(result)
            stdout.write('%{:0.3f} complete...\r'.format(100. * i / n_jobs))
            stdout.flush()
    print
    locs = list(locs)
    locs.sort()
    geo_cache = os.path.join(trec_dir, u'geo_cache.txt.gz')
    with gzip.open(geo_cache, u'w') as f:
        for loc in locs:
            f.write(loc.encode(u'utf-8'))
            f.write('\n')

if __name__ == u'__main__':
    n_procs = 10
    main(data_dir=None, n_procs=n_procs)
