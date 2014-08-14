import os
import sys
import argparse
from cuttsum.event import read_events_xml
from cuttsum.util import gen_dates
from cuttsum.detector import ArticleDetector
import math
import streamcorpus as sc
import re

def main():
    
    event_file, rc_dir, title, out_dir, ad_dir, ldir, nprocs = parse_args()

    event = load_event(title, event_file)
    hours = [dth for dth in gen_dates(event.start, event.end)]
    nhours = len(hours)
    hours_per_proc = int(math.ceil(nhours / float(nprocs)))

    jobs = []
    pid = 1
    for i in xrange(0, nhours, hours_per_proc):
        log_file = os.path.join(ldir, '.rel_extractor_{}.log'.format(pid))
        jobs.append((rc_dir, out_dir, hours[i:i+hours_per_proc], 
                     event, ad_dir, log_file))
        pid += 1  

    if nprocs == 1:
        for job in jobs:
            worker(job)
    else:
        import multiprocessing as mp
        pool = mp.Pool(nprocs)
        x = pool.map_async(worker, jobs)
        x.get()
        pool.close()
        pool.join()


    # COMPILE LOG INFO
    log_data = []
    for fname in os.listdir(ldir):
        if re.search(r'\.rel_extractor', fname):
            with open(os.path.join(ldir, fname), 'r') as f:
                for line in f:
                    items = line.strip().split('\t')
                    if len(items) != 5:
                        continue
                    log_data.append(items)
            os.remove(os.path.join(ldir, fname))

    log_data.sort(key=lambda x: x[0])

    log_fname = os.path.join(ldir, 'rel_log.txt')
    with open(log_fname, 'w') as f:
        f.write("hour\tnum_docs\tnum_sents\tnum_relevant_docs")
        f.write("\tnum_relevant_sents\n")
        f.flush()    
        for date, n_docs, n_sents, n_rel_docs, n_rel_sents in log_data:
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(date, n_docs, n_sents,
                                                  n_rel_docs, n_rel_sents))
            f.flush()


def worker(args):
    rc_dir, out_dir, hours, event, ad_dir, log_file = args
    vct_pkl = os.path.join(ad_dir, 'article_vectorizer.pkl')
    clf_pkl = os.path.join(ad_dir, 'article_clf.pkl')
    artcl_detect = ArticleDetector(vct_pkl, clf_pkl, event)
    lgf = open(log_file, 'w') 


    n_hours = len(hours)
    for h, hour  in enumerate(hours, 1):

        n_docs = 0    
        n_sents = 0
        n_rel_docs = 0
        n_rel_sents = 0
    
        #print u'({}/{}) hour: {}'.format(h, n_hours, hour)
        chunk = os.path.join(rc_dir, '{}.sc.gz'.format(hour))
        opath = str(os.path.join(out_dir, '{}.sc.gz'.format(hour)))
        ochunk = sc.Chunk(path=opath, mode='wb')
        try:
            for si_idx, si in enumerate(sc.Chunk(path=chunk)):

                n_docs += 1
                n_sents += len(si.body.sentences[u'serif'])
                sent_idxs = artcl_detect.find_articles(si) 
                n_idxs = len(sent_idxs)
                if n_idxs > 0:
                    n_rel_docs += 1
                    n_rel_sents += n_idxs
                    rel_sents = []
                    for sent_idx in sent_idxs:
                        rel_sents.append(si.body.sentences['serif'][sent_idx])
                    si.body.sentences['article-clf'] = rel_sents 
                        
                    ochunk.add(si) 
            
            ochunk.close()    
            lgf.write('{}\t{}\t{}\t{}\t{}\n'.format(hour, n_docs, n_sents,
                                                    n_rel_docs, n_rel_sents))
            lgf.flush()
        except IOError, e:
            print str(e)  
    lgf.close()

def load_event(event_title, event_xml):
    events = read_events_xml(event_xml)
    for event in events:
        if event_title == event.title:
            return event
    raise ValueError(("No event title matches \"{}\" " \
                      + "in file: {}").format(event_title, event_xml))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--event-file',
                        help=u'Event xml file.',
                        type=unicode, required=True)

    parser.add_argument('-r', '--rel-chunks-dir',
                        help=u'Relevant Chunks dir',
                        type=str, required=True)

    parser.add_argument('-t', '--event-title',
                        help=u'Event title',
                        type=unicode, required=True)

    parser.add_argument('-o', '--output-dir',
                        help=u'Output Chunks dir',
                        type=str, required=True)

    parser.add_argument('-a', '--article-detector',
                        help=u'Article detector dir',
                        type=str, required=True)

    parser.add_argument('-j', '--jobs',
                        help=u'Number of processes to use',
                        type=int, required=False, default=1)

    parser.add_argument('-l', '--log-dir',
                        help=u'Location to write logs.',
                        type=unicode, required=True)


    args = parser.parse_args()
    event_file = args.event_file
    rc_dir = args.rel_chunks_dir
    event_title = args.event_title
    out_dir = args.output_dir
    num_procs = args.jobs
    ad_dir = args.article_detector
    log_dir = args.log_dir

    if num_procs < 1:
        sys.stderr.write(u'Warning: illegal number of jobs, using 1 instead.\n')
        sys.stderr.flush()
        num_procs = 1

    if log_dir != '' and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(event_file) or os.path.isdir(event_file):
        sys.stderr.write((u'--event-file argument {} either does not exist' \
                          + u' or is a directory!\n').format(event_file))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(rc_dir) or not os.path.isdir(rc_dir):
        sys.stderr.write((u'--rel-chunks-dir argument {} either does not' \
                          + u' exist or is not a directory!\n').format(rc_dir))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(ad_dir) or not os.path.isdir(ad_dir):
        sys.stderr.write((u'--article-detector argument {} either does not' \
                          + u' exist or is not a directory!\n').format(ad_dir))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    return event_file, rc_dir, event_title, out_dir, ad_dir, log_dir, num_procs

if __name__ == u'__main__':
    main()
