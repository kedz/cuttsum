import os
import sys
import argparse
from cuttsum.event import read_events_xml
from cuttsum.util import gen_dates
import streamcorpus as sc
import re
from datetime import datetime

def main():
    event_file, event_title, query, rel_dir, c_dir, cpus, log_dir = parse_args()
    event = load_event(event_title, event_file)
    hours = [hour for hour in gen_dates(event.start, event.end)]
    nhours = len(hours)
    hours_per_cpu = nhours / cpus 
    jobs = []
    pid = 0
    now = datetime.now().strftime('%Y-%m-%d-%H')
    print now
    for i in xrange(0, nhours, hours_per_cpu):
        jobs.append((event, event_title, query, hours[i:i+hours_per_cpu],
                     rel_dir, c_dir, 
                     os.path.join(log_dir, 
                                  '.rel_log_{}_{}_log.txt'.format(now, pid))))
        pid += 1 

    print 'Query Match Relevance Extractor'
    print '==============================='
    print

    print 'SYSTEM INFO'
    print '==========='
    print 'KBA location:', c_dir
    print 'Relevant chunks:', rel_dir
    print 'n-threads:', cpus 
    print

    print 'EVENT INFO'
    print '=========='
    print 'Event Title:', event.title
    print 'Event Type:', event.type
    print 'Date Range: {} -- {}'.format(event.start, event.end)
    print 'Spanning', nhours, 'hours.'
    print 'Query:', query
    print
    
    if cpus == 1:
        for job in jobs:
            worker(job)
    else:
        import multiprocessing as mp
        pool = mp.Pool(cpus)
        pool.map_async(worker, jobs)
        pool.close()
        pool.join()

    log_data = []
    for fname in os.listdir(log_dir):
        if re.search(r'\.rel_log_{}'.format(now), fname):
            with open(os.path.join(log_dir, fname), 'r') as f:
                for line in f:
                    items = line.strip().split('\t')
                    if len(items) != 3:
                        continue
                    log_data.append(items)
            os.remove(os.path.join(log_dir, fname))

    log_data.sort(key=lambda x: x[0])
    log_fname = os.path.join(log_dir, 'rel_log_{}.txt'.format(now))
    with open(log_fname, 'w') as f:
        for date, tot_rel, tot_docs in log_data:
            f.write('{}\t{}\t{}\n'.format(date, tot_rel, tot_docs))
            f.flush()


def worker(args):
    event, event_title, query, hours, rel_dir, c_dir, log_file = args
    msg = sc.StreamItem_v0_2_0
    
    with open(log_file, 'w') as lgf:
        for hour in hours:

            total_docs = 0
            total_rel = 0

            hdir = os.path.join(c_dir, hour)
            opath = str(os.path.join(rel_dir, '{}.sc.gz'.format(hour)))
            if not os.path.exists(hdir):
                continue
            if os.path.exists(opath):
                os.remove(opath)
            #print hdir        

            ochunk = sc.Chunk(path=opath, mode='wb')
            for cname in os.listdir(hdir):
                path = str(os.path.join(hdir, cname))
                for si in sc.Chunk(path=path):
                    total_docs += 1            
                    if si.body.clean_visible is None:
                        continue
                    elif re.search(query, si.body.clean_visible, re.I):
                        total_rel += 1
                        ochunk.add(si)
            ochunk.close()
            lgf.write('{}\t{}\t{}\n'.format(hour, total_rel, total_docs))
            lgf.flush()

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

    parser.add_argument('-r', '--relevant-dir',
                        help=u'Location to write relevant chunks.',
                        type=str, required=True)

    parser.add_argument('-c', '--chunks-dir',
                        help=u'Trec KBA location.',
                        type=str, required=True)

    parser.add_argument('-t', '--event-title',
                        help=u'Event title',
                        type=unicode, required=True)

    parser.add_argument('-q', '--query-string',
                        help=u'Relevance query string.',
                        type=str, required=True)

    parser.add_argument('-l', '--log-dir',
                        help=u'Location to write logs.',
                        type=str, required=True)


    parser.add_argument('-nt', '--n-threads',
                        help=u'Number of threads.',
                        type=int, required=False, default=1)


    args = parser.parse_args()       
    event_file = args.event_file
    rel_dir = args.relevant_dir 
    c_dir = args.chunks_dir
    event_title = args.event_title
    query = args.query_string
    log_dir = args.log_dir
    n_threads = args.n_threads

    if not os.path.exists(event_file) or os.path.isdir(event_file):
        sys.stderr.write(((u'--event-file argument {} either does not exist' \
                           + u' or is a directory!\n').format(event_file)))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(c_dir) or not os.path.isdir(c_dir):
        sys.stderr.write(((u'--chunks-dir argument {} either does not exist' \
                           + u' or is not a directory!\n').format(c_dir)))
        sys.stderr.flush()
        sys.exit()
 
    if rel_dir != '' and not os.path.exists(rel_dir):
        os.makedirs(rel_dir)

    if log_dir != '' and not os.path.exists(log_dir):
        os.makedirs(log_dir)
       

    return event_file, event_title, query, rel_dir, c_dir, n_threads, log_dir

if __name__ == '__main__':
    main()
