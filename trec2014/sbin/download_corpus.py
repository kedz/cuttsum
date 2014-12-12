from cuttsum import kba
from cuttsum import events
import os
from multiprocessing import Pool
import subprocess
import re
import time

domains = set([u'news', u'MAINSTREAM_NEWS'])

def validate_chunk_checksum(path):
    fname = os.path.split(path)[-1]
    check_sums = fname.split('.')[0].split('-')[2:]
    #m = re.search(r'\w+-\d+-(.*?)-(.*?)\.sc', fname)
    #if m is None: 
    
    #    return False

    #else:
    check_sum = check_sums[-1]
        #check_sum = m.groups()[1]
    sp_out = subprocess.check_output(
        "gpg -d {} | xzcat | md5sum".format(path), shell=True)
    actual_check_sum = sp_out.split(' ')[0]
    return check_sum == actual_check_sum

def worker(args):
    path, dest, corpus = args
    date, chunk = path.split('/')
    tgt = os.path.join(dest, date, chunk)

    if os.path.exists(tgt):
        valid = validate_chunk_checksum(tgt)
    
        if valid:
            return path
        else:
            print path, "is not correct downloading again."
            os.remove(tgt)
    
    corpus.download_path(path, dest)
    time.sleep(.5)
    return path

def main(n_procs, trec_dir, download_2013=False):
    pool = Pool(n_procs)

    if download_2013 is True:
        corpus = kba.EnglishAndUnknown2013()
        event_dir = u'2013_english_and_unknown'
        dest = os.path.join(trec_dir, event_dir)
        if not os.path.exists(dest):
            os.makedirs(dest)

        for event in events.get_2013_events():
            print event 
            jobs = [(path, dest, corpus) for ts, dmn, ct, path
                    in corpus.paths(event.start, event.end, domains)]
            for result in pool.imap(worker, jobs):
                print result

    corpus = kba.FilteredTS2014()
    event_dir = u'2014_filtered_ts'
    dest = os.path.join(trec_dir, event_dir)
    if not os.path.exists(dest):
        os.makedirs(dest)

    for event in events.get_2014_events():
        print event 
        jobs = [(path, dest, corpus) for ts, dmn, ct, path
                in corpus.paths(event.start, event.end, domains)]
        
        pool = Pool(n_procs)
        for result in pool.imap(worker, jobs):
            print result

    corpus = kba.SerifOnly2014()
    event_dir = u'2014_serif_only'
    dest = os.path.join(trec_dir, event_dir)
    if not os.path.exists(dest):
        os.makedirs(dest)

    for event in events.get_2014_events():
        print event 
        jobs = [(path, dest, corpus) for ts, dmn, ct, path
                in corpus.paths(event.start, event.end, domains)]
                
        pool = Pool(n_procs)
        for result in pool.imap(worker, jobs):
            print result


if __name__ == u'__main__':
    n_procs = 10
    trec_dir = os.getenv(u"TREC_DATA", None)
    if trec_dir is None:
        trec_dir = os.path.join(os.getenv(u'HOME', u'.'), u'trec-data')
        print u'TREC_DATA env. var not set. Using {} to store data.'.format(
            trec_dir)
    
    if not os.path.exists(trec_dir):
        os.makedirs(trec_dir)
    main(n_procs, trec_dir)
