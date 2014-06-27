import argparse
import os
import sys
import streamcorpus as sc
from collections import defaultdict
import gzip
import multiprocessing as mp

def main():

    chunk_dir, counts_dir = parse_args()
    hours = [hour for hour in os.listdir(chunk_dir)]
    nhours = len(hours)
    cpus = mp.cpu_count()
    jobsize = nhours / cpus
    jobs = []
    pid = 1 
    for i in xrange(0, nhours, jobsize):
        jobs.append((chunk_dir, hours[i:i+jobsize], counts_dir, pid))
        pid += 1 
        
    pool = mp.Pool(cpus)
    pool.map_async(worker, jobs)
    pool.close()
    pool.join()    
    
def worker(args):
    msg = sc.StreamItem_v0_2_0
    chunk_dir, hours, counts_dir, pid = args
    nhours = len(hours)
    for i, hour in enumerate(hours, 1):    
        hdir = os.path.join(chunk_dir, hour)
        chunks = [os.path.join(hdir, fname) for fname in os.listdir(hdir)]
        ofile = os.path.join(counts_dir, '{}.txt.gz'.format(hour))
        print '{}) {} -- {}/{}'.format(pid, hour, i, nhours)
        print '--> {}'.format(ofile)
        
        counts = defaultdict(int)    
        doc_counts = defaultdict(int)
        for chunk in chunks:
            for si in sc.Chunk(path=chunk, message=msg):
                doc_words = set()
                for sentence in si.body.sentences['lingpipe']:
                    for token in sentence.tokens:
                        t = token.token.decode('utf-8')
                        counts[t] += 1
                        doc_words.add(t)
                for word in doc_words:
                    doc_counts[word] += 1        
        with gzip.open(ofile, 'wb') as f:    
            for token, count in counts.iteritems():
                doc_count = doc_counts[token]
                f.write(token.encode('utf-8'))
                f.write('\t') 
                f.write(str(count)) 
                f.write('\t')
                f.write(str(doc_count))             
                f.write('\n') 


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--chunk-dir',
                        help=u'Chunk directory.',
                        type=unicode, required=True)

    parser.add_argument('-d', '--counts-dir',
                        help=u'Write counts to this location',
                        type=unicode, required=True)

    args = parser.parse_args()
    chunk_dir = args.chunk_dir
    counts_dir = args.counts_dir

    if not os.path.exists(chunk_dir) or not os.path.isdir(chunk_dir):
        sys.stderr.write((u'--chunk-dir argument {} either does not exist' \
                          + u' or is not a directory!\n').format(chunk_dir))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(counts_dir):
        os.makedirs(counts_dir)


    return chunk_dir, counts_dir
 
if __name__ == '__main__':
    main()

