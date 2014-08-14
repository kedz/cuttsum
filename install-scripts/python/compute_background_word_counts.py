import argparse
import os
import sys
import streamcorpus as sc
from collections import defaultdict
import gzip
import multiprocessing as mp
import math

def main():

    chunk_dir, wcounts_dir, dcounts_dir, log_dir, num_procs = parse_args()
    hours = [hour for hour in os.listdir(chunk_dir)]
    nhours = len(hours)
    jobsize = int(math.ceil(nhours / num_procs))
    jobs = []
    pid = 1 
    for i in xrange(0, nhours, jobsize):
        jobs.append((chunk_dir, hours[i:i+jobsize], wcounts_dir, dcounts_dir,
                     os.path.join(log_dir, u'word_freq_{}.log'.format(pid))))
        pid += 1 
        
    pool = mp.Pool(jobsize)
    x = pool.map_async(worker, jobs)
    x.get()
    pool.close()
    pool.join()    
    
def worker(args):
    chunk_dir, hours, wcounts_dir, dcounts_dir, log_file = args
    nhours = len(hours)

    with open(log_file, 'w') as lf:
        for i, hour in enumerate(hours, 1):    
            hdir = os.path.join(chunk_dir, hour)
            chunks = [os.path.join(hdir, fname) for fname in os.listdir(hdir)]

            wcfile = os.path.join(wcounts_dir, '{}.txt.gz'.format(hour))
            dcfile = os.path.join(dcounts_dir, '{}.txt'.format(hour))

            lf.write('Counting hour {} ({}/{})\n'.format(hour, i, nhours))
            lf.flush()
            
            counts = defaultdict(int)    
            doc_counts = defaultdict(int)
            num_docs = 0
            for chunk in chunks:
                for si in sc.Chunk(path=chunk):
                    
                    if 'serif' not in si.body.sentences:
                        continue
                    
                    num_docs += 1
                    
                    doc_words = set()
                    for sentence in si.body.sentences['serif']:
                        for token in sentence.tokens:
                            t = token.token.decode('utf-8')
                            counts[t] += 1
                            doc_words.add(t)
                    
                    for word in doc_words:
                        doc_counts[word] += 1
                        
            if len(counts) == 0:
                lf.write(u'Warning: {} contained no words.\n'.format(chunk))
                lf.flush()
                continue

            # Write doc counts for this hour
            with open(dcfile, 'w') as f:
                f.write(str(num_docs))
                f.flush()

            # Write word counts for this hour
            with gzip.open(wcfile, 'wb') as f:    
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

    parser.add_argument('-w', '--word-counts-dir',
                        help=u'Write word counts to this location',
                        type=unicode, required=True)

    parser.add_argument('-d', '--doc-counts-dir',
                        help=u'Write doc counts to this location',
                        type=unicode, required=True)
   
    parser.add_argument('-n', '--n-procs',
                        help=u'Number of processes to use',
                        type=int, required=False, default=1)
 
    parser.add_argument('-l', '--log-dir',
                        help=u'Log location',
                        type=unicode, required=True)
  

    args = parser.parse_args()
    chunk_dir = args.chunk_dir
    wcounts_dir = args.word_counts_dir
    dcounts_dir = args.doc_counts_dir
    num_procs = args.n_procs
    log_dir = args.log_dir
    

    if not os.path.exists(chunk_dir) or not os.path.isdir(chunk_dir):
        sys.stderr.write((u'--chunk-dir argument {} either does not exist' \
                          + u' or is not a directory!\n').format(chunk_dir))
        sys.stderr.flush()
        sys.exit()

    if not os.path.exists(wcounts_dir):
        os.makedirs(wcounts_dir)

    if not os.path.exists(dcounts_dir):
        os.makedirs(dcounts_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if num_procs < 1:
        sys.stderr.write(u'Warning: --n-procs must be > 0, using 1 instead\n.')
        sys.stderr.flush()
        num_procs = 1

    return chunk_dir, wcounts_dir, dcounts_dir, log_dir, num_procs
 
if __name__ == '__main__':
    main()

