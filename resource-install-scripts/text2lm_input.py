import os
import argparse
import re
from collections import defaultdict
import gzip
import codecs
import sys
import nltk.data
from nltk.tokenize.punkt import PunktWordTokenizer
import multiprocessing as mp


def main():
    tdir, ofile = parse_cmdline()

    paths = []
    files = [fname for fname in os.listdir(tdir)]

    nfiles = len(files)            
    cpus = mp.cpu_count()
    jobsize = nfiles / cpus

    jobs = []
    tmpfiles = []
    pid = 0
    for i in xrange(0, nfiles, jobsize):
        tmpfile = ofile+'_{}'.format(pid)
        jobs.append((tdir, files[i:i+jobsize], tmpfile))
        tmpfiles.append(tmpfile)
        pid += 1

    pool = mp.Pool(cpus)
    x = pool.map_async(worker, jobs)
    x.get()
    pool.close()
    pool.join()

    from subprocess import call
    if os.path.exists(ofile):
        os.remove(ofile)

    with open(ofile, 'a') as f:
        for tmpfile in tmpfiles:
            call(['cat', tmpfile], stdout=f)
            os.remove(tmpfile)
    
    print "Completed processing files in ", tdir                                    

def worker(args):

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    word_detector = PunktWordTokenizer()
    def split_sentences(txt):
        sents = sent_detector.tokenize(txt.strip(),
                                       realign_boundaries=True)
        for sent in sents:
            sstr = u' '.join(word for word 
                             in word_detector.tokenize(sent))
            tkns = filter(None, sstr.split(u' '))
            if len(tkns) > 0:
                yield u' '.join(tkns)

    tdir, txt_files, ofile = args
 


    with codecs.open(ofile, 'w', 'utf-8') as of:
        for fname in txt_files:
            txt_file = os.path.join(tdir, fname)
            
            
            with codecs.open(txt_file, 'r', 'utf-8') as f:
                text = u' '.join(f.readlines())

                for sent in split_sentences(text):
                    of.write(sent)
                    of.write(u'\n')
                    of.flush()
            print 'Completed', txt_file

def parse_cmdline():

    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--text-dir',
                        help=u'Location of text data directory.',
                        type=unicode, required=True)

    parser.add_argument('-of', '--output-file',
                        help=u'Output location to write text files.',
                        type=unicode, required=True)
  

    args = parser.parse_args()
    tdir = args.text_dir
    ofile = args.output_file

    if not os.path.exists(tdir) or not os.path.isdir(tdir):
        sys.stderr.write((u'--text-dir argument {} either does not exist '
                          + u'or is not a directory!\n').format(tdir))
        sys.stderr.flush()
        sys.exit()

    odir = os.path.dirname(ofile)
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)

    return tdir, ofile

if __name__ == u'__main__':
    main()
