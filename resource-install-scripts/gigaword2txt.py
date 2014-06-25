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
    gdir, ofile, regex = parse_cmdline()

    paths = []

    for path, dirs, fnames in os.walk(gdir):
        for fname in fnames:
            if re.search(regex, fname):
                paths.append(os.path.join(path, fname))

    npaths = len(paths)            
    cpus = mp.cpu_count()
    jobsize = npaths / cpus

    chunks = []
    for i in xrange(0, npaths, jobsize):
        chunks.append(paths[i:i+jobsize])

    print 'Total Jobs:', len(chunks)
    for i, chunk in enumerate(chunks):
        print 'CPU', i+1, '--', len(chunk), 'file(s)'

    tmpfiles = []
    for i in range(cpus):
        tmpfiles.append(ofile+'_{}'.format(i))
    jobs = zip(chunks, tmpfiles)
    pool = mp.Pool(cpus)
    pool.map_async(worker, jobs)

    pool.close()
    pool.join()

    from subprocess import call
    if os.path.exists(ofile):
        os.remove(ofile)

    with open(ofile, 'a') as f:
        for tmpfile in tmpfiles:
            call(['cat', tmpfile], stdout=f)
            os.remove(tmpfile)
    
                                    

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
            yield u' '.join(tkns)

    chunk, ofile = args
 
    with codecs.open(ofile, 'w', 'utf-8') as of:
        for path in chunk:    
            with gzip.open(path) as f:
                for txt in find_story(f):
                    if txt is None:
                        continue
                    else:
                        for sent in split_sentences(txt):
                            of.write(sent)
                            of.write(u'\n')
                            of.flush()
            print 'Completed', path               

def find_story(f):
    for line in f:
        line = line.decode('utf-8')
        if u'type="story"' in line:
            while not line.startswith(u'<TEXT'):
                line = f.readline().decode('utf-8')
            yield read_text(f)

def read_text(f):
    txt_lines = []
    txt_line = f.readline().decode('utf-8')

    while txt_line != u'</TEXT>\n':
        txt_lines.append(txt_line.strip())
        txt_line = f.readline().decode('utf-8')

    txt = u' '.join(txt_lines)
    txt = txt.replace(u'<P>', u'').replace(u'</P>', u'\n')
    return txt

def parse_cmdline():

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gigaword-dir',
                        help=u'Location of gigaword data directory.',
                        type=unicode, required=True)

    parser.add_argument('-of', '--output-file',
                        help=u'Output location to write text files.',
                        type=unicode, required=True)
  
    parser.add_argument('-r', '--regex', type=str, required=True,
                         help=u'Regex for selecting files.')   

    args = parser.parse_args()
    gdir = args.gigaword_dir
    ofile = args.output_file
    regex = args.regex


    if not os.path.exists(gdir) or not os.path.isdir(gdir):
        sys.stderr.write((u'Gigaword dir argument {} either does not exist '
                          + u'or is not a directory!\n').format(gdir))
        sys.stderr.flush()
        sys.exit()

    odir = os.path.dirname(ofile)
    if odir != '' and not os.path.exists(odir):
        os.makedirs(odir)

    return gdir, ofile, regex

if __name__ == u'__main__':
    main()

