import argparse
import os
import re
import corenlp
from cuttsum.misc import ProgressBar
import gzip
from multiprocessing import Pool

def main(**kwargs):

    gw_dir = kwargs['gigaword_dir']
    o_dir = os.path.join(os.getenv(u"TREC_DATA"), u'gigaword-lm-input')
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    opath = os.path.join(o_dir, u'gigaword.txt.gz')

    paths = [os.path.join(gw_dir, fname) for fname in os.listdir(gw_dir)]

    max_files = len(paths)

    pool = Pool(5)    

    
    with gzip.open(opath, u'w') as f:

        #pb = ProgressBar(max_files)
        #for path in paths:
        #    print path
        #    result = worker(path)
        for result in pool.imap_unordered(worker, paths):
            #pb.update()
            for sent_str in result:
                s = sent_str.encode(u'utf-8')
                f.write(s)

def worker(path):
 
    print path
    patt = re.compile(r'[_]+')
    sent_strings = []
 
    try:
        for doc in corenlp.open_tar_gz(path):
            for sent in doc:
                buf = []
                for token in sent:
                    if token.ne == u'O':
                        words = patt.split(token.lem)
                        for word in words:
                            if word != u'':
                                buf.append(word.lower())
                    else:
                        buf.append(u'__{}__'.format(token.ne.lower()))
                        
                buf = u' '.join(buf)
                buf = buf + u'\n'
                sent_strings.append(buf)
    except IOError, e:
        print e, path
    return sent_strings

if __name__ == u'__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(u'-d', u'--gigaword-dir', type=unicode,
                        help=u'Location of parsed gigaword xml',
                        required=True)

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
