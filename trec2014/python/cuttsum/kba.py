import gzip
from datetime import datetime
import urllib2
from pkg_resources import resource_filename
import os


class _KBAStreamCorpus(object):

    def paths(self, start_dt=None, end_dt=None, domain_filter=None):

        paths_fname = resource_filename(u'cuttsum', self.paths_txt_)
        with gzip.open(paths_fname, u'r') as f:
            for line in f:
                path = line.strip()
                dt_string, doc_string = path.split('/')
                timestamp = datetime.strptime(dt_string, '%Y-%m-%d-%H')
                domain, count = doc_string.split('-')[0:2]
                if start_dt is not None and start_dt > timestamp:
                    continue
                if end_dt is not None and end_dt < timestamp:
                    continue
                if domain_filter is not None and domain not in domain_filter:
                    continue
                yield timestamp, domain, int(count), path

    def download_path(self, path, dest_dir):
        url = '{}{}'.format(self.aws_url_, path)
        dest = os.path.join(dest_dir, path)
        dest_parent = os.path.dirname(dest)
        if not os.path.isdir(dest_parent):
            try:
                os.makedirs(dest_parent)
            except OSError, e:
                if e.errno == errno.EEXIST and os.path.isdir(folder_location):
                    # File exists, and it's a directory,
                    # another process beat us to creating this dir, that's OK.
                    pass
                else:
                    # Our target dir exists as a file, or different error,
                    # reraise the error!
                    raise
        print url
        u = urllib2.urlopen(url)
        f = open(dest, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        #print "Downloading: %s Bytes: %s" % (path, file_size)

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            #status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            #status = status + chr(8)*(len(status)+1)
            #print status,

        f.close() 

    def year(self):
        return self.year_
        
    def version(self):
        return self.ver_


class EnglishAndUnknown2013(_KBAStreamCorpus):
    """This KBA StreamCorpus contains only the English and unknown langauge
documents that werer processed by LingPipe."""

    def __init__(self):
        self.year_ = 2013
        self.ver_ = u'0.2.0'
        self.aws_url_ = u'http://s3.amazonaws.com/' + \
            u'aws-publicdatasets/trec/kba/' + \
            u'kba-streamcorpus-2013-v0_2_0-english-and-unknown-language/'
        self.paths_txt_ = \
            os.path.join(u'2013-data',
                u'kba-streamcorpus-2013-v0_2_0' + \
                U'-english-and-unknown-language.s3-paths.txt.gz')


class SerifOnly2014(_KBAStreamCorpus):
    """This KBA StreamCorpus contains only the English and unkown language
documents that were processed by the BBN Serif tool."""

    def __init__(self):
        self.year_ = 2014
        self.ver_ = u'0.3.0'
        self.aws_url_ = u'http://s3.amazonaws.com/' + \
            u'aws-publicdatasets/trec/kba/' + \
            u'kba-streamcorpus-2014-v0_3_0-serif-only/'
        self.paths_txt_ = \
            os.path.join(u'2014-data',
                u'kba-streamcorpus-2014-v0_3_0-serif-only.s3-paths.txt.gz')

class FilteredTS2014(_KBAStreamCorpus):
    """This KBA StreamCorpus contains only the English and unkown language
documents that were processed by the BBN Serif tool and was filtered with
the TREC TS 2014 event queries."""

    def __init__(self):
        self.year_ = 2014
        self.ver_ = u'0.3.0'
        self.aws_url_ = u'http://s3.amazonaws.com/' + \
            u'aws-publicdatasets/trec/ts/' + \
            u'streamcorpus-2014-v0_3_0-ts-filtered/'
        self.paths_txt_ = \
            os.path.join(u'2014-data',
                u'streamcorpus-2014-v0_3_0-ts-filtered.s3-paths.txt.gz')
