import os
import errno
import re
from datetime import datetime, timedelta
import urllib2
import socket
import gzip
import sys
from multiprocessing import Pool
from collections import defaultdict
import streamcorpus as sc
import marisa_trie
import numpy as np
import signal


def relevant_chunks(event):
    data_dir = os.getenv(u'TREC_DATA', u'.')
    event_ddir = os.path.join(data_dir, u'relevant-chunks', event.fs_name())
    chunks = [os.path.join(event_ddir, chunk)
              for chunk in os.listdir(event_ddir)]
    chunks.sort()
    return chunks

def relevant_chunk_size(event):
    return sum(
        os.path.getsize(chunk) / 1e6 for chunk in relevant_chunks(event))


class Resource(object):

    deps_met_ = {}
    
    def check_coverage(self):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def dependencies(self):
        raise NotImplementedError

    def __str__(self):
        return unicode(self).encode(u'utf-8')

    @classmethod
    def getdependency(self, fun):
        def wrapper(self, event, corpus, **kwargs):            
            is_met = \
                self.deps_met_.get((event.fs_name(), corpus.fs_name()), False)
            if is_met is False:
                for dc in self.dependencies():
                    dep = dc()
                    coverage_per = dep.check_coverage(event, corpus, **kwargs)
                    if coverage_per != 1.:
                        sys.stdout.write(
                            u"{}: Incomplete coverage (%{:0.2f}), "
                            u"retrieving...\n"
                                .format(dep, coverage_per * 100.))
                        if dep.get(event, corpus, **kwargs) is True:
                            sys.stdout.write('OK!\n')
                            sys.stdout.flush()
                        else:
                            sys.stdout.write('FAILED!\n')
                            sys.flush()
                            sys.stderr.write(
                                'Failed to retrieve necessary resource:\n')
                            sys.stderr.write('\t{} for {} / {}'.format(
                                dep, event.query_id, corpus.fs_name())) 
                            sys.stderr.flush()
                            sys.exit(1)
            self.deps_met_[(event.fs_name(), corpus.fs_name())] = True
            return fun(self, event, corpus, **kwargs)
        return wrapper

    @classmethod
    def getsuperdependency(self, fun):
        def wrapper(self, event, corpus, **kwargs):
            if corpus.is_subset() is True:
                corpus = corpus.get_superset()
            return fun(self, event, corpus, **kwargs)
        return wrapper

    def do_work(self, worker, jobs, n_procs, progress_bar):
        if sys.stdout.isatty() is False:
            progress_bar = False

        n_jobs = len(jobs)
        if n_procs == 1:
            for j, job in enumerate(jobs, 1):
                if progress_bar is True:
                    sys.stdout.write('{}: %{:0.3f} complete...\r'.format(
                        self, j * 100. / n_jobs))
                    sys.stdout.flush()
                worker(job)
                
        else:
            try:
                pool = Pool(
                    n_procs, 
                    lambda: signal.signal(
                        signal.SIGINT, signal.SIG_IGN))
                for r, result in enumerate(
                    pool.imap_unordered(worker, jobs), 1):
                    
                    if progress_bar is True:
                        sys.stdout.write('{}: %{:0.3f} complete...\r'.format(
                            self, r * 100. / n_jobs))
                        sys.stdout.flush()

            except KeyboardInterrupt:
                print "\rCaught KeyboardInterrupt, terminating workers..."
                pool.close()
                sys.exit()

        if progress_bar is True:
            sys.stdout.write(' ' * 79 + '\r')
            sys.stdout.flush()
        return True



class KBAChunkResource(Resource):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'kba-chunks')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    @Resource.getdependency
    def check_coverage(self, event, corpus, domains=None, **kwargs):
        n_chunks = 0
        n_covered = 0
        for domain, count, path, url in self.get_chunk_info_paths_urls(
            event, corpus):
            if domains is None or domain in domains:
                n_chunks += 1
                if os.path.exists(path):
                    n_covered += 1    

        if n_chunks == 0:
            return 0
        else:
            return n_covered / float(n_chunks)

    @Resource.getdependency
    def get(self, event, corpus, domains=None, **kwargs):
        jobs = []
        overwrite = kwargs.get(u'overwrite', False)
        for domain, count, path, url in self.get_chunk_info_paths_urls(
            event, corpus):
            if domains is None or domain in domains:
                if overwrite is True or not os.path.exists(path):
                    jobs.append((path, url))

        progress_bar = kwargs.get(u'progress_bar', False)
        n_procs = kwargs.get(u'n_procs', 1)
        n_jobs = len(jobs)
        if n_procs == 1:
            for j, job in enumerate(jobs, 1):
                if progress_bar is True:
                    sys.stdout.write('{}: %{:0.3f} complete...\r'.format(
                        self, j * 100. / n_jobs))
                    sys.stdout.flush()
                _kbachunk_resource_worker(job)
                
        else:
            pool = Pool(n_procs)
            for r, result in enumerate(pool.imap_unordered(
                _kbachunk_resource_worker, jobs), 1):
                if progress_bar is True:
                    sys.stdout.write('{}: %{:0.3f} complete...\r'.format(
                        self, r * 100. / n_jobs))
                    sys.stdout.flush()

        if progress_bar is True:
            sys.stdout.write(' ' * 79 + '\r')
            sys.stdout.flush()
        return True

    def dependencies(self):
        return tuple([UrlListResource])

    def get_chunks_for_hour(self, hour, corpus):
        hour_dir = os.path.join(
            self.dir_, corpus.fs_name(), hour.strftime(u'%Y-%m-%d-%H'))
        if os.path.exists(hour_dir):
            chunks = [os.path.join(hour_dir, fname) for fname
                      in os.listdir(hour_dir)]
            return chunks
        else:
            return []

    def get_chunk_info_paths_urls(self, event, corpus):
        data_dir = os.path.join(self.dir_, corpus.fs_name())
        data = []
        for url_path in UrlListResource().get_event_url_paths(event, corpus):
            with gzip.open(url_path, u'r') as f:
                for line in f:
                    if line == '\n':
                        continue
                    if "index" in line:
                        continue
                    chunk = line.strip()
                    path = os.path.join(data_dir, chunk)
                    url = '{}{}'.format(corpus.aws_url_, chunk)
                    domain, count = os.path.split(path)[-1].split('-')[0:2]
                    data.append((domain, int(count), path, url))
        return data    

    def __unicode__(self):
        return u"cuttsum.data.KBAChunkResource"
   
def _kbachunk_resource_worker(args):
    path, url = args

    parent = os.path.dirname(path)
    if not os.path.exists(parent):
        try:
            os.makedirs(parent)
        except OSError as e: 
            if e.errno == errno.EEXIST and os.path.isdir(parent):
                pass

    try:
        write_url_data(url, path, timeout=60)
    except PageNotFoundException, e:
        print e
  
class UrlListResource(Resource):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'url-lists')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def __unicode__(self):
        return u"cuttsum.data.UrlListResource"

    def check_coverage(self, event, corpus, **kwargs):
        n_hours = 0
        n_covered = 0
        for hour in self.list_event_hours(event):
            path = os.path.join(self.dir_, corpus.fs_name(),
                u'{}.txt.gz'.format(hour.strftime(u'%Y-%m-%d-%H')))
            n_hours += 1
            if os.path.exists(path) and os.path.isfile(path):
                n_covered += 1
        return float(n_covered) / n_hours

    def get(self, event, corpus, 
            overwrite=False, n_procs=1, progress_bar=False, **kwargs):
        data_dir = os.path.join(self.dir_, corpus.fs_name())
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        hours = self.list_event_hours(event)
        n_hours = len(hours)

        jobs = [(hour, data_dir, corpus, overwrite) for hour in hours]

        if n_procs == 1:
            for j, job in enumerate(jobs, 1):
                if progress_bar is True:
                    sys.stdout.write('{}: %{:0.3f} complete...\r'.format(
                        self, j * 100. / n_hours))
                    sys.stdout.flush()
                _urllist_resource_worker(job)            

        else:
            pool = Pool(n_procs)
            for r, result in enumerate(
                pool.imap_unordered(_urllist_resource_worker, jobs), 1):
                
                if progress_bar is True:
                    sys.stdout.write('{}: %{:0.3f} complete...\r'.format(
                        self, r * 100. / n_hours))
                    sys.stdout.flush()

        if progress_bar is True:
            sys.stdout.write(' ' * 79 + '\r')
            sys.stdout.flush()
        return True

    def list_event_hours(self, event):
        start_dt = event.start.replace(minute=0, second=0)
        end_dt = event.end.replace(minute=0, second=0)
        current_dt = start_dt
        hours = []
        while current_dt <= end_dt:
            hours.append(current_dt)
            current_dt += timedelta(hours=1)
        return hours

    def get_event_url_paths(self, event, corpus):
        data_dir = os.path.join(self.dir_, corpus.fs_name())
        paths = []
        for hour in self.list_event_hours(event):
            path = os.path.join(data_dir,
                u'{}.txt.gz'.format(hour.strftime(u'%Y-%m-%d-%H')))
            if os.path.exists(path):
                paths.append(path)
        return paths

    def dependencies(self):
        return ()

def _urllist_resource_worker(args):
    hour, data_dir, corpus, overwrite = args
    hour_str = hour.strftime(u'%Y-%m-%d-%H')
    path = os.path.join(data_dir,
        u'{}.txt.gz'.format(hour_str))
    if overwrite is False and os.path.exists(path):
        return
    url = u'{}{}/index.html'.format(corpus.aws_url_, hour_str)

    try:
        html = get_html_string(url, timeout=60) 
        with gzip.open(path, u'w') as f:
            for link in re.findall(r'<a href="(.*?)">', html):
                if "index.html" in link:
                    continue
                f.write('{}/{}\n'.format(hour_str, link))

    except PageNotFoundException, e:
        print e
        with gzip.open(path, u'w') as f:
            f.write('\n')

def write_url_data(url, path, timeout=None, retries=3):
    n_retries = 0
    while 1:
        try:
            with open(path, u'wb') as f:
                for data in download_url(url, timeout):
                    f.write(data)
            break
        except TimeoutException, e:
            if n_retries > retries:
                print e
                import sys
                sys.exit()
            else:
                n_retries += 1

def get_html_string(url, timeout=None, retries=3):
    n_retries = 0
    while 1:
        try:
            buffer = ''
            for data in download_url(url, timeout):
                buffer += data
            return buffer
        except TimeoutException, e:
            if n_retries > retries:
                print e
                import sys
                sys.exit()
            else:
                n_retries += 1

class TimeoutException(Exception):
    pass

class PageNotFoundException(Exception):
    pass

def download_url(url, timeout):        
    try:
        u = urllib2.urlopen(url, timeout=timeout)
        block_size = 8192
        while 1:
            buffer = u.read(block_size)
            if not buffer:
                break
            yield buffer
    except urllib2.HTTPError, e:
        if e.code == 404:
            raise PageNotFoundException(
                "404 Error: url {} does not exist".format(url))
        else:
            raise e
        #print 'The server couldn\'t fulfill the request.'
        #print 'Error code: ', e.code          
    except urllib2.URLError, e:
    # For Python 2.6
        if isinstance(e.reason, socket.timeout):
            raise TimeoutException("There was an error: %r" % e)
        else:
        # reraise the original error
            raise
    except socket.timeout, e:
    # For Python 2.7
        raise TimeoutException("There was an error: %r" % e)
 
class IdfResource(Resource):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'idf')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)
        

    def __unicode__(self):
        return u"cuttsum.data.IdfResource"

    @Resource.getsuperdependency
    def get_idf_path(self, hour, corpus):
        return os.path.join(
            self.dir_, corpus.fs_name(), u'{}.idf.marisa.gz'.format(
                hour.strftime(u'%Y-%m-%d-%H')))

    @Resource.getsuperdependency
    @Resource.getdependency
    def check_coverage(self, event, corpus, **kwargs):
        domains = kwargs.get(u'domains', None)
        data_dir = os.path.join(self.dir_, corpus.fs_name())
        chunks = KBAChunkResource()

        n_hours = 0
        n_covered = 0
        for hour in event.list_event_hours():
            tfidf_path = os.path.join(data_dir, u"{}.idf.marisa.gz".format(
                hour.strftime(u"%Y-%m-%d-%H")))
            n_hours += 1
            if os.path.exists(tfidf_path):
                n_covered += 1
            
        if n_hours == 0:
            return 0
        else:
            return n_covered / float(n_hours)

    @Resource.getsuperdependency
    @Resource.getdependency
    def get(self, event, corpus, **kwargs):
        data_dir = os.path.join(self.dir_, corpus.fs_name())
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        domains = kwargs.get(u'domains', None)
        chunks = KBAChunkResource()
        
        hour2chunks = defaultdict(list)
        for d, c, path, url in chunks.get_chunk_info_paths_urls(
            event, corpus):
            if domains is None or d in domains:
                hour = os.path.basename(os.path.dirname(path))
                hour2chunks[hour].append(path)

        overwrite = kwargs.get(u'overwrite', False)
        jobs = []
        for hour, paths in hour2chunks.iteritems():
            mpath = os.path.join(data_dir, "{}.idf.marisa.gz".format(hour))
            if overwrite is True or not os.path.exists(mpath):
                jobs.append((mpath, paths, corpus))

        n_procs = kwargs.get(u'n_procs', 1)
        progress_bar = kwargs.get(u'progress_bar', False)
        self.do_work(_idf_resource_worker, jobs, n_procs, progress_bar)

    def dependencies(self):
        return tuple([KBAChunkResource])

def _idf_resource_worker(args):        

    mpath, paths, corpus = args
    n_docs = 0
    counts = defaultdict(int)
    for path in paths:
        for si in sc.Chunk(path=path, mode='rb', 
            message=corpus.sc_msg()):

            sentences = corpus.get_sentences(si)
            if len(sentences) == 0:
                continue 

            n_docs += 1 
            unique_words = set()
            for sentence in sentences:
                for token in sentence.tokens:
                    unique_words.add(
                        token.token.decode(u'utf-8').lower())
            for word in unique_words:
                counts[word] += 1

    n_docs = float(n_docs)
    words = counts.keys() 

    idfs = [tuple([np.log(n_docs / value)])
              for value in counts.values()] 

    trie = marisa_trie.RecordTrie("<d", zip(words, idfs))
    with gzip.open(mpath, u'wb') as f:
        trie.write(f)

class LMResource(Resource):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'lm')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)
        
        self.domain_lm_ = {
            u'accident': u'accidents_3.arpa.gz', 
            u'shooting': u'terrshoot_3.arpa.gz',
            u'storm': u'weather_3.arpa.gz',
            u'earthquake': u'earthquakes_3.arpa.gz',
            u'bombing': u'terrshoot_3.arpa.gz',
            u'riot': u'social_unrest_3.arpa.gz',
            u'protest': u'social_unrest_3.arpa.gz',
            u'hostage': u'terrshoot_3.arpa.gz',
            u'impact event': u'earthquakes_3.arpa.gz',
        }  
        
        self.gigaword_lm_ = u'gigaword_5.arpa.gz'

        self.ports_ = {
            u'gigaword_5.arpa.gz':      9999,
            u'accidents_3.arpa.gz':     9998,
            u'terrshoot_3.arpa.gz':     9997,
            u'weather_3.arpa.gz':       9996,
            u'earthquakes_3.arpa.gz':   9995,
            u'social_unrest_3.arpa.gz': 9994,
        }

    def get_gigaword_port(self):
        return self.ports_[u'gigaword_5.arpa.gz']

    def get_domain_port(self, event):
        lm = self.domain_lm_[event.type]
        return self.ports_[lm]

    def domain_path(self, event):
        return os.path.join(self.dir_, self.domain_lm_[event.type])

    def gigaword_path(self):
        return os.path.join(self.dir_, self.gigaword_lm_)

    def __unicode__(self):
        return u"cuttsum.data.LMResource"

    @Resource.getdependency
    def check_coverage(self, event, corpus, **kwargs):
        coverage = 0
        if os.path.exists(self.domain_path(event)):
            coverage += .5 
        if os.path.exists(self.gigaword_path()):
            coverage += .5 
        
        return coverage

    @Resource.getdependency
    def get(self, event, corpus, **kwargs):
        raise NotImplementedError(
            "I don't know how to make a language model from scratch yet")

    def dependencies(self):
        return tuple([])


