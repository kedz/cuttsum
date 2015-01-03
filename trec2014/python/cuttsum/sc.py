from .data import Resource
import os
import gzip
import signal
import urllib3
from gnupg import GPG
import Queue
import re
import errno
import streamcorpus as sc
import numpy as np
import marisa_trie
from collections import defaultdict

class UrlListResource(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'url-lists')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def __unicode__(self):
        return u"cuttsum.sc.UrlListResource"

    def dependencies(self):
        return tuple([])

    def get_event_url_paths(self, event, corpus, preroll=0,
        must_exist=False):
        paths = []
        for hour in event.list_event_hours(preroll=preroll):
            path = os.path.join(self.dir_, corpus.fs_name(), 
                u'{}.txt.gz'.format(hour.strftime(u'%Y-%m-%d-%H')))
            if must_exist is False or os.path.exists(path):
                paths.append(path)
        return paths

    def check_coverage(self, event, corpus, preroll=0, **kwargs):
        n_hours = 0
        n_covered = 0

        for path in self.get_event_url_paths(event, corpus, preroll=preroll):
            n_hours += 1
            if os.path.exists(path) and os.path.isfile(path):
                n_covered += 1
        return float(n_covered) / n_hours

    def get(self, event, corpus, overwrite=False, n_procs=1, 
            progress_bar=True, preroll=0, **kwargs):

        print "Getting:", corpus
        data_dir = os.path.join(self.dir_, corpus.fs_name())
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        hours = event.list_event_hours(preroll=preroll)
        n_hours = len(hours)

        jobs = []
        for hour in hours:
            ulist_path = os.path.join(
                data_dir, u'{}.txt.gz'.format(hour.strftime(u'%Y-%m-%d-%H')))
            if overwrite is True or not os.path.exists(ulist_path):
                jobs.append((hour, ulist_path)) 

        self.do_work(urllist_worker_, jobs, n_procs, progress_bar, 
                     corpus=corpus)
        return True


def urllist_worker_(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    http = urllib3.PoolManager(timeout=15.0, retries=3)
    corpus = kwargs.get(u'corpus')

    while not job_queue.empty():
        try:
            hour, ulist_path = job_queue.get(block=False)
            hour_str = hour.strftime(u'%Y-%m-%d-%H')
            url = u'{}{}/index.html'.format(corpus.aws_url_, hour_str)

            r = http.request('GET', url)
            with gzip.open(ulist_path, u'w') as f:
                for link in re.findall(r'<a href="(.*?)">', r.data):
                    if "index.html" in link:
                        continue
                    f.write('{}/{}\n'.format(hour_str, link))

            result_queue.put(None)
        except Queue.Empty:
            pass
        
class SCChunkResource(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'sc-chunks')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_chunks_for_hour(self, hour, corpus):
        hour_dir = os.path.join(
            self.dir_, corpus.fs_name(), hour.strftime(u'%Y-%m-%d-%H'))
        if os.path.exists(hour_dir):
            chunks = [os.path.join(hour_dir, fname) for fname
                      in os.listdir(hour_dir)]
            return chunks
        else:
            return []

    def get_chunk_info_paths_urls(self, event, corpus, preroll=0):
        data_dir = os.path.join(self.dir_, corpus.fs_name())
        data = []
        for url_path in UrlListResource().get_event_url_paths(
            event, corpus, preroll=preroll, must_exist=True):
            with gzip.open(url_path, u'r') as f:
                for line in f:
                    if line == '\n':
                        continue
                    if "index" in line:
                        continue
                    chunk = line.strip()
                    chunk = os.path.splitext(chunk)[0]
                    path = os.path.join(data_dir, chunk)
                    url = '{}{}.gpg'.format(corpus.aws_url_, chunk)
                    domain, count = os.path.split(path)[-1].split('-')[0:2]
                    data.append((domain, int(count), path, url))
        return data    

    def __unicode__(self):
        return u"cuttsum.sc.SCChunkResource"
 
    def dependencies(self):
        return tuple(['UrlListResource',])

    def check_coverage(self, event, corpus, domains=None,
        preroll=0, **kwargs):
        n_chunks = 0
        n_covered = 0
        for domain, count, path, url in self.get_chunk_info_paths_urls(
            event, corpus, preroll=preroll):
            if domains is None or domain in domains:
                n_chunks += 1
                if os.path.exists(path):
                    n_covered += 1    

        if n_chunks == 0:
            return 0
        coverage = n_covered / float(n_chunks)
        return coverage

    def get(self, event, corpus, domains=None, overwrite=False, 
            n_procs=1, progress_bar=False, preroll=0, **kwargs):
        n_procs = min(10, n_procs)
        print "processes:", n_procs
        jobs = []
        for domain, count, path, url in self.get_chunk_info_paths_urls(
            event, corpus, preroll=preroll):
            if domains is None or domain in domains:
                if overwrite is True or not os.path.exists(path):
                    jobs.append((path, url))

        self.do_work(scchunk_worker_, jobs, n_procs, progress_bar)
        return True


def scchunk_worker_(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    gpg = GPG()
    http = urllib3.PoolManager(timeout=15.0, retries=3)

    while not job_queue.empty():
        try:
            path, url = job_queue.get(block=False)
            parent = os.path.dirname(path)
            if not os.path.exists(parent):
                try:
                    os.makedirs(parent)
                except OSError as e: 
                    if e.errno == errno.EEXIST and os.path.isdir(parent):
                        pass

            r = http.request('GET', url)
            with open(path, u'wb') as f:
                f.write(str(gpg.decrypt(r.data)))
            result_queue.put(None)
        except Queue.Empty:
            pass

class SuperSetUrlListResource(UrlListResource):
    def __unicode__(self):
        return u"cuttsum.sc.SuperSetUrlListResource"
    
    def get_event_url_paths(self, event, corpus, preroll=0):
        corpus = corpus.get_superset()
        return super(SuperSetUrlListResource, self).get_event_url_paths(
            event, corpus, preroll=preroll)

    def check_coverage(self, event, corpus, preroll=0, **kwargs):
        corpus = corpus.get_superset()
        return super(SuperSetUrlListResource, self).check_coverage(
            event, corpus, preroll=preroll, **kwargs)

    def get(self, event, corpus, overwrite=False, n_procs=1,
            progress_bar=True, preroll=0, **kwargs):
        corpus = corpus.get_superset()
        return super(SuperSetUrlListResource, self).get(
            event, corpus, overwrite=overwrite, n_procs=n_procs,
            progress_bar=progress_bar, preroll=preroll, **kwargs)

class SCSuperSetChunkResource(SCChunkResource):
    def __unicode__(self):
        return u"cuttsum.sc.SCSuperSetChunkResource"

    def dependencies(self):
        return tuple(['SuperSetUrlListResource',])
    
    def get_chunks_for_hour(self, hour, corpus):
        corpus = corpus.get_superset()
        return super(SCSuperSetChunkResource, self).get_chunks_for_hour(
            hour, corpus)
    
    def get_chunk_info_paths_urls(self, event, corpus, preroll=0):
        corpus = corpus.get_superset()
        return super(SCSuperSetChunkResource, 
                     self).get_chunk_info_paths_urls(event, corpus,
                                                     preroll=preroll)

    def check_coverage(self, event, corpus, domains=None, preroll=0,
                       **kwargs):
        corpus = corpus.get_superset()
        return super(SCSuperSetChunkResource, self).check_coverage(
            event, corpus, domains=domains, preroll=preroll, **kwargs)

    def get(self, event, corpus, **kwargs):
        corpus = corpus.get_superset()
        return super(SCSuperSetChunkResource, self).get(
            event, corpus, **kwargs)

class IdfResource(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'idf')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def __unicode__(self):
        return u"cuttsum.sc.IdfResource"

    def dependencies(self):
        return tuple(['SCSuperSetChunkResource',])

    def get_idf_path(self, hour, corpus):
        corpus = corpus.get_superset()
        return os.path.join(
            self.dir_, corpus.fs_name(), u'{}.idf.marisa.gz'.format(
                hour.strftime(u'%Y-%m-%d-%H')))

    def check_coverage(self, event, corpus, preroll=0, **kwargs):
        corpus = corpus.get_superset()
        domains = kwargs.get(u'domains', None)
        data_dir = os.path.join(self.dir_, corpus.fs_name())
        chunks = SCSuperSetChunkResource()

        n_hours = 0
        n_covered = 0
        for hour in event.list_event_hours(preroll=preroll):
            tfidf_path = os.path.join(data_dir, u"{}.idf.marisa.gz".format(
                hour.strftime(u"%Y-%m-%d-%H")))
            n_hours += 1
            if os.path.exists(tfidf_path):
                n_covered += 1
            
        if n_hours == 0:
            return 0
        else:
            return n_covered / float(n_hours)

    def get(self, event, corpus, preroll=0, **kwargs):
        corpus = corpus.get_superset()
        print corpus
        data_dir = os.path.join(self.dir_, corpus.fs_name())
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        domains = kwargs.get(u'domains', None)
        chunks = SCSuperSetChunkResource()
        
        hour2chunks = defaultdict(list)
        for d, c, path, url in chunks.get_chunk_info_paths_urls(
            event, corpus, preroll=preroll):
            if domains is None or d in domains:
                hour = os.path.basename(os.path.dirname(path))
                hour2chunks[hour].append(path)

        overwrite = kwargs.get(u'overwrite', False)
        jobs = []
        for hour, paths in hour2chunks.iteritems():
            mpath = os.path.join(data_dir, "{}.idf.marisa.gz".format(hour))
            if overwrite is True or not os.path.exists(mpath):
                jobs.append((mpath, paths))

        n_procs = kwargs.get(u'n_procs', 1)
        progress_bar = kwargs.get(u'progress_bar', False)
        self.do_work(_idf_resource_worker, jobs,
                     n_procs, progress_bar, corpus=corpus)

def _idf_resource_worker(job_queue, result_queue, **kwargs):

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    corpus = kwargs.get(u'corpus')

    while not job_queue.empty():
        try:
            
            mpath, paths = job_queue.get(block=False)
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

            idfs = [tuple([np.log(n_docs / value) + 1, value])
                      for value in counts.values()] 

            trie = marisa_trie.RecordTrie("<dd", zip(words, idfs))
            with gzip.open(mpath, u'wb') as f:
                trie.write(f)

            result_queue.put(None)
        except Queue.Empty:
            pass
