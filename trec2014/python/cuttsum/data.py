import os
import errno
import re
from datetime import datetime, timedelta
import urllib3
import json
from urllib3 import PoolManager
from gnupg import GPG
from StringIO import StringIO
import socket
import gzip
import sys
import multiprocessing
import Queue
from collections import defaultdict
import streamcorpus as sc
import marisa_trie
import numpy as np
import signal
import pandas as pd
from bs4 import BeautifulSoup
import corenlp.server


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

    def do_work(self, worker, jobs, n_procs,
                progress_bar, result_handler=None):
        from .misc import ProgressBar
        max_jobs = len(jobs)
        job_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        for job in jobs:
            job_queue.put(job)

        pool = []        
        for i in xrange(n_procs):
            p = multiprocessing.Process(target=worker,
                                        args=(job_queue, result_queue))
            p.start()
            pool.append(p)            

            pb = ProgressBar(max_jobs)
        try:
            for n_job in xrange(max_jobs):
                result = result_queue.get(block=True)
                if result_handler is not None:
                    result_handler(result)
                if progress_bar is True:
                    pb.update()

            for p in pool:
                p.join()

        except KeyboardInterrupt:
            pb.clear()
            print "Completing current jobs and shutting down!"
            while not job_queue.empty():
                job_queue.get()
            for p in pool:
                p.join()
            sys.exit()


class UrlListResource(Resource):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'url-lists')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def __unicode__(self):
        return u"cuttsum.data.UrlListResource"

    def get_event_url_paths(self, event, corpus, preroll=0):
        paths = []
        for hour in event.list_event_hours(preroll=preroll):
            path = os.path.join(self.dir_, corpus.fs_name(), 
                u'{}.txt.gz'.format(hour.strftime(u'%Y-%m-%d-%H')))
            if os.path.exists(path):
                paths.append(path)
        return paths

    def check_coverage(self, event, corpus, **kwargs):
        n_hours = 0
        n_covered = 0
        preroll = kwargs.get(u'preroll', 0)

        for hour in event.list_event_hours(preroll=preroll):
            path = os.path.join(self.dir_, corpus.fs_name(),
                u'{}.txt.gz'.format(hour.strftime(u'%Y-%m-%d-%H')))
            n_hours += 1
            if os.path.exists(path) and os.path.isfile(path):
                n_covered += 1
        return float(n_covered) / n_hours

    def get(self, event, corpus, overwrite=False, n_procs=1, 
            progress_bar=True, preroll=0, **kwargs):

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
                jobs.append((hour, ulist_path, corpus)) 

        self.do_work(urllist_worker_, jobs, n_procs, progress_bar)
        return True

    def dependencies(self):
        return tuple([])

def urllist_worker_(job_queue, result_queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    http = PoolManager(timeout=15.0, retries=3)

    while not job_queue.empty():
        try:
            hour, ulist_path, corpus = job_queue.get(block=False)
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
            event, corpus, preroll=preroll):
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
        return u"cuttsum.data.SCChunkResource"
 
    @Resource.getdependency
    def check_coverage(self, event, corpus, domains=None, preroll=0, **kwargs):
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
        else:
            return n_covered / float(n_chunks)

    @Resource.getdependency
    def get(self, event, corpus, domains=None, overwrite=False, 
            n_procs=1, progress_bar=False, preroll=0, **kwargs):
        jobs = []
        for domain, count, path, url in self.get_chunk_info_paths_urls(
            event, corpus, preroll=preroll):
            if domains is None or domain in domains:
                if overwrite is True or not os.path.exists(path):
                    jobs.append((path, url))

        self.do_work(scchunk_worker_, jobs, n_procs, progress_bar)
        return True

    def dependencies(self):
        return tuple([UrlListResource])

def scchunk_worker_(job_queue, result_queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    gpg = GPG()
    http = PoolManager(timeout=15.0, retries=3)

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
                #buf = StringIO(r.data)

                f.write(str(gpg.decrypt(r.data)))
            result_queue.put(None)
        except Queue.Empty:
            pass

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
        for hour in event.list_event_hours(preroll=5):
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
        return tuple([SCChunkResource])

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

class WikiListResource(Resource):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'wiki-lists')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

        self.type2cats = {
            u'accident': ['Category:Accidents'],
            u'impact event': ['Category:Astronomy'],
            u'bombing': ['Category:Terrorism'],
            u'hostage': ['Category:Terrorism'],
            u'shooting': ['Category:Murder'],
            u'protest': ['Category:Activism_by_type'],
            u'riot': ['Category:Activism_by_type'],
            u'storm': ['Category:Natural_disasters'],
            u'earthquake': ['Category:Natural_disasters'],
        }

        self.type2fname = {
            u'accident': 'accident.tsv.gz',
            u'impact event': 'astronomy.tsv.gz',
            u'bombing': 'terrorism.tsv.gz',
            u'hostage': 'terrorism.tsv.gz',
            u'shooting': 'murder.tsv.gz',
            u'protest': 'social-unrest.tsv.gz',
            u'riot': 'social-unrest.tsv.gz',
            u'storm': 'natural-disaster.tsv.gz',
            u'earthquake': 'natural-disaster.tsv.gz',
        }

    def __unicode__(self):
        return u"cuttsum.data.WikiListResource"

    def get_list_path(self, event):
        return os.path.join(self.dir_, self.type2fname[event.type])
        

    def check_coverage(self, event, corpus, **kwargs):
        list_path = self.get_list_path(event)
        if os.path.exists(list_path):
            return 1
        else:
            return 0

    def get(self, event, corpus, max_depth=7, overwrite=False, n_procs=1, 
            **kwargs):
        def query(request, http):
            request['action'] = 'query'
            request['format'] = 'json'
            request['continue'] = ''
            last_continue = dict()
            while True:

                req = request.copy()
                req.update(last_continue)
                r = http.request_encode_url('GET', url, fields=req)
                result = json.loads(r.data)
               
                #print result 
                if 'error' in result:
                    print result['error']
                    break
                if 'warnings' in result: print(result['warnings'])
                if 'query' in result: yield result['query']
                if 'continue' not in result: break
                last_continue = result['continue']


        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)
        list_path = self.get_list_path(event)

        good_pages = []

        queue = [(cat, 0) for cat in self.type2cats[event.type]]
        url = 'http://en.wikipedia.org/w/api.php'
        http = urllib3.connection_from_url(url)
        
        visited = set()
        depth = 0
        while len(queue) > 0:

            cat, depth = queue.pop()
            sys.stdout.write('\r') 
            sys.stdout.write(' ' * 80)
            sys.stdout.write('\r') 
            sys.stdout.write("Categories in queue: {}".format(len(queue)))
            sys.stdout.flush()

            req = {
                'generator':'categorymembers',
                'gcmtitle': cat
            }

            for result in query(req, http):
                if 'pages' in result:
                    for page in result['pages']:
                        m = result['pages'][page]
                        if m['ns'] == 0:
                            title = m['title'].encode('utf-8')
                            if title not in visited:
                                visited.add(title)
                                good_pages.append(
                                    {u'depth': depth, 
                                     u'category': cat,
                                     u'title': title})
                                #print depth, cat, 'Page', title
                        elif m['ns'] == 14:
                            title = m['title'].encode('utf-8')
                            if depth + 1 < max_depth and title not in visited:
                                visited.add(title)
                                queue.append((title, depth + 1))


        good_pages.sort(key=lambda x: x['depth'])
        
        df = pd.DataFrame(good_pages, columns=[u'depth', u'category', u'title'])
        #print df

        with gzip.open(list_path, 'w') as f:
            df.to_csv(f, sep='\t')
        #df.to_csv(list_path, compression=u'gz', sep='\t')
        return True

    def dependencies(self):
        return tuple([])

class DomainLMInputResource(Resource):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'domain-lm-input')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

        self.type2fname = {
            u'accident': 'accident.txt.gz',
            u'impact event': 'astronomy.txt.gz',
            u'bombing': 'terrorism.txt.gz',
            u'hostage': 'terrorism.txt.gz',
            u'shooting': 'murder.txt.gz',
            u'protest': 'social-unrest.txt.gz',
            u'riot': 'social-unrest.txt.gz',
            u'storm': 'natural-disaster.txt.gz',
            u'earthquake': 'natural-disaster.txt.gz',
        }

    def get_domain_lm_input_path(self, event):
        return os.path.join(self.dir_, self.type2fname[event.type])
 
    def __unicode__(self):
        return u"cuttsum.data.DomainLMInputResource"

    @Resource.getdependency
    def check_coverage(self, event, corpus, **kwargs):
        path = self.get_domain_lm_input_path(event)
        if os.path.exists(path):
            return 1
        else:
            return 0

    @Resource.getdependency
    def get(self, event, corpus, overwrite=False, progress_bar=False, 
            n_procs=1, **kwargs):

        n_procs = min(n_procs, 5) # Wikipedia doesn't like us hammering them.

        wikilists = WikiListResource()
        list_path = wikilists.get_list_path(event)
        dmn_lm_input_path = self.get_domain_lm_input_path(event)

        with gzip.open(list_path, u'r') as f:
            df = pd.io.parsers.read_csv(
                f, sep='\t', quoting=3, header=0,
                names=[u'depth', u'category', u'title'])
        titles = df[u'title'].values.astype(list)

        if corenlp.server.check_status() is False:
            print "Starting Corenlp server"
            corenlp.server.start(
                mem="20G", threads=20,
                annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner'])
      
        with gzip.open(dmn_lm_input_path, u'w') as f: 
            self.do_work(domainlminput_worker_, titles, n_procs, 
                progress_bar, result_handler=domainlminput_writer_(f))
        
        return True

    def dependencies(self):
        return tuple([WikiListResource])

def domainlminput_worker_(job_queue, result_queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    url = 'http://en.wikipedia.org/w/api.php'
    http = urllib3.connection_from_url(url)
                                  
    page_url_tmp = 'http://en.wikipedia.org/w/index.php?title={}&oldid={}'

    cnlp = corenlp.server.CoreNLPClient()

    while not job_queue.empty():
        try:

            title = job_queue.get(block=False)
        
            fields = {'action':'query', 'format':'json', 'prop':'revisions',
                      'titles':title, 'rvprop':'timestamp|ids',
                      'rvstart':'20111001000000', 'rvdir':'older',
                      'continue': ''}

            r = http.request_encode_url('GET', url, fields=fields)
            result = json.loads(r.data)
            pid = result['query']['pages'].keys()[0]
            revs = result['query']['pages'][pid].get('revisions', None)
            if revs is None:
                result_queue.put([])
                continue

            if len(revs) == 0:
                result_queue.put([])
                continue
            rev = revs[0]
            page_url = page_url_tmp.format(
                title.replace(' ', '_'), rev['parentid'])
            html_content = http.request_encode_url('GET', page_url).data
            
            soup = BeautifulSoup(html_content)

            results = []
            div = soup.find("div", {"id": "mw-content-text"})
            if div is None:
                result_queue.put([])
                continue
            for tag in div.find_all(True, recursive=False):
                if tag.name == 'p':
                    text = tag.get_text()
                    text = re.sub(r'\[\d+\]', '', tag.get_text())
                    text = cnlp.annotate(text)
                    for sent in text:
                        norm_tokens = []
                        for token in sent:
                            if token.ne != 'O':
                                norm_tokens.append(
                                    u'__{}__'.format(token.ne.lower()))
                            else:
                                norm_tokens.append(token.lem)
                        results.append(
                            (u' '.join(norm_tokens)).encode(u'utf-8'))

            result_queue.put(results)
        except Queue.Empty:
            pass

def domainlminput_writer_(file):
        def wrapper(result):
            for sent in result:
                file.write(sent)
                file.write('\n')
            file.flush()
        return wrapper

class DomainLMResource(Resource):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'lm')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)
        
        self.domain_lm_ = {
            u'accident': u'accidents_3.arpa.gz', 
            u'shooting': u'murder_3.arpa.gz',
            u'storm': u'natural_disaster_3.arpa.gz',
            u'earthquake': u'natural_disaster_3.arpa.gz',
            u'bombing': u'terrorism_3.arpa.gz',
            u'riot': u'social_unrest_3.arpa.gz',
            u'protest': u'social_unrest_3.arpa.gz',
            u'hostage': u'terrorism_3.arpa.gz',
            u'impact event': u'astronomy_3.arpa.gz',
        }  
        

        self.ports_ = {
            u'accidents_3.arpa.gz':     9998,
            u'terrorism_3.arpa.gz':     9997,
            u'natural_disaster_3.arpa.gz':       9996,
            u'murder_3.arpa.gz':   9995,
            u'social_unrest_3.arpa.gz': 9994,
            u'astronomy_3.arpa.gz': 9993,
        }

    def get_port(self, event):
        lm = self.domain_lm_[event.type]
        return self.ports_[lm]

    def get_arpa_path(self, event):
        return os.path.join(self.dir_, self.domain_lm_[event.type])

    def __unicode__(self):
        return u"cuttsum.data.DomainLMResource"

    @Resource.getdependency
    def check_coverage(self, event, corpus, **kwargs):
        if os.path.exists(self.get_arpa_path(event)):
            return 1
        else: 
            return 0

    @Resource.getdependency
    def get(self, event, corpus, **kwargs):
        lminputs = DomainLMInputResource()
        lminput_path = lminputs.get_domain_lm_input_path(event)
        arpa_path = self.get_arpa_path(event)

        cmd = 'ngram-count -order 3 -kndiscount -interpolate ' \
              '-text {} -lm {}'.format(lminput_path, arpa_path)

        print cmd
        os.system(cmd)        

    def dependencies(self):
        return tuple([DomainLMInputResource])


