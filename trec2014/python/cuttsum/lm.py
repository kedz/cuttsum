from .data import Resource
import os
import sys
import pandas as pd
import gzip
import urllib3
import signal
import json
import Queue
import re
import corenlp.server
from bs4 import BeautifulSoup

class WikiListResource(Resource):
    """Manager of wikipedia page lists for each language model/event type."""

    def __init__(self):
        Resource.__init__(self)
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
        return u"cuttsum.lm.WikiListResource"

    def get_list_path(self, event):
        return os.path.join(self.dir_, self.type2fname[event.type])

    def check_coverage(self, event, corpus, **kwargs):
        list_path = self.get_list_path(event)
        if os.path.exists(list_path):
            return 1
        else:
            return 0

    def _query(self, request, http):
        url = 'http://en.wikipedia.org/w/api.php'
        request['action'] = 'query'
        request['format'] = 'json'
        request['continue'] = ''
        last_continue = dict()
        while True:

            req = request.copy()
            req.update(last_continue)
            r = http.request_encode_url('GET', url, fields=req)
            result = json.loads(r.data)
           
            if 'error' in result:
                print result['error']
                break
            if 'warnings' in result: print(result['warnings'])
            if 'query' in result: yield result['query']
            if 'continue' not in result: break
            last_continue = result['continue']


    def get(self, event, corpus, max_depth=7, overwrite=False, n_procs=1, 
            **kwargs):

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
            sys.stdout.write(' ' * 79)
            sys.stdout.write('\r') 
            sys.stdout.write("Categories in queue: {}".format(len(queue)))
            sys.stdout.flush()

            req = {
                'generator':'categorymembers',
                'gcmtitle': cat
            }

            for result in self._query(req, http):
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
                        elif m['ns'] == 14:
                            title = m['title'].encode('utf-8')
                            if depth + 1 < max_depth and title not in visited:
                                visited.add(title)
                                queue.append((title, depth + 1))

        good_pages.sort(key=lambda x: x['depth'])
        
        df = pd.DataFrame(good_pages, columns=[u'depth', u'category', u'title'])

        with gzip.open(list_path, 'w') as f:
            df.to_csv(f, sep='\t')
        return True

    def dependencies(self):
        return tuple([])

class DomainLMInputResource(Resource):
    """Manager of input (one sentence per line docs) 
for each language model."""
    
    def __init__(self):
        Resource.__init__(self)
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
        return u"cuttsum.lm.DomainLMInputResource"

    def check_coverage(self, event, corpus, **kwargs):
        path = self.get_domain_lm_input_path(event)
        if os.path.exists(path):
            return 1
        else:
            return 0

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
        return tuple(['WikiListResource'])

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
                            if token.ne == 'O':
                                words = token.lem.split(u'_')
                                for word in words:
                                    if word != u'':
                                        norm_tokens.append(word.lower())
                            else: 
                                norm_tokens.append(
                                    u'__{}__'.format(token.ne.lower()))
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
    """Manager of arpa files for SRILM."""

    def __init__(self):
        Resource.__init__(self)
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
            u'accidents_3.arpa.gz':        9998,
            u'terrorism_3.arpa.gz':        9997,
            u'natural_disaster_3.arpa.gz': 9996,
            u'murder_3.arpa.gz':           9995,
            u'social_unrest_3.arpa.gz':    9994,
            u'astronomy_3.arpa.gz':        9993,
        }

    def get_port(self, event):
        lm = self.domain_lm_[event.type]
        return self.ports_[lm]

    def get_arpa_path(self, event):
        return os.path.join(self.dir_, self.domain_lm_[event.type])

    def __unicode__(self):
        return u"cuttsum.lm.DomainLMResource"

    def check_coverage(self, event, corpus, **kwargs):
        if os.path.exists(self.get_arpa_path(event)):
            return 1.0
        else: 
            return 0.0

    def get(self, event, corpus, **kwargs):
        lminputs = DomainLMInputResource()
        lminput_path = lminputs.get_domain_lm_input_path(event)
        arpa_path = self.get_arpa_path(event)

        cmd = 'ngram-count -order 3 -kndiscount -interpolate ' \
              '-text {} -lm {}'.format(lminput_path, arpa_path)

        os.system(cmd)        

    def dependencies(self):
        return tuple(['DomainLMInputResource'])

class GigawordLMResource(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'lm')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)
        
        self.gigaword_lm_ = u'gigaword_5.arpa.gz'
        self.ports_ = 9999

    def get_gigaword_port(self):
        return self.ports_

    def gigaword_path(self):
        return os.path.join(self.dir_, self.gigaword_lm_)

    def __unicode__(self):
        return u"cuttsum.lm.GigawordLMResource"

    def check_coverage(self, event, corpus, **kwargs):
        coverage = 0.0
        if os.path.exists(self.gigaword_path()):
            coverage = 1.0
        
        return coverage

    def get(self, event, corpus, **kwargs):
        raise NotImplementedError(
            "I don't know how to make a language model from scratch yet")

    def dependencies(self):
        return tuple([])
