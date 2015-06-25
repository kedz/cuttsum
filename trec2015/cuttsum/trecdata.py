from cuttsum.resources import MultiProcessWorker
import signal
import requests
import urllib3.exceptions
import os
import Queue
import gzip
import re
import streamcorpus as sc
import errno
from gnupg import GPG

class UrlListResource(MultiProcessWorker):
    def __init__(self):
        #Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'url-lists')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def __unicode__(self):
        return unicode(self.__class__.__name__)

    def __str__(self):
        return self.__class__.__name__
    

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



    def get_job_units(self, event, corpus, **kwargs):
        preroll = kwargs.get("preroll", 0)
        overwrite = kwargs.get("overwrite", False)

        data_dir = os.path.join(self.dir_, corpus.fs_name())
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        units = []
        hours = event.list_event_hours(preroll=preroll)
        for i, hour in enumerate(hours):
            ulist_path = os.path.join(
                data_dir, u'{}.txt.gz'.format(hour.strftime(u'%Y-%m-%d-%H')))
            if overwrite is True or not os.path.exists(ulist_path):
                units.append(i)
        return units 

    def do_job_unit(self, event, corpus, unit, **kwargs):
        preroll = kwargs.get("preroll", 0)
        data_dir = os.path.join(self.dir_, corpus.fs_name())
        if not os.path.exists(data_dir):	
            try:
                os.makedirs(data_dir)
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(data_dir):
                    pass

        http = urllib3.PoolManager(timeout=15.0) #, retries=True)
        hours = event.list_event_hours(preroll=preroll)
        for i, hour in enumerate(hours):
            if i != unit:
                continue
            ulist_path = os.path.join(
                data_dir, u'{}.txt.gz'.format(hour.strftime(u'%Y-%m-%d-%H')))
            hour_str = hour.strftime(u'%Y-%m-%d-%H')
            url = u'{}{}/index.html'.format(corpus.aws_url_, hour_str)

            r = http.request('GET', url)
            with gzip.open(ulist_path, u'w') as f:
                for link in re.findall(r'<a href="(.*?)">', r.data):
                    if "index.html" in link:
                        continue
                    if "news" in link or "NEWS" in link:
                        f.write('{}/{}\n'.format(hour_str, link))
           
        


def urllist_worker_(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    http = urllib3.PoolManager(timeout=15.0) #, retries=True)
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
                    if "news" in link or "NEWS" in link:
                        f.write('{}/{}\n'.format(hour_str, link))

            result_queue.put(None)
        except Queue.Empty:
            pass

class SCChunkResource(MultiProcessWorker):
    def __init__(self):
        #Resource.__init__(self)
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

    def streamitem_iter(self, event, corpus):
        for hour in event.list_event_hours():
            for chunk_path in self.get_chunks_for_hour(hour, corpus):
                with sc.Chunk(path=chunk_path, mode="rb", 
                        message=corpus.sc_msg()) as chunk:
                    for si in chunk:
                        yield hour, chunk_path, si   

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
        return unicode(self.__class__.__name__)

    def __str__(self):
        return self.__class__.__name__
 
    def dependencies(self):
        return tuple(['UrlListResource',])


    def get_job_units(self, event, corpus, **kwargs):
        preroll = kwargs.get("preroll", 0)
        overwrite = kwargs.get("overwrite", False)

        n_chunks = 0
        n_covered = 0
        units = []
        for unit, (domain, count, path, url) in enumerate(self.get_chunk_info_paths_urls(
                event, corpus, preroll=preroll)):
            #if domains is None or domain in domains:
            
            n_chunks += 1
            if overwrite is True or not os.path.exists(path):
                units.append(unit)
        return units

    def do_job_unit(self, event, corpus, unit, **kwargs):
        preroll = kwargs.get("preroll", 0)
        
        for i, (domain, count, path, url) in enumerate(self.get_chunk_info_paths_urls(
                event, corpus, preroll=preroll)):
            if i != unit:
                continue
            gpg = GPG()
            
            #http = urllib3.PoolManager()
            parent = os.path.dirname(path)
            if not os.path.exists(parent):
                try:
                    os.makedirs(parent)
                except OSError as e:
                    if e.errno == errno.EEXIST and os.path.isdir(parent):
                        pass

            retries = 3
            while 1:
                try:
                    r = requests.get(url, timeout=30)
                    with open(path, u'wb') as f:
                        f.write(str(gpg.decrypt(r.content)))
                    break
                except requests.exceptions.ConnectionError:
                    retries -= 1
                    if retries == 0:
                        break

                except urllib3.exceptions.ReadTimeoutError:
                    retries -= 1
                    if retries == 0:
                        break

class SCChunkValidatorResource(MultiProcessWorker):
    def __init__(self):
        pass

    def get_job_units(self, event, corpus, **kwargs):
        preroll = kwargs.get("preroll", 0)
        
        num_hours = len(event.list_event_hours(preroll=preroll))
        units = [i for i in xrange(num_hours)]
        return units

    def do_job_unit(self, event, corpus, unit, **kwargs):
        preroll = kwargs.get("preroll", 0)        
        hour = event.list_event_hours(preroll=preroll)[unit]

        res = SCChunkResource()
        import subprocess        

        for path in res.get_chunks_for_hour(hour, corpus):
            gold_checksum = path.split("-")[-1].replace(".sc.xz", "")
            o = subprocess.check_output("xzcat {} | md5sum".format(path), shell=True)
            checksum = o.split(" ")[0].strip()
            if checksum != gold_checksum:
                print checksum,"!=", gold_checksum, "removing path", path
                os.remove(path)
