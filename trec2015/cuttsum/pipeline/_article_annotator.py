from cuttsum.resources import MultiProcessWorker
import signal
import urllib3
import os
import Queue
import gzip
import re
import streamcorpus as sc

class ArticlesResource(MultiProcessWorker):
    def __init__(self):
        #Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'articles')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)


    def streamitem_iter(self, event, corpus):
        for hour in event.list_event_hours():
            path = self.get_chunk_path(event, hour)
            if os.path.exists(path):
            
                with sc.Chunk(path=path, mode="rb", 
                        message=corpus.sc_msg()) as chunk:
                    for si in chunk:
                        yield hour, path, si   




    def check_coverage(self, event, corpus, **kwargs):
        data_dir = os.path.join(self.dir_, event.fs_name())
        hours = event.list_event_hours()
        n_hours = len(hours)
        n_covered = 0
        for hour in hours:
            path = os.path.join(data_dir, '{}.sc.gz'.format(
                hour.strftime(u'%Y-%m-%d-%H')))
            if os.path.exists(path):
                n_covered += 1
        if n_hours == 0:
            return 0
        else:
            return n_covered / float(n_hours)


    def get_chunk_path(self, event, hour):
        data_dir = os.path.join(self.dir_, event.fs_name())
        return os.path.join(data_dir, u'{}.sc.gz'.format(
            hour.strftime(u'%Y-%m-%d-%H')))

#    def get(self, event, corpus, **kwargs):
#        data_dir = os.path.join(self.dir_, event.fs_name())
#        if not os.path.exists(data_dir):
#            os.makedirs(data_dir)
#        chunks = SCChunkResource()
#        
#
#        overwrite = kwargs.get(u'overwrite', False)
#        hours = event.list_event_hours()
#
#        jobs = []
#        for hour in hours: 
#            path = self.get_chunk_path(event, hour)
#            if os.path.exists(path) and overwrite is False:
#                continue
#            chunk_paths = chunks.get_chunks_for_hour(hour, corpus)
#            jobs.append((path, chunk_paths))
#
#        n_procs = kwargs.get(u'n_procs', 1)
#        progress_bar = kwargs.get(u'progress_bar', False)
#        self.do_work(_article_resource_worker, jobs, n_procs, progress_bar,
#                     event=event, corpus=corpus)
#
#    def dependencies(self):
#        return tuple(['SCChunkResource'])
#
    def __unicode__(self):
        return unicode(self.__class__.__name__)

    def __str__(self):
        return self.__class__.__name__
 

#def _article_resource_worker(job_queue, result_queue, **kwargs):
#    signal.signal(signal.SIGINT, signal.SIG_IGN)
#    event = kwargs.get(u'event')
#    corpus = kwargs.get(u'corpus')
#    while not job_queue.empty():
#        try:
#            opath, chunk_paths = job_queue.get(block=False)
#            artcl_detect = ArticleDetector(event)
#            patt = event.regex_pattern()
#            with sc.Chunk(path=opath, mode='wb', message=corpus.sc_msg()) as ochunk:
#                for path in chunk_paths:
#                    for si in sc.Chunk(path=path, message=corpus.sc_msg()):
#                        if si.body.clean_visible is None:
#                            continue
#                        
#                        elif patt.search(si.body.clean_visible, re.I):
#                            
#                            #if corpus.annotator() not in si.body.sentences:
#                            #    continue
#                            sentences = corpus.get_sentences(si)
#                            sent_idxs = artcl_detect.find_articles(
#                                sentences)
#                            if len(sent_idxs) > 0:
#                                rel_sents = []
#                                for sent_idx in sent_idxs:
#                                    #for token in sentences[sent_idx].tokens:
#                                    #    print token.token,
#                                    #print
#                                    rel_sents.append(sentences[sent_idx])
#                                si.body.sentences[u'article-clf'] = rel_sents
#                                ochunk.add(si)
#
#
#            result_queue.put(None)
#        except Queue.Empty:
#            pass
#
#
