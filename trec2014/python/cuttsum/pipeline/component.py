from ..data import Resource, SCChunkResource, IdfResource, LMResource
from ..pipeline import TfIdfExtractor, LMProbExtractor
import os
from cuttsum.detector import ArticleDetector
import streamcorpus as sc
import re
import pandas as pd
from itertools import izip

class ArticlesResource(Resource):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'articles')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    @Resource.getdependency
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

    @Resource.getdependency
    def get(self, event, corpus, **kwargs):
        data_dir = os.path.join(self.dir_, event.fs_name())
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        chunks = SCChunkResource()
        

        overwrite = kwargs.get(u'overwrite', False)
        hours = event.list_event_hours()

        jobs = []
        for hour in hours: 
            path = self.get_chunk_path(event, hour)
            if os.path.exists(path) and overwrite is False:
                continue
            chunk_paths = chunks.get_chunks_for_hour(hour, corpus)
            jobs.append((path, chunk_paths, event, corpus))

        n_procs = kwargs.get(u'n_procs', 1)
        progress_bar = kwargs.get(u'progress_bar', False)
        self.do_work(_article_resource_worker, jobs, n_procs, progress_bar)

    def dependencies(self):
        return tuple([SCChunkResource])

    def __unicode__(self):
        return u"cuttsum.pipeline.ArticlesResource"

def _article_resource_worker(args):
    opath, chunk_paths, event, corpus = args
    artcl_detect = ArticleDetector(event)
    patt = event.regex_pattern()
    with sc.Chunk(path=opath, mode='wb', message=corpus.sc_msg()) as ochunk:
        for path in chunk_paths:
            for si in sc.Chunk(path=path, message=corpus.sc_msg()):
                if si.body.clean_visible is None:
                    continue
                elif patt.search(si.body.clean_visible, re.I):
                    if corpus.annotator() not in si.body.sentences:
                        continue
                    sent_idxs = artcl_detect.find_articles(
                        si, annotator=corpus.annotator())
                    if len(sent_idxs) > 0:
                        rel_sents = []
                        for sent_idx in sent_idxs:
                            rel_sents.append(
                                si.body.sentences[
                                    corpus.annotator()][sent_idx])
                        si.body.sentences[u'article-clf'] = rel_sents
                        ochunk.add(si)


class SentenceFeaturesResource(Resource):
    def __init__(self):
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'sentence-features')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_features_path(self, event, hour):
        data_dir = os.path.join(self.dir_, event.fs_name())
        return os.path.join(data_dir, u'{}.tsv.gz'.format(
            hour.strftime(u'%Y-%m-%d-%H')))

    @Resource.getdependency
    def check_coverage(self, event, corpus, **kwargs):
        
        hours = event.list_event_hours()

        n_covered = 0
        for hour in hours: 
            path = self.get_features_path(event, hour)
            if os.path.exists(path):
                n_covered += 1
        n_hours = len(hours)
        if n_hours == 0:
            return 0
        else:
            return n_covered / float(n_hours)

    @Resource.getdependency
    def get(self, event, corpus, **kwargs):
        hours = event.list_event_hours() 

        idfs = IdfResource()
        lms = LMResource()
        articles = ArticlesResource()
        
        lm_ext = LMProbExtractor(
            lms.get_domain_port(event), 3, lms.get_gigaword_port(), 5)
                          

        for hour in hours:
            articles_path = articles.get_chunk_path(event, hour)
            print hour
            tfidf_ext = TfIdfExtractor(idfs.get_idf_path(hour, corpus))
            
            
#            for si in sc.Chunk(path=articles_path, message=corpus.sc_msg()):
#                
#                freq_features = tfidf_ext.process_article(si, corpus)
#                lm_features = lm_ext.process_article(si)
#
#                features = []
#                for m1, m2 in izip(freq_features, lm_features):
#                    features.append(dict(m1.items() + m2.items()))
#                df = pd.DataFrame(features)    
#                print df 

    def dependencies(self):
        return tuple([LMResource, IdfResource, ArticlesResource])


    def __unicode__(self):
        return u"cuttsum.pipeline.SentenceFeaturesResource"

#        if not os.path.exists(data_dir):
#            os
