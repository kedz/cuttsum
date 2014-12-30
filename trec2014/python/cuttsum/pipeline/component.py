from ..lm import DomainLMResource, GigawordLMResource
from ..data import Resource, get_resource_manager
from ..sc import SCChunkResource, IdfResource
from ..pipeline import TfIdfExtractor, LMProbExtractor, BasicFeaturesExtractor
import os
from cuttsum.detector import ArticleDetector
import streamcorpus as sc
import re
import pandas as pd
from itertools import izip
import signal
import Queue
import cuttsum.srilm
import gzip
from datetime import datetime, timedelta

class ArticlesResource(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'articles')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

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
            jobs.append((path, chunk_paths))

        n_procs = kwargs.get(u'n_procs', 1)
        progress_bar = kwargs.get(u'progress_bar', False)
        self.do_work(_article_resource_worker, jobs, n_procs, progress_bar,
                     event=event, corpus=corpus)

    def dependencies(self):
        return tuple(['SCChunkResource'])

    def __unicode__(self):
        return u"cuttsum.pipeline.ArticlesResource"

def _article_resource_worker(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    event = kwargs.get(u'event')
    corpus = kwargs.get(u'corpus')
    while not job_queue.empty():
        try:
            opath, chunk_paths = job_queue.get(block=False)
            artcl_detect = ArticleDetector(event)
            patt = event.regex_pattern()
            with sc.Chunk(path=opath, mode='wb', message=corpus.sc_msg()) as ochunk:
                for path in chunk_paths:
                    for si in sc.Chunk(path=path, message=corpus.sc_msg()):
                        if si.body.clean_visible is None:
                            continue
                        
                        elif patt.search(si.body.clean_visible, re.I):
                            
                            #if corpus.annotator() not in si.body.sentences:
                            #    continue
                            sentences = corpus.get_sentences(si)
                            sent_idxs = artcl_detect.find_articles(
                                sentences)
                            if len(sent_idxs) > 0:
                                rel_sents = []
                                for sent_idx in sent_idxs:
                                    #for token in sentences[sent_idx].tokens:
                                    #    print token.token,
                                    #print
                                    rel_sents.append(sentences[sent_idx])
                                si.body.sentences[u'article-clf'] = rel_sents
                                ochunk.add(si)


            result_queue.put(None)
        except Queue.Empty:
            pass


class SentenceFeaturesResource(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'sentence-features')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def dependencies(self):
        return tuple([u'DomainLMResource', u'IdfResource',
                      u'ArticlesResource', u'GigawordLMResource',
                      u'SentenceStringsResource',])

    def __unicode__(self):
        return u"cuttsum.pipeline.SentenceFeaturesResource"

    def get_tsv_path(self, event, hour):
        data_dir = os.path.join(self.dir_, event.fs_name())
        return os.path.join(data_dir, u'{}.tsv.gz'.format(
            hour.strftime(u'%Y-%m-%d-%H')))

    def check_coverage(self, event, corpus, **kwargs):
        
        n_hours = 0
        n_covered = 0

        strings = get_resource_manager(u'SentenceStringsResource')

        for hour in event.list_event_hours():
            if os.path.join(strings.get_tsv_path(event, hour)):
                n_hours += 1
                if os.path.exists(self.get_tsv_path(event, hour)):
                    n_covered += 1

        if n_hours == 0:
            return 0
        else:
            return n_covered / float(n_hours)

    def get(self, event, corpus, overwrite=False, n_procs=1, 
            progress_bar=False, preroll=0, **kwargs):
        strings = get_resource_manager(u'SentenceStringsResource')

        data_dir = os.path.join(self.dir_, event.fs_name())
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        jobs = []
        for hour in event.list_event_hours():
            string_tsv_path = strings.get_tsv_path(event, hour)
            feature_tsv_path = self.get_tsv_path(event, hour)
            if os.path.exists(string_tsv_path):
                if overwrite is True or not os.path.exists(feature_tsv_path):
                    jobs.append((string_tsv_path, hour, feature_tsv_path))    

        domainlm = get_resource_manager(u'DomainLMResource')
        domainlm_port = domainlm.get_port(event)
        gigawordlm = get_resource_manager(u'GigawordLMResource')
        gigawordlm_port = gigawordlm.get_gigaword_port()

        if not cuttsum.srilm.check_status(domainlm_port):
            print u"starting domain.lm..."
            cuttsum.srilm.start_lm(
                domainlm.get_arpa_path(event), 3, domainlm_port)

        if not cuttsum.srilm.check_status(gigawordlm_port):
            print u"starting gigaword.lm..."
            cuttsum.srilm.start_lm(
                gigawordlm.get_arpa_path(), 5, gigawordlm_port)

        self.do_work(sentencefeature_worker_, jobs, n_procs, progress_bar,
                     event=event, corpus=corpus, preroll=preroll)

def sentencefeature_worker_(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
 
    event = kwargs.get(u'event')
    corpus = kwargs.get(u'corpus')
    preroll = kwargs.get(u'preroll')

    basic_ext = BasicFeaturesExtractor()

    domainlm = get_resource_manager(u'DomainLMResource')
    domainlm_port = domainlm.get_port(event)
    gigawordlm = get_resource_manager(u'GigawordLMResource')
    gigawordlm_port = gigawordlm.get_gigaword_port()
    lm_ext = LMProbExtractor(domainlm_port, 3, gigawordlm_port, 5)

    idfs = get_resource_manager(u'IdfResource')
    def get_idf_paths(hour):
        return [idfs.get_idf_path(hour - timedelta(hours=i), corpus)
                for i in range(preroll)]

    while not job_queue.empty():
        try:
            string_tsv_path, hour, feature_tsv_path = \
                job_queue.get(block=False)

            idf_paths = get_idf_paths(hour)
            tfidf_ext = TfIdfExtractor(idf_paths[0], idf_paths[1:])

            with gzip.open(string_tsv_path, u'r') as f:
                string_df = pd.io.parsers.read_csv(
                    f, sep='\t', quoting=3, header=0)
            feature_maps = []
            articles = string_df.groupby(u'stream id')
            for _, article in articles:
                #for index, sentence in article.iterrows():
                cnlp_strings = article[u'corenlp'].tolist()
                sc_strings = article[u'streamcorpus'].tolist()
                                        
                tfidf_feats = tfidf_ext.process_streamcorpus_strings(
                    sc_strings)
                lm_feats = lm_ext.process_corenlp_strings(cnlp_strings)
                basic_feats = basic_ext.process_sentences(
                    sc_strings, cnlp_strings)
                for index, (_, sentence) in enumerate(article.iterrows()):
                    assert len(tfidf_feats[index]) == preroll
                    feature_map = {u'stream id': sentence[u'stream id'],
                                   u'sentence id': sentence[u'sentence id']}
                    feature_map.update(basic_feats[index].iteritems())
                    feature_map.update(lm_feats[index].iteritems())
                    feature_map.update(tfidf_feats[index].iteritems())
                    feature_maps.append(feature_map)
            columns = [u'stream id', u'sentence id'] \
                + basic_ext.features + lm_ext.features \
                + tfidf_ext.features
            df = pd.DataFrame(feature_maps, columns=columns)

            with gzip.open(feature_tsv_path, u'w') as f:
                df.to_csv(f, sep='\t', index=False, index_label=False, 
                          na_rep='nan')  


#                    print tfidf_feats
#            for _, sentence in string_df.iterrows():
#                lm_feats = lm_ext.process_corenlp_string(sentence[u'corenlp'])





#                print sentence[u'corenlp']
#                print lm_feats            

#            sentence_sim_data = []
#            for (ii, lvec_row), (_, strings_row) in izip(
#                lvec_df.iterrows(), strings_df.iterrows()):
#                assert lvec_row[u'stream id'] == strings_row[u'stream id']
#                assert lvec_row[u'sentence id'] == strings_row[u'sentence id']
#
#                sentence_sim_dict = {nugget_id_list[nugget_idx]: nugget_sim
#                                     for nugget_idx, nugget_sim
#                                     in enumerate(S[:, ii])}
#                    
#                nugget_ids = check_assessor_matches(
#                    lvec_row[u'stream id'], lvec_row[u'sentence id'])
#
#                sentence_sim_dict.update(nugget_ids.items())
#                sentence_sim_dict[u'stream id'] = lvec_row[u'stream id']
#                sentence_sim_dict[u'sentence id'] = lvec_row[u'sentence id']
#                sentence_sim_data.append(sentence_sim_dict)
#
#                ### Need to remove this when I confirm 2013 matches are good
#                writeme = "/local/nlp/kedzie/check_matches/" + \
#                    event.fs_name() + "." + os.path.basename(lvec_tsv_path)
#                if len(nugget_ids) > 0:
#                    with gzip.open(writeme, u'w') as f:
#                        f.write(strings_row[u'streamcorpus'] + '\n')
#                        for nugget_id in nugget_ids:
#                            nugget_texts = nuggets[ \
#                                nuggets['nugget id'].str.match(
#                                    nugget_id)]['text'].tolist()
#                            for text in nugget_texts:    
#                                f.write("\t" + text + '\n')
# 
            result_queue.put(None)
        except Queue.Empty:
            pass

    return True
#                n_hours += 1
#                if os.path.exists(self.get_tsv_path(event, hour)):
#                    n_covered += 1
#
#
#        hours = event.list_event_hours() 


        #idfs = IdfResource()
        #lms = LMResource()
#        articles = ArticlesResource()
        
#        lm_ext = LMProbExtractor(
#            lms.get_domain_port(event), 3, lms.get_gigaword_port(), 5)
                          

#        for hour in hours:
#            articles_path = articles.get_chunk_path(event, hour)
#            print hour
#            tfidf_ext = TfIdfExtractor(idfs.get_idf_path(hour, corpus))
            
            
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


#        if not os.path.exists(data_dir):
#            os

