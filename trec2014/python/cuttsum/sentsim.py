#from .jtest import get_2014_nuggets
#from .submissions import Updates
from .judgements import (
    get_2013_nuggets, get_2014_nuggets, get_2013_matches, get_2014_matches
)
from .data import Resource, get_resource_manager
from .misc import stringify_corenlp_doc, stringify_streamcorpus_sentence
import os
import signal
import Queue
import wtmf
import gzip
import re
from sklearn.externals import joblib
import streamcorpus as sc
import corenlp.server
from itertools import izip
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class WTMFModelsResource(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'wtmf-models')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

        self.type2fname = {
            u'accident': 'accident.pkl',
            u'impact event': 'astronomy.pkl',
            u'bombing': 'terrorism.pkl',
            u'hostage': 'terrorism.pkl',
            u'shooting': 'murder.pkl',
            u'protest': 'social-unrest.pkl',
            u'riot': 'social-unrest.pkl',
            u'storm': 'natural-disaster.pkl',
            u'earthquake': 'natural-disaster.pkl',
        }

    def check_coverage(self, event, corpus, **kwargs):
        if os.path.exists(self.get_model_path(event)):
            return 1.0
        else:
            return 0.0    
        
        
    def get(self, event, corpus, **kwargs):

        model_path = self.get_model_path(event)
        parent_dir = os.path.dirname(model_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        
        lminputs = get_resource_manager(u'DomainLMInputResource')
        lminput_path = lminputs.get_domain_lm_input_path(event)
        
        with gzip.open(lminput_path, u'r') as f:
            X = f.readlines()
        vectorizer = wtmf.WTMFVectorizer(
            input='content', k=100, w_m=0.01,lam=20, max_iter=20, 
            tokenizer=domain_lm_input_tokenizer,
            tf_threshold=2, verbose=True).fit(X)

        joblib.dump(vectorizer, model_path)


    def get_model_path(self, event):
        return os.path.join(self.dir_, self.type2fname[event.type])

    def dependencies(self):
        return tuple(['DomainLMInputResource'])

    def __unicode__(self):
        return u"cuttsum.sentsim.WTMFModelsResource"

def domain_lm_input_tokenizer(line):
    # need to filter punc
    # also need to add freq filter
    tokens = [token for token in line.split(' ')
              if token != '-lrb-' and token != '-rrb-']
    tokens = [token for token in tokens
              if len(token) > 2 and not re.match(r'__.+__', token)]
    return tokens


class SentenceStringsResource(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'sentence-strings')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def dependencies(self):
        return tuple(['WTMFModelsResource', 'ArticlesResource'])

    def __unicode__(self):
        return u"cuttsum.sentsim.SentenceStringsResource"

    def dataframe_generator(self, event):
        for hour in event.list_event_hours():
            tsv_path = self.get_tsv_path(event, hour)
            if os.path.exists(tsv_path):
                with gzip.open(tsv_path, u'r') as f:
                    df = pd.io.parsers.read_csv(
                        f, sep='\t', quoting=3, header=0)
                    yield df

    def get_dataframe(self, event, hour):
        tsv = self.get_tsv_path(event, hour)
        if not os.path.exists(tsv):
            return None
        else:
            with gzip.open(tsv, u'r') as f:
                df = pd.io.parsers.read_csv(
                    f, sep='\t', quoting=3, header=0)
                return df

    def get_tsv_path(self, event, hour):
        data_dir = os.path.join(self.dir_, event.fs_name())
        return os.path.join(data_dir, u'{}.tsv.gz'.format(
            hour.strftime(u'%Y-%m-%d-%H')))

    def check_coverage(self, event, corpus, **kwargs):
        articles = get_resource_manager(u'ArticlesResource')
        data_dir = os.path.join(self.dir_, event.fs_name())
        
        n_chunks = 0
        n_covered = 0

        hours = event.list_event_hours()
        for hour in hours:
            if os.path.exists(articles.get_chunk_path(event, hour)):
                n_chunks += 1
                if os.path.exists(self.get_tsv_path(event, hour)):
                    n_covered += 1

        if n_chunks == 0:
            return 0
        else:
            return float(n_covered) / n_chunks

    def get(self, event, corpus, n_procs=1, progress_bar=False, **kwargs):
        articles = get_resource_manager(u'ArticlesResource')

        data_dir = os.path.join(self.dir_, event.fs_name())
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        jobs = []
        for hour in event.list_event_hours():
            article_chunk_path = articles.get_chunk_path(event, hour)
            if os.path.exists(article_chunk_path):
                tsv_path = self.get_tsv_path(event, hour)
                if os.path.exists(tsv_path):
                    continue
                jobs.append((article_chunk_path, tsv_path))     
        
        if corenlp.server.check_status() is False:
            print "starting corenlp.server"
            corenlp.server.start(
                mem="20G", threads=n_procs + 15,
                annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner'])

        self.do_work(sentencestring_worker_, jobs, n_procs, progress_bar,
                     corpus=corpus)

from .geo import get_loc_sequences

def sentencestring_worker_(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    corpus = kwargs.get(u'corpus')
    cnlp = corenlp.server.CoreNLPClient()

    while not job_queue.empty():
        try:

            chunk_path, tsv_path = job_queue.get(block=False)

            sent_string_data = []
            for si in sc.Chunk(path=chunk_path, message=corpus.sc_msg()):
 
                sentences = corpus.get_sentences(si)
                str2idx = {}
                for idx, sentence in enumerate(sentences):
                    key = stringify_streamcorpus_sentence(sentence)
                    str2idx[key] = idx

                for sentence in si.body.sentences[u'article-clf']:
                    sc_string = stringify_streamcorpus_sentence(sentence)
                    idx = str2idx[sc_string]

                    #print idx, ")", sc_string
                    doc = cnlp.annotate(sc_string)
                    locs = get_loc_sequences(doc)
                    if len(locs) > 0:
                        locs_string = (u','.join(locs)).encode(u'utf-8')
                    else:
                        locs_string = 'nan'
                    cnlp_string = stringify_corenlp_doc(doc)
                    #print cnlp_string
                    sent_string_data.append({
                        u'stream id': si.stream_id, u'sentence id': idx,
                        u'streamcorpus': sc_string,
                        u'corenlp': cnlp_string,
                        u'locations': locs_string})

            if len(sent_string_data) > 0:                
                df = pd.DataFrame(
                    sent_string_data, 
                    columns=[u'stream id', u'sentence id', 
                             u'streamcorpus', u'corenlp', u'locations'])

                with gzip.open(tsv_path, u'w') as f:
                    df.to_csv(f, sep='\t', index=False, index_label=False)  



            result_queue.put(None)
        except Queue.Empty:
            pass

#        for job in jobs:
#            articles_chunk, tsv_path = job
#            sentence_meta = []
#            sentence_strings = []
#            for si in sc.Chunk(path=articles_chunk, message=corpus.sc_msg()):
#                si.stream_id       
#                
#                sentences = corpus.get_sentences(si)
#                str2idx = {}
#                for idx, sentence in enumerate(sentences):
#                    key = ' '.join(token.token for token in sentence.tokens)
#                    str2idx[key] = idx
#                
#                for sentence in si.body.sentences[u'article-clf']:
#                    key = ' '.join(token.token for token in sentence.tokens)
#                    idx = str2idx[key]
#                    #print idx, ")", key 
#                    doc = cnlp.annotate(key)
#                    norm_tokens = []
#                    for sent in doc:
#                        for token in sent:
#                            if token.ne == 'O':
#                                words = token.lem.split(u'_')
#                                for word in words:
#                                    if word != u'':
#                                        norm_tokens.append(word.lower())
#                            else: 
#                                norm_tokens.append(
#                                    u'__{}__'.format(token.ne.lower()))
#                    sentence_strings.append(
#                        (u' '.join(norm_tokens)).encode(u'utf-8'))
#                    sentence_meta.append((si.stream_id, idx))
 

       


class SentenceLatentVectorsResource(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'sentence-latent-vectors')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def dataframe_generator(self, event):
        for hour in event.list_event_hours():
            tsv_path = self.get_tsv_path(event, hour)
            if os.path.exists(tsv_path):
                with gzip.open(tsv_path, u'r') as f:
                    df = pd.io.parsers.read_csv(
                        f, sep='\t', quoting=3, header=0)
                    yield df

    def get_dataframe(self, event, hour):
        tsv = self.get_tsv_path(event, hour)
        if not os.path.exists(tsv):
            return None
        else:
            with gzip.open(tsv, u'r') as f:
                df = pd.io.parsers.read_csv(
                    f, sep='\t', quoting=3, header=0)
                return df

    def get_tsv_path(self, event, hour):
        data_dir = os.path.join(self.dir_, event.fs_name())
        return os.path.join(data_dir, u'{}.tsv.gz'.format(
            hour.strftime(u'%Y-%m-%d-%H')))

    def dependencies(self):
        return tuple(['SentenceStringsResource'])

    def __unicode__(self):
        return u"cuttsum.sentsim.SentenceLatentVectorsResource"


    def check_coverage(self, event, corpus, **kwargs):
        sentencestrings = get_resource_manager(u'SentenceStringsResource')
        data_dir = os.path.join(self.dir_, event.fs_name())
        
        n_files = 0
        n_covered = 0

        hours = event.list_event_hours()
        for hour in hours:
            if os.path.exists(sentencestrings.get_tsv_path(event, hour)):
                n_files += 1
                if os.path.exists(self.get_tsv_path(event, hour)):
                    n_covered += 1

        if n_files == 0:
            return 0
        else:
            return float(n_covered) / n_files

    def get(self, event, corpus, overwrite=False, n_procs=1, 
            progress_bar=False, **kwargs):
        sentencestrings = get_resource_manager(u'SentenceStringsResource')
        data_dir = os.path.join(self.dir_, event.fs_name())

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        jobs = []
        for hour in event.list_event_hours():
            strings_tsv_path = sentencestrings.get_tsv_path(event, hour)
            lvec_tsv_path = self.get_tsv_path(event, hour)
            if os.path.exists(strings_tsv_path):
                if overwrite is True or not os.path.exists(lvec_tsv_path):
                    jobs.append((strings_tsv_path, lvec_tsv_path))

        self.do_work(sentencelvec_worker_, jobs, n_procs, progress_bar,
                     event=event)

def sentencelvec_worker_(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    event = kwargs.get(u'event')
    wtmf_models = get_resource_manager(u'WTMFModelsResource')
    model_path = wtmf_models.get_model_path(event)
    vectorizer = joblib.load(model_path)

    while not job_queue.empty():
        try:
            
            strings_tsv, lvec_tsv_path = job_queue.get(block=False)
            
            vecs_data = []
            with gzip.open(strings_tsv, u'r') as f:
                strings_df = pd.io.parsers.read_csv(
                    f, sep='\t', quoting=3, header=0)
            strings = strings_df[u'corenlp'].tolist()
            Xstrings = vectorizer.transform(strings)
            for (_, r), xstrings in izip(strings_df.iterrows(), Xstrings):
                vec_data = {idx: val for idx, val in enumerate(xstrings, 0)}
                vec_data[u'stream id'] = r[u'stream id']
                vec_data[u'sentence id'] = r[u'sentence id'] 
                vecs_data.append(vec_data)

            ndims = Xstrings.shape[1]
            names = [u'stream id', u'sentence id' ] + range(ndims)

            df = pd.DataFrame(vecs_data, columns=names)

            with gzip.open(lvec_tsv_path, u'w') as f:
                df.to_csv(f, sep='\t', index=False, index_label=False)  

            result_queue.put(None)
        except Queue.Empty:
            pass

    return True

class NuggetSimilaritiesResource(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'nugget-similarities')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def dependencies(self):
        return tuple([u'SentenceLatentVectorsResource',
                      u'SentenceStringsResource' ])

    def __unicode__(self):
        return u"cuttsum.sentsim.NuggetSimilaritiesResource"

    def get_tsv_path(self, event, hour):
        data_dir = os.path.join(self.dir_, event.fs_name())
        return os.path.join(data_dir, u'{}.tsv.gz'.format(
            hour.strftime(u'%Y-%m-%d-%H')))

    def get_dataframe(self, event, hour):
        tsv = self.get_tsv_path(event, hour)
        if not os.path.exists(tsv):
            return None
        else:
            with gzip.open(tsv, u'r') as f:
                df = pd.io.parsers.read_csv(
                    f, sep='\t', quoting=3, header=0)
                return df

    def get_matches(self, event, corpus):
        if corpus.year_ == 2014:
            df = get_2014_matches()
        elif corpus.year_ == 2013:
            df = get_2013_matches()
        return df.loc[df[u'query id'] == event.query_id]

    def get_nuggets(self, event, corpus):
        if corpus.year_ == 2014:
            df = get_2014_nuggets()
        elif corpus.year_ == 2013:
            df = get_2013_nuggets()
        return df.loc[df[u'query id'] == event.query_id]

    def get_nugget_latent_vectors_(self, nuggets, vectorizer):

        if corenlp.server.check_status() is False:
            print "starting corenlp.server"
            corenlp.server.start(
                mem="20G", threads=20,
                annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner'])

        texts = nuggets[u'text'].tolist()
        processed_texts = []
        cnlp = corenlp.server.CoreNLPClient()
        
        processed_texts = [stringify_corenlp_doc(cnlp.annotate(text))
                           for text in texts]
        
        return vectorizer.transform(processed_texts)

    def check_coverage(self, event, corpus, **kwargs):

        lvecs = get_resource_manager(u'SentenceLatentVectorsResource')
        data_dir = os.path.join(self.dir_, event.fs_name())
        
        n_chunks = 0
        n_covered = 0

        for hour in event.list_event_hours():
            if os.path.exists(lvecs.get_tsv_path(event, hour)):
                n_chunks += 1
                if os.path.exists(self.get_tsv_path(event, hour)):
                    n_covered += 1

        if n_chunks == 0:
            return 0
        else:
            return float(n_covered) / n_chunks

    def get(self, event, corpus, overwrite=False, n_procs=1, 
            progress_bar=False, **kwargs):
        lvecs = get_resource_manager(
            u'SentenceLatentVectorsResource')
        strings = get_resource_manager(u'SentenceStringsResource')
        data_dir = os.path.join(self.dir_, event.fs_name())

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        jobs = []
        for hour in event.list_event_hours():
            lvec_tsv_path = lvecs.get_tsv_path(event, hour)
            strings_tsv_path = strings.get_tsv_path(event, hour)
            nsim_tsv_path = self.get_tsv_path(event, hour)
            if os.path.exists(lvec_tsv_path):
                if overwrite is True or not os.path.exists(nsim_tsv_path):
                    jobs.append(
                        (lvec_tsv_path, strings_tsv_path, nsim_tsv_path))

        self.do_work(nuggetsim_worker_, jobs, n_procs, progress_bar,
                     event=event, corpus=corpus)

def nuggetsim_worker_(job_queue, result_queue, **kwargs):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    event = kwargs.get(u'event')
    corpus = kwargs.get(u'corpus')
    nsims = get_resource_manager(u'NuggetSimilaritiesResource')
    wtmf_models = get_resource_manager(u'WTMFModelsResource')

    model_path = wtmf_models.get_model_path(event)
    vectorizer = joblib.load(model_path)

    nuggets = nsims.get_nuggets(event, corpus)
    nugget_id_list = nuggets[u'nugget id'].tolist()

    Xnuggets = nsims.get_nugget_latent_vectors_(nuggets, vectorizer)
    matches = nsims.get_matches(event, corpus)

    def check_assessor_matches(stream_id, sentence_id):
        update_id = "{}-{}".format(stream_id, sentence_id)
        assessor_matches = matches[matches[u'update id'].str.match(update_id)]
        n_matches = len(assessor_matches)
        if n_matches > 0:
            return {nid: 1.0 for nid 
                    in assessor_matches[u'nugget id'].tolist()}
        return dict()

    while not job_queue.empty():
        try:
            
            lvec_tsv_path, strings_tsv_path, nsim_tsv_path = \
                job_queue.get(block=False)

            with gzip.open(lvec_tsv_path, u'r') as f:
                lvec_df = pd.io.parsers.read_csv(
                    f, sep='\t', quoting=3, header=0)
            
            with gzip.open(strings_tsv_path, u'r') as f:
                strings_df = pd.io.parsers.read_csv(
                    f, sep='\t', quoting=3, header=0)

            Xstrings = lvec_df.ix[:,2:].as_matrix()           
            S = cosine_similarity(Xnuggets, Y=Xstrings)

            sentence_sim_data = []
            for (ii, lvec_row), (_, strings_row) in izip(
                lvec_df.iterrows(), strings_df.iterrows()):
                assert lvec_row[u'stream id'] == strings_row[u'stream id']
                assert lvec_row[u'sentence id'] == strings_row[u'sentence id']

                sentence_sim_dict = {nugget_id_list[nugget_idx]: nugget_sim
                                     for nugget_idx, nugget_sim
                                     in enumerate(S[:, ii])}
                    
                nugget_ids = check_assessor_matches(
                    lvec_row[u'stream id'], lvec_row[u'sentence id'])

                sentence_sim_dict.update(nugget_ids.items())
                sentence_sim_dict[u'stream id'] = lvec_row[u'stream id']
                sentence_sim_dict[u'sentence id'] = lvec_row[u'sentence id']
                sentence_sim_data.append(sentence_sim_dict)

                ### Need to remove this when I confirm 2013 matches are good
                writeme = "/local/nlp/kedzie/check_matches/" + \
                    event.fs_name() + "." + os.path.basename(lvec_tsv_path)
                if len(nugget_ids) > 0:
                    with gzip.open(writeme, u'w') as f:
                        f.write(strings_row[u'streamcorpus'] + '\n')
                        for nugget_id in nugget_ids:
                            nugget_texts = nuggets[ \
                                nuggets['nugget id'].str.match(
                                    nugget_id)]['text'].tolist()
                            for text in nugget_texts:    
                                f.write("\t" + text + '\n')
 
            names = [u'stream id', u'sentence id'] + nugget_id_list
            df = pd.DataFrame(sentence_sim_data, columns=names)

            with gzip.open(nsim_tsv_path, u'w') as f:
                df.to_csv(f, sep='\t', index=False, index_label=False)  

            result_queue.put(None)
        except Queue.Empty:
            pass

    return True
