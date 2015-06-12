from cuttsum.resources import MultiProcessWorker
from cuttsum.trecdata import SCChunkResource
from cuttsum.misc import si2df
import cuttsum.judgements
import signal
import urllib3
import os
import Queue
import gzip
import regex as re
import streamcorpus as sc
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import lxml.etree
import pandas as pd

class ArticlesResource(MultiProcessWorker):
    def __init__(self):
        #Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'articles')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_stats_df(self, event, extractor):
        path = self.get_stats_path(event, extractor)
        if not os.path.exists(path): return None
        with open(path, "r") as f:
            return pd.read_csv(f, sep="\t")


    def get_stats_path(self, event, extractor):
        return os.path.join(
            self.dir_, extractor, event.fs_name(), 
            "{}.stats.tsv".format(event.fs_name()))

    def get_si(self, event, corpus, extractor, hour):
        path = self.get_chunk_path(event, extractor, hour)
        if not os.path.exists(path): return []
        with sc.Chunk(path=path, mode="rb", message=corpus.sc_msg()) as chunk:
            return [si for si in chunk]

    def streamitem_iter(self, event, corpus, extractor):
        for hour in event.list_event_hours():
            path = self.get_chunk_path(event, extractor, hour)
            if os.path.exists(path):
            
                with sc.Chunk(path=path, mode="rb", 
                        message=corpus.sc_msg()) as chunk:
                    for si in chunk:
                        yield hour, path, si   

    def dataframe_iter(self, event, corpus, extractor, include_matches=None):

        if include_matches is not None:

            all_matches = cuttsum.judgements.get_merged_dataframe()
            matches = all_matches[all_matches["query id"] == event.query_id]
        
        if include_matches == "soft":
            from cuttsum.classifiers import NuggetClassifier
            classify_nuggets = NuggetClassifier().get_classifier(event)
            if event.query_id.startswith("TS13"):
                judged = cuttsum.judgements.get_2013_updates() 
                judged = judged[judged["query id"] == event.query_id]
                judged_uids = set(judged["update id"].tolist())
            else:
                raise Exception("Bad corpus!")

        for hour, path, si in self.streamitem_iter(event, corpus, extractor):
            df = si2df(si, extractor=extractor)
                
            if include_matches is not None:
                df["nuggets"] = df["update id"].apply(
                    lambda x: set(
                        matches[
                            matches["update id"] == x]["nugget id"].tolist()))

            if include_matches == "soft":
                ### NOTE BENE: geting an array of indices to index unjudged
                # sentences so I can force pandas to return a view and not a
                # copy.
                I = np.where(
                    df["update id"].apply(lambda x: x not in judged_uids))[0]
                
                unjudged = df[
                    df["update id"].apply(lambda x: x not in judged_uids)]
                unjudged_sents = unjudged["sent text"].tolist()
                assert len(unjudged_sents) == I.shape[0]

                df.loc[I, "nuggets"] = classify_nuggets(unjudged_sents)


            yield df

    def get_job_units(self, event, corpus, **kwargs):

        extractor = kwargs.get("extractor", "gold")
        overwrite = kwargs.get("overwrite", False)
        data_dir = os.path.join(self.dir_, extractor, event.fs_name())
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        chunks_resource = SCChunkResource()

        if extractor == "gold":

            import cuttsum.judgements

            all_matches = cuttsum.judgements.get_matches()
            matches = all_matches[all_matches["query id"] == event.query_id]

            hours = set([datetime.utcfromtimestamp(int(update_id.split("-")[0])).replace(
                             minute=0, second=0)
                         for update_id in matches["update id"].tolist()])
            hours = sorted(list(hours))
           
            units = []
            for h, hour in enumerate(hours):
                output_path = self.get_chunk_path(event, extractor, hour)
                if overwrite is True or not os.path.exists(output_path):
                    units.append(h)
            return units

        elif extractor == "goose":
            hours = event.list_event_hours()
            units = []
            for h, hour in enumerate(hours):
                output_path = self.get_chunk_path(event, extractor, hour)
                if overwrite is True or not os.path.exists(output_path):
                    units.append(h)
            return units

        else:
            raise Exception("extractor: {} not implemented!".format(extractor))

    def do_job_unit(self, event, corpus, unit, **kwargs):
        
        extractor = kwargs.get("extractor", "gold")
        data_dir = os.path.join(self.dir_, extractor, event.fs_name())
        chunks_resource = SCChunkResource()
        
        if extractor == "gold":
            import cuttsum.judgements
            all_matches = cuttsum.judgements.get_matches()
            matches = all_matches[all_matches["query id"] == event.query_id]
            stream_ids = set(
                matches["update id"].apply(
                    lambda x: "-".join(x.split("-")[:-1])).tolist())

            hours = set([datetime.utcfromtimestamp(
                            int(update_id.split("-")[0])).replace(
                             minute=0, second=0)
                         for update_id in matches["update id"].tolist()])
            hours = sorted(list(hours))
            hour = hours[unit]
            output_path = self.get_chunk_path(event, extractor, hour)
            gold_si = []
            for path in chunks_resource.get_chunks_for_hour(hour, corpus):
                with sc.Chunk(path=path, mode="rb", 
                        message=corpus.sc_msg()) as chunk:
                    for si in chunk:
                        if si.stream_id in stream_ids:
                            gold_si.append(si)

            gold_si.sort(key=lambda x: x.stream_id)
            for si in gold_si:
                print si.stream_id

            if os.path.exists(output_path):
                os.remove(path)            
            with sc.Chunk(path=output_path, mode="wb", 
                    message=corpus.sc_msg()) as chunk:
                for si in gold_si:
                    chunk.add(si)

        elif extractor == "goose":

            import nltk
            from nltk.tokenize import WordPunctTokenizer
            sent_tok = nltk.data.load('tokenizers/punkt/english.pickle')
            word_tok = WordPunctTokenizer()           
 
            from goose import Goose, Configuration
            config = Configuration()
            config.enable_image_fetching = False
            g = Goose(config)
            
            hour = event.list_event_hours()[unit]
            output_path = self.get_chunk_path(event, extractor, hour)
            good_si = []

            for path in chunks_resource.get_chunks_for_hour(hour, corpus):
                with sc.Chunk(path=path, mode="rb", 
                        message=corpus.sc_msg()) as chunk:
                    
                    for si in chunk:

                        if si.body.clean_visible is None:
                            continue

                        article_text = self._get_goose_text(g, si)
                        if article_text is None:
                            continue

                        if not self._contains_query(event, article_text):
                            continue
                
                        art_pretty = sent_tok.tokenize(article_text)
                        art_sents = [word_tok.tokenize(sent) 
                                     for sent in art_pretty]

                        df = si2df(si)
                        I = self._map_goose2streamitem(
                            art_sents, df["words"].tolist())
                            
                        if "serif" in si.body.sentences:
                            si_sentences = si.body.sentences["serif"]
                        elif "lingpipe" in si.body.sentences:
                            si_sentences = si.body.sentences["lingpipe"]
                        else:
                            raise Exception("Bad sentence annotator.")
                        
                        ann = sc.Annotator()
                        ann.annotator_id = "goose"
                        si.body.sentences["goose"] = [sc.Sentence() 
                                                      for _ in si_sentences]
                        for i_goose, i_si in enumerate(I):
                            print art_pretty[i_goose]
                            print df.loc[i_si, "sent text"]
                            print
                            tokens = [sc.Token(token=token.encode("utf-8")) 
                                      for token in art_sents[i_goose]]
                            si.body.sentences["goose"][i_si].tokens.extend(
                                tokens)
                        good_si.append(si)
            if len(good_si) == 0:
                print "Nothing in hour:", hour
                return 

            good_si.sort(key=lambda x: x.stream_id)
            for si in good_si:
                print si.stream_id

            if os.path.exists(output_path):
                os.remove(path)            
            print "Writing to", output_path
            with sc.Chunk(path=output_path, mode="wb", 
                    message=corpus.sc_msg()) as chunk:
                for si in good_si:
                    chunk.add(si)
        else:
            raise Exception("extractor: {} not implemented!".format(extractor))


    def _get_goose_text(self, g, si):
        try:
            article = g.extract(raw_html=si.body.clean_html)
            return article.cleaned_text
        except IndexError:
            return None
        except lxml.etree.ParserError:
            return None

    def _contains_query(self, event, article_text):
        for query in event.query: 
            if not re.search(query, article_text, re.I):
                return False
        return True

    def _map_goose2streamitem(self, goose_sents, si_sents):

        goose_wc = []
        for sent in goose_sents:
            c = {}
            for token in sent:
                token = token.lower()
                if isinstance(token, unicode):
                    token = token.encode("utf-8") 
                c[token] = c.get(token, 0) + 1
            goose_wc.append(c)
        si_wc = []
        for sent in si_sents:
            c = {}
            for token in sent:
                token = token.lower()
                if isinstance(token, unicode):
                    token = token.encode("utf-8") 
                c[token] = c.get(token, 0) + 1
            si_wc.append(c)

        vec = DictVectorizer()
        vec.fit(goose_wc + si_wc)
        X_goose = vec.transform(goose_wc)
        X_si = vec.transform(si_wc)
        K = cosine_similarity(X_goose, X_si)
        I = np.argmax(K, axis=1)
        return I                                    

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


    def get_chunk_path(self, event, extractor, hour):
        data_dir = os.path.join(self.dir_, extractor, event.fs_name())
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

