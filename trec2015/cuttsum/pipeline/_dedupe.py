from cuttsum.resources import MultiProcessWorker
import os
from cuttsum.pipeline import ArticlesResource
from sklearn.feature_extraction import FeatureHasher
from cuttsum.misc import si2df
from collections import defaultdict
from itertools import izip
import pandas as pd
import re
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity
import streamcorpus as sc
import numpy as np
import cuttsum.judgements


class DedupedArticlesResource(MultiProcessWorker):
    def __init__(self):
        #Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'deduped-articles')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_stats_df(self, event, corpus, extractor, threshold=.8):
        path = self.get_stats_path(
            event, corpus, extractor, threshold=threshold)
        if not os.path.exists(path): return None
        def _conv(x):
            if x == "":
                return float("nan")
            return float(x)
        with open(path, "r") as f:
            df = pd.read_csv(f, sep="\t",
                converters={
                    "earliest": _conv, 
                    "latest": _conv, 
                    "second": _conv, 
                    "third": _conv,
                },
                dtype={
                    "earliest": np.float64, 
                    "latest": np.float64, 
                    "second": np.float64, 
                    "third": np.float64,
            })
        return df

    def get_stats_path(self, event, corpus, extractor, threshold=.8):
        return os.path.join(
            self.dir_, extractor, corpus.fs_name(), str(threshold), 
            "{}.stats.tsv".format(event.fs_name())) 

    def get_deduped_path_fmt(self, event, corpus, extractor, threshold=.8):
        return os.path.join(
            self.dir_, extractor, corpus.fs_name(), str(threshold), 
            event.fs_name() + ".{}.sc.gz") 


    def streamitem_iter(self, event, corpus, extractor, threshold=.8):
        df = self.get_stats_df(
            event, corpus, extractor, threshold)
        if df is None: return

        import math
        num_chunks = int(math.ceil(len(df) / 1000.))
        tmp = self.get_deduped_path_fmt(
            event, corpus, extractor, threshold)
        for i in xrange(1, num_chunks + 1):
            path = tmp.format(i)
            if os.path.exists(path): 
                with sc.Chunk(path=path, mode="rb", 
                        message=corpus.sc_msg()) as chunk:
                    for si in chunk:
                        yield si

    def dataframe_iter(self, event, corpus, extractor, include_matches=None, threshold=.8):

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

        for si in self.streamitem_iter(event, corpus, extractor, threshold):
            
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
        overwrite = kwargs.get("overwrite", False)
        if overwrite is True: return [0]

        import math
        thresh = kwargs.get("dedupe-sim-threshold", .8)
        extractor = kwargs.get("extractor", "goose")
        df = self.get_stats_df(event, corpus, extractor, thresh)
        if df is None: return [0]

        num_chunks = int(math.ceil(len(df) / 1000.))
        tmp = self.get_deduped_path_fmt(event, corpus, extractor, thresh)
        for i in xrange(1, num_chunks + 1):
            if not os.path.exists(tmp.format(i)): return [0]
                    
        return []

    def do_job_unit(self, event, corpus, unit, **kwargs):
        if unit != 0:
            raise Exception("Job unit {} out of range".format(unit))
        
        res = ArticlesResource()
        thresh = kwargs.get("dedupe-sim-threshold", .8)
        extractor = kwargs.get("extractor", "goose")
        hasher = FeatureHasher(input_type="pair", non_negative=True)
        si_iter = res.streamitem_iter(
            event, corpus, extractor) 

        def to_df(all_ids, all_times, all_matches):
            d = []
            for ids, times, match in izip(all_ids, all_times, all_matches):

                times.sort()
                d.append({
                    "stream ids": ids, "hits": len(ids), "match": match,
                    "earliest": times[0], "latest": times[-1], 
                    "second": times[1] if len(times) >= 2 else None,
                    "third": times[2] if len(times) >= 3 else None,
                })
            return pd.DataFrame(d, columns=["stream ids", "match", "hits", 
                                            "earliest", "latest", 
                                            "second", "third"])    

        def query_in_top20(event, df):
            text = u"\n".join(df["sent text"].tolist()[:20]) 
            for query in event.query:
                if not re.search(query, text, flags=re.I|re.UNICODE):
                    return False
            return True

        def make_time(df):
            return df["timestamp"].tolist()[0]

        def make_counts(df, slimit=20):
            counts = defaultdict(int)
            for words in df["words"].tolist()[:slimit]:
                for word in words:
                    counts[word.lower()] += 1   
            return counts

        def next_chunk_file(chunk_file_num):
            deduped_path_fmt = self.get_deduped_path_fmt(
                event, corpus, extractor, threshold=thresh)
            deduped_path = deduped_path_fmt.format(
                chunk_file_num)
            deduped_dir = os.path.dirname(deduped_path)
            if not os.path.exists(deduped_dir):
                os.makedirs(deduped_dir)
            
            if os.path.exists(deduped_path):
                os.remove(deduped_path)

            return sc.Chunk(path=deduped_path, mode="wb", 
                message=corpus.sc_msg())



        X = None

        chunk_file_num = 1
        chunk = next_chunk_file(chunk_file_num)

        for hour, path, si in si_iter:
            df = si2df(si, extractor=extractor)

            counts = make_counts(df)
            x = hasher.transform([counts.items()])
            x.shape = (1, hasher.n_features)
            
            if X is None:
                X = x
                times = [[make_time(df)]]
                ids = [[si.stream_id]]
                matches = [query_in_top20(event, df)]

                chunk.add(si)
                        
            else:
                K = cosine_similarity(X, x)
                k_argmax = K.argmax()
                
                if K[k_argmax] < thresh:
                    
                    X = vstack([X, x])
                    times.append([make_time(df)])
                    ids.append([si.stream_id])
                    matches.append(query_in_top20(event, df))

                    if X.shape[0] % 1000 == 0:
                        chunk.close()
                        chunk_file_num += 1
                        chunk = next_chunk_file(chunk_file_num)

                    chunk.add(si)
                    
                else:
                    times[k_argmax].append(make_time(df))
                    ids[k_argmax].append(si.stream_id)
               
        chunk.close() 
     
        df = to_df(ids, times, matches)            
        print df

        stats_path = self.get_stats_path(
            event, corpus, extractor, thresh)
        with open(stats_path, "w") as f:
            df.to_csv(f, index=False, sep="\t")

