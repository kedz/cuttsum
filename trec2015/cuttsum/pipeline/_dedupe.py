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
        with open(path, "r") as f:
            df = pd.read_csv(f, sep="\t", engine="python")
        return df

    def get_stats_path(self, event, corpus, extractor, threshold=.8):
        return os.path.join(
            self.dir_, extractor, corpus.fs_name(), str(threshold), 
            "{}.stats.tsv".format(event.fs_name())) 

    def get_deduped_path(self, event, corpus, extractor, threshold=.8):
        return os.path.join(
            self.dir_, extractor, corpus.fs_name(), str(threshold), 
            "{}.sc.gz".format(event.fs_name())) 

    def get_job_units(self, event, corpus, **kwargs):
        return [0]

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

        X = None

        deduped_path = self.get_deduped_path(
            event, corpus, extractor, threshold=thresh)
        deduped_dir = os.path.dirname(deduped_path)
        if not os.path.exists(deduped_dir):
            os.makedirs(deduped_dir)
        if os.path.exists(deduped_path):
            os.remove(deduped_path)

        with sc.Chunk(path=deduped_path, mode="wb", 
                message=corpus.sc_msg()) as chunk:

            for hour, path, si in si_iter:
                df = si2df(si, extractor=extractor)

                counts = make_counts(df)
                x = hasher.transform([counts.items()])
                x.shape = (1, hasher.n_features)
                
                if X is None:
                    X = x
                    times = [[make_time(df)]]
                    ids = [[si.stream_id]]
                    #sis = [si]
                    matches = [query_in_top20(event, df)]
                    chunk.add(si)
                            
                else:
                    K = cosine_similarity(X, x)
                    k_argmax = K.argmax()
                    
                    if K[k_argmax] < thresh:
                        
                        X = vstack([X, x])
                        times.append([make_time(df)])
                        ids.append([si.stream_id])
                        #sis.append(si)
                        matches.append(query_in_top20(event, df))
                        chunk.add(si)
                    else:
                        times[k_argmax].append(make_time(df))
                        ids[k_argmax].append(si.stream_id)
     
        df = to_df(ids, times, matches)            
        print df

        stats_path = self.get_stats_path(
            event, corpus, extractor, thresh)
        with open(stats_path, "w") as f:
            df.to_csv(f, index=False, sep="\t")

