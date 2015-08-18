import os
from cuttsum.resources import MultiProcessWorker
from cuttsum.pipeline import DedupedArticlesResource, SentenceFeaturesResource
import cuttsum.judgements
import gzip
import pandas as pd
import numpy as np

class InputStreamResource(MultiProcessWorker):
    def __init__(self):
        #Resource.__init__(self)
        self.dir_ = os.path.join(
            os.getenv(u'TREC_DATA', u'.'), u'input-streams')
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def get_path(self, event, corpus, extractor, threshold, delay, topk):
        path = os.path.join(
            self.dir_, corpus.fs_name(), extractor, 
            "dedupe-" + str(threshold), "delay-" + str(delay), 
            "top-"+str(topk), "{}.tsv.gz".format(event.fs_name()))
        return path

    def get_dataframes(self, event, corpus, extractor, 
                       threshold, delay, topk):
        path = self.get_path(
            event, corpus, extractor, threshold, delay, topk)
        if not os.path.exists(path): return []
        
        with gzip.open(path, "r") as f:
            df = pd.read_csv(f, converters={"nuggets": eval, "nugget probs": eval, 
                                            "lemmas stopped": eval, "stems": eval}, sep="\t")
            stream = [(sid, group) 
                      for sid, group in df.groupby("stream id")]
            stream.sort(key=lambda x: x[0])
            return [df.reset_index(drop=True) for sid, df in stream]

    def get_job_units(self, event, corpus, **kwargs):
        thresh = kwargs.get("dedupe-sim-threshold", .8)
        extractor = kwargs.get("extractor", "goose")
        delay = kwargs.get("delay", None)
        topk = kwargs.get("top-k", 20)
        
        if delay is not None:
            raise Exception("Delay must be None")
        path = self.get_path(
            event, corpus, extractor, thresh, delay, topk)
        if not os.path.exists(path): return [0]

        return []


    def do_job_unit(self, event, corpus, unit, **kwargs):  

        if unit != 0:
            raise Exception("Job unit {} out of range".format(unit))

        thresh = kwargs.get("dedupe-sim-threshold", .8)
        extractor = kwargs.get("extractor", "goose")
        delay = kwargs.get("delay", None)
        topk = kwargs.get("top-k", 20)
        
        if delay is not None:
            raise Exception("Delay must be None")

        feats_df = SentenceFeaturesResource().get_dataframe(
            event, corpus, extractor, thresh)        
        ded_articles_res = DedupedArticlesResource()
        dfiter = ded_articles_res.dataframe_iter(
            event, corpus, extractor, None, thresh)

        all_matches = cuttsum.judgements.get_merged_dataframe()
        matches = all_matches[all_matches["query id"] == event.query_id]
        
        from cuttsum.classifiers import NuggetClassifier
        classify_nuggets = NuggetClassifier().get_classifier(event)
        if event.query_id.startswith("TS13"):
            judged = cuttsum.judgements.get_2013_updates() 
            judged = judged[judged["query id"] == event.query_id]
            judged_uids = set(judged["update id"].tolist())
        elif event.query_id.startswith("TS14"):
            judged = cuttsum.judgements.get_2014_sampled_updates() 
            judged = judged[judged["query id"] == event.query_id]
            judged_uids = set(judged["update id"].tolist())
        else:
            raise Exception("Bad corpus!")
              
      
        feats_df["nuggets"] = feats_df["update id"].apply(
            lambda x: set(
                matches[matches["update id"] == x]["nugget id"].tolist()))
        feats_df["n conf"] = feats_df["update id"].apply(lambda x: 1 if x in judged_uids else None)
                

            #if include_matches == "soft":
                ### NOTE BENE: geting an array of indices to index unjudged
                # sentences so I can force pandas to return a view and not a
                # copy.
        I = np.where(
            feats_df["update id"].apply(lambda x: x not in judged_uids))[0]
        
        unjudged = feats_df[
            feats_df["update id"].apply(lambda x: x not in judged_uids)]
        #unjudged_sents = unjudged["sent text"].tolist()
        #assert len(unjudged_sents) == I.shape[0]
        feats_df["nugget probs"] = [dict() for x in xrange(len(feats_df))]
        if I.shape[0] > 0:
            nuggets, conf, nugget_probs = classify_nuggets(unjudged)
            feats_df.loc[I, "nuggets"] = nuggets
            feats_df.loc[I, "n conf"] = conf
            feats_df.loc[I, "nugget probs"] = nugget_probs





        path = self.get_path(event, corpus, extractor, thresh, delay, topk)
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname): os.makedirs(dirname)

        cols = ["update id", "stream id", "sent id", "timestamp", 
                "sent text",] 
        nugget_cols = ["nuggets", "n conf", "nugget probs"]

        ling_cols = [
            "pretty text", "tokens", "lemmas", "pos", "ne", "tokens stopped",
            "lemmas stopped"]

        basic_cols = ["BASIC length", "BASIC char length",
            "BASIC doc position", "BASIC all caps ratio",
            "BASIC upper ratio", "BASIC lower ratio",
            "BASIC punc ratio", "BASIC person ratio",
            "BASIC location ratio",
            "BASIC organization ratio", "BASIC date ratio",
            "BASIC time ratio", "BASIC duration ratio",
            "BASIC number ratio", "BASIC ordinal ratio",
            "BASIC percent ratio", "BASIC money ratio",
            "BASIC set ratio", "BASIC misc ratio"]

        lm_cols = ["LM domain lp", "LM domain avg lp",
                   "LM gw lp", "LM gw avg lp"]

        query_cols = [
            "Q_query_sent_cov",
            "Q_sent_query_cov",
            "Q_syn_sent_cov",
            "Q_sent_syn_cov",
            "Q_hyper_sent_cov",
            "Q_sent_hyper_cov",
            "Q_hypo_sent_cov",
            "Q_sent_hypo_cov",
        ]

        sum_cols = [
            "SUM_sbasic_sum",
            "SUM_sbasic_amean",
            "SUM_sbasic_max",
            "SUM_novelty_gmean",
            "SUM_novelty_amean",
            "SUM_novelty_max",
            "SUM_centrality",
            "SUM_pagerank",
        ]
    
        stream_cols = [
            "STREAM_sbasic_sum",
            "STREAM_sbasic_amean",
            "STREAM_sbasic_max",
            "STREAM_per_prob_sum",
            "STREAM_per_prob_max",
            "STREAM_per_prob_amean",
            "STREAM_loc_prob_sum",
            "STREAM_loc_prob_max",
            "STREAM_loc_prob_amean",
            "STREAM_org_prob_sum",
            "STREAM_org_prob_max",
            "STREAM_org_prob_amean",
            "STREAM_nt_prob_sum",
            "STREAM_nt_prob_max",
            "STREAM_nt_prob_amean",
        ]


 
        output_cols = cols + nugget_cols + ling_cols + basic_cols + lm_cols + query_cols + sum_cols + stream_cols

        with gzip.open(path, "w") as f:
            f.write("\t".join(output_cols) + "\n")
            for df in dfiter:
                df = df.head(topk)
                df["sent text"] = df["sent text"].apply(lambda x: x.encode("utf-8"))
                merged = pd.merge(
                    df[cols], 
                    feats_df, 
                    on=["update id", "stream id", "sent id", "timestamp"])
                if len(merged) == 0:
                    print "Warning empty merge"

                merged[output_cols].to_csv(f, index=False, header=False, sep="\t")
        
