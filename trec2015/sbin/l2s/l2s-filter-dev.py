import cuttsum.events
import pandas as pd
import os
import cuttsum.corpora
from cuttsum.pipeline import InputStreamResource
from mpi4py import MPI
from cuttsum.misc import enum
from cuttsum.classifiers import NuggetRegressor
import numpy as np
import pandas as pd
import random
import pyvw
from datetime import datetime
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics.pairwise import cosine_similarity
from itertools import izip
import os
import cuttsum.judgements
from cuttsum.misc import event2semsim
import math

np.random.seed(42)

matches_df = cuttsum.judgements.get_merged_dataframe()


def get_input_stream(event, gold_probs, extractor="goose", thresh=.8, 
        delay=None, topk=20, 
        use_2015F=False, truncate=False):
    max_nuggets = 3
    
    corpus = cuttsum.corpora.get_raw_corpus(event)
    if use_2015F is True and event.query_num > 25:
        corpus = cuttsum.corpora.FilteredTS2015()
    print event, corpus

    res = InputStreamResource()
    df = pd.concat(
        res.get_dataframes(event, corpus, extractor, thresh, delay, topk))

    selector = (df["n conf"] == 1) & (df["nugget probs"].apply(len) == 0)
    df.loc[selector, "nugget probs"] = df.loc[selector, "nuggets"].apply(lambda x: {n:1 for n in x})

    df["true probs"] = df["nugget probs"].apply(lambda x: [val for key, val in x.items()] +[0])
    df["true probs"] = df["true probs"].apply(lambda x: np.max(x))
    df.loc[(df["n conf"] == 1) & (df["nuggets"].apply(len) == 0), "true probs"] = 0

    if gold_probs is True:
        df["probs"] = df["true probs"]
    else:
        df["probs"] = NuggetRegressor().predict(event, df)   
    
    df["nuggets"] = df["nugget probs"].apply(
        lambda x: set([key for key, val in x.items() if val > .9]))


    nid2time = {}
    nids = set(matches_df[matches_df["query id"] == event.query_id]["nugget id"].tolist())
    for nid in nids:
        ts = matches_df[matches_df["nugget id"] == nid]["update id"].apply(lambda x: int(x.split("-")[0])).tolist()
        ts.sort()
        nid2time[nid] = ts[0]

    fltr_nuggets = []
    for name, row in df.iterrows():
        fltr_nuggets.append(
            set([nug for nug in row["nuggets"] if nid2time[nug] <= row["timestamp"]]))
    #print df[["nuggets", "timestamp"]].apply(lambda y: print y[0]) #  datetime.utcfromtimestamp(int(y["timestamp"])))
    #print nids
    df["nuggets"] = fltr_nuggets

    df["nuggets"] = df["nuggets"].apply(lambda x: x if len(x) <= max_nuggets else set([]))


    from cuttsum.pipeline import DedupedArticlesResource
    ded = DedupedArticlesResource()
    stats_df = ded.get_stats_df(event, corpus, extractor, thresh)
    stats_df["stream ids"] = stats_df["stream ids"].apply(lambda x: set(eval(x)))
    sid2match = {}
    for _, row in stats_df.iterrows():
        for sid in row["stream ids"]:
            sid2match[sid] = row["match"]

    all_ts = []
    all_docs = []
    new_docs = []
    for (sid, ts), doc in df.groupby(["stream id", "timestamp"]):
        #if truncate is True:
        doc = doc.iloc[0:20]
#            print sub_doc
        if len(all_ts) > 0:
            assert ts >= all_ts[-1]
        all_ts.append(ts)
        if sid2match[sid] is True:
            new_docs.append(doc)
        all_docs.append(doc)

        
    df = pd.concat(new_docs)    
    print len(all_docs), len(new_docs)
    return df

def get_all_semsim():
    accident_semsim = event2semsim("accident")
    natdis_semsim = event2semsim("earthquake")
    social_semsim = event2semsim("protest")
    terror_semsim = event2semsim("shooting")
    return {
        "accident": accident_semsim,
        "earthquake": natdis_semsim,
        "storm": natdis_semsim,
        "impact event": natdis_semsim,
        "shooting": terror_semsim,                
        "hostage": terror_semsim,                
        "conflict": terror_semsim,                
        "bombing": terror_semsim,                
        "protest": social_semsim,
        "riot": social_semsim,
    }   


def main(output_dir, sim_thresh):
    dev_qids = set([19, 23, 27, 34, 35])

    semsims = get_all_semsim()
    events = [e for e in cuttsum.events.get_events() 
              if e.query_num in dev_qids]

    run_data = []
    for event in events:

        sub_path = os.path.join("crossval", "l2.00001", str(event.query_num),
                "submission.tsv")
        sub_df = pd.read_csv(sub_path, sep="\t", header=None)
        sub_df = sub_df[sub_df[2] == "L2S.F1"]
        good_uids = sub_df[3]+ "-" + sub_df[4].apply(str)
        good_uids = set(good_uids.tolist())
        
        stream_df = get_input_stream(event, True, extractor="goose",
            thresh=.8, delay=None, topk=20, use_2015F=False)

        updates_df = stream_df[stream_df["update id"].isin(good_uids)]
        
        assert len(updates_df) == len(sub_df)
        X = semsims[event.type].transform(
            updates_df["stems"].apply(lambda x: ' '.join(x)).tolist())
       
        K = cosine_similarity(X)

        Isum = [0]
        if K.shape[0] > 1:
            for i in range(1, K.shape[0]):
                #print K[Isum][:,i]
                if K[Isum][:,i].max() < sim_thresh:
                    Isum.append(i)

        fltr_df = updates_df.reset_index(drop=True).iloc[Isum]
        fltr_df = fltr_df[["stream id", "sent id", "timestamp"]]
        fltr_df["query id"] = event.query_num
        fltr_df["team id"] = "l2s.fltr"
        fltr_df["conf"] = .5
        fltr_df["run id"] = "sim{}".format(sim_thresh)
        run_data.append(fltr_df)

    df = pd.concat(run_data)
    cols = ["query id", "team id", "run id", "stream id", "sent id", 
            "timestamp", "conf"]
    print df[cols]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tsv = os.path.join(output_dir, "l2s.fltr.sim{}.tsv".format(sim_thresh))
    df[cols].to_csv(tsv, sep="\t", header=None, index=None)

if __name__ == u"__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(u"--output-dir", type=str,
                        required=True, help="directory to write results.")
    parser.add_argument(
        u"--sim-thresh", type=float, required=True, 
        help=u"cosine sim filter threshold")

    args = parser.parse_args()

    main(args.output_dir, args.sim_thresh)
 
