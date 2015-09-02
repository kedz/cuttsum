import cuttsum.events
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

matches_df = cuttsum.judgements.get_merged_dataframe()
def get_input_stream(event, gold_probs, extractor="goose", thresh=.8, delay=None, topk=20, 
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

    from datetime import datetime
    from collections import defaultdict
   
    ts2count = defaultdict(int) 
    for sids in stats_df["stream ids"].tolist():
        for sid in sids:        
            ts = int(sid.split("-")[0])
            ts2count[ts] += 1


    sid2match = {}
    for _, row in stats_df.iterrows():
        for sid in row["stream ids"]:
            sid2match[sid] = row["match"]

    all_ts = []
    all_docs = []
    new_docs = []
    for (sid, ts), doc in df.groupby(["stream id", "timestamp"]):
        if truncate is True:
            doc = doc.iloc[0:5]
#            print sub_doc
        if len(all_ts) > 0:
            assert ts >= all_ts[-1]
        all_ts.append(ts)
        if sid2match[sid] is True:
            new_docs.append(doc)
        all_docs.append(doc)

        
    df = pd.concat(new_docs)    
    print len(all_docs), len(new_docs)
    return df, ts2count

import matplotlib.pyplot as plt
import sys
from collections import defaultdict
events = cuttsum.events.get_events()
summary_path = os.path.join(sys.argv[1], "summaries.tsv")
with open(summary_path, "r") as f:
    summaries = pd.read_csv(f, sep="\t")
    summaries = summaries[summaries["run"] == "L2S.F1"]
for event_num, summary in summaries.groupby("event"):
    event = [e for e in events if e.query_num == event_num][0]
    sum_doc_counts = defaultdict(int)
    for doc in summary["stream id"].tolist():
        sum_doc_counts[doc] += 1
 
    is_df, ts2count = get_input_stream(event, False, extractor="goose", thresh=.8, delay=None, topk=20, 
        use_2015F=True, truncate=True)
    stream_doc_counts = defaultdict(int)
    for doc in is_df["stream id"].tolist():
        stream_doc_counts[doc] += 1
    doc_counts = stream_doc_counts.items()
    doc_counts.sort(key=lambda x: x[0])
    x_times = []
    y_counts = []
    for doc, counts in doc_counts:
        print doc, "{} / {}".format(sum_doc_counts[doc], counts) 
        timestamp = int(doc.split("-")[0])
        x_times.append(timestamp)
        y_counts.append(sum_doc_counts[doc])
    print

    ts_counts = sorted(ts2count.items(), key=lambda x: x[0])
    x_st = []
    y_sc = []
    for ts, counts in ts_counts:
        for count in range(counts):
            x_st.append(ts)
            y_sc.append(1)    

    ts_counts = np.array(ts_counts)

    x_stream_times = ts_counts[:,0]
    y_stream_counts = ts_counts[:, 1]
    Z = float(y_stream_counts.sum())
    #y_norm_sc = np.max(y_counts) * y_stream_counts / Z
    y_norm_sc = y_stream_counts

    print event
    plt.bar(x_times, y_counts, 60 * 60)
#    plt.hist(x_st, bins=7 * 24, normed=1)
    hist, bin_edges = np.histogram(x_st, bins=7*24)
    hist = np.max(y_counts) * hist / float(hist.max())
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    plt.bar(bin_centers, hist, 60*60, alpha=.2)
    plt.show()
