import cuttsum.events
import cuttsum.corpora 
from cuttsum.pipeline import InputStreamResource
from cuttsum.classifiers import NuggetRegressor
import cuttsum.judgements
import pandas as pd
import numpy as np
from datetime import datetime
from cuttsum.misc import event2semsim
from sklearn.cluster import AffinityPropagation
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from cuttsum.misc import event2lm_name
import math

def sigmoid(x):
    return 1. / (1. + math.exp(-x))


lm2thr = {
    "accidents-lm": 0.15,
    "natural_disaster-lm": 0.10,
    "social_unrest-lm": 0.70,
    "terrorism-lm": 0.40,
}

matches_df = cuttsum.judgements.get_merged_dataframe()
def get_input_stream(event, gold_probs, extractor="goose", thresh=.8, delay=None, topk=20):
    max_nuggets = 3
    corpus = cuttsum.corpora.get_raw_corpus(event)
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

    return df

event2nuggets = defaultdict(set)
event2size = {}



all_results = []
data = []
with open("apsal.tsv", "w") as o:
    for event in cuttsum.events.get_events():
        if event.query_num < 26: continue
        istream = get_input_stream(event, False)

        with open("clusters-2015/{}.tsv".format(event.query_id), "r") as f:
            df = pd.read_csv(f, sep="\t", converters={"stems": eval, "nuggets": eval})

        thresh = lm2thr[event2lm_name(event)]
        thresh = .7

        cache = None
        semsim = event2semsim(event)
        results = []
        for ts, batch in df.groupby("timestamp"):
            X = semsim.transform(batch["stems"].apply(lambda x: ' '.join(x)).tolist())
            for i, (_, row) in enumerate(batch.iterrows()):
                if cache is None:
                    cache = X[i]
                    results.append(row.to_dict())
                    all_results.append(row.to_dict())
                else:
                    K = cosine_similarity(cache, X[i])
                    if (K < thresh).all(): 
                        cache = np.vstack([cache, X[i]])
                        results.append(row.to_dict())
                        all_results.append(row.to_dict())
        for result in results:
            print "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                event.query_num, "cunlp", "APSAL1", "-".join(result["update id"].split("-")[:2]), result["update id"].split("-")[-1], result["timestamp"], sigmoid(result["probs"]))
            o.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                event.query_num, "cunlp", "APSAL1", "-".join(result["update id"].split("-")[:2]), result["update id"].split("-")[-1], result["timestamp"], sigmoid(result["probs"])))

