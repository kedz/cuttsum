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

data = []
for event in cuttsum.events.get_events():
    if event.query_num == 7 or event.query_num > 25: continue
    print event
    istream = get_input_stream(event, False)
    event2size[event] = len(istream)


    for nuggets in istream["nuggets"].tolist():
        event2nuggets[event].update(nuggets)        

    with open("clusters/{}.tsv".format(event.query_id), "r") as f:
        df = pd.read_csv(f, sep="\t", converters={"stems": eval, "nuggets": eval})
    data.append((event, df))
    

all_results = []
for thresh in np.arange(.05, .95, .05):
    print thresh
    results = []

    for event, df in data:
        cache = None
        semsim = event2semsim(event)
        nuggets = set()
        for ts, batch in df.groupby("timestamp"):
            X = semsim.transform(batch["stems"].apply(lambda x: ' '.join(x)).tolist())
            for i, (_, row) in enumerate(batch.iterrows()):
                if cache is None:
                    cache = X[i]
                    nuggets.update(row["nuggets"])
                else:
                    K = cosine_similarity(cache, X[i])
                    if (K < thresh).all(): 
                        cache = np.vstack([cache, X[i]])
                        nuggets.update(row["nuggets"])
                
        egain = len(nuggets) / float(cache.shape[0])     
        comp = len(nuggets) / float(len(event2nuggets[event]))
        f1 = 2 * (egain * comp) / (comp + egain) if comp + egain > 0 else 0
        results.append({"thr": thresh, "event": event.query_id, "E[gain]": egain, "Comp.": comp, "F1": f1, "size": event2size[event]})
        all_results.append({"thr": thresh, "event": event.query_id, "E[gain]": egain, "Comp.": comp, "F1": f1, "size": event2size[event]})
        print pd.DataFrame(results, columns=["thr", "event", "size", "E[gain]", "Comp.", "F1"])
        
df = pd.DataFrame(all_results, columns=["thr", "event", "size", "E[gain]", "Comp.", "F1"])
with open("filters.tsv", "w") as f:
    df.to_csv(f, sep="\t", index=False)

