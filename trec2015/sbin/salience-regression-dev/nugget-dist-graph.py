import cuttsum.events 
import cuttsum.judgements
import cuttsum.corpora
from cuttsum.pipeline import InputStreamResource
from cuttsum.classifiers import NuggetRegressor
import matplotlib.pylab as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np
from datetime import datetime

nuggets = cuttsum.judgements.get_nuggets()
matches_df = cuttsum.judgements.get_merged_dataframe()


def get_input_stream(event, gold_probs, extractor="goose", thresh=.8, delay=None, topk=20,
        max_nuggets=None, is_filter=False):
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
        lambda x: set([key for key, val in x.items() if val > .97]))
    
    if max_nuggets is not None:        

        def sortme(x):    
            l = [(key, val) for key, val in x.items() if val > .5]
            sorted(l, key=lambda y: y[1], reverse=True)
            return [k for k,v in l[:max_nuggets]]
        df["nuggets"] = df["nuggets"].apply(lambda x: x if len(x) <= max_nuggets else set([]))
        #df["nuggets"] = df["nugget probs"].apply(sortme)

    if is_filter:
        nid2time = {}
        nids = set(matches_df[matches_df["query id"] == event.query_id]["nugget id"].tolist())
        for nid in nids:
            ts = matches_df[matches_df["nugget id"] == nid]["update id"].apply(lambda x: int(x.split("-")[0])).tolist()
            ts.sort()
            nid2time[nid] = ts[0]
        #tss = nuggets[nuggets["query id"] == event.query_id]["timestamp"].tolist()
        #ids = nuggets[nuggets["query id"] == event.query_id]["nugget id"].tolist()
        #nt = {nid: ts for ts, nid in zip(tss, ids)}
        fltr_nuggets = []
        for name, row in df.iterrows():
            fltr_nuggets.append(
                set([nug for nug in row["nuggets"] if nid2time[nug] <= row["timestamp"]]))
        #print df[["nuggets", "timestamp"]].apply(lambda y: print y[0]) #  datetime.utcfromtimestamp(int(y["timestamp"])))
        #print nids
        df["nuggets"] = fltr_nuggets
    return df



plt.close("all")
df = cuttsum.judgements.get_merged_dataframe()
i = 1
for event in cuttsum.events.get_events():
    if event.query_num > 25 or event.query_num == 7: continue
    print event.fs_name()
    timestamps = df[df["query id"] == event.query_id]["update id"].apply(lambda x: datetime.utcfromtimestamp(int(x.split("-")[0]))).tolist() 
    z = (event.end - event.start).total_seconds()
    ts_norm = [(ts - event.start).total_seconds() / z for ts in timestamps]
    y = [i] * len(ts_norm)
    plt.plot(ts_norm, y, "x")

    stream = get_input_stream(event, True, max_nuggets=3, is_filter=True)
    timestamps = stream[stream["nuggets"].apply(len) > 0]["timestamp"].tolist()
    timestamps = [datetime.utcfromtimestamp(int(ts)) for ts in timestamps]
    ts_norm = [(ts - event.start).total_seconds() / z for ts in timestamps]
    plt.plot(ts_norm, [i + .5] * len(ts_norm), "x")

#    stream = get_input_stream(event, True, max_nuggets=None)
#    timestamps = stream[stream["nuggets"].apply(len) > 0]["timestamp"].tolist()
#    timestamps = [datetime.utcfromtimestamp(int(ts)) for ts in timestamps]
#    ts_norm = [(ts - event.start).total_seconds() / z for ts in timestamps]
#    plt.plot(ts_norm, [i + .6] * len(ts_norm), "x")



#    if event.query_num == 3:
#        from collections import defaultdict
#        counts = defaultdict(int)
#        for name, row in stream.iterrows():
#            for nid in row["nuggets"]:
#                counts[nid] += 1
#        items = sorted(counts.items(), key=lambda x: x[1])
#        for k, v in items:
#            print k, v, nuggets[nuggets["nugget id"] == k].iloc[0]["text"]
#        top2 = set([k for k,v in items[-2:]])
#        for name, row in stream.iterrows():
#            if len(top2.intersection(row["nuggets"])) > 0:
#                print row["nuggets"]
#                print row["pretty text"]


    i += 1
plt.gca().set_ylim([0, 25])
plt.yticks(range(1,25))
plt.savefig("test.png")
