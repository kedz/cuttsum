import cuttsum.events
import cuttsum.corpora
from random import randint
from cuttsum.pipeline import InputStreamResource
from cuttsum.classifiers import NuggetRegressor
import pandas as pd
import numpy as np


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

results = []

for event in cuttsum.events.get_events():
    if event.query_num > 25 or event.query_num == 7: continue

    summary = []
    istream = get_input_stream(event, False)

    nuggets = set()
    all_nuggets = set()

    y_int_y_hat = 0
    size_y = 0
    size_y_hat = 0

    for _, row in istream.iterrows():
        all_nuggets.update(row["nuggets"])
        gain = len(row["nuggets"].difference(nuggets))
        if gain > 0:
            oracle = "SELECT"
        else: 
            oracle = "SKIP"
        
        action = "SELECT" if randint(0, 1) == 1 else "SKIP"
        if action == "SELECT" and oracle == "SELECT":
            y_int_y_hat += 1
            size_y += 1
            size_y_hat += 1
        elif action == "SELECT" and oracle == "SKIP":
            size_y_hat += 1
        elif action == "SKIP" and oracle == "SELECT":
            size_y += 1
        if action == "SELECT":
            nuggets.update(row["nuggets"])
            summary.append(row["pretty text"])
    with open("{}.txt".format(event.query_id), "w") as f:
        f.write("\n".join(summary))

    loss = 1 - 2 * (y_int_y_hat) / float(size_y + size_y_hat)
    print size_y_hat, size_y
    egain = len(nuggets) / float(size_y_hat)
    comp = len(nuggets) / float(len(all_nuggets))
    f1 = 2 * (egain * comp) / (comp + egain)
    results.append({
        "event": event.query_id, "loss": loss, "E[gain]": egain, 
        "Comp.": comp, "F1": f1})
    df = pd.DataFrame(results, columns=["event", "loss", "E[gain]", "Comp.", "F1"])
    df_u = df.mean().to_frame().T
    df_u["event"] = "mean"
    df = pd.concat([df, df_u[["event", "loss", "E[gain]", "Comp.", "F1"]]])
    df = df.set_index("event")
    print df


        


